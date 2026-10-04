"""Run Stage 1 on individual segments before joining the calibrated visit."""
import dataclasses
import os
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np

from exotedrf.v2 import core, stages
from exotedrf.v2.pipeline import Pipeline, PipelineState

# Join segments before the first visit-wide correction.
JOIN_STEPS = ('BackgroundStep', 'OneOverFStep_int', 'BadPixStep', 'PCAReconstructStep', 'Extract')

DEVICE_RESIDENT_ENV = 'EXOTEDRF_STREAM_DEVICE_JOIN'


def segment_state(state, index):
    """Slice arrays and per-file metadata, retaining exposure-global timing.

    Parameters
    ----------
    state : PipelineState
        Observation arrays and accumulated calibration results.
    index : int
        Zero-based segment index.

    Returns
    -------
    state : PipelineState
        Segment state or joined calibrated state after the requested steps.
    """
    cube = state.cube
    slices = core.segment_slices(cube.meta, cube.data.shape[0])
    seg = slices[index]
    count = seg.stop - seg.start
    extra = dict(cube.meta.extra)
    starts = cube.meta.segment_int_starts
    end = int(starts[index]) + count - 1
    exposure_end = max(int(start) + s.stop - s.start for start, s in zip(starts, slices)) - 1
    extra['exposure_nints'] = max(int(extra.get('exposure_nints', 0)), exposure_end)
    for key in ('segment_headers', 'segment_int_starts', 'segment_int_ends'):
        if key in extra:
            extra[key] = (extra[key][index],)
    extra['segment_int_starts'] = (int(starts[index]),)
    extra['segment_int_ends'] = (end,)
    meta = dataclasses.replace(cube.meta, int_times=cube.meta.int_times[seg],
        segment_edges=np.asarray([count]), filenames=(cube.meta.filenames[index:index + 1]
                   if cube.meta.filenames else ()), extra=extra)
    # Upload each segment once before its detector corrections.
    host_data, host_groupdq = cube.data[seg], cube.groupdq[seg]
    data, groupdq = jnp.asarray(host_data), jnp.asarray(host_groupdq)
    del host_data, host_groupdq
    # Keep pixel flags and observing metadata on the host.
    return PipelineState(core.RampCube(data, groupdq, cube.pixeldq, meta), dict(state.aux))


def segment_context(ctx, cube, seg):
    """Slice visit-length series and masks in a copy of the context for one segment."""
    local = dict(ctx)
    for key in ('soss_timeseries', 'soss_timeseries_o2', 'outlier_mask'):
        value = ctx.get(key)
        if value is not None and value.shape[0] == cube.data.shape[0]:
            local[key] = value[seg]
    return local


def _logger(logger):
    """Wrap the optional logger with the segment-streaming prefix."""
    def log(message):
        """Send a prefixed streaming progress message to the optional logger."""
        if logger is not None:
            logger(f'[v2-stream] {message}')
    return log


def _join_index(names):
    """Locate the first step that requires the joined observation."""
    for index, name in enumerate(names):
        if name in JOIN_STEPS:
            return index
    raise ValueError('segment streaming requires a joined stage-2 step; '
                     f'none of {list(JOIN_STEPS)} is present')


def _check_graph(pipeline, expected, names, store, join_index):
    """Reject any workflow whose segmented form would not be identical."""
    if 'RampFitStep' not in names:
        raise ValueError('segment streaming requires RampFitStep')
    if [(s.name, s.fn) for s in expected] != [(s.name, s.fn) for s in pipeline.steps]:
        raise ValueError(f'segment streaming requires the built-in {pipeline.mode} graph')
    if store is not None and (store.keep is None or store.keep.intersection(names[:join_index])):
        raise ValueError('segment streaming cannot retain ramp checkpoints')


def _gather_last_group(buffer, segment, data, chunk_bytes=32 << 20):
    """Copy one segment's final-group planes into a visit-length host buffer."""
    nints = int(data.shape[0])
    plane_bytes = (int(np.prod(data.shape[-2:])) * np.dtype(data.dtype).itemsize)
    batch = max(1, int(chunk_bytes) // max(1, plane_bytes))
    for lo in range(0, nints, batch):
        hi = min(lo + batch, nints)
        buffer[segment.start + lo:segment.start + hi] = core.to_host(
            data[lo:hi, -1] if data.ndim == 4 else data[lo:hi])


def _device_residency_allowed():
    """Check whether streamed rates may remain on the device."""
    policy = os.environ.get(DEVICE_RESIDENT_ENV, 'auto').strip().lower()
    if policy in ('0', 'false', 'off'):
        return False
    if policy in ('1', 'true', 'on'):
        return True
    if policy != 'auto':
        raise ValueError(f'{DEVICE_RESIDENT_ENV} must be auto, on/true/1, or off/false/0')
    # Use spillable host buffers for CPU joins.
    return jax.default_backend() == 'gpu'


def _rate_trio_bytes(nints, plane_shape, dtype):
    """Calculate the storage needed for rate, uncertainty, and DQ arrays."""
    return (int(nints) * int(np.prod(plane_shape)) * (2 * np.dtype(dtype).itemsize + 4))


def _fits_device(nbytes, *, buffers=3, headroom=.7):
    """Check whether working buffers fit the available device memory."""
    return int(nbytes) * int(buffers) <= int(core.device_memory_bytes() * headroom)


def _open_join(shape, prototype, on_device):
    """Allocate the visit-length buffers that the segment results fill."""
    if on_device:
        return {key: jnp.empty(shape, dtype=getattr(prototype, key).dtype)
                for key in ('data', 'err', 'dq')}
    return {key: core.empty_host_array(shape, getattr(prototype, key).dtype,
        name=f'exotedrf-stream-{key}') for key in ('data', 'err', 'dq')}


def _upload_rate_planes(part):
    """Return one segment's filled rate planes in accelerator memory."""
    cube = part.cube
    return PipelineState(dataclasses.replace(
            cube, data=jnp.asarray(cube.data), err=jnp.asarray(cube.err), dq=jnp.asarray(cube.dq)),
        part.aux)


def _stream_segments(pipeline, state, params, *, join_index, prefix_stop,
                     shared_aux, per_segment, store, logger, observer, started):
    """Run steps ``[0, join_index)`` one original FITS segment at a time."""
    ctx = pipeline.ctx
    cube = state.cube
    slices = core.segment_slices(cube.meta, cube.data.shape[0])
    log = _logger(logger)
    names = [step.name for step in pipeline.steps]
    ramp_stop = names.index('RampFitStep') + 1
    if ramp_stop > join_index:
        raise ValueError('segment streaming requires RampFitStep before the first joined step')
    head_stop = prefix_stop if per_segment is not None else ramp_stop
    resident = _device_residency_allowed()
    nints = int(cube.data.shape[0])
    plane_shape = cube.data.shape[-2:]
    longest = max(seg.stop - seg.start for seg in slices)
    tail_on_device = resident and _fits_device(
        _rate_trio_bytes(longest, plane_shape, cube.data.dtype))
    join_on_device = resident and _fits_device(
        _rate_trio_bytes(nints, plane_shape, cube.data.dtype))
    outputs = None
    final_aux = None
    # Collect each segment's 1/f scaling diagnostics.
    solve_key = stages.OOF_SOLVE_AUX_KEYS['grp']
    solve_parts = []

    def head(index, local, segment_observer):
        """Calibrate one segment through ramp fitting."""
        part = segment_state(state, index)
        if shared_aux:
            part = PipelineState(part.cube, dict(part.aux, **shared_aux))
        prefix = Pipeline(pipeline.steps[:head_stop], pipeline.mode, local)
        part = prefix.run(part, params, observer=segment_observer)
        if per_segment is None:
            return part
        part = per_segment(part, local, segment_observer)
        middle = Pipeline(pipeline.steps[prefix_stop + 1:ramp_stop], pipeline.mode, local)
        return middle.run(part, params, observer=segment_observer)

    def finish(entry):
        """Complete rate filling and add the segment to the joined observation."""
        nonlocal outputs, final_aux
        index, seg, part, local, segment_observer, seg_start = entry
        # Complete rate filling before GainScale and flat-field correction.
        part = stages.await_rampfit_fill(part)
        if tail_on_device:
            part = _upload_rate_planes(part)
        tail = Pipeline(pipeline.steps[ramp_stop:join_index], pipeline.mode, local)
        part = tail.run(part, params, observer=segment_observer)
        if outputs is None:
            outputs = _open_join((nints, *part.cube.data.shape[1:]), part.cube, join_on_device)
        for key in ('data', 'err', 'dq'):
            piece = getattr(part.cube, key)
            if join_on_device:
                # Update the donated device buffer without copying the complete visit.
                outputs[key] = core._device_place(outputs[key],
                    jnp.asarray(piece, dtype=outputs[key].dtype), seg.start, 0)
            else:
                # Copy completed segment rates into the host join buffers.
                outputs[key][seg] = core.to_host(piece)
        final_aux = part.aux
        if final_aux.get(solve_key) is not None:
            solve_parts.append(final_aux[solve_key])
        log(f'segment {index + 1}/{len(slices)} complete: '
            f'{time.perf_counter() - seg_start:.3f}s')
    # Keep one rate-fill task outstanding while the next segment runs.
    pool = (ThreadPoolExecutor(max_workers=1, thread_name_prefix='exotedrf-rate-fill')
            if len(slices) > 1 else None)
    try:
        waiting = []
        for index, seg in enumerate(slices):
            seg_start = time.perf_counter()
            local = segment_context(ctx, cube, seg)
            if pool is not None:
                local[stages._RAMPFIT_FILL_EXECUTOR_KEY] = pool
            segment_observer = (observer.for_segment(index, len(slices))
                                if observer is not None else None)
            entry = (index, seg, head(index, local, segment_observer),
                     local, segment_observer, seg_start)
            if waiting:
                finish(waiting.pop())
            if pool is None:
                finish(entry)
            else:
                waiting.append(entry)
            del entry
        while waiting:
            finish(waiting.pop())
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    # Release local references before the joined pipeline replaces its rate arrays.
    if len(solve_parts) > 1:
        final_aux = dict(final_aux, **{solve_key: stages._concat_solve_diagnostics(solve_parts)})
    solve_parts = None
    pending = [PipelineState(core.RateCube(meta=cube.meta, **outputs), final_aux)]
    outputs = None
    log(f'Stage 1 through the join complete: {time.perf_counter() - started:.3f}s')
    return pipeline.run(pending.pop(), params, start=join_index, store=store, observer=observer)


def run_segmented(pipeline, state, params, *, store=None, logger=None, observer=None):
    """Run the supported stage-1 graph for this mode segment by segment.

    Parameters
    ----------
    pipeline : Pipeline
        Built-in calibration graph and reduction context.
    state : PipelineState
        Observation arrays and accumulated calibration results.
    params : dict
        Calibration or extraction parameter values.
    store : None, CheckpointStore
        Store for retained pipeline checkpoints.
    logger, observer : None, callable
        Callbacks for progress messages and completed-step diagnostics, respectively.

    Returns
    -------
    state : PipelineState
        Segment state or joined calibrated state after the requested steps.
    """
    for prefix, run in (('NIRISS/SOSS', run_soss_segmented),
                        ('NIRSPEC', run_nirspec_segmented), ('MIRI', run_miri_segmented)):
        if pipeline.mode.upper().startswith(prefix):
            return run(pipeline, state, params, store=store, logger=logger, observer=observer)
    raise ValueError(f'segment streaming does not support {pipeline.mode}')


def run_soss_segmented(pipeline, state, params, *, store=None, logger=None, observer=None):
    """Run the built-in SOSS graph with one segment of detector ramps at a time.

    Requires the built-in SOSS graph with CRDS superbias and ramp fitting.

    Parameters
    ----------
    pipeline : Pipeline
        Built-in calibration graph and reduction context.
    state : PipelineState
        Observation arrays and accumulated calibration results.
    params : dict
        Calibration or extraction parameter values.
    store : None, CheckpointStore
        Store for retained pipeline checkpoints.
    logger, observer : None, callable
        Callbacks for progress messages and completed-step diagnostics, respectively.

    Returns
    -------
    state : PipelineState
        Segment state or joined calibrated state after the requested steps.
    """
    if pipeline.mode.upper() != 'NIRISS/SOSS':
        raise ValueError('run_soss_segmented requires the NIRISS/SOSS graph')
    ctx = pipeline.ctx
    if ctx['opts'].get('superbias_method', 'crds') != 'crds':
        raise ValueError('segment streaming requires CRDS superbias')
    names = [step.name for step in pipeline.steps]
    join_index = _join_index(names)
    _check_graph(pipeline, stages.soss_steps(ctx['opts']), names, store, join_index)
    background_index = (names.index('BackgroundStep_grp')
                        if 'BackgroundStep_grp' in names else None)
    cube = state.cube
    slices = core.segment_slices(cube.meta, cube.data.shape[0])
    started = time.perf_counter()
    log = _logger(logger)
    prefix_stop = join_index if background_index is None else background_index
    per_segment = None
    if background_index is not None:
        baseline = stages.baseline_bool_for_meta(cube.meta, cube.data.shape[0])
        if not baseline.any():
            raise ValueError('baseline_ints selects no integrations')
        # Collect baseline integrations after the independent detector corrections.
        selected = core.empty_host_array(
            (int(baseline.sum()), *cube.data.shape[1:]), cube.data.dtype,
            name='exotedrf-stream-baseline')
        offset = 0
        for index, seg in enumerate(slices):
            keep = baseline[seg]
            if not keep.any():
                continue
            part = segment_state(state, index)
            local = segment_context(ctx, cube, seg)
            prefix = Pipeline(pipeline.steps[:background_index], pipeline.mode, local)
            part = prefix.run(part, params)
            count = int(keep.sum())
            # Gather selected baseline ramps in bounded integration blocks.
            indices = np.flatnonzero(keep)
            plane_bytes = int(np.prod(cube.data.shape[1:])) * cube.data.dtype.itemsize
            batch = max(1, (32 << 20) // max(1, plane_bytes))
            for first in range(0, count, batch):
                group = indices[first:first + batch]
                selected[offset + first:offset + first + len(group)] = \
                    core.to_host(part.cube.data[group])
            offset += count
            del part
        mask = np.ones(selected.shape[0], dtype=bool)
        deep = stages._host_deepstack(selected, mask)
        model = np.asarray(ctx['background_model'], dtype=cube.data.dtype)
        region = stages._background_region(*cube.data.shape[-2:])
        calls = stages.soss_background_calls(
            ctx.get('opts', {}), 'soss_background_grp', cube.data.shape[1],
            cube.data.shape[-2], region)
        background = np.stack([stages._scaled_background_model(selected[:, g], deep[g], model,
                                            **calls[g]) for g in range(cube.data.shape[1])])
        # Subtract the background before taking the centroid median to preserve rounding.
        _, centroid_deep = stages.deepstack_pair(selected, mask, background)
        del selected, deep
        shared_aux = stages._attach_auto_centroids(
            dict(state.aux, bkg_grp=background), 'centroids_group', centroid_deep, cube, ctx)
        centroids = stages._centroids_for_stage(
            PipelineState(cube, shared_aux), ctx, 'centroids_group', deepstack=centroid_deep)
        log(f'baseline prepass complete: {time.perf_counter() - started:.3f}s')

        def per_segment(part, local, segment_observer):
            """Apply the prepared background and 1/f correction to a segment."""
            before = part
            data = core.map_over_ints(lambda d: d - jnp.asarray(background, dtype=d.dtype)[None],
                part.cube.data, part.cube.data.shape[0])
            part = PipelineState(dataclasses.replace(part.cube, data=data),
                                 dict(part.aux, **shared_aux))
            part.aux[stages._OOF_GRP_PREP_KEY] = stages._prepare_oneoverf_grp(
                part, local, deep=centroid_deep, centroids=centroids)
            if segment_observer is not None:
                segment_observer(pipeline.steps[background_index], before, part)
            return part
    return _stream_segments(pipeline, state, params, join_index=join_index,
        prefix_stop=prefix_stop, shared_aux=None, per_segment=per_segment,
        store=store, logger=logger, observer=observer, started=started)


def _nirspec_group_centroids(pipeline, state, params, oof_index, logger, started):
    """Reproduce ``OneOverFStep_grp``'s visit-wide NIRSpec trace exactly."""
    ctx = pipeline.ctx
    if ctx.get('centroids') is not None:
        return {}
    cube = state.cube
    slices = core.segment_slices(cube.meta, cube.data.shape[0])
    if len(slices) == 1:
        # Use the step's own deep stack when the segment spans the visit.
        return {}
    log = _logger(logger)
    planes = core.empty_host_array((cube.data.shape[0], *cube.data.shape[-2:]), cube.data.dtype,
        name='exotedrf-stream-nirspec-lastgroup')
    for index, seg in enumerate(slices):
        part = segment_state(state, index)
        local = segment_context(ctx, cube, seg)
        prefix = Pipeline(pipeline.steps[:oof_index], pipeline.mode, local)
        part = prefix.run(part, params)
        _gather_last_group(planes, seg, part.cube.data)
        del part
    centroids = stages._trace_nirspec_deepframe(stages._nirspec_deepstack_all_ints(planes), ctx)
    del planes
    log(f'NIRSpec 1/f trace prepass complete: {time.perf_counter() - started:.3f}s')
    return {'centroids_group': centroids}


def run_nirspec_segmented(pipeline, state, params, *, store=None, logger=None, observer=None):
    """Run the built-in NIRSpec graph one original FITS segment at a time.

    Prepare the visit-wide group trace before running independent segments with CRDS superbias.

    Parameters
    ----------
    pipeline : Pipeline
        Built-in calibration graph and reduction context.
    state : PipelineState
        Observation arrays and accumulated calibration results.
    params : dict
        Calibration or extraction parameter values.
    store : None, CheckpointStore
        Store for retained pipeline checkpoints.
    logger, observer : None, callable
        Callbacks for progress messages and completed-step diagnostics, respectively.

    Returns
    -------
    state : PipelineState
        Segment state or joined calibrated state after the requested steps.
    """
    if not pipeline.mode.upper().startswith('NIRSPEC'):
        raise ValueError('run_nirspec_segmented requires a NIRSpec graph')
    ctx = pipeline.ctx
    if str(ctx['opts'].get('superbias_method', 'crds')).lower() != 'crds':
        raise ValueError('segment streaming requires CRDS superbias: a custom NIRSpec '
            'superbias medians group 0 over the whole visit')
    names = [step.name for step in pipeline.steps]
    join_index = _join_index(names)
    _check_graph(pipeline, stages.nirspec_steps(ctx['opts']), names, store, join_index)
    started = time.perf_counter()
    shared_aux = {}
    if 'OneOverFStep_grp' in names:
        shared_aux = _nirspec_group_centroids(
            pipeline, state, params, names.index('OneOverFStep_grp'), logger, started)
    return _stream_segments(pipeline, state, params, join_index=join_index,
        prefix_stop=join_index, shared_aux=shared_aux, per_segment=None,
        store=store, logger=logger, observer=observer, started=started)


def run_miri_segmented(pipeline, state, params, *, store=None, logger=None, observer=None):
    """Run the built-in MIRI/LRS graph one original FITS segment at a time.

    Restart time-dependent detector corrections at the original FITS segment boundaries.

    Parameters
    ----------
    pipeline : Pipeline
        Built-in calibration graph and reduction context.
    state : PipelineState
        Observation arrays and accumulated calibration results.
    params : dict
        Calibration or extraction parameter values.
    store : None, CheckpointStore
        Store for retained pipeline checkpoints.
    logger, observer : None, callable
        Callbacks for progress messages and completed-step diagnostics, respectively.

    Returns
    -------
    state : PipelineState
        Segment state or joined calibrated state after the requested steps.
    """
    if not pipeline.mode.upper().startswith('MIRI'):
        raise ValueError('run_miri_segmented requires a MIRI graph')
    ctx = pipeline.ctx
    names = [step.name for step in pipeline.steps]
    join_index = _join_index(names)
    _check_graph(pipeline, stages.miri_steps(ctx['opts']), names, store, join_index)
    return _stream_segments(pipeline, state, params, join_index=join_index,
        prefix_stop=join_index, shared_aux={}, per_segment=None, store=store,
        logger=logger, observer=observer, started=time.perf_counter())
