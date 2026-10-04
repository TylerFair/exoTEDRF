"""Assemble detector calibration, spectral calibration, and extraction steps."""

import os
import sys
import time
import warnings
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter

from exotedrf.v2 import config as v2config
from exotedrf.v2 import core, hoststats, refs, trace, upramp
from exotedrf.v2.core import RampCube, RateCube
from exotedrf.v2.pipeline import PipelineState, Step
from exotedrf.v2.kernels import (
    background as k_bkg, badpix as k_badpix, cost as k_cost, detector as k_det, emicorr as k_emi,
    extract as k_ext, jump as k_jump, oneoverf as k_oof, pca as k_pca, ramp as k_ramp,
    reset as k_reset)


def _rp(ctx, key, default=None):
    """Get a reference value from the observation context."""
    rp = ctx.get('refpack')
    if rp is None:
        return default
    if hasattr(rp, 'get'):
        val = rp.get(key)
        return default if val is None else val
    return rp[key] if key in rp else default


def _require_reference(ctx, key, message):
    """Get a required reference or raise the step's missing-reference error."""
    value = _rp(ctx, key)
    if value is None:
        raise ValueError(message)
    return value


def format_out_frames_2(baseline_ints, exposure_nints):
    """Format baseline bounds for the v1 deepstack selection.

    A single positive bound returns [N, -1], selecting the complete exposure.

    Parameters
    ----------
    baseline_ints : array-like(int)
        Baseline bounds from the reduction options.
    exposure_nints : int
        Total number of integrations in the exposure.

    Returns
    -------
    out_frames : np.ndarray(int)
        Lower and upper bounds for the deepstack baseline.
    """
    out_frames = np.atleast_1d(np.asarray(baseline_ints, dtype=int))
    if out_frames.size == 1:
        if out_frames[0] > 0:
            return np.array([out_frames[0], -1])
        return np.array([0, int(exposure_nints) + out_frames[0]])
    if out_frames.size == 2:
        return np.array([out_frames[0], int(exposure_nints) + out_frames[-1]])
    raise ValueError('baseline_ints must have length 1 or 2.')


def baseline_bool(baseline_ints, nints, *, integration_numbers=None,
                  exposure_nints=None):
    """Select baseline integrations using exposure-level integration numbers.

    Parameters
    ----------
    baseline_ints : array-like(int)
        Baseline bounds from the reduction options.
    nints : int
        Number of loaded integrations.
    integration_numbers : None, array-like(int)
        Zero-based exposure integration numbers; defaults to the local array indices.
    exposure_nints : None, int
        Total exposure integrations; defaults to nints.

    Returns
    -------
    baseline_mask : np.ndarray(bool)
        Mask of integrations included in the deepstack baseline.
    """
    if integration_numbers is None:
        integration_numbers = np.arange(nints, dtype=int)
    else:
        integration_numbers = np.asarray(integration_numbers, dtype=int)
        if integration_numbers.shape != (nints,):
            raise ValueError('integration_numbers must match nints')
    if exposure_nints is None:
        exposure_nints = nints
    exposure_nints = int(exposure_nints)
    if exposure_nints < 1:
        raise ValueError('exposure_nints must be positive')

    low, high = format_out_frames_2(baseline_ints, exposure_nints)
    return ((integration_numbers < low) |
            (integration_numbers >= high))


def _exposure_nints(meta, integration_numbers):
    """Resolve exposure-level NINTS without mistaking a segment for it."""
    extra = getattr(meta, 'extra', {}) or {}
    candidates = [extra.get('exposure_nints')]
    header = extra.get('header')
    if hasattr(header, 'get'):
        candidates.append(header.get('NINTS'))
    for segment_header in extra.get('segment_headers', ()):
        if hasattr(segment_header, 'get'):
            candidates.append(segment_header.get('NINTS'))
    segment_ends = extra.get('segment_int_ends')
    if segment_ends is not None:
        candidates.extend(segment_ends)
    # Use exposure-level bounds when segment headers report local NINTS.
    candidates.append(int(np.max(integration_numbers) + 1))
    valid = [int(value) for value in candidates
             if value not in (None, '') and int(value) > 0]
    return max(valid)


def baseline_bool_for_meta(meta, nints):
    """Select baseline integrations from the loaded FITS segments.

    Parameters
    ----------
    meta : ObsMeta
        Observation metadata with baseline bounds and segment integration numbers.
    nints : int
        Number of loaded integrations.

    Returns
    -------
    baseline_mask : np.ndarray(bool)
        Mask of loaded integrations included in the baseline.
    """
    slices = core.segment_slices(meta, n_ints=nints)
    starts = np.asarray(meta.segment_int_starts, dtype=int)
    if starts.shape != (len(slices),):
        raise ValueError('segment INTSTART metadata does not match segments')
    integration_numbers = np.concatenate([
        np.arange(int(start) - 1,
                  int(start) - 1 + segment.stop - segment.start, dtype=int)
        for start, segment in zip(starts, slices)
    ])
    return baseline_bool(
        meta.baseline_ints, nints, integration_numbers=integration_numbers,
        exposure_nints=_exposure_nints(meta, integration_numbers))


def _finite_order2_centroids(ypos_o2):
    """Return finite order-2 centroids in extraction order."""
    ypos_o2 = np.asarray(ypos_o2)
    return ypos_o2[np.isfinite(ypos_o2)]


def _padded_optional_trace(ypos, dimx, dtype=None):
    """Pad an optional SOSS trace with NaNs to the detector width."""
    if ypos is None:
        return None
    ypos = np.asarray(ypos)
    target_dtype = np.dtype(dtype) if dtype is not None else ypos.dtype
    if not np.issubdtype(target_dtype, np.floating):
        target_dtype = np.dtype(np.float32)
    ypos = ypos.astype(target_dtype, copy=False)
    if ypos.ndim != 1 or ypos.size > dimx:
        raise ValueError(
            f'optional trace must be 1D with at most {dimx} entries, got '
            f'{ypos.shape}')
    padded = np.full(dimx, np.nan, dtype=target_dtype)
    padded[:ypos.size] = ypos
    return jnp.asarray(padded)


MIRI_TRACE_CENTER = 36
MIRI_PCA_COLUMNS = (12, 61)

_STEP_CONTROLS = (
    'DQInitStep', 'INLCorrStep', 'SuperBiasStep', 'RefPixStep',
    'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep', 'JumpStep',
    'RampFitStep', 'GainScaleStep', 'AssignWCSStep',
    'FlatFieldStep', 'BackgroundStep', 'OneOverFStep_int', 'BadPixStep',
    'PCAReconstructStep')


def _runs(opts, name):
    """Validate a step control and return whether the step runs."""
    value = opts.get(name, 'run')
    if value not in ('run', 'skip'):
        raise ValueError(
            f"{name} must be exactly lowercase 'run' or 'skip', got {value!r}")
    return value == 'run'


def _updated_state(state, *, aux=None, **arrays):
    """Replace calibrated cube arrays while retaining metadata and unchanged auxiliary data."""
    return PipelineState(cube=replace(state.cube, **arrays), aux=state.aux if aux is None else aux)


def _segment_slices(cube):
    """Return the integration slices for the original FITS segments."""
    return core.segment_slices(cube.meta, n_ints=cube.data.shape[0])


def _resident(*arrays):
    """Check whether all supplied arrays reside on the device."""
    return bool(arrays) and all(core.is_device_array(a) for a in arrays)


def _join_segments(parts, resident):
    """Join segment results on the host or device used by the inputs."""
    if len(parts) == 1:
        return parts[0]
    return (jnp.concatenate(parts, axis=0) if resident else
            np.concatenate(parts, axis=0))


def _store_host_segment(buffer, segment, value, chunk_bytes=32 << 20):
    """Copy one segment result into a host output buffer in bounded pieces."""
    if not core.is_device_array(value):
        buffer[segment] = value
        return
    nints = int(value.shape[0])
    per_int = max(1, int(value.nbytes) // max(1, nints))
    step = max(1, min(nints, int(chunk_bytes) // per_int))
    for lo in range(0, nints, step):
        hi = min(lo + step, nints)
        buffer[segment.start + lo:segment.start + hi] = np.asarray(
            jax.device_get(value[lo:hi]))


def _segment_buffer(shape, dtype, name, resident):
    """Allocate host storage or a list of device arrays for the segment results."""
    return [] if resident else core.empty_host_array(shape, dtype, name=name)


def _segment_store(buffer, segment, value):
    """Record one segment's result in a :func:`_segment_buffer` store."""
    if isinstance(buffer, list):
        buffer.append(value)
    else:
        buffer[segment] = value


def _segment_result(buffer, resident):
    """Close a :func:`_segment_buffer` store into one observation array."""
    return _join_segments(buffer, True) if resident else buffer


def _beside(array, resident):
    """Place a small host auxiliary array beside the cube it is mapped with."""
    if array is None or not resident:
        return array
    return jnp.asarray(array)


def _ramp_segment_readouts(cube):
    """Return per-file TGROUP/NFRAMES/one-read timing used by RampFit."""
    slices = _segment_slices(cube)
    extra = cube.meta.extra
    primary = extra.get('header', {})
    headers = extra.get('segment_headers')
    if headers is None:
        headers = (primary,) * len(slices)
    if len(headers) != len(slices):
        raise ValueError(
            'segment_headers must have one entry per input segment')
    readouts = []
    for header in headers:
        group_time = float(header.get('TGROUP', cube.meta.frame_time) or
                           cube.meta.frame_time)
        nframes = int(header.get('NFRAMES',
                                 primary.get('NFRAMES', 1)) or 1)
        if not np.isfinite(group_time) or group_time <= 0.:
            raise ValueError('RampFit requires a positive finite TGROUP')
        if nframes < 1:
            raise ValueError('RampFit requires NFRAMES >= 1')
        one_group_time = None
        if cube.data.shape[1] == 1:
            frame_time = header.get('TFRAME', primary.get('TFRAME'))
            if frame_time is None and not header and not primary:
                # Use group timing when no FITS readout header is supplied.
                frame_time = group_time
            if frame_time is None or not np.isfinite(frame_time) or \
                    float(frame_time) <= 0.:
                raise ValueError(
                    'NGROUPS=1 requires a positive TFRAME')
            one_group_time = (nframes + 1.) * float(frame_time) / 2.
        readouts.append((group_time, nframes, one_group_time))
    return slices, readouts


_RAMPFIT_DEFAULT_CHUNK_INTS = 16
_RAMPFIT_SPARSE_CHUNK_RAMPS = 8192
_RAMPFIT_SPARSE_MAX_RAMPS = 65536
_RAMPFIT_FILL_EXECUTOR_KEY = 'rampfit_fill_executor'
_RAMPFIT_FILL_FUTURE_KEY = 'rampfit_fill_future'


def _rampfit_budget_chunk_allowed():
    """Check whether device memory may increase the RampFit batch size."""
    policy = os.environ.get('EXOTEDRF_RAMPFIT_DEVICE_CHUNK', 'auto').lower()
    if policy in ('0', 'false', 'off'):
        return False
    if policy in ('1', 'true', 'on'):
        return True
    if policy != 'auto':
        raise ValueError('EXOTEDRF_RAMPFIT_DEVICE_CHUNK must be auto, '
                         'on/true/1, or off/false/0')
    return jax.default_backend() == 'gpu'


def _rampfit_integration_chunk_size(nints, bytes_per_int=None, *,
                                    resident=False):
    """Choose how many integrations RampFit can process safely at once."""
    nints = int(nints)
    if nints < 1:
        raise ValueError('RampFit segments must contain an integration')
    for name in ('EXOTEDRF_RAMPFIT_CHUNK_INTS', 'EXOTEDRF_CHUNK_INTS'):
        requested = os.environ.get(name)
        if requested is None:
            continue
        try:
            chunk_size = int(requested)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be a positive integer') from exc
        if chunk_size < 1:
            raise ValueError(f'{name} must be a positive integer')
        return min(chunk_size, nints)
    chunk_size = min(_RAMPFIT_DEFAULT_CHUNK_INTS, nints)
    if bytes_per_int is None:
        return chunk_size
    budgeted = core.auto_chunk(nints, bytes_per_int, n_buffers=32,
                               headroom=.5)
    if resident and _rampfit_budget_chunk_allowed():
        # Keep the default batch size as the lower bound for resident ramps.
        return min(nints, max(chunk_size, budgeted))
    return min(chunk_size, budgeted)


def _segment_median_rate(data, groupdq, group_time, one_group_time=None,
                         *, return_topology=False, suppress_one_group=True):
    """Measure the segment median rate and optional ramp structure."""
    nints = data.shape[0]
    bytes_per_int = int(np.prod(data.shape[1:])) * (
        np.dtype(data.dtype).itemsize + np.dtype(groupdq.dtype).itemsize)
    chunk_size = _rampfit_integration_chunk_size(
        nints, bytes_per_int, resident=core.is_device_array(data))

    # Accumulate in float64 to preserve the result across batch sizes.
    total = np.zeros(data.shape[2:], dtype=np.float64)
    dtype = np.dtype(data.dtype)
    gt = jnp.asarray(group_time, dtype=dtype)
    invalid_nonfinite = False
    max_segments = 1
    exceptions_out = (core.empty_host_array(
        data.shape[:1] + data.shape[2:], bool,
        name='exotedrf-rampfit-topology') if return_topology else None)
    kw = {} if suppress_one_group else {'suppress_one_group': False}
    for lo in range(0, nints, chunk_size):
        hi = min(lo + chunk_size, nints)
        args = (
            jnp.asarray(data[lo:hi]), jnp.asarray(groupdq[lo:hi]), gt,
            None if one_group_time is None else
            jnp.asarray(one_group_time, dtype=dtype))
        if return_topology:
            med, invalid, chunk_max, exceptions = jax.device_get(
                k_ramp.rampfit_chunk_statistics(*args, **kw))
            invalid_nonfinite |= bool(np.asarray(invalid))
            max_segments = max(max_segments, int(np.asarray(chunk_max)))
            exceptions_out[lo:hi] = np.asarray(exceptions, dtype=bool)
        else:
            med = jax.device_get(
                k_ramp.median_rates_per_integration(*args, **kw))
        total += np.asarray(med, dtype=np.float64).sum(axis=0)
    result = jnp.asarray(total / nints, dtype=dtype)
    if not return_topology:
        return result
    if invalid_nonfinite:
        raise ValueError(
            'RampFit found non-finite SCI without DO_NOT_USE/SATURATED '
            'GROUPDQ; flag upstream non-finite samples before OLS_C')
    return result, max_segments, exceptions_out


def _use_sparse_rampfit(exception_count, total_ramps):
    """Choose whether rare multi-piece ramps should be refit as a sparse set."""
    mode = os.environ.get('EXOTEDRF_RAMPFIT_SPARSE', 'auto').strip().lower()
    if mode in ('0', 'false', 'off'):
        return False
    if mode in ('1', 'true', 'on'):
        return True
    if mode != 'auto':
        raise ValueError('EXOTEDRF_RAMPFIT_SPARSE must be auto, on/true/1, or ' 'off/false/0')
    count = int(exception_count)
    total = int(total_ramps)
    limit = min(max(1, total // 100), _RAMPFIT_SPARSE_MAX_RAMPS)
    return count <= limit


def _writable_host(value, name):
    """Return an updatable host copy of a result produced anywhere."""
    array = np.asarray(core.to_host(value))
    if array.flags.writeable:
        return array
    return core.copy_host_array(array, name=name)


def _gather_detector_values(reference, yy, xx, shape, dtype):
    """Match gain/read-noise reference values to a selected set of ramp pixels."""
    array = np.asarray(jax.device_get(reference), dtype=dtype)
    if array.ndim == 0:
        return np.full(yy.shape, array.item(), dtype=dtype)
    try:
        detector = np.broadcast_to(array, shape)
    except ValueError as exc:
        raise ValueError(
            f'RampFit reference shape {array.shape} cannot broadcast to '
            f'detector shape {shape}') from exc
    return np.asarray(detector[yy, xx], dtype=dtype)


def _apply_sparse_rampfit(rate, err, dq, data, groupdq, readnoise, gain, pixeldq,
        median_rate, exception_mask, group_time, nframes,
        average_dark_current, one_group_time, max_segments,
        chunk_size=_RAMPFIT_SPARSE_CHUNK_RAMPS, suppress_one_group=True):
    """Replace only ramps having a useful later segment with generic fits."""
    exception_mask = np.asarray(exception_mask, dtype=bool)
    if exception_mask.shape != data.shape[:1] + data.shape[2:]:
        raise ValueError('RampFit exception mask has the wrong shape')
    ii, yy, xx = np.nonzero(exception_mask)
    if ii.size == 0:
        return rate, err, dq
    if chunk_size < 1:
        raise ValueError('sparse RampFit chunk_size must be positive')

    # Transfer only the selected ramps from a resident segment.
    resident = core.is_device_array(data)
    host_data = None if resident else np.asarray(data)
    host_groupdq = (None if resident else np.asarray(groupdq, dtype=np.uint8))
    detector_shape = tuple(data.shape[2:])
    dtype = np.dtype(data.dtype)
    values = [_gather_detector_values(ref, yy, xx, detector_shape, kind)
              for ref, kind in ((readnoise, dtype), (gain, dtype), (median_rate, dtype),
                                (average_dark_current, dtype), (pixeldq, np.uint32))]

    rate = _writable_host(rate, 'exotedrf-sparse-rate')
    err = _writable_host(err, 'exotedrf-sparse-error')
    dq = _writable_host(dq, 'exotedrf-sparse-dq')
    ngroups = int(data.shape[1])
    pack = int(chunk_size)
    gt = jnp.asarray(group_time, dtype=dtype)
    nf = jnp.asarray(nframes, dtype=dtype)
    one_time = (None if one_group_time is None else jnp.asarray(one_group_time, dtype=dtype))
    kw = {} if suppress_one_group else {'suppress_one_group': False}

    for lo in range(0, ii.size, pack):
        hi = min(lo + pack, ii.size)
        count = hi - lo
        bi, by, bx = ii[lo:hi], yy[lo:hi], xx[lo:hi]

        packed_data = np.zeros((pack, ngroups, 1, 1), dtype=dtype)
        packed_groupdq = np.full((pack, ngroups, 1, 1), core.DQ_DO_NOT_USE, dtype=np.uint8)
        references = [np.full((pack, 1, 1), fill, dtype=kind)
                      for fill, kind in ((1, dtype), (1, dtype), (0, dtype),
                                         (0, dtype), (0, np.uint32))]

        if resident:
            ramps = (jnp.asarray(bi), slice(None), jnp.asarray(by), jnp.asarray(bx))
            ramp_values = np.asarray(jax.device_get(data[ramps]))
            ramp_groupdq = np.asarray(jax.device_get(groupdq[ramps]), dtype=np.uint8)
        else:
            ramp_values = host_data[bi, :, by, bx]
            ramp_groupdq = host_groupdq[bi, :, by, bx]
        packed_data[:count, :, 0, 0] = ramp_values
        packed_groupdq[:count, :, 0, 0] = ramp_groupdq
        for ref, value in zip(references, values):
            ref[:count, 0, 0] = value[lo:hi]
        packed_rn, packed_gain, packed_median, packed_dark, packed_pixeldq = map(jnp.asarray,
                                                                            references)
        fitted = k_ramp.fit_ramps_packed_stage(
            jnp.asarray(packed_data), jnp.asarray(packed_groupdq), packed_rn, packed_gain, gt,
            packed_pixeldq, median_rate=packed_median, nframes=nf,
            average_dark_current=packed_dark, one_group_time=one_time,
            max_segments=max_segments, **kw)
        for out, fitted_values in zip((rate, err, dq), jax.device_get(fitted)):
            out[bi, by, bx] = np.asarray(fitted_values)[:count, 0, 0]
    return rate, err, dq


def _nearest_fill_workers():
    """Choose the thread count for per-integration nearest-neighbor selection."""
    override = os.environ.get('EXOTEDRF_FILL_WORKERS')
    if override:
        return max(1, int(override))
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    return max(1, min(8, cpus))


def _nearest_selection(finite_mask):
    """Select the nearest finite source index for each invalid pixel."""
    from scipy.interpolate import griddata
    finite = np.where(finite_mask)
    invalid = np.where(~finite_mask)
    source_indices = np.flatnonzero(finite_mask)
    selected = griddata(finite, source_indices, invalid, method='nearest')
    return invalid, np.asarray(selected, dtype=np.intp)


def _nearest_fill_rate_planes(rate, *, copy=True):
    """Fill invalid rate pixels with the nearest finite value, matching v1 griddata."""
    from concurrent.futures import ThreadPoolExecutor

    if copy:
        rate = core.copy_host_array(rate, name='exotedrf-rate-fill')
    elif not isinstance(rate, np.ndarray) or not rate.flags.writeable:
        raise ValueError('in-place nearest fill needs writable host rates')
    nints = rate.shape[0]
    workers = _nearest_fill_workers()
    batch = max(workers * 16, 64)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, nints, batch):
            stop = min(start + batch, nints)
            masks = {}
            order = []
            for integration in range(start, stop):
                plane = rate[integration]
                finite_mask = np.isfinite(plane)
                # Keep an entirely invalid plane as NaNs, matching griddata.
                if not finite_mask.any():
                    plane.fill(np.nan)
                    continue
                if finite_mask.all():
                    continue
                # Use the complete packed mask as the selection-cache key.
                key = np.packbits(finite_mask, bitorder='little').tobytes()
                if key not in masks:
                    masks[key] = finite_mask
                order.append((integration, key))
            if not order:
                continue
            keys = list(masks)
            selections = dict(zip(keys, pool.map(
                _nearest_selection, (masks[k] for k in keys))))
            for integration, key in order:
                invalid, selected = selections[key]
                plane = rate[integration]
                plane[invalid] = plane.reshape(-1)[selected]
            del masks, selections
    return rate


def _reset_max_int(meta):
    """Return the reset-artifact extent for the instrument, grating, and detector."""
    instrument = meta.mode.split('/')[0].upper()
    if instrument != 'NIRSPEC':
        return None
    grating = ''
    if '/' in meta.mode:
        grating = meta.mode.split('/', 1)[1].upper()
    if not grating:
        header = (getattr(meta, 'extra', {}) or {}).get('header', {})
        grating = str(header.get('GRATING', '') or '').upper()
    if not grating:
        raise ValueError(
            'NIRSpec reset-artifact masking requires the grating (from '
            'observing_mode NIRSpec/<grating> or the FITS GRATING header)')
    return k_jump.default_max_reset_int(
        'NIRSPEC', grating, str(meta.detector))


def _timeseries_region(dimy, dimx):
    """Select the SOSS light-curve region or the complete frame for small arrays."""
    return ((20, 60, 1500, 1550) if dimy >= 60 and dimx >= 1550 else
            (0, dimy, 0, dimx))


def _background_region(dimy, dimx):
    """Select a background region for arrays smaller than the SOSS subarray."""
    if (dimy == 96 and dimx >= 401) or (dimy >= 250 and dimx >= 550):
        return None
    y0 = max(0, dimy // 2)
    x0 = min(5, max(0, dimx - 1))
    return (y0, dimy, x0, dimx)


def _host_deepstack(data, baseline_mask):
    """Calculate the baseline median in bounded blocks, preserving the science dtype."""
    baseline_mask = np.asarray(baseline_mask, dtype=bool)
    if baseline_mask.shape != (data.shape[0],):
        raise ValueError('baseline mask does not match the integration axis')
    if not baseline_mask.any():
        raise ValueError('baseline_ints selects no integrations')
    # Inspect the array shape without materializing a lazy ramp.
    reduction_input = (SimpleNamespace(nbytes=int(data.shape[0]) * int(np.prod(data.shape[-2:])) *
        np.dtype(data.dtype).itemsize) if data.ndim == 4 else data)
    if _device_fast_path_fits(reduction_input, buffers=8):
        mask = jnp.asarray(baseline_mask)
        if data.ndim == 4:
            groups = [np.asarray(jax.device_get(
                k_oof.baseline_deepstack(jnp.asarray(data[:, group]), mask)))
                for group in range(data.shape[1])]
            return np.stack(groups, axis=0)
        return np.asarray(jax.device_get(k_oof.baseline_deepstack(jnp.asarray(data), mask)))
    width = _device_deepstack_column_width(data)
    if width is not None:
        # Reduce independent detector columns in bounded device batches.
        return _device_deepstack_columns(data, baseline_mask, width)
    # Bound the baseline gather and median workspace on the host.
    budget = max(1, min(128 << 20, core.host_allocation_budget_bytes() // 8))
    selected = np.flatnonzero(baseline_mask)
    dimy, dimx = data.shape[-2:]
    per_column = max(1, len(selected) * dimy * np.dtype(data.dtype).itemsize)
    width = max(1, min(dimx, budget // (2 * per_column)))
    result = np.empty(data.shape[1:], dtype=data.dtype)
    groups = range(data.shape[1]) if data.ndim == 4 else (None,)
    for group in groups:
        for start in range(0, dimx, width):
            stop = min(start + width, dimx)
            prefix = () if group is None else (group,)
            part = np.asarray(data[(selected, *prefix, slice(None), slice(start, stop))]).copy()
            result[(*prefix, slice(None), slice(start, stop))] = np.nanmedian(
                part, axis=0, overwrite_input=True)
    return result


def _device_deepstack_column_width(data, *, buffers=8, headroom=0.70):
    """Return the detector-column batch size for a device median, or None."""
    policy = os.environ.get('EXOTEDRF_DEVICE_FAST_PATH', 'auto').lower()
    if policy in ('0', 'false', 'off', 'stream', 'streamed'):
        return None
    if jax.default_backend() != 'gpu' and policy not in (
            '1', 'true', 'on', 'device', 'resident'):
        return None
    dimy, dimx = data.shape[-2:]
    per_column = (int(data.shape[0]) * int(dimy) *
                  np.dtype(data.dtype).itemsize * int(buffers))
    budget = int(core.device_memory_bytes() * headroom)
    width = budget // max(per_column, 1)
    if width < 1:
        return None
    return int(min(dimx, width))


def deepstack_pair(data, baseline_mask, subtract):
    """Calculate baseline medians before and after background subtraction.

    Parameters
    ----------
    data : array-like(float)
        Science cube with integration, optional group, and detector axes.
    baseline_mask : array-like(bool)
        Mask of integrations included in the baseline.
    subtract : array-like(float)
        Background matching one detector image per group.

    Returns
    -------
    plain : np.ndarray(float)
        Baseline median of the input science cube.
    corrected : np.ndarray(float)
        Baseline median after background subtraction.
    """
    baseline_mask = np.asarray(baseline_mask, dtype=bool)
    if baseline_mask.shape != (data.shape[0],):
        raise ValueError('baseline mask does not match the integration axis')
    if not baseline_mask.any():
        raise ValueError('baseline_ints selects no integrations')
    subtract = np.asarray(subtract, dtype=data.dtype)
    if subtract.shape != tuple(data.shape[1:]):
        raise ValueError('subtract must match one detector image per group')
    width = _device_deepstack_column_width(data)
    dimx = int(data.shape[-1])
    plain = np.empty(data.shape[1:], dtype=data.dtype)
    corrected = np.empty(data.shape[1:], dtype=data.dtype)
    groups = range(data.shape[1]) if data.ndim == 4 else (None,)
    resident = core.is_device_array(data)
    if width is not None:
        mask = jnp.asarray(baseline_mask)
        for group in groups:
            sub_plane = jnp.asarray(subtract if group is None else subtract[group])
            for start in range(0, dimx, width):
                stop = min(start + width, dimx)
                prefix = () if group is None else (group,)
                index = (*prefix, slice(None), slice(start, stop))
                block = data[(slice(None), *index)]
                block = (block if resident else jnp.asarray(np.ascontiguousarray(block)))
                a = np.asarray(jax.device_get(k_oof.baseline_deepstack(block, mask)))
                b = np.asarray(jax.device_get(k_oof.baseline_deepstack(
                    block - sub_plane[:, start:stop][None], mask)))
                plain[index], corrected[index] = a, b
                del block
        return plain, corrected
    selected = np.flatnonzero(baseline_mask)
    for group in groups:
        sub_plane = subtract if group is None else subtract[group]
        prefix = () if group is None else (group,)
        part = np.asarray(core.to_host(data[(selected, *prefix)]))
        plain[prefix] = np.nanmedian(part, axis=0)
        corrected[prefix] = np.nanmedian(part - sub_plane[None], axis=0)
    return plain, corrected


def _device_deepstack_columns(data, baseline_mask, width):
    """Stream column blocks of every integration through the GPU median."""
    mask = jnp.asarray(np.asarray(baseline_mask, dtype=bool))
    resident = core.is_device_array(data)
    dimx = int(data.shape[-1])
    result = np.empty(data.shape[1:], dtype=data.dtype)
    groups = range(data.shape[1]) if data.ndim == 4 else (None,)
    for group in groups:
        for start in range(0, dimx, width):
            stop = min(start + width, dimx)
            prefix = () if group is None else (group,)
            index = (*prefix, slice(None), slice(start, stop))
            block = data[(slice(None), *index)]
            block = block if resident else jnp.asarray(np.ascontiguousarray(block))
            result[index] = np.asarray(jax.device_get(k_oof.baseline_deepstack(block, mask)))
            del block
    return result


def _device_fast_path_fits(array, *, buffers, headroom=0.70):
    """Check whether a complete observation and known temporary arrays fit on GPU."""
    policy = os.environ.get('EXOTEDRF_DEVICE_FAST_PATH', 'auto').lower()
    if policy in ('0', 'false', 'off', 'stream', 'streamed'):
        return False
    forced = policy in ('1', 'true', 'on', 'device', 'resident')
    if policy not in ('auto', '0', 'false', 'off', 'stream', 'streamed',
                      '1', 'true', 'on', 'device', 'resident'):
        raise ValueError(
            'EXOTEDRF_DEVICE_FAST_PATH must be auto, true/device, or '
            'false/streamed')
    if not forced and jax.default_backend() != 'gpu':
        return False
    required = int(getattr(array, 'nbytes', 0)) * int(buffers)
    return forced or required <= int(core.device_memory_bytes() * headroom)


def _badpix_column_chunk_size(data, box_size, *, halo=0):
    """Choose a detector-column batch size for bad-pixel neighborhood medians."""
    nints, dimy, ncols = (int(value) for value in data.shape)
    specific = os.environ.get('EXOTEDRF_V2_BADPIX_COLUMN_CHUNK')
    generic = os.environ.get('EXOTEDRF_CHUNK_COLS')
    configured = specific if specific is not None else generic
    if configured is not None:
        width = int(configured)
        if width < 1:
            name = ('EXOTEDRF_V2_BADPIX_COLUMN_CHUNK'
                    if specific is not None else 'EXOTEDRF_CHUNK_COLS')
            raise ValueError(f'{name} must be positive')
        return min(ncols, width)

    if jax.default_backend() != 'gpu':
        return ncols

    # Allow for the centre-row neighbors and two complete neighboring rows.
    samples = 6 * int(box_size) + 2
    plane_bytes_per_col = nints * dimy * int(np.dtype(data.dtype).itemsize)
    # Reserve space for sorting, temporal correction, and extraction.
    workspace_per_col = plane_bytes_per_col * (3 * samples + 20)
    budget = int(core.device_memory_bytes() * 0.45)
    read_width = max(1, budget // max(workspace_per_col, 1))
    core_width = max(1, read_width - 2 * int(halo))
    return min(ncols, 256, core_width)


def _trace_soss_deepstack(deepstack, cube, ctx):
    """Trace the SOSS baseline stack with the v1 edgetrigger method."""
    centroids = trace.get_soss_centroids_with_group_fallback(
        np.asarray(deepstack),
        tracetable=ctx.get('centroid_tracetable'),
        subarray=cube.meta.subarray)
    return trace.validate_centroids(centroids, cube.data.shape[-1])


def _stage_centroids(state, ctx, aux_key, deepframe, trace_frame, build_deepframe):
    """Prefer explicit or cached centroids before tracing a processed deepframe."""
    fixed = ctx.get('centroids')
    if fixed is not None:
        return fixed
    cached = state.aux.get(aux_key)
    if cached is not None:
        return cached
    if deepframe is None:
        deepframe = build_deepframe()
    return trace_frame(deepframe)


def _extraction_centroids(state, ctx, mode, instrument):
    """Select centroids using the instrument's intermediate and final trace lifecycle."""
    fixed = ctx.get('centroids')
    if fixed is not None:
        return fixed
    cube = state.cube
    if mode == 'sum':
        cached = state.aux.get('centroids_extract')
        if cached is not None:
            return cached
        deep = _stage3_deepframe(state, ctx)
        if deep is not None:
            if instrument == 'NIRSPEC':
                return _trace_nirspec_deepframe(deep, ctx)
            if instrument == 'MIRI':
                return _trace_miri_deepframe(deep, ctx, stage3=True)
            return _trace_soss_deepstack(deep, cube, ctx)
    if instrument == 'MIRI':
        cached = state.aux.get('centroids_extract')
        return (cached if cached is not None else
                _miri_centroids_for_stage(state, ctx, 'centroids_extract'))
    keys = (('centroids_group',) if mode != 'sum' and cube.data.ndim == 4 else
            ('centroids_int', 'centroids_group'))
    for key in keys:
        value = state.aux.get(key)
        if value is not None:
            return value
    if instrument == 'NIRSPEC':
        return _nirspec_centroids_for_stage(state, ctx, keys[0])
    return _centroids_for_stage(state, ctx, 'centroids_extract' if mode == 'sum' else keys[0])


def _centroids_for_stage(state, ctx, aux_key, deepstack=None):
    """Get explicit centroids or trace the processed baseline stack."""
    cube = state.cube
    return _stage_centroids(
        state, ctx, aux_key, deepstack, lambda deep: _trace_soss_deepstack(deep, cube, ctx),
        lambda: _host_deepstack(cube.data, baseline_bool_for_meta(cube.meta, cube.data.shape[0])))


def _attach_auto_centroids(aux, aux_key, deepstack, cube, ctx):
    """Attach automatic centroids to a copy of the auxiliary data."""
    aux = dict(aux)
    automatic = (ctx.get('centroids_source') == 'automatic-v1-lifecycle' or
                 ('centroids' in ctx and ctx.get('centroids') is None))
    if automatic:
        aux[aux_key] = _trace_soss_deepstack(deepstack, cube, ctx)
    return aux


def _centroids_for_extraction(state, ctx, mode):
    """Select centroids for intermediate or final SOSS extraction."""
    instrument = state.cube.meta.mode.split('/')[0].upper()
    if instrument == 'NIRSPEC':
        return _nirspec_centroids_for_extraction(state, ctx, mode)
    if instrument == 'MIRI':
        return _miri_centroids_for_extraction(state, ctx, mode)
    return _extraction_centroids(state, ctx, mode, 'NIRISS')


def _estimate_segment_timeseries(data, deep, meta, smoothing_scale=None):
    """Estimate the white light curve independently for each FITS segment."""
    parts = []
    region = _timeseries_region(data.shape[-2], data.shape[-1])
    y0, y1, x0, x1 = region
    for seg in core.segment_slices(meta, n_ints=data.shape[0]):
        nseg = seg.stop - seg.start
        if data.ndim == 4:
            postage = data[seg, -1, y0:y1, x0:x1]
            zero_point = deep[-1, y0:y1, x0:x1]
        else:
            postage = data[seg, y0:y1, x0:x1]
            zero_point = deep[y0:y1, x0:x1]
        size = (k_oof.default_smoothing_scale(nseg) if smoothing_scale is None
                else int(smoothing_scale))
        parts.append(np.asarray(k_oof.estimate_timeseries(
            postage, zero_point, size,
            region=(0, y1 - y0, 0, x1 - x0))))
    return np.concatenate(parts, axis=0)


def _segment_temporal_outliers(data, meta):
    """Find temporal outliers independently for each FITS segment."""
    out = core.empty_host_array(data.shape, bool,
                                name='exotedrf-temporal-outliers')
    for seg in core.segment_slices(meta, n_ints=data.shape[0]):
        part = data[seg]
        if _device_fast_path_fits(part, buffers=10):
            out[seg] = np.asarray(jax.device_get(
                k_oof.flag_temporal_outliers(jnp.asarray(part))))
        else:
            out[seg] = core.map_over_cols(
                lambda d: k_oof.flag_temporal_outliers(d), part,
                part.shape[-1])
    return out


def _oneoverf_base_mask(cube, ctx, *, reset=False):
    """Build the combined DQ and outlier mask in bounded blocks."""
    data = cube.data
    is_ramp = data.ndim == 4
    source = data[:, -1] if is_ramp else data
    result = _segment_temporal_outliers(source, cube.meta)
    custom = ctx.get('outlier_mask')
    order0 = ctx.get('order0_mask')
    pixeldq = np.asarray(cube.pixeldq) != 0 if is_ramp else None
    for first in range(0, data.shape[0], 32):
        sl = slice(first, min(first + 32, data.shape[0]))
        dq = cube.groupdq[sl, -1] if is_ramp else cube.dq[sl]
        result[sl] |= np.asarray(dq) != 0
        if pixeldq is not None:
            result[sl] |= pixeldq[None]
        if custom is not None:
            result[sl] |= np.asarray(
                custom if custom.ndim == 2 else custom[sl], dtype=bool)
        if order0 is not None:
            result[sl] |= np.asarray(order0, dtype=bool)[None]
    if reset:
        instrument = cube.meta.mode.split('/')[0].upper()
        limit = _reset_max_int(cube.meta)
        if limit is None:
            limit = k_jump.default_max_reset_int(instrument)
        for seg, first in zip(_segment_slices(cube),
                              cube.meta.segment_int_starts):
            if first >= limit:
                continue
            # Include the terminal unmasked integration in the reset prefix.
            count = min(seg.stop - seg.start, limit - int(first) + 1)
            artifact = k_jump.reset_artifact_mask(
                count, *data.shape[-2:], int_start=int(first),
                instrument=instrument, max_reset_int=limit)
            result[seg.start:seg.start + count] |= artifact
    return result


def _build_soss_masks_chunked(base_mask, o1, o2, o3, inner_width,
                              outer_width, method):
    """Build time-dependent masks without placing their full cube on GPU."""
    if _device_fast_path_fits(base_mask, buffers=6):
        result = k_oof.build_soss_masks(
            jnp.asarray(base_mask), o1, o2, o3, inner_width, outer_width,
            method=method)
        return tuple(np.asarray(jax.device_get(value)) for value in result)
    seed = k_oof.build_soss_masks(
        jnp.asarray(base_mask[:1]), o1, o2, o3, inner_width, outer_width,
        method=method)
    tr1, tr2 = np.asarray(seed[2]), np.asarray(seed[3])

    def build(mask):
        """Build the SOSS masks for one integration batch."""
        result = k_oof.build_soss_masks(
            mask, o1, o2, o3, inner_width, outer_width, method=method)
        return result[0], result[1]

    out1, out2 = core.map_over_ints(
        build, base_mask, base_mask.shape[0])
    return out1, out2, tr1, tr2


_OOF_GRP_PREP_KEY = '_oneoverf_grp_prepared'
_OOF_INT_DEEP_KEY = '_oneoverf_int_deepstack'


def _prepare_oneoverf_grp(state, ctx, *, deep=None, centroids=None):
    """Prepare the SOSS baseline, masks, centroids, and light curve for group 1/f."""
    cached = state.aux.get(_OOF_GRP_PREP_KEY)
    if cached is not None:
        return cached
    cube = state.cube
    data = cube.data
    nints, _ngroups, _dimy, dimx = data.shape
    bmask = baseline_bool_for_meta(cube.meta, nints)
    if deep is None:
        deep = _host_deepstack(data, bmask)
    if centroids is None:
        centroids = _centroids_for_stage(state, ctx, 'centroids_group', deepstack=deep)

    # Omit the temporal mask and light curve for the solve method.
    solve = _is_oof_solve(ctx)
    base_mask = (None if solve else _oneoverf_base_mask(cube, ctx, reset=True))

    timeseries = ctx.get('soss_timeseries')
    if timeseries is None and not solve:
        timeseries = _estimate_segment_timeseries(data, deep, cube.meta,
            smoothing_scale=ctx['opts'].get('oof_smoothing_scale_grp') if 'opts' in ctx else None)
    elif timeseries is not None:
        timeseries = np.asarray(timeseries)
    dtype = np.dtype(data.dtype)
    optional = {}
    for order in (2, 3):
        value = _padded_optional_trace(centroids.get(f'ypos o{order}'), dimx, dtype=dtype)
        optional[order] = np.full(dimx, np.nan, dtype=dtype) if value is None else value
    return {'deep': np.asarray(deep), 'base_mask': (None if base_mask is None else
                      np.asarray(base_mask, dtype=bool)),
        'timeseries': None if timeseries is None else np.asarray(timeseries),
        'centroid_o1': np.asarray(np.nan_to_num(centroids['ypos o1']), dtype=dtype),
        'centroid_o2_padded': np.asarray(optional[2]),
        'centroid_o3_padded': np.asarray(optional[3]), 'centroid_o2_extract': (
            None if centroids.get('ypos o2') is None else
            np.asarray(_finite_order2_centroids(centroids['ypos o2']), dtype=dtype)),
        'background': (None if state.aux.get('bkg_grp') is None else
                       np.asarray(state.aux['bkg_grp'])),}


def _scaled_background_model(sample, deep, model, region, **options):
    """Evaluate SOSS model scaling from a one-integration device sample."""
    _, scaled, *_ = k_bkg.background_soss(
        jnp.asarray(sample[:1]), jnp.asarray(deep), jnp.asarray(model),
        region1=region, **options)
    return np.asarray(scaled)


def soss_background_calls(opts, key, ngroups, dimy, region):
    """Build the background-scaling arguments for each SOSS group.

    Parameters
    ----------
    opts : dict
        Reduction options and translated background-step arguments.
    key : str
        Background options key: soss_background_grp or soss_background_int.
    ngroups : int
        Number of groups to scale.
    dimy : int
        Number of detector rows.
    region : None, tuple[int]
        Default background-scaling region.

    Returns
    -------
    calls : list[dict]
        Background-scaling keyword arguments for each group.
    """
    spec = opts.get(key) or {}
    coords1 = spec.get('background_coords1')
    coords2 = spec.get('background_coords2')
    differential = bool(spec.get('differential', False))
    scales = {name: spec.get(name) for name in ('scale1', 'scale2')}
    for name, scale in scales.items():
        if scale is not None and len(scale) != ngroups:
            raise ValueError(f'BackgroundStep {name} needs one value per group: got '
                f'{len(scale)}, expected {ngroups} (v1 asserts this, ' 'stage2.py:1057-1064)')
    if differential and dimy != 256:
        raise NotImplementedError('differential SOSS background scaling requires SUBSTRIP256, as '
            'in v1 (stage2.py:1116, 1149)')
    calls = []
    for group in range(ngroups):
        call = {'region': region if coords1 is None else tuple(coords1)}
        if differential:
            call['differential'] = True
            if coords2 is not None:
                call['region2'] = tuple(coords2)
            if scales['scale2'] is not None:
                call['scale2'] = float(scales['scale2'][group])
        if scales['scale1'] is not None:
            call['scale1'] = float(scales['scale1'][group])
        calls.append(call)
    return calls


def _option_path(item, name, search_dirs):
    """Find an array-option file in the configured search directories."""
    candidate = os.path.expanduser(os.fspath(item))
    paths = [candidate] if os.path.isabs(candidate) else [
        os.path.join(directory, candidate) for directory in search_dirs]
    paths.append(candidate)
    path = next((path for path in paths if os.path.exists(path)), None)
    if path is None:
        raise FileNotFoundError(f'{name} file not found: {item}')
    return path


def _load_array_option(value, name, search_dirs=()):
    """Resolve a numeric array or one/many ``.npy`` paths once."""
    if value is None:
        return None

    def load_one(item):
        """Load and convert one array input."""
        if not isinstance(item, (str, os.PathLike)):
            return np.asarray(item)
        return np.load(_option_path(item, name, search_dirs), allow_pickle=False)

    if isinstance(value, (list, tuple)) and value and all(
            isinstance(v, (str, os.PathLike)) for v in value):
        return np.concatenate([np.asarray(load_one(v)) for v in value], axis=0)
    return np.asarray(load_one(value))


def _load_hot_pixel_map(value, shape, search_dirs=()):
    """Load the optional DQInit hot-pixel map on the host."""
    hot = _load_array_option(value, 'hot_pixel_map', search_dirs)
    if hot is None:
        return None
    hot = np.asarray(hot)
    if hot.shape != tuple(shape):
        raise ValueError(
            f'hot_pixel_map has shape {hot.shape}, expected {tuple(shape)}')
    return hot.astype(bool, copy=False)


def _load_outlier_maps(value, cube, search_dirs=()):
    """Load detector or integration masks and align them with the FITS segments."""
    if value is None:
        return None

    nints = int(cube.data.shape[0])
    spatial = tuple(cube.data.shape[-2:])
    segments = _segment_slices(cube)
    starts = tuple(int(v) for v in cube.meta.segment_int_starts)
    if len(starts) != len(segments):
        raise ValueError('outlier_maps cannot be aligned: segment INTSTART metadata '
            'does not match the input files')
    max_end = max(start + segment.stop - segment.start - 1
                  for start, segment in zip(starts, segments))

    def load_file(item):
        """Load an outlier-map file as a NumPy array or FITS image."""
        path = _option_path(item, 'outlier_maps', search_dirs)
        try:
            return np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            try:
                return fits.getdata(path)
            except (OSError, ValueError, IndexError, KeyError) as exc:
                raise ValueError(f'Cannot open outlier_maps file: {item}') from exc

    def load_one(item):
        """Load and convert one array input."""
        if isinstance(item, (str, os.PathLike)):
            array = load_file(item)
        else:
            array = np.asarray(item)
        if array.ndim not in (2, 3):
            raise ValueError('each outlier map must be 2D (y, x) or 3D (nint, y, x), '
                f'got {array.shape}')
        if tuple(array.shape[-2:]) != spatial:
            raise ValueError(f'outlier map has detector shape {array.shape[-2:]}, '
                f'expected {spatial}')
        return np.asarray(array).astype(bool, copy=False)

    def looks_like_segment_items(items):
        """Check whether every list item is a path or detector map."""
        if not items:
            return False
        for item in items:
            if isinstance(item, (str, os.PathLike)):
                continue
            try:
                ndim = np.asarray(item).ndim
            except (TypeError, ValueError):
                return False
            if ndim not in (2, 3):
                return False
        return True

    # Distinguish per-segment maps from nested numeric detector arrays.
    per_segment = None
    if isinstance(value, (list, tuple)):
        # Recognize complete numeric cubes before separating segment maps.
        try:
            numeric = np.asarray(value)
        except (TypeError, ValueError):
            numeric = np.asarray(value, dtype=object)
        is_complete_numeric = (numeric.dtype != object and (numeric.ndim == 2 or
             (numeric.ndim == 3 and (numeric.shape[0] == nints or numeric.shape[0] >= max_end))))
        if not is_complete_numeric and looks_like_segment_items(value):
            per_segment = list(value)

    if per_segment is not None and len(per_segment) > 1:
        if len(per_segment) != len(segments):
            raise ValueError('outlier_maps must contain one path/map or exactly one per '
                f'input segment ({len(segments)}), got {len(per_segment)}')
        parts = []
        for index, (item, segment) in enumerate(zip(per_segment, segments)):
            array = load_one(item)
            length = segment.stop - segment.start
            if array.ndim == 2:
                array = np.broadcast_to(array, (length, *spatial))
            elif array.shape[0] != length:
                raise ValueError(f'outlier_maps[{index}] has {array.shape[0]} '
                    f'integrations, expected {length} for that segment')
            parts.append(array)
        return np.concatenate(parts, axis=0).astype(bool, copy=False)

    item = per_segment[0] if per_segment is not None else value
    array = load_one(item)
    if array.ndim == 2:
        return np.broadcast_to(array, (nints, *spatial)).astype(bool, copy=False)
    if array.shape[0] == nints:
        return array

    # Select the loaded segments from an exposure-level mask cube.
    if array.shape[0] >= max_end:
        return np.concatenate([array[start - 1:start - 1 + segment.stop - segment.start]
            for start, segment in zip(starts, segments)], axis=0)

    # Reuse a single integration mask only for segments of matching length.
    lengths = [segment.stop - segment.start for segment in segments]
    if lengths and all(length == array.shape[0] for length in lengths):
        return np.concatenate([array] * len(segments), axis=0)
    raise ValueError(f'one 3D outlier map has {array.shape[0]} integrations; expected '
        f'concatenated nints={nints}, exposure coverage through '
        f'INTEND={max_end}, or one of each equal segment length {lengths}')


def _validate_timeseries(series, name, nints, dimx, require_2d=False,
                         max_int_end=None, dtype=None):
    """Validate the light-curve shape and convert it to the science dtype."""
    if series is None:
        return None
    series = np.asarray(series)
    if dtype is not None:
        series = series.astype(dtype, copy=False)
    elif not np.issubdtype(series.dtype, np.floating):
        series = series.astype(np.float32)
    if series.ndim not in (1, 2) or (series.ndim == 2 and
                                     series.shape[1] != dimx):
        raise ValueError(
            f'{name} must be 1D or have trailing detector width {dimx}, '
            f'got {series.shape}')
    # Accept local light curves or exposure-level arrays covering the loaded segments.
    enough_global = max_int_end is not None and series.shape[0] >= max_int_end
    if series.shape[0] != nints and not enough_global:
        raise ValueError(
            f'{name} leading length must be concatenated nints={nints} or '
            f'cover FITS INTEND={max_int_end}, got {series.shape[0]}')
    if require_2d and series.ndim != 2:
        raise ValueError(f'{name} must be 2D for scale-chromatic')
    if not np.isfinite(series).all():
        # Keep non-finite light-curve values so those pixels remain uncorrected.
        bad = int(np.size(series) - np.count_nonzero(np.isfinite(series)))
        warnings.warn(
            f'{name} contains {bad} non-finite value(s); as in v1, the '
            'affected integration/column 1/f levels are not corrected',
            UserWarning, stacklevel=2)
    return series


def _series_segment(series, cube, seg, int_start):
    """Select a segment from a local or exposure-level light curve."""
    series = np.asarray(series)
    if series.shape[0] == cube.data.shape[0]:
        return series[seg]
    lo = int(int_start) - 1
    hi = lo + (seg.stop - seg.start)
    if lo < 0 or hi > series.shape[0]:
        raise ValueError(
            f'supplied timeseries does not cover INTSTART={int_start} '
            f'through INTEND={int_start + seg.stop - seg.start - 1}')
    return series[lo:hi]


def step_dq_init(state, params, ctx):
    """Initialize DQ and flag saturation plus any supplied hot-pixel map.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    cube = state.cube
    opts = ctx['opts']
    pixeldq = jnp.asarray(cube.pixeldq)
    mask_dq = _require_reference(ctx, 'mask_dq', 'mask_dq reference missing from refpack')
    mask_dq = jnp.asarray(mask_dq, jnp.uint32)
    pixeldq = pixeldq | mask_dq

    # Propagate reference DO_NOT_USE flags into every integration and group.
    mask_dnu = (mask_dq & jnp.uint32(core.DQ_DO_NOT_USE)).astype(jnp.uint8)
    instrument = cube.meta.mode.split('/')[0].upper()

    hot = ctx.get('hot_pixel_map')
    if hot is not None:
        hot = np.asarray(hot, dtype=bool)
        if hot.shape != cube.pixeldq.shape:
            raise ValueError(f'hot_pixel_map has shape {hot.shape}, expected '
                f'{cube.pixeldq.shape}')
        pixeldq = pixeldq | jnp.where(jnp.asarray(hot), core.DQ_HOT, jnp.uint32(0))
    else:
        hot = np.zeros(cube.pixeldq.shape, dtype=bool)

    # Apply the configured full-well threshold without a CRDS saturation plane.
    det_key = cube.meta.detector if instrument == 'NIRSPEC' else None
    full_well = core.FULL_WELL_ADU.get((instrument, det_key), 62070.)
    saturation_threshold = opts.get('saturation_threshold')
    if saturation_threshold is None:
        saturation_threshold = 80

    def flag_chunk(data, groupdq):
        """Initialize group DQ and apply the configured saturation flags."""
        groupdq = groupdq.astype(jnp.uint8) | mask_dnu[None, None]
        if instrument == 'MIRI':
            dnu = jnp.uint8(int(core.DQ_DO_NOT_USE))
            if opts.get('flag_first_miri_frame', True):
                groupdq = groupdq.at[:, 0].set(groupdq[:, 0] | dnu)
            if opts.get('flag_last_miri_frame', True):
                groupdq = groupdq.at[:, -1].set(groupdq[:, -1] | dnu)
        groupdq, _ = k_det.flag_saturated_pixels(
            data, groupdq, jnp.asarray(full_well, dtype=data.dtype),
            saturation_threshold=jnp.asarray(saturation_threshold, dtype=data.dtype),
            flag_neighbours=int(opts.get('flag_neighbours', 1)))
        return groupdq

    groupdq = core.map_over_ints(flag_chunk, (cube.data, cube.groupdq), cube.data.shape[0])
    return _updated_state(state, groupdq=groupdq, pixeldq=np.asarray(pixeldq),
                          aux=dict(state.aux, hot_pixels=hot))


def step_inl(state, params, ctx):
    """Apply the optional NIRISS integral non-linearity correction.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    cube = state.cube
    theta = _rp(ctx, 'inl_theta')
    periods = _rp(ctx, 'inl_periods')
    if theta is None or periods is None:
        raise ValueError('INL theta/periods missing from refpack')
    data = core.map_over_ints(lambda d: k_det.integral_nonlinearity_correct(
            d, jnp.asarray(theta), jnp.asarray(periods)), cube.data, cube.data.shape[0])
    return _updated_state(state, data=data)


def step_superbias(state, params, ctx):
    """Subtract the configured CRDS or visit-derived superbias image.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    cube = state.cube
    method = ctx['opts'].get('superbias_method', 'crds')
    pixeldq = np.asarray(cube.pixeldq, dtype=np.uint32)
    if method == 'crds':
        sb = _require_reference(ctx, 'superbias', 'superbias reference missing from refpack')
        sb = jnp.asarray(sb)
        superbias_dq = _rp(ctx, 'superbias_dq')
        if superbias_dq is not None:
            pixeldq = np.bitwise_or(pixeldq, np.asarray(superbias_dq, dtype=np.uint32))
    else:
        bmask = baseline_bool_for_meta(cube.meta, cube.data.shape[0])
        # Read only baseline first-group images to build the custom superbias.
        selected = np.flatnonzero(np.asarray(bmask))
        sample = cube.data[jnp.asarray(selected) if core.is_device_array(cube.data)
            else selected, 0]
        sb = jnp.asarray(np.nanmedian(np.asarray(core.to_host(sample)), axis=0))
    data = core.map_over_ints(lambda d: k_det.subtract_superbias(d, sb),
        cube.data, cube.data.shape[0])
    return _updated_state(state, data=data, pixeldq=pixeldq)


def step_refpix(state, params, ctx):
    """Subtract NIRISS odd/even electronic offsets measured by reference pixels.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    cube = state.cube
    if str(cube.meta.subarray).upper() != 'SUBSTRIP256':
        return state
    data = core.map_over_ints(lambda d: k_det.refpix_correct(d, jnp.asarray(cube.pixeldq),
                                       nref_side=4, detector_column_axis=-2),
        cube.data, cube.data.shape[0])
    return _updated_state(state, data=data)


def step_dark(state, params, ctx):
    """Subtract the group-matched dark-current reference when enabled.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    cube = state.cube
    dark = _require_reference(ctx, 'dark', 'adapted dark reference missing from refpack')
    pixeldq = np.asarray(cube.pixeldq, dtype=np.uint32)
    dark_dq = _rp(ctx, 'dark_dq')
    if dark_dq is not None:
        pixeldq = np.bitwise_or(pixeldq, np.asarray(dark_dq, dtype=np.uint32))
    data = core.map_over_ints(lambda d: k_det.subtract_dark(d, jnp.asarray(dark, dtype=d.dtype)),
        cube.data, cube.data.shape[0])
    return _updated_state(state, data=data, pixeldq=pixeldq)


def _step_background_soss(state, ctx, group_level):
    """Scale and subtract the SOSS background, then prepare the processed trace stack."""
    cube = state.cube
    model = ctx.get('background_model')
    if model is None:
        step = 'OneOverFStep_grp' if group_level else 'BackgroundStep'
        raise ValueError(f'SOSS background model is required by {step}')
    data = cube.data
    if group_level:
        model = np.asarray(model, dtype=data.dtype)
    bmask = baseline_bool_for_meta(cube.meta, data.shape[0])
    deep = _host_deepstack(data, bmask)
    if not group_level:
        model = np.asarray(model, dtype=data.dtype)
    ngroups = data.shape[1] if group_level else 1
    level = 'grp' if group_level else 'int'
    calls = soss_background_calls(
        ctx.get('opts', {}), 'soss_background_' + level, ngroups, data.shape[-2],
        _background_region(*data.shape[-2:]))
    scaled = (np.stack([_scaled_background_model(data[:1, g], deep[g], model, **calls[g])
                        for g in range(ngroups)], axis=0) if group_level else
              _scaled_background_model(data, deep, model, **calls[0]))
    data = core.map_over_ints(
        lambda d: d - jnp.asarray(scaled, dtype=d.dtype)[None], data, data.shape[0])
    out_cube = replace(cube, data=data)
    centroid_deep = _host_deepstack(data, bmask)
    aux = dict(state.aux, **{'bkg_' + level: scaled})
    if not group_level:
        aux[_OOF_INT_DEEP_KEY] = centroid_deep
    aux = _attach_auto_centroids(
        aux, 'centroids_group' if group_level else 'centroids_int', centroid_deep, out_cube, ctx)
    if group_level:
        aux[_OOF_GRP_PREP_KEY] = _prepare_oneoverf_grp(
            PipelineState(cube=out_cube, aux=aux), ctx, deep=centroid_deep,
            centroids=aux.get('centroids_group') or ctx.get('centroids'))
    return PipelineState(cube=out_cube, aux=aux)


def step_background_grp(state, params, ctx):
    """Subtract the scaled SOSS background from each group.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    return _step_background_soss(state, ctx, True)


def _apply_soss_oneoverf(cube, deep, masks, ts, ts2, method, even_odd,
                         *, background=None, group_level=False):
    """Apply the selected SOSS correction with the original segment and memory bounds."""
    out1, out2, tr1, tr2 = masks
    resident = _resident(cube.data)
    corrected = _segment_buffer(cube.data.shape, cube.data.dtype,
        'exotedrf-oneoverf' if group_level else 'exotedrf-oneoverf-int', resident)

    def correct(data, mask1, *extra):
        """Correct one batch using the selected masks and light curves."""
        options = {'even_odd_rows': even_odd}
        if group_level:
            options['background'] = (None if background is None else
                                     jnp.asarray(background, dtype=data.dtype))
        stack = jnp.asarray(deep, dtype=data.dtype)
        if method == 'achromatic':
            return k_oof.oneoverf_scale_achromatic(data, stack, mask1, *extra, **options)
        kernel = (k_oof.oneoverf_scale_window if method == 'achromatic-window'
                  else k_oof.oneoverf_scale_chromatic)
        mask2, *series = extra
        return kernel(data, stack, mask1, mask2, jnp.asarray(tr1), jnp.asarray(tr2),
                      *series, **options)

    for seg, int_start in zip(_segment_slices(cube), cube.meta.segment_int_starts):
        arrays = [cube.data[seg], out1[seg]]
        if method != 'achromatic':
            arrays.append(out2[seg])
        arrays.append(_series_segment(ts, cube, seg, int_start))
        if method == 'chromatic':
            if ts2 is None:
                raise ValueError('scale-chromatic needs soss_timeseries_o2')
            arrays.append(_series_segment(ts2, cube, seg, int_start))
        if group_level and _device_fast_path_fits(
                cube.data[seg], buffers=6 if method == 'achromatic' else 10):
            part = correct(*(jnp.asarray(a) for a in arrays))
            if not resident:
                part = np.asarray(jax.device_get(part))
        else:
            part = core.map_over_ints(
                correct, (arrays[0], *(_beside(a, resident) for a in arrays[1:])),
                seg.stop - seg.start)
        _segment_store(corrected, seg, part)
    return _segment_result(corrected, resident)


def step_oneoverf_grp(state, params, ctx):
    """Remove column-correlated noise from groups and restore background.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    if _is_oof_solve(ctx):
        return _step_oneoverf_solve(state, params, ctx, 'grp')
    cube = state.cube
    opts = ctx['opts']
    data = cube.data
    prepared = _prepare_oneoverf_grp(state, ctx)
    deep = prepared['deep']
    base_mask = prepared['base_mask']

    method = {'scale-achromatic': 'achromatic', 'scale-achromatic-window': 'achromatic-window',
              'scale-chromatic': 'chromatic'}[opts.get('oof_method', 'scale-achromatic')]
    dtype = np.dtype(data.dtype)
    o1 = jnp.asarray(prepared['centroid_o1'], dtype=dtype)
    o2 = jnp.asarray(prepared['centroid_o2_padded'], dtype=dtype)
    o3 = jnp.asarray(prepared['centroid_o3_padded'], dtype=dtype)
    inner_width = jnp.asarray(params['soss_inner_mask_width'], dtype=dtype)
    outer_width = jnp.asarray(params['soss_outer_mask_width'], dtype=dtype)
    out1, out2, tr1, tr2 = _build_soss_masks_chunked(
        base_mask, o1, o2, o3, inner_width, outer_width, method)

    data = _apply_soss_oneoverf(cube, deep, (out1, out2, tr1, tr2), prepared['timeseries'],
        ctx.get('soss_timeseries_o2'), method, bool(opts.get('oof_even_odd_rows_grp', True)),
        background=prepared['background'], group_level=True)
    aux = dict(state.aux)
    aux.pop(_OOF_GRP_PREP_KEY, None)
    return _updated_state(state, data=data, aux=aux)


def step_linearity(state, params, ctx):
    """Apply the CRDS linearity polynomial to usable ramp samples.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    cube = state.cube
    coeffs = _require_reference(ctx, 'lin_coeffs', 'linearity coefficients missing from refpack')
    coeffs = jnp.asarray(coeffs)
    lin_dq = _require_reference(ctx, 'lin_dq', 'linearity DQ reference missing from refpack')
    coeffs, no_lin, propagated_dq = k_det.prepare_linearity_reference(
        jnp.asarray(coeffs), jnp.asarray(lin_dq, jnp.uint32))
    inverse = _rp(ctx, 'lin_inv_coeffs')
    if inverse is not None:
        slices, readouts = _ramp_segment_readouts(cube)
        read_times = cube.meta.extra.get('read_times')
        if (not any(nframes > 1 for _, nframes, _ in readouts) or
                (read_times is not None and len(read_times) > 0)):
            inverse = None
    if inverse is not None:
        inverse, inverse_skip, inverse_dq = k_det.prepare_linearity_reference(
            jnp.asarray(inverse), jnp.asarray(lin_dq, jnp.uint32))
        propagated_dq |= inverse_dq
        identity = (jnp.arange(coeffs.shape[0]) == 1)[:, None, None]
        forward = jnp.where(no_lin[None], identity, coeffs)
        identity = (jnp.arange(inverse.shape[0]) == 1)[:, None, None]
        inverse = jnp.where(inverse_skip[None], identity, inverse)
    pixeldq = np.bitwise_or(np.asarray(cube.pixeldq, dtype=np.uint32),
        np.asarray(propagated_dq, dtype=np.uint32))
    if inverse is None:
        data = core.map_over_ints(lambda d, g: k_det.linearity_correct(d, g, coeffs,
                                                 no_lin_mask=no_lin),
            (cube.data, cube.groupdq), cube.data.shape[0])
    else:
        parts = []
        for segment, (_, nframes, _) in zip(slices, readouts):
            if nframes > 1:
                correct = lambda d, g: k_det.linearity_correct_reads(
                    d, g, forward, inverse, nframes=nframes)
            else:
                correct = lambda d, g: k_det.linearity_correct(d, g, coeffs, no_lin_mask=no_lin)
            parts.append(core.map_over_ints(correct, (cube.data[segment], cube.groupdq[segment]),
                segment.stop - segment.start))
        data = _join_segments(parts, _resident(cube.data))
    return _updated_state(state, data=data, pixeldq=pixeldq)


def step_jump(state, params, ctx):
    """Run the configured up-ramp and/or across-integration jump searches.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    cube = state.cube
    opts = ctx['opts']
    data = cube.data
    groupdq = cube.groupdq
    nints, ngroups, dimy, dimx = data.shape
    dtype = np.dtype(data.dtype)
    pixeldq = cube.pixeldq
    if _upramp_runs(opts, ngroups):
        groupdq, pixeldq = _apply_upramp_jump(cube, ctx)
    # Run the time-domain search when the up-ramp search cannot run.
    if not opts.get('flag_in_time', True) and _time_jump_runs(opts, ngroups):
        warnings.warn('flag_in_time=False ignored: v1 runs time-domain jump flagging '
            'whenever the JWST up-the-ramp JumpStep does not run ' '(stage1.py:1437-1440)')
    if _time_jump_runs(opts, ngroups):
        instrument = cube.meta.mode.split('/')[0].upper()
        max_reset_int = _reset_max_int(cube.meta)
        resident = _resident(data, groupdq)
        corrected = (_segment_buffer(data.shape, data.dtype, 'exotedrf-jump-science', resident)
                     if ngroups <= 2 else None)
        corrected_dq = _segment_buffer(
            groupdq.shape, groupdq.dtype, 'exotedrf-jump-groupdq', resident)
        for seg, int_start in zip(_segment_slices(cube), cube.meta.segment_int_starts):
            seg_nints = seg.stop - seg.start
            artifact = k_jump.reset_artifact_mask(seg_nints, dimy, dimx, int_start=int(int_start),
                instrument=instrument, max_reset_int=max_reset_int)
            # Exclude the dropped MIRI groups from the time-domain search.
            group_ok = None
            if instrument == 'MIRI':
                group_ok = jnp.asarray(~k_jump.miri_dropped_groups(groupdq[seg]))
            jump_kwargs = {'window': int(params['time_window']), 'artifact': artifact,}
            if group_ok is not None:
                jump_kwargs['group_ok'] = group_ok
            part, part_dq = k_jump.flag_jumps_in_time_chunked(data[seg], groupdq[seg],
                jnp.asarray(params['time_jump_threshold'], dtype=dtype), **jump_kwargs)
            if corrected is not None:
                _segment_store(corrected, seg, part)
            _segment_store(corrected_dq, seg, part_dq)
            del part
        if corrected is not None:
            data = _segment_result(corrected, resident)
        groupdq = _segment_result(corrected_dq, resident)
    return _updated_state(state, data=data, groupdq=groupdq, pixeldq=pixeldq)


def _upramp_runs(opts, ngroups):
    """Check whether the JWST up-the-ramp jump search runs."""
    return bool(opts.get('flag_up_ramp', False)) and int(ngroups) > 2


def _time_jump_runs(opts, ngroups):
    """Check whether the across-integration jump search runs."""
    return bool(opts.get('flag_in_time', True)) or \
        not _upramp_runs(opts, ngroups)


def _upramp_jump_parameters(cube, ctx):
    """Resolve the JWST jump parameters for the observation."""
    cached = ctx.get('upramp_jump_pars')
    if cached is not None:
        return cached
    opts = ctx.get('opts', {})
    crds_pars = ctx.get('upramp_jump_crds_pars')
    source = 'context'
    if crds_pars is None:
        header = (getattr(cube.meta, 'extra', {}) or {}).get('header', {})
        crds_pars, source = upramp.resolve_crds_pars(
            header or {}, opts.get('crds_context'))
    pars = upramp.effective_parameters(
        crds_pars, opts.get('jump_threshold', 15),
        opts.get('upramp_jump_kwargs'))
    ctx['upramp_jump_pars'] = pars
    ctx['upramp_jump_pars_source'] = source
    return pars


def _apply_upramp_jump(cube, ctx):
    """Return ``(groupdq, pixeldq)`` after v1's JWST up-the-ramp JumpStep."""
    opts = ctx.get('opts', {})
    gain, readnoise = _rp(ctx, 'gain'), _rp(ctx, 'readnoise')
    if gain is None or readnoise is None:
        raise ValueError(
            'flag_up_ramp=True requires the readnoise and gain references')
    pars = _upramp_jump_parameters(cube, ctx)
    data, groupdq = cube.data, cube.groupdq
    resident = _resident(data, groupdq)
    slices, readouts = _ramp_segment_readouts(cube)
    out = None if resident else core.empty_host_array(
        groupdq.shape, np.uint8, name='exotedrf-upramp-groupdq')
    parts, pixeldq = [], cube.pixeldq
    for seg, (group_time, nframes, _) in zip(slices, readouts):
        part, pixeldq, _ = upramp.detect_jumps_segment(
            data[seg], groupdq[seg], cube.pixeldq, gain, readnoise, pars,
            group_time=group_time, nframes=nframes,
            cpu_count=opts.get('upramp_cpu_count'))
        if resident:
            parts.append(part)
        else:
            out[seg] = np.asarray(part)
        del part
    return (_join_segments(parts, True) if resident else out), pixeldq


def time_invariant_saturation(groupdq, quantile=0.01, chunk_size=256):
    """Apply a fixed saturation group to each pixel throughout the observation.

    The saturation quantile requires the complete observation.

    Parameters
    ----------
    groupdq : array-like(int)
        Group DQ cube with integration, group, and detector axes.
    quantile : float
        Fraction of integrations required to force a saturation group.
    chunk_size : int
        Number of integrations processed in each batch.

    Returns
    -------
    groupdq : np.ndarray(int)
        Group DQ with forced saturation flags and existing flags retained.
    limit : np.ndarray(int)
        First forced group per pixel, or ngroups when none is forced.
    """
    nints, ngroups = int(groupdq.shape[0]), int(groupdq.shape[1])
    sat_bit = np.uint8(core.DQ_SATURATED)
    counts = np.zeros((ngroups + 1,) + tuple(groupdq.shape[2:]), np.int64)
    for lo in range(0, nints, chunk_size):
        sat = (np.asarray(core.to_host(groupdq[lo:lo + chunk_size]),
                          dtype=np.uint8) & sat_bit) != 0
        first = np.where(sat.any(axis=1), sat.argmax(axis=1), ngroups)
        for k in range(ngroups + 1):
            counts[k] += (first == k).sum(axis=0)
    reached = np.cumsum(counts, axis=0) >= quantile * nints
    limit = reached.argmax(axis=0)
    forced = np.where(np.arange(ngroups)[:, None, None] >= limit[None],
                      sat_bit, np.uint8(0))
    out = core.empty_host_array(groupdq.shape, np.uint8,
                                name='exotedrf-groupdq-saturation')
    for lo in range(0, nints, chunk_size):
        out[lo:lo + chunk_size] = np.asarray(
            core.to_host(groupdq[lo:lo + chunk_size]), dtype=np.uint8) | forced[None]
    return out, limit


def step_rampfit(state, params, ctx):
    """Convert ramps into one count rate, uncertainty, and DQ per integration.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    cube = state.cube
    rn_ref, gain_ref = _rp(ctx, 'readnoise'), _rp(ctx, 'gain')
    if rn_ref is None or gain_ref is None:
        raise ValueError('RampFit requires readnoise and gain references')
    science = cube.data
    dtype = np.dtype(science.dtype)
    rn = jnp.asarray(rn_ref, dtype=dtype)
    gain = jnp.asarray(gain_ref, dtype=dtype)
    pixeldq = jnp.asarray(cube.pixeldq)
    opts = ctx.get('opts', {})
    if cube.meta.mode.split('/')[0].upper() == 'MIRI':
        dark_enabled = (_runs(opts, 'LinearityStep') and opts.get('miri_subtract_dark', True))
    else:
        dark_enabled = _runs(opts, 'DarkCurrentStep')
    if dark_enabled:
        average_dark_ref = _rp(ctx, 'average_dark_current', 0.)
        average_dark = jnp.asarray(average_dark_ref, dtype=dtype)
    else:
        average_dark_ref = 0.
        average_dark = jnp.asarray(0., dtype=dtype)
    slices, readouts = _ramp_segment_readouts(cube)
    # Pass suppress_one_group only when its default is overridden.
    kw = ({} if opts.get('suppress_one_group', True) else {'suppress_one_group': False})
    groupdq = cube.groupdq
    if opts.get('saturation_time_invariant', False):
        # Apply one saturation group per pixel throughout the observation.
        groupdq, _ = time_invariant_saturation(cube.groupdq,
            quantile=float(opts.get('saturation_time_invariant_quantile', 0.01)))
    output_shape = (science.shape[0],) + science.shape[-2:]
    rate = core.empty_host_array(output_shape, dtype, name='exotedrf-rate')
    err = core.empty_host_array(output_shape, dtype, name='exotedrf-error')
    dq = core.empty_host_array(output_shape, np.uint32, name='exotedrf-dq')
    for segment, (group_time, nframes, one_group_time) in zip(slices, readouts):
        segment_data = science[segment]
        segment_dq = groupdq[segment]
        gt = jnp.asarray(group_time, dtype=dtype)
        nf = jnp.asarray(nframes, dtype=dtype)
        med_rate, max_segments, exception_mask = _segment_median_rate(
            segment_data, segment_dq, group_time, one_group_time, return_topology=True, **kw)
        exception_count = int(np.count_nonzero(exception_mask))
        sparse = _use_sparse_rampfit(exception_count, exception_mask.size)
        fit_segments = 1 if sparse else max_segments

        def _fit(d, g):
            """Fit the ramps in one integration batch."""
            return k_ramp.fit_ramps_stage(d, g, rn, gain, gt, pixeldq,
                median_rate=med_rate, nframes=nf, average_dark_current=average_dark,
                one_group_time=(None if one_group_time is None else
                                jnp.asarray(one_group_time, dtype=dtype)),
                max_segments=fit_segments, **kw)

        part_rate, part_err, part_dq = core.map_over_ints(
            _fit, (segment_data, segment_dq), segment.stop - segment.start,
            chunk_size=_rampfit_integration_chunk_size(segment.stop - segment.start,
                int(np.prod(segment_data.shape[1:])) * (
                    dtype.itemsize + np.dtype(segment_dq.dtype).itemsize),
                resident=core.is_device_array(segment_data)))
        # Store fitted rates on the host for nearest-neighbor interpolation.
        _store_host_segment(rate, segment, part_rate)
        _store_host_segment(err, segment, part_err)
        _store_host_segment(dq, segment, part_dq)
        del part_rate, part_err, part_dq
        if sparse and exception_count:
            _apply_sparse_rampfit(rate[segment], err[segment], dq[segment],
                segment_data, segment_dq, rn_ref, gain_ref, cube.pixeldq, med_rate,
                exception_mask, group_time, nframes,
                average_dark_ref, one_group_time, max_segments, **kw)
    # Fill invalid rates in place while retaining their zero uncertainties.
    executor = ctx.get(_RAMPFIT_FILL_EXECUTOR_KEY)
    aux = state.aux
    if executor is None:
        rate = _nearest_fill_rate_planes(rate, copy=False)
    else:
        # Defer the in-place fill until await_rampfit_fill resolves its future.
        aux = dict(aux)
        aux[_RAMPFIT_FILL_FUTURE_KEY] = executor.submit(_nearest_fill_rate_planes, rate, copy=False)
    return PipelineState(cube=RateCube(rate, err, dq, cube.meta), aux=aux)


def await_rampfit_fill(state):
    """Complete a deferred rate fill and remove its scheduling data.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    future = state.aux.get(_RAMPFIT_FILL_FUTURE_KEY)
    if future is None:
        return state
    future.result()
    aux = dict(state.aux)
    del aux[_RAMPFIT_FILL_FUTURE_KEY]
    return PipelineState(cube=state.cube, aux=aux)


def step_gain_scale(state, params, ctx):
    """Apply the exposure-level gain factor to science rates and errors.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    cube = state.cube
    factor = _rp(ctx, 'gain_factor', _rp(ctx, 'GAINFACT'))
    if factor is None:
        raise ValueError('GAINFACT/gain_factor missing from refpack')
    data, err = core.map_over_ints(lambda d, e: (k_det.gain_scale(
            d, jnp.asarray(factor, dtype=d.dtype)), k_det.gain_scale(
                          e, jnp.asarray(factor, dtype=e.dtype))),
        (cube.data, cube.err), cube.data.shape[0])
    return _updated_state(state, data=data, err=err)


def step_assign_wcs(state, params, ctx):
    """Record the SOSS wavelength vectors and point-source metadata.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    waves = ctx.get('waves')
    if not waves or 1 not in waves or 2 not in waves:
        raise ValueError('AssignWCSStep requires SOSS O1/O2 wavelength vectors')
    aux = dict(state.aux, assign_wcs='COMPLETE', source_type='POINT', waves=waves,
               wavemap_provenance=ctx.get('wavemap_provenance'))
    return PipelineState(cube=state.cube, aux=aux)


def step_source_type(state, params, ctx):
    """Record that the time-series target is treated as a point source.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    aux = dict(state.aux, source_type='POINT', source_type_step='COMPLETE')
    return PipelineState(cube=state.cube, aux=aux)


def step_flat(state, params, ctx):
    """Correct NIRISS pixel sensitivity and propagate flat uncertainty and DQ.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    cube = state.cube
    flat = _require_reference(ctx, 'flat', 'flat reference missing from refpack')
    flat = np.asarray(flat)
    flat_err = _rp(ctx, 'flat_err')
    if flat_err is None:
        flat_err = np.zeros(flat.shape, dtype=flat.dtype)
    flat_dq = _rp(ctx, 'flat_dq')
    if flat_dq is None:
        flat_dq = np.zeros(flat.shape, dtype=np.uint32)

    def apply_chunk(data, err, dq):
        """Apply the flat field to one integration batch."""
        return k_det.apply_flat_field(data, err, dq, jnp.asarray(flat, dtype=data.dtype),
            jnp.asarray(flat_err, dtype=data.dtype), jnp.asarray(flat_dq, dtype=jnp.uint32))

    data, err, dq = core.map_over_ints(
        apply_chunk, (cube.data, cube.err, cube.dq), cube.data.shape[0])
    needs_fill = (bool(np.asarray(core.to_host(jnp.any(~jnp.isfinite(data)))))
                  if core.is_device_array(data) else not np.isfinite(data).all())
    if needs_fill:
        resident = core.is_device_array(data)
        data = _nearest_fill_rate_planes(core.to_host(data))
        data = _beside(data, resident)
    return _updated_state(state, data=data, err=err, dq=dq)


def step_background_int(state, params, ctx):
    """Subtract the scaled SOSS background from the integration rates.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    return _step_background_soss(state, ctx, False)


def step_oneoverf_int(state, params, ctx):
    """Remove column-correlated noise from the SOSS integration rates.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    if _is_oof_solve(ctx):
        return _step_oneoverf_solve(state, params, ctx, 'int')
    cube = state.cube
    opts = ctx['opts']
    data = cube.data
    dimx = data.shape[-1]
    bmask = baseline_bool_for_meta(cube.meta, data.shape[0])
    deep = state.aux.get(_OOF_INT_DEEP_KEY)
    if deep is None:
        deep = _host_deepstack(data, bmask)
    cen = _centroids_for_stage(state, ctx, 'centroids_int', deepstack=deep)

    base_mask = _oneoverf_base_mask(cube, ctx)

    method = {'scale-achromatic': 'achromatic', 'scale-achromatic-window': 'achromatic-window',
              'scale-chromatic': 'chromatic'}[opts.get('oof_method', 'scale-achromatic')]
    dtype = np.dtype(data.dtype)
    o1 = jnp.asarray(np.nan_to_num(cen['ypos o1']), dtype=dtype)
    o2 = _padded_optional_trace(cen.get('ypos o2'), dimx, dtype=dtype)
    o3 = _padded_optional_trace(cen.get('ypos o3'), dimx, dtype=dtype)
    out1, out2, tr1, tr2 = _build_soss_masks_chunked(base_mask, o1, o2, o3,
        jnp.asarray(params['soss_inner_mask_width'], dtype=dtype),
        jnp.asarray(params['soss_outer_mask_width'], dtype=dtype), method)

    ts = ctx.get('soss_timeseries')
    if ts is None:
        ts = _estimate_segment_timeseries(data, deep, cube.meta,
            smoothing_scale=opts.get('oof_smoothing_scale_int'))
    else:
        ts = np.asarray(ts)
    data = _apply_soss_oneoverf(
        cube, deep, (out1, out2, tr1, tr2), ts, ctx.get('soss_timeseries_o2'),
        method, bool(opts.get('oof_even_odd_rows_int', True)))
    aux = dict(state.aux)
    aux.pop(_OOF_INT_DEEP_KEY, None)
    return _updated_state(state, data=data, aux=aux)


OOF_SOLVE_AUX_KEYS = {'grp': 'oneoverf_solve_grp', 'int': 'oneoverf_solve_int'}


def _is_oof_solve(ctx):
    """Check whether the SOSS 1/f correction uses the solve method."""
    opts = ctx.get('opts', {}) if hasattr(ctx, 'get') else {}
    return str(opts.get('oof_method', 'scale-achromatic')).lower() == 'solve'


def _solve_centroid_row(values, dimx):
    """Pad float64 centroids with NaNs to the detector width."""
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    row = np.full(dimx, np.nan)
    row[:min(dimx, values.size)] = values[:dimx]
    return row


def _solve_trace_bounds(centroids, width, dimy, dimx):
    """Calculate the order-1 and order-2 windows for the solve method."""
    o1 = _solve_centroid_row(centroids['ypos o1'], dimx)
    b1 = k_oof.solve_trace_bounds(o1, width, dimy)
    o2 = _solve_centroid_row(centroids.get('ypos o2'), dimx)
    if o2 is None:
        zero = np.zeros(dimx, np.int32)
        return b1, (zero, zero)
    return b1, k_oof.solve_trace_bounds(o2, width, dimy)


def _solve_excluded(cube, ctx, seg, int_start):
    """Select the supplied outlier map or group-level reset-artifact mask.

    Apply reset artifacts to every group; the F277W contaminant mask is unused by solve.
    """
    custom = ctx.get('outlier_mask')
    nseg = seg.stop - seg.start
    if custom is not None:
        custom = np.asarray(custom)
        if custom.ndim == 3 and custom.shape[0] == cube.data.shape[0]:
            custom = custom[seg]
        return np.broadcast_to(custom.astype(bool, copy=False),
                               (nseg, *cube.data.shape[-2:]))
    if cube.data.ndim != 4:
        return None
    artifact = k_jump.reset_artifact_mask(
        nseg, *cube.data.shape[-2:], int_start=int(int_start),
        instrument='NIRISS')
    return artifact if artifact.any() else None


def _solve_segment_std(part):
    """Calculate the temporal standard deviation for one FITS segment."""
    std = core.map_over_cols(k_oof.segment_nanstd, part, part.shape[-1])
    return jnp.asarray(std)


def _concat_solve_diagnostics(parts):
    """Concatenate the solve diagnostics along the integration axis."""
    if not parts:
        return {}
    return {order: {key: np.concatenate([p[order][key] for p in parts],
                                        axis=0)
                    for key in parts[0][order]}
            for order in parts[0]}


def _solve_inputs(state, ctx, level):
    """Get the baseline stack, background, and centroids for the solve method."""
    cube = state.cube
    if level == 'grp':
        prepared = _prepare_oneoverf_grp(state, ctx)
        deep = prepared['deep']
        background = prepared['background']
        centroids = _centroids_for_stage(
            state, ctx, 'centroids_group', deepstack=deep)
    else:
        deep = state.aux.get(_OOF_INT_DEEP_KEY)
        if deep is None:
            bmask = baseline_bool_for_meta(cube.meta, cube.data.shape[0])
            deep = _host_deepstack(cube.data, bmask)
        # Omit the background model for integration-level solve correction.
        background = None
        centroids = _centroids_for_stage(
            state, ctx, 'centroids_int', deepstack=deep)
    return deep, background, centroids


def _step_oneoverf_solve(state, params, ctx, level):
    """Apply the SOSS solve correction independently to each FITS segment."""
    cube = state.cube
    data = cube.data
    is_ramp = data.ndim == 4
    dimy, dimx = data.shape[-2:]
    dtype = np.dtype(data.dtype)
    deep, background, centroids = _solve_inputs(state, ctx, level)
    b1, b2 = _solve_trace_bounds(centroids, float(params['soss_outer_mask_width']), dimy, dimx)
    order2 = dimy != 96
    deep_dev = jnp.asarray(deep, dtype=dtype)
    bkg_dev = (None if background is None else jnp.asarray(background, dtype=dtype))
    pixeldq = jnp.asarray(cube.pixeldq) if is_ramp else None
    dq_all = cube.groupdq if is_ramp else cube.dq
    resident = _resident(data)
    corrected = _segment_buffer(data.shape, data.dtype, 'exotedrf-oneoverf-solve', resident)
    diag_parts = []
    for seg, int_start in zip(_segment_slices(cube), cube.meta.segment_int_starts):
        part = data[seg]
        std = _solve_segment_std(part).astype(dtype)
        excluded = _solve_excluded(cube, ctx, seg, int_start)

        def kernel(d, q, ex=None, std=std):
            """Apply the solve correction to one integration batch."""
            return k_oof.oneoverf_solve(d, deep_dev, std, q, b1, b2, excluded=ex, pixeldq=pixeldq,
                background=bkg_dev, order2=order2)

        arrays = (part, dq_all[seg])
        if excluded is not None:
            arrays = arrays + (_beside(np.ascontiguousarray(excluded), resident),)
        out, diag = core.map_over_ints(kernel, arrays, seg.stop - seg.start)
        _segment_store(corrected, seg, out)
        diag_parts.append(jax.tree.map(lambda a: np.asarray(jax.device_get(a)), diag))
        del out, diag, std
    data = _segment_result(corrected, resident)
    diagnostics = _concat_solve_diagnostics(diag_parts)
    if not ctx.get('opts', {}).get('do_plots', True):
        # Retain the scalings when diagnostic 1/f levels are not requested.
        diagnostics = {order: {key: value for key, value in values.items()
                               if key.startswith('scale_')}
                       for order, values in diagnostics.items()}
    aux = dict(state.aux)
    aux[OOF_SOLVE_AUX_KEYS[level]] = diagnostics
    aux.pop(_OOF_GRP_PREP_KEY if is_ramp else _OOF_INT_DEEP_KEY, None)
    return _updated_state(state, data=data, aux=aux)


class PreparedOneOverFSolveScorer:
    """Score final-group ``solve`` corrections by varying the outer trace-mask width."""

    grid_order = ('soss_inner_mask_width', 'soss_outer_mask_width')

    def __init__(self, state, params, ctx):
        """Prepare reusable arrays and candidate caches."""
        cube = state.cube
        self.ctx = ctx
        self.meta = cube.meta
        self.dtype = np.dtype(cube.data.dtype)
        prepared = _prepare_oneoverf_grp(state, ctx)
        state = PipelineState(cube, dict(state.aux, **{_OOF_GRP_PREP_KEY: prepared}))
        deep, background, centroids = _solve_inputs(state, ctx, 'grp')
        self.centroids_solve = centroids
        self.dimy, self.dimx = cube.data.shape[-2:]
        self.deep = jnp.asarray(np.asarray(deep)[-1], dtype=self.dtype)
        self.background = (None if background is None else jnp.asarray(
            np.asarray(background)[-1], dtype=self.dtype))
        self.pixeldq = jnp.asarray(cube.pixeldq)
        self.segments = []
        for seg, int_start in zip(_segment_slices(cube), cube.meta.segment_int_starts):
            data = jnp.asarray(cube.data[seg, -1])
            excluded = _solve_excluded(cube, ctx, seg, int_start)
            self.segments.append((data, jnp.asarray(cube.groupdq[seg, -1]),
                k_oof.segment_nanstd(data), None if excluded is None else jnp.asarray(excluded)))
        self.groupdq = jnp.concatenate([s[1] for s in self.segments], axis=0)
        self.extract_width = self.dtype.type(params['extract_width'])
        self.centroids = {'ypos o1': np.asarray(prepared['centroid_o1'])}
        if prepared['centroid_o2_extract'] is not None:
            self.centroids['ypos o2'] = np.asarray(prepared['centroid_o2_extract'])
        self._result_cache = {}

    def _evaluate(self, outer_width):
        """Calculate or retrieve the cost, scatter, and elapsed time for one width."""
        key = float(np.asarray(outer_width).reshape(()))
        cached = self._result_cache.get(key)
        if cached is not None:
            return cached[0], np.array(cached[1], copy=True), 0.
        started = time.perf_counter()
        b1, b2 = _solve_trace_bounds(self.centroids_solve, key, self.dimy, self.dimx)
        corrected = jnp.concatenate([k_oof.oneoverf_solve(
                data, self.deep, std, dq, b1, b2, excluded=excluded,
                pixeldq=self.pixeldq, background=self.background, order2=self.dimy != 96)[0]
            for data, dq, std, excluded in self.segments], axis=0)
        cost, scatter = _optimizer_cost_from_last_group(
            corrected, self.groupdq, {'extract_width': self.extract_width},
            self.ctx, self.meta, self.centroids)
        cost, scatter = jax.device_get((cost, scatter))
        result = float(np.asarray(cost)), np.asarray(scatter)
        self._result_cache[key] = (result[0], np.array(result[1], copy=True))
        return result[0], result[1], time.perf_counter() - started

    def evaluate_candidates(self, parameter, candidates, params):
        """Score each candidate in its configured order.

        Parameters
        ----------
        parameter : str
            Name of the parameter to vary.
        candidates : array-like(float)
            Candidate values in their configured order.
        params : dict
            Current values of the other reduction parameters.

        Returns
        -------
        results : list[tuple]
            Cost, wavelength scatter, and elapsed seconds for each candidate.
        """
        values = list(candidates)
        if parameter == 'soss_inner_mask_width':
            cost, scatter, duration = self._evaluate(params['soss_outer_mask_width'])
            return [(cost, np.array(scatter, copy=True), duration if index == 0 else 0.)
                    for index, _ in enumerate(values)]
        if parameter != 'soss_outer_mask_width':
            raise KeyError(parameter)
        return [self._evaluate(value) for value in values]


def step_badpix(state, params, ctx):
    """Replace persistent spatial defects and isolated temporal outliers.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    cube = state.cube
    data = cube.data
    bmask = baseline_bool_for_meta(cube.meta, data.shape[0])
    deep = _host_deepstack(data, bmask)
    instrument = cube.meta.mode.split('/')[0].upper()
    resident = _resident(data, cube.err, cube.dq)
    out_data = _segment_buffer(data.shape, data.dtype, 'exotedrf-badpix-data', resident)
    out_dq = _segment_buffer(cube.dq.shape, np.uint32, 'exotedrf-badpix-dq', resident)
    err_in, dq_in = cube.err, cube.dq
    # Reuse finite input errors and copy only when replacing NaNs.
    out_err = err_in
    err_parts, err_changed = ([] if resident else None), False
    box_size = int(params['box_size'])
    # Extend interior halos to preserve the full-detector interpolation-box bounds.
    halo = max(5, box_size + 1 if instrument != 'MIRI' else 0)
    first_badpix = None
    plot_categories = None
    high_var_maps = []
    bp_opts = ctx.get('opts', {})
    median_hv = bp_opts.get('median_high_variance') is True
    preserve_sat = bp_opts.get('preserve_saturated') is True
    clear_dq = bp_opts.get('clear_interpolated_dq') is True
    for seg_index, seg in enumerate(_segment_slices(cube)):
        seg_data = data[seg]
        chunk_size = _badpix_column_chunk_size(seg_data, box_size, halo=halo)
        seg_err = err_in[seg]
        # Replace error NaNs with the detector-wide median before splitting columns.
        plane_bytes = int(np.prod(seg_err.shape[1:])) * seg_err.dtype.itemsize
        error_chunk = max(1, (32 << 20) // max(plane_bytes, 1))
        isnan = jnp.isnan if resident else np.isnan
        if any(bool(isnan(seg_err[first:first + error_chunk]).any())
               for first in range(0, seg_err.shape[0], error_chunk)):
            median_error = hoststats.nanmedian(seg_err)
            if resident:
                seg_err = jnp.where(jnp.isnan(seg_err), median_error, seg_err)
                err_changed = True
            else:
                if out_err is err_in:
                    out_err = core.copy_host_array(err_in, name='exotedrf-badpix-error')
                for first in range(0, seg_err.shape[0], error_chunk):
                    sl = slice(seg.start + first, min(seg.start + first + error_chunk, seg.stop))
                    part_error = out_err[sl]
                    part_error[np.isnan(part_error)] = median_error
        if resident:
            err_parts.append(seg_err)
        seg_dq = dq_in[seg]

        reused_map = ctx.get('badpix_reuse_map')
        if seg_index == 0 and reused_map is not None:
            # Reuse a supplied spatial map without rediscovering defects or clipping negatives.
            first_badpix = np.asarray(reused_map, dtype=bool)
            if first_badpix.shape != tuple(data.shape[-2:]):
                raise ValueError(f'reused hot pixel map has shape {first_badpix.shape}, '
                    f'expected {tuple(data.shape[-2:])}')
        elif seg_index == 0:
            # Discover spatial defects on the complete detector plane before splitting columns.
            saturated_any = np.zeros(data.shape[-2:], dtype=bool)
            # Read only the detector DQ plane and integration 10 for spatial discovery.
            for first in range(0, seg_data.shape[0], 32):
                saturated_any |= np.any((np.asarray(seg_dq[first:first + 32], dtype=np.uint32) &
                     np.uint32(core.DQ_SATURATED)) != 0, axis=0)
            ref_int = min(10, seg_data.shape[0] - 1)
            reference_dq = np.asarray(seg_dq[ref_int], dtype=np.uint32)
            spatial_prepared = k_badpix.prepare_spatial_badpix(
                deep, reference_dq[None], saturated_any, box_size=box_size,
                instrument=instrument, preserve_saturated=preserve_sat)
            first_badpix = np.asarray(k_badpix.spatial_badpix_from_prepared(*spatial_prepared,
                    space_thresh=jnp.asarray(params['space_outlier_threshold'], dtype=data.dtype)),
                dtype=bool)
            if ctx.get('opts', {}).get('do_plots', False):
                dimy, dimx = deep.shape
                yy = np.arange(dimy)[:, None]
                xx = np.arange(dimx)[None, :]
                ymax = dimy - 5 if instrument == 'NIRISS' else dimy
                inbounds = ((xx >= 5) & (xx < dimx - 5) & (yy < ymax))
                hot = ((reference_dq & np.uint32(
                    core.DQ_DO_NOT_USE | core.DQ_HOT | core.DQ_WARM)) != 0)
                hot &= inbounds
                negative = np.isnan(np.asarray(deep)) & inbounds & ~hot
                selected = np.asarray(first_badpix, dtype=bool)
                other = selected & ~hot & ~negative
                plot_categories = {'badpix_plot_hot': hot, 'badpix_plot_nan': negative,
                    'badpix_plot_other': other,}
        # Finish spatial replacement before calculating detector-wide fallback scatter.
        spatial_data, spatial_dq = core.map_over_cols_with_halo(
            lambda d, q, bad: k_badpix.apply_spatial_badpix(
                # Clip negative values only while discovering the first spatial map.
                jnp.where(d < 0, 0, d) if seg_index == 0 and
                reused_map is None else d,
                q, bad, box_size=box_size, instrument=instrument,
                clear_interpolated_dq=clear_dq),
            (seg_data, seg_dq, _beside(first_badpix, resident)),
            data.shape[-1], halo=halo, chunk_size=chunk_size)
        scatter = core.map_over_cols(k_badpix.temporal_scatter, spatial_data, data.shape[-1],
            chunk_size=chunk_size)
        # Apply the zero-scatter fallback and spatial replacement to the assembled plane.
        space_thresh = jnp.asarray(params['space_outlier_threshold'], dtype=data.dtype)
        _, _, scatter = k_badpix.noise_plane_spatial(
            jnp.asarray(scatter), space_thresh, box_size=box_size, instrument=instrument)

        def temporal_chunk(d, q, original_dq, sigma):
            """Apply temporal bad-pixel replacement to one detector-column section."""
            saturated = (original_dq & core.DQ_SATURATED) != 0
            filtered, sigma = k_badpix.prepare_temporal_badpix(
                d, window_size=int(params['window_size']), instrument=instrument, scatter=sigma)
            return k_badpix.apply_temporal_badpix(d, q, saturated, filtered, sigma,
                time_thresh=jnp.asarray(params['time_outlier_threshold'], dtype=d.dtype),
                preserve_saturated=preserve_sat, clear_interpolated_dq=clear_dq)

        data0, dq0 = core.map_over_cols_with_halo(temporal_chunk,
            (spatial_data, spatial_dq, seg_dq, scatter),
            data.shape[-1], halo=halo, chunk_size=chunk_size)

        # Store high-variance flags separately because uint32 DQ cannot hold bit 32.
        tstd = core.map_over_cols(k_badpix.temporal_std, data0, data.shape[-1],
            chunk_size=chunk_size)
        high_var, _, _ = k_badpix.noise_plane_spatial(
            jnp.asarray(tstd), space_thresh, box_size=box_size, instrument=instrument)
        high_var_maps.append(np.asarray(high_var))

        def highvar_chunk(d, q, original_dq, flags):
            """Finalize bad-pixel replacement and calculate the high-variance mask."""
            saturated = (original_dq & core.DQ_SATURATED) != 0
            filtered = k_badpix.temporal_filter(d, window_size=int(params['window_size']),
                instrument=instrument)
            d = k_badpix.apply_high_variance(d, flags, jnp.any(saturated, axis=0),
                median_high_variance=median_hv, preserve_saturated=preserve_sat)
            d, _, q = k_badpix.finalize_badpix(
                d, None, q, filtered, saturated, instrument=instrument,
                clear_interpolated_dq=clear_dq)
            return d, q

        newdata, dq = core.map_over_cols_with_halo(
            highvar_chunk, (data0, dq0, seg_dq, _beside(high_var, resident)),
            data.shape[-1], halo=halo, chunk_size=chunk_size)

        # Apply the full-frame NIRISS edge correction after assembling the detector columns.
        if instrument == 'NIRISS' and data.shape[-1] == 2048 and \
                data.shape[-2] >= 96:
            # Read only the rows needed for the NIRISS edge replacement.
            block = np.concatenate([np.asarray(core.to_host(newdata[:, 82:84, 2018:])),
                 np.asarray(core.to_host(newdata[:, 88:90, 2018:]))], axis=1)
            mm = np.nanmedian(block, axis=1)
            if resident:
                newdata = newdata.at[:, 84:88, 2018:].set(jnp.asarray(mm)[:, None, :])
            else:
                newdata[:, 84:88, 2018:] = mm[:, None, :]
        _segment_store(out_data, seg, newdata)
        _segment_store(out_dq, seg, dq)
    if resident and err_changed:
        out_err = _join_segments(err_parts, True)
    newdata = _segment_result(out_data, resident)
    dq = _segment_result(out_dq, resident)
    # Retain the BadPix input deepframe for final tracing when PCA is skipped.
    aux = dict(state.aux, hot_pixel_map=first_badpix, deepframe=deep)
    if high_var_maps:
        aux['high_variance_maps'] = high_var_maps
        aux['high_variance_map'] = np.logical_or.reduce(high_var_maps)
    if plot_categories is not None:
        aux.update(plot_categories)
    return _updated_state(state, data=newdata, err=out_err, dq=dq, aux=aux)


def _v1_lcestimate(components, baseline_ints, fallback):
    """Normalize the first PCA component using the v1 light-curve baseline."""
    if baseline_ints is None:
        return fallback
    frames = np.atleast_1d(np.asarray(baseline_ints)).astype(int)
    if frames.size == 1:
        n = int(frames[0])
        index = np.arange(n) if n > 0 else np.arange(-n) + n
    elif frames.size == 2:
        first, last = np.abs(frames)
        index = np.concatenate([np.arange(first), np.arange(last) - last])
    else:
        return fallback
    first_pc = np.asarray(components)[0]
    if index.size == 0 or np.any(index >= first_pc.size) or \
            np.any(index < -first_pc.size):
        return fallback
    norm = np.nanmedian(first_pc[index])
    return np.asarray(first_pc / norm, dtype=first_pc.dtype)


def _pca_fit_components(n_comp, remove_components):
    """Choose the component count needed to fit all requested PCA modes."""
    if remove_components is None or not np.atleast_1d(
            remove_components).size:
        return n_comp
    return max(n_comp, int(np.max(np.atleast_1d(remove_components))))


def step_pca(state, params, ctx):
    """Optionally subtract selected detector-wide temporal PCA modes.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    cube = state.cube
    opts = ctx['opts']
    n_comp = int(opts.get('pca_components', 10))
    nints = cube.data.shape[0]
    if nints <= n_comp:
        return state
    source = cube.data
    bmask = baseline_bool_for_meta(cube.meta, nints)
    stage3_deepframe = _host_deepstack(source, bmask)
    # Defer tracing the final deepframe until extraction.
    aux = dict(state.aux, stage3_deepframe=stage3_deepframe)
    if opts.get('skip_pca', False):
        return PipelineState(cube=cube, aux=aux)
    remove = np.zeros(_pca_fit_components(n_comp, opts.get('remove_components')), dtype=bool)
    rc = opts.get('remove_components')
    if rc is not None and np.atleast_1d(rc).size:
        remove[np.asarray(rc, dtype=int) - 1] = True
    # Let the PCA driver stream the input cube in detector-column batches.
    return_plot_data = bool(opts.get('do_plots', False))
    # Trim MIRI columns 12:61 to exclude the lightsaber artifact from the PCA fit.
    instrument = cube.meta.mode.split('/')[0].upper()
    trim = (slice(*MIRI_PCA_COLUMNS) if instrument == 'MIRI' and
            cube.data.shape[-1] > MIRI_PCA_COLUMNS[1] else None)
    pca_input = source if trim is None else source[:, :, trim]
    pca_result = k_pca.pca_reconstruction(pca_input, remove, bmask, n_components=n_comp,
        return_plot_data=return_plot_data)
    if return_plot_data:
        (recon, comps, eigvals, wlc, comps_recon, eigvals_recon,
         projections, projections_recon) = pca_result
    else:
        recon, comps, eigvals, wlc, comps_recon, eigvals_recon = pca_result
    wlc = _v1_lcestimate(comps, cube.meta.baseline_ints, wlc)
    if not core.is_device_array(recon):
        recon = np.asarray(recon)
    if trim is not None:
        if core.is_device_array(source):
            recon = source.at[:, :, trim].set(jnp.asarray(recon, dtype=source.dtype))
        else:
            full = core.copy_host_array(source, name='exotedrf-miri-pca')
            full[:, :, trim] = recon
            recon = full
    aux.update(pca_components=np.asarray(comps), pca_eigvals=np.asarray(eigvals),
               pca_wlc=np.asarray(wlc), pca_components_reconstructed=np.asarray(comps_recon),
               pca_eigvals_reconstructed=np.asarray(eigvals_recon))
    if return_plot_data:
        aux['pca_projections'] = np.asarray(projections)
        if rc is not None:
            aux['pca_projections_reconstructed'] = np.asarray(projections_recon)
    return _updated_state(state, data=recon, aux=aux)


_OPTIMIZE_WIDTHS = {
    'NIRISS': np.linspace(10, 60, 51),
    'NIRSPEC': np.linspace(1, 11, 11),
    'MIRI': np.linspace(2, 12, 11),
}
# Use grating-specific NRS1 start columns for the aperture sweep.
_NIRSPEC_OPTIMIZE_XSTART = {'G395H': 500, 'G395M': 200, 'PRISM': 14}


def _extract_halfwidth(width, dtype):
    """Convert one extraction width to a traced box-aperture half-width."""
    if v2config.is_asymmetric_width(width):
        return jnp.asarray(v2config.extract_width_halves(width), dtype=dtype)
    return jnp.asarray(width, dtype=dtype) / jnp.asarray(2., dtype=dtype)


def _width_cache_key(width):
    """Return a hashable cache key for a scalar or two-sided aperture width."""
    if v2config.is_asymmetric_width(width):
        return v2config.extract_width_halves(width)
    return float(np.asarray(width).reshape(()))


def _scorer_extract_width(width, dtype):
    """Convert a scalar extraction width to the science dtype or keep a width pair."""
    if v2config.is_asymmetric_width(width):
        return width
    return dtype.type(width)


def _broadcast_candidate_halfwidth(fixed, candidates):
    """Repeat one fixed half-width (scalar or pair) for every candidate."""
    fixed = jnp.asarray(fixed)
    return jnp.broadcast_to(fixed, (candidates.shape[0],) + fixed.shape)


def _stage3_deepframe(state, ctx):
    """Get the deep image used for final tracing and optimal extraction."""
    custom = ctx.get('custom_deepframe')
    if custom is not None:
        return custom
    deepframe = state.aux.get('stage3_deepframe')
    if deepframe is None:
        deepframe = state.aux.get('deepframe')
    return deepframe


def _load_custom_deepframe(value, shape, search_dirs=()):
    """Load a custom deepframe from a FITS image."""
    if value is None:
        return None
    candidate = os.path.expanduser(os.fspath(value))
    paths = [candidate] if os.path.isabs(candidate) else [
        os.path.join(directory, candidate) for directory in search_dirs]
    paths.append(candidate)
    path = next((item for item in paths if os.path.exists(item)), None)
    if path is None:
        raise FileNotFoundError(f'deepframe file not found: {value}')
    deepframe = np.asarray(fits.getdata(path), dtype=float)
    if deepframe.shape != tuple(shape):
        raise ValueError(
            f'deepframe {path} has shape {deepframe.shape}, expected the '
            f'2-D detector shape {tuple(shape)}')
    return deepframe


def _select_optimize_width(wlc, instrument):
    """Select the first aperture width with minimum white-light scatter."""
    candidates = _OPTIMIZE_WIDTHS[instrument]
    width, index, scatter = k_ext.select_width_min_white_scatter(
        jax.device_get(wlc), candidates)
    return width, {'widths': np.asarray(candidates), 'scatter': scatter,
                   'index': index, 'width': width}


def _optimize_candidate_halfwidths(instrument, dtype):
    """Return the candidate half-widths for the instrument."""
    return jnp.asarray(_OPTIMIZE_WIDTHS[instrument], dtype=dtype) / \
        jnp.asarray(2., dtype=dtype)


def _box_bitmask(opts, instrument):
    """Select the DQ classes excluded from production box extraction."""
    bitmask = np.uint32(0)
    if opts.get('mask_do_not_use_pixels', False):
        bitmask |= core.DQ_DO_NOT_USE
    mask_sat = bool(opts.get('mask_saturated_pixels', False))
    if instrument == 'NIRISS' and opts.get('saturation_rescue', False):
        mask_sat = False
    if mask_sat:
        bitmask |= core.DQ_SATURATED
    return bitmask


def _resolve_optimize_widths(state, params, ctx, centroids=None):
    """Resolve optimized box widths from the candidate aperture spectra."""
    opts = ctx.get('opts', {})
    width1 = params.get('extract_width')
    instrument = state.cube.meta.mode.split('/')[0].upper()
    width2 = opts.get('extract_width_soss2') if instrument == 'NIRISS' else None
    if width1 != 'optimize' and width2 != 'optimize':
        return params, ctx, None
    cube = state.cube
    if cube.data.ndim != 3:
        raise ValueError("extract_width='optimize' needs a rate cube")
    dtype = np.dtype(cube.data.dtype)
    halfwidths = _optimize_candidate_halfwidths(instrument, dtype)
    bitmask = jnp.uint32(_box_bitmask(opts, instrument))
    centroid_fn = {'NIRISS': _centroids_for_extraction,
                  'NIRSPEC': _nirspec_centroids_for_extraction}.get(
                      instrument, _miri_centroids_for_extraction)
    cen = centroids if centroids is not None else centroid_fn(state, ctx, 'sum')
    apertures = {}
    if instrument == 'NIRISS':
        o1 = jnp.asarray(np.nan_to_num(cen['ypos o1']), dtype=dtype)
        y2 = cen.get('ypos o2')
        o2 = _finite_order2_centroids(y2) if y2 is not None else None
        if width1 == 'optimize':
            apertures[1] = (o1, {'extract_end': min(int(o1.shape[0]), int(cube.data.shape[-1]))})
        if width2 == 'optimize' and o2 is not None and o2.size:
            apertures[2] = (jnp.asarray(o2, dtype), {'extract_end': int(o2.shape[0])})
    elif instrument == 'NIRSPEC':
        detector = str(ctx.get('nirspec_detector', '')).upper()
        grating = str(ctx.get('nirspec_grating', '')).upper()
        xstart = (_NIRSPEC_OPTIMIZE_XSTART.get(grating, _nirspec_xstart(ctx))
                  if detector == 'NRS1' else 0)
        apertures[1] = (jnp.asarray(np.asarray(cen['ypos']), dtype=dtype),
                        {'extract_start': xstart})
    else:
        start, end = miri_extract_bounds(cen)
        apertures[1] = (jnp.asarray(np.asarray(cen['xpos']), dtype=dtype),
                        {'extract_start': start, 'extract_end': end})

    def sweep(data, dq):
        """Extract every optimized aperture from one integration batch."""
        data = jnp.where((dq.astype(jnp.uint32) & bitmask) != 0, jnp.nan, data)
        if instrument not in ('NIRISS', 'NIRSPEC'):
            data = jnp.swapaxes(data, -1, -2)
        return {order: k_ext.box_white_light_sweep(data, track, halfwidths, **bounds)
                for order, (track, bounds) in apertures.items()}

    wlc = core.map_over_ints(sweep, (cube.data, cube.dq), cube.data.shape[0])
    params, selection = dict(params), {}
    if width1 == 'optimize':
        params['extract_width'], selection[1] = _select_optimize_width(wlc[1], instrument)
    if width2 == 'optimize':
        if 2 in wlc:
            chosen, selection[2] = _select_optimize_width(wlc[2], instrument)
        else:
            # Choose the first width when the order-2 trace is empty.
            chosen = float(_OPTIMIZE_WIDTHS[instrument][0])
        ctx = dict(ctx, opts=dict(opts, extract_width_soss2=chosen))
    return params, ctx, selection


def _attach_width_selection(aux, selection):
    """Attach the optimized aperture widths and their scatter curves."""
    if selection:
        aux['extract_width_selected'] = {
            order: item['width'] for order, item in selection.items()}
        aux['extract_width_optimization'] = selection
    return aux


def _extract_apertures(cube, ctx, mode, apertures, extract_widths, *, transpose=False,
                       group=None, instrument='NIRISS'):
    """Extract all orders with shared DQ masking, batching, and candidate stacking."""
    if cube.data.ndim == 4:
        def extract_chunk(data, groupdq):
            """Extract the selected ramp group for every aperture."""
            if group is not None:
                data, groupdq = data[:, group], groupdq[:, group]
            if transpose:
                data, groupdq = (jnp.swapaxes(a, -1, -2) for a in (data, groupdq))
            return {order: (k_ext.extract_from_stage1(data, groupdq, track, width, **bounds),
                            None) for order, (track, width, bounds) in apertures.items()}

        return core.map_over_ints(extract_chunk, (cube.data, cube.groupdq), cube.data.shape[0])
    bitmask = _box_bitmask(ctx['opts'], instrument) if mode == 'sum' else np.uint32(0)

    def extract_chunk(data, err, dq):
        """Mask rate DQ and extract every aperture in one integration batch."""
        dq = dq.astype(jnp.uint32)
        bad = (dq & jnp.uint32(bitmask)) != 0 if mode == 'sum' else dq > 0
        data, err = (jnp.where(bad, jnp.nan, a) for a in (data, err))
        if transpose:
            data, err = (jnp.swapaxes(a, -1, -2) for a in (data, err))
        results = {}
        for order, (track, width, bounds) in apertures.items():
            def extract(halfwidth):
                """Extract one aperture width from the masked rates."""
                return k_ext.box_extract(data, err, track, halfwidth, mode=mode, **bounds)

            results[order] = (extract(width) if extract_widths is None else
                              _stack_candidate_apertures(extract, width))
        return results

    return core.map_over_ints(extract_chunk, (cube.data, cube.err, cube.dq), cube.data.shape[0])


def _aperture_halfwidth(params, cube, extract_widths):
    """Resolve one box half-width or the candidate half-width array."""
    dtype = np.dtype(cube.data.dtype)
    if extract_widths is None:
        return _extract_halfwidth(params['extract_width'], dtype)
    if cube.data.ndim != 3:
        raise NotImplementedError('extract_widths is supported for rate cubes only')
    return _candidate_halfwidths(extract_widths, dtype)


def extract_orders(state, params, ctx, mode='nanaware', extract_widths=None):
    """Extract spectra from the current ramp or rate cube.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.
    mode : str
        Use nanaware for optimizer spectra or sum for production aperture sums.
    extract_widths : None, array-like(float)
        Candidate box widths for rate cubes; adds a candidate axis at position 1.

    Returns
    -------
    spectra : dict
        Flux and optional uncertainty arrays keyed by spectral order.
    """
    instrument = state.cube.meta.mode.split('/')[0].upper()
    if extract_widths is not None and state.cube.data.ndim != 3:
        raise NotImplementedError('extract_widths is supported for rate cubes only')
    if instrument == 'NIRSPEC':
        return _extract_orders_nirspec(state, params, ctx, mode, extract_widths=extract_widths)
    if instrument == 'MIRI':
        return _extract_orders_miri(state, params, ctx, mode, extract_widths=extract_widths)
    cen = _centroids_for_extraction(state, ctx, mode)
    cube = state.cube
    dtype = np.dtype(cube.data.dtype)
    if extract_widths is None:
        halfwidth = _extract_halfwidth(params['extract_width'], dtype)
    else:
        halfwidth = _candidate_halfwidths(extract_widths, dtype)
    o2_width = ctx.get('opts', {}).get('extract_width_soss2')
    o2_halfwidth = (halfwidth if o2_width is None else _extract_halfwidth(o2_width, dtype))
    if extract_widths is not None and o2_width is not None:
        o2_halfwidth = _broadcast_candidate_halfwidth(o2_halfwidth, halfwidth)
    o1 = jnp.asarray(np.nan_to_num(cen['ypos o1']), dtype=dtype)
    y2 = cen.get('ypos o2')
    o2 = _finite_order2_centroids(y2) if y2 is not None else None
    xmax = 0 if o2 is None else int(o2.size)
    o2 = None if not xmax else jnp.asarray(o2, dtype=dtype)

    apertures = {1: (o1, halfwidth, {})}
    if o2 is not None:
        apertures[2] = (o2, o2_halfwidth, {'extract_end': xmax})
    return _extract_apertures(cube, ctx, mode, apertures, extract_widths)


def _cost_from_order_arrays(spectra, opts, baseline_ints):
    """Calculate the spectral cost after joining any SOSS orders."""
    if 2 in spectra:
        wave1, flux1 = spectra[1]
        wave2, flux2 = spectra[2]
        flux, wave = k_cost.stitch_soss_orders(
            flux1, np.asarray(wave1), flux2, np.asarray(wave2))
    else:
        wave, flux = spectra[1]
    return k_cost.cost_function(
        flux, wave=np.asarray(wave),
        baseline_ints=np.atleast_1d(np.asarray(baseline_ints)),
        wave_range=opts.get('wave_range'),
        w1=opts.get('w1', 0.), w2=opts.get('w2', 1.))


def _optimizer_cost_from_last_group(data, groupdq, params, ctx, meta, centroids):
    """Score one corrected single-group image cube without a 4-D state."""
    dtype = np.dtype(data.dtype)
    halfwidth = _extract_halfwidth(params['extract_width'], dtype)
    if str(getattr(meta, 'mode', '')).split('/')[0].upper() == 'MIRI':
        start, end = miri_extract_bounds(centroids)
        xpos = jnp.asarray(np.asarray(centroids['xpos']), dtype=dtype)
        flux = k_ext.extract_from_stage1(
            jnp.swapaxes(data, -1, -2), jnp.swapaxes(groupdq, -1, -2), xpos,
            halfwidth, extract_start=start, extract_end=end)
        wave = np.arange(flux.shape[-1], dtype=float)
        return _cost_from_order_arrays({1: (wave, flux)}, ctx['opts'], meta.baseline_ints)
    if 'ypos o1' not in centroids and 'ypos' in centroids:
        # Use pixel wavelengths and the trace start for intermediate NIRSpec spectra.
        xstart = int(ctx.get('nirspec_xstart', 0))
        ypos = jnp.asarray(np.asarray(centroids['ypos']), dtype=dtype)
        flux = k_ext.extract_from_stage1(data, groupdq, ypos, halfwidth, extract_start=xstart)
        wave = np.arange(flux.shape[-1], dtype=float)
        return _cost_from_order_arrays({1: (wave, flux)}, ctx['opts'], meta.baseline_ints)
    configured_o2 = ctx.get('opts', {}).get('extract_width_soss2')
    o2_halfwidth = (halfwidth if configured_o2 is None else
                    _extract_halfwidth(configured_o2, dtype))
    o1 = jnp.asarray(np.nan_to_num(centroids['ypos o1']), dtype=dtype)
    flux1 = k_ext.extract_from_stage1(data, groupdq, o1, halfwidth)
    arrays = {1: (np.asarray(ctx['waves'][1])[:flux1.shape[-1]], flux1),}
    y2 = centroids.get('ypos o2')
    if y2 is not None:
        o2 = _finite_order2_centroids(y2)
        if o2.size:
            o2 = jnp.asarray(o2, dtype=dtype)
            flux2 = k_ext.extract_from_stage1(
                data, groupdq, o2, o2_halfwidth, extract_end=int(o2.size))
            arrays[2] = (np.asarray(ctx['waves'][2])[:flux2.shape[-1]], flux2)
    return _cost_from_order_arrays(arrays, ctx['opts'], meta.baseline_ints)


# Keep candidate batching opt-in because it can alter aperture-reduction rounding.
SCORER_BATCH_AXIS = os.environ.get('EXOTEDRF_SCORER_BATCH', '0') in (
    '1', 'true', 'on', 'yes')


def _prepared_scorer_fits(cube, *, buffers):
    """Bound resident scorer caches before any complete device conversion."""
    shape = cube.data.shape[:1] + cube.data.shape[-2:]
    bytes_per_cube = int(np.prod(shape)) * np.dtype(cube.data.dtype).itemsize
    batch = 16 if SCORER_BATCH_AXIS else 1
    required = bytes_per_cube * int(buffers) * batch
    budget = int(core.device_memory_bytes() * .5)
    if jax.default_backend() != 'gpu':
        budget = min(budget, core.host_allocation_available_bytes() // 2)
    return required <= budget


def _float_key(value):
    """Convert a scalar candidate to a hashable float cache key."""
    return float(np.asarray(value).reshape(()))


def _cached_score(cache, key, evaluate, *args, started=None):
    """Return a cached score or measure and store one candidate evaluation."""
    cached = cache.get(key)
    if cached is not None:
        return cached[0], np.array(cached[1], copy=True), 0.
    if started is None:
        started = time.perf_counter()
    cost, scatter = jax.device_get(evaluate(*args))
    duration = time.perf_counter() - started
    result = float(np.asarray(cost)), np.asarray(scatter)
    cache[key] = (result[0], np.array(result[1], copy=True))
    return result[0], result[1], duration


def _batched_axis_eval(values, key_fn, cache, device_batch_fn, dtype):
    """Score uncached candidates in batches and preserve their configured order."""
    results = [None] * len(values)
    order = []
    value_for_key = {}
    indices_for_key = {}
    for index, value in enumerate(values):
        key = key_fn(value)
        cached = cache.get(key)
        if cached is not None:
            cost, scatter = cached
            results[index] = (cost, np.array(scatter, copy=True), 0.)
            continue
        if key not in indices_for_key:
            indices_for_key[key] = []
            value_for_key[key] = value
            order.append(key)
        indices_for_key[key].append(index)

    chunk_max = 16
    for start in range(0, len(order), chunk_max):
        chunk_keys = order[start:start + chunk_max]
        chunk_values = [value_for_key[key] for key in chunk_keys]
        started = time.perf_counter()
        candidates = jnp.asarray(chunk_values, dtype=dtype)
        costs, scatters = device_batch_fn(candidates)
        costs, scatters = jax.device_get((costs, scatters))
        duration = time.perf_counter() - started
        per_candidate = duration / len(chunk_values)
        for slot, key in enumerate(chunk_keys):
            cost = float(np.asarray(costs[slot]))
            scatter = np.asarray(scatters[slot])
            cache[key] = (cost, np.array(scatter, copy=True))
            idxs = indices_for_key[key]
            results[idxs[0]] = (cost, scatter, per_candidate)
            for extra in idxs[1:]:
                results[extra] = (cost, np.array(scatter, copy=True), 0.)
    return results


class PreparedOneOverFGroupScorer:
    """Score SOSS achromatic 1/f mask widths from the final group."""

    # Vary the inert outer width before the inner width to reuse cached scores.
    grid_order = ('soss_outer_mask_width', 'soss_inner_mask_width')

    def __init__(self, state, params, ctx):
        """Prepare reusable arrays and candidate caches."""
        cube = state.cube
        prepared = _prepare_oneoverf_grp(state, ctx)
        self.ctx = ctx
        self.meta = cube.meta
        self.opts = ctx['opts']
        self.dtype = np.dtype(cube.data.dtype)
        self.data = jnp.asarray(cube.data[:, -1])
        self.groupdq = jnp.asarray(cube.groupdq[:, -1])
        self.base_mask = jnp.asarray(prepared['base_mask'])
        self.deep = jnp.asarray(prepared['deep'][-1], dtype=self.dtype)
        self.o1 = jnp.asarray(prepared['centroid_o1'], dtype=self.dtype)
        self.o2_padded = jnp.asarray(prepared['centroid_o2_padded'], dtype=self.dtype)
        self.o3_padded = jnp.asarray(prepared['centroid_o3_padded'], dtype=self.dtype)
        self.background = (None if prepared['background'] is None else
                           jnp.asarray(prepared['background'][-1], dtype=self.dtype))
        series = []
        for seg, int_start in zip(_segment_slices(cube), cube.meta.segment_int_starts):
            series.append(_series_segment(prepared['timeseries'], cube, seg, int_start))
        self.timeseries = jnp.asarray(np.concatenate(series), dtype=self.dtype)
        self.extract_width = _scorer_extract_width(params['extract_width'], self.dtype)
        self.centroids = {'ypos o1': np.asarray(prepared['centroid_o1']),}
        if prepared['centroid_o2_extract'] is not None:
            self.centroids['ypos o2'] = np.asarray(prepared['centroid_o2_extract'])
        self._result_cache = {}

    _width_key = staticmethod(_float_key)

    def _evaluate_host(self, inner_width):
        """Calculate or retrieve the cost, scatter, and elapsed time for one candidate."""
        key = self._width_key(inner_width)
        return _cached_score(self._result_cache, key, self._evaluate_width_device, inner_width)

    def _evaluate_width_device(self, inner_width):
        """Calculate the cost and scatter for one mask width on the device."""
        inner_width = jnp.asarray(inner_width, dtype=self.dtype)
        mask = k_oof.build_soss_achromatic_mask(
            self.base_mask, self.o1, self.o2_padded, self.o3_padded, inner_width)
        corrected = k_oof.oneoverf_scale_achromatic(self.data, self.deep, mask, self.timeseries,
            even_odd_rows=bool(self.opts.get('oof_even_odd_rows_grp', True)),
            background=self.background)
        score_params = {'extract_width': self.extract_width}
        return _optimizer_cost_from_last_group(
            corrected, self.groupdq, score_params, self.ctx, self.meta, self.centroids)

    def evaluate_candidates(self, parameter, candidates, params):
        """Score each candidate in its configured order.

        Parameters
        ----------
        parameter : str
            Name of the parameter to vary.
        candidates : array-like(float)
            Candidate values in their configured order.
        params : dict
            Current values of the other reduction parameters.

        Returns
        -------
        results : list[tuple]
            Cost, wavelength scatter, and elapsed seconds for each candidate.
        """
        values = list(candidates)
        if not values:
            return []
        if parameter == 'soss_outer_mask_width':
            # Report each outer-width candidate even though its correction is identical.
            cost, scatter, duration = self._evaluate_host(params['soss_inner_mask_width'])
            return [(float(np.asarray(cost)), np.asarray(scatter), duration if index == 0 else 0.)
                    for index, _ in enumerate(values)]
        if parameter != 'soss_inner_mask_width':
            raise KeyError(parameter)

        if SCORER_BATCH_AXIS:
            return _batched_axis_eval(values, self._width_key, self._result_cache,
                jax.vmap(self._evaluate_width_device), self.dtype)

        results = []
        for value in values:
            results.append(self._evaluate_host(value))
        return results


def prepare_oneoverf_grp_scorer(state, params, ctx):
    """Prepare a final-group scorer for achromatic SOSS 1/f correction.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    scorer : None, PreparedOneOverFGroupScorer, PreparedOneOverFSolveScorer
        Reusable candidate scorer, or None when its requirements are not met.
    """
    solve = _is_oof_solve(ctx)
    method = ctx.get('opts', {}).get('oof_method', 'scale-achromatic')
    if not solve and method != 'scale-achromatic':
        return None
    if not isinstance(state.cube, RampCube) or state.cube.data.ndim != 4:
        return None
    if not _prepared_scorer_fits(state.cube, buffers=16):
        return None
    scorer = PreparedOneOverFSolveScorer if solve else PreparedOneOverFGroupScorer
    return scorer(state, params, ctx)


class PreparedJumpScorer:
    """Score time-domain jump settings using the selected ramp group."""

    # Reuse each temporal-window median across threshold candidates.
    grid_order = ('time_window', 'time_jump_threshold')

    def __init__(self, state, params, ctx):
        """Prepare reusable arrays and candidate caches."""
        cube = state.cube
        self.ctx = ctx
        self.meta = cube.meta
        self.dtype = np.dtype(cube.data.dtype)
        self.centroids = _centroids_for_extraction(state, ctx, 'nanaware')
        self.records = []
        instrument = cube.meta.mode.split('/')[0].upper()
        max_reset_int = _reset_max_int(cube.meta)
        groupdq = cube.groupdq
        if _upramp_runs(ctx.get('opts', {}), cube.data.shape[1]):
            # Compute the fixed up-ramp flags before varying time-domain parameters.
            groupdq, _ = _apply_upramp_jump(cube, ctx)
        group = -1
        if instrument == 'MIRI':
            group = miri_score_group(cube.data, groupdq, params, ctx, cube.meta, self.centroids)
        self.score_group = group % cube.data.shape[1]
        host_data = np.asarray(jax.device_get(cube.data[:, group]))
        host_dq = np.asarray(jax.device_get(groupdq[:, group]))
        for seg, int_start in zip(_segment_slices(cube), cube.meta.segment_int_starts):
            image = host_data[seg]
            data = jnp.asarray(image[:, None])
            groupdq = jnp.asarray(host_dq[seg, :, :][:, None])
            floor = np.asarray([hoststats.nanpercentile(image, 10.)], dtype=self.dtype)
            invariants = k_jump.prepare_jump_invariants(data, flux_floor=jnp.asarray(floor))
            artifact = jnp.asarray(k_jump.reset_artifact_mask(
                seg.stop - seg.start, image.shape[-2], image.shape[-1],
                int_start=int(int_start), instrument=instrument, max_reset_int=max_reset_int))
            self.records.append((data, groupdq, invariants, artifact))
        self._result_cache = {}
        self._prepared_window = {}
        self._prepared_window_value = None

    def _prepared_for_window(self, index, data, invariants, window):
        """Prepare or retrieve jump statistics for one temporal window."""
        window = int(window)
        if window != getattr(self, '_prepared_window_value', None):
            self._prepared_window.clear()
            self._prepared_window_value = window
        key = (index, int(window))
        prepared = self._prepared_window.get(key)
        if prepared is None:
            # Preserve the full time axis while bounding independent detector columns.
            configured = (os.environ.get('EXOTEDRF_JUMP_CHUNK_COLS') or
                          os.environ.get('EXOTEDRF_CHUNK_COLS'))
            ncols = int(data.shape[-1])
            if configured is not None:
                width = int(configured)
                if width < 1:
                    raise ValueError('Jump column chunk must be positive')
            else:
                width = min(512, core.auto_chunk(ncols, data.nbytes // ncols,
                    n_buffers=max(24, 5 * int(window) + 3), headroom=0.4))
            parts = []
            for lo in range(0, ncols, width):
                sl = slice(lo, min(lo + width, ncols))
                local = k_jump.JumpInvariants(invariants.scatter[..., sl], invariants.flux_floor)
                part = k_jump.prepare_jumps_in_time(
                    data[..., sl], window=int(window), invariants=local)
                # Release sort workspace before processing the next column section.
                ready = getattr(part.scale, 'block_until_ready', None)
                if ready is not None:
                    ready()
                parts.append(part)
            prepared = k_jump.PreparedTimeJump(jnp.concatenate([p.scale for p in parts], axis=-1),
                jnp.concatenate([p.cube_filt for p in parts], axis=-1), invariants.flux_floor)
            self._prepared_window[key] = prepared
        return prepared

    def _evaluate_device(self, threshold, window, extract_width):
        """Calculate the cost and scatter for one candidate on the device."""
        images, dqs = [], []
        jump_bit = jnp.uint8(int(core.DQ_JUMP_DET))
        for index, (data, groupdq, invariants, artifact) in enumerate(self.records):
            prepared = self._prepared_for_window(index, data, invariants, window)
            flagged = ((prepared.scale >= jnp.asarray(threshold, dtype=self.dtype)) &
                (data > prepared.flux_floor[None, :, None, None]) & ~artifact[:, None])
            already = (groupdq & jump_bit) != 0
            newdq = jnp.where(flagged & ~already, groupdq | jump_bit, groupdq).astype(jnp.uint8)
            images.append(data[:, 0])
            dqs.append(newdq[:, 0])
        image = jnp.concatenate(images, axis=0)
        dq = jnp.concatenate(dqs, axis=0)
        return _optimizer_cost_from_last_group(
            image, dq, {'extract_width': extract_width}, self.ctx, self.meta, self.centroids)

    def _evaluate_host(self, threshold, window, extract_width):
        """Calculate or retrieve the cost, scatter, and elapsed time for one candidate."""
        key = (float(np.asarray(threshold)), int(window), _width_cache_key(extract_width))
        return _cached_score(self._result_cache, key, self._evaluate_device,
                             threshold, window, extract_width)

    def evaluate_candidates(self, parameter, candidates, params):
        """Score each candidate in its configured order.

        Parameters
        ----------
        parameter : str
            Name of the parameter to vary.
        candidates : array-like(float)
            Candidate values in their configured order.
        params : dict
            Current values of the other reduction parameters.

        Returns
        -------
        results : list[tuple]
            Cost, wavelength scatter, and elapsed seconds for each candidate.
        """
        values = list(candidates)
        if parameter == 'time_jump_threshold':
            window = params['time_window']
            extract_width = params['extract_width']
            if SCORER_BATCH_AXIS and values:
                device_batch_fn = jax.vmap(
                    lambda value: self._evaluate_device(value, window, extract_width))

                def key_fn(value):
                    """Return the result-cache key for one candidate."""
                    return (float(np.asarray(value)), int(window), _width_cache_key(extract_width))

                results = _batched_axis_eval(values, key_fn, self._result_cache, device_batch_fn,
                    self.dtype)
            else:
                results = [self._evaluate_host(value, window, extract_width) for value in values]
        elif parameter == 'time_window':
            # Reuse scatter and flux-floor statistics across temporal-window candidates.
            results = []
            for value in values:
                self._prepared_window.clear()
                results.append(self._evaluate_host(params['time_jump_threshold'], value,
                    params['extract_width']))
            self._prepared_window.clear()
        else:
            raise KeyError(parameter)
        return results


def prepare_jump_scorer(state, params, ctx):
    """Prepare a final-group scorer when time-domain jump flagging is independent.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    scorer : None, PreparedJumpScorer
        Reusable candidate scorer, or None when its requirements are not met.
    """
    opts = ctx.get('opts', {})
    cube = state.cube
    if not isinstance(cube, RampCube) or cube.data.ndim != 4:
        return None
    if cube.data.shape[1] <= 2:
        return None
    if not _time_jump_runs(opts, cube.data.shape[1]):
        return None
    if not _prepared_scorer_fits(cube, buffers=16):
        return None
    return PreparedJumpScorer(state, params, ctx)


# Bound the retained detector-sized temporal-noise planes.
_BADPIX_SCATTER_CACHE_KEYS = 8


class _SpatialRecords(tuple):
    """Store spatial section records with their box-size and threshold cache key."""
    key = None


class PreparedBadPixScorer:
    """Score BadPix box, spatial, time-window, and temporal thresholds."""

    final_consumer_only = True

    # Vary the shared spatial parameters before the temporal window and threshold.
    grid_order = ('box_size', 'space_outlier_threshold', 'window_size', 'time_outlier_threshold')

    def __init__(self, state, params, ctx):
        """Prepare reusable arrays and candidate caches."""
        cube = state.cube
        self.ctx = ctx
        self.meta = cube.meta
        self.instrument = cube.meta.mode.split('/')[0].upper()
        self.dtype = np.dtype(cube.data.dtype)
        self.data = jnp.asarray(cube.data)
        self.dq = jnp.asarray(cube.dq, dtype=jnp.uint32)
        self.saturated = ((self.dq & jnp.uint32(core.DQ_SATURATED)) != 0)
        self.saturated_any = jnp.any(self.saturated, axis=0)
        self.nints, self.dimy, self.dimx = (int(value) for value in self.data.shape)

        baseline = baseline_bool_for_meta(cube.meta, self.nints)
        self.deep = jnp.asarray(_host_deepstack(cube.data, baseline), dtype=self.dtype)
        centroids = _centroids_for_extraction(state, ctx, 'nanaware')
        self.miri_bounds = None
        if self.instrument == 'MIRI':
            # Assemble corrected MIRI columns before extracting the transposed aperture.
            self.miri_bounds = miri_extract_bounds(centroids)
            self.extract_start = 0
            self.o1 = jnp.asarray(np.asarray(centroids['xpos']), dtype=self.dtype)
            y2 = None
        elif 'ypos o1' not in centroids and 'ypos' in centroids:
            # Clamp the NIRSpec centroid track and retain zeros before the trace start.
            self.extract_start = int(ctx.get('nirspec_xstart', 0))
            self.o1 = jnp.asarray(_nirspec_clamped_track(
                centroids, self.dimx, self.extract_start, self.dtype))
            y2 = None
        else:
            self.extract_start = 0
            self.o1 = jnp.asarray(np.nan_to_num(centroids['ypos o1']), dtype=self.dtype)
            y2 = centroids.get('ypos o2')
        compact_o2 = (_finite_order2_centroids(y2) if y2 is not None else np.asarray([]))
        self.o2 = (None if not compact_o2.size else jnp.asarray(compact_o2, dtype=self.dtype))
        self.o2_end = 0 if self.o2 is None else int(self.o2.size)
        self.halfwidth = _extract_halfwidth(params['extract_width'], self.dtype)
        configured_o2 = ctx.get('opts', {}).get('extract_width_soss2')
        self.o2_halfwidth = (self.halfwidth if configured_o2 is None else
            _extract_halfwidth(configured_o2, self.dtype))

        waves = ctx.get('waves') or {}
        n_channels = self.dimy if self.instrument == 'MIRI' else self.dimx
        self.wave1 = np.asarray(waves.get(1, np.arange(n_channels, dtype=float)))[:n_channels]
        self.wave2 = (None if self.o2 is None else np.asarray(
            waves.get(2, np.arange(self.dimx, dtype=float)))[:self.dimx])

        self._layouts = {}
        self._spatial_prepared = {}
        self._spatial_key = None
        self._spatial_records = None
        self._result_cache = {}
        # Cache one box-median cube and a bounded set of temporal-scatter planes.
        self._boxmed_key = None
        self._boxmed = None
        self._cache_box_medians = _prepared_scorer_fits(cube, buffers=18)
        self._scatter_cache = {}
        bp_opts = ctx.get('opts', {})
        self.median_hv = bp_opts.get('median_high_variance') is True
        self.preserve_sat = bp_opts.get('preserve_saturated') is True
        self.clear_dq = bp_opts.get('clear_interpolated_dq') is True
        self._temporal_key = None

    _float_key = staticmethod(_float_key)

    def _layout(self, box_size):
        """Calculate the detector-column sections and halos for one box size."""
        box_size = int(box_size)
        cached = self._layouts.get(box_size)
        if cached is not None:
            return cached
        halo = max(5, box_size + 1 if self.instrument != 'MIRI' else 0)
        chunk = _badpix_column_chunk_size(self.data, box_size, halo=halo)
        layout = []
        for lo in range(0, self.dimx, chunk):
            hi = min(lo + chunk, self.dimx)
            read_lo = max(0, lo - halo)
            read_hi = min(self.dimx, hi + halo)
            layout.append((lo, hi, read_lo, read_hi, lo - read_lo, lo - read_lo + hi - lo))
        cached = tuple(layout)
        self._layouts[box_size] = cached
        return cached

    def _prepare_spatial(self, box_size):
        """Prepare spatial statistics for one box size."""
        box_size = int(box_size)
        cached = self._spatial_prepared.get(box_size)
        if cached is None:
            cached = k_badpix.prepare_spatial_badpix(self.deep, self.dq, self.saturated_any,
                box_size=box_size, instrument=self.instrument, preserve_saturated=self.preserve_sat)
            self._spatial_prepared[box_size] = cached
        return cached

    def _clipped_section(self, layout):
        """Clip negative samples when discovering spatial defects in a column section."""
        _lo, _hi, read_lo, read_hi, _trim_lo, _trim_hi = layout
        source = self.data[..., read_lo:read_hi]
        return jnp.where(source < 0, 0., source)

    def _box_medians(self, box_size):
        """Calculate and cache interpolation-box medians for one box size."""
        box_size = int(box_size)
        if self._boxmed_key == box_size and self._boxmed is not None:
            return self._boxmed
        medians = tuple(k_badpix.spatial_interp_median(
                self._clipped_section(layout), box_size=box_size, instrument=self.instrument)
            for layout in self._layout(box_size))
        if self._cache_box_medians:
            self._boxmed_key = box_size
            self._boxmed = medians
        return medians

    def _tag_records(self, records, box_size, space_thresh):
        """Attach the box size and spatial threshold to the section records."""
        tagged = _SpatialRecords(records)
        tagged.key = (int(box_size), self._float_key(space_thresh))
        return tagged

    def _build_spatial_records(self, box_size, space_thresh):
        """Apply the spatial correction to each detector-column section."""
        box_size = int(box_size)
        spatial_map = k_badpix.spatial_badpix_from_prepared(*self._prepare_spatial(box_size),
            space_thresh=jnp.asarray(space_thresh, dtype=self.dtype))
        records = []
        layout = self._layout(box_size)
        medians = (self._box_medians(box_size) if self._cache_box_medians else
                   (None,) * len(layout))
        for section, boxmed in zip(layout, medians):
            _lo, _hi, read_lo, read_hi, _trim_lo, _trim_hi = section
            inputs = (self._clipped_section(section), self.dq[..., read_lo:read_hi],
                      spatial_map[..., read_lo:read_hi])
            if boxmed is None:
                spatial_data, spatial_dq = k_badpix.apply_spatial_badpix(
                    *inputs, box_size=box_size, instrument=self.instrument,
                    clear_interpolated_dq=self.clear_dq)
            else:
                spatial_data, spatial_dq = k_badpix.apply_spatial_badpix_from_median(
                    *inputs, boxmed, clear_interpolated_dq=self.clear_dq)
            records.append((section, spatial_data, spatial_dq,
                            self.saturated[..., read_lo:read_hi]))
        return self._tag_records(records, box_size, space_thresh)

    def _spatial_for(self, box_size, space_thresh):
        """Get or rebuild the spatial records for one candidate."""
        key = (int(box_size), self._float_key(space_thresh))
        if key != self._spatial_key:
            self._spatial_records = self._build_spatial_records(*key)
            self._spatial_key = key
        return self._spatial_records

    def _scatter_for(self, spatial_records, key):
        """Calculate and cache temporal-scatter planes for one spatial correction."""
        cached = self._scatter_cache.get(key)
        if cached is not None:
            return cached
        # Trim duplicated halos before applying detector-wide scatter replacement.
        raw_scatter = [k_badpix.temporal_scatter(record[1]) for record in spatial_records]
        detector_scatter = jnp.concatenate([sigma[..., record[0][4]:record[0][5]]
            for record, sigma in zip(spatial_records, raw_scatter)], axis=-1)
        box, space = spatial_records.key
        _, _, plane = k_badpix.noise_plane_spatial(
            detector_scatter, jnp.asarray(space, dtype=self.dtype),
            box_size=int(box), instrument=self.instrument)
        sigmas = tuple(plane[..., record[0][2]:record[0][3]] for record in spatial_records)
        if key is not None:
            self._scatter_cache[key] = sigmas
            while len(self._scatter_cache) > _BADPIX_SCATTER_CACHE_KEYS:
                self._scatter_cache.pop(next(iter(self._scatter_cache)))
        return sigmas

    def _prepare_temporal_records(self, spatial_records, window_size, key=None):
        """Prepare the temporal running median for one window size."""
        window_size = int(window_size)
        self._temporal_key = spatial_records.key
        self._temporal_window = window_size
        records = []
        for (layout, spatial_data, spatial_dq, saturated), sigma in zip(
                spatial_records, self._scatter_for(spatial_records, key)):
            cube_filt, std_dev = k_badpix.prepare_temporal_badpix(
                spatial_data, window_size=window_size, instrument=self.instrument, scatter=sigma)
            records.append((layout, spatial_data, spatial_dq, saturated, cube_filt, std_dev))
        return tuple(records)

    def _pass0_chunk(self, record, time_thresh):
        """Apply temporal replacement to one detector-column section."""
        (layout, spatial_data, spatial_dq, saturated, cube_filt, std_dev) = record
        return k_badpix.apply_temporal_badpix(
            spatial_data, spatial_dq, saturated, cube_filt, std_dev,
            time_thresh=jnp.asarray(time_thresh, dtype=self.dtype),
            preserve_saturated=self.preserve_sat, clear_interpolated_dq=self.clear_dq)

    def _extract_corrected_chunk(self, record, pass0, high_variance, window_size):
        """Finalize a corrected section and extract its aperture spectra."""
        (layout, _spatial_data, _spatial_dq, saturated, _cube_filt, _std_dev) = record
        lo, hi, read_lo, read_hi, trim_lo, trim_hi = layout
        corrected, corrected_dq = pass0
        cube_filt = k_badpix.temporal_filter(
            corrected, window_size=window_size, instrument=self.instrument)
        corrected = k_badpix.apply_high_variance(corrected, high_variance[..., read_lo:read_hi],
            jnp.any(saturated, axis=0), median_high_variance=self.median_hv,
            preserve_saturated=self.preserve_sat)
        # Use a scalar error placeholder because optimizer extraction ignores uncertainties.
        corrected, _, corrected_dq = k_badpix.finalize_badpix(
            corrected, jnp.zeros((), dtype=self.dtype), corrected_dq,
            cube_filt, saturated, instrument=self.instrument, clear_interpolated_dq=self.clear_dq)

        if self.instrument == 'NIRISS' and self.dimx == 2048 and \
                self.dimy >= 96 and read_hi > 2018:
            start = max(2018, read_lo) - read_lo
            mm = jnp.nanmedian(jnp.concatenate([corrected[:, 82:84, start:],
                 corrected[:, 88:90, start:]], axis=1), axis=1)
            corrected = corrected.at[:, 84:88, start:].set(mm[:, None, :])

        corrected = corrected[..., trim_lo:trim_hi]
        corrected_dq = corrected_dq[..., trim_lo:trim_hi]
        masked = jnp.where(corrected_dq > 0, jnp.nan, corrected)
        if self.miri_bounds is not None:
            # Defer MIRI extraction until all cross-dispersion columns are assembled.
            return masked, None
        flux1, _ = k_ext.box_extract(masked, None, self.o1[lo:hi], self.halfwidth, mode='nanaware')

        flux2 = None
        if self.o2 is not None and lo < self.o2_end:
            o2_hi = min(hi, self.o2_end)
            local_width = o2_hi - lo
            flux2, _ = k_ext.box_extract(masked[..., :local_width], None, self.o2[lo:o2_hi],
                self.o2_halfwidth, mode='nanaware')
        return flux1, flux2

    def _score_device(self, temporal_records, time_thresh):
        """Calculate the bad-pixel candidate cost and scatter on the device."""
        flux1_parts, flux2_parts = [], []
        pass0 = [self._pass0_chunk(record, time_thresh) for record in temporal_records]
        box, space = self._temporal_key
        plane = jnp.concatenate([k_badpix.temporal_std(p0[0])[..., record[0][4]:record[0][5]]
            for record, p0 in zip(temporal_records, pass0)], axis=-1)
        high_variance, _, _ = k_badpix.noise_plane_spatial(
            plane, jnp.asarray(space, dtype=self.dtype),
            box_size=int(box), instrument=self.instrument)
        window = self._temporal_window
        for record, p0 in zip(temporal_records, pass0):
            flux1, flux2 = self._extract_corrected_chunk(record, p0, high_variance, window)
            flux1_parts.append(flux1)
            if flux2 is not None:
                flux2_parts.append(flux2)
        if self.miri_bounds is not None:
            start, end = self.miri_bounds
            masked = jnp.concatenate(flux1_parts, axis=-1)
            flux1, _ = k_ext.box_extract(
                jnp.swapaxes(masked, -1, -2), None, self.o1, self.halfwidth,
                extract_start=start, extract_end=end, mode='nanaware')
            return _cost_from_order_arrays({1: (self.wave1, flux1)}, self.ctx['opts'],
                self.meta.baseline_ints)
        flux1 = jnp.concatenate(flux1_parts, axis=-1)
        if self.extract_start:
            # Retain zero flux before the NIRSpec trace start.
            columns = jnp.arange(self.dimx)
            flux1 = jnp.where(columns[None, :] < self.extract_start,
                              jnp.zeros((), dtype=flux1.dtype), flux1)
        arrays = {1: (self.wave1, flux1)}
        if self.o2 is not None:
            flux2 = jnp.concatenate(flux2_parts, axis=-1)
            # Keep the full order-2 scatter axis, including columns beyond the compact trace.
            flux2 = jnp.pad(flux2, ((0, 0), (0, self.dimx - self.o2_end)))
            arrays[2] = (self.wave2, flux2)
        return _cost_from_order_arrays(arrays, self.ctx['opts'], self.meta.baseline_ints)

    def _score_host(self, temporal_records, params, *, started=None):
        """Calculate or retrieve the bad-pixel candidate cost, scatter, and elapsed time."""
        key = (self._float_key(params['space_outlier_threshold']),
               self._float_key(params['time_outlier_threshold']),
               int(params['box_size']), int(params['window_size']))
        return _cached_score(self._result_cache, key, self._score_device,
                             temporal_records, params['time_outlier_threshold'], started=started)

    @staticmethod
    def _is_better(cost, best_cost):
        """Check whether a finite cost improves the current minimum."""
        return np.isfinite(cost) and (best_cost is None or cost < best_cost)

    def evaluate_candidates(self, parameter, candidates, params):
        """Score each candidate in its configured order.

        Parameters
        ----------
        parameter : str
            Name of the parameter to vary.
        candidates : array-like(float)
            Candidate values in their configured order.
        params : dict
            Current values of the other reduction parameters.

        Returns
        -------
        results : list[tuple]
            Cost, wavelength scatter, and elapsed seconds for each candidate.
        """
        values = list(candidates)
        results = []
        best_cost = None
        best_key = None
        best_spatial = None

        if parameter in ('space_outlier_threshold', 'box_size'):
            for value in values:
                started = time.perf_counter()
                trial = dict(params)
                trial[parameter] = int(value) if parameter == 'box_size' else value
                box, space = int(trial['box_size']), trial['space_outlier_threshold']
                spatial_key = (box, self._float_key(space))
                spatial = self._build_spatial_records(box, space)
                temporal = self._prepare_temporal_records(
                    spatial, params['window_size'], key=spatial_key)
                result = self._score_host(temporal, trial, started=started)
                results.append(result)
                if self._is_better(result[0], best_cost):
                    best_cost, best_key, best_spatial = result[0], spatial_key, spatial

        elif parameter == 'time_outlier_threshold':
            spatial = self._spatial_for(params['box_size'], params['space_outlier_threshold'])
            started = time.perf_counter()
            temporal = self._prepare_temporal_records(
                spatial, params['window_size'], key=self._spatial_key)
            if SCORER_BATCH_AXIS and values:
                device_batch_fn = jax.vmap(lambda value: self._score_device(temporal, value))

                box = int(params['box_size'])
                window = int(params['window_size'])
                space_key = self._float_key(params['space_outlier_threshold'])

                def key_fn(value):
                    """Return the result-cache key for one candidate."""
                    return (space_key, self._float_key(value), box, window)

                results = _batched_axis_eval(values, key_fn, self._result_cache, device_batch_fn,
                    self.dtype)
            else:
                for index, value in enumerate(values):
                    trial = dict(params, time_outlier_threshold=value)
                    result = self._score_host(temporal, trial,
                        started=started if index == 0 else None)
                    results.append(result)

        elif parameter == 'window_size':
            spatial = self._spatial_for(params['box_size'], params['space_outlier_threshold'])
            spatial_key = self._spatial_key
            for value in values:
                started = time.perf_counter()
                window = int(value)
                temporal = self._prepare_temporal_records(spatial, window, key=spatial_key)
                trial = dict(params, window_size=window)
                results.append(self._score_host(temporal, trial, started=started))
        else:
            raise KeyError(parameter)

        # Retain the first finite minimum and release other candidate intermediates.
        if best_spatial is not None:
            self._spatial_key = best_key
            self._spatial_records = best_spatial
        return results


def prepare_badpix_scorer(state, params, ctx):
    """Prepare a bad-pixel scorer for a single FITS segment.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    scorer : None, PreparedBadPixScorer
        Reusable candidate scorer, or None when its requirements are not met.
    """
    cube = state.cube
    if not isinstance(cube, RateCube) or cube.data.ndim != 3:
        return None
    if len(_segment_slices(cube)) != 1:
        return None
    if not _prepared_scorer_fits(cube, buffers=16):
        return None
    return PreparedBadPixScorer(state, params, ctx)


def evaluate_optimizer_cost(state, params, ctx):
    """Calculate the optimizer cost from NaN-aware aperture spectra.

    Baseline bounds use the raw light-curve slice convention.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    cost : float
        Weighted light-curve scatter.
    scatter : np.ndarray(float)
        Scatter for each wavelength channel.
    """
    opts = ctx['opts']
    spectra = extract_orders(state, params, ctx)
    waves = ctx.get('waves') or {}
    arrays = {}
    for order, (flux, _) in spectra.items():
        wave = waves.get(order)
        if wave is None:
            wave = np.arange(flux.shape[-1], dtype=float)
        arrays[order] = (np.asarray(wave)[:flux.shape[-1]], flux)
    return _cost_from_order_arrays(
        arrays, opts, state.cube.meta.baseline_ints)


def evaluate_production_cost(state, params, ctx):
    """Calculate the cost from the final clipped spectral products.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    cost : float
        Weighted light-curve scatter.
    scatter : np.ndarray(float)
        Scatter for each wavelength channel.
    """
    del params
    products = state.aux.get('spectral_products')
    if not products:
        raise ValueError(
            'production cost requires Extract spectral_products in state.aux')
    arrays = {
        int(order): (np.asarray(product['wave']), product['flux'])
        for order, product in products.items()
    }
    return _cost_from_order_arrays(
        arrays, ctx['opts'], state.cube.meta.baseline_ints)


def evaluate_cost(state, params, ctx):
    """Calculate the cost from final products or intermediate aperture spectra.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    cost : float
        Weighted light-curve scatter.
    scatter : np.ndarray(float)
        Scatter for each wavelength channel.
    """
    if state.aux.get('spectral_products'):
        return evaluate_production_cost(state, params, ctx)
    return evaluate_optimizer_cost(state, params, ctx)


def _report_halfwidth(width):
    """Return float64 aperture half-widths for the DQ report."""
    if v2config.is_asymmetric_width(width):
        return np.asarray(v2config.extract_width_halves(width), dtype=float)
    return np.asarray(float(width) / 2.)


def _dq_report_frame(state):
    """Get the DQ frame used for the final extraction report."""
    cube = state.cube
    first = core.segment_slices(
        cube.meta, n_ints=cube.dq.shape[0])[0]
    index = first.start + k_ext.dq_report_frame_index(first.stop - first.start)
    frame = np.asarray(core.to_host(cube.dq[index]))
    # Merge high-variance flags into bit 32 of the extraction report.
    variance = state.aux.get('high_variance_map') if state.aux else None
    if variance is not None:
        variance = np.asarray(core.to_host(variance))
        plane = variance[index] if variance.ndim == 3 else variance
        frame = frame.astype(np.uint64)
        frame[np.asarray(plane, dtype=bool)] |= core.DQ_HIGH_VARIANCE
    return frame


def _soss_dq_reports(state, params, ctx, cen):
    """Build the SOSS box-aperture DQ report for each order."""
    frame = _dq_report_frame(state)
    hw1 = _report_halfwidth(params['extract_width'])
    reports = {1: k_ext.box_dq_report(
        frame, np.nan_to_num(np.asarray(cen['ypos o1'], dtype=float)), hw1)}
    y2 = cen.get('ypos o2')
    o2 = None if y2 is None else np.asarray(_finite_order2_centroids(y2))
    if o2 is not None and o2.size:
        o2_width = ctx.get('opts', {}).get('extract_width_soss2')
        hw2 = hw1 if o2_width is None else _report_halfwidth(o2_width)
        reports[2] = k_ext.box_dq_report(
            frame, o2, hw2, extract_end=int(o2.size))
    return reports


def _nirspec_dq_report(state, params, ctx, centroids, method):
    """Build the NIRSpec DQ report for box or optimal extraction."""
    frame = _dq_report_frame(state)
    xstart = _nirspec_xstart(ctx)
    ypos = np.asarray(centroids['ypos'], dtype=float)
    width = params.get('extract_width')
    if method == 'optimal':
        dimy, dimx = frame.shape
        idx = np.clip(np.arange(dimx - xstart), 0, max(ypos.shape[0] - 1, 0))
        halfwidth = None if width is None else width / 2.
        mask = np.asarray(k_ext.horne_aperture_mask(
            jnp.asarray(ypos[idx]), None if halfwidth is None else
            jnp.asarray(halfwidth), dimy))
        return k_ext.dq_report_from_mask(frame.T, mask, start=xstart)
    return k_ext.box_dq_report(
        frame, ypos, _report_halfwidth(width),
        extract_start=xstart)


def _miri_dq_report(state, params, ctx, centroids, method):
    """Build the MIRI DQ report for box or optimal extraction."""
    frame = _dq_report_frame(state)
    dimy, dimx = frame.shape
    ypos = np.asarray(centroids['ypos'], dtype=float)
    xpos = np.asarray(centroids['xpos'], dtype=float)
    width = params.get('extract_width')
    if method == 'optimal':
        start, end = int(np.min(ypos)), int(np.max(ypos)) + 1
        idx = np.clip(np.arange(end - start), 0, max(xpos.shape[0] - 1, 0))
        mask = np.asarray(k_ext.horne_aperture_mask(
            jnp.asarray(xpos[idx]), None if width is None else
            jnp.asarray(width / 2.), dimx))
        return k_ext.dq_report_from_mask(frame, mask, start=start)
    start, end = miri_extract_bounds(centroids)
    # Transpose MIRI detector rows into wavelength columns for the report.
    return k_ext.box_dq_report(
        frame.T, xpos, _report_halfwidth(width),
        extract_start=start, extract_end=end)


def _clip_lc(flux, ctx):
    """Clip the final light curves using the configured threshold."""
    thresh = float(ctx.get('opts', {}).get('clip_thresh', 10))
    return sigma_clip_lightcurves(np.asarray(flux), thresh=thresh, window=10)


def sigma_clip_lightcurves(flux, thresh=10, window=10):
    """Interpolate anomalous wavelength channels and clip temporal outliers.

    Parameters
    ----------
    flux : array-like(float)
        Flux array with integration and wavelength axes.
    thresh : float
        Outlier threshold in units of point-to-point scatter.
    window : int
        Median-filter window for the wavelength and temporal passes.

    Returns
    -------
    flux_clipped : np.ndarray(float)
        Flux with anomalous channels interpolated or masked and temporal outliers replaced.
    """
    flux = np.asarray(core.to_host(flux))
    flux_clipped = np.copy(flux)
    nwaves = flux.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scatter = np.nanstd(flux, axis=0)
        scatter = np.where(scatter == 0, np.inf, scatter)
        scatter_filt = median_filter(scatter, window)
        diff = scatter - scatter_filt
        std_dev = np.nanmedian(
            np.abs(0.5 * (diff[0:-2] + diff[2:]) - diff[1:-1]), axis=0)
        with np.errstate(all='ignore'):
            ii = np.where(np.abs(diff) / std_dev >= thresh)[0]
        for chunk in np.split(ii, np.where(np.diff(ii) != 1)[0] + 1):
            if len(chunk) == 0:
                continue
            low = np.nanmax([np.nanmin(chunk) - 1, 0])
            up = np.nanmin([np.nanmax(chunk) + 1, nwaves - 1])
            ll = len(chunk)
            if ll < 10:
                w = 1 / (ll + 1)
                for i in range(ll):
                    flux_clipped[:, chunk[i]] = np.average(
                        [flux[:, low], flux[:, up]],
                        weights=[w * (i + 1), 1 - w * (i + 1)], axis=0)
            else:
                flux_clipped[:, chunk] = np.nan

        flux_filt = median_filter(flux_clipped, (window, 1))
        edge = window // 2
        flux_filt[:edge] = np.median(flux_filt[edge:edge + window], axis=0)
        flux_filt[-edge:] = np.median(
            flux_filt[-(edge + 1 + window):-(edge + 1)], axis=0)
        std_dev = np.median(
            np.abs(0.5 * (flux_clipped[:-2] + flux_clipped[2:])
                   - flux_clipped[1:-1]), axis=0)
        std_dev = np.where(std_dev == 0, np.inf, std_dev)
        scale = np.abs(flux_clipped - flux_filt) / std_dev
        flagged = np.where(scale > thresh)
        flux_clipped[flagged] = flux_filt[flagged]
    return flux_clipped


def step_extract(state, params, ctx):
    """Create final spectra and the exact clipped light curves used for scoring.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    params, ctx, selection = _resolve_optimize_widths(state, params, ctx)
    spectra = extract_orders(state, params, ctx, mode='sum')
    aux = dict(state.aux)
    dq_reports = _soss_dq_reports(state, params, ctx, _centroids_for_extraction(state, ctx, 'sum'))
    aux['spectra'], aux['spectral_products'] = _final_spectral_products(spectra, ctx, dq_reports)
    _attach_width_selection(aux, selection)
    return PipelineState(cube=state.cube, aux=aux)


def step_extract_atoca(state, params, ctx):
    """Extract final NIRISS/SOSS spectra with the installed JWST ATOCA step.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    from exotedrf.v2 import atoca
    spectra, products, info = atoca.extract_state(
        state, params, ctx, log=ctx.get('log', print))
    aux = dict(state.aux)
    aux['spectra'] = spectra
    aux['spectral_products'] = products
    aux['atoca'] = info
    return PipelineState(cube=state.cube, aux=aux)


def _final_spectral_products(spectra, ctx, dq_reports=None):
    """Clip the extracted light curves and sort them by wavelength."""
    from exotedrf.v2 import wavecal
    # Refine wavelengths using the raw extracted flux before clipping.
    spectra, waves, wave_shift = wavecal.soss_stage3_wavelengths(spectra, ctx)
    # Reverse to increasing wavelength before applying the asymmetric median filter.
    clipped = {o: (_clip_lc(np.asarray(f)[..., ::-1], ctx)[..., ::-1],
                   None if e is None else np.asarray(e))
               for o, (f, e) in spectra.items()}
    products = {}
    for order, (flux, ferr) in clipped.items():
        wave = np.asarray(waves[order])
        if wave.ndim != 1 or wave.shape[0] != flux.shape[-1]:
            raise ValueError(
                f'order-{order} wavelength shape {wave.shape} does not match '
                f'extracted flux shape {flux.shape}')
        columns = np.flatnonzero(np.isfinite(wave))
        columns = columns[np.argsort(wave[columns], kind='stable')]
        products[order] = {
            'wave': wave[columns],
            'flux': flux[..., columns],
            'ferr': None if ferr is None else ferr[..., columns],
        }
        if wave_shift is not None:
            products[order]['wave_shift'] = wave_shift
        if dq_reports is not None and order in dq_reports:
            report = np.asarray(dq_reports[order])
            if report.shape[0] == wave.shape[0]:
                products[order]['dq'] = report[columns]
    return clipped, products


def prepared_extract_results(state, params, ctx, candidates):
    """Score candidate box apertures in one pass over the rate cube.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.
    candidates : array-like(float)
        Candidate extraction widths in their configured order.

    Returns
    -------
    results : list[tuple]
        Cost, wavelength scatter, and elapsed seconds for each candidate.
    """
    instrument = state.cube.meta.mode.split('/')[0].upper()
    if instrument in ('NIRSPEC', 'MIRI') and str(ctx.get('opts', {}).get(
            'extract_method', 'box')).lower() != 'box':
        raise NotImplementedError(
            'prepared aperture sweep supports box extraction only')
    extract = {'NIRSPEC': _extract_orders_nirspec, 'MIRI': _extract_orders_miri}.get(instrument)
    if extract is not None:
        centroids = _centroids_for_extraction(state, ctx, 'sum')
    started = time.perf_counter()
    widths = [c if v2config.is_asymmetric_width(c) else float(np.asarray(c)) for c in candidates]
    spectra = (extract(state, params, ctx, mode='sum', centroids=centroids, extract_widths=widths)
               if extract is not None else
               extract_orders(state, params, ctx, mode='sum', extract_widths=widths))
    shared = time.perf_counter() - started
    results = []
    for index in range(len(widths)):
        started = time.perf_counter()
        per = {o: (np.asarray(f[:, index]),
                   None if e is None else np.asarray(e[:, index]))
               for o, (f, e) in spectra.items()}
        if extract is not None:
            products = _single_order_products(state, ctx, centroids, *per[1], instrument.lower())
        else:
            _, products = _final_spectral_products(per, ctx)
        arrays = {int(order): (np.asarray(product['wave']), product['flux'])
                  for order, product in products.items()}
        cost, scatter = _cost_from_order_arrays(
            arrays, ctx['opts'], state.cube.meta.baseline_ints)
        results.append((float(np.asarray(cost)), np.asarray(scatter),
                        shared / len(widths) + time.perf_counter() - started))
    return results


def _nirspec_xstart(ctx):
    """Return the first extraction column for the NIRSpec grating and detector."""
    return int(ctx.get('nirspec_xstart', 0))


def _nirspec_deepstack_all_ints(data):
    """Calculate the NIRSpec tracing median over all integrations."""
    source = data[:, -1] if data.ndim == 4 else data
    return _host_deepstack(source, np.ones(source.shape[0], dtype=bool))


def _trace_nirspec_deepframe(deepframe, ctx):
    """Trace the NIRSpec deepframe with the v1 edgetrigger method."""
    deepframe = np.asarray(deepframe)
    centroids = trace.get_centroids_nirspec(
        deepframe, xstart=_nirspec_xstart(ctx))
    return trace.validate_nirspec_centroids(
        centroids, deepframe.shape[-1], xstart=_nirspec_xstart(ctx))


def _nirspec_centroids_for_stage(state, ctx, aux_key, deepframe=None):
    """Get explicit NIRSpec centroids or trace the processed deepframe."""
    return _stage_centroids(
        state, ctx, aux_key, deepframe, lambda deep: _trace_nirspec_deepframe(deep, ctx),
        lambda: _nirspec_deepstack_all_ints(state.cube.data))


def _nirspec_centroids_for_extraction(state, ctx, mode):
    """Select centroids for intermediate or final NIRSpec extraction."""
    return _extraction_centroids(state, ctx, mode, 'NIRSPEC')


def _nirspec_padded_trace(centroids, dimx, dtype):
    """Pad NIRSpec row centroids with NaNs outside the traced columns."""
    ypos = np.full(dimx, np.nan, dtype=dtype)
    x = np.asarray(centroids['xpos']).astype(int)
    ypos[x] = np.asarray(centroids['ypos'], dtype=dtype)
    return ypos


def _nirspec_clamped_track(centroids, dimx, xstart, dtype):
    """Clamp NIRSpec row centroids to cover the complete detector width."""
    ypos = np.asarray(centroids['ypos'], dtype=dtype)
    index = np.clip(np.arange(dimx) - int(xstart), 0, ypos.size - 1)
    return ypos[index]


def _apply_oneoverf_nirspec(data, base_mask, ypos_padded, width, method):
    """Apply NIRSpec 1/f correction in integration batches."""
    dtype = np.dtype(data.dtype)
    ypos_j = jnp.asarray(ypos_padded, dtype=dtype)
    width_j = jnp.asarray(width, dtype=dtype)
    base_mask = _beside(base_mask, _resident(data))
    return core.map_over_ints(lambda d, m: k_oof.oneoverf_nirspec(
            d, m, ypos_j, width_j, method=method), (data, base_mask), data.shape[0])


def _apply_oneoverf_nirspec_with_segment_traces(
        cube, base_mask, shared_centroids, width, method, ctx):
    """Apply NIRSpec 1/f correction with per-segment tracing for short NRS1 traces."""
    dimx = cube.data.shape[-1]
    short_trace = np.asarray(shared_centroids['ypos']).size < dimx
    if not short_trace:
        ypos = _nirspec_padded_trace(
            shared_centroids, dimx, np.dtype(cube.data.dtype))
        return _apply_oneoverf_nirspec(
            cube.data, base_mask, ypos, width, method)

    resident = _resident(cube.data)
    corrected = _segment_buffer(
        cube.data.shape, cube.data.dtype, 'nirspec_oneoverf_science',
        resident)
    for seg in _segment_slices(cube):
        segment_data = cube.data[seg]
        centroids = _trace_nirspec_deepframe(
            _nirspec_deepstack_all_ints(segment_data), ctx)
        ypos = _nirspec_padded_trace(
            centroids, dimx, np.dtype(cube.data.dtype))
        _segment_store(corrected, seg, _apply_oneoverf_nirspec(
            segment_data, base_mask[seg], ypos, width, method))
    return _segment_result(corrected, resident)


def step_superbias_nirspec(state, params, ctx):
    """Subtract the configured CRDS or visit-derived NIRSpec superbias.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    method = str(ctx['opts'].get('superbias_method', 'crds')).lower()
    if method == 'crds':
        return step_superbias(state, params, ctx)
    if method not in ('custom', 'custom-rescale'):
        raise ValueError(f'unsupported superbias_method {method!r}')
    cube = state.cube
    data = cube.data
    nints, _ngroups, dimy, dimx = data.shape
    dtype = np.dtype(data.dtype)
    superbias = np.nanmedian(np.asarray(core.to_host(data[:, 0])), axis=0)
    aux = dict(state.aux, superbias_custom=superbias)

    if method == 'custom':
        corrected = core.map_over_ints(lambda d: d - jnp.asarray(superbias, dtype=d.dtype),
            data, nints)
        return _updated_state(state, data=corrected, aux=aux)

    dq_bad = np.asarray(cube.pixeldq).astype(bool)
    fixed = ctx.get('centroids')
    resident = _resident(data)
    scale_parts = []
    bias = jnp.asarray(superbias, dtype=dtype)
    corrected_parts = _segment_buffer(data.shape, data.dtype, 'nirspec_superbias_science', resident)
    for seg in _segment_slices(cube):
        # Transfer only the two groups needed for the scale factor and trace.
        seg_group0 = np.asarray(core.to_host(data[seg, 0]))
        if fixed is not None and (np.asarray(fixed['ypos']).size == dimx or
                ctx['opts'].get('superbias_override_centroids', False)):
            centroids = fixed
        else:
            # Retrace each segment when its NRS1 centroid table omits the leading columns.
            centroids = _trace_nirspec_deepframe(
                np.nanmedian(np.asarray(core.to_host(data[seg, -1])), axis=0), ctx)
        ypos = np.asarray(centroids['ypos'], dtype=float)
        xpos = np.asarray(centroids['xpos']).astype(int)
        mask_width = ctx['opts'].get('superbias_mask_width', 10)
        low = np.maximum(np.zeros_like(ypos), ypos - mask_width / 2).astype(int)
        up = np.minimum(dimy * np.ones_like(ypos), ypos + mask_width / 2).astype(int)
        tracemask = np.ones((dimy, dimx))
        for i, x in enumerate(xpos):
            tracemask[low[i]:up[i], x] = 0
        # Mask only pixels that are both DQ-flagged and inside the trace.
        mask = ~dq_bad | tracemask.astype(bool)
        mask = np.where(mask == 0, np.nan, mask.astype(float))
        group0 = mask[None] * seg_group0
        scale = np.nanmedian(group0 / superbias[None], axis=(1, 2))
        scale_parts.append(scale)
        # Multiply before subtracting to preserve float32 rounding.
        part = core.map_over_ints(lambda ramp, factor: ramp - factor[:, None, None, None] *
            bias[None, None], (data[seg], _beside(scale.astype(dtype), resident)),
            seg.stop - seg.start)
        _segment_store(corrected_parts, seg, part)
    corrected = _segment_result(corrected_parts, resident)
    aux['superbias_scale_factors'] = np.concatenate(scale_parts, axis=0)
    return _updated_state(state, data=corrected, aux=aux)


def _step_oneoverf_nirspec(state, params, ctx, aux_key):
    """Apply NIRSpec 1/f correction and retain automatically traced centroids."""
    cube = state.cube
    centroids = _nirspec_centroids_for_stage(state, ctx, aux_key)
    mask = _oneoverf_base_mask(cube, ctx, reset=True)
    method = str(ctx['opts'].get('oof_method', 'median')).lower()
    data = _apply_oneoverf_nirspec_with_segment_traces(
        cube, mask, centroids, params['nirspec_mask_width'], method, ctx)
    aux = dict(state.aux)
    if ctx.get('centroids') is None:
        aux[aux_key] = centroids
    return _updated_state(state, data=data, aux=aux)


def step_oneoverf_grp_nirspec(state, params, ctx):
    """Remove NIRSpec 1/f noise from each group.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    return _step_oneoverf_nirspec(state, params, ctx, 'centroids_group')


def step_oneoverf_int_nirspec(state, params, ctx):
    """Remove NIRSpec 1/f noise from the integration rates.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    return _step_oneoverf_nirspec(state, params, ctx, 'centroids_int')


def step_assign_wcs_nirspec(state, params, ctx):
    """Record the NIRSpec slit wavelengths and point-source metadata.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    if ctx.get('nirspec_wave_map') is None:
        raise ValueError(
            'AssignWCSStep requires the refpack NIRSpec wave_map')
    aux = dict(state.aux, assign_wcs='COMPLETE', source_type='POINT',
               wavemap_provenance=ctx.get('wavemap_provenance'))
    return PipelineState(cube=state.cube, aux=aux)


def step_extract2d_nirspec(state, params, ctx):
    """Header-only for the supported BOTS subarrays (slit == subarray).

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    return PipelineState(cube=state.cube,
                         aux=dict(state.aux, extract2d='COMPLETE'))


def step_wavecorr_nirspec(state, params, ctx):
    """Wavelength zero-point correction is folded into the refpack map.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    return PipelineState(cube=state.cube,
                         aux=dict(state.aux, wavecorr='COMPLETE'))


def _stack_candidate_apertures(extract, halfwidths):
    """Extract each aperture sequentially and stack results on the candidate axis.

    Sequential extraction preserves the rounding of a committed single-width extraction.
    """
    evaluated = [extract(halfwidths[index])
                 for index in range(int(halfwidths.shape[0]))]
    stacked = []
    for position in range(len(evaluated[0])):
        pieces = [candidate[position] for candidate in evaluated]
        stacked.append(None if pieces[0] is None else
                       jnp.stack(pieces, axis=1))
    return tuple(stacked)


def _candidate_halfwidths(extract_widths, dtype):
    """Convert candidate aperture widths to traced half-widths."""
    if any(v2config.is_asymmetric_width(width) for width in extract_widths):
        return jnp.asarray(
            [v2config.extract_width_halves(width)
             for width in extract_widths], dtype=dtype)
    return jnp.asarray(np.asarray(extract_widths, dtype=dtype),
                       dtype=dtype) / jnp.asarray(2., dtype=dtype)


def _extract_orders_nirspec(state, params, ctx, mode='nanaware',
                            centroids=None, extract_widths=None):
    """Extract NIRSpec spectra from the current ramp or rate cube."""
    cen = (centroids if centroids is not None else
           _nirspec_centroids_for_extraction(state, ctx, mode))
    cube = state.cube
    halfwidth = _aperture_halfwidth(params, cube, extract_widths)
    ypos = jnp.asarray(np.asarray(cen['ypos']), dtype=np.dtype(cube.data.dtype))
    return _extract_apertures(
        cube, ctx, mode, {1: (ypos, halfwidth, {'extract_start': _nirspec_xstart(ctx)})},
        extract_widths, instrument='NIRSPEC')


def _extract_optimal(state, params, ctx, centroids, instrument):
    """Extract raw rates with Horne profiles and pad the native wavelength axis with zeros."""
    data = state.cube.data
    nints, dimy, dimx = data.shape
    dtype = np.dtype(data.dtype)
    miri = instrument == 'miri'
    if miri:
        ypos = np.asarray(centroids['ypos'], dtype=float)
        start, end = int(np.min(ypos)), int(np.max(ypos)) + 1
        track = np.asarray(centroids['xpos'], dtype=float)
        across, channels = dimx, dimy
        data_horne = data[:, start:end, :]
    else:
        start, end = _nirspec_xstart(ctx), dimx
        track = np.asarray(centroids['ypos'], dtype=float)
        across, channels = dimy, dimx
        swapaxes = jnp.swapaxes if core.is_device_array(data) else np.swapaxes
        source = data if core.is_device_array(data) else np.asarray(data)
        data_horne = swapaxes(source[:, :, start:], 1, 2)
    index = np.clip(np.arange(end - start), 0, max(track.shape[0] - 1, 0))
    width = params.get('extract_width')
    halfwidth = (None if width is None else
                 jnp.asarray(width, dtype=dtype) / jnp.asarray(2., dtype=dtype))
    mask = k_ext.horne_aperture_mask(jnp.asarray(track[index], dtype=dtype), halfwidth, across)
    deepframe = _stage3_deepframe(state, ctx)
    if deepframe is None:
        raise ValueError('optimal extraction requires a deepframe (v1 '
            'do_optimal_extraction, stage3.py:2036)')
    deepframe = jnp.asarray(deepframe, dtype=dtype)
    deep_prep = (k_ext.miri_optimal_deepframe_prep(deepframe) if miri else
                 k_ext.horne_deepframe_prep(deepframe, dispersion_axis=1))
    profile = k_ext.miri_optimal_profile(deep_prep[start:end], mask, mask_output=False)
    opts = ctx.get('opts', {})
    flux_slice, ferr_slice, clipped_counts = k_ext.horne_optimal_extract(
        data_horne, profile, aperture_mask=mask, max_iter=int(opts.get('opt_max_iter', 25)),
        var_thresh=float(opts.get('opt_var_thresh', 25)))
    outputs = []
    for values, suffix in ((flux_slice, 'flux'), (ferr_slice, 'error')):
        out = core.empty_host_array((nints, channels), dtype,
                                    name=f'exotedrf-{instrument}-optimal-{suffix}')
        out.fill(0.)
        out[:, start:end] = np.asarray(values)
        outputs.append(out)
    return *outputs, clipped_counts


def _single_order_products(state, ctx, centroids, flux, ferr, instrument):
    """Build wavelengths and clipped products on one instrument's native detector axis."""
    raw_flux = flux
    flux = _clip_lc(flux, ctx)
    ferr = None if ferr is None else np.asarray(ferr)
    wave_map = np.asarray(ctx[instrument + '_wave_map'], dtype=float)
    x = np.asarray(centroids['xpos']).astype(int)
    y = np.asarray(centroids['ypos']).astype(int)
    axis = -2 if instrument == 'miri' else -1
    wave = np.full(state.cube.data.shape[axis], np.nan)
    wave[y if instrument == 'miri' else x] = wave_map[y, x]
    product = {'wave': wave, 'flux': flux, 'ferr': ferr}
    if instrument == 'nirspec':
        from exotedrf.v2 import wavecal
        product['wave'], shift = wavecal.nirspec_stage3_wavelengths(wave, raw_flux, ctx)
        if shift is not None:
            product['wave_shift'] = shift
    return {1: product}


def _step_extract_single_order(state, params, ctx, instrument):
    """Assemble final box or optimal spectra, DQ reports, and extraction diagnostics."""
    miri = instrument == 'miri'
    centroid_fn = _miri_centroids_for_extraction if miri else _nirspec_centroids_for_extraction
    centroids = centroid_fn(state, ctx, 'sum')
    method = str(ctx.get('opts', {}).get('extract_method', 'box')).lower()
    selection = clipped_counts = None
    if method == 'optimal':
        optimal = _extract_miri_optimal if miri else _extract_nirspec_optimal
        flux, ferr, clipped_counts = optimal(state, params, ctx, centroids)
    else:
        params, ctx, selection = _resolve_optimize_widths(state, params, ctx, centroids)
        extract = _extract_orders_miri if miri else _extract_orders_nirspec
        flux, ferr = extract(state, params, ctx, mode='sum', centroids=centroids)[1]
    products = _single_order_products(state, ctx, centroids, flux, ferr, instrument)
    report = _miri_dq_report if miri else _nirspec_dq_report
    products[1]['dq'] = report(state, params, ctx, centroids,
                              'optimal' if method == 'optimal' else 'box')
    aux = dict(state.aux, spectra={1: (products[1]['flux'], products[1]['ferr'])},
               spectral_products=products)
    if ctx.get('centroids') is None:
        aux['centroids_extract'] = centroids
    if clipped_counts is not None:
        aux['optimal_clipped_counts'] = [clipped_counts]
    _attach_width_selection(aux, selection)
    return PipelineState(cube=state.cube, aux=aux)


def _nirspec_final_products(state, ctx, centroids, flux, ferr):
    """Build the final NIRSpec wavelength and clipped light-curve products."""
    return _single_order_products(state, ctx, centroids, flux, ferr, 'nirspec')


def step_extract_nirspec(state, params, ctx):
    """Extract final NIRSpec box or optimal spectra and clip the light curves.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    return _step_extract_single_order(state, params, ctx, 'nirspec')


def _extract_nirspec_optimal(state, params, ctx, centroids):
    """Extract NIRSpec spectra with the Horne optimal-extraction method.

    Use raw rates without DQ masking and retain zero flux before the trace start.
    """
    return _extract_optimal(state, params, ctx, centroids, 'nirspec')


class PreparedNirspecOneOverFScorer:
    """Score NIRSpec 1/f trace-mask widths from the final group."""

    grid_order = ('nirspec_mask_width',)

    def __init__(self, state, params, ctx):
        """Prepare reusable arrays and candidate caches."""
        cube = state.cube
        self.ctx = ctx
        self.meta = cube.meta
        self.dtype = np.dtype(cube.data.dtype)
        self.method = str(ctx['opts'].get('oof_method', 'median')).lower()
        self.data = jnp.asarray(cube.data[:, -1])
        self.groupdq = jnp.asarray(cube.groupdq[:, -1])
        self.base_mask = jnp.asarray(_oneoverf_base_mask(cube, ctx, reset=True))
        self.centroids = _nirspec_centroids_for_stage(state, ctx, 'centroids_group')
        self.ypos_padded = jnp.asarray(_nirspec_padded_trace(
            self.centroids, cube.data.shape[-1], self.dtype))
        self.extract_width = _scorer_extract_width(params['extract_width'], self.dtype)
        self._result_cache = {}

    def _evaluate_width_device(self, width):
        """Calculate the cost and scatter for one mask width on the device."""
        corrected = k_oof.oneoverf_nirspec(self.data, self.base_mask, self.ypos_padded,
            jnp.asarray(width, dtype=self.dtype), method=self.method)
        return _optimizer_cost_from_last_group(
            corrected, self.groupdq, {'extract_width': self.extract_width},
            self.ctx, self.meta, self.centroids)

    def _evaluate_host(self, width):
        """Calculate or retrieve the cost, scatter, and elapsed time for one candidate."""
        return _cached_score(self._result_cache, _float_key(width),
                             self._evaluate_width_device, width)

    def evaluate_candidates(self, parameter, candidates, params):
        """Score each candidate in its configured order.

        Parameters
        ----------
        parameter : str
            Name of the parameter to vary.
        candidates : array-like(float)
            Candidate values in their configured order.
        params : dict
            Current values of the other reduction parameters.

        Returns
        -------
        results : list[tuple]
            Cost, wavelength scatter, and elapsed seconds for each candidate.
        """
        if parameter != 'nirspec_mask_width':
            raise KeyError(parameter)
        values = list(candidates)
        if SCORER_BATCH_AXIS and values:
            return _batched_axis_eval(
                values, _float_key, self._result_cache, jax.vmap(self._evaluate_width_device),
                self.dtype)
        return [self._evaluate_host(value) for value in values]


def prepare_nirspec_oneoverf_scorer(state, params, ctx):
    """Prepare a NIRSpec mask-width scorer when its arrays fit in memory.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    scorer : None, PreparedNirspecOneOverFScorer
        Reusable candidate scorer, or None when its requirements are not met.
    """
    if not isinstance(state.cube, RampCube) or state.cube.data.ndim != 4:
        return None
    if str(ctx.get('opts', {}).get('oof_method', 'median')).lower() not in \
            ('median', 'slope'):
        return None
    if np.asarray(state.cube.meta.segment_edges).size > 1:
        # Use per-segment tracing when one shared NRS1 trace cannot cover the leading columns.
        return None
    if not _prepared_scorer_fits(state.cube, buffers=16):
        return None
    return PreparedNirspecOneOverFScorer(state, params, ctx)


_NIRSPEC_STEP_CONTROLS = (
    'DQInitStep', 'SuperBiasStep', 'DarkCurrentStep', 'OneOverFStep_grp',
    'LinearityStep', 'JumpStep', 'RampFitStep', 'GainScaleStep',
    'AssignWCSStep', 'OneOverFStep_int', 'BadPixStep',
    'PCAReconstructStep')


def _validate_pca_options(opts):
    """Validate the PCA component count and one-based removal indices."""
    n_comp = int(opts.get('pca_components', 10))
    if n_comp < 1:
        raise ValueError('pca_components must be positive')
    remove = opts.get('remove_components')
    if remove is not None:
        values = np.atleast_1d(remove)
        if (not np.issubdtype(values.dtype, np.integer) or
                np.any(values < 1)):
            raise ValueError(
                'remove_components must contain 1-based (>= 1) indices')


def _validate_nirspec_context(ctx):
    """Validate NIRSpec controls and references and release the input cube."""
    opts = ctx.get('opts', {})
    for name in _NIRSPEC_STEP_CONTROLS:
        _runs(opts, name)

    method = str(opts.get('oof_method', 'median')).lower()
    if method not in ('median', 'slope'):
        raise ValueError(f"NIRSpec oof_method must be 'median' or 'slope', got {method!r}")
    extract_method = str(opts.get('extract_method', 'box')).lower()
    if extract_method not in ('box', 'optimal'):
        raise NotImplementedError(f"NIRSpec v2 supports extract_method='box' or 'optimal', got "
            f'{extract_method!r}')
    sb_method = str(opts.get('superbias_method', 'crds')).lower()
    if sb_method not in ('crds', 'custom', 'custom-rescale'):
        raise NotImplementedError(f'superbias_method={sb_method!r} is not supported for NIRSpec')

    post_ramp = ('GainScaleStep', 'AssignWCSStep',
                 'OneOverFStep_int', 'BadPixStep', 'PCAReconstructStep')
    return _finish_context(ctx, post_ramp)


def _enabled_steps(opts, specs, extract):
    """Build ordered steps and validate controls with no separate science operation."""
    opts = {} if opts is None else opts
    steps = []
    for control, entries in specs:
        if _runs(opts, control):
            steps.extend(Step(*entry) for entry in entries)
    steps.append(Step('Extract', extract, ('extract_width',)))
    return steps


_JUMP_PARAMS = ('time_jump_threshold', 'time_window')
_BADPIX_PARAMS = ('space_outlier_threshold', 'time_outlier_threshold', 'box_size', 'window_size')


def nirspec_steps(opts=None):
    """Build the enabled NIRSpec steps in v1 order.

    Parameters
    ----------
    opts : None, dict
        Reduction options controlling the enabled steps.

    Returns
    -------
    steps : list[Step]
        Enabled calibration and extraction steps.
    """
    return _enabled_steps(opts, [('DQInitStep', [('DQInitStep', step_dq_init)]),
        ('SuperBiasStep', [('SuperBiasStep', step_superbias_nirspec)]),
        ('DarkCurrentStep', [('DarkCurrentStep', step_dark)]),
        ('OneOverFStep_grp', [('OneOverFStep_grp', step_oneoverf_grp_nirspec,
                              ('nirspec_mask_width',))]),
        ('LinearityStep', [('LinearityStep', step_linearity)]),
        ('JumpStep', [('JumpStep', step_jump, _JUMP_PARAMS)]),
        ('RampFitStep', [('RampFitStep', step_rampfit)]),
        ('GainScaleStep', [('GainScaleStep', step_gain_scale)]),
        ('AssignWCSStep', [('AssignWCSStep', step_assign_wcs_nirspec)]),
        ('Extract2DStep', []), ('SourceTypeStep', []), ('WaveCorrStep', []),
        ('OneOverFStep_int', [('OneOverFStep_int', step_oneoverf_int_nirspec,
                              ('nirspec_mask_width',))]),
        ('BadPixStep', [('BadPixStep', step_badpix, _BADPIX_PARAMS)]),
        ('PCAReconstructStep', [('PCAReconstructStep', step_pca)]),], step_extract_nirspec)


def _initial_context(cube, opts, refpack, search_dirs):
    """Resolve references and load the static hot-pixel and temporal-outlier masks."""
    refpack = refs.resolve_refpack(cube, opts, refpack=refpack)
    refs.validate_refpack(refpack, cube, opts, require_waves=True)
    ctx = {'opts': opts, 'refpack': refpack, 'cube_shape': cube.data.shape, 'input_cube': cube}
    ctx['hot_pixel_map'] = _load_hot_pixel_map(
        opts.get('hot_pixel_map'), cube.data.shape[-2:], search_dirs)
    ctx['outlier_mask'] = _load_outlier_maps(opts.get('outlier_maps'), cube, search_dirs)
    return ctx


def _context_wave_map(ctx, cube, instrument):
    """Validate the wavelength plane and retain its instrument-specific context key."""
    wave_map = _require_reference(ctx, 'wave_map', f'{instrument} refpack must provide wave_map')
    wave_map = np.asarray(wave_map, dtype=float)
    if wave_map.shape != cube.data.shape[-2:]:
        raise ValueError(f'refpack wave_map shape {wave_map.shape} does not match the '
            f'detector {cube.data.shape[-2:]}')
    ctx[instrument.lower() + '_wave_map'] = wave_map


def _context_centroids(ctx, search_dirs, validate, size, **kwargs):
    """Load and validate explicit centroid inputs."""
    value = ctx['opts'].get('centroids')
    if value is not None:
        if not isinstance(value, dict):
            if isinstance(value, (str, os.PathLike)) and not os.path.isabs(os.fspath(value)):
                value = next((path for directory in search_dirs
                              if os.path.exists(path := os.path.join(directory, os.fspath(value)))),
                             value)
            value = trace.load_centroids_csv(value)
        value = validate(value, size, **kwargs)
    ctx['centroids'] = value
    ctx['centroids_source'] = 'explicit' if value is not None else 'automatic-v1-lifecycle'


def _finish_context(ctx, post_ramp):
    """Validate post-ramp dependencies and PCA options, then release the input cube."""
    opts = ctx.get('opts', {})
    if not _runs(opts, 'RampFitStep') and any(_runs(opts, key) for key in post_ramp) and \
            opts.get('restart_input_ndim') != 3:
        raise ValueError('RampFitStep cannot be skipped while stage-2 steps run')
    _validate_pca_options(opts)
    cube = ctx.pop('input_cube', None)
    if cube is not None:
        refs.validate_refpack(ctx.get('refpack'), cube, opts, require_waves=True)
    return ctx


def prepare_nirspec_context(cube, opts, refpack=None):
    """Assemble static references and explicit centroid inputs.

    Explicit centroids stay fixed; automatic centroids follow the processed stages.

    Parameters
    ----------
    cube : RampCube, RateCube
        Input science cube and observation metadata.
    opts : dict
        Reduction options.
    refpack : None, RefPack
        Reference arrays; resolve them when None.

    Returns
    -------
    ctx : dict
        Validated options, references, and explicit centroid inputs.
    """
    opts = dict(opts)
    search_dirs = (opts.get('input_dir', ''), os.getcwd())
    ctx = _initial_context(cube, opts, refpack, search_dirs)

    mode = str(opts.get('mode') or cube.meta.mode)
    header = (getattr(cube.meta, 'extra', {}) or {}).get('header', {})
    grating = ''
    if '/' in mode:
        grating = mode.split('/', 1)[1].upper()
    if not grating:
        grating = str(header.get('GRATING', '') or '').upper()
    if not grating:
        raise ValueError('NIRSpec requires the grating (observing_mode NIRSpec/<grating> '
            'or a FITS GRATING header)')
    detector = str(cube.meta.detector or opts.get('filter_detector', '')).upper()
    if detector not in ('NRS1', 'NRS2'):
        raise ValueError(f'NIRSpec requires detector NRS1 or NRS2, got {detector!r}')
    ctx['nirspec_grating'] = grating
    ctx['nirspec_detector'] = detector
    ctx['nirspec_xstart'] = trace.nirspec_trace_start(detector, cube.meta.subarray, grating)

    _context_wave_map(ctx, cube, 'NIRSpec')
    # Use pixel wavelengths for optimizer spectra and the wavelength map for final products.
    ctx['waves'] = None
    ctx['wavemap_provenance'] = _rp(ctx, 'meta_file_wavemap')

    _context_centroids(ctx, search_dirs, trace.validate_nirspec_centroids,
                       cube.data.shape[-1], xstart=ctx['nirspec_xstart'])
    ctx['custom_deepframe'] = _load_custom_deepframe(
        opts.get('deepframe'), cube.data.shape[-2:], search_dirs)
    _validate_nirspec_context(ctx)
    return ctx


def miri_extract_bounds(centroids):
    """Return the MIRI box-extraction bounds in the transposed frame.

    Parameters
    ----------
    centroids : dict
        MIRI trace positions with dispersion rows in ypos.

    Returns
    -------
    start : int
        First extracted detector row.
    end : int
        Exclusive final detector row.
    """
    ypos = np.asarray(centroids['ypos'])
    return int(np.min(ypos)), int(np.max(ypos))


def miri_score_group(data, groupdq, params, ctx, meta, centroids):
    """Select the latest usable MIRI group with a finite optimizer cost.

    JUMP_DET alone does not disqualify a group. Raise ValueError if no group is scoreable.

    Parameters
    ----------
    data : array-like(float)
        MIRI ramp cube.
    groupdq : array-like(int)
        Group DQ flags for the ramp cube.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.
    meta : ObsMeta
        Observation metadata.
    centroids : dict
        MIRI trace positions.

    Returns
    -------
    group : int
        Latest usable group with a finite aperture score.
    """
    if groupdq.ndim != 4:
        raise ValueError('MIRI score-group selection requires 4-D GROUPDQ')
    ngroups = groupdq.shape[1]
    jump_bit = np.uint8(core.DQ_JUMP_DET)
    # Check one group at a time without materializing the complete ramp.
    xp = jnp if core.is_device_array(groupdq) else np
    for group in range(ngroups - 1, -1, -1):
        group_dq = (groupdq[:, group] & ~jump_bit).astype(groupdq.dtype)
        if bool(xp.all(group_dq > 0)):
            continue
        cost, _ = _optimizer_cost_from_last_group(
            jnp.asarray(data[:, group]), jnp.asarray(group_dq),
            params, ctx, meta, centroids)
        if np.isfinite(np.asarray(cost)):
            return group
    raise ValueError(
        'No scoreable MIRI group: every group is either fully flagged '
        '(ignoring DQ_JUMP_DET) or yields a non-finite optimizer cost.')


def _trace_miri_deepframe(deepframe, ctx, stage3=False):
    """Trace the MIRI deepframe, allowing a slope only for final extraction."""
    deepframe = np.asarray(deepframe)
    opts = ctx.get('opts', {})
    centroids = trace.get_centroids_miri(
        deepframe, ystart=int(ctx.get('miri_ystart', trace.MIRI_TRACE_YSTART)),
        allow_slope=bool(opts.get('allow_miri_slope', False) or (
            stage3 and opts.get('stage3_allow_miri_slope', False))))
    return trace.validate_miri_centroids(
        centroids, deepframe.shape[-2],
        ystart=int(ctx.get('miri_ystart', trace.MIRI_TRACE_YSTART)))


def _miri_centroids_for_stage(state, ctx, aux_key, deepframe=None):
    """Get explicit MIRI centroids or trace the processed deepframe."""
    data = state.cube.data
    return _stage_centroids(
        state, ctx, aux_key, deepframe, lambda deep: _trace_miri_deepframe(deep, ctx),
        lambda: np.nanmedian(np.asarray(core.to_host(data[:, -1] if data.ndim == 4 else data)),
                             axis=0))


def _miri_centroids_for_extraction(state, ctx, mode):
    """Select centroids for intermediate or final MIRI extraction."""
    return _extraction_centroids(state, ctx, mode, 'MIRI')


def step_emicorr_miri(state, params, ctx):
    """Correct MIRI electromagnetic interference independently for each FITS segment.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    del params
    cube = state.cube
    frequencies = _rp(ctx, 'emicorr_frequencies')
    waves = _rp(ctx, 'emicorr_reference_waves')
    lengths = _rp(ctx, 'emicorr_reference_wave_lengths')
    rowclocks = _rp(ctx, 'emicorr_rowclocks')
    frameclocks = _rp(ctx, 'emicorr_frameclocks')
    if frequencies is None or rowclocks is None or frameclocks is None:
        raise ValueError('EmiCorrStep requires the refpack EMI reference')
    if waves is not None:
        waves = np.asarray(waves, dtype=float)
        if lengths is None:
            lengths = np.full(waves.shape[0], waves.shape[1], dtype=int)
        waves = [wave[:int(length)] for wave, length in zip(waves, np.asarray(lengths))]

    # Correct EMI on the host and return each segment to its original host or device.
    data = cube.data
    resident = _resident(data)
    corrected = _segment_buffer(data.shape, data.dtype, 'miri_emicorr_science', resident)
    extra = getattr(cube.meta, 'extra', {}) or {}
    headers = extra.get('segment_headers')
    if headers is None:
        headers = tuple(extra.get('header', {}) for _ in _segment_slices(cube))
    algorithm = str(ctx.get('opts', {}).get('emicorr_algorithm', 'joint'))
    pixeldq = (None if algorithm != 'joint' or waves is None else
               np.asarray(core.to_host(cube.pixeldq)))
    for seg, header in zip(_segment_slices(cube), headers):
        readpatt = str(header.get('READPATT', 'FASTR1') or 'FASTR1')
        nsamples = int(header.get('NSAMPLES', 1) or 1)
        if pixeldq is not None:
            part = k_emi.apply_emicorr_joint(np.asarray(core.to_host(data[seg])), pixeldq,
                np.asarray(frequencies, dtype=float), waves, int(np.asarray(rowclocks).reshape(())),
                int(np.asarray(frameclocks).reshape(())), readpatt=readpatt, nsamples=nsamples)
        else:
            part = k_emi.apply_emicorr(np.asarray(core.to_host(data[seg])),
                np.asarray(frequencies, dtype=float), waves, int(np.asarray(rowclocks).reshape(())),
                int(np.asarray(frameclocks).reshape(())), readpatt=readpatt, nsamples=nsamples,
                xstart=int(header.get('SUBSTRT1', 1) or 1))
        # Cast corrected samples explicitly for both host and device storage.
        part = np.asarray(part, dtype=data.dtype)
        _segment_store(corrected, seg, _beside(part, resident))
    corrected = _segment_result(corrected, resident)
    aux = dict(state.aux, emicorr='COMPLETE')
    return _updated_state(state, data=corrected, aux=aux)


step_emicorr = step_emicorr_miri


def _chunked_int_offsets(nints, bytes_per_int):
    """Return integration batch boundaries for kernels requiring absolute indices."""
    chunk = core.auto_chunk(nints, max(int(bytes_per_int), 1))
    return [(lo, min(lo + chunk, nints)) for lo in range(0, nints, chunk)]


def step_reset_miri(state, params, ctx):
    """Correct the MIRI reset anomaly using exposure-level integration indices.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    del params
    cube = state.cube
    reset_data = _rp(ctx, 'reset_data')
    reset_dq = _rp(ctx, 'reset_dq')
    if reset_data is None:
        raise ValueError('ResetStep requires the refpack reset reference')
    if reset_dq is None:
        reset_dq = np.zeros(cube.data.shape[-2:], np.uint32)
    reset_j = jnp.asarray(reset_data)
    dq_j = jnp.asarray(np.asarray(reset_dq, np.uint32))
    pixeldq = jnp.asarray(np.asarray(cube.pixeldq, np.uint32))
    data = cube.data
    resident = _resident(data)
    bytes_per_int = data.nbytes // max(data.shape[0], 1)
    corrected_all = _segment_buffer(data.shape, data.dtype, 'miri_reset_science', resident)
    for seg, int_start in zip(_segment_slices(cube), cube.meta.segment_int_starts):
        offset = int(int_start) - 1
        seg_nints = seg.stop - seg.start
        seg_data = data[seg]
        for lo, hi in _chunked_int_offsets(seg_nints, bytes_per_int):
            corrected, _ = k_reset.reset_correct(
                jnp.asarray(seg_data[lo:hi]), pixeldq, reset_j, dq_j, int_start=offset + lo)
            _segment_store(corrected_all, slice(seg.start + lo, seg.start + hi), corrected)
    corrected_all = _segment_result(corrected_all, resident)
    _, pixeldq = k_reset.reset_correct(jnp.zeros((1, 1) + tuple(cube.data.shape[-2:]),
                  dtype=cube.data.dtype), pixeldq, reset_j, dq_j, int_start=0)
    return _updated_state(state, data=corrected_all, pixeldq=np.asarray(pixeldq))


step_reset = step_reset_miri


def _subtract_miri_dark(cube, dark, dark_dq):
    """Subtract the matched MIRI dark groups and merge reference pixel DQ."""
    if dark is None:
        raise ValueError('the MIRI dark reference is required by LinearityStep '
            '(miri_subtract_dark)')
    dark = np.asarray(dark, dtype=np.float32)
    dark_nints = dark.shape[0]
    ngroups = cube.data.shape[1]
    data = cube.data
    resident = _resident(data)
    bytes_per_int = data.nbytes // max(data.shape[0], 1)
    corrected_all = _segment_buffer(data.shape, data.dtype, 'miri_dark_science', resident)
    for seg, int_start in zip(_segment_slices(cube), cube.meta.segment_int_starts):
        seg_nints = seg.stop - seg.start
        if int(int_start) == 1:
            index = np.minimum(np.arange(seg_nints), dark_nints - 1)
        else:
            index = np.full(seg_nints, dark_nints - 1)
        seg_data = data[seg]
        # Select dark-reference integrations within each batch.
        for lo, hi in _chunked_int_offsets(seg_nints, bytes_per_int):
            correction = jnp.asarray(dark[index[lo:hi], :ngroups], dtype=seg_data.dtype)
            part = core.map_over_ints(lambda science, reference: science - reference,
                (seg_data[lo:hi], correction), hi - lo)
            _segment_store(corrected_all, slice(seg.start + lo, seg.start + hi), part)
    corrected_all = _segment_result(corrected_all, resident)
    pixeldq = np.asarray(cube.pixeldq, np.uint32)
    if dark_dq is not None:
        pixeldq = np.bitwise_or(pixeldq, np.asarray(dark_dq, np.uint32))
    return corrected_all, pixeldq


def _miri_dark_correction(cube, ctx):
    """Apply the MIRI dark reference supplied in the context."""
    status = _rp(ctx, 'meta_dark_status')
    if status is not None and \
            str(np.asarray(status).reshape(()).item()).upper() == 'SKIPPED':
        return cube.data, np.asarray(cube.pixeldq, np.uint32)
    return _subtract_miri_dark(
        cube, _rp(ctx, 'dark'), _rp(ctx, 'dark_dq'))


def step_linearity_miri(state, params, ctx):
    """Correct MIRI linearity, subtract dark current, and flag the dropped groups.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    opts = ctx['opts']
    state = step_linearity(state, params, ctx)
    cube = state.cube
    data, pixeldq = cube.data, np.asarray(cube.pixeldq, np.uint32)
    dark_subtracted = False
    if opts.get('miri_subtract_dark', True):
        data, pixeldq = _miri_dark_correction(cube, ctx)
        dark_subtracted = True
    groupdq = cube.groupdq
    drop = opts.get('miri_drop_groups', 12)
    ngroups = cube.data.shape[1]
    if drop is not None and int(drop) > 0 and ngroups > int(drop) + 4:
        dnu = jnp.uint8(int(core.DQ_DO_NOT_USE))
        groupdq = jnp.asarray(groupdq, dtype=jnp.uint8)
        # Flag groups 1 through miri_drop_groups after DQInit flags group zero.
        groupdq = groupdq.at[:, 1:int(drop) + 1].set(groupdq[:, 1:int(drop) + 1] | dnu)
        if not _resident(cube.groupdq):
            groupdq = np.asarray(groupdq)
    aux = dict(state.aux, miri_dark_subtracted=dark_subtracted)
    return _updated_state(state, data=data, groupdq=groupdq, pixeldq=pixeldq, aux=aux)


def _apply_background_miri(data, params, ctx):
    """Apply MIRI background correction in integration batches."""
    method = str(ctx['opts'].get('miri_background_method', 'median')).lower()
    dtype = np.dtype(data.dtype)
    trace_width = jnp.asarray(params['miri_trace_width'], dtype=dtype)
    background_width = jnp.asarray(params['miri_background_width'],
                                   dtype=dtype)
    center = int(ctx.get('miri_trace_center', MIRI_TRACE_CENTER))
    return core.map_over_ints(
        lambda d: k_bkg.background_miri(
            d, trace_width, background_width, method=method,
            trace_center=center),
        data, data.shape[0])


def step_background_miri(state, params, ctx):
    """Subtract the row-wise MIRI background from the integration rates.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    cube = state.cube
    data, _bkg = _apply_background_miri(cube.data, params, ctx)
    return _updated_state(state, data=data)


def step_assign_wcs_miri(state, params, ctx):
    """Record the MIRI wavelength map and point-source metadata.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    del params
    if ctx.get('miri_wave_map') is None:
        raise ValueError('AssignWCSStep requires the refpack MIRI wave_map')
    aux = dict(state.aux, assign_wcs='COMPLETE', source_type='POINT',
               wavemap_provenance=ctx.get('wavemap_provenance'))
    return PipelineState(cube=state.cube, aux=aux)


def _extract_orders_miri(state, params, ctx, mode='nanaware', centroids=None, extract_widths=None):
    """Extract MIRI spectra from the current ramp or rate cube."""
    cen = (centroids if centroids is not None else _miri_centroids_for_extraction(state, ctx, mode))
    cube = state.cube
    start, end = miri_extract_bounds(cen)
    halfwidth = _aperture_halfwidth(params, cube, extract_widths)
    xpos = jnp.asarray(np.asarray(cen['xpos']), dtype=np.dtype(cube.data.dtype))
    group = (miri_score_group(cube.data, cube.groupdq, params, ctx, cube.meta, cen)
             if cube.data.ndim == 4 else None)
    return _extract_apertures(
        cube, ctx, mode, {1: (xpos, halfwidth, {'extract_start': start, 'extract_end': end})},
        extract_widths, transpose=True, group=group, instrument='MIRI')


def _extract_miri_optimal(state, params, ctx, centroids):
    """Extract MIRI spectra with the Horne optimal-extraction method.

    Use raw rates without DQ masking and include the final traced detector row.
    """
    return _extract_optimal(state, params, ctx, centroids, 'miri')


def step_extract_miri(state, params, ctx):
    """Extract final MIRI box or optimal spectra and clip the light curves.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    state : PipelineState
        Science cube and auxiliary products after the step.
    """
    return _step_extract_single_order(state, params, ctx, 'miri')


def _miri_final_products(state, ctx, centroids, flux, ferr):
    """Build the final MIRI wavelength and clipped light-curve products."""
    return _single_order_products(state, ctx, centroids, flux, ferr, 'miri')


class PreparedMiriBackgroundScorer:
    """Score MIRI trace-mask and background-strip widths on the rate cube."""

    grid_order = ('miri_trace_width', 'miri_background_width')

    def __init__(self, state, params, ctx):
        """Prepare reusable arrays and candidate caches."""
        cube = state.cube
        self.ctx = ctx
        self.meta = cube.meta
        self.dtype = np.dtype(cube.data.dtype)
        self.state = state
        self.params = dict(params)
        self.centroids = _miri_centroids_for_extraction(state, ctx, 'nanaware')
        self._result_cache = {}

    def _evaluate_device(self, trace_width, background_width, extract_width):
        """Calculate the cost and scatter for one candidate on the device."""
        params = dict(self.params, miri_trace_width=trace_width,
                      miri_background_width=background_width, extract_width=extract_width)
        corrected, _ = _apply_background_miri(self.state.cube.data, params, self.ctx)
        cube = self.state.cube
        trial = PipelineState(cube=RateCube(corrected, cube.err, cube.dq, cube.meta),
            aux=self.state.aux)
        spectra = _extract_orders_miri(trial, params, self.ctx, mode='nanaware',
            centroids=self.centroids)
        flux = spectra[1]
        flux = flux[0] if isinstance(flux, tuple) else flux
        wave = np.arange(np.shape(flux)[-1], dtype=float)
        return _cost_from_order_arrays({1: (wave, flux)}, self.ctx['opts'], self.meta.baseline_ints)

    def _evaluate_host(self, trace_width, background_width, extract_width):
        """Calculate or retrieve the cost, scatter, and elapsed time for one candidate."""
        key = (float(np.asarray(trace_width).reshape(())),
               float(np.asarray(background_width).reshape(())), _width_cache_key(extract_width))
        return _cached_score(self._result_cache, key, self._evaluate_device,
                             trace_width, background_width, extract_width)

    def evaluate_candidates(self, parameter, candidates, params):
        """Score each candidate in its configured order.

        Parameters
        ----------
        parameter : str
            Name of the parameter to vary.
        candidates : array-like(float)
            Candidate values in their configured order.
        params : dict
            Current values of the other reduction parameters.

        Returns
        -------
        results : list[tuple]
            Cost, wavelength scatter, and elapsed seconds for each candidate.
        """
        if parameter == 'miri_trace_width':
            return [self._evaluate_host(value, params['miri_background_width'],
                params['extract_width']) for value in candidates]
        if parameter == 'miri_background_width':
            return [self._evaluate_host(params['miri_trace_width'], value, params['extract_width'])
                for value in candidates]
        raise KeyError(parameter)


def prepare_miri_background_scorer(state, params, ctx):
    """Prepare a MIRI background scorer when its arrays fit in memory.

    Parameters
    ----------
    state : PipelineState
        Current science cube and auxiliary products.
    params : dict
        Reduction parameters, including the candidate extraction or mask widths.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    scorer : None, PreparedMiriBackgroundScorer
        Reusable candidate scorer, or None when its requirements are not met.
    """
    if not isinstance(state.cube, RateCube) or state.cube.data.ndim != 3:
        return None
    if str(ctx.get('opts', {}).get('miri_background_method',
                                   'median')).lower() not in \
            ('median', 'slope'):
        return None
    if not _prepared_scorer_fits(state.cube, buffers=16):
        return None
    return PreparedMiriBackgroundScorer(state, params, ctx)


_MIRI_STEP_CONTROLS = (
    'DQInitStep', 'EmiCorrStep', 'LinearityStep', 'JumpStep',
    'RampFitStep', 'GainScaleStep', 'AssignWCSStep',
    'FlatFieldStep', 'BackgroundStep', 'BadPixStep', 'PCAReconstructStep')


def _validate_miri_context(ctx):
    """Validate MIRI controls and references and release the input cube."""
    opts = ctx.get('opts', {})
    for name in _MIRI_STEP_CONTROLS:
        _runs(opts, name)

    method = str(opts.get('miri_background_method', 'median')).lower()
    if method not in ('median', 'slope'):
        raise ValueError(f"miri_background_method must be 'median' or 'slope', got " f'{method!r}')
    extract_method = str(opts.get('extract_method', 'box')).lower()
    if extract_method not in ('box', 'optimal'):
        raise NotImplementedError(f"MIRI v2 supports extract_method='box' or 'optimal', got "
            f'{extract_method!r}')
    drop = opts.get('miri_drop_groups', 12)
    if drop is not None and (isinstance(drop, bool) or not isinstance(drop, (int, np.integer)) or
                             drop < 0):
        raise ValueError(f'miri_drop_groups must be None or a non-negative integer, got '
            f'{drop!r}')

    post_ramp = ('GainScaleStep', 'AssignWCSStep', 'FlatFieldStep', 'BackgroundStep', 'BadPixStep',
                 'PCAReconstructStep')
    return _finish_context(ctx, post_ramp)


def miri_steps(opts=None):
    """Build the enabled MIRI steps in v1 order.

    Parameters
    ----------
    opts : None, dict
        Reduction options controlling the enabled steps.

    Returns
    -------
    steps : list[Step]
        Enabled calibration and extraction steps.
    """
    return _enabled_steps(opts, [('DQInitStep', [('DQInitStep', step_dq_init)]),
        ('EmiCorrStep', [('EmiCorrStep', step_emicorr)]), ('ResetStep', []),
        ('LinearityStep', [('LinearityStep', step_linearity_miri)]),
        ('JumpStep', [('JumpStep', step_jump, _JUMP_PARAMS)]),
        ('RampFitStep', [('RampFitStep', step_rampfit)]),
        ('GainScaleStep', [('GainScaleStep', step_gain_scale)]),
        ('AssignWCSStep', [('AssignWCSStep', step_assign_wcs_miri)]), ('SourceTypeStep', []),
        ('FlatFieldStep', [('FlatFieldStep', step_flat)]),
        ('BackgroundStep', [('BackgroundStep', step_background_miri,
                            ('miri_trace_width', 'miri_background_width'))]),
        ('BadPixStep', [('BadPixStep', step_badpix, _BADPIX_PARAMS)]),
        ('PCAReconstructStep', [('PCAReconstructStep', step_pca)]),], step_extract_miri)


def prepare_miri_context(cube, opts, refpack=None):
    """Assemble static references and explicit centroid inputs.

    Explicit centroids stay fixed; automatic centroids follow the processed stages.

    Parameters
    ----------
    cube : RampCube, RateCube
        Input science cube and observation metadata.
    opts : dict
        Reduction options.
    refpack : None, RefPack
        Reference arrays; resolve them when None.

    Returns
    -------
    ctx : dict
        Validated options, references, and explicit centroid inputs.
    """
    opts = dict(opts)
    search_dirs = (opts.get('input_dir', ''), os.getcwd())
    ctx = _initial_context(cube, opts, refpack, search_dirs)
    ctx['miri_trace_center'] = MIRI_TRACE_CENTER
    ctx['miri_ystart'] = trace.MIRI_TRACE_YSTART

    _context_wave_map(ctx, cube, 'MIRI')
    from exotedrf.v2 import wavecal
    wavecal.warn_miri_stellar(opts)
    ctx['waves'] = None
    ctx['wavemap_provenance'] = _rp(ctx, 'meta_file_wavemap')

    _context_centroids(ctx, search_dirs, trace.validate_miri_centroids, cube.data.shape[-2])
    ctx['custom_deepframe'] = _load_custom_deepframe(
        opts.get('deepframe'), cube.data.shape[-2:], search_dirs)
    _validate_miri_context(ctx)
    return ctx


def _validate_soss_context(ctx):
    """Validate SOSS controls and references and release the input cube."""
    opts = ctx.get('opts', {})
    for name in _STEP_CONTROLS:
        _runs(opts, name)


    method = opts.get('oof_method', 'scale-achromatic')
    supported_oof = ('scale-achromatic', 'scale-achromatic-window',
                     'scale-chromatic', 'solve')
    if method not in supported_oof:
        raise ValueError(
            f'oof_method must be one of {supported_oof}, got {method!r}')

    extract_method = opts.get('extract_method', 'box')
    if extract_method not in ('box', 'atoca'):
        raise ValueError(
            "SOSS v2 supports extract_method='box' or 'atoca', got "
            f'{extract_method!r}')
    for key in ('extract_width_soss2',):
        value = opts.get(key)
        if value is None or value == 'optimize' or \
                isinstance(value, v2config.ExtractAperture):
            continue
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"{key} must be a positive scalar, 'optimize', a two-sided "
                '(lower_width, upper_width) aperture, or None')

    sb_method = opts.get('superbias_method', 'crds')
    if sb_method != 'crds':
        raise NotImplementedError(
            f'superbias_method={sb_method!r} is not supported in SOSS v2 '
            '(v1 always uses crds for NIRISS)')

    post_ramp = ('GainScaleStep', 'AssignWCSStep',
                 'FlatFieldStep', 'BackgroundStep', 'OneOverFStep_int',
                 'BadPixStep', 'PCAReconstructStep')
    if not _runs(opts, 'RampFitStep') and any(
            _runs(opts, key) for key in post_ramp) and \
            opts.get('restart_input_ndim') != 3:
        raise ValueError('RampFitStep cannot be skipped while stage-2 steps run')
    if (_runs(opts, 'OneOverFStep_grp') or _runs(opts, 'BackgroundStep')) and \
            ctx.get('background_model') is None:
        raise ValueError('enabled SOSS background/1/f steps require a background model')
    if method == 'scale-chromatic' and (
            ctx.get('soss_timeseries') is None or
            ctx.get('soss_timeseries_o2') is None):
        raise ValueError(
            'scale-chromatic requires 2D soss_timeseries and soss_timeseries_o2')
    _validate_pca_options(opts)

    cube = ctx.pop('input_cube', None)
    if cube is not None:
        refs.validate_refpack(ctx.get('refpack'), cube, opts,
                              require_waves=True)
    del cube
    return ctx

def soss_steps(opts=None):
    """Build the enabled SOSS steps in v1 order.

    Group-level background subtraction runs with OneOverFStep_grp.

    Parameters
    ----------
    opts : None, dict
        Reduction options controlling the enabled steps.

    Returns
    -------
    steps : list[Step]
        Enabled calibration and extraction steps.
    """
    opts = {} if opts is None else opts
    extract = (step_extract_atoca if str(opts.get('extract_method', 'box')).lower() == 'atoca'
               else step_extract)
    oof_params = ('soss_inner_mask_width', 'soss_outer_mask_width')
    return _enabled_steps(opts, [('DQInitStep', [('DQInitStep', step_dq_init)]),
        ('INLCorrStep', [('INLCorrStep', step_inl)]),
        ('SuperBiasStep', [('SuperBiasStep', step_superbias)]),
        ('RefPixStep', [('RefPixStep', step_refpix)]),
        ('DarkCurrentStep', [('DarkCurrentStep', step_dark)]),
        ('OneOverFStep_grp', [('BackgroundStep_grp', step_background_grp),
                            ('OneOverFStep_grp', step_oneoverf_grp, oof_params)]),
        ('LinearityStep', [('LinearityStep', step_linearity)]),
        ('JumpStep', [('JumpStep', step_jump, _JUMP_PARAMS)]),
        ('RampFitStep', [('RampFitStep', step_rampfit)]),
        ('GainScaleStep', [('GainScaleStep', step_gain_scale)]),
        ('AssignWCSStep', [('AssignWCSStep', step_assign_wcs)]), ('SourceTypeStep', []),
        ('FlatFieldStep', [('FlatFieldStep', step_flat)]),
        ('BackgroundStep', [('BackgroundStep', step_background_int)]),
        ('OneOverFStep_int', [('OneOverFStep_int', step_oneoverf_int, oof_params)]),
        ('BadPixStep', [('BadPixStep', step_badpix, _BADPIX_PARAMS)]),
        ('PCAReconstructStep', [('PCAReconstructStep', step_pca)]),], extract)


def build_pipeline(mode, ctx):
    """Create the ordered reduction for one prepared observation.

    Parameters
    ----------
    mode : str
        Instrument and observing mode.
    ctx : dict
        Observation options, reference arrays, and trace information.

    Returns
    -------
    pipeline : Pipeline
        Enabled calibration and extraction steps with the observation context.
    """
    from exotedrf.v2.pipeline import Pipeline
    handlers = {'NIRISS': (_validate_soss_context, soss_steps),
                'NIRSPEC': (_validate_nirspec_context, nirspec_steps),
                'MIRI': (_validate_miri_context, miri_steps)}
    for instrument, (validate, steps) in handlers.items():
        if mode.upper().startswith(instrument):
            validate(ctx)
            return Pipeline(steps(ctx['opts']), mode, ctx)
    raise NotImplementedError(f'{mode}: unsupported observing mode')


def prepare_context(cube, opts, refpack=None, files_dir=None):
    """Assemble static references and explicit centroid inputs.

    Explicit centroids stay fixed; automatic centroids follow the processed stages.

    Parameters
    ----------
    cube : RampCube, RateCube
        Input science cube and observation metadata.
    opts : dict
        Reduction options.
    refpack : None, RefPack
        Reference arrays; resolve them when None.
    files_dir : None, str
        Directory containing the bundled SOSS reference files.

    Returns
    -------
    ctx : dict
        Validated options, references, and explicit centroid inputs.
    """
    mode = str(opts.get('mode') or cube.meta.mode)
    if mode.upper().startswith('NIRSPEC'):
        return prepare_nirspec_context(cube, opts, refpack=refpack)
    if mode.upper().startswith('MIRI'):
        return prepare_miri_context(cube, opts, refpack=refpack)
    opts = dict(opts)
    if files_dir is None:
        # Find bundled reference files in the checkout or installed environment.
        checkout_files = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'files')
        installed_files = os.path.join(sys.prefix, 'files')
        files_dir = next((path for path in (checkout_files, installed_files)
             if os.path.isdir(path)), checkout_files)
    search_dirs = (opts.get('input_dir', ''), files_dir, os.getcwd())
    ctx = _initial_context(cube, opts, refpack, search_dirs)

    model = None
    bg = opts.get('soss_background_file')
    if bg is not None:
        model = _load_array_option(bg, 'soss_background_file', search_dirs)
        model = np.asarray(model, dtype=cube.data.dtype)
        if model.shape != cube.data.shape[-2:]:
            raise ValueError(f'SOSS background model has shape {model.shape}, expected '
                f'{cube.data.shape[-2:]}')
    ctx['background_model'] = model

    f277w = _load_array_option(opts.get('f277w'), 'f277w', search_dirs)
    order0 = None
    if f277w is not None:
        f277w = np.asarray(f277w, dtype=float)
        if f277w.shape != cube.data.shape[-2:]:
            raise ValueError(f'f277w exposure has shape {f277w.shape}, expected '
                f'{cube.data.shape[-2:]}')
        order0 = trace.make_order0_mask_from_f277w(f277w)
    # Retain the F277W detector plane for contaminant diagnostics.
    ctx['f277w'] = f277w
    ctx['order0_mask'] = order0

    nints, dimx = cube.data.shape[0], cube.data.shape[-1]
    slices = _segment_slices(cube)
    max_int_end = max(int(start) + seg.stop - seg.start - 1
                      for start, seg in zip(cube.meta.segment_int_starts, slices))
    method = opts.get('oof_method', 'scale-achromatic')
    for key in ('soss_timeseries', 'soss_timeseries_o2'):
        ctx[key] = _validate_timeseries(
            _load_array_option(opts.get(key), key, search_dirs), key, nints, dimx,
            require_2d=(method == 'scale-chromatic'), max_int_end=max_int_end,
            dtype=cube.data.dtype)

    tracetable = trace.resolve_soss_tracetable(cube.meta.subarray, (files_dir,))
    ctx['centroid_tracetable'] = tracetable
    if ctx['centroid_tracetable'] is None and opts.get('centroids') is None:
        warnings.warn(f'SOSS tracetable {trace.soss_tracetable_name(cube.meta.subarray)}'
            f' not found in {files_dir} or $CRDS_PATH/references/jwst/niriss '
            'and could not be downloaded from GitHub (v1 2.5.0 source); '
            'automatic order-2 centroids will be traced without it and will '
            'differ from a reduction that has the file', RuntimeWarning, stacklevel=2)
    _context_centroids(ctx, search_dirs, trace.validate_centroids, dimx)

    ctx['waves'] = {1: np.asarray(_rp(ctx, 'wave_o1'), dtype=float),
        2: np.asarray(_rp(ctx, 'wave_o2'), dtype=float),}
    ctx['wavemap_provenance'] = _rp(ctx, 'meta_file_wavemap')
    github_waves = trace.soss_github_wave_vectors(
        files_dir, 'SUBSTRIP96' if cube.data.shape[-2] == 96 else 'SUBSTRIP256')
    if github_waves is not None and github_waves[1].shape[0] == dimx:
        ctx['waves'] = github_waves
        ctx['wavemap_provenance'] = github_waves.pop('name')
    if opts.get('use_pastasoss', False):
        import warnings
        warnings.warn('use_pastasoss is no longer supported (exoTEDRF 2.5.0); '
                      'falling back on the default wavelength solution.',
                      RuntimeWarning, stacklevel=2)
    ctx['custom_deepframe'] = _load_custom_deepframe(
        opts.get('deepframe'), cube.data.shape[-2:], search_dirs)
    _validate_soss_context(ctx)
    return ctx
