"""Segment scheduling: device residency and exactness against the joined run."""

import numpy as np
import pytest

from exotedrf.v2 import core, products, stages, streaming
from exotedrf.v2.pipeline import CheckpointStore, PipelineState

from .test_optimal_diagnostics import _ramps, _state
from .test_soss_e2e import build_synthetic_soss_case


def _case(group_oof=True):
    """Return case."""
    cube, ctx, params = build_synthetic_soss_case(group_oof=group_oof)
    return cube, ctx, params


def test_segment_state_uploads_science_arrays_once_per_segment():
    """Check segment state uploads science arrays once per segment."""
    cube, _, _ = _case()
    state = PipelineState(cube)
    slices = core.segment_slices(cube.meta, cube.data.shape[0])

    for index, seg in enumerate(slices):
        part = streaming.segment_state(state, index)
        assert core.is_device_array(part.cube.data)
        assert core.is_device_array(part.cube.groupdq)
        # Detector-level flags and metadata stay on the host.
        assert isinstance(part.cube.pixeldq, np.ndarray)
        assert not core.is_device_array(part.cube.pixeldq)
        # The upload is a copy, not a reinterpretation.
        assert part.cube.data.dtype == cube.data.dtype
        assert part.cube.groupdq.dtype == cube.groupdq.dtype
        np.testing.assert_array_equal(core.to_host(part.cube.data),
                                      cube.data[seg])
        np.testing.assert_array_equal(core.to_host(part.cube.groupdq),
                                      cube.groupdq[seg])
        np.testing.assert_array_equal(part.cube.pixeldq, cube.pixeldq)
        # Segment-local bookkeeping, exposure-global timing.
        count = seg.stop - seg.start
        assert part.cube.meta.segment_edges.tolist() == [count]
        np.testing.assert_array_equal(part.cube.meta.int_times,
                                      cube.meta.int_times[seg])
        assert part.cube.meta.filenames == cube.meta.filenames[index:index + 1]


def test_to_host_round_trip_of_a_segment_state_is_lossless():
    """Check to host round trip of a segment state is lossless."""
    cube, _, _ = _case()
    part = streaming.segment_state(PipelineState(cube), 0)
    back = core.to_host(part.cube)
    assert isinstance(back.data, np.ndarray) and isinstance(
        back.groupdq, np.ndarray)
    np.testing.assert_array_equal(back.data, cube.data[:6])
    np.testing.assert_array_equal(back.groupdq, cube.groupdq[:6])
    assert back.meta is part.cube.meta


@pytest.mark.parametrize('group_oof', [False, True])
def test_run_soss_segmented_equals_joined_pipeline_run(group_oof, monkeypatch):
    """Check run SOSS segmented equals joined pipeline run."""
    monkeypatch.setenv(streaming.DEVICE_RESIDENT_ENV, 'off')
    cube, ctx, params = _case(group_oof=group_oof)
    pipeline = stages.build_pipeline('NIRISS/SOSS', ctx)

    joined = pipeline.run(PipelineState(cube), params)
    streamed = streaming.run_soss_segmented(
        pipeline, PipelineState(cube), params,
        store=CheckpointStore(keep={'Extract'}))

    # Bit-exact: the schedule may not change a single calibrated value.
    for name in ('data', 'err', 'dq'):
        streamed_array = getattr(streamed.cube, name)
        joined_array = getattr(joined.cube, name)
        # Default policy on a host backend: spillable host join buffers.
        assert not core.is_device_array(streamed_array)
        assert streamed_array.dtype == joined_array.dtype
        np.testing.assert_array_equal(streamed_array, joined_array)
    for order in (1, 2):
        for name in ('wave', 'flux', 'ferr'):
            np.testing.assert_array_equal(
                streamed.aux['spectral_products'][order][name],
                joined.aux['spectral_products'][order][name])
    np.testing.assert_array_equal(streamed.aux['pca_components'],
                                  joined.aux['pca_components'])
    np.testing.assert_array_equal(np.asarray(streamed.aux['deepframe']),
                                  np.asarray(joined.aux['deepframe']))


@pytest.mark.parametrize('overrides', [
    {'oof_method': 'scale-achromatic-window'},
    {'saturation_rescue': True},
    {'mask_do_not_use_pixels': False},
])
def test_segmented_run_is_exact_for_other_supported_soss_options(overrides):
    """Option branches the production YAML does not use must stream too."""
    cube, ctx, params = _case()
    ctx['opts'].update(overrides)
    pipeline = stages.build_pipeline('NIRISS/SOSS', ctx)
    joined = pipeline.run(PipelineState(cube), params)
    streamed = streaming.run_soss_segmented(
        pipeline, PipelineState(cube), params,
        store=CheckpointStore(keep={'Extract'}))
    for name in ('data', 'err', 'dq'):
        np.testing.assert_array_equal(getattr(streamed.cube, name),
                                      getattr(joined.cube, name))
    for order in (1, 2):
        np.testing.assert_array_equal(
            streamed.aux['spectral_products'][order]['flux'],
            joined.aux['spectral_products'][order]['flux'])


def test_compatibility_capture_for_segment_selects_first_and_final_dq(
        tmp_path):
    """Standard-mode quick looks follow the same v1 segment selection."""
    before, after = _ramps()
    first, last = _state(before), _state(after, start=25, saturated=True)
    capture = products.CompatibilityCapture(
        products.output_layout({}, output_dir=tmp_path),
        {'opts': {'do_plots': True}})
    assert not capture.segmented
    first_observer = capture.for_segment(0, 3)
    middle_observer = capture.for_segment(1, 3)
    last_observer = capture.for_segment(2, 3)
    assert capture.segmented
    assert first_observer is capture

    bias = type('Step', (), {'name': 'SuperBiasStep'})()
    dq = type('Step', (), {'name': 'DQInitStep'})()
    first_observer(bias, first, first)
    first_observer(dq, first, first)
    recorded = capture.records['SuperBiasStep']
    # Only the first segment contributes stage-1 panels .
    middle_observer(bias, last, last)
    last_observer(bias, last, last)
    assert capture.records['SuperBiasStep'] is recorded
    assert not capture.records['DQInitStep'].any()
    # ... and only the last contributes v1's final-integration saturation map.
    middle_observer(dq, last, last)
    assert not capture.records['DQInitStep'].any()
    last_observer(dq, last, last)
    assert capture.records['DQInitStep'].all()

    with pytest.raises(ValueError, match='segment index'):
        capture.for_segment(3, 3)
    with pytest.raises(ValueError, match='segment index'):
        capture.for_segment(0, 0)


def test_disabled_compatibility_capture_still_supports_segments(tmp_path):
    """Check disabled compatibility capture still supports segments."""
    before, _ = _ramps()
    state = _state(before)
    capture = products.CompatibilityCapture(
        products.output_layout({}, output_dir=tmp_path),
        {'opts': {'do_plots': False}})
    observer = capture.for_segment(1, 2)
    observer(type('Step', (), {'name': 'DQInitStep'})(), state, state)
    assert capture.records == {}


# Overlapped rate fill and an accelerator-resident join.

def test_rampfit_defers_its_rate_fill_to_a_supplied_pool():
    """With a pool the step hands back unfilled planes plus a future."""
    from concurrent.futures import ThreadPoolExecutor

    cube, ctx, params = _case()
    pipeline = stages.build_pipeline('NIRISS/SOSS', ctx)
    index = [step.name for step in pipeline.steps].index('RampFitStep')
    before = pipeline.run(PipelineState(cube), params, stop=index)

    inline = stages.step_rampfit(before, params, ctx)
    assert stages._RAMPFIT_FILL_FUTURE_KEY not in inline.aux

    with ThreadPoolExecutor(max_workers=1) as pool:
        local = dict(ctx)
        local[stages._RAMPFIT_FILL_EXECUTOR_KEY] = pool
        deferred = stages.step_rampfit(before, params, local)
        future = deferred.aux[stages._RAMPFIT_FILL_FUTURE_KEY]
        assert not future.cancelled()
        resolved = stages.await_rampfit_fill(deferred)

    # Identical planes, and no scheduling bookkeeping left in aux.
    assert stages._RAMPFIT_FILL_FUTURE_KEY not in resolved.aux
    assert resolved.aux.keys() == inline.aux.keys()
    for name in ('data', 'err', 'dq'):
        assert np.array_equal(np.asarray(getattr(resolved.cube, name)),
                              np.asarray(getattr(inline.cube, name)),
                              equal_nan=True), name


def test_await_rampfit_fill_is_a_no_op_without_a_deferred_fill():
    """Check await rampfit fill is a no op without a deferred fill."""
    state = PipelineState(object(), {'centroids': 'kept'})
    assert stages.await_rampfit_fill(state) is state


def test_await_rampfit_fill_reraises_a_failed_fill():
    """Check await rampfit fill reraises a failed fill."""
    from concurrent.futures import ThreadPoolExecutor

    def explode():
        raise RuntimeError('griddata refused')

    with ThreadPoolExecutor(max_workers=1) as pool:
        state = PipelineState(
            object(), {stages._RAMPFIT_FILL_FUTURE_KEY: pool.submit(explode)})
        with pytest.raises(RuntimeError, match='griddata refused'):
            stages.await_rampfit_fill(state)


def test_streamed_fill_overlaps_the_following_segment(monkeypatch):
    """Segment i's KD-tree fill runs while segment i+1 is being calibrated."""
    import threading

    original_fill = stages._nearest_fill_rate_planes
    original_rampfit = stages.step_rampfit
    started, release = threading.Event(), threading.Event()
    overlapped = []
    ramps = []

    def blocking_fill(rate, *, copy=True):
        started.set()
        release.wait(5.)
        return original_fill(rate, copy=copy)

    def watched_rampfit(state, params_, context):
        ramps.append(context.get(stages._RAMPFIT_FILL_EXECUTOR_KEY))
        if len(ramps) == 2:
            overlapped.append(started.is_set() or started.wait(5.))
            release.set()
        return original_rampfit(state, params_, context)

    monkeypatch.setattr(stages, '_nearest_fill_rate_planes', blocking_fill)
    monkeypatch.setattr(stages, 'step_rampfit', watched_rampfit)
    try:
        cube, ctx, params = _case()
        # Built after the patch so the graph identity check still matches.
        pipeline = stages.build_pipeline('NIRISS/SOSS', ctx)
        streamed = streaming.run_soss_segmented(
            pipeline, PipelineState(cube), params)
    finally:
        release.set()

    assert overlapped == [True]
    assert len(ramps) == 2 and all(pool is not None for pool in ramps)

    monkeypatch.undo()
    joined = stages.build_pipeline('NIRISS/SOSS', ctx).run(
        PipelineState(cube), params)
    for name in ('data', 'err', 'dq'):
        assert np.array_equal(np.asarray(getattr(streamed.cube, name)),
                              np.asarray(getattr(joined.cube, name)),
                              equal_nan=True), name


def test_single_segment_visits_fill_inline(monkeypatch):
    """Nothing to overlap with, so no pool and no deferred future."""
    import dataclasses

    cube, ctx, params = _case()
    meta = dataclasses.replace(
        cube.meta, segment_edges=np.asarray([cube.data.shape[0]]),
        filenames=cube.meta.filenames[:1],
        extra=dict(cube.meta.extra, segment_int_starts=(1,)))
    single = dataclasses.replace(cube, meta=meta)

    seen = []
    original = stages.step_rampfit

    def watched(state, params_, context):
        seen.append(context.get(stages._RAMPFIT_FILL_EXECUTOR_KEY))
        return original(state, params_, context)

    monkeypatch.setattr(stages, 'step_rampfit', watched)
    pipeline = stages.build_pipeline('NIRISS/SOSS', ctx)
    streaming.run_soss_segmented(pipeline, PipelineState(single), params)
    assert seen == [None]


@pytest.mark.parametrize('group_oof', [False, True])
def test_device_resident_join_is_bit_exact(monkeypatch, group_oof):
    """Joining in accelerator memory changes residency, never a value."""
    monkeypatch.setenv(streaming.DEVICE_RESIDENT_ENV, 'on')
    cube, ctx, params = _case(group_oof=group_oof)
    pipeline = stages.build_pipeline('NIRISS/SOSS', ctx)
    streamed = streaming.run_soss_segmented(
        pipeline, PipelineState(cube), params,
        store=CheckpointStore(keep={'Extract'}))

    monkeypatch.setenv(streaming.DEVICE_RESIDENT_ENV, 'off')
    host = streaming.run_soss_segmented(
        pipeline, PipelineState(cube), params,
        store=CheckpointStore(keep={'Extract'}))

    # Stage 2 ran on the accelerator and its result stayed there.
    for name in ('data', 'err', 'dq'):
        assert core.is_device_array(getattr(streamed.cube, name))
        assert not core.is_device_array(getattr(host.cube, name))
        assert np.array_equal(core.to_host(getattr(streamed.cube, name)),
                              getattr(host.cube, name), equal_nan=True), name
    for order in (1, 2):
        for name in ('wave', 'flux', 'ferr'):
            assert np.array_equal(
                np.asarray(streamed.aux['spectral_products'][order][name]),
                np.asarray(host.aux['spectral_products'][order][name]),
                equal_nan=True), (order, name)


def test_device_resident_join_falls_back_when_it_does_not_fit(monkeypatch):
    """Check device resident join falls back when it does not fit."""
    monkeypatch.setenv(streaming.DEVICE_RESIDENT_ENV, 'on')
    monkeypatch.setenv('EXOTEDRF_MAX_DEVICE_BYTES', '4096')
    cube, ctx, params = _case()
    pipeline = stages.build_pipeline('NIRISS/SOSS', ctx)
    streamed = streaming.run_soss_segmented(
        pipeline, PipelineState(cube), params,
        store=CheckpointStore(keep={'Extract'}))
    assert not core.is_device_array(streamed.cube.data)


def test_device_join_policy_rejects_unknown_values(monkeypatch):
    """Check device join policy rejects unknown values."""
    monkeypatch.setenv(streaming.DEVICE_RESIDENT_ENV, 'sometimes')
    with pytest.raises(ValueError, match=streaming.DEVICE_RESIDENT_ENV):
        streaming._device_residency_allowed()


def test_device_resident_products_match_the_host_join(monkeypatch, tmp_path):
    """The standard products read a resident final state without change."""
    from astropy.io import fits

    cube, ctx, params = _case()
    pipeline = stages.build_pipeline('NIRISS/SOSS', ctx)
    written = {}
    for policy, tag in (('on', 'device'), ('off', 'host')):
        monkeypatch.setenv(streaming.DEVICE_RESIDENT_ENV, policy)
        final = streaming.run_soss_segmented(
            pipeline, PipelineState(cube), params,
            store=CheckpointStore(keep={'Extract'}))
        paths = products.output_layout({'name_tag': tag},
                                       output_dir=tmp_path / tag)
        written[tag] = products.write_final_products(
            final, params, ctx, paths, output_mode='standard')
    assert 'rate' in written['device'] and 'spectra' in written['device']
    for key in ('rate', 'spectra'):
        with fits.open(written['device'][key]) as device, \
                fits.open(written['host'][key]) as host:
            assert len(device) == len(host)
            for left, right in zip(device[1:], host[1:]):
                if left.data is None or left.data.dtype.names is not None:
                    continue
                assert np.array_equal(np.asarray(left.data),
                                      np.asarray(right.data),
                                      equal_nan=True), key
