"""Check stages."""

import jax
import numpy as np
import pytest

from exotedrf.v2 import core, stages
from exotedrf.v2.core import ObsMeta, RampCube, RateCube
from exotedrf.v2.pipeline import PipelineState


STEP_KEYS = (
    'DQInitStep', 'INLCorrStep', 'SuperBiasStep', 'RefPixStep',
    'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep', 'JumpStep',
    'RampFitStep', 'GainScaleStep', 'AssignWCSStep',
    'FlatFieldStep', 'BackgroundStep', 'OneOverFStep_int', 'BadPixStep',
    'PCAReconstructStep')


def controls(value='skip', **updates):
    """Create step controls with the supplied overrides.

    Parameters
    ----------
    value : str
        Default run or skip control.
    updates : dict
        Overrides for the returned configuration.

    Returns
    -------
    options : dict
        Step controls with overrides applied.
    """
    opts = {key: value for key in STEP_KEYS}
    opts.update(updates)
    return opts


def meta(nints, edges=None, starts=None, ngroups=3, dimx=4):
    """Create metadata for a synthetic observation.

    Parameters
    ----------
    nints : int
        Number of integrations.
    edges : None, array-like(int)
        Integration boundaries between segments.
    starts : None, array-like(int)
        First integration index of each segment.
    ngroups : int
        Number of groups per integration.
    dimx : int
        Number of detector columns.

    Returns
    -------
    meta : ObsMeta
        Synthetic observation metadata.
    """
    edges = np.asarray(edges if edges is not None else [nints])
    extra = {} if starts is None else {'segment_int_starts': tuple(starts)}
    return ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP96', 2.214, ngroups,
                   np.arange(nints, dtype=float), np.array([nints]), edges,
                   tuple(f'seg{i}.fits' for i in range(len(edges))), extra)


def test_soss_graph_has_exact_enabled_order_and_nested_group_background():
    """Check SOSS graph has exact enabled order and nested group background."""
    opts = controls('run')
    names = [step.name for step in stages.soss_steps(opts)]
    assert names == [
        'DQInitStep', 'INLCorrStep', 'SuperBiasStep', 'RefPixStep',
        'DarkCurrentStep', 'BackgroundStep_grp', 'OneOverFStep_grp',
        'LinearityStep', 'JumpStep', 'RampFitStep', 'GainScaleStep',
        'AssignWCSStep', 'FlatFieldStep',
        'BackgroundStep', 'OneOverFStep_int', 'BadPixStep',
        'PCAReconstructStep', 'Extract']

    opts['OneOverFStep_grp'] = 'skip'
    names = [step.name for step in stages.soss_steps(opts)]
    assert 'OneOverFStep_grp' not in names
    assert 'BackgroundStep_grp' not in names


def test_soss_graph_rejects_nonlowercase_control():
    """Check SOSS graph rejects nonlowercase control."""
    with pytest.raises(ValueError, match='exactly lowercase'):
        stages.soss_steps(controls(DQInitStep='RUN'))


def test_jump_temporal_kernel_resets_at_segment_edges(monkeypatch):
    """Check jump temporal kernel resets at segment edges."""
    data = np.zeros((5, 3, 4, 4), np.float32)
    cube = RampCube(data, np.zeros_like(data, np.uint8),
                    np.zeros((4, 4), np.uint32),
                    meta(5, edges=[3, 5], starts=[10, 50]))
    calls = []

    def fake(data_part, dq_part, threshold, window, artifact):
        calls.append((data_part.shape[0], np.asarray(artifact)))
        return data_part, dq_part

    monkeypatch.setattr(stages.k_jump, 'flag_jumps_in_time_chunked', fake)
    state = stages.step_jump(
        PipelineState(cube), {'time_jump_threshold': 10, 'time_window': 5},
        {'opts': {'flag_up_ramp': False, 'flag_in_time': True}})
    # For NGROUPS > 2 the time-domain detector changes only GROUPDQ.
    assert state.cube.data is data
    assert state.cube.data.shape == data.shape
    assert [length for length, _ in calls] == [3, 2]
    assert calls[0][1].shape[0] == 3
    assert calls[1][1].shape[0] == 2


def test_two_group_jump_forces_time_domain_like_v1(monkeypatch):
    """Check two group jump forces time domain like v1."""
    data = np.zeros((3, 2, 2, 2), np.float32)
    cube = RampCube(data, np.zeros_like(data, np.uint8),
                    np.zeros((2, 2), np.uint32), meta(3, ngroups=2))
    calls = []

    def fake(data_part, dq_part, threshold, window, artifact):
        calls.append(data_part.shape[0])
        return data_part, dq_part

    monkeypatch.setattr(stages.k_jump, 'flag_jumps_in_time_chunked', fake)
    stages.step_jump(
        PipelineState(cube), {'time_jump_threshold': 10, 'time_window': 5},
        {'opts': {'flag_up_ramp': False, 'flag_in_time': False}})
    assert calls == [3]


def test_prepared_jump_scorer_matches_full_step_costs(monkeypatch):
    """Check prepared jump scorer matches full step costs."""
    rng = np.random.default_rng(0x123041a)
    nints, ngroups, dimy, dimx = 9, 3, 10, 12
    data = rng.normal(100., .2, (nints, ngroups, dimy, dimx)).astype(
        np.float32)
    data[4, -1, 5, 6] += 25.
    data[6, -1, 4, 8] += 15.
    groupdq = np.zeros_like(data, np.uint8)
    cube = RampCube(
        data, groupdq, np.zeros((dimy, dimx), np.uint32),
        meta(nints, ngroups=ngroups, dimx=dimx))
    state = PipelineState(cube)
    ctx = {
        'opts': {'flag_up_ramp': False, 'flag_in_time': True,
                 'extract_width_soss2': None,
                 'wave_range': None, 'w1': 1., 'w2': 1.},
        'centroids': {'ypos o1': np.linspace(4.5, 5.5, dimx)},
        'waves': {1: np.linspace(.9, 2., dimx)},
    }
    params = {'time_jump_threshold': 7., 'time_window': 5,
              'extract_width': 4.}
    scorer = stages.prepare_jump_scorer(state, params, ctx)
    monkeypatch.setenv('EXOTEDRF_CHUNK_COLS', '4')

    for parameter, candidates in (
            ('time_jump_threshold', [5., 7., 9.]),
            ('time_window', [3, 5, 7])):
        fast = scorer.evaluate_candidates(parameter, candidates, params)
        for value, (fast_cost, fast_scatter, _) in zip(candidates, fast):
            trial = dict(params, **{parameter: value})
            full = stages.step_jump(state, trial, ctx)
            cost, scatter = stages.evaluate_optimizer_cost(full, trial, ctx)
            np.testing.assert_allclose(
                fast_cost, cost, rtol=1e-5, atol=1e-7)
            assert fast_scatter.shape == np.asarray(scatter).shape
            np.testing.assert_allclose(
                fast_scatter, scatter, rtol=1e-5, atol=1e-7,
                equal_nan=True)


def test_dq_init_only_applies_explicit_hot_map(monkeypatch):
    """Check DQ init only applies explicit hot map."""
    data = np.ones((2, 2, 4, 12), np.float32)
    cube = RampCube(data, np.zeros_like(data, np.uint8),
                    np.zeros((4, 12), np.uint32),
                    meta(2, ngroups=2, dimx=12))
    hot = np.zeros((4, 12), bool)
    hot[1, 6] = True

    def forbidden(*args, **kwargs):
        raise AssertionError('DQInit must not auto-detect hot pixels')

    monkeypatch.setattr(stages.k_det, 'flag_hot_pixels', forbidden)
    base_ctx = {
        'opts': {'saturation_threshold': 80},
        'refpack': {
            'mask_dq': np.zeros((4, 12), np.uint32),
        },
    }
    without = stages.step_dq_init(PipelineState(cube), {}, base_ctx)
    assert not np.asarray(without.cube.pixeldq).any()
    with_hot = stages.step_dq_init(
        PipelineState(cube), {}, dict(base_ctx, hot_pixel_map=hot))
    assert np.asarray(with_hot.cube.pixeldq)[1, 6] == stages.core.DQ_HOT
    np.testing.assert_array_equal(with_hot.aux['hot_pixels'], hot)


def test_dq_init_propagates_only_mask_do_not_use_to_every_group():
    """Match jwst DQInitStep's mask PIXELDQ/GROUPDQ contract."""
    data = np.ones((2, 3, 4, 12), np.float32)
    groupdq = np.zeros_like(data, np.uint8)
    groupdq[0, 1, 0, 0] = core.DQ_JUMP_DET
    pixeldq = np.zeros((4, 12), np.uint32)
    pixeldq[3, 9] = core.DQ_DO_NOT_USE
    cube = RampCube(data, groupdq, pixeldq,
                    meta(2, ngroups=3, dimx=12))
    mask = np.zeros((4, 12), np.uint32)
    mask[1, 5] = core.DQ_DO_NOT_USE | core.DQ_HOT
    mask[2, 7] = core.DQ_HOT

    out = stages.step_dq_init(
        PipelineState(cube), {},
        {'opts': {'saturation_threshold': 80},
         'refpack': {'mask_dq': mask}})

    np.testing.assert_array_equal(
        np.asarray(out.cube.pixeldq), np.bitwise_or(pixeldq, mask))
    propagated = np.asarray(out.cube.groupdq)
    assert np.all(
        propagated[:, :, 1, 5] & np.uint8(core.DQ_DO_NOT_USE))
    # Other mask flags remain detector-level, exactly as jwst.dq_init does.
    assert not np.any(propagated[:, :, 2, 7])
    assert not np.any(propagated[:, :, 3, 9])
    # Existing GROUPDQ flags are preserved.
    assert propagated[0, 1, 0, 0] & np.uint8(core.DQ_JUMP_DET)


def test_dq_init_ignores_crds_saturation_plane_but_keeps_custom_pass():
    """Check DQ init ignores CRDS saturation plane but keeps custom pass."""
    data = np.zeros((1, 2, 4, 12), np.float32)
    data[0, 1, 2, 6] = 50000.
    cube = RampCube(data, np.zeros_like(data, np.uint8),
                    np.zeros((4, 12), np.uint32),
                    meta(1, ngroups=2, dimx=12))
    common = {
        'opts': {'saturation_threshold': 80},
        'refpack': {'mask_dq': np.zeros((4, 12), np.uint32)},
    }
    without_ref = stages.step_dq_init(PipelineState(cube), {}, common)
    with_extreme_ref = stages.step_dq_init(
        PipelineState(cube), {},
        {'opts': common['opts'],
         'refpack': dict(common['refpack'],
                         saturation=np.zeros((4, 12), np.float32))})

    np.testing.assert_array_equal(
        with_extreme_ref.cube.groupdq, without_ref.cube.groupdq)
    saturated = (np.asarray(without_ref.cube.groupdq) & 2) != 0
    assert saturated[0, 1, 2, 6]
    # The separate custom v1 pass grows the saturated pixel by one pixel.
    assert saturated[0, 1, 1:4, 5:8].all()
    assert saturated.sum() == 9


def test_badpix_streams_columns_and_reuses_first_segment_map(monkeypatch):
    """Check bad-pixel correction streams columns and reuses first segment map."""
    rng = np.random.default_rng(22)
    nints, dimy, dimx = 24, 10, 28
    data = rng.normal(100., 0.2, (nints, dimy, dimx)).astype(np.float32)
    err = np.ones_like(data)
    dq = np.zeros_like(data, np.uint32)
    # First segment establishes x=10.
    dq[10, 2, 10] = stages.core.DQ_HOT
    dq[12:, 2, 10] = stages.core.DQ_HOT
    dq[22, 2, 17] = stages.core.DQ_HOT
    cube = RateCube(
        data, err, dq, meta(nints, edges=[12, 24], ngroups=3, dimx=dimx))
    monkeypatch.setenv('EXOTEDRF_CHUNK_COLS', '7')
    sized_shapes = []
    choose_chunk = stages._badpix_column_chunk_size

    def segment_chunk(array, *args, **kwargs):
        sized_shapes.append(array.shape)
        return choose_chunk(array, *args, **kwargs)

    monkeypatch.setattr(stages, '_badpix_column_chunk_size', segment_chunk)
    out = stages.step_badpix(
        PipelineState(cube),
        {'space_outlier_threshold': 1e6, 'time_outlier_threshold': 1e6,
         'box_size': 2, 'window_size': 3},
        {'opts': {'clear_interpolated_dq': True}})

    first_map = out.aux['hot_pixel_map']
    assert first_map[2, 10]
    assert not first_map[2, 17]
    # Legacy clear_interpolated_dq=True; v1 2.5.0's default keeps the DQ.
    assert np.all(out.cube.dq[12:, 2, 10] == 0)
    default = stages.step_badpix(
        PipelineState(cube),
        {'space_outlier_threshold': 1e6, 'time_outlier_threshold': 1e6,
         'box_size': 2, 'window_size': 3}, {})
    assert np.all(default.cube.dq[12:, 2, 10] == stages.core.DQ_HOT)
    assert 'high_variance_map' in default.aux
    assert out.cube.dq[22, 2, 17] & stages.core.DQ_HOT
    assert isinstance(out.cube.data, np.ndarray)
    assert sized_shapes[:2] == [(12, dimy, dimx), (12, dimy, dimx)]


def test_badpix_specific_chunk_cap_is_gpu_bounded_and_overridable(
        monkeypatch):
    """Check bad-pixel correction specific chunk cap is GPU bounded and overridable."""
    class ShapeOnly:
        shape = (105, 256, 2048)
        dtype = np.dtype(np.float32)

    monkeypatch.delenv('EXOTEDRF_CHUNK_COLS', raising=False)
    monkeypatch.delenv('EXOTEDRF_V2_BADPIX_COLUMN_CHUNK', raising=False)
    monkeypatch.setattr(stages.jax, 'default_backend', lambda: 'gpu')
    monkeypatch.setattr(stages.core, 'device_memory_bytes', lambda: 40 << 30)
    automatic = stages._badpix_column_chunk_size(
        ShapeOnly(), 8, halo=9)
    assert 1 <= automatic <= 256
    assert automatic < ShapeOnly.shape[-1]

    monkeypatch.setenv('EXOTEDRF_CHUNK_COLS', '32')
    assert stages._badpix_column_chunk_size(
        ShapeOnly(), 8, halo=9) == 32
    monkeypatch.setenv('EXOTEDRF_V2_BADPIX_COLUMN_CHUNK', '123')
    assert stages._badpix_column_chunk_size(
        ShapeOnly(), 8, halo=9) == 123


def test_prepared_badpix_scorer_matches_literal_greedy_coordinates(
        monkeypatch):
    """Check prepared bad-pixel correction scorer matches literal greedy coordinates."""
    rng = np.random.default_rng(0x1352770)
    nints, dimy, dimx = 13, 12, 24
    data = rng.normal(100., .25, (nints, dimy, dimx)).astype(np.float32)
    data[5, 4, 11] += 20.
    data[8, 5, 16] -= 15.
    err = np.ones_like(data)
    err[2, 3, 9] = np.nan
    dq = np.zeros_like(data, np.uint32)
    dq[10, 3, 8] = core.DQ_HOT
    dq[7, 6, 14] = core.DQ_SATURATED
    cube = RateCube(data, err, dq, meta(nints, dimx=dimx))
    state = PipelineState(cube)
    ctx = {
        'opts': {'extract_width_soss2': None, 'wave_range': None,
                 'w1': 1., 'w2': 1.},
        'centroids': {
            'ypos o1': np.linspace(4.5, 5.5, dimx),
            'ypos o2': np.concatenate([
                np.linspace(8., 9., 17), np.full(dimx - 17, np.nan)]),
        },
        'waves': {1: np.linspace(.9, 2., dimx),
                  2: np.linspace(.6, 1., dimx)},
    }
    params = {
        'space_outlier_threshold': 8.,
        'time_outlier_threshold': 8.,
        'box_size': 2,
        'window_size': 3,
        'extract_width': 4.,
    }
    # Exercise multiple resident chunks and their artificial boundaries.
    monkeypatch.setenv('EXOTEDRF_V2_BADPIX_COLUMN_CHUNK', '6')
    scorer = stages.prepare_badpix_scorer(state, params, ctx)
    assert scorer.final_consumer_only

    coordinates = (
        ('space_outlier_threshold', [6., 8., 10.]),
        ('time_outlier_threshold', [6., 8., 10.]),
        ('box_size', [2, 3]),
        ('window_size', [3, 5]),
    )
    for parameter, candidates in coordinates:
        fast = scorer.evaluate_candidates(parameter, candidates, params)
        reference_costs = []
        for value, (fast_cost, fast_scatter, _duration) in zip(
                candidates, fast):
            trial = dict(params, **{parameter: value})
            full = stages.step_badpix(state, trial, ctx)
            cost, scatter = stages.evaluate_optimizer_cost(full, trial, ctx)
            reference_costs.append(float(np.asarray(cost)))
            np.testing.assert_allclose(
                fast_cost, cost, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(
                fast_scatter, scatter, rtol=1e-5, atol=1e-7,
                equal_nan=True)
        finite = np.flatnonzero(np.isfinite(reference_costs))
        winner = finite[np.argmin(np.asarray(reference_costs)[finite])]
        params[parameter] = candidates[int(winner)]


def test_supplied_timeseries_uses_fits_intstart_per_segment():
    """Check supplied timeseries uses FITS intstart per segment."""
    data = np.zeros((5, 3, 4, 4), np.float32)
    cube = RampCube(data, np.zeros_like(data, np.uint8),
                    np.zeros((4, 4), np.uint32),
                    meta(5, edges=[3, 5], starts=[10, 50]))
    full_exposure_series = np.arange(60, dtype=np.float32)
    slices = stages._segment_slices(cube)
    first = stages._series_segment(full_exposure_series, cube, slices[0], 10)
    second = stages._series_segment(full_exposure_series, cube, slices[1], 50)
    np.testing.assert_array_equal(first, [9, 10, 11])
    np.testing.assert_array_equal(second, [49, 50])


def test_hot_pixel_map_loader_accepts_array_or_relative_npy(tmp_path):
    """Check hot pixel map loader accepts array or relative npy."""
    hot = np.zeros((3, 4), dtype=bool)
    hot[1, 2] = True
    np.save(tmp_path / 'hot.npy', hot)

    np.testing.assert_array_equal(
        stages._load_hot_pixel_map(hot.astype(np.uint8), hot.shape), hot)
    np.testing.assert_array_equal(
        stages._load_hot_pixel_map('hot.npy', hot.shape, (tmp_path,)), hot)
    assert stages._load_hot_pixel_map(None, hot.shape) is None
    with pytest.raises(ValueError, match='expected'):
        stages._load_hot_pixel_map(np.zeros((2, 4)), hot.shape)


def test_group_and_integration_oof_get_phase_local_processed_traces(
        monkeypatch):
    """Check group and integration 1/f get phase local processed traces."""
    traced = []

    def fake_trace(deepstack, cube, ctx):
        traced.append(np.asarray(deepstack).copy())
        dimx = cube.data.shape[-1]
        return {'ypos o1': np.full(dimx, len(traced), dtype=float)}

    backgrounds = iter((1., 2., 3.))
    monkeypatch.setattr(
        stages, '_scaled_background_model',
        lambda sample, deep, model, region: np.full_like(
            model, next(backgrounds)))
    monkeypatch.setattr(stages, '_trace_soss_deepstack', fake_trace)
    auto_ctx = {
        'centroids': None,
        'centroids_source': 'automatic-v1-lifecycle',
        'background_model': np.ones((4, 4), dtype=np.float32),
    }

    ramp_data = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(
        3, 2, 4, 4)
    ramp = RampCube(
        ramp_data, np.zeros_like(ramp_data, np.uint8),
        np.zeros((4, 4), np.uint32), meta(3, ngroups=2))
    group_state = stages.step_background_grp(
        PipelineState(ramp), {}, auto_ctx)
    expected_ramp = ramp_data.copy()
    expected_ramp[:, 0] -= 1.
    expected_ramp[:, 1] -= 2.
    np.testing.assert_array_equal(group_state.cube.data, expected_ramp)
    np.testing.assert_array_equal(group_state.cube.groupdq, ramp.groupdq)
    np.testing.assert_allclose(traced[0], np.median(expected_ramp, axis=0))
    assert group_state.aux['centroids_group']['ypos o1'][0] == 1.

    rate_data = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    rate = RateCube(
        rate_data, np.ones_like(rate_data),
        np.zeros_like(rate_data, np.uint32), meta(3))
    int_state = stages.step_background_int(
        PipelineState(rate), {}, auto_ctx)
    expected_rate = rate_data - 3.
    np.testing.assert_allclose(traced[1], np.median(expected_rate, axis=0))
    assert int_state.aux['centroids_int']['ypos o1'][0] == 2.

    def forbidden(*args, **kwargs):
        raise AssertionError('a phase-local trace must be reused by OOF')

    monkeypatch.setattr(stages, '_trace_soss_deepstack', forbidden)
    assert stages._centroids_for_stage(
        group_state, auto_ctx, 'centroids_group') is \
        group_state.aux['centroids_group']
    assert stages._centroids_for_stage(
        int_state, auto_ctx, 'centroids_int') is \
        int_state.aux['centroids_int']


@pytest.mark.parametrize('deepframe_key', ['stage3_deepframe', 'deepframe'])
def test_final_extraction_retraces_stage3_deepframe_not_oof_centroids(
        monkeypatch, deepframe_key):
    """Check final extraction retraces stage3 deepframe not 1/f centroids."""
    data = np.ones((3, 5, 4), np.float32)
    cube = RateCube(
        data, np.ones_like(data), np.zeros_like(data, np.uint32), meta(3))
    deepframe = np.full((5, 4), 17., np.float32)
    int_centroids = {'ypos o1': np.full(4, 1.)}
    final_centroids = {'ypos o1': np.full(4, 3.)}
    state = PipelineState(
        cube, {'centroids_int': int_centroids, deepframe_key: deepframe})
    traced = []

    def fake_trace(deepstack, traced_cube, ctx):
        traced.append((np.asarray(deepstack), traced_cube))
        return final_centroids

    monkeypatch.setattr(stages, '_trace_soss_deepstack', fake_trace)
    ctx = {'centroids': None,
           'centroids_source': 'automatic-v1-lifecycle'}

    resolved = stages._centroids_for_extraction(state, ctx, mode='sum')
    assert resolved is final_centroids
    assert len(traced) == 1
    np.testing.assert_array_equal(traced[0][0], deepframe)
    assert traced[0][1] is cube

    assert stages._centroids_for_extraction(
        state, ctx, mode='nanaware') is int_centroids
    assert len(traced) == 1


def test_explicit_centroids_remain_fixed_across_every_lifecycle(monkeypatch):
    """Check explicit centroids remain fixed across every lifecycle."""
    fixed = {'ypos o1': np.full(4, 2.)}
    data = np.ones((3, 5, 4), np.float32)
    cube = RateCube(
        data, np.ones_like(data), np.zeros_like(data, np.uint32), meta(3))
    state = PipelineState(
        cube, {'centroids_int': {'ypos o1': np.full(4, 1.)},
               'stage3_deepframe': np.ones((5, 4))})

    def forbidden(*args, **kwargs):
        raise AssertionError('explicit YAML centroids must never be retraced')

    monkeypatch.setattr(stages, '_trace_soss_deepstack', forbidden)
    ctx = {'centroids': fixed, 'centroids_source': 'explicit'}
    assert stages._centroids_for_stage(
        state, ctx, 'centroids_int') is fixed
    assert stages._centroids_for_extraction(state, ctx, 'sum') is fixed


def test_pca_step_unpacks_all_kernel_outputs():
    """Check PCA step unpacks all kernel outputs."""
    rng = np.random.default_rng(1)
    data = rng.normal(size=(6, 3, 4)).astype(np.float32)
    cube = RateCube(data, np.ones_like(data), np.zeros_like(data, np.uint32),
                    meta(6, ngroups=3))
    out = stages.step_pca(
        PipelineState(cube), {},
        {'opts': {'pca_components': 2, 'remove_components': None}})
    assert out.cube.data.shape == data.shape
    assert out.aux['pca_components'].shape == (2, 6)
    assert out.aux['pca_wlc'].shape == (6,)
    assert out.aux['pca_components_reconstructed'].shape == (2, 6)
    assert 'pca_projections' not in out.aux
    assert 'pca_projections_reconstructed' not in out.aux
    np.testing.assert_allclose(
        out.aux['stage3_deepframe'], np.median(data, axis=0))

    plotted = stages.step_pca(
        PipelineState(cube), {},
        {'opts': {'pca_components': 2, 'remove_components': None,
                  'do_plots': True}})
    assert plotted.aux['pca_projections'].shape == (2, 3, 4)
    assert 'pca_projections_reconstructed' not in plotted.aux


def test_final_extract_uses_sum_selective_dq_and_separate_o2_width():
    """Check final extract uses sum selective DQ and separate o2 width."""
    data = np.zeros((11, 10, 4), np.float32)
    data[:, 2] = 10
    data[:, 3] = 2
    data[:, 5:9] = 1
    err = np.ones_like(data)
    dq = np.zeros_like(data, np.uint32)
    dq[:, 2, 0] = 1
    dq[:, 2, 1] = 2
    dq[:, 2, 2] = 4
    cube = RateCube(data, err, dq, meta(11))
    ctx = {
        'opts': {'extract_width_soss2': 4, 'mask_do_not_use_pixels': True,
                 'mask_saturated_pixels': True, 'saturation_rescue': False},
        'centroids': {'ypos o1': np.full(4, 3.),
                      'ypos o2': np.full(4, 7.)},
        'waves': {1: np.arange(4.), 2: np.arange(4.) + 10},
    }
    params = {'extract_width': 2}
    out = stages.step_extract(PipelineState(cube), params, ctx)
    flux1, ferr1 = out.aux['spectra'][1]
    flux2, ferr2 = out.aux['spectra'][2]
    np.testing.assert_allclose(flux1[0], [2, 2, 12, 12])
    np.testing.assert_allclose(ferr1[0], [1, 1, np.sqrt(2), np.sqrt(2)])
    np.testing.assert_allclose(flux2, 4.)
    np.testing.assert_allclose(ferr2, 2.)

    # Optimizer semantics remain a NaN-aware mean and reject every DQ bit.
    opt, _ = stages.extract_orders(PipelineState(cube), params, ctx)[1]
    assert opt[0, 2] == pytest.approx(1.)


def test_phase1_ramp_extract_uses_last_groupdq_and_ignores_pixeldq():
    """Check phase1 ramp extract uses last GROUPDQ and ignores PIXELDQ."""
    data = np.broadcast_to(
        np.arange(1., 7., dtype=np.float32)[None, None, :, None],
        (1, 2, 6, 1)).copy()
    groupdq = np.zeros_like(data, np.uint8)
    groupdq[:, 0] = np.uint8(core.DQ_DO_NOT_USE)
    pixeldq = np.full((6, 1), core.DQ_DO_NOT_USE, np.uint32)
    cube = RampCube(data, groupdq, pixeldq, meta(1, ngroups=2, dimx=1))
    ctx = {'opts': {}, 'centroids': {'ypos o1': np.asarray([3.])}}

    flux, _ = stages.extract_orders(
        PipelineState(cube), {'extract_width': 2}, ctx)[1]
    assert np.isfinite(np.asarray(flux)[0, 0])

    last_group_bad = groupdq.copy()
    last_group_bad[:, -1] = np.uint8(core.DQ_DO_NOT_USE)
    flagged = RampCube(
        data, last_group_bad, np.zeros_like(pixeldq), cube.meta)
    flagged_flux, _ = stages.extract_orders(
        PipelineState(flagged), {'extract_width': 2}, ctx)[1]
    assert np.isnan(np.asarray(flagged_flux)[0, 0])


def test_prepared_group_oof_scorer_matches_full_4d_candidate_costs(
        monkeypatch):
    """Check prepared group 1/f scorer matches full 4d candidate costs."""
    rng = np.random.default_rng(0x135276f)
    nints, ngroups, dimy, dimx = 9, 3, 12, 16
    data = rng.normal(100., .3, (nints, ngroups, dimy, dimx)).astype(
        np.float32)
    # Give every integration/group a different even/odd column offset.
    offsets = rng.normal(0., .5, (nints, ngroups, 2, dimx)).astype(np.float32)
    data[:, :, 0::2] += offsets[:, :, 0, None]
    data[:, :, 1::2] += offsets[:, :, 1, None]
    groupdq = np.zeros_like(data, np.uint8)
    groupdq[3, -1, 5, 7] = np.uint8(core.DQ_DO_NOT_USE)
    pixeldq = np.zeros((dimy, dimx), np.uint32)
    pixeldq[2, 3] = core.DQ_HOT
    cube = RampCube(
        data, groupdq, pixeldq,
        meta(nints, ngroups=ngroups, dimx=dimx))
    background = rng.normal(.2, .01, (ngroups, dimy, dimx)).astype(np.float32)
    state = PipelineState(cube, {'bkg_grp': background})
    centroids = {'ypos o1': np.linspace(5.5, 6.5, dimx)}
    ctx = {
        'opts': {'oof_method': 'scale-achromatic',
                 'extract_width_soss2': None,
                 'wave_range': None, 'w1': 1., 'w2': 1.},
        'centroids': centroids,
        'soss_timeseries': np.linspace(.99, 1.01, nints).astype(np.float32),
        'soss_timeseries_o2': None,
        'outlier_mask': None,
        'order0_mask': None,
        'waves': {1: np.linspace(.9, 2., dimx)},
    }
    base_params = {
        'soss_inner_mask_width': 5,
        'soss_outer_mask_width': 10,
        'extract_width': 4,
    }
    # Exercise the same streamed reference path used on smaller devices.
    monkeypatch.setenv('EXOTEDRF_DEVICE_FAST_PATH', 'false')
    scorer = stages.prepare_oneoverf_grp_scorer(state, base_params, ctx)
    widths = [3., 5., 7.]
    fast = scorer.evaluate_candidates(
        'soss_inner_mask_width', widths, base_params)

    streamed_states = {}
    for width, (fast_cost, fast_scatter, _) in zip(widths, fast):
        params = dict(base_params, soss_inner_mask_width=width)
        full = stages.step_oneoverf_grp(state, params, ctx)
        streamed_states[width] = full
        cost, scatter = stages.evaluate_optimizer_cost(full, params, ctx)
        np.testing.assert_allclose(fast_cost, cost, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(
            fast_scatter, scatter, rtol=1e-5, atol=1e-7, equal_nan=True)

    outer_params = dict(base_params, soss_inner_mask_width=5.)
    outer = scorer.evaluate_candidates(
        'soss_outer_mask_width', [8., 10., 12.], outer_params)
    assert len(outer) == 3
    assert [row[0] for row in outer] == pytest.approx([fast[1][0]] * 3)
    assert outer[0][2] == 0.
    assert all(row[2] == 0. for row in outer[1:])

    monkeypatch.setenv('EXOTEDRF_DEVICE_FAST_PATH', 'true')
    resident = stages.step_oneoverf_grp(
        state, dict(base_params, soss_inner_mask_width=5.), ctx)
    np.testing.assert_allclose(
        resident.cube.data, streamed_states[5.].cube.data,
        rtol=1e-6, atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(
        resident.cube.groupdq, streamed_states[5.].cube.groupdq)
    np.testing.assert_array_equal(
        resident.cube.pixeldq, streamed_states[5.].cube.pixeldq)


def test_production_cost_requires_extract_products():
    """Check production cost requires extract products."""
    data = np.ones((8, 3, 1), np.float32)
    cube = RateCube(data, np.ones_like(data),
                    np.zeros_like(data, np.uint32), meta(8, dimx=1))
    with pytest.raises(ValueError, match='requires Extract spectral_products'):
        stages.evaluate_production_cost(
            PipelineState(cube), {'extract_width': 2},
            {'opts': {'w1': 1., 'w2': 0.}})


def test_final_lightcurve_sigma_clip_matches_v1_defaults():
    """Check final lightcurve sigma clip matches v1 defaults."""
    first = np.asarray(
        [1.00, 1.01, .99, 1.02, .98, 10., 1.01, .99, 1.02, .98, 1.00])
    flux = np.column_stack([first, np.arange(first.size, dtype=float)])
    expected = flux.copy()
    expected[5, 0] = 1.01

    clipped = stages.sigma_clip_lightcurves(flux)
    np.testing.assert_allclose(clipped, expected)


def test_order2_internal_nan_is_compacted_in_both_extraction_paths():
    """Check order2 internal NaN is compacted in both extraction paths."""
    data = np.zeros((1, 10, 4), np.float32)
    data[:, 6:8, 0] = 4
    data[:, 1:3, 1] = 5
    centroids = {'ypos o1': np.full(4, 4.),
                 'ypos o2': np.asarray([7., np.nan, 2., np.nan])}
    ctx = {'opts': {'extract_width_soss2': 1.5},
           'centroids': centroids}
    params = {'extract_width': 1.5}

    rate = RateCube(data, np.ones_like(data),
                    np.zeros_like(data, np.uint32), meta(1))
    rate_flux, _ = stages.extract_orders(
        PipelineState(rate), params, ctx, mode='sum')[2]
    assert rate_flux[0, 1] > 0
    np.testing.assert_array_equal(rate_flux[0, 2:], 0)

    ramp_data = np.repeat(data[:, None], 2, axis=1)
    ramp = RampCube(ramp_data, np.zeros_like(ramp_data, np.uint8),
                    np.zeros((10, 4), np.uint32), meta(1, ngroups=2))
    ramp_flux, _ = stages.extract_orders(PipelineState(ramp), params, ctx)[2]
    assert ramp_flux[0, 1] > 0
    np.testing.assert_array_equal(ramp_flux[0, 2:], 0)


def test_optional_trace_nans_do_not_create_row_zero_oof_apertures():
    """Check optional trace NaNs do not create row zero 1/f apertures."""
    centroids = stages._padded_optional_trace([7., np.nan, 2.], dimx=4)
    mask = np.asarray(stages.k_oof.make_trace_mask(centroids, 2., dimy=10))
    assert mask[:, 0].any()
    assert not mask[:, 1].any()
    assert mask[:, 2].any()
    assert not mask[:, 3].any()


def test_final_spectral_products_trim_and_sort_wavelengths():
    """Check final spectral products trim and sort wavelengths."""
    data = np.zeros((11, 8, 4), np.float32)
    for column in range(4):
        data[:, 2:4, column] = column + 1
        data[:, 5:7, column] = 10 * (column + 1)
    cube = RateCube(data, np.ones_like(data),
                    np.zeros_like(data, np.uint32), meta(11))
    ctx = {
        'opts': {'extract_width_soss2': 2},
        'centroids': {'ypos o1': np.full(4, 3.),
                      'ypos o2': np.full(4, 6.)},
        'waves': {1: np.asarray([3., np.nan, 1., 2.]),
                  2: np.asarray([30., 10., np.nan, 20.])},
    }
    out = stages.step_extract(
        PipelineState(cube), {'extract_width': 2}, ctx)
    o1 = out.aux['spectral_products'][1]
    np.testing.assert_array_equal(o1['wave'], [1., 2., 3.])
    np.testing.assert_array_equal(o1['flux'][0], [6., 8., 2.])
    np.testing.assert_allclose(o1['ferr'][0], np.sqrt(2.))
    o2 = out.aux['spectral_products'][2]
    np.testing.assert_array_equal(o2['wave'], [10., 20., 30.])
    np.testing.assert_array_equal(o2['flux'][0], [40., 80., 20.])
    np.testing.assert_allclose(o2['ferr'][0], np.sqrt(2.))


def test_stage_wrapper_preserves_float64_science_dtype():
    """Check stage wrapper preserves float64 science dtype."""
    old = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        data = np.ones((2, 2, 3, 4), np.float64)
        cube = RampCube(data, np.zeros_like(data, np.uint8),
                        np.zeros((3, 4), np.uint32), meta(2, ngroups=2))
        out = stages.step_superbias(
            PipelineState(cube), {},
            {'opts': {'superbias_method': 'crds'},
             'refpack': {'superbias': np.full((3, 4), .25, np.float32)}})
        assert np.asarray(out.cube.data).dtype == np.float64
        np.testing.assert_allclose(out.cube.data, .75)
    finally:
        jax.config.update('jax_enable_x64', old)


def test_superbias_and_dark_steps_or_reference_dq_and_skip_nan_values():
    """Check superbias and dark steps or reference DQ and skip NaN values."""
    data = np.full((2, 2, 2, 3), 10., np.float32)
    pixeldq = np.zeros((2, 3), np.uint32)
    pixeldq[0, 0] = 2
    cube = RampCube(data, np.zeros_like(data, np.uint8), pixeldq,
                    meta(2, ngroups=2))

    superbias = np.ones((2, 3), np.float32)
    superbias[0, 2] = np.nan
    superbias_dq = np.zeros((2, 3), np.uint32)
    superbias_dq[0, 1] = 2048
    biased = stages.step_superbias(
        PipelineState(cube), {},
        {'opts': {'superbias_method': 'crds'},
         'refpack': {'superbias': superbias,
                     'superbias_dq': superbias_dq}})
    np.testing.assert_allclose(biased.cube.data[..., 0, 0], 9.)
    np.testing.assert_allclose(biased.cube.data[..., 0, 2], 10.)
    assert biased.cube.pixeldq[0, 0] == 2
    assert biased.cube.pixeldq[0, 1] == 2048

    dark = np.full((2, 2, 3), .5, np.float32)
    dark[1, 1, 2] = np.nan
    dark_dq = np.zeros((2, 3), np.uint32)
    dark_dq[1, 0] = 4096
    darked = stages.step_dark(
        biased, {}, {'refpack': {'dark': dark, 'dark_dq': dark_dq}})
    np.testing.assert_allclose(darked.cube.data[:, 0, 0, 0], 8.5)
    np.testing.assert_allclose(darked.cube.data[:, 1, 1, 2], 9.)
    assert darked.cube.pixeldq[0, 1] == 2048
    assert darked.cube.pixeldq[1, 0] == 4096


def test_linearity_step_propagates_all_dq_but_skips_only_no_lin_corr():
    """Check linearity step propagates all DQ but skips only no lin corr."""
    no_lin = np.uint32(1 << 20)
    data = np.full((1, 2, 2, 3), 3., np.float32)
    cube = RampCube(data, np.zeros_like(data, np.uint8),
                    np.zeros((2, 3), np.uint32), meta(1, ngroups=2))
    coeffs = np.zeros((3, 2, 3), np.float32)
    coeffs[1] = 2.
    coeffs[2, 0, 2] = np.nan
    coeffs[1, 1, 0] = 0.
    lin_dq = np.zeros((2, 3), np.uint32)
    lin_dq[0, 0] = 2048
    lin_dq[0, 1] = no_lin

    out = stages.step_linearity(
        PipelineState(cube), {},
        {'refpack': {'lin_coeffs': coeffs, 'lin_dq': lin_dq}})
    result = np.asarray(out.cube.data)
    np.testing.assert_allclose(result[..., 0, 0], 6.)
    np.testing.assert_allclose(result[..., 0, 1], 3.)
    np.testing.assert_allclose(result[..., 0, 2], 3.)
    np.testing.assert_allclose(result[..., 1, 0], 3.)
    assert out.cube.pixeldq[0, 0] == 2048
    assert out.cube.pixeldq[0, 1] == no_lin
    assert out.cube.pixeldq[0, 2] & no_lin
    assert out.cube.pixeldq[1, 0] & no_lin


@pytest.mark.parametrize('read_times', [None, np.array([]), np.array([1., 2.])])
def test_linearity_step_selects_read_level_correction(read_times):
    """Check linearity step selects read level correction."""
    from stcal.linearity.linearity import linearity_correction

    data = np.arange(60, dtype=np.float32).reshape(2, 5, 2, 3) * 30.
    groupdq = np.zeros_like(data, np.uint8)
    groupdq[0, 3:, 0, 1] = 2
    pixeldq = np.zeros((2, 3), np.uint32)
    metadata = meta(2, ngroups=5)
    metadata.extra.update(header={'NFRAMES': 2}, read_times=read_times)
    cube = RampCube(data, groupdq, pixeldq, metadata)
    coeffs = np.zeros((3, 2, 3), np.float32)
    coeffs[1] = 1.
    coeffs[2] = 1e-5
    inverse = coeffs.copy()
    inverse[2] *= -1
    lin_dq = np.zeros((2, 3), np.uint32)
    use_inverse = read_times is None or len(read_times) == 0
    expected, expected_dq, _ = linearity_correction(
        data.copy(), groupdq.copy(), pixeldq.copy(), coeffs.copy(),
        lin_dq.copy(), {'SATURATED': 2, 'NO_LIN_CORR': 1 << 20},
        ilin_coeffs=inverse.copy() if use_inverse else None,
        read_pattern=[[2 * g + 1, 2 * g + 2] for g in range(5)])
    actual = stages.step_linearity(PipelineState(cube), {}, {
        'refpack': {'lin_coeffs': coeffs, 'lin_inv_coeffs': inverse,
                    'lin_dq': lin_dq}}).cube
    np.testing.assert_allclose(actual.data, expected, rtol=2e-6, atol=2e-4)
    np.testing.assert_array_equal(actual.pixeldq, expected_dq)


def test_flat_step_propagates_reference_dq_and_flat_uncertainty():
    """Check flat step propagates reference DQ and flat uncertainty."""
    from scipy.interpolate import griddata
    no_flat = np.uint32(1 << 18)
    data = np.arange(12, dtype=np.float32).reshape(2, 2, 3) + 10.
    err = np.full_like(data, 2.)
    dq = np.zeros_like(data, np.uint32)
    dq[0, 0, 0] = 2
    cube = RateCube(data, err, dq, meta(2))
    flat = np.asarray([[2., 0., np.nan], [4., 5., 2.]], np.float32)
    flat_err = np.full(flat.shape, .1, np.float32)
    flat_dq = np.zeros(flat.shape, np.uint32)
    flat_dq[1, 1] = 1
    flat_dq[1, 2] = no_flat

    out = stages.step_flat(
        PipelineState(cube), {},
        {'refpack': {'flat': flat, 'flat_err': flat_err,
                     'flat_dq': flat_dq}})

    ref_dq = flat_dq.copy()
    invalid = np.isnan(flat) | (flat == 0)
    ref_dq[invalid] |= np.uint32(1) | no_flat
    ref_dq[(ref_dq & no_flat) != 0] |= np.uint32(1)
    bad = (ref_dq & 1) != 0
    safe = np.where(bad, 1., flat)
    expected_data = data / safe
    expected_err = np.sqrt(
        (err / safe) ** 2
        + expected_data ** 2 / safe ** 2 * flat_err ** 2)
    # Match JWST's NaN/DQ handling and v1's cosmetic SCI interpolation.
    expected_dq = dq | ref_dq
    invalid = (expected_dq & 1) != 0
    expected_data[invalid] = np.nan
    expected_err[invalid] = np.nan
    yy, xx = np.indices(flat.shape)
    for plane in expected_data:
        good = np.where(np.isfinite(plane))
        plane[:] = griddata(good, plane[good], (yy, xx), method='nearest')
    np.testing.assert_allclose(out.cube.data, expected_data, rtol=1e-6)
    np.testing.assert_allclose(out.cube.err, expected_err, rtol=1e-6)
    np.testing.assert_array_equal(out.cube.dq, expected_dq)

    # Older synthetic packs may omit optional ERR/DQ companions.
    neutral = stages.step_flat(
        PipelineState(cube), {}, {'refpack': {'flat': np.full((2, 3), 2.)}})
    np.testing.assert_allclose(neutral.cube.data, data / 2.)
    np.testing.assert_allclose(neutral.cube.err, err / 2.)
    np.testing.assert_array_equal(neutral.cube.dq, dq)


def test_float64_auxiliary_arrays_follow_science_dtype():
    """Check float64 auxiliary arrays follow science dtype."""
    old = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        series = stages._validate_timeseries(
            [1, 2], 'series', 2, 8, dtype=np.float64)
        trace = stages._padded_optional_trace(
            [2., np.nan], dimx=8, dtype=np.float64)
        assert series.dtype == np.float64
        assert np.asarray(trace).dtype == np.float64

        data = np.full((2, 6, 8), 10., np.float64)
        cube = RateCube(data, np.ones_like(data),
                        np.zeros_like(data, np.uint32), meta(2, dimx=8))
        out = stages.step_background_int(
            PipelineState(cube), {},
            {'background_model': np.full(
                (6, 8), 0.123456789012, np.float64)})
        assert out.cube.data.dtype == np.float64
        assert out.aux['bkg_int'].dtype == np.float64
        assert out.aux[stages._OOF_INT_DEEP_KEY].dtype == np.float64

        extracted = stages.extract_orders(
            out, {'extract_width': np.float64(2.0000000001)},
            {'opts': {},
             'centroids': {'ypos o1': np.full(8, 3., np.float64)}})
        assert extracted[1][0].dtype == np.float64
        assert extracted[1][1].dtype == np.float64
    finally:
        jax.config.update('jax_enable_x64', old)


def test_integration_oof_reuses_background_output_deepstack(monkeypatch):
    """Check integration 1/f reuses background output deepstack."""
    rng = np.random.default_rng(0x1352771)
    nints, dimy, dimx = 13, 10, 12
    data = rng.normal(100., .1, (nints, dimy, dimx)).astype(np.float32)
    cube = RateCube(data, np.ones_like(data),
                    np.zeros_like(data, np.uint32),
                    meta(nints, dimx=dimx))
    deep = np.nanmedian(data, axis=0)
    state = PipelineState(cube, {stages._OOF_INT_DEEP_KEY: deep})
    ctx = {
        'opts': {'oof_method': 'scale-achromatic'},
        'centroids': {'ypos o1': np.full(dimx, 5., np.float32)},
        'soss_timeseries': np.ones(nints, np.float32),
    }

    def forbidden(*args, **kwargs):
        raise AssertionError('BackgroundStep deepstack was recomputed')

    monkeypatch.setattr(stages, '_host_deepstack', forbidden)
    out = stages.step_oneoverf_int(
        state, {'soss_inner_mask_width': 4.,
                'soss_outer_mask_width': 8.}, ctx)

    assert out.cube.data.shape == data.shape
    assert stages._OOF_INT_DEEP_KEY not in out.aux


def test_rampfit_x64_preserves_group_time_precision():
    """Check rampfit x64 preserves group time precision."""
    old = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        group_time = 1.00000006
        slope = 2.5
        groups = np.arange(1, 4, dtype=np.float64)
        ramp_1d = 59000. + slope * groups * group_time
        data = np.broadcast_to(
            ramp_1d[None, :, None, None], (2, 3, 2, 2)).copy()
        obs_meta = meta(2, ngroups=3)
        obs_meta.frame_time = group_time
        cube = RampCube(
            data, np.zeros_like(data, np.uint8),
            np.zeros((2, 2), np.uint32), obs_meta)
        out = stages.step_rampfit(
            PipelineState(cube), {},
            {'refpack': {'readnoise': np.ones((2, 2), np.float32),
                         'gain': np.ones((2, 2), np.float32)}})
        assert np.asarray(out.cube.data).dtype == np.float64
        np.testing.assert_allclose(out.cube.data, slope, rtol=1e-11,
                                   atol=1e-11)
    finally:
        jax.config.update('jax_enable_x64', old)


def test_rampfit_collapses_group_saturation_like_v1_wrapper():
    """v1 restores any/early saturation bits after JWST RampFit."""
    group_time = 2.
    groups = np.arange(1, 5, dtype=np.float32)
    ramp_1d = 20. + 3. * groups * group_time
    data = np.broadcast_to(
        ramp_1d[None, :, None, None], (1, 4, 2, 3)).copy()
    groupdq = np.zeros_like(data, np.uint8)
    groupdq[0, 3, 0, 0] = np.uint8(core.DQ_SATURATED)
    groupdq[0, 1:, 0, 1] = np.uint8(core.DQ_SATURATED)
    obs_meta = meta(1, ngroups=4)
    obs_meta.frame_time = group_time
    cube = RampCube(data, groupdq, np.zeros((2, 3), np.uint32), obs_meta)

    out = stages.step_rampfit(
        PipelineState(cube), {},
        {'refpack': {'readnoise': np.ones((2, 3), np.float32),
                     'gain': np.ones((2, 3), np.float32)}})
    dq = np.asarray(out.cube.dq)
    assert dq[0, 0, 0] == core.DQ_SATURATED
    assert dq[0, 0, 1] == (core.DQ_SATURATED | core.DQ_DO_NOT_USE)
    assert dq[0, 0, 2] == 0


def test_rampfit_nearest_fills_invalid_rate_but_preserves_nan_error():
    """Check rampfit nearest fills invalid rate but preserves NaN error."""
    group_time = 2.
    groups = np.arange(1, 5, dtype=np.float32)
    data = np.broadcast_to(
        (20. + 3. * groups * group_time)[None, :, None, None],
        (1, 4, 3, 3)).copy()
    groupdq = np.zeros_like(data, np.uint8)
    # The center has one good group and is suppressed by pinned RampFit.
    groupdq[0, 1:, 1, 1] = np.uint8(core.DQ_SATURATED)
    obs_meta = meta(1, ngroups=4)
    obs_meta.frame_time = group_time
    cube = RampCube(data, groupdq, np.zeros((3, 3), np.uint32), obs_meta)

    out = stages.step_rampfit(
        PipelineState(cube), {},
        {'refpack': {'readnoise': np.ones((3, 3), np.float32),
                     'gain': np.ones((3, 3), np.float32)}})
    assert np.asarray(out.cube.data)[0, 1, 1] == pytest.approx(3.)
    assert np.asarray(out.cube.err)[0, 1, 1] == 0.
    assert np.asarray(out.cube.dq)[0, 1, 1] == (
        core.DQ_SATURATED | core.DQ_DO_NOT_USE)


def test_rampfit_all_invalid_plane_is_not_fabricated():
    """Check rampfit all invalid plane is not fabricated."""
    filled = stages._nearest_fill_rate_planes(
        np.full((1, 2, 3), np.nan, np.float32))
    assert np.isnan(filled).all()


def test_rampfit_nearest_fill_reuses_exact_mask_geometry(monkeypatch):
    """Check rampfit nearest fill reuses exact mask geometry."""
    from scipy import interpolate

    rng = np.random.default_rng(0x135276f)
    rate = rng.normal(size=(4, 7, 9)).astype(np.float32)
    shared = np.zeros((7, 9), dtype=bool)
    shared[1, 2] = True
    shared[4, 5] = True
    rate[:3, shared] = np.nan
    rate[3, 2, 6] = np.nan

    expected = rate.copy()
    px, py = np.meshgrid(np.arange(rate.shape[-1]),
                         np.arange(rate.shape[-2]))
    for integration in range(rate.shape[0]):
        finite = np.where(np.isfinite(expected[integration]))
        expected[integration] = interpolate.griddata(
            finite, expected[integration][finite], (py, px),
            method='nearest')

    calls = 0
    original = interpolate.griddata

    def counted_griddata(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(interpolate, 'griddata', counted_griddata)
    actual = stages._nearest_fill_rate_planes(rate)

    np.testing.assert_array_equal(actual, expected)
    # Three integrations share one geometry; the fourth has one distinct invalid mask.
    assert calls == 2


def test_rampfit_pixeldq_invalidates_before_science_nearest_fill():
    """Check rampfit PIXELDQ invalidates before science nearest fill."""
    group_time = 1.
    data = np.broadcast_to(
        np.asarray([100, 110, 120, 130], np.float32)[None, :, None, None],
        (1, 4, 3, 3)).copy()
    pixeldq = np.zeros((3, 3), np.uint32)
    pixeldq[1, 1] = core.DQ_DO_NOT_USE
    obs_meta = meta(1, ngroups=4)
    obs_meta.frame_time = group_time
    cube = RampCube(data, np.zeros_like(data, np.uint8), pixeldq, obs_meta)
    out = stages.step_rampfit(
        PipelineState(cube), {},
        {'refpack': {'readnoise': np.ones((3, 3), np.float32) * 6,
                     'gain': np.ones((3, 3), np.float32) * 1.6}})
    assert np.asarray(out.cube.data)[0, 1, 1] == pytest.approx(10.)
    assert np.asarray(out.cube.err)[0, 1, 1] == 0.
    assert np.asarray(out.cube.dq)[0, 1, 1] == core.DQ_DO_NOT_USE


def test_rampfit_ngroup1_uses_nframes_frame_time_and_rejects_unflagged_nan():
    """Check rampfit ngroup1 uses nframes frame time and rejects unflagged NaN."""
    data = np.full((1, 1, 2, 2), 100., np.float32)
    obs_meta = meta(1, ngroups=1)
    obs_meta.frame_time = 10.
    header = {'TGROUP': 10., 'TFRAME': 2., 'NFRAMES': 4}
    obs_meta.extra = {'header': header, 'segment_headers': (header,)}
    cube = RampCube(data, np.zeros_like(data, np.uint8),
                    np.zeros((2, 2), np.uint32), obs_meta)
    ctx = {'refpack': {
        'readnoise': np.ones((2, 2), np.float32) * 6,
        'gain': np.ones((2, 2), np.float32) * 1.6}}
    out = stages.step_rampfit(PipelineState(cube), {}, ctx)
    np.testing.assert_allclose(out.cube.data, 20., rtol=2e-6)
    np.testing.assert_allclose(out.cube.err, 1.6911534071, rtol=2e-6)

    bad = data.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='non-finite SCI'):
        stages.step_rampfit(
            PipelineState(RampCube(
                bad, np.zeros_like(bad, np.uint8),
                np.zeros((2, 2), np.uint32), obs_meta)), {}, ctx)


def test_context_rejects_solve_and_atoca_boundaries():
    """Check context rejects solve and ATOCA boundaries."""
    ctx = {
        'opts': controls(oof_method='scale-achromatic', extract_method='box'),
        'background_model': None,
        'soss_timeseries': None,
        'soss_timeseries_o2': None,
    }
    # Oof_method='solve' is supported (tests/v2/test_oneoverf_solve_*.py).
    ctx['opts']['extract_method'] = 'atoca'
    stages._validate_soss_context(ctx)
    ctx['opts']['extract_method'] = 'doublegauss'
    with pytest.raises(ValueError, match='atoca'):
        stages._validate_soss_context(ctx)


def test_context_accepts_up_ramp_jump_surface():
    # The JWST up-the-ramp JumpStep is ported (kernels/upramp_jump.py).
    """Check context accepts up ramp jump surface."""
    ctx = {
        'opts': controls(flag_up_ramp=True, oof_method='scale-achromatic',
                         extract_method='box'),
        'background_model': None,
        'soss_timeseries': None,
        'soss_timeseries_o2': None,
    }
    stages._validate_soss_context(ctx)


def test_context_accepts_window_method_without_supplied_timeseries():
    """Check context accepts window method without supplied timeseries."""
    ctx = {
        'opts': controls(oof_method='scale-achromatic-window',
                         extract_method='box'),
        'background_model': None,
        'soss_timeseries': None,
        'soss_timeseries_o2': None,
    }
    assert stages._validate_soss_context(ctx) is ctx
