"""Focused MIRI/LRS graph, wrapper, and extraction integration tests."""

import numpy as np
import pytest

from exotedrf.v2 import core, stages
from exotedrf.v2.core import ObsMeta, RampCube, RateCube
from exotedrf.v2.pipeline import PipelineState


def _meta(nints, ngroups, edges=None, starts=None, headers=None,
          baseline_ints=None):
    """Return meta."""
    edges = np.asarray([nints] if edges is None else edges, dtype=int)
    if starts is None:
        starts = tuple([1] + [int(edge) + 1 for edge in edges[:-1]])
    if headers is None:
        headers = tuple({
            'TGROUP': 2., 'TFRAME': 2., 'NFRAMES': 1, 'GROUPGAP': 0,
            'NINTS': nints, 'READPATT': 'FASTR1', 'NSAMPLES': 1,
            'SUBSTRT1': 1,
        } for _ in edges)
    return ObsMeta(
        mode='MIRI/LRS', detector='MIRIMAGE', subarray='SLITLESSPRISM',
        frame_time=2., ngroups=ngroups,
        int_times=60000. + np.arange(nints) * 1e-4,
        baseline_ints=np.asarray(
            [nints] if baseline_ints is None else baseline_ints),
        segment_edges=edges,
        filenames=tuple(f'seg{i + 1:03d}_uncal.fits'
                        for i in range(len(edges))),
        extra={
            'header': headers[0],
            'segment_headers': tuple(headers),
            'segment_int_starts': tuple(starts),
            'exposure_nints': nints,
        },
    )


def _ramp(data, *, edges=None, starts=None, headers=None, groupdq=None,
          pixeldq=None):
    """Return ramp."""
    nints, ngroups, dimy, dimx = data.shape
    if groupdq is None:
        groupdq = np.zeros(data.shape, np.uint8)
    if pixeldq is None:
        pixeldq = np.zeros((dimy, dimx), np.uint32)
    return RampCube(
        np.asarray(data, np.float32), np.asarray(groupdq, np.uint8),
        np.asarray(pixeldq, np.uint32),
        _meta(nints, ngroups, edges=edges, starts=starts, headers=headers))


def test_miri_graph_has_exact_v1_order():
    """Check MIRI graph has exact v1 order."""
    opts = {name: 'run' for name in stages._MIRI_STEP_CONTROLS}

    assert [step.name for step in stages.miri_steps(opts)] == [
        'DQInitStep', 'EmiCorrStep', 'LinearityStep',
        'JumpStep', 'RampFitStep', 'GainScaleStep', 'AssignWCSStep',
        'FlatFieldStep', 'BackgroundStep', 'BadPixStep',
        'PCAReconstructStep', 'Extract',
    ]

    # V1 2.5.0 removed ResetStep/SourceTypeStep: stale switches are ignored.
    opts['ResetStep'] = 'run'
    assert 'ResetStep' not in [step.name for step in stages.miri_steps(opts)]
    opts['EmiCorrStep'] = 'RUN'
    with pytest.raises(ValueError, match='exactly lowercase'):
        stages.miri_steps(opts)


def test_miri_dqinit_marks_first_and_last_groups_do_not_use():
    """Check MIRI DQ initialization marks first and last groups do not use."""
    data = np.zeros((2, 5, 3, 4), np.float32)
    mask = np.zeros((3, 4), np.uint32)
    mask[1, 2] = core.DQ_DO_NOT_USE
    out = stages.step_dq_init(
        PipelineState(_ramp(data)), {},
        {'opts': {'saturation_threshold': 100},
         'refpack': {'mask_dq': mask}})

    dq = np.asarray(out.cube.groupdq)
    assert ((dq[:, 0] & core.DQ_DO_NOT_USE) != 0).all()
    assert ((dq[:, -1] & core.DQ_DO_NOT_USE) != 0).all()
    assert ((dq[:, 1:-1, 1, 2] & core.DQ_DO_NOT_USE) != 0).all()
    assert not ((dq[:, 1:-1, 0, 0] & core.DQ_DO_NOT_USE) != 0).any()


def test_miri_gpu_optimal_extraction_produces_full_products():
    """Check MIRI GPU optimal extraction produces full products."""
    nints, dimy, dimx = 15, 20, 12
    ypos = np.arange(5, 15, dtype=float)
    xpos = np.full(ypos.shape, 6., dtype=float)
    spatial = np.exp(-0.5 * ((np.arange(dimx) - 6.) / 1.2) ** 2)
    spectrum = 500. + 20. * np.arange(dimy)
    data = np.empty((nints, dimy, dimx), np.float32)
    for integration in range(nints):
        data[integration] = (
            (1. + integration * 1e-3) * spectrum[:, None] * spatial[None])
    deepframe = np.median(data, axis=0)
    cube = RateCube(
        data, np.ones_like(data), np.zeros_like(data, np.uint32),
        _meta(nints, 1))
    wave_map = np.broadcast_to(
        np.arange(dimy, dtype=float)[:, None], (dimy, dimx))
    ctx = {
        'opts': {
            'extract_method': 'optimal', 'opt_max_iter': 1,
            'opt_var_thresh': 25,
        },
        'centroids': {'xpos': xpos, 'ypos': ypos},
        'miri_wave_map': wave_map,
    }
    state = PipelineState(cube, {'stage3_deepframe': deepframe})
    out = stages.step_extract_miri(
        state, {'extract_width': 6.}, ctx)

    product = out.aux['spectral_products'][1]
    assert product['wave'].shape == (dimy,)
    assert product['flux'].shape == (nints, dimy)
    assert product['ferr'].shape == (nints, dimy)
    assert np.all(product['flux'][:, :5] == 0)
    assert np.all(product['flux'][:, 15:] == 0)
    assert np.isfinite(product['flux'][:, 5:15]).all()
    assert np.isfinite(product['ferr'][:, 5:15]).all()
    assert len(out.aux['optimal_clipped_counts']) == 1


def test_miri_emicorr_restarts_for_each_segment_and_uses_its_header(
        monkeypatch):
    """Check MIRI emicorr restarts for each segment and uses its header."""
    headers = (
        {'READPATT': 'FASTR1', 'NSAMPLES': 1, 'SUBSTRT1': 1},
        {'READPATT': 'SLOWR1', 'NSAMPLES': 9, 'SUBSTRT1': 5},
    )
    data = np.zeros((4, 4, 2, 16), np.float32)
    cube = _ramp(
        data, edges=[2, 4], starts=[1, 3], headers=headers)
    calls = []
    allocations = []

    def allocate(shape, dtype, *, name, **kwargs):
        allocations.append((tuple(shape), np.dtype(dtype), name))
        return np.empty(shape, dtype=dtype)

    def fake_emicorr(part, pixeldq, frequencies, waves, rowclocks, frameclocks,
                     **kwargs):
        np.testing.assert_array_equal(pixeldq, cube.pixeldq)
        calls.append((np.array(part), np.array(frequencies),
                      tuple(np.array(wave) for wave in waves),
                      rowclocks, frameclocks, kwargs))
        return np.asarray(part) + len(calls)

    monkeypatch.setattr(stages.k_emi, 'apply_emicorr_joint', fake_emicorr)
    monkeypatch.setattr(stages.core, 'empty_host_array', allocate)
    refpack = {
        'emicorr_frequencies': np.array([390.625, 218.52055]),
        'emicorr_reference_waves': np.array([
            [0., 1., 0., -1.], [1., 0., -1., 0.]]),
        'emicorr_reference_wave_lengths': np.array([4, 3]),
        'emicorr_rowclocks': np.int64(28),
        'emicorr_frameclocks': np.int64(15904),
    }

    out = stages.step_emicorr_miri(
        PipelineState(cube), {}, {'refpack': refpack})

    assert [call[0].shape[0] for call in calls] == [2, 2]
    assert calls[0][-1] == {
        'readpatt': 'FASTR1', 'nsamples': 1}
    assert calls[1][-1] == {
        'readpatt': 'SLOWR1', 'nsamples': 9}
    assert [wave.size for wave in calls[0][2]] == [4, 3]
    np.testing.assert_array_equal(calls[0][2][1], [1., 0., -1.])
    np.testing.assert_array_equal(np.asarray(out.cube.data[:2]), 1.)
    np.testing.assert_array_equal(np.asarray(out.cube.data[2:]), 2.)
    assert out.aux['emicorr'] == 'COMPLETE'
    assert allocations == [
        (data.shape, data.dtype, 'miri_emicorr_science')]


def test_miri_reset_uses_absolute_intstart_and_last_reference_integration(
        monkeypatch):
    """Check MIRI reset uses absolute intstart and last reference integration."""
    nints, ngroups, dimy, dimx = 4, 4, 2, 3
    cube = _ramp(
        np.zeros((nints, ngroups, dimy, dimx), np.float32),
        edges=[2, 4], starts=[1, 3])
    reset_data = np.stack([
        np.full((2, dimy, dimx), value, np.float32)
        for value in (10., 20., 30.)
    ])
    reset_dq = np.full((dimy, dimx), core.DQ_PERSISTENCE, np.uint32)
    allocations = []

    def allocate(shape, dtype, *, name, **kwargs):
        allocations.append((tuple(shape), np.dtype(dtype), name))
        return np.empty(shape, dtype=dtype)

    monkeypatch.setattr(stages.core, 'empty_host_array', allocate)

    out = stages.step_reset_miri(
        PipelineState(cube), {},
        {'refpack': {'reset_data': reset_data, 'reset_dq': reset_dq}})

    np.testing.assert_array_equal(
        np.asarray(out.cube.data[:, 0, 0, 0]), [-10., -20., -30., -30.])
    np.testing.assert_array_equal(np.asarray(out.cube.data[:, 1, 0, 0]),
                                  [-10., -20., -30., -30.])
    np.testing.assert_array_equal(np.asarray(out.cube.data[:, 2:]), 0.)
    np.testing.assert_array_equal(np.asarray(out.cube.pixeldq), reset_dq)
    assert allocations == [
        (cube.data.shape, np.dtype(cube.data.dtype), 'miri_reset_science')]


def test_miri_linearity_links_segmented_dark_and_rscd_group_flags():
    """Check MIRI linearity links segmented dark and rscd group flags."""
    nints, ngroups, dimy, dimx = 4, 7, 2, 3
    data = np.broadcast_to(
        np.arange(ngroups, dtype=np.float32)[None, :, None, None],
        (nints, ngroups, dimy, dimx)).copy()
    groupdq = np.zeros_like(data, np.uint8)
    groupdq[:, 0] = core.DQ_DO_NOT_USE
    groupdq[:, -1] = core.DQ_DO_NOT_USE
    cube = _ramp(
        data, edges=[2, 4], starts=[1, 3], groupdq=groupdq)
    dark = np.stack([
        np.full((ngroups, dimy, dimx), value, np.float32)
        for value in (1., 2., 3.)
    ])
    coeffs = np.zeros((2, dimy, dimx), np.float32)
    coeffs[1] = 1.
    dark_dq = np.full((dimy, dimx), core.DQ_HOT, np.uint32)
    ctx = {
        'opts': {'miri_subtract_dark': True, 'miri_drop_groups': 2},
        'refpack': {
            'lin_coeffs': coeffs,
            'lin_dq': np.zeros((dimy, dimx), np.uint32),
            'dark': dark,
            'dark_dq': dark_dq,
        },
    }

    out = stages.step_linearity_miri(PipelineState(cube), {}, ctx)

    np.testing.assert_array_equal(
        np.asarray(out.cube.data[:, 0, 0, 0]), [-1., -2., -3., -3.])
    expected_dnu = np.array([True, True, True, False, False, False, True])
    np.testing.assert_array_equal(
        (np.asarray(out.cube.groupdq[0, :, 0, 0]) &
         core.DQ_DO_NOT_USE) != 0,
        expected_dnu)
    np.testing.assert_array_equal(np.asarray(out.cube.pixeldq), dark_dq)
    assert out.aux['miri_dark_subtracted'] is True


def test_miri_dark_preallocates_one_segment_filled_output(monkeypatch):
    """Check MIRI dark preallocates one segment filled output."""
    cube = _ramp(
        np.zeros((4, 4, 2, 3), np.float32), edges=[2, 4], starts=[1, 3])
    dark = np.ones((2, 4, 2, 3), np.float32)
    allocations = []

    def allocate(shape, dtype, *, name, **kwargs):
        allocations.append((tuple(shape), np.dtype(dtype), name))
        return np.empty(shape, dtype=dtype)

    monkeypatch.setattr(stages.core, 'empty_host_array', allocate)
    monkeypatch.setattr(
        stages.core, 'map_over_ints',
        lambda fn, arrays, nints: np.asarray(arrays[0]) - np.asarray(arrays[1]))

    result, _ = stages._subtract_miri_dark(cube, dark, None)

    np.testing.assert_array_equal(result, -1.)
    assert allocations == [
        (cube.data.shape, np.dtype(cube.data.dtype), 'miri_dark_science')]


def test_miri_jump_passes_zero_reset_artifact_and_skips_dropped_groups(
        monkeypatch):
    """Check MIRI jump passes zero reset artifact and skips dropped groups."""
    data = np.zeros((4, 7, 2, 3), np.float32)
    groupdq = np.zeros_like(data, np.uint8)
    groupdq[:, 0] = core.DQ_DO_NOT_USE
    groupdq[:, 1] = core.DQ_DO_NOT_USE
    groupdq[:, -1] = core.DQ_DO_NOT_USE
    cube = _ramp(
        data, edges=[2, 4], starts=[1, 3], groupdq=groupdq)
    calls = []

    def fake_jump(part, dq, threshold, *, window, artifact, group_ok):
        calls.append((np.asarray(artifact), np.asarray(group_ok)))
        return part, dq

    monkeypatch.setattr(
        stages.k_jump, 'flag_jumps_in_time_chunked', fake_jump)
    stages.step_jump(
        PipelineState(cube),
        {'time_jump_threshold': 10., 'time_window': 5},
        {'opts': {'flag_up_ramp': False, 'flag_in_time': True}})

    assert len(calls) == 2
    assert all(not artifact.any() for artifact, _ in calls)
    expected_ok = np.array([False, False, True, True, True, True, False])
    for _, group_ok in calls:
        np.testing.assert_array_equal(group_ok, expected_ok)


def _miri_score_context(dimy):
    """Return MIRI score context."""
    return {
        'opts': {
            'flag_up_ramp': False,
            'flag_in_time': True,
            'wave_range': None,
            'w1': 0.,
            'w2': 1.,
        },
        'centroids': {
            'xpos': np.full(dimy, 2.5),
            'ypos': np.arange(dimy, dtype=float),
        },
    }


def test_miri_jump_scorer_uses_newest_scoreable_group():
    """Check MIRI jump scorer uses newest scoreable group."""
    rng = np.random.default_rng(0x1da1f1a)
    nints, ngroups, dimy, dimx = 9, 5, 8, 6
    data = rng.normal(
        100., 0.5, (nints, ngroups, dimy, dimx)).astype(np.float32)
    groupdq = np.zeros_like(data, np.uint8)
    groupdq[:, 0] = core.DQ_DO_NOT_USE
    groupdq[:, -1] = core.DQ_DO_NOT_USE
    cube = _ramp(data, groupdq=groupdq)
    ctx = _miri_score_context(dimy)
    params = {
        'time_jump_threshold': 7.,
        'time_window': 5,
        'extract_width': 4.,
    }

    scorer = stages.prepare_jump_scorer(
        PipelineState(cube), params, ctx)

    assert scorer.score_group == ngroups - 2
    results = scorer.evaluate_candidates(
        'time_jump_threshold', [5., 7., 9.], params)
    assert all(np.isfinite(cost) for cost, _, _ in results)


def test_miri_score_group_falls_back_and_ignores_jump_for_eligibility():
    """Check MIRI score group falls back and ignores jump for eligibility."""
    rng = np.random.default_rng(0x1da1f1b)
    nints, ngroups, dimy, dimx = 9, 6, 8, 6
    data = rng.normal(
        100., 0.5, (nints, ngroups, dimy, dimx)).astype(np.float32)
    groupdq = np.zeros_like(data, np.uint8)
    groupdq[:, -1] = core.DQ_DO_NOT_USE
    groupdq[:, -2] = core.DQ_DO_NOT_USE
    groupdq[0, -3, 0, 0] = core.DQ_JUMP_DET
    cube = _ramp(data, groupdq=groupdq)
    ctx = _miri_score_context(dimy)
    params = {'extract_width': 4.}

    selected = stages.miri_score_group(
        cube.data, cube.groupdq, params, ctx, cube.meta,
        ctx['centroids'])

    assert selected == ngroups - 3


def test_miri_score_group_rejects_fully_unusable_ramp():
    """Check MIRI score group rejects fully unusable ramp."""
    nints, ngroups, dimy, dimx = 9, 4, 8, 6
    data = np.ones((nints, ngroups, dimy, dimx), np.float32)
    groupdq = np.full_like(data, core.DQ_DO_NOT_USE, dtype=np.uint8)
    cube = _ramp(data, groupdq=groupdq)
    ctx = _miri_score_context(dimy)

    with pytest.raises(ValueError, match='No scoreable MIRI group'):
        stages.miri_score_group(
            cube.data, cube.groupdq, {'extract_width': 4.}, ctx,
            cube.meta, ctx['centroids'])


def test_miri_pca_only_reconstructs_detector_columns_12_through_60(
        monkeypatch):
    """Check MIRI PCA only reconstructs detector columns 12 through 60."""
    nints, dimy, dimx = 5, 3, 72
    data = np.arange(nints * dimy * dimx, dtype=np.float32).reshape(
        nints, dimy, dimx)
    cube = RateCube(
        data, np.ones_like(data), np.zeros_like(data, np.uint32),
        _meta(nints, ngroups=1))
    captured = {}

    def fake_pca(pca_input, remove, baseline, *, n_components,
                 return_plot_data):
        captured.update(
            pca_input=np.array(pca_input), remove=np.array(remove),
            baseline=np.array(baseline), n_components=n_components,
            return_plot_data=return_plot_data)
        components = np.zeros((n_components, nints), np.float32)
        eigvals = np.zeros(n_components, np.float32)
        wlc = np.zeros(nints, np.float32)
        return (np.asarray(pca_input) + 1000., components, eigvals, wlc,
                components.copy(), eigvals.copy())

    monkeypatch.setattr(stages.k_pca, 'pca_reconstruction', fake_pca)
    out = stages.step_pca(
        PipelineState(cube), {},
        {'opts': {'pca_components': 2, 'remove_components': [1],
                  'do_plots': False}})

    np.testing.assert_array_equal(captured['pca_input'], data[:, :, 12:61])
    np.testing.assert_array_equal(captured['remove'], [True, False])
    np.testing.assert_array_equal(np.asarray(out.cube.data[:, :, :12]),
                                  data[:, :, :12])
    np.testing.assert_array_equal(np.asarray(out.cube.data[:, :, 61:]),
                                  data[:, :, 61:])
    np.testing.assert_array_equal(np.asarray(out.cube.data[:, :, 12:61]),
                                  data[:, :, 12:61] + 1000.)
    np.testing.assert_array_equal(
        out.aux['stage3_deepframe'], np.median(data, axis=0))


def test_miri_vertical_extraction_keeps_native_y_wavelength_axis():
    """Check MIRI vertical extraction keeps native y wavelength axis."""
    nints, dimy, dimx = 12, 6, 8
    ypos, xpos = np.mgrid[:dimy, :dimx]
    plane = 100. * ypos + 10. * xpos
    data = np.broadcast_to(plane, (nints, dimy, dimx)).astype(np.float32)
    cube = RateCube(
        data, np.ones_like(data), np.zeros_like(data, np.uint32),
        _meta(nints, ngroups=1))
    centroids = {
        'xpos': np.full(4, 3., dtype=float),
        'ypos': np.arange(1., 5.),
    }
    wave_map = 1000. * ypos + xpos
    ctx = {
        'opts': {'mask_do_not_use_pixels': True, 'mask_saturated_pixels': True},
        'centroids': centroids,
        'miri_wave_map': wave_map,
    }

    out = stages.step_extract_miri(
        PipelineState(cube), {'extract_width': 2.}, ctx)
    product = out.aux['spectral_products'][1]

    assert product['flux'].shape == (nints, dimy)
    assert product['ferr'].shape == (nints, dimy)
    np.testing.assert_array_equal(
        product['flux'][0], [0., 250., 450., 650., 0., 0.])
    np.testing.assert_allclose(
        product['ferr'][0], [0., np.sqrt(2.), np.sqrt(2.), np.sqrt(2.), 0., 0.])
    np.testing.assert_allclose(
        product['wave'], [np.nan, 1003., 2003., 3003., 4003., np.nan],
        equal_nan=True)


def test_small_miri_prepare_build_and_run_path():
    """Run representative Stage 1/2/3 kernels across two segments."""
    nints, ngroups, dimy, dimx = 12, 8, 8, 48
    yy, xx = np.mgrid[:dimy, :dimx]
    trace_profile = 80. * np.exp(-0.5 * ((xx - 36.) / 1.2) ** 2)
    slope = 20. + trace_profile
    integration = np.arange(nints, dtype=np.float32)[:, None, None, None]
    group = np.arange(ngroups, dtype=np.float32)[None, :, None, None]
    data = (100. + .01 * integration + group * slope[None, None]) \
        .astype(np.float32)
    cube = _ramp(data, edges=[6, 12], starts=[1, 7])

    coeffs = np.zeros((2, dimy, dimx), np.float32)
    coeffs[1] = 1.
    wave_map = 5. + .1 * yy + .001 * xx
    zeros_u32 = np.zeros((dimy, dimx), np.uint32)
    zeros_f32 = np.zeros((dimy, dimx), np.float32)
    refpack = {
        'mask_dq': zeros_u32,
        'reset_data': np.zeros((1, ngroups, dimy, dimx), np.float32),
        'reset_dq': zeros_u32,
        'lin_coeffs': coeffs,
        'lin_dq': zeros_u32,
        'dark': np.zeros((1, ngroups, dimy, dimx), np.float32),
        'dark_dq': zeros_u32,
        'average_dark_current': zeros_f32,
        'readnoise': np.ones((dimy, dimx), np.float32),
        'gain': np.ones((dimy, dimx), np.float32),
        'gain_factor': np.float32(1.),
        'flat': np.ones((dimy, dimx), np.float32),
        'flat_dq': zeros_u32,
        'flat_err': zeros_f32,
        'wave_map': wave_map,
    }

    opts = {name: 'skip' for name in stages._MIRI_STEP_CONTROLS}
    for name in (
            'DQInitStep', 'LinearityStep', 'RampFitStep',
            'GainScaleStep', 'AssignWCSStep',
            'FlatFieldStep', 'BackgroundStep'):
        opts[name] = 'run'
    # Reference validation shares the instrument-agnostic step controls.
    opts.update({
        'INLCorrStep': 'skip', 'SuperBiasStep': 'skip',
        'RefPixStep': 'skip', 'DarkCurrentStep': 'skip',
        'OneOverFStep_grp': 'skip', 'OneOverFStep_int': 'skip',
        'mode': 'MIRI/LRS', 'extract_method': 'box',
        'miri_subtract_dark': True, 'miri_drop_groups': 1,
        'miri_background_method': 'median',
        'saturation_threshold': 100, 'mask_do_not_use_pixels': True, 'mask_saturated_pixels': True,
        'hot_pixel_map': None, 'outlier_maps': None,
        'centroids': {
            'xpos': np.full(dimy, 36., dtype=float),
            'ypos': np.arange(dimy, dtype=float),
        },
    })
    params = {
        'miri_trace_width': 4., 'miri_background_width': 4.,
        'extract_width': 4.,
    }

    ctx = stages.prepare_miri_context(cube, opts, refpack=refpack)
    pipeline = stages.build_pipeline('MIRI/LRS', ctx)
    out = pipeline.run(PipelineState(cube), params)

    assert [step.name for step in pipeline.steps] == [
        'DQInitStep', 'LinearityStep', 'RampFitStep',
        'GainScaleStep', 'AssignWCSStep', 'FlatFieldStep',
        'BackgroundStep', 'Extract',
    ]
    assert isinstance(out.cube, RateCube)
    assert np.isfinite(np.asarray(out.cube.data)).all()
    product = out.aux['spectral_products'][1]
    assert product['wave'].shape == (dimy,)
    assert product['flux'].shape == (nints, dimy)
    assert product['ferr'].shape == (nints, dimy)
    np.testing.assert_allclose(product['wave'], wave_map[:, 36])
    # Check the excluded row before v1 2.5.0's channel interpolation.
    raw = stages._extract_orders_miri(
        out, params, ctx, mode='sum', centroids=ctx['centroids'])[1][0]
    assert np.all(raw[:, -1] == 0.)
    from .stellar_wave_oracle import load_v1
    expected = load_v1().utils.sigma_clip_lightcurves(raw, thresh=10, window=10)
    np.testing.assert_array_equal(product['flux'], expected)
    # 2.5.0 clipping NaNs runs of >= 10 high-variance channels (v1 rule).
    assert np.isfinite(product['flux']).mean() > 0.8
