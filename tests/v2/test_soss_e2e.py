"""Small, real-kernel SOSS graph smoke test."""

import numpy as np
import pytest

from exotedrf.v2 import stages
from exotedrf.v2.core import ObsMeta, RampCube, RateCube
from exotedrf.v2.pipeline import CheckpointStore, PipelineState


def build_synthetic_soss_case(group_oof=True):
    """Cube, context and parameters for the two-segment synthetic graph.

    Parameters
    ----------
    group_oof : bool
        Group oof option.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    nints, ngroups, dimy, dimx = 12, 3, 16, 32
    yy, xx = np.mgrid[:dimy, :dimx]
    spatial = 0.05 * yy + 0.02 * xx
    data = np.empty((nints, ngroups, dimy, dimx), np.float32)
    for integration in range(nints):
        for group in range(ngroups):
            data[integration, group] = (
                1000. + 75. * group + spatial + 0.1 * integration)

    meta = ObsMeta(
        mode='NIRISS/SOSS', detector='NIS', subarray='SUBSTRIP96',
        frame_time=2.214, ngroups=ngroups,
        int_times=60000. + np.arange(nints) * 1e-4,
        baseline_ints=np.asarray([4, -4]),
        segment_edges=np.asarray([6, 12]),
        filenames=('seg001_uncal.fits', 'seg002_uncal.fits'),
        extra={'segment_int_starts': (1, 20)},
    )
    cube = RampCube(
        data=data,
        groupdq=np.zeros_like(data, np.uint8),
        pixeldq=np.zeros((dimy, dimx), np.uint32),
        meta=meta,
    )

    lin_coeffs = np.zeros((2, dimy, dimx), np.float32)
    lin_coeffs[1] = 1.
    wave_o1 = np.linspace(2.8, 0.8, dimx)
    wave_o2 = np.linspace(1.4, 0.6, dimx)
    refpack = {
        'mask_dq': np.zeros((dimy, dimx), np.uint32),
        'inl_theta': np.zeros(6, np.float32),
        'inl_periods': np.asarray([1024 / 3, 512, 1024], np.float32),
        'superbias': np.zeros((dimy, dimx), np.float32),
        'dark': np.zeros((ngroups, dimy, dimx), np.float32),
        'average_dark_current': np.zeros((dimy, dimx), np.float32),
        'lin_coeffs': lin_coeffs,
        'lin_dq': np.zeros((dimy, dimx), np.uint32),
        'readnoise': np.full((dimy, dimx), 6., np.float32),
        'gain': np.ones((dimy, dimx), np.float32),
        'gain_factor': np.float32(1.),
        'flat': np.ones((dimy, dimx), np.float32),
        'wave_o1': wave_o1,
        'wave_o2': wave_o2,
    }
    controls = {
        name: 'run' for name in stages._STEP_CONTROLS
    }
    controls['OneOverFStep_grp'] = 'run' if group_oof else 'skip'
    opts = {
        **controls,
        'oof_method': 'scale-achromatic',
        'superbias_method': 'crds',
        'extract_method': 'box',
        'extract_width_soss2': 2,
        'pca_components': 2,
        'remove_components': None,
        'flag_up_ramp': False,
        'flag_in_time': True,
        'jump_threshold': 15,
        'saturation_threshold': 80,
        'mask_do_not_use_pixels': True, 'mask_saturated_pixels': True,
        'saturation_rescue': False,
    }
    ctx = {
        'opts': opts,
        'refpack': refpack,
        'input_cube': cube,
        'background_model': np.full((dimy, dimx), 0.1, np.float32),
        'soss_timeseries': np.ones(nints, np.float32),
        'soss_timeseries_o2': None,
        'centroids': {
            'ypos o1': np.full(dimx, 5., np.float32),
            'ypos o2': np.full(dimx, 10., np.float32),
        },
        'waves': {1: wave_o1, 2: wave_o2},
        'wavemap_provenance': 'synthetic-test',
    }
    params = {
        'soss_inner_mask_width': 2,
        'soss_outer_mask_width': 6,
        'time_jump_threshold': 1e6,
        'time_window': 3,
        'space_outlier_threshold': 1e6,
        'time_outlier_threshold': 1e6,
        'box_size': 2,
        'window_size': 3,
        'extract_width': 2,
    }
    return cube, ctx, params


@pytest.mark.parametrize('automatic', [False, True])
@pytest.mark.parametrize('group_oof', [False, True])
def test_all_enabled_soss_steps_run_end_to_end_on_two_segments(
        monkeypatch, tmp_path, automatic, group_oof):
    """Check all enabled SOSS steps run end to end on two segments."""
    cube, ctx, params = build_synthetic_soss_case(group_oof=group_oof)
    nints, dimy, dimx = (cube.data.shape[0], *cube.data.shape[-2:])
    if automatic:
        # Trace geometry depends on the processed global baseline.
        ctx['centroids'] = None
        ctx['soss_timeseries'] = None

        def trace_deep(deep, cube, ctx):
            shift = float(np.nanmedian(deep)) / 1e5
            return {'ypos o1': np.full(dimx, 5. + shift, np.float32),
                    'ypos o2': np.full(dimx, 10. + shift, np.float32)}

        monkeypatch.setattr(stages, '_trace_soss_deepstack', trace_deep)

    pipeline = stages.build_pipeline('NIRISS/SOSS', ctx)
    store = CheckpointStore()
    out = pipeline.run(PipelineState(cube), params, store=store)

    assert isinstance(out.cube, RateCube)
    assert out.cube.data.shape == (nints, dimy, dimx)
    assert np.isfinite(np.asarray(out.cube.data)).all()
    expected_steps = [
        'DQInitStep', 'INLCorrStep', 'SuperBiasStep', 'RefPixStep',
        'DarkCurrentStep', 'BackgroundStep_grp', 'OneOverFStep_grp',
        'LinearityStep', 'JumpStep', 'RampFitStep', 'GainScaleStep',
        'AssignWCSStep', 'FlatFieldStep',
        'BackgroundStep', 'OneOverFStep_int', 'BadPixStep',
        'PCAReconstructStep', 'Extract',
    ]
    if not group_oof:
        expected_steps.remove('BackgroundStep_grp')
        expected_steps.remove('OneOverFStep_grp')
    assert [step.name for step in pipeline.steps] == expected_steps
    assert all(name in store for name in ('JumpStep', 'BadPixStep', 'Extract'))
    assert ('OneOverFStep_grp' in store) == group_oof
    assert out.aux['pca_components'].shape == (2, nints)
    for order in (1, 2):
        product = out.aux['spectral_products'][order]
        assert product['flux'].shape[0] == nints
        assert np.isfinite(product['flux']).all()
        assert np.all(np.diff(product['wave']) > 0)

    from exotedrf.v2.streaming import run_soss_segmented
    streamed = run_soss_segmented(pipeline, PipelineState(cube), params,
                                 store=CheckpointStore(keep={'Extract'}))
    for name in ('data', 'err', 'dq'):
        np.testing.assert_array_equal(getattr(streamed.cube, name),
                                      getattr(out.cube, name))
    for order in (1, 2):
        for name in ('wave', 'flux', 'ferr'):
            np.testing.assert_array_equal(
                streamed.aux['spectral_products'][order][name],
                out.aux['spectral_products'][order][name])

    monkeypatch.setenv('EXOTEDRF_MAX_HOST_BYTES', '1B')
    monkeypatch.setenv('EXOTEDRF_SCRATCH_DIR', str(tmp_path))
    spilled = pipeline.run(PipelineState(cube), params)
    for name in ('data', 'err', 'dq'):
        np.testing.assert_array_equal(getattr(spilled.cube, name),
                                      getattr(out.cube, name))
    for order in (1, 2):
        np.testing.assert_array_equal(
            spilled.aux['spectral_products'][order]['flux'],
            out.aux['spectral_products'][order]['flux'])
