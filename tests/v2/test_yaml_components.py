"""Check the kernels selected by the NIRISS optimizer configuration."""

from pathlib import Path

import numpy as np
import pytest

from exotedrf.v2 import config as v2config
from exotedrf.v2 import products, stages
from exotedrf.v2.core import ObsMeta, RampCube, RateCube
from exotedrf.v2.pipeline import CheckpointStore, PipelineState

from .niriss_acceptance import apply_current_niriss_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
NIRISS_YAML = REPO_ROOT / 'tests' / 'v2' / 'data' / 'run_optimize_niriss.yaml'


def load_niriss_cfg():
    """Load the NIRISS optimizer test configuration.

    Returns
    -------
    config : dict
        NIRISS optimizer configuration.
    """
    return apply_current_niriss_settings(v2config.load_config(NIRISS_YAML))


def make_meta(nints, edges=None, starts=None, ngroups=3, dimy=4, dimx=4,
              subarray='SUBSTRIP96', baseline_ints=(4, -4)):
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
    dimy : int
        Number of detector rows.
    dimx : int
        Number of detector columns.
    subarray : str
        Subarray name.
    baseline_ints : tuple[int]
        Baseline ints.

    Returns
    -------
    meta : ObsMeta
        Synthetic observation metadata.
    """
    edges = np.asarray(edges if edges is not None else [nints])
    extra = {} if starts is None else {'segment_int_starts': tuple(starts)}
    return ObsMeta('NIRISS/SOSS', 'NIS', subarray, 2.214, ngroups,
                   np.arange(nints, dtype=float),
                   np.asarray(baseline_ints), edges,
                   tuple(f'seg{i}.fits' for i in range(len(edges))), extra)


# YAML parsing / option resolution.

def test_yaml_dead_saturation_keys_are_parsed_but_never_consumed():
    """Check YAML dead saturation keys are parsed but never consumed."""
    cfg = load_niriss_cfg()
    assert cfg['saturation_fraction'] == 0.8
    assert cfg['propagate_saturation'] is True
    assert v2config.validate_supported_config(cfg) is cfg
    opts = v2config.fixed_options(cfg)
    assert 'saturation_fraction' not in opts
    assert 'propagate_saturation' not in opts


def test_yaml_fixed_options_resolve_v1_defaults():
    """Check YAML fixed options resolve v1 defaults."""
    cfg = load_niriss_cfg()
    opts = v2config.fixed_options(cfg)

    assert opts['wave_range'] == [1.0, 2.0]
    assert opts['w1'] == 0.0 and opts['w2'] == 1.0

    assert opts['saturation_rescue'] is True
    assert opts['mask_do_not_use_pixels'] is True

    assert opts['INLCorrStep'] == 'run'
    assert opts['DarkCurrentStep'] == 'skip'
    assert opts['OneOverFStep_grp'] == 'run'
    assert opts['PCAReconstructStep'] == 'run'

    assert opts['flag_up_ramp'] is False
    assert opts['flag_in_time'] is True

    assert opts['pca_components'] == 10
    assert opts['remove_components'] is None
    assert opts['superbias_method'] == 'crds'
    assert opts['generate_lc'] is True

    assert opts['do_plots'] is True
    # Planet_letter only names v1 Stage-3 files and stays unforwarded.
    assert 'planet_letter' not in opts

    assert opts['saturation_threshold'] == 80


def test_yaml_sweep_plan_matches_v1_greedy_order_and_initials():
    """Check YAML sweep plan matches v1 greedy order and initials."""
    cfg = load_niriss_cfg()
    plan, initial = v2config.build_sweep_plan(cfg, 'NIRISS/SOSS')
    assert [name for name, _ in plan] == [
        'OneOverFStep_grp', 'JumpStep', 'BadPixStep', 'Extract']
    grouped = {name: [p for p, _ in group] for name, group in plan}
    assert grouped['OneOverFStep_grp'] == ['soss_inner_mask_width',
                                           'soss_outer_mask_width']
    assert grouped['JumpStep'] == ['time_jump_threshold', 'time_window']
    assert grouped['BadPixStep'] == [
        'space_outlier_threshold', 'time_outlier_threshold', 'box_size',
        'window_size']
    assert initial['soss_inner_mask_width'] == 40
    assert initial['soss_outer_mask_width'] == 70
    assert initial['time_jump_threshold'] == 7
    assert initial['time_window'] == 7
    assert initial['space_outlier_threshold'] == 8
    assert initial['time_outlier_threshold'] == 8
    assert initial['box_size'] == 5
    assert initial['window_size'] == 7
    assert initial['extract_width'] == 30
    # Phase-1 width is the upper-middle candidate, not the mean.
    assert v2config.phase1_extract_width(cfg, initial) == 30


# Stage 1 — DQInit saturation threshold (yaml literal None).

def _dq_init_ctx(saturation_threshold):
    """Return DQ init ctx."""
    return {
        'opts': {'saturation_threshold': saturation_threshold},
        'refpack': {'mask_dq': np.zeros((4, 12), np.uint32)},
    }


def _tiny_ramp_state():
    """Return tiny ramp state."""
    data = np.zeros((1, 2, 4, 12), np.float32)
    data[0, 1, 2, 6] = 55000.
    cube = RampCube(data, np.zeros_like(data, np.uint8),
                    np.zeros((4, 12), np.uint32),
                    make_meta(1, ngroups=2, dimx=12))
    return PipelineState(cube)


@pytest.mark.parametrize('threshold', [None, 80])
def test_dq_init_default_saturation_threshold(threshold):
    """Flag and dilate saturation using the default full-well fraction."""
    out = stages.step_dq_init(_tiny_ramp_state(), {}, _dq_init_ctx(threshold))
    saturated = (np.asarray(out.cube.groupdq) & 2) != 0
    assert saturated[0, 1, 2, 6]
    assert saturated[0, 1, 1:4, 5:8].all()


# Stage 1 — RefPix under both SOSS subarrays.

def test_step_refpix_only_corrects_substrip256():
    """Check step reference-pixel correction only corrects substrip256."""
    nints, ngroups, dimy, dimx = 2, 2, 8, 6
    rng = np.random.default_rng(7)
    data = rng.normal(10., 1., (nints, ngroups, dimy, dimx)).astype(
        np.float32)
    data[:, :, -4:, :] = 5.

    cube256 = RampCube(data.copy(), np.zeros_like(data, np.uint8),
                       np.zeros((dimy, dimx), np.uint32),
                       make_meta(nints, ngroups=ngroups, dimy=dimy,
                                 dimx=dimx, subarray='SUBSTRIP256'))
    out256 = stages.step_refpix(PipelineState(cube256), {}, {})
    # Even/odd reference means are both exactly 5 -> subtract 5 everywhere.
    np.testing.assert_allclose(np.asarray(out256.cube.data), data - 5.,
                               rtol=1e-6, atol=1e-5)

    cube96 = RampCube(data.copy(), np.zeros_like(data, np.uint8),
                      np.zeros((dimy, dimx), np.uint32),
                      make_meta(nints, ngroups=ngroups, dimy=dimy,
                                dimx=dimx, subarray='SUBSTRIP96'))
    state96 = PipelineState(cube96)
    assert stages.step_refpix(state96, {}, {}) is state96


# Stage 3 — saturation_rescue=True with mask_do_not_use_pixels=True.

def _extract_fixture():
    """Return extract fixture."""
    data = np.zeros((11, 10, 4), np.float32)
    data[:, 2] = 10.
    data[:, 3] = 2.
    err = np.ones_like(data)
    dq = np.zeros_like(data, np.uint32)
    dq[:, 2, 0] = 1
    dq[:, 2, 1] = 2
    dq[:, 2, 2] = 4
    cube = RateCube(data, err, dq, make_meta(11, dimy=10, dimx=4))
    return PipelineState(cube)


def test_final_extract_saturation_rescue_keeps_saturated_pixels():
    """Check final extract saturation rescue keeps saturated pixels."""
    ctx = {
        'opts': {'mask_do_not_use_pixels': True, 'mask_saturated_pixels': True,
                 'saturation_rescue': True},
        'centroids': {'ypos o1': np.full(4, 3.)},
        'waves': {1: np.arange(4.)},
    }
    out = stages.step_extract(_extract_fixture(), {'extract_width': 2}, ctx)
    flux, ferr = out.aux['spectra'][1]
    # Col0: DNU masked -> 2; col1: SATURATED rescued -> 12; col2/3 -> 12.
    np.testing.assert_allclose(flux[0], [2., 12., 12., 12.])
    np.testing.assert_allclose(
        ferr[0], [1., np.sqrt(2.), np.sqrt(2.), np.sqrt(2.)])

    ctx_norescue = dict(ctx, opts={'mask_do_not_use_pixels': True,
                                   'mask_saturated_pixels': True,
                                   'saturation_rescue': False})
    out2 = stages.step_extract(_extract_fixture(), {'extract_width': 2},
                               ctx_norescue)
    np.testing.assert_allclose(out2.aux['spectra'][1][0][0],
                               [2., 2., 12., 12.])


# Full graph under the exact yaml branch set.

def test_yaml_branch_soss_graph_runs_end_to_end():
    """Run every branch the shipped yaml selects, on a tiny synthetic cube."""
    nints, ngroups, dimy, dimx = 16, 3, 16, 32
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
        segment_edges=np.asarray([8, 16]),
        filenames=('seg001_uncal.fits', 'seg002_uncal.fits'),
        extra={'segment_int_starts': (1, 30)},
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
        'lin_coeffs': lin_coeffs,
        'lin_dq': np.zeros((dimy, dimx), np.uint32),
        'readnoise': np.full((dimy, dimx), 6., np.float32),
        'gain': np.ones((dimy, dimx), np.float32),
        'gain_factor': np.float32(1.),
        'flat': np.ones((dimy, dimx), np.float32),
        'wave_o1': wave_o1,
        'wave_o2': wave_o2,
    }
    controls = {name: 'run' for name in stages._STEP_CONTROLS}
    controls['DarkCurrentStep'] = 'skip'
    opts = {
        **controls,
        'oof_method': 'scale-achromatic',
        'superbias_method': 'crds',
        'extract_method': 'box',
        'pca_components': 10,
        'remove_components': None,
        'flag_up_ramp': False,
        'flag_in_time': True,
        'jump_threshold': 15,
        'saturation_threshold': 80,
        'mask_do_not_use_pixels': True,
        'saturation_rescue': True,
        'generate_lc': True,
        'wave_range': [1.0, 2.0],
        'w1': 0.0,
        'w2': 1.0,
    }
    order0 = np.zeros((dimy, dimx), bool)
    order0[3:6, 20:23] = True
    ctx = {
        'opts': opts,
        'refpack': refpack,
        'input_cube': cube,
        'background_model': np.full((dimy, dimx), 0.1, np.float32),
        'soss_timeseries': None,
        'soss_timeseries_o2': None,
        'order0_mask': order0,
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

    pipeline = stages.build_pipeline('NIRISS/SOSS', ctx)
    names = [step.name for step in pipeline.steps]
    assert 'DarkCurrentStep' not in names
    assert names[0] == 'DQInitStep' and names[-1] == 'Extract'

    store = CheckpointStore()
    out = pipeline.run(PipelineState(cube), params, store=store)

    assert isinstance(out.cube, RateCube)
    assert out.cube.data.shape == (nints, dimy, dimx)
    assert np.isfinite(np.asarray(out.cube.data)).all()
    # PCA executed (nints=16 > pca_components=10).
    assert out.aux['pca_components'].shape == (10, nints)
    assert out.aux['pca_wlc'].shape == (nints,)
    for order in (1, 2):
        product = out.aux['spectral_products'][order]
        assert product['flux'].shape[0] == nints
        assert np.isfinite(product['flux']).all()
    cost, scatter = stages.evaluate_production_cost(out, params, ctx)
    assert np.isfinite(float(cost))
    assert np.asarray(scatter).ndim == 1


# PCAReconstructStep small-cube guard (documented v2-only behavior).

def test_pca_step_silently_skips_when_nints_not_above_components():
    """Check PCA step silently skips when nints not above components."""
    rng = np.random.default_rng(3)
    data = rng.normal(size=(6, 3, 4)).astype(np.float32)
    cube = RateCube(data, np.ones_like(data), np.zeros_like(data, np.uint32),
                    make_meta(6, dimy=3, dimx=4))
    state = PipelineState(cube)
    out = stages.step_pca(
        state, {}, {'opts': {'pca_components': 10,
                             'remove_components': None}})
    assert out is state
    assert 'pca_wlc' not in out.aux


# Generate_lc=True — v1-compatible PCA light-curve product.

def test_write_final_products_emits_lcestimate_product(tmp_path):
    """Check write final products emits lcestimate product."""
    nints, dimy, dimx = 6, 4, 4
    data = np.ones((nints, dimy, dimx), np.float32)
    cube = RateCube(data, np.ones_like(data),
                    np.zeros_like(data, np.uint32),
                    make_meta(nints, dimy=dimy, dimx=dimx))
    wave = np.linspace(1.0, 2.0, dimx)
    aux = {
        'pca_wlc': np.ones(nints, np.float32),
        'spectral_products': {
            1: {'wave': wave, 'flux': np.ones((nints, dimx), np.float32),
                'ferr': np.ones((nints, dimx), np.float32)},
        },
    }
    state = PipelineState(cube, aux=aux)
    ctx = {'opts': {'generate_lc': True}, 'waves': {1: wave}}
    paths = products.output_layout(
        {'name_tag': ''}, output_dir=tmp_path, create=True)

    written = products.write_final_products(state, {}, ctx, paths)
    assert set(written) == {'lcestimate', 'rate', 'spectra'}
    assert Path(written['lcestimate']).name == 'seg0_lcestimate.npy'
    np.testing.assert_array_equal(
        np.load(written['lcestimate'], allow_pickle=False), aux['pca_wlc'])
