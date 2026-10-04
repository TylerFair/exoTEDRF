"""v1 OneOverFStep kwargs ``smoothing_scale`` / ``even_odd_rows`` parity."""
import ast
import warnings
from pathlib import Path
from types import SimpleNamespace

import bottleneck as bn
import numpy as np
import pytest
from astropy.io import fits
from scipy.ndimage import median_filter

from exotedrf.v2 import config, stages
from exotedrf.v2.core import ObsMeta, RampCube, RateCube
from exotedrf.v2.pipeline import PipelineState

REPO = Path(__file__).resolve().parents[2]
DIMY, DIMX = 64, 1560
Y1 = 32.4


def _load(relative, names, namespace):
    """Return load."""
    path = REPO / relative
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'),
         namespace)
    return namespace


def _v1_scale():
    """Return v1 scale."""
    helpers = {'np': np, 'fits': fits, 'median_filter': median_filter,
               'warnings': warnings, 'fancyprint': lambda *a, **k: None,
               'get_instrument_name': lambda value: 'NIRISS',
               'get_soss_subarray': lambda value: 'SUBSTRIP96',
               'open_filetype': lambda value: SimpleNamespace()}
    _load('exotedrf/utils.py', ['scatter_normalize_cube', 'make_soss_tracemask',
                                'mask_reset_artifact'], helpers)
    utils = SimpleNamespace(**helpers)
    return _load('exotedrf/stage1.py', ['oneoverfstep_scale'], {
        'np': np, 'bn': bn, 'fits': fits, 'utils': utils, 'warnings': warnings,
        'median_filter': median_filter, 'tqdm': lambda x: x,
        'fancyprint': lambda *a, **k: None})['oneoverfstep_scale']


def _signal(rng, nints, extra_shape=()):
    """Return signal."""
    yy = np.arange(DIMY)[:, None]
    prof = 800. * np.exp(-0.5 * ((yy - Y1) / 2.) ** 2) + 5.
    lc = 1. - 0.03 * ((np.arange(nints) > nints // 3) &
                      (np.arange(nints) < 2 * nints // 3))
    lc = lc + rng.normal(0., 0.002, nints)
    shape = (nints, *extra_shape, DIMY, DIMX)
    scale = lc.reshape((nints,) + (1,) * (len(shape) - 1))
    grow = (np.arange(1, extra_shape[0] + 1).reshape(-1, 1, 1)
            if extra_shape else 1.)
    oof = rng.normal(0., 3., (nints, *extra_shape, 2, DIMX))
    rows = np.arange(DIMY) % 2
    data = scale * prof * grow + np.take(oof, rows, axis=-2)
    return (data + rng.normal(0., 1., shape)).astype(np.float32)


def _meta(nints, ngroups):
    """Return meta."""
    return ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP96', 5.494, ngroups,
                   np.arange(nints, dtype=float), np.array([3, -3]),
                   np.array([nints]), ('jwx-seg001_nis_uncal.fits',),
                   {'segment_int_starts': (300,)})


def _ctx(opts):
    """Return ctx."""
    return {'opts': {'oof_method': 'scale-achromatic', **opts},
            'centroids': {'ypos o1': np.full(DIMX, Y1)},
            'outlier_mask': None, 'order0_mask': None,
            'soss_timeseries': None, 'soss_timeseries_o2': None}


def _write(path, sci, dq, pixeldq=None, nints=None):
    """Return write."""
    primary = fits.PrimaryHDU()
    primary.header['FILENAME'] = path.name
    primary.header['INTSTART'] = 300
    primary.header['INTEND'] = 300 + sci.shape[0] - 1
    hdus = [primary, fits.ImageHDU(sci, name='SCI')]
    if pixeldq is not None:
        hdus += [fits.ImageHDU(pixeldq, name='PIXELDQ'),
                 fits.ImageHDU(dq, name='GROUPDQ')]
    else:
        hdus += [fits.ImageHDU(np.ones_like(sci), name='ERR'),
                 fits.ImageHDU(dq, name='DQ')]
    fits.HDUList(hdus).writeto(path)
    return str(path)


def test_estimate_timeseries_smoothing_matches_scipy_median_filter():
    """Check estimate timeseries smoothing matches scipy median filter."""
    rng = np.random.default_rng(1)
    data = rng.normal(100., 5., (40, 8, 12)).astype(np.float32)
    deep = np.median(data, axis=0)
    meta = ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP96', 1., 1,
                   np.arange(40.), np.array([3, -3]), np.array([25, 40]),
                   (), {'segment_int_starts': (1, 26)})
    for scale in (None, 1, 4, 7):
        got = stages._estimate_segment_timeseries(data, deep, meta,
                                                  smoothing_scale=scale)
        parts = []
        for lo, hi in ((0, 25), (25, 40)):
            ts = np.nansum(data[lo:hi], axis=(1, 2)) / np.nansum(deep)
            size = max(1, int(0.02 * (hi - lo))) if scale is None else scale
            parts.append(median_filter(ts, size))
        np.testing.assert_allclose(got, np.concatenate(parts), rtol=2e-6)


@pytest.mark.parametrize('smoothing', [None, 4, 5])
def test_integration_level_smoothing_matches_actual_v1(tmp_path, smoothing):
    """Check integration level smoothing matches actual v1."""
    rng = np.random.default_rng(7)
    nints = 30
    sci = _signal(rng, nints)
    dq = np.zeros(sci.shape, np.uint32)
    dq[rng.random(sci.shape) < 0.002] = 4
    path = _write(tmp_path / 'rate.fits', sci, dq)
    deep = bn.nanmedian(sci, axis=0)
    centroids = {'xpos': np.arange(DIMX), 'ypos o1': np.full(DIMX, Y1),
                 'ypos o2': np.full(DIMX, np.nan),
                 'ypos o3': np.full(DIMX, np.nan)}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        result, _ = _v1_scale()(path, deep, inner_mask_width=12,
                                save_results=False, method='achromatic',
                                centroids=centroids,
                                smoothing_scale=smoothing)
    cfg = {'observing_mode': 'NIRISS/SOSS',
           'stage2_kwargs': {'OneOverFStep': {'smoothing_scale': smoothing}}}
    opts = config.fixed_options(cfg)
    state = PipelineState(
        RateCube(sci.copy(), np.ones_like(sci), dq.copy(), _meta(nints, 1)),
        {stages._OOF_INT_DEEP_KEY: deep})
    out = stages.step_oneoverf_int(
        state, {'soss_inner_mask_width': 12, 'soss_outer_mask_width': 30},
        _ctx({key: opts[key] for key in opts if key.startswith('oof_')}))
    np.testing.assert_allclose(np.asarray(out.cube.data), result.data,
                               rtol=0, atol=2e-3)


def test_group_level_even_odd_false_and_smoothing_match_actual_v1(tmp_path):
    """Check group level even odd false and smoothing match actual v1."""
    rng = np.random.default_rng(9)
    nints, ngroups = 20, 2
    sci = _signal(rng, nints, (ngroups,))
    groupdq = np.zeros(sci.shape, np.uint8)
    groupdq[rng.random(sci.shape) < 0.002] = 4
    pixeldq = np.zeros((DIMY, DIMX), np.uint32)
    pixeldq[rng.random((DIMY, DIMX)) < 0.003] = 1
    path = _write(tmp_path / 'ramp.fits', sci, groupdq, pixeldq=pixeldq)
    deep = bn.nanmedian(sci, axis=0)
    centroids = {'xpos': np.arange(DIMX), 'ypos o1': np.full(DIMX, Y1),
                 'ypos o2': np.full(DIMX, np.nan),
                 'ypos o3': np.full(DIMX, np.nan)}
    background = np.full((ngroups, DIMY, DIMX), 0.25, np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        result, _ = _v1_scale()(path, deep, inner_mask_width=12,
                                even_odd_rows=False, background=background,
                                save_results=False, method='achromatic',
                                centroids=centroids, smoothing_scale=3)
    cfg = {'observing_mode': 'NIRISS/SOSS', 'stage1_kwargs': {
        'OneOverFStep': {'smoothing_scale': 3, 'even_odd_rows': False}}}
    opts = config.fixed_options(cfg)
    ctx = _ctx({key: opts[key] for key in opts if key.startswith('oof_')})
    cube = RampCube(sci.copy(), groupdq.copy(), pixeldq.copy(),
                    _meta(nints, ngroups))
    state = PipelineState(cube, {'bkg_grp': background})
    state.aux[stages._OOF_GRP_PREP_KEY] = stages._prepare_oneoverf_grp(
        state, ctx, deep=deep)
    out = stages.step_oneoverf_grp(
        state, {'soss_inner_mask_width': 12, 'soss_outer_mask_width': 30},
        ctx)
    np.testing.assert_allclose(np.asarray(out.cube.data), result.data,
                               rtol=0, atol=5e-3)
    # The kwargs genuinely change the answer.
    default = stages.step_oneoverf_grp(
        PipelineState(cube, {'bkg_grp': background, stages._OOF_GRP_PREP_KEY:
                             stages._prepare_oneoverf_grp(
                                 PipelineState(cube, {'bkg_grp': background}),
                                 _ctx({}), deep=deep)}),
        {'soss_inner_mask_width': 12, 'soss_outer_mask_width': 30}, _ctx({}))
    assert np.nanmax(np.abs(np.asarray(default.cube.data) -
                            result.data)) > 0.05


def test_v1_integration_level_even_odd_false_crashes(tmp_path):
    """Check v1 integration level even odd false crashes."""
    rng = np.random.default_rng(3)
    sci = _signal(rng, 12)
    path = _write(tmp_path / 'rate.fits', sci, np.zeros(sci.shape, np.uint32))
    deep = bn.nanmedian(sci, axis=0)
    centroids = {'xpos': np.arange(DIMX), 'ypos o1': np.full(DIMX, Y1),
                 'ypos o2': np.full(DIMX, np.nan),
                 'ypos o3': np.full(DIMX, np.nan)}
    with pytest.raises(IndexError):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            _v1_scale()(path, deep, inner_mask_width=12, even_odd_rows=False,
                        save_results=False, method='achromatic',
                        centroids=centroids, timeseries=np.ones(12))


def test_non_finite_timeseries_values_leave_columns_uncorrected_like_v1():
    """Check non finite timeseries values leave columns uncorrected like v1."""
    rng = np.random.default_rng(12)
    nints, dimy, dimx = 6, 8, 10
    series = np.ones((nints, dimx), np.float32)
    series[2, 4] = np.nan
    with pytest.warns(UserWarning, match='non-finite'):
        checked = stages._validate_timeseries(series, 'soss_timeseries',
                                              nints, dimx, require_2d=True)
    data = rng.normal(100., 3., (nints, dimy, dimx)).astype(np.float32)
    deep = np.median(data, axis=0)
    outliers = np.zeros(data.shape, bool)
    from exotedrf.v2.kernels import oneoverf
    import jax.numpy as jnp
    got = np.asarray(oneoverf.oneoverf_scale_achromatic(
        jnp.asarray(data), jnp.asarray(deep), jnp.asarray(outliers),
        jnp.asarray(checked)))
    np.testing.assert_array_equal(got[2, :, 4], data[2, :, 4])
    finite = np.asarray(oneoverf.oneoverf_scale_achromatic(
        jnp.asarray(data), jnp.asarray(deep), jnp.asarray(outliers),
        jnp.asarray(np.nan_to_num(checked, nan=1.))))
    keep = np.ones(data.shape, bool)
    keep[2, :, 4] = False
    np.testing.assert_array_equal(got[keep], finite[keep])
