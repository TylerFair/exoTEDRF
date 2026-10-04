"""Tests for exotedrf.v2.kernels.oneoverf against numpy references."""

import warnings

import numpy as np
import pytest
from scipy.ndimage import median_filter

import jax.numpy as jnp

from exotedrf.v2.kernels import oneoverf

NINTS, NGROUPS, DIMY, DIMX = 8, 3, 16, 32


# Numpy references (v1 transcriptions).

def ref_make_soss_tracemask(ypix, mask_width, dimy, dimx):
    """Build the v1 SOSS trace mask.

    Parameters
    ----------
    ypix : array-like(float)
        Ypix array.
    mask_width : int, float
        Trace-mask width.
    dimy : int
        Number of detector rows.
    dimx : int
        Number of detector columns.

    Returns
    -------
    mask : np.ndarray(float)
        Trace mask in the reference convention.
    """
    low = np.max([np.zeros_like(ypix), ypix - mask_width / 2], axis=0
                 ).astype(int)
    up = np.min([dimy * np.ones_like(ypix), ypix + mask_width / 2], axis=0
                ).astype(int)
    mask = np.zeros((dimy, dimx))
    for i in range(len(ypix)):
        mask[low[i]:up[i], i] = 1
    return mask


def ref_build_soss_masks(base, y1, y2, y3, inner, outer, method, o2cut):
    """Build inner and outer masks for the three SOSS orders.

    Parameters
    ----------
    base : array-like(bool)
        Base detector mask.
    y1 : array-like(float)
        Y1 array.
    y2 : array-like(float)
        Y2 array.
    y3 : array-like(float)
        Y3 array.
    inner : int
        Inner trace-mask width.
    outer : int
        Outer trace-mask width.
    method : str
        Correction method.
    o2cut : int
        First detector column included for order 2.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    dimy, dimx = base.shape[-2:]
    m1_in = ref_make_soss_tracemask(y1, inner, dimy, dimx)
    m1_out = ref_make_soss_tracemask(y1, outer, dimy, dimx)
    m2_in = ref_make_soss_tracemask(y2, inner, dimy, dimx)
    m2_out = ref_make_soss_tracemask(y2, outer, dimy, dimx)
    m3 = ref_make_soss_tracemask(y3, inner, dimy, dimx)
    tracemask = m1_in.astype(bool) | m2_in.astype(bool) | m3.astype(bool)
    if method == 'achromatic':
        out1 = base | tracemask
        out2 = out1.copy()
    else:
        window1 = ~(m1_out - m1_in).astype(bool)
        window2 = ~(m2_out - m2_in).astype(bool)
        out1 = base | window1 | tracemask
        out2 = base | window2 | tracemask
        out2[:, :, :o2cut] = True
    return out1, out2, m1_in, m2_in


def ref_oneoverfstep_scale(cube, deepstack, outliers_nan_list, ts2d_list,
                           even_odd_rows, trace_in_list=None,
                           background=None):
    """Calculate scaled SOSS 1/f corrections with the v1 equations.

    Parameters
    ----------
    cube : array-like(float)
        Input observation cube.
    deepstack : array-like(float)
        Deepstack array.
    outliers_nan_list : list
        Outliers nan list.
    ts2d_list : list
        Ts2d list.
    even_odd_rows : bool
        Whether to correct odd and even rows separately.
    trace_in_list : list
        Trace in list.
    background : array-like(float)
        Background array.

    Returns
    -------
    result : np.ndarray(float)
        Calculated reference array.
    """
    cube = cube.copy()
    nint = cube.shape[0]
    if trace_in_list is None:
        trace_in_list = [None] * len(outliers_nan_list)
    for outlier, ts, trace_in in zip(outliers_nan_list, ts2d_list,
                                     trace_in_list):
        for i in range(nint):
            if cube.ndim == 4:
                sub = cube[i] - deepstack * ts[i, None, None, :]
            else:
                sub = cube[i] - deepstack * ts[i, None, :]
            sub = sub * outlier[i, :, :]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                dc = np.zeros_like(sub)
                if even_odd_rows:
                    if cube.ndim == 4:
                        dc[:, ::2] = np.nanmedian(sub[:, ::2], axis=1)[:, None, :]
                        dc[:, 1::2] = np.nanmedian(sub[:, 1::2], axis=1)[:, None, :]
                    else:
                        dc[::2] = np.nanmedian(sub[::2], axis=0)[None, :]
                        dc[1::2] = np.nanmedian(sub[1::2], axis=0)[None, :]
                else:
                    if cube.ndim == 4:
                        dc[:, :, :] = np.nanmedian(sub, axis=1)[:, None, :]
                    else:
                        dc[:, :] = np.nanmedian(sub, axis=0)[None, :]
            dc = np.where(np.isfinite(dc), dc, 0)
            if trace_in is None:
                cube[i] -= dc
            else:
                cube[i] -= dc * trace_in
    if background is not None:
        cube += background
    return cube


def ref_oneoverfstep_nirspec(cube, outliers_nan, method):
    """Calculate NIRSpec 1/f corrections with NumPy.

    Parameters
    ----------
    cube : array-like(float)
        Input observation cube.
    outliers_nan : array-like(float)
        Multiplicative outlier mask with NaNs at excluded pixels.
    method : str
        Correction method.

    Returns
    -------
    result : np.ndarray(float)
        Calculated reference array.
    """
    cube = cube.copy()
    cube_corr = cube.copy()
    nint = cube.shape[0]
    dimy, dimx = cube.shape[-2:]
    for i in range(nint):
        cube[i] = cube[i] * outliers_nan[i, :, :]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            dc = np.zeros_like(cube[i])
            if method == 'median':
                if cube.ndim == 4:
                    dc[:, :, :] = np.nanmedian(cube[i], axis=1)[:, None, :]
                else:
                    dc[:, :] = np.nanmedian(cube[i], axis=0)[None, :]
            elif method == 'slope':
                if cube.ndim == 4:
                    ngroup = cube.shape[1]
                    for xx in range(dimx):
                        xpos = np.arange(dimy)
                        ypos = cube[i][:, :, xx]
                        if np.all(np.isnan(ypos[-1])):
                            continue
                        xxpos = xpos[~np.isnan(ypos[-1])]
                        yypos = ypos[~np.isnan(ypos)].reshape(ngroup,
                                                              len(xxpos))
                        pp = np.polyfit(xxpos, yypos.T, 1)
                        dc[:, :, xx] = np.polyval(
                            pp, np.repeat(xpos[:, np.newaxis], ngroup,
                                          axis=1)).T
                else:
                    for xx in range(dimx):
                        xpos = np.arange(dimy)
                        ypos = cube[i][:, xx]
                        if np.all(np.isnan(ypos)):
                            continue
                        xxpos = xpos[~np.isnan(ypos)]
                        yypos = ypos[~np.isnan(ypos)]
                        pp = np.polyfit(xxpos, yypos, 1)
                        dc[:, xx] = np.polyval(pp, xpos)
        dc = np.where(np.isfinite(dc), dc, 0)
        cube_corr[i] -= dc
    return cube_corr


def nan_map(mask):
    """Convert a boolean mask to a multiplicative NaN mask.

    Parameters
    ----------
    mask : array-like(float)
        Mask array.

    Returns
    -------
    mask : np.ndarray(float)
        Multiplicative mask with NaNs at excluded pixels.
    """
    return np.where(mask, np.nan, 1.).astype(np.float32)


def make_data(seed=0, ndim=4, with_nans=True):
    """Create a synthetic observation with optional non-finite samples.

    Parameters
    ----------
    seed : int
        Random seed.
    ndim : int
        Number of dimensions in the synthetic cube.
    with_nans : bool
        With nans option.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    rng = np.random.default_rng(seed)
    shape = ((NINTS, NGROUPS, DIMY, DIMX) if ndim == 4
             else (NINTS, DIMY, DIMX))
    cube = (100. + 10. * rng.standard_normal(shape)).astype(np.float32)
    if with_nans:
        nan_idx = rng.random(shape) < 0.02
        cube[nan_idx] = np.nan
    # Baseline stack of the first 6 integrations (v1 utils.make_deepstack).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        deepstack = np.nanmedian(cube[:6], axis=0)
    outliers = (rng.random((NINTS, DIMY, DIMX)) < 0.25)
    ts = (1. + 0.01 * rng.standard_normal(NINTS)).astype(np.float32)
    return cube, deepstack, outliers, ts


# Trace masks & deep stack.

@pytest.mark.parametrize('width', [3.0, 4.0, 5.5, 7.0, 40.0])
def test_make_trace_mask_matches_v1(width):
    """Check make trace mask matches v1."""
    rng = np.random.default_rng(0)
    ypos = rng.uniform(1.5, 14.5, DIMX).astype(np.float32)
    ref = ref_make_soss_tracemask(ypos, width, DIMY, DIMX)
    got = oneoverf.make_trace_mask(jnp.asarray(ypos), width, DIMY)
    np.testing.assert_array_equal(np.asarray(got).astype(int), ref)


def test_make_trace_mask_nan_columns_unmasked():
    """Check make trace mask NaN columns unmasked."""
    ypos = np.full(DIMX, 8., dtype=np.float32)
    ypos[10:20] = np.nan
    got = np.asarray(oneoverf.make_trace_mask(jnp.asarray(ypos), 6., DIMY))
    assert got[:, :10].any() and got[:, 20:].any()
    assert not got[:, 10:20].any()


def test_baseline_deepstack_matches_nanmedian():
    """Check baseline deepstack matches nanmedian."""
    cube, deepstack, _, _ = make_data(seed=1)
    bmask = np.zeros(NINTS, dtype=bool)
    bmask[:6] = True
    got = oneoverf.baseline_deepstack(jnp.asarray(cube), jnp.asarray(bmask))
    np.testing.assert_allclose(np.asarray(got), deepstack, rtol=1e-5,
                               equal_nan=True)


def test_build_soss_masks_matches_v1():
    """Check build SOSS masks matches v1."""
    rng = np.random.default_rng(2)
    base = rng.random((NINTS, DIMY, DIMX)) < 0.2
    y1 = rng.uniform(9., 12., DIMX).astype(np.float32)
    y2 = rng.uniform(5., 7., DIMX).astype(np.float32)
    y3 = rng.uniform(2., 3.5, DIMX).astype(np.float32)
    for method in ['achromatic', 'achromatic-window', 'chromatic']:
        ref = ref_build_soss_masks(base.copy(), y1, y2, y3, 3., 7., method, 8)
        got = oneoverf.build_soss_masks(jnp.asarray(base), jnp.asarray(y1),
                                        jnp.asarray(y2), jnp.asarray(y3),
                                        3., 7., method=method, o2_red_cut=8)
        for g, r in zip(got, ref):
            np.testing.assert_array_equal(np.asarray(g).astype(int),
                                          np.asarray(r).astype(int))


def test_minimal_achromatic_mask_matches_full_branch_and_ignores_outer():
    """Check minimal achromatic mask matches full branch and ignores outer."""
    rng = np.random.default_rng(0x135276f)
    base = rng.random((NINTS, DIMY, DIMX)) < 0.2
    y1 = rng.uniform(9., 12., DIMX).astype(np.float32)
    y2 = rng.uniform(5., 7., DIMX).astype(np.float32)
    y3 = rng.uniform(2., 3.5, DIMX).astype(np.float32)
    inner = 5.5

    minimal = oneoverf.build_soss_achromatic_mask(
        jnp.asarray(base), jnp.asarray(y1), jnp.asarray(y2),
        jnp.asarray(y3), inner)
    full_narrow = oneoverf.build_soss_masks(
        jnp.asarray(base), jnp.asarray(y1), jnp.asarray(y2),
        jnp.asarray(y3), inner, 7., method='achromatic')[0]
    full_wide = oneoverf.build_soss_masks(
        jnp.asarray(base), jnp.asarray(y1), jnp.asarray(y2),
        jnp.asarray(y3), inner, 40., method='achromatic')[0]

    np.testing.assert_array_equal(minimal, full_narrow)
    np.testing.assert_array_equal(minimal, full_wide)


# Scale-achromatic.

@pytest.mark.parametrize('even_odd', [True, False])
@pytest.mark.parametrize('ndim', [4, 3])
def test_scale_achromatic_matches_v1(ndim, even_odd):
    """Check scale achromatic matches v1."""
    cube, deepstack, outliers, ts = make_data(seed=3, ndim=ndim)
    ts2d = np.repeat(ts[:, np.newaxis], DIMX, axis=1)
    ref = ref_oneoverfstep_scale(cube, deepstack, [nan_map(outliers)],
                                 [ts2d], even_odd)
    got = oneoverf.oneoverf_scale_achromatic(
        jnp.asarray(cube), jnp.asarray(deepstack), jnp.asarray(outliers),
        jnp.asarray(ts), even_odd_rows=even_odd)
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-5, atol=1e-4,
                               equal_nan=True)


def test_scale_achromatic_background_added_back():
    """Check scale achromatic background added back."""
    cube, deepstack, outliers, ts = make_data(seed=4)
    rng = np.random.default_rng(5)
    background = (2. * rng.random((DIMY, DIMX))).astype(np.float32)
    ts2d = np.repeat(ts[:, np.newaxis], DIMX, axis=1)
    ref = ref_oneoverfstep_scale(cube, deepstack, [nan_map(outliers)],
                                 [ts2d], True, background=background)
    got = oneoverf.oneoverf_scale_achromatic(
        jnp.asarray(cube), jnp.asarray(deepstack), jnp.asarray(outliers),
        jnp.asarray(ts), even_odd_rows=True,
        background=jnp.asarray(background))
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-5, atol=1e-4,
                               equal_nan=True)


def test_synthetic_oneoverf_removed_and_trace_respected():
    """Check synthetic 1/f correction removed and trace respected."""
    rng = np.random.default_rng(6)
    scene = np.full((DIMY, DIMX), 10., dtype=np.float32)
    trace_rows = (slice(6, 10), slice(None))
    scene[trace_rows] = 200.
    deepstack = np.broadcast_to(scene, (NGROUPS, DIMY, DIMX)).copy()
    # Per-(int, group, column) even/odd 1/f offsets.
    a_e = rng.normal(0., 0.5, (NINTS, NGROUPS, DIMX)).astype(np.float32)
    a_o = rng.normal(0., 0.5, (NINTS, NGROUPS, DIMX)).astype(np.float32)
    odd = (np.arange(DIMY) % 2).astype(bool)
    noise = np.where(odd[None, None, :, None], a_o[:, :, None, :],
                     a_e[:, :, None, :])
    # Large in-trace flux variation (transit-like), inside the mask.
    dip = np.linspace(0., -20., NINTS).astype(np.float32)
    signal = np.zeros((NINTS, DIMY, DIMX), dtype=np.float32)
    signal[:, 6:10, :] = dip[:, None, None]
    cube = scene[None, None] + signal[:, None] + noise
    outliers = np.zeros((NINTS, DIMY, DIMX), dtype=bool)
    outliers[:, 5:11, :] = True
    got = oneoverf.oneoverf_scale_achromatic(
        jnp.asarray(cube), jnp.asarray(deepstack), jnp.asarray(outliers),
        jnp.ones(NINTS, dtype=jnp.float32), even_odd_rows=True)
    expected = np.broadcast_to(scene[None, None] + signal[:, None],
                               cube.shape)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-5,
                               atol=1e-4)


# Scale-achromatic-window & scale-chromatic.

def _window_setup(seed, ndim=4):
    """Return window setup."""
    cube, deepstack, base, ts = make_data(seed=seed, ndim=ndim)
    rng = np.random.default_rng(seed + 100)
    y1 = rng.uniform(10., 12., DIMX).astype(np.float32)
    y2 = rng.uniform(4., 6., DIMX).astype(np.float32)
    y3 = rng.uniform(1.5, 2.5, DIMX).astype(np.float32)
    inner, outer = 3., 9.
    ref_masks = ref_build_soss_masks(base.copy(), y1, y2, y3, inner, outer,
                                     'achromatic-window', 8)
    got_masks = oneoverf.build_soss_masks(
        jnp.asarray(base), jnp.asarray(y1), jnp.asarray(y2),
        jnp.asarray(y3), inner, outer, method='achromatic-window',
        o2_red_cut=8)
    return cube, deepstack, ts, ref_masks, got_masks


@pytest.mark.parametrize('ndim', [4, 3])
def test_scale_window_matches_v1(ndim):
    """Check scale window matches v1."""
    cube, deepstack, ts, ref_masks, got_masks = _window_setup(7, ndim=ndim)
    out1, out2, m1_in, m2_in = ref_masks
    ts2d = np.repeat(ts[:, np.newaxis], DIMX, axis=1)
    ref = ref_oneoverfstep_scale(cube, deepstack,
                                 [nan_map(out1), nan_map(out2)],
                                 [ts2d, ts2d], True,
                                 trace_in_list=[m1_in, m2_in])
    o1, o2, t1, t2 = got_masks
    got = oneoverf.oneoverf_scale_window(
        jnp.asarray(cube), jnp.asarray(deepstack), o1, o2, t1, t2,
        jnp.asarray(ts), even_odd_rows=True)
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-5, atol=1e-4,
                               equal_nan=True)


def test_scale_chromatic_matches_v1():
    """Check scale chromatic matches v1."""
    cube, deepstack, ts, ref_masks, got_masks = _window_setup(8)
    out1, out2, m1_in, m2_in = ref_masks
    rng = np.random.default_rng(9)
    ts1 = (1. + 0.01 * rng.standard_normal((NINTS, DIMX))).astype(np.float32)
    ts2 = (1. + 0.01 * rng.standard_normal((NINTS, DIMX))).astype(np.float32)
    ref = ref_oneoverfstep_scale(cube, deepstack,
                                 [nan_map(out1), nan_map(out2)],
                                 [ts1, ts2], True,
                                 trace_in_list=[m1_in, m2_in])
    o1, o2, t1, t2 = got_masks
    got = oneoverf.oneoverf_scale_chromatic(
        jnp.asarray(cube), jnp.asarray(deepstack), o1, o2, t1, t2,
        jnp.asarray(ts1), jnp.asarray(ts2), even_odd_rows=True)
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-5, atol=1e-4,
                               equal_nan=True)


def test_scale_chromatic_requires_2d_timeseries():
    """Check scale chromatic requires 2d timeseries."""
    cube, deepstack, ts, _, got_masks = _window_setup(10)
    o1, o2, t1, t2 = got_masks
    with pytest.raises(ValueError):
        oneoverf.oneoverf_scale_chromatic(
            jnp.asarray(cube), jnp.asarray(deepstack), o1, o2, t1, t2,
            jnp.asarray(ts), jnp.asarray(ts))


# Timeseries estimation.

@pytest.mark.parametrize('ndim', [4, 3])
@pytest.mark.parametrize('smooth', [1, 2, 3])
def test_estimate_timeseries_matches_v1(ndim, smooth):
    """Check estimate timeseries matches v1."""
    cube, deepstack, _, _ = make_data(seed=11, ndim=ndim, with_nans=True)
    region = (2, 14, 5, 25)
    y0, y1_, x0, x1_ = region
    if ndim == 4:
        postage = cube[:, -1, y0:y1_, x0:x1_]
        zero_point = deepstack[-1, y0:y1_, x0:x1_]
    else:
        postage = cube[:, y0:y1_, x0:x1_]
        zero_point = deepstack[y0:y1_, x0:x1_]
    ts = np.nansum(postage, axis=(1, 2)) / np.nansum(zero_point)
    ref = median_filter(ts, smooth)
    got = oneoverf.estimate_timeseries(jnp.asarray(cube),
                                       jnp.asarray(deepstack),
                                       smoothing_scale=smooth, region=region)
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-5, atol=1e-5)


def test_default_smoothing_scale():
    """Check default smoothing scale."""
    assert oneoverf.default_smoothing_scale(10) == 1
    assert oneoverf.default_smoothing_scale(500) == 10
    assert oneoverf.default_smoothing_scale(160) == 3


# NIRSpec median / slope.

def _nirspec_setup(seed, ndim, with_nans):
    """Return NIRSpec setup."""
    cube, _, _, _ = make_data(seed=seed, ndim=ndim, with_nans=with_nans)
    rng = np.random.default_rng(seed + 200)
    base = rng.random((NINTS, DIMY, DIMX)) < 0.15
    base[:, :2, :] = False
    base[:, :, 13] = True
    ypos = rng.uniform(6., 10., DIMX).astype(np.float32)
    width = 4.
    tracemask = ref_make_soss_tracemask(ypos, width, DIMY, DIMX)
    outliers = base | tracemask.astype(bool)[None]
    return cube, base, ypos, width, outliers


@pytest.mark.parametrize('ndim', [4, 3])
def test_nirspec_median_matches_v1(ndim):
    """Check NIRSpec median matches v1."""
    cube, base, ypos, width, outliers = _nirspec_setup(12, ndim, True)
    ref = ref_oneoverfstep_nirspec(cube, nan_map(outliers), 'median')
    got = oneoverf.oneoverf_nirspec(jnp.asarray(cube), jnp.asarray(base),
                                    jnp.asarray(ypos), width,
                                    method='median')
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-5, atol=1e-4,
                               equal_nan=True)


@pytest.mark.parametrize('ndim', [4, 3])
def test_nirspec_slope_matches_v1(ndim):
    """Check NIRSpec slope matches v1."""
    cube, base, ypos, width, outliers = _nirspec_setup(13, ndim, False)
    ref = ref_oneoverfstep_nirspec(cube, nan_map(outliers), 'slope')
    got = oneoverf.oneoverf_nirspec(jnp.asarray(cube), jnp.asarray(base),
                                    jnp.asarray(ypos), width,
                                    method='slope')
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-5, atol=2e-4)


# Solve presence & recompilation checks.

def test_solve_is_implemented():
    # Full v1 parity coverage lives in test_oneoverf_solve_kernel.py.
    """Check solve is implemented."""
    assert callable(oneoverf.oneoverf_solve)
    assert oneoverf.SOLVE_O2_RED_CUT == 1100


def test_width_sweeps_do_not_recompile():
    """Check width sweeps do not recompile."""
    if not hasattr(oneoverf.build_soss_masks, '_cache_size'):
        pytest.skip('jit cache introspection unavailable in this jax')
    rng = np.random.default_rng(14)
    base = jnp.asarray(rng.random((NINTS, DIMY, DIMX)) < 0.2)
    y1 = jnp.asarray(rng.uniform(9., 12., DIMX).astype(np.float32))
    y2 = jnp.asarray(rng.uniform(5., 7., DIMX).astype(np.float32))
    y3 = jnp.asarray(rng.uniform(2., 3.5, DIMX).astype(np.float32))
    oneoverf.build_soss_masks(base, y1, y2, y3, 3., 7.,
                              method='achromatic-window', o2_red_cut=8)
    n0 = oneoverf.build_soss_masks._cache_size()
    oneoverf.build_soss_masks(base, y1, y2, y3, 4.5, 9.,
                              method='achromatic-window', o2_red_cut=8)
    assert oneoverf.build_soss_masks._cache_size() == n0

    cube, _, _, _ = make_data(seed=15, ndim=3, with_nans=False)
    cube_j = jnp.asarray(cube)
    base3 = jnp.asarray(rng.random((NINTS, DIMY, DIMX)) < 0.15)
    ypos = jnp.asarray(rng.uniform(6., 10., DIMX).astype(np.float32))
    oneoverf.oneoverf_nirspec(cube_j, base3, ypos, 4., method='median')
    n1 = oneoverf.oneoverf_nirspec._cache_size()
    oneoverf.oneoverf_nirspec(cube_j, base3, ypos, 6., method='median')
    assert oneoverf.oneoverf_nirspec._cache_size() == n1
