"""Check background subtraction against v1 reference calculations."""

import jax
import numpy as np

from exotedrf.v2.kernels import background


# Numpy references.

def ref_estimate_scale(bkg_ratio):
    """Estimate the background scale using the v1 selection rules.

    Parameters
    ----------
    bkg_ratio : array-like(float)
        Bkg ratio array.

    Returns
    -------
    scale : float
        Background scale estimate.
    """
    finite = np.isfinite(bkg_ratio)
    if not np.any(finite):
        return np.nan
    vals = bkg_ratio[finite]
    q1, q2 = np.nanpercentile(vals, [25, 50])
    use = finite & (bkg_ratio > q1) & (bkg_ratio < q2)
    if not np.any(use):
        use = finite
    return np.nanmedian(bkg_ratio[use])


def ref_background_soss(cube, deep, model, region1, region2=None,
                        differential=False):
    """Subtract the SOSS background using the v1 equations.

    Parameters
    ----------
    cube : array-like(float)
        Input observation cube.
    deep : array-like(float)
        Deep array.
    model : array-like(float)
        Model array.
    region1 : None, tuple[int]
        Detector bounds for background scaling.
    region2 : None, tuple[int]
        Detector bounds for background scaling.
    differential : bool
        Differential option.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    rl, rh, cl, ch = region1
    shift = 0.0
    scale1 = -1000.
    while scale1 < 0:
        ratio = (deep[rl:rh, cl:ch] + shift) / model[rl:rh, cl:ch]
        scale1 = ref_estimate_scale(ratio)
        if not np.isfinite(scale1):
            scale1 = 0.0
            break
        if scale1 < 0:
            mm = np.nanmedian(model[rl:rh, cl:ch])
            if not np.isfinite(mm) or mm == 0:
                scale1 = 0.0
                break
            shift -= scale1 * mm

    if differential:
        rl2, rh2, cl2, ch2 = region2
        ratio = (deep[rl2:rh2, cl2:ch2] + shift) / model[rl2:rh2, cl2:ch2]
        scale2 = ref_estimate_scale(ratio)
        if not np.isfinite(scale2):
            scale2 = 0.0
        elif scale2 < 0:
            scale2 = 0.
        grad = np.gradient(model, axis=1)
        step_pos = np.argmax(grad[:, 10:-10], axis=1) + 10 - 4
        ms = np.zeros_like(model)
        for j in range(model.shape[0]):
            ms[j, :step_pos[j]] = model[j, :step_pos[j]] * scale1 - shift
            ms[j, step_pos[j]:] = model[j, step_pos[j]:] * scale2 - shift
    else:
        scale2 = scale1
        ms = model * scale1 - shift
    return cube - ms[None], ms, scale1, scale2, shift


def ref_background_miri(cube, trace_width, background_width, method,
                        trace_center=36):
    """Subtract the MIRI background using the v1 equations.

    Parameters
    ----------
    cube : array-like(float)
        Input observation cube.
    trace_width : int, float
        Width of the trace exclusion region.
    background_width : int, float
        Width of the background region.
    method : str
        Correction method.
    trace_center : float
        Trace center in detector rows.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    cube = np.asarray(cube, np.float64)
    nint, dimy, dimx = cube.shape
    thw = int(trace_width / 2)
    bw = int(background_width)
    c = trace_center
    if method == 'median':
        bkg_cube = np.concatenate(
            [cube[:, :, (c - thw - bw):(c - thw + 1)],
             cube[:, :, (c + thw):(c + thw + bw + 1)]], axis=2)
        bkg = np.nanmedian(bkg_cube, axis=2)
        bkg = np.broadcast_to(bkg[:, :, None], cube.shape)
    else:
        bkg_cube = cube.copy()
        bkg_cube[:, :, :(c - thw - bw + 1)] = np.nan
        bkg_cube[:, :, (c + thw + bw):] = np.nan
        bkg_cube[:, :, (c - thw):(c + thw + 1)] = np.nan
        bkg = np.zeros_like(cube)
        xx = np.arange(dimx)
        for i in range(nint):
            for j in range(dimy):
                y = bkg_cube[i, j]
                ii = np.isfinite(y)
                xf, yf = xx[ii], y[ii]
                mask = np.ones_like(xf, bool)
                for _ in range(5):
                    param = np.polyfit(xf[mask], yf[mask], 1)
                    res = yf - np.polyval(param, xf)
                    mask = np.abs(res) <= 3. * np.std(res)
                param = np.polyfit(xf[mask], yf[mask], 1)
                bkg[i, j] = np.polyval(param, xx)
    return cube - bkg, bkg


# SOSS tests.

def _soss_model(dimy=16, dimx=32, jump_col=17):
    """Return SOSS model."""
    yy, xx = np.mgrid[:dimy, :dimx]
    model = 1.0 + 0.02 * yy + np.where(xx >= jump_col, 1.0, 0.0)
    return model.astype(np.float32)


REGION1 = (2, 14, 3, 15)
REGION2 = (2, 14, 20, 32)


def test_soss_known_scale_recovered():
    """Check SOSS known scale recovered."""
    rng = np.random.default_rng(1)
    model = _soss_model()
    deep = (1.7 * model).astype(np.float32)
    cube = (deep[None] + rng.uniform(-.01, .01, (6,) + deep.shape)
            ).astype(np.float32)

    out = background.background_soss(cube, deep, model, region1=REGION1)
    cube_corr, ms, s1, s2, shift = [np.asarray(o) for o in out]
    np.testing.assert_allclose(s1, 1.7, rtol=1e-5)
    np.testing.assert_allclose(s2, 1.7, rtol=1e-5)
    np.testing.assert_allclose(shift, 0.0, atol=1e-7)
    np.testing.assert_allclose(cube_corr, cube - 1.7 * model, rtol=1e-5,
                               atol=1e-5)

    ref = ref_background_soss(cube, deep, model, REGION1)
    np.testing.assert_allclose(ms, ref[1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cube_corr, ref[0], rtol=1e-5, atol=1e-5)


def test_soss_noisy_scale_second_quartile():
    """Check SOSS noisy scale second quartile."""
    rng = np.random.default_rng(2)
    model = _soss_model()
    deep = (1.7 * model + rng.uniform(-.05, .05, model.shape)
            ).astype(np.float32)
    cube = np.repeat(deep[None], 5, axis=0).astype(np.float32)

    out = background.background_soss(cube, deep, model, region1=REGION1)
    cube_corr, ms, s1, s2, shift = [np.asarray(o) for o in out]
    ref_corr, ref_ms, ref_s1, _, ref_shift = ref_background_soss(
        cube, deep, model, REGION1)
    np.testing.assert_allclose(s1, ref_s1, rtol=1e-5)
    np.testing.assert_allclose(shift, ref_shift, atol=1e-7)
    np.testing.assert_allclose(ms, ref_ms, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cube_corr, ref_corr, rtol=1e-5, atol=1e-5)
    # Scale close to the injected one.
    np.testing.assert_allclose(s1, 1.7, rtol=2e-2)


def test_soss_negative_scale_shift_loop():
    """Check SOSS negative scale shift loop."""
    model = np.full((16, 32), 2.0, np.float32)
    deep = (np.float32(-0.4) * model).astype(np.float32)
    cube = np.zeros((3, 16, 32), np.float32)

    out = background.background_soss(cube, deep, model, region1=REGION1)
    cube_corr, ms, s1, s2, shift = [np.asarray(o) for o in out]
    ref_corr, ref_ms, ref_s1, _, ref_shift = ref_background_soss(
        cube, deep, model, REGION1)
    np.testing.assert_allclose(s1, ref_s1, atol=1e-7)
    np.testing.assert_allclose(shift, ref_shift, rtol=1e-6)
    assert s1 == 0.0
    assert shift > 0.7
    np.testing.assert_allclose(ms, ref_ms, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cube_corr, ref_corr, rtol=1e-5, atol=1e-6)


def test_soss_differential_two_zone():
    """Check SOSS differential two zone."""
    rng = np.random.default_rng(3)
    jump_col = 17
    model = _soss_model(jump_col=jump_col)
    deep = model.copy()
    deep[:, :jump_col] *= 1.5
    deep[:, jump_col:] *= 2.0
    deep = deep.astype(np.float32)
    cube = (deep[None] + rng.uniform(-.01, .01, (4,) + deep.shape)
            ).astype(np.float32)

    out = background.background_soss(cube, deep, model, region1=REGION1,
                                     region2=REGION2, differential=True)
    cube_corr, ms, s1, s2, shift = [np.asarray(o) for o in out]
    ref_corr, ref_ms, ref_s1, ref_s2, ref_shift = ref_background_soss(
        cube, deep, model, REGION1, REGION2, differential=True)

    np.testing.assert_allclose(s1, ref_s1, rtol=1e-5)
    np.testing.assert_allclose(s2, ref_s2, rtol=1e-5)
    np.testing.assert_allclose(s1, 1.5, rtol=1e-5)
    np.testing.assert_allclose(s2, 2.0, rtol=1e-5)
    np.testing.assert_allclose(ms, ref_ms, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cube_corr, ref_corr, rtol=1e-5, atol=1e-5)
    resid = cube_corr - (cube - np.where(
        np.arange(32)[None, None] < jump_col, 1.5 * model, 2.0 * model))
    assert np.abs(resid[:, :, :jump_col - 8]).max() < 1e-5
    assert np.abs(resid[:, :, jump_col:]).max() < 1e-5


def test_soss_float64_while_loop_preserves_carry_dtype():
    """Check SOSS float64 while loop preserves carry dtype."""
    old = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        model = _soss_model().astype(np.float64)
        deep = 1.25 * model
        cube = np.repeat(deep[None], 2, axis=0)
        corrected, scaled, scale, _, _ = background.background_soss(
            cube, deep, model, region1=REGION1)
        assert np.asarray(corrected).dtype == np.float64
        assert np.asarray(scaled).dtype == np.float64
        np.testing.assert_allclose(np.asarray(scale), 1.25, rtol=1e-12)
    finally:
        jax.config.update('jax_enable_x64', old)


# MIRI tests.

def _miri_dataset(seed=4):
    """Return MIRI dataset."""
    rng = np.random.default_rng(seed)
    nints, dimy, dimx = 8, 16, 32
    ii, jj, xx = np.mgrid[:nints, :dimy, :dimx].astype(np.float64)
    # Per-(integration, row) linear background.
    a = 2. + 0.05 * ii + 0.1 * jj
    b = 0.03 + 0.002 * jj
    bkg_true = a + b * xx
    trace = np.where(np.abs(xx - 16) <= 2,
                     50. * np.exp(-((xx - 16.) / 1.5) ** 2), 0.)
    noise = rng.uniform(-.05, .05, (nints, dimy, dimx))
    cube = (bkg_true + trace + noise).astype(np.float32)
    return cube, bkg_true, trace


def test_miri_median():
    """Check MIRI median."""
    cube, bkg_true, trace = _miri_dataset()
    corr, bkg = background.background_miri(
        cube, trace_width=6., background_width=5., method='median',
        trace_center=16)
    corr, bkg = np.asarray(corr), np.asarray(bkg)
    ref_corr, ref_bkg = ref_background_miri(cube, 6, 5, 'median', 16)
    np.testing.assert_allclose(bkg, ref_bkg, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(corr, ref_corr, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(bkg[:, :, 0], bkg_true[:, :, 16], atol=0.5)


def test_miri_median_known_level():
    """Flat background: the median recovers the injected level exactly."""
    nints, dimy, dimx = 6, 16, 32
    ii, jj, xx = np.mgrid[:nints, :dimy, :dimx].astype(np.float64)
    level = 3. + 0.1 * ii + 0.2 * jj
    trace = np.where(np.abs(xx - 16) <= 2, 40., 0.)
    cube = (level + trace).astype(np.float32)
    corr, bkg = background.background_miri(
        cube, trace_width=6., background_width=5., method='median',
        trace_center=16)
    np.testing.assert_allclose(np.asarray(bkg), level, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(corr), trace, atol=1e-4)


def test_miri_slope():
    """Check MIRI slope."""
    cube, bkg_true, trace = _miri_dataset()
    corr, bkg = background.background_miri(
        cube, trace_width=6., background_width=5., method='slope',
        trace_center=16)
    corr, bkg = np.asarray(corr), np.asarray(bkg)
    ref_corr, ref_bkg = ref_background_miri(cube, 6, 5, 'slope', 16)
    np.testing.assert_allclose(bkg, ref_bkg, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(corr, ref_corr, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(bkg, bkg_true, atol=0.5)
    # And the trace region is background-subtracted but keeps the trace.
    np.testing.assert_allclose(corr[:, :, 14:19], trace[:, :, 14:19],
                               atol=0.5)


def test_miri_traced_widths_no_recompile():
    """Check MIRI traced widths no recompile."""
    cube = _miri_dataset()[0]
    f = background.background_miri
    _ = f(cube, trace_width=6., background_width=5., method='median',
          trace_center=16)
    misses_before = f._cache_size()
    _ = f(cube, trace_width=8., background_width=4., method='median',
          trace_center=16)
    assert f._cache_size() == misses_before
