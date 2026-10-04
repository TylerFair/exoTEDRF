"""Tests for exotedrf.v2.kernels.detector."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from exotedrf.v2 import core
from exotedrf.v2.kernels import detector

warnings.filterwarnings('ignore', message='All-NaN')
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='invalid value')
warnings.filterwarnings('ignore', message='Degrees of freedom')

SAT, DNU, FLOOR = 2, 1, 64
HOT, WARM = 2048, 4096


def _cube(rng, shape=(3, 4, 8, 12), loc=100., scale=30., nan_frac=0.02):
    """Return cube."""
    data = rng.normal(loc, scale, size=shape).astype(np.float32)
    nans = rng.random(shape) < nan_frac
    data[nans] = np.nan
    return data


# Flag_saturated_pixels.

def ref_flag_saturated(data, groupdq, full_well, thresh_pct, fn):
    """Calculate saturation flags and neighbor dilation with NumPy.

    Parameters
    ----------
    data : array-like(float)
        Data array.
    groupdq : array-like(int)
        Groupdq array.
    full_well : float
        Detector full-well level.
    thresh_pct : float
        Saturation threshold as a percentage of full well.
    fn : callable
        Neighbor-dilation filter.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    sat_adu = full_well * thresh_pct / 100.
    inds = data >= sat_adu
    nints, ngroups, ydim, xdim = data.shape
    inds_expanded = np.copy(inds)
    ii = np.where(inds)
    for pix in range(len(ii[0])):
        i, g, y, x = ii[0][pix], ii[1][pix], ii[2][pix], ii[3][pix]
        y0, y1 = max(0, y - fn), min(ydim, y + fn + 1)
        x0, x1 = max(0, x - fn), min(xdim, x + fn + 1)
        inds_expanded[i, g, y0:y1, x0:x1] = True
    gdq = np.bitwise_or(groupdq, 2 * inds_expanded.astype(np.uint8))
    gdq = np.bitwise_or(gdq, 64 * (data < 0).astype(np.uint8))
    return gdq, inds_expanded


def test_flag_saturated_matches_v1_reference():
    """Check flag saturated matches v1 reference."""
    rng = np.random.default_rng(0)
    data = _cube(rng, loc=40000., scale=15000.)
    data[0, 1, 0, 0] = 99999.
    data[1, 2, -1, 5] = 99999.
    data[2, 0, 3, -1] = 99999.
    data[0, 0, 2, 2] = -50.
    groupdq = rng.integers(0, 4, size=data.shape).astype(np.uint8)
    full_well, thresh = 62070., 80.

    gdq_ref, mask_ref = ref_flag_saturated(data, groupdq, full_well, thresh, 1)
    gdq, mask = detector.flag_saturated_pixels(
        data, groupdq, full_well, saturation_threshold=thresh,
        flag_neighbours=1)
    assert np.array_equal(np.asarray(gdq), gdq_ref)
    assert np.array_equal(np.asarray(mask), mask_ref)


def test_flag_saturated_neighbour_box_2():
    """Check flag saturated neighbour box 2."""
    rng = np.random.default_rng(1)
    data = _cube(rng, shape=(2, 3, 10, 14), loc=1000., scale=100.)
    data[1, 1, 5, 7] = 1e6
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    gdq_ref, mask_ref = ref_flag_saturated(data, groupdq, 62070., 80., 2)
    gdq, mask = detector.flag_saturated_pixels(
        data, groupdq, 62070., saturation_threshold=80., flag_neighbours=2)
    assert np.array_equal(np.asarray(gdq), gdq_ref)
    assert np.array_equal(np.asarray(mask), mask_ref)


def test_flag_saturated_zero_neighbours_is_identity_dilation():
    """Check flag saturated zero neighbours is identity dilation."""
    data = np.zeros((1, 2, 4, 5), np.float32)
    data[0, 1, 2, 3] = 101.
    dq = np.zeros_like(data, np.uint8)
    out, mask = detector.flag_saturated_pixels(
        data, dq, 100., saturation_threshold=100., flag_neighbours=0)
    expected = data >= 100.
    assert np.asarray(mask).shape == data.shape
    assert np.array_equal(np.asarray(mask), expected)
    assert np.array_equal((np.asarray(out) & SAT) != 0, expected)


# Flag_hot_pixels.

def ref_flag_hot(pixeldq, deepframe, thresh, box_half, n_edge_cols):
    """Calculate hot-pixel flags with the v1 selection rules.

    Parameters
    ----------
    pixeldq : array-like(int)
        Pixeldq array.
    deepframe : array-like(float)
        Deepframe array.
    thresh : float
        Detection threshold.
    box_half : int
        Half-width of the spatial comparison box.
    n_edge_cols : int
        Number of edge columns to exclude.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    dimy, dimx = deepframe.shape
    bh = box_half
    padded = np.full((dimy + 2 * bh, dimx + 2 * bh), np.nan, dtype=np.float32)
    padded[bh:bh + dimy, bh:bh + dimx] = deepframe
    med = np.zeros_like(deepframe)
    std = np.zeros_like(deepframe)
    for j in range(dimy):
        for i in range(dimx):
            win = padded[j:j + 2 * bh + 1, i:i + 2 * bh + 1].copy()
            med[j, i] = np.nanmedian(win)
            win[bh, bh] = np.nan
            std[j, i] = np.nanstd(win.astype(np.float64))
    global_med = np.nanmedian(deepframe)
    already = (pixeldq & (HOT | WARM)) != 0
    cols = np.arange(dimx)
    col_ok = (cols >= n_edge_cols) & (cols < dimx - n_edge_cols)
    hot = (np.abs(deepframe - med) >= thresh * std) \
        & (deepframe > global_med) & ~already & col_ok[None, :]
    pdq = np.bitwise_or(pixeldq, np.where(hot, np.uint32(HOT | DNU),
                                          np.uint32(0)))
    return pdq, hot


def test_flag_hot_matches_reference():
    """Check flag hot matches reference."""
    rng = np.random.default_rng(2)
    deep = rng.normal(50., 5., size=(16, 32)).astype(np.float32)
    deep[rng.random(deep.shape) < 0.03] = np.nan
    # A few genuinely hot pixels.
    deep[7, 10] = 1000.
    deep[3, 20] = 800.
    deep[12, 25] = 900.
    pixeldq = np.zeros(deep.shape, dtype=np.uint32)
    pixeldq[3, 20] = HOT
    pixeldq[5, 5] = WARM

    pdq_ref, hot_ref = ref_flag_hot(pixeldq, deep, 10., 2, 4)
    pdq, hot = detector.flag_hot_pixels(pixeldq, deep, thresh=10.,
                                        box_half=2, n_edge_cols=4)
    assert np.array_equal(np.asarray(pdq), pdq_ref)
    assert np.array_equal(np.asarray(hot), hot_ref)


def test_flag_hot_functional():
    """Check flag hot functional."""
    rng = np.random.default_rng(3)
    deep = rng.normal(50., 2., size=(16, 32)).astype(np.float32)
    deep[7, 10] = 1000.
    deep[9, 2] = 1000.
    deep[11, 15] = -1000.
    pixeldq = np.zeros(deep.shape, dtype=np.uint32)
    pdq, hot = detector.flag_hot_pixels(pixeldq, deep, thresh=10.,
                                        box_half=2, n_edge_cols=4)
    hot = np.asarray(hot)
    assert hot[7, 10]
    assert not hot[9, 2]
    assert not hot[11, 15]
    assert np.asarray(pdq)[7, 10] == HOT | DNU


# Superbias.

def test_integral_nonlinearity_matches_v1_fourier_formula():
    """Check integral nonlinearity matches v1 fourier formula."""
    rng = np.random.default_rng(44)
    data = rng.uniform(0, 5000, (2, 3, 4, 5)).astype(np.float32)
    periods = np.asarray([1024 / 3, 512, 1024], np.float32)
    theta = np.asarray([1e-3, -2e-3, 3e-4, 4e-4, -8e-4, 2e-4],
                       np.float32)
    correction = np.zeros_like(data)
    for k, period in enumerate(periods):
        phase = 2 * np.pi / period * data
        correction += theta[2 * k] * np.sin(phase)
        correction += theta[2 * k + 1] * np.cos(phase)
    expected = data / (1 + correction)
    actual = detector.integral_nonlinearity_correct(data, theta, periods)
    assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-4)

def test_make_custom_superbias():
    """Check make custom superbias."""
    rng = np.random.default_rng(4)
    data = _cube(rng, shape=(6, 3, 8, 12))
    baseline = np.array([1, 1, 0, 0, 1, 1], dtype=bool)
    ref = np.nanmedian(data[baseline, 0], axis=0)
    out = detector.make_custom_superbias(data, baseline)
    assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)
    # All-True mask reproduces v1 exactly.
    out_all = detector.make_custom_superbias(data, np.ones(6, bool))
    assert_allclose(np.asarray(out_all), np.nanmedian(data[:, 0], axis=0),
                    rtol=1e-5, atol=1e-5)


def test_subtract_superbias():
    """Check subtract superbias."""
    rng = np.random.default_rng(5)
    data = _cube(rng)
    sb = rng.normal(90., 10., size=data.shape[2:]).astype(np.float32)
    sb[1, 2] = np.nan
    out = detector.subtract_superbias(data, sb)
    safe = np.where(np.isfinite(sb), sb, 0.)
    assert_allclose(np.asarray(out), data - safe, rtol=1e-5, atol=1e-4)


def test_subtract_superbias_rescale_matches_reference():
    """Check subtract superbias rescale matches reference."""
    rng = np.random.default_rng(6)
    data = _cube(rng, shape=(4, 3, 8, 12), loc=100., scale=10.)
    sb = rng.normal(100., 5., size=data.shape[2:]).astype(np.float32)
    use = rng.random(data.shape[2:]) < 0.7
    ratio = np.where(use[None], data[:, 0] / sb[None], np.nan)
    scale_ref = np.nanmedian(ratio, axis=(1, 2))
    out_ref = data - scale_ref[:, None, None, None] * sb[None, None]
    out, scale = detector.subtract_superbias_rescale(data, sb, use)
    assert_allclose(np.asarray(scale), scale_ref, rtol=1e-5, atol=1e-6)
    assert_allclose(np.asarray(out), out_ref, rtol=1e-5, atol=1e-3)


def test_subtract_superbias_rescale_exact_scale():
    # If group 0 is exactly s_i * superbias, the recovered scales are s_i.
    """Check subtract superbias rescale exact scale."""
    rng = np.random.default_rng(7)
    sb = rng.normal(100., 5., size=(8, 12)).astype(np.float32)
    scales = np.array([0.9, 1.0, 1.1], dtype=np.float32)
    data = np.zeros((3, 2, 8, 12), dtype=np.float32)
    data[:, 0] = scales[:, None, None] * sb[None]
    data[:, 1] = data[:, 0] + 50.
    use = np.ones(sb.shape, bool)
    out, scale = detector.subtract_superbias_rescale(data, sb, use)
    assert_allclose(np.asarray(scale), scales, rtol=1e-5)
    assert_allclose(np.asarray(out)[:, 0], np.zeros((3,) + sb.shape),
                    atol=1e-2)


# Refpix_correct.

def _refpix_mean(values):
    """Compute the clipped mean used by JWST's reference-pixel step."""
    from scipy.stats import sigmaclip

    if np.std(values, dtype=np.float64) == 0:
        return values.mean(dtype=np.float64)
    return sigmaclip(values, 3, 3)[0].mean()


def ref_refpix(data, nref_top, nref_bottom, odd_even):
    """Subtract reference-pixel offsets with NumPy.

    Parameters
    ----------
    data : array-like(float)
        Data array.
    nref_top : int
        Number of top reference rows.
    nref_bottom : int
        Number of bottom reference rows.
    odd_even : bool
        Whether to correct odd and even rows separately.

    Returns
    -------
    result : np.ndarray(float)
        Calculated reference array.
    """
    out = data.astype(np.float64).copy()
    nints, ngroups, dimy, dimx = data.shape
    for i in range(nints):
        for g in range(ngroups):
            parts = []
            if nref_bottom > 0:
                parts.append(data[i, g, :nref_bottom])
            if nref_top > 0:
                parts.append(data[i, g, dimy - nref_top:])
            ref = np.concatenate(parts, axis=0)
            if odd_even:
                out[i, g, :, 0::2] -= _refpix_mean(ref[:, 0::2])
                out[i, g, :, 1::2] -= _refpix_mean(ref[:, 1::2])
            else:
                out[i, g] -= _refpix_mean(ref)
    return out.astype(np.float32)


def test_refpix_matches_reference():
    """Check reference-pixel correction matches reference."""
    rng = np.random.default_rng(8)
    data = _cube(rng, shape=(3, 4, 12, 16), loc=10., scale=5.)
    for oe in (True, False):
        ref = ref_refpix(data, 4, 0, oe)
        out = detector.refpix_correct(data, nref_top=4, nref_bottom=0,
                                      odd_even_columns=oe)
        assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-4)
    # Bottom rows variant.
    ref = ref_refpix(data, 0, 4, True)
    out = detector.refpix_correct(data, nref_top=0, nref_bottom=4)
    assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-4)


def test_refpix_removes_odd_even_offsets():
    """Check reference-pixel correction removes odd even offsets."""
    rng = np.random.default_rng(9)
    nints, ngroups, dimy, dimx = 2, 3, 12, 16
    signal = rng.normal(20., 3., size=(nints, ngroups, dimy, dimx)
                        ).astype(np.float32)
    signal[:, :, dimy - 4:, :] = 0.
    even_off, odd_off = 7.5, -3.25
    parity = (np.arange(dimx) % 2 == 0)
    data = signal + np.where(parity, even_off, odd_off).astype(np.float32)
    out = np.asarray(detector.refpix_correct(data, nref_top=4))
    assert_allclose(out, signal, rtol=1e-4, atol=1e-3)


def ref_refpix_dq(data, dq, nref_top, nref_bottom, odd_even):
    """JWST reference mean after DQ exclusion and iterative clipping.

    Parameters
    ----------
    data : array-like(float)
        Data array.
    dq : array-like(int)
        Dq array.
    nref_top : int
        Number of top reference rows.
    nref_bottom : int
        Number of bottom reference rows.
    odd_even : bool
        Whether to correct odd and even rows separately.

    Returns
    -------
    result : np.ndarray(float)
        Calculated reference array.
    """
    out = data.astype(np.float64).copy()
    nints, ngroups, dimy, dimx = data.shape
    for i in range(nints):
        for g in range(ngroups):
            parts = []
            dqparts = []
            if nref_bottom > 0:
                parts.append(data[i, g, :nref_bottom])
                dqparts.append(dq[:nref_bottom])
            if nref_top > 0:
                parts.append(data[i, g, dimy - nref_top:])
                dqparts.append(dq[dimy - nref_top:])
            ref = np.concatenate(parts, axis=0)
            refdq = np.concatenate(dqparts, axis=0)
            good = (refdq & DNU) == 0
            if odd_even:
                out[i, g, :, 0::2] -= _refpix_mean(ref[:, 0::2][good[:, 0::2]])
                out[i, g, :, 1::2] -= _refpix_mean(ref[:, 1::2][good[:, 1::2]])
            else:
                out[i, g] -= _refpix_mean(ref[good])
    return out.astype(np.float32)


@pytest.mark.parametrize('odd_even', [False, True])
def test_refpix_clips_reference_outliers_until_converged(odd_even):
    """Check reference-pixel correction clips reference outliers until converged."""
    rng = np.random.default_rng(41)
    data = rng.normal(500, 2, (2, 3, 12, 128)).astype(np.float32)
    data[..., -4, :20] += 15
    data[..., -3, :4] += 1000
    dq = np.zeros(data.shape[-2:], np.uint32)
    dq[-1, 4] = DNU
    expected = ref_refpix_dq(data, dq, 4, 0, odd_even)
    actual = np.asarray(detector.refpix_correct(
        data, dq, odd_even_columns=odd_even))
    assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)


def test_niriss_refpix_uses_flagged_sides_and_detector_column_parity():
    """Check NIRISS reference-pixel correction uses flagged sides and detector column parity."""
    reference_pixels = pytest.importorskip('jwst.refpix.reference_pixels')
    dataset = object.__new__(reference_pixels.Dataset)
    dataset.siglimit = 3
    rng = np.random.default_rng(19)
    data = rng.normal(50, 2, (2, 3, 16, 128)).astype(np.float32)
    data[..., 0::2, :] += 10
    data[..., -3, :4] += 1000
    dq = np.zeros(data.shape[-2:], np.uint32)
    dq[-4:] = np.uint32(1 << 31)
    dq[:, :4] = np.uint32(1 << 31)
    dq[:, -4:] = np.uint32(1 << 31)
    dq[-1, 5] |= DNU
    reference = data.copy()
    good = ((dq & np.uint32(1 << 31)) != 0) & ((dq & DNU) == 0)
    for i in range(data.shape[0]):
        for g in range(data.shape[1]):
            for parity in (0, 1):
                values = data[i, g, parity::2][good[parity::2]]
                offset = dataset.sigma_clip(values, np.zeros(values.shape, np.uint32))
                reference[i, g, parity::2] -= offset
    actual = np.asarray(detector.refpix_correct(
        data, dq, nref_side=4, detector_column_axis=-2))
    assert_allclose(actual, reference, rtol=1e-5, atol=1e-4)


def test_refpix_pixeldq_zero_matches_none():
    """Check reference-pixel correction PIXELDQ zero matches none."""
    rng = np.random.default_rng(11)
    data = _cube(rng, shape=(2, 3, 12, 16), loc=10., scale=5.)
    dimy, dimx = data.shape[-2:]
    zero_dq = np.zeros((dimy, dimx), np.uint32)
    out_none = np.asarray(detector.refpix_correct(data, nref_top=4))
    out_zero = np.asarray(detector.refpix_correct(data, zero_dq,
                                                   nref_top=4))
    np.testing.assert_array_equal(out_none, out_zero)


def test_refpix_excludes_do_not_use_reference_pixels():
    """Check reference-pixel correction excludes do not use reference pixels."""
    rng = np.random.default_rng(12)
    nints, ngroups, dimy, dimx = 2, 2, 12, 16
    data = _cube(rng, shape=(nints, ngroups, dimy, dimx), loc=10.,
                scale=2., nan_frac=0.)
    dq = np.zeros((dimy, dimx), np.uint32)
    dq[dimy - 4, 2] = DNU
    dq[dimy - 1, 5] = DNU
    ref = ref_refpix_dq(data, dq, 4, 0, True)
    out = np.asarray(detector.refpix_correct(data, dq, nref_top=4))
    assert_allclose(out, ref, rtol=1e-5, atol=1e-4)
    unmasked = np.asarray(detector.refpix_correct(data, nref_top=4))
    assert not np.allclose(out, unmasked, atol=1e-6)


# Linearity_correct.

def ref_linearity(data, groupdq, coeffs, no_lin):
    """Apply the reference linearity polynomial to usable groups.

    Parameters
    ----------
    data : array-like(float)
        Data array.
    groupdq : array-like(int)
        Groupdq array.
    coeffs : array-like(float)
        Coeffs array.
    no_lin : array-like(bool)
        Pixels excluded from linearity correction.

    Returns
    -------
    result : np.ndarray(float)
        Calculated reference array.
    """
    acc = np.broadcast_to(coeffs[-1], data.shape).astype(np.float32).copy()
    for k in range(coeffs.shape[0] - 2, -1, -1):
        acc = acc * data + coeffs[k]
    skip = (groupdq & SAT) != 0
    if no_lin is not None:
        skip = skip | no_lin[None, None]
    return np.where(skip, data, acc)


def test_linearity_matches_reference():
    """Check linearity matches reference."""
    rng = np.random.default_rng(10)
    data = _cube(rng, loc=5000., scale=2000.)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    groupdq[rng.random(data.shape) < 0.05] = SAT
    groupdq[rng.random(data.shape) < 0.03] = DNU
    coeffs = np.stack([
        rng.normal(0., 1., data.shape[2:]),
        rng.normal(1., 0.01, data.shape[2:]),
        rng.normal(0., 1e-6, data.shape[2:]),
        rng.normal(0., 1e-11, data.shape[2:]),
    ]).astype(np.float32)
    no_lin = rng.random(data.shape[2:]) < 0.05

    ref = ref_linearity(data, groupdq, coeffs, no_lin)
    out = detector.linearity_correct(data, groupdq, coeffs, no_lin)
    assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-2)
    outn = np.asarray(out)
    satm = (groupdq & SAT) != 0
    assert_allclose(outn[satm], data[satm], rtol=0, atol=0)
    assert_allclose(outn[:, :, no_lin], data[:, :, no_lin], rtol=0, atol=0)
    # No_lin_mask=None variant.
    ref2 = ref_linearity(data, groupdq, coeffs, None)
    out2 = detector.linearity_correct(data, groupdq, coeffs)
    assert_allclose(np.asarray(out2), ref2, rtol=1e-5, atol=1e-2)


def test_prepare_linearity_reference_flags_only_official_skip_cases():
    """Check prepare linearity reference flags only official skip cases."""
    no_lin_bit = np.uint32(1 << 20)
    coeffs = np.zeros((3, 2, 3), np.float32)
    coeffs[1] = 1.
    coeffs[2, 0, 1] = np.nan
    coeffs[1, 0, 2] = 0.
    ref_dq = np.zeros((2, 3), np.uint32)
    ref_dq[1, 0] = HOT
    ref_dq[1, 1] = no_lin_bit

    safe, skip, propagated = detector.prepare_linearity_reference(
        coeffs, ref_dq)
    assert np.isfinite(np.asarray(safe)).all()
    expected_skip = np.zeros((2, 3), bool)
    expected_skip[0, 1] = expected_skip[0, 2] = True
    expected_skip[1, 1] = True
    np.testing.assert_array_equal(np.asarray(skip), expected_skip)
    expected_dq = ref_dq.copy()
    expected_dq[0, 1] |= no_lin_bit
    expected_dq[0, 2] |= no_lin_bit
    np.testing.assert_array_equal(np.asarray(propagated), expected_dq)


# Subtract_dark & gain_scale.

def test_subtract_dark():
    """Check subtract dark."""
    rng = np.random.default_rng(11)
    data = _cube(rng)
    dark = rng.normal(1., 0.3, size=data.shape[1:]).astype(np.float32)
    dark[1, 2, 3] = np.nan
    out = detector.subtract_dark(data, dark)
    safe = np.where(np.isfinite(dark), dark, 0.)
    assert_allclose(np.asarray(out), data - safe[None], rtol=1e-5, atol=1e-4)


def test_apply_flat_field_matches_reference_dq_and_uncertainty():
    """Check apply flat field matches reference DQ and uncertainty."""
    no_flat = np.uint32(1 << 18)
    data = np.arange(12, dtype=np.float32).reshape(2, 2, 3) + 10.
    err = np.full_like(data, 2.)
    dq = np.zeros_like(data, np.uint32)
    dq[0, 0, 0] = SAT
    flat = np.asarray([[2., 0., np.nan], [4., 5., 2.]], np.float32)
    flat_err = np.full(flat.shape, .1, np.float32)
    flat_err[0, 2] = np.inf
    flat_dq = np.zeros(flat.shape, np.uint32)
    flat_dq[1, 1] = DNU
    flat_dq[1, 2] = no_flat

    actual = detector.apply_flat_field(
        data, err, dq, flat, flat_err, flat_dq)

    ref_dq = flat_dq.copy()
    invalid = ~np.isfinite(flat) | (flat == 0)
    ref_dq[invalid] |= np.uint32(DNU) | no_flat
    ref_dq[(ref_dq & no_flat) != 0] |= np.uint32(DNU)
    bad = (ref_dq & DNU) != 0
    safe_flat = np.where(bad, 1., flat)
    expected_data = data / safe_flat
    expected_err = np.sqrt(
        (err / safe_flat) ** 2
        + expected_data ** 2 / safe_flat ** 2 * flat_err ** 2)
    expected_dq = dq | ref_dq
    invalid = (np.isnan(expected_data) | np.isnan(expected_err) |
               ((expected_dq & DNU) != 0))
    expected_dq[invalid] |= DNU
    expected_data[invalid] = np.nan
    expected_err[invalid] = np.nan

    assert_allclose(np.asarray(actual[0]), expected_data, rtol=1e-6)
    assert_allclose(np.asarray(actual[1]), expected_err, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual[2]), expected_dq)


def test_gain_scale():
    """Check gain scale."""
    rng = np.random.default_rng(12)
    data = _cube(rng)
    out = detector.gain_scale(data, 1.23)
    assert_allclose(np.asarray(out), data * np.float32(1.23),
                    rtol=1e-5, atol=1e-4)
    # Factor 1 is a pass-through.
    out1 = np.asarray(detector.gain_scale(data, 1.0))
    assert_allclose(out1, data, rtol=0, atol=0)


@pytest.mark.parametrize('nframes', [2, 4])
def test_read_level_linearity_matches_stcal(nframes):
    """Compare inferred reads and saturated groups against STCAL 1.20."""
    official = pytest.importorskip('stcal.linearity.linearity')
    rng = np.random.default_rng(71)
    data = rng.uniform(100., 2000., (2, 5, 3, 4)).astype(np.float32)
    gdq = np.zeros(data.shape, np.uint8)
    gdq[0, 3:, 1, 2] = SAT
    coeffs = np.zeros((3, 3, 4), np.float32)
    coeffs[1] = 1.
    coeffs[2] = 1e-5
    inverse = coeffs.copy()
    inverse[2] *= -1
    pattern = [list(range(g * nframes + 1, (g + 1) * nframes + 1))
               for g in range(data.shape[1])]
    expected, _, _ = official.linearity_correction(
        data.copy(), gdq.copy(), np.zeros((3, 4), np.uint32),
        coeffs.copy(), np.zeros((3, 4), np.uint32),
        {'SATURATED': SAT, 'NO_LIN_CORR': 1 << 20},
        ilin_coeffs=inverse.copy(), read_pattern=pattern)
    actual = detector.linearity_correct_reads(
        data, gdq, coeffs, inverse, nframes=nframes)
    assert_allclose(np.asarray(actual), expected, rtol=2e-6, atol=2e-4)
    np.testing.assert_array_equal(np.asarray(actual)[gdq == SAT],
                                  data[gdq == SAT])
