"""Check extraction kernels against v1 reference calculations."""

import numpy as np
import pytest

import jax.numpy as jnp
from scipy.ndimage import median_filter

from exotedrf.v2.kernels import extract


# V1 reference implementations (numpy).

def _ref_box_extraction_nanaware(cube, ypos, width, extract_start=0,
                                 extract_end=None):
    """Calculate box extraction nanaware with the reference equations."""
    nint, dimy, dimx = np.shape(cube)
    if extract_end is None:
        extract_end = dimx
    f = np.zeros((nint, dimx))

    lower_width = upper_width = float(width) / 2
    edge_up = np.min([ypos + upper_width, np.ones_like(ypos) * dimy], axis=0)
    edge_low = np.max([ypos - lower_width, np.zeros_like(ypos)], axis=0)

    for i in range(nint):
        for x in range(extract_start, extract_end):
            xx = x - extract_start
            if xx >= len(ypos):
                xx = len(ypos) - 1

            up_whole = np.floor(edge_up[xx]).astype(int)
            low_whole = np.ceil(edge_low[xx]).astype(int)

            box = cube[i, low_whole:up_whole, x]
            total_flux = np.nansum(box)
            total_area = np.sum(np.isfinite(box))

            if edge_up[xx] < (dimy - 1) and edge_low[xx] > 0:
                up_part = edge_up[xx] % 1
                low_part = 1 - edge_low[xx] % 1
                up_val = cube[i, up_whole, x]
                low_val = cube[i, low_whole - 1, x]
                if np.isfinite(up_val):
                    total_flux += up_part * up_val
                    total_area += up_part
                if np.isfinite(low_val):
                    total_flux += low_part * low_val
                    total_area += low_part

            if total_area > 0:
                f[i, x] = total_flux / total_area
            else:
                f[i, x] = np.nan
    return f


def _ref_get_aperture_pixels(edge_low, edge_up, dimy):
    """Calculate get aperture pixels with the reference equations."""
    row_start = max(int(np.floor(edge_low)), 0)
    row_end = min(int(np.ceil(edge_up)), dimy)
    rows = np.arange(row_start, row_end)
    if len(rows) == 0:
        return rows.astype(int), np.array([], dtype=float)
    weights = np.minimum(rows + 1, edge_up) - np.maximum(rows, edge_low)
    weights = np.clip(weights, 0, 1)
    ii = np.where(weights > 0)[0]
    return rows[ii].astype(int), weights[ii]


def _ref_box_extraction(cube, err, ypos, width, extract_start=0,
                        extract_end=None):
    """Calculate box extraction with the reference equations."""
    nint, dimy, dimx = np.shape(cube)
    if extract_end is None:
        extract_end = dimx
    f, ferr = np.zeros((nint, dimx)), np.zeros((nint, dimx))

    lower_half = upper_half = float(width) / 2
    ypos = np.asarray(ypos, dtype=float)
    edge_up = np.min([ypos + upper_half,
                      np.ones_like(ypos, dtype=float) * dimy], axis=0)
    edge_low = np.max([ypos - lower_half,
                       np.zeros_like(ypos, dtype=float)], axis=0)

    for x in range(extract_start, extract_end):
        xx = x - extract_start
        rows, weights = _ref_get_aperture_pixels(edge_low[xx], edge_up[xx],
                                                 dimy)
        if len(rows) == 0:
            continue
        weighted_cube = cube[:, rows, x] * weights[None, :]
        weighted_err = err[:, rows, x] * weights[None, :]
        f[:, x] = np.nansum(weighted_cube, axis=1)
        ferr[:, x] = np.sqrt(np.nansum(weighted_err**2, axis=1))
    return f, ferr


def _ref_horne(data, profile, max_iter=25, var_thresh=25):
    """Vectorized transcription of stage3.do_optimal_extraction."""
    data = np.array(data, dtype=float, copy=True)
    flux = np.nansum(data, axis=2)
    variance0 = np.nanstd(data, axis=0) ** 2
    variance = (variance0[None] +
                np.abs(flux[:, :, None] * profile[None]))

    def solve(values, var):
        numerator = np.nansum(profile[None] * values / var, axis=2)
        denominator = np.nansum(profile[None] ** 2 / var, axis=2)
        result = numerator / denominator
        result_var = np.nansum(profile, axis=1)[None] / denominator
        return result, result_var

    flux, flux_variance = solve(data, variance)
    counts = []
    previous = None
    for _ in range(max_iter):
        filtered = median_filter(data, (11, 1, 1))
        bad = ((data - flux[:, :, None] * profile[None]) ** 2 /
               variance > var_thresh)
        data[bad] = filtered[bad]
        count = int(np.sum(bad))
        counts.append(count)
        variance0 = np.nanstd(data, axis=0) ** 2
        variance = np.broadcast_to(variance0[None], data.shape).copy()
        variance[variance == 0] = np.inf
        flux, flux_variance = solve(data, variance)
        if count == 0 or count == previous:
            break
        previous = count
    return flux, np.sqrt(flux_variance), tuple(counts)


# Fixtures (tiny cubes: 8 GB dev machine).

NINTS, DIMY, DIMX = 6, 16, 32
RTOL = 1e-5


def _make_cube(seed=0, nan_frac=0.08, with_err=False):
    """Create cube for the test observation."""
    rng = np.random.default_rng(seed)
    data = (50.0 + 10.0 * rng.standard_normal((NINTS, DIMY, DIMX)))
    nan_mask = rng.random(data.shape) < nan_frac
    data[nan_mask] = np.nan
    data = data.astype(np.float32)
    if not with_err:
        return data
    err = (1.0 + rng.random(data.shape)).astype(np.float32)
    err[rng.random(err.shape) < 0.05] = np.nan
    return data, err


def _make_centroid(seed=1):
    """Create centroid for the test observation."""
    x = np.arange(DIMX)
    cen = 8.0 + 1.5 * np.sin(2 * np.pi * x / DIMX) + 0.13
    return cen.astype(np.float32)


# Box_extract, nanaware mode (per-trial optimizer semantics).

def test_uniform_column_flux():
    """Check uniform column flux."""
    value = 3.0
    data = np.full((2, DIMY, DIMX), value, dtype=np.float32)
    cen = np.full(DIMX, 8.25, dtype=np.float32)
    for hw in (1.5, 2.6, 3.0, 4.75):
        flux, _ = extract.box_extract(data, None, cen, hw, mode='sum')
        np.testing.assert_allclose(np.asarray(flux), 2 * hw * value,
                                   rtol=RTOL)
        flux, _ = extract.box_extract(data, None, cen, hw, mode='nanaware')
        np.testing.assert_allclose(np.asarray(flux), value, rtol=RTOL)


@pytest.mark.parametrize('halfwidth', [1.7, 2.5, 3.3, 4.6, 6.9])
def test_nanaware_parity_random(halfwidth):
    """Check nanaware parity random."""
    data = _make_cube(seed=2)
    cen = _make_centroid()
    ref = _ref_box_extraction_nanaware(data.astype(float), cen.astype(float),
                                       width=2 * halfwidth)
    flux, ferr = extract.box_extract(data, None, cen, halfwidth,
                                     mode='nanaware')
    assert ferr is None
    np.testing.assert_allclose(np.asarray(flux), ref, rtol=RTOL,
                               equal_nan=True)


def test_nanaware_subrange_and_centroid_clamp():
    """Check nanaware subrange and centroid clamp."""
    data = _make_cube(seed=3)
    ncen, start, end = 20, 4, 28
    cen = _make_centroid()[:ncen]
    hw = 2.5
    ref = _ref_box_extraction_nanaware(data.astype(float), cen.astype(float),
                                       width=2 * hw, extract_start=start,
                                       extract_end=end)
    flux, _ = extract.box_extract(data, None, cen, hw, extract_start=start,
                                  extract_end=end, mode='nanaware')
    flux = np.asarray(flux)
    np.testing.assert_allclose(flux, ref, rtol=RTOL, equal_nan=True)
    assert np.all(flux[:, :start] == 0.0)
    assert np.all(flux[:, end:] == 0.0)


def test_nanaware_integer_edge_quirk():
    """Check nanaware integer edge quirk."""
    data = _make_cube(seed=4, nan_frac=0.0)
    cen = np.full(DIMX, 8.0, dtype=np.float32)
    hw = 3.0
    ref = _ref_box_extraction_nanaware(data.astype(float), cen.astype(float),
                                       width=2 * hw)
    flux, _ = extract.box_extract(data, None, cen, hw, mode='nanaware')
    np.testing.assert_allclose(np.asarray(flux), ref, rtol=RTOL)
    # Explicit: the quirk row (row 4) participates with full weight.
    expected = data[:, 4:11, :].mean(axis=1)
    np.testing.assert_allclose(np.asarray(flux), expected, rtol=RTOL)


def test_nanaware_all_nan_column():
    """Check nanaware all NaN column."""
    data = _make_cube(seed=5, nan_frac=0.0)
    data[:, :, 10] = np.nan
    cen = _make_centroid()
    flux, _ = extract.box_extract(data, None, cen, 2.5, mode='nanaware')
    assert np.all(np.isnan(np.asarray(flux)[:, 10]))


# Box_extract, sum mode (production stage3 semantics + error propagation).

@pytest.mark.parametrize('halfwidth', [1.7, 2.5, 3.0, 4.6])
def test_sum_mode_parity_and_error_propagation(halfwidth):
    """Check sum mode parity and error propagation."""
    data, err = _make_cube(seed=6, with_err=True)
    cen = _make_centroid()
    ref_f, ref_e = _ref_box_extraction(data.astype(float), err.astype(float),
                                       cen.astype(float), width=2 * halfwidth)
    flux, ferr = extract.box_extract(data, err, cen, halfwidth, mode='sum')
    np.testing.assert_allclose(np.asarray(flux), ref_f, rtol=RTOL,
                               atol=1e-4, equal_nan=True)
    np.testing.assert_allclose(np.asarray(ferr), ref_e, rtol=RTOL,
                               atol=1e-4, equal_nan=True)


def test_sum_mode_err_none():
    """Check sum mode ERR none."""
    data = _make_cube(seed=7)
    cen = _make_centroid()
    flux, ferr = extract.box_extract(data, None, cen, 3.3, mode='sum')
    assert ferr is None
    assert np.asarray(flux).shape == (NINTS, DIMX)


# Vmap sweep over halfwidths.

def test_sweep_matches_individual_calls():
    """Check sweep matches individual calls."""
    data = _make_cube(seed=8)
    cen = _make_centroid()
    widths = np.array([1.7, 2.2, 2.5, 3.0, 3.8, 4.6], dtype=np.float32)
    swept, _ = extract.box_extract_sweep(data, None, cen, widths,
                                         mode='nanaware')
    swept = np.asarray(swept)
    assert swept.shape == (len(widths), NINTS, DIMX)
    for k, hw in enumerate(widths):
        single, _ = extract.box_extract(data, None, cen, float(hw),
                                        mode='nanaware')
        np.testing.assert_allclose(swept[k], np.asarray(single), rtol=RTOL,
                                   equal_nan=True)


# Extract_from_stage1 (optimizer mid-pipeline path).

def test_extract_from_stage1_last_group_and_dq():
    """Check extract from stage1 last group and DQ."""
    rng = np.random.default_rng(9)
    ngroups = 3
    data4 = (40.0 + 5.0 * rng.standard_normal(
        (5, ngroups, DIMY, DIMX))).astype(np.float32)
    dq4 = np.zeros((5, ngroups, DIMY, DIMX), dtype=np.uint32)
    # Flags in the last group -> NaNed; flags only in group 0 -> ignored.
    last_bad = rng.random((5, DIMY, DIMX)) < 0.1
    dq4[:, -1][last_bad] = 1
    dq4[:, -1][rng.random((5, DIMY, DIMX)) < 0.03] |= 2048
    early_only = (rng.random((5, DIMY, DIMX)) < 0.1) & ~(dq4[:, -1] > 0)
    dq4[:, 0][early_only] = 4
    cen = _make_centroid()
    hw = 2.5

    expected_img = data4[:, -1].astype(float).copy()
    expected_img[dq4[:, -1] > 0] = np.nan
    ref = _ref_box_extraction_nanaware(expected_img, cen.astype(float),
                                       width=2 * hw)

    flux = extract.extract_from_stage1(data4, dq4, cen, hw)
    np.testing.assert_allclose(np.asarray(flux), ref, rtol=RTOL,
                               equal_nan=True)
    dq_clear = dq4.copy()
    dq_clear[:, 0] = 0
    flux2 = extract.extract_from_stage1(data4, dq_clear, cen, hw)
    np.testing.assert_array_equal(np.asarray(flux), np.asarray(flux2))


def test_extract_from_stage1_3d_and_2d_dq():
    """Check extract from stage1 3d and 2d DQ."""
    data = _make_cube(seed=10, nan_frac=0.0)
    cen = _make_centroid()
    hw = 3.0
    rng = np.random.default_rng(11)
    dq3 = (rng.random(data.shape) < 0.1).astype(np.uint32) * 16
    img = data.astype(float).copy()
    img[dq3 > 0] = np.nan
    ref = _ref_box_extraction_nanaware(img, cen.astype(float), width=2 * hw)
    flux = extract.extract_from_stage1(data, dq3, cen, hw)
    np.testing.assert_allclose(np.asarray(flux), ref, rtol=RTOL,
                               equal_nan=True)

    dq2 = (rng.random(data.shape[1:]) < 0.1).astype(np.uint32)
    img = data.astype(float).copy()
    img[:, dq2 > 0] = np.nan
    ref = _ref_box_extraction_nanaware(img, cen.astype(float), width=2 * hw)
    flux = extract.extract_from_stage1(data, dq2, cen, hw)
    np.testing.assert_allclose(np.asarray(flux), ref, rtol=RTOL,
                               equal_nan=True)


def test_halfwidth_traced_no_python_branching():
    """Check halfwidth traced no python branching."""
    data = _make_cube(seed=12)
    cen = _make_centroid()
    flux, _ = extract.box_extract(data, None, cen, jnp.float32(2.5),
                                  mode='nanaware')
    ref, _ = extract.box_extract(data, None, cen, 2.5, mode='nanaware')
    np.testing.assert_array_equal(np.asarray(flux), np.asarray(ref))


# GPU Horne extraction.

@pytest.mark.parametrize('max_iter', [0, 3])
def test_horne_optimal_extract_v1_parity(max_iter):
    """Check horne optimal extract v1 parity."""
    rng = np.random.default_rng(21)
    nints, nwave, naperture = 23, 7, 5
    profile = rng.uniform(0.1, 1., (nwave, naperture))
    profile /= profile.sum(axis=1, keepdims=True)
    spectra = rng.uniform(500., 1500., (nints, nwave))
    data = spectra[:, :, None] * profile[None]
    data += rng.normal(0., 2., data.shape)
    data[5, 2, 3] += 80.
    data[14, 6, 1] -= 90.
    data = data.astype(np.float32)
    profile = profile.astype(np.float32)

    ref_flux, ref_ferr, ref_counts = _ref_horne(
        data, profile, max_iter=max_iter, var_thresh=25)
    flux, ferr, counts = extract.horne_optimal_extract(
        data, profile, max_iter=max_iter, var_thresh=25)
    np.testing.assert_allclose(np.asarray(flux), ref_flux, rtol=2e-5,
                               atol=2e-4)
    np.testing.assert_allclose(np.asarray(ferr), ref_ferr, rtol=2e-5,
                               atol=2e-5)
    assert counts == ref_counts


@pytest.mark.parametrize(
    'kwargs, match',
    [({'max_iter': -1}, 'non-negative'),
     ({'max_iter': True}, 'non-negative'),
     ({'var_thresh': 0}, 'positive')])
def test_horne_optimal_extract_rejects_invalid_controls(kwargs, match):
    """Check horne optimal extract rejects invalid controls."""
    data = np.ones((3, 2, 2), np.float32)
    profile = np.full((2, 2), .5, np.float32)
    with pytest.raises(ValueError, match=match):
        extract.horne_optimal_extract(data, profile, **kwargs)
