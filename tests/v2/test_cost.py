"""Tests for exotedrf.v2.kernels.cost against the v1 cost function."""

import warnings

import numpy as np
import pytest

from exotedrf.v2.kernels import cost


def _ref_cost_function(st3, baseline_ints=None, wave_range=None, w1=0.0,
                       w2=1.0, tol=0.05, niriss=False):
    """Calculate cost function with the reference equations."""
    if niriss:
        flux_O1 = np.asarray(st3['Flux O1'], float)
        flux_O2 = np.asarray(st3['Flux O2'], float)
        wave_O1 = np.asarray(st3['Wave O1'], float)
        wave_O2 = np.asarray(st3['Wave O2'], float)
        cutoff = 0.85
        i2 = np.where(wave_O2 <= cutoff)[0]
        i1 = np.where(wave_O1 > cutoff)[0]
        if i2.size == 0 or i1.size == 0:
            raise ValueError("Cutoff produces empty segment: "
                             f"O2<= {cutoff}: {i2.size}, O1> {cutoff}: {i1.size}")
        idx2 = i2[-1]
        idx1 = i1[0]
        wave = np.concatenate([wave_O2[:idx2+1], wave_O1[idx1:]])
        flux = np.concatenate([flux_O2[:, :idx2+1], flux_O1[:, idx1:]], axis=1)
        s = np.argsort(wave)
        wave = wave[s]
        flux = flux[:, s]
    else:
        flux = np.asarray(st3['Flux'], float)
        wave = np.asarray(st3['Wave'], float)
        if wave.ndim == 2:
            if wave.shape == flux.shape:
                wave = np.nanmedian(wave, axis=0)
            elif 1 in wave.shape:
                wave = np.ravel(wave)

    white = np.nansum(flux, axis=1)
    white = white[~np.isnan(white)]
    norm_white = white / np.median(white)
    d2_white = 0.5*(norm_white[:-2] + norm_white[2:]) - norm_white[1:-1]
    ptp2_white = np.nanmedian(np.abs(d2_white))

    wave_meds = np.nanmedian(flux, axis=0, keepdims=True)
    norm_spec = flux / wave_meds
    d2_spec = 0.5*(norm_spec[:-2] + norm_spec[2:]) - norm_spec[1:-1]

    if baseline_ints is None:
        ptp2_spec_wave = np.nanmedian(np.abs(d2_spec), axis=0)
    elif len(baseline_ints) == 1:
        N = int(baseline_ints[0])
        ptp2_spec_wave = np.nanmedian(np.abs(d2_spec[:N]), axis=0)
    elif len(baseline_ints) == 2:
        Nlow, Nhigh = map(int, baseline_ints)
        low_term = np.nanmedian(np.abs(d2_spec[:Nlow]), axis=0)
        high_term = np.nanmedian(np.abs(d2_spec[Nhigh:]), axis=0)
        ptp2_spec_wave = 0.5 * (low_term + high_term)

    if wave_range is None:
        ptp2_spec = np.nanmedian(ptp2_spec_wave)
    else:
        lo, hi = wave_range
        finite = np.isfinite(wave)
        wave_min = np.nanmin(wave[finite])
        wave_max = np.nanmax(wave[finite])
        if lo is None:
            lo = wave_min
        if hi is None:
            hi = wave_max
        dist_lo = np.abs(wave - lo); dist_lo[~finite] = np.inf
        dist_hi = np.abs(wave - hi); dist_hi[~finite] = np.inf
        idx_lo = int(np.argmin(dist_lo))
        idx_hi = int(np.argmin(dist_hi))
        i0, i1 = sorted((idx_lo, idx_hi))
        sub = ptp2_spec_wave[i0:i1+1]
        ptp2_spec = np.nanmedian(sub)

    cost_val = 0.0
    if w1 != 0:
        cost_val += w1 * ptp2_white
    if w2 != 0:
        cost_val += w2 * ptp2_spec
    return cost_val, ptp2_spec_wave


# Fixtures (tiny arrays: 8 GB dev machine).

NINTS, NWAVE = 20, 30
RTOL = 1e-5


def _make_flux(seed=0, nan_frac=0.03, scatter=0.05):
    """Create flux for the test observation."""
    rng = np.random.default_rng(seed)
    meds = 500.0 + 1000.0 * rng.random(NWAVE)
    flux = meds[None, :] * (1.0 + scatter * rng.standard_normal(
        (NINTS, NWAVE)))
    flux[rng.random(flux.shape) < nan_frac] = np.nan
    return flux.astype(np.float32)


def _wave():
    """Return wave."""
    return np.linspace(0.6, 2.8, NWAVE)


# Hand-built scatter: exact analytic values.

def test_known_alternating_scatter():
    """Check known alternating scatter."""
    a = 0.05
    t = np.arange(NINTS)
    meds = 100.0 + 10.0 * np.arange(NWAVE)
    flux = (meds[None, :] * (1.0 + a * (-1.0)**t)[:, None]).astype(np.float32)

    c, scatter = cost.cost_function(flux, wave=_wave(), w1=1.0, w2=1.0)
    np.testing.assert_allclose(np.asarray(scatter), 2 * a, rtol=RTOL)
    np.testing.assert_allclose(float(c), 4 * a, rtol=RTOL)

    # Spectral-only / white-only weightings.
    c_spec, _ = cost.cost_function(flux, wave=_wave(), w1=0.0, w2=1.0)
    np.testing.assert_allclose(float(c_spec), 2 * a, rtol=RTOL)
    c_white, _ = cost.cost_function(flux, wave=_wave(), w1=1.0, w2=0.0)
    np.testing.assert_allclose(float(c_white), 2 * a, rtol=RTOL)


# Parity with the v1 reference.

@pytest.mark.parametrize('baseline_ints', [None, [8], [6, -6], [6, 14],
                                           [-5]])
def test_parity_baselines(baseline_ints):
    """Check parity baselines."""
    flux = _make_flux(seed=1)
    st3 = {'Flux': flux, 'Wave': _wave()}
    ref_c, ref_s = _ref_cost_function(st3, baseline_ints=baseline_ints,
                                      w1=0.3, w2=0.7)
    c, s = cost.cost_function(flux, wave=_wave(),
                              baseline_ints=baseline_ints, w1=0.3, w2=0.7)
    np.testing.assert_allclose(np.asarray(s), ref_s, rtol=RTOL, atol=1e-7,
                               equal_nan=True)
    np.testing.assert_allclose(float(c), ref_c, rtol=RTOL, atol=1e-7)


@pytest.mark.parametrize('wave_range', [[1.0, 2.0], [None, 2.0],
                                        [1.0, None], [0.97, 2.03]])
def test_parity_wave_range(wave_range):
    """Check parity wave range."""
    flux = _make_flux(seed=2)
    st3 = {'Flux': flux, 'Wave': _wave()}
    ref_c, ref_s = _ref_cost_function(st3, baseline_ints=[6, -6],
                                      wave_range=wave_range, w1=0.4, w2=0.6)
    c, s = cost.cost_function(flux, wave=_wave(), baseline_ints=[6, -6],
                              wave_range=wave_range, w1=0.4, w2=0.6)
    np.testing.assert_allclose(np.asarray(s), ref_s, rtol=RTOL, atol=1e-7,
                               equal_nan=True)
    np.testing.assert_allclose(float(c), ref_c, rtol=RTOL, atol=1e-7)


def test_parity_2d_wave():
    """Check parity 2d wave."""
    flux = _make_flux(seed=3)
    wave2d = np.repeat(_wave()[None, :], NINTS, axis=0)
    st3 = {'Flux': flux, 'Wave': wave2d}
    ref_c, ref_s = _ref_cost_function(st3, baseline_ints=[8],
                                      wave_range=[1.0, 2.5], w1=0.0, w2=1.0)
    c, s = cost.cost_function(flux, wave=wave2d, baseline_ints=[8],
                              wave_range=[1.0, 2.5], w1=0.0, w2=1.0)
    np.testing.assert_allclose(float(c), ref_c, rtol=RTOL, atol=1e-7)
    np.testing.assert_allclose(np.asarray(s), ref_s, rtol=RTOL, atol=1e-7,
                               equal_nan=True)


# NIRISS O1/O2 stitching at 0.85 um.

def test_soss_stitch_and_cost_parity():
    """Check SOSS stitch and cost parity."""
    rng = np.random.default_rng(4)
    n_o1, n_o2 = 25, 18
    wave_o1 = np.linspace(0.85, 2.8, n_o1)
    wave_o2 = np.linspace(0.6, 1.4, n_o2)
    flux_o1 = (1000.0 * (1 + 0.05 * rng.standard_normal(
        (NINTS, n_o1)))).astype(np.float32)
    flux_o2 = (300.0 * (1 + 0.05 * rng.standard_normal(
        (NINTS, n_o2)))).astype(np.float32)

    flux, wave = cost.stitch_soss_orders(flux_o1, wave_o1, flux_o2, wave_o2)

    st3 = {'Flux O1': flux_o1, 'Flux O2': flux_o2,
           'Wave O1': wave_o1, 'Wave O2': wave_o2}
    ref_c, ref_s = _ref_cost_function(st3, baseline_ints=[6, -6],
                                      w1=0.5, w2=0.5, niriss=True)
    # Reference merge for the wave grid itself.
    i2 = np.where(wave_o2 <= 0.85)[0]
    i1 = np.where(wave_o1 > 0.85)[0]
    ref_wave = np.concatenate([wave_o2[:i2[-1]+1], wave_o1[i1[0]:]])
    ref_wave = ref_wave[np.argsort(ref_wave)]
    np.testing.assert_array_equal(wave, ref_wave)
    assert np.all(wave[wave <= 0.85] <= 0.85)

    c, s = cost.cost_function(flux, wave=wave, baseline_ints=[6, -6],
                              w1=0.5, w2=0.5)
    np.testing.assert_allclose(float(c), ref_c, rtol=RTOL, atol=1e-7)
    np.testing.assert_allclose(np.asarray(s), ref_s, rtol=RTOL, atol=1e-7,
                               equal_nan=True)


def test_soss_stitch_empty_segment_raises():
    """Check SOSS stitch empty segment raises."""
    wave_o2 = np.linspace(0.9, 1.4, 10)
    wave_o1 = np.linspace(0.85, 2.8, 10)
    flux = np.ones((4, 10), dtype=np.float32)
    with pytest.raises(ValueError, match='empty segment'):
        cost.stitch_soss_orders(flux, wave_o1, flux, wave_o2)


def test_zero_weight_nan_guards():
    """Check zero weight NaN guards."""
    flux = np.full((NINTS, NWAVE), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref_c00, _ = _ref_cost_function({'Flux': flux, 'Wave': _wave()},
                                        w1=0.0, w2=0.0)
        ref_c10, _ = _ref_cost_function({'Flux': flux, 'Wave': _wave()},
                                        w1=1.0, w2=0.0)
        c00, _ = cost.cost_function(flux, wave=_wave(), w1=0.0, w2=0.0)
        c10, _ = cost.cost_function(flux, wave=_wave(), w1=1.0, w2=0.0)
    # Both weights zero: exactly 0.0, never NaN (matches v1).
    assert ref_c00 == 0.0
    assert float(c00) == 0.0 and not np.isnan(float(c00))
    # Nonzero weight on the NaN term: NaN, as in v1.
    assert np.isnan(ref_c10) and np.isnan(float(c10))


def test_w1_zero_ignores_white_term():
    """Check w1 zero ignores white term."""
    flux = _make_flux(seed=5)
    c_both, s = cost.cost_function(flux, wave=_wave(), baseline_ints=[8],
                                   w1=0.0, w2=2.0)
    ref_c, _ = _ref_cost_function({'Flux': flux, 'Wave': _wave()},
                                  baseline_ints=[8], w1=0.0, w2=2.0)
    np.testing.assert_allclose(float(c_both), ref_c, rtol=RTOL, atol=1e-7)
    # And it is exactly 2x the w2=1 cost.
    c_one, _ = cost.cost_function(flux, wave=_wave(), baseline_ints=[8],
                                  w1=0.0, w2=1.0)
    np.testing.assert_allclose(float(c_both), 2 * float(c_one), rtol=1e-6)


def test_baseline_ints_validation():
    """Check baseline ints validation."""
    flux = _make_flux(seed=6)
    with pytest.raises(ValueError, match='length 1 or 2'):
        cost.cost_function(flux, wave=_wave(), baseline_ints=[1, 2, 3])
    with pytest.raises(ValueError, match='length-2'):
        cost.cost_function(flux, wave=_wave(), wave_range=[1.0, 2.0, 3.0])
