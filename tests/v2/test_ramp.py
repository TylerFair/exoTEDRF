"""Tests for exotedrf.v2.kernels.ramp."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from exotedrf.v2 import core
from exotedrf.v2.kernels import ramp

warnings.filterwarnings('ignore', message='All-NaN')
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='invalid value')

DNU, SAT, JUMP = 1, 2, 4
GT = 5.494


def _random_cube(rng, shape, slope_scale=20., rn=6., nan_frac=0.02,
                 sat_frac=0.03, dnu_frac=0.02, jump_frac=0.05):
    """Random linear-ish ramps with noise, NaNs and pre-set DQ flags."""
    nints, ngroups, dimy, dimx = shape
    t = (np.arange(ngroups) + 1.) * GT
    slopes = rng.uniform(0.5, slope_scale, size=(nints, dimy, dimx))
    data = slopes[:, None] * t[None, :, None, None] \
        + rng.normal(0., rn, size=shape)
    data = data.astype(np.float32)
    data[rng.random(shape) < nan_frac] = np.nan
    gdq = np.zeros(shape, dtype=np.uint8)
    gdq[rng.random(shape) < sat_frac] |= SAT
    gdq[rng.random(shape) < dnu_frac] |= DNU
    jl = rng.random(shape) < jump_frac
    jl[:, 0] = False
    gdq[jl] |= JUMP
    return data, gdq


# Flag_jumps_upramp: numpy reference.

def ref_flag_jumps(data, groupdq, readnoise, thresh):
    """Detect up-the-ramp jumps with the reference equations.

    Parameters
    ----------
    data : array-like(float)
        Data array.
    groupdq : array-like(int)
        Groupdq array.
    readnoise : array-like(float)
        Readnoise array.
    thresh : float
        Detection threshold.

    Returns
    -------
    flags : np.ndarray(int)
        Updated group DQ flags.
    """
    ngroups = data.shape[1]
    if ngroups <= 2:
        return groupdq.copy()
    ndiffs = ngroups - 1
    bad = ((groupdq & (SAT | DNU)) != 0) | ~np.isfinite(data)
    d = (data[:, 1:] - data[:, :-1]).astype(np.float64)
    d[bad[:, 1:] | bad[:, :-1]] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if ndiffs == 2:
            base = np.nanmin(d, axis=1)
        else:
            base = np.nanmedian(d, axis=1)
        sigma = np.sqrt(np.clip(base, 0., None)
                        + 2. * np.float64(readnoise) ** 2)
        if ndiffs >= 4:
            mad = np.nanmedian(np.abs(d - base[:, None]), axis=1)
            sigma = np.maximum(sigma, 1.4826 * mad)
    with np.errstate(invalid='ignore'):
        jump = np.abs(d - base[:, None]) / sigma[:, None] > thresh
    jump = np.nan_to_num(jump)
    flags = np.zeros(data.shape, dtype=np.uint8)
    flags[:, 1:][jump] = JUMP
    return groupdq | flags


def test_flag_jumps_matches_reference():
    """Check flag jumps matches reference."""
    rng = np.random.default_rng(20)
    rn = 6.
    for ngroups in (3, 4, 6):
        data, gdq = _random_cube(rng, (4, ngroups, 8, 12), jump_frac=0.)
        # Inject some genuine jumps.
        jmask = rng.random((4, 8, 12)) < 0.1
        jg = rng.integers(1, ngroups, size=(4, 8, 12))
        for i in range(4):
            for y in range(8):
                for x in range(12):
                    if jmask[i, y, x]:
                        data[i, jg[i, y, x]:, y, x] += 5000.
        ref = ref_flag_jumps(data, gdq, rn, 10.)
        out = np.asarray(ramp.flag_jumps_upramp(data, gdq, np.float32(rn),
                                                rejection_threshold=10.))
        assert np.array_equal(out, ref), 'ngroups={}'.format(ngroups)


def test_flag_jumps_readnoise_map():
    """Check flag jumps readnoise map."""
    rng = np.random.default_rng(21)
    data, gdq = _random_cube(rng, (3, 5, 8, 12), jump_frac=0.)
    data[1, 3:, 4, 6] += 4000.
    rn_map = rng.uniform(4., 12., size=(8, 12)).astype(np.float32)
    ref = ref_flag_jumps(data, gdq, rn_map, 10.)
    out = np.asarray(ramp.flag_jumps_upramp(data, gdq, rn_map,
                                            rejection_threshold=10.))
    assert np.array_equal(out, ref)


def test_flag_jumps_detects_injected_jump():
    """Check flag jumps detects injected jump."""
    nints, ngroups, dimy, dimx = 2, 6, 6, 8
    t = (np.arange(ngroups) + 1.) * GT
    data = np.broadcast_to(10. * t[None, :, None, None],
                           (nints, ngroups, dimy, dimx)).astype(np.float32)
    data = data.copy()
    data[1, 3:, 2, 4] += 2000.
    gdq = np.zeros(data.shape, dtype=np.uint8)
    out = np.array(ramp.flag_jumps_upramp(data, gdq, 6.,
                                          rejection_threshold=10.))
    assert out[1, 3, 2, 4] == JUMP
    out[1, 3, 2, 4] = 0
    assert np.all(out == 0)


def test_flag_jumps_ngroups2_noop():
    """Check flag jumps ngroups2 noop."""
    rng = np.random.default_rng(22)
    data, gdq = _random_cube(rng, (3, 2, 6, 8))
    out = np.asarray(ramp.flag_jumps_upramp(data, gdq, 6.,
                                            rejection_threshold=10.))
    assert np.array_equal(out, gdq)


# Fit_ramps: frozen JWST 3.0.0 / STCAL 1.20.0 golden values.

@pytest.mark.parametrize(
    'values,group_flags,expected_rate,expected_err,expected_dq',
    [
        ([100, 111, 121, 134, 139, 154], [0, 0, 0, 0, 0, 0],
         10.507720947265625, 1.550345540046692, 0),
        # A JUMP group begins and is included in the second semiramp.
        ([100, 111, 121, 634, 644, 654], [0, 0, 0, JUMP, 0, 0],
         10.25, 2.462214469909668, JUMP),
        ([100, 101, 999, 130, 140, 150], [0, 0, DNU, 0, 0, 0],
         8.199999809265137, 3.0694077014923096, 0),
        ([100, 1000, 2000, 2020], [0, 8, 0, 0],
         20.0, 24.464259645450134, 0),
        ([60000, 100, 110, 120], [SAT, 0, 0, 0],
         np.nan, 0., SAT | DNU),
        ([55, 999, 77, 999], [0, DNU, 0, DNU],
         55., 8.388980867781259, 0),
        ([999, 888, 55, 777, 66], [JUMP, DNU, 0, DNU, 0],
         55., 25.6977615, JUMP),
        ([100, 200, 300, 400], [JUMP, JUMP, JUMP, JUMP],
         0., 0., JUMP),
        ([100, 200, 300, 400], [8, 8, 8, 8],
         0., 0., 0),
        # RampFitStep defaults suppress_one_group=True for NGROUPS > 1.
        ([55, 60000, 60000, 60000], [0, SAT, SAT, SAT],
         np.nan, 0., DNU | SAT),
    ],
    ids=('optimal-clean', 'jump-split', 'isolated-dnu', 'dropout-median-policy',
         'first-group-saturated', 'singleton-prune',
         'orphan-jump-singleton', 'all-jump', 'all-dropout',
         'one-group-suppressed'))
def test_fit_ramps_matches_pinned_stcal_golden(
        values, group_flags, expected_rate, expected_err, expected_dq):
    """Check fit ramps matches pinned stcal golden."""
    data = np.asarray(values, np.float32)[None, :, None, None]
    gdq = np.asarray(group_flags, np.uint8)[None, :, None, None]
    rate, err, dq = ramp.fit_ramps(
        data, gdq, np.float32(6.), np.float32(1.6), np.float32(1.))
    assert_allclose(np.asarray(rate)[0, 0, 0], expected_rate,
                    rtol=2e-6, atol=2e-6, equal_nan=True)
    assert_allclose(np.asarray(err)[0, 0, 0], expected_err,
                    rtol=2e-6, atol=2e-6, equal_nan=True)
    assert int(np.asarray(dq)[0, 0, 0]) == expected_dq
    if group_flags == [0, 8, 0, 0]:
        med = ramp.median_rates_per_integration(data, gdq, np.float32(1.))
        assert_allclose(np.asarray(med)[0, 0, 0], 900., rtol=0., atol=0.)


def test_fit_ramps_one_group_rate_when_not_suppressed():
    """Check fit ramps one group rate when not suppressed."""
    data = np.asarray([55, 60000, 60000, 60000], np.float32)[None, :, None, None]
    gdq = np.asarray([0, SAT, SAT, SAT], np.uint8)[None, :, None, None]
    gt, rn, gain = np.float32(2.), np.float32(6.), np.float32(1.6)
    pixeldq = np.zeros((1, 1), np.uint32)
    rate, err, dq = ramp.fit_ramps(data, gdq, rn, gain, gt,
                                   suppress_one_group=False)
    # Singleton variance: median rate / (gt * gain) + rn^2 / (nframes gt^2).
    assert_allclose(np.asarray(rate)[0, 0, 0], 27.5, rtol=2e-6)
    assert_allclose(np.asarray(err)[0, 0, 0],
                    np.sqrt(27.5 / (2. * 1.6) + 36. / 4.), rtol=2e-6)
    assert int(np.asarray(dq)[0, 0, 0]) == SAT


    med = ramp.median_rates_per_integration(data, gdq, gt,
                                            suppress_one_group=False)
    assert_allclose(np.asarray(med)[0, 0, 0], 27.5, rtol=2e-6)
    rate, _, dq = ramp.fit_ramps_stage(data, gdq, rn, gain, gt, pixeldq,
                                       suppress_one_group=False)
    assert_allclose(np.asarray(rate)[0, 0, 0], 27.5, rtol=2e-6)
    assert int(np.asarray(dq)[0, 0, 0]) == SAT
    # The default still suppresses the ramp.
    rate, _, dq = ramp.fit_ramps_stage(data, gdq, rn, gain, gt, pixeldq)
    assert np.isnan(np.asarray(rate)[0, 0, 0])
    assert int(np.asarray(dq)[0, 0, 0]) == DNU | SAT
    # Group-zero saturation stays unusable either way.
    rate, _, dq = ramp.fit_ramps_stage(data, np.full_like(gdq, SAT), rn,
                                       gain, gt, pixeldq,
                                       suppress_one_group=False)
    assert np.isnan(np.asarray(rate)[0, 0, 0])
    assert int(np.asarray(dq)[0, 0, 0]) == DNU | SAT


@pytest.mark.parametrize('values,gain,readnoise,expected_rate', [
    ([5.439324855804443, 64.90570068359375, 85.76891326904297,
      122.9776611328125, 167.5601348876953, 199.99574279785156,
      236.7843780517578, 291.05511474609375, 325.6883850097656],
     1.5992568731307983, 10.597169876098633, 7.135429382324219),
    ([-27.938718795776367, 14.672967910766602, 44.20784378051758,
      82.19129180908203, 131.35206604003906, 168.67578125,
      221.87905883789062, 261.97784423828125, 292.3504638671875],
     1.6111301183700562, 10.739189147949219, 7.414943695068359),
])
def test_fit_ramps_weights_near_snr_threshold(
        values, gain, readnoise, expected_rate):
    # Preserve JWST 3.0 golden slopes on either side of the SNR=20 boundary.
    """Check fit ramps weights near snr threshold."""
    data = np.asarray(values, np.float32)[None, :, None, None]
    groupdq = np.zeros_like(data, dtype=np.uint8)
    rate, _, dq = ramp.fit_ramps(
        data, groupdq, np.float32(readnoise), np.float32(gain),
        np.float32(GT))
    assert_allclose(np.asarray(rate)[0, 0, 0], expected_rate,
                    rtol=2e-6, atol=2e-6)
    assert int(np.asarray(dq)[0, 0, 0]) == 0


def test_fit_ramps_nframes_readnoise_scaling_and_bad_gain_dq():
    """Check fit ramps nframes readnoise scaling and bad gain DQ."""
    data = np.asarray(
        [100, 111, 121, 134, 139, 154], np.float32)[None, :, None, None]
    gdq = np.zeros_like(data, np.uint8)
    rate1, err1, _ = ramp.fit_ramps(
        data, gdq, 6., 1.6, 1., nframes=np.float32(1))
    rate4, err4, _ = ramp.fit_ramps(
        data, gdq, 6., 1.6, 1., nframes=np.float32(4))
    assert_allclose(rate4, rate1, rtol=2e-6)
    # Frozen pinned result: var_P=1.375 and var_R=1.0285714/NFRAMES.
    assert_allclose(np.asarray(err1)[0, 0, 0], 1.550345540046692,
                    rtol=2e-6)
    assert_allclose(np.asarray(err4)[0, 0, 0], 1.2775534594747684,
                    rtol=2e-6)

    bad_rate, bad_err, bad_dq = ramp.fit_ramps(data, gdq, 6., 0., 1.)
    assert np.isnan(np.asarray(bad_rate)[0, 0, 0])
    assert np.asarray(bad_err)[0, 0, 0] == 0.
    assert int(np.asarray(bad_dq)[0, 0, 0]) == (DNU | 524288)


@pytest.mark.parametrize(
    'values,flags,expected_rate,expected_err',
    [
        ([100, 110, 121, 131], [0, 0, 0, 0],
         10.3778772354, 2.3926970959),
        # STCAL 1.20 read-noise-weighted semiramps (1.11.1: 11.1621618271).
        ([100, 110, 999, 130, 142, 153], [0, 0, DNU, 0, 0, 0],
         11.1999998093, 3.1120226383),
    ])
def test_fit_ramps_average_dark_current_pinned_goldens(
        values, flags, expected_rate, expected_err):
    """Check fit ramps average dark current pinned goldens."""
    data = np.asarray(values, np.float32)[None, :, None, None]
    gdq = np.asarray(flags, np.uint8)[None, :, None, None]
    rate, err, dq = ramp.fit_ramps(
        data, gdq, 6., 1.6, 1., average_dark_current=np.float32(.2))
    assert_allclose(np.asarray(rate)[0, 0, 0], expected_rate, rtol=2e-6)
    assert_allclose(np.asarray(err)[0, 0, 0], expected_err, rtol=2e-6)
    assert int(np.asarray(dq)[0, 0, 0]) == 0


@pytest.mark.parametrize('values,expected_rate', [
    ([100, 100, 100, 100], 0.),
    ([100, 99, 98, 97], -1.),
])
def test_fit_ramps_zero_poisson_variance_uses_readnoise(
        values, expected_rate):
    """Check fit ramps zero poisson variance uses readnoise."""
    data = np.asarray(values, np.float32)[None, :, None, None]
    gdq = np.zeros_like(data, np.uint8)
    rate, err, dq = ramp.fit_ramps(data, gdq, 6., 1.6, 1.)
    assert_allclose(np.asarray(rate)[0, 0, 0], expected_rate, atol=2e-6)
    assert_allclose(np.asarray(err)[0, 0, 0], 1.897366643, rtol=2e-6)
    assert int(np.asarray(dq)[0, 0, 0]) == 0


def test_ols_c_fixsen_thresholds_and_ngroup1_uses_frame_timing():
    """Check OLS c fixsen thresholds and ngroup1 uses frame timing."""
    snr = np.asarray([5., 5.001, 10., 10.001, 20., 20.001,
                      50., 50.001, 100., 100.001], np.float32)
    np.testing.assert_array_equal(
        np.asarray(ramp._fixsen_power(snr)),
        np.asarray([.4, .4, 1., 1., 3., 3., 6., 6., 10., 10.],
                   np.float32))

    data = np.asarray([100.], np.float32)[None, :, None, None]
    gdq = np.zeros_like(data, np.uint8)
    rate, err, dq = ramp.fit_ramps(
        data, gdq, 6., 1.6, 5., nframes=np.float32(4),
        one_group_time=np.float32(2.5))
    assert_allclose(np.asarray(rate)[0, 0, 0], 40., rtol=0., atol=0.)
    assert_allclose(np.asarray(err)[0, 0, 0], np.sqrt(11.44), rtol=2e-6)
    assert int(np.asarray(dq)[0, 0, 0]) == 0


def test_fit_ramps_clean_ramp_recovers_slope():
    """Check fit ramps clean ramp recovers slope."""
    rng = np.random.default_rng(32)
    nints, ngroups, dimy, dimx = 3, 6, 6, 8
    t = (np.arange(ngroups) + 1.) * GT
    slopes = rng.uniform(1., 50., size=(nints, dimy, dimx))
    data = (slopes[:, None] * t[None, :, None, None]).astype(np.float32)
    gdq = np.zeros(data.shape, dtype=np.uint8)
    rate, err, dq = ramp.fit_ramps(data, gdq, 6., 1.6, GT)
    assert_allclose(np.asarray(rate), slopes, rtol=1e-4)
    assert np.all(np.asarray(err) > 0)
    assert np.all(np.asarray(dq) == 0)


def test_fit_ramps_float32_large_offset_is_centered():
    """Large raw-count offsets must not erase a shallow ramp in float32."""
    ngroups, slope, offset = 6, 1.0, 59000.0
    t = (np.arange(ngroups, dtype=np.float32) + 1.) * np.float32(GT)
    data = (offset + slope * t)[None, :, None, None].astype(np.float32)
    gdq = np.zeros(data.shape, dtype=np.uint8)
    rate, _, _ = ramp.fit_ramps(data, gdq, 6., 1.6, GT)
    assert_allclose(np.asarray(rate), slope, rtol=1e-4, atol=1e-4)


def test_fit_ramps_noisy_ramp_within_noise():
    """Check fit ramps noisy ramp within noise."""
    rng = np.random.default_rng(33)
    nints, ngroups, dimy, dimx = 6, 4, 16, 32
    t = (np.arange(ngroups) + 1.) * GT
    slope, rn = 10., 6.
    data = (slope * t[None, :, None, None]
            + rng.normal(0., rn, size=(nints, ngroups, dimy, dimx))
            ).astype(np.float32)
    gdq = np.zeros(data.shape, dtype=np.uint8)
    rate, err, dq = ramp.fit_ramps(data, gdq, np.float32(rn), 1.6, GT)
    rate, err = np.asarray(rate), np.asarray(err)
    assert np.all(np.abs(rate - slope) < 6. * err)
    assert abs(rate.mean() - slope) < err.mean()


def test_fit_ramps_jump_split_inverse_variance_combination():
    """Check fit ramps jump split inverse variance combination."""
    ngroups, gt = 6, GT
    s1, s2, rn, gain = 10., 30., 6., 1.6
    t = (np.arange(ngroups) + 1.) * gt
    d = np.zeros(ngroups)
    d[:3] = s1 * t[:3]
    d[3] = d[2] + 500. + s2 * gt
    d[4] = d[3] + s2 * gt
    d[5] = d[4] + s2 * gt
    data = np.tile(d[None, :, None, None], (1, 1, 2, 2)).astype(np.float32)
    gdq = np.zeros(data.shape, dtype=np.uint8)
    gdq[:, 3] = JUMP
    rate, err, dq = ramp.fit_ramps(data, gdq, rn, gain, gt)
    # Pinned STCAL uses one whole-exposure median rate for both segments' Poisson variance.
    median_rate = 20.
    var_r = 6. * rn ** 2 / ((27 - 3) * gt ** 2)
    var_p = median_rate / (gain * gt * 2.)
    expected = (s1 + s2) / 2.
    assert_allclose(np.asarray(rate), expected, rtol=1e-4)
    assert_allclose(np.asarray(err), np.sqrt((var_r + var_p) / 2.),
                    rtol=1e-4)
    assert np.all(np.asarray(dq) == JUMP)
    # Same slope on both sides of an offset-only jump -> slope recovered.
    d2 = np.zeros(ngroups)
    d2[:3] = s1 * t[:3]
    d2[3:] = s1 * t[3:] + 500.
    data2 = np.tile(d2[None, :, None, None], (1, 1, 2, 2)).astype(np.float32)
    rate2, _, _ = ramp.fit_ramps(data2, gdq, rn, gain, gt)
    assert_allclose(np.asarray(rate2), s1, rtol=1e-4)


def test_fit_ramps_saturation_truncation():
    # Saturated tail groups are excluded; the slope comes from the clean leading groups only.
    """Check fit ramps saturation truncation."""
    ngroups = 6
    t = (np.arange(ngroups) + 1.) * GT
    slope = 20.
    d = slope * t
    d[4:] = 60000.
    data = np.tile(d[None, :, None, None], (2, 1, 3, 4)).astype(np.float32)
    gdq = np.zeros(data.shape, dtype=np.uint8)
    gdq[:, 4:] = SAT
    rate, err, dq = ramp.fit_ramps(data, gdq, 6., 1.6, GT)
    assert_allclose(np.asarray(rate), slope, rtol=1e-4)
    assert np.all(np.asarray(dq) == SAT)


def test_fit_ramps_fully_saturated():
    """Check fit ramps fully saturated."""
    data = np.full((2, 4, 3, 4), 60000., dtype=np.float32)
    gdq = np.full(data.shape, SAT, dtype=np.uint8)
    rate, err, dq = ramp.fit_ramps(data, gdq, 6., 1.6, GT)
    assert np.all(np.isnan(np.asarray(rate)))
    assert np.all(np.asarray(err) == 0.)
    assert np.all(np.asarray(dq) == (SAT | DNU))


def test_fit_ramps_single_good_group():
    # RampFitStep's default suppresses the sole good group for NGROUPS > 1.
    """Check fit ramps single good group."""
    ngroups = 4
    data = np.full((2, ngroups, 3, 4), 100., dtype=np.float32)
    data[:, 0] = 55.
    gdq = np.zeros(data.shape, dtype=np.uint8)
    gdq[:, 1:] = DNU
    rate, err, dq = ramp.fit_ramps(data, gdq, 6., 1.6, GT)
    assert np.all(np.isnan(np.asarray(rate)))
    assert np.all(np.asarray(err) == 0.)
    assert np.all(np.asarray(dq) == DNU)


def test_fit_ramps_ngroups_1_and_2():
    """Check fit ramps ngroups 1 and 2."""
    rng = np.random.default_rng(34)
    # Ngroups == 1: rate = value / group_time.
    data1 = rng.normal(100., 10., size=(2, 1, 3, 4)).astype(np.float32)
    gdq1 = np.zeros(data1.shape, dtype=np.uint8)
    rate1, err1, dq1 = ramp.fit_ramps(data1, gdq1, 6., 1.6, GT)
    assert_allclose(np.asarray(rate1), data1[:, 0] / GT, rtol=1e-5)
    assert np.all(np.asarray(dq1) == 0)
    # Ngroups == 2: slope = (g1 - g0) / group_time.
    data2 = np.zeros((2, 2, 3, 4), dtype=np.float32)
    data2[:, 0] = rng.normal(50., 5., size=(2, 3, 4))
    diff = rng.normal(80., 5., size=(2, 3, 4)).astype(np.float32)
    data2[:, 1] = data2[:, 0] + diff
    gdq2 = np.zeros(data2.shape, dtype=np.uint8)
    rate2, err2, dq2 = ramp.fit_ramps(data2, gdq2, 6., 1.6, GT)
    assert_allclose(np.asarray(rate2), diff / GT, rtol=1e-4)


def test_fit_ramps_chunked_over_ints_matches():
    """Check fit ramps chunked over ints matches."""
    rng = np.random.default_rng(35)
    data, gdq = _random_cube(rng, (5, 4, 6, 8))
    rn, gain = np.float32(6.), np.float32(1.6)
    median_rate = np.asarray(ramp.median_rates_per_integration(
        data, gdq, GT)).mean(axis=0)
    whole = ramp.fit_ramps(
        data, gdq, rn, gain, GT, median_rate=median_rate)
    chunked = core.map_over_ints(
        lambda d, g: ramp.fit_ramps(
            d, g, rn, gain, GT, median_rate=median_rate), (data, gdq),
        n_ints=5, chunk_size=2)
    for w, c in zip(whole[:2], chunked[:2]):
        assert_allclose(np.asarray(w), np.asarray(c), rtol=0., atol=0.,
                        equal_nan=True)
    assert np.array_equal(np.asarray(whole[2]), np.asarray(chunked[2]))
