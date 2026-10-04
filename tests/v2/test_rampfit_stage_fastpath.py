"""Exact-parity tests for the bounded/fused RampFit stage fast path."""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from exotedrf.v2 import core, stages
from exotedrf.v2.kernels import ramp


def _assert_float_bits_equal(actual, expected):
    """Compare floating outputs including signed zero and NaN payload bits."""
    actual = np.ascontiguousarray(np.asarray(actual))
    expected = np.ascontiguousarray(np.asarray(expected))
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    uint_dtype = np.dtype(f'u{actual.dtype.itemsize}')
    np.testing.assert_array_equal(
        actual.view(uint_dtype), expected.view(uint_dtype))


def _required_segment_loops_per_ramp(groupdq):
    """Independent NumPy statement of each ramp's useful-segment bound."""
    groupdq = np.asarray(groupdq, dtype=np.uint8)
    if groupdq.shape[1] == 1:
        return np.ones(groupdq.shape[:1] + groupdq.shape[2:], np.uint8)

    suppressed = groupdq.copy()
    one_good = np.sum(suppressed == 0, axis=1) == 1
    suppressed = np.where(
        one_good[:, None],
        suppressed | np.uint8(core.DQ_DO_NOT_USE), suppressed)

    pure_jump = suppressed == np.uint8(core.DQ_JUMP_DET)
    exact_good = suppressed == 0
    next_good = np.concatenate(
        (exact_good[:, 1:], np.zeros_like(exact_good[:, :1])), axis=1)
    usable = exact_good | (pure_jump & next_good)
    usable &= ~(
        (suppressed[:, :1] & np.uint8(core.DQ_SATURATED)) != 0)

    previous = np.concatenate(
        (np.ones_like(usable[:, :1]), usable[:, :-1]), axis=1)
    boundary = usable & (pure_jump | ~previous)
    boundary[:, 0] = False
    segment_id = np.cumsum(boundary, axis=1, dtype=np.uint8)

    maximum = np.ones(groupdq.shape[:1] + groupdq.shape[2:], np.uint8)
    for segment in range(groupdq.shape[1]):
        count = np.sum(
            usable & (segment_id == segment), axis=1, dtype=np.uint8)
        maximum = np.where(
            count >= 2, np.maximum(maximum, segment + 1), maximum)
    return maximum


def test_segment_bound_exhaustive_supported_groupdq_topologies():
    """The host scan never drops a segment that can affect a fit."""
    codes = np.asarray([
        0,
        core.DQ_JUMP_DET,
        core.DQ_DO_NOT_USE,
        core.DQ_SATURATED,
        core.DQ_DROPOUT,
        core.DQ_JUMP_DET | core.DQ_DO_NOT_USE,
    ], dtype=np.uint8)
    sequences = np.asarray(
        list(itertools.product(codes, repeat=7)), dtype=np.uint8)
    groupdq = sequences[:, :, None, None]

    bounds = _required_segment_loops_per_ramp(groupdq)[:, 0, 0]

    for expected in np.unique(bounds):
        selected = groupdq[bounds == expected]
        data = np.zeros(selected.shape, dtype=np.float32)
        _, invalid, device_bound, exceptions = \
            ramp.rampfit_chunk_statistics(
            data, selected, np.float32(2.214))
        assert not bool(np.asarray(invalid))
        assert int(np.asarray(device_bound)) == int(expected)
        expected_exceptions = bounds[bounds == expected] > 1
        np.testing.assert_array_equal(
            np.asarray(exceptions)[:, 0, 0], expected_exceptions)


def test_bounded_fit_is_bitwise_equal_to_all_segment_iterations():
    """Check bounded fit is bitwise equal to all segment iterations."""
    rng = np.random.default_rng(0x1352770)
    nints, ngroups, dimy, dimx = 4, 9, 3, 6
    increments = rng.normal(
        20., 3., size=(nints, ngroups, dimy, dimx)).astype(np.float32)
    data = np.cumsum(increments, axis=1, dtype=np.float32)
    groupdq = np.zeros(data.shape, dtype=np.uint8)

    # Three useful semiramps establish a bound of three.
    groupdq[0, 2, 0, 0] = np.uint8(core.DQ_JUMP_DET)
    groupdq[0, 5, 0, 0] = np.uint8(core.DQ_JUMP_DET)
    # A leading gap means the first useful semiramp can have ID one.
    groupdq[1, :2, 0, 1] = np.uint8(core.DQ_DO_NOT_USE)
    # Later one-read semiramps must not increase the useful bound.
    groupdq[2, 2::2, 0, 2] = np.uint8(core.DQ_DO_NOT_USE)
    # Non-finite data are accepted only when already excluded by DQ.
    data[3, 4, 1, 1] = np.nan
    groupdq[3, 4, 1, 1] = np.uint8(core.DQ_SATURATED)

    maximum = int(np.asarray(ramp.rampfit_chunk_statistics(
        data, groupdq, np.float32(2.214))[2]))
    assert maximum == int(_required_segment_loops_per_ramp(groupdq).max())
    assert maximum == 3

    readnoise = rng.uniform(4., 8., (dimy, dimx)).astype(np.float32)
    gain = rng.uniform(1.2, 1.8, (dimy, dimx)).astype(np.float32)
    median_rate = np.asarray(ramp.median_rates_per_integration(
        data, groupdq, np.float32(2.214))).mean(axis=0)
    kwargs = dict(
        median_rate=median_rate,
        nframes=np.float32(2.),
        average_dark_current=np.float32(0.13))

    full = ramp.fit_ramps(
        data, groupdq, readnoise, gain, np.float32(2.214),
        max_segments=ngroups, **kwargs)
    bounded = ramp.fit_ramps(
        data, groupdq, readnoise, gain, np.float32(2.214),
        max_segments=maximum, **kwargs)

    _assert_float_bits_equal(bounded[0], full[0])
    _assert_float_bits_equal(bounded[1], full[1])
    np.testing.assert_array_equal(np.asarray(bounded[2]), np.asarray(full[2]))


def test_fused_stage_wrapper_is_bitwise_equal_to_legacy_composition():
    """Check fused stage wrapper is bitwise equal to legacy composition."""
    rng = np.random.default_rng(7319)
    nints, ngroups, dimy, dimx = 3, 9, 4, 5
    data = np.cumsum(
        rng.uniform(10., 25., (nints, ngroups, dimy, dimx)).astype(
            np.float32),
        axis=1, dtype=np.float32)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    groupdq[0, 4, 0, 0] = np.uint8(core.DQ_JUMP_DET)
    groupdq[0, 7:, 0, 1] = np.uint8(core.DQ_SATURATED)
    groupdq[1, 1:, 0, 2] = np.uint8(core.DQ_SATURATED)
    groupdq[2, 3, 1, 0] = np.uint8(core.DQ_DO_NOT_USE)

    pixeldq = np.zeros((dimy, dimx), dtype=np.uint32)
    pixeldq[2, 1] = core.DQ_DO_NOT_USE | core.DQ_HOT
    pixeldq[2, 2] = core.DQ_SATURATED
    pixeldq[3, 4] = core.DQ_HOT
    readnoise = rng.uniform(4., 8., (dimy, dimx)).astype(np.float32)
    gain = rng.uniform(1.2, 1.8, (dimy, dimx)).astype(np.float32)
    maximum = int(np.asarray(ramp.rampfit_chunk_statistics(
        data, groupdq, np.float32(2.214))[2]))
    median_rate = np.asarray(ramp.median_rates_per_integration(
        data, groupdq, np.float32(2.214))).mean(axis=0)
    fit_kwargs = dict(
        median_rate=median_rate,
        nframes=np.float32(2.),
        average_dark_current=np.float32(0.07),
        max_segments=maximum)

    legacy_rate, legacy_err, legacy_dq = ramp.fit_ramps(
        data, groupdq, readnoise, gain, np.float32(2.214), **fit_kwargs)
    invalid_pixel = ((jnp.asarray(pixeldq) & (
        core.DQ_DO_NOT_USE | core.DQ_SATURATED)) != 0)
    legacy_rate = jnp.where(invalid_pixel[None], jnp.nan, legacy_rate)
    legacy_err = jnp.where(invalid_pixel[None], 0., legacy_err)
    saturated = ((jnp.asarray(groupdq).astype(jnp.uint32) &
                  core.DQ_SATURATED) != 0)
    sat_any = jnp.any(saturated, axis=1)
    first_sat = jnp.argmax(saturated, axis=1)
    sat_early = sat_any & (first_sat < 2)
    legacy_dq = legacy_dq | jnp.where(
        sat_any, core.DQ_SATURATED, jnp.uint32(0))
    legacy_dq = legacy_dq | jnp.where(
        sat_early, core.DQ_DO_NOT_USE, jnp.uint32(0))
    legacy_dq = legacy_dq | jnp.asarray(pixeldq)[None]

    fused = ramp.fit_ramps_stage(
        data, groupdq, readnoise, gain, np.float32(2.214), pixeldq,
        **fit_kwargs)
    _assert_float_bits_equal(fused[0], legacy_rate)
    _assert_float_bits_equal(fused[1], legacy_err)
    np.testing.assert_array_equal(np.asarray(fused[2]), np.asarray(legacy_dq))


@pytest.mark.parametrize('flag', [0, core.DQ_JUMP_DET])
def test_segment_scan_rejects_unexcluded_nonfinite_science(flag):
    """Check segment scan rejects unexcluded nonfinite science."""
    data = np.zeros((2, 4, 2, 3), dtype=np.float32)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    data[1, 2, 0, 1] = np.nan
    groupdq[1, 2, 0, 1] = np.uint8(flag)

    with pytest.raises(ValueError, match='non-finite SCI'):
        stages._segment_median_rate(data, groupdq, 2.214, return_topology=True)


@pytest.mark.parametrize('flag', [core.DQ_DO_NOT_USE, core.DQ_SATURATED])
def test_segment_scan_accepts_dq_excluded_nonfinite_science(flag):
    """Check segment scan accepts DQ excluded nonfinite science."""
    data = np.zeros((2, 4, 2, 3), dtype=np.float32)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    data[1, 2, 0, 1] = np.nan
    groupdq[1, 2, 0, 1] = np.uint8(flag)

    _, maximum, _ = stages._segment_median_rate(
        data, groupdq, 2.214, return_topology=True)
    assert maximum >= 1
