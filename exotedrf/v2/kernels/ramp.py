"""Up-the-ramp jump detection and weighted ramp fitting."""

import functools

import jax
import jax.numpy as jnp

from exotedrf.v2 import core

_BAD_GROUP = jnp.uint8(int(core.DQ_SATURATED) | int(core.DQ_DO_NOT_USE))
_DQ_NO_GAIN_VALUE = jnp.uint32(524288)
_LARGE_VARIANCE_THRESHOLD = 1.e6


@jax.jit
def flag_jumps_upramp(data, groupdq, readnoise, rejection_threshold=15.):
    """Flag jumps using a single pass over group differences.

    Parameters
    ----------
    data, groupdq : array-like(float), array-like(int)
        Detector samples and group flags with shape (nints, ngroups, dimy, dimx).
    readnoise : array-like(float), float
        Single-read noise in DN.
    rejection_threshold : float
        Jump detection threshold in scatter units. Traced.

    Returns
    -------
    groupdq : array-like(int)
        Updated group flags with the input shape.
    """
    ngroups = data.shape[1]
    if ngroups <= 2:
        return groupdq
    ndiffs = ngroups - 1
    bad = ((groupdq & _BAD_GROUP) != 0) | ~jnp.isfinite(data)
    diffs = data[:, 1:] - data[:, :-1]
    diffs = jnp.where(bad[:, 1:] | bad[:, :-1], jnp.nan, diffs)
    if ndiffs == 2:
        base = jnp.nanmin(diffs, axis=1)
    else:
        base = jnp.nanmedian(diffs, axis=1)
    sigma = jnp.sqrt(jnp.clip(base, 0., None) + 2. * readnoise ** 2)
    if ndiffs >= 4:
        mad = jnp.nanmedian(jnp.abs(diffs - base[:, None]), axis=1)
        sigma = jnp.maximum(sigma, 1.4826 * mad)
    ratio = jnp.abs(diffs - base[:, None]) / sigma[:, None]
    jump = ratio > rejection_threshold
    flags = jnp.where(jump, jnp.uint8(core.DQ_JUMP_DET), jnp.uint8(0))
    flags = jnp.concatenate([jnp.zeros_like(flags[:, :1]), flags], axis=1)
    return (groupdq | flags).astype(groupdq.dtype)


def _suppress_one_good_group(groupdq, suppress=True):
    """Flag ramps with a single good group when suppression is enabled."""
    if not suppress or groupdq.shape[1] == 1:
        return groupdq
    one_good = jnp.sum(groupdq == 0, axis=1) == 1
    return jnp.where(one_good[:, None], groupdq | jnp.asarray(core.DQ_DO_NOT_USE, groupdq.dtype),
        groupdq)


def _ramp_segments(data, groupdq):
    """Mark usable reads and segment boundaries, excluding early saturation."""
    pure_jump = groupdq == jnp.asarray(core.DQ_JUMP_DET, groupdq.dtype)
    exact_good = groupdq == 0
    next_good = jnp.concatenate((exact_good[:, 1:], jnp.zeros_like(exact_good[:, :1])), axis=1)
    usable = (exact_good | (pure_jump & next_good)) & jnp.isfinite(data)
    first_saturated = (groupdq[:, 0] & jnp.asarray(core.DQ_SATURATED, groupdq.dtype)) != 0
    usable &= ~first_saturated[:, None]
    previous = jnp.concatenate((jnp.ones_like(usable[:, :1]), usable[:, :-1]), axis=1)
    boundary = usable & (pure_jump | ~previous)
    # Assign the first usable read to segment zero.
    boundary = jnp.concatenate((jnp.zeros_like(boundary[:, :1]), boundary[:, 1:]), axis=1)
    return usable, boundary, jnp.cumsum(boundary.astype(jnp.int32), axis=1), first_saturated


@functools.partial(jax.jit, static_argnames=('suppress_one_group',))
def median_rates_per_integration(data, groupdq, group_time,
                                 one_group_time=None, suppress_one_group=True):
    """Calculate median rates per integration for Poisson variance.

    Parameters
    ----------
    data, groupdq : array-like(float), array-like(int)
        Detector samples and group flags with shape (nints, ngroups, dimy, dimx).
    group_time : float
        Elapsed seconds per group. Traced.
    one_group_time : None, float
        Elapsed seconds for a single-group ramp; None uses group_time. Traced.
    suppress_one_group : bool
        Suppress single good groups in multi-group ramps. Static: changes trigger a recompile.

    Returns
    -------
    medians : array-like(float)
        Median rate planes with shape (nints, dimy, dimx).
    """
    groupdq = _suppress_one_good_group(groupdq, suppress_one_group)
    invalid = ((groupdq & _BAD_GROUP) != 0) | ~jnp.isfinite(data)
    if data.shape[1] == 1 and one_group_time is not None:
        rate_time = one_group_time
    else:
        rate_time = group_time
    scaled = jnp.where(invalid, jnp.nan, data / rate_time)
    if data.shape[1] == 1:
        diffs = scaled
    else:
        diffs = scaled[:, 1:] - scaled[:, :-1]
        # Use the first read relative to reset when the first difference is unavailable.
        first = jnp.where(jnp.isnan(diffs[:, 0]), scaled[:, 0], diffs[:, 0])
        diffs = jnp.concatenate((first[:, None], diffs[:, 1:]), axis=1)
        jump = ((groupdq[:, 1:] & jnp.asarray(core.DQ_JUMP_DET, groupdq.dtype)) != 0)
        diffs = jnp.where(jump, jnp.nan, diffs)
        all_nan = ~jnp.isfinite(diffs).any(axis=1)
        first = jnp.where(all_nan & jnp.isfinite(scaled[:, 0]), scaled[:, 0], diffs[:, 0])
        diffs = jnp.concatenate((first[:, None], diffs[:, 1:]), axis=1)
    med = jnp.nanmedian(diffs, axis=1)
    return jnp.where(jnp.isfinite(med), med, 0.).astype(data.dtype)


@functools.partial(jax.jit, static_argnames=('suppress_one_group',))
def rampfit_chunk_statistics(data, groupdq, group_time,
                             one_group_time=None, suppress_one_group=True):
    """Calculate median rates and ramp segmentation statistics for a chunk.

    Parameters
    ----------
    data, groupdq : array-like(float), array-like(int)
        Detector samples and group flags with shape (nints, ngroups, dimy, dimx).
    group_time : float
        Elapsed seconds per group. Traced.
    one_group_time : None, float
        Elapsed seconds for a single-group ramp; None uses group_time. Traced.
    suppress_one_group : bool
        Suppress single good groups in multi-group ramps. Static: changes trigger a recompile.

    Returns
    -------
    medians : array-like(float)
        Per-integration median rate planes.
    invalid_nonfinite : bool
        True when non-finite samples lack exclusion flags.
    maximum : int
        Maximum useful segment count.
    exception_mask : array-like(bool)
        Pixels with a useful multi-read segment after the first segment.
    """
    medians = median_rates_per_integration(data, groupdq, group_time, one_group_time,
        suppress_one_group=suppress_one_group)
    finite = jnp.isfinite(data)
    excluded = ((groupdq & _BAD_GROUP) != 0)
    invalid_nonfinite = jnp.any(~finite & ~excluded)
    suppressed = _suppress_one_good_group(groupdq, suppress_one_group)
    usable, boundary, segment_id, _ = _ramp_segments(data, suppressed)

    if data.shape[1] < 3:
        exception_mask = jnp.zeros(data.shape[:1] + data.shape[2:], bool)
        maximum = jnp.asarray(1, dtype=jnp.int32)
    else:
        # Count later segments only when another usable read follows their boundary.
        later_start = (boundary[:, 1:-1] & usable[:, 2:] & ~boundary[:, 2:])
        exception_mask = jnp.any(later_start, axis=1)
        required = jnp.where(later_start, segment_id[:, 1:-1] + 1,
            jnp.ones_like(segment_id[:, 1:-1]))
        maximum = jnp.max(required).astype(jnp.int32)
    return medians, invalid_nonfinite, maximum, exception_mask


def _fixsen_power(snr):
    """Get the piecewise SNR exponent used for OLS weights."""
    power = jnp.zeros_like(snr)
    power = jnp.where(snr >= 5., .4, power)
    power = jnp.where(snr >= 10., 1., power)
    power = jnp.where(snr >= 20., 3., power)
    power = jnp.where(snr >= 50., 6., power)
    return jnp.where(snr >= 100., 10., power)


def _segment_power(first, last, readnoise, gain, nframes):
    """Calculate segment SNR weights using double precision reference scaling."""
    # Match the reference scaling precision used by the C fitter.
    with jax.enable_x64():
        gain64 = gain.astype(jnp.float64)
        noise = (readnoise.astype(jnp.float64) * gain64 /
                 jnp.sqrt(2. * nframes.astype(jnp.float64)))
        noise = noise.astype(readnoise.dtype).astype(jnp.float64)
        signal = (last.astype(jnp.float64) - first.astype(jnp.float64)) * gain64
        variance = noise ** 2 + signal
        snr = jnp.where(variance > 0., signal / jnp.sqrt(jnp.maximum(variance, 0.)), 0.)
        return _fixsen_power(jnp.maximum(snr, 0.)).astype(first.dtype)


@functools.partial(jax.jit, static_argnames=('max_segments', 'suppress_one_group'))
def fit_ramps(data, groupdq, readnoise, gain, group_time,
              median_rate=None, nframes=1, average_dark_current=0.,
              one_group_time=None, max_segments=None, suppress_one_group=True):
    """Fit ramp segments with optimal OLS weights and combine their rates.

    Combine slopes with inverse read-noise variance and errors with inverse total variance.

    Parameters
    ----------
    data, groupdq : array-like(float), array-like(int)
        Detector samples and group flags with shape (nints, ngroups, dimy, dimx).
    readnoise, gain : array-like(float), float
        Single-read noise in DN and gain in electrons per DN, respectively.
    group_time : float
        Elapsed seconds per group. Traced.
    median_rate : None, array-like(float)
        Full-segment mean of integration medians, including when chunking integrations.
    nframes : int
        Reads averaged into each group. Traced.
    average_dark_current : array-like(float), float
        Mean dark current in DN per second. Traced.
    one_group_time : None, float
        Elapsed seconds for a single-group ramp; None uses group_time. Traced.
    max_segments : None, int
        Maximum segments; None uses ngroups. Static: changes trigger a recompile.
    suppress_one_group : bool
        Suppress single good groups in multi-group ramps. Static: changes trigger a recompile.

    Returns
    -------
    rate, err, dq : array-like(float), array-like(int)
        Integration rates, uncertainties and collapsed uint32 quality flags, respectively.
    """
    ngroups = data.shape[1]
    if max_segments is None:
        max_segments = ngroups
    max_segments = int(max_segments)
    if not 1 <= max_segments <= ngroups:
        raise ValueError(f'max_segments must be between 1 and {ngroups}, got ' f'{max_segments}')
    dtype = data.dtype
    groupdq = _suppress_one_good_group(groupdq, suppress_one_group)
    if median_rate is None:
        median_rate = jnp.mean(median_rates_per_integration(
                data, groupdq, group_time, one_group_time,
                suppress_one_group=suppress_one_group), axis=0)
    med_rate = jnp.asarray(median_rate, dtype)
    average_dark_current = jnp.asarray(average_dark_current, dtype)
    gain = jnp.asarray(gain, dtype)
    readnoise = jnp.asarray(readnoise, dtype)
    nframes = jnp.asarray(nframes, dtype)
    group_time = jnp.asarray(group_time, dtype)
    good_gain = jnp.isfinite(gain) & (gain > 0.)
    good_rn = jnp.isfinite(readnoise) & (readnoise >= 0.)
    good_timing = (jnp.isfinite(group_time) & (group_time > 0.) &
                   jnp.isfinite(nframes) & (nframes > 0.))
    good_cal = good_gain & good_rn & good_timing
    safe_gain = jnp.where(good_gain, gain, 1.)
    safe_rn = jnp.where(good_rn, readnoise, 0.)
    safe_nf = jnp.where(nframes > 0., nframes, 1.)
    safe_gt = jnp.where(group_time > 0., group_time, 1.)
    jump_bit = jnp.asarray(core.DQ_JUMP_DET, groupdq.dtype)
    usable, _, seg_id, first_saturated = _ramp_segments(data, groupdq)
    group_index = jnp.arange(ngroups, dtype=dtype)[None, :, None, None]
    output_shape = data.shape[:1] + data.shape[2:]
    slope_num = jnp.zeros(output_shape, dtype)
    inv_total = jnp.zeros_like(slope_num)
    inv_rnoise = jnp.zeros_like(slope_num)
    has_segment = jnp.zeros(output_shape, bool)
    ref_slope = jnp.zeros_like(slope_num)
    rn2 = safe_rn ** 2

    for segment in range(max_segments):
        mask = usable & (seg_id == segment)
        count = mask.sum(axis=1).astype(dtype)
        good_segment = count >= 2.
        safe_count = jnp.where(good_segment, count, 2.)
        first_index = jnp.argmax(mask, axis=1)
        last_index = ngroups - 1 - jnp.argmax(mask[:, ::-1], axis=1)
        first_value = jnp.take_along_axis(data, first_index[:, None], axis=1)[:, 0]
        last_value = jnp.take_along_axis(data, last_index[:, None], axis=1)[:, 0]
        power = _segment_power(first_value, last_value, safe_rn, safe_gain, safe_nf)
        first_f = first_index.astype(dtype)
        local_index = group_index - first_f[:, None]
        midpoint = (safe_count - 1.) / 2.
        relative = (jnp.abs(local_index - midpoint[:, None]) / midpoint[:, None])
        optimal_weight = relative ** power[:, None]
        fit_weight = jnp.where(mask, jnp.where((count > 2.)[:, None], optimal_weight, 1.), 0.)
        weight_sum = fit_weight.sum(axis=1)
        safe_weight_sum = jnp.where(weight_sum > 0., weight_sum, 1.)
        x_mean = (fit_weight * group_index).sum(axis=1) / safe_weight_sum
        clean_data = jnp.where(mask, data, 0.)
        y_mean = (fit_weight * clean_data).sum(axis=1) / safe_weight_sum
        dx = group_index - x_mean[:, None]
        dy = jnp.where(mask, data - y_mean[:, None], 0.)
        denominator = (fit_weight * dx * dx).sum(axis=1)
        covariance = (fit_weight * dx * dy).sum(axis=1)
        good_segment = good_segment & (denominator > 0.) & good_cal
        slope = jnp.where(good_segment, covariance / jnp.where(
                denominator > 0., denominator, 1.) / safe_gt, 0.)
        n_minus_one = jnp.maximum(count - 1., 1.)
        n3_minus_n = jnp.where(count > 1., count ** 3 - count, 6.)
        var_r = (6. * rn2 / (safe_nf * n3_minus_n * safe_gt ** 2))
        var_p = ((jnp.maximum(med_rate, 0.) + average_dark_current) /
                 (safe_gt * safe_gain * n_minus_one))
        positive_p = jnp.isfinite(var_p) & (var_p > 0.)
        # Use read-noise-only weighting for a zero Poisson estimate.
        weight_var_p = jnp.where(positive_p, var_p, 0.)
        total_var = var_r + weight_var_p
        segment_weight = jnp.where(good_segment & jnp.isfinite(total_var) & (total_var > 0.),
            1. / total_var, 0.)
        # Accumulate slope differences to preserve equal segment slopes exactly.
        ref_slope = jnp.where(good_segment & ~has_segment, slope, ref_slope)
        rnoise_weight = jnp.where(good_segment, 1. / var_r, 0.)
        slope_num += rnoise_weight * (slope - ref_slope)
        inv_rnoise += rnoise_weight
        inv_total += segment_weight
        has_segment |= good_segment

    safe_inv_rnoise = jnp.where(has_segment, inv_rnoise, 1.)
    # Compare inverse variance directly to preserve the rate operation order.
    large_rnoise = safe_inv_rnoise <= 1. / _LARGE_VARIANCE_THRESHOLD
    rate = jnp.where(has_segment, jnp.where(large_rnoise, 0. * slope_num,
        ref_slope + slope_num / safe_inv_rnoise), jnp.nan)
    has_total = inv_total > 0.
    total_var_int = jnp.where(has_total, 1. / jnp.where(has_total, inv_total, 1.), 0.)
    # Retain the first singleton only when no multi-read segment survives.
    any_usable = usable.any(axis=1)
    first_usable = jnp.argmax(usable, axis=1)
    first_value = jnp.take_along_axis(data, first_usable[:, None], axis=1)[:, 0]
    singleton_fallback = ~has_segment & any_usable & good_cal
    singleton_rate = first_value / safe_gt
    singleton_var = ((jnp.maximum(med_rate, 0.) + average_dark_current) /
        (safe_gt * safe_gain) + rn2 / (safe_nf * safe_gt ** 2))
    rate = jnp.where(singleton_fallback, singleton_rate, rate)
    total_var_int = jnp.where(singleton_fallback, singleton_var, total_var_int)
    has_segment |= singleton_fallback

    if ngroups == 1:
        single = usable[:, 0] & good_cal
        if one_group_time is None:
            single_time = safe_gt
        else:
            single_time = jnp.asarray(one_group_time, dtype)
            single_time = jnp.where(single_time > 0., single_time, safe_gt)
        rate = jnp.where(single, data[:, 0] / single_time, jnp.nan)
        total_var_int = jnp.where(single, (jnp.maximum(med_rate, 0.) + average_dark_current) /
            (single_time * safe_gain) + rn2 / (safe_nf * single_time ** 2), 0.)
        has_segment = single

    err = jnp.sqrt(total_var_int)
    all_bad = ((groupdq & _BAD_GROUP) != 0).all(axis=1)
    any_jump = ((groupdq & jump_bit) != 0).any(axis=1)
    # Return zero for absent slopes whose flags do not exclude the ramp.
    empty_flagged = ~has_segment & ~all_bad & ~first_saturated & good_cal
    rate = jnp.where(empty_flagged, 0., rate)
    err = jnp.where(empty_flagged, 0., err)
    dq = jnp.where(any_jump, core.DQ_JUMP_DET, jnp.uint32(0))
    any_sat = ((groupdq & jnp.asarray(core.DQ_SATURATED, groupdq.dtype)) != 0).any(axis=1)
    dq |= jnp.where(any_sat, core.DQ_SATURATED, jnp.uint32(0))
    dq |= jnp.where(all_bad, core.DQ_DO_NOT_USE, jnp.uint32(0))
    dq |= jnp.where(first_saturated, core.DQ_SATURATED | core.DQ_DO_NOT_USE, jnp.uint32(0))
    dq |= jnp.where(~good_gain, _DQ_NO_GAIN_VALUE | core.DQ_DO_NOT_USE, jnp.uint32(0))
    dq |= jnp.where(~good_rn | ~good_timing, core.DQ_DO_NOT_USE, jnp.uint32(0))
    return rate.astype(dtype), err.astype(dtype), dq.astype(jnp.uint32)


def _finalize_ramps_stage(rate, err, dq, groupdq, pixeldq, suppress_one_group):
    """Apply pixel flags and early saturation handling to fitted ramps."""
    invalid_pixel = ((pixeldq & (core.DQ_DO_NOT_USE | core.DQ_SATURATED)) != 0)
    rate = jnp.where(invalid_pixel, jnp.nan, rate)
    err = jnp.where(invalid_pixel, 0., err)
    saturated = ((groupdq.astype(jnp.uint32) & core.DQ_SATURATED) != 0)
    sat_any = jnp.any(saturated, axis=1)
    first_sat = jnp.argmax(saturated, axis=1)
    sat_early = sat_any & (first_sat < (2 if suppress_one_group else 1))
    dq = dq | jnp.where(sat_any, core.DQ_SATURATED, jnp.uint32(0))
    dq = dq | jnp.where(sat_early, core.DQ_DO_NOT_USE, jnp.uint32(0))
    return rate, err, dq | pixeldq


@functools.partial(jax.jit, static_argnames=('max_segments', 'suppress_one_group'))
def fit_ramps_stage(data, groupdq, readnoise, gain, group_time, pixeldq,
                    median_rate=None, nframes=1, average_dark_current=0.,
                    one_group_time=None, max_segments=None, suppress_one_group=True):
    """Fit ramps and apply pixel flags and early saturation handling.

    Other parameters follow ``fit_ramps``.

    Parameters
    ----------
    pixeldq : array-like(int)
        Pixel flags with shape (dimy, dimx).

    Returns
    -------
    rate, err, dq : array-like(float), array-like(int)
        Rates (invalid pixels NaN), errors (invalid pixels zero) and combined uint32 flags.
    """
    rate, err, dq = fit_ramps(data, groupdq, readnoise, gain, group_time,
        median_rate=median_rate, nframes=nframes, average_dark_current=average_dark_current,
        one_group_time=one_group_time, max_segments=max_segments,
        suppress_one_group=suppress_one_group)
    return _finalize_ramps_stage(
        rate, err, dq, groupdq, jnp.asarray(pixeldq, jnp.uint32)[None], suppress_one_group)


@functools.partial(jax.jit, static_argnames=('max_segments', 'suppress_one_group'))
def fit_ramps_packed_stage(data, groupdq, readnoise, gain, group_time,
                           pixeldq, median_rate=None, nframes=1,
                           average_dark_current=0., one_group_time=None,
                           max_segments=None, suppress_one_group=True):
    """Fit independently gathered ramps and apply their pixel flags.

    Other parameters follow ``fit_ramps``.

    Parameters
    ----------
    data, groupdq : array-like(float), array-like(int)
        Gathered samples and group flags with shape (nramps, ngroups, 1, 1).
    pixeldq : array-like(int)
        Gathered pixel flags with shape (nramps, 1, 1).
    median_rate : None, array-like(float)
        Segment-wide median rate estimates gathered to shape (nramps, 1, 1).

    Returns
    -------
    rate, err, dq : array-like(float), array-like(int)
        Integration rates, uncertainties and collapsed uint32 quality flags, respectively.
    """
    rate, err, dq = fit_ramps(data, groupdq, readnoise, gain, group_time,
        median_rate=median_rate, nframes=nframes, average_dark_current=average_dark_current,
        one_group_time=one_group_time, max_segments=max_segments,
        suppress_one_group=suppress_one_group)
    return _finalize_ramps_stage(
        rate, err, dq, groupdq, jnp.asarray(pixeldq, jnp.uint32), suppress_one_group)
