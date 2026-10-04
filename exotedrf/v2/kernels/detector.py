"""Detector calibration and flat-field correction kernels."""

from functools import partial

import jax
import jax.numpy as jnp

from exotedrf.v2 import core
from exotedrf.v2.mathutils import (bitwise_or_reduce_uint, dilate_mask, nan_stats_filter_2d)


DQ_NO_FLAT_FIELD = jnp.uint32(1 << 18)
DQ_NO_LIN_CORR = jnp.uint32(1 << 20)


def _center_excluded_std(img, med, box_half):
    """Calculate local scatter excluding the central pixel."""
    size = 2 * box_half + 1
    dimy, dimx = img.shape[-2], img.shape[-1]
    pad = ((0, 0),) * (img.ndim - 2) + ((box_half, box_half), (box_half, box_half))
    padded = jnp.pad(img, pad, constant_values=jnp.nan)
    # Accumulate centered moments to retain float32 precision.
    n = jnp.zeros(img.shape, jnp.float32)
    s1 = jnp.zeros(img.shape, jnp.float32)
    s2 = jnp.zeros(img.shape, jnp.float32)
    for dy in range(size):
        for dx in range(size):
            if dy == box_half and dx == box_half:
                continue
            w = padded[..., dy:dy + dimy, dx:dx + dimx]
            f = jnp.isfinite(w)
            v = jnp.where(f, w - med, 0.)
            n = n + f
            s1 = s1 + v
            s2 = s2 + v * v
    safe_n = jnp.where(n > 0, n, 1.)
    mean = s1 / safe_n
    var = s2 / safe_n - mean * mean
    return jnp.where(n > 0, jnp.sqrt(jnp.clip(var, 0., None)), jnp.nan)


def _dilate_mask_clipped(mask, box_half):
    """Expand a mask without wrapping across detector edges."""
    if box_half < 0:
        raise ValueError('box_half must be non-negative')
    if box_half == 0:
        return mask
    pad = ((0, 0),) * (mask.ndim - 2) + ((box_half, box_half), (box_half, box_half))
    padded = jnp.pad(mask, pad, constant_values=False)
    dilated = dilate_mask(padded, box_half)
    return dilated[..., box_half:-box_half, box_half:-box_half]


@jax.jit
def integral_nonlinearity_correct(data, theta, periods):
    """Correct integral non-linearity using the Fourier count model.

    Parameters
    ----------
    data : array-like(float)
        Detector samples with shape (nints, ngroups, dimy, dimx).
    theta : array-like(float)
        Alternating sine and cosine amplitudes, with two coefficients per period.
    periods : array-like(float)
        Integral non-linearity periods in counts.

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    """
    theta = jnp.asarray(theta, dtype=data.dtype)
    periods = jnp.asarray(periods, dtype=data.dtype)
    if theta.ndim != 1 or periods.ndim != 1:
        raise ValueError('theta and periods must be one-dimensional')
    if theta.shape[0] != 2 * periods.shape[0]:
        raise ValueError('theta must contain two coefficients per INL period')
    phase = (2.0 * jnp.pi / periods).reshape((periods.shape[0],) + (1,) * data.ndim) * data[None]
    sine = theta[0::2].reshape((-1,) + (1,) * data.ndim)
    cosine = theta[1::2].reshape((-1,) + (1,) * data.ndim)
    correction = jnp.sum(sine * jnp.sin(phase) + cosine * jnp.cos(phase), axis=0)
    return data / (1.0 + correction)


@partial(jax.jit, static_argnames=('flag_neighbours',))
def flag_saturated_pixels(data, groupdq, full_well_adu,
                          saturation_threshold=80., flag_neighbours=1):
    """Flag saturated pixels, their neighbours and negative counts.

    Parameters
    ----------
    data, groupdq : array-like(float), array-like(int)
        Detector samples and group flags with shape (nints, ngroups, dimy, dimx).
    full_well_adu : float
        Detector full well in ADU. Traced.
    saturation_threshold : float
        Saturation threshold as a percentage of full well. Traced.
    flag_neighbours : int
        Half-width of the box to flag around saturated pixels. Static: changes trigger a recompile.

    Returns
    -------
    groupdq : array-like(int)
        Updated group flags with the input shape.
    sat_mask : array-like(bool)
        Expanded saturation mask matching the data.
    """
    thresh_adu = full_well_adu * (saturation_threshold / 100.)
    sat = data >= thresh_adu
    sat = _dilate_mask_clipped(sat, flag_neighbours)
    groupdq = bitwise_or_reduce_uint(
        groupdq, jnp.where(sat, jnp.uint8(core.DQ_SATURATED), jnp.uint8(0)))
    floor = data < 0.
    groupdq = bitwise_or_reduce_uint(
        groupdq, jnp.where(floor, jnp.uint8(core.DQ_AD_FLOOR), jnp.uint8(0)))
    return groupdq, sat


@partial(jax.jit, static_argnames=('box_half', 'n_edge_cols'))
def flag_hot_pixels(pixeldq, deepframe, thresh=15., box_half=10, n_edge_cols=4):
    """Flag hot pixels using local median and scatter estimates.

    Local scatter excludes the central pixel; the median includes it.

    Parameters
    ----------
    pixeldq : array-like(int)
        Pixel flags with shape (dimy, dimx).
    deepframe : array-like(float)
        Median detector image.
    thresh : float
        Rejection threshold in scatter units. Traced.
    box_half : int
        Local statistics box half-width in pixels. Static: changes trigger a recompile.
    n_edge_cols : int
        Columns excluded at each detector edge. Static: changes trigger a recompile.

    Returns
    -------
    pixeldq : array-like(int)
        Pixel flags with HOT and DO_NOT_USE on new hot pixels.
    hot_mask : array-like(bool)
        Newly flagged detector pixels.
    """
    med, _ = nan_stats_filter_2d(deepframe, box_half)
    std = _center_excluded_std(deepframe, med, box_half)
    global_med = jnp.nanmedian(deepframe)
    already = (pixeldq & (core.DQ_HOT | core.DQ_WARM)) != 0
    dimx = deepframe.shape[-1]
    col_ok = ((jnp.arange(dimx) >= n_edge_cols) & (jnp.arange(dimx) < dimx - n_edge_cols))[None, :]
    hot = (jnp.abs(deepframe - med) >= thresh * std) & (deepframe > global_med) & ~already & col_ok
    pixeldq = bitwise_or_reduce_uint(pixeldq, jnp.where(hot, core.DQ_HOT | core.DQ_DO_NOT_USE,
                           jnp.uint32(0)))
    return pixeldq, hot


@jax.jit
def make_custom_superbias(data, baseline_mask):
    """Build a superbias from the first group of baseline integrations.

    Parameters
    ----------
    data : array-like(float)
        Detector samples with shape (nints, ngroups, dimy, dimx).
    baseline_mask : array-like(bool)
        True for baseline integrations.

    Returns
    -------
    superbias : array-like(float)
        Median first-group detector frame.
    """
    group0 = jnp.where(baseline_mask[:, None, None], data[:, 0], jnp.nan)
    return jnp.nanmedian(group0, axis=0)


@jax.jit
def subtract_superbias(data, superbias):
    """Subtract a superbias reference from every detector group.

    Non-finite reference values give zero correction.

    Parameters
    ----------
    data : array-like(float)
        Detector samples with shape (nints, ngroups, dimy, dimx).
    superbias : array-like(float)
        Superbias reference with shape (dimy, dimx).

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    """
    safe_superbias = jnp.where(jnp.isfinite(superbias), superbias, 0.)
    return data - safe_superbias[None, None]


@jax.jit
def subtract_superbias_rescale(data, superbias, use_mask):
    """Scale and subtract the superbias per integration.

    Parameters
    ----------
    data : array-like(float)
        Detector samples with shape (nints, ngroups, dimy, dimx).
    superbias : array-like(float)
        Superbias reference with shape (dimy, dimx).
    use_mask : array-like(bool)
        Pixels included in the superbias scale estimate.

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    scale : array-like(float)
        Superbias scale factor per integration.
    """
    ratio = jnp.where(use_mask[None], data[:, 0] / superbias[None], jnp.nan)
    scale = jnp.nanmedian(ratio, axis=(1, 2))
    out = data - scale[:, None, None, None] * superbias[None, None]
    return out, scale


@partial(jax.jit, static_argnames=('nref_top', 'nref_bottom', 'odd_even_columns', 'nref_side',
                                   'detector_column_axis'))
def refpix_correct(data, pixeldq=None, *, nref_top=4, nref_bottom=0,
                   odd_even_columns=True, nref_side=0, detector_column_axis=-1):
    """Subtract sigma-clipped reference pixel offsets from a NIR subarray.

    Parameters
    ----------
    data : array-like(float)
        Detector samples with shape (nints, ngroups, dimy, dimx).
    pixeldq : None, array-like(int)
        Pixel flags excluding DO_NOT_USE; None uses configured sections.
    nref_top : int
        Reference rows at the top of the subarray. Static: changes trigger a recompile.
    nref_bottom : int
        Reference rows at the bottom of the subarray. Static: changes trigger a recompile.
    odd_even_columns : bool
        Separate even and odd column offsets. Static: changes trigger a recompile.
    nref_side : int
        Candidate reference columns on each side. Static: changes trigger a recompile.
    detector_column_axis : int
        Detector column axis (-1 or -2). Static: changes trigger a recompile.

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    """
    dimy, dimx = data.shape[-2:]
    coordinate = (jnp.arange(dimx)[None, :] if detector_column_axis == -1
                  else jnp.arange(dimy)[:, None])
    is_even = jnp.broadcast_to(coordinate % 2 == 0, (dimy, dimx))
    sections = []
    if nref_bottom > 0:
        sections.append((slice(0, nref_bottom), slice(None)))
    if nref_top > 0:
        sections.append((slice(dimy - nref_top, dimy), slice(None)))
    row_sections = len(sections)
    if nref_side > 0:
        sides = min(nref_side, dimx // 2)
        rows = slice(nref_bottom, dimy - nref_top)
        sections.extend([(rows, slice(0, sides)), (rows, slice(dimx - sides, dimx))])
    parts = [data[..., rows, cols].reshape(*data.shape[:-2], 1, -1) for rows, cols in sections]
    ref = jnp.concatenate(parts, axis=-1)
    parity = jnp.concatenate([is_even[rows, cols].reshape(1, -1)
                              for rows, cols in sections], axis=-1)
    valid = jnp.ones(ref.shape, dtype=bool)
    if pixeldq is not None:
        dq = jnp.concatenate([pixeldq[rows, cols].reshape(1, -1)
                              for rows, cols in sections], axis=-1)
        do_not_use = (jnp.asarray(dq, jnp.uint32) & jnp.uint32(core.DQ_DO_NOT_USE)) != 0
        reference = (jnp.asarray(dq, jnp.uint32) & jnp.uint32(1 << 31)) != 0
        # Use configured reference rows when no REFERENCE_PIXEL bits are available.
        row_mask = jnp.concatenate([
            jnp.full(is_even[rows, cols].size, i < row_sections)
            for i, (rows, cols) in enumerate(sections)])[None, :]
        eligible = jnp.where(jnp.any(reference), reference, row_mask)
        valid = jnp.broadcast_to((eligible & ~do_not_use)[None, None], ref.shape)

    def clipped_mean(values, mask):
        """Calculate a sigma-clipped reference pixel mean."""
        axes = (-2, -1)

        def mean(support):
            """Calculate the mean of retained reference pixels."""
            count = jnp.sum(support, axis=axes, keepdims=True)
            return jnp.sum(jnp.where(support, values, 0), axis=axes, keepdims=True) / count

        def clip(state):
            """Apply one round of reference pixel sigma clipping."""
            support, _ = state
            center = mean(support)
            count = jnp.sum(support, axis=axes, keepdims=True)
            variance = jnp.sum(jnp.where(support, (values - center) ** 2, 0),
                               axis=axes, keepdims=True) / count
            spread = 3 * jnp.sqrt(variance)
            retained = support & (values >= center - spread) & (values <= center + spread)
            return retained, jnp.any(retained != support)

        support, _ = jax.lax.while_loop(lambda state: state[1], clip, (mask, jnp.asarray(True)))
        return mean(support).squeeze(axis=axes)

    if odd_even_columns:
        even = clipped_mean(ref, valid & parity)
        odd = clipped_mean(ref, valid & ~parity)
        corr = jnp.where(is_even, even[..., None, None], odd[..., None, None])
    else:
        corr = clipped_mean(ref, valid)[..., None, None]
    return data - corr


@jax.jit
def linearity_correct(data, groupdq, lin_coeffs, no_lin_mask=None):
    """Apply the reference linearity polynomial to eligible pixels.

    Saturated groups and pixels flagged NO_LIN_CORR retain their input values.

    Parameters
    ----------
    data, groupdq : array-like(float), array-like(int)
        Detector samples and group flags with shape (nints, ngroups, dimy, dimx).
    lin_coeffs : array-like(float)
        Linearity coefficients with shape (ncoeffs, dimy, dimx), lowest order first.
    no_lin_mask : None, array-like(bool)
        Pixels to skip because the reference flags NO_LIN_CORR.

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    """
    ncoeffs = lin_coeffs.shape[0]
    acc = jnp.broadcast_to(lin_coeffs[ncoeffs - 1], data.shape)
    for k in range(ncoeffs - 2, -1, -1):
        acc = acc * data + lin_coeffs[k]
    skip = (groupdq & jnp.uint8(core.DQ_SATURATED)) != 0
    if no_lin_mask is not None:
        skip = skip | no_lin_mask[None, None]
    return jnp.where(skip, data, acc)


@jax.jit
def prepare_linearity_reference(lin_coeffs, lin_dq):
    """Prepare finite linearity coefficients and reference quality flags.

    Parameters
    ----------
    lin_coeffs : array-like(float)
        Linearity coefficients with shape (ncoeffs, dimy, dimx), lowest order first.
    lin_dq : array-like(int)
        Linearity reference pixel flags.

    Returns
    -------
    safe_coeffs : array-like(float)
        Coefficients with non-finite values replaced by zero.
    no_lin_mask : array-like(bool)
        Pixels flagged NO_LIN_CORR.
    propagated_dq : array-like(int)
        Reference flags including invalid-coefficient flags.
    """
    invalid = ~jnp.all(jnp.isfinite(lin_coeffs), axis=0)
    if lin_coeffs.shape[0] > 1:
        invalid = invalid | (lin_coeffs[1] == 0.)
    propagated_dq = jnp.asarray(lin_dq, jnp.uint32) | jnp.where(
        invalid, DQ_NO_LIN_CORR, jnp.uint32(0))
    no_lin_mask = (propagated_dq & DQ_NO_LIN_CORR) != 0
    safe_coeffs = jnp.where(jnp.isfinite(lin_coeffs), lin_coeffs, 0.)
    return safe_coeffs, no_lin_mask, propagated_dq


@partial(jax.jit, static_argnames=('nframes',))
def linearity_correct_reads(data, groupdq, lin_coeffs, inv_coeffs, nframes):
    """Correct individual reads inferred from group-averaged counts.

    Parameters
    ----------
    data, groupdq : array-like(float), array-like(int)
        Detector samples and group flags with shape (nints, ngroups, dimy, dimx).
    lin_coeffs, inv_coeffs : array-like(float)
        Linearity and inverse polynomial coefficients, lowest order first.
    nframes : int
        Reads averaged into each group. Static: changes trigger a recompile.

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    """
    ngroups = data.shape[1]
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    times = (jnp.arange(ngroups, dtype=dtype) * nframes + (nframes + 1.) / 2.)
    saturated = (groupdq & jnp.uint8(core.DQ_SATURATED)) != 0
    last = jnp.maximum(jnp.sum(~saturated, axis=1) - 1, 2) - 1
    last_data = jnp.take_along_axis(data, last[:, None], axis=1)
    last_dq = jnp.take_along_axis(groupdq, last[:, None], axis=1)
    first = linearity_correct(data[:, :1], groupdq[:, :1], lin_coeffs)
    final = linearity_correct(last_data, last_dq, lin_coeffs)
    delta = (times[last] - times[0])[:, None]
    rate = (final - first) / delta
    corrected = []
    for group in range(ngroups):
        offsets = (jnp.arange(nframes, dtype=dtype) + 1 + group * nframes - times[0])
        reads = first + rate * offsets[None, :, None, None]
        reads = linearity_correct(reads, jnp.zeros_like(reads, jnp.uint8), inv_coeffs)
        reads += (data[:, group:group + 1] - jnp.mean(reads, axis=1, keepdims=True))
        reads = linearity_correct(reads, jnp.zeros_like(reads, jnp.uint8), lin_coeffs)
        value = jnp.mean(reads, axis=1).astype(data.dtype)
        corrected.append(jnp.where(saturated[:, group], data[:, group], value))
    return jnp.stack(corrected, axis=1)


@jax.jit
def subtract_dark(data, dark):
    """Subtract the per-group dark reference.

    Reference NaNs give zero correction; infinities are preserved.

    Parameters
    ----------
    data : array-like(float)
        Detector samples with shape (nints, ngroups, dimy, dimx).
    dark : array-like(float)
        Dark reference with shape (ngroups, dimy, dimx).

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    """
    safe_dark = jnp.where(jnp.isnan(dark), 0., dark)
    return data - safe_dark[None]


@jax.jit
def apply_flat_field(data, err, dq, flat, flat_err, flat_dq):
    """Apply an image flat with uncertainty and quality flag propagation.

    Parameters
    ----------
    data : array-like(float)
        Flux cube with shape (nints, dimy, dimx).
    err : array-like(float)
        Uncertainty cube matching the data.
    dq, flat_dq : array-like(int)
        Science and flat-field reference quality flags, respectively.
    flat, flat_err : array-like(float)
        Flat-field reference and uncertainty images with shape (dimy, dimx).

    Returns
    -------
    corrected, corrected_err : array-like(float)
        Corrected data and uncertainty cubes, respectively.
    corrected_dq : array-like(int)
        Combined science and reference flags.
    """
    flat = jnp.asarray(flat, dtype=data.dtype)
    flat_err = jnp.asarray(flat_err, dtype=data.dtype)
    flat_dq = jnp.asarray(flat_dq, dtype=jnp.uint32)
    invalid = jnp.isnan(flat) | (flat == 0.)
    bad_flag = jnp.uint32(core.DQ_DO_NOT_USE) | DQ_NO_FLAT_FIELD
    flat_dq = flat_dq | jnp.where(invalid, bad_flag, jnp.uint32(0))
    no_flat = (flat_dq & DQ_NO_FLAT_FIELD) != 0
    flat_dq = flat_dq | jnp.where(no_flat, jnp.uint32(core.DQ_DO_NOT_USE), jnp.uint32(0))
    ref_bad = (flat_dq & jnp.uint32(core.DQ_DO_NOT_USE)) != 0
    safe_flat = jnp.where(ref_bad, 1., flat)
    corrected = data / safe_flat
    prior_var = (err / safe_flat) ** 2
    # Preserve the multiplication order of the JWST flat variance term.
    flat_var = corrected ** 2 / safe_flat ** 2 * flat_err ** 2
    corrected_err = jnp.sqrt(prior_var + flat_var)
    corrected_dq = jnp.asarray(dq, jnp.uint32) | flat_dq
    invalid = (jnp.isnan(corrected) | jnp.isnan(corrected_err) |
               ((corrected_dq & core.DQ_DO_NOT_USE) != 0))
    corrected_dq |= jnp.where(invalid, core.DQ_DO_NOT_USE, jnp.uint32(0))
    corrected = jnp.where(invalid, jnp.nan, corrected)
    corrected_err = jnp.where(invalid, jnp.nan, corrected_err)
    return corrected, corrected_err, corrected_dq


@jax.jit
def gain_scale(data, gain_factor):
    """Multiply detector samples by the gain scale factor.

    Parameters
    ----------
    data : array-like(float)
        Detector values to scale.
    gain_factor : float
        Gain scale factor. Traced.

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    """
    return data * gain_factor
