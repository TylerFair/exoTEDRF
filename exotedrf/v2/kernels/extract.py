"""Partial-pixel box and Horne optimal spectral extraction."""

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

from exotedrf.v2 import core

__all__ = [
    'box_dq_report', 'dq_report_from_bounds', 'dq_report_from_mask', 'dq_report_frame_index',
    'box_extract', 'box_extract_sweep', 'box_white_light_sweep',
    'extract_from_stage1', 'horne_aperture_mask', 'horne_deepframe_prep',
    'horne_optimal_extract', 'miri_optimal_aperture_mask',
    'miri_optimal_deepframe_prep', 'miri_optimal_profile', 'select_width_min_white_scatter', ]


def _oddeven_merge_pairs(n):
    """Build comparator pairs for a power-of-two odd-even merge sort."""
    pairs, p = [], 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return tuple(pairs)


_MEDIAN_NETWORK = _oddeven_merge_pairs(16)


def _median_of_windows(values):
    """Calculate the median of up to fifteen arrays using a sorting network."""
    count = len(values)
    lanes = list(values) + [jnp.full_like(values[0], jnp.inf)] * (16 - count)
    for i, j in _MEDIAN_NETWORK:
        a, b = lanes[i], lanes[j]
        lanes[i], lanes[j] = jnp.minimum(a, b), jnp.maximum(a, b)
    return lanes[count // 2]


@jax.jit
def _horne_extract(profile, data, variance, aperture_mask=None):
    """Calculate the Horne inverse-variance flux and variance estimates."""
    if aperture_mask is not None:
        profile = jnp.where(aperture_mask, profile, jnp.nan)
    numerator = jnp.nansum(profile[None] * data / variance, axis=2)
    denominator = jnp.nansum(profile[None] ** 2 / variance, axis=2)
    flux = numerator / denominator
    flux_variance = jnp.nansum(profile, axis=1)[None] / denominator
    flux_variance = jnp.broadcast_to(flux_variance, flux.shape)
    return flux, flux_variance


@jax.jit
def _horne_initialize(data, profile, aperture_mask=None):
    """Initialize the Horne flux and empirical plus photon variance."""
    aperture_data = (data if aperture_mask is None else
                     jnp.where(aperture_mask[None], data, jnp.nan))
    flux0 = jnp.nansum(aperture_data, axis=2)
    variance0 = jnp.nanstd(data, axis=0) ** 2
    photon = jnp.abs(flux0[:, :, None] * profile[None])
    if aperture_mask is not None:
        # Add photon variance only inside the extraction aperture.
        photon = jnp.where(aperture_mask[None], photon, 0.)
    variance = variance0[None] + photon
    flux, flux_variance = _horne_extract(profile, data, variance, aperture_mask)
    return flux, flux_variance, variance


@jax.jit
def _horne_clip_step(data, flux, variance, profile, var_thresh,
                     aperture_mask=None, median_override=None, median_indices=None):
    """Replace temporal outliers and refit the Horne flux."""
    padded = jnp.pad(data, ((5, 5), (0, 0), (0, 0)), mode='symmetric')
    rolling = _median_of_windows([padded[offset:offset + data.shape[0]] for offset in range(11)])
    if median_override is not None:
        # Use SciPy rank selection for NaN-containing time series.
        yy, xx = median_indices
        rolling = rolling.at[:, yy, xx].set(median_override)
    bad = ((data - flux[:, :, None] * profile[None]) ** 2 / variance > var_thresh)
    filtered = jnp.where(bad, rolling, data)
    # Use empirical variance after the initial photon estimate.
    variance0 = jnp.nanstd(filtered, axis=0) ** 2
    next_variance = jnp.where(variance0 == 0, jnp.inf, variance0)
    next_flux, next_flux_variance = _horne_extract(profile, filtered, next_variance, aperture_mask)
    return (filtered, next_flux, next_flux_variance, next_variance, jnp.sum(bad))


def _horne_chunk_width(data, chunk_size):
    """Choose a wavelength chunk width within host and device memory limits."""
    nwave = int(data.shape[1])
    per_wave = (int(data.shape[0]) * int(data.shape[2]) * np.dtype(data.dtype).itemsize)
    device_width = core.auto_chunk(nwave, per_wave, n_buffers=40, headroom=.45)
    host_width = max(1, core.host_allocation_budget_bytes() // max(8 * per_wave, 1))
    requested = (chunk_size if chunk_size is not None else
                 os.environ.get('EXOTEDRF_HORNE_CHUNK_WAVES'))
    if requested is None:
        requested = nwave
    try:
        width = int(requested)
    except (TypeError, ValueError) as exc:
        raise ValueError('Horne wavelength chunk size must be positive') from exc
    if width < 1:
        raise ValueError('Horne wavelength chunk size must be positive')
    return min(nwave, width, device_width, host_width)


def _horne_nan_pixels(data):
    """Find pixel time series containing NaNs."""
    plane = np.asarray(jax.device_get(jnp.any(jnp.isnan(data), axis=0)))
    indices = np.nonzero(plane)
    return None if not indices[0].size else indices


def _horne_nan_medians(data, indices):
    """Filter NaN-containing time series using SciPy median semantics."""
    if indices is None:
        return None
    from scipy.ndimage import median_filter
    selected = np.asarray(jax.device_get(data[:, indices[0], indices[1]]))
    filtered = median_filter(selected, (11, 1), mode='reflect')
    return jnp.asarray(filtered, dtype=data.dtype)


def _horne_streamed(data, profile, aperture_mask, *, max_iter, var_thresh, chunk_size):
    """Extract wavelength chunks using the global rejection stopping rule."""
    nints, nwave, naperture = data.shape
    dtype = np.dtype(np.float64 if np.dtype(data.dtype) == np.float64 and
                     jax.config.x64_enabled else np.float32)
    flux = core.empty_host_array((nints, nwave), dtype, name='exotedrf-horne-flux')
    ferr = core.empty_host_array((nints, nwave), dtype, name='exotedrf-horne-error')
    variance = np.empty((nwave, naperture), dtype=dtype)
    threshold = jnp.asarray(var_thresh, dtype=dtype)
    # Retain a copy only for wavelength chunks altered by rejection.
    corrected, nan_pixels = {}, {}
    counts, previous = [], None
    for iteration in range(max(1, max_iter)):
        clipped_total = 0
        for lo in range(0, nwave, chunk_size):
            hi = min(lo + chunk_size, nwave)
            source = corrected.get(lo)
            if source is None:
                source = data[:, lo:hi, :]
            block = jnp.asarray(source, dtype=dtype)
            p = jnp.asarray(profile[lo:hi], dtype=block.dtype)
            mask = (None if aperture_mask is None else
                    jnp.asarray(aperture_mask[lo:hi], dtype=bool))
            if iteration == 0:
                f, fv, var = _horne_initialize(block, p, mask)
                if max_iter:
                    nan_pixels[lo] = _horne_nan_pixels(block)
            else:
                f = jnp.asarray(flux[:, lo:hi])
                var = jnp.asarray(variance[lo:hi])
            if max_iter:
                filtered, f, fv, var, rejected = _horne_clip_step(
                    block, f, var, p, threshold, mask, _horne_nan_medians(block, nan_pixels[lo]),
                    nan_pixels[lo])
                clipped = int(rejected.item())
                clipped_total += clipped
                if clipped:
                    target = corrected.get(lo)
                    if target is None:
                        target = core.empty_host_array(filtered.shape, dtype,
                            name='exotedrf-horne-rejected')
                        corrected[lo] = target
                    target[...] = np.asarray(jax.device_get(filtered))
                variance[lo:hi] = np.asarray(jax.device_get(var))
            f_host, e_host = jax.device_get((f, jnp.sqrt(fv)))
            flux[:, lo:hi] = np.asarray(f_host)
            ferr[:, lo:hi] = np.asarray(e_host)
        if not max_iter:
            break
        counts.append(clipped_total)
        if clipped_total == 0 or clipped_total == previous:
            break
        previous = clipped_total
    return flux, ferr, tuple(counts)


def horne_optimal_extract(data, profile, *, max_iter=25, var_thresh=25,
                          aperture_mask=None, chunk_size=None):
    """Extract spectra using the Horne inverse-variance estimator.

    Wavelength chunks share the full time axis and the global rejection stopping rule.

    Parameters
    ----------
    data : array-like(float)
        Samples with shape (nints, nwave, naperture), including pixels outside the aperture.
    profile : array-like(float)
        Normalized spatial profile with shape (nwave, naperture).
    max_iter : int
        Maximum temporal outlier rejection iterations; zero skips rejection.
    var_thresh : float
        Squared standardized residual threshold.
    aperture_mask : None, array-like(bool)
        Flux and initial photon aperture; rejection uses the full frame.
    chunk_size : None, int
        Maximum chunk width; None selects a width within memory limits.

    Returns
    -------
    flux, ferr : array-like(float)
        Extracted flux and its estimated uncertainty, respectively.
    clipped_counts : tuple[int]
        Rejected sample count per completed iteration.
    """
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError('max_iter must be a non-negative integer')
    if not np.isfinite(var_thresh) or var_thresh <= 0:
        raise ValueError('var_thresh must be finite and positive')

    if data.ndim != 3:
        raise ValueError(f'data must have shape (nints, nwave, naperture), got ' f'{data.shape}')
    if any(size < 1 for size in data.shape):
        raise ValueError('Horne data axes must contain at least one sample')
    if profile.shape != data.shape[1:]:
        raise ValueError(f'profile shape {profile.shape} does not match aperture '
            f'{data.shape[1:]}')
    if aperture_mask is not None and aperture_mask.shape != profile.shape:
        raise ValueError('aperture_mask shape must match the spatial profile')
    width = _horne_chunk_width(data, chunk_size)
    if width < data.shape[1]:
        return _horne_streamed(data, profile, aperture_mask, max_iter=max_iter,
            var_thresh=var_thresh, chunk_size=width)

    data = jnp.asarray(data)
    profile = jnp.asarray(profile, dtype=data.dtype)
    aperture_mask = (None if aperture_mask is None else jnp.asarray(aperture_mask, dtype=bool))
    flux, flux_variance, variance = _horne_initialize(data, profile, aperture_mask)
    nan_pixels = _horne_nan_pixels(data) if max_iter else None
    previous = None
    clipped_counts = []
    threshold = jnp.asarray(var_thresh, dtype=data.dtype)
    for _ in range(max_iter):
        data, flux, flux_variance, variance, clipped = _horne_clip_step(
            data, flux, variance, profile, threshold, aperture_mask,
            _horne_nan_medians(data, nan_pixels), nan_pixels)
        clipped = int(clipped.item())
        clipped_counts.append(clipped)
        if clipped == 0 or clipped == previous:
            break
        previous = clipped
    return flux, jnp.sqrt(flux_variance), tuple(clipped_counts)


def _nanaware_weights(centroid_cols, lower, upper, dimy):
    """Calculate whole and partial pixel weights for NaN-aware extraction."""
    edge_up = jnp.minimum(centroid_cols + upper, float(dimy))
    edge_low = jnp.maximum(centroid_cols - lower, 0.0)
    up_whole = jnp.floor(edge_up)
    low_whole = jnp.ceil(edge_low)
    rows = jnp.arange(dimy, dtype=edge_up.dtype)[:, None]
    weights = ((rows >= low_whole) & (rows < up_whole)).astype(edge_up.dtype)
    gate = ((edge_up < dimy - 1) & (edge_low > 0)).astype(edge_up.dtype)
    up_part = edge_up - up_whole
    low_part = 1.0 - (edge_low - jnp.floor(edge_low))
    # Include the extra lower pixel when the lower edge is an integer, as in v1.
    partial = ((rows == up_whole).astype(edge_up.dtype) * up_part
               + (rows == low_whole - 1.0).astype(edge_up.dtype) * low_part)
    return weights + gate * partial


def _row_sum(values, weights):
    """Sum weighted detector rows in a fixed sequential order."""
    # Keep the row reduction order independent of integration chunk size.
    def body(row, acc):
        """Update the loop state."""
        return acc + values[:, row, :] * weights[row]
    init = jnp.zeros((values.shape[0], values.shape[2]), jnp.result_type(values, weights))
    return jax.lax.fori_loop(0, values.shape[1], body, init)


def _asymmetric_sum_weights(centroid_cols, lower, upper, dimy):
    """Calculate overlap weights with separate lower and upper half-widths."""
    y = jnp.arange(dimy, dtype=centroid_cols.dtype)[:, None]
    low = centroid_cols[None, :] - lower
    high = centroid_cols[None, :] + upper
    return jnp.clip(jnp.minimum(y + 0.5, high) - jnp.maximum(y - 0.5, low), 0.0, 1.0)


DQ_REPORT_BITS = ((1, np.uint64(1)), (2, np.uint64(2)),
                  (4, np.uint64(2048 | 4096)), (8, np.uint64(1 << 32)))


def dq_report_frame_index(n_first_segment):
    """Get the integration index used for the spectral quality report.

    Parameters
    ----------
    n_first_segment : int
        Integration count in the first segment.

    Returns
    -------
    index : int
        Integration 10, or the final integration of a shorter segment.
    """
    return min(10, int(n_first_segment) - 1)


def dq_report_from_bounds(dq_frame, low, up, xmin, xmax):
    """Report aperture quality flags over whole-pixel row bounds.

    Parameters
    ----------
    dq_frame : array-like(int)
        Quality flags from the report integration with shape (dimy, dimx).
    low, up : array-like(int), int
        Inclusive lower and exclusive upper whole-pixel row bounds per column.
    xmin : int
        First column included in the report.
    xmax : None, int
        Exclusive final column; None includes the detector edge.

    Returns
    -------
    report : np.ndarray(float)
        Quality report per column, zero outside the requested range.
    """
    dq_frame = np.asarray(dq_frame).astype(np.uint64)
    dimy, dimx = dq_frame.shape
    xmax = dimx if xmax is None else int(xmax)
    xmin = int(xmin)
    report = np.zeros(dimx)
    n = max(xmax - xmin, 0)
    if n == 0:
        return report
    low = np.broadcast_to(np.asarray(low), (n,))
    up = np.broadcast_to(np.asarray(up), (n,))
    rows = np.arange(dimy)[:, None]
    inside = (rows >= low[None, :]) & (rows < up[None, :])
    block = dq_frame[:, xmin:xmax]
    for value, mask in DQ_REPORT_BITS:
        hit = np.any(inside & ((block & mask) != 0), axis=0)
        report[xmin:xmax] += value * hit
    return report


def dq_report_from_mask(dq_wave_major, mask, start=0):
    """Report quality flags within a whole-pixel optimal aperture.

    Parameters
    ----------
    dq_wave_major : array-like(int)
        Quality flags with shape (nwave, naperture).
    mask : array-like(bool)
        Whole-pixel aperture mask with shape (nwave, naperture).
    start : int
        First wavelength row covered by the aperture mask.

    Returns
    -------
    report : np.ndarray(float)
        Quality report per wavelength, zero outside the mask range.
    """
    frame = np.asarray(dq_wave_major).astype(np.uint64)
    mask = np.asarray(mask, dtype=bool)
    ncols = mask.shape[0]
    report = np.zeros(frame.shape[0])
    block = frame[start:start + ncols]
    for value, bits in DQ_REPORT_BITS:
        report[start:start + ncols] += value * np.any(mask & ((block & bits) != 0), axis=1)
    return report


def box_dq_report(dq_frame, centroid_y, halfwidth, extract_start=0, extract_end=None):
    """Report quality flags within the whole pixels of a box aperture.

    Parameters
    ----------
    dq_frame : array-like(int)
        Quality flags from the report integration with shape (dimy, dimx).
    centroid_y : array-like(float)
        Centroids from extract_start; short arrays repeat their final value.
    halfwidth : float, array-like(float)
        Scalar or (lower, upper) aperture half-widths. Traced.
    extract_start : int
        First detector column to extract.
    extract_end : None, int
        Exclusive final extraction column; None uses the detector edge.

    Returns
    -------
    report : np.ndarray(float)
        Quality report per detector column.
    """
    dq_frame = np.asarray(dq_frame)
    dimy, dimx = dq_frame.shape
    end = dimx if extract_end is None else int(extract_end)
    start = int(extract_start)
    cen = np.asarray(centroid_y, dtype=float)
    # Index centroids relative to the extraction start.
    cen = cen[np.clip(np.arange(start, end) - start, 0, cen.size - 1)]
    hw = np.asarray(halfwidth, dtype=float)
    lower, upper = (hw, hw) if hw.ndim == 0 else (hw[0], hw[1])
    edge_up = np.minimum(cen + upper, dimy)
    edge_low = np.maximum(cen - lower, 0)
    low = np.ceil(edge_low).astype(int)
    up = np.floor(edge_up).astype(int)
    return dq_report_from_bounds(dq_frame, low, up, start, end)


@functools.partial(jax.jit, static_argnames=('extract_start', 'extract_end', 'mode'))
def box_extract(data, err, centroid_y, halfwidth, extract_start=0,
                extract_end=None, mode='nanaware'):
    """Extract weighted box flux or a NaN-aware aperture mean.

    Partial edge pixels enter nanaware extraction only for fully interior apertures.

    Parameters
    ----------
    data : array-like(float)
        Flux cube with shape (nints, dimy, dimx).
    err : None, array-like(float)
        Uncertainty cube matching the data; None skips error propagation.
    centroid_y : array-like(float)
        Centroids from extract_start; short arrays repeat their final value.
    halfwidth : float, array-like(float)
        Scalar or (lower, upper) aperture half-widths. Traced.
    extract_start : int
        First detector column to extract. Static: changes trigger a recompile.
    extract_end : None, int
        Exclusive end column; None uses the edge. Static: changes trigger a recompile.
    mode : str
        Sum or nanaware mean extraction. Static: changes trigger a recompile.

    Returns
    -------
    flux : array-like(float)
        Flux with shape (nints, dimx).
    ferr : None, array-like(float)
        Propagated errors, or None when no errors were supplied.
    """
    _, dimy, dimx = data.shape
    if extract_end is None:
        extract_end = dimx
    if mode not in ('nanaware', 'sum'):
        raise ValueError(f"mode must be 'nanaware' or 'sum', got {mode!r}")

    centroid_y = jnp.asarray(centroid_y, data.dtype)
    ncen = centroid_y.shape[0]
    xcols = jnp.arange(dimx)
    cen_idx = jnp.clip(xcols - extract_start, 0, ncen - 1)
    centroid_cols = centroid_y[cen_idx]
    col_ok = (xcols >= extract_start) & (xcols < extract_end)
    halfwidth = jnp.asarray(halfwidth)
    if halfwidth.ndim == 0:
        lower = upper = halfwidth
    elif halfwidth.shape == (2,):
        lower, upper = halfwidth[0], halfwidth[1]
    else:
        raise ValueError('halfwidth must be a scalar or a (lower, upper) pair, got shape '
            f'{halfwidth.shape}')
    # Convert the v1 pixel convention to centered pixels for summed flux.
    weights = (_asymmetric_sum_weights(centroid_cols - 0.5, lower, upper, dimy)
               if mode == 'sum' else _nanaware_weights(centroid_cols, lower, upper, dimy))
    weights = weights * col_ok.astype(weights.dtype)[None, :]
    finite = jnp.isfinite(data)
    data0 = jnp.where(finite, data, 0.0)
    flux_sum = _row_sum(data0, weights)

    if err is not None:
        err0 = jnp.where(jnp.isfinite(err), err, 0.0)
        err2_sum = _row_sum(err0 * err0, weights * weights)
        ferr_sum = jnp.sqrt(err2_sum)
    else:
        ferr_sum = None

    if mode == 'sum':
        return flux_sum, ferr_sum

    area = _row_sum(finite.astype(weights.dtype), weights)
    flux = jnp.where(area > 0, flux_sum / area, jnp.nan)
    flux = jnp.where(col_ok[None, :], flux, 0.0)
    if ferr_sum is None:
        return flux, None
    ferr = jnp.where(area > 0, ferr_sum / area, jnp.nan)
    ferr = jnp.where(col_ok[None, :], ferr, 0.0)
    return flux, ferr


@functools.partial(jax.jit, static_argnames=('extract_start', 'extract_end', 'mode'))
def box_extract_sweep(data, err, centroid_y, halfwidths, extract_start=0,
                      extract_end=None, mode='nanaware'):
    """Extract box spectra for each candidate aperture width.

    Evaluate widths sequentially to preserve the single-candidate reduction order.

    Other parameters follow ``box_extract``.

    Parameters
    ----------
    halfwidths : array-like(float)
        Candidate half-widths with shape (nwidths,) or (nwidths, 2). Traced.

    Returns
    -------
    flux : array-like(float)
        Flux with shape (nwidths, nints, dimx).
    ferr : None, array-like(float)
        Errors with the same shape, or None when no errors were supplied.
    """
    fn = functools.partial(box_extract, extract_start=extract_start,
                           extract_end=extract_end, mode=mode)
    return jax.lax.map(lambda hw: fn(data, err, centroid_y, hw), jnp.asarray(halfwidths))


@functools.partial(jax.jit, static_argnames=('extract_start', 'extract_end'))
def box_white_light_sweep(data, centroid_y, halfwidths, extract_start=0, extract_end=None):
    """Calculate white-light curves for candidate box widths.

    Other parameters follow ``box_extract``.

    Parameters
    ----------
    halfwidths : array-like(float)
        Candidate half-widths with shape (nwidths,) or (nwidths, 2). Traced.

    Returns
    -------
    wlc : array-like(float)
        White-light curves with shape (nints, nwidths).
    """
    fn = functools.partial(box_extract, extract_start=extract_start,
                           extract_end=extract_end, mode='sum')
    wlc = jax.lax.map(lambda halfwidth: jnp.sum(fn(data, None, centroid_y, halfwidth)[0], axis=1),
        jnp.asarray(halfwidths))
    return wlc.T


def select_width_min_white_scatter(wlc, candidates):
    """Select the first aperture width with minimum white-light scatter.

    Parameters
    ----------
    wlc : array-like(float)
        White-light curves with shape (nints, nwidths).
    candidates : array-like(float)
        Candidate full aperture widths.

    Returns
    -------
    width : float
        Selected full aperture width.
    index : int
        Selected candidate index.
    scatter : np.ndarray(float)
        Point-to-point scatter per candidate.
    """
    wlc = np.asarray(wlc, dtype=np.float64)
    if wlc.ndim != 2 or wlc.shape[1] != len(candidates):
        raise ValueError(f'wlc shape {wlc.shape} does not match {len(candidates)} ' 'candidates')
    scatter = np.median(np.abs(0.5 * (wlc[0:-2] + wlc[2:]) - wlc[1:-1]), axis=0)
    index = int(np.argmin(scatter))
    return float(np.asarray(candidates, dtype=np.float64)[index]), index, scatter


@functools.partial(jax.jit, static_argnames=('extract_start', 'extract_end'))
def extract_from_stage1(data, dq, centroid_y, halfwidth, extract_start=0, extract_end=None):
    """Extract a NaN-aware spectrum from the last group of a ramp.

    Other parameters follow ``box_extract``.

    Parameters
    ----------
    data : array-like(float)
        Ramp cube with shape (nints, ngroups, dimy, dimx), or an integration cube.
    dq : array-like(int)
        Group, integration or pixel flags; the last group is used for group flags.

    Returns
    -------
    flux : array-like(float)
        Extracted flux with shape (nints, dimx).
    """
    if data.ndim == 4:
        data = data[:, -1]
    if dq.ndim == 4:
        dq = dq[:, -1]
    img = jnp.where(dq > 0, jnp.nan, data)
    flux, _ = box_extract(img, None, centroid_y, halfwidth, extract_start=extract_start,
                          extract_end=extract_end, mode='nanaware')
    return flux


def miri_optimal_aperture_mask(xpos, halfwidth, naperture):
    """Build a rounded whole-pixel aperture for MIRI optimal extraction.

    Parameters
    ----------
    xpos : array-like(float)
        Cross-dispersion aperture centers per wavelength.
    halfwidth : float
        Aperture half-width in pixels. Traced.
    naperture : int
        Full cross-dispersion detector size.

    Returns
    -------
    mask : array-like(bool)
        Aperture mask with shape (nwave, naperture).
    """
    xpos = jnp.asarray(xpos)
    halfwidth = jnp.asarray(halfwidth, dtype=xpos.dtype)
    upper = jnp.round(jnp.minimum(xpos + halfwidth, jnp.asarray(naperture, xpos.dtype)))
    lower = jnp.round(jnp.maximum(xpos - halfwidth, jnp.asarray(0.0, xpos.dtype)))
    rows = jnp.arange(naperture, dtype=xpos.dtype)[None, :]
    return (rows >= lower[:, None]) & (rows < upper[:, None])


def miri_optimal_deepframe_prep(deepframe):
    """Clip negative values and replace MIRI dispersion edge rows.

    Parameters
    ----------
    deepframe : array-like(float)
        Median detector image.

    Returns
    -------
    deepframe : array-like(float)
        Prepared deep image in the native MIRI orientation.
    """
    deepframe = jnp.asarray(deepframe)
    deepframe = jnp.where(deepframe < 0, 0.0, deepframe)
    deepframe = deepframe.at[:5].set(deepframe[5])
    deepframe = deepframe.at[-6:].set(deepframe[-6])
    return deepframe


def miri_optimal_profile(deepframe, mask, *, mask_output=True):
    """Normalize the deep image by flux within each wavelength aperture.

    Parameters
    ----------
    deepframe : array-like(float)
        Prepared deep image with shape (nwave, naperture).
    mask : array-like(bool)
        Whole-pixel aperture mask with shape (nwave, naperture).
    mask_output : bool
        Set the profile outside the aperture to NaN; False retains it for rejection.

    Returns
    -------
    profile : array-like(float)
        Spatial profile with shape (nwave, naperture).
    """
    deepframe = jnp.asarray(deepframe)
    windowed = jnp.where(mask, deepframe, jnp.nan)
    total = jnp.nansum(windowed, axis=1, keepdims=True)
    return (windowed if mask_output else deepframe) / total


def horne_aperture_mask(centers, halfwidth, naperture):
    """Build a whole-pixel Horne aperture with the wavelength axis first.

    Parameters
    ----------
    centers : array-like(float)
        Cross-dispersion aperture centers per wavelength.
    halfwidth : None, float
        Aperture half-width in pixels; None includes the full cross-dispersion frame. Traced.
    naperture : int
        Full cross-dispersion detector size.

    Returns
    -------
    mask : array-like(bool)
        Aperture mask with shape (nwave, naperture).
    """
    centers = jnp.asarray(centers)
    if halfwidth is None:
        return jnp.ones((centers.shape[0], int(naperture)), dtype=bool)
    return miri_optimal_aperture_mask(centers, halfwidth, naperture)


def horne_deepframe_prep(deepframe, dispersion_axis=0):
    """Prepare the optimal deep image and put the wavelength axis first.

    Parameters
    ----------
    deepframe : array-like(float)
        Median detector image.
    dispersion_axis : int
        Dispersion axis in the input image, either 0 or 1.

    Returns
    -------
    deepframe : array-like(float)
        Prepared deep image with shape (nwave, naperture).
    """
    deepframe = jnp.asarray(deepframe)
    if dispersion_axis == 1:
        deepframe = deepframe.T
    elif dispersion_axis != 0:
        raise ValueError('dispersion_axis must be 0 or 1')
    return miri_optimal_deepframe_prep(deepframe)
