"""Trace masks and 1/f noise corrections for SOSS and NIRSpec."""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from exotedrf.v2.kernels import jump

__all__ = ['make_trace_mask', 'baseline_deepstack', 'flag_temporal_outliers',
    'build_soss_masks', 'build_soss_achromatic_mask',
    'estimate_timeseries', 'oneoverf_scale_achromatic',
    'oneoverf_scale_window', 'oneoverf_scale_chromatic', 'oneoverf_nirspec',
    'oneoverf_solve', 'line_mle', 'segment_nanstd', 'solve_trace_bounds', ]


@functools.partial(jax.jit, static_argnames=('dimy',))
def make_trace_mask(ypos, width, dimy):
    """Build a trace mask from truncated centroid bounds.

    Parameters
    ----------
    ypos : array-like(float)
        Trace centroids per detector column; NaN indicates no trace.
    width : float
        Full trace mask width in pixels. Traced.
    dimy : int
        Number of detector rows. Static: changes trigger a recompile.

    Returns
    -------
    mask : array-like(bool)
        Trace mask with shape (dimy, dimx); NaN centroids remain unmasked.
    """
    ypos = jnp.asarray(ypos, dtype=jnp.float32)
    low = jnp.floor(jnp.maximum(0., ypos - width / 2.))
    up = jnp.floor(jnp.minimum(float(dimy), ypos + width / 2.))
    y = jnp.arange(dimy, dtype=jnp.float32)[:, None]
    return (y >= low[None, :]) & (y < up[None, :])


def _soss_trace_masks(positions, width, dimy, dimx):
    """Build per-order trace masks, leaving missing orders empty."""
    return tuple(make_trace_mask(pos, width, dimy) if pos is not None else
                 jnp.zeros((dimy, dimx), dtype=bool) for pos in positions)


@jax.jit
def baseline_deepstack(cube, baseline_mask):
    """Calculate the median stack of baseline integrations.

    Parameters
    ----------
    cube : array-like(float)
        Cube with shape (nints, [ngroups,] dimy, dimx).
    baseline_mask : array-like(bool)
        True for baseline integrations.

    Returns
    -------
    deepstack : array-like(float)
        NaN-aware median with the integration axis removed.
    """
    bm = baseline_mask.reshape((-1,) + (1,) * (cube.ndim - 1))
    return jnp.nanmedian(jnp.where(bm, cube, jnp.nan), axis=0)


@functools.partial(jax.jit, static_argnames=('window',))
def flag_temporal_outliers(cube, thresh=10.0, window=5):
    """Flag pixels above the normalized temporal deviation threshold.

    Parameters
    ----------
    cube : array-like(float)
        Cube with shape (nints, [ngroups,] dimy, dimx).
    thresh : float
        Rejection threshold in scatter units. Traced.
    window : int
        Time median window. Static: changes trigger a recompile.

    Returns
    -------
    outliers : array-like(bool)
        Temporal outlier mask matching the input cube.
    """
    scale, _ = jump.scatter_normalize(cube, window)
    return scale > thresh


@functools.partial(jax.jit, static_argnames=('method', 'o2_red_cut'))
def build_soss_masks(base_mask, ypos_o1, ypos_o2, ypos_o3, inner_width,
                     outer_width, method='achromatic', o2_red_cut=1100):
    """Assemble SOSS exclusions and inner trace regions for 1/f correction.

    Parameters
    ----------
    base_mask : array-like(bool)
        Existing exclusions with shape (nints, dimy, dimx).
    ypos_o1 : array-like(float)
        Order 1 trace centroids per column.
    ypos_o2, ypos_o3 : None, array-like(float)
        Order 2 and 3 centroids per column; NaNs indicate absent traces.
    inner_width, outer_width : float
        Full inner and outer trace mask widths in pixels, respectively. Traced.
    method : str
        Achromatic, achromatic-window or chromatic exclusions. Static: triggers a recompile.
    o2_red_cut : int
        Exclude order 2 columns below this index. Static: changes trigger a recompile.

    Returns
    -------
    outliers1, outliers2 : array-like(bool)
        Order exclusions with shape (nints, dimy, dimx).
    trace1_in, trace2_in : array-like(bool)
        Inner trace masks with shape (dimy, dimx).
    """
    _, dimy, dimx = base_mask.shape
    m1_in = make_trace_mask(ypos_o1, inner_width, dimy)
    m2_in, m3 = _soss_trace_masks((ypos_o2, ypos_o3), inner_width, dimy, dimx)
    tracemask = m1_in | m2_in | m3
    if method == 'achromatic':
        outliers1 = base_mask | tracemask[None]
        outliers2 = outliers1
    else:
        m1_out = make_trace_mask(ypos_o1, outer_width, dimy)
        m2_out, = _soss_trace_masks((ypos_o2,), outer_width, dimy, dimx)
        window1 = ~(m1_out & ~m1_in)
        window2 = ~(m2_out & ~m2_in)
        outliers1 = base_mask | (window1 | tracemask)[None]
        outliers2 = base_mask | (window2 | tracemask)[None]
        # Exclude the red end of order 2 from the background estimate.
        outliers2 = outliers2.at[:, :, :o2_red_cut].set(True)
    return outliers1, outliers2, m1_in, m2_in


@jax.jit
def build_soss_achromatic_mask(base_mask, ypos_o1, ypos_o2, ypos_o3, inner_width):
    """Assemble achromatic SOSS exclusions from all three trace cores.

    Parameters
    ----------
    base_mask : array-like(bool)
        Existing exclusions with shape (nints, dimy, dimx).
    ypos_o1 : array-like(float)
        Order 1 trace centroids per column.
    ypos_o2, ypos_o3 : array-like(float)
        Order 2 and 3 centroids per column; NaNs indicate absent traces.
    inner_width : float
        Full inner trace mask width in pixels. Traced.

    Returns
    -------
    outliers : array-like(bool)
        Combined exclusions with shape (nints, dimy, dimx).
    """
    dimy = base_mask.shape[-2]
    m1, m2, m3 = (make_trace_mask(pos, inner_width, dimy) for pos in (ypos_o1, ypos_o2, ypos_o3))
    return base_mask | (m1 | m2 | m3)[None]


def _median_filter_1d(x, size):
    """Median-filter a series with symmetric padding and upper-middle rank."""
    if size <= 1:
        return x
    n = x.shape[0]
    lo, hi = size // 2, size - size // 2 - 1
    padded = jnp.pad(x, (lo, hi), mode='symmetric')
    windows = jnp.stack([jax.lax.dynamic_slice_in_dim(padded, k, n, axis=0)
                         for k in range(size)], axis=0)
    return jnp.sort(windows, axis=0)[size // 2]


@functools.partial(jax.jit, static_argnames=('smoothing_scale', 'region'))
def estimate_timeseries(cube, deepstack, smoothing_scale, region=(20, 60, 1500, 1550)):
    """Estimate flux scaling from a median-filtered detector region.

    Parameters
    ----------
    cube : array-like(float)
        Cube with shape (nints, [ngroups,] dimy, dimx).
    deepstack : array-like(float)
        Baseline median stack with the integration axis removed.
    smoothing_scale : int
        Median filter window length in integrations. Static: changes trigger a recompile.
    region : tuple[int]
        Region (row_low, row_high, col_low, col_high). Static: changes trigger a recompile.

    Returns
    -------
    timeseries : array-like(float)
        Normalized flux scaling per integration.
    """
    y0, y1, x0, x1 = region
    if cube.ndim == 4:
        cube, deepstack = cube[:, -1], deepstack[-1]
    postage = cube[:, y0:y1, x0:x1]
    zero_point = deepstack[y0:y1, x0:x1]
    ts = jnp.nansum(postage, axis=(1, 2)) / jnp.nansum(zero_point)
    return _median_filter_1d(ts, int(smoothing_scale))


def default_smoothing_scale(nints):
    """Get the default flux smoothing window from the integration count.

    Parameters
    ----------
    nints : int
        Number of integrations.

    Returns
    -------
    scale : int
        Two percent of integrations, truncated and bounded below by one.
    """
    return int(max(0.02 * nints, 1))


def _colwise_masked_median(sub, even_odd_rows):
    """Calculate column medians, optionally separating even and odd rows."""
    row_axis = sub.ndim - 2
    if even_odd_rows:
        med_e = jnp.nanmedian(sub[..., 0::2, :], axis=row_axis)
        med_o = jnp.nanmedian(sub[..., 1::2, :], axis=row_axis)
        odd = (jnp.arange(sub.shape[row_axis]) % 2).astype(bool)
        dc = jnp.where(odd[:, None], jnp.expand_dims(med_o, row_axis),
                       jnp.expand_dims(med_e, row_axis))
    else:
        dc = jnp.nanmedian(sub, axis=row_axis, keepdims=True)
        dc = jnp.broadcast_to(dc, sub.shape)
    return jnp.where(jnp.isfinite(dc), dc, 0.)


def _ts_2d(timeseries, nints, dimx):
    """Broadcast a white-light time series over detector columns."""
    ts = jnp.asarray(timeseries)
    if ts.ndim == 1:
        ts = jnp.broadcast_to(ts[:, None], (nints, dimx))
    return ts


def _oneoverf_pass(cube, deepstack, outliers, ts2d, even_odd_rows, apply_mask=None):
    """Subtract the column median of scaled deepstack residuals."""
    shape = (ts2d.shape[0],) + (1,) * (cube.ndim - 2) + (ts2d.shape[-1],)
    sub = cube - deepstack[None] * ts2d.reshape(shape)
    if cube.ndim == 4:
        outliers = outliers[:, None]
    sub = jnp.where(outliers, jnp.nan, sub)
    dc = _colwise_masked_median(sub, even_odd_rows)
    if apply_mask is not None:
        dc = dc * apply_mask.astype(cube.dtype)
    return cube - dc


@functools.partial(jax.jit, static_argnames=('even_odd_rows',))
def oneoverf_scale_achromatic(cube, deepstack, outliers, timeseries,
                              even_odd_rows=True, background=None):
    """Subtract an achromatic SOSS 1/f estimate over the full frame.

    Parameters
    ----------
    cube : array-like(float)
        Cube with shape (nints, [ngroups,] dimy, dimx).
    deepstack : array-like(float)
        Baseline median stack with the integration axis removed.
    outliers : array-like(bool)
        Excluded pixels with shape (nints, dimy, dimx).
    timeseries : array-like(float)
        Normalized flux scaling with shape (nints,) or (nints, dimx).
    even_odd_rows : bool
        Separate even and odd row levels. Static: changes trigger a recompile.
    background : None, array-like(float)
        Background model to add back after correction.

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    """
    ts2d = _ts_2d(timeseries, cube.shape[0], cube.shape[-1])
    out = _oneoverf_pass(cube, deepstack, outliers, ts2d, even_odd_rows)
    if background is not None:
        out = out + background
    return out


@functools.partial(jax.jit, static_argnames=('even_odd_rows',))
def oneoverf_scale_window(cube, deepstack, outliers1, outliers2, trace1_in,
                          trace2_in, timeseries, timeseries_o2=None,
                          even_odd_rows=True, background=None):
    """Subtract SOSS 1/f estimates within each order trace window.

    Correct order 1 first, then fit order 2 to the corrected data.

    Parameters
    ----------
    cube : array-like(float)
        Cube with shape (nints, [ngroups,] dimy, dimx).
    deepstack : array-like(float)
        Baseline median stack with the integration axis removed.
    outliers1, outliers2 : array-like(bool)
        Order 1 and 2 exclusion masks, respectively.
    trace1_in, trace2_in : array-like(bool)
        Order 1 and 2 inner trace application masks, respectively.
    timeseries : array-like(float)
        Normalized flux scaling with shape (nints,) or (nints, dimx).
    timeseries_o2 : None, array-like(float)
        Order 2 flux scaling; None reuses the order 1 time series.
    even_odd_rows : bool
        Separate even and odd row levels. Static: changes trigger a recompile.
    background : None, array-like(float)
        Background model to add back after correction.

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    """
    nints, dimx = cube.shape[0], cube.shape[-1]
    ts1 = _ts_2d(timeseries, nints, dimx)
    ts2 = ts1 if timeseries_o2 is None else _ts_2d(timeseries_o2, nints, dimx)
    out = _oneoverf_pass(cube, deepstack, outliers1, ts1, even_odd_rows, apply_mask=trace1_in)
    # Fit order 2 after correcting order 1.
    out = _oneoverf_pass(out, deepstack, outliers2, ts2, even_odd_rows, apply_mask=trace2_in)
    if background is not None:
        out = out + background
    return out


def oneoverf_scale_chromatic(cube, deepstack, outliers1, outliers2,
                             trace1_in, trace2_in, timeseries, timeseries_o2, even_odd_rows=True,
                             background=None):
    """Subtract windowed SOSS 1/f using per-column flux scalings.

    Other parameters follow ``oneoverf_scale_window``.

    Parameters
    ----------
    timeseries, timeseries_o2 : array-like(float)
        Order 1 and 2 normalized per-column scalings with shape (nints, dimx).

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    """
    timeseries = jnp.asarray(timeseries)
    timeseries_o2 = jnp.asarray(timeseries_o2)
    if timeseries.ndim != 2 or timeseries_o2.ndim != 2:
        raise ValueError('2D light curves are required for the chromatic '
                         'correction (v1 stage1.py:2539-2552).')
    return oneoverf_scale_window(cube, deepstack, outliers1, outliers2,
                                 trace1_in, trace2_in, timeseries, timeseries_o2=timeseries_o2,
                                 even_odd_rows=even_odd_rows, background=background)


def _slope_dc(masked, dimy):
    """Fit a linear trend over rows, returning zero for fewer than two valid rows."""
    y = jnp.arange(dimy, dtype=masked.dtype)
    yb = y.reshape((dimy, 1))
    valid = jnp.isfinite(masked)
    w = valid.astype(masked.dtype)
    v = jnp.where(valid, masked, 0.)
    ax = masked.ndim - 2
    n = jnp.sum(w, axis=ax)
    sx = jnp.sum(w * yb, axis=ax)
    sxx = jnp.sum(w * yb * yb, axis=ax)
    sy = jnp.sum(v, axis=ax)
    sxy = jnp.sum(v * yb, axis=ax)
    den = n * sxx - sx * sx
    ok = (n >= 2) & (den > 0)
    den = jnp.where(ok, den, 1.)
    slope = jnp.where(ok, (n * sxy - sx * sy) / den, 0.)
    icpt = jnp.where(ok, (sy - slope * sx) / jnp.where(n > 0, n, 1.), 0.)
    return (jnp.expand_dims(slope, ax) * yb + jnp.expand_dims(icpt, ax))


@functools.partial(jax.jit, static_argnames=('method',))
def oneoverf_nirspec(cube, base_outliers, ypos, mask_width, method='median'):
    """Subtract a column median or linear row trend from NIRSpec data.

    Parameters
    ----------
    cube : array-like(float)
        Cube with shape (nints, [ngroups,] dimy, dimx).
    base_outliers : array-like(bool)
        Existing exclusions with shape (nints, dimy, dimx).
    ypos : array-like(float)
        Trace centroids per detector column; NaN indicates no trace.
    mask_width : float
        Full trace exclusion width in pixels. Traced.
    method : str
        Use median or slope column correction. Static: changes trigger a recompile.

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    """
    dimy = cube.shape[-2]
    tracemask = make_trace_mask(ypos, mask_width, dimy)
    outliers = base_outliers | tracemask[None]
    if cube.ndim == 4:
        masked = jnp.where(outliers[:, None], jnp.nan, cube)
    else:
        masked = jnp.where(outliers, jnp.nan, cube)
    if method == 'median':
        row_axis = masked.ndim - 2
        dc = jnp.nanmedian(masked, axis=row_axis, keepdims=True)
        dc = jnp.broadcast_to(dc, masked.shape)
    elif method == 'slope':
        dc = _slope_dc(masked, dimy)
    else:
        raise ValueError('Unrecognized 1/f method {}'.format(method))
    dc = jnp.where(jnp.isfinite(dc), dc, 0.)
    return cube - dc


SOLVE_O2_RED_CUT = 1100


def solve_trace_bounds(ypos, width, dimy):
    """Calculate integer SOSS trace bounds using host double precision.

    Parameters
    ----------
    ypos : array-like(float)
        Trace centroids per detector column; NaN indicates no trace.
    width : float
        Full trace mask width in pixels.
    dimy : int
        Number of detector rows.

    Returns
    -------
    low, up : np.ndarray(int)
        Per-column bounds with Python slice normalization.
    """
    if ypos is None:
        raise ValueError('solve_trace_bounds needs a centroid array')
    y = np.asarray(ypos, dtype=np.float64)
    finite = np.isfinite(y)
    half = float(width) / 2.
    with np.errstate(invalid='ignore'):
        low = np.trunc(np.maximum(0., np.where(finite, y - half, 0.)))
        up = np.trunc(np.minimum(float(dimy), np.where(finite, y + half, 0.)))
    low = low.astype(np.int64)
    up = up.astype(np.int64)
    # Normalize bounds using Python slice semantics.
    low = np.where(low < 0, np.maximum(dimy + low, 0), np.minimum(low, dimy))
    up = np.where(up < 0, np.maximum(dimy + up, 0), np.minimum(up, dimy))
    low = np.where(finite, low, 0)
    up = np.where(finite, up, 0)
    return low.astype(np.int32), up.astype(np.int32)


@jax.jit
def segment_nanstd(cube):
    """Calculate temporal scatter over one complete segment.

    Parameters
    ----------
    cube : array-like(float)
        Cube with shape (nints, [ngroups,] dimy, dimx).

    Returns
    -------
    std : array-like(float)
        Population standard deviation with the integration axis removed.
    """
    return jnp.nanstd(cube, axis=0)


def _solve_rows_in(bounds, dimy):
    """Build a row mask from per-column integer bounds."""
    low, up = bounds
    rows = jnp.arange(dimy, dtype=jnp.int32)[:, None]
    return (rows >= jnp.asarray(low, jnp.int32)[None, :]) & \
        (rows < jnp.asarray(up, jnp.int32)[None, :])


def line_mle(x, y, e):
    """Fit weighted lines separately to even and odd detector rows.

    Reduce weighted terms independently with nansum, preserving finite-error weights at NaN samples.

    Parameters
    ----------
    x : array-like(float)
        Independent line-fit samples, broadcast to (..., dimy, dimx).
    y : array-like(float)
        Dependent line-fit samples, broadcast to (..., dimy, dimx).
    e : array-like(float)
        Sample errors; infinite errors give zero weight.

    Returns
    -------
    m_e, b_e, m_o, b_o : array-like(float)
        Even and odd slopes and intercepts with shape (..., dimx).
    """
    def half(xs, ys, es):
        """Fit a weighted line to one row parity."""
        e2 = es ** 2
        sx = jnp.nansum(xs / e2, axis=-2)
        sxx = jnp.nansum((xs / es) ** 2, axis=-2)
        sy = jnp.nansum(ys / e2, axis=-2)
        sxy = jnp.nansum(xs * ys / e2, axis=-2)
        s = jnp.nansum(1 / e2, axis=-2)
        m = (s * sxy - sx * sy) / (s * sxx - sx ** 2)
        b = (sy - m * sx) / s
        return m, b

    shape = jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y), jnp.shape(e))
    x = jnp.broadcast_to(x, shape)
    y = jnp.broadcast_to(y, shape)
    e = jnp.broadcast_to(e, shape)
    m_e, b_e = half(x[..., 0::2, :], y[..., 0::2, :], e[..., 0::2, :])
    m_o, b_o = half(x[..., 1::2, :], y[..., 1::2, :], e[..., 1::2, :])
    return m_e, b_e, m_o, b_o


def _solve_order(cube, deepstack, err, apply_mask):
    """Fit and subtract one order, returning scalings and 1/f levels."""
    m_e, b_e, m_o, b_o = line_mle(deepstack[None], cube, err)
    row_axis = cube.ndim - 2
    odd = (jnp.arange(cube.shape[row_axis]) % 2).astype(bool)[:, None]
    oof = jnp.where(odd, jnp.expand_dims(b_o, row_axis), jnp.expand_dims(b_e, row_axis))
    oof = jnp.where(jnp.isfinite(oof), oof, jnp.zeros((), cube.dtype))
    corrected = jnp.where(apply_mask, cube - oof, cube)
    finite_b_e = jnp.where(jnp.isfinite(b_e), b_e, jnp.zeros((), b_e.dtype))
    finite_b_o = jnp.where(jnp.isfinite(b_o), b_o, jnp.zeros((), b_o.dtype))
    return corrected, {'scale_e': m_e, 'scale_o': m_o, 'oof_e': finite_b_e, 'oof_o': finite_b_o}


@functools.partial(jax.jit, static_argnames=('order2', 'o2_red_cut'))
def oneoverf_solve(cube, deepstack, std, dq, trace1_bounds, trace2_bounds,
                   excluded=None, pixeldq=None, background=None,
                   order2=True, o2_red_cut=SOLVE_O2_RED_CUT):
    """Fit and subtract even and odd SOSS 1/f levels per order.

    Correct order 1 first, then fit order 2 to the corrected data.

    Parameters
    ----------
    cube, dq : array-like(float), array-like(int)
        Flux and quality cubes with shape (nints, [ngroups,] dimy, dimx).
    deepstack : array-like(float)
        Baseline median stack with the integration axis removed.
    std : array-like(float)
        Temporal scatter of the complete segment, matching deepstack.
    trace1_bounds, trace2_bounds : tuple[array-like(int)]
        Order 1 and 2 lower and upper row bounds per column, respectively.
    excluded : None, array-like(bool)
        Additional exclusions with shape (nints, dimy, dimx).
    pixeldq : None, array-like(int)
        Additional detector pixel flags for group-level data.
    background : None, array-like(float)
        Background model to add back after correction.
    order2 : bool
        Fit and subtract the order 2 correction. Static: changes trigger a recompile.
    o2_red_cut : int
        Exclude order 2 columns below this index. Static: changes trigger a recompile.

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    diagnostics : dict
        Order o1 and optional o2 scalings and levels: scale_e, scale_o, oof_e and oof_o.
    """
    dimy, dimx = cube.shape[-2], cube.shape[-1]
    is_ramp = cube.ndim == 4
    inf = jnp.asarray(jnp.inf, dtype=cube.dtype)
    bad = jnp.asarray(dq) != 0
    if pixeldq is not None:
        bad = bad | (jnp.asarray(pixeldq) != 0)
    if excluded is not None:
        ex = jnp.asarray(excluded, dtype=bool)
        bad = bad | (ex[:, None] if is_ramp else ex)
    bad = bad | (std == 0)[None]
    tr1 = _solve_rows_in(trace1_bounds, dimy)
    tr2 = _solve_rows_in(trace2_bounds, dimy)
    err1 = jnp.where(bad | ~tr1, inf, std[None])
    out, diag1 = _solve_order(cube, deepstack, err1, tr1)
    diagnostics = {'o1': diag1}
    if order2:
        red = jnp.arange(dimx)[None, :] < o2_red_cut
        err2 = jnp.where(bad | ~tr2 | red, inf, std[None])
        # Fit order 2 after correcting order 1.
        out, diag2 = _solve_order(out, deepstack, err2, tr2)
        diagnostics['o2'] = diag2
    if background is not None:
        out = out + background
    return out, diagnostics
