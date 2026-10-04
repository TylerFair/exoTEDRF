"""Bad pixel detection and interpolation for integration cubes."""

import functools

import jax
import jax.numpy as jnp

from exotedrf.v2 import core, mathutils


def _shifted_row(img, row_indices, dx):
    """Sample shifted columns and selected rows, filling clipped columns with NaN."""
    dimx = img.shape[-1]
    x = jnp.arange(dimx) + dx
    valid = (x >= 0) & (x < dimx - 1)
    x = jnp.clip(x, 0, max(dimx - 1, 0))
    sampled = jnp.take(jnp.take(img, row_indices, axis=-2), x, axis=-1)
    mask_shape = (1,) * (sampled.ndim - 1) + (dimx,)
    return jnp.where(valid.reshape(mask_shape), sampled, jnp.nan)


def _v1_interp_stack(img, ybox_half, xbox_half):
    """Collect the asymmetric interpolation samples used by v1 ``get_interp_box``."""
    dimy = img.shape[-2]
    rows = jnp.arange(dimy)
    low = jnp.clip(rows - ybox_half, 0, dimy - 1)
    high = jnp.clip(rows + ybox_half + 1, 0, dimy - 1)
    samples = []
    for dx in range(-xbox_half, xbox_half + 1):
        if dx != 0:
            samples.append(_shifted_row(img, rows, dx))
    # Keep repeated samples to match the interpolation median weights.
    for selected_rows in (low, high):
        for dx in range(-xbox_half, xbox_half + 1):
            samples.append(_shifted_row(img, selected_rows, dx))
    return jnp.stack(samples, axis=0)


def _box_median(img, ybox_half, xbox_half):
    """Calculate the NaN-aware interpolation box median."""
    stack = _v1_interp_stack(img, ybox_half, xbox_half)
    return jnp.nanmedian(stack, axis=0)


def _box_stats_mad(img, ybox_half, xbox_half):
    """Calculate the interpolation box median and MAD scatter."""
    stack = _v1_interp_stack(img, ybox_half, xbox_half)
    med = jnp.nanmedian(stack, axis=0)
    mad = jnp.nanmedian(jnp.abs(stack - med), axis=0)
    return med, mad / 0.6745


def _detector_geometry(dimy, box_size, instrument):
    """Get detector limits and interpolation box half-widths."""
    instrument = instrument.upper()
    if instrument == 'NIRISS':
        return dimy - 5, 0, box_size
    if instrument == 'NIRSPEC':
        return dimy, 0, box_size
    return dimy, box_size, 0


def _detector_inbounds(dimy, dimx, ymax):
    """Exclude reference columns and rows beyond the instrument limit."""
    yy = jnp.arange(dimy)[:, None]
    xx = jnp.arange(dimx)[None, :]
    return (xx >= 5) & (xx < dimx - 5) & (yy < ymax)


def _replace_flagged(data, dq, flags, replacement, clear_dq):
    """Interpolate flagged samples and optionally clear their quality flags."""
    if clear_dq:
        dq = jnp.where(flags, jnp.uint32(0), dq)
    return jnp.where(flags, replacement, data), dq.astype(jnp.uint32)


@jax.jit
def prepare_badpix_common(err, dq):
    """Replace uncertainty NaNs and prepare saturation masks.

    Parameters
    ----------
    err : array-like(float)
        Uncertainty cube matching the data.
    dq : array-like(int)
        Quality flags matching the data.

    Returns
    -------
    err_out : array-like(float)
        Error cube with NaNs replaced by its global median.
    saturated, saturated_any : array-like(bool)
        Saturation masks per integration and across integrations, respectively.
    """
    saturated = (dq & core.DQ_SATURATED) != 0
    saturated_any = jnp.any(saturated, axis=0)
    err_out = jnp.where(jnp.isnan(err), jnp.nanmedian(err), err)
    return err_out, saturated, saturated_any


@functools.partial(jax.jit, static_argnames=('box_size', 'instrument', 'preserve_saturated'))
def prepare_spatial_badpix(deepframe, dq, saturated_any, *, box_size=5,
                           instrument='NIRISS', preserve_saturated=False):
    """Prepare fixed flags and local statistics for spatial rejection.

    MIRI spatial deviation detection is disabled; DQ and NaN detection remain active.

    Parameters
    ----------
    deepframe, dq : array-like(float), array-like(int)
        Median image (dimy, dimx) and quality flags (nints, dimy, dimx), respectively.
    saturated_any : array-like(bool)
        Pixels saturated in any integration.
    box_size : int
        Interpolation box half-width in pixels. Static: changes trigger a recompile.
    instrument : str
        Instrument name. Static: changes trigger a recompile.
    preserve_saturated : bool
        Exclude saturated pixels from interpolation. Static: changes trigger a recompile.

    Returns
    -------
    fixed_badpix, threshold_eligible : array-like(bool)
        Fixed DQ/NaN flags and eligibility for spatial threshold detection, respectively.
    deviation, scatter : array-like(float)
        Absolute local-median deviation and local MAD scatter, respectively.
    """
    instrument = instrument.upper()
    nints, dimy, dimx = dq.shape
    ymax, ybox, xbox = _detector_geometry(dimy, box_size, instrument)
    # Use integration 10, or the final integration of a shorter segment.
    ref_int = min(10, nints - 1)
    hot = ((dq[ref_int] & (core.DQ_DO_NOT_USE | core.DQ_HOT | core.DQ_WARM)) != 0)
    inbounds = _detector_inbounds(dimy, dimx, ymax)
    med, scatter = _box_stats_mad(deepframe, ybox, xbox)
    is_nan = jnp.isnan(deepframe)
    fixed_badpix = inbounds & (hot | is_nan)
    threshold_eligible = inbounds & ~hot & ~is_nan
    if preserve_saturated:
        fixed_badpix = fixed_badpix & ~saturated_any
        threshold_eligible = threshold_eligible & ~saturated_any
    if instrument == 'MIRI':
        threshold_eligible = jnp.zeros_like(threshold_eligible)
    deviation = jnp.abs(deepframe - med)
    return fixed_badpix, threshold_eligible, deviation, scatter


@jax.jit
def spatial_badpix_from_prepared(fixed_badpix, threshold_eligible,
                                 deviation, scatter, space_thresh=15.):
    """Apply a spatial threshold to prepared statistics.

    Parameters
    ----------
    fixed_badpix, threshold_eligible : array-like(bool)
        Fixed DQ/NaN flags and eligibility for spatial threshold detection, respectively.
    deviation, scatter : array-like(float)
        Absolute local-median deviation and local MAD scatter, respectively.
    space_thresh : float
        Spatial rejection threshold in MAD scatter units. Traced.

    Returns
    -------
    badpix : array-like(bool)
        Fixed and threshold-selected detector flags.
    """
    # Keep zero-scatter threshold comparisons separate from division.
    deviant = deviation >= space_thresh * scatter
    return fixed_badpix | (deviant & threshold_eligible)


@functools.partial(jax.jit, static_argnames=('box_size', 'instrument', 'clear_interpolated_dq'))
def apply_spatial_badpix(cube, dq, badpix, *, box_size=5,
                         instrument='NIRISS', clear_interpolated_dq=False):
    """Replace spatially flagged pixels with interpolation box medians.

    Supply negative-clipped data when discovering a new spatial map.

    Other parameters follow ``apply_spatial_badpix_from_median``.

    Parameters
    ----------
    box_size : int
        Interpolation box half-width in pixels. Static: changes trigger a recompile.
    instrument : str
        Instrument name. Static: changes trigger a recompile.

    Returns
    -------
    newdata, newdq : array-like(float), array-like(int)
        Interpolated data cube and updated uint32 flags, respectively.
    """
    return apply_spatial_badpix_from_median(cube, dq, badpix,
        spatial_interp_median(cube, box_size=box_size, instrument=instrument),
        clear_interpolated_dq=clear_interpolated_dq)


@functools.partial(jax.jit, static_argnames=('box_size', 'instrument'))
def spatial_interp_median(cube, *, box_size=5, instrument='NIRISS'):
    """Calculate interpolation box medians for every integration.

    Parameters
    ----------
    cube : array-like(float)
        Flux cube with shape (nints, dimy, dimx).
    box_size : int
        Interpolation box half-width in pixels. Static: changes trigger a recompile.
    instrument : str
        Instrument name. Static: changes trigger a recompile.

    Returns
    -------
    boxmed : array-like(float)
        Replacement medians with the input shape.
    """
    _, ybox, xbox = _detector_geometry(cube.shape[-2], box_size, instrument)
    return _box_median(cube, ybox, xbox)


@functools.partial(jax.jit, static_argnames=('clear_interpolated_dq',))
def apply_spatial_badpix_from_median(cube, dq, badpix, boxmed, clear_interpolated_dq=False):
    """Replace flagged pixels using prepared box medians.

    Parameters
    ----------
    cube, dq : array-like(float), array-like(int)
        Flux and quality cubes with shape (nints, dimy, dimx).
    badpix : array-like(bool)
        Detector map of pixels to interpolate.
    boxmed : array-like(float)
        Prepared interpolation medians matching the data.
    clear_interpolated_dq : bool
        Clear flags on interpolated pixels. Static: changes trigger a recompile.

    Returns
    -------
    newdata, newdq : array-like(float), array-like(int)
        Interpolated data cube and updated uint32 flags, respectively.
    """
    return _replace_flagged(cube, dq, badpix[None], boxmed, clear_interpolated_dq)


@jax.jit
def temporal_scatter(newdata):
    """Calculate per-pixel point-to-point scatter over integrations.

    Parameters
    ----------
    newdata : array-like(float)
        Interpolated data with shape (nints, dimy, dimx).

    Returns
    -------
    scatter : array-like(float)
        Raw scatter plane before zero replacement.
    """
    return jnp.nanmedian(jnp.abs(0.5 * (newdata[0:-2] + newdata[2:]) - newdata[1:-1]), axis=0)


@functools.partial(jax.jit, static_argnames=('window_size', 'instrument'))
def temporal_filter(newdata, *, window_size=5, instrument='NIRISS'):
    """Calculate a running time median and replace edge integrations.

    Parameters
    ----------
    newdata : array-like(float)
        Interpolated data with shape (nints, dimy, dimx).
    window_size : int
        Time median window. Static: changes trigger a recompile.
    instrument : str
        Instrument name. Static: changes trigger a recompile.

    Returns
    -------
    cube_filt : array-like(float)
        Filtered cube with instrument-specific edge replacement.
    """
    cube_filt = mathutils.running_median_time(newdata, window_size)
    if instrument.upper() == 'NIRISS':
        cube_filt = cube_filt.at[:2].set(jnp.median(cube_filt[2:7], axis=0))
        cube_filt = cube_filt.at[-2:].set(jnp.median(cube_filt[-8:-3], axis=0))
    else:
        cube_filt = cube_filt.at[:5].set(jnp.median(cube_filt[5:15], axis=0))
        cube_filt = cube_filt.at[-5:].set(jnp.median(cube_filt[-16:-6], axis=0))
    return cube_filt


@functools.partial(jax.jit, static_argnames=('box_size', 'instrument'))
def noise_plane_spatial(std_dev, space_thresh=15., *, box_size=5, instrument='NIRISS'):
    """Flag spatial outliers in a temporal scatter plane.

    Parameters
    ----------
    std_dev : array-like(float)
        Temporal scatter per detector pixel.
    space_thresh : float
        Spatial rejection threshold in MAD scatter units. Traced.
    box_size : int
        Interpolation box half-width in pixels. Static: changes trigger a recompile.
    instrument : str
        Instrument name. Static: changes trigger a recompile.

    Returns
    -------
    flags : array-like(bool)
        Spatial outlier map.
    filled, replaced : array-like(float)
        Zero-filled scatter and scatter with spatial outliers replaced, respectively.
    """
    instrument = instrument.upper()
    dimy, dimx = std_dev.shape
    ymax, ybox, xbox = _detector_geometry(dimy, box_size, instrument)
    filled = jnp.where(std_dev == 0, jnp.nanmedian(std_dev), std_dev)
    thresh = 1e6 if instrument == 'MIRI' else space_thresh
    med, scatter = _box_stats_mad(filled, ybox, xbox)
    inbounds = _detector_inbounds(dimy, dimx, ymax)
    flags = inbounds & (jnp.abs(filled - med) >= thresh * scatter)
    replaced = jnp.where(flags, med, filled)
    return flags, filled, replaced


@jax.jit
def temporal_std(newdata):
    """Calculate per-pixel standard deviation over integrations.

    Parameters
    ----------
    newdata : array-like(float)
        Interpolated data with shape (nints, dimy, dimx).

    Returns
    -------
    std : array-like(float)
        NaN-aware standard deviation with the integration axis removed.
    """
    return jnp.nanstd(newdata, axis=0)


@functools.partial(jax.jit, static_argnames=('window_size', 'instrument', 'box_size'))
def prepare_temporal_badpix(newdata, *, window_size=5, instrument='NIRISS', scatter=None,
                            space_thresh=15., box_size=5):
    """Prepare the running median and scatter for temporal rejection.

    Parameters
    ----------
    newdata : array-like(float)
        Interpolated data with shape (nints, dimy, dimx).
    window_size : int
        Time median window. Static: changes trigger a recompile.
    instrument : str
        Instrument name. Static: changes trigger a recompile.
    scatter : None, array-like(float)
        Prepared scatter plane, used directly when supplied.
    space_thresh : float
        Spatial rejection threshold in MAD scatter units. Traced.
    box_size : int
        Interpolation box half-width in pixels. Static: changes trigger a recompile.

    Returns
    -------
    cube_filt, std_dev : array-like(float)
        Running median cube and prepared detector scatter plane, respectively.
    """
    instrument = instrument.upper()
    cube_filt = temporal_filter(newdata, window_size=window_size, instrument=instrument)
    if scatter is None:
        _, _, std_dev = noise_plane_spatial(
            temporal_scatter(newdata), space_thresh, box_size=int(box_size), instrument=instrument)
    else:
        std_dev = jnp.asarray(scatter, dtype=newdata.dtype)
    return cube_filt, std_dev


@functools.partial(jax.jit, static_argnames=('preserve_saturated', 'clear_interpolated_dq'))
def apply_temporal_badpix(newdata, newdq, saturated, cube_filt, std_dev,
                          time_thresh=10., preserve_saturated=False, clear_interpolated_dq=False):
    """Replace temporal outliers using the prepared running median.

    Parameters
    ----------
    newdata, newdq : array-like(float), array-like(int)
        Interpolated data cube and updated uint32 flags, respectively.
    saturated : array-like(bool)
        Per-integration saturation mask matching the data.
    cube_filt, std_dev : array-like(float)
        Running median cube and prepared detector scatter plane, respectively.
    time_thresh : float
        Temporal rejection threshold in scatter units. Traced.
    preserve_saturated, clear_interpolated_dq : bool
        Preserve saturation and clear replaced flags, respectively. Static: triggers a recompile.

    Returns
    -------
    newdata, newdq : array-like(float), array-like(int)
        Interpolated data cube and updated uint32 flags, respectively.
    """
    scale = jnp.abs(newdata - cube_filt) / std_dev
    tflag = scale > time_thresh
    if preserve_saturated:
        tflag = tflag & ~saturated
    return _replace_flagged(newdata, newdq, tflag, cube_filt, clear_interpolated_dq)


@functools.partial(jax.jit, static_argnames=('median_high_variance', 'preserve_saturated'))
def apply_high_variance(newdata, flags, saturated_any, *, median_high_variance=False,
                        preserve_saturated=False):
    """Optionally replace high-variance pixels with their time median.

    Carry high-variance flags separately because the corresponding v1 bit exceeds uint32.

    Parameters
    ----------
    newdata : array-like(float)
        Interpolated data with shape (nints, dimy, dimx).
    flags, saturated_any : array-like(bool)
        High-variance flags and pixels saturated in any integration, respectively.
    median_high_variance : bool
        Replace high-variance pixels with their time median. Static: changes trigger a recompile.
    preserve_saturated : bool
        Exclude saturated pixels from interpolation. Static: changes trigger a recompile.

    Returns
    -------
    newdata : array-like(float)
        Data after any high-variance replacement.
    """
    if not median_high_variance:
        return newdata
    replace = flags
    if preserve_saturated:
        replace = replace & ~saturated_any
    stack = jnp.nanmedian(newdata, axis=0)
    return jnp.where(replace[None], stack[None], newdata)


@functools.partial(jax.jit, static_argnames=('instrument', 'clear_interpolated_dq'))
def finalize_badpix(newdata, err_out, newdq, cube_filt, saturated, *,
                    instrument='NIRISS', clear_interpolated_dq=False):
    """Replace residual NaNs and negative values and clean NIRISS artifacts.

    Supply the running median computed after the first temporal pass for final NaN replacement.

    Parameters
    ----------
    newdata, err_out, newdq, cube_filt : array-like(float), array-like(int)
        Interpolated flux, uncertainty, flags and running median cubes with matching shapes.
    saturated : array-like(bool)
        Per-integration saturation mask matching the data.
    instrument : str
        Instrument name. Static: changes trigger a recompile.
    clear_interpolated_dq : bool
        Clear flags on interpolated pixels. Static: changes trigger a recompile.

    Returns
    -------
    newdata, err_out, newdq : array-like(float), array-like(int)
        Corrected flux, uncertainty (NaNs replaced) and uint32 quality cubes, respectively.
    """
    instrument = instrument.upper()
    dimy, dimx = newdata.shape[-2:]
    newdata = jnp.where(jnp.isnan(newdata), cube_filt, newdata)
    newdata = jnp.where(newdata < 0, 0., newdata)
    newdata = jnp.where(jnp.isnan(newdata), 0., newdata)

    # Replace the NIRISS artifact near the right detector edge.
    if instrument == 'NIRISS' and dimx == 2048 and dimy >= 96:
        mm = jnp.nanmedian(jnp.concatenate(
            [newdata[:, 82:84, 2018:], newdata[:, 88:90, 2018:]], axis=1), axis=1)
        newdata = newdata.at[:, 84:88, 2018:].set(mm[:, None, :])
    # Zero the NIRISS reference pixels.
    if instrument == 'NIRISS':
        newdata = newdata.at[:, :, :5].set(0.)
        newdata = newdata.at[:, :, -5:].set(0.)
        newdata = newdata.at[:, -5:].set(0.)

    if clear_interpolated_dq:
        newdq = newdq | jnp.where(saturated, core.DQ_SATURATED, jnp.uint32(0))
    return newdata, err_out, newdq.astype(jnp.uint32)


@functools.partial(jax.jit, static_argnames=('box_size', 'window_size', 'instrument',
                                    'median_high_variance', 'preserve_saturated',
                                    'clear_interpolated_dq', 'return_high_variance'))
def badpixstep(cube, err, dq, deepframe, space_thresh=15., time_thresh=10.,
               *, box_size=5, window_size=5, instrument='NIRISS',
               spatial_badpix=None, median_high_variance=False,
               preserve_saturated=False, clear_interpolated_dq=False, return_high_variance=False):
    """Detect and interpolate spatial and temporal bad pixels.

    Run a spatial pass followed by temporal outlier and high-variance passes.

    Parameters
    ----------
    cube, err, dq : array-like(float), array-like(int)
        Flux, uncertainty and quality cubes with shape (nints, dimy, dimx).
    deepframe : array-like(float)
        Median detector image.
    space_thresh, time_thresh : float
        Spatial MAD and temporal scatter rejection thresholds, respectively. Traced.
    box_size : int
        Interpolation box half-width in pixels. Static: changes trigger a recompile.
    window_size : int
        Time median window. Static: changes trigger a recompile.
    instrument : str
        Instrument name. Static: changes trigger a recompile.
    spatial_badpix : None, array-like(bool)
        Spatial map from an earlier segment; None discovers a map and clips negative inputs.
    median_high_variance : bool
        Replace high-variance pixels with their time median. Static: changes trigger a recompile.
    preserve_saturated, clear_interpolated_dq : bool
        Preserve saturation and clear replaced flags, respectively. Static: triggers a recompile.
    return_high_variance : bool
        Append the high-variance map (v1 bit beyond uint32). Static: changes trigger a recompile.

    Returns
    -------
    newdata, err_out, newdq : array-like(float), array-like(int)
        Corrected flux, uncertainty (NaNs replaced) and uint32 quality cubes, respectively.
    badpix, high_variance : array-like(bool)
        Spatial and high-variance maps; high_variance requires return_high_variance=True.
    """
    instrument = instrument.upper()
    _, dimy, dimx = cube.shape
    err_out, saturated, saturated_any = prepare_badpix_common(err, dq)

    if spatial_badpix is None:
        newdata = jnp.where(cube < 0, 0., cube)
        prepared_spatial = prepare_spatial_badpix(deepframe, dq, saturated_any, box_size=box_size,
            instrument=instrument, preserve_saturated=preserve_saturated)
        badpix = spatial_badpix_from_prepared(*prepared_spatial, space_thresh=space_thresh)
    else:
        newdata = cube
        badpix = jnp.asarray(spatial_badpix, dtype=bool)
        if badpix.shape != (dimy, dimx):
            raise ValueError('spatial_badpix must match the detector plane')

    newdata, newdq = apply_spatial_badpix(
        newdata, dq, badpix, box_size=box_size, instrument=instrument,
        clear_interpolated_dq=clear_interpolated_dq)
    # Replace temporal outliers using the prepared running median.
    cube_filt, std_dev = prepare_temporal_badpix(
        newdata, window_size=window_size, instrument=instrument,
        space_thresh=space_thresh, box_size=box_size)
    newdata, newdq = apply_temporal_badpix(newdata, newdq, saturated, cube_filt, std_dev,
        time_thresh=time_thresh, preserve_saturated=preserve_saturated,
        clear_interpolated_dq=clear_interpolated_dq)
    cube_filt = temporal_filter(newdata, window_size=window_size, instrument=instrument)
    # Flag high-variance pixels after temporal outlier replacement.
    high_variance, _, _ = noise_plane_spatial(
        temporal_std(newdata), space_thresh, box_size=box_size,
        instrument=instrument)
    newdata = apply_high_variance(newdata, high_variance, saturated_any,
        median_high_variance=median_high_variance, preserve_saturated=preserve_saturated)
    newdata, err_out, newdq = finalize_badpix(newdata, err_out, newdq, cube_filt, saturated,
        instrument=instrument, clear_interpolated_dq=clear_interpolated_dq)
    if return_high_variance:
        return newdata, err_out, newdq, badpix, high_variance
    return newdata, err_out, newdq, badpix
