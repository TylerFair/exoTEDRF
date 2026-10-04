"""Array operations shared by detector corrections and spectral extraction."""

import jax
import jax.numpy as jnp


def masked_to_nan(x, bad_mask):
    """Replace flagged measurements with NaN so statistics ignore them."""
    return jnp.where(bad_mask, jnp.nan, x)


def _pixel_windows(img, box_half):
    """Collect NaN-padded square windows around detector pixels."""
    size = 2 * box_half + 1
    pad = ((0, 0),) * (img.ndim - 2) + ((box_half, box_half), (box_half, box_half))
    padded = jnp.pad(img, pad, constant_values=jnp.nan)
    windows = []
    for dy in range(size):
        for dx in range(size):
            windows.append(jax.lax.dynamic_slice_in_dim(
                jax.lax.dynamic_slice_in_dim(padded, dy, img.shape[-2], axis=-2),
                dx, img.shape[-1], axis=-1))
    return jnp.stack(windows, axis=0)


def nanmedian_filter_2d(img, box_half):
    """Measure the local median around every detector pixel while ignoring NaN.

    Parameters
    ----------
    img : array-like(float)
        Detector image; any leading axes are retained.
    box_half : int
        Neighboring rows and columns on each side. Static: changing it triggers a recompile.

    Returns
    -------
    median : array-like(float)
        Local NaN-aware median for each detector pixel.
    """
    return jnp.nanmedian(_pixel_windows(img, box_half), axis=0)


def nan_stats_filter_2d(img, box_half):
    """Measure local median and scatter around every valid detector pixel.

    Parameters
    ----------
    img : array-like(float)
        Detector image; any leading axes are retained.
    box_half : int
        Neighboring rows and columns on each side. Static: changing it triggers a recompile.

    Returns
    -------
    median, scatter : array-like(float)
        Local NaN-aware median and standard deviation, respectively.
    """
    stack = _pixel_windows(img, box_half)
    return jnp.nanmedian(stack, axis=0), jnp.nanstd(stack, axis=0)


def running_median_time(cube, window):
    """Calculate v1's running median through the integration axis.

    Reflect the temporal edges as in SciPy; this median does not ignore NaN samples.

    Parameters
    ----------
    cube : array-like(float)
        Input cube with integration as its leading axis.
    window : int
        Temporal window length. Static: changing it triggers a recompile.

    Returns
    -------
    filtered : array-like(float)
        Running integration median with reflected temporal edges.
    """
    half = window // 2
    n = cube.shape[0]
    padded = jnp.pad(cube, ((half, half),) + ((0, 0),) * (cube.ndim - 1), mode='symmetric')
    windows = [jax.lax.dynamic_slice_in_dim(padded, k, n, axis=0) for k in range(window)]
    return jnp.median(jnp.stack(windows, axis=0), axis=0)


def dilate_mask(mask, box_half):
    """Expand each flagged pixel to nearby pixels, as used for saturation spill.

    Callers must mask detector borders to avoid wrapped flags.

    Parameters
    ----------
    mask : array-like(bool)
        Flagged detector pixels.
    box_half : int
        Neighboring rows and columns on each side. Static: changing it triggers a recompile.

    Returns
    -------
    expanded : array-like(bool)
        Mask expanded by the requested pixel neighborhood.
    """
    out = mask
    for dy in range(-box_half, box_half + 1):
        for dx in range(-box_half, box_half + 1):
            if dy == 0 and dx == 0:
                continue
            out = out | jnp.roll(jnp.roll(mask, dy, axis=-2), dx, axis=-1)
    return out


def box_aperture_weights(centroid_y, halfwidth, dimy):
    """Calculate fractional pixel weights for a box extraction aperture.

    Parameters
    ----------
    centroid_y : array-like(float)
        Trace center in each detector column.
    halfwidth : float
        Distance from the trace center to each aperture edge in pixels.
    dimy : int
        Number of detector rows. Static: changing it triggers a recompile.

    Returns
    -------
    weights : array-like(float)
        Pixel overlap fractions with detector row and column axes.
    """
    y = jnp.arange(dimy, dtype=centroid_y.dtype)[:, None]
    low = centroid_y[None, :] - halfwidth
    high = centroid_y[None, :] + halfwidth
    overlap = jnp.clip(jnp.minimum(y + 0.5, high) - jnp.maximum(y - 0.5, low), 0.0, 1.0)
    return overlap


def bitwise_or_reduce_uint(dq, flags):
    """Merge DQ flags without changing the observation's integer DQ type."""
    return (dq | flags).astype(dq.dtype)
