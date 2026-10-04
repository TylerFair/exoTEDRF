"""Background subtraction for SOSS and MIRI integration cubes."""

import functools

import jax
import jax.numpy as jnp


def _estimate_scale(region_deep, region_model, shift):
    """Estimate the background scale from second-quartile flux ratios."""
    ratio = (region_deep + shift) / region_model
    ratio = jnp.where(jnp.isfinite(ratio), ratio, jnp.nan)
    q = jnp.nanpercentile(ratio, jnp.array([25., 50.]))
    use = (ratio > q[0]) & (ratio < q[1])
    med_use = jnp.nanmedian(jnp.where(use, ratio, jnp.nan))
    med_all = jnp.nanmedian(ratio)
    # Use all finite ratios when the second quartile is empty.
    return jnp.where(jnp.any(use), med_use, med_all)


def _scale_with_shift(region_deep, region_model):
    """Estimate a non-negative background scale and its additive shift."""
    mm = jnp.nanmedian(region_model)
    good_mm = jnp.isfinite(mm) & (mm != 0)

    def cond(state):
        """Check whether the loop should continue."""
        scale, _ = state
        return (scale < 0) & good_mm

    def body(state):
        """Update the loop state."""
        scale, shift = state
        shift = shift - scale * mm
        return _estimate_scale(region_deep, region_model, shift), shift

    # Match the loop carry to the deepstack precision.
    zero = jnp.zeros((), dtype=region_deep.dtype)
    scale0 = _estimate_scale(region_deep, region_model, zero)
    scale, shift = jax.lax.while_loop(cond, body, (scale0, zero))
    scale = jnp.where(jnp.isfinite(scale), scale, 0.)
    scale = jnp.where(scale < 0, 0., scale)
    return scale, shift


def _default_regions(dimy):
    """Get the default SOSS background scaling regions."""
    if dimy == 96:
        region1 = (5, 21, 5, 401)
        region2 = None
    else:
        region1 = (230, 250, 350, 550)
        region2 = (235, 250, 715, 750)
    return region1, region2


@functools.partial(jax.jit, static_argnames=('region1', 'region2', 'differential'))
def background_soss(cube, deepstack, background_model, region1=None,
                    region2=None, differential=False, scale1=None, scale2=None):
    """Scale and subtract the SOSS background model.

    Parameters
    ----------
    cube : array-like(float)
        Flux cube with shape (nints, dimy, dimx).
    deepstack, background_model : array-like(float)
        Baseline image and SOSS background model with shape (dimy, dimx).
    region1, region2 : None, tuple[int]
        Boxes (row_low, row_high, col_low, col_high); None uses defaults. Static: recompiles.
    differential : bool
        Scale both sides of the background step independently. Static: changes trigger a recompile.
    scale1, scale2 : None, float
        Scale overrides for the first and differential regions; scale1 skips shifting. Traced.

    Returns
    -------
    cube_corr : array-like(float)
        Background-subtracted cube.
    model_scaled : array-like(float)
        Subtracted detector background.
    scale1, scale2, shift : float
        Scale factors and shift; scale2 equals scale1 outside differential mode.
    """
    dimy, dimx = deepstack.shape
    r1_default, r2_default = _default_regions(dimy)
    if region1 is None:
        region1 = r1_default
    if region2 is None:
        region2 = r2_default
    if differential and region2 is None:
        raise NotImplementedError('No default differential region for SUBSTRIP96 (v1 '
            'stage2.py:1116).')

    rl, rh, cl, ch = region1
    if scale1 is None:
        scale1, shift = _scale_with_shift(deepstack[rl:rh, cl:ch], background_model[rl:rh, cl:ch])
    else:
        scale1 = jnp.asarray(scale1, dtype=deepstack.dtype)
        shift = jnp.zeros((), dtype=deepstack.dtype)

    if differential:
        if scale2 is None:
            rl2, rh2, cl2, ch2 = region2
            scale2 = _estimate_scale(deepstack[rl2:rh2, cl2:ch2],
                                     background_model[rl2:rh2, cl2:ch2], shift)
            scale2 = jnp.where(jnp.isfinite(scale2), scale2, 0.)
            scale2 = jnp.where(scale2 < 0, 0., scale2)
        else:
            scale2 = jnp.asarray(scale2, dtype=deepstack.dtype)

        # Locate the background step from the model gradient.
        grad_bkg = jnp.gradient(background_model, axis=1)
        step_pos = jnp.argmax(grad_bkg[:, 10:-10], axis=1) + 10 - 4
        left = jnp.arange(dimx)[None, :] < step_pos[:, None]
        model_scaled = jnp.where(left, background_model * scale1 - shift,
                                 background_model * scale2 - shift)
    else:
        scale2 = scale1
        model_scaled = background_model * scale1 - shift

    cube_corr = cube - model_scaled[None]
    return cube_corr, model_scaled, scale1, scale2, shift


def _masked_linfit(w, y, xx):
    """Fit a weighted line along the last axis."""
    wf = w.astype(y.dtype)
    yw = jnp.where(w, y, 0.)
    n = wf.sum(-1)
    sx = (wf * xx).sum(-1)
    sy = yw.sum(-1)
    sxx = (wf * xx * xx).sum(-1)
    sxy = (yw * xx).sum(-1)
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return intercept, slope


@functools.partial(jax.jit, static_argnames=('method', 'trace_center'))
def background_miri(cube, trace_width=20., background_width=14., *,
                    method='slope', trace_center=36):
    """Subtract a median or linear background from MIRI data.

    The slope method uses five clipping rounds and a final ordinary least-squares fit.

    Parameters
    ----------
    cube : array-like(float)
        Flux cube with shape (nints, dimy, dimx).
    trace_width, background_width : float
        Full trace width and width of each background region, in pixels. Traced.
    method : str
        Use median or slope background subtraction. Static: changes trigger a recompile.
    trace_center : int
        Column at the center of the MIRI trace. Static: changes trigger a recompile.

    Returns
    -------
    cube_corr, bkg : array-like(float)
        Background-subtracted cube and subtracted background with the input shape.
    """
    _, _, dimx = cube.shape
    c = trace_center
    thw = jnp.floor(trace_width / 2.)
    bw = background_width
    xx = jnp.arange(dimx, dtype=cube.dtype)

    # Include background region endpoints for the median method.
    if method == 'median':
        colmask = (((xx >= c - thw - bw) & (xx <= c - thw)) |
                   ((xx >= c + thw) & (xx <= c + thw + bw)))
        bkg_cols = jnp.where(colmask[None, None, :], cube, jnp.nan)
        bkg_row = jnp.nanmedian(bkg_cols, axis=2)
        bkg = jnp.broadcast_to(bkg_row[:, :, None], cube.shape)
    # Exclude background region endpoints for the slope method.
    elif method == 'slope':
        colmask = (((xx > c - thw - bw) & (xx < c - thw)) | ((xx > c + thw) & (xx < c + thw + bw)))
        valid = colmask[None, None, :] & jnp.isfinite(cube)
        w = valid
        for _ in range(5):
            a, b = _masked_linfit(w, cube, xx)
            res = cube - (a[..., None] + b[..., None] * xx)
            resv = jnp.where(valid, res, jnp.nan)
            stddev = jnp.nanstd(resv, axis=-1)
            w = valid & (jnp.abs(res) <= 3. * stddev[..., None])
        a, b = _masked_linfit(w, cube, xx)
        bkg = a[..., None] + b[..., None] * xx
    else:
        raise ValueError('Unknown method: {}.'.format(method))

    cube_corr = cube - bkg
    return cube_corr, bkg
