"""Subtract the MIRI reset anomaly reference from detector ramps."""

import jax
import jax.numpy as jnp

__all__ = ['reset_correct']


@jax.jit
def reset_correct(data, pixeldq, reset_data, reset_dq, int_start=0):
    """Subtract the reset anomaly reference from a science ramp segment.

    Parameters
    ----------
    data : array-like(float)
        Detector samples with shape (nints, ngroups, dimy, dimx).
    pixeldq, reset_dq : array-like(int)
        Science and reset reference pixel flags with shape (dimy, dimx).
    reset_data : array-like(float)
        Reference (reset_nints, reset_ngroups, dimy, dimx); NaNs give zero correction.
    int_start : int
        Zero-based absolute index of the first science integration. Traced.

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    pixeldq : array-like(int)
        Science pixel flags combined with reset reference flags.
    """
    nints, ngroups = data.shape[0], data.shape[1]
    reset_nints, reset_ngroups = reset_data.shape[0], reset_data.shape[1]
    igroup = min(ngroups, reset_ngroups)
    safe_reset = jnp.where(jnp.isnan(reset_data), 0., reset_data).astype(data.dtype)
    # Reuse the final reference integration after the reference ends.
    ref_idx = jnp.clip(int_start + jnp.arange(nints), 0, reset_nints - 1)
    corr = safe_reset[ref_idx][:, :igroup]
    data = data.at[:, :igroup].add(-corr)
    pixeldq = jnp.asarray(pixeldq, jnp.uint32) | jnp.asarray(reset_dq, jnp.uint32)
    return data, pixeldq
