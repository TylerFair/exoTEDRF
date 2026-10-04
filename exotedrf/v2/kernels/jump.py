"""Time-domain jump detection and detector reset artifact masks."""

import functools
import os
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from exotedrf.v2 import core, hoststats, mathutils


_DEFAULT_JUMP_CHUNK_COLS = 512


class JumpInvariants(NamedTuple):
    """Store scatter and flux floors shared by jump threshold and window choices.

    Attributes
    ----------
    scatter : array-like(float)
        Temporal point-to-point scatter per group and pixel.
    flux_floor : array-like(float)
        Global tenth-percentile flux per group.
    """

    scatter: Any
    flux_floor: Any


class PreparedTimeJump(NamedTuple):
    """Store normalized deviations, filtered data and group flux floors.

    Attributes
    ----------
    scale : array-like(float)
        Absolute normalized deviations, matching the detector cube.
    cube_filt : array-like(float)
        Running median cube with edge integrations replaced.
    flux_floor : array-like(float)
        Global tenth-percentile flux per group.
    """

    scale: Any
    cube_filt: Any
    flux_floor: Any


@jax.jit
def point_to_point_scatter(cube):
    """Calculate temporal point-to-point scatter for jump detection.

    Parameters
    ----------
    cube : array-like(float)
        Data with integrations first and detector rows and columns last.

    Returns
    -------
    scatter : array-like(float)
        Median absolute second difference; zero scatter becomes infinity.
    """
    scatter = jnp.median(jnp.abs(0.5 * (cube[0:-2] + cube[2:]) - cube[1:-1]), axis=0)
    return jnp.where(scatter == 0, jnp.inf, scatter)


@functools.partial(jax.jit, static_argnames=('window',))
def _scatter_normalize_with_scatter(cube, scatter, window=5):
    """Normalize running-median deviations by the supplied scatter."""
    cube_filt = mathutils.running_median_time(cube, window)
    # Read both edge replacements before updating the filtered cube.
    first, last = cube_filt[3], cube_filt[-3]
    cube_filt = cube_filt.at[:2].set(first[None])
    cube_filt = cube_filt.at[-2:].set(last[None])
    scale = jnp.abs(cube - cube_filt) / scatter[None]
    return scale, cube_filt


@functools.partial(jax.jit, static_argnames=('window',))
def scatter_normalize(cube, window=5):
    """Normalize temporal running-median deviations by point-to-point scatter.

    Parameters
    ----------
    cube : array-like(float)
        Data with integrations first and detector rows and columns last.
    window : int
        Time median window. Static: changes trigger a recompile.

    Returns
    -------
    scale : array-like(float)
        Absolute normalized deviations matching the input cube.
    cube_filt : array-like(float)
        Running median with edge integrations replaced.
    """
    scatter = point_to_point_scatter(cube)
    return _scatter_normalize_with_scatter(cube, scatter, window=window)


@jax.jit
def group_flux_floor(data):
    """Calculate the global tenth-percentile flux for each group.

    Compute on the complete segment before chunking detector columns.

    Parameters
    ----------
    data : array-like(float)
        Detector samples with shape (nints, ngroups, dimy, dimx).

    Returns
    -------
    flux_floor : array-like(float)
        Flux floor with shape (ngroups,).
    """
    return jnp.nanpercentile(data, 10.0, axis=(0, 2, 3))


@jax.jit
def prepare_jump_invariants(data, flux_floor=None):
    """Prepare scatter and flux floors shared by temporal jump choices.

    Parameters
    ----------
    data : array-like(float)
        Detector samples with shape (nints, ngroups, dimy, dimx).
    flux_floor : None, array-like(float)
        Full-segment tenth-percentile flux per group, including when chunking columns.

    Returns
    -------
    invariants : JumpInvariants
        Temporal scatter and global flux floors.
    """
    if flux_floor is None:
        flux_floor = group_flux_floor(data)
    return JumpInvariants(point_to_point_scatter(data), flux_floor)


@functools.partial(jax.jit, static_argnames=('window',))
def prepare_jumps_in_time(data, window=5, invariants=None, flux_floor=None):
    """Prepare one running median window for temporal jump detection.

    Supplied invariants take precedence over flux_floor.

    Parameters
    ----------
    data : array-like(float)
        Detector samples with shape (nints, ngroups, dimy, dimx).
    window : int
        Time median window. Static: changes trigger a recompile.
    invariants : None, JumpInvariants
        Prepared scatter and flux floors; None computes them from the data.
    flux_floor : None, array-like(float)
        Full-segment tenth-percentile flux per group, including when chunking columns.

    Returns
    -------
    prepared : PreparedTimeJump
        Normalized deviations, filtered data and flux floors.
    """
    if invariants is None:
        invariants = prepare_jump_invariants(data, flux_floor=flux_floor)
    scale, cube_filt = _scatter_normalize_with_scatter(data, invariants.scatter, window=window)
    return PreparedTimeJump(scale, cube_filt, invariants.flux_floor)


def default_max_reset_int(instrument, grating=None, detector=None):
    """Get the last integration affected by the detector reset artifact.

    Parameters
    ----------
    instrument : str
        Instrument name.
    grating, detector : None, str
        NIRSpec grating and detector names, respectively.

    Returns
    -------
    max_reset_int : int
        Instrument and grating-dependent artifact limit.
    """
    if instrument == 'NIRISS':
        return 256
    if grating in ['G395H', 'G235H', 'G140H']:
        if detector is not None and detector.lower() == 'nrs2':
            return 58
        return 62
    if grating in ['G395M', 'G235M', 'G140M']:
        return 81
    return 68


def reset_artifact_mask(nints, dimy, dimx, int_start=1, instrument='NIRISS', max_reset_int=None):
    """Build a mask of bright reset artifact rows per integration.

    Parameters
    ----------
    nints : int
        Number of integrations.
    dimy, dimx : int
        Detector row and column counts, respectively.
    int_start : int
        One-based integration index of the segment start.
    instrument : str
        Instrument name.
    max_reset_int : None, int
        Exclusive final integration affected by the artifact; None uses the instrument default.

    Returns
    -------
    artifact : np.ndarray(bool)
        Mask with shape (nints, dimy, dimx); the final segment integration is unmasked.
    """
    if max_reset_int is None:
        max_reset_int = default_max_reset_int(instrument)
    artifact = np.zeros((nints, dimy, dimx), dtype=bool)
    int_end = min(int_start + nints - 1, max_reset_int)
    if int_start < max_reset_int:
        for j, jj in enumerate(range(int_start, int_end)):
            if instrument == 'NIRISS':
                min_row = max(max_reset_int - (jj + 3), 0)
                max_row = min((max_reset_int + 2) - jj, dimy)
            else:
                min_row = max(max_reset_int - (jj + 2), 0)
                max_row = min(max_reset_int - jj, dimy)
            artifact[j, min_row:max_row, :] = True
    return artifact


def miri_dropped_groups(groupdq):
    """Identify MIRI groups whose first integration is entirely DO_NOT_USE.

    Parameters
    ----------
    groupdq : array-like(int)
        Group flags with shape (nints, ngroups, dimy, dimx).

    Returns
    -------
    dropped : np.ndarray(bool)
        Dropped group mask with shape (ngroups,).
    """
    dq0 = np.asarray(groupdq)[0]
    return ((dq0 & int(core.DQ_DO_NOT_USE)) != 0).all(axis=(-2, -1))


@jax.jit
def apply_jumps_in_time(data, groupdq, thresh, prepared, artifact=None, group_ok=None):
    """Apply a temporal jump threshold to prepared deviations.

    Short ramps replace only values whose flags equal zero or JUMP_DET.

    Parameters
    ----------
    data, groupdq : array-like(float), array-like(int)
        Detector samples and group flags with shape (nints, ngroups, dimy, dimx).
    thresh : float
        Rejection threshold in scatter units. Traced.
    prepared : PreparedTimeJump
        Normalized deviations and running median from the same input data.
    artifact : None, array-like(bool)
        Reset artifact mask with shape (nints, dimy, dimx); True excludes pixels.
    group_ok : None, array-like(bool)
        True for groups eligible for jump detection.

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    groupdq : array-like(int)
        Updated group flags with the input shape.
    """
    ngroups = data.shape[1]
    ii = ((prepared.scale >= thresh) & (data > prepared.flux_floor[None, :, None, None]))
    if artifact is not None:
        ii = ii & ~artifact[:, None]
    if group_ok is not None:
        ii = ii & group_ok[None, :, None, None]

    if ngroups <= 2:
        # Replace short-ramp jumps only when DQ equals zero or JUMP_DET.
        jj = (groupdq == 0) | (groupdq == jnp.uint8(int(core.DQ_JUMP_DET)))
        replace = ii & jj
        data = jnp.where(replace, prepared.cube_filt, data)
        groupdq = jnp.where(replace, jnp.uint8(0), groupdq)
    else:
        jump_bit = jnp.uint8(int(core.DQ_JUMP_DET))
        already = (groupdq & jump_bit) != 0
        to_flag = ii & ~already
        groupdq = jnp.where(to_flag, groupdq | jump_bit, groupdq).astype(jnp.uint8)
    return data, groupdq


@functools.partial(jax.jit, static_argnames=('window',))
def flag_jumps_in_time(data, groupdq, thresh, window=5, artifact=None,
                       group_ok=None, flux_floor=None):
    """Detect jumps across integrations using temporal deviations.

    Ramps with at most two groups replace eligible jumps; longer ramps set JUMP_DET.

    Other parameters follow ``apply_jumps_in_time``.

    Parameters
    ----------
    window : int
        Time median window. Static: changes trigger a recompile.
    flux_floor : None, array-like(float)
        Full-segment tenth-percentile flux per group, including when chunking columns.

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    groupdq : array-like(int)
        Updated group flags with the input shape.
    """
    prepared = prepare_jumps_in_time(data, window=window, flux_floor=flux_floor)
    return apply_jumps_in_time(data, groupdq, thresh, prepared,
                               artifact=artifact, group_ok=group_ok)


@functools.partial(jax.jit, static_argnames=('window',))
def flag_jump_dq_in_time(data, groupdq, thresh, window=5, artifact=None,
                         group_ok=None, flux_floor=None):
    """Flag temporal jumps in ramps with more than two groups.

    Other parameters follow ``flag_jumps_in_time``.

    Returns
    -------
    groupdq : array-like(int)
        Updated group flags with the input shape.
    """
    if data.shape[1] <= 2:
        raise ValueError('DQ-only time-domain Jump requires ngroups > 2')
    prepared = prepare_jumps_in_time(data, window=window, flux_floor=flux_floor)
    _, corrected_dq = apply_jumps_in_time(data, groupdq, thresh, prepared, artifact=artifact,
        group_ok=group_ok)
    return corrected_dq


def _nbytes(array):
    """Get the array size without materializing its contents."""
    size = getattr(array, 'nbytes', None)
    if size is not None:
        return int(size)
    return int(np.prod(array.shape)) * int(np.dtype(array.dtype).itemsize)


def flag_jumps_in_time_chunked(data, groupdq, thresh, window=5, artifact=None, group_ok=None,
                               chunk_size=None):
    """Detect temporal jumps in detector column chunks.

    Keep every integration in each chunk and compute flux floors over the complete segment.

    Parameters
    ----------
    data, groupdq : array-like(float), array-like(int)
        Detector samples and group flags with shape (nints, ngroups, dimy, dimx).
    thresh : float
        Rejection threshold in scatter units. Traced.
    window : int
        Running median window length in integrations.
    artifact : None, array-like(bool)
        Reset artifact mask with shape (nints, dimy, dimx); True excludes pixels.
    group_ok : None, array-like(bool)
        True for groups eligible for jump detection.
    chunk_size : None, int
        Maximum chunk width; None selects a width within memory limits.

    Returns
    -------
    corrected : array-like(float)
        Corrected data with the input shape.
    groupdq : array-like(int)
        Updated group flags with the input shape.
    """
    # Reduce each group on the host before transferring detector column chunks.
    floor = np.asarray([
        hoststats.nanpercentile_fast(data[:, group], 10.)
        for group in range(data.shape[1])], dtype=data.dtype)
    if artifact is None:
        artifact = np.zeros((data.shape[0],) + data.shape[2:], dtype=bool)
    # Keep artifact masks on the device for resident detector data.
    if core.is_device_array(data) and core.is_device_array(groupdq):
        artifact = jnp.asarray(artifact)

    if chunk_size is None:
        specific = os.environ.get('EXOTEDRF_JUMP_CHUNK_COLS')
        global_override = os.environ.get('EXOTEDRF_CHUNK_COLS')
        if specific is not None:
            chunk_size = int(specific)
        elif global_override is not None:
            chunk_size = int(global_override)
        else:
            bytes_per_col = (_nbytes(data) + _nbytes(groupdq) +
                _nbytes(artifact)) // int(data.shape[-1])
            # Account for the running median window when bounding the workspace.
            live_buffers = max(24, 5 * int(window) + 3)
            chunk_size = min(_DEFAULT_JUMP_CHUNK_COLS, core.auto_chunk(
                    int(data.shape[-1]), bytes_per_col, n_buffers=live_buffers, headroom=0.6))

    dq_only = data.shape[1] > 2
    kernel = flag_jump_dq_in_time if dq_only else flag_jumps_in_time

    def apply_chunk(d, dq, art):
        """Apply temporal jump detection to one detector column chunk."""
        return kernel(d, dq, thresh, window=window, artifact=art, group_ok=group_ok,
            flux_floor=jnp.asarray(floor, dtype=d.dtype))

    result = core.map_over_cols(apply_chunk, (data, groupdq, artifact), data.shape[-1],
        chunk_size=chunk_size)
    return (data, result) if dq_only else result
