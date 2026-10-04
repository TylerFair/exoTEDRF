"""Calculate exact scalar statistics with bounded host memory."""

import itertools
import os
import warnings

import numpy as np

from exotedrf.v2 import core


def _blocks(array, chunk_bytes):
    """Yield host array tiles bounded by the requested byte count."""
    shape = tuple(array.shape)
    tile = list(shape)
    count = max(1, chunk_bytes // np.dtype(array.dtype).itemsize)
    for axis in range(len(shape)):
        if int(np.prod(tile)) <= count:
            break
        following = int(np.prod(tile[axis + 1:]))
        tile[axis] = min(tile[axis], max(1, count // following))
    starts = (range(0, size, width) for size, width in zip(shape, tile))
    for offset in itertools.product(*starts):
        key = tuple(slice(first, min(first + width, size))
            for first, width, size in zip(offset, tile, shape))
        yield np.asarray(array[key])


def _statistic_indices(count, q, median):
    """Return bracketing sample indices and the percentile interpolation weight."""
    if median:
        return (count - 1) // 2, count // 2, None
    position = (count - 1) * (float(q) / 100.)
    lo, hi = int(np.floor(position)), int(np.ceil(position))
    return lo, hi, position - lo


def _interpolate_statistic(pair, lo, hi, gamma):
    """Interpolate a percentile pair, or average a median pair when gamma is None."""
    if gamma is None:
        # Preserve the input floating dtype when averaging the median pair.
        return np.mean(pair[:1] if lo == hi else pair)
    return np.quantile(pair, gamma)


def _scalar_statistic(array, q, *, median=False):
    """Calculate an exact scalar statistic using host blocks and in-place partitioning."""
    dtype = np.dtype(array.dtype)
    size = int(np.prod(array.shape))
    # Reserve memory for pipeline arrays and device transfers.
    chunk_bytes = max(dtype.itemsize, min(32 << 20, core.host_allocation_budget_bytes() // 8))
    direct_limit = min(128 << 20, chunk_bytes * 2)
    if size * dtype.itemsize <= direct_limit:
        source = np.asarray(array)
        return (np.nanmedian(source) if median else np.nanpercentile(source, q))
    sample = core.empty_host_array((size,), dtype, name='exotedrf-order-statistic',
        max_bytes=max(1, core.host_allocation_available_bytes() // 4))
    count = 0
    for block in _blocks(array, chunk_bytes):
        values = block.reshape(-1)
        values = values[~np.isnan(values)]
        sample[count:count + values.size] = values
        count += values.size
    if not count:
        warnings.warn('All-NaN slice encountered', RuntimeWarning, stacklevel=2)
        return dtype.type(np.nan)
    lo, hi, gamma = _statistic_indices(count, q, median)
    compact = sample[:count]
    compact.partition((lo, hi))
    pair = np.asarray([compact[lo], compact[hi]], dtype=dtype)
    return _interpolate_statistic(pair, lo, hi, gamma)


def _device_order_statistic_pair(array, lo, hi):
    """Select two order statistics on the device, or return None if unavailable."""
    import jax
    import jax.numpy as jnp
    resident = core.is_device_array(array)
    if not resident:
        policy = os.environ.get('EXOTEDRF_DEVICE_FAST_PATH', 'auto').lower()
        if policy in ('0', 'false', 'off', 'stream', 'streamed'):
            return None
        if jax.default_backend() != 'gpu':
            return None
    nbytes = int(np.prod(array.shape)) * np.dtype(array.dtype).itemsize
    # Allow space for the input, sorted output, and sort temporaries.
    if nbytes * 6 > core.device_memory_bytes() * 0.7:
        return None
    flat = (array.reshape(-1) if resident else jnp.asarray(np.ascontiguousarray(array)).reshape(-1))
    ordered = jnp.sort(flat)
    pair = np.asarray(jax.device_get(ordered[jnp.asarray([lo, hi])]))
    del ordered, flat
    return pair


def _resident_nan_count(array):
    """Count NaN samples of a device-resident array without a host copy."""
    import jax.numpy as jnp
    if not np.issubdtype(np.dtype(array.dtype), np.inexact):
        return 0
    return int(jnp.count_nonzero(jnp.isnan(array)))


def _resident_statistic(array, q, *, median=False):
    """Calculate a statistic by device sorting for resident or host arrays."""
    dtype = np.dtype(array.dtype)
    resident = core.is_device_array(array)
    nan_count = (_resident_nan_count(array) if resident else int(np.count_nonzero(np.isnan(array))))
    count = int(np.prod(array.shape)) - nan_count
    if not count:
        if resident:
            warnings.warn('All-NaN slice encountered', RuntimeWarning, stacklevel=3)
            return dtype.type(np.nan)
        return None
    lo, hi, gamma = _statistic_indices(count, q, median)
    pair = _device_order_statistic_pair(array, lo, hi)
    if pair is None:
        return None
    pair = np.asarray(pair, dtype=dtype)
    return _interpolate_statistic(pair, lo, hi, gamma)


def nanpercentile_fast(array, q):
    """Calculate the NumPy linear nanpercentile using device sorting when available.

    Parameters
    ----------
    array : array-like(float)
        Input samples. NaNs are excluded; infinities are retained.
    q : float
        Scalar percentile between 0 and 100.

    Returns
    -------
    value : float
        Percentile matching NumPy linear interpolation.
    """
    if core.is_device_array(array):
        # Reduce device arrays before transferring values to the host.
        result = _resident_statistic(array, q)
        if result is not None:
            return result
        array = core.to_host(array)
    dtype = np.dtype(array.dtype)
    size = int(np.prod(array.shape))
    if size * dtype.itemsize <= (128 << 20):
        return np.nanpercentile(np.asarray(array), q)
    result = _resident_statistic(array, q)
    if result is None:
        return nanpercentile(array, q)
    return result


def nanmedian(array):
    """Calculate the NumPy nanmedian with bounded host memory.

    Parameters
    ----------
    array : array-like(float)
        Input samples. NaNs are excluded; infinities are retained.

    Returns
    -------
    value : float
        Median using the input floating dtype for the central mean.
    """
    if core.is_device_array(array):
        result = _resident_statistic(array, 50., median=True)
        if result is not None:
            return result
        # Read oversized device arrays in bounded blocks.
    return _scalar_statistic(array, 50., median=True)


def nanpercentile(array, q):
    """Calculate the NumPy linear nanpercentile with bounded host memory.

    Parameters
    ----------
    array : array-like(float)
        Input samples. NaNs are excluded; infinities are retained.
    q : float
        Scalar percentile between 0 and 100.

    Returns
    -------
    value : float
        Percentile matching NumPy linear interpolation.
    """
    if not np.isscalar(q) or not 0 <= float(q) <= 100:
        raise ValueError('q must be a scalar percentile between 0 and 100')
    if core.is_device_array(array):
        result = _resident_statistic(array, q)
        if result is not None:
            return result
    return _scalar_statistic(array, q)
