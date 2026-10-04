"""Observation containers, data quality flags, and memory allocation tools."""

import contextlib
import functools
import contextvars
import dataclasses
import itertools
import os
import re
import threading
import weakref

_YNN_FLAG = 'xla_cpu_experimental_ynn_fusion_type'


def _disable_xla_cpu_ynn_fusions():
    """Disable experimental CPU fusions that can change ramp-fit results."""
    flags = os.environ.get('XLA_FLAGS', '')
    if _YNN_FLAG in flags:
        return
    import importlib.metadata
    version = importlib.metadata.version('jaxlib')
    major, minor = (int(x) for x in version.split('.')[:2])
    if (major, minor) < (0, 10):
        return
    os.environ['XLA_FLAGS'] = f'{flags} --{_YNN_FLAG}='.strip()


_disable_xla_cpu_ynn_fusions()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def setup(cache_dir=None):
    """Prepare JAX for an exoTEDRF run.

    The default cache directory is selected by EXOTEDRF_JAX_CACHE.

    Parameters
    ----------
    cache_dir : None, str
        Directory in which to store compiled JAX calculations.
    """
    if cache_dir is None:
        cache_dir = os.environ.get('EXOTEDRF_JAX_CACHE',
                                   os.path.expanduser('~/.exotedrf_jax_cache'))
    jax.config.update('jax_compilation_cache_dir', cache_dir)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.5)


def enable_x64():
    """Enable 64-bit JAX arithmetic."""
    jax.config.update('jax_enable_x64', True)


DQ_DO_NOT_USE = np.uint32(1)
DQ_SATURATED = np.uint32(2)
DQ_JUMP_DET = np.uint32(4)
DQ_DROPOUT = np.uint32(8)
DQ_OUTLIER = np.uint32(16)
DQ_PERSISTENCE = np.uint32(32)
DQ_AD_FLOOR = np.uint32(64)
DQ_DEAD = np.uint32(1024)
DQ_HOT = np.uint32(2048)
DQ_WARM = np.uint32(4096)
DQ_REFERENCE_PIXEL = np.uint32(2147483648)
DQ_HIGH_VARIANCE = np.uint64(1 << 32)

FULL_WELL_ADU = {('NIRISS', None): 62070., ('NIRSPEC', 'NRS1'): 61537.,
    ('NIRSPEC', 'NRS2'): 60000., ('MIRI', None): 56600.,}

FRAME_TIME_S = {'NIRISS/SOSS-SUBSTRIP256': 5.494, 'NIRISS/SOSS-SUBSTRIP96': 2.214,
    'NIRSPEC/PRISM': 0.226, 'NIRSPEC/G395': 0.902,}


@dataclasses.dataclass
class ObsMeta:
    """Observing metadata retained throughout calibration."""
    mode: str
    detector: str
    subarray: str
    frame_time: float
    ngroups: int
    int_times: np.ndarray
    baseline_ints: np.ndarray
    segment_edges: np.ndarray
    filenames: tuple
    extra: dict = dataclasses.field(default_factory=dict)

    @property
    def segment_int_starts(self):
        """Return the exposure integration number where each segment begins.

        Returns
        -------
        starts : np.ndarray(int)
            One-based FITS integration number of the first sample in each segment.
        """
        stored = self.extra.get('segment_int_starts')
        if stored is not None:
            starts = np.asarray(stored, dtype=int)
            if starts.shape != np.asarray(self.segment_edges).shape:
                raise ValueError('segment_int_starts must have one entry per segment edge')
            if np.any(starts < 1):
                raise ValueError('FITS INTSTART values must be one-indexed')
            return starts
        edges = np.asarray(self.segment_edges, dtype=int)
        return np.concatenate(([1], edges[:-1] + 1))


@dataclasses.dataclass
class RampCube:
    """Observation counts and quality flags before ramp fitting.

    Data and group flags have integration, group, row, and column axes.
    """
    data: jax.Array
    groupdq: jax.Array
    pixeldq: jax.Array
    meta: ObsMeta = dataclasses.field(metadata=dict(static=True))


@dataclasses.dataclass
class RateCube:
    """Observation count rates, uncertainties, and quality flags after ramp fitting.

    Data, errors, and flags have integration, row, and column axes.
    """
    data: jax.Array
    err: jax.Array
    dq: jax.Array
    meta: ObsMeta = dataclasses.field(metadata=dict(static=True))


# Keep observing metadata out of compiled arithmetic.
jax.tree_util.register_dataclass(RampCube, data_fields=('data', 'groupdq', 'pixeldq'),
    meta_fields=('meta',))
jax.tree_util.register_dataclass(
    RateCube, data_fields=('data', 'err', 'dq'), meta_fields=('meta',))


def device_memory_bytes():
    """Estimate working memory currently available for a calculation.

    Returns
    -------
    nbytes : int
        Memory available for device working arrays, respecting EXOTEDRF_MAX_DEVICE_BYTES.
    """
    override = os.environ.get('EXOTEDRF_MAX_DEVICE_BYTES')
    if override:
        return int(override)
    dev = jax.devices()[0]
    stats = None
    try:
        stats = dev.memory_stats()
    except Exception:
        pass
    if stats and 'bytes_limit' in stats:
        return max(0, int(stats['bytes_limit'] - stats.get('bytes_in_use', 0)))
    if getattr(dev, 'platform', None) == 'cpu':
        # Limit CPU chunks to the job headroom and array allowance.
        return max(1, min(_available_job_memory_bytes(), host_allocation_budget_bytes()))
    # Use a conservative allowance when device telemetry is unavailable.
    return 1 << 30


_HOST_BYTE_UNITS = {'': 1, 'B': 1, 'KB': 1000, 'MB': 1000 ** 2, 'GB': 1000 ** 3, 'KIB': 1 << 10,
    'MIB': 1 << 20, 'GIB': 1 << 30,}

_HOST_BUDGET_CONTEXT = contextvars.ContextVar('exotedrf_host_budget', default=None)
_SCRATCH_DIR_CONTEXT = contextvars.ContextVar('exotedrf_scratch_dir', default=None)


def parse_host_bytes(value):
    """Interpret a user memory limit such as ``250MB`` or ``1.5 GiB``.

    Parameters
    ----------
    value : int, str
        Memory limit in bytes or a string with byte units.

    Returns
    -------
    nbytes : int
        Parsed memory limit in bytes.
    """
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if value < 0:
            raise ValueError('host byte count must be non-negative')
        return int(value)
    match = re.fullmatch('\\s*(\\d+(?:\\.\\d+)?)\\s*([KMG]?I?B)?\\s*', str(value), re.IGNORECASE)
    if match is None:
        raise ValueError(f'Invalid host byte count {value!r}; supported units are B, KB, '
            'MB, GB, KiB, MiB, and GiB')
    number, unit = match.groups()
    return int(float(number) * _HOST_BYTE_UNITS[(unit or '').upper()])


@contextlib.contextmanager
def host_memory_budget(max_bytes=None, scratch_dir=None):
    """Apply a reduction's host settings without changing process variables.

    Allocator arguments take precedence; enclosing settings are restored on exit.

    Parameters
    ----------
    max_bytes : None, int, str
        Maximum memory allowed for managed host arrays.
    scratch_dir : None, str
        Directory for temporary arrays that exceed the host memory limit.

    Yields
    ------
    None : None
        Scoped host allocation and scratch settings.
    """
    tokens = []
    try:
        if max_bytes is not None:
            tokens.append((_HOST_BUDGET_CONTEXT, _HOST_BUDGET_CONTEXT.set(
                parse_host_bytes(max_bytes))))
        if scratch_dir is not None:
            tokens.append((_SCRATCH_DIR_CONTEXT, _SCRATCH_DIR_CONTEXT.set(os.fspath(scratch_dir))))
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


def _process_resident_bytes():
    """Return current resident memory, rather than the historical RSS peak."""
    try:
        with open('/proc/self/statm', encoding='ascii') as stream:
            return int(stream.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError):
        return 0


def _cgroup_available_memory_bytes():
    """Read the effective remaining cgroup v1/v2 memory allowance on Linux."""
    try:
        with open('/proc/self/cgroup', encoding='ascii') as stream:
            memberships = [line.strip().split(':', 2) for line in stream]
    except OSError:
        return None
    remaining = []
    for fields in memberships:
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        if not controllers:
            root = '/sys/fs/cgroup'
            limit_file, used_file = 'memory.max', 'memory.current'
        elif 'memory' in controllers.split(','):
            root = '/sys/fs/cgroup/memory'
            limit_file = 'memory.limit_in_bytes'
            used_file = 'memory.usage_in_bytes'
        else:
            continue
        current = os.path.join(root, relative.lstrip('/'))
        while current == root or current.startswith(root + '/'):
            try:
                with open(os.path.join(current, limit_file), encoding='ascii') as stream:
                    limit = int(stream.read().strip())
                with open(os.path.join(current, used_file), encoding='ascii') as stream:
                    used = int(stream.read().strip())
                # Ignore the cgroup unlimited-memory sentinel.
                if 0 <= limit < 1 << 60:
                    remaining.append(max(0, limit - used))
            except (OSError, ValueError):
                pass
            if current == root:
                break
            current = os.path.dirname(current)
    return min(remaining) if remaining else None


def _system_available_memory_bytes():
    """Estimate free system RAM independently of job/container constraints."""
    try:
        with open('/proc/meminfo', encoding='ascii') as stream:
            fields = {key.rstrip(':'): int(value)
                for key, value, *_ in (line.split() for line in stream)}
        if 'MemAvailable' in fields:
            return fields['MemAvailable'] * 1024
    except (OSError, ValueError):
        pass
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
        if available > 0:
            return available
    except (ImportError, AttributeError, OSError, ValueError):
        pass
    try:
        page_size = os.sysconf('SC_PAGE_SIZE')
        try:
            pages = os.sysconf('SC_AVPHYS_PAGES')
            fraction = 1.0
        except (OSError, ValueError):
            # Reserve half of total memory when free memory is unknown.
            pages = os.sysconf('SC_PHYS_PAGES')
            fraction = 0.5
        available = int(page_size * pages * fraction)
        if available > 0:
            return available
    except (OSError, ValueError):
        pass
    return 1 << 30


def _available_job_memory_bytes():
    """Read actual free system/job memory independently of a v2 allowance."""
    available = [_system_available_memory_bytes()]
    cgroup_available = _cgroup_available_memory_bytes()
    if cgroup_available is not None:
        available.append(cgroup_available)
    try:
        slurm_mb = int(os.environ.get('SLURM_MEM_PER_NODE', '0'))
        if not slurm_mb:
            slurm_mb = (int(os.environ.get('SLURM_MEM_PER_CPU', '0')) *
                        int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
        if slurm_mb > 0:
            available.append(max(0, (slurm_mb << 20) - _process_resident_bytes()))
    except ValueError:
        pass
    return min(available)


def available_host_memory_bytes():
    """Return free host bytes within system/job limits or EXOTEDRF_MAX_HOST_BYTES."""
    override = os.environ.get('EXOTEDRF_MAX_HOST_BYTES')
    if override:
        return parse_host_bytes(override)
    return _available_job_memory_bytes()


def host_allocation_budget_bytes(explicit=None):
    """Choose how much system memory v2 may use for observation arrays.

    Live arrays share the allowance; the automatic limit reserves memory for temporary buffers.

    Parameters
    ----------
    explicit : None, int, str
        Maximum memory allowed for managed host arrays.

    Returns
    -------
    nbytes : int
        Shared host-array allocation ceiling in bytes.
    """
    if explicit is not None:
        return parse_host_bytes(explicit)
    scoped = _HOST_BUDGET_CONTEXT.get()
    if scoped is not None:
        return scoped
    fraction = float(os.environ.get('EXOTEDRF_HOST_MEMORY_FRACTION', '0.6'))
    if not 0.0 < fraction <= 1.0:
        raise ValueError('EXOTEDRF_HOST_MEMORY_FRACTION must be in (0, 1]')
    available = available_host_memory_bytes()
    if os.environ.get('EXOTEDRF_MAX_HOST_BYTES'):
        # Apply the configured fraction to the explicit memory allowance.
        return int(available * fraction)
    global _AUTOMATIC_HOST_BUDGET
    with _HOST_ALLOCATION_LOCK:
        used = _HOST_ALLOCATED_BYTES
        if (_AUTOMATIC_HOST_BUDGET is None or used == 0 or _AUTOMATIC_HOST_BUDGET[0] != fraction):
            _AUTOMATIC_HOST_BUDGET = (fraction, int(available * fraction))
        ceiling = _AUTOMATIC_HOST_BUDGET[1]
        # Limit additional arrays by current free memory without subtracting live arrays twice.
        return min(ceiling, used + int(available * fraction))


_HOST_ALLOCATION_LOCK = threading.Lock()
_HOST_ALLOCATED_BYTES = 0
_AUTOMATIC_HOST_BUDGET = None


def _release_host_allocation(nbytes):
    """Release the memory allowance held by an expired array."""
    global _HOST_ALLOCATED_BYTES, _AUTOMATIC_HOST_BUDGET
    with _HOST_ALLOCATION_LOCK:
        _HOST_ALLOCATED_BYTES -= nbytes
        if _HOST_ALLOCATED_BYTES == 0:
            _AUTOMATIC_HOST_BUDGET = None


def host_allocation_available_bytes(max_bytes=None):
    """Return the host allocation budget minus bytes reserved by managed arrays."""
    budget = host_allocation_budget_bytes(max_bytes)
    with _HOST_ALLOCATION_LOCK:
        return max(0, budget - _HOST_ALLOCATED_BYTES)


def empty_host_array(shape, dtype, *, name='exotedrf', max_bytes=None, scratch_dir=None):
    """Allocate within a shared live-array RAM budget, spilling excess.

    Array views retain their allocation budget until the last owner is released.

    Parameters
    ----------
    shape : tuple[int]
        Array dimensions.
    dtype : dtype
        Data type of the array.
    name : str
        Name used for temporary array files.
    max_bytes : None, int, str
        Maximum memory allowed for managed host arrays.
    scratch_dir : None, str
        Directory for temporary arrays that exceed the host memory limit.

    Returns
    -------
    result : np.ndarray
        Allocated host array; a temporary memory mapping when the budget is exceeded.
    """
    global _HOST_ALLOCATED_BYTES
    dtype = np.dtype(dtype)
    required = int(np.prod(shape)) * dtype.itemsize
    budget = host_allocation_budget_bytes(max_bytes)
    with _HOST_ALLOCATION_LOCK:
        use_ram = required <= max(0, budget - _HOST_ALLOCATED_BYTES)
        if use_ram:
            _HOST_ALLOCATED_BYTES += required
    if use_ram:
        try:
            result = np.empty(shape, dtype=dtype)
            weakref.finalize(result, _release_host_allocation, required)
            return result
        except MemoryError:
            # Release the reservation before falling back to scratch storage.
            _release_host_allocation(required)
        except BaseException:
            _release_host_allocation(required)
            raise
    import tempfile
    scratch_dir = (scratch_dir or _SCRATCH_DIR_CONTEXT.get() or
        os.environ.get('EXOTEDRF_SCRATCH_DIR') or tempfile.gettempdir())
    os.makedirs(scratch_dir, exist_ok=True)
    backing = tempfile.TemporaryFile(prefix=f'{name}-', dir=scratch_dir)
    try:
        result = np.memmap(backing, mode='w+', dtype=dtype, shape=tuple(shape))
    finally:
        backing.close()
    return result


def copy_host_array(value, *, name='exotedrf-copy', max_bytes=None,
                    scratch_dir=None, chunk_bytes=32 << 20):
    """Copy a NumPy, JAX, or lazy array without materializing a full cube.

    Parameters
    ----------
    value : array-like
        NumPy, JAX, or lazy input array to copy.
    name : str
        Name used for temporary array files.
    max_bytes : None, int, str
        Maximum memory allowed for managed host arrays.
    scratch_dir : None, str
        Directory for temporary arrays that exceed the host memory limit.
    chunk_bytes : int
        Maximum bytes to convert or transfer at a time.

    Returns
    -------
    result : np.ndarray
        Allocated host array; a temporary memory mapping when the budget is exceeded.
    """
    if chunk_bytes < 1:
        raise ValueError('chunk_bytes must be positive')
    shape = tuple(value.shape)
    dtype = np.dtype(value.dtype)
    result = empty_host_array(shape, dtype, name=name, max_bytes=max_bytes, scratch_dir=scratch_dir)
    if any(size == 0 for size in shape):
        return result
    if not shape:
        result[...] = _host_array(value)
        return result
    tile_shape = list(shape)
    max_elements = max(1, int(chunk_bytes) // dtype.itemsize)
    for axis in range(len(shape)):
        if int(np.prod(tile_shape)) <= max_elements:
            break
        following = int(np.prod(tile_shape[axis + 1:]))
        tile_shape[axis] = min(tile_shape[axis], max(1, max_elements // following))
    starts = (range(0, size, tile) for size, tile in zip(shape, tile_shape))
    for offset in itertools.product(*starts):
        tile = tuple(slice(lo, min(lo + width, size))
                     for lo, width, size in zip(offset, tile_shape, shape))
        result[tile] = _host_array(value[tile])
    return result


def auto_chunk(n_items, bytes_per_item, n_buffers=None, headroom=0.75):
    """Choose how many integrations or columns can be processed together.

    Parameters
    ----------
    n_items : int
        Number of items along the mapped axis.
    bytes_per_item : int
        Input bytes per mapped item.
    n_buffers : None, int
        Number of working buffers to allow per item.
    headroom : float
        Fraction of available device memory to use.

    Returns
    -------
    size : int
        Number of items to process per chunk.
    """
    if n_buffers is None:
        n_buffers = int(os.environ.get('EXOTEDRF_DEVICE_BUFFER_FACTOR', '24'))
    if n_buffers < 1:
        raise ValueError('device buffer factor must be positive')
    budget = device_memory_bytes() * headroom
    per = bytes_per_item * n_buffers
    return max(1, min(n_items, int(budget // max(per, 1))))


def _host_array(value):
    """Copy a completed device calculation to host memory."""
    return np.asarray(jax.device_get(value))


def is_device_array(value):
    """Return whether the supplied value is a JAX array."""
    return isinstance(value, jax.Array)


def to_host(tree):
    """Copy JAX array leaves of a nested tree to NumPy; other leaves pass through."""
    return jax.tree_util.tree_map(
        lambda leaf: _host_array(leaf) if is_device_array(leaf) else leaf, tree)


def _device_stream_chunks(fn, arrs, n_items, chunk_size, *, axis, halo=0):
    """Process and join chunks without moving arrays off the device."""
    chunk_size = min(int(chunk_size), int(n_items))
    pieces = None
    treedef = None
    for lo in range(0, n_items, chunk_size):
        hi = min(lo + chunk_size, n_items)
        read_lo = max(0, lo - halo)
        read_hi = min(n_items, hi + halo)
        chunks = []
        for array in arrs:
            array_sl = [slice(None)] * array.ndim
            array_sl[axis] = slice(read_lo, read_hi)
            chunks.append(array[tuple(array_sl)])
        result = fn(*chunks)
        leaves, result_tree = jax.tree_util.tree_flatten(result)
        del result
        if pieces is None:
            treedef = result_tree
            pieces = [[] for _ in leaves]
            for leaf in leaves:
                if leaf.ndim == 0:
                    raise ValueError('chunked function outputs must retain the mapped axis')
        elif result_tree != treedef or len(leaves) != len(pieces):
            raise ValueError('chunked function returned inconsistent nested result shapes')
        for store, leaf in zip(pieces, leaves):
            leaf_axis = axis % leaf.ndim
            trim_lo = lo - read_lo
            trim_hi = trim_lo + (hi - lo)
            if trim_lo != 0 or trim_hi != leaf.shape[leaf_axis]:
                source = [slice(None)] * leaf.ndim
                source[leaf_axis] = slice(trim_lo, trim_hi)
                leaf = leaf[tuple(source)]
            store.append(leaf)
    joined = [_device_join(store, axis) for store in pieces]
    return jax.tree_util.tree_unflatten(treedef, joined)


@functools.partial(jax.jit, static_argnames=('axis',), donate_argnums=0)
def _device_place(buffer, piece, start, axis):
    """Place a chunk into a donated device buffer along a static axis."""
    return jax.lax.dynamic_update_slice_in_dim(buffer, piece, start, axis)


def _device_join(store, axis):
    """Join device pieces along ``axis`` without a second full-size copy."""
    if len(store) == 1:
        piece = store[0]
        store.clear()
        return piece
    leaf_axis = axis % store[0].ndim
    total = sum(int(piece.shape[leaf_axis]) for piece in store)
    shape = list(store[0].shape)
    shape[leaf_axis] = total
    buffer = jnp.empty(shape, dtype=store[0].dtype)
    offset = 0
    while store:
        piece = store.pop(0)
        buffer = _device_place(buffer, piece, offset, leaf_axis)
        offset += int(piece.shape[leaf_axis])
        del piece
    return buffer


def _stream_chunks(fn, arrs, n_items, chunk_size, *, axis, halo=0):
    """Process consecutive pieces of an observation and reassemble them."""
    if n_items < 1:
        raise ValueError('chunked axes must contain at least one item')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if arrs and all(is_device_array(a) for a in arrs):
        return _device_stream_chunks(fn, arrs, n_items, chunk_size, axis=axis, halo=halo)
    chunk_size = min(int(chunk_size), int(n_items))

    def upload(lo):
        """Upload and launch one integration or column chunk."""
        hi = min(lo + chunk_size, n_items)
        read_lo = max(0, lo - halo)
        read_hi = min(n_items, hi + halo)
        chunks = []
        for array in arrs:
            array_sl = [slice(None)] * array.ndim
            array_sl[axis] = slice(read_lo, read_hi)
            chunks.append(jnp.asarray(array[tuple(array_sl)]))
        # Overlap the next chunk calculation with the host copy.
        result = fn(*chunks)
        del chunks
        return lo, hi, read_lo, result

    buffers = None
    treedef = None
    pending = None
    for lo in range(0, n_items, chunk_size):
        launched = upload(lo)
        if pending is not None:
            buffers, treedef = _land_chunk(pending, buffers, treedef, n_items, axis)
        pending = launched
    if pending is not None:
        buffers, treedef = _land_chunk(pending, buffers, treedef, n_items, axis)

    return jax.tree_util.tree_unflatten(treedef, buffers)


def _land_chunk(pending, buffers, treedef, n_items, axis):
    """Copy one finished device piece into the host output buffers."""
    lo, hi, read_lo, result = pending
    leaves, result_tree = jax.tree_util.tree_flatten(jax.tree.map(_host_array, result))
    del result, pending

    if buffers is None:
        treedef = result_tree
        buffers = []
        for leaf in leaves:
            if leaf.ndim == 0:
                raise ValueError('chunked function outputs must retain the mapped axis')
            leaf_axis = axis % leaf.ndim
            shape = list(leaf.shape)
            shape[leaf_axis] = n_items
            buffers.append(empty_host_array(shape, leaf.dtype, name='exotedrf-stream'))
    elif result_tree != treedef or len(leaves) != len(buffers):
        raise ValueError('chunked function returned inconsistent nested result shapes')
    for buffer, leaf in zip(buffers, leaves):
        leaf_axis = axis % leaf.ndim
        trim_lo = lo - read_lo
        trim_hi = trim_lo + (hi - lo)
        source = [slice(None)] * leaf.ndim
        source[leaf_axis] = slice(trim_lo, trim_hi)
        target = [slice(None)] * buffer.ndim
        target[leaf_axis] = slice(lo, hi)
        buffer[tuple(target)] = leaf[tuple(source)]
    return buffers, treedef


def _mapped_arrays(arrays, n_items, axis, kind):
    """Normalize mapped inputs and validate the requested integration or column axis."""
    arrs = tuple(arrays) if isinstance(arrays, (tuple, list)) else (arrays,)
    if n_items < 1 or not arrs:
        raise ValueError(f'at least one {kind} and array are required')
    if any(a.ndim < 1 or int(a.shape[axis]) != int(n_items) for a in arrs):
        edge = 'leading' if axis == 0 else 'trailing'
        raise ValueError(f'all arrays must share the requested {edge} axis')
    return arrs


def _mapped_chunk_size(arrs, n_items, chunk_size, bytes_per_item, variable):
    """Choose the requested, environment, or memory-budget chunk size."""
    if chunk_size is None:
        env = os.environ.get(variable)
        if env:
            return int(env)
        if bytes_per_item is None:
            bytes_per_item = sum(a.nbytes // n_items for a in arrs)
        return auto_chunk(n_items, bytes_per_item)
    return chunk_size


def map_over_ints(fn, arrays, n_ints, chunk_size=None, bytes_per_int=None):
    """Apply an integration-independent calibration to a long visit.

    Parameters
    ----------
    fn : callable
        Function to apply to each array chunk.
    arrays : array-like, tuple
        Input arrays sharing the mapped axis.
    n_ints : int
        Number of integrations in the observation.
    chunk_size, bytes_per_int : None, int
        Integration chunk length and input bytes per integration; calculated if omitted.

    Returns
    -------
    result : object
        Joined function results with the same nested structure and array residency.
    """
    arrs = _mapped_arrays(arrays, n_ints, 0, 'integration')
    chunk_size = _mapped_chunk_size(arrs, n_ints, chunk_size, bytes_per_int, 'EXOTEDRF_CHUNK_INTS')
    return _stream_chunks(fn, arrs, n_ints, chunk_size, axis=0)


def map_over_cols(fn, arrays, n_cols, chunk_size=None, bytes_per_col=None):
    """Process detector columns in pieces while retaining the full time axis.

    Parameters
    ----------
    fn : callable
        Function to apply to each array chunk.
    arrays : array-like, tuple
        Input arrays sharing the mapped axis.
    n_cols : int
        Number of detector columns.
    chunk_size, bytes_per_col : None, int
        Column chunk length and input bytes per column; calculated if omitted.

    Returns
    -------
    result : object
        Joined function results with the same nested structure and array residency.
    """
    return map_over_cols_with_halo(fn, arrays, n_cols, 0, chunk_size, bytes_per_col)


def map_over_cols_with_halo(fn, arrays, n_cols, halo, chunk_size=None, bytes_per_col=None):
    """Process columns while including the neighboring pixels they require.

    Parameters
    ----------
    fn : callable
        Function to apply to each array chunk.
    arrays : array-like, tuple
        Input arrays sharing the mapped axis.
    n_cols : int
        Number of detector columns.
    halo : int
        Number of neighboring columns to include around each chunk.
    chunk_size, bytes_per_col : None, int
        Column chunk length and input bytes per column; calculated if omitted.

    Returns
    -------
    result : object
        Joined function results with the same nested structure and array residency.
    """
    if halo < 0:
        raise ValueError('halo must be non-negative')
    arrs = _mapped_arrays(arrays, n_cols, -1, 'column')
    chunk_size = _mapped_chunk_size(arrs, n_cols, chunk_size, bytes_per_col, 'EXOTEDRF_CHUNK_COLS')
    return _stream_chunks(fn, arrs, n_cols, chunk_size, axis=-1, halo=int(halo))


def segment_slices(meta_or_edges, n_ints=None):
    """Return the integration ranges belonging to each original FITS file.

    Parameters
    ----------
    meta_or_edges : ObsMeta, array-like(int)
        Observing metadata or cumulative segment integration counts.
    n_ints : None, int
        Total integrations to check against the final segment edge, if supplied.

    Returns
    -------
    segments : tuple[slice]
        Integration slices for the original input segments.
    """
    edges = getattr(meta_or_edges, 'segment_edges', meta_or_edges)
    edges = np.asarray(edges, dtype=int)
    if edges.ndim != 1 or edges.size == 0:
        raise ValueError('segment_edges must be a non-empty 1D array')
    if np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError('segment_edges must be strictly increasing')
    if n_ints is not None and int(edges[-1]) != int(n_ints):
        raise ValueError(f'last segment edge {edges[-1]} does not match nints={n_ints}')
    starts = np.concatenate(([0], edges[:-1]))
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(starts, edges))


def map_over_segments(fn, arrays, meta_or_edges):
    """Run a time-dependent correction separately on each input segment.

    Parameters
    ----------
    fn : callable
        Function to apply to each array chunk.
    arrays : array-like, tuple
        Input arrays sharing the mapped axis.
    meta_or_edges : ObsMeta, array-like(int)
        Observing metadata or cumulative segment integration counts.

    Returns
    -------
    result : object
        Joined function results with the same nested structure and array residency.
    """
    single = not isinstance(arrays, (tuple, list))
    arrs = (arrays,) if single else tuple(arrays)
    if not arrs:
        raise ValueError('at least one array is required')
    n_ints = int(arrs[0].shape[0])
    if any(int(a.shape[0]) != n_ints for a in arrs):
        raise ValueError('all segmented arrays must share their leading axis')
    resident = all(is_device_array(a) for a in arrs)
    buffers = treedef = None
    for seg in segment_slices(meta_or_edges, n_ints=n_ints):
        result = fn(*(a[seg] for a in arrs))
        if not resident:
            result = jax.tree.map(_host_array, result)
        leaves, result_tree = jax.tree_util.tree_flatten(result)
        del result
        if any(leaf.ndim == 0 or leaf.shape[0] != seg.stop - seg.start for leaf in leaves):
            raise ValueError('segmented function outputs must retain the integration axis')
        if buffers is None:
            treedef = result_tree
            buffers = ([[] for leaf in leaves] if resident else
                       [empty_host_array((n_ints, *leaf.shape[1:]), leaf.dtype,
                                         name='exotedrf-segments') for leaf in leaves])
        elif result_tree != treedef or (not resident and any(
                leaf.shape[1:] != buffer.shape[1:] or leaf.dtype != buffer.dtype
                for leaf, buffer in zip(leaves, buffers))):
            raise ValueError('segmented function returned inconsistent nested results')
        for buffer, leaf in zip(buffers, leaves):
            if resident:
                buffer.append(leaf)
            else:
                buffer[seg] = leaf
        del leaves
    joined = [_device_join(buffer, 0) for buffer in buffers] if resident else buffers
    return jax.tree_util.tree_unflatten(treedef, joined)
