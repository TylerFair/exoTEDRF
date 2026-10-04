"""Check core."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import gc
import io
import os
from types import SimpleNamespace

from exotedrf.v2 import core


@pytest.mark.parametrize(
    ('cube_type', 'field_names'),
    [(core.RampCube, ('data', 'groupdq', 'pixeldq')),
     (core.RateCube, ('data', 'err', 'dq'))])
def test_cube_pytrees_keep_only_science_arrays_as_dynamic_leaves(
        cube_type, field_names):
    """Check cube pytrees keep only science arrays as dynamic leaves."""
    meta = core.ObsMeta(
        mode='NIRISS/SOSS', detector='NIS', subarray='SUBSTRIP96',
        frame_time=2.214, ngroups=2, int_times=np.arange(2.),
        baseline_ints=np.asarray([2]), segment_edges=np.asarray([2]),
        filenames=('seg001.fits',))
    if cube_type is core.RampCube:
        arrays = {
            'data': np.zeros((2, 2, 2, 2), np.float32),
            'groupdq': np.zeros((2, 2, 2, 2), np.uint8),
            'pixeldq': np.zeros((2, 2), np.uint32),
        }
    else:
        arrays = {
            'data': np.zeros((2, 2, 2), np.float32),
            'err': np.zeros((2, 2, 2), np.float32),
            'dq': np.zeros((2, 2, 2), np.uint32),
        }
    cube = cube_type(**arrays, meta=meta)

    leaves, structure = jax.tree_util.tree_flatten(cube)
    assert len(leaves) == len(field_names)
    rebuilt = jax.tree_util.tree_unflatten(structure, leaves)
    assert rebuilt.meta is meta
    for name in field_names:
        np.testing.assert_array_equal(getattr(rebuilt, name), arrays[name])


def test_map_over_ints_transfers_only_chunks_and_returns_host_tree():
    """Check map over ints transfers only chunks and returns host tree."""
    data = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    seen = []

    def kernel(chunk):
        seen.append((type(chunk), chunk.shape))
        return {'science': chunk + 1, 'nested': (chunk * 2, None)}

    out = core.map_over_ints(kernel, data, n_ints=5, chunk_size=2)
    assert [shape[0] for _, shape in seen] == [2, 2, 1]
    assert all(issubclass(kind, jax.Array) for kind, _ in seen)
    assert isinstance(out['science'], np.ndarray)
    assert isinstance(out['nested'][0], np.ndarray)
    np.testing.assert_array_equal(out['science'], data + 1)
    np.testing.assert_array_equal(out['nested'][0], data * 2)


def test_map_over_ints_one_chunk_still_returns_host_and_preserves_x64():
    """Check map over ints one chunk still returns host and preserves x64."""
    old = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        data = np.arange(12, dtype=np.float64).reshape(4, 3)
        out = core.map_over_ints(lambda chunk: chunk / 3., data, 4,
                                 chunk_size=99)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, data / 3.)
    finally:
        jax.config.update('jax_enable_x64', old)


def test_stream_outputs_spill_and_keep_halo_values(monkeypatch, tmp_path):
    """Check stream outputs spill and keep halo values."""
    monkeypatch.setenv('EXOTEDRF_MAX_HOST_BYTES', '1B')
    monkeypatch.setenv('EXOTEDRF_SCRATCH_DIR', str(tmp_path))
    data = np.arange(21, dtype=np.float32).reshape(3, 7)
    out = core.map_over_cols_with_halo(
        lambda x: x * 2, data, n_cols=7, halo=1, chunk_size=3)
    assert isinstance(out, np.memmap)
    np.testing.assert_array_equal(out, data * 2)


def test_map_over_cols_with_halo_trims_overlap_into_host_buffer():
    """Check map over cols with halo trims overlap into host buffer."""
    data = np.arange(2 * 7, dtype=np.float32).reshape(2, 7)
    widths = []

    def kernel(chunk):
        widths.append(chunk.shape[-1])
        return jnp.sin(chunk), chunk.astype(jnp.int32)

    sine, integer = core.map_over_cols_with_halo(
        kernel, data, n_cols=7, halo=1, chunk_size=3)
    assert widths == [4, 5, 2]
    assert isinstance(sine, np.ndarray)
    assert isinstance(integer, np.ndarray)
    np.testing.assert_allclose(sine, np.sin(data), rtol=1e-6)
    np.testing.assert_array_equal(integer, data.astype(np.int32))


def test_chunk_mappers_validate_shared_axis():
    """Check chunk mappers validate shared axis."""
    with np.testing.assert_raises_regex(ValueError, 'leading axis'):
        core.map_over_ints(lambda a, b: a, (np.zeros((3, 2)),
                                            np.zeros((2, 2))), 3)
    with np.testing.assert_raises_regex(ValueError, 'trailing axis'):
        core.map_over_cols(lambda a, b: a, (np.zeros((2, 3)),
                                            np.zeros((2, 4))), 3)


def test_auto_chunk_uses_conservative_configurable_buffer_factor(monkeypatch):
    """Check auto chunk uses conservative configurable buffer factor."""
    monkeypatch.setattr(core, 'device_memory_bytes', lambda: 2400)
    monkeypatch.delenv('EXOTEDRF_DEVICE_BUFFER_FACTOR', raising=False)
    assert core.auto_chunk(100, bytes_per_item=10, headroom=1.) == 10
    monkeypatch.setenv('EXOTEDRF_DEVICE_BUFFER_FACTOR', '6')
    assert core.auto_chunk(100, bytes_per_item=10, headroom=1.) == 40
    monkeypatch.setenv('EXOTEDRF_DEVICE_BUFFER_FACTOR', '0')
    with pytest.raises(ValueError, match='buffer factor'):
        core.auto_chunk(100, bytes_per_item=10, headroom=1.)


def test_cpu_chunk_memory_respects_job_headroom_and_configured_budget(
        monkeypatch):
    """Check CPU chunk memory respects job headroom and configured budget."""
    monkeypatch.delenv('EXOTEDRF_MAX_DEVICE_BYTES', raising=False)
    device = SimpleNamespace(platform='cpu', memory_stats=lambda: None)
    monkeypatch.setattr(core.jax, 'devices', lambda: [device])
    monkeypatch.setattr(core, '_available_job_memory_bytes', lambda: 20 << 30)
    with core.host_memory_budget('30GiB'):
        assert core.device_memory_bytes() == 20 << 30
    with core.host_memory_budget('2GiB'):
        assert core.device_memory_bytes() == 2 << 30
        assert core.auto_chunk(1000, 1 << 20, headroom=1.) == 85


@pytest.mark.parametrize('stats', [None, {}, {'bytes_in_use': 123}])
def test_unknown_gpu_memory_uses_small_conservative_allowance(monkeypatch,
                                                            stats):
    """Check unknown GPU memory uses small conservative allowance."""
    monkeypatch.delenv('EXOTEDRF_MAX_DEVICE_BYTES', raising=False)
    device = SimpleNamespace(platform='gpu', memory_stats=lambda: stats)
    monkeypatch.setattr(core.jax, 'devices', lambda: [device])
    assert core.device_memory_bytes() == 1 << 30
    monkeypatch.setenv('EXOTEDRF_MAX_DEVICE_BYTES', str(7 << 30))
    assert core.device_memory_bytes() == 7 << 30


def test_device_telemetry_subtracts_live_allocations(monkeypatch):
    """Check device telemetry subtracts live allocations."""
    monkeypatch.delenv('EXOTEDRF_MAX_DEVICE_BYTES', raising=False)
    device = SimpleNamespace(platform='gpu', memory_stats=lambda: {
        'bytes_limit': 16 << 30, 'bytes_in_use': 5 << 30})
    monkeypatch.setattr(core.jax, 'devices', lambda: [device])
    assert core.device_memory_bytes() == 11 << 30


def test_host_memory_budget_accepts_unit_strings_and_explicit_override(
        monkeypatch):
    """Check host memory budget accepts unit strings and explicit override."""
    assert core.parse_host_bytes('1.5 GiB') == int(1.5 * (1 << 30))
    assert core.parse_host_bytes('250MB') == 250_000_000
    with pytest.raises(ValueError, match='units'):
        core.parse_host_bytes('twelve')

    monkeypatch.setenv('EXOTEDRF_MAX_HOST_BYTES', '8MiB')
    monkeypatch.setenv('EXOTEDRF_HOST_MEMORY_FRACTION', '0.25')
    assert core.available_host_memory_bytes() == 8 << 20
    assert core.host_allocation_budget_bytes() == 2 << 20
    assert core.host_allocation_budget_bytes('3MiB') == 3 << 20


def test_host_allocator_caps_combined_live_arrays_and_retains_view_budget(
        tmp_path):
    """Check host allocator caps combined live arrays and retains view budget."""
    before = core._HOST_ALLOCATED_BYTES
    allowance = before + 48
    first = core.empty_host_array((8,), np.float32, max_bytes=allowance,
                                   scratch_dir=tmp_path)
    assert not isinstance(first, np.memmap)
    assert core.host_allocation_available_bytes(allowance) == 16
    second = core.empty_host_array((8,), np.float32, max_bytes=allowance,
                                    scratch_dir=tmp_path)
    assert isinstance(second, np.memmap)
    assert core.host_allocation_available_bytes(allowance) == 16
    view = first[1:]
    del first
    gc.collect()
    assert core.host_allocation_available_bytes(allowance) == 16
    del view
    gc.collect()
    assert core.host_allocation_available_bytes(allowance) == 48


def test_copy_host_array_tiles_lazy_noncontiguous_values_without_materializing(
        tmp_path):
    """Check copy host array tiles lazy noncontiguous values without materializing."""
    original = np.arange(2 * 4 * 7, dtype='>f8').reshape(2, 4, 7)[:, :, ::-1]
    sizes = []

    class SliceOnly:
        shape, dtype = original.shape, original.dtype

        def __array__(self, *args, **kwargs):
            raise AssertionError('a complete lazy cube must not materialize')

        def __getitem__(self, index):
            result = original[index]
            sizes.append(result.nbytes)
            return result

    result = core.copy_host_array(SliceOnly(), max_bytes=0,
                                  scratch_dir=tmp_path, chunk_bytes=24)
    assert isinstance(result, np.memmap)
    assert result.dtype == original.dtype
    assert max(sizes) <= 24
    np.testing.assert_array_equal(result, original)
    result[:] = -1
    assert not np.all(original == -1)


def test_host_allocator_falls_back_if_ram_allocation_fails(monkeypatch, tmp_path):
    """Check host allocator falls back if ram allocation fails."""
    before = core._HOST_ALLOCATED_BYTES

    def unavailable(*args, **kwargs):
        raise MemoryError('another process consumed the free memory')

    monkeypatch.setattr(core.np, 'empty', unavailable)
    result = core.empty_host_array((4,), np.float32,
                                   max_bytes=before + 100,
                                   scratch_dir=tmp_path)
    assert isinstance(result, np.memmap)
    assert core._HOST_ALLOCATED_BYTES == before


def test_automatic_host_budget_does_not_subtract_live_resident_arrays_twice(
        monkeypatch, tmp_path):
    """Check automatic host budget does not subtract live resident arrays twice."""
    monkeypatch.delenv('EXOTEDRF_MAX_HOST_BYTES', raising=False)
    monkeypatch.setenv('EXOTEDRF_HOST_MEMORY_FRACTION', '0.5')
    monkeypatch.setattr(core, '_AUTOMATIC_HOST_BUDGET', None)
    before = core._HOST_ALLOCATED_BYTES
    # Account for any live test fixtures without changing the global tracker.
    available = [2 * (before + 100)]
    monkeypatch.setattr(core, 'available_host_memory_bytes',
                        lambda: available[0])
    assert core.host_allocation_budget_bytes() == before + 100
    first = core.empty_host_array((64,), np.uint8, scratch_dir=tmp_path)
    assert not isinstance(first, np.memmap)
    # OS memory reports fall after touching the new array.
    first[:] = 1
    available[0] -= 64
    assert core.host_allocation_budget_bytes() == before + 100
    assert core.host_allocation_available_bytes() == 36
    second = core.empty_host_array((64,), np.uint8, scratch_dir=tmp_path)
    assert isinstance(second, np.memmap)
    available[0] = 20
    assert core.host_allocation_available_bytes() == 10
    third = core.empty_host_array((16,), np.uint8, scratch_dir=tmp_path)
    assert isinstance(third, np.memmap)
    del first


def test_copy_host_array_handles_device_values_scalar_and_empty():
    """Check copy host array handles device values scalar and empty."""
    values = jnp.arange(15, dtype=jnp.float32).reshape(3, 5)
    copied = core.copy_host_array(values, chunk_bytes=16)
    np.testing.assert_array_equal(copied, np.asarray(values))
    assert copied.dtype == np.float32
    scalar = core.copy_host_array(np.asarray(7., np.float64), chunk_bytes=8)
    assert scalar.shape == () and scalar.dtype == np.float64
    assert scalar.item() == 7.
    assert core.copy_host_array(np.empty((2, 0, 5))).shape == (2, 0, 5)
    with pytest.raises(ValueError, match='chunk_bytes'):
        core.copy_host_array(values, chunk_bytes=0)


def test_host_memory_detects_job_limits_on_large_shared_node(monkeypatch):
    """Check host memory detects job limits on large shared node."""
    monkeypatch.delenv('EXOTEDRF_MAX_HOST_BYTES', raising=False)
    monkeypatch.setattr(core, '_system_available_memory_bytes',
                        lambda: 256 << 30)
    monkeypatch.setattr(core, '_cgroup_available_memory_bytes',
                        lambda: 40 << 30)
    monkeypatch.setattr(core, '_process_resident_bytes', lambda: 4 << 30)
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '51200')
    assert core.available_host_memory_bytes() == 40 << 30
    monkeypatch.setattr(core, '_cgroup_available_memory_bytes', lambda: None)
    assert core.available_host_memory_bytes() == 46 << 30
    monkeypatch.setenv('EXOTEDRF_MAX_HOST_BYTES', '30GiB')
    assert core.available_host_memory_bytes() == 30 << 30


@pytest.mark.parametrize('version', [1, 2])
def test_cgroup_memory_includes_finite_ancestor_limit(monkeypatch, version):
    """Check cgroup memory includes finite ancestor limit."""
    if version == 2:
        root = '/sys/fs/cgroup'
        membership = '0::/job/step\n'
        limit, used = 'memory.max', 'memory.current'
    else:
        root = '/sys/fs/cgroup/memory'
        membership = '3:cpu,cpuacct:/other\n4:memory:/job/step\n'
        limit, used = 'memory.limit_in_bytes', 'memory.usage_in_bytes'
    files = {'/proc/self/cgroup': membership,
             f'{root}/job/step/{limit}': 'max',
             f'{root}/job/{limit}': str(50 << 30),
             f'{root}/job/{used}': str(15 << 30),
             f'{root}/{limit}': str(200 << 30),
             f'{root}/{used}': str(190 << 30)}

    def fake_open(path, *args, **kwargs):
        if str(path) not in files:
            raise FileNotFoundError(path)
        return io.StringIO(files[str(path)])

    monkeypatch.setattr('builtins.open', fake_open)
    assert core._cgroup_available_memory_bytes() == 10 << 30


def test_map_over_segments_streams_nested_outputs_with_boundary_resets(
        monkeypatch, tmp_path):
    """Check map over segments streams nested outputs with boundary resets."""
    monkeypatch.setenv('EXOTEDRF_MAX_HOST_BYTES', '1B')
    monkeypatch.setenv('EXOTEDRF_SCRATCH_DIR', str(tmp_path))
    values = np.arange(7 * 3, dtype=np.float64).reshape(7, 3)
    seen = []

    def operation(part):
        seen.append(part.shape[0])
        return {'centered': part - part.mean(axis=0),
                'nested': (part.astype(np.uint32), None)}

    result = core.map_over_segments(operation, values, [2, 6, 7])
    assert seen == [2, 4, 1]
    assert isinstance(result['centered'], np.memmap)
    expected = values.copy()
    for segment in (slice(0, 2), slice(2, 6), slice(6, 7)):
        expected[segment] -= expected[segment].mean(axis=0)
    np.testing.assert_array_equal(result['centered'], expected)
    np.testing.assert_array_equal(result['nested'][0], values.astype(np.uint32))
    assert result['nested'][1] is None


def test_map_over_segments_rejects_outputs_that_drop_integrations():
    """Check map over segments rejects outputs that drop integrations."""
    with pytest.raises(ValueError, match='integration axis'):
        core.map_over_segments(lambda part: part.mean(axis=0),
                               np.zeros((5, 3)), [2, 5])


def test_host_budget_scope_restores_nested_settings_without_env_changes(
        monkeypatch, tmp_path):
    """Check host budget scope restores nested settings without env changes."""
    monkeypatch.setenv('EXOTEDRF_MAX_HOST_BYTES', '8MiB')
    monkeypatch.setenv('EXOTEDRF_HOST_MEMORY_FRACTION', '0.5')
    before = dict(os.environ)
    scratch = tmp_path / 'scoped'
    with core.host_memory_budget('1MiB', scratch):
        assert core.host_allocation_budget_bytes() == 1 << 20
        assert core.host_allocation_budget_bytes('2MiB') == 2 << 20
        with core.host_memory_budget():
            assert core.host_allocation_budget_bytes() == 1 << 20
        with pytest.raises(RuntimeError):
            with core.host_memory_budget(0):
                result = core.empty_host_array((10,), np.float32)
                assert isinstance(result, np.memmap)
                assert scratch.is_dir()
                raise RuntimeError('failure still restores scoped settings')
        assert core.host_allocation_budget_bytes() == 1 << 20
    assert core.host_allocation_budget_bytes() == 4 << 20
    assert core._SCRATCH_DIR_CONTEXT.get() is None
    assert dict(os.environ) == before
