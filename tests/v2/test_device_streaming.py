"""Device-resident chunk streaming must equal the host-streamed path exactly."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from exotedrf.v2 import core


def _fn_ints(d, g):
    """Return fn ints."""
    return {'a': d * 2 + g.astype(d.dtype), 'b': (g + 1).astype(jnp.uint8)}


def _fn_cols(d):
    # Column-local smoothing needing a halo of 1.
    """Return fn cols."""
    return 0.25 * d[..., :-2] + 0.5 * d[..., 1:-1] + 0.25 * d[..., 2:]


@pytest.mark.parametrize('chunk', [1, 3, 7, 100])
def test_map_over_ints_device_matches_host(chunk):
    """Check map over ints device matches host."""
    rng = np.random.default_rng(0)
    d = rng.normal(size=(7, 2, 4, 6)).astype(np.float32)
    g = rng.integers(0, 4, size=d.shape).astype(np.uint8)
    host = core.map_over_ints(_fn_ints, (d, g), 7, chunk_size=chunk)
    dev = core.map_over_ints(_fn_ints, (jnp.asarray(d), jnp.asarray(g)), 7,
                             chunk_size=chunk)
    assert isinstance(host['a'], np.ndarray)
    assert core.is_device_array(dev['a']) and core.is_device_array(dev['b'])
    np.testing.assert_array_equal(np.asarray(dev['a']), host['a'])
    np.testing.assert_array_equal(np.asarray(dev['b']), host['b'])
    back = core.to_host(dev)
    assert isinstance(back['a'], np.ndarray) and isinstance(back['b'], np.ndarray)


@pytest.mark.parametrize('chunk', [2, 5, 64])
def test_map_over_cols_with_halo_device_matches_host(chunk):
    """Check map over cols with halo device matches host."""
    rng = np.random.default_rng(1)
    d = rng.normal(size=(3, 5, 17)).astype(np.float32)
    # The kernel consumes a halo of one column on each side.
    def fn(x):
        return jnp.pad(_fn_cols(x), ((0, 0), (0, 0), (1, 1)))
    host = core.map_over_cols_with_halo(fn, d, 17, halo=1, chunk_size=chunk)
    dev = core.map_over_cols_with_halo(fn, jnp.asarray(d), 17, halo=1,
                                       chunk_size=chunk)
    assert core.is_device_array(dev)
    np.testing.assert_array_equal(np.asarray(dev), host)


def test_map_over_segments_device_matches_host():
    """Check map over segments device matches host."""
    rng = np.random.default_rng(2)
    d = rng.normal(size=(9, 4, 5)).astype(np.float32)
    edges = np.array([4, 9])
    fn = lambda x: x - jnp.nanmedian(x, axis=0)[None]
    host = core.map_over_segments(fn, d, edges)
    dev = core.map_over_segments(fn, jnp.asarray(d), edges)
    assert core.is_device_array(dev)
    np.testing.assert_array_equal(np.asarray(dev), host)


def test_mixed_inputs_use_host_path():
    """Check mixed inputs use host path."""
    d = np.ones((4, 2, 2), np.float32)
    out = core.map_over_ints(lambda x, y: x + y, (d, jnp.asarray(d)), 4,
                             chunk_size=2)
    assert isinstance(out, np.ndarray)
