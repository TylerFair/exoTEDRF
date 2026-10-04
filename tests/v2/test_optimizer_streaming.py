"""Check streamed optimizer calculations and array lifetimes."""
import weakref

import numpy as np
import pytest

from exotedrf.v2 import core, optimize, stages
from exotedrf.v2.core import ObsMeta, RateCube
from exotedrf.v2.pipeline import CheckpointStore, Pipeline, PipelineState, Step


def _meta(nints, ngroups=3):
    """Return meta."""
    return ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP96', 2.214, ngroups,
                   np.arange(nints, dtype=float), np.array([nints]),
                   np.asarray([nints]), ('seg0.fits',), {})


@pytest.mark.parametrize('ndim', [3, 4])
@pytest.mark.parametrize('width', [1, 5, 12])
def test_device_deepstack_columns_matches_masked_nanmedian(ndim, width):
    """Check device deepstack columns matches masked nanmedian."""
    rng = np.random.default_rng(3)
    shape = (40, 2, 8, 12) if ndim == 4 else (40, 8, 12)
    data = rng.normal(size=shape).astype(np.float32)
    data[rng.random(shape) < 0.05] = np.nan
    mask = rng.random(40) < 0.6
    mask[:3] = True
    expected = np.nanmedian(data[mask], axis=0)
    got = stages._device_deepstack_columns(data, mask, width)
    assert got.dtype == data.dtype and got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6,
                               equal_nan=True)
    np.testing.assert_allclose(stages._host_deepstack(data, mask), expected,
                               rtol=1e-6, atol=1e-6, equal_nan=True)


def test_device_deepstack_column_width_policy(monkeypatch):
    """Check device deepstack column width policy."""
    data = np.zeros((10, 4, 6), np.float32)
    monkeypatch.setenv('EXOTEDRF_DEVICE_FAST_PATH', 'off')
    assert stages._device_deepstack_column_width(data) is None
    monkeypatch.setenv('EXOTEDRF_DEVICE_FAST_PATH', 'device')
    monkeypatch.setattr(core, 'device_memory_bytes', lambda: 10 * 4 * 4 * 8 * 2)
    # Budget 0.7 x (2 columns' worth) -> one column per block.
    assert stages._device_deepstack_column_width(data) == 1
    monkeypatch.setattr(core, 'device_memory_bytes', lambda: 1 << 40)
    assert stages._device_deepstack_column_width(data) == 6
    monkeypatch.setattr(core, 'device_memory_bytes', lambda: 1)
    assert stages._device_deepstack_column_width(data) is None
    monkeypatch.setenv('EXOTEDRF_DEVICE_FAST_PATH', 'auto')
    monkeypatch.setattr(core, 'device_memory_bytes', lambda: 1 << 40)
    import jax
    if jax.default_backend() != 'gpu':
        assert stages._device_deepstack_column_width(data) is None


def _extract_fixture():
    """Return extract fixture."""
    rng = np.random.default_rng(7)
    data = rng.normal(loc=5., size=(14, 10, 4)).astype(np.float32)
    data[:, 2] = 10
    data[:, 3] = 2
    data[:, 5:9] = 1
    err = np.ones_like(data)
    dq = np.zeros_like(data, np.uint32)
    dq[:, 2, 0] = 1
    dq[:, 2, 1] = 2
    dq[:, 2, 2] = 4
    cube = RateCube(data, err, dq, _meta(14))
    ctx = {
        'opts': {'extract_width_soss2': 4, 'mask_do_not_use_pixels': True,
                 'mask_saturated_pixels': True,
                 'saturation_rescue': False},
        'centroids': {'ypos o1': np.full(4, 3.),
                      'ypos o2': np.full(4, 7.)},
        'waves': {1: np.array([0.9, 1.4, 2.0, 2.6]),
                  2: np.array([0.6, 0.7, 0.8, 0.84])},
    }
    return PipelineState(cube), ctx


@pytest.mark.parametrize('o2_fixed', [True, False])
def test_prepared_extract_results_match_sequential_extract(o2_fixed):
    """Check prepared extract results match sequential extract."""
    state, ctx = _extract_fixture()
    if not o2_fixed:
        ctx['opts'].pop('extract_width_soss2')
    widths = [2, 3, 5]
    prepared = stages.prepared_extract_results(state, {}, ctx, widths)
    assert len(prepared) == len(widths)
    for width, (cost, scatter, duration) in zip(widths, prepared):
        out = stages.step_extract(state, {'extract_width': width}, ctx)
        expected_cost, expected_scatter = stages.evaluate_production_cost(
            out, {}, ctx)
        assert cost == float(expected_cost)
        np.testing.assert_array_equal(scatter, expected_scatter)
        assert duration >= 0.0

    spectra = stages.extract_orders(state, {}, ctx, mode='sum',
                                    extract_widths=[2, 3])
    assert spectra[1][0].shape == (14, 2, 4)
    assert spectra[2][0].shape == (14, 2, 4)


def test_final_graph_releases_raw_cube_during_run():
    """Check final graph releases raw cube during run."""
    raw = np.ones((6, 1, 1), np.float32)
    state = PipelineState(RateCube(raw, np.ones_like(raw),
                                   np.zeros_like(raw, np.uint32), _meta(6)))
    ref = weakref.ref(raw)
    seen = {}

    def first(state, params, ctx):
        # A fresh cube replaces the raw one; the raw array must then die.
        new = np.asarray(state.cube.data) * 2
        return PipelineState(RateCube(new, state.cube.err, state.cube.dq,
                                      state.cube.meta), state.aux)

    def second(state, params, ctx):
        seen['raw_alive'] = ref() is not None
        return state

    pipeline = Pipeline([Step('A', first, ()), Step('B', second, ())],
                        'NIRISS/SOSS', {'opts': {}})
    store = CheckpointStore(keep=set())
    del raw
    holder = [state]
    del state
    optimize._run_final_graph(pipeline, holder, {}, store, None, None)
    assert seen['raw_alive'] is False


def test_device_percentile_pair_is_bit_identical_to_numpy(monkeypatch):
    """Check device percentile pair is bit identical to NumPy."""
    from exotedrf.v2 import hoststats
    rng = np.random.default_rng(11)
    data = rng.normal(size=(30, 40, 50)).astype(np.float32)
    data[rng.random(data.shape) < 0.03] = np.nan
    expected = np.nanpercentile(data, 10.)
    # Force the device selection path even on CPU and for a small array.
    monkeypatch.setattr(hoststats, '_device_order_statistic_pair',
                        lambda array, lo, hi: _sorted_pair(array, lo, hi))
    got = hoststats._resident_statistic(data, 10.)
    assert np.asarray(got, np.float64) == np.asarray(expected, np.float64)
    assert hoststats.nanpercentile_fast(data, 10.) == expected


def _sorted_pair(array, lo, hi):
    """Return sorted pair."""
    import jax.numpy as jnp
    ordered = jnp.sort(jnp.asarray(np.ascontiguousarray(array)).reshape(-1))
    return np.asarray(ordered[jnp.asarray([lo, hi])])


@pytest.mark.parametrize('workers', ['1', '4'])
def test_threaded_nearest_fill_matches_v1_griddata(monkeypatch, workers):
    """Check threaded nearest fill matches v1 griddata."""
    from scipy.interpolate import griddata
    monkeypatch.setenv('EXOTEDRF_FILL_WORKERS', workers)
    rng = np.random.default_rng(5)
    rate = rng.normal(size=(70, 12, 16)).astype(np.float32)
    rate[:, 3, 4] = np.nan
    rate[:, 8:10, 10] = np.nan
    for i in range(70):
        rate[i, rng.integers(0, 12), rng.integers(0, 16)] = np.nan
    rate[7] = np.nan
    rate[11] = rng.normal(size=(12, 16)).astype(np.float32)
    expected = rate.copy()
    py, px = np.mgrid[0:12, 0:16]
    for j in range(70):
        ii = np.where(np.isfinite(expected[j]))
        if ii[0].size:
            expected[j] = griddata(ii, expected[j][ii], (py, px),
                                   method='nearest')
        else:
            expected[j] = np.nan
    got = stages._nearest_fill_rate_planes(rate, copy=True)
    np.testing.assert_array_equal(got, expected)
    keep = np.ones(70, bool); keep[11] = False
    assert np.isnan(rate[keep, 3, 4]).all()


@pytest.mark.parametrize('ndim', [3, 4])
@pytest.mark.parametrize('resident', [False, True])
def test_deepstack_pair_matches_subtract_then_median(monkeypatch, ndim, resident):
    """Check deepstack pair matches subtract then median."""
    import jax.numpy as jnp
    rng = np.random.default_rng(9)
    shape = (30, 2, 6, 10) if ndim == 4 else (30, 6, 10)
    data = (rng.normal(size=shape) * 50 + 1000).astype(np.float32)
    data[rng.random(shape) < 0.05] = np.nan
    sub = (rng.normal(size=shape[1:]) * 20).astype(np.float32)
    mask = rng.random(30) < 0.7
    mask[:2] = True
    expected_plain = np.nanmedian(data[mask], axis=0)
    expected_corr = np.nanmedian((data - sub[None]).astype(np.float32)[mask], axis=0)
    for width in (None, 3, 10):
        if width is None:
            monkeypatch.setenv('EXOTEDRF_DEVICE_FAST_PATH', 'off')
        else:
            monkeypatch.setenv('EXOTEDRF_DEVICE_FAST_PATH', 'device')
            monkeypatch.setattr(stages, '_device_deepstack_column_width',
                                lambda d, **k: width)
        arr = jnp.asarray(data) if resident else data
        plain, corr = stages.deepstack_pair(arr, mask, sub)
        np.testing.assert_array_equal(plain, expected_plain)
        np.testing.assert_array_equal(corr, expected_corr)
        np.testing.assert_array_equal(plain, stages._host_deepstack(data, mask))
