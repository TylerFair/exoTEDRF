"""Optimal PNGs preserve quantities without holding detector observations."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exotedrf.v2 import plotting_compat, products


class SliceGuard:
    """Array stand-in that forbids whole-array host conversion."""

    def __init__(self, data, *, allow_4d=False):
        self.data = data
        self.shape, self.ndim, self.dtype = data.shape, data.ndim, data.dtype
        self.allow_4d = allow_4d
        self.reads = []

    def __array__(self, *args, **kwargs):
        raise AssertionError('Diagnostic materialized a whole source array')

    def __getitem__(self, item):
        selected = self.data[item]
        self.reads.append((item, selected.shape))
        if selected.ndim == 4 and not self.allow_4d:
            raise AssertionError('Diagnostic selected a complete ramp segment')
        return selected


def _ramps():
    """Return ramps."""
    shape = (24, 5, 6, 80)
    ints = np.arange(shape[0], dtype=np.float32)[:, None, None, None]
    groups = np.arange(shape[1], dtype=np.float32)[None, :, None, None]
    pixels = np.arange(shape[2] * shape[3], dtype=np.float32).reshape(
        1, 1, shape[2], shape[3])
    before = 11000 + ints + 500 * groups + pixels + .7 * groups**2
    after = before - .5 * groups**2
    return before.astype(np.float32), after.astype(np.float32)


def test_inl_reads_only_the_ten_plotted_integrations():
    """Check inverse linearity reads only the ten plotted integrations."""
    before, after = _ramps()
    left, right = SliceGuard(before, allow_4d=True), SliceGuard(
        after, allow_4d=True)
    result = plotting_compat.capture_inl(left, right, 23, npix_to_bin=17)
    expected = plotting_compat.capture_inl(before, after, 23, npix_to_bin=17)
    for key in ('data_pre', 'res_pre', 'data_post', 'res_post'):
        np.testing.assert_array_equal(result[key], expected[key])
    assert left.reads == [(slice(10, 20), (10, 5, 6, 80))]
    assert right.reads == left.reads


@pytest.mark.parametrize('backend', ['numpy', 'jax'])
def test_linearity_gathers_short_ramps_without_full_segment_transfers(backend):
    """Check linearity gathers short ramps without full segment transfers."""
    before, after = _ramps()
    expected = plotting_compat.capture_linearity(
        before, after, 23, randint=np.random.RandomState(325).randint)
    if backend == 'jax':
        import jax.numpy as jnp
        before, after = jnp.asarray(before), jnp.asarray(after)
    left, right = SliceGuard(before), SliceGuard(after)
    actual = plotting_compat.capture_linearity(
        left, right, 23, randint=np.random.RandomState(325).randint)
    for plot, quantities in (('plot1', ('before', 'after', 'locs')),
                             ('plot2', ('before', 'after', 'x'))):
        for key in quantities:
            np.testing.assert_array_equal(actual[plot][key], expected[plot][key])
    assert len(left.reads) == 2
    assert len(right.reads) == 4


def test_last_group_median_columns_preserve_exact_all_integration_median():
    """Check last group median columns preserve exact all integration median."""
    before, _ = _ramps()
    before[4, -1, 2, 5] = np.nan
    source = SliceGuard(before)
    result = plotting_compat.last_group_median(
        source, 21, rows=slice(2, 5), max_bytes=21 * 3 * 4 * 7)
    expected = np.nanmedian(before[:21, -1, 2:5], axis=0)
    np.testing.assert_array_equal(result, expected)
    assert len(source.reads) == 12
    assert all(shape[0] == 21 and shape[-1] <= 7
               for _, shape in source.reads)


def _state(data, *, start=1, saturated=False):
    """Return state."""
    dq = np.full(data.shape, 2 if saturated else 0, dtype=np.uint8)
    meta = SimpleNamespace(
        mode='NIRISS/SOSS', detector='NIS',
        segment_edges=np.array([data.shape[0]]),
        segment_int_starts=np.array([start]), frame_time=1.)
    return SimpleNamespace(
        cube=SimpleNamespace(data=data, groupdq=dq,
                             pixeldq=np.zeros(data.shape[-2:], np.uint32),
                             meta=meta), aux={})


def test_panel_records_own_their_small_arrays_instead_of_pinning_cubes():
    """Check panel records own their small arrays instead of pinning cubes."""
    before, _ = _ramps()
    state = _state(before)
    record = products._capture_basic_nine_panels(
        state.cube, randint=np.random.RandomState(305).randint)
    assert len(record['panels']) == 9
    for plane in record['panels']:
        assert not np.shares_memory(plane, before)
        assert plane.base is None


def test_segmented_optimal_capture_selects_first_stage1_and_final_dq(
        tmp_path, monkeypatch):
    """Check segmented optimal capture selects first stage1 and final DQ."""
    before, after = _ramps()
    first, last = _state(before), _state(after, start=25, saturated=True)
    capture = products.OptimalCapture(
        products.output_layout({}, output_dir=tmp_path),
        {'opts': {'do_plots': True, 'output_mode': 'optimal'}})
    first_observer = capture.for_segment(0, 3)
    middle_observer = capture.for_segment(1, 3)
    last_observer = capture.for_segment(2, 3)
    bias = SimpleNamespace(name='SuperBiasStep')
    dq = SimpleNamespace(name='DQInitStep')
    first_observer(bias, first, first)
    first_observer(dq, first, first)
    recorded = capture.records['SuperBiasStep']
    middle_observer(bias, last, last)
    last_observer(bias, last, last)
    assert capture.records['SuperBiasStep'] is recorded
    assert not capture.records['DQInitStep'].any()
    last_observer(dq, last, last)
    assert capture.records['DQInitStep'].all()

    def forbidden(*args, **kwargs):
        raise AssertionError('Recomputed a visit-wide mask for a PNG')

    monkeypatch.setattr(products, '_capture_oneoverf_record', forbidden)
    capture(SimpleNamespace(name='OneOverFStep_int'), first, last)
    assert 'OneOverFStep_int' not in capture.records
    assert 'OneOverFStep_int' in capture.summary['skipped']
    assert 'first FITS segment' in capture.summary['stage1_sample']


def test_optimal_disabled_plots_do_no_work(tmp_path, monkeypatch):
    """Check optimal disabled plots do no work."""
    paths = products.output_layout({}, output_dir=tmp_path)
    ctx = {'opts': {'output_mode': 'optimal', 'do_plots': False}}

    def forbidden(*args, **kwargs):
        raise AssertionError('do_plots:false attempted to plot')

    monkeypatch.setattr(products, '_plot_cost_trials', forbidden)
    capture = products.OptimalCapture(paths, ctx)
    capture(SimpleNamespace(name='BadPixStep'), None, None)
    assert capture.render(None) == {}
    assert products.write_diagnostic_plots(None, {}, ctx, paths, []) == {}


@pytest.mark.parametrize('step_name', [
    'DQInitStep', 'INLCorrStep', 'LinearityStep', 'JumpStep',
])
def test_optimal_custom_rate_graph_skips_inapplicable_ramp_plots(
        tmp_path, step_name):
    """Check optimal custom rate graph skips inapplicable ramp plots."""
    capture = products.OptimalCapture(
        products.output_layout({}, output_dir=tmp_path),
        {'opts': {'do_plots': True, 'output_mode': 'optimal'}})
    state = SimpleNamespace(
        cube=SimpleNamespace(data=np.zeros((10, 3, 4))), aux={})
    capture(SimpleNamespace(name=step_name), state, state)
    assert step_name not in capture.records
    assert 'compatible ramp' in capture.summary['skipped'][step_name]


@pytest.mark.parametrize('mode,detector,bounds', [
    ('NIRISS/SOSS', 'NIS', (.6, 2.8)),
    ('NIRSpec/G395H', 'NRS1', (2.9, 3.9)),
    ('NIRSpec/G395H', 'NRS2', (3.8, 5.0)),
    ('MIRI/LRS', 'MIRIMAGE', (5., 12.)),
])
def test_bounded_final_summaries_match_full_diagnostics(mode, detector, bounds):
    """Check bounded final summaries match full diagnostics."""
    rng = np.random.default_rng(47)
    wave = np.linspace(bounds[0] - .05, bounds[1] + .05, 31)[::-1]
    wave[5] = wave[4]
    flux = rng.normal(100, 2, (17, wave.size)).astype(np.float32)
    flux[3, 6] = np.nan
    flux[:, 10] = 0
    spectral = {1: {'wave': wave, 'flux': flux}}
    if mode == 'NIRISS/SOSS':
        spectral[2] = {'wave': np.linspace(.6, .9, 13),
                       'flux': rng.normal(10, .1, (17, 13))}
    expected_wave, expected_flux = products._diagnostic_spectrum(
        spectral, mode, detector)
    actual = products._bounded_spectral_summary(
        spectral, mode, detector, max_image_rows=5, max_bytes=128)
    np.testing.assert_array_equal(actual['wave'], expected_wave)
    # Changing the matrix's memory order can change the float64 summation order by an ULP.
    np.testing.assert_allclose(
        actual['white'], np.nansum(expected_flux, axis=1), rtol=4e-16)
    median = np.nanmedian(expected_flux, axis=0)
    np.testing.assert_array_equal(actual['column_median'], median)
    expected_image = np.asarray([
        np.nanmean(expected_flux[lo:hi], axis=0) / median
        for lo, hi in zip(actual['time_edges'][:-1],
                          actual['time_edges'][1:])])
    np.testing.assert_allclose(actual['image'], expected_image, rtol=1e-14)
    assert actual['image'].shape[0] <= 5
    assert actual['time_edges'][-1] == flux.shape[0]


def test_per_order_scatter_chunking_preserves_quantities():
    """Check per order scatter chunking preserves quantities."""
    rng = np.random.default_rng(431)
    flux = rng.normal(5, .1, (33, 23))
    flux[5, 3] = np.nan
    expected = products._per_order_scatter(flux, [5, -5])
    guarded = SliceGuard(flux)
    actual = products._per_order_scatter(
        guarded, [5, -5], max_bytes=33 * 8 * 3)
    np.testing.assert_array_equal(actual, expected)
    assert len(guarded.reads) == 8
    assert all(shape[1] <= 3 for _, shape in guarded.reads)


def test_optimal_spectrum_plots_never_materialize_a_full_spectral_matrix():
    """Check optimal spectrum plots never materialize a full spectral matrix."""
    rng = np.random.default_rng(442)
    flux = SliceGuard(rng.normal(100, 2, (47, 31)).astype(np.float32))
    spectral = {1: {'wave': np.linspace(.9, 2.8, 31), 'flux': flux}}
    summary = products._bounded_spectral_summary(
        spectral, 'NIRISS/SOSS', 'NIS', max_image_rows=7, max_bytes=47 * 8 * 3)
    assert summary['image'].shape[0] <= 7
    for _, shape in flux.reads:
        # Either an exact median over a few columns, or one display bin's full-band rows.
        assert shape[1] <= 3 or shape[0] <= 7


def test_optimal_flux_image_labels_temporal_averaging(tmp_path, monkeypatch):
    """Check optimal flux image labels temporal averaging."""
    spectral = {1: {'wave': np.array([1., 1.1]),
                    'flux': np.ones((1201, 2))}}
    saved = {}

    def save(figure, path, **kwargs):
        saved[Path(path).name] = [ax.get_title() for ax in figure.axes]
        figure.canvas.draw()

    monkeypatch.setattr(products, '_save_figure', save)
    paths = products.output_layout({'name_tag': 'long'}, output_dir=tmp_path)
    written = products._plot_optimal_spectral_diagnostics(
        paths, spectral, [20, -20])
    assert set(written) == {'white_plot', 'flux_plot'}
    assert 'mean of up to 2 integrations per bin' in saved[
        'flux_img_long.png'][0]
