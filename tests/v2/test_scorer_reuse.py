"""Check cached candidate scores against complete reduction steps."""

import numpy as np
import pytest

from exotedrf.v2 import core, stages
from exotedrf.v2.core import ObsMeta, RateCube
from exotedrf.v2.pipeline import PipelineState


# 1.

def _soss_meta(nints, dimx):
    """Return SOSS meta."""
    return ObsMeta(
        mode='NIRISS/SOSS', detector='NIS', subarray='SUBSTRIP96',
        frame_time=5.494, ngroups=5,
        int_times=60000. + np.arange(nints) * 1e-4,
        baseline_ints=np.asarray([nints // 3]),
        segment_edges=np.asarray([nints]),
        filenames=('seg001_uncal.fits',),
        extra={'header': {'FILTER': 'CLEAR', 'NINTS': nints},
               'segment_headers': ({'NINTS': nints},),
               'segment_int_starts': (1,)})


def _badpix_fixture():
    """Return bad-pixel correction fixture."""
    rng = np.random.default_rng(0x135283a)
    nints, dimy, dimx = 13, 12, 24
    data = rng.normal(100., .25, (nints, dimy, dimx)).astype(np.float32)
    data[5, 4, 11] += 20.
    data[8, 5, 16] -= 15.
    err = np.ones_like(data)
    dq = np.zeros_like(data, np.uint32)
    dq[10, 3, 8] = core.DQ_HOT
    dq[7, 6, 14] = core.DQ_SATURATED
    cube = RateCube(data, err, dq, _soss_meta(nints, dimx))
    ctx = {
        'opts': {'extract_width_soss2': None, 'wave_range': None,
                 'w1': 1., 'w2': 1.},
        'centroids': {
            'ypos o1': np.linspace(4.5, 5.5, dimx),
            'ypos o2': np.concatenate([
                np.linspace(8., 9., 17), np.full(dimx - 17, np.nan)])},
        'waves': {1: np.linspace(.9, 2., dimx),
                  2: np.linspace(.6, 1., dimx)},
    }
    params = {'space_outlier_threshold': 8., 'time_outlier_threshold': 8.,
              'box_size': 2, 'window_size': 3, 'extract_width': 4.}
    return PipelineState(cube), params, ctx


def _disable_reuse(scorer):
    """Reproduce the original per-candidate work pattern exactly."""
    scorer._cache_box_medians = False
    original = scorer._prepare_temporal_records

    def no_scatter_reuse(records, window, key=None):
        return original(records, window, key=None)

    scorer._prepare_temporal_records = no_scatter_reuse


COORDINATES = (
    ('box_size', [2, 3]),
    ('space_outlier_threshold', [6., 8., 10.]),
    ('window_size', [3, 5]),
    ('time_outlier_threshold', [6., 8., 10.]),
)


def test_badpix_scorer_reuse_is_bit_identical_to_recomputing(monkeypatch):
    """Check bad-pixel correction scorer reuse is bit identical to recomputing."""
    monkeypatch.setenv('EXOTEDRF_V2_BADPIX_COLUMN_CHUNK', '6')

    def sweep(reuse):
        state, params, ctx = _badpix_fixture()
        scorer = stages.prepare_badpix_scorer(state, params, ctx)
        assert scorer is not None
        if not reuse:
            _disable_reuse(scorer)
        trial = dict(params)
        out = []
        for parameter, candidates in COORDINATES:
            results = scorer.evaluate_candidates(parameter, candidates, trial)
            costs = [float(result[0]) for result in results]
            out.append((costs, [np.asarray(result[1]) for result in results]))
            finite = [index for index, cost in enumerate(costs)
                      if np.isfinite(cost)]
            winner = min(finite, key=lambda index: costs[index])
            trial[parameter] = candidates[winner]
        return out, trial

    fast, fast_params = sweep(True)
    slow, slow_params = sweep(False)
    assert fast_params == slow_params
    for (fast_costs, fast_scatter), (slow_costs, slow_scatter) in zip(
            fast, slow):
        assert fast_costs == slow_costs
        for one, other in zip(fast_scatter, slow_scatter):
            np.testing.assert_array_equal(one, other)


def test_badpix_scorer_reuse_matches_the_full_science_step(monkeypatch):
    """Check bad-pixel correction scorer reuse matches the full science step."""
    monkeypatch.setenv('EXOTEDRF_V2_BADPIX_COLUMN_CHUNK', '6')
    state, params, ctx = _badpix_fixture()
    scorer = stages.prepare_badpix_scorer(state, params, ctx)
    for parameter, candidates in COORDINATES:
        results = scorer.evaluate_candidates(parameter, candidates, params)
        for value, (cost, scatter, _duration) in zip(candidates, results):
            trial = dict(params, **{parameter: value})
            full = stages.step_badpix(state, trial, ctx)
            expected, expected_scatter = stages.evaluate_optimizer_cost(
                full, trial, ctx)
            np.testing.assert_allclose(cost, expected, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(scatter, expected_scatter, rtol=1e-5,
                                       atol=1e-7, equal_nan=True)


def test_badpix_box_median_cache_holds_one_box_size(monkeypatch):
    """Check bad-pixel correction box median cache holds one box size."""
    monkeypatch.setenv('EXOTEDRF_V2_BADPIX_COLUMN_CHUNK', '6')
    state, params, ctx = _badpix_fixture()
    scorer = stages.prepare_badpix_scorer(state, params, ctx)
    assert scorer._cache_box_medians
    scorer.evaluate_candidates('box_size', [2, 3], params)
    assert scorer._boxmed_key == 3
    assert len(scorer._layouts) == 2
    assert isinstance(scorer._boxmed, tuple)


def test_badpix_box_medians_match_the_fused_kernel():
    """Check bad-pixel correction box medians match the fused kernel."""
    from exotedrf.v2.kernels import badpix as k_badpix
    rng = np.random.default_rng(11)
    cube = rng.normal(100., .3, (6, 10, 16)).astype(np.float32)
    dq = np.zeros(cube.shape, np.uint32)
    bad = rng.random((10, 16)) < .1
    for box in (2, 3, 5):
        fused = k_badpix.apply_spatial_badpix(
            cube, dq, bad, box_size=box, instrument='NIRISS')
        split = k_badpix.apply_spatial_badpix_from_median(
            cube, dq, bad, k_badpix.spatial_interp_median(
                cube, box_size=box, instrument='NIRISS'))
        np.testing.assert_array_equal(np.asarray(fused[0]),
                                      np.asarray(split[0]))
        np.testing.assert_array_equal(np.asarray(fused[1]),
                                      np.asarray(split[1]))


# 2.

def _nirspec_meta(nints, dimx):
    """Return NIRSpec meta."""
    return ObsMeta(
        mode='NIRSpec/G395H', detector='NRS1', subarray='SUB2048',
        frame_time=.902, ngroups=3,
        int_times=60000. + np.arange(nints) * 1e-4,
        baseline_ints=np.asarray([4]),
        segment_edges=np.asarray([nints]),
        filenames=('seg001_uncal.fits',),
        extra={'header': {'NINTS': nints}, 'segment_headers': (
            {'NINTS': nints},), 'segment_int_starts': (1,)})


def _nirspec_extract_fixture():
    """Return NIRSpec extract fixture."""
    rng = np.random.default_rng(5)
    nints, dimy, dimx = 14, 9, 12
    data = rng.normal(200., 1., (nints, dimy, dimx)).astype(np.float32)
    err = np.full_like(data, .5)
    dq = np.zeros_like(data, np.uint32)
    dq[:, 4, 3] = core.DQ_DO_NOT_USE
    dq[:, 5, 7] = core.DQ_SATURATED
    cube = RateCube(data, err, dq, _nirspec_meta(nints, dimx))
    wave_map = np.broadcast_to(
        3. + .05 * np.arange(dimx, dtype=float)[None, :], (dimy, dimx))
    ctx = {
        'opts': {'mask_do_not_use_pixels': True, 'mask_saturated_pixels': True, 'wave_range': None,
                 'w1': 1., 'w2': 1.},
        'centroids': {'xpos': np.arange(2, dimx, dtype=float),
                      'ypos': np.full(dimx - 2, 4.5)},
        'nirspec_wave_map': np.asarray(wave_map, dtype=float),
        'nirspec_xstart': 2,
    }
    return PipelineState(cube), ctx


def _miri_extract_fixture():
    """Return MIRI extract fixture."""
    rng = np.random.default_rng(6)
    nints, dimy, dimx = 14, 10, 8
    data = rng.normal(150., 1., (nints, dimy, dimx)).astype(np.float32)
    err = np.full_like(data, .4)
    dq = np.zeros_like(data, np.uint32)
    dq[:, 3, 4] = core.DQ_DO_NOT_USE
    dq[:, 6, 2] = core.DQ_SATURATED
    meta = ObsMeta(
        mode='MIRI/LRS', detector='MIRIMAGE', subarray='SLITLESSPRISM',
        frame_time=2., ngroups=3,
        int_times=60000. + np.arange(nints) * 1e-4,
        baseline_ints=np.asarray([4]),
        segment_edges=np.asarray([nints]),
        filenames=('seg001_uncal.fits',),
        extra={'header': {'NINTS': nints},
               'segment_headers': ({'NINTS': nints},),
               'segment_int_starts': (1,)})
    cube = RateCube(data, err, dq, meta)
    ypos = np.arange(2., 9.)
    wave_map = np.broadcast_to(
        5. + .3 * np.arange(dimy, dtype=float)[:, None], (dimy, dimx))
    ctx = {
        'opts': {'mask_do_not_use_pixels': True, 'mask_saturated_pixels': True,
                 'extract_method': 'box',
                 'wave_range': None, 'w1': 1., 'w2': 1.},
        'centroids': {'xpos': np.full(ypos.size, 3.5), 'ypos': ypos},
        'miri_wave_map': np.asarray(wave_map, dtype=float),
    }
    return PipelineState(cube), ctx


@pytest.mark.parametrize('instrument', ['nirspec', 'miri'])
def test_prepared_aperture_sweep_matches_sequential_extract(instrument):
    """Check prepared aperture sweep matches sequential extract."""
    if instrument == 'nirspec':
        state, ctx = _nirspec_extract_fixture()
        step = stages.step_extract_nirspec
    else:
        state, ctx = _miri_extract_fixture()
        step = stages.step_extract_miri
    widths = [2., 3., 5.]

    prepared = stages.prepared_extract_results(state, {}, ctx, widths)
    assert len(prepared) == len(widths)
    for width, (cost, scatter, duration) in zip(widths, prepared):
        out = step(state, {'extract_width': width}, ctx)
        expected_cost, expected_scatter = stages.evaluate_production_cost(
            out, {}, ctx)
        assert cost == float(expected_cost)
        np.testing.assert_array_equal(scatter, expected_scatter)
        assert duration >= 0.

    spectra = stages.extract_orders(state, {}, ctx, mode='sum',
                                    extract_widths=widths)
    flux, ferr = spectra[1]
    assert flux.shape[1] == len(widths)
    assert ferr.shape == flux.shape


@pytest.mark.parametrize('instrument', ['nirspec', 'miri'])
def test_swept_candidate_axis_holds_the_single_width_spectra(instrument):
    """Check swept candidate axis holds the single width spectra."""
    if instrument == 'nirspec':
        state, ctx = _nirspec_extract_fixture()
        extract = stages._extract_orders_nirspec
    else:
        state, ctx = _miri_extract_fixture()
        extract = stages._extract_orders_miri
    widths = [2., 4.]
    swept = extract(state, {}, ctx, mode='sum', extract_widths=widths)
    for index, width in enumerate(widths):
        single = extract(state, {'extract_width': width}, ctx, mode='sum')
        np.testing.assert_array_equal(
            np.asarray(swept[1][0][:, index]), np.asarray(single[1][0]))
        np.testing.assert_array_equal(
            np.asarray(swept[1][1][:, index]), np.asarray(single[1][1]))


def test_miri_optimal_extraction_declines_the_prepared_sweep():
    """Check MIRI optimal extraction declines the prepared sweep."""
    state, ctx = _miri_extract_fixture()
    ctx['opts']['extract_method'] = 'optimal'
    with pytest.raises(NotImplementedError):
        stages.prepared_extract_results(state, {}, ctx, [3., 5.])


def test_prepared_sweep_rejects_four_dimensional_cubes():
    """Check prepared sweep rejects four dimensional cubes."""
    from exotedrf.v2.core import RampCube
    data = np.zeros((4, 3, 6, 8), np.float32)
    meta = _nirspec_meta(4, 8)
    state = PipelineState(RampCube(
        data, np.zeros_like(data, np.uint8),
        np.zeros((6, 8), np.uint32), meta))
    ctx = {'opts': {}, 'centroids': {'xpos': np.arange(8.),
                                     'ypos': np.full(8, 3.)}}
    with pytest.raises(NotImplementedError):
        stages.extract_orders(state, {'extract_width': 3.}, ctx,
                              extract_widths=[3., 5.])


@pytest.mark.parametrize('instrument', ['nirspec', 'miri'])
def test_optimizer_accepts_nirspec_and_miri_extract_steps(instrument):
    """Check optimizer accepts NIRSpec and MIRI extract steps."""
    from exotedrf.v2 import optimize
    from exotedrf.v2.pipeline import CheckpointStore, Pipeline, Step

    if instrument == 'nirspec':
        state, ctx = _nirspec_extract_fixture()
        step_fn, mode = stages.step_extract_nirspec, 'NIRSpec/G395H'
    else:
        state, ctx = _miri_extract_fixture()
        step_fn, mode = stages.step_extract_miri, 'MIRI/LRS'
    pipeline = Pipeline(
        [Step('Extract', step_fn, ('extract_width',))], mode, ctx)
    store = CheckpointStore(keep={'Extract'})
    store.put('Extract', state)
    results = optimize._prepared_extract_results(
        pipeline, 0, 'extract_width', [2., 3.], {}, store, None, None)
    assert results is not None and len(results) == 2
    for width, (cost, _scatter, _duration) in zip([2., 3.], results):
        out = step_fn(state, {'extract_width': width}, ctx)
        expected, _ = stages.evaluate_production_cost(out, {}, ctx)
        assert cost == float(expected)


def test_optimizer_declines_the_prepared_sweep_for_miri_optimal():
    """Check optimizer declines the prepared sweep for MIRI optimal."""
    from exotedrf.v2 import optimize
    from exotedrf.v2.pipeline import CheckpointStore, Pipeline, Step

    state, ctx = _miri_extract_fixture()
    ctx['opts']['extract_method'] = 'optimal'
    pipeline = Pipeline(
        [Step('Extract', stages.step_extract_miri, ('extract_width',))],
        'MIRI/LRS', ctx)
    store = CheckpointStore(keep={'Extract'})
    store.put('Extract', state)
    assert optimize._prepared_extract_results(
        pipeline, 0, 'extract_width', [2., 3.], {}, store, None,
        None) is None


# 3.

def test_flux_image_is_unbinned_below_the_row_limit():
    """Check flux image is unbinned below the row limit."""
    from exotedrf.v2 import products
    rng = np.random.default_rng(3)
    flux = 1000. + rng.normal(0., 1., (40, 7))
    column_median = np.nanmedian(flux, axis=0)
    image, edges, bin_size = products._binned_flux_image(flux, column_median)
    assert bin_size == 1
    np.testing.assert_array_equal(image, flux / column_median)
    np.testing.assert_array_equal(edges, np.arange(41))


def test_long_visit_flux_image_bins_every_integration_exactly():
    """Check long visit flux image bins every integration exactly."""
    from exotedrf.v2 import products
    rng = np.random.default_rng(4)
    nints, nwave = 25, 5
    flux = 1000. + rng.normal(0., 1., (nints, nwave))
    flux[3, 2] = np.nan
    flux[:, 4] = np.nan
    column_median = np.nanmedian(flux, axis=0)
    image, edges, bin_size = products._binned_flux_image(
        flux, column_median, max_image_rows=6)
    assert bin_size == 5
    np.testing.assert_array_equal(edges, [0, 5, 10, 15, 20, 25])
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        block = flux[low:high]
        counts = np.sum(~np.isnan(block), axis=0)
        expected = np.divide(np.nansum(block, axis=0), counts,
                             out=np.full(nwave, np.nan), where=counts != 0)
        np.testing.assert_array_equal(image[index], expected / column_median)
    assert np.isnan(image[:, 4]).all()


def test_flux_image_bin_count_never_exceeds_the_row_limit():
    """Check flux image bin count never exceeds the row limit."""
    from exotedrf.v2 import products
    flux = np.ones((6100, 3))
    _image, edges, bin_size = products._binned_flux_image(
        flux, np.ones(3))
    assert bin_size == 6
    assert edges.size - 1 <= products._MAX_IMAGE_ROWS
    assert edges[-1] == 6100


def test_concurrent_saves_write_every_png_and_flush_before_returning(
        tmp_path, monkeypatch):
    """Check concurrent saves write every png and flush before returning."""
    from exotedrf.v2 import products
    monkeypatch.setenv('EXOTEDRF_V2_PLOT_THREADS', '3')
    paths = []
    with products._concurrent_saves():
        for index in range(6):
            figure = products._new_figure(figsize=(1., 1.))
            axes = figure.subplots()
            axes.plot([0, 1], [index, index + 1])
            path = tmp_path / f'plot_{index}.png'
            products._save_figure(figure, path, dpi=40)
            paths.append(path)
        assert products._PLOT_SAVE_CONTEXT is not None
    assert products._PLOT_SAVE_CONTEXT is None
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_single_worker_keeps_the_inline_save_path(tmp_path, monkeypatch):
    """Check single worker keeps the inline save path."""
    from exotedrf.v2 import products
    monkeypatch.setenv('EXOTEDRF_V2_PLOT_THREADS', '1')
    with products._concurrent_saves():
        assert products._PLOT_SAVE_CONTEXT is None
        figure = products._new_figure(figsize=(1., 1.))
        figure.subplots().plot([0, 1], [0, 1])
        path = tmp_path / 'inline.png'
        products._save_figure(figure, path, dpi=40)
        # Inline saves are complete as soon as the call returns.
        assert path.exists()


def test_deferred_save_failure_is_retried_on_the_calling_thread(
        tmp_path, monkeypatch):
    """Check deferred save failure is retried on the calling thread."""
    from exotedrf.v2 import products
    monkeypatch.setenv('EXOTEDRF_V2_PLOT_THREADS', '2')
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError('worker failed')
        (tmp_path / 'retried.png').write_bytes(b'ok')

    with products._concurrent_saves():
        products._submit_save(flaky)
    assert len(attempts) == 2
    assert (tmp_path / 'retried.png').read_bytes() == b'ok'


def test_concurrent_png_output_matches_the_inline_render(tmp_path,
                                                         monkeypatch):
    """Check concurrent png output matches the inline render."""
    from exotedrf.v2 import products

    def draw(path):
        figure = products._new_figure(figsize=(3., 2.))
        axes = figure.subplots()
        axes.pcolormesh(np.arange(5), np.arange(4),
                        np.arange(12.).reshape(3, 4), rasterized=True)
        axes.set_title('Normalized Flux Image')
        products._save_figure(figure, path, dpi=60, bbox_inches=None)

    monkeypatch.setenv('EXOTEDRF_V2_PLOT_THREADS', '1')
    draw(tmp_path / 'inline.png')
    monkeypatch.setenv('EXOTEDRF_V2_PLOT_THREADS', '4')
    with products._concurrent_saves():
        draw(tmp_path / 'threaded.png')
    assert (tmp_path / 'inline.png').read_bytes() == \
        (tmp_path / 'threaded.png').read_bytes()


def test_nested_concurrent_saves_do_not_deadlock(tmp_path, monkeypatch):
    """Check nested concurrent saves do not deadlock."""
    from exotedrf.v2 import products
    monkeypatch.setenv('EXOTEDRF_V2_PLOT_THREADS', '2')
    with products._concurrent_saves():
        outer = products._PLOT_SAVE_CONTEXT
        with products._concurrent_saves():
            assert products._PLOT_SAVE_CONTEXT is outer
            figure = products._new_figure(figsize=(1., 1.))
            figure.subplots().plot([0, 1], [1, 0])
            products._save_figure(figure, tmp_path / 'nested.png', dpi=40)
        assert products._PLOT_SAVE_CONTEXT is outer
    assert (tmp_path / 'nested.png').exists()


def test_plot_thread_setting_rejects_non_integers(monkeypatch):
    """Check plot thread setting rejects non integers."""
    from exotedrf.v2 import products
    monkeypatch.setenv('EXOTEDRF_V2_PLOT_THREADS', 'many')
    with pytest.raises(ValueError, match='EXOTEDRF_V2_PLOT_THREADS'):
        products._plot_save_workers()
    monkeypatch.delenv('EXOTEDRF_V2_PLOT_THREADS')
    assert products._plot_save_workers() == 1
