"""Check products."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from exotedrf.v2 import products


def test_output_layout_defaults_below_pipeline_v2(tmp_path):
    """Check output layout defaults below pipeline v2."""
    layout = products.output_layout(
        {'pipeline_outputs_directory': str(tmp_path), 'name_tag': ''})
    assert layout['root'] == tmp_path / 'v2'
    assert layout['cost'] == tmp_path / 'v2' / 'Files' / 'Cost_.txt'
    assert layout['decision_ranking'] == (
        tmp_path / 'v2' / 'Files' / 'Decision_ranking_.txt')
    assert layout['cost_plot'] == tmp_path / 'v2' / 'Files' / 'Cost_.png'
    assert layout['scatter_plot'] == (
        tmp_path / 'v2' / 'Files' / 'Scatter_Plot_.png')
    assert layout['flux_plot'] == tmp_path / 'v2' / 'Files' / 'flux_img_.png'
    assert layout['white_plot'] == tmp_path / 'v2' / 'Files' / 'norm_white_.png'
    assert layout['centroid_plot'] == tmp_path / 'v2' / 'Stage3' / \
        'centroiding.png'
    assert layout['spectra'] == (
        tmp_path / 'v2' / 'Stage3' / 'soss_box_spectra_fullres.fits')
    assert all(layout[name].parent.exists()
               for name in ('cost', 'rate', 'lcestimate', 'spectra'))


def test_output_layout_preserves_v1_output_tag_root_isolation(tmp_path):
    """Check output layout preserves v1 output tag root isolation."""
    layout = products.output_layout({
        'pipeline_outputs_directory': str(tmp_path / 'pipeline'),
        'output_tag': 'visit2',
    })
    assert layout['root'] == tmp_path / 'pipeline_visit2' / 'v2'


def test_output_layout_preserves_v1_trailing_slash_tag_join(tmp_path):
    """Check output layout preserves v1 trailing slash tag join."""
    configured = f'{tmp_path / "pipeline"}/'
    layout = products.output_layout({
        'pipeline_outputs_directory': configured,
        'output_tag': 'visit2',
    })
    assert layout['root'] == tmp_path / 'pipeline' / '_visit2' / 'v2'


def test_v1_text_logs_and_json_summary(tmp_path):
    """Check v1 text logs and json summary."""
    paths = products.output_layout({'name_tag': 'target'}, output_dir=tmp_path)
    trials = [
        {'params': {'width': 3, 'window': 5}, 'duration_s': 1.25,
         'cost': 0.125, 'scatter': [1.0, 2.5]},
        {'params': {'width': 4, 'window': 5}, 'duration_s': 2.0,
         'cost': float('nan'), 'scatter': [3.0]},
    ]
    products.write_optimizer_logs(
        paths, trials, ['width', 'window'],
        {'winners': {'width': 3}, 'array': [1, 2],
         'nonfinite': [float('nan'), float('inf')]})

    cost_lines = paths['cost'].read_text(encoding='utf-8').splitlines()
    assert cost_lines[0] == 'width\twindow\tduration_s\tcost'
    assert cost_lines[1] == '3\t5\t1.2\t0.125000000000'
    assert cost_lines[2].endswith('\t2.0\tnan')
    assert paths['scatter'].read_text(encoding='utf-8').splitlines() == [
        '1 2.5', '3']
    summary_text = paths['summary'].read_text(encoding='utf-8')
    summary = json.loads(summary_text)
    assert summary['winners'] == {'width': 3}
    assert summary['nonfinite'] == [None, None]
    assert 'NaN' not in summary_text and 'Infinity' not in summary_text
    assert isinstance(paths['summary'], Path)
    # No decision_sensitivity key in the summary: no ranking product.
    assert not paths['decision_ranking'].exists()


def test_write_decision_ranking_writes_parseable_tsv(tmp_path):
    """Check write decision ranking writes parseable tsv."""
    paths = products.output_layout({'name_tag': 'target'}, output_dir=tmp_path)
    ranking = [
        {'parameter': 'time_window', 'rank': 1, 'n_pairs': 2,
         'mean_rel': 0.325, 'max_rel': 0.6, 'median_rel': 0.325,
         'mean_rel_scatter': 0.075, 'prune_candidate': False,
         'winning_values': {5: 1, 7: 1}},
        {'parameter': 'soss_inner_mask_width', 'rank': 2, 'n_pairs': 3,
         'mean_rel': 0.0002, 'max_rel': 0.0003, 'median_rel': 0.0003,
         'mean_rel_scatter': float('nan'), 'prune_candidate': True,
         'winning_values': {1: 1}},
    ]

    returned = products.write_decision_ranking(paths, ranking)
    assert returned == paths['decision_ranking']
    lines = paths['decision_ranking'].read_text(
        encoding='utf-8').splitlines()
    header = lines[0].split('\t')
    assert header == ['rank', 'parameter', 'n_pairs', 'mean_rel', 'max_rel',
                      'median_rel', 'mean_rel_scatter', 'prune_candidate',
                      'winning_values']
    rows = [dict(zip(header, line.split('\t'))) for line in lines[1:]]
    assert [row['parameter'] for row in rows] == \
        ['time_window', 'soss_inner_mask_width']
    assert rows[0]['prune_candidate'] == 'False'
    assert rows[1]['prune_candidate'] == 'True'
    assert json.loads(rows[0]['winning_values']) == {'5': 1, '7': 1}
    # NaN is written as plain text, not JSON-illegal, since this is a TSV.
    assert rows[1]['mean_rel_scatter'] == 'nan'

    # Write_optimizer_logs calls through when the summary carries a ranking.
    products.write_optimizer_logs(
        paths, [], [], {'decision_sensitivity': ranking})
    assert paths['decision_ranking'].exists()


def test_final_writer_prefers_sorted_trimmed_spectral_products(
        tmp_path, monkeypatch):
    """Check final writer prefers sorted trimmed spectral products."""
    captured = {}
    monkeypatch.setattr(
        'exotedrf.v2.io.save_rate_cube',
        lambda cube, path, extra_header=None: None)

    def capture_spectra(path, orders, meta, extra_header=None):
        captured.update(orders)

    monkeypatch.setattr('exotedrf.v2.io.save_spectra', capture_spectra)
    cube = SimpleNamespace(
        data=np.zeros((1, 2, 4), np.float32),
        err=np.zeros((1, 2, 4), np.float32),
        dq=np.zeros((1, 2, 4), np.uint32),
        meta=SimpleNamespace(),
    )
    state = SimpleNamespace(
        cube=cube,
        aux={
            'spectra': {1: (np.asarray([[10., 20., 30., 40.]]), None)},
            'spectral_products': {
                1: {
                    'wave': np.asarray([1., 2., 3.]),
                    'flux': np.asarray([[30., 40., 10.]]),
                    'ferr': np.asarray([[3., 4., 1.]]),
                },
            },
        },
    )
    paths = products.output_layout({'name_tag': 'target'}, output_dir=tmp_path)
    products.write_final_products(
        state, {}, {'waves': {1: np.asarray([3., np.nan, 1., 2.])}}, paths)

    np.testing.assert_array_equal(captured['O1']['wave'], [1., 2., 3.])
    np.testing.assert_array_equal(captured['O1']['flux'], [[30., 40., 10.]])
    np.testing.assert_array_equal(captured['O1']['ferr'], [[3., 4., 1.]])


def test_final_writer_rejects_misaligned_custom_spectrum(tmp_path, monkeypatch):
    """Check final writer rejects misaligned custom spectrum."""
    monkeypatch.setattr(
        'exotedrf.v2.io.save_rate_cube',
        lambda cube, path, extra_header=None: None)
    monkeypatch.setattr(
        'exotedrf.v2.io.save_spectra',
        lambda path, orders, meta, extra_header=None: None)
    cube = SimpleNamespace(
        data=np.zeros((1, 2, 4), np.float32),
        err=np.zeros((1, 2, 4), np.float32),
        dq=np.zeros((1, 2, 4), np.uint32), meta=SimpleNamespace())
    state = SimpleNamespace(
        cube=cube,
        aux={'spectra': {1: (np.ones((1, 4)), np.ones((1, 4)))}})
    paths = products.output_layout({'name_tag': 'target'}, output_dir=tmp_path)
    with pytest.raises(ValueError, match='wavelength shape'):
        products.write_final_products(
            state, {}, {'waves': {1: np.asarray([1., 2., 3.])}}, paths)


@pytest.mark.parametrize('generate_lc', [True, False])
def test_final_writer_conditionally_writes_pca_light_curve(
        tmp_path, monkeypatch, generate_lc):
    """Check final writer conditionally writes PCA light curve."""
    monkeypatch.setattr(
        'exotedrf.v2.io.save_rate_cube',
        lambda cube, path, extra_header=None: None)
    cube = SimpleNamespace(
        data=np.zeros((4, 2, 3), np.float32),
        err=np.zeros((4, 2, 3), np.float32),
        dq=np.zeros((4, 2, 3), np.uint32), meta=SimpleNamespace())
    wlc = np.asarray([0.98, 1.01, 1.02, 0.99], np.float32)
    state = SimpleNamespace(cube=cube, aux={'pca_wlc': wlc})
    paths = products.output_layout({'name_tag': 'visit'}, output_dir=tmp_path)

    written = products.write_final_products(
        state, {}, {'opts': {'generate_lc': generate_lc}}, paths)

    if generate_lc:
        assert Path(written['lcestimate']).name == 'visit_lcestimate.npy'
        np.testing.assert_array_equal(
            np.load(written['lcestimate'], allow_pickle=False), wlc)
    else:
        assert 'lcestimate' not in written
        assert not paths['lcestimate'].exists()


def test_final_spectrum_retains_v1_science_metadata_and_v2_provenance(
        tmp_path, monkeypatch):
    """Check final spectrum retains v1 science metadata and v2 provenance."""
    monkeypatch.setattr(
        'exotedrf.v2.io.save_rate_cube',
        lambda cube, path, extra_header=None: None)
    meta = SimpleNamespace(
        mode='NIRISS/SOSS',
        int_times=np.asarray([60000.0]),
        extra={'header': {'TARGNAME': 'TOI-674', 'INSTRUME': 'NIRISS'}},
    )
    cube = SimpleNamespace(
        data=np.zeros((1, 2, 3), np.float32),
        err=np.zeros((1, 2, 3), np.float32),
        dq=np.zeros((1, 2, 3), np.uint32),
        meta=meta,
    )
    state = SimpleNamespace(cube=cube, aux={'spectral_products': {
        1: {
            'wave': np.asarray([1.0, 1.1, 1.2]),
            'flux': np.ones((1, 3), np.float32),
            'ferr': np.full((1, 3), 0.1, np.float32),
        },
    }})
    paths = products.output_layout({'name_tag': 'target'}, output_dir=tmp_path)

    written = products.write_final_products(
        state, {'extract_width': np.int64(34)},
        {'opts': {'extract_method': 'box', 'generate_lc': False}}, paths)

    spectrum_path = Path(written['spectra'])
    assert spectrum_path == tmp_path / 'Stage3' / \
        'TOI-674_box_spectra_fullres.fits'
    header = fits.getheader(spectrum_path, 0)
    assert header['TARGET'] == 'TOI-674'
    assert header['INST'] == 'NIRISS/SOSS'
    assert header['PIPELINE'] == 'exoTEDRF v2'
    assert header['CONTENTS'] == 'Full resolution stellar spectra'
    assert header['METHOD'] == 'box'
    assert header['WIDTH'] == 34
    assert header['OBSMODE'] == 'NIRISS/SOSS'
    assert header['OPTIMIZE']
    with fits.open(spectrum_path) as hdus:
        assert [hdu.name for hdu in hdus[1:]] == [
            'WAVE O1', 'WAVE ERR O1', 'FLUX O1', 'FLUX ERR O1', 'TIME',
            'DQ REPORT O1']
        np.testing.assert_allclose(hdus['Wave Err O1'].data,
                                   [0.05, 0.05, 0.05])
        np.testing.assert_array_equal(hdus['Time'].data, [60000.0])
        assert hdus['Wave O1'].header['UNITS'] == 'Micron'
        assert hdus['Flux O1'].header['UNITS'] == 'DN/s'
        assert hdus['Time'].header['UNITS'] == 'MJD_TDB'


def test_reusable_sidecars_follow_v1_names_and_formats(tmp_path, monkeypatch):
    """Check reusable sidecars follow v1 names and formats."""
    monkeypatch.setattr(
        'exotedrf.v2.io.save_rate_cube',
        lambda cube, path, extra_header=None: None)
    meta = SimpleNamespace(
        mode='NIRISS/SOSS', subarray='SUBSTRIP96',
        filenames=('jwvisit-seg001_nis_uncal.fits',),
        int_times=np.arange(4.),
        extra={'header': {'FILENAME': 'jwvisit-seg001_nis_uncal.fits'}})
    cube = SimpleNamespace(
        data=np.zeros((4, 3, 5), np.float32),
        err=np.ones((4, 3, 5), np.float32),
        dq=np.zeros((4, 3, 5), np.uint32), meta=meta)
    centroids = {
        'xpos': np.arange(5.),
        'ypos o1': np.full(5, 1.),
        'ypos o2': np.full(5, 2.),
    }
    state = SimpleNamespace(cube=cube, aux={
        'pca_wlc': np.arange(4.),
        'hot_pixel_map': np.eye(3, 5, dtype=bool),
        'bkg_int': np.ones((3, 5)),
        'stage3_deepframe': np.arange(15.).reshape(3, 5),
        'pca_components': np.arange(8.).reshape(2, 4),
    })
    paths = products.output_layout({'name_tag': ''}, output_dir=tmp_path)

    written = products.write_final_products(
        state, {}, {'centroids': centroids, 'order0_mask': np.eye(3, 5),
                    'opts': {'generate_lc': True}}, paths)

    assert Path(written['centroids']).name == 'jwvisit_nis_centroids.csv'
    assert Path(written['deepframe']).name == 'jwvisit_nis_deepframe.fits'
    assert Path(written['hot_pixels']).name == 'jwvisit_nis_hot_pixels.npy'
    assert Path(written['background']).name == 'jwvisit_nis_background.npy'
    assert Path(written['stability']).name == 'jwvisit_nis_stability.csv'
    assert Path(written['lcestimate']).name == 'jwvisit_nis_lcestimate.npy'
    assert Path(written['contaminant_mask']).name == 'contaminant_mask.npy'
    loaded = np.load(written['hot_pixels'], allow_pickle=False)
    np.testing.assert_array_equal(loaded, state.aux['hot_pixel_map'])
    from astropy.io import fits
    np.testing.assert_array_equal(
        fits.getdata(written['deepframe']), state.aux['stage3_deepframe'])
    from exotedrf.v2 import trace
    loaded_centroids = trace.load_centroids_csv(written['centroids'])
    np.testing.assert_array_equal(loaded_centroids['ypos o1'],
                                  centroids['ypos o1'])
    assert Path(written['stability']).read_text().splitlines()[0] == \
        'Component 1,Component 2'


def test_nirspec_stability_sidecar_uses_v1_detector_suffix(
        tmp_path, monkeypatch):
    """Check NIRSpec stability sidecar uses v1 detector suffix."""
    monkeypatch.setattr(
        'exotedrf.v2.io.save_rate_cube',
        lambda cube, path, extra_header=None: None)
    meta = SimpleNamespace(
        mode='NIRSpec/G395M', detector='NRS1',
        filenames=('jwvisit-seg001_nrs1_uncal.fits',),
        extra={'header': {'FILENAME':
                          'jwvisit-seg001_nrs1_uncal.fits'}})
    state = SimpleNamespace(
        cube=SimpleNamespace(
            data=np.zeros((2, 3, 4), np.float32),
            err=np.ones((2, 3, 4), np.float32),
            dq=np.zeros((2, 3, 4), np.uint32), meta=meta),
        aux={'pca_components': np.arange(4.).reshape(2, 2)})
    paths = products.output_layout({'name_tag': ''}, output_dir=tmp_path)

    written = products.write_final_products(
        state, {}, {'centroids': {'xpos': np.arange(4.),
                                  'ypos': np.ones(4)},
                    'opts': {'generate_lc': False}}, paths)

    assert Path(written['stability']).name == \
        'jwvisit_nrs1_stability_nrs1.csv'


def test_diagnostic_stitch_uses_o2_below_cutoff_and_o1_above():
    """Check diagnostic stitch uses o2 below cutoff and o1 above."""
    wave, flux = products._diagnostic_spectrum({
        1: {
            'wave': np.asarray([0.8, 0.9, 1.0, 1.0]),
            'flux': np.asarray([[80., 90., 100., 101.],
                                [81., 91., 101., 102.]]),
        },
        2: {
            'wave': np.asarray([0.6, 0.8, 0.9]),
            'flux': np.asarray([[60., 80., 90.],
                                [61., 81., 91.]]),
        },
    })

    np.testing.assert_array_equal(wave, [0.6, 0.8, 0.9, 1.0])
    np.testing.assert_array_equal(
        flux, [[60., 80., 90., 100.], [61., 81., 91., 101.]])


@pytest.mark.parametrize(
    ('mode', 'detector', 'input_wave', 'expected_wave'), [
        ('NIRSpec/G395M', 'NRS1',
         [2.8, 2.9, 3.4, 3.9, 4.0], [2.9, 3.4, 3.9]),
        ('NIRSpec/G395M', 'NRS2',
         [5.1, 5.0, 4.4, 3.8, 3.7], [3.8, 4.4, 5.0]),
        ('MIRI/LRS', 'MIRIMAGE',
         [13.0, 12.0, 10.0, 5.0, 4.0], [5.0, 10.0, 12.0]),
    ])
def test_single_trace_diagnostic_uses_v1_instrument_band(
        mode, detector, input_wave, expected_wave):
    """Check single trace diagnostic uses v1 instrument band."""
    input_wave = np.asarray(input_wave)
    input_flux = np.stack((input_wave * 10., input_wave * 10. + 1.))

    wave, flux = products._diagnostic_spectrum(
        {1: {'wave': input_wave, 'flux': input_flux}},
        mode=mode, detector=detector)

    np.testing.assert_array_equal(wave, expected_wave)
    np.testing.assert_array_equal(flux[0], np.asarray(expected_wave) * 10.)


def test_niriss_scatter_plot_uses_v1_two_order_layout(tmp_path, monkeypatch):
    """Check NIRISS scatter plot uses v1 two order layout."""
    captured = {}

    def save(figure, path, *, dpi=300):
        figure.canvas.draw()
        captured['dpi'] = dpi
        captured['titles'] = [axis.get_title() for axis in figure.axes]
        captured['xlabels'] = [axis.get_xlabel() for axis in figure.axes]
        captured['ylabels'] = [axis.get_ylabel() for axis in figure.axes]
        captured['line_counts'] = [len(axis.lines) for axis in figure.axes]

    monkeypatch.setattr(products, '_save_figure', save)
    integrations = np.arange(6., dtype=float)[:, None]
    spectral = {
        1: {
            'wave': np.asarray([1.2, 1.0, .9]),
            'flux': 100. + integrations * np.asarray([1., 2., 3.]),
        },
        2: {
            'wave': np.asarray([.85, .7]),
            'flux': 80. + integrations * np.asarray([2., 1.]),
        },
    }
    assert products._plot_scatter_diagnostic(
        tmp_path / 'Scatter_Plot_target.png', spectral,
        scatter=np.zeros(4), smooth=2, baseline_ints=[2, -2])

    assert captured == {
        'dpi': 150,
        'titles': ['Order 2', 'Order 1'],
        'xlabels': ['Wavelength [μm]', 'Wavelength [μm]'],
        'ylabels': ['Scatter [ppm]', 'Scatter [ppm]'],
        'line_counts': [2, 2],
    }


def test_diagnostic_writer_emits_v1_named_products_once(
        tmp_path, monkeypatch):
    """Check diagnostic writer emits v1 named products once."""
    paths = products.output_layout({'name_tag': 'toi674'},
                                   output_dir=tmp_path)
    cube = SimpleNamespace(
        data=np.zeros((2, 3, 4), np.float32),
        meta=SimpleNamespace(baseline_ints=np.asarray([1]),
                             subarray='SUBSTRIP256'),
    )
    state = SimpleNamespace(cube=cube, aux={
        'spectral_products': {1: {
            'wave': np.asarray([1., 1.1]),
            'flux': np.ones((2, 2)),
        }},
        'stage3_deepframe': np.ones((3, 4)),
    })
    calls = []

    def cost(path, trials, winners):
        calls.append(('cost', path, tuple(trials), dict(winners)))
        return True

    def spectra(these_paths, spectral_products, baseline_ints, **kwargs):
        calls.append(('spectra', spectral_products, tuple(baseline_ints)))
        return {
            'white_plot': str(these_paths['white_plot']),
            'flux_plot': str(these_paths['flux_plot']),
        }

    def centroids(path, this_state, params, ctx):
        calls.append(('centroids', path, this_state))
        return True

    def scatter(path, spectral_products, values, **kwargs):
        calls.append(('scatter', path, tuple(values), kwargs))
        return True

    monkeypatch.setattr(products, '_plot_cost_trials', cost)
    monkeypatch.setattr(products, '_plot_spectral_diagnostics', spectra)
    monkeypatch.setattr(products, '_plot_scatter_diagnostic', scatter)
    monkeypatch.setattr(products, '_plot_centroids', centroids)
    written = products.write_diagnostic_plots(
        state, {'extract_width': 32}, {'opts': {'do_plots': True}}, paths,
        [{'parameter': 'extract_width', 'value': 32, 'cost': 1.}],
        scatter=np.asarray([1., 2.]))

    assert [call[0] for call in calls] == [
        'cost', 'spectra', 'scatter', 'centroids']
    assert written == {
        'cost_plot': str(tmp_path / 'Files' / 'Cost_toi674.png'),
        'white_plot': str(tmp_path / 'Files' / 'norm_white_toi674.png'),
        'flux_plot': str(tmp_path / 'Files' / 'flux_img_toi674.png'),
        'scatter_plot': str(
            tmp_path / 'Files' / 'Scatter_Plot_toi674.png'),
        'centroid_plot': str(tmp_path / 'Stage3' / 'centroiding.png'),
    }


def test_nirspec_centroid_diagnostic_uses_detector_suffix(
        tmp_path, monkeypatch):
    """Check NIRSpec centroid diagnostic uses detector suffix."""
    paths = products.output_layout({'name_tag': ''}, output_dir=tmp_path)
    state = SimpleNamespace(
        cube=SimpleNamespace(meta=SimpleNamespace(
            mode='NIRSpec/G395M', detector='NRS2',
            baseline_ints=np.asarray([1]))),
        aux={})
    captured = []
    monkeypatch.setattr(products, '_plot_cost_trials',
                        lambda *args, **kwargs: False)
    monkeypatch.setattr(products, '_plot_spectral_diagnostics',
                        lambda *args, **kwargs: {})
    monkeypatch.setattr(products, '_plot_scatter_diagnostic',
                        lambda *args, **kwargs: False)
    monkeypatch.setattr(
        products, '_plot_centroids',
        lambda path, *args, **kwargs: captured.append(Path(path)) or True)

    written = products.write_diagnostic_plots(
        state, {'extract_width': 8}, {'opts': {'do_plots': True}}, paths, [])

    assert captured == [tmp_path / 'Stage3' / 'centroiding_nrs2.png']
    assert Path(written['centroid_plot']).name == 'centroiding_nrs2.png'


def test_diagnostic_figures_render_without_image_inspection(
        tmp_path, monkeypatch):
    """Check diagnostic figures render without image inspection."""
    saved = []

    def save(figure, path, *, dpi=300, **kwargs):
        figure.canvas.draw()
        saved.append((Path(path), dpi))

    monkeypatch.setattr(products, '_save_figure', save)
    assert products._plot_cost_trials(
        tmp_path / 'Cost_target.png', [
            {'phase': 1, 'checkpoint': 'JumpStep',
             'parameter': 'time_window', 'value': 3, 'cost': 2.},
            {'phase': 1, 'checkpoint': 'JumpStep',
             'parameter': 'time_window', 'value': 5, 'cost': 1.},
        ], {'time_window': 5})

    spectral = {1: {
        'wave': np.asarray([0.9, 1.0, 1.1]),
        'flux': np.asarray([[9., 10., 11.],
                            [9.1, 10.1, 11.1],
                            [9.2, 10.2, 11.2]]),
    }}
    paths = products.output_layout({'name_tag': 'target'},
                                   output_dir=tmp_path)
    assert set(products._plot_spectral_diagnostics(
        paths, spectral, np.asarray([1, -1]))) == {
            'white_plot', 'flux_plot'}
    assert products._plot_scatter_diagnostic(
        paths['scatter_plot'], spectral,
        np.asarray([1e-4, 2e-4, 3e-4]))

    centroids = {
        'xpos': np.arange(4, dtype=float),
        'ypos o1': np.asarray([1., 1., 1., 1.]),
        'ypos o2': np.asarray([2., 2., np.nan, np.nan]),
    }
    state = SimpleNamespace(
        cube=SimpleNamespace(meta=SimpleNamespace(subarray='SUBSTRIP256')),
        aux={'stage3_deepframe': np.arange(12.).reshape(3, 4)})
    assert products._plot_centroids(
        paths['centroid_plot'], state, {'extract_width': 2},
        {'centroids': centroids,
         'opts': {'extract_width_soss2': 1}})

    assert [path.name for path, _ in saved] == [
        'Cost_target.png', 'norm_white_target.png',
        'flux_img_target.png', 'Scatter_Plot_target.png']
    assert paths['centroid_plot'].exists()


def test_compatibility_capture_renders_full_v1_named_soss_diagnostic_set(
        tmp_path, monkeypatch):
    """Check compatibility capture renders full v1 named SOSS diagnostic set."""
    paths = products.output_layout({'name_tag': ''}, output_dir=tmp_path)
    ctx = {'opts': {'do_plots': True, 'remove_components': [2]},
           'order0_mask': np.eye(3, 5, dtype=bool)}
    capture = products.CompatibilityCapture(paths, ctx)
    image = np.arange(15., dtype=float).reshape(3, 5)
    correction = (np.arange(20., dtype=float),
                  np.arange(20., dtype=float) + 0.1)
    temporal = {
        'before_image': image,
        'after_image': image + 1,
        'before_series': np.asarray([1., 2., 1., 2.]),
        'after_series': np.asarray([1., 1.1, 1., 1.1]),
    }
    capture.records = {
        'DQInitStep': np.eye(3, 5, dtype=bool),
        'INLCorrStep': correction,
        'SuperBiasStep': image,
        'BackgroundStep_grp': temporal,
        'OneOverFStep_grp': temporal,
        'LinearityStep': correction,
        'JumpStep': np.eye(3, 5, dtype=np.uint32),
        'BackgroundStep': temporal,
        'OneOverFStep_int': temporal,
        'BadPixStep': {'deepframe': image,
                       'mask': np.eye(3, 5, dtype=bool)},
    }
    final_state = SimpleNamespace(
        cube=SimpleNamespace(meta=SimpleNamespace(frame_time=2.214)),
        aux={
            'pca_components': np.arange(12.).reshape(3, 4),
            'pca_eigvals': np.asarray([0.8, 0.15, 0.05]),
            'pca_components_reconstructed': np.arange(12.).reshape(3, 4),
            'pca_eigvals_reconstructed': np.asarray([0.7, 0.2, 0.1]),
        })

    def save(figure, path, *, dpi=300):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()
        figure.clear()

    monkeypatch.setattr(products, '_save_figure', save)
    capture.render(final_state)

    expected_stage1 = {
        'dqinitstep.png', 'inlcorrstep_1.png', 'inlcorrstep_2.png',
        'superbiasstep_1.png',
        'backgroundstep_1.png', 'backgroundstep_2.png',
        'oneoverfstep_1.png', 'oneoverfstep_2.png',
        'linearitystep_1.png', 'linearitystep_2.png', 'jump.png',
        'contminant_mask.png',
    }
    expected_stage2 = {
        'backgroundstep_1.png', 'backgroundstep_2.png',
        'oneoverfstep_1.png', 'oneoverfstep_2.png', 'badpixstep.png',
        'stability_pca.png', 'stability_pca_reconstructed.png',
    }
    assert {path.name for path in paths['stage1'].glob('*.png')} == \
        expected_stage1
    assert {path.name for path in paths['stage2'].glob('*.png')} == \
        expected_stage2


def test_nirspec_compatibility_diagnostics_use_detector_suffix(
        tmp_path, monkeypatch):
    """Check NIRSpec compatibility diagnostics use detector suffix."""
    paths = products.output_layout({'name_tag': ''}, output_dir=tmp_path)
    capture = products.CompatibilityCapture(
        paths, {'opts': {'do_plots': True, 'remove_components': [2]}})
    image = np.arange(15., dtype=float).reshape(3, 5)
    correction = (np.arange(20., dtype=float),
                  np.arange(20., dtype=float) + 0.1)
    temporal = {
        'before_image': image,
        'after_image': image + 1,
        'before_series': np.asarray([1., 2., 1., 2.]),
        'after_series': np.asarray([1., 1.1, 1., 1.1]),
    }
    capture.records = {
        'DQInitStep': np.eye(3, 5, dtype=bool),
        'INLCorrStep': correction,
        'SuperBiasStep': image,
        'OneOverFStep_grp': temporal,
        'LinearityStep': correction,
        'JumpStep': np.eye(3, 5, dtype=np.uint32),
        'OneOverFStep_int': temporal,
        'BadPixStep': {'deepframe': image,
                       'mask': np.eye(3, 5, dtype=bool)},
    }
    final_state = SimpleNamespace(
        cube=SimpleNamespace(meta=SimpleNamespace(
            mode='NIRSpec/G395M', detector='NRS1', frame_time=.902)),
        aux={
            'pca_components': np.arange(12.).reshape(3, 4),
            'pca_eigvals': np.asarray([0.8, 0.15, 0.05]),
            'pca_components_reconstructed': np.arange(12.).reshape(3, 4),
            'pca_eigvals_reconstructed': np.asarray([0.7, 0.2, 0.1]),
        })

    def touch(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def touch_outfile(*args, outfile=None, **kwargs):
        touch(outfile)

    def touch_correction(path1, path2, *args, **kwargs):
        touch(path1)
        touch(path2)

    from exotedrf import plotting as v1_plotting
    monkeypatch.setattr(v1_plotting, 'plot_saturated_pixels', touch_outfile)
    monkeypatch.setattr(v1_plotting, 'make_badpix_plot', touch_outfile)
    monkeypatch.setattr(products, '_plot_correction_fallback',
                        touch_correction)
    monkeypatch.setattr(products, '_plot_image',
                        lambda path, *args, **kwargs: touch(path))
    monkeypatch.setattr(products, '_plot_series',
                        lambda path, *args, **kwargs: touch(path))
    monkeypatch.setattr(products, '_plot_psd',
                        lambda path, *args, **kwargs: touch(path))
    monkeypatch.setattr(
        products.CompatibilityCapture, '_plot_pca',
        staticmethod(lambda path, *args, **kwargs: touch(path)))

    written = capture.render(final_state)

    expected = {
        'dqinit_plot': 'dqinitstep_nrs1.png',
        'inlcorrstep_plot_1': 'inlcorrstep_1_nrs1.png',
        'inlcorrstep_plot_2': 'inlcorrstep_2_nrs1.png',
        'superbias_plot': 'superbiasstep_1_nrs1.png',
        'OneOverFStep_grp_plot_1': 'oneoverfstep_1_nrs1.png',
        'OneOverFStep_grp_plot_2': 'oneoverfstep_2_nrs1.png',
        'linearitystep_plot_1': 'linearitystep_1_nrs1.png',
        'linearitystep_plot_2': 'linearitystep_2_nrs1.png',
        'jump_plot': 'jump_nrs1.png',
        'OneOverFStep_int_plot_1': 'oneoverfstep_1_nrs1.png',
        'OneOverFStep_int_plot_2': 'oneoverfstep_2_nrs1.png',
        'badpix_plot': 'badpixstep_nrs1.png',
        'pca_plot': 'stability_pca_nrs1.png',
        'pca_reconstructed_plot': 'stability_pca_reconstructed_nrs1.png',
    }
    assert {key: Path(value).name for key, value in written.items()} == expected


def test_compatibility_capture_reuses_adjacent_background_summaries(
        tmp_path, monkeypatch):
    """Check compatibility capture reuses adjacent background summaries."""
    paths = products.output_layout({'name_tag': ''}, output_dir=tmp_path)
    capture = products.CompatibilityCapture(
        paths, {'opts': {'do_plots': True}})
    calls = []

    def image(cube):
        calls.append(('image', cube.label))
        return np.full((2, 3), cube.value)

    def series(cube):
        calls.append(('series', cube.label))
        return np.full(4, cube.value)

    monkeypatch.setattr(products, '_representative_image', image)
    monkeypatch.setattr(products, '_diagnostic_series', series)
    cubes = [SimpleNamespace(label=label, value=index)
             for index, label in enumerate(('raw', 'background', 'oof'))]
    states = [SimpleNamespace(cube=cube) for cube in cubes]

    capture(SimpleNamespace(name='BackgroundStep_grp'), states[0], states[1])
    capture(SimpleNamespace(name='OneOverFStep_grp'), states[1], states[2])

    assert calls == [
        ('image', 'raw'), ('series', 'raw'),
        ('image', 'background'), ('series', 'background'),
        ('image', 'oof'), ('series', 'oof'),
    ]
    np.testing.assert_array_equal(
        capture.records['OneOverFStep_grp']['before_image'],
        capture.records['BackgroundStep_grp']['after_image'])
