"""Check diagnostic content and plotting options against v1."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from exotedrf import plotting as v1_plotting
from exotedrf.v2 import products


def _paths(tmp_path):
    """Return paths."""
    return products.output_layout({'name_tag': ''}, output_dir=tmp_path)


def _touch_outfile(kwargs):
    """Return touch outfile."""
    path = Path(kwargs['outfile'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _assert_plot_kwargs(actual, outfile, **expected):
    """Compare plot semantics without requiring Path versus str identity."""
    actual = dict(actual)
    assert Path(actual.pop('outfile')) == Path(outfile)
    assert actual == expected


def test_dqinit_and_order0_delegate_to_the_original_v1_plotters(
        tmp_path, monkeypatch):
    """The two detector maps retain v1's fixed limits and panel geometry."""
    saturated = np.asarray([[False, True, False], [True, False, False]])
    order0 = np.asarray([[True, False, False], [False, True, False]])
    f277w = np.asarray([[-1., -0.5, 0.], [0.25, 0.5, 1.]])
    calls = {}

    def saturated_plot(pixels, **kwargs):
        calls['saturated'] = (np.asarray(pixels), kwargs)
        _touch_outfile(kwargs)

    def order0_plot(mask, exposure, **kwargs):
        calls['order0'] = (np.asarray(mask), np.asarray(exposure), kwargs)
        _touch_outfile(kwargs)

    monkeypatch.setattr(v1_plotting, 'plot_saturated_pixels', saturated_plot)
    monkeypatch.setattr(v1_plotting, 'make_order0_mask_plot', order0_plot)

    paths = _paths(tmp_path)
    capture = products.CompatibilityCapture(
        paths, {'opts': {'do_plots': True},
                'order0_mask': order0, 'f277w': f277w})
    capture.records['DQInitStep'] = saturated
    state = SimpleNamespace(
        cube=SimpleNamespace(meta=SimpleNamespace(frame_time=5.494)), aux={})
    capture.render(state)

    np.testing.assert_array_equal(calls['saturated'][0], saturated)
    _assert_plot_kwargs(
        calls['saturated'][1], paths['stage1'] / 'dqinitstep.png',
        show_plot=False)
    np.testing.assert_array_equal(calls['order0'][0], order0)
    np.testing.assert_array_equal(calls['order0'][1], f277w)
    _assert_plot_kwargs(
        calls['order0'][2], paths['stage1'] / 'contminant_mask.png',
        show_plot=False)


def test_badpix_delegates_distinct_hot_negative_and_other_categories(
        tmp_path, monkeypatch):
    """A combined red mask loses the meaning encoded by v1's three colors."""
    deepframe = np.arange(20., dtype=float).reshape(4, 5)
    deepframe[1, 3] = np.nan
    hot = np.zeros((4, 5), dtype=bool)
    negative = np.zeros_like(hot)
    other = np.zeros_like(hot)
    hot[0, 1] = True
    negative[1, 3] = True
    other[2, 4] = True
    calls = []

    def badpix_plot(deep, hotpix, nanpix, otherpix, **kwargs):
        calls.append((np.asarray(deep), hotpix, nanpix, otherpix, kwargs))
        _touch_outfile(kwargs)

    monkeypatch.setattr(v1_plotting, 'make_badpix_plot', badpix_plot)

    paths = _paths(tmp_path)
    capture = products.CompatibilityCapture(
        paths, {'opts': {'do_plots': True}})
    capture.records['BadPixStep'] = {
        'deepframe': deepframe,
        'hotpix': hot,
        'nanpix': negative,
        'otherpix': other,
    }
    state = SimpleNamespace(
        cube=SimpleNamespace(meta=SimpleNamespace(frame_time=5.494)), aux={})
    capture.render(state)

    assert len(calls) == 1
    plotted, hotpix, nanpix, otherpix, kwargs = calls[0]
    expected_deep = np.nan_to_num(deepframe, nan=0.)
    np.testing.assert_array_equal(plotted, expected_deep)
    for actual, expected in ((hotpix, hot), (nanpix, negative),
                             (otherpix, other)):
        actual_mask = np.zeros_like(expected)
        actual_mask[actual] = True
        np.testing.assert_array_equal(actual_mask, expected)
    _assert_plot_kwargs(
        kwargs, paths['stage2'] / 'badpixstep.png', show_plot=False,
        miri_scale=False)


def test_pca_delegates_components_variance_and_spatial_projection_panels(
        tmp_path, monkeypatch):
    """The legacy PCA diagnostic is N components by two scientific panels."""
    components = np.arange(15., dtype=float).reshape(3, 5)
    variance = np.asarray([0.8, 0.15, 0.05])
    projections = np.arange(72., dtype=float).reshape(3, 4, 6)
    reconstructed_components = components + 100.
    reconstructed_variance = variance[::-1]
    reconstructed_projections = projections + 1000.
    calls = []

    def pca_plot(pcs, var, spatial, **kwargs):
        calls.append((np.asarray(pcs), np.asarray(var), np.asarray(spatial),
                      kwargs))
        _touch_outfile(kwargs)

    monkeypatch.setattr(v1_plotting, 'make_pca_plot', pca_plot)

    paths = _paths(tmp_path)
    capture = products.CompatibilityCapture(
        paths, {'opts': {'do_plots': True, 'remove_components': [2]}})
    state = SimpleNamespace(
        cube=SimpleNamespace(meta=SimpleNamespace(frame_time=5.494)),
        aux={
            'pca_components': components,
            'pca_eigvals': variance,
            'pca_projections': projections,
            'pca_components_reconstructed': reconstructed_components,
            'pca_eigvals_reconstructed': reconstructed_variance,
            'pca_projections_reconstructed': reconstructed_projections,
        })
    capture.render(state)

    assert len(calls) == 2
    expected = (
        (components, variance, projections,
         paths['stage2'] / 'stability_pca.png'),
        (reconstructed_components, reconstructed_variance,
         reconstructed_projections,
         paths['stage2'] / 'stability_pca_reconstructed.png'),
    )
    for call, (pcs, var, spatial, outfile) in zip(calls, expected):
        np.testing.assert_array_equal(call[0], pcs)
        np.testing.assert_array_equal(call[1], var)
        np.testing.assert_array_equal(call[2], spatial)
        _assert_plot_kwargs(call[3], outfile, show_plot=False)


def test_centroid_diagnostic_delegates_v1_niriss_aperture_style(
        tmp_path, monkeypatch):
    """Centroid traces use v1's red dashed apertures and order-width rules."""
    deepframe = np.arange(24., dtype=float).reshape(4, 6)
    xpos = np.arange(6., dtype=float)
    ypos1 = np.linspace(1., 2., 6)
    ypos2 = np.linspace(2., 3., 6)
    ypos3 = np.linspace(0.5, 1.5, 6)
    centroids = {
        'xpos': xpos,
        'ypos o1': ypos1,
        'ypos o2': ypos2,
        'ypos o3': ypos3,
    }
    calls = []

    def centroid_plot(image, traces, instrument, **kwargs):
        calls.append((np.asarray(image), traces, instrument, kwargs))
        _touch_outfile(kwargs)

    monkeypatch.setattr(v1_plotting, 'make_centroiding_plot', centroid_plot)

    paths = _paths(tmp_path)
    state = SimpleNamespace(
        cube=SimpleNamespace(meta=SimpleNamespace(
            subarray='SUBSTRIP256', mode='NIRISS/SOSS')),
        aux={'stage3_deepframe': deepframe})
    ctx = {
        'centroids': centroids,
        'opts': {'do_plots': True, 'extract_width_soss2': 17},
    }
    assert products._plot_centroids(
        paths['centroid_plot'], state, {'extract_width': 31}, ctx)

    assert len(calls) == 1
    plotted, traces, instrument, kwargs = calls[0]
    np.testing.assert_array_equal(plotted, deepframe)
    assert instrument == 'NIRISS'
    assert len(traces) == 3
    np.testing.assert_array_equal(traces[0][0], xpos)
    np.testing.assert_array_equal(traces[0][1], ypos1)
    np.testing.assert_array_equal(traces[1][0], xpos)
    np.testing.assert_array_equal(traces[1][1], ypos2)
    np.testing.assert_array_equal(traces[2][0], xpos)
    np.testing.assert_array_equal(traces[2][1], ypos3)
    _assert_plot_kwargs(
        kwargs, paths['centroid_plot'], show_plot=False,
        extract_width=31, extract_width_soss2=17)
