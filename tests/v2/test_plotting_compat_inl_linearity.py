"""Focused parity tests for compact INL/linearity plot capture."""

from pathlib import Path
from types import SimpleNamespace

import bottleneck as bn
import numpy as np

from exotedrf.v2 import plotting_compat, products


def _legacy_bin(data, npix):
    """Return legacy bin."""
    flat = data.flatten()
    remainder = flat.size % npix
    return np.nanmedian(
        flat[remainder:].reshape((-1, npix)), axis=1)


def _legacy_inl(before, after, npix):
    """Return legacy inverse linearity."""
    def residuals(data):
        groups = np.arange(data.shape[1], dtype=np.float32)
        centered = groups - np.mean(groups)
        slope = np.tensordot(
            data, centered, axes=([1], [0])) / np.sum(centered**2)
        intercept = np.mean(data, axis=1) - slope * np.mean(groups)
        model = (intercept[:, None, :, :] +
                 slope[:, None, :, :] * groups[None, :, None, None])
        return data - model

    pre = before[10:20]
    post = after[10:20]
    rpre = residuals(pre)
    rpost = residuals(post)
    dpre = pre.flatten()
    rpre = rpre.flatten()
    order = np.argsort(dpre)
    dpre = dpre[order]
    rpre = rpre[order]
    mask = (dpre >= 10000) & (dpre <= 20000)
    dpre = dpre[mask]
    rpre = rpre[mask]
    dpost = post.flatten()[order][mask]
    rpost = rpost.flatten()[order][mask]
    return {
        'data_pre': _legacy_bin(dpre, npix),
        'res_pre': _legacy_bin(rpre, npix),
        'data_post': _legacy_bin(dpost, npix),
        'res_post': _legacy_bin(rpost, npix),
    }


def test_capture_inl_is_v1_exact_and_uses_only_first_segment():
    """Check capture inverse linearity is v1 exact and uses only first segment."""
    nint, ngroup, dimy, dimx = 16, 4, 3, 5
    integration = np.arange(nint, dtype=np.float32)[:, None, None, None]
    group = np.arange(ngroup, dtype=np.float32)[None, :, None, None]
    pixel = np.arange(dimy * dimx, dtype=np.float32).reshape(
        1, 1, dimy, dimx)
    before = 11000 + 17 * integration + 900 * group + 3 * pixel
    after = before - 0.4 * np.sin(before / 127.0).astype(np.float32)

    # Anything after the first segment must be irrelevant, just as in v1.
    after[13:] = -1e20
    actual = plotting_compat.capture_inl(
        before, after, first_segment_stop=13, npix_to_bin=7)
    expected = _legacy_inl(before[:13], after[:13], 7)

    assert actual['npix_to_bin'] == 7
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)
        assert not np.shares_memory(actual[key], before)
        assert not np.shares_memory(actual[key], after)


def test_render_inl_delegates_to_unchanged_v1_array_plotters(
        tmp_path, monkeypatch):
    """Check render inverse linearity delegates to unchanged v1 array plotters."""
    from exotedrf import plotting as v1_plotting

    record = {
        'data_pre': np.asarray([1., 2., 3.]),
        'res_pre': np.asarray([0.1, 0.2, 0.3]),
        'data_post': np.asarray([0.9, 1.9, 2.9]),
        'res_post': np.asarray([0.01, 0.02, 0.03]),
    }
    calls = []

    def plot1(data_pre, res_pre, data_post, **kwargs):
        calls.append(('plot1', data_pre, res_pre, data_post, kwargs))
        Path(kwargs['outfile']).touch()

    def plot2(data_pre, res_pre, data_post, res_post, **kwargs):
        calls.append(
            ('plot2', data_pre, res_pre, data_post, res_post, kwargs))
        Path(kwargs['outfile']).touch()

    monkeypatch.setattr(v1_plotting, 'make_inl_plot', plot1)
    monkeypatch.setattr(v1_plotting, 'make_inl_plot2', plot2)
    path1 = tmp_path / 'inlcorrstep_1.png'
    path2 = tmp_path / 'inlcorrstep_2.png'
    assert plotting_compat.render_inl(record, path1, path2) == (path1, path2)

    assert [call[0] for call in calls] == ['plot1', 'plot2']
    np.testing.assert_array_equal(calls[0][1], record['data_pre'])
    np.testing.assert_array_equal(calls[0][2], record['res_pre'])
    np.testing.assert_array_equal(calls[0][3], record['data_post'])
    assert calls[0][4] == {'outfile': str(path1), 'show_plot': False}
    np.testing.assert_array_equal(calls[1][4], record['res_post'])
    assert calls[1][5] == {'outfile': str(path2), 'show_plot': False}


def _legacy_linearity(before, after, randint):
    """Return legacy linearity."""
    cube = after
    old_cube = before
    nint, ngroup, _dimy, _dimx = cube.shape

    stack = bn.nanmedian(
        cube[randint(0, nint, 25), -1], axis=0)
    bright = np.where(
        (stack >= np.nanpercentile(stack, 80)) &
        (stack < np.nanpercentile(stack, 99)))
    new_diffs = np.zeros((ngroup - 1, len(bright[0])))
    old_diffs = np.zeros((ngroup - 1, len(bright[0])))
    for index in range(min(10000, len(bright[0]))):
        ypos, xpos = bright[0][index], bright[1][index]
        integration = randint(0, nint)
        new_diffs[:, index] = np.diff(
            cube[integration, :, ypos, xpos])
        old_diffs[:, index] = np.diff(
            old_cube[integration, :, ypos, xpos])
    new_med = np.mean(new_diffs, axis=1)
    old_med = np.mean(old_diffs, axis=1)

    stack = bn.nanmedian(cube[:, -1], axis=0)
    bright = np.where(
        (stack >= np.nanpercentile(stack, 80)) &
        (stack < np.nanpercentile(stack, 99)))
    selected = randint(0, len(bright[0]), 1000)
    ypix = bright[0][selected]
    xpix = bright[1][selected]
    integrations = randint(0, nint, 1000)
    old_residuals = np.zeros((1000, ngroup))
    new_residuals = np.zeros((1000, ngroup))
    for index in range(1000):
        old_ramp = old_cube[
            integrations[index], :, ypix[index], xpix[index]]
        old_line = np.linspace(
            np.min(old_ramp), np.max(old_ramp), ngroup)
        old_residuals[index] = (
            (old_ramp - old_line) / np.max(old_ramp) * 100)
        new_ramp = cube[
            integrations[index], :, ypix[index], xpix[index]]
        new_line = np.linspace(
            np.min(new_ramp), np.max(new_ramp), ngroup)
        new_residuals[index] = (
            (new_ramp - new_line) / np.max(new_ramp) * 100)

    return {
        'plot1_before': old_med - np.mean(old_med),
        'plot1_after': new_med - np.mean(new_med),
        'plot2_before': np.nanmedian(old_residuals, axis=0),
        'plot2_after': np.nanmedian(new_residuals, axis=0),
    }


def _legacy_linearity_miri(before, after, groupdq, randint):
    """Literal MIRI branches of v1's two linearity plotting routines."""
    cube = after
    old_cube = before
    nint, ngroup, dimy, dimx = cube.shape
    assert dimx >= 60

    dropped = []
    for group_index in range(ngroup):
        dnu = (groupdq[0, group_index] & np.uint8(1)) != 0
        if np.sum(dnu) == dimy * dimx:
            dropped.append(group_index)
    good_groups = np.delete(np.arange(ngroup), dropped)

    stack = bn.nanmedian(cube[randint(0, nint, 25), -1], axis=0)
    stack = stack[:, 20:60]
    bright = np.where(
        (stack >= np.nanpercentile(stack, 80)) &
        (stack < np.nanpercentile(stack, 99)))
    new_diffs = np.zeros((ngroup - 1, len(bright[0])))
    old_diffs = np.zeros((ngroup - 1, len(bright[0])))
    for index in range(min(10_000, len(bright[0]))):
        ypos = bright[0][index]
        # Convert the 20:60 cutout coordinate back to detector x.
        xpos = bright[1][index] + 20
        integration = randint(0, nint)
        new_diffs[:, index] = np.diff(
            cube[integration, :, ypos, xpos])
        old_diffs[:, index] = np.diff(
            old_cube[integration, :, ypos, xpos])
    good_differences = good_groups - 1
    new_med = np.mean(new_diffs, axis=1)[good_differences]
    old_med = np.mean(old_diffs, axis=1)[good_differences]
    plot1_before = old_med - np.mean(old_med)
    plot1_after = new_med - np.mean(new_med)
    difference_locs = np.arange(ngroup - 1)[good_differences]
    difference_labels = np.asarray([
        f'{index + 2}-{index + 1}' for index in range(ngroup - 1)
    ])[good_differences]

    stack = bn.nanmedian(cube[:, -1], axis=0)[:, 20:60]
    bright = np.where(
        (stack >= np.nanpercentile(stack, 80)) &
        (stack < np.nanpercentile(stack, 99)))
    selected = randint(0, len(bright[0]), 1000)
    ypix = bright[0][selected]
    xpix = bright[1][selected] + 20
    integrations = randint(0, nint, 1000)
    old_residuals = np.zeros((1000, ngroup))
    new_residuals = np.zeros((1000, ngroup))
    for index in range(1000):
        old_ramp = old_cube[
            integrations[index], good_groups, ypix[index], xpix[index]]
        old_line = np.linspace(
            np.min(old_ramp), np.max(old_ramp), len(good_groups))
        old_residuals[index, good_groups] = (
            (old_ramp - old_line) / np.max(old_ramp) * 100)
        new_ramp = cube[
            integrations[index], good_groups, ypix[index], xpix[index]]
        new_line = np.linspace(
            np.min(new_ramp), np.max(new_ramp), len(good_groups))
        new_residuals[index, good_groups] = (
            (new_ramp - new_line) / np.max(new_ramp) * 100)

    return {
        'plot1_locs': difference_locs,
        'plot1_labels': difference_labels,
        'plot1_before': plot1_before,
        'plot1_after': plot1_after,
        'plot2_x': (np.arange(ngroup) + 1)[good_groups],
        'plot2_tick_locs': np.arange(ngroup)[good_groups],
        'plot2_tick_labels': (np.arange(ngroup) + 1).astype(str)[good_groups],
        'plot2_before': np.nanmedian(old_residuals, axis=0)[good_groups],
        'plot2_after': np.nanmedian(new_residuals, axis=0)[good_groups],
    }


def test_capture_linearity_matches_v1_rng_order_and_quantities():
    """Check capture linearity matches v1 rng order and quantities."""
    nint, ngroup, dimy, dimx = 15, 5, 6, 8
    integration = np.arange(nint, dtype=np.float32)[:, None, None, None]
    group = np.arange(ngroup, dtype=np.float32)[None, :, None, None]
    pixel = np.arange(dimy * dimx, dtype=np.float32).reshape(
        1, 1, dimy, dimx)
    before = (1000 + 0.5 * pixel + 2 * integration +
              200 * group + 1.7 * group**2).astype(np.float32)
    after = (1000 + 0.5 * pixel + 2 * integration +
             200 * group + 0.3 * group**2).astype(np.float32)

    rng_actual = np.random.RandomState(738)
    actual = plotting_compat.capture_linearity(
        before, after, first_segment_stop=12,
        randint=rng_actual.randint)
    rng_expected = np.random.RandomState(738)
    expected = _legacy_linearity(
        before[:12], after[:12], rng_expected.randint)

    np.testing.assert_array_equal(
        actual['plot1']['before'], expected['plot1_before'])
    np.testing.assert_array_equal(
        actual['plot1']['after'], expected['plot1_after'])
    np.testing.assert_array_equal(
        actual['plot2']['before'], expected['plot2_before'])
    np.testing.assert_array_equal(
        actual['plot2']['after'], expected['plot2_after'])
    assert actual['plot1']['tick_labels'].tolist() == [
        '2-1', '3-2', '4-3', '5-4']
    np.testing.assert_array_equal(actual['plot2']['tick_locs'], np.arange(5))
    assert actual['plot2']['tick_labels'].tolist() == [
        '1', '2', '3', '4', '5']


def test_render_linearity_writes_v1_named_figures(tmp_path):
    """Check render linearity writes v1 named figures."""
    record = {
        'plot1': {
            'locs': np.arange(3),
            'tick_locs': np.arange(3),
            'tick_labels': np.asarray(['2-1', '3-2', '4-3']),
            'after': np.asarray([-0.1, 0.0, 0.1]),
            'before': np.asarray([-0.2, 0.0, 0.2]),
            'ylim': (-0.22, 0.22),
        },
        'plot2': {
            'x': np.arange(1, 5),
            'tick_locs': np.arange(4),
            'tick_labels': np.asarray(['1', '2', '3', '4']),
            'before': np.asarray([0., 0.2, -0.1, 0.]),
            'after': np.asarray([0., 0.1, -0.05, 0.]),
        },
    }
    path1 = tmp_path / 'linearitystep_1.png'
    path2 = tmp_path / 'linearitystep_2.png'
    assert plotting_compat.render_linearity(record, path1, path2) == (
        path1, path2)
    assert path1.stat().st_size > 0
    assert path2.stat().st_size > 0


def test_capture_linearity_miri_matches_v1_dropped_group_quantities():
    """Check capture linearity MIRI matches v1 dropped group quantities."""
    nint, ngroup, dimy, dimx = 14, 7, 6, 72
    integration = np.arange(nint, dtype=np.float32)[:, None, None, None]
    group = np.arange(ngroup, dtype=np.float32)[None, :, None, None]
    ypos = np.arange(dimy, dtype=np.float32)[None, None, :, None]
    xpos = np.arange(dimx, dtype=np.float32)[None, None, None, :]
    before = (900 + 3 * integration + 2 * ypos + .7 * xpos +
              group * (120 + .8 * xpos) +
              group**2 * (2.5 + .03 * ypos)).astype(np.float32)
    after = (900 + 3 * integration + 2 * ypos + .7 * xpos +
             group * (120 + .8 * xpos) +
             group**2 * (.4 + .01 * ypos)).astype(np.float32)

    groupdq = np.zeros(before.shape, dtype=np.uint8)
    dropped = np.asarray([0, 2, 6])
    groupdq[0, dropped] = np.uint8(1)
    # Only the first integration defines a dropped group in v1.
    groupdq[1:, 1] = np.uint8(1)
    groupdq[0, 4, :3] = np.uint8(1)

    first_stop = 11
    rng_actual = np.random.RandomState(947)
    actual = plotting_compat.capture_linearity(
        before, after, first_segment_stop=first_stop,
        instrument='MIRI', groupdq=groupdq,
        randint=rng_actual.randint)
    rng_expected = np.random.RandomState(947)
    expected = _legacy_linearity_miri(
        before[:first_stop], after[:first_stop], groupdq[:first_stop],
        rng_expected.randint)

    np.testing.assert_array_equal(
        actual['plot1']['locs'], expected['plot1_locs'])
    np.testing.assert_array_equal(
        actual['plot1']['tick_locs'], expected['plot1_locs'])
    np.testing.assert_array_equal(
        actual['plot1']['tick_labels'], expected['plot1_labels'])
    np.testing.assert_array_equal(
        actual['plot1']['before'], expected['plot1_before'])
    np.testing.assert_array_equal(
        actual['plot1']['after'], expected['plot1_after'])
    np.testing.assert_array_equal(actual['plot2']['x'], expected['plot2_x'])
    np.testing.assert_array_equal(
        actual['plot2']['tick_locs'], expected['plot2_tick_locs'])
    np.testing.assert_array_equal(
        actual['plot2']['tick_labels'], expected['plot2_tick_labels'])
    np.testing.assert_array_equal(
        actual['plot2']['before'], expected['plot2_before'])
    np.testing.assert_array_equal(
        actual['plot2']['after'], expected['plot2_after'])


def test_compatibility_capture_passes_unsliced_miri_groupdq(
        monkeypatch):
    """Check compatibility capture passes unsliced MIRI GROUPDQ."""
    capture = products.CompatibilityCapture(
        {}, {'opts': {'do_plots': True}})
    shape = (5, 4, 2, 72)
    before_data = np.zeros(shape, dtype=np.float32)
    after_data = np.ones(shape, dtype=np.float32)
    after_groupdq = np.arange(np.prod(shape), dtype=np.uint32).reshape(
        shape).astype(np.uint8)
    meta = SimpleNamespace(
        mode='MIRI/LRS', segment_edges=np.asarray([3, 5]),
        segment_int_starts=np.asarray([1, 4]))
    before = SimpleNamespace(cube=SimpleNamespace(
        data=before_data, groupdq=np.zeros(shape, dtype=np.uint8), meta=meta))
    after = SimpleNamespace(cube=SimpleNamespace(
        data=after_data, groupdq=after_groupdq, meta=meta))
    observed = {}

    def capture_linearity(before_arg, after_arg, first_stop, **kwargs):
        observed.update(
            before=before_arg, after=after_arg, first_stop=first_stop,
            **kwargs)
        return {'captured': True}

    monkeypatch.setattr(plotting_compat, 'capture_linearity',
                        capture_linearity)
    capture(SimpleNamespace(name='LinearityStep'), before, after)

    assert observed['before'] is before_data
    assert observed['after'] is after_data
    assert observed['first_stop'] == 3
    assert observed['instrument'] == 'MIRI'
    assert observed['groupdq'] is after_groupdq
    assert np.shares_memory(observed['groupdq'], after_groupdq)


def test_batched_lombscargle_matches_astropy_psd_normalization():
    """Check batched lombscargle matches astropy psd normalization."""
    from astropy.timeseries import LombScargle

    rng = np.random.default_rng(91)
    count = 512
    pixels = np.arange(count)
    timestamps = (1e-5 * (pixels + 1) +
                  1.2e-4 * (pixels // 16))
    values = rng.normal(size=(3, count))
    valid = rng.random((3, count)) > .2
    frequencies = np.logspace(np.log10(1 / 5.494),
                              np.log10(80_000), 23)

    actual = plotting_compat.batched_lombscargle_psd(
        timestamps, values, valid, frequencies)
    expected = np.stack([
        LombScargle(timestamps[mask], row[mask]).power(
            frequencies, normalization='psd', method='chi2')
        for row, mask in zip(values, valid)
    ])
    # XLA and NumPy use different correctly-rounded sin/matmul reductions.
    np.testing.assert_allclose(
        actual[:, 1:], expected[:, 1:], rtol=5e-6, atol=2e-8)
    np.testing.assert_allclose(
        actual[:, 0], expected[:, 0], rtol=7e-5, atol=2e-8)
