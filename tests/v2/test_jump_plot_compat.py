"""V1-style Jump diagnostic semantics without 4-D host transfers."""

from types import SimpleNamespace

import numpy as np

from exotedrf.v2 import core, products


class _SliceOnlyArray:
    """Array proxy which fails if a caller materializes the whole cube."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.ndim = self.values.ndim
        self.requests = []

    def __getitem__(self, key):
        self.requests.append(key)
        return self.values[key]

    def __array__(self, dtype=None, copy=None):
        del dtype, copy
        raise AssertionError('full observation must not cross to host')


def _cube(*, instrument='NIRISS', nints=7, ngroups=4, dimy=3, dimx=5):
    """Return cube."""
    data = np.arange(
        nints * ngroups * dimy * dimx, dtype=np.float32).reshape(
            nints, ngroups, dimy, dimx)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    # Every sampled panel exposes all three final-state marker categories.
    groupdq[:, :, 0, 0] |= np.uint8(core.DQ_JUMP_DET)
    groupdq[:, :, 1, 1] |= np.uint8(core.DQ_DO_NOT_USE)
    pixeldq = np.zeros((dimy, dimx), dtype=np.uint32)
    pixeldq[2, 2] = core.DQ_HOT
    meta = SimpleNamespace(
        mode=f'{instrument}/TEST', segment_edges=np.asarray([3, nints]),
        segment_int_starts=np.asarray([11, 41]))
    return (SimpleNamespace(data=_SliceOnlyArray(data),
                            groupdq=_SliceOnlyArray(groupdq),
                            pixeldq=pixeldq, meta=meta),
            data, groupdq)


def _random_state_randint(seed, calls):
    """Return random state randint."""
    rng = np.random.RandomState(seed)

    def randint(*args, **kwargs):
        calls.append((args, kwargs))
        return rng.randint(*args, **kwargs)

    return randint


def test_capture_matches_v1_sampling_order_and_final_flag_semantics():
    """Check capture matches v1 sampling order and final flag semantics."""
    cube, data, _ = _cube()
    calls = []
    record = products._capture_jump_panels(
        cube, randint=_random_state_randint(1947, calls))

    expected_rng = np.random.RandomState(1947)
    files = expected_rng.randint(0, 2, 9)
    edges = np.asarray([3, 7])
    offsets = np.asarray([0, 3])
    starts = np.asarray([11, 41])
    expected = []
    for segment in files:
        local = expected_rng.randint(0, edges[segment] - offsets[segment])
        group = expected_rng.randint(1, data.shape[1])
        integration = offsets[segment] + local
        expected.append((segment, local, group, integration))

    assert calls[0] == ((0, 2, 9), {})
    assert len(calls) == 19
    assert len(record['panels']) == 9
    assert record['instrument'] == 'NIRISS'
    assert record['hot'][2, 2]
    for panel, (segment, local, group, integration) in zip(
            record['panels'], expected):
        assert panel['segment'] == segment
        assert panel['local_integration'] == local
        assert panel['integration'] == starts[segment] + local
        assert panel['group'] == group
        np.testing.assert_array_equal(
            panel['difference'],
            data[integration, group] - data[integration, group - 1])
        assert panel['jump'][0, 0]
        assert panel['dnu'][1, 1]

    # Duplicate random panels may share cached arrays.
    assert cube.data.requests
    assert all(isinstance(key, tuple) and len(key) == 2 and
               isinstance(key[1], slice) and
               key[1].stop - key[1].start == 2
               for key in cube.data.requests)
    panel_dq_requests = [key for key in cube.groupdq.requests
                         if isinstance(key, tuple)]
    assert panel_dq_requests
    assert all(len(key) == 2 and isinstance(key[0], (int, np.integer)) and
               isinstance(key[1], (int, np.integer))
               for key in panel_dq_requests)


def test_miri_rejects_dropped_group_with_v1_random_draw_sequence():
    """Check MIRI rejects dropped group with v1 random draw sequence."""
    nints, ngroups, dimy, dimx = 2, 4, 3, 5
    data = np.arange(
        nints * ngroups * dimy * dimx, dtype=np.float32).reshape(
            nints, ngroups, dimy, dimx)
    groupdq = np.zeros_like(data, dtype=np.uint8)
    groupdq[0, 1] = np.uint8(core.DQ_DO_NOT_USE)
    cube = SimpleNamespace(
        data=_SliceOnlyArray(data), groupdq=_SliceOnlyArray(groupdq),
        pixeldq=np.zeros((dimy, dimx), np.uint32),
        meta=SimpleNamespace(
            mode='MIRI/LRS', segment_edges=np.asarray([nints]),
            segment_int_starts=np.asarray([101])))
    responses = iter((np.asarray([0]), 0, 1, 2))
    calls = []

    def randint(*args):
        calls.append(args)
        return next(responses)

    record = products._capture_jump_panels(
        cube, panel_count=1, randint=randint)

    assert calls == [(0, 1, 1), (0, 2), (1, 4), (1, 4)]
    assert len(record['panels']) == 1
    assert record['panels'][0]['group'] == 2
    assert record['panels'][0]['integration'] == 101


def test_render_recreates_nine_panel_layout_with_batched_ellipses(
        tmp_path, monkeypatch):
    """Check render recreates nine panel layout with batched ellipses."""
    difference = np.arange(15., dtype=float).reshape(3, 5)
    hot = np.zeros((3, 5), bool)
    hot[0, 0] = True
    panels = []
    for index in range(9):
        jump = np.zeros((3, 5), bool)
        dnu = np.zeros((3, 5), bool)
        jump[1, 1] = True
        dnu[2, 2] = True
        panels.append({
            'difference': difference + index,
            'jump': jump,
            'dnu': dnu,
            'integration': 100 + index,
            'group': 2,
        })
    record = {'instrument': 'NIRISS', 'hot': hot,
              'panels': tuple(panels)}
    saved = {}

    def save(figure, path, *, dpi=300):
        figure.canvas.draw()
        saved.update(figure=figure, path=path, dpi=dpi)

    monkeypatch.setattr(products, '_save_figure', save)
    path = tmp_path / 'jump.png'
    assert products._plot_jump_panels(path, record)

    figure = saved['figure']
    assert saved['path'] == path
    assert saved['dpi'] == 100
    assert len(figure.axes) == 9
    first = figure.axes[0]
    assert len(first.images) == 1
    np.testing.assert_array_equal(first.images[0].get_array(), difference)
    assert first.images[0].get_clim() == (0., np.nanpercentile(difference, 85))
    assert [item.get_label() for item in first.collections] == [
        'Hot Pixel', 'Bad Pixel', 'Cosmic Ray']
    np.testing.assert_array_equal(first.collections[0].get_widths(), [21.])
    np.testing.assert_array_equal(first.collections[0].get_heights(), [3.])
    assert first.texts[0].get_text() == '(100, 2)'
    assert [text.get_text() for text in first.get_legend().get_texts()] == [
        'Hot Pixel', 'Bad Pixel', 'Cosmic Ray']


def test_capture_observer_uses_post_jump_planes_not_before_after_xor(tmp_path,
                                                                    monkeypatch):
    """Check capture observer uses post jump planes not before after xor."""
    cube, _, _ = _cube(nints=7)
    paths = products.output_layout({'name_tag': ''}, output_dir=tmp_path)
    capture = products.CompatibilityCapture(
        paths, {'opts': {'do_plots': True}})
    monkeypatch.setattr(
        products.np.random, 'randint',
        _random_state_randint(20, []))
    state = SimpleNamespace(cube=cube)
    capture(SimpleNamespace(name='JumpStep'), state, state)

    record = capture.records['JumpStep']
    assert isinstance(record, dict)
    assert len(record['panels']) == 9
    assert all(panel['jump'][0, 0] for panel in record['panels'])
