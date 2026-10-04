"""Check batched candidate scores against sequential calculations."""

import numpy as np
import pytest

from exotedrf.v2 import core, stages
from exotedrf.v2.core import ObsMeta, RampCube, RateCube
from exotedrf.v2.pipeline import PipelineState


def _niriss_meta(nints, edges=None, starts=None, ngroups=3, dimx=16):
    """Return NIRISS meta."""
    edges = np.asarray(edges if edges is not None else [nints])
    extra = {} if starts is None else {'segment_int_starts': tuple(starts)}
    return ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP96', 2.214, ngroups,
                   np.arange(nints, dtype=float), np.array([nints]), edges,
                   tuple(f'seg{i}.fits' for i in range(len(edges))), extra)


def _assert_batched_matches_loop(scorer, parameter, candidates, params,
                                 monkeypatch):
    """Assert the vmap-batched axis reproduces the sequential loop."""
    monkeypatch.setattr(stages, 'SCORER_BATCH_AXIS', False)
    scorer._result_cache.clear()
    loop = scorer.evaluate_candidates(parameter, list(candidates), params)

    monkeypatch.setattr(stages, 'SCORER_BATCH_AXIS', True)
    scorer._result_cache.clear()
    batched = scorer.evaluate_candidates(parameter, list(candidates), params)

    assert len(batched) == len(loop) == len(candidates)
    for (b_cost, b_scatter, b_dur), (l_cost, l_scatter, l_dur) in zip(
            batched, loop):
        np.testing.assert_allclose(b_cost, l_cost, rtol=1e-5, atol=1e-7)
        assert np.asarray(b_scatter).shape == np.asarray(l_scatter).shape
        np.testing.assert_allclose(
            b_scatter, l_scatter, rtol=1e-5, atol=1e-7, equal_nan=True)
        assert b_dur >= 0. and l_dur >= 0.
    return batched, loop


def test_oneoverf_group_scorer_batched_matches_loop(monkeypatch):
    """Check 1/f correction group scorer batched matches loop."""
    rng = np.random.default_rng(0x135282d)
    nints, ngroups, dimy, dimx = 7, 3, 10, 16
    data = rng.normal(100., .3, (nints, ngroups, dimy, dimx)).astype(
        np.float32)
    groupdq = np.zeros_like(data, np.uint8)
    pixeldq = np.zeros((dimy, dimx), np.uint32)
    cube = RampCube(data, groupdq, pixeldq,
                    _niriss_meta(nints, ngroups=ngroups, dimx=dimx))
    state = PipelineState(cube)
    centroids = {'ypos o1': np.linspace(4.5, 5.5, dimx)}
    ctx = {
        'opts': {'oof_method': 'scale-achromatic',
                 'extract_width_soss2': None,
                 'wave_range': None, 'w1': 1., 'w2': 1.},
        'centroids': centroids,
        'soss_timeseries': np.linspace(.99, 1.01, nints).astype(np.float32),
        'soss_timeseries_o2': None,
        'outlier_mask': None,
        'order0_mask': None,
        'waves': {1: np.linspace(.9, 2., dimx)},
    }
    params = {'soss_inner_mask_width': 5, 'soss_outer_mask_width': 10,
             'extract_width': 4}
    scorer = stages.prepare_oneoverf_grp_scorer(state, params, ctx)
    assert scorer is not None

    _assert_batched_matches_loop(
        scorer, 'soss_inner_mask_width', [3., 4., 5., 6., 7.], params,
        monkeypatch)


def test_jump_scorer_batched_matches_loop(monkeypatch):
    """Check jump scorer batched matches loop."""
    rng = np.random.default_rng(0x123041a)
    nints, ngroups, dimy, dimx = 9, 3, 10, 12
    data = rng.normal(100., .2, (nints, ngroups, dimy, dimx)).astype(
        np.float32)
    data[4, -1, 5, 6] += 25.
    data[6, -1, 4, 8] += 15.
    groupdq = np.zeros_like(data, np.uint8)
    cube = RampCube(
        data, groupdq, np.zeros((dimy, dimx), np.uint32),
        _niriss_meta(nints, ngroups=ngroups, dimx=dimx))
    state = PipelineState(cube)
    ctx = {
        'opts': {'flag_up_ramp': False, 'flag_in_time': True,
                 'extract_width_soss2': None,
                 'wave_range': None, 'w1': 1., 'w2': 1.},
        'centroids': {'ypos o1': np.linspace(4.5, 5.5, dimx)},
        'waves': {1: np.linspace(.9, 2., dimx)},
    }
    params = {'time_jump_threshold': 7., 'time_window': 5,
             'extract_width': 4.}
    scorer = stages.prepare_jump_scorer(state, params, ctx)
    assert scorer is not None

    _assert_batched_matches_loop(
        scorer, 'time_jump_threshold', [4., 5., 6., 7., 8., 9.], params,
        monkeypatch)


def test_badpix_scorer_batched_matches_loop(monkeypatch):
    """Check bad-pixel correction scorer batched matches loop."""
    rng = np.random.default_rng(0x1352770)
    nints, dimy, dimx = 13, 12, 24
    data = rng.normal(100., .25, (nints, dimy, dimx)).astype(np.float32)
    data[5, 4, 11] += 20.
    data[8, 5, 16] -= 15.
    err = np.ones_like(data)
    dq = np.zeros_like(data, np.uint32)
    dq[10, 3, 8] = core.DQ_HOT
    dq[7, 6, 14] = core.DQ_SATURATED
    cube = RateCube(data, err, dq, _niriss_meta(nints, dimx=dimx))
    state = PipelineState(cube)
    ctx = {
        'opts': {'extract_width_soss2': None, 'wave_range': None,
                 'w1': 1., 'w2': 1.},
        'centroids': {
            'ypos o1': np.linspace(4.5, 5.5, dimx),
            'ypos o2': np.concatenate([
                np.linspace(8., 9., 17), np.full(dimx - 17, np.nan)]),
        },
        'waves': {1: np.linspace(.9, 2., dimx),
                  2: np.linspace(.6, 1., dimx)},
    }
    params = {
        'space_outlier_threshold': 8., 'time_outlier_threshold': 8.,
        'box_size': 2, 'window_size': 3, 'extract_width': 4.,
    }
    monkeypatch.setenv('EXOTEDRF_V2_BADPIX_COLUMN_CHUNK', '6')
    scorer = stages.prepare_badpix_scorer(state, params, ctx)
    assert scorer is not None

    _assert_batched_matches_loop(
        scorer, 'time_outlier_threshold', [4., 6., 8., 10., 12.], params,
        monkeypatch)


def _nirspec_meta(nints, ngroups, edges, dimx):
    """Return NIRSpec meta."""
    return ObsMeta(
        mode='NIRSpec/G395H', detector='NRS1', subarray='SUB2048',
        frame_time=0.902, ngroups=ngroups,
        int_times=60000. + np.arange(nints) * 1e-4,
        baseline_ints=np.asarray([4]),
        segment_edges=np.asarray(edges),
        filenames=('seg001_uncal.fits',),
        extra={
            'header': {
                'EXP_TYPE': 'NRS_BRIGHTOBJ', 'GRATING': 'G395H',
                'NFRAMES': 1, 'TFRAME': 0.902, 'TGROUP': 0.902,
                'NINTS': nints,
            },
            'segment_headers': ({
                'EXP_TYPE': 'NRS_BRIGHTOBJ', 'GRATING': 'G395H',
                'NFRAMES': 1, 'TFRAME': 0.902, 'TGROUP': 0.902,
                'NINTS': nints,
            },),
            'segment_int_starts': (1,),
        },
    )


def test_nirspec_oneoverf_scorer_batched_matches_loop(monkeypatch):
    """Check NIRSpec 1/f correction scorer batched matches loop."""
    rng = np.random.default_rng(0x135282d)
    nints, ngroups, dimy, dimx = 6, 3, 10, 16
    data = rng.normal(100., .3, (nints, ngroups, dimy, dimx)).astype(
        np.float32)
    groupdq = np.zeros_like(data, np.uint8)
    pixeldq = np.zeros((dimy, dimx), np.uint32)
    cube = RampCube(
        data, groupdq, pixeldq, _nirspec_meta(nints, ngroups, [nints], dimx))
    state = PipelineState(cube)
    ctx = {
        'opts': {'oof_method': 'median'},
        'centroids': {
            'xpos': np.arange(dimx, dtype=float),
            'ypos': np.full(dimx, 5.5, dtype=float),
        },
        'outlier_mask': None,
    }
    params = {'nirspec_mask_width': 4, 'extract_width': 4.}
    scorer = stages.prepare_nirspec_oneoverf_scorer(state, params, ctx)
    assert scorer is not None

    _assert_batched_matches_loop(
        scorer, 'nirspec_mask_width', [2., 3., 4., 5., 6.], params,
        monkeypatch)


def test_miri_background_scorer_ignores_batch_switch(monkeypatch):
    """Not batchable (map_over_ints host round-trip); switch is a no-op."""
    rng = np.random.default_rng(0x135282d)
    nints, dimy, dimx = 6, 24, 32
    data = rng.normal(50., .1, (nints, dimy, dimx)).astype(np.float32)
    err = np.ones_like(data)
    dq = np.zeros_like(data, np.uint32)
    meta = ObsMeta(
        mode='MIRI/LRS', detector='MIRIMAGE', subarray='SLITLESSPRISM',
        frame_time=0.159, ngroups=3,
        int_times=np.arange(nints, dtype=float),
        baseline_ints=np.asarray([nints]),
        segment_edges=np.asarray([nints]),
        filenames=('seg1.fits',), extra={})
    cube = RateCube(data, err, dq, meta)
    state = PipelineState(cube)
    ctx = {
        'opts': {'miri_background_method': 'median', 'wave_range': None,
                 'w1': 1., 'w2': 1.},
        'centroids': {
            'xpos': np.full(dimy, 16., dtype=float),
            'ypos': np.arange(dimy, dtype=float),
        },
    }
    params = {'miri_trace_width': 6., 'miri_background_width': 8.,
             'extract_width': 4.}
    scorer = stages.prepare_miri_background_scorer(state, params, ctx)
    assert scorer is not None

    batched, loop = _assert_batched_matches_loop(
        scorer, 'miri_background_width', [6., 8., 10.], params, monkeypatch)
    for (b_cost, _, _), (l_cost, _, _) in zip(batched, loop):
        assert b_cost == l_cost
