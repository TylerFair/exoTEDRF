"""Stage wiring of v1's ``flag_up_ramp`` JWST JumpStep."""

import warnings

import numpy as np
import pytest

from exotedrf.v2 import core, stages, upramp
from exotedrf.v2.core import ObsMeta, RampCube
from exotedrf.v2.pipeline import PipelineState

from tests.v2.test_upramp_jump import NIRSPEC, NIRISS, stcal_detect, synth

TGROUPS = (2.0, 0.5)


def cube_and_ctx(mode_pars, *, flag_in_time, nints=(5, 4), shape=(10, 16, 40),
                 seed=2, mode='NIRSPEC/G395H'):
    """Create a segmented ramp cube and jump-detection context.

    Parameters
    ----------
    mode_pars : tuple
        Observing mode and jump-detection parameters.
    flag_in_time : bool
        Flag in time option.
    nints : int
        Number of integrations.
    shape : tuple[int]
        Observation array shape.
    seed : int
        Random seed.
    mode : str
        Observing mode.

    Returns
    -------
    context : dict
        Reduction options and reference arrays.
    """
    ngroups, ny, nx = shape
    parts = [synth(n, ngroups, ny, nx, seed=seed + k)
             for k, n in enumerate(nints)]
    data = np.concatenate([p[0] for p in parts])
    gdq = np.concatenate([p[1] for p in parts])
    gain, rn = parts[0][2], parts[0][3]
    edges = np.cumsum(nints)
    headers = tuple({'TGROUP': t, 'NFRAMES': 1} for t in TGROUPS[:len(nints)])
    meta = ObsMeta(mode, 'NRS1', 'SUB2048', TGROUPS[0], ngroups,
                   np.arange(edges[-1], dtype=float), np.array([2]), edges,
                   tuple(f'seg{i}.fits' for i in range(len(nints))),
                   {'header': {'TGROUP': TGROUPS[0], 'NFRAMES': 1},
                    'segment_headers': headers})
    cube = RampCube(data, gdq, np.zeros((ny, nx), np.uint32), meta)
    ctx = {'opts': {'flag_up_ramp': True, 'flag_in_time': flag_in_time,
                    'jump_threshold': 5, 'upramp_cpu_count': 12},
           'refpack': {'gain': gain, 'readnoise': rn},
           'upramp_jump_crds_pars': mode_pars}
    return cube, ctx


def expected_segments(cube, ctx):
    """Calculate expected jump flags for each observation segment.

    Parameters
    ----------
    cube : array-like(float)
        Input observation cube.
    ctx : dict
        Reduction context.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    pars = upramp.effective_parameters(ctx['upramp_jump_crds_pars'], 5)
    gain, rn = ctx['refpack']['gain'], ctx['refpack']['readnoise']
    out = []
    for seg, tgroup in zip(core.segment_slices(cube.meta), TGROUPS):
        args = upramp.stcal_arguments(pars, gain, tgroup, 1)
        # 12 CPUs -> maximum_cores='quarter' -> 3 row slices, like v1 there.
        gdq, pdq = stcal_detect(cube.data[seg], cube.groupdq[seg], gain, rn,
                                pars, args, '3')
        out.append(gdq)
    return np.concatenate(out), pdq


@pytest.mark.parametrize('pars', [NIRSPEC, NIRISS], ids=['nirspec', 'niriss'])
def test_step_jump_up_ramp_matches_stcal_per_segment(pars, monkeypatch):
    """Check step jump up ramp matches stcal per segment."""
    cube, ctx = cube_and_ctx(pars, flag_in_time=False)
    called = []
    monkeypatch.setattr(stages.k_jump, 'flag_jumps_in_time_chunked',
                        lambda *a, **k: called.append(1))
    params = {'time_jump_threshold': 10, 'time_window': 5}
    state = stages.step_jump(PipelineState(cube), params, ctx)
    want_dq, want_pdq = expected_segments(cube, ctx)
    assert np.array_equal(np.asarray(state.cube.groupdq), want_dq)
    assert np.array_equal(np.asarray(state.cube.pixeldq), want_pdq)
    assert state.cube.data is cube.data
    assert not called
    assert ctx['upramp_jump_pars']['rejection_threshold'] == 5.


def test_up_ramp_then_time_domain_sees_up_ramp_flags(monkeypatch):
    """Check up ramp then time domain sees up ramp flags."""
    cube, ctx = cube_and_ctx(NIRISS, flag_in_time=True)
    seen = []

    def fake(data, groupdq, thresh, **kw):
        seen.append(np.asarray(groupdq).copy())
        return data, groupdq

    monkeypatch.setattr(stages.k_jump, 'flag_jumps_in_time_chunked', fake)
    stages.step_jump(PipelineState(cube),
                     {'time_jump_threshold': 10, 'time_window': 5}, ctx)
    want_dq, _ = expected_segments(cube, ctx)
    assert np.array_equal(np.concatenate(seen), want_dq)


def test_flag_in_time_false_is_forced_like_v1_without_up_ramp(monkeypatch):
    """Check flag in time false is forced like v1 without up ramp."""
    cube, ctx = cube_and_ctx(NIRISS, flag_in_time=False)
    ctx['opts']['flag_up_ramp'] = False
    calls = []
    monkeypatch.setattr(
        stages.k_jump, 'flag_jumps_in_time_chunked',
        lambda data, groupdq, thresh, **kw: (calls.append(1), (data, groupdq))[1])
    with pytest.warns(UserWarning, match='flag_in_time=False ignored'):
        state = stages.step_jump(
            PipelineState(cube), {'time_jump_threshold': 10,
                                  'time_window': 5}, ctx)
    assert len(calls) == 2
    assert np.array_equal(state.cube.groupdq, cube.groupdq)
    assert 'upramp_jump_pars' not in ctx


def test_two_group_ramp_skips_jwst_step_and_forces_time_domain(monkeypatch):
    """Check two group ramp skips JWST step and forces time domain."""
    cube, ctx = cube_and_ctx(NIRISS, flag_in_time=False, shape=(2, 8, 12))
    calls = []
    monkeypatch.setattr(
        stages.k_jump, 'flag_jumps_in_time_chunked',
        lambda data, groupdq, thresh, **kw: (calls.append(1), (data, groupdq))[1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stages.step_jump(PipelineState(cube),
                         {'time_jump_threshold': 10, 'time_window': 5}, ctx)
    assert len(calls) == 2
    assert 'upramp_jump_pars' not in ctx


def test_up_ramp_requires_gain_reference():
    """Check up ramp requires gain reference."""
    cube, ctx = cube_and_ctx(NIRISS, flag_in_time=False)
    del ctx['refpack']['gain']
    with pytest.raises(ValueError, match='gain'):
        stages.step_jump(PipelineState(cube),
                         {'time_jump_threshold': 10, 'time_window': 5}, ctx)


def test_prepared_jump_scorer_with_up_ramp_matches_full_step():
    """Check prepared jump scorer with up ramp matches full step."""
    rng = np.random.default_rng(7)
    nints, ngroups, dimy, dimx = 9, 5, 10, 12
    ramp = np.arange(ngroups, dtype=np.float32)[None, :, None, None]
    data = (100. + 40. * ramp + rng.normal(0., .5, (nints, ngroups, dimy,
                                                       dimx))).astype(np.float32)
    data[4, 2:, 5, 6] += 400.
    data[6, -1, 4, 8] += 15.
    meta = ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP96', 2.0, ngroups,
                   np.arange(nints, dtype=float), np.array([nints]),
                   np.array([nints]), ('seg0.fits',),
                   {'header': {'TGROUP': 2.0, 'NFRAMES': 1}})
    cube = RampCube(data, np.zeros_like(data, np.uint8),
                    np.zeros((dimy, dimx), np.uint32), meta)
    state = PipelineState(cube)
    ctx = {
        'opts': {'flag_up_ramp': True, 'flag_in_time': True,
                 'jump_threshold': 6, 'extract_width_soss2': None,
                 'wave_range': None, 'w1': 1., 'w2': 1.},
        'refpack': {'gain': np.full((dimy, dimx), 1.6, np.float32),
                    'readnoise': np.full((dimy, dimx), 2., np.float32)},
        'upramp_jump_crds_pars': NIRISS,
        'centroids': {'ypos o1': np.linspace(4.5, 5.5, dimx)},
        'waves': {1: np.linspace(.9, 2., dimx)},
    }
    params = {'time_jump_threshold': 7., 'time_window': 5,
              'extract_width': 4.}
    scorer = stages.prepare_jump_scorer(state, params, ctx)
    assert scorer is not None
    for value, (fast_cost, _, _) in zip(
            [5., 7.], scorer.evaluate_candidates(
                'time_jump_threshold', [5., 7.], params)):
        trial = dict(params, time_jump_threshold=value)
        full = stages.step_jump(state, trial, ctx)
        assert np.any(np.asarray(full.cube.groupdq)[:, -1] & 4)
        cost, _ = stages.evaluate_optimizer_cost(full, trial, ctx)
        np.testing.assert_allclose(fast_cost, cost, rtol=1e-5, atol=1e-7)


def test_jumpstep_kwargs_options():
    """Check jumpstep keywords options."""
    from exotedrf.v2.config import jumpstep_kwargs_options as j
    assert j({'flag_4_neighbors': True, 'after_jump_flag_time1': 5}, {}) == {
        'flag_4_neighbors': True, 'after_jump_flag_time1': 5.0}
    with __import__('pytest').raises(ValueError):
        j({'rejection_threshold': 3}, {})
    with __import__('pytest').raises(NotImplementedError):
        j({'bogus_key': 1}, {})


def test_jumpstep_kwargs_reach_fixed_options_through_the_registry():
    """stage1_kwargs JumpStep keys survive load-time normalization."""
    from exotedrf.v2 import config
    import warnings
    cfg = {'observing_mode': 'NIRISS/SOSS', 'filter_detector': 'CLEAR',
           'flag_up_ramp': True,
           'stage1_kwargs': {'JumpStep': {'flag_4_neighbors': False}}}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        config.validate_supported_config(dict(cfg))
        opts = config.fixed_options(dict(cfg))
    assert opts['upramp_jump_kwargs'] == config.jumpstep_kwargs_options(
        {'flag_4_neighbors': False}, cfg)
    assert opts['upramp_jump_kwargs']
