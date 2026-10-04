"""Check optimize."""

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from exotedrf.v2 import config as v2config
from exotedrf.v2 import stages, validate
from exotedrf.v2.core import ObsMeta, RateCube
from exotedrf.v2.optimize import (TrialResult, _builtin_outer_width_is_inactive,
                                  _checkpoint_device_reserve,
                                  _configure_crds_environment,
                                  _default_checkpoint_store_factory,
                                  _first_segment_state,
                                  _phase1_device_state,
                                  _phase2_store_factory,
                                  _phase1_consumer_groups,
                                  _stream_without_memory_pressure,
                                  _sweep_parameter, decision_sensitivity,
                                  run_optimizer, select_first_finite_minimum)
from exotedrf.v2.pipeline import (CheckpointStore, Pipeline, PipelineState,
                                  Step)


@pytest.mark.parametrize('storage,budget,borrow', [
    ('host', None, True), ('device', None, False),
    ('device_if_fits', 32, True), ('device_if_fits', 1024, False)])
def test_terminal_extraction_borrows_only_when_not_resident(storage, budget,
                                                          borrow):
    """Check terminal extraction borrows only when not resident."""
    import functools
    factory = functools.partial(CheckpointStore, storage=storage,
                                max_device_bytes=budget)
    pipeline = Pipeline([Step('Extract', stages.step_extract,
                               ('extract_width',))], 'NIRISS/SOSS', {})
    selected = _phase2_store_factory(factory, pipeline, _state(), {'Extract'},
                                      allow_borrow=True)
    assert (selected(keep={'Extract'}).storage == 'borrowed_readonly') == borrow
    assert _phase2_store_factory(factory, pipeline, _state(), {'Extract'},
                                 allow_borrow=False) is factory
    custom_pipeline = _pipelines()[1]
    assert _phase2_store_factory(factory, custom_pipeline, _state(), {'Extract'},
                                 allow_borrow=True) is factory


def _state(nints=5):
    """Return state."""
    meta = ObsMeta(
        mode='NIRISS/SOSS', detector='CLEAR', subarray='SUBSTRIP96',
        frame_time=1.0, ngroups=2,
        int_times=np.arange(nints, dtype=float), baseline_ints=np.array([2]),
        segment_edges=np.array([2, nints]),
        filenames=('seg002_uncal.fits', 'seg010_uncal.fits'))
    cube = RateCube(
        data=np.zeros((nints, 1, 1), np.float32),
        err=np.ones((nints, 1, 1), np.float32),
        dq=np.zeros((nints, 1, 1), np.uint32), meta=meta)
    return PipelineState(cube)


def _replace_data(state, data, last):
    """Return replace data."""
    return PipelineState(
        dataclasses.replace(state.cube, data=np.asarray(data, np.float32)),
        dict(state.aux, last=last))


def _pipelines():
    """Return pipelines."""
    def oof(state, params, ctx):
        return _replace_data(state, state.cube.data +
                             params['soss_inner_mask_width'], 'oof')

    def scale(state, params, ctx):
        return _replace_data(state, state.cube.data * 10, 'scale')

    def jump(state, params, ctx):
        return _replace_data(state, state.cube.data + params['time_window'],
                             'jump')

    def extract(state, params, ctx):
        nints = state.cube.data.shape[0]
        flux = np.full((nints, 1), params['extract_width'], np.float32)
        return PipelineState(
            state.cube,
            dict(state.aux, last='extract', width=params['extract_width'],
                 spectral_products={
                     1: {'wave': np.asarray([1.], np.float64),
                         'flux': flux, 'ferr': np.ones_like(flux)}}))

    steps = [
        Step('OneOverFStep_grp', oof, ('soss_inner_mask_width',)),
        Step('Scale', scale),
        Step('JumpStep', jump, ('time_window',)),
        Step('Extract', extract, ('extract_width',)),
    ]
    return (Pipeline(steps, 'NIRISS/SOSS', {'which': 'phase1'}),
            Pipeline(steps, 'NIRISS/SOSS', {'which': 'full'}))


def _config():
    """Return config."""
    return {
        'observing_mode': 'NIRISS/SOSS',
        'extract_method': 'box',
        'oof_method': 'scale-achromatic',
        'soss_inner_mask_width': [9, 3, 3, 1],
        'optimize_soss_inner_mask_width': True,
        'soss_outer_mask_width': 70,
        'optimize_soss_outer_mask_width': False,
        'time_jump_threshold': 7,
        'optimize_time_jump_threshold': False,
        'time_window': [7, 5, 3],
        'optimize_time_window': True,
        'extract_width': [8, 4],
        'optimize_extract_width': True,
        'extract_width_soss2': 12,
        'name_tag': '',
    }


@pytest.mark.parametrize('strategy', ['greedy', 'joint', 'beam'])
@pytest.mark.parametrize('plots', [False, True])
def test_optimal_mode_preserves_results_and_writes_only_end_products(
        tmp_path, monkeypatch, strategy, plots):
    """Check optimal mode preserves results and writes only end products."""
    from astropy.io import fits
    from exotedrf.v2 import products

    cfg = dict(_config(), v2_search_strategy=strategy, do_plots=plots)

    def evaluator(state, params, ctx):
        cost = float(np.mean(state.cube.data) ** 2 +
                     (params['extract_width'] - 4) ** 2)
        return cost, np.array([cost])

    def run(mode):
        p1, full = _pipelines()
        return run_optimizer(dict(cfg, v2_output_mode=mode),
                             full_state=_state(), phase1_pipeline=p1,
                             full_pipeline=full, evaluator=evaluator,
                             output_dir=tmp_path / mode, logger=None)

    standard = run('standard')

    def forbidden(*args, **kwargs):
        raise AssertionError('optimal mode attempted an intermediate product')

    monkeypatch.setattr(products, 'CompatibilityCapture', forbidden)
    monkeypatch.setattr(products, '_write_reusable_sidecars', forbidden)
    monkeypatch.setattr('exotedrf.v2.io.save_rate_cube', forbidden)
    optimal = run('optimal')
    assert optimal.winners == standard.winners
    assert optimal.final_cost == standard.final_cost
    np.testing.assert_array_equal(optimal.final_scatter, standard.final_scatter)
    if plots:
        assert any(Path(p).suffix == '.png' for p in optimal.output_paths.values())
        assert all(Path(p).suffix in ('.png', '.json', '.fits')
                   for p in optimal.output_paths.values())
    else:
        assert set(optimal.output_paths) == {'summary', 'spectra'}
    assert set(p for p in (tmp_path / 'optimal').rglob('*') if p.is_file()) == {
        Path(p) for p in optimal.output_paths.values()}
    summary = json.loads(Path(optimal.output_paths['summary']).read_text())
    assert summary['output_mode'] == 'optimal'
    assert summary['trials']
    with fits.open(standard.output_paths['spectra']) as a, \
            fits.open(optimal.output_paths['spectra']) as b:
        assert len(a) == len(b)
        for x, y in zip(a[1:], b[1:]):
            np.testing.assert_array_equal(x.data, y.data)


def test_two_phase_order_commit_and_products(tmp_path):
    """Check two phase order commit and products."""
    phase1_pipeline, full_pipeline = _pipelines()
    seen = []

    def evaluator(state, params, ctx):
        seen.append((state.cube.data.shape[0], state.aux['last'],
                     params['extract_width']))
        mean = float(np.mean(state.cube.data))
        if state.aux['last'] == 'oof':
            cost = (mean - 3.0) ** 2
        elif state.aux['last'] == 'jump':
            cost = (mean - 35.0) ** 2
        else:
            cost = (mean - 35.0) ** 2 + \
                   (float(params['extract_width']) - 4.0) ** 2
        return cost, np.array([cost, cost + 1])

    product_calls = []

    def product_callback(state, params, ctx, paths):
        product_calls.append((state.cube.data.shape[0], dict(params), ctx))
        return {'custom': str(paths['root'] / 'custom.fits')}

    result = run_optimizer(
        _config(), full_state=_state(), phase1_pipeline=phase1_pipeline,
        full_pipeline=full_pipeline, evaluator=evaluator,
        product_callback=product_callback, output_dir=tmp_path, logger=None,
        checkpoint_path=tmp_path / 'checkpoints.npz')

    assert result.winners['soss_inner_mask_width'] == 3
    assert result.winners['time_window'] == 5
    assert result.winners['extract_width'] == 4
    assert result.winners['extract_width_soss2'] == 12
    assert result.initial_params['soss_inner_mask_width'] == 4
    assert result.initial_params['extract_width'] == 6
    assert result.final_cost == 0.0

    # Literal order and duplicate candidates are both retained.
    inner = [trial for trial in result.trials
             if trial.parameter == 'soss_inner_mask_width']
    assert [trial.value for trial in inner] == [9, 3, 3, 1]
    phase1_seen = seen[:4 + 3]
    assert all(nints == 2 and width == 4
               for nints, _, width in phase1_seen)
    assert all(trial.phase == 2 for trial in result.trials[-2:])
    assert product_calls[0][0] == 5
    assert product_calls[0][1]['extract_width'] == 4

    cost_lines = (tmp_path / 'Files' / 'Cost_.txt').read_text().splitlines()
    assert len(cost_lines) == 1 + len(result.trials)
    summary = json.loads(
        (tmp_path / 'Files' / 'Optimization_.json').read_text())
    assert summary['phase1_segment'] == 'seg002_uncal.fits'
    assert summary['corrected_v1_defects'] == [
        'commit_true_winner_downstream',
        'propagate_structural_window_winners',
        'reject_nonfinite_trial_costs',
        'miri_jump_score_usable_group',
    ]
    checkpoints = validate.load_checkpoint_npz(tmp_path / 'checkpoints.npz')
    assert checkpoints['Cost']['value'] == 0.0
    assert checkpoints['Optimizer']['winner.extract_width'] == 4
    np.testing.assert_array_equal(
        checkpoints['Extract']['aux.spectral_products.1.flux'],
        np.full((5, 1), 4., np.float32))
    assert result.output_paths['checkpoints'].endswith('checkpoints.npz')


def test_validation_checkpoint_bundle_is_strictly_opt_in(tmp_path):
    """Check validation checkpoint bundle is strictly opt in."""
    phase1_pipeline, full_pipeline = _pipelines()
    result = run_optimizer(
        _config(), full_state=_state(), phase1_pipeline=phase1_pipeline,
        full_pipeline=full_pipeline,
        evaluator=lambda state, params, ctx: (0.0, np.zeros(1)),
        product_callback=lambda state, params, ctx, paths: {},
        output_dir=tmp_path, logger=None, write_logs=False)

    assert result.summary['checkpoints'] is None
    assert 'checkpoints' not in result.output_paths
    assert not list(tmp_path.rglob('*.npz'))


def test_default_diagnostics_run_once_after_final_winner(
        tmp_path, monkeypatch):
    """Check default diagnostics run once after final winner."""
    phase1_pipeline, full_pipeline = _pipelines()
    calls = []

    monkeypatch.setattr(
        'exotedrf.v2.products.write_final_products',
        lambda state, params, ctx, paths: {'spectra': 'final.fits'})

    def diagnostics(state, params, ctx, paths, trials, *, scatter=None):
        calls.append((state.aux['last'], dict(params), len(trials),
                      np.asarray(scatter).tolist()))
        return {'cost_plot': str(paths['cost_plot'])}

    monkeypatch.setattr(
        'exotedrf.v2.products.write_diagnostic_plots', diagnostics)
    result = run_optimizer(
        _config(), full_state=_state(), phase1_pipeline=phase1_pipeline,
        full_pipeline=full_pipeline,
        evaluator=lambda state, params, ctx: (0., np.asarray([0.])),
        output_dir=tmp_path, logger=None, write_logs=False)

    assert calls == [
        ('extract', result.winners, len(result.trials), [0.])]
    assert result.products == {
        'spectra': 'final.fits',
        'cost_plot': str(tmp_path / 'Files' / 'Cost_.png'),
    }


def test_first_finite_minimum_rejects_nonfinite_and_keeps_first_tie():
    """Check first finite minimum rejects nonfinite and keeps first tie."""
    assert select_first_finite_minimum(
        [np.nan, np.inf, 2.0, 2.0, 3.0], parameter='width') == (2, 2.0)
    with pytest.raises(ValueError, match='No finite costs'):
        select_first_finite_minimum([np.nan, np.inf, -np.inf],
                                    parameter='width')


def _trial(phase, checkpoint, parameter, value, params, cost, scatter,
          beam=0, candidate_index=0):
    """Return trial."""
    return TrialResult(
        phase=phase, checkpoint=checkpoint, parameter=parameter,
        candidate_index=candidate_index, value=value, params=dict(params),
        duration_s=0.1, cost=cost, scatter=np.asarray(scatter, dtype=float),
        beam=beam)


def test_decision_sensitivity_ranks_pairs_within_context_only():
    """Check decision sensitivity ranks pairs within context only."""
    trials = [
        _trial(1, 'OneOverFStep_grp', 'w', 1, {'w': 1, 'x': 5, 'y': 5},
              1.0, [1.0, 1.0]),
        _trial(1, 'OneOverFStep_grp', 'w', 2, {'w': 2, 'x': 5, 'y': 5},
              1.0003, [1.0, 1.0]),
        _trial(1, 'OneOverFStep_grp', 'w', 3, {'w': 3, 'x': 5, 'y': 5},
              1.0, [1.0, 1.0]),

        _trial(1, 'JumpStep', 'y', 5, {'w': 1, 'x': 5, 'y': 5},
              0.5, [1.0, 2.0, 3.0], beam=0),
        _trial(1, 'JumpStep', 'y', 7, {'w': 1, 'x': 5, 'y': 7},
              0.8, [1.1, 2.2, 3.3], beam=0),
        _trial(1, 'JumpStep', 'y', 5, {'w': 1, 'x': 5, 'y': 5},
              10.5, [1.0, 2.0, 3.0], beam=1),
        _trial(1, 'JumpStep', 'y', 7, {'w': 1, 'x': 5, 'y': 7},
              10.0, [1.05, 2.1, 3.15], beam=1),

        _trial(2, 'Extract', 'extract_width', 4,
              {'w': 1, 'x': 5, 'y': 5, 'extract_width': 4}, 2.0, [1.0]),
        _trial(2, 'Extract', 'extract_width', 6,
              {'w': 1, 'x': 5, 'y': 5, 'extract_width': 6}, 2.5, [1.0]),
    ]

    ranking = decision_sensitivity(trials)
    by_name = {row['parameter']: row for row in ranking}
    assert set(by_name) == {'w', 'y', 'extract_width'}

    # Ranked by mean_rel descending: y (0.6, 0.05) > extract_width (0.25) > w (~0.0002).
    assert [row['parameter'] for row in ranking] == \
        ['y', 'extract_width', 'w']
    assert [row['rank'] for row in ranking] == [1, 2, 3]

    y_row = by_name['y']
    assert y_row['n_pairs'] == 2
    assert y_row['mean_rel'] == pytest.approx((0.6 + 0.05) / 2)
    assert y_row['max_rel'] == pytest.approx(0.6)
    assert y_row['median_rel'] == pytest.approx((0.6 + 0.05) / 2)
    assert y_row['prune_candidate'] is False
    assert y_row['mean_rel_scatter'] == pytest.approx((0.1 + 0.05) / 2)
    # Beam0's cheaper trial wins at y=5; beam1's cheaper trial wins at y=7.
    assert y_row['winning_values'] == {5: 1, 7: 1}

    w_row = by_name['w']
    assert w_row['n_pairs'] == 3
    assert w_row['max_rel'] == pytest.approx(0.0003, abs=1e-9)
    assert w_row['prune_candidate'] is True
    assert w_row['winning_values'] == {1: 1}

    extract_row = by_name['extract_width']
    assert extract_row['n_pairs'] == 1
    assert extract_row['mean_rel'] == pytest.approx(0.25)
    assert extract_row['winning_values'] == {4: 1}

    # A stricter negligible_rel threshold demotes 'w' from a prune candidate.
    strict = decision_sensitivity(trials, negligible_rel=1e-4)
    assert {row['parameter']: row['prune_candidate']
           for row in strict}['w'] is False


def test_all_nan_prepared_sweep_is_rejected():
    """Check all NaN prepared sweep is rejected."""
    phase1_pipeline, _ = _pipelines()
    store = CheckpointStore()
    store.put('JumpStep', _state())
    params = {'time_window': 7}
    trials = []
    prepared = [
        (np.nan, np.asarray([np.nan]), 0.),
        (np.nan, np.asarray([np.nan]), 0.),
    ]

    with pytest.raises(ValueError, match='No finite costs'):
        _sweep_parameter(
            phase=1, checkpoint='JumpStep', parameter='time_window',
            candidates=[3, 5], pipeline=phase1_pipeline, store=store,
            evaluation_stop=3, commit_stop=None, params=params,
            evaluator=None, trials=trials, logger=None, commit_winner=False,
            prepared_results=prepared)

    assert params['time_window'] == 7
    assert all(np.isnan(trial.cost) for trial in trials)


def test_first_segment_baseline_keeps_exposure_global_egress():
    """Stage/deepstack masks are global; v1 cost slices remain raw YAML."""
    nints = 400
    obs_meta = ObsMeta(
        mode='NIRISS/SOSS', detector='CLEAR', subarray='SUBSTRIP96',
        frame_time=1., ngroups=2, int_times=np.arange(nints, dtype=float),
        baseline_ints=np.asarray([100, -100]),
        segment_edges=np.asarray([200, 400]),
        filenames=('seg001_uncal.fits', 'seg002_uncal.fits'),
        # A file-local NINTS must not override the exposure-global INTEND.
        extra={'header': {'NINTS': 200}, 'exposure_nints': 200,
               'segment_int_starts': (1, 201),
               'segment_int_ends': (200, 400)})
    data = np.zeros((nints, 1, 1), np.float32)
    full = PipelineState(RateCube(
        data, np.ones_like(data), np.zeros_like(data, np.uint32), obs_meta))

    full_mask = stages.baseline_bool_for_meta(full.cube.meta, nints)
    first = _first_segment_state(full)
    first_mask = stages.baseline_bool_for_meta(first.cube.meta, 200)

    np.testing.assert_array_equal(
        np.flatnonzero(full_mask),
        np.concatenate([np.arange(100), np.arange(300, 400)]))
    np.testing.assert_array_equal(np.flatnonzero(first_mask), np.arange(100))
    assert first.cube.meta.extra['exposure_nints'] == 400


def test_extract_sweep_scores_exact_production_products(tmp_path):
    """Check extract sweep scores exact production products."""
    nints = 11
    obs_meta = ObsMeta(
        mode='NIRISS/SOSS', detector='CLEAR', subarray='SUBSTRIP96',
        frame_time=1., ngroups=2, int_times=np.arange(nints, dtype=float),
        baseline_ints=np.asarray([nints]), segment_edges=np.asarray([nints]),
        filenames=('seg001_uncal.fits',))
    data = np.ones((nints, 6, 1), np.float32)
    err = np.ones_like(data)
    dq = np.zeros_like(data, np.uint32)
    dq[[2, 5, 8], 1, 0] = stages.core.DQ_DO_NOT_USE
    state = PipelineState(RateCube(data, err, dq, obs_meta))
    ctx = {
        'opts': {'extract_width_soss2': None,
                 'mask_do_not_use_pixels': True, 'mask_saturated_pixels': True,
                 'saturation_rescue': False,
                 'wave_range': None, 'w1': 1., 'w2': 0.},
        'centroids': {'ypos o1': np.asarray([3.])},
        'waves': {1: np.asarray([1.])},
    }
    pipeline = Pipeline([
        Step('Extract', stages.step_extract, ('extract_width',)),
    ], mode='NIRISS/SOSS', ctx=ctx)

    optimizer_costs = []
    production_costs = []
    for width in (4, 2):
        optimizer_costs.append(float(stages.evaluate_optimizer_cost(
            state, {'extract_width': width}, ctx)[0]))
        extracted = pipeline.run(state, {'extract_width': width})
        production_costs.append(float(stages.evaluate_production_cost(
            extracted, {'extract_width': width}, ctx)[0]))
    np.testing.assert_allclose(optimizer_costs, [0., 0.])
    assert production_costs[0] > production_costs[1] == 0.

    cfg = _config()
    cfg.update({
        'soss_inner_mask_width': 3,
        'optimize_soss_inner_mask_width': False,
        'time_window': 5,
        'optimize_time_window': False,
        'extract_width': [4, 2],
        'optimize_extract_width': True,
        'baseline_ints': [nints],
        'wave_range': None,
        'w1': 1.,
        'w2': 0.,
    })
    result = run_optimizer(
        cfg, full_state=state, phase1_pipeline=pipeline,
        full_pipeline=pipeline, output_dir=tmp_path, logger=None,
        write_logs=False, write_products=False)
    assert result.winners['extract_width'] == 2
    assert result.final_cost == 0.


def test_crds_environment_uses_yaml_defaults_without_overwriting(monkeypatch):
    """Check CRDS environment uses YAML defaults without overwriting."""
    for name in ('CRDS_PATH', 'CRDS_SERVER_URL', 'CRDS_CONTEXT'):
        monkeypatch.delenv(name, raising=False)
    _configure_crds_environment({
        'crds_cache_path': '/configured/cache',
        'crds_context': 'jwst_test.pmap',
    })
    assert os.environ['CRDS_PATH'] == '/configured/cache'
    assert os.environ['CRDS_CONTEXT'] == 'jwst_test.pmap'
    assert os.environ['CRDS_SERVER_URL'] == \
        'https://jwst-crds.stsci.edu'

    monkeypatch.setenv('CRDS_CONTEXT', 'already-set.pmap')
    _configure_crds_environment({'crds_context': 'ignored.pmap'})
    assert os.environ['CRDS_CONTEXT'] == 'already-set.pmap'


def test_default_checkpoint_factory_is_budgeted_on_gpu(monkeypatch):
    """Check default checkpoint factory is budgeted on GPU."""
    class FakeDevice:
        platform = 'gpu'

    device = FakeDevice()
    monkeypatch.delenv('EXOTEDRF_DEVICE_CHECKPOINTS', raising=False)
    monkeypatch.delenv('EXOTEDRF_CHECKPOINT_DEVICE_BYTES', raising=False)
    monkeypatch.setattr('exotedrf.v2.optimize.jax.devices', lambda: [device])
    monkeypatch.setattr('exotedrf.v2.core.device_memory_bytes',
                        lambda: 40 << 30)
    factory = _default_checkpoint_store_factory()
    assert factory.func == CheckpointStore.device_if_fits
    assert factory.keywords['max_device_bytes'] == 10 << 30
    assert factory.keywords['device'] is device

    monkeypatch.setenv('EXOTEDRF_DEVICE_CHECKPOINTS', 'host')
    assert _default_checkpoint_store_factory() is CheckpointStore


def test_all_nonfinite_sweep_fails_instead_of_committing(tmp_path):
    """Check all nonfinite sweep fails instead of committing."""
    phase1_pipeline, full_pipeline = _pipelines()

    def evaluator(state, params, ctx):
        return np.nan, np.array([np.nan])

    with pytest.raises(ValueError, match='soss_inner_mask_width'):
        run_optimizer(
            _config(), full_state=_state(),
            phase1_pipeline=phase1_pipeline, full_pipeline=full_pipeline,
            evaluator=evaluator, output_dir=tmp_path, logger=None,
            write_products=False)


def test_inactive_outer_width_reuses_first_result_but_keeps_trial_rows():
    """Check inactive outer width reuses first result but keeps trial rows."""
    calls = {'step': 0, 'evaluate': 0}

    def inactive_outer(state, params, ctx):
        calls['step'] += 1
        return _replace_data(state, state.cube.data + 1, 'oof')

    pipeline = Pipeline([
        Step('OneOverFStep_grp', inactive_outer,
             ('soss_outer_mask_width',)),
    ], 'NIRISS/SOSS', {})
    params = {'soss_outer_mask_width': 75}
    store = CheckpointStore(keep={'OneOverFStep_grp'})
    pipeline.run(_state(2), params, store=store)

    def evaluator(state, trial_params, ctx):
        calls['evaluate'] += 1
        return 4.25, np.array([1., 2.])

    trials = []
    logs = []
    final_state, winner_cost = _sweep_parameter(
        phase=1, checkpoint='OneOverFStep_grp',
        parameter='soss_outer_mask_width', candidates=[36, 50, 70],
        pipeline=pipeline, store=store, evaluation_stop=1, commit_stop=1,
        params=params, evaluator=evaluator, trials=trials, logger=logs.append,
        reuse_first_result=True)

    assert calls == {'step': 3, 'evaluate': 1}
    assert [trial.value for trial in trials] == [36, 50, 70]
    assert [trial.candidate_index for trial in trials] == [0, 1, 2]
    assert [trial.params['soss_outer_mask_width'] for trial in trials] == \
        [36, 50, 70]
    assert [trial.cost for trial in trials] == [4.25, 4.25, 4.25]
    assert all(np.array_equal(trial.scatter, [1., 2.]) for trial in trials)
    assert [line.split(':', 1)[0] for line in logs[:-1]] == [
        '[v2] soss_outer_mask_width=36',
        '[v2] soss_outer_mask_width=50',
        '[v2] soss_outer_mask_width=70',
    ]
    assert logs[-1].startswith('[v2] winner soss_outer_mask_width=36 ')
    assert params['soss_outer_mask_width'] == 36
    assert winner_cost == 4.25
    assert final_state.aux['last'] == 'oof'


def test_outer_width_collapse_gate_is_strictly_builtin_scale_achromatic():
    """Check outer width collapse gate is strictly builtin scale achromatic."""
    builtin = Pipeline([
        Step('OneOverFStep_grp', stages.step_oneoverf_grp,
             ('soss_inner_mask_width', 'soss_outer_mask_width')),
    ], 'NIRISS/SOSS', {})
    cfg = {'oof_method': 'scale-achromatic'}

    assert _builtin_outer_width_is_inactive(cfg, builtin, None)
    assert not _builtin_outer_width_is_inactive(
        {'oof_method': 'scale-achromatic-window'}, builtin, None)
    assert not _builtin_outer_width_is_inactive(
        {'oof_method': 'scale-chromatic'}, builtin, None)
    assert not _builtin_outer_width_is_inactive(cfg, builtin, lambda *_: 0.)

    custom = Pipeline([
        Step('OneOverFStep_grp', lambda state, params, ctx: state,
             ('soss_outer_mask_width',)),
    ], 'NIRISS/SOSS', {})
    assert not _builtin_outer_width_is_inactive(cfg, custom, None)


def test_injected_evaluator_executes_every_outer_width_candidate(tmp_path):
    """Check injected evaluator executes every outer width candidate."""
    cfg = _config()
    cfg.update({
        'soss_inner_mask_width': 20,
        'optimize_soss_inner_mask_width': False,
        'soss_outer_mask_width': [36, 50, 70],
        'optimize_soss_outer_mask_width': True,
        'time_window': 5,
        'optimize_time_window': False,
        'extract_width': 4,
        'optimize_extract_width': False,
    })

    def outer(state, params, ctx):
        return _replace_data(
            state, state.cube.data + params['soss_outer_mask_width'], 'oof')

    pipeline1 = Pipeline([
        Step('OneOverFStep_grp', outer,
             ('soss_inner_mask_width', 'soss_outer_mask_width')),
    ], 'NIRISS/SOSS', {'which': 'phase1'})
    pipeline2 = Pipeline([
        Step('OneOverFStep_grp', outer,
             ('soss_inner_mask_width', 'soss_outer_mask_width')),
    ], 'NIRISS/SOSS', {'which': 'full'})
    phase1_calls = []

    def evaluator(state, params, ctx):
        if ctx['which'] == 'phase1':
            phase1_calls.append(params['soss_outer_mask_width'])
        return (params['soss_outer_mask_width'] - 50) ** 2, np.array([0.])

    result = run_optimizer(
        cfg, full_state=_state(), phase1_pipeline=pipeline1,
        full_pipeline=pipeline2, evaluator=evaluator, output_dir=tmp_path,
        logger=None, write_logs=False, write_products=False)

    assert phase1_calls == [36, 50, 70]
    assert result.winners['soss_outer_mask_width'] == 50


def test_phase1_advances_lazily_and_commits_each_consumer_once(tmp_path):
    """Check phase1 advances lazily and commits each consumer once."""
    calls = []

    def oof(state, params, ctx):
        if ctx['which'] == 'phase1':
            calls.append(('oof', params['soss_inner_mask_width']))
        return _replace_data(state, state.cube.data +
                             params['soss_inner_mask_width'], 'oof')

    def scale(state, params, ctx):
        if ctx['which'] == 'phase1':
            calls.append(('scale', None))
        return _replace_data(state, state.cube.data * 10, 'scale')

    def jump(state, params, ctx):
        if ctx['which'] == 'phase1':
            calls.append(('jump', params['time_window']))
        return _replace_data(state, state.cube.data + params['time_window'],
                             'jump')

    def extract(state, params, ctx):
        if ctx['which'] == 'phase1':
            calls.append(('extract', params['extract_width']))
        return PipelineState(state.cube, dict(
            state.aux, last='extract', width=params['extract_width']))

    steps = [
        Step('OneOverFStep_grp', oof, ('soss_inner_mask_width',)),
        Step('Scale', scale),
        Step('JumpStep', jump, ('time_window',)),
        Step('Extract', extract, ('extract_width',)),
    ]
    phase1 = Pipeline(steps, 'NIRISS/SOSS', {'which': 'phase1'})
    full = Pipeline(steps, 'NIRISS/SOSS', {'which': 'full'})

    def evaluator(state, params, ctx):
        mean = float(np.mean(state.cube.data))
        if state.aux['last'] == 'oof':
            cost = (mean - 3.) ** 2
        elif state.aux['last'] == 'jump':
            cost = (mean - 35.) ** 2
        else:
            cost = (mean - 35.) ** 2 + (params['extract_width'] - 4.) ** 2
        return cost, np.asarray([cost])

    result = run_optimizer(
        _config(), full_state=_state(), phase1_pipeline=phase1,
        full_pipeline=full, evaluator=evaluator, output_dir=tmp_path,
        logger=None, write_logs=False, write_products=False)

    # No eager pass evaluates a consumer with the mean-derived initial value.
    assert calls == [
        ('oof', 9), ('oof', 3), ('oof', 3), ('oof', 1), ('oof', 3),
        ('scale', None),
        ('jump', 7), ('jump', 5), ('jump', 3), ('jump', 5),
    ]
    assert result.winners['soss_inner_mask_width'] == 3
    assert result.winners['time_window'] == 5
    assert result.winners['extract_width'] == 4


def test_phase1_commits_shared_consumer_after_all_coordinate_winners(tmp_path):
    """Check phase1 commits shared consumer after all coordinate winners."""
    cfg = _config()
    cfg.update({
        'soss_inner_mask_width': [1, 3],
        'optimize_soss_inner_mask_width': True,
        'soss_outer_mask_width': [10, 20],
        'optimize_soss_outer_mask_width': True,
        'time_window': 5,
        'optimize_time_window': False,
        'extract_width': 4,
        'optimize_extract_width': False,
    })
    calls = []

    def oof(state, params, ctx):
        inner = params['soss_inner_mask_width']
        outer = params['soss_outer_mask_width']
        if ctx['which'] == 'phase1':
            calls.append((inner, outer))
        cost = (inner - 3.) ** 2 + (outer - 20.) ** 2
        return _replace_data(state, np.full_like(state.cube.data, cost),
                             'oof')

    step = Step(
        'OneOverFStep_grp', oof,
        ('soss_inner_mask_width', 'soss_outer_mask_width'))
    phase1 = Pipeline([step], 'NIRISS/SOSS', {'which': 'phase1'})
    full = Pipeline([step], 'NIRISS/SOSS', {'which': 'full'})

    def evaluator(state, params, ctx):
        return float(np.mean(state.cube.data)), np.asarray([0.])

    result = run_optimizer(
        cfg, full_state=_state(), phase1_pipeline=phase1,
        full_pipeline=full, evaluator=evaluator, output_dir=tmp_path,
        logger=None, write_logs=False, write_products=False)

    # The second coordinate sees the first coordinate's winner.
    assert calls == [(1, 15), (3, 15), (3, 10), (3, 20), (3, 20)]
    assert result.winners['soss_inner_mask_width'] == 3
    assert result.winners['soss_outer_mask_width'] == 20


def test_prepared_final_badpix_omits_dead_phase1_materialization(
        tmp_path, monkeypatch):
    """Check prepared final bad-pixel correction omits dead phase1 materialization."""
    cfg = _config()
    cfg.update({
        'soss_inner_mask_width': 3,
        'optimize_soss_inner_mask_width': False,
        'time_window': 5,
        'optimize_time_window': False,
        'extract_width': 4,
        'optimize_extract_width': False,
        'space_outlier_threshold': [5, 7, 9],
        'optimize_space_outlier_threshold': True,
        'time_outlier_threshold': 8,
        'optimize_time_outlier_threshold': False,
        'box_size': 4,
        'optimize_box_size': False,
        'window_size': 7,
        'optimize_window_size': False,
    })
    calls = []

    def fake_badpix(state, params, ctx):
        calls.append(ctx['which'])
        return _replace_data(state, state.cube.data + 1., 'badpix')

    monkeypatch.setattr(stages, 'step_badpix', fake_badpix)
    step = Step(
        'BadPixStep', stages.step_badpix, ('space_outlier_threshold',))
    phase1 = Pipeline([step], 'NIRISS/SOSS', {'which': 'phase1'})
    full = Pipeline([step], 'NIRISS/SOSS', {'which': 'full'})

    class Prepared:
        final_consumer_only = True

        @staticmethod
        def evaluate_candidates(parameter, candidates, params):
            assert parameter == 'space_outlier_threshold'
            return [((float(value) - 7.) ** 2, np.asarray([value]), 0.)
                    for value in candidates]

    monkeypatch.setattr(
        stages, 'prepare_badpix_scorer', lambda state, params, ctx: Prepared())

    def evaluator(state, params, ctx):
        return 0., np.asarray([0.])

    monkeypatch.setattr(stages, 'evaluate_optimizer_cost', evaluator)
    monkeypatch.setattr(stages, 'evaluate_production_cost', evaluator)

    result = run_optimizer(
        cfg, full_state=_state(), phase1_pipeline=phase1,
        full_pipeline=full, output_dir=tmp_path, logger=None,
        write_logs=False, write_products=False)

    assert result.winners['space_outlier_threshold'] == 7
    assert calls == ['full']


def test_phase1_schedule_rejects_checkpoint_with_split_consumers():
    """Check phase1 schedule rejects checkpoint with split consumers."""
    pipeline = Pipeline([
        Step('First', lambda state, params, ctx: state, ('left',)),
        Step('Second', lambda state, params, ctx: state, ('right',)),
    ], 'NIRISS/SOSS', {})
    plan = [('SharedCheckpoint', [
        ('left', np.asarray([1])), ('right', np.asarray([2])),
    ])]

    with pytest.raises(ValueError, match='must have one shared consumer'):
        _phase1_consumer_groups(plan, pipeline)


def test_search_options_validates_negligible_rel_cost_and_leaf_tolerance():
    # Defaults are accepted silently.
    """Check search options validates negligible rel cost and leaf tolerance."""
    v2config.search_options({})
    v2config.search_options({'v2_negligible_rel_cost': 0.5,
                             'v2_leaf_tolerance': 2.0})
    v2config.search_options({'v2_leaf_tolerance': 0.0})

    with pytest.raises(ValueError, match='v2_negligible_rel_cost'):
        v2config.search_options({'v2_negligible_rel_cost': 0.0})
    with pytest.raises(ValueError, match='v2_negligible_rel_cost'):
        v2config.search_options({'v2_negligible_rel_cost': -1e-3})
    with pytest.raises(ValueError, match='v2_negligible_rel_cost'):
        v2config.search_options({'v2_negligible_rel_cost': True})
    with pytest.raises(ValueError, match='v2_negligible_rel_cost'):
        v2config.search_options({'v2_negligible_rel_cost': 'big'})

    with pytest.raises(ValueError, match='v2_leaf_tolerance'):
        v2config.search_options({'v2_leaf_tolerance': -0.1})
    with pytest.raises(ValueError, match='v2_leaf_tolerance'):
        v2config.search_options({'v2_leaf_tolerance': True})
    with pytest.raises(ValueError, match='v2_leaf_tolerance'):
        v2config.search_options({'v2_leaf_tolerance': float('nan')})

    # Validate_supported_config runs the same check.
    cfg = {'observing_mode': 'NIRISS/SOSS', 'extract_method': 'box',
          'oof_method': 'scale-achromatic', 'v2_leaf_tolerance': -1.0}
    with pytest.raises(ValueError, match='v2_leaf_tolerance'):
        v2config.validate_supported_config(cfg)


def test_default_search_strategy_is_greedy_with_empty_search_summary(
        tmp_path):
    """Check default search strategy is greedy with empty search summary."""
    phase1_pipeline, full_pipeline = _pipelines()
    result = run_optimizer(
        _config(), full_state=_state(), phase1_pipeline=phase1_pipeline,
        full_pipeline=full_pipeline,
        evaluator=lambda state, params, ctx: (0., np.asarray([0.])),
        output_dir=tmp_path, logger=None, write_logs=False,
        write_products=False)

    assert result.summary['search_strategy'] == 'greedy'
    assert result.summary['beam_width'] == 3
    assert result.summary['final_candidates'] == 3
    assert result.summary['group_max_evals'] == 4096
    assert result.summary['phase1_beam'] == []
    assert result.summary['phase2_candidates'] == []
    assert result.summary['search_evaluations'] == len(result.trials)


def _interaction_cost_pipelines():
    """A single OneOverFStep_grp group whose two parameters interact."""
    cost_table = {
        (0, 0): 5., (0, 10): 5., (0, 20): 5.,
        (10, 0): 5., (10, 10): 4., (10, 20): 5.,
        (20, 0): 5., (20, 10): 5., (20, 20): 0.,
    }

    def oof(state, params, ctx):
        return state

    step = Step(
        'OneOverFStep_grp', oof,
        ('soss_inner_mask_width', 'soss_outer_mask_width'))
    phase1 = Pipeline([step], 'NIRISS/SOSS', {'which': 'phase1'})
    full = Pipeline([step], 'NIRISS/SOSS', {'which': 'full'})

    def evaluator(state, params, ctx):
        key = (int(params['soss_inner_mask_width']),
               int(params['soss_outer_mask_width']))
        cost = cost_table[key]
        return cost, np.asarray([cost])

    cfg = _config()
    cfg.update({
        'soss_inner_mask_width': [0, 10, 20],
        'optimize_soss_inner_mask_width': True,
        'soss_outer_mask_width': [0, 10, 20],
        'optimize_soss_outer_mask_width': True,
        'time_window': 5,
        'optimize_time_window': False,
        'extract_width': 4,
        'optimize_extract_width': False,
    })
    return cfg, phase1, full, evaluator


def test_joint_search_finds_global_minimum_greedy_misses(tmp_path):
    """Check joint search finds global minimum greedy misses."""
    cfg, phase1, full, evaluator = _interaction_cost_pipelines()

    greedy = run_optimizer(
        cfg, full_state=_state(), phase1_pipeline=phase1, full_pipeline=full,
        evaluator=evaluator, output_dir=tmp_path / 'greedy', logger=None,
        write_logs=False, write_products=False)
    assert greedy.summary['search_strategy'] == 'greedy'
    assert (greedy.winners['soss_inner_mask_width'],
            greedy.winners['soss_outer_mask_width']) == (10, 10)
    assert greedy.final_cost == 4.0

    cfg_joint = dict(cfg, v2_search_strategy='joint')
    joint = run_optimizer(
        cfg_joint, full_state=_state(), phase1_pipeline=phase1,
        full_pipeline=full, evaluator=evaluator,
        output_dir=tmp_path / 'joint', logger=None, write_logs=False,
        write_products=False)
    assert joint.summary['search_strategy'] == 'joint'
    assert joint.summary['beam_width'] == 1
    assert (joint.winners['soss_inner_mask_width'],
            joint.winners['soss_outer_mask_width']) == (20, 20)
    assert joint.final_cost == 0.0
    assert joint.final_cost < greedy.final_cost

    grid_trials = [t for t in joint.trials
                   if t.checkpoint == 'OneOverFStep_grp']
    assert len(grid_trials) == 9
    assert joint.summary['phase1_beam'][0]['checkpoint'] == \
        'OneOverFStep_grp'
    assert len(joint.summary['phase1_beam'][0]['kept']) == 1
    assert joint.summary['phase1_beam'][0]['kept'][0]['cost'] == 0.0


def test_group_max_evals_smaller_than_grid_falls_back_to_iterated_descent(
        tmp_path):
    """Check group max evals smaller than grid falls back to iterated descent."""
    cfg, phase1, full, evaluator = _interaction_cost_pipelines()
    cfg.update({'v2_search_strategy': 'joint', 'v2_group_max_evals': 4})
    logs = []

    result = run_optimizer(
        cfg, full_state=_state(), phase1_pipeline=phase1, full_pipeline=full,
        evaluator=evaluator, output_dir=tmp_path, logger=logs.append,
        write_logs=False, write_products=False)

    assert any('v2_group_max_evals' in line for line in logs)
    assert any('iterated coordinate descent' in line for line in logs)
    assert (result.winners['soss_inner_mask_width'],
            result.winners['soss_outer_mask_width']) == (10, 10)
    assert result.final_cost == 4.0


def test_beam_width_two_keeps_two_lineages_and_phase2_overrides_proxy_rank(
        tmp_path):
    """Check beam width two keeps two lineages and phase2 overrides proxy rank."""
    phase1_costs = {
        (1, 6): 1.0, (3, 6): 2.0,
        (1, 5): 0.5, (1, 7): 0.9,
        (3, 5): 0.8, (3, 7): 0.95,
    }
    phase2_costs = {(1, 5): 5.0, (3, 5): 1.0}

    def oof(state, params, ctx):
        return state

    def jump(state, params, ctx):
        return state

    steps = [
        Step('OneOverFStep_grp', oof, ('soss_inner_mask_width',)),
        Step('JumpStep', jump, ('time_window',)),
    ]
    phase1_pipeline = Pipeline(steps, 'NIRISS/SOSS', {'which': 'phase1'})
    full_pipeline = Pipeline(steps, 'NIRISS/SOSS', {'which': 'full'})

    def evaluator(state, params, ctx):
        key = (int(params['soss_inner_mask_width']),
               int(params['time_window']))
        table = phase1_costs if ctx['which'] == 'phase1' else phase2_costs
        cost = table[key]
        return cost, np.asarray([cost])

    cfg = _config()
    cfg.update({
        'soss_inner_mask_width': [1, 3],
        'optimize_soss_inner_mask_width': True,
        'soss_outer_mask_width': 70,
        'optimize_soss_outer_mask_width': False,
        'time_jump_threshold': 7,
        'optimize_time_jump_threshold': False,
        'time_window': [5, 7],
        'optimize_time_window': True,
        'extract_width': 4,
        'optimize_extract_width': False,
        'v2_search_strategy': 'beam',
        'v2_beam_width': 2,
        'v2_final_candidates': 2,
    })

    result = run_optimizer(
        cfg, full_state=_state(), phase1_pipeline=phase1_pipeline,
        full_pipeline=full_pipeline, evaluator=evaluator,
        output_dir=tmp_path, logger=None, write_logs=False,
        write_products=False)

    assert result.summary['search_strategy'] == 'beam'
    checkpoints = [row['checkpoint'] for row in result.summary['phase1_beam']]
    assert checkpoints == ['OneOverFStep_grp', 'JumpStep']
    # Both lineages (a=1 and a=3) survive the first (width-2) group.
    first_kept = result.summary['phase1_beam'][0]['kept']
    assert sorted(row['params']['soss_inner_mask_width']
                 for row in first_kept) == [1, 3]

    second_kept = result.summary['phase1_beam'][1]['kept']
    assert len(second_kept) == 2
    assert (second_kept[0]['params']['soss_inner_mask_width'],
            second_kept[0]['params']['time_window']) == (1, 5)
    assert (second_kept[1]['params']['soss_inner_mask_width'],
            second_kept[1]['params']['time_window']) == (3, 5)

    phase2_rows = result.summary['phase2_candidates']
    assert len(phase2_rows) == 2
    assert phase2_rows[0]['final_cost'] == 5.0
    assert phase2_rows[1]['final_cost'] == 1.0
    assert result.winners['soss_inner_mask_width'] == 3
    assert result.winners['time_window'] == 5
    assert result.final_cost == 1.0

    assert [row['rank'] for row in phase2_rows] == [0, 1]
    assert phase2_rows[0]['rel_cost'] == pytest.approx(4.0)
    assert phase2_rows[1]['rel_cost'] == 0.0
    assert phase2_rows[0]['leaf'] is False
    assert phase2_rows[1]['leaf'] is True
    assert result.summary['leaves'] == [{
        'rank': 1, 'beam': phase2_rows[1]['beam'],
        'params': phase2_rows[1]['params'], 'final_cost': 1.0,
        'rel_cost': 0.0,
    }]


def test_beam_leaf_tolerance_keeps_runner_up_and_decision_ranking_written(
        tmp_path):
    """Check beam leaf tolerance keeps runner up and decision ranking written."""
    phase1_costs = {
        (1, 6): 1.0, (3, 6): 2.0,
        (1, 5): 0.5, (1, 7): 0.9,
        (3, 5): 0.8, (3, 7): 0.95,
    }
    phase2_costs = {(1, 5): 5.0, (3, 5): 1.0}

    def oof(state, params, ctx):
        return state

    def jump(state, params, ctx):
        return state

    steps = [
        Step('OneOverFStep_grp', oof, ('soss_inner_mask_width',)),
        Step('JumpStep', jump, ('time_window',)),
    ]
    phase1_pipeline = Pipeline(steps, 'NIRISS/SOSS', {'which': 'phase1'})
    full_pipeline = Pipeline(steps, 'NIRISS/SOSS', {'which': 'full'})

    def evaluator(state, params, ctx):
        key = (int(params['soss_inner_mask_width']),
               int(params['time_window']))
        table = phase1_costs if ctx['which'] == 'phase1' else phase2_costs
        cost = table[key]
        return cost, np.asarray([cost])

    cfg = _config()
    cfg.update({
        'soss_inner_mask_width': [1, 3],
        'optimize_soss_inner_mask_width': True,
        'soss_outer_mask_width': 70,
        'optimize_soss_outer_mask_width': False,
        'time_jump_threshold': 7,
        'optimize_time_jump_threshold': False,
        'time_window': [5, 7],
        'optimize_time_window': True,
        'extract_width': 4,
        'optimize_extract_width': False,
        'v2_search_strategy': 'beam',
        'v2_beam_width': 2,
        'v2_final_candidates': 2,
        'v2_leaf_tolerance': 5.0,
    })

    result = run_optimizer(
        cfg, full_state=_state(), phase1_pipeline=phase1_pipeline,
        full_pipeline=full_pipeline, evaluator=evaluator,
        output_dir=tmp_path, logger=None, write_products=False)

    phase2_rows = result.summary['phase2_candidates']
    assert len(phase2_rows) == 2
    assert all(row['leaf'] for row in phase2_rows)
    leaves = sorted(result.summary['leaves'], key=lambda row: row['rank'])
    assert [row['rank'] for row in leaves] == [0, 1]
    assert [row['final_cost'] for row in leaves] == [5.0, 1.0]
    assert leaves[0]['rel_cost'] == pytest.approx(4.0)
    assert leaves[1]['rel_cost'] == 0.0
    for row in leaves:
        assert set(row) == {'rank', 'beam', 'params', 'final_cost',
                            'rel_cost'}

    ranking = result.summary['decision_sensitivity']
    assert {row['parameter'] for row in ranking} == \
        {'soss_inner_mask_width', 'time_window'}
    assert [row['rank'] for row in ranking] == list(
        range(1, len(ranking) + 1))
    assert result.output_paths['decision_ranking'] == \
        str(tmp_path / 'Files' / 'Decision_ranking_.txt')
    lines = Path(result.output_paths['decision_ranking']).read_text(
        encoding='utf-8').splitlines()
    header = lines[0].split('\t')
    assert header == ['rank', 'parameter', 'n_pairs', 'mean_rel', 'max_rel',
                      'median_rel', 'mean_rel_scatter', 'prune_candidate',
                      'winning_values']
    rows = [dict(zip(header, line.split('\t'))) for line in lines[1:]]
    assert {row['parameter'] for row in rows} == \
        {'soss_inner_mask_width', 'time_window'}
    for row in rows:
        json.loads(row['winning_values'])


def test_beam_grid_uses_scorer_grid_order_and_records_every_point(
        tmp_path, monkeypatch):
    """Check beam grid uses scorer grid order and records every point."""
    cfg = _config()
    cfg.update({
        'soss_inner_mask_width': 3,
        'optimize_soss_inner_mask_width': False,
        'time_window': 5,
        'optimize_time_window': False,
        'extract_width': 4,
        'optimize_extract_width': False,
        'space_outlier_threshold': [5, 7],
        'optimize_space_outlier_threshold': True,
        'time_outlier_threshold': 8,
        'optimize_time_outlier_threshold': False,
        'box_size': [4, 6],
        'optimize_box_size': True,
        'window_size': 7,
        'optimize_window_size': False,
        'v2_search_strategy': 'joint',
    })

    def fake_badpix(state, params, ctx):
        return state

    monkeypatch.setattr(stages, 'step_badpix', fake_badpix)
    step = Step('BadPixStep', stages.step_badpix,
               ('space_outlier_threshold', 'box_size'))
    phase1 = Pipeline([step], 'NIRISS/SOSS', {'which': 'phase1'})
    full = Pipeline([step], 'NIRISS/SOSS', {'which': 'full'})

    class FakeScorer:
        grid_order = ('box_size', 'space_outlier_threshold')
        final_consumer_only = False

        @staticmethod
        def evaluate_candidates(parameter, candidates, params):
            assert parameter == 'space_outlier_threshold'
            box = params['box_size']
            return [(float(box) + float(value), np.asarray([box, value]),
                     0.1) for value in candidates]

    monkeypatch.setattr(
        stages, 'prepare_badpix_scorer',
        lambda state, params, ctx: FakeScorer())
    monkeypatch.setattr(
        stages, 'evaluate_optimizer_cost',
        lambda state, params, ctx: (0., np.asarray([0.])))
    monkeypatch.setattr(
        stages, 'evaluate_production_cost',
        lambda state, params, ctx: (0., np.asarray([0.])))

    result = run_optimizer(
        cfg, full_state=_state(), phase1_pipeline=phase1, full_pipeline=full,
        output_dir=tmp_path, logger=None, write_logs=False,
        write_products=False)

    grid_trials = [t for t in result.trials if t.checkpoint == 'BadPixStep']
    assert len(grid_trials) == 4
    assert all(t.parameter == 'space_outlier_threshold' for t in grid_trials)
    assert [t.value for t in grid_trials] == [5, 7, 5, 7]
    assert [t.params['box_size'] for t in grid_trials] == [4, 4, 6, 6]
    assert [t.cost for t in grid_trials] == [9., 11., 11., 13.]
    assert [t.grid_index for t in grid_trials] == [0, 1, 2, 3]
    assert result.winners['box_size'] == 4
    assert result.winners['space_outlier_threshold'] == 5


def test_cli_search_overrides_take_precedence_over_yaml(
        tmp_path, monkeypatch):
    """Check cli search overrides take precedence over YAML."""
    from exotedrf.v2 import core as core_module
    from exotedrf.v2 import optimize as optimize_module

    cfg_path = tmp_path / 'run_optimize.yaml'
    cfg_path.write_text(
        'observing_mode: NIRISS/SOSS\nv2_search_strategy: greedy\n',
        encoding='utf-8')
    monkeypatch.setattr(core_module, 'setup', lambda: None)
    captured = {}

    class FakeResult:
        winners = {}
        final_cost = 0.0
        output_paths = {}

    def fake_run_optimizer(config_arg, **kwargs):
        captured['config_arg'] = config_arg
        return FakeResult()

    monkeypatch.setattr(optimize_module, 'run_optimizer', fake_run_optimizer)

    optimize_module.main([
        '--config', str(cfg_path), '--no-products', '--search', 'beam',
        '--beam-width', '5', '--final-candidates', '2'])
    cfg_arg = captured['config_arg']
    assert cfg_arg['v2_search_strategy'] == 'beam'
    assert cfg_arg['v2_beam_width'] == 5
    assert cfg_arg['v2_final_candidates'] == 2

    captured.clear()
    optimize_module.main(['--config', str(cfg_path), '--no-products'])
    # No CLI overrides: the YAML path is passed through untouched.
    assert captured['config_arg'] == str(cfg_path)


def test_cost_columns_follow_yaml_flag_order_not_execution_order(tmp_path):
    """Check cost columns follow YAML flag order not execution order."""
    cfg = _config()
    reordered = {
        key: value for key, value in cfg.items()
        if key not in ('optimize_extract_width',
                       'optimize_soss_inner_mask_width',
                       'optimize_time_window')
    }
    cfg = {
        'optimize_extract_width': True,
        'optimize_time_window': True,
        'optimize_soss_inner_mask_width': True,
        **reordered,
    }
    phase1_pipeline, full_pipeline = _pipelines()

    def evaluator(state, params, ctx):
        mean = float(np.mean(state.cube.data))
        if state.aux['last'] == 'oof':
            cost = (mean - 3) ** 2
        elif state.aux['last'] == 'jump':
            cost = (mean - 35) ** 2
        else:
            cost = (mean - 35) ** 2 + (params['extract_width'] - 4) ** 2
        return cost, np.array([cost])

    result = run_optimizer(
        cfg, full_state=_state(), phase1_pipeline=phase1_pipeline,
        full_pipeline=full_pipeline, evaluator=evaluator,
        output_dir=tmp_path, logger=None, write_products=False)
    header = (tmp_path / 'Files' / 'Cost_.txt').read_text().splitlines()[0]
    assert header.split('\t')[:3] == [
        'extract_width', 'time_window', 'soss_inner_mask_width']
    assert set(result.output_paths) == {
        'cost', 'scatter', 'summary', 'decision_ranking'}


def test_tree_search_keeps_branches_within_tolerance(tmp_path):
    """``tree`` keeps every in-group branch within ``v2_tree_tolerance``."""
    cfg, phase1, full, evaluator = _interaction_cost_pipelines()

    narrow = run_optimizer(
        dict(cfg, v2_search_strategy='tree', v2_tree_tolerance=0.0),
        full_state=_state(), phase1_pipeline=phase1, full_pipeline=full,
        evaluator=evaluator, output_dir=tmp_path / 'narrow', logger=None,
        write_logs=False, write_products=False)
    assert narrow.summary['search_strategy'] == 'tree'
    assert narrow.summary['beam_width'] == 8
    assert narrow.summary['tree_tolerance'] == 0.0
    assert len(narrow.summary['phase1_beam'][0]['kept']) == 1
    assert narrow.final_cost == 0.0

    def offset_evaluator(state, params, ctx):
        cost, scatter = evaluator(state, params, ctx)
        return cost + 1.0, scatter + 1.0

    wide = run_optimizer(
        dict(cfg, v2_search_strategy='tree', v2_tree_tolerance=10.0,
             v2_beam_width=4),
        full_state=_state(), phase1_pipeline=phase1, full_pipeline=full,
        evaluator=offset_evaluator, output_dir=tmp_path / 'wide',
        logger=None, write_logs=False, write_products=False)
    kept = wide.summary['phase1_beam'][0]['kept']
    assert len(kept) == 4
    assert [row['cost'] for row in kept] == [1.0, 5.0, 6.0, 6.0]
    assert len(wide.summary['phase2_candidates']) == 4
    assert wide.final_cost == 1.0

    mid = run_optimizer(
        dict(cfg, v2_search_strategy='tree', v2_tree_tolerance=4.5,
             v2_beam_width=4),
        full_state=_state(), phase1_pipeline=phase1, full_pipeline=full,
        evaluator=offset_evaluator, output_dir=tmp_path / 'mid',
        logger=None, write_logs=False, write_products=False)
    assert [row['cost'] for row in mid.summary['phase1_beam'][0]['kept']] \
        == [1.0, 5.0]

    default = run_optimizer(
        dict(cfg, v2_search_strategy='tree'),
        full_state=_state(), phase1_pipeline=phase1, full_pipeline=full,
        evaluator=evaluator, output_dir=tmp_path / 'default', logger=None,
        write_logs=False, write_products=False)
    assert default.summary['tree_tolerance'] == 1e-3
    assert len(default.summary['phase1_beam'][0]['kept']) == 1


def _ramp_state(nints=4, ngroups=2, dimy=3, dimx=5):
    """Return ramp state."""
    from exotedrf.v2.core import RampCube
    meta = ObsMeta(
        mode='NIRISS/SOSS', detector='CLEAR', subarray='SUBSTRIP96',
        frame_time=1., ngroups=ngroups,
        int_times=np.arange(nints, dtype=float),
        baseline_ints=np.asarray([nints]),
        segment_edges=np.asarray([nints]), filenames=('seg001_uncal.fits',))
    return PipelineState(RampCube(
        np.ones((nints, ngroups, dimy, dimx), np.float32),
        np.zeros((nints, ngroups, dimy, dimx), np.uint8),
        np.zeros((dimy, dimx), np.uint32), meta))


def test_phase1_stays_on_the_host_without_an_accelerator(monkeypatch):
    """Check phase1 stays on the host without an accelerator."""
    from exotedrf.v2 import core, optimize
    monkeypatch.delenv('EXOTEDRF_PHASE1_DEVICE', raising=False)
    monkeypatch.setattr(optimize, '_gpu_backend', lambda: False)
    state = _phase1_device_state(_ramp_state(), CheckpointStore)
    assert not core.is_device_array(state.cube.data)


def test_phase1_uploads_the_first_segment_when_the_device_holds_it(
        monkeypatch):
    """Check phase1 uploads the first segment when the device holds it."""
    from exotedrf.v2 import core, optimize
    monkeypatch.delenv('EXOTEDRF_PHASE1_DEVICE', raising=False)
    monkeypatch.setattr(optimize, '_gpu_backend', lambda: True)
    monkeypatch.setattr(core, 'device_memory_bytes', lambda: 1 << 30)
    original = _ramp_state()
    state = _phase1_device_state(original, CheckpointStore)
    # SCI and GROUPDQ move; detector flags and metadata stay on the host.
    assert core.is_device_array(state.cube.data)
    assert core.is_device_array(state.cube.groupdq)
    assert isinstance(state.cube.pixeldq, np.ndarray)
    assert state.cube.meta is original.cube.meta
    np.testing.assert_array_equal(np.asarray(state.cube.data),
                                  original.cube.data)
    # An already-resident state is returned untouched.
    assert _phase1_device_state(state, CheckpointStore) is state


def test_phase1_residency_reserves_room_for_kernels_and_checkpoints(
        monkeypatch):
    """Check phase1 residency reserves room for kernels and checkpoints."""
    import functools

    from exotedrf.v2 import core, optimize
    monkeypatch.delenv('EXOTEDRF_PHASE1_DEVICE', raising=False)
    monkeypatch.setattr(optimize, '_gpu_backend', lambda: True)
    state = _ramp_state()
    raw = state.cube.data.nbytes + state.cube.groupdq.nbytes

    # Six working copies of SCI+GROUPDQ must fit beside the checkpoint budget.
    monkeypatch.setattr(core, 'device_memory_bytes', lambda: 6 * raw - 1)
    assert not core.is_device_array(
        _phase1_device_state(state, CheckpointStore).cube.data)
    monkeypatch.setattr(core, 'device_memory_bytes', lambda: 6 * raw)
    assert core.is_device_array(
        _phase1_device_state(state, CheckpointStore).cube.data)

    budgeted = functools.partial(CheckpointStore.device_if_fits,
                                 max_device_bytes=raw)
    assert _checkpoint_device_reserve(budgeted) == raw
    assert not core.is_device_array(
        _phase1_device_state(state, budgeted).cube.data)
    monkeypatch.setattr(core, 'device_memory_bytes', lambda: 7 * raw)
    assert core.is_device_array(
        _phase1_device_state(state, budgeted).cube.data)

    # Unbounded device checkpoints cannot be sized, so residency is refused.
    assert _checkpoint_device_reserve(CheckpointStore.device_resident) is None
    assert not core.is_device_array(
        _phase1_device_state(state, CheckpointStore.device_resident).cube.data)


def test_phase1_residency_policy_is_explicitly_overridable(monkeypatch):
    """Check phase1 residency policy is explicitly overridable."""
    from exotedrf.v2 import core, optimize
    state = _ramp_state()
    monkeypatch.setattr(optimize, '_gpu_backend', lambda: False)
    monkeypatch.setenv('EXOTEDRF_PHASE1_DEVICE', 'true')
    assert core.is_device_array(
        _phase1_device_state(state, CheckpointStore).cube.data)
    monkeypatch.setenv('EXOTEDRF_PHASE1_DEVICE', 'auto')
    monkeypatch.setattr(optimize, '_gpu_backend', lambda: True)
    monkeypatch.setattr(core, 'device_memory_bytes', lambda: 1 << 30)
    assert not core.is_device_array(
        _phase1_device_state(state, CheckpointStore, builtin=False).cube.data)
    monkeypatch.setenv('EXOTEDRF_PHASE1_DEVICE', 'off')
    assert not core.is_device_array(
        _phase1_device_state(state, CheckpointStore).cube.data)
    monkeypatch.setenv('EXOTEDRF_PHASE1_DEVICE', 'sometimes')
    with pytest.raises(ValueError, match='EXOTEDRF_PHASE1_DEVICE'):
        _phase1_device_state(state, CheckpointStore)


def test_auto_streaming_follows_the_backend_then_the_host_budget(monkeypatch):
    """Check auto streaming follows the backend then the host budget."""
    from exotedrf.v2 import optimize
    monkeypatch.delenv('EXOTEDRF_STREAM_STAGE1', raising=False)
    monkeypatch.setattr(optimize, '_gpu_backend', lambda: True)
    assert _stream_without_memory_pressure() is True
    monkeypatch.setattr(optimize, '_gpu_backend', lambda: False)
    assert _stream_without_memory_pressure() is False
    monkeypatch.setenv('EXOTEDRF_STREAM_STAGE1', '1')
    assert _stream_without_memory_pressure() is True
    monkeypatch.setenv('EXOTEDRF_STREAM_STAGE1', 'false')
    monkeypatch.setattr(optimize, '_gpu_backend', lambda: True)
    assert _stream_without_memory_pressure() is False
