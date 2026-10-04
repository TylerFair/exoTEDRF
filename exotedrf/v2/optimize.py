"""Select reduction settings from light-curve scatter costs."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import inspect
import itertools
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exotedrf.v2 import config as v2config
from exotedrf.v2.config import _log
from exotedrf.v2.pipeline import CheckpointStore, PipelineState
from exotedrf.v2 import products as v2products


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """Record one optimizer trial and its search context.

    Parameters
    ----------
    phase, candidate_index, beam, grid_index : int
        Optimizer phase, candidate index, search branch and grid position.
    checkpoint, parameter : str
        Restart step and swept parameter name.
    value : object
        Candidate value.
    params : dict
        Reduction parameter values.
    duration_s, cost : float
        Trial duration in seconds and light-curve scatter cost.
    scatter : array-like(float)
        Light-curve scatter values.
    varied : tuple[str]
        Parameters varied jointly for this trial.
    """
    phase: int
    checkpoint: str
    parameter: str
    candidate_index: int
    value: object
    params: dict
    duration_s: float
    cost: float
    scatter: np.ndarray = dataclasses.field(repr=False)
    beam: int = 0
    grid_index: int = -1
    varied: tuple = ()


@dataclasses.dataclass
class OptimizationResult:
    """Store the winning reduction, trial history and saved products.

    Parameters
    ----------
    winners, initial_params : dict
        Winning and initial reduction parameter values.
    trials : tuple[TrialResult]
        Ordered trial history.
    final_cost : float
        Light-curve scatter cost.
    final_scatter : array-like(float)
        Light-curve scatter values.
    final_state : PipelineState
        Winning reduction state.
    products, output_paths, summary : dict
        Saved products, output paths and run summary.
    """
    winners: dict
    initial_params: dict
    trials: tuple[TrialResult, ...]
    final_cost: float
    final_scatter: np.ndarray
    final_state: PipelineState = dataclasses.field(repr=False)
    products: dict = dataclasses.field(default_factory=dict)
    output_paths: dict = dataclasses.field(default_factory=dict)
    summary: dict = dataclasses.field(default_factory=dict)

    @property
    def cost(self):
        """Get the final scatter cost of the winning reduction."""
        return self.final_cost


def select_first_finite_minimum(costs, *, parameter='parameter'):
    """Select the first finite trial with the lowest cost.

    Ties retain the earlier candidate; all-invalid sweeps raise ValueError.

    Parameters
    ----------
    costs : array-like(float)
        Trial costs in candidate order.
    parameter : str
        Sweep name to include if every cost is invalid.

    Returns
    -------
    index : int
        Index of the winning trial.
    cost : float
        Winning finite cost.
    """
    best_index = None
    best_cost = None
    for index, raw in enumerate(costs):
        value = float(np.asarray(raw))
        if not np.isfinite(value):
            continue
        if best_index is None or value < best_cost:
            best_index = index
            best_cost = value
    if best_index is None:
        raise ValueError(f'No finite costs for sweep {parameter!r}')
    return best_index, best_cost


def _python_scalar(value):
    """Convert a NumPy scalar to its Python value."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _evaluate(evaluator, state, params, pipeline):
    """Evaluate a state and validate the scalar cost and scatter."""
    result = evaluator(state, params, pipeline.ctx)
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError('evaluator must return cost or (cost, scatter)')
        cost, scatter = result
    else:
        cost, scatter = result, np.empty(0, dtype=float)
    cost_arr = np.asarray(cost)
    if cost_arr.size != 1:
        raise ValueError('evaluator cost must be scalar')
    return float(cost_arr.reshape(())), np.asarray(scatter, dtype=float)


def _configure_crds_environment(cfg):
    """Set CRDS defaults without replacing existing environment values."""
    os.environ.setdefault('CRDS_PATH', os.fspath(cfg.get('crds_cache_path', './crds_cache')))
    os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
    os.environ.setdefault('CRDS_CONTEXT', str(cfg.get('crds_context', 'jwst_1322.pmap')))


def _default_checkpoint_store_factory():
    """Choose budgeted accelerator checkpoints or host snapshots."""
    policy = os.environ.get('EXOTEDRF_DEVICE_CHECKPOINTS', 'auto').lower()
    if policy in ('0', 'false', 'off', 'host'):
        return CheckpointStore
    if policy not in ('auto', '1', 'true', 'on', 'device'):
        raise ValueError('EXOTEDRF_DEVICE_CHECKPOINTS must be auto/device or false/host')
    try:
        devices = [device for device in jax.devices() if device.platform == 'gpu']
    except RuntimeError:
        devices = []
    if not devices:
        return CheckpointStore
    from exotedrf.v2 import core
    configured = os.environ.get('EXOTEDRF_CHECKPOINT_DEVICE_BYTES')
    # Reserve accelerator memory for temporary calculations.
    budget = (int(configured) if configured is not None else int(core.device_memory_bytes() * 0.25))
    return functools.partial(CheckpointStore.device_if_fits, max_device_bytes=budget,
        device=devices[0])


def _gpu_backend():
    """Check whether JAX uses a GPU backend."""
    try:
        return jax.default_backend() == 'gpu'
    except RuntimeError:
        return False


def _env_policy(name):
    """Read an on, off or automatic environment override."""
    value = os.environ.get(name, 'auto').lower()
    if value in ('0', 'false', 'off', 'host', 'no'):
        return False
    if value in ('1', 'true', 'on', 'device', 'stream', 'yes'):
        return True
    if value != 'auto':
        raise ValueError(f'{name} must be auto, true, or false')
    return None


def _stream_without_memory_pressure():
    """Check whether segment scheduling is enabled without host memory pressure."""
    policy = _env_policy('EXOTEDRF_STREAM_STAGE1')
    return _gpu_backend() if policy is None else policy


# Allow space for kernel temporaries and a saved input state.
_PHASE1_DEVICE_FACTOR = 6


def _checkpoint_device_reserve(store_factory):
    """Get the accelerator memory reserved for phase-1 checkpoints."""
    if store_factory is None:
        return 0
    try:
        prototype = _call_factory(store_factory, keep=set())
    except (TypeError, ValueError):
        # Keep the custom factory's allocation policy.
        return None
    storage = getattr(prototype, 'storage', 'host')
    if storage == 'device_if_fits':
        return int(getattr(prototype, 'max_device_bytes', 0) or 0)
    if storage == 'device':
        # Leave unbounded accelerator stores under caller control.
        return None
    return 0


def _phase1_device_state(state, store_factory=None, logger=None, builtin=True):
    """Upload the first segment when it and the required temporaries fit."""
    from exotedrf.v2 import core
    policy = _env_policy('EXOTEDRF_PHASE1_DEVICE')
    if policy is False:
        return state
    forced = policy is True
    cube = state.cube
    names = [name for name in ('data', 'groupdq') if getattr(cube, name, None) is not None]
    arrays = [getattr(cube, name) for name in names]
    arrays = [value for value in arrays if hasattr(value, 'nbytes')]
    if not arrays or any(core.is_device_array(value) for value in arrays):
        return state
    if not forced:
        # Use the custom pipeline's storage policy.
        if not builtin or not _gpu_backend():
            return state
        reserve = _checkpoint_device_reserve(store_factory)
        required = _PHASE1_DEVICE_FACTOR * sum(int(value.nbytes) for value in arrays)
        if reserve is None or required + reserve > core.device_memory_bytes():
            _log(logger, '[v2] phase 1 runs from system memory: the first '
                         'segment and its temporaries do not fit the device')
            return state
    updates = {name: jnp.asarray(getattr(cube, name)) for name in names
               if hasattr(getattr(cube, name), 'nbytes')}
    if not updates:
        return state
    _log(logger, '[v2] phase 1 first segment resident on the accelerator')
    return PipelineState(dataclasses.replace(cube, **updates), dict(state.aux))


def _sweep_parameter(*, phase, checkpoint, parameter, candidates, pipeline,
                     store, evaluation_stop, commit_stop, params, evaluator,
                     trials, logger, reuse_first_result=False,
                     commit_winner=True, prepared_results=None):
    """Evaluate candidates and optionally commit the first finite minimum."""
    if prepared_results is not None:
        prepared_results = list(prepared_results)
        if len(prepared_results) != len(candidates):
            raise ValueError(f'prepared result count for {parameter!r} does not match '
                'candidate count')
        scored = ((dict(params, **{parameter: _python_scalar(candidate)}), result)
                  for candidate, result in zip(candidates, prepared_results))
    else:
        run = lambda point: pipeline.rerun_for([parameter], point, store, stop=evaluation_stop)
        scored = _candidate_results(pipeline, params, parameter, candidates,
                                    evaluator, run, reuse_first_result)
    costs = []
    for candidate_index, (candidate, (trial_params, result)) in enumerate(zip(candidates, scored)):
        candidate = _python_scalar(candidate)
        cost, scatter, duration = result
        cost = float(np.asarray(cost))
        scatter = np.asarray(scatter, dtype=float)
        duration = float(duration)
        costs.append(cost)
        trials.append(TrialResult(phase=phase, checkpoint=checkpoint, parameter=parameter,
            candidate_index=candidate_index, value=candidate,
            params=trial_params, duration_s=duration, cost=cost, scatter=scatter))
        _log(logger, f'[v2] {parameter}={candidate}: cost={cost:.12g} ({duration:.1f}s)')

    winner_index, winner_cost = select_first_finite_minimum(costs, parameter=parameter)
    winner = _python_scalar(candidates[winner_index])
    params[parameter] = winner
    final_state = None
    if commit_winner:
        # Refresh downstream checkpoints with the winning value.
        final_state = pipeline.commit_from([parameter], params, store, stop=commit_stop)
    _log(logger, f'[v2] winner {parameter}={winner} (cost={winner_cost:.12g})')
    return final_state, winner_cost


def _phase1_consumer_groups(plan, pipeline):
    """Validate checkpoint groups and resolve their shared consuming steps."""
    resolved = []
    previous_consumer = -1
    for checkpoint, group in plan:
        consumers = {pipeline.first_consumer(parameter) for parameter, _ in group}
        if len(consumers) != 1:
            details = ', '.join(f'{parameter}={pipeline.first_consumer(parameter)}'
                for parameter, _ in group)
            raise ValueError(f'Phase-1 checkpoint {checkpoint!r} must have one shared '
                f'consumer ({details})')
        consumer = consumers.pop()
        if consumer <= previous_consumer:
            raise ValueError('Phase-1 checkpoint consumers must be in strictly increasing '
                f'pipeline order; {checkpoint!r} resolves to step {consumer} '
                f'after step {previous_consumer}')
        resolved.append((checkpoint, group, consumer))
        previous_consumer = consumer
    return resolved


def _prepare_group_scorer(consumer_step, consumer_name, state, params, ctx,
                          mode, evaluator, logger):
    """Prepare a shared scorer for a built-in phase-1 step."""
    if evaluator is not None:
        # Use custom evaluators for both optimization phases.
        return None
    from exotedrf.v2 import stages as v2stages
    preparers = (
        ('OneOverFStep_grp', v2stages.step_oneoverf_grp, v2stages.prepare_oneoverf_grp_scorer),
        ('OneOverFStep_grp', v2stages.step_oneoverf_grp_nirspec,
         v2stages.prepare_nirspec_oneoverf_scorer),
        ('JumpStep', v2stages.step_jump, v2stages.prepare_jump_scorer),
        ('BackgroundStep', v2stages.step_background_miri, v2stages.prepare_miri_background_scorer),
        ('BadPixStep', v2stages.step_badpix, v2stages.prepare_badpix_scorer),)
    prepare = next((prepare for name, fn, prepare in preparers
                    if name == consumer_name and fn is consumer_step.fn), None)
    if prepare is None:
        return None
    scorer = prepare(state, params, ctx)
    score_group = getattr(scorer, 'score_group', None)
    if consumer_name == 'JumpStep' and score_group is not None and mode.upper().startswith('MIRI'):
        _log(logger, f'[v2] MIRI Jump scoring newest usable group {score_group} (zero-based)')
    return scorer


def _prepared_extract_results(pipeline, consumer, parameter, candidates,
                              params, store, evaluator, logger):
    """Score aperture candidates in one pass when supported."""
    if evaluator is not None or parameter != 'extract_width':
        return None
    from exotedrf.v2 import stages as v2stages
    step = pipeline.steps[consumer]
    if step.name != 'Extract' or step.fn not in (
            v2stages.step_extract, v2stages.step_extract_nirspec, v2stages.step_extract_miri):
        return None
    state = store.get(step.name)
    if state.cube.data.ndim != 3:
        return None
    try:
        results = v2stages.prepared_extract_results(state, params, pipeline.ctx, candidates)
    except NotImplementedError:
        return None
    _log(logger, f'[v2] Extract: scored {len(results)} aperture widths in '
                 'one pass over the rate cube')
    return results


def _params_key(params):
    """Get a hashable parameter identity with rounded float values."""
    items = []
    for name in sorted(params):
        value = _python_scalar(params[name])
        if isinstance(value, float):
            value = round(value, 12)
        items.append((name, value))
    return tuple(items)


def _axis_results(scorer, parameter, candidates, params, *, pipeline,
                  start_state, start_index, evaluation_stop, evaluator, cfg):
    """Score one search axis with a prepared scorer or pipeline reruns."""
    if scorer is not None:
        return scorer.evaluate_candidates(parameter, candidates, params)
    reuse = (parameter == 'soss_outer_mask_width' and
             _builtin_outer_width_is_inactive(cfg, pipeline, evaluator))
    run = lambda point: pipeline.run(start_state, point, start=start_index, stop=evaluation_stop)
    return [result for _, result in
            _candidate_results(pipeline, params, parameter, candidates, evaluator, run, reuse)]


def _candidate_results(pipeline, params, parameter, candidates, evaluator,
                       run, reuse_first_result=False):
    """Yield candidate costs and timings, optionally reusing the first result."""
    reused = None
    for candidate in candidates:
        point = dict(params, **{parameter: _python_scalar(candidate)})
        if reused is None:
            started = time.perf_counter()
            state = run(point)
            cost, scatter = _evaluate(evaluator, state, point, pipeline)
            duration = time.perf_counter() - started
            if reuse_first_result:
                reused = cost, np.array(scatter, copy=True)
        else:
            cost, scatter = reused
            scatter = np.array(scatter, copy=True)
            duration = 0.0
        yield point, (cost, scatter, duration)


def _group_param_order(scorer, group_params):
    """Order jointly varied parameters for candidate-grid iteration."""
    grid_order = getattr(scorer, 'grid_order', None)
    if not grid_order:
        return list(group_params)
    ordered = [p for p in grid_order if p in group_params]
    ordered += [p for p in group_params if p not in ordered]
    return ordered


def _record_grid_trial(trials, *, phase, checkpoint, parameter, full_params,
                       value, cost, scatter, duration, grid_index, beam, varied, logger):
    """Record the cost, scatter and search context of a grid candidate."""
    cost = float(np.asarray(cost))
    scatter = np.asarray(scatter, dtype=float)
    trials.append(TrialResult(phase=phase, checkpoint=checkpoint, parameter=parameter,
        candidate_index=grid_index, value=_python_scalar(value),
        params=full_params, duration_s=float(duration), cost=cost,
        scatter=scatter, beam=beam, grid_index=grid_index, varied=varied))
    shown = {name: full_params[name] for name in varied}
    _log(logger, f'[v2] grid[{grid_index}] beam={beam} {shown}: '
                 f'cost={cost:.12g} ({duration:.1f}s)')
    return cost, scatter


def _evaluate_group_iterated(scorer, checkpoint, ordered, candidates_map,
                             params, *, phase, pipeline, start_state,
                             start_index, evaluation_stop, evaluator,
                             trials, logger, cfg, beam=0, max_passes=4):
    """Use repeated coordinate sweeps when a candidate grid exceeds its limit."""
    trial_params = dict(params)
    visited = []
    grid_index = 0
    changed = True
    passes = 0
    while changed and passes < max_passes:
        changed = False
        passes += 1
        for parameter in ordered:
            candidates = list(candidates_map[parameter])
            axis_results = _axis_results(scorer, parameter, candidates, trial_params,
                pipeline=pipeline, start_state=start_state, start_index=start_index,
                evaluation_stop=evaluation_stop, evaluator=evaluator, cfg=cfg)
            costs = []
            for value, (cost, scatter, duration) in zip(candidates, axis_results):
                full_params = dict(trial_params)
                full_params[parameter] = _python_scalar(value)
                cost, scatter = _record_grid_trial(trials, phase=phase, checkpoint=checkpoint,
                    parameter=parameter, full_params=full_params,
                    value=value, cost=cost, scatter=scatter,
                    duration=duration, grid_index=grid_index, beam=beam,
                    varied=tuple(ordered), logger=logger)
                visited.append((full_params, cost, scatter, float(duration), grid_index))
                costs.append(cost)
                grid_index += 1
            winner_index, _ = select_first_finite_minimum(costs, parameter=parameter)
            winner_value = _python_scalar(candidates[winner_index])
            if _python_scalar(trial_params.get(parameter)) != winner_value:
                changed = True
            trial_params[parameter] = winner_value
    _log(logger, f'[v2] {checkpoint}: v2_group_max_evals fallback (iterated '
                 f'coordinate descent) converged after {passes} pass(es)')
    return visited


def _evaluate_group_grid(scorer, checkpoint, group, params, *, phase,
                         pipeline, start_state, start_index,
                         evaluation_stop, evaluator, trials, logger, cfg,
                         beam=0, group_max_evals=4096):
    """Evaluate the joint candidate grid for a checkpoint group."""
    group_params = [p for p, _ in group]
    candidates_map = {p: c for p, c in group}
    ordered = _group_param_order(scorer, group_params)

    total = 1
    for name in ordered:
        total *= len(candidates_map[name])
    if total > group_max_evals:
        _log(logger, f'[v2] {checkpoint}: cartesian grid size {total} '
                     f'exceeds v2_group_max_evals={group_max_evals}; '
                     'falling back to iterated coordinate descent for this ' 'group')
        return _evaluate_group_iterated(
            scorer, checkpoint, ordered, candidates_map, params, phase=phase,
            pipeline=pipeline, start_state=start_state,
            start_index=start_index, evaluation_stop=evaluation_stop,
            evaluator=evaluator, trials=trials, logger=logger, cfg=cfg, beam=beam)

    outer_params = ordered[:-1]
    inner_param = ordered[-1]
    inner_candidates = list(candidates_map[inner_param])
    outer_lists = [list(candidates_map[p]) for p in outer_params]
    combos = itertools.product(*outer_lists) if outer_params else [()]

    results = []
    grid_index = 0
    for outer_combo in combos:
        trial_params = dict(params)
        for name, value in zip(outer_params, outer_combo):
            trial_params[name] = _python_scalar(value)
        axis_results = _axis_results(scorer, inner_param, inner_candidates, trial_params,
            pipeline=pipeline, start_state=start_state, start_index=start_index,
            evaluation_stop=evaluation_stop, evaluator=evaluator, cfg=cfg)
        if len(axis_results) != len(inner_candidates):
            raise ValueError(f'prepared grid result count for {inner_param!r} does not '
                'match candidate count')
        for value, (cost, scatter, duration) in zip(inner_candidates, axis_results):
            full_params = dict(trial_params)
            full_params[inner_param] = _python_scalar(value)
            cost, scatter = _record_grid_trial(trials, phase=phase, checkpoint=checkpoint,
                parameter=inner_param, full_params=full_params, value=value,
                cost=cost, scatter=scatter, duration=duration,
                grid_index=grid_index, beam=beam, varied=tuple(ordered), logger=logger)
            results.append((full_params, cost, scatter, float(duration), grid_index))
            grid_index += 1
    return results


def _debug_prepared_results(scorer, parameter, candidates, params):
    """Repeat the first candidate result to match v1 debug-mode caching."""
    if scorer is None or len(candidates) == 0:
        return None
    first = scorer.evaluate_candidates(parameter, list(candidates[:1]), params)[0]
    cost, scatter, duration = first
    return [first] + [(cost, np.array(scatter, copy=True), 0.0) for _ in candidates[1:]]


def _run_phase1_greedy(*, phase1_schedule, phase1_pipeline, phase1_state,
                       phase1_params, params, mode, store_factory, evaluator,
                       phase1_evaluator, trials, logger, cfg):
    """Select detector settings by ordered coordinate sweeps."""
    phase1_keep = {phase1_pipeline.steps[consumer].name for _, _, consumer in phase1_schedule}
    store1 = _call_factory(store_factory, keep=phase1_keep)
    _log(logger, '[v2] phase 1: first segment only')
    # Reuse the first candidate result when v1 debug caching is enabled.
    debug = bool(cfg.get('debug_mode', False)) if cfg is not None else False
    phase1_cursor = 0
    phase1_committed_state = phase1_state
    for checkpoint, group, consumer in phase1_schedule:
        # Advance to the consuming step before saving its input.
        phase1_committed_state = phase1_pipeline.run(phase1_committed_state, phase1_params,
            start=phase1_cursor, stop=consumer)
        consumer_name = phase1_pipeline.steps[consumer].name
        store1.put(consumer_name, phase1_committed_state)

        consumer_step = phase1_pipeline.steps[consumer]
        prepared_scorer = _prepare_group_scorer(
            consumer_step, consumer_name, phase1_committed_state,
            phase1_params, phase1_pipeline.ctx, mode, evaluator, logger)

        for parameter, candidates in group:
            if debug:
                prepared_results = _debug_prepared_results(
                    prepared_scorer, parameter, candidates, phase1_params)
            else:
                prepared_results = (None if prepared_scorer is None else
                    prepared_scorer.evaluate_candidates(parameter, candidates, phase1_params))
            _sweep_parameter(phase=1, checkpoint=checkpoint, parameter=parameter,
                candidates=candidates, pipeline=phase1_pipeline,
                store=store1, evaluation_stop=consumer + 1, commit_stop=None,
                params=phase1_params, evaluator=phase1_evaluator, trials=trials, logger=logger,
                reuse_first_result=debug or (prepared_results is None and
                    parameter == 'soss_outer_mask_width' and _builtin_outer_width_is_inactive(
                        cfg, phase1_pipeline, evaluator)), commit_winner=False,
                prepared_results=prepared_results)
            params[parameter] = phase1_params[parameter]

        # Apply the group winners before advancing to the next consuming step.
        skip_final_materialization = (consumer == phase1_schedule[-1][2] and
            bool(getattr(prepared_scorer, 'final_consumer_only', False)))
        del prepared_scorer
        store1.drop(consumer_name)
        if not skip_final_materialization:
            phase1_committed_state = phase1_pipeline.run(phase1_committed_state, phase1_params,
                start=consumer, stop=consumer + 1)
        phase1_cursor = consumer + 1
        # Release the completed group's input checkpoint.
    # Release first-segment data before the independent full-observation pass.
    del phase1_committed_state


def _run_phase1_beam(*, cfg, mode, phase1_schedule, phase1_pipeline,
                     phase1_state, phase1_params, store_factory,
                     phase1_evaluator, evaluator, trials, logger,
                     beam_width, group_max_evals, keep_tolerance=None):
    """Retain the best joint-search branches across detector-setting groups."""
    if not phase1_schedule:
        only = [{'beam': 0, 'parent_beam': -1, 'params': dict(phase1_params), 'cost': None}]
        return only, []

    entries = [{'beam': 0, 'parent': -1, 'params': dict(phase1_params), 'state': phase1_state}]
    cursor = 0
    next_id = 1
    history = []

    for group_index, (checkpoint, group, consumer) in enumerate(phase1_schedule):
        consumer_name = phase1_pipeline.steps[consumer].name
        is_last_group = group_index == len(phase1_schedule) - 1
        pool = []
        stores = {}
        final_consumer_only = False
        for entry in entries:
            state_before = phase1_pipeline.run(
                entry['state'], entry['params'], start=cursor, stop=consumer)
            name = f"{consumer_name}#{entry['beam']}"
            gen_store = _call_factory(store_factory, keep={name})
            gen_store.put(name, state_before)
            stores[entry['beam']] = gen_store

            consumer_step = phase1_pipeline.steps[consumer]
            scorer = _prepare_group_scorer(
                consumer_step, consumer_name, state_before, entry['params'],
                phase1_pipeline.ctx, mode, evaluator, logger)
            if scorer is not None and getattr(scorer, 'final_consumer_only', False):
                final_consumer_only = True
            grid_results = _evaluate_group_grid(scorer, checkpoint, group, entry['params'], phase=1,
                pipeline=phase1_pipeline, start_state=state_before,
                start_index=consumer, evaluation_stop=consumer + 1,
                evaluator=phase1_evaluator, trials=trials, logger=logger,
                cfg=cfg, beam=entry['beam'], group_max_evals=group_max_evals)
            for full_params, cost, _scatter, _duration, grid_index in grid_results:
                pool.append((cost, entry['beam'], grid_index, full_params))
            del scorer

        finite_pool = [row for row in pool if np.isfinite(row[0])]
        if not finite_pool:
            raise ValueError(f'No finite costs for checkpoint {checkpoint!r}')
        finite_pool.sort(key=lambda row: (row[0], row[1], row[2]))
        kept = []
        seen = set()
        for cost, parent_beam, _grid_index, full_params in finite_pool:
            key = _params_key(full_params)
            if key in seen:
                continue
            seen.add(key)
            if keep_tolerance is not None and kept and \
                    _pair_relative_cost(cost, finite_pool[0][0]) > keep_tolerance:
                break
            kept.append((cost, parent_beam, full_params))
            if len(kept) >= beam_width:
                break

        _log(logger, f'[v2] beam {checkpoint}: kept {len(kept)} of '
                     f'{len(pool)} (best cost={kept[0][0]:.12g}, worst kept='
                     f'{kept[-1][0]:.12g})')

        skip_final = is_last_group and final_consumer_only
        new_entries = []
        kept_rows = []
        for cost, parent_beam, full_params in kept:
            parent_state = stores[parent_beam].get(f'{consumer_name}#{parent_beam}')
            if skip_final:
                new_state = parent_state
            else:
                new_state = phase1_pipeline.run(parent_state, full_params, start=consumer,
                    stop=consumer + 1)
            new_entries.append({'beam': next_id, 'parent': parent_beam,
                'params': full_params, 'state': new_state})
            kept_rows.append({'beam': next_id, 'parent_beam': parent_beam,
                'params': dict(full_params), 'cost': cost})
            _log(logger, f'[v2]   kept beam {next_id} <- parent '
                         f'{parent_beam}: cost={cost:.12g}')
            next_id += 1
        history.append({'checkpoint': checkpoint, 'kept': kept_rows})

        for beam_id, gen_store in stores.items():
            gen_store.drop(f'{consumer_name}#{beam_id}')
        del stores, entries
        entries = new_entries
        cursor = consumer + 1
    return history[-1]['kept'], history


def _clean_phase2_params(base_params, entry_params, swept_names):
    """Combine fixed full-observation parameters with a phase-1 branch."""
    return dict(base_params, **{name: entry_params[name]
                                for name in swept_names if name in entry_params})


def _run_phase2_multi(*, cfg, params_base, swept_names, candidates,
                      full_pipeline, full_state, extract_groups,
                      extract_keep, store_factory, phase2_evaluator,
                      trials, logger, paths, product_callback, write_products, group_max_evals):
    """Score phase-1 finalists on the complete observation and select the best."""
    _log(logger, f'[v2] phase 2: scoring {len(candidates)} candidate '
                 'configuration(s) on the full data')
    best = None
    summary_rows = []
    for rank, candidate in enumerate(candidates):
        candidate_params = _clean_phase2_params(params_base, candidate['params'], swept_names)
        store2 = _call_factory(store_factory, keep=extract_keep)
        compatibility = None
        if product_callback is None and write_products:
            compatibility = _product_capture(v2config.output_mode(cfg), paths, full_pipeline.ctx,
                candidate_params)
        final_state = _run_final_graph(full_pipeline, full_state, candidate_params, store2,
            compatibility, logger)
        for group in extract_groups:
            consumer = full_pipeline.first_consumer(group[0][0])
            start_state = store2.get(full_pipeline.steps[consumer].name)
            grid_results = _evaluate_group_grid(None, 'Extract', group, candidate_params, phase=2,
                pipeline=full_pipeline, start_state=start_state,
                start_index=consumer, evaluation_stop=consumer + 1,
                evaluator=phase2_evaluator, trials=trials, logger=logger,
                cfg=cfg, beam=candidate['beam'], group_max_evals=group_max_evals)
            finite_rows = [row for row in grid_results if np.isfinite(row[1])]
            if not finite_rows:
                raise ValueError('No finite costs for Extract sweep')
            best_row = min(finite_rows, key=lambda row: (row[1], row[4]))
            candidate_params = dict(best_row[0])
            varied = [name for name, _ in group]
            final_state = full_pipeline.commit_from(varied, candidate_params, store2, stop=None)
        final_cost, final_scatter = _evaluate(
            phase2_evaluator, final_state, candidate_params, full_pipeline)
        _log(logger, f'[v2] candidate beam={candidate["beam"]} rank={rank}: '
                     f'final cost={final_cost:.12g}')
        summary_rows.append({'beam': candidate['beam'], 'rank': rank,
            'params': dict(candidate_params), 'final_cost': final_cost})
        if np.isfinite(final_cost) and (best is None or final_cost < best[0]):
            best = (final_cost, final_scatter, final_state,
                    dict(candidate_params), compatibility, rank)
        del store2
        if extract_groups:
            del start_state

    if best is None:
        raise ValueError('Final committed pipeline state has non-finite cost for every '
            'phase-2 candidate')
    final_cost, final_scatter, final_state, params, compatibility, rank = best
    best_cost = final_cost
    leaf_tolerance = float(cfg.get('v2_leaf_tolerance', 0.0))
    leaves = []
    for row in summary_rows:
        row['rel_cost'] = _pair_relative_cost(row['final_cost'], best_cost) \
            if np.isfinite(row['final_cost']) else float('inf')
        row['leaf'] = bool(np.isfinite(row['rel_cost']) and row['rel_cost'] <= leaf_tolerance)
        if row['leaf']:
            leaves.append({'rank': row['rank'], 'beam': row['beam'], 'params': dict(row['params']),
                'final_cost': row['final_cost'], 'rel_cost': row['rel_cost']})
    leaves.sort(key=lambda entry: entry['rank'])
    _log(logger, f'[v2] leaves within tolerance {leaf_tolerance:.12g}: {len(leaves)}')
    for entry in leaves:
        _log(logger, f"[v2]   leaf rank={entry['rank']} "
                     f"beam={entry['beam']} cost={entry['final_cost']:.12g} "
                     f"rel_cost={entry['rel_cost']:.6g}")
    _log(logger, f'[v2] phase 2 winner: beam={candidates[rank]["beam"]} '
                 f'final cost={final_cost:.12g}')
    return final_state, params, final_cost, final_scatter, compatibility, summary_rows, leaves


def _builtin_outer_width_is_inactive(cfg, pipeline, evaluator):
    """Check whether the built-in group 1/f step ignores the outer mask width."""
    if evaluator is not None or str(cfg.get('oof_method', '')).lower() != 'scale-achromatic':
        return False
    try:
        step = pipeline.steps[pipeline.first_consumer('soss_outer_mask_width')]
    except KeyError:
        return False
    from exotedrf.v2.stages import step_oneoverf_grp
    return step.fn is step_oneoverf_grp


def _first_segment_state(state):
    """Slice an initial state to its first input segment."""
    cube = state.cube
    meta = cube.meta
    edges = np.asarray(meta.segment_edges, dtype=int)
    if edges.size == 0:
        end = cube.data.shape[0]
    else:
        end = int(edges[0])
    if end <= 0:
        raise ValueError('First segment contains no integrations')

    cube_updates = {}
    nints = cube.data.shape[0]
    for field in dataclasses.fields(cube):
        name = field.name
        value = getattr(cube, name)
        if name == 'meta':
            continue
        if name in ('data', 'groupdq', 'err', 'dq') and \
                hasattr(value, 'shape') and value.ndim > 0 and value.shape[0] == nints:
            cube_updates[name] = value[:end]
    filenames = tuple(meta.filenames[:1]) if meta.filenames else ()
    extra = dict(meta.extra)
    header = extra.get('header')
    header_nints = header.get('NINTS') if hasattr(header, 'get') else None
    segment_ends = extra.get('segment_int_ends')
    if segment_ends is None:
        segment_ends = ()
    candidates = [nints, *segment_ends]
    for value in (extra.get('exposure_nints'), header_nints):
        if value not in (None, ''):
            candidates.append(value)
    # Normalize local integration bounds before removing later segments.
    extra['exposure_nints'] = max(int(value) for value in candidates)
    for key in ('segment_headers', 'segment_int_starts', 'segment_int_ends'):
        if key in extra:
            extra[key] = tuple(extra[key][:1])
    meta_first = dataclasses.replace(meta, int_times=np.asarray(meta.int_times)[:end],
        segment_edges=np.asarray([end]), filenames=filenames, extra=extra)
    cube_first = dataclasses.replace(cube, meta=meta_first, **cube_updates)

    aux = {}
    for key, value in state.aux.items():
        if hasattr(value, 'shape') and value.ndim > 0 and value.shape[0] == nints:
            aux[key] = value[:end]
        else:
            aux[key] = value
    return PipelineState(cube=cube_first, aux=aux)


def _call_factory(factory, **available):
    """Call a factory with the supplied keywords it accepts."""
    signature = inspect.signature(factory)
    if any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
        return factory(**available)
    kwargs = {name: value for name, value in available.items() if name in signature.parameters}
    return factory(**kwargs)


def _phase1_options(opts, phase1_state, full_state):
    """Slice time-dependent 1/f inputs and outlier maps to the first segment."""
    phase_opts = dict(opts)
    # Use wavelength filtering only for SOSS first-segment trials.
    if not str(opts.get('mode', '')).upper().startswith('NIRISS'):
        phase_opts['wave_range'] = None
    first_nints = phase1_state.cube.data.shape[0]
    full_nints = full_state.cube.data.shape[0]
    search_dirs = (opts.get('input_dir', ''), os.getcwd())
    for name in ('soss_timeseries', 'soss_timeseries_o2'):
        value = opts.get(name)
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and value and all(
                isinstance(item, (str, os.PathLike)) for item in value):
            # Select the first segment's auxiliary file.
            phase_opts[name] = [value[0]]
            continue
        if isinstance(value, (str, os.PathLike)):
            candidate = os.path.expanduser(os.fspath(value))
            paths = ([candidate] if os.path.isabs(candidate) else
                     [os.path.join(directory, candidate)
                      for directory in search_dirs] + [candidate])
            existing = next((path for path in paths if os.path.exists(path)), None)
            if existing is None:
                continue
            array = np.load(existing, allow_pickle=False)
        else:
            array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == full_nints:
            phase_opts[name] = array[:first_nints]
        else:
            phase_opts[name] = array

    # Slice time-dependent outlier maps to the first segment.
    value = opts.get('outlier_maps')
    if value is not None:
        segment_count = len(np.atleast_1d(full_state.cube.meta.segment_edges))

        def is_map_item(item):
            """Check whether an item names or contains an outlier map."""
            if isinstance(item, (str, os.PathLike)):
                return True
            try:
                return np.asarray(item).ndim in (2, 3)
            except (TypeError, ValueError):
                return False

        if isinstance(value, (list, tuple)) and value and \
                len(value) == segment_count and segment_count > 1 and \
                all(is_map_item(item) for item in value):
            phase_opts['outlier_maps'] = [value[0]]
        else:
            if isinstance(value, (list, tuple)) and len(value) == 1 and is_map_item(value[0]):
                value = value[0]
            array = None
            if isinstance(value, (str, os.PathLike)):
                candidate = os.path.expanduser(os.fspath(value))
                paths = ([candidate] if os.path.isabs(candidate) else
                         [os.path.join(directory, candidate)
                          for directory in search_dirs] + [candidate])
                existing = next((path for path in paths if os.path.exists(path)), None)
                if existing is not None:
                    try:
                        array = np.load(existing, allow_pickle=False)
                    except (OSError, ValueError):
                        from astropy.io import fits
                        try:
                            array = fits.getdata(existing)
                        except (OSError, ValueError, IndexError, KeyError):
                            array = None
            else:
                array = np.asarray(value)
            if array is not None and array.ndim == 3 and array.shape[0] == full_nints:
                phase_opts['outlier_maps'] = array[:first_nints]
    return phase_opts


def _auto_dependencies(cfg, opts, *, full_state=None, phase1_state=None, full_pipeline=None,
                       phase1_pipeline=None, pipeline_factory=None,
                       refpack=None, files=None, context_factory=None):
    """Load missing observations and build their reduction pipelines."""
    from exotedrf.v2 import core, io, stages
    requested_streaming = opts.get('stream_stage1', False)
    automatic = requested_streaming == 'auto'
    eligible = v2config.supports_stage1_streaming(cfg)
    # Keep the custom pipeline's scheduling policy.
    if automatic and (full_pipeline is not None or pipeline_factory is not None):
        eligible = False

    if full_state is None:
        files = list(files) if files is not None else _call_factory(
            io.find_segments, input_dir=opts['input_dir'],
            filetag=opts['input_filetag'], mode=opts['mode'],
            filter_detector=opts.get('filter_detector'))
        if not files:
            raise RuntimeError(f"No FITS found in {opts['input_dir']}")
        required = io.estimate_ramp_storage_bytes(
            files, opts.get('dtype', np.float32)) if automatic and eligible else 0
        # Allow space for raw data, corrected data and a checkpoint.
        opts['stream_stage1'] = (eligible and (_stream_without_memory_pressure() or 3 * required >
                          core.host_allocation_budget_bytes(opts.get('max_host_bytes')))
            if automatic else bool(requested_streaming))
        cube = _call_factory(io.load_ramp_cube, files=files,
            baseline_ints=opts['baseline_ints'], mode=opts['mode'],
            filter_detector=opts.get('filter_detector'),
            science_dtype=opts.get('dtype', np.float32), lazy=opts.get('stream_stage1', False),
            max_host_bytes=opts.get('max_host_bytes'), scratch_dir=opts.get('scratch_dir'))
        full_state = PipelineState(cube=cube)
    else:
        if automatic:
            required = sum(int(getattr(getattr(full_state.cube, name, None), 'nbytes', 0))
                           for name in ('data', 'groupdq'))
            opts['stream_stage1'] = eligible and (
                _stream_without_memory_pressure() or 3 * required >
                core.host_allocation_budget_bytes(opts.get('max_host_bytes')))
        else:
            opts['stream_stage1'] = bool(requested_streaming)
        files = list(files) if files is not None else list(
            getattr(full_state.cube.meta, 'filenames', ()))

    if phase1_state is None:
        phase1_state = _first_segment_state(full_state)

    if full_pipeline is not None and phase1_pipeline is not None:
        return phase1_state, full_state, phase1_pipeline, full_pipeline

    if context_factory is None:
        context_factory = stages.prepare_context

    phase_opts = _phase1_options(opts, phase1_state, full_state)
    if opts['output_mode'] == 'optimal':
        # Disable committed diagnostic products during trials.
        phase_opts = dict(phase_opts, do_plots=False)

    def build_for(state, phase):
        """Build a reduction pipeline for the requested phase."""
        these_opts = phase_opts if phase == 1 else opts
        ctx = _call_factory(context_factory, cube=state.cube, opts=these_opts, refpack=refpack,
            config=cfg, phase=phase)
        factory = pipeline_factory or stages.build_pipeline
        return _call_factory(factory, mode=opts['mode'], ctx=ctx,
                             state=state, opts=these_opts, refpack=refpack, config=cfg, phase=phase)

    if phase1_pipeline is None:
        phase1_pipeline = build_for(phase1_state, 1)
    if full_pipeline is None:
        full_pipeline = build_for(full_state, 2)
    return phase1_state, full_state, phase1_pipeline, full_pipeline


def _invoke_product_callback(callback, state, params, ctx, paths):
    """Write products through the configured callback."""
    result = _call_factory(callback, state=state, params=dict(params), ctx=ctx, paths=paths,
                           output_dir=paths['root'])
    if result is None:
        return {}
    if not isinstance(result, Mapping):
        raise TypeError('product_callback must return a mapping or None')
    return dict(result)


def _log_parameter_columns(cfg, plan):
    """Get enabled sweep names in YAML declaration order."""
    executed = {parameter for _, group in plan for parameter, _ in group}
    return [key[len('optimize_'):] for key, enabled in cfg.items()
            if key.startswith('optimize_') and enabled and key[len('optimize_'):] in executed]


def _pair_relative_cost(cost1, cost2):
    """Get the relative cost difference between two trials."""
    lo = min(cost1, cost2)
    if lo == 0.0:
        return 0.0 if cost1 == cost2 else float('inf')
    return abs(cost1 - cost2) / lo


def _pair_relative_scatter(scatter1, scatter2):
    """Get the median finite relative difference between scatter arrays."""
    s1 = np.asarray(scatter1, dtype=float)
    s2 = np.asarray(scatter2, dtype=float)
    if s1.shape != s2.shape or s1.size == 0:
        return None
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.abs(s1 - s2) / s1
    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def decision_sensitivity(trials, *, negligible_rel=1e-3):
    """Rank parameters by relative cost differences between trials.

    Compare finite trials that differ in one parameter within the same phase, checkpoint and beam.

    Parameters
    ----------
    trials : list[TrialResult]
        Optimizer trial history.
    negligible_rel : float
        Maximum relative cost gap for a negligible parameter effect.

    Returns
    -------
    results : list[dict]
        Per-parameter cost and scatter gaps, winning values and pruning flags.
    """
    contexts = {}
    for trial in trials:
        cost = float(np.asarray(trial.cost))
        if not np.isfinite(cost):
            continue
        key = (trial.phase, trial.checkpoint, trial.beam)
        contexts.setdefault(key, []).append(trial)

    rel_costs = {}
    rel_scatters = {}
    winners = {}
    for context_trials in contexts.values():
        n = len(context_trials)
        varying = set()
        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = context_trials[i], context_trials[j]
                names = set(t1.params) | set(t2.params)
                diffs = []
                for name in names:
                    v1 = _python_scalar(t1.params.get(name))
                    v2 = _python_scalar(t2.params.get(name))
                    if v1 != v2:
                        diffs.append(name)
                        if len(diffs) > 1:
                            break
                if len(diffs) != 1:
                    continue
                parameter = diffs[0]
                varying.add(parameter)
                c1, c2 = float(t1.cost), float(t2.cost)
                rel_costs.setdefault(parameter, []).append(_pair_relative_cost(c1, c2))
                scatter_rel = _pair_relative_scatter(t1.scatter, t2.scatter)
                if scatter_rel is not None:
                    rel_scatters.setdefault(parameter, []).append(scatter_rel)

        # Count winning values among trials with the same other parameters.
        for parameter in varying:
            buckets = {}
            for trial in context_trials:
                fingerprint = tuple(sorted((name, _python_scalar(value))
                    for name, value in trial.params.items() if name != parameter))
                buckets.setdefault(fingerprint, []).append(trial)
            for bucket in buckets.values():
                best_trial = min(bucket, key=lambda t: float(t.cost))
                value = _python_scalar(best_trial.params.get(parameter))
                tally = winners.setdefault(parameter, {})
                tally[value] = tally.get(value, 0) + 1

    results = []
    for parameter, rels in rel_costs.items():
        rels_arr = np.asarray(rels, dtype=float)
        max_rel = float(np.max(rels_arr))
        results.append({'parameter': parameter, 'n_pairs': int(rels_arr.size),
            'mean_rel': float(np.mean(rels_arr)), 'max_rel': max_rel,
            'median_rel': float(np.median(rels_arr)), 'mean_rel_scatter': (
                float(np.mean(rel_scatters[parameter]))
                if rel_scatters.get(parameter) else float('nan')),
            'winning_values': dict(winners.get(parameter, {})),
            'prune_candidate': bool(max_rel < negligible_rel)})
    results.sort(key=lambda row: row['mean_rel'], reverse=True)
    for rank, row in enumerate(results, start=1):
        row['rank'] = rank
    return results


def _with_host_budget(function):
    """Apply the configured host allocation budget during the reduction."""
    @functools.wraps(function)
    def bounded(config_or_path, *args, **kwargs):
        """Run the reduction within its configured host allocation budget."""
        from exotedrf.v2 import core
        cfg = v2config.coerce_config(config_or_path)
        with core.host_memory_budget(cfg.get('v2_max_host_bytes'), cfg.get('v2_scratch_dir')):
            return function(cfg, *args, **kwargs)
    return bounded


def _product_capture(mode, paths, ctx, params):
    """Create a diagnostic capture for standard output mode."""
    if mode == 'optimal':
        if not ctx.get('opts', {}).get('do_plots', True):
            return None
        return v2products.OptimalCapture(paths, ctx, params=params)
    return v2products.CompatibilityCapture(paths, ctx, params=params)


def _run_final_graph(pipeline, state, params, store, observer, logger):
    # Consume the input state so calibrated steps can release the raw arrays.
    """Run the final reduction, releasing consumed input states."""
    if isinstance(state, list):
        state = state.pop()
    holder = [state]
    del state
    if pipeline.ctx.get('opts', {}).get('stream_stage1', False):
        from exotedrf.v2.streaming import run_segmented
        return run_segmented(pipeline, holder.pop(), params, store=store, observer=observer,
                             logger=logger)
    return pipeline.run(holder.pop(), params, store=store, observer=observer)


def _phase2_store_factory(factory, pipeline, state, keep, *, allow_borrow):
    """Borrow host arrays for a built-in read-only extraction tail when appropriate."""
    if not allow_borrow or keep != {'Extract'} or not pipeline.steps:
        return factory
    from exotedrf.v2 import stages
    final = pipeline.steps[-1]
    if final.name != 'Extract' or final.fn not in (stages.step_extract, stages.step_extract_nirspec,
            stages.step_extract_miri, stages.step_extract_atoca):
        return factory
    prototype = _call_factory(factory, keep=keep)
    shape = state.cube.data.shape
    rate_bytes = (int(shape[0]) * int(np.prod(shape[-2:])) *
                  (2 * np.dtype(state.cube.data.dtype).itemsize + 4))
    if prototype.storage == 'host' or (prototype.storage == 'device_if_fits' and
            rate_bytes > prototype.max_device_bytes):
        return CheckpointStore.borrowed_readonly
    return factory


@_with_host_budget
def run_optimizer(config_or_path, *, phase1_state=None, full_state=None,
        phase1_pipeline=None, full_pipeline=None, pipeline_factory=None,
        evaluator=None, product_callback=None, output_dir=None, logger=print,
        enable_x64_mode=False, checkpoint_path=None, checkpoint_steps=None, **dependency_overrides):
    """Run detector-setting trials followed by a full-observation aperture sweep.

    Parameters
    ----------
    config_or_path : dict, str
        Reduction configuration or path to a YAML file.
    phase1_state, full_state : None, PipelineState
        First-segment and full-observation input states.
    phase1_pipeline, full_pipeline : None, Pipeline
        First-segment and full-observation reduction pipelines.
    pipeline_factory : None, callable
        Factory for missing phase pipelines.
    evaluator : None, callable
        Function returning cost or (cost, scatter) from state, params and ctx.
    product_callback : None, callable
        Function writing products from state, params, ctx and paths.
    output_dir, checkpoint_path : None, str
        Output root and optional validation bundle path.
    logger : None, callable
        Function to receive progress and warning messages.
    enable_x64_mode : bool
        If True, enable JAX float64 calculations.
    checkpoint_steps : None, list[str]
        After-step states to include in the validation NPZ.
    dependency_overrides : dict
        Overrides for refpack, files, context_factory, checkpoint_store_factory,
        write_logs and write_products.

    Returns
    -------
    result : OptimizationResult
        Winning parameters, trial history, final state and product paths.
    """
    allowed = {'refpack', 'files', 'context_factory', 'checkpoint_store_factory',
        'write_logs', 'write_products'}
    unknown = set(dependency_overrides) - allowed
    if unknown:
        raise TypeError(f'Unknown dependency override(s): {sorted(unknown)}')

    cfg = v2config.coerce_config(config_or_path)
    if v2config.ad_hoc_mode(cfg) is not None:
        # Run only extraction or PCA and extraction from saved products.
        from exotedrf.v2 import adhoc
        if any(value is not None for value in (
                phase1_state, full_state, phase1_pipeline, full_pipeline,
                pipeline_factory, evaluator, product_callback)):
            raise TypeError('ad-hoc optimizer modes do not accept injected '
                            'states, pipelines, or callbacks')
        _configure_crds_environment(cfg)
        return adhoc.run_ad_hoc(cfg, output_dir=output_dir, logger=logger,
            enable_x64_mode=enable_x64_mode, refpack=dependency_overrides.get('refpack'),
            write_products=dependency_overrides.get('write_products', True),
            write_logs=dependency_overrides.get('write_logs', True))
    v2config.validate_supported_config(cfg)
    if cfg.get('debug_mode', False):
        _log(logger, '[v2] WARNING: DEBUG MODE ENABLED: phase-1 sweeps reuse '
                     "each step's first-candidate result (v1 cached outputs, " 'force_redo=False)')
    if v2config.stream_stage1_mode(cfg) is True and checkpoint_steps:
        raise ValueError('stage-1 streaming requires bounded final checkpoints; '
                         'disable it for whole-visit intermediate captures')
    _configure_crds_environment(cfg)
    if enable_x64_mode or cfg.get('v2_enable_x64', False):
        # Enable the calculation dtype before loading FITS or reference arrays.
        from exotedrf.v2 import core
        core.enable_x64()
    opts = v2config.fixed_options(cfg)
    if checkpoint_steps:
        opts['stream_stage1'] = False
    output_mode = opts['output_mode']
    opts['dtype'] = (np.float64 if enable_x64_mode or
                     cfg.get('v2_enable_x64', False) else np.float32)
    mode = opts['mode']
    plan, initial_params = v2config.build_sweep_plan(cfg, mode)
    search_opts = v2config.search_options(cfg)
    params = dict(initial_params)
    params['extract_width_soss2'] = cfg.get('extract_width_soss2')
    phase1_width = v2config.phase1_extract_width(cfg, params)
    paths = v2products.output_layout(cfg, output_dir=output_dir, create=True)
    if opts.get('stellar_model_dir') is None and None not in (
            opts.get('st_teff'), opts.get('st_logg'), opts.get('st_met')):
        # Cache PHOENIX models under the Stage-3 output directory.
        opts['stellar_model_dir'] = str(paths['stage3'] / 'phoenix_models')
    write_products = dependency_overrides.get('write_products', True)

    restart_refpack = None
    from exotedrf.v2 import restart as v2restart
    if v2restart.is_restart(cfg) and full_state is None:
        # Load intermediate products at the configured restart point.
        full_state, restart_files = v2restart.load_restart_state(
            opts, files=dependency_overrides.get('files'), logger=logger)
        opts = v2restart.restart_opts(opts, full_state.cube)
        dependency_overrides['files'] = restart_files
        if dependency_overrides.get('refpack') is None:
            restart_refpack = v2restart.restart_refpack(
                full_state.cube, opts, restart_files, logger=logger)
    builtin_phase1 = phase1_pipeline is None and pipeline_factory is None
    phase1_state, full_state, phase1_pipeline, full_pipeline = _auto_dependencies(
            cfg, opts, full_state=full_state,
            phase1_state=phase1_state, full_pipeline=full_pipeline, phase1_pipeline=phase1_pipeline,
            pipeline_factory=pipeline_factory, refpack=(dependency_overrides.get('refpack')
                     if restart_refpack is None else restart_refpack),
            files=dependency_overrides.get('files'),
            context_factory=dependency_overrides.get('context_factory'))
    # Save reusable ATOCA profiles and estimates under Stage 3.
    if isinstance(getattr(full_pipeline, 'ctx', None), dict):
        full_pipeline.ctx.setdefault('atoca_output_dir', paths['stage3'])
    if output_mode == 'optimal':
        full_pipeline.ctx['opts'] = dict(full_pipeline.ctx.get('opts', {}),
            output_mode=output_mode, do_plots=opts['do_plots'], stream_stage1=opts['stream_stage1'])
    else:
        full_pipeline.ctx['opts'] = dict(full_pipeline.ctx.get('opts', {}),
                                         stream_stage1=opts['stream_stage1'])
    phase1_segment = (str(phase1_state.cube.meta.filenames[0])
                      if phase1_state.cube.meta.filenames else None)

    if evaluator is None:
        from exotedrf.v2.stages import (evaluate_optimizer_cost, evaluate_production_cost)
        phase1_evaluator = evaluate_optimizer_cost
        phase2_evaluator = evaluate_production_cost
    else:
        # Use the supplied evaluator for both phases.
        phase1_evaluator = phase2_evaluator = evaluator
    store_factory = dependency_overrides.get('checkpoint_store_factory')
    if store_factory is None:
        store_factory = _default_checkpoint_store_factory()
    # Upload the first segment once when the accelerator has enough memory.
    phase1_state = _phase1_device_state(phase1_state, store_factory, logger, builtin=builtin_phase1)
    trials = []

    # Select detector settings one shared step at a time.
    phase1_plan = [(checkpoint, group) for checkpoint, group in plan if checkpoint != 'Extract']
    phase1_params = dict(params)
    phase1_params['extract_width'] = phase1_width
    strategy = search_opts['strategy']
    phase1_schedule = (_phase1_consumer_groups(phase1_plan, phase1_pipeline) if phase1_plan else [])
    phase1_beam_history = []
    phase1_beam_final = []

    if strategy == 'greedy':
        if phase1_plan:
            _run_phase1_greedy(phase1_schedule=phase1_schedule,
                phase1_pipeline=phase1_pipeline, phase1_state=phase1_state,
                phase1_params=phase1_params, params=params, mode=mode,
                store_factory=store_factory, evaluator=evaluator,
                phase1_evaluator=phase1_evaluator, trials=trials, logger=logger, cfg=cfg)
    else:
        beam_width = search_opts['beam_width']
        phase1_beam_final, phase1_beam_history = _run_phase1_beam(
            cfg=cfg, mode=mode, phase1_schedule=phase1_schedule,
            phase1_pipeline=phase1_pipeline, phase1_state=phase1_state,
            phase1_params=phase1_params, store_factory=store_factory,
            phase1_evaluator=phase1_evaluator, evaluator=evaluator,
            trials=trials, logger=logger, beam_width=beam_width,
            group_max_evals=search_opts['group_max_evals'],
            keep_tolerance=search_opts.get('tree_tolerance'))

    # Release the first-segment pipeline and retain its scalar provenance.
    del phase1_state, phase1_pipeline

    # Apply detector winners to the full observation before sweeping apertures.
    _log(logger, '[v2] phase 2: all segments')
    extract_groups = [group for checkpoint, group in plan if checkpoint == 'Extract']
    extract_keep = {full_pipeline.steps[full_pipeline.first_consumer(parameter)].name
        for group in extract_groups for parameter, _ in group}
    # Borrow the calibrated cube for built-in read-only extraction.
    phase2_store_factory = _phase2_store_factory(
        store_factory, full_pipeline, full_state, extract_keep,
        allow_borrow=(evaluator is None and product_callback is None and dependency_overrides.get(
                          'checkpoint_store_factory') is None))
    phase2_candidates_summary = []
    leaves_summary = []
    if strategy == 'greedy':
        store2 = _call_factory(phase2_store_factory, keep=extract_keep)
        compatibility = None
        if product_callback is None and write_products:
            compatibility = _product_capture(output_mode, paths, full_pipeline.ctx, params)
        # Release raw data when no validation rerun needs it.
        holder = [full_state]
        if checkpoint_steps is None:
            full_state = None
        final_state = _run_final_graph(full_pipeline, holder, params, store2, compatibility, logger)
        del holder
        for group in extract_groups:
            for parameter, candidates in group:
                consumer = full_pipeline.first_consumer(parameter)
                stop = consumer + 1
                prepared_results = _prepared_extract_results(
                    full_pipeline, consumer, parameter, candidates, params,
                    store2, evaluator, logger)
                final_state, _ = _sweep_parameter(
                    phase=2, checkpoint='Extract', parameter=parameter,
                    candidates=candidates, pipeline=full_pipeline,
                    store=store2, evaluation_stop=stop, commit_stop=None,
                    params=params, evaluator=phase2_evaluator, trials=trials, logger=logger,
                    prepared_results=prepared_results)
        del store2
    else:
        swept_names = [p for _, group in phase1_plan for p, _ in group]
        n_final = min(search_opts['final_candidates'], len(phase1_beam_final))
        top_candidates = phase1_beam_final[:n_final]
        final_state, params, _, _, compatibility, \
            phase2_candidates_summary, leaves_summary = _run_phase2_multi(
                cfg=cfg, params_base=params, swept_names=swept_names,
                candidates=top_candidates, full_pipeline=full_pipeline,
                full_state=full_state, extract_groups=extract_groups,
                extract_keep=extract_keep, store_factory=phase2_store_factory,
                phase2_evaluator=phase2_evaluator, trials=trials,
                logger=logger, paths=paths, product_callback=product_callback,
                write_products=write_products, group_max_evals=search_opts['group_max_evals'])

    # Score the committed state and winning parameters.
    final_cost, final_scatter = _evaluate(phase2_evaluator, final_state, params, full_pipeline)
    if not np.isfinite(final_cost):
        raise ValueError('Final committed pipeline state has non-finite cost')
    # Record the aperture chosen by automatic extraction.
    width_selected = final_state.aux.get('extract_width_selected')
    if width_selected:
        _log(logger, '[v2] extract_width optimize selected: ' + ', '.join(
            f'order {order}={width:g}' for order, width in sorted(width_selected.items())))

    checkpoint_artifact = None
    if checkpoint_path is not None:
        from exotedrf.v2.validate import (capture_pipeline_steps, save_checkpoint_npz)
        if checkpoint_steps is None:
            # Save final validation products without repeating detector steps.
            checkpoints = {}
            spectral_products = final_state.aux.get('spectral_products')
            if spectral_products is not None:
                checkpoints['Extract'] = {'aux': {'spectral_products': spectral_products}}
        else:
            checkpoints = capture_pipeline_steps(
                full_pipeline, full_state, params, steps=checkpoint_steps)
        checkpoints['Cost'] = {'value': np.asarray(final_cost),
            'scatter': np.asarray(final_scatter)}
        checkpoints['Optimizer'] = {f'winner.{name}': np.asarray(value)
            for name, value in params.items() if value is not None}
        checkpoint_artifact = str(save_checkpoint_npz(checkpoints, checkpoint_path))

    output_products = {}
    output_timing = {}
    if product_callback is not None:
        output_products = _invoke_product_callback(
            product_callback, final_state, params, full_pipeline.ctx, paths)
    elif write_products:
        started = time.perf_counter()
        _log(logger, f'[v2] final products: {output_mode} mode')
        product_options = ({'output_mode': output_mode} if output_mode == 'optimal' else {})
        output_products = v2products.write_final_products(
            final_state, params, full_pipeline.ctx, paths, **product_options)
        # Render diagnostic plots after committing the winning reduction.
        output_products.update(v2products.write_diagnostic_plots(
            final_state, params, full_pipeline.ctx, paths, trials, scatter=final_scatter))
        if compatibility is not None:
            output_products.update(compatibility.render(final_state))
            output_timing.update(compatibility.timings)
        output_timing['post_winner_products'] = time.perf_counter() - started
        output_timing['total_output_overhead'] = (output_timing['post_winner_products'] +
            output_timing.get('diagnostic_capture', 0.))
        _log(logger, '[v2] final products complete')

    swept_columns = _log_parameter_columns(cfg, plan)
    decision_ranking = decision_sensitivity(
        trials, negligible_rel=cfg.get('v2_negligible_rel_cost', 1e-3))
    if decision_ranking:
        _log(logger, '[v2] decision ranking:')
        for row in decision_ranking:
            _log(logger, f"[v2]   {row['rank']}\t{row['parameter']}\t"
                 f"mean_rel={row['mean_rel']:.6g}\tmax_rel="
                 f"{row['max_rel']:.6g}\tn_pairs={row['n_pairs']}\t"
                 f"prune_candidate={row['prune_candidate']}\t"
                 f"winning_values={row['winning_values']}")
    summary = v2products.jsonable({'schema_version': 1, 'mode': mode, 'output_mode': output_mode,
        'stream_stage1': opts.get('stream_stage1', False),
        'diagnostic_policy': getattr(compatibility, 'summary', None),
        'phase1_segment': phase1_segment, 'phase1_extract_width': _python_scalar(phase1_width),
        'extract_width_selected': width_selected, 'initial_params': initial_params,
        'winners': params, 'final_cost': final_cost, 'trial_count': len(trials), 'trials': trials,
        'products': output_products, 'output_timing_s': output_timing,
        'checkpoints': checkpoint_artifact, 'search_strategy': strategy,
        'beam_width': search_opts['beam_width'],
        'final_candidates': search_opts['final_candidates'],
        'group_max_evals': search_opts['group_max_evals'],
        'tree_tolerance': search_opts.get('tree_tolerance'), 'phase1_beam': phase1_beam_history,
        'phase2_candidates': phase2_candidates_summary, 'leaves': leaves_summary,
        'decision_sensitivity': decision_ranking, 'search_evaluations': len(trials),
        'corrected_v1_defects': ['commit_true_winner_downstream',
            'propagate_structural_window_winners', 'reject_nonfinite_trial_costs',
            'miri_jump_score_usable_group']})
    write_logs = dependency_overrides.get('write_logs', True)
    if write_logs:
        v2products.write_optimizer_logs(paths, trials, swept_columns, summary)

    result_paths = {}
    if write_logs:
        log_keys = (('summary',) if output_mode == 'optimal' else ('cost', 'scatter', 'summary'))
        result_paths.update({key: str(paths[key]) for key in log_keys})
        if output_mode == 'standard' and summary.get('decision_sensitivity'):
            result_paths['decision_ranking'] = str(paths['decision_ranking'])
    result_paths.update({key: str(value) for key, value in output_products.items()})
    if checkpoint_artifact is not None:
        result_paths['checkpoints'] = checkpoint_artifact
    result = OptimizationResult(winners=dict(params), initial_params=dict(initial_params),
        trials=tuple(trials), final_cost=final_cost,
        final_scatter=final_scatter, final_state=final_state,
        products=output_products, output_paths=result_paths, summary=summary)
    _log(logger, f'[v2] complete: cost={final_cost:.12g}; winners='
                 f'{json.dumps(v2products.jsonable(params), sort_keys=True)}')
    if not v2products_null_like(cfg.get('archive_to_longterm_storage')) \
            and not cfg.get('v2_archive', False):
        # Archive only when explicitly requested.
        _log(logger, '[v2] archive_to_longterm_storage is set but not '
                     'applied; add v2_archive: true (or --archive) to move '
                     'input_dir and the outputs there')
    elif not v2products_null_like(cfg.get('archive_to_longterm_storage')):
        # Move the input and output directories to long-term storage.
        from exotedrf.v2 import archive
        archive_root = (v2products.v1_output_root(cfg.get('pipeline_outputs_directory',
                        'pipeline_outputs_directory'), cfg.get('output_tag', ''))
            if output_dir is None else Path(output_dir))
        result.summary['archived'] = archive.archive_run(cfg, archive_root, logger=logger)
    return result


def v2products_null_like(value):
    """Check whether an archive destination is unset."""
    return v2config._is_null_like(value)


def main(argv=None):
    """Run the optimizer command-line interface.

    Parameters
    ----------
    argv : None, list[str]
        Command-line arguments. If None, read the process arguments.

    Returns
    -------
    result : OptimizationResult
        Completed optimization result.
    """
    parser = argparse.ArgumentParser(description='exoTEDRF v2 SOSS optimizer')
    parser.add_argument('--config', default='run_optimize.yaml',
                        help='v1-compatible run_optimize YAML')
    parser.add_argument('--output-dir', default=None,
                        help='explicit v2 output root (default: configured '
                             'pipeline_outputs_directory[_output_tag]/v2)')
    parser.add_argument('--no-products', action='store_true',
                        help=('write optimizer logs but skip final FITS, '
                              'PNG, CSV, and NPY products'))
    parser.add_argument('--output-mode', choices=['standard', 'optimal'],
                        default=None, help='optimal saves final spectra, '
                        'optimizer JSON and bounded diagnostic PNGs; large '
                        'arrays use scratch as needed during processing')
    parser.add_argument('--x64', action='store_true',
                        help='enable JAX float64 for strict validation')
    parser.add_argument('--archive', action='store_true',
        help='after a normal run, move input_dir and the outputs to '
             'archive_to_longterm_storage (verified copy before deletion)')
    parser.add_argument('--checkpoint-npz', default=None,
        help='optional bounded validation bundle (final spectra, cost, scatter, and winners)')
    parser.add_argument('--checkpoint-steps', action='append', default=None,
        help='after-step arrays to add to the bundle; repeat or use commas '
             '(large detector steps can consume substantial host RAM/disk)')
    parser.add_argument('--search', choices=['greedy', 'joint', 'beam', 'tree'], default=None,
        help='override v2_search_strategy from the YAML (default: greedy, '
             'v1-parity coordinate descent)')
    parser.add_argument('--beam-width', type=int, default=None,
        help='override v2_beam_width from the YAML (beam search only; joint forces width 1)')
    parser.add_argument('--final-candidates', type=int, default=None,
        help='override v2_final_candidates from the YAML (number of '
             'phase-1 finalists scored on the full data in phase 2)')
    parser.add_argument('--tree-tolerance', type=float, default=None,
        help='override v2_tree_tolerance from the YAML (tree search only; '
             'relative phase-1 cost gap within which branches survive)')
    args = parser.parse_args(argv)

    from exotedrf.v2 import core
    core.setup()
    checkpoint_steps = None
    if args.checkpoint_steps is not None:
        checkpoint_steps = tuple(name.strip()
            for group in args.checkpoint_steps for name in group.split(',') if name.strip())
        if not checkpoint_steps:
            parser.error('--checkpoint-steps did not contain a step name')

    config_arg = args.config
    override_names = {'output_mode': 'v2_output_mode', 'search': 'v2_search_strategy',
                      'beam_width': 'v2_beam_width', 'final_candidates': 'v2_final_candidates',
                      'tree_tolerance': 'v2_tree_tolerance'}
    cli_overrides = {key: getattr(args, arg) for arg, key in override_names.items()
                     if getattr(args, arg) is not None}
    if args.archive:
        cli_overrides['v2_archive'] = True
    if cli_overrides:
        config_arg = dict(v2config.load_config(args.config))
        config_arg.update(cli_overrides)

    result = run_optimizer(config_arg, output_dir=args.output_dir,
        write_products=not args.no_products, enable_x64_mode=args.x64,
        checkpoint_path=args.checkpoint_npz, checkpoint_steps=checkpoint_steps)
    print(json.dumps({'winners': result.winners, 'final_cost': result.final_cost,
                      'output_paths': result.output_paths}, indent=2, default=v2products.jsonable))
    return result


if __name__ == '__main__':
    main()


__all__ = ['OptimizationResult', 'TrialResult', 'decision_sensitivity', 'main',
    'run_optimizer', 'select_first_finite_minimum']
