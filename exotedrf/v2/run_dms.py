"""Run a fixed reduction recipe from a run_DMS YAML configuration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from exotedrf.v2 import config as v2config
from exotedrf.v2.config import _log

# Use the v1 stage step lists.
DMS_STAGE1_STEPS = ('DQInitStep', 'INLCorrStep', 'EmiCorrStep', 'SuperBiasStep',
    'RefPixStep', 'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep',
    'JumpStep', 'RampFitStep', 'GainScaleStep')
DMS_STAGE2_STEPS = ('AssignWCSStep', 'FlatFieldStep', 'OneOverFStep_int', 'BackgroundStep',
    'BadPixStep', 'PCAReconstructStep')

# Use the v1 reduction defaults for omitted parameters.
V1_PARAM_DEFAULTS = {'soss_inner_mask_width': 40, 'soss_outer_mask_width': 70,
    'nirspec_mask_width': 16, 'time_jump_threshold': 10, 'time_window': 5, 'miri_trace_width': 20,
    'miri_background_width': 14, 'space_outlier_threshold': 15, 'time_outlier_threshold': 10,
    'box_size': 5, 'window_size': 5, 'extract_width': 40}


def normalize_dms_config(cfg, *, logger=print):
    """Normalize the fixed recipe and requested pipeline stages.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    logger : None, callable
        Function to receive progress and warning messages.

    Returns
    -------
    cfg : dict
        Configuration with excluded stages skipped.
    run_stages : list[int]
        Sorted pipeline stages to run.
    deepframe : None, str, np.ndarray(float)
        Stage-3 tracing frame override.
    """
    cfg = dict(cfg)
    # Accept both INL amplitude spellings, giving the v1 key precedence.
    if 'inl_amplitudes_file' in cfg:
        value = cfg.pop('inl_amplitudes_file')
        if cfg.get('inl_amplitude_file') not in (None, value):
            _log(logger, '[v2] WARNING: both inl_amplitudes_file and '
                         'inl_amplitude_file are set; using '
                         'inl_amplitudes_file, the key v1 run_DMS.py reads')
        else:
            _log(logger, '[v2] WARNING: inl_amplitudes_file is v1 run_DMS.py\'s '
                         'misspelling of inl_amplitude_file; accepted')
        cfg['inl_amplitude_file'] = value
    elif 'inl_amplitude_file' in cfg:
        _log(logger, '[v2] WARNING: v1 run_DMS.py reads inl_amplitudes_file '
                     '(misspelled) and raises KeyError for this YAML; v2 uses '
                     'inl_amplitude_file')

    run_stages = cfg.get('run_stages', [1, 2, 3])
    if isinstance(run_stages, (int, np.integer)):
        run_stages = [int(run_stages)]
    if not isinstance(run_stages, (list, tuple)) or not run_stages or any(
            isinstance(stage, bool) or stage not in (1, 2, 3) for stage in run_stages):
        raise ValueError(f'run_stages must list stages from [1, 2, 3], got {run_stages!r}')
    run_stages = sorted({int(stage) for stage in run_stages})
    if 1 not in run_stages:
        cfg.update({name: 'skip' for name in DMS_STAGE1_STEPS})
    if 2 not in run_stages:
        cfg.update({name: 'skip' for name in DMS_STAGE2_STEPS})
    deepframe = cfg.pop('deepframe', None)
    for key in list(cfg):
        if key.startswith('optimize_') and cfg[key]:
            raise ValueError(f'{key}=True: run_dms runs one fixed recipe; use '
                'python -m exotedrf.v2.optimize for parameter sweeps')
    width = cfg.get('extract_width')
    if isinstance(width, str) and width.strip().lower() == 'optimize':
        raise NotImplementedError("extract_width: 'optimize' (v1 box_extract_soss 10-60 px "
            'white-light search) is not implemented by v2 run_dms; use the '
            'v2 optimizer with optimize_extract_width')
    for name in V1_PARAM_DEFAULTS:
        if isinstance(cfg.get(name), list):
            raise ValueError(f'{name} must be a single value for run_dms')
    for name in ('save_results', 'force_redo', 'do_plots'):
        if name in cfg and not isinstance(cfg[name], (bool, np.bool_)):
            raise TypeError(f'{name} must be a boolean')
    return cfg, run_stages, deepframe


def output_root(cfg, output_dir=None):
    """Get the tagged reduction output directory.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    output_dir : None, str
        Override for the output directory.

    Returns
    -------
    root : Path
        Reduction output directory.
    """
    from exotedrf.v2 import products
    if output_dir is not None:
        return Path(os.path.expanduser(os.fspath(output_dir)))
    return products.v1_output_root(
        cfg.get('pipeline_outputs_directory', 'pipeline_outputs_directory'),
        cfg.get('output_tag', ''))


def save_config(config_file, root):
    """Copy the YAML configuration and append the run time.

    Parameters
    ----------
    config_file, root : str
        Source YAML path and output directory.

    Returns
    -------
    path : Path
        Saved configuration path, numbered if the filename already exists.
    """
    directory = Path(root) / 'config_files'
    directory.mkdir(parents=True, exist_ok=True)
    name = os.path.basename(os.fspath(config_file))
    copy = directory / name
    index = 0
    while copy.exists():
        index += 1
        copy = directory / f"{name.split('.yaml')[0]}_{index}.yaml"
    shutil.copy(config_file, copy)
    stamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=' ', timespec='minutes')
    with open(copy, 'a') as stream:
        stream.write(f'\nRun at {stamp}.')
    return copy


def _params(cfg):
    """Resolve fixed reduction parameters with v1 defaults."""
    params = {name: cfg.get(name, default) for name, default in V1_PARAM_DEFAULTS.items()}
    params['extract_width_soss2'] = cfg.get('extract_width_soss2')
    return params


def _resume_from_cache(opts, files, paths, *, logger=None):
    """Find the last complete saved step, preferring Stage-2 products."""
    from exotedrf.v2 import io, restart
    if opts.get('force_redo', False) or not opts.get('save_results', True):
        return None
    names = restart._graph_step_names(opts['mode'], opts)
    roots = io.v1_fileroots(files)
    instrument = str(opts['mode']).split('/')[0].upper()
    best = None
    for stage_key, stage_steps in (('stage1', restart.STAGE1_STEPS),
                                   ('stage2', restart.STAGE2_STEPS)):
        stage_names = [name for name in names if name in stage_steps]
        hit = restart.cached_resume_point(stage_names, paths[stage_key], roots,
            save_results=opts.get('save_results', True), instrument=instrument,
            remove_components=opts.get('remove_components'))
        if hit is not None:
            name = stage_names[hit[0]]
            best = (name, [os.fspath(Path(paths[stage_key]) / f'{root}{hit[1]}.fits')
                           for root in roots])
    if best is not None:
        _log(logger, f'[v2] force_redo=False: v1 cached {best[0]} products '
                     f'found; resuming after it ({len(best[1])} file(s))')
    return best


def run_dms(config_or_path, *, output_dir=None, logger=print,
            enable_x64_mode=False, refpack=None, config_file=None):
    """Run the configured stages once and save reduction products.

    Parameters
    ----------
    config_or_path : dict, str
        Reduction configuration or path to a YAML file.
    output_dir : None, str
        Override for the output directory.
    logger : None, callable
        Function to receive progress and warning messages.
    enable_x64_mode : bool
        If True, enable JAX float64 calculations.
    refpack : None, str
        Path to a prebuilt reference pack.
    config_file : None, str
        Path to the YAML file to copy into the output directory.

    Returns
    -------
    result : dict
        Final state, parameters, product paths, summary and output layout.
    """
    from exotedrf.v2 import (core, io, optimize, products, restart, stages)
    from exotedrf.v2.pipeline import PipelineState
    started = time.perf_counter()
    raw = dict(v2config.coerce_config(config_or_path))
    if config_file is None and isinstance(config_or_path, (str, os.PathLike)):
        config_file = config_or_path
    cfg, run_stages, deepframe_cfg = normalize_dms_config(raw, logger=logger)
    # Set the CRDS cache and server as in run_DMS.
    if cfg.get('crds_cache_path') is not None:
        os.environ['CRDS_PATH'] = os.fspath(cfg['crds_cache_path'])
    os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'
    optimize._configure_crds_environment(cfg)
    v2config.validate_supported_config(cfg)
    if enable_x64_mode or cfg.get('v2_enable_x64', False):
        core.enable_x64()

    root = output_root(cfg, output_dir)
    paths = products.output_layout(cfg, output_dir=root, create=True)
    if config_file is not None:
        save_config(config_file, root)

    opts = v2config.fixed_options(cfg)
    opts['save_results'] = bool(cfg.get('save_results', True))
    opts['force_redo'] = bool(cfg.get('force_redo', False))
    opts['do_plots'] = bool(cfg.get('do_plots', False))
    opts['dtype'] = (np.float64 if enable_x64_mode or
                     cfg.get('v2_enable_x64', False) else np.float32)
    # Keep the reference cache under the output directory.
    opts['output_dir'] = os.fspath(root.parent)
    opts['output_tag'] = root.name
    if output_dir is None:
        opts['output_dir'] = cfg.get('pipeline_outputs_directory', 'pipeline_outputs_directory')
        opts['output_tag'] = cfg.get('output_tag', '')
    mode = opts['mode']
    params = _params(cfg)
    tag = str(opts.get('input_filetag') or 'uncal')

    # Find the configured input products.
    files = io.find_products(opts['input_dir'], tag, mode=mode,
                             filter_detector=opts.get('filter_detector'))
    if not files:
        raise RuntimeError(f'No FITS found in {opts["input_dir"]} matching '
                           f'input_filetag={tag!r}')
    label = '' if str(mode).upper() == 'MIRI/LRS' else opts.get('filter_detector', '')
    _log(logger, f'[v2] Identified {len(files)} {label} {mode} observation segment(s)')
    for path in files:
        _log(logger, f'[v2]  {path}')

    stage3 = 3 in run_stages
    cached_stage3 = None
    if stage3 and not opts['force_redo']:
        # Reuse an existing spectrum when force_redo is False.
        candidates = sorted(Path(paths['stage3']).glob(
            f"*_{opts['extract_method']}_spectra_fullres.fits"))
        if candidates:
            cached_stage3 = str(candidates[0])

    resume = _resume_from_cache(opts, files, paths, logger=logger)
    if resume is not None:
        name, cached_files = resume
        order = list(restart.STAGE1_STEPS + restart.STAGE2_STEPS)
        for step in order[:order.index(name) + 1]:
            if step in opts or step in v2config.STEP_DEFAULTS:
                opts[step] = 'skip'
        files = cached_files
        opts['input_filetag'] = restart.STEP_TAGS[name]

    loading = time.perf_counter()
    if opts['input_filetag'] == 'uncal' and resume is None:
        cube = io.load_ramp_cube(files, baseline_ints=opts['baseline_ints'], mode=mode,
            filter_detector=opts.get('filter_detector'), science_dtype=opts['dtype'],
            max_host_bytes=opts.get('max_host_bytes'), scratch_dir=opts.get('scratch_dir'))
        state = PipelineState(cube=cube)
        refpack_path = refpack
        opts['stream_stage1'] = _streaming(cfg, opts, files, run_stages)
    else:
        state, files = restart.load_restart_state(opts, files=files, logger=logger)
        opts = restart.restart_opts(opts, state.cube)
        refpack_path = refpack if refpack is not None else \
            restart.restart_refpack(state.cube, opts, files, logger=logger)
    if resume is not None and resume[0] == 'BackgroundStep_grp':
        noseg = io.v1_fileroot_noseg(io.v1_fileroots(files))
        background = np.load(Path(paths['stage1']) / f'{noseg}background.npy', allow_pickle=False)
        state = PipelineState(cube=state.cube, aux=dict(state.aux, bkg_grp=background))

    ctx = stages.prepare_context(state.cube, opts, refpack=refpack_path)
    if opts['save_results'] and not opts['force_redo'] and \
            opts.get('BadPixStep', v2config.STEP_DEFAULTS['BadPixStep']) == 'run':
        reused = restart.reusable_hot_pixels(
            paths['stage2'], io.v1_fileroots(files), save_results=True)
        if reused is not None:
            _log(logger, '[v2] BadPixStep: reusing the cached hot_pixels.npy '
                         'map (v1 force_redo=False behaviour)')
            ctx['badpix_reuse_map'] = reused
    pipeline = stages.build_pipeline(mode, ctx)
    load_seconds = time.perf_counter() - loading

    if resume is not None and resume[0] == 'BackgroundStep_grp':
        # Omit group background subtraction already present in the cached product.
        from exotedrf.v2.pipeline import Pipeline
        pipeline = Pipeline([step for step in pipeline.steps if step.name != 'BackgroundStep_grp'],
                            mode, ctx)

    capture = None
    if opts['do_plots'] and opts['save_results']:
        capture = products.CompatibilityCapture(paths, ctx, params=params)
    graph_stop = len(pipeline.steps) - 1
    _log(logger, '[v2] running: ' + ', '.join(step.name for step in pipeline.steps[:graph_stop]))
    running = time.perf_counter()
    # Use segment scheduling for a complete built-in reduction.
    streamed = bool(opts.get('stream_stage1', False) and stage3 and
                    cached_stage3 is None and deepframe_cfg is None)
    if streamed:
        state = optimize._run_final_graph(pipeline, [state], params, None, capture, logger)
    elif graph_stop > 0:
        state = pipeline.run(state, params, stop=graph_stop, observer=capture)
    calibrate_seconds = time.perf_counter() - running

    written = {}
    if deepframe_cfg is not None:
        # Apply the configured Stage-3 deepframe override.
        from astropy.io import fits
        frame = (fits.getdata(deepframe_cfg) if isinstance(deepframe_cfg, (str, os.PathLike))
                 else deepframe_cfg)
        state = PipelineState(cube=state.cube, aux=dict(
            state.aux, stage3_deepframe=np.asarray(frame, dtype=float)))
    extract_seconds = 0.
    if stage3 and cached_stage3 is not None:
        _log(logger, f'[v2] File {cached_stage3} already exists; skipping '
                     'Stage 3 (force_redo=False)')
        written['spectra'] = cached_stage3
        stage3 = False
    elif stage3 and not streamed:
        has_deep = (state.aux.get('stage3_deepframe') is not None or
                    state.aux.get('deepframe') is not None)
        if not has_deep and opts['extract_method'] in ('box', 'optimal'):
            # Require a deepframe even when centroids are supplied.
            raise ValueError('Deepframe must be provided for box extraction.')
        extracting = time.perf_counter()
        state = pipeline.steps[-1].fn(state, params, ctx)
        extract_seconds = time.perf_counter() - extracting

    if opts['save_results'] or stage3:
        written.update(_write_products(state, params, ctx, paths, files,
                                       run_stages, stage3, opts, logger))
    if capture is not None:
        written.update(capture.render(state))
    if stage3 and opts['do_plots']:
        plots = products._plot_centroids(_centroid_plot_path(paths, state), state, params, ctx) \
            if ctx.get('centroids') is None else False
        if plots:
            written['centroid_plot'] = str(_centroid_plot_path(paths, state))

    total = time.perf_counter() - started
    summary = products.jsonable({'schema_version': 1, 'mode': mode, 'run_stages': run_stages,
        'input_products': list(files), 'resumed_after': None if resume is None else resume[0],
        'params': params, 'steps': [step.name for step in pipeline.steps[:graph_stop]] +
                 (['Extract'] if stage3 else []), 'products': written,
        'timing_s': {'load_and_context': load_seconds, 'calibration': calibrate_seconds,
                     'extraction': extract_seconds, 'total': total}})
    summary_path = root / f'run_dms_summary{_suffix(cfg)}.json'
    with summary_path.open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    written['summary'] = str(summary_path)
    _log(logger, f'[v2] run_dms complete in {total:.1f}s')
    return {'state': state, 'params': params, 'products': written,
            'summary': summary, 'paths': {k: str(v) for k, v in paths.items()}}


def _suffix(cfg):
    """Get the optional name-tag suffix."""
    name = cfg.get('name_tag')
    return f'_{name}' if name else ''


def _streaming(cfg, opts, files, run_stages):
    """Select segment scheduling for a complete raw-ramp reduction."""
    from exotedrf.v2 import core, io, optimize
    requested = opts.get('stream_stage1', 'auto')
    if requested is False or run_stages != [1, 2, 3] or not v2config.supports_stage1_streaming(cfg):
        return False
    if requested is True:
        return True
    required = io.estimate_ramp_storage_bytes(files, opts['dtype'])
    return bool(optimize._stream_without_memory_pressure() or 3 * required >
                core.host_allocation_budget_bytes(opts.get('max_host_bytes')))


def _centroid_plot_path(paths, state):
    """Get the instrument-specific centroid plot path."""
    path = Path(paths['centroid_plot'])
    meta = state.cube.meta
    if str(meta.mode).split('/')[0].upper() == 'NIRSPEC':
        path = path.with_name(f'centroiding_{str(meta.detector).lower()}.png')
    return path


def _write_products(state, params, ctx, paths, files, run_stages, stage3, opts, logger):
    """Write stage products, spectra and reusable sidecars."""
    from exotedrf.v2 import io, products, restart
    written = {}
    cube = state.cube
    meta = cube.meta
    stage_key = 'stage2' if 2 in run_stages else 'stage1'
    if opts['save_results'] and (1 in run_stages or 2 in run_stages):
        last = [step for step in restart._graph_step_names(opts['mode'], opts) if step != 'Extract']
        tag = restart.STEP_TAGS.get(last[-1]) if last else None
        segment_files = []
        if tag is not None:
            segment_files = restart.write_segment_products(
                cube, files, paths[stage_key], tag, logger=logger)
        if segment_files:
            written[tag] = segment_files
        elif cube.data.ndim == 3:
            prefix = products._v1_fileroot_noseg(meta)
            rate_path = Path(paths[stage_key]) / f'{prefix}rateints_v2.fits'
            io.save_rate_cube(cube, rate_path, extra_header={'OPTIMIZE': False})
            written['rate'] = str(rate_path)
    if stage3:
        written.update(products.write_final_products(
            state, params, ctx, dict(paths), output_mode='optimal'))
    if opts['save_results']:
        sidecar_ctx = dict(ctx)
        sidecars = products._write_reusable_sidecars(state, sidecar_ctx, paths)
        centroid_file = sidecars.pop('centroids', None)
        if centroid_file is not None and (not stage3 or ctx.get('centroids') is not None):
            # Save centroid tables only when Stage 3 traces them.
            os.unlink(centroid_file)
        elif centroid_file is not None:
            written['centroids'] = centroid_file
        # Remove sidecars belonging to stages excluded from this reduction.
        dropped = set()
        if 2 not in run_stages:
            dropped |= {'deepframe', 'hot_pixels', 'background', 'stability'}
        if 1 not in run_stages and 2 not in run_stages:
            dropped.add('contaminant_mask')
        for key in dropped & set(sidecars):
            os.unlink(sidecars.pop(key))
        written.update(sidecars)
        if 2 in run_stages and opts.get('generate_lc') is True and \
                state.aux.get('pca_wlc') is not None:
            prefix = products._v1_fileroot_noseg(meta)
            lc_path = Path(paths['stage2']) / f'{prefix}lcestimate.npy'
            np.save(lc_path, np.asarray(state.aux['pca_wlc']), allow_pickle=False)
            written['lcestimate'] = str(lc_path)
    return written


def main(argv=None):
    """Run the fixed-recipe command-line interface.

    Parameters
    ----------
    argv : None, list[str]
        Command-line arguments. If None, read the process arguments.

    Returns
    -------
    result : dict
        Completed reduction result.
    """
    parser = argparse.ArgumentParser(description='exoTEDRF v2 single-pass reduction (run_DMS.yaml)')
    parser.add_argument('config_positional', nargs='?', default=None,
                        help='run_DMS YAML (v1 positional form)')
    parser.add_argument('--config', '-c', default=None, help='run_DMS YAML')
    parser.add_argument('--output-dir', default=None, help='explicit output root (default: '
                             'pipeline_outputs_directory[_output_tag])')
    parser.add_argument('--x64', action='store_true',
                        help='enable JAX float64 for strict validation')
    args = parser.parse_args(argv)
    config = args.config or args.config_positional
    if config is None:
        parser.error('Config file must be provided')
    from exotedrf.v2 import core
    core.setup()
    result = run_dms(config, output_dir=args.output_dir, enable_x64_mode=args.x64)
    from exotedrf.v2 import products
    print(json.dumps({'products': result['products']}, indent=2, default=products.jsonable))
    print('[v2] Done')
    return result


if __name__ == '__main__':
    main()


__all__ = ['DMS_STAGE1_STEPS', 'DMS_STAGE2_STEPS', 'main',
           'normalize_dms_config', 'output_root', 'run_dms', 'save_config']
