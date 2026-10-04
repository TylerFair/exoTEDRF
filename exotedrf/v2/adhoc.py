"""Rerun PCA and spectral extraction from saved reduction products."""

from __future__ import annotations

import ast
import glob
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

from exotedrf.v2 import config as v2config
from exotedrf.v2.config import _log


def is_null_like(value):
    """Check for a v1 null value."""
    return value in [None, 'None', 'none', 'null', 'NULL', '']


def _null_like_width(value):
    """Check recognized null widths without comparing array-like apertures."""
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return False
    return value in [None, 'None', 'null', '']


def parse_extract_width_metadata(width_value):
    """Parse an aperture width from saved metadata.

    Parameters
    ----------
    width_value : object
        Aperture width from YAML, a FITS header or a cost log.

    Returns
    -------
    width : None, float, dict, list, str
        Parsed width or unchanged unrecognized text.
    """
    if _null_like_width(width_value):
        return None
    if isinstance(width_value, dict):
        return width_value
    if isinstance(width_value, str):
        text = width_value.strip()
    elif np.isscalar(width_value):
        return width_value
    else:
        text = None
    if isinstance(width_value, (list, tuple)):
        if len(width_value) == 2:
            return {'lower': float(width_value[0]), 'upper': float(width_value[1])}
        return list(width_value)
    if text is None:
        text = str(width_value).strip()
    if text in ['', 'None', 'null']:
        return None

    if (text.startswith('{') and text.endswith('}')) or (
            text.startswith('[') and text.endswith(']')):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, dict) and text.startswith('{'):
            return parsed
        if isinstance(parsed, (list, tuple)) and text.startswith('['):
            if len(parsed) == 2:
                return {'lower': float(parsed[0]), 'upper': float(parsed[1])}
            return list(parsed)

    match = re.fullmatch(r'lower\s*=\s*([-+]?\d*\.?\d+)\s*,\s*upper\s*=\s*([-+]?\d*\.?\d+)', text)
    if match:
        return {'lower': float(match.group(1)), 'upper': float(match.group(2))}

    try:
        scalar = float(text)
    except ValueError:
        return text

    if scalar.is_integer():
        return int(scalar)
    return scalar


def format_log_value(value):
    """Format a parameter value for the v1 cost log.

    Parameters
    ----------
    value : object
        Value to format or check.

    Returns
    -------
    text : str
        Value with v1 list and missing-value formatting.
    """
    import pandas as pd
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(str(v) for v in value) + ']'
    if value is None:
        return 'None'
    if isinstance(value, dict):
        return str(value)
    if pd.isna(value):
        return ''
    return str(value)


def prepare_cost_log(cost_path, required_param_cols):
    """Create or extend a cost log with the required columns.

    Parameters
    ----------
    cost_path : str
        Path to the tab-separated cost log.
    required_param_cols : list[str]
        Parameter columns required for the new trials.

    Returns
    -------
    param_cols : list[str]
        Parameter columns in log order.
    row_offset : int
        Number of existing rows.
    best_logged : dict
        Parameter values from the lowest-cost existing row.
    """
    import pandas as pd
    cost_path = Path(cost_path)
    if cost_path.exists() and cost_path.stat().st_size > 0:
        df = pd.read_csv(cost_path, sep='\t', keep_default_na=False)
    else:
        df = pd.DataFrame()

    existing_param_cols = [c for c in df.columns if c not in ['duration_s', 'cost']]
    param_cols = existing_param_cols.copy()
    for col in required_param_cols:
        if col not in param_cols:
            param_cols.append(col)

    if df.empty:
        df = pd.DataFrame(columns=param_cols + ['duration_s', 'cost'])
    else:
        for col in param_cols + ['duration_s', 'cost']:
            if col not in df.columns:
                df[col] = ''
        df = df[param_cols + ['duration_s', 'cost']]

    cost_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cost_path, sep='\t', index=False)

    best_logged = {}
    if len(df) > 0:
        numeric_cost = pd.to_numeric(df['cost'], errors='coerce')
        if numeric_cost.notna().any():
            best_logged = df.loc[numeric_cost.idxmin(), param_cols].to_dict()
    return param_cols, len(df), best_logged


def append_cost_log_row(cost_path, param_cols, row_values, duration_s, cost):
    """Append one trial to the cost log.

    Parameters
    ----------
    cost_path : str
        Path to the tab-separated cost log.
    param_cols : list[str]
        Parameter columns in log order.
    row_values : dict
        Parameter values for the trial row.
    duration_s, cost : float
        Trial duration in seconds and light-curve scatter cost.
    """
    fields = [format_log_value(row_values.get(col, '')) for col in param_cols]
    fields.extend([f'{duration_s:.1f}', f'{cost:.12f}'])
    with open(cost_path, 'a') as logf:
        logf.write('\t'.join(fields) + '\n')


def append_scatter_log_row(scatter_path, scatter):
    """Append one trial to the scatter log.

    Parameters
    ----------
    scatter_path : str
        Path to the scatter log.
    scatter : array-like(float)
        Light-curve scatter values.
    """
    with open(scatter_path, 'a') as logs:
        logs.write(' '.join(f'{x:.10g}' for x in np.ravel(scatter)) + '\n')


def find_best_logged_extract_width(cost_paths):
    """Read the best aperture from the first existing cost log.

    Parameters
    ----------
    cost_paths : list[str]
        Candidate cost-log paths in search order.

    Returns
    -------
    width : float, dict, list, str
        Parsed extraction width from the lowest-cost valid row.
    """
    import pandas as pd
    if isinstance(cost_paths, (str, os.PathLike)):
        cost_paths = [cost_paths]
    cost_path = next((Path(path) for path in cost_paths if Path(path).exists()), None)
    if cost_path is None:
        raise FileNotFoundError(f'No optimizer cost log found at {Path(cost_paths[0])}')

    df = pd.read_csv(cost_path, sep='\t', keep_default_na=False)
    if 'extract_width' not in df.columns or 'cost' not in df.columns:
        raise ValueError(f'{cost_path} does not contain extract_width and cost columns.')

    cost = pd.to_numeric(df['cost'], errors='coerce')
    valid = cost.notna() & (df['extract_width'].astype(str).str.strip() != '')
    if not bool(valid.any()):
        raise ValueError(f'{cost_path} does not contain any valid logged extract_width values.')

    best_idx = cost[valid].idxmin()
    width = parse_extract_width_metadata(df.loc[best_idx, 'extract_width'])
    if _null_like_width(width):
        raise ValueError(f'Could not parse extract_width from best row of {cost_path}')
    return width


def find_stage3_spectrum_file(stage3_dirs, extract_method):
    """Find the first spectrum for the requested extraction method.

    Parameters
    ----------
    stage3_dirs : list[str]
        Candidate Stage-3 directories in search order.
    extract_method : str
        Spectral extraction method.

    Returns
    -------
    path : str
        Path to the matching Stage-3 spectrum.
    """
    for directory in stage3_dirs:
        pattern = os.path.join(os.fspath(directory), f'*_{extract_method}_spectra_fullres.fits')
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    pattern = os.path.join(os.fspath(stage3_dirs[0]), f'*_{extract_method}_spectra_fullres.fits')
    raise FileNotFoundError(f'No Stage 3 spectrum file found matching {pattern}')


def resolve_ad_hoc_extract_width(cfg, *, stage3_dirs, cost_paths, logger=None):
    """Resolve the aperture for a Stage-3 rerun.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    stage3_dirs, cost_paths : list[str]
        Spectrum directories and cost-log paths in search order.
    logger : None, callable
        Function to receive progress and warning messages.

    Returns
    -------
    width : float, dict, list
        Configured aperture or first-pass width from a spectrum or cost log.
    """
    if cfg.get('optimize_extract_width', False):
        return cfg.get('extract_width')

    width = cfg.get('extract_width')
    if not _null_like_width(width):
        return width

    if cfg.get('reuse_first_pass_extract_width', False):
        from astropy.io import fits
        source_method = cfg.get('first_pass_extract_method', 'box')
        try:
            specfile = find_stage3_spectrum_file(stage3_dirs, source_method)
        except FileNotFoundError:
            if logger is not None:
                logger('[v2] No first-pass Stage 3 spectrum file found; '
                       'falling back to optimizer cost log for ' 'extract_width.')
            width = find_best_logged_extract_width(cost_paths)
            if logger is not None:
                logger(f'[v2] Reusing best logged extract_width: {width}')
            return width
        header = fits.getheader(specfile)
        width = parse_extract_width_metadata(header.get('WIDTH'))
        if _null_like_width(width):
            raise ValueError(f'WIDTH header missing or unreadable in {specfile}')
        if logger is not None:
            logger(f'[v2] Reusing first-pass extract_width from {specfile}: {width}')
        return width

    raise ValueError('No extract_width specified for the Stage-3 rerun. Set '
        'extract_width, or set reuse_first_pass_extract_width=True to read '
        'it from an existing first-pass Stage 3 box spectrum.')


def ad_hoc_widths(cfg):
    """Get aperture candidates for an ad-hoc reduction.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.

    Returns
    -------
    widths : list
        Ordered candidates, or the single fixed aperture.
    """
    if cfg.get('optimize_extract_width', False):
        widths = cfg['extract_width']
        if not isinstance(widths, list):
            raise ValueError('extract_width must be a list when optimize_extract_width=True')
        return list(widths)
    widths = cfg.get('extract_width')
    if isinstance(widths, list):
        return [widths[0]]
    return [widths]


def find_existing_stage2_outputs(patterns, error_message, logger=None):
    """Find the first matching set of Stage-2 FITS products.

    Parameters
    ----------
    patterns : list[str]
        FITS glob patterns in search order.
    error_message : str
        Error message if no pattern matches.
    logger : None, callable
        Function to receive progress and warning messages.

    Returns
    -------
    files : list[str]
        Sorted matching paths.
    """
    for pattern in patterns:
        found = sorted(glob.glob(pattern))
        if found:
            if logger is not None:
                logger(f'[v2] Found {len(found)} file(s) matching: {pattern}')
            return found
    raise FileNotFoundError(error_message)


def _first_glob(directories, suffix):
    """Find the first sorted suffix match in directory search order."""
    for directory in directories:
        if directory is None:
            continue
        found = sorted(glob.glob(os.path.join(os.fspath(directory), f'*{suffix}')))
        if found:
            return found[0]
    return None


def load_ad_hoc_centroids(cfg, centroid_dirs, logger=None):
    """Get configured centroids or the first saved centroid table.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    centroid_dirs : list[str]
        Candidate centroid directories in search order.
    logger : None, callable
        Function to receive progress and warning messages.

    Returns
    -------
    centroids : None, str, dict
        Centroid path or data, or None to trace the deepframe.
    """
    configured = cfg.get('centroids')
    if configured not in [None, 'None', 'null', '']:
        if logger is not None:
            logger(f'[v2] Using centroids from config: {configured}')
        return configured
    found = _first_glob(centroid_dirs, 'centroids.csv')
    if found is not None and logger is not None:
        logger(f'[v2] Loading centroids from {found}')
    if found is None and logger is not None:
        logger('[v2] No centroid table found in config, Stage 3, or Stage 2. '
               'Stage 3 will trace centroids from the deepframe.')
    return found


def resolve_ad_hoc_deepframe(cfg, stage2_dir, logger=None):
    """Get the configured or first saved Stage-2 deepframe.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    stage2_dir : str
        Stage-2 output directory.
    logger : None, callable
        Function to receive progress and warning messages.

    Returns
    -------
    deepframe : None, str
        Deepframe path, if available.
    """
    deepframe = cfg.get('deepframe')
    if deepframe not in [None, 'None', 'null', '']:
        return deepframe
    found = _first_glob([stage2_dir], 'deepframe.fits')
    if found is not None and logger is not None:
        logger(f'[v2] Using deepframe from: {found}')
    return found


def ad_hoc_directories(cfg, paths):
    """Get the v1 and v2 directories for saved products and logs.

    Parameters
    ----------
    cfg, paths : dict
        Reduction configuration and product paths.

    Returns
    -------
    directories : dict
        Candidate Stage-2, Stage-3 and log directories.
    """
    v1_root = Path(os.path.expanduser(os.fspath(cfg.get(
        'pipeline_outputs_directory', 'pipeline_outputs_directory'))))
    return {'v1_stage2': v1_root / 'Stage2', 'v1_stage3': v1_root / 'Stage3',
        'v1_files': v1_root / 'Files', 'v2_stage2': Path(paths['stage2']),
        'v2_stage3': Path(paths['stage3']), 'v2_files': Path(paths['files'])}


def extract_only_source_dir(cfg, dirs, logger=None):
    """Find the directory containing BadPix or PCA products.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    dirs : dict
        Candidate v1 and v2 output directories.
    logger : None, callable
        Function to receive progress and warning messages.

    Returns
    -------
    directory : Path
        Selected Stage-2 product directory.
    """
    def has_products(directory):
        """Check for saved BadPix or PCA FITS products."""
        return bool(glob.glob(os.path.join(directory, '*_badpixstep.fits')) or
                    glob.glob(os.path.join(directory, '*_pcareconstructstep.fits')))

    input_dir = cfg.get('input_dir')
    if input_dir not in [None, 'None', 'null', '']:
        for candidate in (input_dir, os.path.join(input_dir, 'Stage2')):
            candidate = os.path.expanduser(os.fspath(candidate))
            if os.path.isdir(candidate) and has_products(candidate):
                if logger is not None:
                    logger(f'[v2] Detected Stage 2 outputs in input_dir: {candidate}')
                return Path(candidate)
    for candidate in (dirs['v2_stage2'], dirs['v1_stage2']):
        if candidate.is_dir() and has_products(os.fspath(candidate)):
            return candidate
    return dirs['v1_stage2']


def stage2_patterns(cfg, source_dir):
    """Order Stage-2 product patterns for the configured PCA settings.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    source_dir : str
        Directory containing the saved Stage-2 products.

    Returns
    -------
    patterns : list[str]
        BadPix and PCA FITS patterns in search order.
    """
    source = os.fspath(source_dir)
    pca_step = cfg.get('PCAReconstructStep', 'run')
    remove = cfg.get('remove_components')
    no_components = (remove is None or remove in ['None', 'null', ''] or
                     (isinstance(remove, list) and not remove))
    badpix = os.path.join(source, '*_badpixstep.fits')
    pca = os.path.join(source, '*_pcareconstructstep.fits')
    if pca_step == 'skip' or no_components:
        return [badpix, pca]
    return [pca, badpix]


def _seed_logs(paths, dirs, name):
    """Copy existing v1 logs when the current run has no logs."""
    for key, prefix in (('cost', 'Cost_'), ('scatter', 'Scatter_')):
        target = Path(paths[key])
        source = dirs['v1_files'] / f'{prefix}{name}.txt'
        if not target.exists() and source.is_file() and source.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


_SKIPPED_STEPS = ('DQInitStep', 'INLCorrStep', 'EmiCorrStep', 'ResetStep', 'SuperBiasStep',
    'RefPixStep', 'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep',
    'JumpStep', 'RampFitStep', 'GainScaleStep', 'AssignWCSStep',
    'Extract2DStep', 'SourceTypeStep', 'WaveCorrStep', 'FlatFieldStep',
    'BackgroundStep', 'OneOverFStep_int', 'BadPixStep', 'PCAReconstructStep')


def _scalar_width(width):
    """Check whether an aperture is a scalar numeric value."""
    try:
        return np.ndim(width) == 0 and not isinstance(width, (dict, str))
    except TypeError:
        return False


def score_extract_widths(pipeline, state, params, widths, *, logger=None):
    """Score apertures on the state immediately before extraction.

    Parameters
    ----------
    pipeline : Pipeline
        Ordered reduction steps and their context.
    state : PipelineState
        Observation and auxiliary data.
    params : dict
        Reduction parameter values.
    widths : list
        Aperture candidates in trial order.
    logger : None, callable
        Function to receive progress and warning messages.

    Returns
    -------
    results : list[tuple]
        Cost, scatter and duration in seconds for each candidate.
    """
    from exotedrf.v2 import stages
    extract = pipeline.steps[-1]
    if extract.name != 'Extract':
        raise ValueError('pipeline must end with its Extract step')
    builtin = extract.fn in (stages.step_extract, stages.step_extract_nirspec,
                             stages.step_extract_miri)
    if builtin and all(_scalar_width(width) for width in widths):
        try:
            results = stages.prepared_extract_results(state, params, pipeline.ctx, widths)
        except NotImplementedError:
            results = None
        if results is not None:
            _log(logger, f'[v2] Extract: scored {len(widths)} aperture '
                         'width(s) in one pass over the rate cube')
            return results
    results = []
    for width in widths:
        started = time.perf_counter()
        trial = extract.fn(state, dict(params, extract_width=width), pipeline.ctx)
        cost, scatter = stages.evaluate_production_cost(trial, params, pipeline.ctx)
        results.append((float(np.asarray(cost)), np.asarray(scatter),
                        time.perf_counter() - started))
    return results


def _ad_hoc_options(cfg, mode_name, files_tag):
    """Enable only the steps needed for the selected ad-hoc rerun."""
    opts = v2config.fixed_options(cfg)
    for name in _SKIPPED_STEPS:
        opts[name] = 'skip'
    if mode_name == 'from_pca_only':
        # Rerun PCA regardless of the original step switch.
        opts['PCAReconstructStep'] = 'run'
    opts['input_filetag'] = files_tag
    opts['restart_input_ndim'] = 3
    opts['stream_stage1'] = False
    # Clear inputs consumed only by skipped steps.
    for name in ('hot_pixel_map', 'outlier_maps', 'soss_timeseries',
                 'soss_timeseries_o2', 'f277w', 'soss_background_file'):
        opts[name] = None
    return opts


def run_ad_hoc(config_or_path, *, output_dir=None, logger=print,
               enable_x64_mode=False, refpack=None, write_products=True, write_logs=True):
    """Rerun extraction, with optional PCA, from saved Stage-2 products.

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
    write_products, write_logs : bool
        Whether to save products and append optimizer logs.

    Returns
    -------
    result : OptimizationResult
        Winning aperture, trial history, final state and product paths.
    """
    from astropy.io import fits
    from exotedrf.v2 import io, products, restart, stages
    from exotedrf.v2.optimize import (OptimizationResult, TrialResult, select_first_finite_minimum)
    started_total = time.perf_counter()
    cfg = dict(v2config.coerce_config(config_or_path))
    mode_name = v2config.ad_hoc_mode(cfg)
    if mode_name is None:
        raise ValueError('run_ad_hoc requires optimize_extract_width_only or from_pca_only')
    # Set aside the Stage-3 tracing frame.
    deepframe_cfg = cfg.pop('deepframe', None)
    # Accept candidate lists for disabled sweeps in ad-hoc mode.
    for key in [key for key, value in cfg.items() if key.startswith('optimize_') and not value and
                key not in ('optimize_extract_width_only', 'optimize_from_pca_only')]:
        del cfg[key]
    # Allow an unset width when reusing the first-pass aperture.
    validate_cfg = cfg
    if cfg.get('reuse_first_pass_extract_width', False) and cfg.get('extract_width') is None:
        validate_cfg = {k: v for k, v in cfg.items() if k != 'extract_width'}
    v2config.validate_supported_config(validate_cfg)
    if validate_cfg is not cfg:
        # Keep the validators' in-place method canonicalizations.
        cfg.update(validate_cfg)
    if enable_x64_mode or cfg.get('v2_enable_x64', False):
        from exotedrf.v2 import core
        core.enable_x64()
    paths = products.output_layout(cfg, output_dir=output_dir, create=True)
    dirs = ad_hoc_directories(cfg, paths)
    name = str(cfg.get('name_tag', 'default_run'))
    extract_method = str(cfg.get('extract_method', 'box')).lower()
    banner = '=' * 60

    if mode_name == 'extract_width_only':
        _log(logger, f'\n{banner}\nEXTRACT WIDTH ONLY MODE ENABLED\n'
                     'Skipping directly to Stage 3 using existing Stage 2 ' f'outputs\n{banner}\n')
        source_dir = extract_only_source_dir(cfg, dirs, logger)
        _log(logger, f'[v2] Looking for existing Stage 2 outputs in {source_dir}...')
        files = find_existing_stage2_outputs(stage2_patterns(cfg, source_dir),
            f'No Stage 2 outputs found in {source_dir}. Please run the full '
            'pipeline first before using optimize_extract_width_only mode.', logger)
    else:
        _log(logger, f'\n{banner}\nFROM PCA ONLY MODE ENABLED\nRestarting '
                     'from existing BadPix outputs and rerunning PCA/Stage 3 ' f'only\n{banner}\n')
        source_dir = next((directory for directory in (dirs['v2_stage2'], dirs['v1_stage2'])
             if glob.glob(os.path.join(os.fspath(directory), '*_badpixstep.fits'))),
            dirs['v1_stage2'])
        files = find_existing_stage2_outputs(
            [os.path.join(os.fspath(source_dir), '*_badpixstep.fits')],
            f'No BadPix Step outputs found in {source_dir}. Please run the '
            'optimizer through BadPixStep before using from_pca_only mode.', logger)
    files_tag = ('pcareconstructstep' if files[0].endswith(
        '_pcareconstructstep.fits') else 'badpixstep')
    centroid_dirs = [dirs['v2_stage3'], dirs['v1_stage3'], source_dir]
    centroids = load_ad_hoc_centroids(cfg, centroid_dirs, logger)
    deepframe_path = None
    if mode_name == 'extract_width_only':
        deepframe_path = resolve_ad_hoc_deepframe(
            dict(cfg, deepframe=deepframe_cfg), source_dir, logger)

    rerun_cfg = dict(cfg)
    rerun_cfg['extract_width'] = resolve_ad_hoc_extract_width(
        cfg, stage3_dirs=[dirs['v2_stage3'], dirs['v1_stage3']],
        cost_paths=[paths['cost'], dirs['v1_files'] / f'Cost_{name}.txt'], logger=logger)
    widths = ad_hoc_widths(rerun_cfg)

    opts = _ad_hoc_options(cfg, mode_name, files_tag)
    opts['centroids'] = centroids
    opts['dtype'] = (np.float64 if enable_x64_mode or
                     cfg.get('v2_enable_x64', False) else np.float32)
    loaded = time.perf_counter()
    state, files = restart.load_restart_state(opts, files=files, logger=logger)
    refpack_path = refpack if refpack is not None else \
        restart.restart_refpack(state.cube, opts, files, logger=logger)
    ctx = stages.prepare_context(state.cube, opts, refpack=refpack_path)
    pipeline = stages.build_pipeline(opts['mode'], ctx)
    load_seconds = time.perf_counter() - loaded

    params = {'extract_width': widths[0], 'extract_width_soss2': cfg.get('extract_width_soss2')}
    capture = None
    if mode_name == 'from_pca_only':
        remove = cfg.get('remove_components')
        _log(logger, f'[v2] Rerunning PCAReconstructStep with remove_components={remove}')
        if write_products and opts.get('do_plots', False):
            capture = products.CompatibilityCapture(paths, ctx, params=params)
        pca_started = time.perf_counter()
        state = pipeline.run(state, params, stop=len(pipeline.steps) - 1, observer=capture)
        pca_seconds = time.perf_counter() - pca_started
        _log(logger, f'[v2] PCAReconstructStep: {pca_seconds:.1f}s')
    else:
        pca_seconds = 0.
        if deepframe_path is None:
            # Require a deepframe for box or optimal extraction even with supplied centroids.
            raise ValueError('Deepframe must be provided for box extraction.')
        deepframe = (np.asarray(fits.getdata(deepframe_path), dtype=float)
                     if isinstance(deepframe_path, (str, os.PathLike))
                     else np.asarray(deepframe_path, dtype=float))
        state = type(state)(cube=state.cube, aux=dict(state.aux, stage3_deepframe=deepframe))

    # Merge the best earlier parameters into every new log row.
    if write_logs:
        _seed_logs(paths, dirs, name)
        required = ['ad_hoc_mode', 'remove_components', 'extract_width']
        param_cols, row_offset, best_logged = prepare_cost_log(paths['cost'], required)
    else:
        param_cols, row_offset, best_logged = [], 0, {}
    base_row = {'ad_hoc_mode': mode_name, 'remove_components': cfg.get('remove_components')}
    merged = dict(best_logged)
    merged.update(base_row)

    results = score_extract_widths(pipeline, state, params, widths, logger=logger)
    trials = []
    costs = []
    for index, (width, (cost, scatter, duration)) in enumerate(zip(widths, results)):
        cost = float(np.asarray(cost))
        costs.append(cost)
        trial_params = dict(params, extract_width=width)
        trials.append(TrialResult(phase=2, checkpoint='Extract', parameter='extract_width',
            candidate_index=index, value=width, params=trial_params,
            duration_s=float(duration), cost=cost, scatter=np.asarray(scatter, dtype=float)))
        if write_logs:
            row = dict(merged, extract_width=width)
            append_cost_log_row(paths['cost'], param_cols, row, duration, cost)
            append_scatter_log_row(paths['scatter'], scatter)
        _log(logger, f'extract_width={width}: cost={cost:.12f} ({duration:.1f}s)')
    # Select the first finite minimum while retaining candidate order.
    best_index, best_cost = select_first_finite_minimum(costs, parameter='extract_width')
    best_width = widths[best_index]
    _log(logger, f'\n*** Best extract_width={best_width} with cost={best_cost:.6f} ***\n')
    params['extract_width'] = best_width
    final_state = pipeline.steps[-1].fn(state, params, ctx)
    final_cost, final_scatter = stages.evaluate_production_cost(final_state, params, ctx)
    final_cost = float(np.asarray(final_cost))
    final_scatter = np.asarray(final_scatter, dtype=float)

    written = {}
    if write_products:
        written.update(products.write_final_products(
            final_state, params, ctx, paths, output_mode='optimal'))
        written.update(_write_ad_hoc_sidecars(mode_name, final_state, ctx, paths, files, cfg,
            logger=logger))
        plot_ctx = ctx
        if ctx.get('centroids') is not None:
            # Plot centroids only when Stage 3 traces them.
            plot_ctx = dict(ctx, opts=dict(ctx['opts'], do_plots=False))
        written.update(products.write_diagnostic_plots(
            final_state, params, plot_ctx, paths, trials, scatter=final_scatter))
        if capture is not None:
            written.update(capture.render(final_state))

    total = time.perf_counter() - started_total
    summary = products.jsonable({'schema_version': 1, 'ad_hoc_mode': mode_name,
        'mode': opts['mode'], 'input_products': [os.fspath(path) for path in files],
        'centroids': centroids if not isinstance(centroids, dict) else 'config-dict',
        'deepframe': (os.fspath(deepframe_path) if isinstance(deepframe_path, (str, os.PathLike))
                      else None), 'remove_components': cfg.get('remove_components'),
        'extract_widths': widths, 'winners': params, 'final_cost': final_cost, 'trials': trials,
        'products': written, 'timing_s': {'load_and_context': load_seconds, 'pca': pca_seconds,
                     'total': total}, 'cost_log_row_offset': row_offset})
    if write_logs:
        summary_path = Path(paths['files']) / (f'AdHoc_{mode_name}_{name}.json')
        with summary_path.open('w', encoding='utf-8') as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        written['ad_hoc_summary'] = str(summary_path)
    hours, rest = divmod(int(total), 3600)
    minutes, seconds = divmod(rest, 60)
    label = ('OPTIMAL EXTRACT_WIDTH' if cfg.get('optimize_extract_width', False) is True
             else 'USED EXTRACT_WIDTH')
    _log(logger, f'\n{banner}\nTOTAL RUNTIME: {hours}h {minutes:02d}min '
                 f'{seconds:02d}s\n{label}: {best_width}')
    if mode_name == 'from_pca_only':
        _log(logger, f"REMOVE_COMPONENTS: {format_log_value(cfg.get('remove_components'))}")
    _log(logger, banner)
    output_paths = {key: str(value) for key, value in written.items()}
    if write_logs:
        output_paths.update(cost=str(paths['cost']), scatter=str(paths['scatter']))
    return OptimizationResult(winners=dict(params), initial_params={}, trials=tuple(trials),
        final_cost=final_cost, final_scatter=final_scatter,
        final_state=final_state, products=written, output_paths=output_paths, summary=summary)


def _write_ad_hoc_sidecars(mode_name, state, ctx, paths, files, cfg, logger=None):
    """Write reusable products for an ad-hoc reduction."""
    from exotedrf.v2 import io, products, trace
    written = {}
    if mode_name == 'extract_width_only':
        if ctx.get('centroids') is None:
            centroids = products._final_centroids(state, ctx, state.aux.get('stage3_deepframe'))
            if centroids:
                prefix = products._v1_fileroot_noseg(state.cube.meta)
                path = Path(paths['stage3']) / f'{prefix}centroids.csv'
                trace.save_centroids_csv(path, centroids)
                written['centroids'] = str(path)
        return written
    from exotedrf.v2 import restart
    segment_files = restart.write_segment_products(
        state.cube, files, paths['stage2'], 'pcareconstructstep', logger=logger)
    if segment_files:
        written['pcareconstructstep'] = segment_files
    else:
        # Save the reconstructed cube as a combined rate product.
        io.save_rate_cube(state.cube, paths['rate'], extra_header={'OPTIMIZE': True})
        written['rate'] = str(paths['rate'])
    sidecars = products._write_reusable_sidecars(state, dict(ctx, order0_mask=None), paths)
    if ctx.get('centroids') is not None and 'centroids' in sidecars:
        # Save only centroids traced during this reduction.
        os.unlink(sidecars.pop('centroids'))
    written.update(sidecars)
    if cfg.get('generate_lc') is True and state.aux.get('pca_wlc') is not None:
        prefix = products._v1_fileroot_noseg(state.cube.meta)
        path = Path(paths['stage2']) / f'{prefix}lcestimate.npy'
        np.save(path, np.asarray(state.aux['pca_wlc']), allow_pickle=False)
        written['lcestimate'] = str(path)
    return written


__all__ = ['ad_hoc_widths', 'append_cost_log_row', 'append_scatter_log_row',
    'find_best_logged_extract_width', 'find_existing_stage2_outputs',
    'find_stage3_spectrum_file', 'format_log_value', 'is_null_like',
    'load_ad_hoc_centroids', 'parse_extract_width_metadata',
    'prepare_cost_log', 'resolve_ad_hoc_deepframe',
    'resolve_ad_hoc_extract_width', 'run_ad_hoc', 'score_extract_widths']
