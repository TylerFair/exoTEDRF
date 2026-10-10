#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 00:00 2025

@author: PSD, modified by TRF

Script to run the exoTEDRF pipeline optimizer.
"""

import argparse
import ast
from astropy.io import fits
import glob
import numpy as np
import os
import pandas as pd
import re
import sys
import time
import yaml

from exotedrf import utils

# ===== Setup =====
early = argparse.ArgumentParser(add_help=False)
early.add_argument(
    "--config", "-c",
    default="run_optimize.yaml",
    help="Path to your DMS config YAML"
)
args, remaining = early.parse_known_args()


try:
    cfg_early = yaml.safe_load(open(args.config))
except FileNotFoundError:
    sys.exit(f"ERROR: config file '{args.config}' not found.")

os.environ.setdefault(
    "CRDS_PATH",
    cfg_early.get("crds_cache_path", "./crds_cache")
)
os.environ.setdefault(
    "CRDS_SERVER_URL",
    "https://jwst-crds.stsci.edu"
)

from exotedrf.utils import parse_config, unpack_input_dir, fancyprint
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3
import exotedrf.optimize_utils as opt_utils
from exotedrf.optimize_plotting import make_diagnostic_plot, plot_scatter, plot_cost

# ===== Define Global Variables =====
# All Pipeline Steps
stage1_steps = ['DQInitStep', 'INLCorrStep', 'EmiCorrStep', 'SuperBiasStep', 'RefPixStep',
                'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep', 'JumpStep', 'RampFitStep',
                'GainScaleStep']
stage2_steps = ['AssignWCSStep', 'FlatFieldStep', 'OneOverFStep_int', 'BackgroundStep',
                'BadPixStep', 'PCAReconstructStep']

# Downstream Steps
_STAGE1_AFTER_1OVERF = ['linearitystep', 'jump', 'rampfitstep', 'gainscalestep']
_STAGE2_ALL = ['assignwcsstep', 'extract2dstep', 'sourcetypestep', 'wavecorrstep',
               'flatfieldstep', 'photomstep', 'backgroundstep', 'oneoverfstep', 'badpixstep',
               'pcareconstructstep']
_STAGE2_AFTER_BKG = ['oneoverfstep', 'badpixstep', 'pcareconstructstep']

# Optimization Checkpoints
all_checkpoints = [
    # Stage 1 checkpoints
    {
        'name': 'OneOverFStep_grp',
        'stage': 1,
        'params': ['soss_inner_mask_width', 'soss_outer_mask_width', 'nirspec_mask_width'],
        'skip_before': ['DQInitStep', 'INLCorrStep', 'EmiCorrStep', 'SuperBiasStep', 'RefPixStep',
                        'DarkCurrentStep'],
        'skip_after': ['LinearityStep', 'JumpStep', 'RampFitStep', 'GainScaleStep'],
    },
    {
        'name': 'JumpStep',
        'stage': 1,
        'params': ['time_jump_threshold', 'time_window'],
        'skip_before': ['DQInitStep', 'INLCorrStep', 'EmiCorrStep', 'SuperBiasStep', 'RefPixStep',
                        'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep'],
        'skip_after': ['RampFitStep', 'GainScaleStep'],
    },
    # Stage 2 checkpoints
    {
        'name': 'BackgroundStep',
        'stage': 2,
        'params': ['miri_trace_width', 'miri_background_width'],
        'skip_before': ['AssignWCSStep', 'FlatFieldStep'],
        'skip_after': ['OneOverFStep_int', 'BadPixStep', 'PCAReconstructStep'],
    },
    {
        'name': 'BadPixStep',
        'stage': 2,
        'params': ['space_outlier_threshold', 'time_outlier_threshold', 'box_size', 'window_size'],
        'skip_before': ['AssignWCSStep', 'FlatFieldStep', 'BackgroundStep', 'OneOverFStep_int'],
        'skip_after': ['PCAReconstructStep'],
    },
    # Stage 3 checkpoint - only for Phase 2 (full dataset)
    {
        'name': 'Extract',
        'stage': 3,
        'params': ['extract_width'],
        'skip_before': [],
        'skip_after': [],
        'phase_2_only': True,  # Only optimize in Phase 2
    },
]

# Output Directories
root_dir = cfg_early.get('root_dir', './')
# The stages write to pipeline_outputs_directory + '_' + output_tag (expanding '~'), so mirror
# that here so cached outputs are found and invalidated in the right place.
_output_tag = cfg_early.get('output_tag', '')
_output_tag = '_' + _output_tag if _output_tag != '' else ''
full_outdir = os.path.join(root_dir, 'pipeline_outputs_directory' + _output_tag)

# Define where to store outputs for each pipeline stage
outdir    = full_outdir
outdir_f  = os.path.join(full_outdir, 'Optimizer_Files')
outdir_s1 = os.path.join(full_outdir, 'Stage1/')
outdir_s2 = os.path.join(full_outdir, 'Stage2/')
outdir_s3 = os.path.join(full_outdir, 'Stage3/')
utils.verify_path(full_outdir)
utils.verify_path(outdir_f)
utils.verify_path(outdir_s1)
utils.verify_path(outdir_s2)
utils.verify_path(outdir_s3)

# ======== OBSERVING CONFIG PARAMETERS ========
# Observation mode in lowercase (e.g., 'niriss', 'nirspec', 'miri')
obs_early = (cfg_early.get('observing_mode') or '').lower()
# Detector filter in lowercase (e.g., 'clear', 'nrs1', 'nrs2')
filter_early = (cfg_early.get('filter_detector') or '').lower()
# Wavelength range limits for analysis and plotting (if provided in config)
wave_range_early      = cfg_early.get('wave_range', None)
wave_range_plot_early = cfg_early.get('wave_range_plot', None)
# Weighting factors for cost function or metrics,
# w1 = whitelight weight, w2 = spectral weight
w1 = cfg_early.get('w1', 0.0)
w2 = cfg_early.get('w2', 1.0)

# Allowed wavelength coverage for each instrument (microns)
bands = {
    'miri':    (5.0, 12.0),
    'nirspec': (1.0, 5.0),
    'niriss':  (0.6, 2.8)
}

# Loop through instruments to find the matching one for this observation
for key, (lo, hi) in bands.items():
    if key in obs_early:
        for name, rng in (('wave_range', wave_range_early),
                          ('wave_range_plot', wave_range_plot_early)):
            if rng is not None and not (lo <= min(rng) and max(rng) <= hi):
                raise ValueError(f"{name}={rng!r} out of allowed band [{lo}, {hi}]")
        break
# If no instrument key matched the observation mode, throw an error
else:
    raise ValueError(f"Unrecognized observing_mode: {cfg_early.get('observing_mode')}")


def is_null_like(value):
    """Return True for config values that should behave like None."""
    return value in [None, 'None', 'none', 'null', 'NULL', '']


def default_spectral_wave_range(observing_mode, filter_detector=None):
    """Return the high-signal wavelength range to use for spectral optimization."""
    obs = (observing_mode or '').lower()
    det = (filter_detector or '').lower()

    if 'niriss' in obs:
        return [1.0, 2.0]
    if 'nirspec' in obs:
        if det == 'nrs1':
            return [3.0, 3.5]
        if det == 'nrs2':
            return [4.0, 4.5]
        raise ValueError('NIRSpec spectral optimization requires filter_detector=NRS1 or NRS2.')
    if 'miri' in obs:
        return [5.0, 10.0]

    return None


def resolve_spectral_wave_range(cfg, w2):
    """Use the configured wavelength range, or an instrument default when spectral cost is active."""

    configured = cfg.get('wave_range', None)
    if not is_null_like(configured):
        return configured
    if w2 == 0:
        return None

    wave_range = default_spectral_wave_range(
        cfg.get('observing_mode'),
        cfg.get('filter_detector')
    )
    if wave_range is not None:
        fancyprint('Using default spectral optimization wave_range={} for {} {}.'
                   .format(wave_range, cfg.get('observing_mode'),
                           cfg.get('filter_detector')))
    return wave_range


def phase1_spectral_wave_range(instrument, wave_range):
    """Return the wavelength range usable by the fast optimizer-side extraction."""

    if is_null_like(wave_range):
        return None
    if instrument == 'NIRISS':
        return wave_range

    return None


def cost_function(st3, baseline_ints=None, wave_range=None, w1=0.0, w2=1.0, tol=0.05):
    """
    Compute a combined white-light + spectral P2P (point-to-point) metric.

    Parameters
    ----------
    st3 : dict-like
        Must contain:
          - 'Flux' (or 'Flux O1'/'Flux O2' for NIRISS) -> 2D array (n_int, n_wave)
          - 'Wave' (or 'Wave O1'/'Wave O2') -> 1D array (n_wave,)
    baseline_ints : list of 1 or 2 ints
        Integration indices defining baseline(s) for the spectral term.
    wave_range : None or [min, max]
        If given, restrict spectral term to this wavelength range (within ±tol).
    w1, w2 : float
        Weights for white-light and spectral terms in final cost.
    tol : float
        Allowed deviation when matching wave_range endpoints.

    Returns
    -------
    cost : float
        Combined cost = w1*ptp2_white + w2*ptp2_spec
    ptp2_spec_wave : np.ndarray
        Per-wavelength ptp2 metric values.
    """

    # ======== NIRISS-SPECIFIC WAVE + FLUX MERGE ========
    if 'niriss' in obs_early:
        wave, flux = opt_utils.stitch_soss_orders(st3['Wave O1'], st3['Wave O2'],
                                                  st3['Flux O1'], st3['Flux O2'])

    else:
        # For non-NIRISS: take flux/wave arrays directly
        flux = np.asarray(st3['Flux'], float)
        wave = np.asarray(st3['Wave'], float)
        if wave.ndim == 2:
            # MIRI/NIRSpec Stage 3 outputs store the same wavelength grid for each integration.
            if wave.shape == flux.shape:
                wave = np.nanmedian(wave, axis=0)
            elif 1 in wave.shape:
                wave = np.ravel(wave)
            else:
                raise ValueError(
                    f"Expected 1D wavelength axis or 2D array matching flux; got wave.shape={wave.shape} "
                    f"and flux.shape={flux.shape}"
                )
        elif wave.ndim != 1:
            raise ValueError(f"Expected 1D wavelength axis, got wave.ndim={wave.ndim}")

    # ======== WHITE-LIGHT TERM ========
    # Collapse all wavelengths into single white-light curve
    white      = np.nansum(flux, axis=1)
    white      = white[~np.isnan(white)]
    norm_white = white / np.median(white)
    # 2nd finite difference (neighbour avg - centre)
    d2_white   = 0.5*(norm_white[:-2] + norm_white[2:]) - norm_white[1:-1]
    ptp2_white = np.nanmedian(np.abs(d2_white))

    # ======== SPECTRAL TERM (PER-WAVELENGTH P2P) ========
    wave_meds = np.nanmedian(flux, axis=0, keepdims=True)
    norm_spec = flux / wave_meds
    d2_spec   = 0.5*(norm_spec[:-2] + norm_spec[2:]) - norm_spec[1:-1]

    # Select baseline integrations for spectral metric
    if baseline_ints is None:
        ptp2_spec_wave = np.nanmedian(np.abs(d2_spec), axis=0)
    elif len(baseline_ints) == 1:
        N = int(baseline_ints[0])
        ptp2_spec_wave = np.nanmedian(np.abs(d2_spec[:N]), axis=0)
    elif len(baseline_ints) == 2:
        Nlow, Nhigh = map(int, baseline_ints)
        low, high = d2_spec[:Nlow], d2_spec[Nhigh:]
        # Phase 1 only uses the first segment, so a positive Nhigh can lie beyond its end. Fall
        # back on whichever baseline is available rather than returning an all-NaN metric.
        if len(low) > 0 and len(high) > 0:
            low_term  = np.nanmedian(np.abs(low), axis=0)
            high_term = np.nanmedian(np.abs(high), axis=0)
            ptp2_spec_wave = 0.5 * (low_term + high_term)
        elif len(low) > 0 or len(high) > 0:
            ptp2_spec_wave = np.nanmedian(np.abs(low if len(low) > 0 else high), axis=0)
        else:
            raise ValueError(f"baseline_ints {baseline_ints} select no integrations "
                             f"(only {len(d2_spec) + 2} available).")
    else:
        raise ValueError(f"baseline_ints must be length 1 or 2, got {len(baseline_ints)}")

    # ======== WAVELENGTH RANGE FILTER (OPTIONAL) ========
    if wave_range is None:
        ptp2_spec = np.nanmedian(ptp2_spec_wave)

    elif isinstance(wave_range, (list, tuple)) and len(wave_range) == 2:
        lo, hi = wave_range
        finite = np.isfinite(wave)
        if not finite.any():
            raise ValueError("All entries in wave are NaN!")

        wave_min = np.nanmin(wave[finite])
        wave_max = np.nanmax(wave[finite])

        # Handle None values (means use data min/max)
        if lo is None:
            lo = wave_min
        if hi is None:
            hi = wave_max

        # Distances from requested range edges
        dist_lo = np.abs(wave - lo); dist_lo[~finite] = np.inf
        dist_hi = np.abs(wave - hi); dist_hi[~finite] = np.inf

        idx_lo = int(np.argmin(dist_lo))
        idx_hi = int(np.argmin(dist_hi))

        # If requested wavelengths not found within tolerance, use closest available
        if dist_lo[idx_lo] > tol or dist_hi[idx_hi] > tol:
            actual_lo = wave[idx_lo]
            actual_hi = wave[idx_hi]

            # Clip to available range and use closest wavelengths
            fancyprint(
                f"Requested wave_range {wave_range} not found within ±{tol} µm tolerance.\n"
                f"  Available data range: {wave_min:.3f} to {wave_max:.3f} µm\n"
                f"  Using closest wavelengths: {actual_lo:.3f} to {actual_hi:.3f} µm",
                msg_type='WARNING'
            )

        # Slice range in correct order
        i0, i1 = sorted((idx_lo, idx_hi))
        sub = ptp2_spec_wave[i0:i1+1]
        if np.all(np.isnan(sub)):
            raise ValueError(f"No valid ptp2_spec values in wave range {wave_range}")
        ptp2_spec = np.nanmedian(sub)

    else:
        raise ValueError("wave_range must be None or a length-2 list/tuple")

    # ======== FINAL COST COMBINATION ========
    # Avoid allowing a zero-weighted NaN term to poison the selected metric
    # (IEEE arithmetic makes 0.0 * NaN evaluate to NaN).
    cost = 0.0
    if w1 != 0:
        cost += w1 * ptp2_white
    if w2 != 0:
        cost += w2 * ptp2_spec

    return cost, ptp2_spec_wave


def format_log_value(value):
    """Format optimizer values for TSV logging."""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(str(v) for v in value) + ']'
    if value is None:
        return 'None'
    if pd.isna(value):
        return ''
    return str(value)


def prepare_cost_log(name_str, required_param_cols):
    """Ensure the cost log exists and can store the requested parameter columns."""

    cost_path = f"{outdir_f}/Cost_Summary{name_str}.txt"
    if os.path.exists(cost_path) and os.path.getsize(cost_path) > 0:
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
        for col in param_cols:
            if col not in df.columns:
                df[col] = ''
        for col in ['duration_s', 'cost']:
            if col not in df.columns:
                df[col] = ''
        df = df[param_cols + ['duration_s', 'cost']]

    df.to_csv(cost_path, sep='\t', index=False)

    best_logged = {}
    if len(df) > 0:
        numeric_cost = pd.to_numeric(df['cost'], errors='coerce')
        if numeric_cost.notna().any():
            best_logged = df.loc[numeric_cost.idxmin(), param_cols].to_dict()

    return cost_path, param_cols, len(df), best_logged


def append_cost_log_row(cost_path, param_cols, row_values, duration_s, cost):
    """Append one optimizer result row to the cost log."""

    fields = [format_log_value(row_values.get(col, '')) for col in param_cols]
    fields.extend([f"{duration_s:.1f}", f"{cost:.12f}"])
    with open(cost_path, 'a') as logf:
        logf.write('\t'.join(fields) + '\n')


def append_scatter_log_row(name_str, scatter):
    """Append one scatter spectrum row to the scatter log."""

    scatter_path = f"{outdir_f}/LightCurve_Scatter{name_str}.txt"
    with open(scatter_path, 'a') as logs:
        logs.write(' '.join(f"{x:.10g}" for x in scatter) + '\n')
    return scatter_path


def load_ad_hoc_centroids(cfg, stage2_source_dir=None):
    """Load centroids from the config or existing pipeline outputs."""

    centroids_path = cfg.get('centroids')
    if centroids_path not in [None, 'None', 'null', '']:
        fancyprint(f"  Using centroids from config: {centroids_path}")
        if isinstance(centroids_path, str):
            return pd.read_csv(centroids_path, comment='#')
        return centroids_path

    s2_dir = stage2_source_dir if stage2_source_dir is not None else outdir_s2
    centroid_patterns = [
        (outdir_s3, 'Stage 3'),
        (s2_dir, 'Stage 2'),
    ]
    for outdir, label in centroid_patterns:
        centroid_files = sorted(glob.glob(f'{outdir}*centroids.csv'))
        if centroid_files:
            centroid_file = centroid_files[0]
            fancyprint(f"  Loading centroids from {label}: {centroid_file}")
            return pd.read_csv(centroid_file, comment='#')

    fancyprint("  No centroid table found in config, Stage 3, or Stage 2. Stage 3 will trace "
               "centroids from the deepframe.")
    return None


def resolve_existing_centroids(cfg, fileroot_noseg=None):
    """Resolve the centroid table for a Stage 3 extraction or rerun. If `fileroot_noseg` is
    given, only centroid tables written for that dataset are considered."""

    centroids_path = cfg.get('centroids')
    if centroids_path not in [None, 'None', 'null', '']:
        fancyprint(f"  Using centroids from config: {centroids_path}")
        if isinstance(centroids_path, str):
            return pd.read_csv(centroids_path, comment='#')
        return centroids_path

    centroid_patterns = [
        (outdir_s3, 'Stage 3'),
        (outdir_s2, 'Stage 2'),
    ]
    for outdir, label in centroid_patterns:
        if fileroot_noseg is not None:
            centroid_files = sorted(glob.glob(f'{outdir}{glob.escape(fileroot_noseg)}centroids.csv'))
        else:
            centroid_files = sorted(glob.glob(f'{outdir}*centroids.csv'))
        if centroid_files:
            fancyprint(f"  Loading centroids from {label}: {centroid_files[0]}")
            return pd.read_csv(centroid_files[0], comment='#')

    raise FileNotFoundError(
        "No centroid table available for Stage 3. Set 'centroids' in the config or provide "
        "a Stage 3/Stage 2 centroids.csv output."
    )


def find_existing_stage2_outputs(patterns, error_message):
    """Return the first matching set of Stage 2 files from a list of glob patterns."""

    for pattern in patterns:
        found_files = sorted(glob.glob(pattern))
        if found_files:
            fancyprint(f"  Found {len(found_files)} file(s) matching: {pattern}")
            return found_files
    raise FileNotFoundError(error_message)


def resolve_ad_hoc_deepframe(cfg, stage2_source_dir=None):
    """Resolve the deepframe path for ad hoc Stage 3 runs."""

    deepframe = cfg.get('deepframe')
    if deepframe not in [None, 'None', 'null', '']:
        return deepframe

    s2_dir = stage2_source_dir if stage2_source_dir is not None else outdir_s2
    deepframe_files = sorted(glob.glob(f'{s2_dir}*deepframe.fits'))
    if deepframe_files:
        fancyprint(f"  Using deepframe from: {deepframe_files[0]}")
        return deepframe_files[0]

    return None


def resolve_extract1d_kwargs(cfg):
    """Return the Stage 3 Extract1dStep kwargs block, if present."""

    return cfg.get('stage3_kwargs', {}).get('Extract1dStep', {})


def find_stage3_spectrum_file(extract_method):
    """Return the first Stage 3 full-resolution spectrum file for the requested method."""

    pattern = os.path.join(outdir_s3, f"*_{extract_method}_spectra_fullres.fits")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No Stage 3 spectrum file found matching {pattern}")
    return matches[0]


def parse_extract_width_metadata(width_value):
    """Parse an extraction width from YAML/header metadata into a scalar or asymmetric dict."""

    if width_value in [None, 'None', 'null', '']:
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

    if text.startswith('{') and text.endswith('}'):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    if text.startswith('[') and text.endswith(']'):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            if len(parsed) == 2:
                return {'lower': float(parsed[0]), 'upper': float(parsed[1])}
            return list(parsed)

    match = re.fullmatch(
        r'lower\s*=\s*([-+]?\d*\.?\d+)\s*,\s*upper\s*=\s*([-+]?\d*\.?\d+)', text
    )
    if match:
        return {'lower': float(match.group(1)), 'upper': float(match.group(2))}

    try:
        scalar = float(text)
    except ValueError:
        return text

    if scalar.is_integer():
        return int(scalar)
    return scalar


def find_best_logged_extract_width(name_str):
    """Read the best logged extract width from the optimizer cost table."""

    cost_path = f"{outdir_f}/Cost_Summary{name_str}.txt"
    if os.path.exists(cost_path) is not True:
        raise FileNotFoundError(f"No optimizer cost log found at {cost_path}")

    df = pd.read_csv(cost_path, sep='\t', keep_default_na=False)
    if 'extract_width' not in df.columns or 'cost' not in df.columns:
        raise ValueError(f"{cost_path} does not contain extract_width and cost columns.")

    cost = pd.to_numeric(df['cost'], errors='coerce')
    valid = cost.notna() & (df['extract_width'].astype(str).str.strip() != '')
    if not valid.any():
        raise ValueError(f"{cost_path} does not contain any valid logged extract_width values.")

    best_idx = cost[valid].idxmin()
    width = parse_extract_width_metadata(df.loc[best_idx, 'extract_width'])
    if width in [None, 'None', 'null', '']:
        raise ValueError(f"Could not parse extract_width from best row of {cost_path}")
    fancyprint(f"  Reusing best logged extract_width from {cost_path}: {width}")
    return width


def resolve_ad_hoc_extract_width(cfg):
    """Resolve the extraction width to use for ad hoc Stage-3 reruns."""

    if cfg.get('optimize_extract_width', False):
        return cfg.get('extract_width')

    width = cfg.get('extract_width')
    if width not in [None, 'None', 'null', '']:
        return width

    if cfg.get('reuse_first_pass_extract_width', False):
        source_method = cfg.get('first_pass_extract_method', 'box')
        try:
            specfile = find_stage3_spectrum_file(source_method)
        except FileNotFoundError:
            name_str = cfg.get('run_name', 'default_run')
            if name_str != '':
                name_str = '_' + name_str
            fancyprint("  No first-pass Stage 3 spectrum file found; falling back to optimizer "
                       "cost log for extract_width.")
            return find_best_logged_extract_width(name_str)
        header = fits.getheader(specfile)
        width = parse_extract_width_metadata(header.get('WIDTH'))
        if width in [None, 'None', 'null', '']:
            raise ValueError(f"WIDTH header missing or unreadable in {specfile}")
        fancyprint(f"  Reusing first-pass extract_width from {specfile}: {width}")
        return width

    raise ValueError(
        "No extract_width specified for the Stage-3 rerun. Set extract_width, or set "
        "reuse_first_pass_extract_width=True to read it from an existing first-pass Stage 3 "
        "box spectrum."
    )


def run_stage3_for_width(stage2_inputs, cfg, centroids, deepframe, extract_width):
    """Run Stage 3 once for a specific extraction width."""

    return run_stage3(
        stage2_inputs,
        save_results=True,
        force_redo=True,
        extract_method=cfg['extract_method'],
        soss_specprofile=cfg.get('soss_specprofile'),
        centroids=centroids,
        extract_width=extract_width,
        extract_width_soss2=cfg.get('extract_width_soss2'),
        st_teff=cfg.get('st_teff'),
        st_logg=cfg.get('st_logg'),
        st_met=cfg.get('st_met'),
        planet_letter=cfg.get('planet_letter'),
        output_tag=cfg['output_tag'],
        do_plot=cfg.get('do_plots', False),
        deepframe=deepframe,
        saturation_rescue=cfg.get('saturation_rescue', False),
        mask_do_not_use_pixels=cfg.get('mask_do_not_use_pixels', True),
        root_dir=root_dir,
        **cfg.get('stage3_kwargs', {})
    )


def select_best_trial(costs, param_name='parameter'):
    """Return the index of the first finite minimum cost.

    np.argmin returns the index of a NaN if one is present, so a failed trial
    could otherwise be selected as the winner. Non-finite costs are skipped;
    ties keep the earliest candidate.
    """
    best_idx = None
    best_cost = None
    for idx, cost in enumerate(costs):
        cost = float(cost)
        if not np.isfinite(cost):
            continue
        if best_idx is None or cost < best_cost:
            best_idx, best_cost = idx, cost
    if best_idx is None:
        raise ValueError(f'All candidate values for {param_name} produced non-finite costs.')
    return best_idx


def delete_checkpoint_outputs(checkpoint_name, outdir_s1, outdir_s2):
    """Delete a checkpoint step's cached outputs, and those of every step downstream of it, so
    the next pipeline call recomputes them."""
    patterns = []
    downstream = []
    if checkpoint_name == 'OneOverFStep_grp':
        patterns.append(f"{outdir_s1}*_oneoverfstep.fits")
        downstream += [f"{outdir_s1}*_{t}.fits" for t in _STAGE1_AFTER_1OVERF]
        downstream += [f"{outdir_s2}*_{t}.fits" for t in _STAGE2_ALL]
        downstream.append(f"{outdir_s2}*hot_pixels.npy")
    elif checkpoint_name == 'JumpStep':
        patterns.append(f"{outdir_s1}*_jump.fits")
        downstream += [f"{outdir_s1}*_{t}.fits" for t in ['rampfitstep', 'gainscalestep']]
        downstream += [f"{outdir_s2}*_{t}.fits" for t in _STAGE2_ALL]
        downstream.append(f"{outdir_s2}*hot_pixels.npy")
    elif checkpoint_name == 'BackgroundStep':
        patterns.append(f"{outdir_s2}*_backgroundstep.fits")
        downstream += [f"{outdir_s2}*_{t}.fits" for t in _STAGE2_AFTER_BKG]
        downstream.append(f"{outdir_s2}*hot_pixels.npy")
    elif checkpoint_name == 'BadPixStep':
        patterns.append(f"{outdir_s2}*_badpixstep.fits")
        # Also delete cached hot_pixels.npy to force spatial outlier
        # redetection with new parameters (space_thresh, box_size).
        patterns.append(f"{outdir_s2}*hot_pixels.npy")
        downstream.append(f"{outdir_s2}*_pcareconstructstep.fits")
    deleted = 0
    for pattern in patterns + downstream:
        files_to_delete = glob.glob(pattern)
        if files_to_delete:
            fancyprint(f"Deleting {len(files_to_delete)} cached file(s) for {checkpoint_name}:")
        for cached_file in files_to_delete:
            fancyprint(f"  Deleting: {cached_file}")
            os.remove(cached_file)
            if pattern in patterns:
                deleted += 1
    if patterns and deleted == 0:
        fancyprint(f"WARNING: No cached files found matching: {patterns}", msg_type='WARNING')


def stage1_kwargs_with_winners(run_cfg):
    """Stage 1 kwargs with the current scalar time_window forwarded to JumpStep.

    time_window was previously only forwarded while it was itself being swept,
    so later sweeps and the Phase 2 run silently fell back to the step default
    instead of the current best (or fixed) value.
    """
    kwargs = dict(run_cfg.get('stage1_kwargs') or {})
    time_window = run_cfg.get('time_window')
    if isinstance(time_window, (int, float, np.integer, np.floating)):
        step_kwargs = dict(kwargs.get('JumpStep') or {})
        step_kwargs['time_window'] = time_window
        kwargs['JumpStep'] = step_kwargs
    return kwargs


def stage2_kwargs_with_winners(run_cfg):
    """Stage 2 kwargs with current scalar box_size/window_size forwarded to
    BadPixStep (same defect and fix as stage1_kwargs_with_winners)."""
    kwargs = dict(run_cfg.get('stage2_kwargs') or {})
    step_kwargs = dict(kwargs.get('BadPixStep') or {})
    for key in ('box_size', 'window_size'):
        value = run_cfg.get(key)
        if isinstance(value, (int, float, np.integer, np.floating)):
            step_kwargs[key] = value
    if step_kwargs:
        kwargs['BadPixStep'] = step_kwargs
    return kwargs


def run_ad_hoc_extract_width_search(stage2_inputs, cfg, centroids, deepframe, baseline_ints,
                                    wave_range, w1, w2, name_str, base_row_values):
    """Append an ad hoc Stage 3 extraction sweep to the optimizer logs."""

    if cfg.get('optimize_extract_width', False):
        extract_widths = cfg['extract_width']
        if not isinstance(extract_widths, list):
            raise ValueError("extract_width must be a list when optimize_extract_width=True")
    else:
        extract_widths = cfg.get('extract_width')
        if isinstance(extract_widths, list):
            extract_widths = [extract_widths[0]]
        else:
            extract_widths = [extract_widths]

    required_cols = ['ad_hoc_mode', 'remove_components']
    if 'extract_width' not in required_cols:
        required_cols.append('extract_width')
    cost_path, param_cols, row_offset, best_logged = prepare_cost_log(name_str, required_cols)
    if best_logged:
        merged_row_values = best_logged.copy()
        merged_row_values.update(base_row_values)
    else:
        merged_row_values = base_row_values.copy()

    extract_costs = []
    appended_rows = []
    best_stage3_results = None

    for idx, width in enumerate(extract_widths):
        fancyprint(f"\n{'='*60}")
        fancyprint(f"Testing extract_width={width}")
        fancyprint(f"{'='*60}\n")
        t0 = time.perf_counter()

        stage3_results = run_stage3_for_width(stage2_inputs, cfg, centroids, deepframe, width)
        cost, scatter = cost_function(
            stage3_results,
            baseline_ints=baseline_ints,
            wave_range=wave_range,
            w1=w1,
            w2=w2
        )

        dt = time.perf_counter() - t0
        extract_costs.append(cost)
        appended_rows.append(row_offset + idx)
        this_row = merged_row_values.copy()
        this_row['extract_width'] = width
        append_cost_log_row(cost_path, param_cols, this_row, dt, cost)
        append_scatter_log_row(name_str, scatter)

        fancyprint(f"extract_width={width}: cost={cost:.12f} ({dt:.1f}s)")

        if best_stage3_results is None or cost <= np.nanmin(extract_costs):
            best_stage3_results = stage3_results

    best_idx = select_best_trial(extract_costs, 'extract_width')
    best_extract_width = extract_widths[best_idx]
    best_cost = extract_costs[best_idx]
    best_row_idx = appended_rows[best_idx]

    fancyprint(f"\n*** Best extract_width={best_extract_width} with cost={best_cost:.6f} ***\n")

    final_stage3_results = run_stage3_for_width(stage2_inputs, cfg, centroids, deepframe,
                                                best_extract_width)

    return final_stage3_results, best_extract_width, best_cost, best_row_idx


def run_optimizer():
    """Run the optimizer.
    """

    # ===== Initial Setup =====
    parser = argparse.ArgumentParser(description="exoTEDRF Optimizer")
    parser.add_argument("--config", "-c", default="run_optimize.yaml", help="Config YAML")
    args = parser.parse_args()

    cfg = parse_config(args.config)
    # Fail fast rather than after Phase 1: Stage 3 only supports these extraction methods.
    if cfg.get('extract_method') not in ['box', 'atoca', 'optimal']:
        raise ValueError("extract_method must be one of 'box', 'atoca', or 'optimal'; got "
                         "{!r}.".format(cfg.get('extract_method')))
    obs = (cfg.get('observing_mode') or '').lower()
    filter_det = (cfg.get('filter_detector') or '').lower()
    instrument = obs.split('/')[0].upper() if '/' in obs else obs.upper()

    # Key parameters
    baseline_ints = cfg.get('baseline_ints', [100, -100])
    name_str = cfg.get('run_name', 'default_run')
    if name_str != '':
        name_str = '_' + name_str
    wave_range_plot = cfg.get('wave_range_plot', None)
    ylim_plot = cfg.get('ylim_plot', None)
    w1 = cfg.get('w1', 0.0)
    w2 = cfg.get('w2', 1.0)
    wave_range = resolve_spectral_wave_range(cfg, w2)

    if wave_range_plot is None:
        wave_range_plot = wave_range

    t0_total = time.perf_counter()
    optimize_extract_width_only = cfg.get('optimize_extract_width_only', False)
    from_pca_only = cfg.get('from_pca_only', cfg.get('optimize_from_pca_only', False))
    extract_method = cfg.get('extract_method', 'box')

    if optimize_extract_width_only and from_pca_only:
        raise ValueError("optimize_extract_width_only and from_pca_only cannot both be True.")

    # ==========================================================
    # ===== Special Case 1: Optimize Extraction Width Only =====
    # ==========================================================
    if optimize_extract_width_only:
        fancyprint(f"\n{'='*60}")
        fancyprint("EXTRACT WIDTH ONLY MODE ENABLED")
        fancyprint("Skipping directly to Stage 3 using existing Stage 2 outputs")
        fancyprint(f"{'='*60}\n")

        optimize_flags = [
            k for k in cfg.keys()
            if k.startswith('optimize_') and k not in ['optimize_extract_width_only',
                                                       'optimize_from_pca_only']
        ]
        for flag in optimize_flags:
            if flag == 'optimize_extract_width':
                continue
            elif cfg[flag]:
                raise ValueError(
                    f"{flag} must be False when optimize_extract_width_only=True. "
                    "Only the Stage 3 extraction may be rerun in this mode."
                )

        # Determine the source directory for Stage 2 inputs
        stage2_source_dir = outdir_s2
        input_dir_cfg = cfg.get('input_dir')
        if input_dir_cfg not in [None, 'None', 'null', '']:
            possible_dirs = [
                input_dir_cfg,
                os.path.join(input_dir_cfg, 'Stage2'),
                os.path.join(input_dir_cfg, 'Stage2/'),
            ]
            for p_dir in possible_dirs:
                if os.path.isdir(p_dir):
                    if glob.glob(os.path.join(p_dir, '*_badpixstep.fits')) or glob.glob(os.path.join(p_dir, '*_pcareconstructstep.fits')):
                        stage2_source_dir = p_dir
                        if not stage2_source_dir.endswith('/'):
                            stage2_source_dir += '/'
                        fancyprint(f"Detected Stage 2 outputs in input_dir: {stage2_source_dir}")
                        break

        fancyprint(f"Looking for existing Stage 2 outputs in {stage2_source_dir}...")
        pca_step = cfg.get('PCAReconstructStep', 'run')
        if pca_step == 'skip' or cfg.get('remove_components') in [None, 'None', 'null', '', []]:
            patterns = [
                f'{stage2_source_dir}*_badpixstep.fits',
                f'{stage2_source_dir}*_pcareconstructstep.fits',
            ]
        else:
            patterns = [
                f'{stage2_source_dir}*_pcareconstructstep.fits',
                f'{stage2_source_dir}*_badpixstep.fits',
            ]
        stage2_files = find_existing_stage2_outputs(
            patterns,
            f"No Stage 2 outputs found in {stage2_source_dir}. "
            "Please run the full pipeline first before using optimize_extract_width_only mode."
        )
        fancyprint("Looking for centroids file...")
        centroids_df = load_ad_hoc_centroids(cfg, stage2_source_dir=stage2_source_dir)
        deepframe = resolve_ad_hoc_deepframe(cfg, stage2_source_dir=stage2_source_dir)

        fancyprint(f"Using Stage 2 outputs: {stage2_files}")
        base_row_values = {
            'ad_hoc_mode': 'extract_width_only',
            'remove_components': cfg.get('remove_components'),
        }
        rerun_cfg = cfg.copy()
        rerun_cfg['extract_width'] = resolve_ad_hoc_extract_width(cfg)
        stage3_results, best_extract_width, _, best_row_idx = run_ad_hoc_extract_width_search(
            stage2_files, rerun_cfg, centroids_df, deepframe, baseline_ints, wave_range, w1, w2,
            name_str, base_row_values
        )

        fancyprint("Generating optimization plots...")
        plot_cost(name_str, outdir=outdir_f)
        make_diagnostic_plot(stage3_results, name_str, baseline_ints=baseline_ints, obs=obs_early,
                             filter=filter_early, outdir=outdir_f)

        outfile = os.path.join(outdir_f, f"LightCurve_Scatter{name_str}.txt")
        specfile = find_stage3_spectrum_file(extract_method)
        plot_scatter(
            txtfile=outfile,
            rows=[best_row_idx],
            wave_range=wave_range_plot,
            smooth=10,
            spectrum_files=[specfile],
            ylim=ylim_plot,
            style="line",
            save_path=os.path.join(outdir_f, f"Scatter_Plot{name_str}.png"),
        )

        t1 = time.perf_counter() - t0_total
        h, m = divmod(int(t1), 3600)
        m, s = divmod(m, 60)
        width_label = 'OPTIMAL EXTRACT_WIDTH'
        if cfg.get('optimize_extract_width', False) is not True:
            width_label = 'USED EXTRACT_WIDTH'
        fancyprint(f"\n{'='*60}")
        fancyprint(f"TOTAL RUNTIME: {h}h {m:02d}min {s:02d}s")
        fancyprint(f"{width_label}: {best_extract_width}")
        fancyprint(f"{'='*60}\n")
        return

    # ===========================================================
    # ===== Special Case 2: PCA + Optimize Extraction Width =====
    # ===========================================================
    if from_pca_only:
        fancyprint(f"\n{'='*60}")
        fancyprint("FROM PCA ONLY MODE ENABLED")
        fancyprint("Restarting from existing BadPix outputs and rerunning PCA/Stage 3 only")
        fancyprint(f"{'='*60}\n")

        optimize_flags = [
            k for k in cfg.keys()
            if k.startswith('optimize_') and k not in ['optimize_extract_width',
                                                       'optimize_extract_width_only',
                                                       'optimize_from_pca_only']
        ]
        for flag in optimize_flags:
            if cfg[flag]:
                raise ValueError(
                    f"{flag} must be False when from_pca_only=True. "
                    "Only optimize_extract_width may be True in this mode."
                )

        remove_components = cfg.get('remove_components')
        if remove_components in [None, 'None', 'null', '']:
            raise ValueError("remove_components must be set when from_pca_only=True")

        fancyprint("Looking for existing BadPix Step outputs...")
        badpix_files = find_existing_stage2_outputs(
            [f'{outdir_s2}*_badpixstep.fits'],
            f"No BadPix Step outputs found in {outdir_s2}. "
            "Please run the optimizer through BadPixStep before using from_pca_only mode."
        )
        fancyprint("Looking for centroids file...")
        centroids_df = load_ad_hoc_centroids(cfg)

        pca_skip_steps = ['AssignWCSStep', 'FlatFieldStep', 'BackgroundStep', 'OneOverFStep',
                          'BadPixStep']
        fancyprint(f"Rerunning PCAReconstructStep with remove_components={remove_components}")
        stage2_results, deepframe = run_stage2(
            badpix_files,
            mode=cfg['observing_mode'],
            soss_background_model=cfg.get('soss_background_file'),
            baseline_ints=cfg['baseline_ints'],
            save_results=True,
            force_redo=True,
            space_thresh=cfg.get('space_outlier_threshold'),
            time_thresh=cfg.get('time_outlier_threshold'),
            remove_components=remove_components,
            pca_components=cfg.get('pca_components'),
            soss_timeseries=cfg.get('soss_timeseries'),
            soss_timeseries_o2=cfg.get('soss_timeseries_o2'),
            oof_method=cfg.get('oof_method'),
            output_tag=cfg['output_tag'],
            skip_steps=pca_skip_steps,
            generate_lc=cfg.get('generate_lc'),
            soss_inner_mask_width=cfg.get('soss_inner_mask_width'),
            soss_outer_mask_width=cfg.get('soss_outer_mask_width'),
            nirspec_mask_width=cfg.get('nirspec_mask_width'),
            pixel_masks=cfg.get('outlier_maps'),
            f277w=cfg.get('f277w'),
            do_plot=cfg.get('do_plots', False),
            centroids=cfg.get('centroids'),
            miri_trace_width=cfg.get('miri_trace_width'),
            miri_background_width=cfg.get('miri_background_width'),
            miri_background_method=cfg.get('miri_background_method'),
            root_dir=root_dir,
            **cfg.get('stage2_kwargs', {})
        )
        if deepframe is None:
            deepframe = resolve_ad_hoc_deepframe(cfg)

        base_row_values = {
            'ad_hoc_mode': 'from_pca_only',
            'remove_components': remove_components,
        }
        rerun_cfg = cfg.copy()
        rerun_cfg['extract_width'] = resolve_ad_hoc_extract_width(cfg)
        stage3_results, best_extract_width, _, best_row_idx = run_ad_hoc_extract_width_search(
            stage2_results, rerun_cfg, centroids_df, deepframe, baseline_ints, wave_range, w1, w2,
            name_str, base_row_values
        )

        fancyprint("Generating optimization plots...")
        plot_cost(name_str, outdir=outdir_f)
        make_diagnostic_plot(stage3_results, name_str, baseline_ints=baseline_ints, obs=obs_early,
                             filter=filter_early, outdir=outdir_f)

        outfile = os.path.join(outdir_f, f"LightCurve_Scatter{name_str}.txt")
        specfile = find_stage3_spectrum_file(extract_method)
        plot_scatter(
            txtfile=outfile,
            rows=[best_row_idx],
            wave_range=wave_range_plot,
            smooth=10,
            spectrum_files=[specfile],
            ylim=ylim_plot,
            style="line",
            save_path=os.path.join(outdir_f, f"Scatter_Plot{name_str}.png"),
        )

        t1 = time.perf_counter() - t0_total
        h, m = divmod(int(t1), 3600)
        m, s = divmod(m, 60)
        width_label = 'OPTIMAL EXTRACT_WIDTH'
        if cfg.get('optimize_extract_width', False) is not True:
            width_label = 'USED EXTRACT_WIDTH'
        fancyprint(f"\n{'='*60}")
        fancyprint(f"TOTAL RUNTIME: {h}h {m:02d}min {s:02d}s")
        fancyprint(f"{width_label}: {best_extract_width}")
        fancyprint(f"REMOVE_COMPONENTS: {format_log_value(remove_components)}")
        fancyprint(f"{'='*60}\n")
        return

    # ==============================================
    # ===== Normal Mode: Run Full Optimization =====
    # ==============================================
    # Load input files
    input_files = unpack_input_dir(
        cfg["input_dir"],
        mode=cfg["observing_mode"],
        filetag=cfg["input_filetag"],
        filter_detector=cfg["filter_detector"],
    )
    if isinstance(input_files, np.ndarray):
        input_files = input_files.tolist()

    if not input_files:
        raise RuntimeError(f"No FITS found in {cfg['input_dir']}")

    fancyprint(f"Found {len(input_files)} segment(s) from {cfg['input_dir']}")
    fancyprint(f"=== PHASE 1: OPTIMIZATION ON FIRST SEGMENT ONLY ===")

    # use only first segment for optimization
    single_segment = [input_files[0]]

    param_ranges = {}  # parametrs to optimize
    fixed_params = {}  # fixed parameters

    optimizer_control_flags = {
        'optimize_extract_width_only',
        'optimize_from_pca_only',
    }
    for k, v in cfg.items():
        if k.startswith("optimize_"):
            if k in optimizer_control_flags:
                continue

            param_name = k[len("optimize_"):]

            # Special handling for extract_width - optimize in Phase 2 using custom cost function
            if param_name == 'extract_width':
                if v:
                    vals = cfg[param_name]
                    if not isinstance(vals, list):
                        raise ValueError(f"{param_name} must be list when optimize_{param_name}=True")
                    fancyprint(f"Will optimize: {param_name} in Phase 2 over {vals} (using spectral scatter cost)")
                    # Add to param_ranges so it shows up in logs and plots
                    param_ranges[param_name] = vals
                else:
                    fixed_params[param_name] = cfg[param_name]
                continue  # Skip the normal processing below

            if param_name not in cfg:
                fancyprint(
                    f'Skipping optimizer control flag "{k}" because "{param_name}" is not a config parameter.',
                    msg_type='WARNING'
                )
                continue

            if v:  # true = optimize (sweep)
                vals = cfg[param_name]
                if not isinstance(vals, list):
                    raise ValueError(f"{param_name} must be list when optimize_{param_name}=True")
                param_ranges[param_name] = vals
                fancyprint(f"Will optimize: {param_name} over {vals}")
            else:
                val = cfg[param_name]
                if isinstance(val, list):
                    raise ValueError(f"{param_name} must be single value when optimize_{param_name}=False")
                fixed_params[param_name] = val

    # Initialize swept parameters with the middle candidate until their own sweep runs (an
    # int-truncated mean can fall outside the candidate list, e.g. for float thresholds).
    current_best = {k: v[len(v) // 2] for k, v in param_ranges.items()}
    current_best.update(fixed_params)

    logf = open(f"{outdir_f}/Cost_Summary{name_str}.txt", "w")
    logs = open(f"{outdir_f}/LightCurve_Scatter{name_str}.txt", "w")
    logf.write("\t".join(param_ranges.keys()) + "\tduration_s\tcost\n")

    # Filter checkpoints to only include those with parameters being optimized
    optimization_checkpoints = []
    for checkpoint in all_checkpoints:
        # Check if any params at this checkpoint are being optimized
        params_to_optimize = [p for p in checkpoint['params'] if p in param_ranges]
        if params_to_optimize:
            # Skip Phase 2-only checkpoints during Phase 1
            if checkpoint.get('phase_2_only', False):
                fancyprint(f"Skipping {checkpoint['name']} - will optimize in Phase 2 on full dataset")
                continue
            optimization_checkpoints.append(checkpoint)
            fancyprint(f"Including checkpoint: {checkpoint['name']} with params {params_to_optimize}")

    # Cache for centroids (generated once, reused)
    centroids = None

    # ~~~ OPTIMIZE EACH CHECKPOINT ~~~
    for checkpoint in optimization_checkpoints:
        # check if any params at this checkpoint need optimization
        params_to_optimize = [p for p in checkpoint['params'] if p in param_ranges]

        if not params_to_optimize:
            fancyprint(f"Skipping {checkpoint['name']}: no parameters to optimize")
            continue

        fancyprint(f"\n{'='*60}")
        fancyprint(f"OPTIMIZING AT: {checkpoint['name']} (Stage {checkpoint['stage']})")
        fancyprint(f"Parameters: {params_to_optimize}")
        fancyprint(f"{'='*60}\n")

        # for each parameter at this checkpoint
        for param_name in params_to_optimize:
            param_values = param_ranges[param_name]
            fancyprint(f"\n--- Sweeping {param_name}: {param_values} ---")

            costs = []
            scatters = []

            # sweep through parameter values
            for param_value in param_values:
                t0 = time.perf_counter()

                # updaete config with current parameter
                run_cfg = cfg.copy()
                run_cfg.update(current_best)  # use best values from previous optimizations
                run_cfg[param_name] = param_value  # Current  value

                fancyprint(f"\nTesting {param_name}={param_value}")

                # Delete cached output for the optimization step to force rerun from that step
                delete_checkpoint_outputs(checkpoint['name'], outdir_s1, outdir_s2)

                # run pipeline up to (including this step)
                if checkpoint['stage'] == 1:
                    # Build skip list: skip everything after this step
                    skip_list = checkpoint['skip_after'].copy()

                    # ALSO add user's skip preferences from YAML config
                    for step in stage1_steps:
                        if run_cfg.get(step) == 'skip' and step not in skip_list:
                            if step == 'OneOverFStep_grp':
                                skip_list.append('OneOverFStep')
                            else:
                                skip_list.append(step)

                    # Forward the current time_window (candidate while sweeping it,
                    # winner/fixed value otherwise) to JumpStep.
                    s1_kwargs = stage1_kwargs_with_winners(run_cfg)

                    # Run Stage 1 with force_redo=False (deleted file will trigger rerun from that step)
                    stage1_results = run_stage1(
                        single_segment,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg.get('soss_background_file'),
                        baseline_ints=run_cfg['baseline_ints'],
                        oof_method=run_cfg.get('oof_method'),
                        superbias_method=run_cfg.get('superbias_method'),
                        soss_timeseries=run_cfg.get('soss_timeseries'),
                        soss_timeseries_o2=run_cfg.get('soss_timeseries_o2'),
                        save_results=True,
                        pixel_masks=run_cfg.get('outlier_maps'),
                        force_redo=False,
                        flag_up_ramp=run_cfg.get('flag_up_ramp', False),
                        rejection_threshold=run_cfg.get('jump_threshold', 15),
                        flag_in_time=run_cfg.get('flag_in_time', True),
                        time_rejection_threshold=run_cfg.get('time_jump_threshold'),
                        output_tag=run_cfg['output_tag'],
                        skip_steps=skip_list,
                        do_plot=run_cfg.get('do_plots', False),
                        soss_inner_mask_width=run_cfg.get('soss_inner_mask_width'),
                        soss_outer_mask_width=run_cfg.get('soss_outer_mask_width'),
                        nirspec_mask_width=run_cfg.get('nirspec_mask_width'),
                        centroids=run_cfg.get('centroids'),
                        hot_pixel_map=run_cfg.get('hot_pixel_map'),
                        miri_drop_groups=run_cfg.get('miri_drop_groups'),
                        saturation_threshold=run_cfg.get('saturation_threshold', 80),
                        f277w=run_cfg.get('f277w'),
                        inl_amplitude_file=run_cfg.get('inl_amplitude_file'),
                        inl_periods=run_cfg.get('inl_periods'),
                        root_dir=root_dir,
                        **s1_kwargs
                    )

                    # Extract from Stage 1 output
                    datafile = stage1_results[0]

                elif checkpoint['stage'] == 2:
                    # First, need Stage 1 results (use cached)
                    # Build skip list for Stage 1 based on user config
                    stage1_skip_for_s2 = []
                    for step in stage1_steps:
                        if run_cfg.get(step) == 'skip':
                            if step == 'OneOverFStep_grp':
                                stage1_skip_for_s2.append('OneOverFStep')
                            else:
                                stage1_skip_for_s2.append(step)

                    stage1_results = run_stage1(
                        single_segment,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg.get('soss_background_file'),
                        baseline_ints=run_cfg['baseline_ints'],
                        oof_method=run_cfg.get('oof_method'),
                        superbias_method=run_cfg.get('superbias_method'),
                        soss_timeseries=run_cfg.get('soss_timeseries'),
                        soss_timeseries_o2=run_cfg.get('soss_timeseries_o2'),
                        save_results=True,
                        pixel_masks=run_cfg.get('outlier_maps'),
                        force_redo=False,  # Use cached Stage 1 results
                        flag_up_ramp=run_cfg.get('flag_up_ramp', False),
                        rejection_threshold=run_cfg.get('jump_threshold', 15),
                        flag_in_time=run_cfg.get('flag_in_time', True),
                        time_rejection_threshold=run_cfg.get('time_jump_threshold'),
                        output_tag=run_cfg['output_tag'],
                        skip_steps=stage1_skip_for_s2,
                        do_plot=run_cfg.get('do_plots', False),
                        soss_inner_mask_width=run_cfg.get('soss_inner_mask_width'),
                        soss_outer_mask_width=run_cfg.get('soss_outer_mask_width'),
                        nirspec_mask_width=run_cfg.get('nirspec_mask_width'),
                        centroids=run_cfg.get('centroids'),
                        hot_pixel_map=run_cfg.get('hot_pixel_map'),
                        miri_drop_groups=run_cfg.get('miri_drop_groups'),
                        root_dir=root_dir,
                        saturation_threshold=run_cfg.get('saturation_threshold', 80),
                        f277w=run_cfg.get('f277w'),
                        inl_amplitude_file=run_cfg.get('inl_amplitude_file'),
                        inl_periods=run_cfg.get('inl_periods'),

                        **stage1_kwargs_with_winners(run_cfg)
                    )

                    # Build skip list for Stage 2
                    skip_list = checkpoint['skip_after'].copy()

                    # ALSO add user's skip preferences from YAML config
                    for step in stage2_steps:
                        if run_cfg.get(step) == 'skip' and step not in skip_list:
                            if step == 'OneOverFStep_int':
                                skip_list.append('OneOverFStep')
                            else:
                                skip_list.append(step)

                    # Forward the current box_size/window_size (candidate while
                    # sweeping them, winner/fixed values otherwise) to BadPixStep.
                    s2_kwargs = stage2_kwargs_with_winners(run_cfg)

                    # Run Stage 2 with force_redo=False
                    # The deleted cached file will trigger rerun from that step onward
                    stage2_results, _ = run_stage2(
                        stage1_results,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg.get('soss_background_file'),
                        baseline_ints=run_cfg['baseline_ints'],
                        save_results=True,
                        force_redo=False,  # Use cached until missing file triggers rerun
                        space_thresh=run_cfg.get('space_outlier_threshold'),
                        time_thresh=run_cfg.get('time_outlier_threshold'),
                        remove_components=run_cfg.get('remove_components'),
                        pca_components=run_cfg.get('pca_components'),
                        soss_timeseries=run_cfg.get('soss_timeseries'),
                        soss_timeseries_o2=run_cfg.get('soss_timeseries_o2'),
                        oof_method=run_cfg.get('oof_method'),
                        output_tag=run_cfg['output_tag'],
                        skip_steps=skip_list,
                        generate_lc=run_cfg.get('generate_lc'),
                        soss_inner_mask_width=run_cfg.get('soss_inner_mask_width'),
                        soss_outer_mask_width=run_cfg.get('soss_outer_mask_width'),
                        nirspec_mask_width=run_cfg.get('nirspec_mask_width'),
                        pixel_masks=run_cfg.get('outlier_maps'),
                        f277w=run_cfg.get('f277w'),
                        do_plot=run_cfg.get('do_plots', False),
                        centroids=run_cfg.get('centroids'),
                        miri_trace_width=run_cfg.get('miri_trace_width'),
                        miri_background_width=run_cfg.get('miri_background_width'),
                        miri_background_method=run_cfg.get('miri_background_method'),
                        root_dir=root_dir,
                        **s2_kwargs
                    )

                    datafile = stage2_results[0]

                elif checkpoint['stage'] == 3:
                    # Need Stage 1 and 2 completed first (use cached)
                    # Build skip list for Stage 1 based on user config
                    stage1_skip_for_s3 = []
                    for step in stage1_steps:
                        if run_cfg.get(step) == 'skip':
                            if step == 'OneOverFStep_grp':
                                stage1_skip_for_s3.append('OneOverFStep')
                            else:
                                stage1_skip_for_s3.append(step)

                    stage1_results = run_stage1(
                        single_segment,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg.get('soss_background_file'),
                        baseline_ints=run_cfg['baseline_ints'],
                        oof_method=run_cfg.get('oof_method'),
                        superbias_method=run_cfg.get('superbias_method'),
                        soss_timeseries=run_cfg.get('soss_timeseries'),
                        soss_timeseries_o2=run_cfg.get('soss_timeseries_o2'),
                        save_results=True,
                        pixel_masks=run_cfg.get('outlier_maps'),
                        force_redo=False,
                        flag_up_ramp=run_cfg.get('flag_up_ramp', False),
                        rejection_threshold=run_cfg.get('jump_threshold', 15),
                        flag_in_time=run_cfg.get('flag_in_time', True),
                        time_rejection_threshold=run_cfg.get('time_jump_threshold'),
                        output_tag=run_cfg['output_tag'],
                        skip_steps=stage1_skip_for_s3,
                        do_plot=run_cfg.get('do_plots', False),
                        soss_inner_mask_width=run_cfg.get('soss_inner_mask_width'),
                        soss_outer_mask_width=run_cfg.get('soss_outer_mask_width'),
                        nirspec_mask_width=run_cfg.get('nirspec_mask_width'),
                        centroids=run_cfg.get('centroids'),
                        hot_pixel_map=run_cfg.get('hot_pixel_map'),
                        miri_drop_groups=run_cfg.get('miri_drop_groups'),
                        root_dir=root_dir,
                        saturation_threshold=run_cfg.get('saturation_threshold', 80),
                        f277w=run_cfg.get('f277w'),
                        inl_amplitude_file=run_cfg.get('inl_amplitude_file'),
                        inl_periods=run_cfg.get('inl_periods'),

                        **stage1_kwargs_with_winners(run_cfg)
                    )

                    # Build skip list for Stage 2 based on user config
                    stage2_skip_for_s3 = []
                    for step in stage2_steps:
                        if run_cfg.get(step) == 'skip':
                            if step == 'OneOverFStep_int':
                                stage2_skip_for_s3.append('OneOverFStep')
                            else:
                                stage2_skip_for_s3.append(step)

                    stage2_results, _ = run_stage2(
                        stage1_results,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg.get('soss_background_file'),
                        baseline_ints=run_cfg['baseline_ints'],
                        save_results=True,
                        force_redo=False,
                        space_thresh=run_cfg.get('space_outlier_threshold'),
                        time_thresh=run_cfg.get('time_outlier_threshold'),
                        remove_components=run_cfg.get('remove_components'),
                        pca_components=run_cfg.get('pca_components'),
                        soss_timeseries=run_cfg.get('soss_timeseries'),
                        soss_timeseries_o2=run_cfg.get('soss_timeseries_o2'),
                        oof_method=run_cfg.get('oof_method'),
                        output_tag=run_cfg['output_tag'],
                        skip_steps=stage2_skip_for_s3,
                        generate_lc=run_cfg.get('generate_lc'),
                        soss_inner_mask_width=run_cfg.get('soss_inner_mask_width'),
                        soss_outer_mask_width=run_cfg.get('soss_outer_mask_width'),
                        nirspec_mask_width=run_cfg.get('nirspec_mask_width'),
                        pixel_masks=run_cfg.get('outlier_maps'),
                        f277w=run_cfg.get('f277w'),
                        do_plot=run_cfg.get('do_plots', False),
                        centroids=run_cfg.get('centroids'),
                        miri_trace_width=run_cfg.get('miri_trace_width'),
                        miri_background_width=run_cfg.get('miri_background_width'),
                        miri_background_method=run_cfg.get('miri_background_method'),
                        root_dir=root_dir,
                        **stage2_kwargs_with_winners(run_cfg)
                    )

                    datafile = stage2_results[0]

                # Extract and compute cost <- new function
                # For Phase 1, use a fixed extract_width (will be optimized in Phase 2)
                phase1_extract_width = cfg.get('extract_width')
                if isinstance(phase1_extract_width, list):
                    # If it's a list (optimize_extract_width=True), use middle value for Phase 1
                    phase1_extract_width = phase1_extract_width[len(phase1_extract_width) // 2]
                    fancyprint(f"  Using extract_width={phase1_extract_width} for Phase 1 (will optimize in Phase 2)")

                spectral_dict, centroids = opt_utils.extract_at_step(
                    datafile=datafile,
                    instrument=instrument,
                    extract_width=phase1_extract_width,
                    centroids=centroids,  # Reuse cached
                    baseline_ints=baseline_ints,
                    output_dir=outdir_s2,
                    extract_method=cfg.get('extract_method', 'box'),
                    extract_width_soss2=cfg.get('extract_width_soss2'),
                    extract_step_kwargs=resolve_extract1d_kwargs(cfg)
                )

                # The fast NIRSpec/MIRI optimizer-side extraction uses pixel indices as
                # placeholder wavelengths, so micron-space filtering is only valid here for SOSS.
                phase1_wave_range = phase1_spectral_wave_range(instrument, wave_range)

                cost, scatter = cost_function(
                    spectral_dict,
                    baseline_ints=baseline_ints,
                    wave_range=phase1_wave_range,
                    w1=w1,
                    w2=w2
                )

                # Debug cost details
                fancyprint(f"  Cost function: w1={w1}, w2={w2}, wave_range={phase1_wave_range}")
                fancyprint(f"  Scatter: min={np.nanmin(scatter):.6e}, max={np.nanmax(scatter):.6e}, median={np.nanmedian(scatter):.6e}")
                fancyprint(f"  Valid scatter values: {np.sum(np.isfinite(scatter))}/{len(scatter)}")

                dt = time.perf_counter() - t0
                costs.append(cost)
                scatters.append(scatter)

                fancyprint(f"{param_name}={param_value}: cost={cost:.12f} ({dt:.1f}s)")

                # Log results
                log_line = "\t".join(str(run_cfg.get(p, '')) for p in param_ranges.keys())
                logf.write(f"{log_line}\t{dt:.1f}\t{cost:.12f}\n")
                logf.flush()

                scatter_line = " ".join(f"{x:.10g}" for x in scatter)
                logs.write(f"{scatter_line}\n")
                logs.flush()

            # Find best value for this parameter (non-finite costs cannot win)
            best_idx = select_best_trial(costs, param_name)
            best_value = param_values[best_idx]
            best_cost = costs[best_idx]

            current_best[param_name] = best_value
            fancyprint(f"\n*** Best {param_name}={best_value} with cost={best_cost:.6f} ***\n")

            # The cached step outputs on disk belong to the LAST value tested,
            # not necessarily the winner. Delete them so the next pipeline call
            # (the following sweep, or Phase 2) regenerates this checkpoint --
            # and, lazily, its downstream caches -- with the winning value.
            if best_idx != len(param_values) - 1:
                delete_checkpoint_outputs(checkpoint['name'], outdir_s1, outdir_s2)

    logf.close()
    logs.close()

    # Only plot if Phase 1 actually logged a sweep (e.g., not when only extract_width is
    # being optimized).
    phase1_costs = pd.read_csv(f"{outdir_f}/Cost_Summary{name_str}.txt", sep="\t")
    if len(phase1_costs) > 0:
        fancyprint("\n=== Plotting optimization results ===")
        plot_cost(name_str, outdir=outdir_f)

    # ===== PHASE 2: FULL PIPELINE WITH OPTIMAL PARAMETERS =====
    fancyprint(f"\n{'='*60}")
    fancyprint("PHASE 2: FULL PIPELINE WITH OPTIMAL PARAMETERS")
    fancyprint(f"Using ALL {len(input_files)} segments")
    fancyprint(f"Optimal parameters: {current_best}")
    fancyprint(f"{'='*60}\n")

    #  set up config of full pipeline with optimal parameters
    final_cfg = cfg.copy()
    final_cfg.update(current_best)

    # Build skip lists for Stage 1 and Stage 2 based on config settings
    stage1_skip = []
    for step in stage1_steps:
        if final_cfg.get(step) == 'skip':
            if step == 'OneOverFStep_grp':
                stage1_skip.append('OneOverFStep')
            else:
                stage1_skip.append(step)

    fancyprint(f"Stage 1 steps to skip: {stage1_skip}")

    # Stage 1
    stage1_results = run_stage1(
        input_files,
        mode=final_cfg['observing_mode'],
        soss_background_model=final_cfg.get('soss_background_file'),
        baseline_ints=final_cfg['baseline_ints'],
        oof_method=final_cfg.get('oof_method'),
        superbias_method=final_cfg.get('superbias_method'),
        soss_timeseries=final_cfg.get('soss_timeseries'),
        soss_timeseries_o2=final_cfg.get('soss_timeseries_o2'),
        save_results=True,
        pixel_masks=final_cfg.get('outlier_maps'),
        force_redo=True,
        flag_up_ramp=final_cfg.get('flag_up_ramp', False),
        rejection_threshold=final_cfg.get('jump_threshold', 15),
        flag_in_time=final_cfg.get('flag_in_time', True),
        time_rejection_threshold=final_cfg.get('time_jump_threshold'),
        output_tag=final_cfg['output_tag'],
        skip_steps=stage1_skip,
        do_plot=final_cfg.get('do_plots', False),
        soss_inner_mask_width=final_cfg.get('soss_inner_mask_width'),
        soss_outer_mask_width=final_cfg.get('soss_outer_mask_width'),
        nirspec_mask_width=final_cfg.get('nirspec_mask_width'),
        centroids=final_cfg.get('centroids'),
        hot_pixel_map=final_cfg.get('hot_pixel_map'),
        miri_drop_groups=final_cfg.get('miri_drop_groups'),
        root_dir=root_dir,
        saturation_threshold=final_cfg.get('saturation_threshold', 80),
        f277w=final_cfg.get('f277w'),
        inl_amplitude_file=final_cfg.get('inl_amplitude_file'),
        inl_periods=final_cfg.get('inl_periods'),

        **stage1_kwargs_with_winners(final_cfg)
    )

    # Build skip list for Stage 2
    stage2_skip = []
    for step in stage2_steps:
        if final_cfg.get(step) == 'skip':
            if step == 'OneOverFStep_int':
                stage2_skip.append('OneOverFStep')
            else:
                stage2_skip.append(step)

    fancyprint(f"Stage 2 steps to skip: {stage2_skip}")

    # Stage 2
    stage2_results, final_deepframe = run_stage2(
        stage1_results,
        mode=final_cfg['observing_mode'],
        soss_background_model=final_cfg.get('soss_background_file'),
        baseline_ints=final_cfg['baseline_ints'],
        save_results=True,
        force_redo=True,
        space_thresh=final_cfg.get('space_outlier_threshold'),
        time_thresh=final_cfg.get('time_outlier_threshold'),
        remove_components=final_cfg.get('remove_components'),
        pca_components=final_cfg.get('pca_components'),
        soss_timeseries=final_cfg.get('soss_timeseries'),
        soss_timeseries_o2=final_cfg.get('soss_timeseries_o2'),
        oof_method=final_cfg.get('oof_method'),
        output_tag=final_cfg['output_tag'],
        skip_steps=stage2_skip,
        generate_lc=final_cfg.get('generate_lc'),
        soss_inner_mask_width=final_cfg.get('soss_inner_mask_width'),
        soss_outer_mask_width=final_cfg.get('soss_outer_mask_width'),
        nirspec_mask_width=final_cfg.get('nirspec_mask_width'),
        pixel_masks=final_cfg.get('outlier_maps'),
        f277w=final_cfg.get('f277w'),
        do_plot=final_cfg.get('do_plots', False),
        centroids=final_cfg.get('centroids'),
        miri_trace_width=final_cfg.get('miri_trace_width'),
        miri_background_width=final_cfg.get('miri_background_width'),
        miri_background_method=final_cfg.get('miri_background_method'),
        root_dir=root_dir,
        **stage2_kwargs_with_winners(final_cfg)
    )

    # new_stage2.run_stage2 now returns (results, deepframe), not centroids.
    # If no centroids are explicitly provided (or already saved on disk), let new_stage3 trace
    # them directly from the deepframe during Stage 3 extraction.
    # Only consider centroid tables written for this dataset (e.g., not the other NIRSpec
    # detector's when NRS1 and NRS2 share an output directory).
    final_fileroot = utils.get_filename_root_noseg(utils.get_filename_root(stage2_results))
    try:
        this_centroid = resolve_existing_centroids(final_cfg, fileroot_noseg=final_fileroot)
    except FileNotFoundError:
        fancyprint("No Stage 3 or Stage 2 centroid table found. Stage 3 will trace centroids "
                   "from the deepframe.")
        this_centroid = None

    # Use the deepframe Stage 2 just produced unless one is explicitly set in the config.
    this_deepframe = final_cfg.get('deepframe')
    if this_deepframe in [None, 'None', 'null', '']:
        this_deepframe = final_deepframe

    # ===== OPTIMIZE EXTRACT_WIDTH IF REQUESTED =====
    if cfg.get('optimize_extract_width', False):
        fancyprint(f"\n{'='*60}")
        fancyprint("OPTIMIZING EXTRACT_WIDTH ON FULL DATASET")
        fancyprint(f"Uses same cost function as Phase 1 (spectral scatter)")
        fancyprint(f"{'='*60}\n")

        extract_widths = cfg['extract_width']
        if not isinstance(extract_widths, list):
            extract_widths = [extract_widths]

        extract_costs = []

        # Reopen log files to append extract_width optimization results
        logf = open(f"{outdir_f}/Cost_Summary{name_str}.txt", "a")
        logs = open(f"{outdir_f}/LightCurve_Scatter{name_str}.txt", "a")

        for width in extract_widths:
            fancyprint(f"\nTesting extract_width={width}")
            t0 = time.perf_counter()

            # Run Stage 3 with this extract width
            stage3_results = run_stage3(
                stage2_results,
                save_results=True,
                force_redo=True,
                extract_method=final_cfg['extract_method'],
                soss_specprofile=final_cfg.get('soss_specprofile'),
                centroids=this_centroid,
                extract_width=width,
                extract_width_soss2=final_cfg.get('extract_width_soss2'),
                st_teff=final_cfg.get('st_teff'),
                st_logg=final_cfg.get('st_logg'),
                st_met=final_cfg.get('st_met'),
                planet_letter=final_cfg.get('planet_letter'),
                output_tag=final_cfg['output_tag'],
                do_plot=final_cfg.get('do_plots', False),
                deepframe=this_deepframe,
                saturation_rescue=final_cfg.get('saturation_rescue', False),
                mask_do_not_use_pixels=final_cfg.get('mask_do_not_use_pixels', True),
                root_dir=root_dir,
                **final_cfg.get('stage3_kwargs', {})
            )

            # Compute cost using same function as Phase 1
            cost, scatter = cost_function(
                stage3_results,
                baseline_ints=baseline_ints,
                wave_range=wave_range,
                w1=w1,
                w2=w2
            )

            dt = time.perf_counter() - t0
            extract_costs.append(cost)

            fancyprint(f"extract_width={width}: cost={cost:.12f} ({dt:.1f}s)")

            # Log results to files (same format as Phase 1)
            # Create a temporary config with this extract_width for logging
            log_cfg = current_best.copy()
            log_cfg['extract_width'] = width
            log_line = "\t".join(str(log_cfg.get(p, '')) for p in param_ranges.keys())
            logf.write(f"{log_line}\t{dt:.1f}\t{cost:.12f}\n")
            logf.flush()

            scatter_line = " ".join(f"{x:.10g}" for x in scatter)
            logs.write(f"{scatter_line}\n")
            logs.flush()

        # Select best extract_width (non-finite costs cannot win)
        best_width_idx = select_best_trial(extract_costs, 'extract_width')
        best_extract_width = extract_widths[best_width_idx]
        best_extract_cost = extract_costs[best_width_idx]

        # Close log files
        logf.close()
        logs.close()

        fancyprint(f"\n*** Best extract_width={best_extract_width} with cost={best_extract_cost:.6f} ***\n")

        # Update final config and run one more time with best width
        final_cfg['extract_width'] = best_extract_width
        current_best['extract_width'] = best_extract_width

        # Regenerate plot with extract_width optimization results
        fancyprint("\n=== Updating optimization plot with extract_width results ===")
        plot_cost(name_str, outdir=outdir_f)
        # Final Stage 3 with optimal width
        stage3_results = run_stage3(
            stage2_results,
            save_results=True,
            force_redo=True,
            extract_method=final_cfg['extract_method'],
            soss_specprofile=final_cfg.get('soss_specprofile'),
            centroids=this_centroid,
            extract_width=best_extract_width,
            extract_width_soss2=final_cfg.get('extract_width_soss2'),
            st_teff=final_cfg.get('st_teff'),
            st_logg=final_cfg.get('st_logg'),
            st_met=final_cfg.get('st_met'),
            planet_letter=final_cfg.get('planet_letter'),
            output_tag=final_cfg['output_tag'],
            do_plot=final_cfg.get('do_plots', False),
            deepframe=this_deepframe,
            saturation_rescue=final_cfg.get('saturation_rescue', False),
            mask_do_not_use_pixels=final_cfg.get('mask_do_not_use_pixels', True),
            root_dir=root_dir,
            **final_cfg.get('stage3_kwargs', {})
        )
    else:
        # No optimization, just run Stage 3 once with fixed width
        extract_width_to_use = final_cfg.get('extract_width')
        if isinstance(extract_width_to_use, list):
            extract_width_to_use = extract_width_to_use[0]
        fancyprint(f"\nUsing fixed extract_width={extract_width_to_use}")

        stage3_results = run_stage3(
            stage2_results,
            save_results=True,
            force_redo=True,
            extract_method=final_cfg['extract_method'],
            soss_specprofile=final_cfg.get('soss_specprofile'),
            centroids=this_centroid,
            extract_width=extract_width_to_use,
            extract_width_soss2=final_cfg.get('extract_width_soss2'),
            st_teff=final_cfg.get('st_teff'),
            st_logg=final_cfg.get('st_logg'),
            st_met=final_cfg.get('st_met'),
            planet_letter=final_cfg.get('planet_letter'),
            output_tag=final_cfg['output_tag'],
            do_plot=final_cfg.get('do_plots', False),
            deepframe=this_deepframe,
            saturation_rescue=final_cfg.get('saturation_rescue', False),
            mask_do_not_use_pixels=final_cfg.get('mask_do_not_use_pixels', True),
            root_dir=root_dir,
            **final_cfg.get('stage3_kwargs', {})
        )

    #  Diagnostics
    make_diagnostic_plot(stage3_results, name_str, baseline_ints=baseline_ints, obs=obs_early,
                         filter=filter_early, outdir=outdir_f)

    #  scatter plot of the final Stage 3 spectrum. Phase 1 costs are computed on a single segment
    #  (and possibly a single group), so they are not comparable with the full-dataset Stage 3
    #  products and must not be plotted on the final wavelength axis.
    _, final_scatter = cost_function(stage3_results, baseline_ints=baseline_ints,
                                     wave_range=wave_range, w1=w1, w2=w2)
    outfile = os.path.join(outdir_f, f"LightCurve_Scatter_Final{name_str}.txt")
    with open(outfile, "w") as f:
        f.write(" ".join(f"{x:.10g}" for x in final_scatter) + "\n")
    specfile = find_stage3_spectrum_file(final_cfg['extract_method'])

    plot_scatter(
        txtfile=outfile,
        rows=[0],
        wave_range=wave_range_plot,
        smooth=10,
        spectrum_files=[specfile],
        ylim=ylim_plot,
        style="line",
        save_path=os.path.join(outdir_f, f"Scatter_Plot{name_str}.png"),
    )

    # ===== ARCHIVE TO LONG-TERM STORAGE =====
    archive_dest = cfg.get('archive_to_longterm_storage')
    if archive_dest and archive_dest not in [None, 'None', 'null', '']:
        fancyprint(f"\n{'='*60}")
        fancyprint("ARCHIVING TO LONG-TERM STORAGE")
        fancyprint(f"{'='*60}\n")

        import shutil

        # Full output directory path (pipeline_outputs_directory + output_tag)
        full_output_dir = full_outdir

        # Get input directory
        input_dir = cfg['input_dir']

        # Trust that archive_dest exists (don't try to create parent dirs like /cds2)
        # User must ensure the archive destination directory exists before running

        # Archive input directory
        if os.path.exists(input_dir):
            input_basename = os.path.basename(input_dir.rstrip('/'))
            archive_input = os.path.join(archive_dest, input_basename)
            try:
                fancyprint(f"Moving input data:")
                fancyprint(f"  From: {input_dir}")
                fancyprint(f"  To:   {archive_input}")
                shutil.move(input_dir, archive_input)
                fancyprint("  ✓ Input data archived successfully")
            except Exception as e:
                fancyprint(f"  ✗ Failed to archive input data: {e}", msg_type='WARNING')
        else:
            fancyprint(f"Input directory not found (already moved?): {input_dir}", msg_type='WARNING')

        # Archive output directory
        if os.path.exists(full_output_dir):
            output_basename = os.path.basename(full_output_dir.rstrip('/'))
            archive_output = os.path.join(archive_dest, output_basename)
            try:
                fancyprint(f"\nMoving pipeline outputs:")
                fancyprint(f"  From: {full_output_dir}")
                fancyprint(f"  To:   {archive_output}")
                shutil.move(full_output_dir, archive_output)
                fancyprint("  ✓ Pipeline outputs archived successfully")
            except Exception as e:
                fancyprint(f"  ✗ Failed to archive outputs: {e}", msg_type='WARNING')
        else:
            fancyprint(f"Output directory not found: {full_output_dir}", msg_type='WARNING')

        fancyprint(f"\n{'='*60}")
        fancyprint("ARCHIVING COMPLETE")
        fancyprint(f"{'='*60}\n")

    #  timing
    t1 = time.perf_counter() - t0_total
    h, m = divmod(int(t1), 3600)
    m, s = divmod(m, 60)
    fancyprint(f"\n{'='*60}")
    fancyprint(f"TOTAL RUNTIME: {h}h {m:02d}min {s:02d}s")
    fancyprint(f"OPTIMAL PARAMETERS: {current_best}")
    fancyprint(f"{'='*60}\n")


# ===== Do Stuff =====
if __name__ == "__main__":
    run_optimizer()
    fancyprint('Done')
