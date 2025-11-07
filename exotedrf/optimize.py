#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 00:00 2025

@author: PSD

Script to run the exoTEDRF pipeline optimizer.
"""

# ======== STANDARD LIBRARY IMPORTS ========
import os       # For file paths and environment variable handling
import sys      # For exiting with error messages
import glob     # For matching file patterns
import time     # For timing operations
import argparse # For parsing command-line arguments
import yaml     # For reading YAML configuration files

# ======== PROJECT IMPORTS ========
from exotedrf import utils  # Utility functions for exoTEDRF 

# --------------------------------------------------------
# 1) EARLY CONFIG PARSING
# --------------------------------------------------------
# Create a lightweight ArgumentParser that only looks for --config/-c
# This is done *before* the main parser so we can load config and set env vars
early = argparse.ArgumentParser(add_help=False)
early.add_argument(
    "--config", "-c",
    default="run_optimize.yaml",       # Default config file if none given
    help="Path to your DMS config YAML"
)
# parse_known_args() -> returns parsed args plus the remaining unparsed args
args, remaining = early.parse_known_args()

# --------------------------------------------------------
# 2) LOAD CONFIG & SET JWST CRDS ENVIRONMENT VARIABLES
# --------------------------------------------------------
# Read the YAML config file to get CRDS path/context before importing JWST modules
try:
    cfg_early = yaml.safe_load(open(args.config))
except FileNotFoundError:
    sys.exit(f"ERROR: config file '{args.config}' not found.")

# Set CRDS cache path (local storage for JWST calibration files)
os.environ.setdefault(
    "CRDS_PATH",
    cfg_early.get("crds_cache_path", "./crds_cache")
)
# Set CRDS server URL (location of JWST calibration data online)
os.environ.setdefault(
    "CRDS_SERVER_URL",
    "https://jwst-crds.stsci.edu"
)
# Set CRDS context (specific calibration reference mapping to use)
os.environ.setdefault(
    "CRDS_CONTEXT",
    cfg_early.get("crds_context", "jwst_1322.pmap")
)

# --------------------------------------------------------
# 3) NUMERICAL / PLOTTING IMPORTS
# --------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits

# --------------------------------------------------------
# 4) ADDITIONAL PROJECT IMPORTS
# --------------------------------------------------------
from exotedrf.utils import parse_config, unpack_input_dir, fancyprint
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3, do_box_extraction
from exotedrf.optimize_helpers import extract_at_step


# ======== OUTPUT DIRECTORY DEFINITIONS ========
# Define where to store outputs for each pipeline stage
outdir    = 'pipeline_outputs_directory'         # Main output root
outdir_f  = 'pipeline_outputs_directory/Files'   # Generic files (tables, logs, etc.)
outdir_s1 = 'pipeline_outputs_directory/Stage1/' # Stage 1 calibrated data
outdir_s2 = 'pipeline_outputs_directory/Stage2/' # Stage 2 calibrated data
outdir_s3 = 'pipeline_outputs_directory/Stage3/' # Stage 3 processed data
outdir_s4 = 'pipeline_outputs_directory/Stage4/' # Stage 4 final results

# Ensure that all required output directories exist (create if missing)
utils.verify_path('pipeline_outputs_directory')
utils.verify_path('pipeline_outputs_directory/Files')
utils.verify_path('pipeline_outputs_directory/Stage1')
utils.verify_path('pipeline_outputs_directory/Stage2')
utils.verify_path('pipeline_outputs_directory/Stage3')
utils.verify_path('pipeline_outputs_directory/Stage4')

# ======== OBSERVING CONFIG PARAMETERS ========
# Observation mode in lowercase (e.g., 'niriss', 'nirspec', 'miri')
obs_early = (cfg_early.get('observing_mode') or '').lower()
# Detector filter in lowercase (e.g., 'clear', 'nrs1', 'nrs2')
filter_early = (cfg_early.get('filter_detector') or '').lower()
# Wavelength range limits for analysis and plotting (if provided in config)
wave_range_early      = cfg_early.get('wave_range', None)
wave_range_plot_early = cfg_early.get('wave_range_plot', None)
# Weighting factors for cost function or metrics
w1 = cfg_early.get('w1', 0.0)
w2 = cfg_early.get('w2', 1.0)

# ======== INSTRUMENT WAVELENGTH LIMITS ========
# Allowed wavelength coverage for each instrument (microns)
bands = {
    'miri':    (5.0, 13.0),
    'nirspec': (1.0, 5.0),
    'niriss':  (1.0, 2.8)
}

# ======== VALIDATION: CHECK WAVELENGTH RANGE AGAINST INSTRUMENT LIMITS ========
# Loop through instruments to find the matching one for this observation
for key, (lo, hi) in bands.items():
    if key in obs_early:
        # Validate that provided ranges (if any) fall within allowed limits
        for name, rng in (('wave_range', wave_range_early),
                          ('wave_range_plot', wave_range_plot_early)):
            if rng is not None and not (lo <= min(rng) and max(rng) <= hi):
                raise ValueError(f"{name}={rng!r} out of allowed band [{lo}, {hi}]")
        break
# If no instrument key matched the observation mode, throw an error
else:
    raise ValueError(f"Unrecognized observing_mode: {cfg_early.get('observing_mode')}")


# ----------------------------------------
# Plot the cost values from a parameter sweep
# ----------------------------------------
def plot_cost(name_str, table_height=0.4):
    """
    Reads a tab-delimited cost file, detects parameter sweeps, highlights 
    the best parameter set(s), and produces a figure showing cost trends.

    Parameters
    ----------
    name_str : str
        Identifier used to find the cost file (Cost_<name_str>.txt).
    table_height : float
        Fraction of the figure height to allocate to the table display.
    """

    # ======== LOAD AND CLEAN DATA ========
    # Read cost file for the given run name
    df = pd.read_csv(f"pipeline_outputs_directory/Files/Cost_{name_str}.txt",
                     delimiter="\t")

    # Remove rows where 'cost' is not numeric
    df = df[pd.to_numeric(df["cost"], errors="coerce").notna()].reset_index(drop=True)

    # Get all parameter columns (exclude 'duration_s' and 'cost' at the end)
    param_cols = df.columns[:-2]

    # ======== DETECT WHICH PARAMETER CHANGED PER ROW (sweep-aware) ========
    changed_param_per_row = [None] * len(df)
    
    # current sweep = first differing column between row 0 and 1 (fallback to first varying col)
    if len(df) > 1:
        diffs01 = [c for c in param_cols if df.at[1, c] != df.at[0, c]]
        if diffs01:
            current_param = diffs01[0]
        else:
            # fallback: first column that varies anywhere
            vary = [c for c in param_cols if df[c].nunique(dropna=False) > 1]
            current_param = vary[0] if vary else param_cols[0]
    else:
        current_param = param_cols[0]
    
    changed_param_per_row[0] = current_param
    changed_param_per_row[1 if len(df) > 1 else 0] = current_param
    
    # find sweep boundaries: as soon as ANY other parameter changes, the next sweep starts
    sweep_lines = []  # indices where a new sweep begins
    for i in range(1, len(df)):
        diffs = [c for c in param_cols if df.at[i, c] != df.at[i-1, c]]
        if not diffs:  # nothing changed -> stay in current sweep
            changed_param_per_row[i] = current_param
            continue
    
        if current_param in diffs and len(diffs) == 1:
            # only the active param changed -> still same sweep
            changed_param_per_row[i] = current_param
        else:
            # another param appeared (possibly with the current one reverting)
            # new sweep starts at this row
            new_param = next((c for c in diffs if c != current_param), diffs[0])
            sweep_lines.append(i)
            current_param = new_param
            changed_param_per_row[i] = current_param
    
    # first row label belongs to the first detected sweep
    if len(df) >= 2 and changed_param_per_row[0] is None:
        changed_param_per_row[0] = changed_param_per_row[1] or param_cols[0]
    
    sweep_boundaries = [0] + sweep_lines + [len(df)]

    # ======== BUILD LABELS AND FIND SWEEP BOUNDARIES ========
    labels = []
    sweep_lines = []  # indices where a new parameter sweep starts
    last_changed_param = None
    for idx, row in df.iterrows():
        changed_param = changed_param_per_row[idx]
        # Start a new sweep if parameter changes
        if changed_param != last_changed_param and last_changed_param is not None:
            sweep_lines.append(idx)

        # Format value (use integer if no fractional part)
        value = row[changed_param]
        try:
            fv = float(value)
            value = int(fv) if fv.is_integer() else fv
        except Exception:
            pass

        labels.append(f"{changed_param}={value}")
        last_changed_param = changed_param

    df["changed_label"] = labels

    # ======== HIGHLIGHT BEST COST PER SWEEP ========
    sweep_boundaries = [0] + sweep_lines + [len(df)]
    colors = ['gray'] * len(df)  # default color
    for i in range(len(sweep_boundaries) - 1):
        start = sweep_boundaries[i]
        end = sweep_boundaries[i+1]
        # Get index of min cost in this sweep
        min_idx = df.iloc[start:end]["cost"].idxmin()
        colors[min_idx] = 'green'

    # ======== BEST OVERALL PARAMETERS ========
    best_row = df.loc[df["cost"].idxmin(), param_cols.tolist() + ["cost"]].copy()
    # Pretty-print numeric values
    for col in best_row.index:
        val = best_row[col]
        try:
            fv = float(val)
            best_row[col] = int(fv) if fv.is_integer() else fv
        except Exception:
            best_row[col] = val
    best_df = pd.DataFrame([best_row]).reset_index(drop=True)

    # ======== FIGURE LAYOUT ========
    fig = plt.figure(figsize=(max(14, len(df) * 0.25), 10))
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[1 - table_height, table_height])
    ax_plot = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    # Scatter plot of cost
    ax_plot.scatter(range(len(df)), df["cost"].values, color=colors)
    # Vertical dashed lines for sweep boundaries
    for x in sweep_lines:
        ax_plot.axvline(x=x - 0.5, color='gray', linestyle='--', linewidth=1)

    # X-axis tick labels -> just the value part of "param=value"
    values = [lbl.split('=', 1)[1] for lbl in df["changed_label"]]
    ax_plot.set_xticks(range(len(df)))
    ax_plot.set_xticklabels(values, rotation=0, fontsize=8)

    # Drop parameter names under x-axis, alternating heights to avoid overlap
    ymin, ymax = ax_plot.get_ylim()
    base_y = ymin - 0.08 * (ymax - ymin)
    alt_y  = ymin - 0.15 * (ymax - ymin)
    for i, (start, end) in enumerate(zip(sweep_boundaries[:-1], sweep_boundaries[1:])):
        param_name = df.loc[start, "changed_label"].split("=", 1)[0]
        center = (start + end - 1) / 2
        y_pos = base_y if i % 2 == 0 else alt_y
        ax_plot.text(center, y_pos, param_name, ha="center", va="top", fontsize=10)

    fig.subplots_adjust(bottom=0.30)
    ax_plot.set_ylabel("Cost (ppm)")
    ax_plot.set_title(f"Cost by Single Parameter Sweep: {name_str}")

    # ======== TABLE OF BEST PARAMETERS ========
    ax_table.axis("off")
    ax_table.text(0.5, 0.65, "Best Parameters", ha="center", va="bottom", fontsize=12)
    table = ax_table.table(
        cellText=best_df.values,
        colLabels=best_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.scale(1.0, 1.8)
    table.auto_set_font_size(False)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(7)   # header
        else:
            cell.set_fontsize(10)  # data

    # Save final plot to PNG
    fig.savefig(f"pipeline_outputs_directory/Files/Cost_{name_str}.png",
                dpi=300, bbox_inches='tight')

# ----------------------------------------
# create filenames
# ----------------------------------------
def make_step_filenames(input_files, output_dir, possible_steps, 
                        output_dir_2nd=None, possible_steps_2nd=None):
    """
    Search for files in output_dir matching any of the given step suffixes.
    If found, return regenerated filenames aligned to input_files.
    If not found and a second dir/list are given, search there.
    If still nothing, raise FileNotFoundError.

    Parameters
    ----------
    input_files : list[str]
        List of original input file paths.
    output_dir : str
        Primary directory to search for processed files.
    possible_steps : list[str]
        Ordered list of step suffixes to try (e.g., ['darkcurrentstep', 'refpixstep']).
    output_dir_2nd : str, optional
        Secondary directory to search if nothing found in primary.
    possible_steps_2nd : list[str], optional
        Steps to try in secondary directory.

    Returns
    -------
    list[str]
        Paths to regenerated filenames for the found step.
    """

    # Helper function to regenerate full output filenames
    def _regen(dirpath, step):
        """
        Given a directory and a step suffix, build output filenames
        by replacing the suffix of each input file with the given step.
        """
        out = []
        for f in input_files:
            base = os.path.basename(f)                # just filename, no path
            root = base[: base.rfind("_")]            # remove everything after last underscore
            out.append(os.path.join(dirpath, f"{root}_{step}.fits"))
        return out

    # 1) Primary search: loop over possible steps and check for matches in output_dir
    for step in possible_steps:
        if glob.glob(os.path.join(output_dir, f"*_{step}.fits")):
            print(f"Found step '{step}' in {output_dir}")
            return _regen(output_dir, step)

    # 2) Secondary search: same logic, but in output_dir_2nd if provided
    if output_dir_2nd and possible_steps_2nd:
        for step in possible_steps_2nd:
            if glob.glob(os.path.join(output_dir_2nd, f"*_{step}.fits")):
                print(f"Found step '{step}' in {output_dir_2nd}")
                return _regen(output_dir_2nd, step)

    # 3) No match found in either directory -> raise error
    raise FileNotFoundError(
        f"No matching step files found in '{output_dir}'"
        + (f" or '{output_dir_2nd}'" if output_dir_2nd else "")
    )


# ----------------------------------------
# cost function (P2P-based)
# ----------------------------------------
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
        flux_O1 = np.asarray(st3['Flux O1'], float)  # Order 1 flux
        flux_O2 = np.asarray(st3['Flux O2'], float)  # Order 2 flux
        wave_O1 = np.asarray(st3['Wave O1'], float)  # Order 1 wavelengths
        wave_O2 = np.asarray(st3['Wave O2'], float)  # Order 2 wavelengths

        cutoff = 0.85  # μm — wavelength boundary between O2 and O1 segments

        # Find O2 indices up to cutoff
        i2 = np.where(wave_O2 <= cutoff)[0]
        # Find O1 indices above cutoff
        i1 = np.where(wave_O1 > cutoff)[0]

        if i2.size == 0 or i1.size == 0:
            raise ValueError("Cutoff produces empty segment: "
                             f"O2<= {cutoff}: {i2.size}, O1> {cutoff}: {i1.size}")

        idx2 = i2[-1]  # last valid O2 index
        idx1 = i1[0]   # first valid O1 index

        # Concatenate O2 segment + O1 segment along wavelength axis
        wave = np.concatenate([wave_O2[:idx2+1],        wave_O1[idx1:]])
        flux = np.concatenate([flux_O2[:, :idx2+1],     flux_O1[:, idx1:]], axis=1)

        # Sort by wavelength just in case
        s = np.argsort(wave)
        wave = wave[s]
        flux = flux[:, s]

    else:
        # For non-NIRISS: take flux/wave arrays directly
        flux = np.asarray(st3['Flux'], float)
        wave = np.asarray(st3['Wave'], float)

    # ======== WHITE-LIGHT TERM ========
    # Collapse all wavelengths into single white-light curve
    white      = np.nansum(flux, axis=1)
    white      = white[~np.isnan(white)]
    norm_white = white / np.median(white)
    # 2nd finite difference (neighbor avg - center)
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
        low_term  = np.nanmedian(np.abs(d2_spec[:Nlow]), axis=0)
        high_term = np.nanmedian(np.abs(d2_spec[Nhigh:]), axis=0)
        ptp2_spec_wave = 0.5 * (low_term + high_term)
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

        # Distances from requested range edges
        dist_lo = np.abs(wave - lo); dist_lo[~finite] = np.inf
        dist_hi = np.abs(wave - hi); dist_hi[~finite] = np.inf

        idx_lo = int(np.argmin(dist_lo))
        idx_hi = int(np.argmin(dist_hi))

        # Check tolerance
        if dist_lo[idx_lo] > tol or dist_hi[idx_hi] > tol:
            raise ValueError(f"wave_range {wave_range} not found within ±{tol}")

        # Slice range in correct order
        i0, i1 = sorted((idx_lo, idx_hi))
        sub = ptp2_spec_wave[i0:i1+1]
        if np.all(np.isnan(sub)):
            raise ValueError(f"No valid ptp2_spec values in wave range {wave_range}")
        ptp2_spec = np.nanmedian(sub)

    else:
        raise ValueError("wave_range must be None or a length-2 list/tuple")

    # ======== FINAL COST COMBINATION ========
    cost = w1 * ptp2_white + w2 * ptp2_spec

    return cost, ptp2_spec_wave



# ----------------------------------------
# diagnostic plot
# ----------------------------------------
def diagnostic_plot(st3, name_str, baseline_ints, outdir=outdir_f):
    """
    Create two diagnostic plots from Stage-3 data:
      1) Normalized white-light curve
      2) Normalized flux image with true wavelength mapping

    Parameters
    ----------
    st3 : dict-like
        Stage-3 outputs containing flux and wavelength arrays.
        For NIRISS/SOSS: requires 'Flux_O1', 'Flux_O2', 'Wave_O1', 'Wave_O2'.
        For others: requires 'Flux', 'Wave'.
    name_str : str
        Identifier used in output filenames.
    baseline_ints : list[int]
        One or two integers for baseline integrations:
            [N] -> normalize by median of first N integrations
            [Nlow, Nhigh] -> normalize by mean of medians of start and end segments
    outdir : str
        Output directory for saved figures.
    """

    os.makedirs(outdir, exist_ok=True)

    # ======== WAVELENGTH RANGE SELECTION BASED ON MODE/FILTER ========
    # obs_early and filter_early must be defined globally before calling
    if 'miri' in obs_early:
        wave_min, wave_max = 5.0, 12.0
    elif 'niriss' in obs_early:
        wave_min, wave_max = 0.6, 2.8
    elif 'nirspec' in obs_early:
        if filter_early == 'nrs1':
            wave_min, wave_max = 2.9, None
        elif filter_early == 'nrs2':
            wave_min, wave_max = None, 2.9
        else:
            raise ValueError(f"Unknown nirspec filter_detector: {filter_early}")
    else:
        raise ValueError(f"Unknown observing_mode: {obs_early}")

    # --- Build stitched spectrum ---
    if 'niriss' in obs_early:
        # Load flux and wavelength for both spectral orders
        flux_O1 = np.asarray(st3['Flux O1'], float)
        flux_O2 = np.asarray(st3['Flux O2'], float)
        wave_O1 = np.asarray(st3['Wave O1'], float)
        wave_O2 = np.asarray(st3['Wave O2'], float)

        # Cutoff wavelength separating orders
        cutoff = 0.85  # µm

        # Indices: O2 wavelengths ≤ cutoff, O1 wavelengths > cutoff
        i2 = np.where(wave_O2 <= cutoff)[0]
        i1 = np.where(wave_O1 > cutoff)[0]
        if i2.size == 0 or i1.size == 0:
            raise ValueError(
                f"Cutoff {cutoff} yields empty segment: "
                f"O2<= {i2.size}, O1> {i1.size}"
            )

        # Concatenate both orders along wavelength axis
        wave = np.concatenate([wave_O2[:i2[-1]+1], wave_O1[i1[0]:]])
        flux = np.concatenate([flux_O2[:, :i2[-1]+1], flux_O1[:, i1[0]:]], axis=1)
    else:
        # Non-NIRISS: directly load single flux/wavelength arrays
        flux = np.asarray(st3['Flux'], float)
        wave = np.asarray(st3['Wave'], float)

    # --- Apply wavelength range filter ---
    mask = np.isfinite(wave)
    if wave_min is not None:
        mask &= wave >= wave_min
    if wave_max is not None:
        mask &= wave <= wave_max
    wave = wave[mask]
    flux = flux[:, mask]

    # --- Sort by wavelength ---
    # mergesort preserves order for equal wavelengths (stable sort)
    s = np.argsort(wave, kind='mergesort')
    wave = wave[s]
    flux = flux[:, s]

    # --- Drop bad columns and enforce strictly increasing wavelengths ---
    # Column median across time for each spectral channel
    col_med = np.nanmedian(flux, axis=0)
    # Keep only finite wavelengths, finite medians, and non-zero medians
    good = np.isfinite(wave) & np.isfinite(col_med) & (col_med != 0)
    wave = wave[good]
    flux = flux[:, good]

    # --- Collapse duplicate wavelengths ---
    # Round wavelengths to tolerance to handle floating-point noise
    w_round = np.round(wave, 12)
    _, keep_idx = np.unique(w_round, return_index=True)
    keep_idx.sort()  # keep in ascending order
    wave = wave[keep_idx]
    flux = flux[:, keep_idx]

    # --- White-light curve ---
    # Sum flux over all spectral channels for each integration
    white = np.nansum(flux, axis=1)
    if len(baseline_ints) == 1:
        # Normalize by median of first N integrations
        N = int(baseline_ints[0])
        norm_white = white / np.median(white[:N])
    else:
        # Normalize by mean of medians from start and end segments
        Nlow, Nhigh = map(int, baseline_ints)
        base = 0.5 * (
            np.median(white[:Nlow]) +
            np.median(white[Nhigh:])
        )
        norm_white = white / base

    # --- Plot normalized white-light curve ---
    plt.figure()
    plt.plot(norm_white, marker='.')
    plt.xlabel("Integration Number")
    plt.ylabel("Normalized White Flux")
    plt.title("Normalized White-light Curve")
    plt.grid(True)
    plt.savefig(f"{outdir}/norm_white_{name_str}.png", dpi=300)
    plt.close()

    # --- Normalized flux image with true wavelength mapping ---
    # Normalize each column by its time median (post-cleaning)
    img = np.full_like(flux, np.nan, dtype=float)
    img[:, :] = flux / col_med[good][keep_idx]  # safe: filtered for finite non-zero values

    n_int, n_pix = img.shape

    # Require strictly increasing wavelength for pcolormesh bin edges
    if not np.all(np.diff(wave) > 0):
        raise ValueError("wave must be strictly increasing for pcolormesh")

    # Compute wavelength bin edges for pcolormesh
    dw = np.diff(wave)
    edges = np.empty(n_pix + 1, float)
    edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])  # midpoints
    edges[0] = wave[0] - dw[0] / 2              # lower bound
    edges[-1] = wave[-1] + dw[-1] / 2           # upper bound

    # Integration edges for x-axis
    x = np.arange(n_int + 1)

    # Plot normalized flux image
    plt.figure()
    plt.pcolormesh(x, edges, img.T, shading="auto", vmin=0.98, vmax=1.02)
    plt.xlabel("Integration Number")
    plt.ylabel("Wavelength (µm)")
    plt.title("Normalized Flux Image")
    plt.colorbar(label="Relative Flux")
    plt.savefig(f"{outdir}/flux_img_{name_str}.png", dpi=300)
    plt.close()



# ----------------------------------------
# Plot Scatter
# ----------------------------------------
def plot_scatter(  
    txtfile, rows,
    wave_range=None, smooth=None,
    spectrum_files=None,
    style='line', ylim=None, save_path=None,
    tol=0.05
):
    """
    Plot point-to-point (P2P) scatter vs wavelength for selected rows from a scatter table.

    Overlays for each selected row:
      1) Smoothed series using a moving-average window (`smooth`) if provided
      2) Raw (unsmoothed) series

    Photon-noise curves are intentionally excluded from this plot.

    Parameters
    ----------
    txtfile : str
        Path to the whitespace-delimited scatter table.
    rows : list[int]
        Indices of the table rows to plot. Negative indices count from the end.
    wave_range : tuple(float, float), optional
        Wavelength range to plot (μm), with tolerance `tol`.
    smooth : int, optional
        Window size (in pixels) for moving-average smoothing.
    spectrum_files : list[str]
        List of spectrum FITS files to retrieve wavelength axis from.
    style : {'line', 'scatter'}
        Plotting style.
    ylim : tuple(float, float), optional
        y-axis limits.
    save_path : str, optional
        If given, save the plot to this file.
    tol : float
        Allowed margin when applying wave_range filtering.
    """

    # --- Load scatter table ---
    # Read whitespace-delimited table, replace NaNs with 0.0
    df = pd.read_csv(txtfile, sep=r'\s+', header=None).fillna(0.0)
    n_rows, n_cols = df.shape

    # --- Validate requested rows ---
    valid = []
    for r in rows:
        # Convert negative indices to positive equivalents
        i = r if r >= 0 else n_rows + r
        if 0 <= i < n_rows:
            valid.append(i)
        else:
            print(f"Warning: row {r} out of range, skipping.")
    if not valid:
        raise ValueError("No valid rows to plot.")

    # --- Load wavelength grid to match scatter columns ---
    if not spectrum_files:
        raise ValueError("`spectrum_files` is required to read the wavelength axis.")
    with fits.open(spectrum_files[0]) as hdus:
        # Create dict mapping sanitized HDU names to HDU objects
        name_map = {h.name.replace(" ", "_"): h
                    for h in hdus if h.data is not None and h.name != "PRIMARY"}

        # Special handling for NIRISS with two orders
        if ("Wave_O1" in name_map) and ("Wave_O2" in name_map):
            wave_O1 = np.asarray(name_map["Wave_O1"].data, float)
            wave_O2 = np.asarray(name_map["Wave_O2"].data, float)
            cutoff = 0.85  # μm: boundary between orders
            # Select O2 wavelengths <= cutoff
            i2 = np.where(np.isfinite(wave_O2) & (wave_O2 <= cutoff))[0]
            # Select O1 wavelengths > cutoff
            i1 = np.where(np.isfinite(wave_O1) & (wave_O1 > cutoff))[0]
            if i2.size == 0 or i1.size == 0:
                raise ValueError(f"Cutoff {cutoff} yields empty segment: "
                                 f"O2<={i2.size}, O1>{i1.size}")
            # Concatenate the valid segments
            wave_full = np.concatenate([wave_O2[:i2[-1]+1], wave_O1[i1[0]:]])
        else:
            # Fallback: read first extension array as wavelength grid
            wave_full = np.asarray(hdus[1].data, float)

    # --- Ensure monotonic wavelength and align scatter columns ---
    # Sort indices by wavelength, keeping equal values in original order (stable)
    s = np.argsort(wave_full, kind="mergesort")
    wave_sorted = wave_full[s]

    # Wavelength length must match scatter table column count
    if wave_sorted.size != n_cols:
        raise ValueError(f"Wavelength length {wave_sorted.size} != scatter columns {n_cols}")

    # Build boolean mask for desired wavelength range
    if wave_range is not None:
        wmin, wmax = wave_range
        mask = np.isfinite(wave_sorted) & (wave_sorted >= wmin - tol) & (wave_sorted <= wmax + tol)
    else:
        mask = np.isfinite(wave_sorted)
    if not mask.any():
        raise ValueError(f"No finite wavelengths within selected range {wave_range}.")

    # Final x-axis values
    x = wave_sorted[mask]

    # --- Plot ---
    plt.figure(figsize=(8, 4))

    for i in valid:
        # Extract row data and reorder columns to match wavelength order
        y_full = df.iloc[i, :].to_numpy(float)
        y_ord = y_full[s]

        # Raw series (convert to ppm)
        y_raw = (y_ord[mask]) * 1e6
        if style == 'line':
            plt.plot(x, y_raw, linewidth=0.6, linestyle='-', alpha=0.5,
                     color='grey', label="Best Parameter configuration (raw)")
        else:
            plt.scatter(x, y_raw, s=3, alpha=0.8,
                        label="Best Parameter configuration (raw)")

        # Smoothed series (moving average)
        if smooth and int(smooth) > 1:
            w = int(smooth)
            kern = np.ones(w, dtype=float) / w
            y_sm_all = np.convolve(y_ord, kern, mode='same')
            y_sm = (y_sm_all[mask]) * 1e6
            if style == 'line':
                plt.plot(x, y_sm, linewidth=1.0,
                         label=f"Best Parameter configuration (smoothed:{w})")
            else:
                plt.scatter(x, y_sm, s=6,
                            label=f"Best Parameter configuration (smoothed:{w})")

    # --- Finalize plot ---
    plt.xlim(x.min(), x.max())
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Scatter (ppm)")
    plt.legend(ncol=2, fontsize='small')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    plt.show()



# ----------------------------------------
# skip step list
# ----------------------------------------
def get_stage_skips(cfg, steps, always_skip=None, special_one_over_f=False):
    """
    Build a list of pipeline steps to skip based on a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Configuration mapping step names to actions (e.g., {'DarkCurrentStep': 'run'}).
    steps : list[str]
        Candidate step names to check.
    always_skip : list[str], optional
        Steps to skip unconditionally, regardless of cfg settings.
    special_one_over_f : bool
        If True, treat any step whose name starts with 'OneOverFStep' as 'OneOverFStep'
        when adding to skip list. Useful if different variants exist.

    Returns
    -------
    list[str]
        Steps to skip for this run.
    """

    # Initialize skip set from always_skip (if given)
    skips = set(always_skip or [])

    # Check each candidate step in config
    for step in steps:
        # If the config marks this step to 'skip'
        if cfg.get(step, 'run') == 'skip':
            # Special handling for OneOverFStep variants
            if step.startswith('OneOverFStep'):
                step = 'OneOverFStep'
            skips.add(step)

    # Return as a list (order not guaranteed since set used)
    return list(skips)




# ----------------------------------------
# main
# ----------------------------------------

def main():
    # ===== SETUP =====
    parser = argparse.ArgumentParser(description="exoTEDRF Optimizer")
    parser.add_argument("--config", default="run_optimize.yaml", help="Config YAML")
    args = parser.parse_args()

    cfg = parse_config(args.config)
    obs = (cfg.get('observing_mode') or '').lower()
    filter_det = (cfg.get('filter_detector') or '').lower()
    instrument = obs.split('/')[0].upper() if '/' in obs else obs.upper()

    # Key parameters
    baseline_ints = cfg.get('baseline_ints', [100, -100])
    wave_range = cfg.get('wave_range', None)
    name_str = cfg.get('name_tag', 'default_run')
    wave_range_plot = cfg.get('wave_range_plot', None)
    ylim_plot = cfg.get('ylim_plot', None)
    w1 = cfg.get('w1', 0.0)
    w2 = cfg.get('w2', 1.0)
    debug_mode = cfg.get('debug_mode', False)

    if debug_mode:
        fancyprint("DEBUG MODE ENABLED: Will use cached results (force_redo=False) for all stages", msg_type='WARNING')

    # Set default wave ranges by instrument
    if wave_range is None:
        if 'nrs1' in filter_det:
            wave_range = (2.9, 5.0)
        elif 'niriss' in obs:
            wave_range = (0.6, 2.8)
        elif 'miri' in obs:
            wave_range = (5, 12)
        else:
            wave_range = None

    if wave_range_plot is None:
        wave_range_plot = wave_range

    t0_total = time.perf_counter()

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

    for k, v in cfg.items():
        if k.startswith("optimize_"):
            param_name = k[len("optimize_"):]

            # Special handling for extract_width - optimize in Phase 2 using custom cost function
            if param_name == 'extract_width':
                if v:
                    vals = cfg[param_name]
                    if not isinstance(vals, list):
                        raise ValueError(f"{param_name} must be list when optimize_{param_name}=True")
                    fancyprint(f"Will optimize: {param_name} in Phase 2 over {vals} (using spectral scatter cost)")
                else:
                    fixed_params[param_name] = cfg[param_name]
                continue  # Don't add to param_ranges for Phase 1

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

    # Initialize with mean values (?)
    current_best = {k: int(np.mean(v)) for k, v in param_ranges.items()}
    current_best.update(fixed_params)

    logf = open(f"{outdir_f}/Cost_{name_str}.txt", "w")
    logs = open(f"{outdir_f}/Scatter_{name_str}.txt", "w")
    logf.write("\t".join(param_ranges.keys()) + "\tduration_s\tcost\n")

    # ===== OPTIMIZATION CHECKPOINTS =====
    # Define all possible optimization checkpoints
    # These will be filtered based on which parameters are actually being optimized

    all_checkpoints = [
        # Stage 1 checkpoints
        {
            'name': 'OneOverFStep_grp',
            'stage': 1,
            'params': ['soss_inner_mask_width', 'soss_outer_mask_width', 'nirspec_mask_width'],
            'skip_before': ['DQInitStep', 'EmiCorrStep', 'SaturationStep', 'ResetStep',
                           'SuperBiasStep', 'RefPixStep', 'DarkCurrentStep'],
            'skip_after': ['LinearityStep', 'JumpStep', 'RampFitStep', 'GainScaleStep'],
        },
        {
            'name': 'JumpStep',
            'stage': 1,
            'params': ['time_jump_threshold', 'time_window'],
            'skip_before': ['DQInitStep', 'EmiCorrStep', 'SaturationStep', 'ResetStep',
                           'SuperBiasStep', 'RefPixStep', 'DarkCurrentStep',
                           'OneOverFStep_grp', 'LinearityStep'],
            'skip_after': ['RampFitStep', 'GainScaleStep'],
        },
        # Stage 2 checkpoints
        {
            'name': 'BackgroundStep',
            'stage': 2,
            'params': ['miri_trace_width', 'miri_background_width'],
            'skip_before': ['AssignWCSStep', 'Extract2DStep', 'SourceTypeStep',
                           'WaveCorrStep', 'FlatFieldStep'],
            'skip_after': ['OneOverFStep_int', 'BadPixStep', 'PCAReconstructStep', 'TracingStep'],
        },
        {
            'name': 'BadPixStep',
            'stage': 2,
            'params': ['space_outlier_threshold', 'time_outlier_threshold', 'box_size', 'window_size'],
            'skip_before': ['AssignWCSStep', 'Extract2DStep', 'SourceTypeStep',
                           'WaveCorrStep', 'FlatFieldStep', 'BackgroundStep', 'OneOverFStep_int'],
            'skip_after': ['PCAReconstructStep', 'TracingStep'],
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
                if not debug_mode:
                    step_output_pattern = None
                    if checkpoint['name'] == 'OneOverFStep_grp':
                        step_output_pattern = f"{outdir_s1}*_oneoverfstep.fits"
                    elif checkpoint['name'] == 'JumpStep':
                        step_output_pattern = f"{outdir_s1}*_jump.fits"
                    elif checkpoint['name'] == 'BackgroundStep':
                        step_output_pattern = f"{outdir_s2}*_backgroundstep.fits"
                    elif checkpoint['name'] == 'BadPixStep':
                        step_output_pattern = f"{outdir_s2}*_badpixstep.fits"

                    if step_output_pattern:
                        files_to_delete = glob.glob(step_output_pattern)
                        if files_to_delete:
                            fancyprint(f"Deleting {len(files_to_delete)} cached file(s) for {checkpoint['name']}:")
                            for cached_file in files_to_delete:
                                fancyprint(f"  Deleting: {cached_file}")
                                os.remove(cached_file)
                        else:
                            fancyprint(f"WARNING: No cached files found matching: {step_output_pattern}", msg_type='WARNING')

                # run pipeline up to (including this step)
                if checkpoint['stage'] == 1:
                    # Build skip list: skip everything after this step
                    skip_list = checkpoint['skip_after'].copy()

                    # Pass time_window parameter to JumpStep if optimizing it
                    s1_kwargs = run_cfg.get('stage1_kwargs', {}).copy()
                    if checkpoint['name'] == 'JumpStep' and param_name == 'time_window':
                        if 'JumpStep' not in s1_kwargs:
                            s1_kwargs['JumpStep'] = {}
                        s1_kwargs['JumpStep']['time_window'] = param_value

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
                        force_redo=False if not debug_mode else False,
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
                        **s1_kwargs
                    )

                    # Extract from Stage 1 output
                    datafile = stage1_results[0]

                elif checkpoint['stage'] == 2:
                    # First, need Stage 1 results (use cached)
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
                        do_plot=run_cfg.get('do_plots', False),
                        soss_inner_mask_width=run_cfg.get('soss_inner_mask_width'),
                        soss_outer_mask_width=run_cfg.get('soss_outer_mask_width'),
                        nirspec_mask_width=run_cfg.get('nirspec_mask_width'),
                        centroids=run_cfg.get('centroids'),
                        hot_pixel_map=run_cfg.get('hot_pixel_map'),
                        miri_drop_groups=run_cfg.get('miri_drop_groups'),
                        **run_cfg.get('stage1_kwargs', {})
                    )

                    # Build skip list for Stage 2
                    skip_list = checkpoint['skip_after'].copy()

                    # Pass step-specific parameters if optimizing them
                    s2_kwargs = run_cfg.get('stage2_kwargs', {}).copy()
                    if checkpoint['name'] == 'BadPixStep':
                        if 'BadPixStep' not in s2_kwargs:
                            s2_kwargs['BadPixStep'] = {}
                        if param_name == 'box_size':
                            s2_kwargs['BadPixStep']['box_size'] = param_value
                        if param_name == 'window_size':
                            s2_kwargs['BadPixStep']['window_size'] = param_value

                    # Run Stage 2 with force_redo=False
                    # The deleted cached file will trigger rerun from that step onward
                    stage2_results, stage2_centroids = run_stage2(
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
                        smoothing_scale=run_cfg.get('smoothing_scale'),
                        skip_steps=skip_list,
                        generate_lc=run_cfg.get('generate_lc'),
                        soss_inner_mask_width=run_cfg.get('soss_inner_mask_width'),
                        soss_outer_mask_width=run_cfg.get('soss_outer_mask_width'),
                        nirspec_mask_width=run_cfg.get('nirspec_mask_width'),
                        pixel_masks=run_cfg.get('outlier_maps'),
                        generate_order0_mask=run_cfg.get('generate_order0_mask'),
                        f277w=run_cfg.get('f277w'),
                        do_plot=run_cfg.get('do_plots', False),
                        centroids=run_cfg.get('centroids'),
                        miri_trace_width=run_cfg.get('miri_trace_width'),
                        miri_background_width=run_cfg.get('miri_background_width'),
                        miri_background_method=run_cfg.get('miri_background_method'),
                        **s2_kwargs
                    )

                    if isinstance(stage2_centroids, np.ndarray):
                        stage2_centroids = pd.DataFrame(stage2_centroids.T, columns=["xpos", "ypos"])

                    datafile = stage2_results[0]

                elif checkpoint['stage'] == 3:
                    # Need Stage 1 and 2 completed first (use cached)
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
                        do_plot=run_cfg.get('do_plots', False),
                        soss_inner_mask_width=run_cfg.get('soss_inner_mask_width'),
                        soss_outer_mask_width=run_cfg.get('soss_outer_mask_width'),
                        nirspec_mask_width=run_cfg.get('nirspec_mask_width'),
                        centroids=run_cfg.get('centroids'),
                        hot_pixel_map=run_cfg.get('hot_pixel_map'),
                        miri_drop_groups=run_cfg.get('miri_drop_groups'),
                        **run_cfg.get('stage1_kwargs', {})
                    )

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
                        smoothing_scale=run_cfg.get('smoothing_scale'),
                        generate_lc=run_cfg.get('generate_lc'),
                        soss_inner_mask_width=run_cfg.get('soss_inner_mask_width'),
                        soss_outer_mask_width=run_cfg.get('soss_outer_mask_width'),
                        nirspec_mask_width=run_cfg.get('nirspec_mask_width'),
                        pixel_masks=run_cfg.get('outlier_maps'),
                        generate_order0_mask=run_cfg.get('generate_order0_mask'),
                        f277w=run_cfg.get('f277w'),
                        do_plot=run_cfg.get('do_plots', False),
                        centroids=run_cfg.get('centroids'),
                        miri_trace_width=run_cfg.get('miri_trace_width'),
                        miri_background_width=run_cfg.get('miri_background_width'),
                        miri_background_method=run_cfg.get('miri_background_method'),
                        **run_cfg.get('stage2_kwargs', {})
                    )

                    datafile = stage2_results[0]

                # Extract and compute cost <- new function
                # For Phase 1, use a fixed extract_width (will be optimized in Phase 2)
                phase1_extract_width = cfg.get('extract_width')
                if isinstance(phase1_extract_width, list):
                    # If it's a list (optimize_extract_width=True), use middle value for Phase 1
                    phase1_extract_width = phase1_extract_width[len(phase1_extract_width) // 2]
                    fancyprint(f"  Using extract_width={phase1_extract_width} for Phase 1 (will optimize in Phase 2)")

                spectral_dict, centroids = extract_at_step(
                    datafile=datafile,
                    instrument=instrument,
                    extract_width=phase1_extract_width,
                    centroids=centroids,  # Reuse cached
                    baseline_ints=baseline_ints,
                    output_dir=outdir_s2
                )

                # Compute cost
                # For Phase 1, skip wave_range filtering (use all channels for relative comparison)
                # Accurate wavelengths not available from Stage 1 files
                phase1_wave_range = None

                cost, scatter = cost_function(
                    spectral_dict,
                    baseline_ints=baseline_ints,
                    wave_range=phase1_wave_range,
                    w1=w1,
                    w2=w2
                )

                # Debug cost details
                fancyprint(f"  Cost function: w1={w1}, w2={w2}, wave_range={wave_range}")
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

            # Find best value for this parameter
            best_idx = np.argmin(costs)
            best_value = param_values[best_idx]
            best_cost = costs[best_idx]

            current_best[param_name] = best_value
            fancyprint(f"\n*** Best {param_name}={best_value} with cost={best_cost:.6f} ***\n")

    logf.close()
    logs.close()

 
    fancyprint("\n=== Plotting optimization results ===")
    plot_cost(name_str)

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
    stage1_steps = ['DQInitStep', 'EmiCorrStep', 'SaturationStep', 'ResetStep', 'SuperBiasStep',
                    'RefPixStep', 'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep', 'JumpStep',
                    'RampFitStep', 'GainScaleStep']
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
        **final_cfg.get('stage1_kwargs', {})
    )

    # Build skip list for Stage 2
    stage2_steps = ['AssignWCSStep', 'Extract2DStep', 'SourceTypeStep', 'WaveCorrStep',
                    'FlatFieldStep', 'OneOverFStep_int', 'BackgroundStep', 'TracingStep',
                    'BadPixStep', 'PCAReconstructStep']
    stage2_skip = []
    for step in stage2_steps:
        if final_cfg.get(step) == 'skip':
            if step == 'OneOverFStep_int':
                stage2_skip.append('OneOverFStep')
            else:
                stage2_skip.append(step)

    fancyprint(f"Stage 2 steps to skip: {stage2_skip}")

    # Stage 2
    stage2_results, final_centroids = run_stage2(
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
        smoothing_scale=final_cfg.get('smoothing_scale'),
        generate_lc=final_cfg.get('generate_lc'),
        soss_inner_mask_width=final_cfg.get('soss_inner_mask_width'),
        soss_outer_mask_width=final_cfg.get('soss_outer_mask_width'),
        nirspec_mask_width=final_cfg.get('nirspec_mask_width'),
        pixel_masks=final_cfg.get('outlier_maps'),
        generate_order0_mask=final_cfg.get('generate_order0_mask'),
        f277w=final_cfg.get('f277w'),
        do_plot=final_cfg.get('do_plots', False),
        centroids=final_cfg.get('centroids'),
        miri_trace_width=final_cfg.get('miri_trace_width'),
        miri_background_width=final_cfg.get('miri_background_width'),
        miri_background_method=final_cfg.get('miri_background_method'),
        **final_cfg.get('stage2_kwargs', {})
    )

    if isinstance(final_centroids, np.ndarray):
        final_centroids = pd.DataFrame(final_centroids.T, columns=["xpos", "ypos"])

    this_centroid = final_cfg.get('centroids') if final_cfg.get('centroids') is not None else final_centroids

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
                st_teff=final_cfg.get('st_teff'),
                st_logg=final_cfg.get('st_logg'),
                st_met=final_cfg.get('st_met'),
                planet_letter=final_cfg.get('planet_letter'),
                output_tag=final_cfg['output_tag'],
                do_plot=final_cfg.get('do_plots', False),
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

        # Select best extract_width
        best_width_idx = np.argmin(extract_costs)
        best_extract_width = extract_widths[best_width_idx]
        best_extract_cost = extract_costs[best_width_idx]

        fancyprint(f"\n*** Best extract_width={best_extract_width} with cost={best_extract_cost:.6f} ***\n")

        # Update final config and run one more time with best width
        final_cfg['extract_width'] = best_extract_width
        current_best['extract_width'] = best_extract_width

        # Final Stage 3 with optimal width
        stage3_results = run_stage3(
            stage2_results,
            save_results=True,
            force_redo=True,
            extract_method=final_cfg['extract_method'],
            soss_specprofile=final_cfg.get('soss_specprofile'),
            centroids=this_centroid,
            extract_width=best_extract_width,
            st_teff=final_cfg.get('st_teff'),
            st_logg=final_cfg.get('st_logg'),
            st_met=final_cfg.get('st_met'),
            planet_letter=final_cfg.get('planet_letter'),
            output_tag=final_cfg['output_tag'],
            do_plot=final_cfg.get('do_plots', False),
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
            st_teff=final_cfg.get('st_teff'),
            st_logg=final_cfg.get('st_logg'),
            st_met=final_cfg.get('st_met'),
            planet_letter=final_cfg.get('planet_letter'),
            output_tag=final_cfg['output_tag'],
            do_plot=final_cfg.get('do_plots', False),
            **final_cfg.get('stage3_kwargs', {})
        )

    #  diagnostics
    diagnostic_plot(stage3_results, name_str, baseline_ints=baseline_ints, outdir=outdir_f)

    #  scatter plot
    outfile = os.path.join(outdir_f, f"Scatter_{name_str}.txt")
    specfile = glob.glob(os.path.join(outdir_s3, "*_box_spectra_fullres.fits"))[0]
    best_idx = pd.read_csv(os.path.join(outdir_f, f"Cost_{name_str}.txt"), sep="\t")['cost'].idxmin()

    plot_scatter(
        txtfile=outfile,
        rows=[best_idx],
        wave_range=wave_range_plot,
        smooth=10,
        spectrum_files=[specfile],
        ylim=ylim_plot,
        style="line",
        save_path=os.path.join(outdir_f, f"Scatter_Plot_{name_str}.png"),
    )

    #  timing
    t1 = time.perf_counter() - t0_total
    h, m = divmod(int(t1), 3600)
    m, s = divmod(m, 60)
    fancyprint(f"\n{'='*60}")
    fancyprint(f"TOTAL RUNTIME: {h}h {m:02d}min {s:02d}s")
    fancyprint(f"OPTIMAL PARAMETERS: {current_best}")
    fancyprint(f"{'='*60}\n")

if __name__ == "__main__":
    main() 
