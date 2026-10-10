#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 21:32 2026

@author: MCR, PSD, TRF

Optimizer plotting routines.
"""

from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import pandas as pd
from scipy.ndimage import uniform_filter1d

from exotedrf.utils import fancyprint
from exotedrf.optimize_utils import stitch_soss_orders


def make_diagnostic_plot(st3, name_str, baseline_ints, obs, filter, outdir):
    """Create two diagnostic plots from Stage-3 data:
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
            [nn] -> normalize by median of first nn integrations
            [nlow, nhigh] -> normalize by mean of medians of start and end segments
    obs : str
        Observing mode identifier.
    filter : str
        Filter/detector identifier.
    outdir : str
        Output directory for saved figures.
    """

    os.makedirs(outdir, exist_ok=True)

    # ======== WAVELENGTH RANGE SELECTION BASED ON MODE/FILTER ========
    if 'miri' in obs:
        wave_min, wave_max = 5.0, 12.0
    elif 'niriss' in obs:
        wave_min, wave_max = 0.6, 2.8
    elif 'nirspec' in obs:
        if filter == 'nrs1':
            wave_min, wave_max = 2.9, 3.9  # NRS1 covers lower wavelengths (~2.9-3.8 µm)
        elif filter == 'nrs2':
            wave_min, wave_max = 3.8, 5.0  # NRS2 covers higher wavelengths (~3.8-5.2 µm)
        else:
            raise ValueError(f"Unknown nirspec filter_detector: {filter}")
    else:
        raise ValueError(f"Unknown observing_mode: {obs}")

    # --- Build stitched spectrum ---
    if 'niriss' in obs:
        # Stitch both spectral orders onto one wavelength axis
        wave, flux = stitch_soss_orders(st3['Wave O1'], st3['Wave O2'],
                                        st3['Flux O1'], st3['Flux O2'])
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
        # Normalize by median of first nn integrations
        nn = int(baseline_ints[0])
        norm_white = white / np.median(white[:nn])
    else:
        # Normalize by mean of medians from start and end segments
        nlow, nhigh = map(int, baseline_ints)
        base = 0.5 * (np.median(white[:nlow]) + np.median(white[nhigh:]))
        norm_white = white / base

    # --- Plot normalized white-light curve ---
    plt.figure()
    plt.plot(norm_white, 'k.', markersize=2, alpha=0.5)
    plt.xlabel("Integration Number")
    plt.ylabel("Normalized White Flux")
    plt.title("Normalized White Light Curve")
    plt.savefig(f"{outdir}/WhiteLightCurve{name_str}.png", dpi=300)
    plt.close()

    # --- Normalized flux image with true wavelength mapping ---
    # Normalize each column by its time median (post-cleaning)
    img = np.full_like(flux, np.nan, dtype=float)
    img[:, :] = flux / col_med[good][keep_idx]  # safe: filtered for finite non-zero values

    n_int, n_pix = img.shape

    # Check if wavelength array is empty (can happen with bad extractions)
    if wave.size == 0 or n_pix == 0:
        fancyprint("WARNING: Wavelength array is empty, skipping diagnostic flux "
                   "image plot", msg_type='WARNING')
        return

    # Require strictly increasing wavelength for pcolormesh bin edges
    if not np.all(np.diff(wave) > 0):
        fancyprint("WARNING: Wavelength not strictly increasing, skipping diagnostic flux "
                   "image plot", msg_type='WARNING')
        return

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
    plt.savefig(f"{outdir}/2D_LightCurves{name_str}.png", dpi=300)
    plt.close()


def plot_cost(name_str, outdir, table_height=0.4):
    """Reads a tab-delimited cost file, detects parameter sweeps, highlights
    the best parameter set(s), and produces a figure showing cost trends.

    Parameters
    ----------
    name_str : str
        Identifier used to find the cost file (Cost_<name_str>.txt).
    table_height : float
        Fraction of the figure height to allocate to the table display.
    outdir : str
        Direcory to which to save outputs.
    """

    df = pd.read_csv(f"{outdir}/Cost_Summary{name_str}.txt",
                     delimiter="\t", keep_default_na=False)

    # Remove rows where 'cost' is not numeric, then keep the surviving values
    # numeric so per-sweep normalization does arithmetic instead of string math.
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df = df[df["cost"].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No finite numeric costs found in {outdir}/Cost_Summar{name_str}.txt")

    # Get all parameter columns (exclude 'duration_s' and 'cost' at the end)
    param_cols = df.columns[:-2]

    # detect which parameter changed per row
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

    # Find sweep boundaries: as soon as any other parameter changes, the next sweep starts
    sweep_lines = []
    for i in range(1, len(df)):
        diffs = [c for c in param_cols if df.at[i, c] != df.at[i - 1, c]]
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

    # First row label belongs to the first detected sweep
    if len(df) >= 2 and changed_param_per_row[0] is None:
        changed_param_per_row[0] = changed_param_per_row[1] or param_cols[0]

    # Labels and sweep boundaries
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

    # Normalize cost and highlight best
    sweep_boundaries = [0] + sweep_lines + [len(df)]
    colors = ['gray'] * len(df)  # default colour
    normalized_costs = np.zeros(len(df))

    for i in range(len(sweep_boundaries) - 1):
        start = sweep_boundaries[i]
        end = sweep_boundaries[i + 1]

        # Get costs for this sweep
        sweep_costs = df.iloc[start:end]["cost"].values

        # normalize to [0, 1] within this sweep
        min_cost = np.nanmin(sweep_costs)
        max_cost = np.nanmax(sweep_costs)
        if max_cost > min_cost:
            # scale to [0, 1]: 0 = best (lowest cost), 1 = worst (highest cost)
            normalized_sweep = (sweep_costs - min_cost) / (max_cost - min_cost)
        else:
            # all costs are the same in this sweep
            normalized_sweep = np.zeros(len(sweep_costs))

        normalized_costs[start:end] = normalized_sweep

        # highlight the best (minimum cost) in this sweep
        min_idx = df.iloc[start:end]["cost"].idxmin()
        colors[min_idx] = 'royalblue'

    best_row = df.loc[df["cost"].idxmin(), param_cols.tolist() + ["cost"]].copy()
    for col in best_row.index:
        val = best_row[col]
        try:
            fv = float(val)
            best_row[col] = int(fv) if fv.is_integer() else fv
        except Exception:
            best_row[col] = val
    best_df = pd.DataFrame([best_row]).reset_index(drop=True)

    fig = plt.figure(figsize=(max(14, len(df) * 0.25), 10))
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[1 - table_height, table_height])
    ax_plot = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    ax_plot.scatter(range(len(df)), normalized_costs, color=colors)
    for x in sweep_lines:
        ax_plot.axvline(x=x - 0.5, color='gray', linestyle='--', linewidth=1)

    values = [lbl.split('=', 1)[1] for lbl in df["changed_label"]]
    ax_plot.set_xticks(range(len(df)))
    ax_plot.set_xticklabels(values, rotation=0, fontsize=8)

    ymin, ymax = ax_plot.get_ylim()
    base_y = ymin - 0.08 * (ymax - ymin)
    alt_y = ymin - 0.15 * (ymax - ymin)
    for i, (start, end) in enumerate(zip(sweep_boundaries[:-1], sweep_boundaries[1:])):
        param_name = df.loc[start, "changed_label"].split("=", 1)[0]
        center = (start + end - 1) / 2
        y_pos = base_y if i % 2 == 0 else alt_y
        ax_plot.text(center, y_pos, param_name, ha="center", va="top", fontsize=10)

    fig.subplots_adjust(bottom=0.30)
    ax_plot.set_ylabel("Relative Cost (normalized per sweep)")
    ax_plot.set_title(f"Cost by Single Parameter Sweep: {name_str}")
    ax_plot.set_ylim(-0.05, 1.05)

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
            cell.set_fontsize(7)  # header
        else:
            cell.set_fontsize(10)  # data

    fig.savefig(f"{outdir}/Cost_Summary{name_str}.png",
                dpi=300, bbox_inches='tight')


# TODO: Figure out why this isn't plotting order 2.
def plot_scatter(txtfile, rows, wave_range=None, smooth=None, spectrum_files=None, style='line',
                 ylim=None, save_path=None, tol=0.05):
    """Plot point-to-point (P2P) scatter vs wavelength for selected rows from a scatter table.

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

        # Special handling for NIRISS with two orders - plot separately
        if ("Wave_O1" in name_map) and ("Wave_O2" in name_map):
            # Scatter rows are on the stitched, wavelength-sorted axis used by cost_function.
            wave_o1 = np.asarray(name_map["Wave_O1"].data, float)
            wave_o2 = np.asarray(name_map["Wave_O2"].data, float)
            if wave_o1.ndim == 2:
                wave_o1 = np.nanmedian(wave_o1, axis=0)
                wave_o2 = np.nanmedian(wave_o2, axis=0)
            wave_stitched = stitch_soss_orders(wave_o1, wave_o2)[0]
            is_niriss_two_orders = True
            # Plot each order's segment of the stitched axis separately
            orders = [
                {'sel': wave_stitched <= 0.85, 'name': 'Order 2'},
                {'sel': wave_stitched > 0.85, 'name': 'Order 1'}
            ]
        else:
            # Fallback: read first extension array as wavelength grid
            wave_full = np.asarray(hdus[1].data, float)
            is_niriss_two_orders = False

    # --- Plot for NIRISS with two orders (side-by-side subplots) ---
    if is_niriss_two_orders:
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))

        if wave_stitched.size != n_cols:
            raise ValueError(f"Stitched SOSS wavelength axis ({wave_stitched.size}) does not match "
                             f"scatter columns ({n_cols}).")
        for idx, order_info in enumerate(orders):
            ax = axes[idx]
            order_name = order_info['name']

            # Build mask for this order's segment and the wavelength range
            mask = order_info['sel'] & np.isfinite(wave_stitched)
            if wave_range is not None:
                wmin, wmax = wave_range
                mask &= (wave_stitched >= wmin - tol) & (wave_stitched <= wmax + tol)

            if not mask.any():
                fancyprint(f"Warning: No finite wavelengths in {order_name} within range"
                           f" {wave_range}", msg_type='WARNING')
                continue

            x = wave_stitched[mask]

            # Plot each valid row
            for i in valid:
                # Extract data for this order from scatter table
                y_full = df.iloc[i, :].to_numpy(float)
                y_raw = (y_full[mask]) * 1e6

                if style == 'line':
                    ax.plot(x, y_raw, linewidth=0.6, linestyle='-', alpha=0.5,
                            color='grey', label="Best config (raw)" if idx == 0 else "")
                else:
                    ax.scatter(x, y_raw, s=3, alpha=0.8,
                               label="Best config (raw)" if idx == 0 else "")

                # Apply smoothing if requested
                if smooth and smooth > 1:
                    y_sm = uniform_filter1d(y_raw, size=smooth, mode='nearest')
                    if style == 'line':
                        ax.plot(x, y_sm, linewidth=1.2, linestyle='-',
                                label=f"Best config (smoothed, window={smooth})" if idx == 0 else "")
                    else:
                        ax.scatter(x, y_sm, s=5,
                                   label=f"Best config (smoothed, window={smooth})" if idx == 0 else "")

            ax.set_xlabel("Wavelength [μm]", fontsize=11)
            ax.set_ylabel("Scatter [ppm]", fontsize=11)
            ax.set_title(f"{order_name}", fontsize=12)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    # --- Plot for single-order instruments ---
    # Sort wavelengths
    s = np.argsort(wave_full, kind="mergesort")
    wave_sorted = wave_full[s]

    # Check size match
    if wave_sorted.size != n_cols:
        min_size = min(wave_sorted.size, n_cols)
        fancyprint(
            f"WARNING: Wavelength array size ({wave_sorted.size}) != scatter columns ({n_cols}).\n"
            f"  Truncating to {min_size} elements.",
            msg_type='WARNING'
        )
        wave_sorted = wave_sorted[:min_size]
        s = s[:min_size]

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
    plt.xlim(np.min(x), np.max(x))
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
