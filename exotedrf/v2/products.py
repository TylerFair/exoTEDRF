"""Save spectra, logs, summaries, and diagnostic figures from a v2 run."""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from itertools import groupby

import numpy as np


def v1_output_root(configured, output_tag='', *, cwd=None):
    """Reproduce v1's exact output-directory and output-tag naming rule.

    Parameters
    ----------
    configured : str, Path
        Configured pipeline output directory.
    output_tag : str
        Suffix appended to the output directory.
    cwd : None, str, Path
        Directory used to resolve relative output paths.

    Returns
    -------
    root : Path
        Tagged pipeline output directory.
    """
    base = os.path.expanduser(os.fspath(configured))
    if cwd is not None and not os.path.isabs(base):
        base = os.path.join(os.fspath(cwd), base)
    tag = str(output_tag or '')
    return Path(base + (f'_{tag}' if tag else ''))


def output_layout(cfg, output_dir=None, create=True):
    """Choose the Stage1/Stage2/Stage3, log, plot, and final-product paths.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration.
    output_dir : None, str, Path
        Directory to which to save products.
    create : bool
        If True, create the output directories.

    Returns
    -------
    paths : dict
        Directories and filenames for reduction products.
    """
    if output_dir is None:
        # Apply the output tag before selecting the v2 directory.
        root = v1_output_root(cfg.get('pipeline_outputs_directory', 'pipeline_outputs_directory'),
            cfg.get('output_tag', '')) / 'v2'
    else:
        root = Path(os.path.expanduser(os.fspath(output_dir)))
    files = root / 'Files'
    stage1 = root / 'Stage1'
    stage2 = root / 'Stage2'
    stage3 = root / 'Stage3'
    if create:
        for directory in (root, files, stage1, stage2, stage3):
            directory.mkdir(parents=True, exist_ok=True)
    name = str(cfg.get('name_tag', 'default_run'))
    stem = name or 'soss'
    extract_method = str(cfg.get('extract_method', 'box')).lower()
    return {'root': root, 'files': files, 'stage1': stage1, 'stage2': stage2, 'stage3': stage3,
        'cost': files / f'Cost_{name}.txt', 'scatter': files / f'Scatter_{name}.txt',
        'decision_ranking': files / f'Decision_ranking_{name}.txt',
        'summary': files / f'Optimization_{name}.json', 'cost_plot': files / f'Cost_{name}.png',
        'scatter_plot': files / f'Scatter_Plot_{name}.png',
        'flux_plot': files / f'flux_img_{name}.png', 'white_plot': files / f'norm_white_{name}.png',
        'centroid_plot': stage3 / 'centroiding.png', 'rate': stage2 / f'{stem}_rateints_v2.fits',
        'lcestimate': stage2 / f'{stem}_lcestimate.npy',
        # Use the configured name when target metadata is unavailable.
        'spectra': stage3 / f'{stem}_{extract_method}_spectra_fullres.fits'}


def _v1_spectrum_path(path, meta, extract_method):
    """Return v1's ``<target>[_<detector>]_<method>_spectra_fullres.fits``."""
    path = Path(path)
    header = (getattr(meta, 'extra', {}) or {}).get('header', {})
    target = next((header.get(key) for key in ('TARGNAME', 'TARGET', 'OBJECT')
        if header.get(key) not in (None, '')), None)
    if target is None:
        return path
    instrument = str(getattr(meta, 'mode', '')).split('/')[0].upper()
    detector = ''
    if instrument == 'NIRSPEC':
        detector = f'{str(getattr(meta, "detector", "")).lower()}_'
    return path.with_name(f'{str(target)}_{detector}{str(extract_method).lower()}'
        '_spectra_fullres.fits')


def _v1_fileroot_noseg(meta, fallback='soss'):
    """Derive v1's reusable-product prefix from the first input filename."""
    filenames = tuple(getattr(meta, 'filenames', ()) or ())
    header = (getattr(meta, 'extra', {}) or {}).get('header', {})
    source = header.get('FILENAME')
    if source in (None, '') and filenames:
        source = filenames[0]
    if source in (None, ''):
        return f'{fallback}_'
    name = Path(os.fspath(source)).name
    chunks = name.split('_')
    root = '_'.join(chunks[:-1]) + '_' if len(chunks) > 1 else Path(name).stem + '_'
    # Remove the segment identifier as in utils.get_filename_root_noseg.
    return re.sub(r'[-_]seg\d+', '', root, count=1, flags=re.IGNORECASE)


def _v1_oofscaling_root(meta, fallback='soss'):
    """Derive the solve sidecar prefix from the first segment filename."""
    filenames = tuple(getattr(meta, 'filenames', ()) or ())
    header = (getattr(meta, 'extra', {}) or {}).get('header', {})
    source = filenames[0] if filenames else header.get('FILENAME')
    if source not in (None, ''):
        chunks = Path(os.fspath(source)).name.split('_')
        if len(chunks) > 1:
            root = '_'.join(chunks[:-1]) + '_'
            if re.fullmatch(r'.+[-_]seg\d{3}_nis_', root):
                return root[:-12] + '_nis_'
    return _v1_fileroot_noseg(meta, fallback=fallback)


def _write_oneoverf_solve_sidecars(state, paths):
    """Save even- and odd-row solve scalings for each stage and spectral order."""
    from exotedrf.v2 import stages as v2stages
    written = {}
    for level, directory in (('grp', 'stage1'), ('int', 'stage2')):
        record = state.aux.get(v2stages.OOF_SOLVE_AUX_KEYS[level])
        if not record:
            continue
        root = _v1_oofscaling_root(state.cube.meta)
        for order_key in sorted(record):
            order = int(order_key.lstrip('o'))
            for parity, name in (('even', 'scale_e'), ('odd', 'scale_o')):
                path = Path(paths[directory]) / (f'{root}oofscaling_{parity}_order{order}.npy')
                np.save(path, np.asarray(record[order_key][name], dtype=np.float64),
                    allow_pickle=False)
                written[f'oofscaling_{level}_{parity}_order{order}'] = str(path)
    return written


def _write_reusable_sidecars(state, ctx, paths):
    """Write the small v1 products used to reinterpret a reduction later."""
    from exotedrf.v2 import trace
    paths = {key: Path(value) for key, value in paths.items()}
    cube = state.cube
    prefix = _v1_fileroot_noseg(cube.meta)
    written = {}
    deepframe = state.aux.get('stage3_deepframe')
    if deepframe is None:
        deepframe = state.aux.get('deepframe')
    centroids = _final_centroids(state, ctx, deepframe)
    if centroids:
        centroid_path = paths['stage3'] / f'{prefix}centroids.csv'
        trace.save_centroids_csv(centroid_path, centroids)
        written['centroids'] = str(centroid_path)

    # Save the baseline deep stack for Stage 3 reuse.
    if deepframe is not None:
        from astropy.io import fits
        deepframe = np.asarray(deepframe)
        if deepframe.ndim != 2:
            raise ValueError(f'deepframe must be 2D, got shape {deepframe.shape}')
        deepframe_path = paths['stage2'] / f'{prefix}deepframe.fits'
        fits.PrimaryHDU(deepframe).writeto(deepframe_path, overwrite=True)
        written['deepframe'] = str(deepframe_path)

    for key, directory, filename, value, dtype in (
        ('hot_pixels', 'stage2', f'{prefix}hot_pixels.npy',
        state.aux.get('hot_pixel_map'), None),
        ('contaminant_mask', 'stage1', 'contaminant_mask.npy',
        ctx.get('order0_mask'), bool)):
        if value is not None:
            path = paths[directory] / filename
            np.save(path, np.asarray(value, dtype=dtype), allow_pickle=False)
            written[key] = str(path)
    written.update(_write_oneoverf_solve_sidecars(state, paths))
    background = state.aux.get('bkg_int')
    if background is not None:
        path = paths['stage2'] / f'{prefix}background.npy'
        np.save(path, np.asarray(background), allow_pickle=False)
        written['background'] = str(path)

    # Save the refitted components when component removal is enabled.
    removed = ctx.get('opts', {}).get('remove_components')
    components = (state.aux.get('pca_components_reconstructed') if removed is not None else
        state.aux.get('pca_components'))
    if components is not None:
        components = np.asarray(components)
        if components.ndim != 2:
            raise ValueError(f'pca_components must be 2D, got shape {components.shape}')
        instrument = str(getattr(cube.meta, 'mode', '')).split('/')[0].upper()
        detector_suffix = (f'_{str(getattr(cube.meta, "detector", "")).lower()}'
            if instrument == 'NIRSPEC' else '')
        stability_path = paths['stage2'] / (f'{prefix}stability{detector_suffix}.csv')
        with stability_path.open('w', encoding='utf-8', newline='') as stream:
            stream.write(','.join(f'Component {index + 1}'
                for index in range(components.shape[0])) + '\n')
            for row in components.T:
                stream.write(','.join(f'{float(value):.18g}' for value in row) + '\n')
        written['stability'] = str(stability_path)
    return written


def format_log_value(value):
    """Format scalar and list values for the v1 Cost table.

    Parameters
    ----------
    value : object
        Value to format.

    Returns
    -------
    formatted : str
        Value formatted as in the v1 Cost table.
    """
    from exotedrf.v2 import config as v2config
    if isinstance(value, v2config.ExtractAperture):
        # Preserve spaces in asymmetric aperture values.
        return str(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(str(item) for item in value) + ']'
    if value is None:
        return 'None'
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _trial_value(trial, name, default=None):
    """Read a trial field from a mapping or an object."""
    if isinstance(trial, dict):
        return trial.get(name, default)
    return getattr(trial, name, default)


def write_decision_ranking(paths, ranking):
    """Write the decision-sensitivity table as a TSV.

    Parameters
    ----------
    paths : dict
        Output paths returned by output_layout.
    ranking : list[dict]
        Decision-sensitivity rows ordered by mean relative cost change.

    Returns
    -------
    path : Path
        Path to the decision-sensitivity table.
    """
    paths = {key: Path(value) for key, value in paths.items()}
    paths['files'].mkdir(parents=True, exist_ok=True)
    columns = ['rank', 'parameter', 'n_pairs', 'mean_rel', 'max_rel',
        'median_rel', 'mean_rel_scatter', 'prune_candidate', 'winning_values']
    with paths['decision_ranking'].open('w', encoding='utf-8', newline='') as stream:
        stream.write('\t'.join(columns) + '\n')
        for row in ranking:
            values = [str(row.get('rank', '')), str(row.get('parameter', '')),
                str(row.get('n_pairs', '')), format_log_value(row.get('mean_rel')),
                format_log_value(row.get('max_rel')), format_log_value(row.get('median_rel')),
                format_log_value(row.get('mean_rel_scatter')), str(row.get('prune_candidate', '')),
                json.dumps(jsonable(row.get('winning_values', {})), sort_keys=True)]
            stream.write('\t'.join(values) + '\n')
    return paths['decision_ranking']


def write_optimizer_logs(paths, trials, parameter_columns, summary):
    """Write v1-style Cost/Scatter logs and the structured JSON summary.

    Parameters
    ----------
    paths : dict
        Output paths returned by output_layout.
    trials : list
        Optimizer trial records.
    parameter_columns : list[str]
        Parameter names in Cost table order.
    summary : dict
        Optimizer summary to save as JSON.

    Returns
    -------
    paths : dict
        Output paths used to write the logs and summary.
    """
    paths = {key: Path(value) for key, value in paths.items()}
    paths['files'].mkdir(parents=True, exist_ok=True)

    if summary.get('output_mode') == 'optimal':
        with paths['summary'].open('w', encoding='utf-8') as stream:
            json.dump(jsonable(summary), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        return paths

    # Log the phase-1 seed width as in v1.
    phase1_width = (summary.get('initial_params') or {}).get('extract_width')
    with paths['cost'].open('w', encoding='utf-8', newline='') as stream:
        stream.write('\t'.join([*parameter_columns, 'duration_s', 'cost']) + '\n')
        for trial in trials:
            params = _trial_value(trial, 'params', {})
            if _trial_value(trial, 'phase') == 1 and 'extract_width' in parameter_columns and \
                    phase1_width is not None:
                params = dict(params, extract_width=phase1_width)
            values = [format_log_value(params.get(name, '')) for name in parameter_columns]
            duration = float(_trial_value(trial, 'duration_s', 0.0))
            cost = float(_trial_value(trial, 'cost', np.nan))
            values.extend([f'{duration:.1f}', f'{cost:.12f}'])
            stream.write('\t'.join(values) + '\n')

    with paths['scatter'].open('w', encoding='utf-8', newline='') as stream:
        for trial in trials:
            scatter = np.asarray(_trial_value(trial, 'scatter', []), dtype=float).ravel()
            stream.write(' '.join(f'{value:.10g}' for value in scatter) + '\n')

    with paths['summary'].open('w', encoding='utf-8') as stream:
        json.dump(jsonable(summary), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')

    if summary.get('decision_sensitivity'):
        write_decision_ranking(paths, summary['decision_sensitivity'])
    return paths


def _wavelength_solution_header(spectral_products, ctx):
    """Return header cards for the wavelength shift and PASTASOSS solution."""
    header = {}
    shift = next((product.get('wave_shift') for product in (spectral_products or {}).values()
        if isinstance(product, dict) and product.get('wave_shift') is not None), None)
    if shift is not None:
        header['WAVESHFT'] = (float(shift), 'Stellar-model CCF wavelength shift [micron]')
    solution = (ctx or {}).get('pastasoss_solution')
    if solution is not None:
        header['WAVESOLN'] = ('pastasoss', 'Stage-3 wavelength solution')
        header['PWCPOS'] = (float(solution['pwcpos']), 'Pupil wheel position used by PASTASOSS')
    return header


def write_final_products(state, params, ctx, paths, *, output_mode=None):
    """Write the final rate cube, reusable sidecars, and extracted spectra.

    Parameters
    ----------
    state : State
        Final calibrated pipeline state.
    params : dict
        Selected calibration and extraction parameters.
    ctx : dict
        Pipeline context and options.
    paths : dict
        Output paths returned by output_layout.
    output_mode : None, str
        Product mode; either "standard" or "optimal".

    Returns
    -------
    written : dict
        Names and paths of the products written.
    """
    from exotedrf.v2 import io
    paths = {key: Path(value) for key, value in paths.items()}
    written = {}
    cube = state.cube
    mode = output_mode or ctx.get('opts', {}).get('output_mode', 'standard')
    if mode not in ('standard', 'optimal'):
        raise ValueError('output_mode must be standard or optimal')
    optimal = mode == 'optimal'
    data = cube.data
    if not optimal and data.ndim == 3 and all(hasattr(cube, name)
            for name in ('err', 'dq', 'meta')):
        io.save_rate_cube(cube, paths['rate'], extra_header={'OPTIMIZE': True})
        written['rate'] = str(paths['rate'])


    # Save the baseline-normalized first PCA component.
    if not optimal and ctx.get('opts', {}).get('generate_lc', True):
        pca_wlc = state.aux.get('pca_wlc')
        if pca_wlc is not None:
            pca_wlc = np.asarray(pca_wlc)
            if pca_wlc.ndim != 1:
                raise ValueError(f'pca_wlc must be 1D, got shape {pca_wlc.shape}')
            if pca_wlc.shape[0] != data.shape[0]:
                raise ValueError(f'pca_wlc length {pca_wlc.shape[0]} does not match '
                    f'integration count {data.shape[0]}')
            fallback = paths['lcestimate'].name.removesuffix('lcestimate.npy').rstrip('_') or 'soss'
            prefix = _v1_fileroot_noseg(cube.meta, fallback=fallback)
            lcestimate_path = paths['stage2'] / f'{prefix}lcestimate.npy'
            np.save(lcestimate_path, pca_wlc, allow_pickle=False)
            written['lcestimate'] = str(lcestimate_path)
    spectra = state.aux.get('spectra')
    spectral_products = state.aux.get('spectral_products')
    if (spectral_products or spectra) and hasattr(cube, 'meta'):
        waves = ctx.get('waves') or {}
        instrument = str(getattr(cube.meta, 'mode', '')).split('/')[0].upper()

        def order_label(order):
            """Choose the spectral extension suffix for the instrument."""
            return '' if instrument in ('NIRSPEC', 'MIRI') else f'O{order}'
        orders = {}
        # Use the wavelength axes prepared by the extraction steps.
        for order, value in (spectral_products or spectra).items():
            flux, ferr = ((value['flux'], value.get('ferr')) if spectral_products else value)
            flux = np.asarray(flux)
            ferr = (np.full_like(flux, np.nan, dtype=np.float32) if ferr is None
                else np.asarray(ferr))
            if spectral_products:
                product = {'wave': np.asarray(value['wave']), 'dq': value.get('dq')}
            else:
                wave = waves.get(order)
                if wave is None:
                    wave = np.arange(flux.shape[-1], dtype=float)
                product = {'wave': np.asarray(wave)[:flux.shape[-1]]}
            orders[order_label(order)] = {**product, 'flux': flux, 'ferr': ferr}
        for label, value in orders.items():
            wave = np.asarray(value['wave'])
            flux = np.asarray(value['flux'])
            ferr = np.asarray(value['ferr'])
            if flux.ndim != 2:
                raise ValueError(f'{label} flux must be 2D (nints, nwave), got {flux.shape}')
            if wave.shape != (flux.shape[-1],):
                raise ValueError(f'{label} wavelength shape {wave.shape} does not match '
                    f'flux wavelength axis {flux.shape[-1]}')
            if ferr.shape != flux.shape:
                raise ValueError(f'{label} error shape {ferr.shape} does not match flux '
                    f'shape {flux.shape}')
        extract_method = str(ctx.get('opts', {}).get('extract_method', 'box')).lower()
        width = _v1_width_header(params.get('extract_width', ''), extract_method, state)
        spectra_path = _v1_spectrum_path(paths['spectra'], cube.meta, extract_method)
        extra_header = {'OPTIMIZE': True, 'METHOD': extract_method, 'WIDTH': width}
        extra_header.update(_wavelength_solution_header(spectral_products, ctx))
        io.save_spectra(spectra_path, orders, cube.meta, extra_header=extra_header)
        written['spectra'] = str(spectra_path)
    if state.aux.get('atoca'):
        # Save the APPLESOSS profile and ATOCA spectrum estimate.
        from exotedrf.v2 import atoca
        written.update(atoca.write_atoca_products(state, ctx, paths))
    if not optimal:
        written.update(_write_reusable_sidecars(state, ctx, paths))
    return written


def _new_figure(*, figsize=(6.4, 4.8)):
    """Create an off-screen Matplotlib figure without changing its backend."""
    # Import Matplotlib when a figure is needed.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    return figure


_PLOT_SAVE_CONTEXT = None
_PLOT_SAVE_LOCK = threading.Lock()


def _plot_save_workers():
    """Return the PNG save thread count, defaulting to one."""
    raw = os.environ.get('EXOTEDRF_V2_PLOT_THREADS')
    if raw is None:
        return 1
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        raise ValueError('EXOTEDRF_V2_PLOT_THREADS must be an integer')


@contextlib.contextmanager
def _concurrent_saves():
    """Encode the figures saved inside this block on a small thread pool."""
    global _PLOT_SAVE_CONTEXT
    workers = _plot_save_workers()
    context = None
    with _PLOT_SAVE_LOCK:
        # Reuse an outer save pool for nested blocks.
        if workers >= 2 and _PLOT_SAVE_CONTEXT is None:
            context = {'executor': ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix='exotedrf-png'), 'pending': []}
            _PLOT_SAVE_CONTEXT = context
    if context is None:
        yield
        return
    try:
        yield
    finally:
        with _PLOT_SAVE_LOCK:
            _PLOT_SAVE_CONTEXT = None
        try:
            _flush_saves(context)
        finally:
            context['executor'].shutdown(wait=True)


def _flush_saves(context):
    """Wait for deferred saves and retry failed saves on the calling thread."""
    failures = []
    for render, future in context['pending']:
        try:
            future.result()
        except Exception:                          # pragma: no cover - rare
            # Retry failed background saves on the calling thread.
            try:
                render()
            except Exception as error:             # pragma: no cover - rare
                failures.append(error)
    context['pending'] = []
    if failures:
        raise failures[0]


def _submit_save(render):
    """Save a figure on the active thread pool or the calling thread."""
    context = _PLOT_SAVE_CONTEXT
    if context is None:
        render()
        return
    context['pending'].append((render, context['executor'].submit(render)))


def _save_figure(figure, path, *, dpi=300, **save_options):
    """Save a cropped diagnostic figure and clear its contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def render():
        """Render and clear the diagnostic figure."""
        figure.savefig(path, dpi=dpi, **{'bbox_inches': 'tight', **save_options})
        figure.clear()
    _submit_save(render)


def _representative_image(cube, *, max_integrations=16):
    """Calculate a median diagnostic image from sparsely sampled integrations."""
    data = cube.data
    stride = max(1, int(np.ceil(data.shape[0] / max_integrations)))
    sample = data[::stride]
    if sample.ndim == 4:
        sample = sample[:, -1]
    # Sample on the source device before transferring to the host.
    return np.nanmedian(np.asarray(sample), axis=0)


def _diagnostic_series(cube):
    """Calculate a scalar time series from sparsely sampled detector pixels."""
    data = cube.data
    if data.ndim == 4:
        data = data[:, -1]
    return np.nanmedian(np.asarray(data[:, ::4, ::32]), axis=(1, 2))


def _segment_geometry(cube):
    """Return cumulative edges, zero-based offsets, and FITS INTSTARTs."""
    nints = int(cube.data.shape[0])
    edges = np.asarray(getattr(cube.meta, 'segment_edges', (nints,)), dtype=int)
    if (edges.ndim != 1 or not edges.size or edges[-1] != nints or
            np.any(edges <= 0) or np.any(np.diff(edges) <= 0)):
        raise ValueError('diagnostic capture requires valid segment_edges')
    offsets = np.concatenate(([0], edges[:-1]))
    starts = np.asarray(getattr(cube.meta, 'segment_int_starts', offsets + 1), dtype=int)
    if starts.shape != edges.shape:
        raise ValueError('diagnostic capture requires one INTSTART per segment')
    return edges, offsets, starts


def _capture_basic_nine_panels(cube, *, panel_count=9, randint=None):
    """Capture v1 ``basic_nine_panel_plot`` inputs without copying a cube."""
    if randint is None:
        randint = np.random.randint
    data = cube.data
    if data.ndim not in (3, 4):
        return None
    edges, offsets, starts = _segment_geometry(cube)
    selected = np.asarray(randint(0, len(edges), int(panel_count)), dtype=int)
    panels, labels = [], []
    for segment in selected:
        segment = int(segment)
        lo, hi = int(offsets[segment]), int(edges[segment])
        # Exclude the final integration as in v1, except for single-integration segments.
        local = int(randint(0, max(1, hi - lo - 1)))
        integration = lo + local
        if data.ndim == 4:
            group = int(randint(0, data.shape[1]))
            panels.append(np.array(data[integration, group], copy=True))
            labels.append(f'({int(starts[segment] + local)}, {group})')
        else:
            panels.append(np.array(data[integration], copy=True))
            labels.append(f'({int(starts[segment] + local)})')
    return {'panels': tuple(panels), 'labels': tuple(labels)}


def _capture_background_record(before, after, background_model):
    """Capture both v1 SOSS background diagnostics from in-memory cubes."""
    record = _capture_basic_nine_panels(after.cube)
    if record is None:
        return None
    edges, _, _ = _segment_geometry(before.cube)
    first_stop = int(edges[0])
    dimy = before.cube.data.shape[-2]
    row_start, row_end = ((5, 21) if dimy == 96 else (230, 251))

    def row_stack(cube):
        """Calculate the first-segment median of the background reference rows."""
        from exotedrf.v2.plotting_compat import last_group_median
        return last_group_median(cube.data, first_stop, rows=slice(row_start, row_end))
    pre = row_stack(before.cube)
    post = row_stack(after.cube)
    model = np.asarray(background_model)
    if model.ndim == 3:
        model_plane = model[-1] if before.cube.data.ndim == 4 else model[0]
    elif model.ndim == 2:
        model_plane = model
    else:
        raise ValueError(f'background model must be 2D/3D, got shape {model.shape}')
    record.update(before_row=np.nanmedian(pre, axis=0), after_row=np.nanmedian(post, axis=0),
        background_row=np.nanmedian(model_plane[row_start:row_end], axis=0),
        subarray96=(dimy == 96))
    return record


def _plot_background_row(path, record, *, scale=1.):
    """Plot background reference rows as in v1 make_background_row_plot."""
    before = np.asarray(record['before_row'])
    after = np.asarray(record['after_row'])
    background = np.asarray(record['background_row'])
    dimx = before.size
    split = min(700, dimx)
    x = np.arange(dimx)
    figure = _new_figure(figsize=(5., 3.))
    ax = figure.subplots()
    ax.plot(before)
    ax.plot(x[:split], background[:split], c='black', ls='--')
    ax.plot(after)
    residual = np.empty(0, dtype=float)
    if split < dimx:
        scaled = (scale * (background[split:] - background[split]) + background[split])
        ax.plot(x[split:], scaled, c='black', ls='--')
        residual = before[split:] - scaled
        ax.plot(x[split:], residual)
    ax.axvline(700, ls=':', c='grey')
    ax.axhline(0, ls=':', c='grey')
    candidates = [np.nanmin(after)]
    if residual.size:
        candidates.append(np.nanmin(residual))
    upper_percentile = 75 if record.get('subarray96') else 95
    lower = np.nanmin(candidates)
    upper = np.nanpercentile(before, upper_percentile)
    if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
        ax.set_ylim(lower, upper)
    ax.set_xlabel('Spectral Pixel', fontsize=12)
    ax.set_ylabel('Counts', fontsize=12)
    _save_figure(figure, path, dpi=100)


def _host_trace_mask(ypos, width, dimy):
    """Construct a detector mask from trace centers and a full width."""
    ypos = np.asarray(ypos, dtype=float)
    low = np.floor(np.maximum(0., ypos - float(width) / 2.))
    high = np.floor(np.minimum(float(dimy), ypos + float(width) / 2.))
    rows = np.arange(dimy, dtype=float)[:, None]
    return ((rows >= low[None]) & (rows < high[None]) & np.isfinite(ypos)[None])


def _oof_trace_mask(prepared, params, method, dimy):
    """Return v1's order-1 PSD exclusion mask as one detector plane."""
    inner = float(params.get('soss_inner_mask_width', 40))
    outer = float(params.get('soss_outer_mask_width', 70))
    o1 = np.asarray(prepared['centroid_o1'])
    if method == 'solve':
        # Mask pixels outside the order-1 trace window.
        return ~_host_trace_mask(o1, outer, dimy)
    o2 = np.asarray(prepared.get('centroid_o2_padded', np.full_like(o1, np.nan)))
    o3 = np.asarray(prepared.get('centroid_o3_padded', np.full_like(o1, np.nan)))
    m1_in = _host_trace_mask(o1, inner, dimy)
    m2_in = _host_trace_mask(o2, inner, dimy)
    m3_in = _host_trace_mask(o3, inner, dimy)
    trace = m1_in | m2_in | m3_in
    if method == 'achromatic':
        return trace
    m1_out = _host_trace_mask(o1, outer, dimy)
    return trace | ~(m1_out & ~m1_in)


def _plot_timeseries_value(timeseries, cube, segment, local):
    """Index a plot light curve with v1's historical INTSTART convention."""
    if timeseries is None:
        return 1.
    values = np.asarray(timeseries)
    if values.ndim > 1:
        values = np.nanmedian(values, axis=1)
    _, offsets, starts = _segment_geometry(cube)
    # Use the v1 INTSTART index when the supplied series contains it.
    legacy_index = int(starts[segment] + local)
    if legacy_index < values.shape[0]:
        return values[legacy_index]
    return values[int(offsets[segment] + local)]


def _capture_oneoverf_record(before, after, ctx, params, *, randint=None):
    """Capture v1's residual-frame and detector-readout PSD diagnostics."""
    if randint is None:
        randint = np.random.randint
    from exotedrf.v2 import stages as v2stages
    data_before = before.cube.data
    data_after = after.cube.data
    if data_before.ndim not in (3, 4) or data_after.shape != data_before.shape:
        return None
    nints, dimy, dimx = (data_after.shape[0], data_after.shape[-2], data_after.shape[-1])
    baseline = v2stages.baseline_bool_for_meta(after.cube.meta, nints)
    deep_after = v2stages._host_deepstack(data_after, baseline)
    prepared = before.aux.get('_oneoverf_grp_prepared')
    if prepared is not None:
        deep_before = np.asarray(prepared['deep'])
    else:
        deep_before = before.aux.get('_oneoverf_int_deepstack')
        if deep_before is None:
            deep_before = v2stages._host_deepstack(data_before, baseline)
        deep_before = np.asarray(deep_before)
    solve = v2stages._is_oof_solve(ctx)
    if solve:
        # Combine the custom pixel mask with the order-1 trace exclusion.
        custom = ctx.get('outlier_mask')
        base_mask = (np.zeros((nints, dimy, dimx), dtype=bool) if custom is None else
            np.broadcast_to(np.asarray(custom, dtype=bool), (nints, dimy, dimx)))
        centroids = (before.aux.get('centroids_group') if data_before.ndim == 4 else
            before.aux.get('centroids_int')) or \
            after.aux.get('centroids_int') or ctx.get('centroids') or {}
        o1 = np.asarray(centroids.get('ypos o1', np.full(dimx, np.nan)), dtype=float)
        prepared = {'centroid_o1': np.pad(o1[:dimx], (0, max(0, dimx - o1.size)),
            constant_values=np.nan)[:dimx]}
    elif prepared is not None:
        base_mask = np.asarray(prepared['base_mask'], dtype=bool)
    else:
        if hasattr(before.cube, 'dq'):
            dq = np.asarray(before.cube.dq, dtype=np.uint32)
            base_mask = dq != 0
        else:
            # Combine PIXELDQ with the final-group GROUPDQ.
            base_mask = ((np.asarray(before.cube.pixeldq, dtype=np.uint32) != 0)[None]
                | (np.asarray(before.cube.groupdq[:, -1], dtype=np.uint32) != 0))
        custom = ctx.get('outlier_mask')
        if custom is not None:
            base_mask |= np.asarray(custom, dtype=bool)
        order0 = ctx.get('order0_mask')
        if order0 is not None:
            base_mask |= np.asarray(order0, dtype=bool)[None]
        temporal_source = (data_before[:, -1] if data_before.ndim == 4 else data_before)
        base_mask |= v2stages._segment_temporal_outliers(temporal_source, before.cube.meta)
        centroids = (before.aux.get('centroids_int') or after.aux.get('centroids_int') or
            ctx.get('centroids'))
        if centroids is None:
            centroids = {}
        o1 = np.asarray(centroids.get('ypos o1', np.full(dimx, np.nan)), dtype=float)
        prepared = {'centroid_o1': np.pad(o1[:dimx], (0, max(0, dimx - o1.size)),
            constant_values=np.nan)[:dimx], 'centroid_o2_padded': np.full(dimx, np.nan),
            'centroid_o3_padded': np.full(dimx, np.nan)}
        for order in (2, 3):
            values = centroids.get(f'ypos o{order}')
            if values is not None:
                values = np.asarray(values, dtype=float)
                padded = np.full(dimx, np.nan)
                padded[:min(dimx, values.size)] = values[:dimx]
                prepared[f'centroid_o{order}_padded'] = padded
    method = {'scale-achromatic': 'achromatic', 'scale-achromatic-window': 'achromatic-window',
        'scale-chromatic': 'chromatic', 'solve': 'solve',
        }.get(ctx.get('opts', {}).get('oof_method', 'scale-achromatic'), 'achromatic')
    trace_mask = _oof_trace_mask(prepared, params, method, dimy)
    timeseries = ctx.get('soss_timeseries')
    edges, offsets, starts = _segment_geometry(after.cube)

    def residual(source, deep, segment, local, group=None):
        """Subtract the scaled deep stack from a sampled detector plane."""
        integration = int(offsets[segment] + local)
        scale = _plot_timeseries_value(timeseries, after.cube, segment, local)
        if source.ndim == 4:
            plane = np.asarray(source[integration, group])
            reference = np.asarray(deep[group])
        else:
            plane = np.asarray(source[integration])
            reference = np.asarray(deep)
        if np.ndim(scale):
            scale = np.asarray(scale)[None, :]
        return plane - reference * scale
    selected = np.asarray(randint(0, len(edges), 9), dtype=int)
    panels, labels = [], []
    for segment in selected:
        segment = int(segment)
        length = int(edges[segment] - offsets[segment])
        local = int(randint(0, max(1, length - 1)))
        if data_after.ndim == 4:
            group = int(randint(0, data_after.shape[1]))
            panels.append(residual(data_after, deep_after, segment, local, group))
            labels.append(f'({int(starts[segment] + local)}, {group})')
        else:
            panels.append(residual(data_after, deep_after, segment, local))
            labels.append(f'({int(starts[segment] + local)})')
    psd_samples = []
    selected = np.asarray(randint(0, len(edges), 10), dtype=int)
    for segment in selected:
        segment = int(segment)
        length = int(edges[segment] - offsets[segment])
        local = int(randint(0, max(1, length - 1)))
        integration = int(offsets[segment] + local)
        group = (int(randint(0, data_after.shape[1])) if data_after.ndim == 4 else None)
        old = residual(data_before, deep_before, segment, local, group)
        new = residual(data_after, deep_after, segment, local, group)
        mask = np.asarray(base_mask[integration], dtype=bool) | trace_mask
        psd_samples.append({'before': old, 'after': new, 'mask': mask})
    return {'panels': tuple(panels), 'labels': tuple(labels),
        'vmin': np.nanpercentile(panels[-1], 5), 'vmax': np.nanpercentile(panels[-1], 95),
        'psd_samples': tuple(psd_samples),
        'frame_time': float(getattr(after.cube.meta, 'frame_time', 5.494))}


def _plot_oneoverf_psd(path, record, *, tpix=1e-5, tgap=1.2e-4):
    """Plot detector readout noise power as in v1 make_oneoverf_psd."""
    from exotedrf.v2 import plotting_compat
    samples = tuple(record.get('psd_samples', ()))
    if not samples:
        return False
    tframe = float(record.get('frame_time', 5.494))
    freqs = np.logspace(np.log10(1. / tframe), np.log10(1. / tpix), 100)
    before_values, after_values, valid = [], [], []
    first_shape = np.asarray(samples[0]['before']).shape
    count = int(np.prod(first_shape))
    pixels = np.arange(count)
    timestamps = (tpix * (pixels + 1) + tgap * np.floor_divide(pixels, first_shape[0]))
    for sample in samples:
        if np.asarray(sample['before']).shape != first_shape:
            raise ValueError('all 1/f PSD detector planes must match')
        before = np.asarray(sample['before']).flatten('F')[::-1]
        after = np.asarray(sample['after']).flatten('F')[::-1]
        mask = np.asarray(sample['mask'], dtype=bool).flatten('F')[::-1]
        before_values.append(before)
        after_values.append(after)
        valid.append(~mask & np.isfinite(before) & np.isfinite(after))
    before_values = np.asarray(before_values)
    after_values = np.asarray(after_values)
    valid = np.asarray(valid, dtype=bool)
    powers = plotting_compat.batched_lombscargle_psd(
        timestamps, np.concatenate((before_values, after_values), axis=0),
        np.concatenate((valid, valid), axis=0), freqs)
    before_power, after_power = np.split(powers, 2, axis=0)
    figure = _new_figure(figsize=(7., 3.))
    ax = figure.subplots()
    for index in range(len(samples)):
        ax.plot(freqs[:-1], before_power[index, :-1], c='salmon', alpha=.1)
        ax.plot(freqs[:-1], after_power[index, :-1], c='royalblue', alpha=.1)
    ax.plot(freqs[:-1], np.nanmedian(before_power, axis=0)[:-1], c='red',
        lw=2, label='Before Correction')
    ax.plot(freqs[:-1], np.nanmedian(after_power, axis=0)[:-1], c='blue',
        lw=2, label='After Correction')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency [Hz]', fontsize=12)
    ax.set_yscale('log')
    finite_after = after_power[np.isfinite(after_power) & (after_power > 0)]
    finite_before = before_power[np.isfinite(before_power)]
    if finite_after.size and finite_before.size:
        lower = np.nanpercentile(finite_after, .1)
        upper = np.nanmax(finite_before)
        if np.isfinite(lower) and np.isfinite(upper) and upper > lower > 0:
            ax.set_ylim(lower, upper)
    ax.set_ylabel('PSD', fontsize=12)
    ax.legend(loc=1)
    _save_figure(figure, path, dpi=100)
    return True


def _plot_image(path, image, title, *, cmap='viridis'):
    """Plot a detector image with percentile color limits."""
    figure = _new_figure(figsize=(9., 3.5))
    ax = figure.subplots()
    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [5, 95])
    else:
        vmin, vmax = 0., 1.
    shown = ax.imshow(image, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    ax.set_title(title)
    figure.colorbar(shown, ax=ax)
    _save_figure(figure, path, dpi=150)


def _plot_image_pair(path, before, after, title):
    """Plot detector images before and after correction."""
    figure = _new_figure(figsize=(12., 4.))
    axes = figure.subplots(1, 2)
    for ax, image, label in zip(axes, (before, after), ('Before', 'After')):
        finite = image[np.isfinite(image)]
        if finite.size:
            vmin, vmax = np.percentile(finite, [5, 95])
        else:
            vmin, vmax = 0., 1.
        shown = ax.imshow(image, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel('X Pixel')
        ax.set_ylabel('Y Pixel')
        figure.colorbar(shown, ax=ax)
    figure.suptitle(title)
    _save_figure(figure, path, dpi=150)


def _plot_profile_pair(path, before, after, title):
    """Plot median row profiles before and after correction."""
    figure = _new_figure(figsize=(9., 4.))
    ax = figure.subplots()
    ax.plot(np.nanmedian(before, axis=1), label='Before')
    ax.plot(np.nanmedian(after, axis=1), label='After')
    ax.set_xlabel('Y Pixel')
    ax.set_ylabel('Median Signal')
    ax.set_title(title)
    ax.legend()
    _save_figure(figure, path, dpi=150)


def _plot_series(path, before, after, title):
    """Plot scalar time series before and after correction."""
    figure = _new_figure(figsize=(9., 4.))
    ax = figure.subplots()
    ax.plot(before, alpha=0.75, label='Before')
    ax.plot(after, alpha=0.75, label='After')
    ax.set_xlabel('Integration')
    ax.set_ylabel('Median Signal')
    ax.set_title(title)
    ax.legend()
    _save_figure(figure, path, dpi=150)


def _plot_psd(path, before, after, title, frame_time=1.):
    """Plot Fourier power spectra before and after correction."""
    figure = _new_figure(figsize=(9., 4.))
    ax = figure.subplots()
    for values, label in ((before, 'Before'), (after, 'After')):
        values = np.asarray(values, dtype=float)
        values = values - np.nanmedian(values)
        finite = np.isfinite(values)
        values = np.where(finite, values, 0.)
        freq = np.fft.rfftfreq(values.size, d=float(frame_time))
        power = np.abs(np.fft.rfft(values)) ** 2
        use = freq > 0
        ax.loglog(freq[use], power[use], label=label)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power')
    ax.set_title(title)
    ax.legend()
    _save_figure(figure, path, dpi=150)


def _plot_correction_fallback(path1, path2, before, after, *, relative_second):
    """Plot signal differences from paired diagnostic samples."""
    before = np.asarray(before)
    after = np.asarray(after)
    figure = _new_figure(figsize=(7., 5.))
    ax = figure.subplots()
    ax.scatter(before, after - before, s=1, alpha=.2)
    ax.set_xlabel('Input Signal')
    ax.set_ylabel('Correction (after - before)')
    _save_figure(figure, path1, dpi=150)
    figure = _new_figure(figsize=(7., 5.))
    ax = figure.subplots()
    values = after - before
    if relative_second:
        values = np.divide(values, before, out=np.full_like(values, np.nan),
            where=before != 0) * 100.
    ax.scatter(before, values, s=1, alpha=.2)
    ax.set_xlabel('Input Signal')
    ax.set_ylabel('Correction [%]' if relative_second else 'Correction')
    _save_figure(figure, path2, dpi=150)


def _capture_jump_panels(cube, *, panel_count=9, randint=None):
    """Capture sampled group differences and flags as in v1 make_jump_location_plot."""
    if randint is None:
        randint = np.random.randint
    data = cube.data
    groupdq = cube.groupdq
    if data.ndim != 4 or groupdq.ndim != 4 or data.shape != groupdq.shape:
        return None
    nints, ngroups, _, _ = data.shape
    # Require two groups to form a group difference.
    if nints < 1 or ngroups < 2:
        return None
    meta = cube.meta
    edges = np.asarray(getattr(meta, 'segment_edges', (nints,)), dtype=int)
    if edges.ndim != 1 or not edges.size or edges[-1] != nints or \
            np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError('Jump diagnostic requires valid cumulative segment_edges')
    offsets = np.concatenate(([0], edges[:-1]))
    starts = np.asarray(getattr(meta, 'segment_int_starts', offsets + 1), dtype=int)
    if starts.shape != edges.shape:
        raise ValueError('Jump diagnostic requires one INTSTART per segment')
    instrument = str(getattr(meta, 'mode', '')).split('/')[0].upper()
    selected_segments = np.asarray(randint(0, len(edges), int(panel_count)), dtype=int)
    pixeldq = np.asarray(cube.pixeldq, dtype=np.uint32)
    hot = ((pixeldq & np.uint32((1 << 11) | (1 << 12))) != 0)
    dropped_cache = {}
    plane_cache = {}
    panels = []
    for segment_index in selected_segments:
        segment_index = int(segment_index)
        lo, hi = int(offsets[segment_index]), int(edges[segment_index])
        local_integration = int(randint(0, hi - lo))
        dropped = ()
        if instrument == 'MIRI':
            dropped = dropped_cache.get(segment_index)
            if dropped is None:
                first_dq = np.asarray(groupdq[lo], dtype=np.uint8)
                dropped = tuple(np.flatnonzero(((first_dq & np.uint8(1)) != 0).all(axis=(-2, -1))))
                dropped_cache[segment_index] = dropped
        group = int(randint(1, ngroups))
        # Skip segments with no plottable groups before rejection sampling.
        if len(dropped) >= ngroups - 1:
            continue
        while group in dropped:
            group = int(randint(1, ngroups))
        integration = lo + local_integration
        cache_key = (integration, group)
        cached = plane_cache.get(cache_key)
        if cached is None:
            science = np.asarray(data[integration, group - 1:group + 1])
            difference = science[1] - science[0]
            dq = np.asarray(groupdq[integration, group], dtype=np.uint8)
            cached = {'difference': difference, 'jump': (dq & np.uint8(4)) != 0,
                'dnu': (dq & np.uint8(1)) != 0}
            plane_cache[cache_key] = cached
        panels.append({**cached, 'segment': segment_index, 'local_integration': local_integration,
            'integration': int(starts[segment_index] + local_integration), 'group': group})
    return {'instrument': instrument, 'hot': hot, 'panels': tuple(panels)}


def _plot_jump_panels(path, record):
    """Render the compact capture in v1's nine-panel Jump style."""
    from matplotlib.collections import EllipseCollection
    from matplotlib.patches import Ellipse
    from matplotlib.ticker import NullFormatter
    panels = tuple(record.get('panels', ()))
    if not panels:
        return False
    figure = _new_figure(figsize=(15., 9.))
    figure.set_facecolor('white')
    axes = np.asarray(figure.subplots(3, 3)).reshape(-1)
    instrument = str(record.get('instrument', '')).upper()
    # Orient flag ellipses along the spectral direction.
    width, height = ((3., 21.) if instrument == 'MIRI' else (21., 3.))
    hot = np.asarray(record.get('hot'), dtype=bool)

    def add_flags(ax, mask, color, label):
        """Draw a flag collection and return its ellipse legend handle."""
        ypos, xpos = np.where(np.asarray(mask, dtype=bool))
        if not xpos.size:
            return None
        count = xpos.size
        collection = EllipseCollection(
            np.full(count, width), np.full(count, height), np.zeros(count),
            units='xy', offsets=np.column_stack((xpos, ypos)),
            transOffset=ax.transData, facecolors='none', edgecolors=color, label=label)
        ax.add_collection(collection)
        # Use a single ellipse as the legend handle for each flag collection.
        return Ellipse((0., 0.), width, height, facecolor='none', edgecolor=color, label=label)

    for index, ax in enumerate(axes):
        if index >= len(panels):
            ax.set_visible(False)
            continue
        panel = panels[index]
        difference = np.asarray(panel['difference'])
        ax.imshow(difference, aspect='auto', origin='lower', vmin=0,
            vmax=np.nanpercentile(difference, 85))
        legend_handles = [add_flags(ax, hot, 'blue', 'Hot Pixel'),
            add_flags(ax, panel['dnu'], 'dodgerblue', 'Bad Pixel'),
            add_flags(ax, panel['jump'], 'red', 'Cosmic Ray')]
        ax.text(0.05 * difference.shape[-1], 0.9 * difference.shape[-2],
            f"({panel['integration']}, {panel['group']})", color='white', fontsize=12)
        row, col = divmod(index, 3)
        if col:
            ax.yaxis.set_major_formatter(NullFormatter())
        else:
            ax.tick_params(axis='y', labelsize=10)
        if row != 2:
            ax.xaxis.set_major_formatter(NullFormatter())
        else:
            ax.tick_params(axis='x', labelsize=10)
        if index == 0:
            ax.legend(handles=[handle for handle in legend_handles if handle is not None], loc=1)
    figure.subplots_adjust(hspace=0.1)
    _save_figure(figure, path, dpi=100)
    return True


def _fallback_capture(before, after, previous=None):
    """Capture image and time-series summaries for observer states without detector data."""
    return {
        'before_image': previous['after_image'] if previous else _representative_image(before),
        'before_series': previous['after_series'] if previous else _diagnostic_series(before),
        'after_image': _representative_image(after), 'after_series': _diagnostic_series(after),}


class CompatibilityCapture:
    """Capture compact final-pass summaries for v1-style diagnostics."""

    def __init__(self, paths, ctx, params=None):
        """Initialize the diagnostic capture."""
        self.paths = {key: Path(value) for key, value in paths.items()}
        self.ctx = ctx
        self.params = {} if params is None else params
        self.enabled = bool(ctx.get('opts', {}).get('do_plots', False))
        self.records = {}
        self.written = {}
        self.segmented = False
        self.timings = {'diagnostic_capture': 0., 'diagnostic_render': 0.}

    def __call__(self, step, before, after):
        """Capture diagnostic arrays before and after a calibration step."""
        name = step.name
        if not self.enabled:
            return
        started = time.perf_counter()

        if name == 'DQInitStep':
            dq = np.asarray(after.cube.groupdq[-1, -1], dtype=np.uint32)
            self.records[name] = (dq & np.uint32(2)) != 0
        elif name in ('INLCorrStep', 'LinearityStep'):
            from exotedrf.v2 import plotting_compat
            edges, _, _ = _segment_geometry(before.cube)
            first_stop = int(edges[0])
            if name == 'INLCorrStep':
                self.records[name] = plotting_compat.capture_inl(
                    before.cube.data, after.cube.data, first_stop)
            else:
                instrument = str(getattr(after.cube.meta, 'mode', '')).split('/')[0].upper()
                self.records[name] = plotting_compat.capture_linearity(
                    before.cube.data, after.cube.data, first_stop, instrument=instrument,
                    groupdq=(after.cube.groupdq if instrument == 'MIRI' else None))
        elif name == 'SuperBiasStep':
            panels = _capture_basic_nine_panels(after.cube)
            if panels is not None:
                self.records[name] = panels
        elif name in ('BackgroundStep_grp', 'BackgroundStep'):
            model_key = ('bkg_grp' if name.endswith('_grp') else 'bkg_int')
            model = getattr(after, 'aux', {}).get(model_key)
            # Capture the SOSS background reference rows for NIRISS only.
            meta = getattr(after.cube, 'meta', None)
            instrument = str(getattr(meta, 'mode', 'NIRISS')).split('/')[0].upper()
            if instrument != 'NIRISS':
                model = None
            if model is not None:
                record = _capture_background_record(before, after, model)
                if record is not None:
                    self.records[name] = record
            elif not hasattr(after.cube, 'data'):
                self.records[name] = _fallback_capture(before.cube, after.cube)
        elif name in ('OneOverFStep_grp', 'OneOverFStep_int'):
            if hasattr(after.cube, 'data'):
                record = _capture_oneoverf_record(before, after, self.ctx, self.params)
                if record is not None:
                    self.records[name] = record
            else:
                preceding = ('BackgroundStep_grp' if name.endswith('_grp') else 'BackgroundStep')
                previous = self.records.get(preceding)
                self.records[name] = _fallback_capture(before.cube, after.cube, previous)
        elif name == 'JumpStep':
            required = ('data', 'groupdq', 'pixeldq', 'meta')
            panels = (_capture_jump_panels(after.cube)
                if all(hasattr(after.cube, key) for key in required) else None)
            if panels is not None:
                self.records[name] = panels
            else:
                # Count changed jump flags when detector planes are unavailable.
                old = before.cube.groupdq
                new = after.cube.groupdq
                counts = np.zeros(old.shape[-2:], dtype=np.uint32)
                for lo in range(0, old.shape[0], 8):
                    old_chunk = np.asarray(old[lo:lo + 8], dtype=np.uint8)
                    new_chunk = np.asarray(new[lo:lo + 8], dtype=np.uint8)
                    changed = ((old_chunk ^ new_chunk) & np.uint8(4)) != 0
                    counts += np.sum(changed, axis=(0, 1), dtype=np.uint32)
                self.records[name] = counts
        elif name == 'BadPixStep':
            deepframe = after.aux.get('deepframe')
            if deepframe is None:
                deepframe = _representative_image(before.cube)
            hot_pixel_map = after.aux.get('hot_pixel_map')
            if hot_pixel_map is None:
                hot_pixel_map = np.zeros(np.asarray(deepframe).shape, dtype=bool)
            self.records[name] = {'deepframe': np.asarray(deepframe),
                'mask': np.asarray(hot_pixel_map, dtype=bool), 'hotpix': np.asarray(after.aux.get(
                'badpix_plot_hot', hot_pixel_map), dtype=bool),
                'nanpix': np.asarray(after.aux.get('badpix_plot_nan', np.zeros_like(hot_pixel_map)),
                dtype=bool), 'otherpix': np.asarray(after.aux.get(
                'badpix_plot_other', np.zeros_like(hot_pixel_map)), dtype=bool)}
        self.timings['diagnostic_capture'] += time.perf_counter() - started

    def for_segment(self, index, total):
        """Return a segment observer, capturing diagnostics first and the final DQ plane last.

        Parameters
        ----------
        index : int
            Zero-based FITS segment index.
        total : int
            Number of FITS segments.

        Returns
        -------
        observer : callable
            Observer for the selected FITS segment.
        """
        index, total = int(index), int(total)
        if total < 1 or not 0 <= index < total:
            raise ValueError('segment index must be within [0, total)')
        self.segmented = True
        if index == 0:
            return self

        def observe(step, before, after):
            """Capture the final segment saturation flags."""
            if index == total - 1 and step.name == 'DQInitStep':
                self(step, before, after)
        return observe

    def render(self, final_state):
        """Render v1-named diagnostics once, after winner selection.

        Parameters
        ----------
        final_state : State
            Final calibrated pipeline state.

        Returns
        -------
        written : dict
            Names and paths of the diagnostic plots written.
        """
        if not self.enabled:
            return dict(self.written)
        # Wait for deferred PNG saves before returning.
        with _concurrent_saves():
            return self._render(final_state)

    def _render(self, final_state):
            # Render the captured arrays with the v1 plotting functions.
        """Render captured diagnostics with instrument-specific filenames."""
        from exotedrf import plotting as v1_plotting
        started = time.perf_counter()
        stage1, stage2 = self.paths['stage1'], self.paths['stage2']
        meta = getattr(final_state.cube, 'meta', None)
        instrument = str(getattr(meta, 'mode', '')).split('/')[0].upper()
        detector_suffix = (f'_{str(getattr(meta, "detector", "")).lower()}'
            if instrument == 'NIRSPEC' else '')

        def diagnostic_name(stem):
            """Append the detector suffix to a diagnostic filename."""
            return f'{stem}{detector_suffix}.png'
        saturated = self.records.get('DQInitStep')
        if saturated is not None:
            # Start a new figure before drawing the saturation image.
            import matplotlib.pyplot as plt
            plt.figure()
            path = stage1 / diagnostic_name('dqinitstep')
            v1_plotting.plot_saturated_pixels(np.asarray(saturated), outfile=path, show_plot=False)
            self.written['dqinit_plot'] = str(path)
        from exotedrf.v2 import plotting_compat
        for step, stem, render, relative in (
            ('INLCorrStep', 'inlcorrstep', plotting_compat.render_inl, False),
            ('LinearityStep', 'linearitystep', plotting_compat.render_linearity, True)):
            record = self.records.get(step)
            if record is None:
                continue
            path1 = stage1 / diagnostic_name(f'{stem}_1')
            path2 = stage1 / diagnostic_name(f'{stem}_2')
            if isinstance(record, dict):
                render(record, path1, path2)
            else:
                _plot_correction_fallback(path1, path2, *record, relative_second=relative)
            self.written[f'{stem}_plot_1'] = str(path1)
            self.written[f'{stem}_plot_2'] = str(path2)
        superbias = self.records.get('SuperBiasStep')
        if superbias is not None:
            path = stage1 / diagnostic_name('superbiasstep_1')
            if isinstance(superbias, dict) and 'panels' in superbias:
                v1_plotting.nine_panel_plot(list(superbias['panels']), list(superbias['labels']),
                    outfile=path, show_plot=False)
            else:
                _plot_image(path, superbias, 'Superbias Reference')
            self.written['superbias_plot'] = str(path)

        for name, directory, stem, title in (('BackgroundStep_grp', stage1, 'backgroundstep',
            'Group-level Background Correction'), ('BackgroundStep', stage2, 'backgroundstep',
            'Integration-level Background Correction')):
            record = self.records.get(name)
            if record is None:
                continue
            path1 = directory / f'{stem}_1.png'
            path2 = directory / f'{stem}_2.png'
            if 'panels' in record:
                v1_plotting.nine_panel_plot(list(record['panels']), list(record['labels']),
                    outfile=path1, show_plot=False, max_percentile=70)
                _plot_background_row(path2, record)
            else:
                _plot_image_pair(path1, record['before_image'], record['after_image'], title)
                _plot_profile_pair(path2, record['before_image'], record['after_image'],
                    f'{title}: Row Profile')
            self.written[f'{name}_plot_1'] = str(path1)
            self.written[f'{name}_plot_2'] = str(path2)

        for name, directory, title in (('OneOverFStep_grp', stage1, 'Group-level 1/f Correction'),
            ('OneOverFStep_int', stage2, 'Integration-level 1/f Correction')):
            record = self.records.get(name)
            if record is None:
                continue
            path1 = directory / diagnostic_name('oneoverfstep_1')
            path2 = directory / diagnostic_name('oneoverfstep_2')
            if 'panels' in record:
                v1_plotting.nine_panel_plot(list(record['panels']), list(record['labels']),
                    outfile=path1, show_plot=False, vmin=record['vmin'], vmax=record['vmax'])
                _plot_oneoverf_psd(path2, record)
            else:
                _plot_series(path1, record['before_series'], record['after_series'], title)
                _plot_psd(path2, record['before_series'], record['after_series'],
                    f'{title}: Power Spectrum', frame_time=getattr(final_state.cube.meta,
                    'frame_time', 1.))
            self.written[f'{name}_plot_1'] = str(path1)
            self.written[f'{name}_plot_2'] = str(path2)
        self._render_oneoverf_solve(final_state, stage1, stage2)
        jump = self.records.get('JumpStep')
        if jump is not None:
            path = stage1 / diagnostic_name('jump')
            if isinstance(jump, dict):
                rendered = _plot_jump_panels(path, jump)
            else:
                _plot_image(path, np.asarray(jump, dtype=float),
                    'Time-domain Jump Flags', cmap='magma')
                rendered = True
            if rendered:
                self.written['jump_plot'] = str(path)
        order0 = self.ctx.get('order0_mask')
        f277w = self.ctx.get('f277w')
        if order0 is not None:
            # Preserve the v1 product filename.
            path = stage1 / 'contminant_mask.png'
            if f277w is not None:
                v1_plotting.make_order0_mask_plot(
                    np.asarray(order0), np.asarray(f277w), outfile=path, show_plot=False)
            else:
                _plot_image(path, np.asarray(order0, dtype=float),
                    'F277W Order-0 Contaminant Mask', cmap='magma')
            self.written['contaminant_plot'] = str(path)
        badpix = self.records.get('BadPixStep')
        if badpix is not None:
            path = stage2 / diagnostic_name('badpixstep')
            image = np.nan_to_num(np.asarray(badpix['deepframe']), nan=0.)
            hot = np.asarray(badpix.get('hotpix', badpix.get('mask')), dtype=bool)
            nanpix = np.asarray(badpix.get('nanpix', np.zeros_like(hot)), dtype=bool)
            other = np.asarray(badpix.get('otherpix', hot & False), dtype=bool)
            instrument = str(getattr(final_state.cube.meta, 'mode', '')).split('/')[0].upper()
            v1_plotting.make_badpix_plot(image, np.where(hot), np.where(nanpix), np.where(other),
                outfile=path, show_plot=False, miri_scale=(instrument == 'MIRI'))
            self.written['badpix_plot'] = str(path)
        components = final_state.aux.get('pca_components')
        eigvals = final_state.aux.get('pca_eigvals')
        if components is not None and eigvals is not None:
            variants = [('', 'TSO Stability PCA', 'pca_plot')]
            if self.ctx.get('opts', {}).get('remove_components') is not None:
                variants.append(('_reconstructed', 'Reconstructed TSO Stability PCA',
                    'pca_reconstructed_plot'))
            for suffix, title, key in variants:
                components = final_state.aux.get(f'pca_components{suffix}')
                eigvals = final_state.aux.get(f'pca_eigvals{suffix}')
                projections = final_state.aux.get(f'pca_projections{suffix}')
                if components is None or eigvals is None:
                    continue
                path = stage2 / diagnostic_name(f'stability_pca{suffix}')
                if projections is not None:
                    v1_plotting.make_pca_plot(
                        np.asarray(components), np.asarray(eigvals), np.asarray(projections),
                        outfile=path, show_plot=False)
                else:
                    self._plot_pca(path, components, eigvals, title)
                self.written[key] = str(path)
        self.timings['diagnostic_render'] += time.perf_counter() - started
        return dict(self.written)

    def _render_oneoverf_solve(self, final_state, stage1, stage2):
        """Plot even- and odd-row solve scalings for each stage and spectral order."""
        from exotedrf import plotting as v1_plotting
        from exotedrf.v2 import stages as v2stages
        aux = getattr(final_state, 'aux', {}) or {}
        for level, directory in (('grp', stage1), ('int', stage2)):
            record = aux.get(v2stages.OOF_SOLVE_AUX_KEYS[level])
            if not record:
                continue
            for order_key in sorted(record):
                values = record[order_key]
                if any(values.get(key) is None for key in ('scale_e', 'scale_o', 'oof_e', 'oof_o')):
                    continue
                scale_e = np.array(values['scale_e'], dtype=np.float64)
                plot_group = scale_e.shape[1] if scale_e.ndim == 3 else 1
                path = directory / f'oneoverfstep_{order_key}_3.png'
                v1_plotting.make_oneoverf_chromatic_plot(
                    scale_e, np.array(values['scale_o'], dtype=np.float64),
                    np.array(values['oof_e'], dtype=np.float64),
                    np.array(values['oof_o'], dtype=np.float64),
                    plot_group, outfile=path, show_plot=False)
                self.written[f'oneoverf_solve_{level}_{order_key}_plot'] = str(path)

    @staticmethod
    def _plot_pca(path, components, eigvals, title):
        """Plot PCA components and their explained variance."""
        components = np.asarray(components)
        eigvals = np.asarray(eigvals)
        figure = _new_figure(figsize=(10., 6.))
        axes = figure.subplots(2, 1)
        count = min(5, components.shape[0])
        for index in range(count):
            values = components[index]
            scale = np.nanstd(values)
            normalized = values if not np.isfinite(scale) or scale == 0 else values / scale
            axes[0].plot(normalized + 3 * index, label=f'Component {index + 1}')
        axes[0].set_xlabel('Integration')
        axes[0].set_ylabel('Normalized Component + offset')
        axes[0].legend(fontsize='small', ncol=2)
        axes[1].bar(np.arange(eigvals.size) + 1, eigvals)
        axes[1].set_xlabel('Component')
        axes[1].set_ylabel('Explained Variance Ratio')
        figure.suptitle(title)
        _save_figure(figure, path, dpi=150)


class OptimalCapture(CompatibilityCapture):
    """Capture diagnostic arrays without detector cubes or full-visit 1/f recomputation."""

    def __init__(self, paths, ctx, params=None):
        """Initialize the diagnostic capture."""
        super().__init__(paths, ctx, params=params)
        self.enabled = bool(ctx.get('opts', {}).get('do_plots', True))
        self.skipped = {}

    def __call__(self, step, before, after):
        """Capture supported diagnostics and record skipped plots."""
        if not self.enabled:
            return
        name = step.name
        # Require the ramp fields needed by each diagnostic.
        if name in ('DQInitStep', 'JumpStep', 'INLCorrStep', 'LinearityStep'):
            required = (('data', 'groupdq', 'pixeldq', 'meta') if name == 'JumpStep' else
                (('groupdq',) if name == 'DQInitStep' else ('data',)))
            field = 'groupdq' if name == 'DQInitStep' else 'data'
            if (not all(hasattr(after.cube, key) for key in required) or
                    getattr(getattr(after.cube, field, None), 'ndim', 0) != 4):
                self.skipped[name] = 'The observer state has no compatible ramp.'
                return
        if name in ('OneOverFStep_grp', 'OneOverFStep_int'):
            self.skipped[name] = ('Would recompute full-visit masks and deep stacks solely '
                'for the diagnostic.')
            return
        if name == 'INLCorrStep':
            edges, _, _ = _segment_geometry(before.cube)
            if int(edges[0]) <= 10 or before.cube.data.shape[1] < 2:
                self.skipped[name] = ('The v1 INL diagnostic requires integrations 10:20 '
                    'and at least two groups.')
                return
        if name == 'LinearityStep' and before.cube.data.shape[1] < 2:
            self.skipped[name] = 'The v1 linearity diagnostic needs two groups.'
            return
        try:
            super().__call__(step, before, after)
        except ValueError as error:
            # Skip linearity diagnostics without usable samples.
            if (name == 'LinearityStep' and str(error) in (
                    'linearity diagnostic found no bright trace pixels',
                    'every MIRI group is dropped; nothing to plot')):
                self.skipped[name] = str(error)
                return
            raise
        if name in self.records:
            # Copy captured arrays so their views do not retain detector cubes.
            started = time.perf_counter()
            self.records[name] = self._owned_record(self.records[name])
            self.timings['diagnostic_capture'] += time.perf_counter() - started

    @staticmethod
    def _owned_record(value):
        """Copy diagnostic arrays recursively to release references to detector cubes."""
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)
        if isinstance(value, dict):
            return {key: OptimalCapture._owned_record(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(OptimalCapture._owned_record(item) for item in value)
        if isinstance(value, list):
            return [OptimalCapture._owned_record(item) for item in value]
        return value

    @property
    def summary(self):
        """Describe diagnostic selection and flux-image binning.

        Returns
        -------
        summary : dict
            Diagnostic selection, skipped plots, and image binning metadata.
        """
        return {'mode': 'optimal', 'enabled': self.enabled,
            'stage1_sample': ('first FITS segment; final DQ plane from last '
            'segment' if self.segmented else
            'v1 per-diagnostic segment selection'), 'skipped': dict(self.skipped),
            'captured_steps': list(self.records), 'flux_image': {'maximum_time_bins': 1200,
            'aggregation': 'mean of consecutive integrations',
            'full_resolution_science_fits': True}}


def _contiguous_trial_groups(trials):
    """Group adjacent trials by optimization phase, checkpoint, and parameter."""
    key = lambda trial: (_trial_value(trial, 'phase'), _trial_value(trial, 'checkpoint'),
        _trial_value(trial, 'parameter'),)
    return [(identity, list(group)) for identity, group in groupby(trials, key)]


def _plot_cost_trials(path, trials, winners):
    """Render the v1 Cost PNG directly from structured trial records."""
    trials = list(trials)
    if not trials:
        return False
    normalized = np.full(len(trials), np.nan, dtype=float)
    colors = np.full(len(trials), 'gray', dtype=object)
    labels = []
    boundaries = []
    group_labels = []
    offset = 0
    for (_, _, parameter), group in _contiguous_trial_groups(trials):
        costs = np.asarray([_trial_value(trial, 'cost', np.nan) for trial in group], dtype=float)
        finite = np.isfinite(costs)
        if finite.any():
            lo = np.min(costs[finite])
            hi = np.max(costs[finite])
            normalized[offset:offset + len(group)][finite] = (
                0. if hi == lo else (costs[finite] - lo) / (hi - lo))
            first_best = np.flatnonzero(finite & (costs == lo))[0]
            colors[offset + first_best] = 'green'
        labels.extend(format_log_value(_trial_value(trial, 'value')) for trial in group)
        group_labels.append((offset, offset + len(group), parameter))
        offset += len(group)
        if offset < len(trials):
            boundaries.append(offset)
    figure = _new_figure(figsize=(max(14., len(trials) * 0.25), 10.))
    axes = figure.subplots(2, 1, gridspec_kw={'height_ratios': (.6, .4)})
    ax, table_ax = axes
    x = np.arange(len(trials))
    ax.scatter(x, normalized, c=colors, s=25)
    for boundary in boundaries:
        ax.axvline(boundary - 0.5, color='gray', ls='--', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Relative Cost (normalized per sweep)')
    run_name = Path(path).stem.removeprefix('Cost_')
    ax.set_title(f'Cost by Single Parameter Sweep: {run_name}')
    ax.set_ylim(-0.05, 1.05)
    for group_number, (start, end, parameter) in enumerate(group_labels):
        y = -0.14 if group_number % 2 == 0 else -0.22
        ax.text((start + end - 1) / 2, y, parameter, ha='center', va='top',
            fontsize=9, transform=ax.get_xaxis_transform())
    table_ax.axis('off')
    # Include only swept parameters in the winner table.
    winner_names = list(dict.fromkeys(_trial_value(trial, 'parameter') for trial in trials
        if _trial_value(trial, 'parameter') in winners))
    if winner_names:
        finite_costs = np.asarray([_trial_value(trial, 'cost', np.nan) for trial in trials],
            dtype=float)
        best_cost = (np.nanmin(finite_costs) if np.isfinite(finite_costs).any() else np.nan)
        columns = [*winner_names, 'cost']
        table = table_ax.table(cellText=[[
            *[format_log_value(winners[name]) for name in winner_names],
                # Round the cost to the precision used in the Cost table.
            format_log_value(float(f'{best_cost:.12f}'))]],
            colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        for (row, _column), cell in table.get_celld().items():
            cell.set_fontsize(7 if row == 0 else 10)
        table.scale(1., 1.8)
        table_ax.text(.5, .65, 'Best Parameters', ha='center', va='bottom', fontsize=12)
    figure.subplots_adjust(bottom=.30)
    _save_figure(figure, path)
    return True


def _diagnostic_wave_bounds(mode, detector):
    """Return the diagnostic wavelength band for an instrument and detector."""
    instrument = str(mode).split('/')[0].upper()
    if instrument == 'NIRSPEC':
        return (2.9, 3.9) if str(detector).upper() == 'NRS1' else (3.8, 5.0)
    return (5., 12.) if instrument == 'MIRI' else (.6, 2.8)


def _diagnostic_spectrum(spectral_products, mode='NIRISS/SOSS', detector=''):
    """Build the v1 diagnostic spectrum for the active instrument band."""
    if not spectral_products or 1 not in spectral_products:
        return None, None
    order1 = spectral_products[1]
    wave1 = np.asarray(order1['wave'], dtype=float)
    flux1 = np.asarray(order1['flux'], dtype=float)
    if flux1.ndim != 2 or flux1.shape[-1] != wave1.size:
        return None, None
    order2 = spectral_products.get(2)
    if order2 is not None:
        wave2 = np.asarray(order2['wave'], dtype=float)
        flux2 = np.asarray(order2['flux'], dtype=float)
        use2 = np.flatnonzero(np.isfinite(wave2) & (wave2 <= 0.85))
        use1 = np.flatnonzero(np.isfinite(wave1) & (wave1 > 0.85))
        if (flux2.ndim == 2 and flux2.shape[0] == flux1.shape[0] and
                flux2.shape[-1] == wave2.size and use2.size and use1.size):
            wave = np.concatenate((wave2[use2], wave1[use1]))
            flux = np.concatenate((flux2[:, use2], flux1[:, use1]), axis=1)
        else:
            wave, flux = wave1, flux1
    else:
        wave, flux = wave1, flux1
    bounds = _diagnostic_wave_bounds(mode, detector)
    keep = np.isfinite(wave) & (wave >= bounds[0]) & (wave <= bounds[1])
    wave, flux = wave[keep], flux[:, keep]
    if not wave.size:
        return None, None
    order = np.argsort(wave, kind='stable')
    wave, flux = wave[order], flux[:, order]
    med = np.nanmedian(flux, axis=0)
    keep = np.isfinite(med) & (med != 0)
    wave, flux = wave[keep], flux[:, keep]
    if not wave.size:
        return None, None
    _, unique = np.unique(np.round(wave, 12), return_index=True)
    unique.sort()
    return wave[unique], flux[:, unique]


_MAX_IMAGE_ROWS = 1200


def _binned_flux_image(flux, column_median, *, max_image_rows=None):
    """Normalize flux columns and average consecutive integrations for the image."""
    nints, width = flux.shape
    limit = _MAX_IMAGE_ROWS if max_image_rows is None else int(max_image_rows)
    bin_size = max(1, int(np.ceil(nints / max(1, limit))))
    if bin_size == 1:
        return flux / column_median, np.arange(nints + 1), 1
    time_edges = np.append(np.arange(0, nints, bin_size, dtype=int), nints)
    image = np.empty((time_edges.size - 1, width), dtype=float)
    for index, (low, high) in enumerate(zip(time_edges[:-1], time_edges[1:])):
        block = flux[int(low):int(high)]
        counts = np.sum(~np.isnan(block), axis=0)
        image[index] = np.divide(np.nansum(block, axis=0), counts,
            out=np.full(width, np.nan), where=counts != 0)
    return image / column_median, time_edges, bin_size


def _plot_white_flux(path, normalized):
    """Plot the baseline-normalized white-light curve."""
    figure = _new_figure()
    ax = figure.subplots()
    ax.plot(normalized, 'k.', markersize=2, alpha=0.5)
    ax.set_xlabel('Integration Number')
    ax.set_ylabel('Normalized White Flux')
    ax.set_title('Normalized White-light Curve')
    _save_figure(figure, path, bbox_inches=None)


def _plot_flux_image(path, wave, image, time_edges, bin_size):
    """Plot a normalized flux image with wavelength and integration bin edges."""
    edges = np.empty(wave.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
    edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
    figure = _new_figure()
    ax = figure.subplots()
    mesh = ax.pcolormesh(time_edges, edges, image.T,
        shading='auto', vmin=0.98, vmax=1.02, rasterized=True)
    ax.set_xlabel('Integration Number')
    ax.set_ylabel('Wavelength (µm)')
    title = 'Normalized Flux Image'
    if bin_size > 1:
        title += f' (mean of up to {bin_size} integrations per bin)'
    ax.set_title(title)
    figure.colorbar(mesh, ax=ax, label='Relative Flux')
    _save_figure(figure, path, bbox_inches=None)


def _plot_spectral_diagnostics(paths, spectral_products, baseline_ints,
        mode='NIRISS/SOSS', detector=''):
    """Plot normalized white flux and the spectral flux image."""
    wave, flux = _diagnostic_spectrum(spectral_products, mode=mode, detector=detector)
    if wave is None:
        return {}
    white = np.nansum(flux, axis=1)
    bounds = np.atleast_1d(baseline_ints).astype(int)
    high = None if bounds.size == 1 else bounds[1]
    written = _plot_normalized_white(paths, white, bounds[0], high)
    if wave.size >= 2:
        image, edges, bin_size = _binned_flux_image(flux, np.nanmedian(flux, axis=0))
        _plot_flux_image(paths['flux_plot'], wave, image, edges, bin_size)
        written['flux_plot'] = str(paths['flux_plot'])
    return written


def _bounded_spectral_summary(spectral_products, mode, detector, *,
        max_image_rows=1200, max_bytes=32 << 20):
    """Calculate white flux and a normalized flux image in bounded chunks."""
    if not spectral_products or 1 not in spectral_products:
        return None
    order1 = spectral_products[1]
    wave1 = np.asarray(order1['wave'], dtype=float)
    flux1 = order1['flux']
    if (len(flux1.shape) != 2 or wave1.ndim != 1 or
            flux1.shape[-1] != wave1.size or flux1.shape[0] < 1):
        return None
    nints = int(flux1.shape[0])
    selections = [(1, np.arange(wave1.size))]
    wave_by_order = {1: wave1}
    order2 = spectral_products.get(2)
    if order2 is not None:
        wave2 = np.asarray(order2['wave'], dtype=float)
        flux2 = order2['flux']
        use2 = np.flatnonzero(np.isfinite(wave2) & (wave2 <= .85))
        use1 = np.flatnonzero(np.isfinite(wave1) & (wave1 > .85))
        if (wave2.ndim == 1 and len(flux2.shape) == 2 and flux2.shape == (nints, wave2.size) and
                use1.size and use2.size):
            selections = [(2, use2), (1, use1)]
            wave_by_order[2] = wave2
    bounds = _diagnostic_wave_bounds(mode, detector)
    waves, members, medians = [], [], []
    col_step = max(1, int(max_bytes) // max(1, nints * 8))
    for order, selection in selections:
        wave = wave_by_order[order]
        selection = selection[np.isfinite(wave[selection]) & (wave[selection] >= bounds[0]) &
            (wave[selection] <= bounds[1])]
        for start in range(0, selection.size, col_step):
            columns = selection[start:start + col_step]
            values = np.asarray(spectral_products[order]['flux'][:, columns], dtype=float)
            median = np.nanmedian(values, axis=0)
            keep = np.isfinite(median) & (median != 0)
            waves.extend(wave[columns[keep]])
            members.extend((order, int(column)) for column in columns[keep])
            medians.extend(median[keep])
    if not waves:
        return None
    permutation = np.argsort(waves, kind='stable')
    wave = np.asarray(waves)[permutation]
    members = np.asarray(members, dtype=int)[permutation]
    medians = np.asarray(medians)[permutation]
    _, unique = np.unique(np.round(wave, 12), return_index=True)
    unique.sort()
    wave, members, medians = wave[unique], members[unique], medians[unique]
    width = wave.size
    bin_size = max(1, int(np.ceil(nints / max(1, int(max_image_rows)))))
    edges = np.arange(0, nints, bin_size, dtype=int)
    edges = np.append(edges, nints)
    image = np.empty((edges.size - 1, width), dtype=float)
    white = np.empty(nints, dtype=float)
    row_step = max(1, int(max_bytes) // max(1, width * 8))
    by_order = [(order, np.flatnonzero(members[:, 0] == order))
        for order in np.unique(members[:, 0])]
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        sums = np.zeros(width)
        counts = np.zeros(width, dtype=np.int64)
        for start in range(int(low), int(high), row_step):
            end = min(int(high), start + row_step)
            values = np.empty((end - start, width), dtype=float)
            for order, positions in by_order:
                values[:, positions] = np.asarray(spectral_products[order]['flux'][
                    start:end, members[positions, 1]], dtype=float)
            white[start:end] = np.nansum(values, axis=1)
            sums += np.nansum(values, axis=0)
            counts += np.sum(~np.isnan(values), axis=0)
        image[index] = np.divide(sums, counts, out=np.full(width, np.nan), where=counts != 0)
    image /= medians
    return {'wave': wave, 'white': white, 'image': image,
        'time_edges': edges, 'column_median': medians, 'bin_size': bin_size}


def _plot_optimal_spectral_diagnostics(paths, spectral_products, baseline_ints,
        mode='NIRISS/SOSS', detector=''):
    """Plot white flux and a flux image from bounded spectral summaries."""
    summary = _bounded_spectral_summary(spectral_products, mode, detector)
    if summary is None:
        return {}
    wave, white = summary['wave'], summary['white']
    bounds = np.atleast_1d(baseline_ints).astype(int)
    high = bounds[1] if bounds.size > 1 else None
    written = _plot_normalized_white(paths, white, bounds[0], high)
    if wave.size >= 2:
        _plot_flux_image(paths['flux_plot'], wave, summary['image'], summary['time_edges'],
            summary['bin_size'])
        written['flux_plot'] = str(paths['flux_plot'])
    return written


def _plot_normalized_white(paths, white, low, high):
    """Normalize and plot white flux using the selected baseline bounds."""
    baseline = np.nanmedian(white[:low])
    if high is not None:
        baseline = .5 * (baseline + np.nanmedian(white[high:]))
    normalized = (white / baseline if np.isfinite(baseline) and baseline != 0
        else np.full_like(white, np.nan, dtype=float))
    _plot_white_flux(paths['white_plot'], normalized)
    return {'white_plot': str(paths['white_plot'])}


def _scatter_wavelength_axis(spectral_products):
    """Return the wavelength axis used by the optimizer's final scatter."""
    if not spectral_products or 1 not in spectral_products:
        return None
    wave1 = np.asarray(spectral_products[1].get('wave'), dtype=float)
    if wave1.ndim != 1:
        return None
    order2 = spectral_products.get(2)
    if order2 is None:
        return wave1
    wave2 = np.asarray(order2.get('wave'), dtype=float)
    if wave2.ndim != 1:
        return None
    i2 = np.flatnonzero(wave2 <= 0.85)
    i1 = np.flatnonzero(wave1 > 0.85)
    if not i2.size or not i1.size:
        return None
    wave = np.concatenate((wave2[:i2[-1] + 1], wave1[i1[0]:]))
    return wave[np.argsort(wave)]


def _per_order_scatter(flux, baseline_ints, *, max_bytes=32 << 20):
    """Return v1's per-wavelength P2P statistic for one spectral order."""
    import warnings
    if not hasattr(flux, 'shape'):
        flux = np.asarray(flux)
    if len(flux.shape) != 2 or flux.shape[0] < 3:
        return None
    cols = max(1, int(max_bytes) // max(1, flux.shape[0] * 8))
    if flux.shape[1] > cols:
        # Process wavelength chunks while retaining every integration.
        result = np.empty(flux.shape[1], dtype=float)
        for start in range(0, flux.shape[1], cols):
            end = min(flux.shape[1], start + cols)
            result[start:end] = _per_order_scatter(
                flux[:, start:end], baseline_ints, max_bytes=max_bytes)
        return result
    flux = np.asarray(flux, dtype=float)
    median = np.nanmedian(flux, axis=0, keepdims=True)
    normalized = np.divide(flux, median, out=np.full_like(flux, np.nan), where=median != 0)
    difference = (0.5 * (normalized[:-2] + normalized[2:]) - normalized[1:-1])
    # Suppress all-NaN slice warnings for empty detector columns.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered',
            category=RuntimeWarning)
        if baseline_ints is None:
            return np.nanmedian(np.abs(difference), axis=0)
        bounds = tuple(int(value) for value in np.atleast_1d(baseline_ints))
        if len(bounds) == 1:
            return np.nanmedian(np.abs(difference[:bounds[0]]), axis=0)
        if len(bounds) != 2:
            raise ValueError('baseline_ints must contain one or two values')
        return 0.5 * (np.nanmedian(np.abs(difference[:bounds[0]]), axis=0) +
            np.nanmedian(np.abs(difference[bounds[1]:]), axis=0))


def _plot_scatter_diagnostic(path, spectral_products, scatter, *, smooth=10,
        wave_range=None, ylim=None, baseline_ints=None, tol=0.05):
    """Render the winning per-wavelength P2P scatter in v1 units."""
    # Calculate per-order scatter from the final spectra for the two SOSS panels.
    if spectral_products and 1 in spectral_products and 2 in spectral_products:
        from scipy.ndimage import uniform_filter1d
        figure = _new_figure(figsize=(16., 4.))
        axes = np.atleast_1d(figure.subplots(1, 2))
        rendered = False
        for index, order in enumerate((2, 1)):
            product = spectral_products[order]
            wave_order = np.asarray(product.get('wave'), dtype=float)
            order_scatter = _per_order_scatter(product.get('flux'), baseline_ints)
            if (wave_order.ndim != 1 or order_scatter is None or
                    order_scatter.shape != wave_order.shape):
                continue
            permutation = np.argsort(wave_order, kind='mergesort')
            wave_sorted = wave_order[permutation]
            raw = np.asarray(order_scatter)[permutation] * 1e6
            keep = np.isfinite(wave_sorted) & np.isfinite(raw)
            if wave_range is not None:
                if (not isinstance(wave_range, (list, tuple)) or len(wave_range) != 2):
                    raise ValueError('wave_range_plot must be a length-2 sequence')
                low, high = wave_range
                if low is not None:
                    keep &= wave_sorted >= float(low) - float(tol)
                if high is not None:
                    keep &= wave_sorted <= float(high) + float(tol)
            if not keep.any():
                continue
            ax = axes[index]
            ax.plot(wave_sorted[keep], raw[keep], linewidth=.6, linestyle='-',
                alpha=.5, color='grey', label='Best config (raw)' if index == 0 else None)
            window = min(max(int(smooth), 1), raw.size)
            if window > 1:
                # Replace NaNs with zeros before smoothing as in v1.
                smoothed = uniform_filter1d(np.nan_to_num(raw, nan=0.), size=window, mode='nearest')
                ax.plot(wave_sorted[keep], smoothed[keep], linewidth=1.2, linestyle='-',
                    label=(f'Best config (smoothed, window={window})' if index == 0 else None))
            ax.set_xlabel('Wavelength [μm]', fontsize=11)
            ax.set_ylabel('Scatter [ppm]', fontsize=11)
            ax.set_title(f'Order {order}', fontsize=12)
            if ylim is not None:
                if not isinstance(ylim, (list, tuple)) or len(ylim) != 2:
                    raise ValueError('ylim_plot must be a length-2 sequence')
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=.3)
            if index == 0:
                ax.legend(fontsize=8)
            rendered = True
        if rendered:
            figure.tight_layout()
            _save_figure(figure, path, dpi=150)
            return True
        figure.clear()
    wave = _scatter_wavelength_axis(spectral_products)
    if wave is None or scatter is None:
        return False
    scatter = np.asarray(scatter, dtype=float).ravel()
    if scatter.size != wave.size:
        raise ValueError(f'final scatter length {scatter.size} does not match its '
            f'wavelength axis {wave.size}')
    keep = np.isfinite(wave) & np.isfinite(scatter)
    if wave_range is not None:
        if not isinstance(wave_range, (list, tuple)) or len(wave_range) != 2:
            raise ValueError('wave_range_plot must be a length-2 sequence')
        low, high = wave_range
        if low is not None:
            keep &= wave >= float(low)
        if high is not None:
            keep &= wave <= float(high)
    if not keep.any():
        return False
    scatter_ppm = scatter * 1e6
    figure = _new_figure(figsize=(8., 4.))
    ax = figure.subplots()
    ax.plot(wave[keep], scatter_ppm[keep], color='grey', alpha=0.5,
        linewidth=0.6, label='Best configuration (raw)')
    window = min(max(int(smooth), 1), scatter.size)
    if window > 1:
        kernel = np.ones(window, dtype=float)
        finite = np.isfinite(scatter_ppm)
        numerator = np.convolve(np.where(finite, scatter_ppm, 0.), kernel, mode='same')
        denominator = np.convolve(finite.astype(float), kernel, mode='same')
        smoothed = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
            where=denominator > 0)
        smooth_keep = keep & np.isfinite(smoothed)
        ax.plot(wave[smooth_keep], smoothed[smooth_keep], linewidth=1.,
            label=f'Best configuration (smoothed: {window})')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Scatter (ppm)')
    ax.set_title('Point-to-point Spectral Scatter')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small')
    if ylim is not None:
        if not isinstance(ylim, (list, tuple)) or len(ylim) != 2:
            raise ValueError('ylim_plot must be a length-2 sequence')
        ax.set_ylim(*ylim)
    _save_figure(figure, path)
    return True


def _v1_width_header(width, extract_method, state):
    """Format the spectral WIDTH header as in v1."""
    from exotedrf.v2 import config as v2config
    if extract_method == 'optimal':
        return 'N/A'
    if isinstance(width, str) and width == 'optimize':
        selected = (state.aux.get('extract_width_selected') or {}).get(1)
        return '' if selected is None else int(selected)
    if v2config.is_asymmetric_width(width):
        return v2config.format_extract_width(width)
    width_array = np.asarray(width)
    if width_array.ndim == 0:
        return width_array.item()
    return format_log_value(width)


def _plot_width(width, selected=None):
    """Return the full aperture width for centroid plotting, or None."""
    from exotedrf.v2 import config as v2config
    if isinstance(width, str) and width == 'optimize':
        width = selected
    if width is None or v2config.is_asymmetric_width(width):
        return None
    return float(np.asarray(width).reshape(()))


def _final_centroids(state, ctx, deepframe):
    """Select or calculate the final spectral trace centroids."""
    fixed = ctx.get('centroids')
    if fixed is not None:
        return fixed
    cached = state.aux.get('centroids_extract')
    if cached is not None:
        return cached
    if ctx.get('custom_deepframe') is not None:
        deepframe = ctx['custom_deepframe']
    if deepframe is None:
        return (state.aux.get('centroids_int') or state.aux.get('centroids_group'))
    instrument = str(getattr(state.cube.meta, 'mode', '')).split('/')[0].upper()
    if instrument == 'NIRSPEC':
        from exotedrf.v2 import stages as v2stages
        return v2stages._trace_nirspec_deepframe(deepframe, ctx)
    if instrument == 'MIRI':
        from exotedrf.v2 import stages as v2stages
        return v2stages._trace_miri_deepframe(deepframe, ctx, stage3=True)
    from exotedrf.v2 import trace
    return trace.get_soss_centroids_with_group_fallback(
        deepframe, tracetable=ctx.get('centroid_tracetable'), subarray=state.cube.meta.subarray)


def _plot_centroids(path, state, params, ctx):
    """Plot spectral traces and extraction apertures over the deep frame."""
    deepframe = ctx.get('custom_deepframe')
    if deepframe is None:
        deepframe = state.aux.get('stage3_deepframe')
    if deepframe is None:
        deepframe = state.aux.get('deepframe')
    if deepframe is None:
        return False
    deepframe = np.asarray(deepframe, dtype=float)
    if deepframe.ndim != 2 or not deepframe.size:
        return False
    centroids = _final_centroids(state, ctx, deepframe)
    if not centroids:
        return False
    from exotedrf import plotting as v1_plotting
    default_x = np.arange(deepframe.shape[-1])
    xpos = np.asarray(centroids.get('xpos', default_x), dtype=float)
    selected = state.aux.get('extract_width_selected') or {}
    width1 = _plot_width(params.get('extract_width', 0.), selected.get(1))
    width2 = ctx.get('opts', {}).get('extract_width_soss2')
    width2 = width1 if width2 is None else _plot_width(width2, selected.get(2))
    traces = []
    order_keys = ['ypos o1', 'ypos o2', 'ypos o3']
    if 'ypos o1' not in centroids and 'ypos' in centroids:
        order_keys = ['ypos']
    for key in order_keys:
        y = centroids.get(key)
        if y is None:
            continue
        y = np.asarray(y, dtype=float)
        x = xpos[:y.size] if xpos.size >= y.size else default_x[:y.size]
        keep = np.isfinite(x) & np.isfinite(y)
        if not keep.any():
            continue
        traces.append((x[keep], y[keep]))
    if not traces:
        return False
    instrument = str(getattr(state.cube.meta, 'mode', '')).split('/')[0].upper() or 'NIRISS'
    if order_keys == ['ypos']:
        # Pass the single trace as an (xpos, ypos) pair.
        v1_plotting.make_centroiding_plot(
            deepframe, traces[0], instrument, outfile=path, show_plot=False,
            miri_scale=(instrument == 'MIRI'), extract_width=width1, extract_width_soss2=width2)
        return True
    v1_plotting.make_centroiding_plot(deepframe, traces, instrument, outfile=path, show_plot=False,
        extract_width=width1, extract_width_soss2=width2)
    return True


def write_diagnostic_plots(state, params, ctx, paths, trials, *, scatter=None):
    """Write the final cost, spectrum, scatter, and requested trace diagnostics.

    Parameters
    ----------
    state : State
        Final calibrated pipeline state.
    params : dict
        Selected calibration and extraction parameters.
    ctx : dict
        Pipeline context and options.
    paths : dict
        Output paths returned by output_layout.
    trials : list
        Optimizer trial records.
    scatter : None, array-like(float)
        Final per-wavelength scatter.

    Returns
    -------
    written : dict
        Names and paths of the diagnostic plots written.
    """
    opts = ctx.get('opts', {})
    if opts.get('output_mode') == 'optimal' and not opts.get('do_plots', True):
        return {}
    with _concurrent_saves():
        return _write_diagnostic_plots(state, params, ctx, paths, trials, scatter=scatter)


def _write_diagnostic_plots(state, params, ctx, paths, trials, *, scatter=None):
    """Render the final optimizer diagnostics."""
    opts = ctx.get('opts', {})
    paths = {key: Path(value) for key, value in paths.items()}
    written = {}
    if _plot_cost_trials(paths['cost_plot'], trials, params):
        written['cost_plot'] = str(paths['cost_plot'])
    meta = state.cube.meta
    mode = getattr(meta, 'mode', 'NIRISS/SOSS')
    detector = getattr(meta, 'detector', '')
    spectral_plotter = (_plot_optimal_spectral_diagnostics
        if opts.get('output_mode') == 'optimal' else _plot_spectral_diagnostics)
    written.update(spectral_plotter(paths, state.aux.get('spectral_products'),
        meta.baseline_ints, mode=mode, detector=detector))
    if _plot_scatter_diagnostic(paths['scatter_plot'], state.aux.get('spectral_products'), scatter,
            wave_range=opts.get('wave_range_plot'), ylim=opts.get('ylim_plot'),
            baseline_ints=state.cube.meta.baseline_ints):
        written['scatter_plot'] = str(paths['scatter_plot'])
    centroid_path = paths['centroid_plot']
    if str(mode).split('/')[0].upper() == 'NIRSPEC':
        centroid_path = centroid_path.with_name(f'centroiding_{str(detector).lower()}.png')
    atoca_info = state.aux.get('atoca')
    if atoca_info is not None:
        # Plot ATOCA decontamination residuals when requested.
        if opts.get('do_plots', False):
            from exotedrf.v2 import atoca
            atoca_path = paths['stage3'] / 'extract1dstep_atoca.png'
            if atoca.plot_decontamination(atoca_info, atoca_path):
                written['atoca_decontamination_plot'] = str(atoca_path)
    elif opts.get('do_plots', False) and _plot_centroids(centroid_path, state, params, ctx):
        written['centroid_plot'] = str(centroid_path)
    if opts.get('do_plots', False):
        written.update(_plot_aperture_optimization(paths, state))
    return written


def _plot_aperture_optimization(paths, state):
    """Plot aperture-width scatter with v1 make_soss_width_plot."""
    selection = state.aux.get('extract_width_optimization') or {}
    if not selection:
        return {}
    from exotedrf import plotting as v1_plotting
    instrument = str(getattr(state.cube.meta, 'mode', '')).split('/')[0].upper()
    written = {}
    for order, item in sorted(selection.items()):
        name = ('aperture_optimization_order{}.png'.format(order)
            if instrument == 'NIRISS' else 'aperture_optimization.png')
        path = Path(paths['stage3']) / name
        v1_plotting.make_soss_width_plot(np.asarray(item['widths']), np.asarray(item['scatter']),
            int(item['index']), outfile=str(path), show_plot=False)
        written[f'aperture_optimization_{order}'] = str(path)
    return written


def jsonable(value):
    """Convert NumPy arrays, paths, and dataclass fields to JSON values.

    Parameters
    ----------
    value : object
        Value to format.

    Returns
    -------
    converted : object
        Converted value with nonfinite floats represented by None.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, '__dataclass_fields__'):
        return {name: jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    return value


__all__ = ['format_log_value', 'jsonable', 'output_layout', 'write_final_products',
    'write_decision_ranking', 'write_diagnostic_plots', 'write_optimizer_logs', 'v1_output_root']
