"""Load v1 Stage-3 extraction functions with instrument stubs."""

import ast
import contextlib
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[2] / 'exotedrf'

STAGE3_FUNCTIONS = (
    '_get_dq_mask', '_mask_dq_pixels', '_load_box_extraction_cubes',
    'box_extract_miri', 'box_extract_nirspec', 'box_extract_soss',
    '_format_extract_width', '_parse_extraction_width',
    '_get_extraction_edges', '_get_aperture_pixels', 'do_box_extraction',
    'do_optimal_extraction', 'extract_optimal', 'get_spatial_prof_opt',
    'optimal_extract_miri', 'optimal_extract_nirspec',
)


def _functions(path, names):
    """Return functions."""
    tree = ast.parse(Path(path).read_text())
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    missing = set(names) - {node.name for node in nodes}
    if missing:
        raise RuntimeError(f'v1 functions not found in {path}: {missing}')
    for node in nodes:
        if node.name == 'do_optimal_extraction':
            # Bind the DQ return when v1 2.5.0 performs zero iterations.
            node.body.insert(1, ast.parse('dq = None').body[0])
        elif node.name == 'optimal_extract_miri':
            # Guard v1 2.5.0's transpose when DQ reporting is disabled.
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
                        and call.func.id == 'do_optimal_extraction':
                    keyword = next(k for k in call.keywords if k.arg == 'dq_cube')
                    keyword.value = ast.IfExp(
                        test=ast.Compare(left=ast.Name(id='dqcube', ctx=ast.Load()),
                                         ops=[ast.IsNot()],
                                         comparators=[ast.Constant(None)]),
                        body=keyword.value, orelse=ast.Constant(None))
    return nodes


def _compile(nodes, path, namespace):
    """Return compile."""
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


@contextlib.contextmanager
def _open_filetype(item):
    """Return open filetype."""
    yield item


def segment(data, err=None, dq=None):
    """A minimal in-memory 'datamodel' for v1's non-path input branch.

    Parameters
    ----------
    data : array-like(float)
        Data array.
    err : array-like(float)
        Err array.
    dq : array-like(int)
        Dq array.

    Returns
    -------
    result : types.SimpleNamespace
        Loaded functions or synthetic observation records.
    """
    samples = np.array(data)
    flags = np.zeros(samples.shape, np.uint32) if dq is None else np.array(dq)
    return SimpleNamespace(data=samples, err=None if err is None
                           else np.array(err), dq=flags)


def load(detector='nrs1', grating='G395H', subarray='SUB2048', dimx=None,
         trace_start=None):
    """Return a namespace holding the v1 functions with instrument stubs.

    Parameters
    ----------
    detector : str
        Detector name.
    grating : str
        Disperser name.
    subarray : str
        Subarray name.
    dimx : int
        Number of detector columns.
    trace_start : None, int
        First detector column covered by the trace.

    Returns
    -------
    result : types.SimpleNamespace
        Loaded functions or synthetic observation records.
    """
    utils_ns = _compile(_functions(ROOT / 'utils.py',
                                   ('get_nrs_trace_start',
                                    'get_dq_flag_metrics')),
                        ROOT / 'utils.py', {'np': np})
    get_trace_start = utils_ns['get_nrs_trace_start']
    if trace_start is not None:
        def get_trace_start(*args, **kwargs):
            return trace_start
    utils = SimpleNamespace(
        open_filetype=_open_filetype,
        get_nrs_detector_name=lambda datafile: detector,
        get_nrs_grating=lambda datafile: grating,
        get_soss_subarray=lambda datafile: subarray,
        get_nrs_trace_start=get_trace_start,
        get_dq_flag_metrics=utils_ns['get_dq_flag_metrics'],
    )
    namespace = {
        'np': np, 'os': os, 'median_filter': median_filter,
        'fancyprint': lambda *args, **kwargs: None,
        'tqdm': lambda iterable, *args, **kwargs: iterable,
        'utils': utils,
        'plotting': SimpleNamespace(
            make_soss_width_plot=lambda *args, **kwargs: None),
        'fits': SimpleNamespace(getdata=None),
    }
    _compile(_functions(ROOT / 'stage3.py', STAGE3_FUNCTIONS),
             ROOT / 'stage3.py', namespace)

    # Wavelength lookups need CRDS/datamodels; parity is on flux/errors.
    def wave_soss(datafile, *args, **kwargs):
        width = dimx if dimx is not None else datafile.data.shape[-1]
        return np.arange(width, dtype=float), np.arange(width, dtype=float)
    namespace['get_wave_soss'] = wave_soss
    namespace['get_wave_nirspec'] = lambda *args, **kwargs: None
    namespace['get_wave_miri'] = lambda *args, **kwargs: None

    helpers = _compile(
        _functions(ROOT / 'optimize_helpers.py',
                   ('_parse_width', 'do_box_extraction_nanaware')),
        ROOT / 'optimize_helpers.py',
        {'np': np, 'tqdm': lambda iterable, *a, **k: iterable})
    namespace['_parse_width'] = helpers['_parse_width']
    namespace['do_box_extraction_nanaware'] = \
        helpers['do_box_extraction_nanaware']
    optimize = _compile(
        _functions(ROOT / 'optimize.py', ('parse_extract_width_metadata',)),
        ROOT / 'optimize.py', {'np': np, 'ast': ast, 're': re})
    namespace['parse_extract_width_metadata'] = \
        optimize['parse_extract_width_metadata']
    return SimpleNamespace(**namespace)


def centroid_frame(**columns):
    """v1 reads centroids through ``DataFrame[...].values``.

    Parameters
    ----------
    columns : dict
        Centroid table columns.

    Returns
    -------
    centroids : pandas.DataFrame
        Centroid table with named columns.
    """
    return pd.DataFrame({key: np.asarray(value, dtype=float)
                         for key, value in columns.items()})
