"""Create synthetic v1 products for optimizer restart tests."""

import ast
import glob
import os
import re
from pathlib import Path

import numpy as np
from astropy.io import fits

DIMY, DIMX = 96, 128
Y_O1, Y_O2 = 20., 40.
SEGMENTS = (64, 56)
ROOT = 'jwsynth_04102_00001'
REPO = Path(__file__).resolve().parents[2]


def soss_header(nints_total, int_start, nints, segment, filename):
    """Create a FITS header for a synthetic SOSS segment.

    Parameters
    ----------
    nints_total : int
        Total number of integrations in the exposure.
    int_start : int
        First integration index.
    nints : int
        Number of integrations.
    segment : int
        Segment number.
    filename : str
        Segment filename.

    Returns
    -------
    header : astropy.io.fits.Header
        Synthetic segment header.
    """
    header = fits.Header()
    header['INSTRUME'] = 'NIRISS'
    header['DETECTOR'] = 'NIS'
    header['SUBARRAY'] = 'SUBSTRIP96'
    header['EXP_TYPE'] = 'NIS_SOSS'
    header['FILTER'] = 'CLEAR'
    header['PUPIL'] = 'GR700XD'
    header['TARGNAME'] = 'SYNTH-1'
    header['TGROUP'] = 5.494
    header['TFRAME'] = 5.494
    header['NFRAMES'] = 1
    header['GROUPGAP'] = 0
    header['NGROUPS'] = 2
    header['NINTS'] = nints_total
    header['INTSTART'] = int_start
    header['INTEND'] = int_start + nints - 1
    header['EXSEGNUM'] = segment
    header['FILENAME'] = filename
    return header


def synthetic_rates(seed=3, nints=sum(SEGMENTS)):
    """A two-order SOSS-like rate cube with a transit and a few outliers.

    Parameters
    ----------
    seed : int
        Random seed.
    nints : int
        Number of integrations.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    rng = np.random.RandomState(seed)
    yy = np.arange(DIMY, dtype=float)[:, None]
    profile = (600. * np.exp(-0.5 * ((yy - Y_O1) / 3.) ** 2) +
               250. * np.exp(-0.5 * ((yy - Y_O2) / 3.) ** 2) + 20.)
    shape = 1. + 0.25 * np.sin(np.arange(DIMX) / 13.)
    depth = np.ones(nints)
    depth[50:70] = 0.99
    trend = 1. + 1e-4 * np.arange(nints)
    sci = (profile * shape)[None] * (depth * trend)[:, None, None]
    sci = sci + rng.normal(0., 2., sci.shape)
    sci[7, 30, 40] += 5000.
    err = np.sqrt(np.abs(sci)) + 1.
    dq = np.zeros(sci.shape, np.uint32)
    dq[:, 5, 7] = 1
    dq[3, 50, 60] = np.uint32(1 << 31)
    return (sci.astype(np.float32), err.astype(np.float32), dq)


def write_rate_products(directory, tag, cube=None, *, segments=SEGMENTS,
                        root=ROOT, suffix='nis', wavelength=None):
    """Write one v1 CubeModel product per segment; return their paths.

    Parameters
    ----------
    directory : str, pathlib.Path
        Directory for generated files.
    tag : str
        Product filename tag.
    cube : array-like(float)
        Input observation cube.
    segments : tuple[int]
        Integration counts for each segment.
    root : str
        Product filename prefix.
    suffix : str
        Instrument filename suffix.
    wavelength : array-like(float)
        Wavelength array.

    Returns
    -------
    paths : list[str]
        Written segment paths.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sci, err, dq = synthetic_rates() if cube is None else cube
    total = sum(segments)
    paths = []
    start = 0
    for index, nints in enumerate(segments, start=1):
        name = f'{root}-seg{index:03d}_{suffix}_{tag}.fits'
        header = soss_header(total, start + 1, nints, index, name)
        sl = slice(start, start + nints)
        times = fits.BinTableHDU.from_columns(fits.ColDefs([fits.Column(
            name='int_mid_BJD_TDB', format='D',
            array=60000. + (start + np.arange(nints)) * 1e-3)]),
            name='INT_TIMES')
        hdus = [fits.PrimaryHDU(header=header),
                fits.ImageHDU(sci[sl], name='SCI'),
                fits.ImageHDU(err[sl], name='ERR'),
                fits.ImageHDU(dq[sl], name='DQ'), times]
        if wavelength is not None:
            hdus.append(fits.ImageHDU(wavelength, name='WAVELENGTH'))
        path = directory / name
        fits.HDUList(hdus).writeto(path)
        paths.append(str(path))
        start += nints
    return paths


def soss_refpack():
    """Return reference arrays for restarted Stage-2 and Stage-3 graphs.

    Returns
    -------
    references : dict
        Flat, wavelength and gain references.
    """
    return {
        'flat': np.ones((DIMY, DIMX), np.float32),
        'wave_o1': np.linspace(2.75, 0.87, DIMX),
        'wave_o2': np.linspace(1.4, 0.6, DIMX),
        'gain_factor': np.float32(1.),
    }


def write_centroids(path):
    """Write a CSV table of synthetic SOSS trace coordinates.

    Parameters
    ----------
    path : str, pathlib.Path
        Output file path.

    Returns
    -------
    path : str
        Written centroid-table path.
    """
    from exotedrf.v2 import trace
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    trace.save_centroids_csv(str(path), {
        'xpos': np.arange(DIMX, dtype=float),
        'ypos o1': np.full(DIMX, Y_O1),
        'ypos o2': np.full(DIMX, Y_O2)})
    return str(path)


def write_deepframe(path, cube=None):
    """Write the median integration image to a FITS file.

    Parameters
    ----------
    path : str, pathlib.Path
        Output file path.
    cube : array-like(float)
        Input observation cube.

    Returns
    -------
    path : str
        Written median-image path.
    """
    sci = synthetic_rates()[0] if cube is None else cube
    fits.PrimaryHDU(np.median(sci, axis=0)).writeto(path)
    return str(path)


def base_cfg(**updates):
    """A SOSS restart configuration with every upstream step skipped.

    Parameters
    ----------
    updates : dict
        Overrides for the returned configuration.

    Returns
    -------
    config : dict
        Restart configuration with overrides applied.
    """
    cfg = {
        'observing_mode': 'NIRISS/SOSS', 'filter_detector': 'CLEAR',
        'baseline_ints': [50, -50], 'w1': 0.0, 'w2': 1.0,
        'oof_method': 'scale-achromatic', 'extract_method': 'box',
        'saturation_rescue': True, 'generate_lc': True,
        'pca_components': 10, 'do_plots': False, 'name_tag': 'synth',
        'soss_background_file': None,
    }
    for name in ('DQInitStep', 'INLCorrStep', 'SuperBiasStep', 'RefPixStep',
                 'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep',
                 'JumpStep', 'RampFitStep', 'GainScaleStep', 'AssignWCSStep',
                 'SourceTypeStep', 'FlatFieldStep', 'BackgroundStep',
                 'OneOverFStep_int', 'BadPixStep', 'PCAReconstructStep'):
        cfg[name] = 'skip'
    cfg.update(updates)
    return cfg


def v1_optimize_functions(*names, **globals_):
    """Execute selected functions from v1 ``exotedrf/optimize.py``.

    Parameters
    ----------
    names : list[str]
        Names of functions or arrays to select.
    globals_ : dict
        Additional globals for the loaded v1 functions.

    Returns
    -------
    namespace : dict
        Loaded v1 functions and their globals.
    """
    import pandas as pd
    source = (REPO / 'exotedrf' / 'optimize.py').read_text()
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in names]
    missing = set(names) - {node.name for node in nodes}
    assert not missing, missing
    namespace = {'np': np, 'pd': pd, 're': re, 'ast': ast, 'os': os,
                 'glob': glob, 'fits': fits,
                 'fancyprint': lambda *args, **kwargs: None}
    namespace.update(globals_)
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 str(REPO / 'exotedrf' / 'optimize.py'), 'exec'), namespace)
    return namespace
