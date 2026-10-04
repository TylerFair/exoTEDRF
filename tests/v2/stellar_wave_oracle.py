"""Load v1 Stage-3 wavelength functions with isolated dependencies."""

import ast
import os
import types
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]

UTILS_FUNCTIONS = (
    'download_stellar_spectra', 'get_stellar_param_grid',
    'interpolate_stellar_model_grid', 'sigma_clip_lightcurves',
    'get_default_header', 'save_extracted_spectra', 'verify_path',
    'convert_flux_units',
)
STAGE3_FUNCTIONS = (
    'do_ccf', 'format_soss_spectra', 'format_nirspec_spectra',
    'format_miri_spectra', 'flux_calibrate',
)


def _exec_functions(path, names, namespace):
    """Execute selected functions in the supplied namespace."""
    tree = ast.parse(Path(path).read_text())
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    missing = set(names) - {node.name for node in nodes}
    assert not missing, f'v1 functions not found in {path}: {missing}'
    for node in nodes:
        if node.name == 'do_ccf':
            # Align continuum wavelengths after v1 2.5.0 removes NaN flux.
            position = next(i for i, stmt in enumerate(node.body)
                            if isinstance(stmt, ast.Assign)
                            and isinstance(stmt.value, ast.Call)
                            and isinstance(stmt.value.func, ast.Attribute)
                            and stmt.value.func.attr == 'delete'
                            and stmt.targets[0].id == 'thismod')
            node.body.insert(position + 1, ast.parse(
                'norm_wave = np.delete(new_wave, ii)').body[0])
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
                        and call.func.id == 'continuum_normalize':
                    call.args[0] = ast.Name(id='norm_wave', ctx=ast.Load())
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


def _no_network(command):
    """Reject network access from the reference functions."""
    raise AssertionError(f'v1 attempted a download: {command}')


def load_v1(*, system=None, exists=None, quiet=True):
    """Return a namespace with the v1 functions (``.utils`` holds utils).

    Parameters
    ----------
    system : None, callable
        Replacement system-command function.
    exists : None, callable
        Replacement file-existence check.
    quiet : bool
        Quiet option.

    Returns
    -------
    result : types.SimpleNamespace
        Reference wavelength and utility functions.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import median_filter
    from numpy.polynomial import chebyshev
    from scipy.signal import butter, correlate, filtfilt
    import spectres
    from spectres.spectral_resampling import make_bins
    try:
        import pastasoss
    except ImportError:  # pragma: no cover - optional
        pastasoss = None

    path = os.path if exists is None else types.SimpleNamespace(
        exists=exists, join=os.path.join)
    fake_os = types.SimpleNamespace(path=path, mkdir=os.mkdir,
                                    system=system or _no_network,
                                    environ=os.environ)
    fancyprint = (lambda *args, **kwargs: None) if quiet else print
    utils_ns = {'np': np, 'fits': fits, 'os': fake_os, 'datetime': datetime,
                'RegularGridInterpolator': RegularGridInterpolator,
                'median_filter': median_filter, 'fancyprint': fancyprint}
    _exec_functions(ROOT / 'exotedrf/utils.py', UTILS_FUNCTIONS, utils_ns)
    utils = types.SimpleNamespace(
        **{name: utils_ns[name] for name in UTILS_FUNCTIONS})
    stage3_ns = {'np': np, 'fits': fits, 'utils': utils, 'spectres': spectres,
                 'make_bins': make_bins, 'butter': butter,
                 'filtfilt': filtfilt, 'correlate': correlate,
                 'chebyshev': chebyshev,
                 'pastasoss': pastasoss, 'fancyprint': fancyprint}
    _exec_functions(ROOT / 'exotedrf/stage3.py', STAGE3_FUNCTIONS, stage3_ns)
    return types.SimpleNamespace(
        utils=utils, **{name: stage3_ns[name] for name in STAGE3_FUNCTIONS})


# Synthetic PHOENIX grid.

def synthetic_stellar_flux(wave_um, teff, logg, met, seed=11):
    """Smooth continuum with ~300 resolved absorption lines.

    Parameters
    ----------
    wave_um : array-like(float)
        Model wavelengths in microns.
    teff : float
        Stellar effective temperature in kelvin.
    logg : float
        Stellar surface gravity in log cgs units.
    met : float
        Stellar metallicity.
    seed : int
        Random seed.

    Returns
    -------
    flux : np.ndarray(float)
        Synthetic stellar flux.
    """
    rng = np.random.default_rng(seed)
    centers = np.sort(rng.uniform(0.55, 5.4, 300))
    sigmas = rng.uniform(0.0015, 0.004, 300) * centers
    depths = rng.uniform(0.05, 0.5, 300)
    scale = (1 + 2e-4 * (teff - 5000)) * (1 + 0.05 * (logg - 4)) * \
        (1 + 0.1 * met)
    lines = np.zeros_like(wave_um)
    for center, sigma, depth in zip(centers, sigmas, depths):
        window = np.abs(wave_um - center) < 6 * sigma
        lines[window] += depth * scale * np.exp(
            -0.5 * ((wave_um[window] - center) / sigma) ** 2)
    continuum = 1e14 * (wave_um / 1.0) ** (-2 - 1e-4 * (teff - 5000))
    return continuum * np.clip(1 - lines, 0.05, None)


def write_phoenix_grid(directory, params, npoints=60000):
    """Write a PHOENIX-format wave file and the grid models for ``params``.

    Parameters
    ----------
    directory : str, pathlib.Path
        Directory for generated files.
    params : list[tuple]
        Stellar parameter triples for the model grid.
    npoints : int
        Number of model wavelength samples.

    Returns
    -------
    wave : np.ndarray(float)
        Model wavelengths in microns.
    """
    from exotedrf.v2 import wavecal
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    wave_aa = np.geomspace(5000., 55000., npoints)
    fits.PrimaryHDU(wave_aa).writeto(
        directory / wavecal.PHOENIX_WAVE_FILE, overwrite=True)
    for st_teff, st_logg, st_met in params:
        teffs, loggs, mets = wavecal.get_stellar_param_grid(
            st_teff, st_logg, st_met)
        for teff in teffs:
            for logg in loggs:
                for met in mets:
                    name, _ = wavecal.phoenix_model_name(teff, logg, met)
                    flux = synthetic_stellar_flux(wave_aa / 1e4, teff, logg,
                                                  met).astype(np.float32)
                    fits.PrimaryHDU(flux).writeto(directory / name,
                                                  overwrite=True)
    return wave_aa / 1e4


def observed_spectrum(wave, true_shift, params, nints=24, seed=3,
                      nan_fraction=0.0):
    """Time series whose stellar lines sit ``true_shift`` microns redward.

    Parameters
    ----------
    wave : array-like(float)
        Wave array.
    true_shift : float
        Injected wavelength shift in microns.
    params : tuple[float]
        Effective temperature, surface gravity and metallicity.
    nints : int
        Number of integrations.
    seed : int
        Random seed.
    nan_fraction : float
        Fraction of samples replaced with NaNs.

    Returns
    -------
    spectra : tuple
        Synthetic flux and uncertainty arrays.
    """
    rng = np.random.default_rng(seed)
    wave = np.asarray(wave, dtype=float)
    fine = np.geomspace(0.5, 5.5, 400000)
    star = synthetic_stellar_flux(fine, *params)
    finite = np.isfinite(wave)
    spectrum = np.zeros_like(wave)
    spectrum[finite] = np.interp(wave[finite] + true_shift, fine, star)
    spectrum = 5e3 * spectrum / np.median(spectrum[finite])
    flux = spectrum[None, :] * (1 + 1e-3 * rng.standard_normal((nints, 1)))
    flux = flux + rng.normal(0, 3, flux.shape)
    flux[:, ~finite] = 0.
    if nan_fraction:
        flux[rng.random(flux.shape) < nan_fraction] = np.nan
    ferr = np.full_like(flux, 3.)
    return flux, ferr
