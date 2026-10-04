"""Refine extracted wavelength solutions with stellar models or PASTASOSS."""

from __future__ import annotations

import os
import threading
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

PHOENIX_URL = 'ftp://phoenix.astro.physik.uni-goettingen.de/'
PHOENIX_WAVE_FILE = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
PHOENIX_GRID = 'PHOENIX-ACES-AGSS-COND-2011'
PHOENIX_SUFFIX = '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
SOSS_CCF_OVERSAMPLE = 5
NIRSPEC_CCF_OVERSAMPLE = 1
_FETCH_TIMEOUT_S = 120


def stellar_params(opts):
    """Return ``(st_teff, st_logg, st_met)`` from fixed options."""
    opts = opts or {}
    return opts.get('st_teff'), opts.get('st_logg'), opts.get('st_met')


def stellar_refinement_enabled(opts):
    """Check whether all three stellar parameters are specified."""
    return None not in stellar_params(opts)


def validate_stellar_options(cfg):
    """Check the stellar-parameter YAML keys the way v1 consumes them.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.

    Returns
    -------
    enabled : bool
        Whether all stellar parameters are supplied and wavelength refinement will run.
    """
    values = {name: cfg.get(name) for name in ('st_teff', 'st_logg', 'st_met')}
    for name, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)) or \
                not np.isfinite(value):
            raise ValueError(f'{name} must be a finite number or None, got {value!r}')
    given = [name for name, value in values.items() if value is not None]
    if given and len(given) < 3:
        warnings.warn(f'only {given} of st_teff/st_logg/st_met are set; as in v1 the '
            'default wavelength solution is used without stellar-model '
            'refinement', RuntimeWarning, stacklevel=3)
        return False
    if not given:
        return False
    teff, logg = float(values['st_teff']), float(values['st_logg'])
    if teff <= 0 or logg < 0:
        raise ValueError('st_teff must be positive and st_logg non-negative')
    model_dir = cfg.get('v2_stellar_model_dir')
    if model_dir is not None and not isinstance(model_dir, (str, os.PathLike)):
        raise ValueError('v2_stellar_model_dir must be a directory path')
    return True


def stage3_extract_kwargs(cfg):
    """Read the Extract1dStep configuration from Stage 3 options."""
    stage3 = cfg.get('stage3_kwargs') or {}
    if not isinstance(stage3, dict):
        return {}
    extract = stage3.get('Extract1dStep') or {}
    return extract if isinstance(extract, dict) else {}


def use_pastasoss_option(cfg):
    """Resolve v1's ``Extract1DStep.run(use_pastasoss=...)`` flag."""
    value = stage3_extract_kwargs(cfg).get('use_pastasoss', False)
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f'stage3_kwargs.Extract1dStep.use_pastasoss must be a boolean, '
            f'got {value!r}')
    return bool(value)


def highpass_filter(signal, order=3, freq=0.05):
    """Apply the high-pass filter used by v1 do_ccf.

    Parameters
    ----------
    signal : array-like(float)
        Input spectrum to filter.
    order : int
        Butterworth filter order.
    freq : float
        Cutoff frequency as a fraction of the Nyquist frequency.

    Returns
    -------
    filtered : np.ndarray(float)
        High-pass filtered spectrum.
    """
    from scipy.signal import butter, filtfilt
    b, a = butter(order, freq, btype='high')
    return filtfilt(b, a, signal)


def oversample_wave(wave, oversample):
    """Interpolate evenly spaced samples between adjacent wavelengths.

    Parameters
    ----------
    wave : array-like(float)
        Wavelength samples in microns.
    oversample : int
        Number of interpolated samples per wavelength interval.

    Returns
    -------
    wave : np.ndarray(float)
        Oversampled wavelength axis retaining the input endpoints.
    """
    wave = np.asarray(wave)
    if wave.shape[0] < 2:
        return wave.copy()
    step = (wave[1:] - wave[:-1]) / oversample
    sub = np.arange(1, oversample).astype(step.dtype)
    grid = np.empty((wave.shape[0] - 1, oversample), dtype=step.dtype)
    grid[:, 0] = wave[:-1]
    grid[:, 1:] = wave[:-1, None] + sub[None, :] * step[:, None]
    return np.concatenate([grid.ravel(), wave[-1:].astype(step.dtype)])


def continuum_normalize(wave, flux):
    """Normalize a spectrum with an iteratively clipped Chebyshev continuum.

    Parameters
    ----------
    wave, flux : array-like(float)
        Wavelength samples in microns and their flux, respectively.

    Returns
    -------
    normalized : np.ndarray(float)
        Flux divided by the fitted continuum.
    """
    from numpy.polynomial import chebyshev
    mask = np.isfinite(flux)
    for _ in range(10):
        coeffs = chebyshev.chebfit(wave[mask], flux[mask], 4)
        continuum = chebyshev.chebval(wave, coeffs)
        resid = flux - continuum
        sigma = np.std(resid[mask])
        mask = (resid > -1.5 * sigma) & (resid < 3 * sigma) & np.isfinite(flux)
    coeffs = chebyshev.chebfit(wave[mask], flux[mask], 4)
    continuum = chebyshev.chebval(wave, coeffs)
    return flux / continuum


def do_ccf(wave, flux, mod_flux, oversample=5):
    """Measure the wavelength shift between extracted and model stellar spectra.

    Matches v1 do_ccf, including continuum normalization and the sign of the wavelength offset.

    Parameters
    ----------
    wave, flux, mod_flux : array-like(float)
        Wavelength samples in microns, observed flux, and model flux, respectively.
    oversample : int
        Number of interpolated samples per wavelength interval.

    Returns
    -------
    shift_wave, shift_steps : float
        Measured offset in microns and oversampled grid steps, respectively.
    """
    from scipy.signal import correlate
    wave = np.asarray(wave)
    ii = np.argsort(wave)
    thiswave = wave[ii]
    thisflux = np.asarray(flux)[ii]
    thismod = np.asarray(mod_flux)[ii]
    if oversample != 1:
        new_wave = oversample_wave(thiswave, oversample)
        thisflux = np.interp(new_wave, thiswave, thisflux)
        thismod = np.interp(new_wave, thiswave, thismod)
    else:
        new_wave = thiswave
    # Remove NaN data samples and retain the complete grid for the median wavelength step.
    keep = ~np.isnan(thisflux)
    thisflux = thisflux[keep]
    thismod = thismod[keep]
    norm_wave = np.asarray(new_wave)[keep]
    # Normalize each spectrum by its maximum and fitted continuum.
    thisflux = continuum_normalize(norm_wave, thisflux / np.nanmax(thisflux))
    thismod = continuum_normalize(norm_wave, thismod / np.nanmax(thismod))
    ccf = correlate(highpass_filter(thisflux), highpass_filter(thismod))
    ll = len(thisflux)
    steps = np.linspace(-ll + 1, ll - 1, 2 * ll - 1)
    shift_steps = steps[np.argmax(ccf)]
    return (-1 * shift_steps * np.median(np.diff(new_wave)), shift_steps)


def get_stellar_param_grid(st_teff, st_logg, st_met):
    """Find neighboring PHOENIX grid points for the stellar parameters.

    Parameters
    ----------
    st_teff, st_logg, st_met : float
        Stellar effective temperature (K), log surface gravity, and metallicity.

    Returns
    -------
    teffs : list[int]
        Neighboring temperature grid points.
    loggs, mets : list[float]
        Neighboring log-gravity and metallicity grid points, respectively.
    """
    teff_lw = int(np.floor(st_teff / 100) * 100)
    teff_up = int(np.ceil(st_teff / 100) * 100)
    teffs = [teff_lw] if teff_lw == teff_up else [teff_lw, teff_up]
    logg_lw = np.floor(st_logg / 0.5) * 0.5
    logg_up = np.ceil(st_logg / 0.5) * 0.5
    loggs = [logg_lw] if logg_lw == logg_up else [logg_lw, logg_up]
    met_lw, met_up = np.floor(st_met), np.ceil(st_met)
    # Use positive zero for the upper metallicity grid point.
    if -1 < st_met < 0:
        met_up = 0.0
    mets = [met_lw] if met_lw == met_up else [met_lw, met_up]
    return teffs, loggs, mets


def phoenix_model_name(teff, logg, met):
    """Get the PHOENIX filename and metallicity directory for a grid point.

    Parameters
    ----------
    teff, logg, met : float
        Stellar effective temperature (K), log surface gravity, and metallicity.

    Returns
    -------
    filename : str
        PHOENIX spectrum filename.
    directory : str
        Metallicity directory within the archive.
    """
    # Normalize negative zero in PHOENIX filenames.
    met = float(met) + 0.0
    tstr = f'lte{int(teff):05d}-{float(logg)}0'
    if met > 0:
        return f'{tstr}+{met}{PHOENIX_SUFFIX}', f'Z+{met}'
    if met == 0:
        return f'{tstr}-{met}{PHOENIX_SUFFIX}', f'Z-{met}'
    return f'{tstr}{met}{PHOENIX_SUFFIX}', f'Z{met}'


def _looks_like_fits(path):
    """Check the FITS signature and block size of a cached file."""
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as handle:
            return (size > 0 and size % 2880 == 0 and handle.read(9) == b'SIMPLE  =')
    except OSError:
        return False


def _urlretrieve(url, destination):
    """Download a stellar model in bounded blocks."""
    with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_S) as response, \
            open(destination, 'wb') as handle:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)


def _ensure_file(path, url, fetch):
    """Download ``url`` to ``path`` atomically unless a valid copy exists."""
    if os.path.exists(path) and _looks_like_fits(path):
        return path
    directory = os.path.dirname(os.path.abspath(path))
    # Download to a private partial file while retaining the process umask.
    temporary = os.path.join(directory, f'.{os.path.basename(path)}.{os.getpid()}.'
                   f'{threading.get_ident()}.part')
    try:
        try:
            fetch(url, temporary)
        except Exception as exc:
            raise RuntimeError(f'could not download PHOENIX model {url}: {exc}. Without '
                'network access, copy the PHOENIX files into the stellar '
                'model directory (v2_stellar_model_dir) beforehand.') from exc
        if not _looks_like_fits(temporary):
            raise RuntimeError(f'PHOENIX download {url} did not return a FITS file; check '
                'that the stellar parameters lie on the PHOENIX grid')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def phoenix_model_paths(st_teff, st_logg, st_met, outdir, base_url=PHOENIX_URL):
    """Get local paths and archive URLs for PHOENIX model files.

    Parameters
    ----------
    st_teff, st_logg, st_met : float
        Stellar effective temperature (K), log surface gravity, and metallicity.
    outdir : str
        Directory for calibration or stellar model files.
    base_url : str
        Root URL of the PHOENIX model archive.

    Returns
    -------
    wave : tuple[str]
        Local wavelength-file path and archive URL.
    models : list[tuple]
        Local model-file paths and URLs in interpolation grid order.
    """
    outdir = os.fspath(outdir)
    wave = (os.path.join(outdir, PHOENIX_WAVE_FILE), f'{base_url}HiResFITS/{PHOENIX_WAVE_FILE}')
    models = []
    teffs, loggs, mets = get_stellar_param_grid(st_teff, st_logg, st_met)
    # Retain temperature, gravity, then metallicity ordering for grid interpolation.
    for teff in teffs:
        for logg in loggs:
            for met in mets:
                name, subdir = phoenix_model_name(teff, logg, met)
                models.append((os.path.join(outdir, name),
                    f'{base_url}HiResFITS/{PHOENIX_GRID}/{subdir}/{name}'))
    return wave, models


def download_stellar_spectra(st_teff, st_logg, st_met, outdir, *,
                             base_url=PHOENIX_URL, fetch=None, workers=4):
    """Fetch the PHOENIX files v1 ``utils.download_stellar_spectra`` uses.

    Parameters
    ----------
    st_teff, st_logg, st_met : float
        Stellar effective temperature (K), log surface gravity, and metallicity.
    outdir : str
        Directory for calibration or stellar model files.
    base_url : str
        Root URL of the PHOENIX model archive.
    fetch : None, callable
        Download function taking a URL and destination path.
    workers : int
        Number of concurrent workers.

    Returns
    -------
    wave_file : str
        Local wavelength-file path.
    model_files : list[str]
        Local model-file paths in interpolation grid order.
    """
    if outdir is None:
        raise ValueError('a stellar model directory is required')
    outdir = os.fspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    fetch = _urlretrieve if fetch is None else fetch
    wave, models = phoenix_model_paths(st_teff, st_logg, st_met, outdir, base_url=base_url)
    entries = [wave] + models
    missing = [entry for entry in entries
               if not (os.path.exists(entry[0]) and _looks_like_fits(entry[0]))]
    if missing:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(missing)))) as pool:
            list(pool.map(lambda entry: _ensure_file(entry[0], entry[1], fetch), missing))
    return wave[0], [path for path, _ in models]


def interpolate_stellar_model_grid(model_files, st_teff, st_logg, st_met):
    """Interpolate PHOENIX spectra to the requested stellar parameters.

    Parameters
    ----------
    model_files : list[str]
        Model spectra in temperature, gravity, then metallicity grid order.
    st_teff, st_logg, st_met : float
        Stellar effective temperature (K), log surface gravity, and metallicity.

    Returns
    -------
    flux : np.ndarray(float)
        Stellar model interpolated to the requested temperature, gravity, and metallicity.
    """
    from astropy.io import fits
    from scipy.interpolate import RegularGridInterpolator
    teffs, loggs, mets = get_stellar_param_grid(st_teff, st_logg, st_met)
    pts = (np.array(teffs), np.array(loggs), np.array(mets))
    specs = [fits.getdata(model) for model in model_files]
    vals = np.zeros((len(teffs), len(loggs), len(mets), len(specs[0])))
    tot_i = 0
    for i in range(pts[0].shape[0]):
        for j in range(pts[1].shape[0]):
            for k in range(pts[2].shape[0]):
                vals[i, j, k] = specs[tot_i]
                tot_i += 1
    grid = RegularGridInterpolator(pts, vals)
    return grid([st_teff, st_logg, st_met])[0]


def bin_model(new_wave, mod_wave, mod_flux):
    """Bin model flux at the supplied ascending wavelengths using v1 flux-conserving resampling."""
    import spectres
    return spectres.spectres(np.asarray(new_wave), np.asarray(mod_wave), np.asarray(mod_flux))


def load_stellar_model(st_teff, st_logg, st_met, model_dir, *, fetch=None):
    """Load an interpolated PHOENIX spectrum and its wavelength axis.

    Parameters
    ----------
    st_teff, st_logg, st_met : float
        Stellar effective temperature (K), log surface gravity, and metallicity.
    model_dir : str
        Directory for calibration or stellar model files.
    fetch : None, callable
        Download function taking a URL and destination path.

    Returns
    -------
    wave, flux : np.ndarray(float)
        Wavelength samples in microns and their flux, respectively.
    """
    from astropy.io import fits
    wave_file, flux_files = download_stellar_spectra(
        st_teff, st_logg, st_met, model_dir, fetch=fetch)
    mod_flux = interpolate_stellar_model_grid(flux_files, st_teff, st_logg, st_met)
    mod_wave = fits.getdata(wave_file) / 1e4
    return mod_wave, mod_flux


def _cache(ctx):
    """Get or initialize the wavelength calibration cache."""
    cache = ctx.get('wavecal_cache')
    if cache is None:
        cache = {}
        ctx['wavecal_cache'] = cache
    return cache


def _model_dir(opts):
    """Resolve the configured stellar model directory."""
    model_dir = opts.get('stellar_model_dir')
    if model_dir is None:
        raise ValueError('stellar wavelength refinement needs a model directory: set '
            'v2_stellar_model_dir (the optimizer defaults it to ' '<output>/Stage3/phoenix_models)')
    return os.fspath(model_dir)


def _interpolated_model(ctx):
    """Cache the PHOENIX spectrum for the configured stellar parameters."""
    opts = ctx.get('opts', {})
    teff, logg, met = stellar_params(opts)
    model_dir = _model_dir(opts)
    key = ('model', float(teff), float(logg), float(met), model_dir)
    cache = _cache(ctx)
    if key not in cache:
        cache[key] = load_stellar_model(teff, logg, met, model_dir,
                                        fetch=ctx.get('stellar_model_fetch'))
    return cache[key]


def binned_model(ctx, new_wave):
    """Return cached stellar model flux binned at the supplied ascending wavelengths."""
    new_wave = np.asarray(new_wave)
    key = ('binned', new_wave.dtype.str, new_wave.tobytes())
    cache = _cache(ctx)
    if key not in cache:
        mod_wave, mod_flux = _interpolated_model(ctx)
        cache[key] = bin_model(new_wave, mod_wave, mod_flux)
    return cache[key]


def refine_soss_wavelengths(wave_o1, wave_o2, flux_o1, model_o1, oversample=SOSS_CCF_OVERSAMPLE):
    """Apply the order-1 stellar wavelength shift to both SOSS orders.

    Parameters
    ----------
    wave_o1, flux_o1, model_o1 : array-like(float)
        Order-1 wavelengths in microns, extracted flux, and model flux, respectively.
    wave_o2 : None, array-like(float)
        Order-2 wavelength axis in microns, if extracted.
    oversample : int
        Number of interpolated samples per wavelength interval.

    Returns
    -------
    wave_o1 : np.ndarray(float)
        Shifted order-1 wavelength axis.
    wave_o2 : None, np.ndarray(float)
        Shifted order-2 axis, if supplied.
    shift : float
        Applied wavelength offset in microns.
    """
    x1d_flux = np.nansum(flux_o1, axis=0)
    shift, _ = do_ccf(np.asarray(wave_o1), x1d_flux, model_o1, oversample=oversample)
    out1 = np.array(wave_o1, copy=True)
    out1 += shift
    out2 = None
    if wave_o2 is not None:
        out2 = np.array(wave_o2, copy=True)
        out2 += shift
    return out1, out2, shift


def soss_model_for_wave(ctx, wave_o1):
    """Return the cached stellar model binned in SOSS detector-column order."""
    wave_o1 = np.asarray(wave_o1)
    return binned_model(ctx, wave_o1[::-1])[::-1]


def nirspec_ccf_columns(wave1d, detector='', grating=''):
    """Select the NIRSpec wavelength columns used for stellar correlation.

    Parameters
    ----------
    wave1d : array-like(float)
        Wavelength samples in microns.
    detector : None, str
        Detector identifier for the observation.
    grating : str
        NIRSpec disperser identifier.

    Returns
    -------
    columns : np.ndarray(int)
        Detector columns retained for the stellar cross-correlation.
    """
    wave1d = np.asarray(wave1d)
    keep = np.isfinite(wave1d)
    if str(detector).upper() == 'NRS1':
        grating = str(grating).upper()
        low = 3.1 if 'G395' in grating else 1.8 if 'G235' in grating else 1.0
        with np.errstate(invalid='ignore'):
            keep &= wave1d >= low
    return np.where(keep)[0]


def refine_nirspec_wavelengths(wave1d, flux, model_trim, columns=None):
    """Apply a stellar wavelength shift to the NIRSpec trace axis.

    Parameters
    ----------
    wave1d, flux, model_trim : array-like(float)
        Wavelengths in microns, extracted flux, and model flux on the selected columns.
    columns : None, array-like(int)
        Columns to include in the cross-correlation; defaults to finite wavelengths.

    Returns
    -------
    wave : np.ndarray(float)
        NIRSpec wavelength axis after refinement.
    shift : None, float
        Applied wavelength offset in microns, if refinement ran.
    """
    wave1d = np.asarray(wave1d)
    ii = np.where(np.isfinite(wave1d))[0] if columns is None else columns
    x1d_flux = np.nansum(flux, axis=0)[ii]
    shift, _ = do_ccf(wave1d[ii], x1d_flux, model_trim, oversample=NIRSPEC_CCF_OVERSAMPLE)
    out = np.array(wave1d, copy=True)
    out += shift
    return out, shift


def pastasoss_solution(pwcpos, dimx=2048):
    """Get SOSS wavelength vectors at the supplied pupil-wheel position.

    The solution requires a full 2048-column SOSS detector.

    Parameters
    ----------
    pwcpos : float
        Pupil-wheel position from the exposure header.
    dimx : int
        Number of detector columns.

    Returns
    -------
    solution : dict
        Order wavelength vectors, order-2 detector columns, and pupil-wheel position.
    """
    if int(dimx) != 2048:
        raise ValueError(f'use_pastasoss requires 2048 detector columns, got {dimx}')
    import pastasoss
    order1 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='1', interp=True)
    order2 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='2', interp=True)
    wave_o1 = np.asarray(order1.wavelength)
    if wave_o1.shape[0] != 2040:
        raise ValueError(f'PASTASOSS order-1 solution has {wave_o1.shape[0]} columns; v1 '
            'assumes detector columns 4-2043 (2040 columns)')
    return {'wave_o1': wave_o1, 'wave_o2': np.asarray(order2.wavelength),
            'xpos_o2': np.asarray(order2.x).astype(int), 'pwcpos': float(pwcpos)}


def pupil_wheel_position(meta):
    """Read the pupil-wheel position from the first segment header."""
    header = (getattr(meta, 'extra', {}) or {}).get('header', {}) or {}
    value = header.get('PWCPOS')
    if value is None:
        raise ValueError('use_pastasoss requires the PWCPOS header keyword')
    return float(value)


def _trim_pair(pair, columns):
    """Select matching wavelength columns of flux and uncertainty arrays."""
    flux, ferr = pair
    flux = np.asarray(flux)[..., columns]
    ferr = None if ferr is None else np.asarray(ferr)[..., columns]
    return flux, ferr


def soss_stage3_wavelengths(spectra, ctx):
    """Apply v1's optional SOSS wavelength-solution changes.

    Parameters
    ----------
    spectra : dict
        Extracted flux and uncertainty arrays by spectral order.
    ctx : dict
        Reduction context containing options and prepared calibration data.

    Returns
    -------
    spectra : dict
        Flux and uncertainty arrays, trimmed to the selected wavelength coverage.
    waves : dict
        Wavelength vectors for the extracted orders.
    shift : None, float
        Applied stellar wavelength offset in microns, if refinement ran.
    """
    opts = ctx.get('opts', {})
    waves = ctx['waves']
    solution = ctx.get('pastasoss_solution')
    refine = stellar_refinement_enabled(opts)
    if solution is None and not refine:
        return spectra, waves, None
    spectra = dict(spectra)
    waves = dict(waves)
    if solution is not None:
        waves[1] = solution['wave_o1']
        spectra[1] = _trim_pair(spectra[1], slice(4, -4))
        if 2 in spectra:
            waves[2] = solution['wave_o2']
            spectra[2] = _trim_pair(spectra[2], solution['xpos_o2'])
    shift = None
    if refine:
        flux_o1 = np.asarray(spectra[1][0], dtype=np.float64)
        model = soss_model_for_wave(ctx, waves[1])
        wave_o2 = waves.get(2) if 2 in spectra else None
        waves[1], wave_o2, shift = refine_soss_wavelengths(waves[1], wave_o2, flux_o1, model)
        if wave_o2 is not None:
            waves[2] = wave_o2
    return spectra, waves, (None if shift is None else float(shift))


def nirspec_stage3_wavelengths(wave1d, flux, ctx):
    """Apply v1's optional NIRSpec stellar refinement to the trace axis.

    Parameters
    ----------
    wave1d, flux : array-like(float)
        Wavelength samples in microns and extracted stellar flux, respectively.
    ctx : dict
        Reduction context containing options and prepared calibration data.

    Returns
    -------
    wave : np.ndarray(float)
        NIRSpec wavelength axis after refinement.
    shift : None, float
        Applied wavelength offset in microns, if refinement ran.
    """
    if not stellar_refinement_enabled(ctx.get('opts', {})):
        return wave1d, None
    wave1d = np.asarray(wave1d)
    ii = nirspec_ccf_columns(wave1d, ctx.get('nirspec_detector', ''),
                             ctx.get('nirspec_grating', ''))
    model = binned_model(ctx, wave1d[ii])
    out, shift = refine_nirspec_wavelengths(
        wave1d, np.asarray(flux, dtype=np.float64), model, columns=ii)
    return out, float(shift)


def warn_miri_stellar(opts):
    """Warn when stellar wavelength calibration is requested for MIRI."""
    teff, logg, met = stellar_params(opts)
    if teff is not None or logg is not None or met is not None:
        warnings.warn('Wavelength calibration not implemented for MIRI; '
                      'using the default wavelength solution (as in v1).',
                      RuntimeWarning, stacklevel=2)


__all__ = ['PHOENIX_URL', 'binned_model', 'bin_model', 'do_ccf', 'continuum_normalize',
    'download_stellar_spectra', 'get_stellar_param_grid', 'highpass_filter',
    'interpolate_stellar_model_grid', 'load_stellar_model',
    'nirspec_stage3_wavelengths', 'oversample_wave', 'pastasoss_solution',
    'phoenix_model_name', 'phoenix_model_paths', 'pupil_wheel_position',
    'refine_nirspec_wavelengths', 'nirspec_ccf_columns', 'refine_soss_wavelengths',
    'soss_stage3_wavelengths', 'stellar_refinement_enabled',
    'use_pastasoss_option', 'validate_stellar_options', 'warn_miri_stellar',]
