"""Check stellar wavelength refinement against v1 functions."""

import os
import types

import numpy as np
import pytest
from astropy.io import fits

from exotedrf.v2 import stages, wavecal

from .stellar_wave_oracle import (load_v1, observed_spectrum,
                                  write_phoenix_grid)

FAEDI = (5400., 4.45, -0.12)
ERS = (5512., 4.47, 0.0)


@pytest.fixture(scope='module')
def v1():
    """Load the v1 wavelength-refinement functions.

    Returns
    -------
    result : types.SimpleNamespace
        Loaded functions or synthetic observation records.
    """
    return load_v1()


@pytest.fixture(scope='module')
def phoenix(tmp_path_factory):
    """Create a local synthetic PHOENIX model grid.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for temporary test directories.

    Returns
    -------
    directory : pathlib.Path
        Directory containing the synthetic stellar models.
    """
    base = tmp_path_factory.mktemp('stellar_wave')
    model_dir = base / 'phoenix_models'
    write_phoenix_grid(model_dir, [FAEDI, ERS, (4800., 4.588, 0.14)])
    return base


def _v1_loop_oversample(thiswave, oversample):
    """Return v1 loop oversample."""
    new_wave = []
    for i in range(len(thiswave)):
        new_wave.append(thiswave[i])
        if i < len(thiswave) - 1:
            step = thiswave[i + 1] - thiswave[i]
            step /= oversample
            for s in range(1, oversample):
                new_wave.append(thiswave[i] + s * step)
    return np.asarray(new_wave)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('oversample', [2, 5, 7])
def test_oversampled_grid_is_bitwise_v1(dtype, oversample):
    """Check oversampled grid is bitwise v1."""
    rng = np.random.default_rng(0)
    wave = np.sort(rng.uniform(0.8, 2.9, 300)).astype(dtype)
    got = wavecal.oversample_wave(wave, oversample)
    expected = _v1_loop_oversample(wave, oversample)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('oversample', [1, 5])
@pytest.mark.parametrize('descending', [True, False])
@pytest.mark.parametrize('nan_fraction', [0.0, 0.02])
def test_do_ccf_is_exactly_v1(v1, phoenix, dtype, oversample, descending,
                              nan_fraction):
    """Check do CCF is exactly v1."""
    wave = np.linspace(0.9, 2.7, 1500).astype(dtype)
    if descending:
        wave = wave[::-1].copy()
    flux, _ = observed_spectrum(wave, 0.0021, FAEDI, nints=1, seed=5)
    flux = flux[0]
    if nan_fraction:
        flux[np.random.default_rng(1).random(flux.size) < nan_fraction] = \
            np.nan
    ctx = {'opts': {'st_teff': FAEDI[0], 'st_logg': FAEDI[1],
                    'st_met': FAEDI[2],
                    'stellar_model_dir': str(phoenix / 'phoenix_models')}}
    order = np.argsort(wave)
    model = np.empty(wave.size)
    model[order] = wavecal.binned_model(ctx, wave[order])
    expected = v1.do_ccf(wave.copy(), flux.copy(), model.copy(),
                         oversample=oversample)
    got = wavecal.do_ccf(wave, flux, model, oversample=oversample)
    # ExoTEDRF 2.5.0: do_ccf returns (shift_wave, shift_steps).
    assert type(got) is type(expected) is tuple
    assert got == expected
    got = got[0]
    if not nan_fraction:
        # Sanity: the injected offset is recovered to the CCF resolution.
        step = np.median(np.diff(np.sort(wave))) / oversample
        assert abs(got - 0.0021) <= 1.5 * step


@pytest.mark.parametrize('params', [
    FAEDI, ERS, (5400., 4.5, 0.0), (4800., 4.588, 0.14), (6123., 3.9, -1.5),
    (5000., 4.0, -0.5), (3999.9, 5.01, 0.99), (5550., 4.25, -0.0)])
def test_stellar_param_grid_is_v1(v1, params):
    """Check stellar param grid is v1."""
    assert wavecal.get_stellar_param_grid(*params) == \
        v1.utils.get_stellar_param_grid(*params)


@pytest.mark.parametrize('params', [
    FAEDI, ERS, (5400., 4.5, 0.0), (4800., 4.588, 0.14), (3500., 5.2, -2.4),
    (6100., 3.5, 0.3)])
def test_phoenix_names_and_urls_are_v1(params, tmp_path):
    """Check phoenix names and urls are v1."""
    commands = []
    v1 = load_v1(system=commands.append, exists=lambda path: False)
    wfile, ffiles = v1.utils.download_stellar_spectra(
        *params, outdir=str(tmp_path), silent=True)
    (wave_path, wave_url), models = wavecal.phoenix_model_paths(
        *params, str(tmp_path))
    assert os.path.normpath(wfile) == os.path.normpath(wave_path)
    assert [os.path.normpath(p) for p in ffiles] == \
        [os.path.normpath(p) for p, _ in models]
    # V1: 'wget -q -O <file> <url>' for the wave file and every model.
    urls = [command.split()[-1] for command in commands]
    assert urls == [wave_url] + [url for _, url in models]


def test_v1_naming_quirks_are_corrected():
    # Names v1 can download are identical; impossible v1 spellings fixed.
    """Check v1 naming quirks are corrected."""
    assert wavecal.phoenix_model_name(5400, 4.5, 0.0)[0].startswith(
        'lte05400-4.50-0.0.')
    assert wavecal.phoenix_model_name(10200, 4.0, 0.0)[0].startswith(
        'lte10200-4.00-0.0.')
    assert wavecal.phoenix_model_name(5400, 4.5, -0.0) == \
        wavecal.phoenix_model_name(5400, 4.5, 0.0)
    assert wavecal.phoenix_model_name(5400, 4.5, 1.0)[1] == 'Z+1.0'
    assert wavecal.phoenix_model_name(5400, 4.5, -1.0)[1] == 'Z-1.0'


@pytest.mark.parametrize('params', [FAEDI, ERS, (4800., 4.588, 0.14)])
def test_interpolated_model_is_exactly_v1(v1, phoenix, params):
    """Check interpolated model is exactly v1."""
    model_dir = str(phoenix / 'phoenix_models')
    wfile, ffiles = v1.utils.download_stellar_spectra(
        *params, outdir=model_dir, silent=True)
    expected = v1.utils.interpolate_stellar_model_grid(ffiles, *params)
    mod_wave, got = wavecal.load_stellar_model(*params, model_dir)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(mod_wave, fits.getdata(wfile) / 1e4)


def _soss_inputs(dtype=np.float32, nints=30, shift=0.0013):
    """Return SOSS inputs."""
    wave_o1 = np.linspace(2.83, 0.85, 2048).astype(dtype)
    wave_o2 = np.linspace(1.41, 0.50, 2048).astype(dtype)
    flux_o1, ferr_o1 = observed_spectrum(wave_o1, shift, FAEDI, nints=nints,
                                         seed=7, nan_fraction=0.001)
    flux_o2, ferr_o2 = observed_spectrum(wave_o2, shift, FAEDI, nints=nints,
                                         seed=8)
    flux_o1[5, 100] += 4e4
    return wave_o1, flux_o1, ferr_o1, wave_o2, flux_o2, ferr_o2


def _run_v1_soss(v1, phoenix, inputs, params, use_pastasoss=False,
                 pwcpos=None):
    """Return run v1 SOSS."""
    w1, f1, e1, w2, f2, e2 = (np.array(item, copy=True) for item in inputs)
    copies = (w1, f1, e1, -np.ones(w1.size), w2, f2, e2, -np.ones(w2.size))
    nints = inputs[1].shape[0]
    st = params if params is not None else (None, None, None)
    return v1.format_soss_spectra(
        copies, np.arange(nints, dtype=float),
        {'method': 'box', 'extract_width': 30}, 'SYNTH b', *st,
        pwcpos=pwcpos, output_dir=str(phoenix) + '/', save_results=False,
        use_pastasoss=use_pastasoss, clip_thresh=10)


def _soss_ctx(phoenix, inputs, params, pwcpos=None):
    """Return SOSS ctx."""
    opts = {'stellar_model_dir': str(phoenix / 'phoenix_models')}
    if params is not None:
        opts.update(st_teff=params[0], st_logg=params[1], st_met=params[2])
    ctx = {'opts': opts, 'waves': {1: inputs[0].copy(), 2: inputs[3].copy()}}
    if pwcpos is not None:
        ctx['pastasoss_solution'] = wavecal.pastasoss_solution(pwcpos)
    return ctx


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('params', [FAEDI, ERS])
def test_soss_products_are_exactly_v1(v1, phoenix, dtype, params):
    """Check SOSS products are exactly v1."""
    inputs = _soss_inputs(dtype)
    expected = _run_v1_soss(v1, phoenix, inputs, params)
    ctx = _soss_ctx(phoenix, inputs, params)
    spectra = {1: (inputs[1], inputs[2]), 2: (inputs[4], inputs[5])}
    _, products = stages._final_spectral_products(spectra, ctx)
    for order, label in ((1, 'O1'), (2, 'O2')):
        product = products[order]
        assert product['wave'].dtype == expected[f'Wave {label}'].dtype
        np.testing.assert_array_equal(product['wave'],
                                      expected[f'Wave {label}'])
        np.testing.assert_array_equal(product['flux'],
                                      expected[f'Flux {label}'])
        np.testing.assert_array_equal(product['ferr'],
                                      expected[f'Flux Err {label}'])
    # One shift for both orders, measured on order 1, recorded.
    shift = products[1]['wave_shift']
    assert shift == products[2]['wave_shift']
    assert abs(shift - 0.0013) < 2e-4
    # The input context is not modified.
    np.testing.assert_array_equal(ctx['waves'][1], inputs[0])


@pytest.mark.parametrize('params', [None, FAEDI])
def test_soss_pastasoss_falls_back_like_v1_2_5_0(v1, phoenix, params):
    """2.5.0: use_pastasoss only warns and keeps the default solution."""
    inputs = _soss_inputs(np.float32)
    expected = _run_v1_soss(v1, phoenix, inputs, params, use_pastasoss=True,
                            pwcpos=245.79)
    ctx = _soss_ctx(phoenix, inputs, params)
    spectra = {1: (inputs[1], inputs[2]), 2: (inputs[4], inputs[5])}
    clipped, products = stages._final_spectral_products(spectra, ctx)
    assert clipped[1][0].shape == (inputs[1].shape[0], 2048)
    for order, label in ((1, 'O1'), (2, 'O2')):
        np.testing.assert_array_equal(products[order]['wave'],
                                      expected[f'Wave {label}'])
        np.testing.assert_array_equal(products[order]['flux'],
                                      expected[f'Flux {label}'])
    assert ('wave_shift' in products[1]) == (params is not None)


def test_soss_default_products_unchanged(phoenix):
    """Check SOSS default products unchanged."""
    inputs = _soss_inputs(np.float32)
    ctx = _soss_ctx(phoenix, inputs, None)
    spectra = {1: (inputs[1], inputs[2]), 2: (inputs[4], inputs[5])}
    _, products = stages._final_spectral_products(spectra, ctx)
    np.testing.assert_array_equal(products[1]['wave'], inputs[0][::-1])
    assert 'wave_shift' not in products[1]
    assert 'wavecal_cache' not in ctx


def _nirspec_inputs(nints=25, shift=-0.0009):
    """Return NIRSpec inputs."""
    wave1d = np.full(2048, np.nan)
    wave1d[500:] = np.linspace(2.87, 3.72, 1548)
    flux, ferr = observed_spectrum(wave1d, shift, FAEDI, nints=nints, seed=9,
                                   nan_fraction=0.0005)
    return wave1d, flux, ferr


@pytest.mark.parametrize('params', [FAEDI, ERS])
def test_nirspec_products_are_exactly_v1(v1, phoenix, params):
    """Check NIRSpec products are exactly v1."""
    wave1d, flux, ferr = _nirspec_inputs()
    nints = flux.shape[0]
    wave2d = np.repeat(wave1d[None, :], nints, axis=0)
    expected = v1.format_nirspec_spectra(
        (wave2d.copy(), flux.copy(), ferr.copy(), -np.ones(2048)),
        np.arange(nints, dtype=float),
        {'method': 'box', 'extract_width': 12}, 'SYNTH b', 'nrs1', 'g395h',
        *params, output_dir=str(phoenix) + '/', save_results=False,
        clip_thresh=10)

    dimy = 32
    wave_map = np.full((dimy, 2048), np.nan)
    ypos = np.full(1548, 15.4)
    wave_map[15, 500:] = wave1d[500:]
    centroids = {'xpos': np.arange(500, 2048), 'ypos': ypos}
    state = types.SimpleNamespace(cube=types.SimpleNamespace(
        data=np.zeros((1, dimy, 2048), np.float32)))
    ctx = {'opts': {'st_teff': params[0], 'st_logg': params[1],
                    'st_met': params[2],
                    'stellar_model_dir': str(phoenix / 'phoenix_models')},
           'nirspec_wave_map': wave_map, 'nirspec_detector': 'NRS1',
           'nirspec_grating': 'G395H'}
    products = stages._nirspec_final_products(state, ctx, centroids, flux,
                                              ferr)
    np.testing.assert_array_equal(products[1]['wave'], expected['Wave'])
    np.testing.assert_array_equal(products[1]['flux'], expected['Flux'])
    np.testing.assert_array_equal(products[1]['ferr'], expected['Flux Err'])
    assert np.isnan(products[1]['wave'][:500]).all()
    assert abs(products[1]['wave_shift'] + 0.0009) < 5e-4

    # Without stellar parameters the product is the plain trace axis.
    ctx['opts'] = {}
    plain = stages._nirspec_final_products(state, ctx, centroids, flux, ferr)
    np.testing.assert_array_equal(plain[1]['wave'], wave1d)
    assert 'wave_shift' not in plain[1]


def test_miri_v1_and_v2_only_warn(v1):
    """Check MIRI v1 and v2 only warn."""
    nints = 12
    wave = np.repeat(np.linspace(5, 12, 64)[None], nints, axis=0)
    flux = np.random.default_rng(2).normal(100, 1, (nints, 64))
    expected = v1.format_miri_spectra(
        (wave.copy(), flux, flux, -np.ones(64)), np.arange(nints, dtype=float),
        {'method': 'box', 'extract_width': 8}, 'SYNTH b', *FAEDI,
        save_results=False, clip_thresh=10)
    np.testing.assert_array_equal(expected['Wave'], wave[0])
    with pytest.warns(RuntimeWarning, match='not implemented for MIRI'):
        wavecal.warn_miri_stellar(
            {'st_teff': FAEDI[0], 'st_logg': None, 'st_met': None})


def test_model_preparation_is_cached_per_wavelength_axis(phoenix,
                                                         monkeypatch):
    """Check model preparation is cached per wavelength axis."""
    calls = []
    original = wavecal.bin_model

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(wavecal, 'bin_model', counted)
    inputs = _soss_inputs(np.float32, nints=12)
    ctx = _soss_ctx(phoenix, inputs, FAEDI)
    spectra = {1: (inputs[1], inputs[2]), 2: (inputs[4], inputs[5])}
    shifts = set()
    for scale in (1.0, 0.8, 1.2):
        scaled = {o: (f * scale, e) for o, (f, e) in spectra.items()}
        _, products = stages._final_spectral_products(scaled, ctx)
        shifts.add(products[1]['wave_shift'])
    assert len(calls) == 1
    assert len(shifts) == 1


# Download helper (never touches the network).

def _fake_fetch(record):
    """Return fake fetch."""
    def fetch(url, destination):
        record.append(url)
        fits.PrimaryHDU(np.arange(4, dtype=np.float32)).writeto(destination)
    return fetch


def test_download_fetches_missing_files_once(tmp_path):
    """Check download fetches missing files once."""
    record = []
    wfile, ffiles = wavecal.download_stellar_spectra(
        *FAEDI, tmp_path / 'models', fetch=_fake_fetch(record))
    (_, wave_url), models = wavecal.phoenix_model_paths(*FAEDI,
                                                        tmp_path / 'models')
    assert sorted(record) == sorted([wave_url] + [u for _, u in models])
    assert all(os.path.exists(path) for path in [wfile] + ffiles)
    assert not [p for p in os.listdir(tmp_path / 'models')
                if p.endswith('.part')]
    record.clear()
    wavecal.download_stellar_spectra(*FAEDI, tmp_path / 'models',
                                     fetch=_fake_fetch(record))
    assert record == []
    # A truncated/empty file (v1's failed wget) is fetched again.
    open(ffiles[0], 'wb').close()
    wavecal.download_stellar_spectra(*FAEDI, tmp_path / 'models',
                                     fetch=_fake_fetch(record))
    assert record == [models[0][1]]


def test_download_failures_are_clear_and_leave_no_partial_file(tmp_path):
    """Check download failures are clear and leave no partial file."""
    def offline(url, destination):
        with open(destination, 'wb') as handle:
            handle.write(b'partial')
        raise OSError('network unreachable')

    with pytest.raises(RuntimeError, match='v2_stellar_model_dir'):
        wavecal.download_stellar_spectra(*FAEDI, tmp_path, fetch=offline)
    assert os.listdir(tmp_path) == []

    def html(url, destination):
        with open(destination, 'wb') as handle:
            handle.write(b'<html>not found</html>')

    with pytest.raises(RuntimeError, match='did not return a FITS'):
        wavecal.download_stellar_spectra(*FAEDI, tmp_path, fetch=html)
    assert os.listdir(tmp_path) == []


def test_missing_model_dir_is_an_error():
    """Check missing model dir is an error."""
    ctx = {'opts': {'st_teff': 5400., 'st_logg': 4.5, 'st_met': 0.0},
           'waves': {1: np.linspace(2.8, 0.9, 64)}}
    spectra = {1: (np.ones((5, 64)), None)}
    with pytest.raises(ValueError, match='v2_stellar_model_dir'):
        stages._final_spectral_products(spectra, ctx)


def test_v1_flux_calibrate_runs_unchanged_on_v2_products(v1, tmp_path):
    """Check v1 flux calibrate runs unchanged on v2 products."""
    from exotedrf.v2 import io
    from exotedrf.v2.core import ObsMeta
    nints = 6
    wave = np.linspace(2.9, 3.7, 40)
    flux = np.random.default_rng(4).uniform(1, 2, (nints, 40))
    meta = ObsMeta('NIRSpec/G395H', 'NRS1', 'SUB2048', 1.0, 2,
                   np.arange(nints, dtype=float), np.array([2]),
                   np.array([nints]), ('seg.fits',), {})
    path = tmp_path / 'SYNTH_nrs1_box_spectra_fullres.fits'
    io.save_spectra(path, {'': {'wave': wave, 'flux': flux, 'ferr': flux}},
                    meta)
    v1.flux_calibrate(str(path))
    with fits.open(str(path)[:-5] + '_FluxCalibrated.fits') as hdul:
        assert hdul[3].header['UNITS'] == 'erg/s/cm2/um'
        np.testing.assert_allclose(
            hdul[3].data, v1.utils.convert_flux_units(
                wave, flux.astype(np.float32)), rtol=1e-6)
