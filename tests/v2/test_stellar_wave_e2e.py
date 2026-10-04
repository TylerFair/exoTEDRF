"""Filesystem-to-products optimizer runs with v1's Stage-3 wavelength options."""

import os
import shutil

import numpy as np
import pytest
from astropy.io import fits

from exotedrf.v2 import config as v2config
from exotedrf.v2 import trace, wavecal
from exotedrf.v2.optimize import run_optimizer

from .niriss_acceptance import apply_current_niriss_settings
from .stellar_wave_oracle import synthetic_stellar_flux, write_phoenix_grid
from .test_yaml_niriss_e2e import _yaml_path

STELLAR = (5400., 4.45, -0.12)
TRUE_SHIFT = 0.004
NGROUPS = 2
DIMY = 96
Y_O1, Y_O2 = 20., 40.


def _waves(dimx):
    """Return waves."""
    return np.linspace(1.62, 1.38, dimx), np.linspace(0.84, 0.62, dimx)


def _write_segment(path, nints, int_start, exsegnum, exposure_nints, dimx,
                   rng):
    """Return write segment."""
    wave_o1, _ = _waves(dimx)
    lines = synthetic_stellar_flux(wave_o1 + TRUE_SHIFT, *STELLAR)
    lines = lines / np.median(lines)
    yy = np.arange(DIMY, dtype=float)[:, None]
    rate = (600. * np.exp(-0.5 * ((yy - Y_O1) / 3.) ** 2) * lines[None] +
            250. * np.exp(-0.5 * ((yy - Y_O2) / 3.) ** 2) + 20.)
    groups = (1. + np.arange(NGROUPS))[:, None, None]
    data = 1500. + rate[None, None] * groups[None] + rng.normal(
        0., 3., (nints, NGROUPS, DIMY, dimx))
    sci = np.clip(np.round(data), 0, 65535).astype(np.uint16)
    ph = fits.PrimaryHDU()
    for key, value in {
            'INSTRUME': 'NIRISS', 'DETECTOR': 'NIS', 'SUBARRAY': 'SUBSTRIP96',
            'EXP_TYPE': 'NIS_SOSS', 'FILTER': 'CLEAR', 'PUPIL': 'GR700XD',
            'PWCPOS': 245.79, 'TGROUP': 5.494, 'TFRAME': 5.494,
            'NFRAMES': 1, 'GROUPGAP': 0, 'NGROUPS': NGROUPS,
            'NINTS': exposure_nints, 'INTSTART': int_start,
            'INTEND': int_start + nints - 1, 'EXSEGNUM': exsegnum,
            'TARGNAME': 'SYNTH'}.items():
        ph.header[key] = value
    times = fits.BinTableHDU.from_columns(fits.ColDefs([fits.Column(
        name='int_mid_BJD_TDB', format='D',
        array=60000. + (int_start - 1 + np.arange(nints)) * 1e-4)]),
        name='INT_TIMES')
    fits.HDUList([ph, fits.ImageHDU(sci, name='SCI'), times]).writeto(path)


def _refpack(dimx):
    """Return refpack."""
    lin = np.zeros((2, DIMY, dimx), np.float32)
    lin[1] = 1.
    wave_o1, wave_o2 = _waves(dimx)
    return {
        'mask_dq': np.zeros((DIMY, dimx), np.uint32),
        'inl_theta': np.zeros(6, np.float32),
        'inl_periods': np.asarray([1024 / 3, 512, 1024], np.float32),
        'superbias': np.full((DIMY, dimx), 1500., np.float32),
        'lin_coeffs': lin, 'lin_dq': np.zeros((DIMY, dimx), np.uint32),
        'readnoise': np.full((DIMY, dimx), 6., np.float32),
        'gain': np.ones((DIMY, dimx), np.float32),
        'gain_factor': np.float32(1.),
        'flat': np.ones((DIMY, dimx), np.float32),
        'wave_o1': wave_o1, 'wave_o2': wave_o2,
    }


def _run(base, dimx, segment_nints, *, stellar=True, pastasoss=False,
         extract_widths=(8, 10, 12), model_dir=None):
    """Return run."""
    input_dir = base / 'inputs'
    input_dir.mkdir(parents=True)
    rng = np.random.RandomState(4)
    total = sum(segment_nints)
    start = 1
    for index, nints in enumerate(segment_nints, start=1):
        _write_segment(input_dir / f'jwsynth-seg{index:03d}_uncal.fits',
                       nints, start, index, total, dimx, rng)
        start += nints
    np.save(base / 'background.npy', np.ones((DIMY, dimx), np.float32))
    trace.save_centroids_csv(str(base / 'centroids.csv'), {
        'xpos': np.arange(dimx, dtype=float),
        'ypos o1': np.full(dimx, Y_O1), 'ypos o2': np.full(dimx, Y_O2)})

    cfg = apply_current_niriss_settings(v2config.load_config(_yaml_path()))
    # Fix every detector sweep to one candidate; keep the aperture sweep.
    for key in list(cfg):
        if key.startswith('optimize_') and cfg[key] and \
                key != 'optimize_extract_width':
            param = key[len('optimize_'):]
            cfg[key] = False
            cfg[param] = cfg[param][len(cfg[param]) // 2]
    cfg.update({
        'input_dir': str(input_dir),
        'pipeline_outputs_directory': str(base / 'outputs'),
        'crds_cache_path': str(base / 'crds'),
        'f277w': None, 'soss_background_file': str(base / 'background.npy'),
        'centroids': str(base / 'centroids.csv'),
        'baseline_ints': [8, -8], 'pca_components': 2,
        'soss_inner_mask_width': 12, 'soss_outer_mask_width': 30,
        'extract_width': list(extract_widths), 'do_plots': False,
        'wave_range': None,
    })
    if stellar:
        cfg.update(st_teff=STELLAR[0], st_logg=STELLAR[1], st_met=STELLAR[2])
    if model_dir is not None:
        cfg['v2_stellar_model_dir'] = str(model_dir)
    if pastasoss:
        cfg['stage3_kwargs'] = {'Extract1dStep': {'use_pastasoss': True}}
    return cfg, run_optimizer(cfg, logger=None, refpack=_refpack(dimx))


def _spectra_file(base):
    """Return spectra file."""
    stage3 = base / 'outputs' / 'v2' / 'Stage3'
    files = sorted(stage3.glob('*_box_spectra_fullres.fits'))
    assert len(files) == 1
    return files[0]


@pytest.fixture(scope='module')
def grid(tmp_path_factory):
    """Create a local synthetic stellar-model grid.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for temporary test directories.

    Returns
    -------
    directory : pathlib.Path
        Directory containing the synthetic stellar models.
    """
    source = tmp_path_factory.mktemp('phoenix_source')
    write_phoenix_grid(source, [STELLAR])
    return source


@pytest.fixture
def copy_fetch(grid, monkeypatch):
    """Serve 'downloads' from the local synthetic grid; record the URLs.

    Parameters
    ----------
    grid : pathlib.Path
        Local synthetic stellar-model grid.
    monkeypatch : pytest.MonkeyPatch
        Fixture for temporary replacements.

    Returns
    -------
    requests : list
        Recorded stellar-model download requests.
    """
    record = []

    def fetch(url, destination):
        record.append(url)
        shutil.copyfile(grid / url.rsplit('/', 1)[-1], destination)

    monkeypatch.setattr(wavecal, '_urlretrieve', fetch)
    return record


def test_soss_stellar_refinement_end_to_end(tmp_path, copy_fetch,
                                            monkeypatch):
    """Check SOSS stellar refinement end to end."""
    calls = []
    original = wavecal.do_ccf

    def counted(*args, **kwargs):
        calls.append(kwargs.get('oversample'))
        return original(*args, **kwargs)

    monkeypatch.setattr(wavecal, 'do_ccf', counted)
    widths = (8, 10, 12)
    cfg, result = _run(tmp_path, 128, (20, 16), extract_widths=widths)
    assert np.isfinite(result.cost)

    # The grid was fetched into v1's default location below Stage 3.
    cache = tmp_path / 'outputs' / 'v2' / 'Stage3' / 'phoenix_models'
    assert len(copy_fetch) == 5
    assert sorted(os.listdir(cache)) == sorted(
        url.rsplit('/', 1)[-1] for url in copy_fetch)

    assert len(calls) >= len(widths) + 1
    assert set(calls) == {wavecal.SOSS_CCF_OVERSAMPLE}

    with fits.open(_spectra_file(tmp_path)) as hdul:
        shift = hdul[0].header['WAVESHFT']
        wave_o1 = hdul['Wave O1'].data
        wave_o2 = hdul['Wave O2'].data
    assert abs(shift - TRUE_SHIFT) < 1e-3
    default_o1, default_o2 = _waves(128)
    np.testing.assert_array_equal(wave_o1, np.sort(default_o1 + shift))
    np.testing.assert_array_equal(wave_o2, np.sort(default_o2 + shift))


def test_soss_without_stellar_parameters_is_unchanged(tmp_path, copy_fetch):
    """Check SOSS without stellar parameters is unchanged."""
    _, result = _run(tmp_path, 128, (20, 16), stellar=False,
                     extract_widths=(10,))
    assert np.isfinite(result.cost)
    assert copy_fetch == []
    assert not (tmp_path / 'outputs' / 'v2' / 'Stage3' /
                'phoenix_models').exists()
    with fits.open(_spectra_file(tmp_path)) as hdul:
        assert 'WAVESHFT' not in hdul[0].header
        assert 'WAVESOLN' not in hdul[0].header
        np.testing.assert_array_equal(hdul['Wave O1'].data,
                                      np.sort(_waves(128)[0]))


def test_soss_pastasoss_and_explicit_model_dir_end_to_end(tmp_path, grid,
                                                          monkeypatch):
    """Check SOSS pastasoss and explicit model dir end to end."""
    def offline(url, destination):
        raise AssertionError('a pre-populated model directory must be used')

    monkeypatch.setattr(wavecal, '_urlretrieve', offline)
    with pytest.warns(RuntimeWarning, match='use_pastasoss'):
        _, result = _run(tmp_path, 2048, (8, 8), pastasoss=True,
                         extract_widths=(10,), model_dir=grid)
    assert np.isfinite(result.cost)
    with fits.open(_spectra_file(tmp_path)) as hdul:
        assert hdul[0].header.get('WAVESOLN') != 'pastasoss'
        for label in ('O1', 'O2'):
            assert hdul[f'Flux {label}'].data.shape[-1] == \
                hdul[f'Wave {label}'].data.shape[-1] > 1900
