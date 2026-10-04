"""Check miri refs."""

import sys
import types

import numpy as np
import pytest
from astropy.io import fits

from exotedrf.v2 import refs
from exotedrf.v2.core import ObsMeta, RampCube


def _miri_cube():
    """Return MIRI cube."""
    meta = ObsMeta(
        'MIRI/LRS', 'MIRIMAGE', 'SLITLESSPRISM', 2.775, 2,
        np.asarray([1.]), np.asarray([0]), np.asarray([1]), ('miri.fits',))
    return RampCube(
        np.zeros((1, 2, 2, 4), np.float32),
        np.zeros((1, 2, 2, 4), np.uint8),
        np.zeros((2, 4), np.uint32), meta)


def _all_skipped(**updates):
    """Return all skipped."""
    opts = {key: 'skip' for key in (
        'DQInitStep', 'INLCorrStep', 'EmiCorrStep', 'ResetStep',
        'SuperBiasStep', 'RefPixStep', 'DarkCurrentStep',
        'OneOverFStep_grp', 'LinearityStep', 'JumpStep', 'RampFitStep',
        'GainScaleStep', 'AssignWCSStep', 'Extract2DStep', 'WaveCorrStep',
        'SourceTypeStep', 'FlatFieldStep', 'BackgroundStep',
        'OneOverFStep_int', 'BadPixStep', 'PCAReconstructStep')}
    opts.update(updates)
    return opts


def test_miri_dark_adaptation_matches_pinned_four_dimensional_rules():
    """Check MIRI dark adaptation matches pinned four dimensional rules."""
    science = fits.Header({
        'NINTS': 1, 'NGROUPS': 2, 'NFRAMES': 2, 'GROUPGAP': 1})
    reference = fits.Header({'NFRAMES': 1, 'GROUPGAP': 0})
    dark = np.arange(2 * 5 * 2 * 3, dtype=np.float32).reshape(2, 5, 2, 3)
    dark[0, 0, 0, 0] = np.nan

    adapted, dark_dq, status, reason = refs._adapt_miri_dark_reference(
        dark, None, science, reference)
    safe = np.nan_to_num(dark, nan=0.)
    expected = np.zeros((2, 2, 2, 3), np.float32)
    expected[0, 0] = safe[0, 0:2].mean(axis=0, dtype=np.float32)
    expected[0, 1] = safe[0, 3:5].mean(axis=0, dtype=np.float32)
    np.testing.assert_array_equal(adapted, expected)
    np.testing.assert_array_equal(dark_dq, 0)
    assert status == 'COMPLETE'
    assert reason == ''

    direct_science = fits.Header({
        'NINTS': 3, 'NGROUPS': 2, 'NFRAMES': 1, 'GROUPGAP': 0})
    direct, dark_dq, status, _ = refs._adapt_miri_dark_reference(
        dark, None, direct_science, reference)
    np.testing.assert_array_equal(direct, safe[:, :2])
    np.testing.assert_array_equal(dark_dq, 0)
    assert status == 'COMPLETE'

    too_long = fits.Header({
        'NINTS': 1, 'NGROUPS': 3, 'NFRAMES': 2, 'GROUPGAP': 1})
    neutral, dark_dq, status, reason = refs._adapt_miri_dark_reference(
        dark, None, too_long, reference)
    expected = np.zeros((2, 3, 2, 3), np.float32)
    expected[0, :2] = adapted[0]
    expected[0, 2] = (dark[0, 4] + 2.5 * (dark[0, 4] - dark[0, 3]))
    np.testing.assert_array_equal(neutral, expected)
    np.testing.assert_array_equal(dark_dq, 0)
    assert status == 'COMPLETE'
    assert reason == ''


def test_miri_dark_dq_collapse_uses_group_zero_only():
    """Check MIRI dark DQ collapse uses group zero only."""
    dq = np.zeros((2, 3, 2, 4), np.uint32)
    dq[0, 0, 0, 0] = 1
    dq[1, 0, 0, 0] = 4
    dq[0, 2, 1, 1] = 8
    collapsed = refs._collapse_miri_dark_dq(dq, (2, 4))
    assert collapsed[0, 0] == 5
    assert collapsed[1, 1] == 0


def test_extract_miri_emicorr_reference_uses_builder_only_jwst_helpers(
        tmp_path, monkeypatch):
    """Check extract MIRI emicorr reference uses builder only JWST helpers."""
    class FakeEmiModel:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def get_subarcase(model, subarray, readpatt, detector):
        assert isinstance(model, FakeEmiModel)
        assert (subarray, readpatt, detector) == (
            'SLITLESSPRISM', 'FASTR1', 'MIRIMAGE')
        return 'SLITLESSPRISM', 28, 15904, ['Hz390', 'Hz218']

    waves = {
        'Hz390': (390.625, np.asarray([0., 1., 0., -1.])),
        'Hz218': (218.52055, np.asarray([1., 0., -1.])),
    }
    jwst = types.ModuleType('jwst')
    jwst.datamodels = types.SimpleNamespace(EmiModel=FakeEmiModel)
    emicorr_package = types.ModuleType('jwst.emicorr')
    emicorr_module = types.ModuleType('jwst.emicorr.emicorr')
    emicorr_module.get_subarcase = get_subarcase
    emicorr_module.get_frequency_info = lambda model, name: waves[name]
    monkeypatch.setitem(sys.modules, 'jwst', jwst)
    monkeypatch.setitem(sys.modules, 'jwst.emicorr', emicorr_package)
    monkeypatch.setitem(sys.modules, 'jwst.emicorr.emicorr', emicorr_module)

    path = tmp_path / 'emicorr.asdf'
    path.touch()
    header = fits.Header({
        'SUBARRAY': 'SLITLESSPRISM', 'READPATT': 'FASTR1',
        'DETECTOR': 'MIRIMAGE'})
    packed = refs._extract_miri_emicorr_reference(path, header)

    np.testing.assert_array_equal(
        packed['emicorr_frequencies'], [390.625, 218.52055])
    np.testing.assert_array_equal(
        packed['emicorr_reference_waves'],
        [[0., 1., 0., -1.], [1., 0., -1., 0.]])
    np.testing.assert_array_equal(
        packed['emicorr_reference_wave_lengths'], [4, 3])
    assert packed['emicorr_frequencies'].dtype == np.float64
    assert packed['emicorr_reference_waves'].dtype == np.float64
    assert packed['emicorr_rowclocks'] == np.int64(28)
    assert packed['emicorr_frameclocks'] == np.int64(15904)


def test_load_miri_wave_map_supports_npy_and_fits(tmp_path):
    """Check load MIRI wave map supports npy and FITS."""
    wave = np.arange(8, dtype=np.float64).reshape(2, 4)
    npy = tmp_path / 'wave.npy'
    np.save(npy, wave)
    np.testing.assert_array_equal(refs._load_miri_wave_map(npy, (2, 4)), wave)

    path = tmp_path / 'wave.fits'
    fits.HDUList([
        fits.PrimaryHDU(), fits.ImageHDU(wave, name='WAVELENGTH')
    ]).writeto(path)
    np.testing.assert_array_equal(
        refs._load_miri_wave_map(path, (2, 4)), wave)
    with pytest.raises(ValueError, match='does not match'):
        refs._load_miri_wave_map(npy, (4, 2))


def test_compute_miri_wave_map_uses_builder_only_assign_wcs(monkeypatch):
    """Check compute MIRI wave map uses builder only assign wcs."""
    expected = np.linspace(5., 12., 8).reshape(2, 4)
    calls = []

    class Ramp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Rate:
        def __init__(self, data):
            self.data = data
            self.meta = types.SimpleNamespace(
                exposure=types.SimpleNamespace(type=None))

        def update(self, ramp):
            calls.append(('update', ramp))

    class AssignWcsStep:
        @staticmethod
        def call(rate, **kwargs):
            calls.append(('assign_wcs', rate.data.shape,
                          rate.meta.exposure.type, kwargs))
            return types.SimpleNamespace(wavelength=expected)

    jwst = types.ModuleType('jwst')
    jwst.__path__ = []
    jwst.datamodels = types.SimpleNamespace(
        open=lambda path: Ramp(), CubeModel=Rate)
    assign_wcs = types.ModuleType('jwst.assign_wcs')
    assign_wcs.AssignWcsStep = AssignWcsStep
    monkeypatch.setitem(sys.modules, 'jwst', jwst)
    monkeypatch.setitem(sys.modules, 'jwst.assign_wcs', assign_wcs)

    result = refs._compute_miri_wave_map('miri_uncal.fits', (2, 4))
    np.testing.assert_array_equal(result, expected)
    assert calls[-1] == (
        'assign_wcs', (1, 2, 4), 'MIR_LRS-SLITLESS',
        {'slit_y_low': -0.55, 'slit_y_high': 0.55})


def test_build_miri_refpack_packs_reset_emi_dark_and_wave(
        tmp_path, monkeypatch):
    """Check build MIRI refpack packs reset emi dark and wave."""
    shape = (2, 4)
    header = fits.Header({
        'INSTRUME': 'MIRI', 'DETECTOR': 'MIRIMAGE',
        'SUBARRAY': 'SLITLESSPRISM', 'READPATT': 'FASTR1',
        'EXP_TYPE': 'MIR_LRS-SLITLESS', 'SUBSTRT1': 1, 'SUBSTRT2': 1,
        'SUBSIZE1': shape[1], 'SUBSIZE2': shape[0], 'NINTS': 1,
        'NGROUPS': 2, 'NFRAMES': 2, 'GROUPGAP': 1, 'NSAMPLES': 1})
    uncal = tmp_path / 'miri_uncal.fits'
    fits.HDUList([
        fits.PrimaryHDU(header=header),
        fits.ImageHDU(np.zeros((1, 2) + shape, np.float32), name='SCI'),
    ]).writeto(uncal)

    reset_data = np.arange(2 * 3 * 2 * 4, dtype=np.float32).reshape(
        2, 3, 2, 4)
    reset_dq = np.arange(8, dtype=np.uint32).reshape(shape)
    reset = tmp_path / 'reset.fits'
    fits.HDUList([
        fits.PrimaryHDU(), fits.ImageHDU(reset_data, name='SCI'),
        fits.ImageHDU(reset_dq, name='DQ'),
    ]).writeto(reset)

    dark_data = np.arange(2 * 5 * 2 * 4, dtype=np.float32).reshape(
        2, 5, 2, 4)
    dark_dq = np.zeros((2, 5) + shape, np.uint32)
    dark_dq[0, 0, 0, 0] = 1
    dark_dq[1, 0, 0, 0] = 4
    dark_dq[0, 4, 1, 1] = 8
    dark = tmp_path / 'dark.fits'
    fits.HDUList([
        fits.PrimaryHDU(header=fits.Header({
            'NFRAMES': 1, 'GROUPGAP': 0})),
        fits.ImageHDU(dark_data, name='SCI'),
        fits.ImageHDU(dark_dq, name='DQ'),
        fits.ImageHDU(np.full(shape, .25, np.float32), name='AVDRKCUR'),
    ]).writeto(dark)

    emicorr = tmp_path / 'emicorr.asdf'
    emicorr.touch()
    wave = np.linspace(5., 12., 8).reshape(shape)
    wave_file = tmp_path / 'wave.npy'
    np.save(wave_file, wave)
    paths = {'emicorr': str(emicorr), 'reset': str(reset), 'dark': str(dark)}
    calls = []

    def getreferences(*args, **kwargs):
        calls.append(kwargs['reftypes'])
        return paths

    monkeypatch.setitem(
        sys.modules, 'crds', types.SimpleNamespace(getreferences=getreferences))
    monkeypatch.setattr(
        refs, '_extract_miri_emicorr_reference', lambda path, hdr: {
            'emicorr_frequencies': np.asarray([390.625]),
            'emicorr_reference_waves': np.asarray([[0., 1., 0., -1.]]),
            'emicorr_rowclocks': np.int64(28),
            'emicorr_frameclocks': np.int64(15904),
        })

    output = refs.build_refpack(
        uncal, out=tmp_path / 'refpack.npz', wavemap_file=wave_file,
        include_inl=False)
    assert calls == [refs.MIRI_REFTYPES]
    assert 'superbias' not in refs.MIRI_REFTYPES
    with np.load(output, allow_pickle=False) as pack:
        np.testing.assert_array_equal(pack['reset_data'], reset_data)
        np.testing.assert_array_equal(pack['reset_dq'], reset_dq)
        assert pack['reset_data'].dtype == np.float32
        assert pack['reset_dq'].dtype == np.uint32
        assert pack['dark'].shape == (2, 2) + shape
        np.testing.assert_array_equal(pack['dark'][1], 0.)
        assert pack['dark_dq'][0, 0] == 5
        assert pack['dark_dq'][1, 1] == 0
        np.testing.assert_allclose(pack['average_dark_current'], .25)
        np.testing.assert_array_equal(pack['wave_map'], wave)
        assert pack['meta_file_emicorr'] == 'emicorr.asdf'


def _valid_miri_pack():
    """Return valid MIRI pack."""
    shape = (2, 4)
    return {
        'mask_dq': np.zeros(shape, np.uint32),
        'emicorr_frequencies': np.asarray([390.625]),
        'emicorr_reference_waves': np.asarray([[0., 1., 0., -1.]]),
        'emicorr_reference_wave_lengths': np.asarray([4], np.int64),
        'emicorr_rowclocks': np.int64(28),
        'emicorr_frameclocks': np.int64(15904),
        'reset_data': np.zeros((2, 3) + shape, np.float32),
        'reset_dq': np.zeros(shape, np.uint32),
        'lin_coeffs': np.zeros((3,) + shape, np.float32),
        'lin_dq': np.zeros(shape, np.uint32),
        'dark': np.zeros((2, 2) + shape, np.float32),
        'dark_dq': np.zeros(shape, np.uint32),
        'average_dark_current': np.zeros(shape, np.float32),
        'readnoise': np.ones(shape, np.float32),
        'gain': np.ones(shape, np.float32),
        'gain_factor': np.float32(1.),
        'flat': np.ones(shape, np.float32),
        'wave_map': np.ones(shape, np.float64),
    }


def test_validate_miri_refpack_requires_and_checks_miri_payloads():
    """Check validate MIRI refpack requires and checks MIRI payloads."""
    opts = _all_skipped(
        DQInitStep='run', EmiCorrStep='run', ResetStep='run',
        LinearityStep='run', RampFitStep='run', GainScaleStep='run',
        AssignWCSStep='run', FlatFieldStep='run')
    pack = _valid_miri_pack()
    assert refs.validate_refpack(pack, _miri_cube(), opts) is pack

    for key in ('emicorr_frequencies', 'dark', 'wave_map'):
        broken = dict(pack)
        broken.pop(key)
        with pytest.raises(ValueError, match=key):
            refs.validate_refpack(broken, _miri_cube(), opts)

    without_reset = dict(pack)
    without_reset.pop('reset_data')
    without_reset.pop('reset_dq')
    assert refs.validate_refpack(without_reset, _miri_cube(), opts) is without_reset

    broken = dict(pack, dark=np.zeros((2, 2, 4), np.float32))
    with pytest.raises(ValueError, match='dark has shape'):
        refs.validate_refpack(broken, _miri_cube(), opts)
    broken = dict(pack, reset_data=np.zeros((3, 2, 4), np.float32))
    with pytest.raises(ValueError, match='reset_data'):
        refs.validate_refpack(broken, _miri_cube(), opts)
    broken = dict(pack, emicorr_rowclocks=np.float64(28.))
    with pytest.raises(ValueError, match='positive integer scalar'):
        refs.validate_refpack(broken, _miri_cube(), opts)
    broken = dict(pack, emicorr_reference_waves=np.ones((2, 4)))
    with pytest.raises(ValueError, match='one row per frequency'):
        refs.validate_refpack(broken, _miri_cube(), opts)
    broken = dict(
        pack, emicorr_reference_wave_lengths=np.asarray([5], np.int64))
    with pytest.raises(ValueError, match='padded wave width'):
        refs.validate_refpack(broken, _miri_cube(), opts)
    broken = dict(
        pack, emicorr_reference_wave_lengths=np.asarray([4.], np.float64))
    with pytest.raises(ValueError, match='integer array'):
        refs.validate_refpack(broken, _miri_cube(), opts)


def test_miri_linearity_requires_linked_dark_unless_disabled():
    """Check MIRI linearity requires linked dark unless disabled."""
    pack = {
        'lin_coeffs': np.zeros((3, 2, 4), np.float32),
        'lin_dq': np.zeros((2, 4), np.uint32),
    }
    opts = _all_skipped(LinearityStep='run')
    with pytest.raises(ValueError, match='dark'):
        refs.validate_refpack(pack, _miri_cube(), opts, require_waves=False)
    opts['miri_subtract_dark'] = False
    assert refs.validate_refpack(
        pack, _miri_cube(), opts, require_waves=False) is pack
