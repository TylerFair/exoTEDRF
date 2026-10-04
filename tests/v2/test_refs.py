"""Check refs."""

import dataclasses
import numpy as np
import pathlib
import pytest
import sys
import types

from astropy.io import fits

from exotedrf.v2 import refs
from exotedrf.v2.core import ObsMeta, RampCube


def _cube():
    """Return cube."""
    meta = ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP96', 2.214, 3,
                   np.arange(2.), np.array([1]), np.array([2]), ('x.fits',))
    return RampCube(np.zeros((2, 3, 4, 8), np.float32),
                    np.zeros((2, 3, 4, 8), np.uint8),
                    np.zeros((4, 8), np.uint32), meta)


def _all_skipped(**updates):
    """Return all skipped."""
    opts = {key: 'skip' for key in (
        'DQInitStep', 'INLCorrStep', 'SuperBiasStep', 'RefPixStep',
        'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep', 'JumpStep',
        'RampFitStep', 'GainScaleStep', 'AssignWCSStep', 'SourceTypeStep',
        'FlatFieldStep', 'BackgroundStep', 'OneOverFStep_int', 'BadPixStep',
        'PCAReconstructStep')}
    opts.update(updates)
    return opts


def test_refpack_roundtrip(tmp_path):
    """Check refpack roundtrip."""
    p = tmp_path / 'refpack.npz'
    np.savez_compressed(
        p,
        superbias=np.ones((16, 32), np.float32),
        lin_coeffs=np.zeros((5, 16, 32), np.float32),
        mask_dq=np.zeros((16, 32), np.uint32),
        meta_context='jwst_test.pmap')
    rp = refs.RefPack(str(p))
    assert 'superbias' in rp
    assert 'gain' not in rp
    assert rp.get('gain') is None
    assert rp['lin_coeffs'].shape == (5, 16, 32)


def test_cut_to_subarray():
    """Check cut to subarray."""
    from astropy.io import fits
    hdr = fits.Header()
    hdr['SUBSTRT1'], hdr['SUBSTRT2'] = 1, 1793
    hdr['SUBSIZE1'], hdr['SUBSIZE2'] = 2048, 256
    full = np.arange(2048 * 2048, dtype=np.float32).reshape(2048, 2048)
    cut = refs._cut_to_subarray(full, hdr)
    assert cut.shape == (256, 2048)
    np.testing.assert_array_equal(cut, full[1792:2048, :])
    # Already-cut refs pass through untouched.
    np.testing.assert_array_equal(refs._cut_to_subarray(cut, hdr), cut)


def test_cut_to_subarray_translates_partial_reference_origin():
    """Check cut to subarray translates partial reference origin."""
    from astropy.io import fits

    science = fits.Header({
        'SUBSTRT1': 1, 'SUBSTRT2': 946,
        'SUBSIZE1': 8, 'SUBSIZE2': 2,
    })
    reference = fits.Header({
        'SUBSTRT1': 1, 'SUBSTRT2': 895,
        'SUBSIZE1': 8, 'SUBSIZE2': 60,
    })
    coeffs = np.arange(3 * 60 * 8).reshape(3, 60, 8)
    dq = np.arange(60 * 8).reshape(60, 8)

    np.testing.assert_array_equal(
        refs._cut_to_subarray(coeffs, science, reference),
        coeffs[:, 51:53])
    np.testing.assert_array_equal(
        refs._cut_to_subarray(dq, science, reference), dq[51:53])


def test_build_refpack_slices_payload_and_companion_from_reference_origin(
        tmp_path, monkeypatch):
    """Check build refpack slices payload and companion from reference origin."""
    from astropy.io import fits

    science_header = fits.Header({
        'INSTRUME': 'FGS', 'DETECTOR': 'GUIDER1', 'SUBARRAY': 'TEST',
        'READPATT': 'FGSRAPID', 'SUBSTRT1': 3, 'SUBSTRT2': 4,
        'SUBSIZE1': 3, 'SUBSIZE2': 2, 'NGROUPS': 2,
    })
    uncal = tmp_path / 'uncal.fits'
    fits.HDUList([
        fits.PrimaryHDU(header=science_header),
        fits.ImageHDU(np.zeros((1, 2, 2, 3), np.float32), name='SCI'),
    ]).writeto(uncal)

    reference_header = fits.Header({
        'SUBARRAY': 'GENERIC', 'SUBSTRT1': 2, 'SUBSTRT2': 2,
        'SUBSIZE1': 6, 'SUBSIZE2': 5,
    })
    coeffs = np.arange(3 * 5 * 6, dtype=np.float32).reshape(3, 5, 6)
    dq = np.arange(5 * 6, dtype=np.uint32).reshape(5, 6)
    linearity = tmp_path / 'linearity.fits'
    fits.HDUList([
        fits.PrimaryHDU(header=reference_header),
        fits.ImageHDU(coeffs, name='COEFFS'),
        fits.ImageHDU(dq, name='DQ'),
    ]).writeto(linearity)
    monkeypatch.setitem(
        sys.modules, 'crds',
        types.SimpleNamespace(getreferences=lambda *a, **k: {
            'linearity': str(linearity)}))

    out = refs.build_refpack(
        uncal, out=tmp_path / 'refpack.npz', reftypes=['linearity'],
        include_inl=False)

    with np.load(out) as pack:
        np.testing.assert_array_equal(
            pack['lin_coeffs'], coeffs[:, 2:4, 1:4])
        np.testing.assert_array_equal(pack['lin_dq'], dq[2:4, 1:4])


def test_crds_parameter_mapping():
    """Check CRDS parameter mapping."""
    from astropy.io import fits
    hdr = fits.Header()
    hdr['INSTRUME'] = 'NIRISS'
    hdr['DETECTOR'] = 'NIS'
    hdr['SUBARRAY'] = 'SUBSTRIP256'
    params = refs._crds_parameters(hdr)
    assert params['META.INSTRUMENT.NAME'] == 'NIRISS'
    assert params['META.SUBARRAY.NAME'] == 'SUBSTRIP256'
    assert 'META.INSTRUMENT.GRATING' not in params


def _install_fake_wcs_modules(monkeypatch, wave):
    """Install the small JWST surface used by builder-only WCS helpers."""
    calls = []

    class Exposure:
        type = None

    class Model:
        def __init__(self, data=None):
            self.data = data
            self.meta = types.SimpleNamespace(exposure=Exposure())
            self.wavelength = np.asarray(wave)

        def update(self, other):
            del other

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class AssignWcsStep:
        @staticmethod
        def call(model, **kwargs):
            bbox = sys.modules['jwst.assign_wcs.nirspec'].generate_compound_bbox
            calls.append(('assign', kwargs))
            if getattr(bbox, 'keywords', None):
                calls.append(('bbox', bbox.keywords))
            return model

    class Extract2dStep:
        @staticmethod
        def call(model, **kwargs):
            calls.append(('extract_2d', kwargs))
            return model

    class WavecorrStep:
        @staticmethod
        def call(model, **kwargs):
            calls.append(('wavecorr', kwargs))
            return model

    jwst = types.ModuleType('jwst')
    jwst.__path__ = []
    jwst.__version__ = '3.0.0'
    jwst.datamodels = types.SimpleNamespace(
        open=lambda path: Model(), CubeModel=Model)
    assign_wcs = types.ModuleType('jwst.assign_wcs')
    assign_wcs.__path__ = []
    assign_wcs.AssignWcsStep = AssignWcsStep
    nirspec = types.ModuleType('jwst.assign_wcs.nirspec')
    def nrs_wcs_set_input(model, slit_name):
        calls.append(('nrs_wcs_set_input', slit_name))
        return lambda x, y: (x, y, np.asarray(wave)[y, x])

    def generate_compound_bbox(*args, **kwargs):
        return None

    nirspec.nrs_wcs_set_input = nrs_wcs_set_input
    nirspec.generate_compound_bbox = generate_compound_bbox
    extract_2d = types.ModuleType('jwst.extract_2d')
    extract_2d.Extract2dStep = Extract2dStep
    wavecorr = types.ModuleType('jwst.wavecorr')
    wavecorr.WavecorrStep = WavecorrStep
    jwst.assign_wcs = assign_wcs
    monkeypatch.setitem(sys.modules, 'jwst', jwst)
    monkeypatch.setitem(sys.modules, 'jwst.assign_wcs', assign_wcs)
    monkeypatch.setitem(sys.modules, 'jwst.assign_wcs.nirspec', nirspec)
    monkeypatch.setitem(sys.modules, 'jwst.extract_2d', extract_2d)
    monkeypatch.setitem(sys.modules, 'jwst.wavecorr', wavecorr)
    return calls, nirspec


def test_compute_nirspec_wave_map_matches_v1_wcs_overrides(monkeypatch):
    """Check compute NIRSpec wave map matches v1 wcs overrides."""
    wave = np.arange(12, dtype=float).reshape(3, 4)
    calls, nirspec = _install_fake_wcs_modules(monkeypatch, wave)

    actual = refs._compute_nirspec_wave_map('nrs_uncal.fits', wave.shape)

    np.testing.assert_array_equal(actual, wave)
    assert calls == [
        ('assign', {'slit_y_low': -50, 'slit_y_high': 50}),
        ('bbox', {'wavelength_range': [6e-08, 6e-06]}),
        ('nrs_wcs_set_input', 'S1600A1'),
    ]
    assert not hasattr(nirspec.generate_compound_bbox, 'keywords')


def test_compute_miri_wave_map_matches_v1_slit_bounds(monkeypatch):
    """Check compute MIRI wave map matches v1 slit bounds."""
    wave = np.arange(12, dtype=float).reshape(3, 4)
    calls, _ = _install_fake_wcs_modules(monkeypatch, wave)

    actual = refs._compute_miri_wave_map('miri_uncal.fits', wave.shape)

    np.testing.assert_array_equal(actual, wave)
    assert calls == [
        ('assign', {'slit_y_low': -0.55, 'slit_y_high': 0.55})]


def test_adapt_dark_reference_reconstructs_science_groups():
    """Check adapt dark reference reconstructs science groups."""
    from astropy.io import fits

    hdr = fits.Header({'NGROUPS': 2, 'NFRAMES': 2, 'GROUPGAP': 1})
    dark_hdr = fits.Header({'NFRAMES': 1, 'GROUPGAP': 0})
    frames = np.arange(5, dtype=np.float32)[:, None, None]
    adapted, status, reason = refs._adapt_dark_reference(
        frames, hdr, dark_hdr)
    np.testing.assert_allclose(adapted[:, 0, 0], [0.5, 3.5])
    assert status == 'COMPLETE'
    assert reason == ''

    integrations = np.stack([frames, frames + 10], axis=0)
    with pytest.raises(ValueError, match='must be 3D'):
        refs._adapt_dark_reference(integrations, hdr, dark_hdr)


def test_adapt_dark_reference_direct_extrapolation_and_skip_paths():
    """Check adapt dark reference direct extrapolation and skip paths."""
    from astropy.io import fits

    hdr = fits.Header({'NGROUPS': 2, 'NFRAMES': 1, 'GROUPGAP': 0})
    dark_hdr = fits.Header({'NFRAMES': 1, 'GROUPGAP': 0})
    direct = np.arange(3, dtype=np.float32)[:, None, None]
    adapted, status, _ = refs._adapt_dark_reference(
        direct, hdr, dark_hdr)
    np.testing.assert_array_equal(
        adapted, direct[:2])
    assert status == 'COMPLETE'

    direct[0, 0, 0] = np.nan
    adapted, status, _ = refs._adapt_dark_reference(
        direct, hdr, dark_hdr)
    assert adapted[0, 0, 0] == 0.
    assert status == 'COMPLETE'

    # STCAL 1.20 extends the final dark difference before averaging.
    long_science = fits.Header(
        {'NGROUPS': 2, 'NFRAMES': 2, 'GROUPGAP': 1})
    neutral, status, reason = refs._adapt_dark_reference(
        direct, long_science, dark_hdr)
    np.testing.assert_array_equal(neutral[:, 0, 0], [0.5, 3.5])
    assert status == 'COMPLETE'
    assert reason == ''

    # It also skips when the reference's grouping values exceed science.
    larger_dark_groups = fits.Header({'NFRAMES': 2, 'GROUPGAP': 1})
    neutral, status, reason = refs._adapt_dark_reference(
        np.zeros((2, 1, 1), np.float32), hdr, larger_dark_groups)
    np.testing.assert_array_equal(neutral, 0.)
    assert status == 'SKIPPED'
    assert reason == 'dark_readout_exceeds_science'

    spatial = np.asarray([[4.]], np.float32)
    with pytest.raises(ValueError, match='must be 3D'):
        refs._adapt_dark_reference(spatial, hdr, dark_hdr)
    with pytest.raises(ValueError, match='must provide NFRAMES'):
        refs._adapt_dark_reference(direct, hdr, fits.Header())


def test_build_refpack_preserves_reference_dq_and_flat_err(tmp_path,
                                                          monkeypatch):
    """Check build refpack preserves reference DQ and flat ERR."""
    from astropy.io import fits

    shape = (2, 3)
    hdr = fits.Header()
    hdr['INSTRUME'] = 'FGS'
    hdr['DETECTOR'] = 'GUIDER1'
    hdr['SUBARRAY'] = 'FULL'
    hdr['READPATT'] = 'FGSRAPID'
    hdr['SUBSTRT1'] = hdr['SUBSTRT2'] = 1
    hdr['SUBSIZE1'], hdr['SUBSIZE2'] = shape[1], shape[0]
    hdr['NGROUPS'] = 2
    hdr['NFRAMES'] = 1
    hdr['GROUPGAP'] = 0
    hdr['GAINFACT'] = 9.0
    uncal = tmp_path / 'uncal.fits'
    fits.HDUList([
        fits.PrimaryHDU(header=hdr),
        fits.ImageHDU(np.zeros((1, 2) + shape, np.float32), name='SCI'),
    ]).writeto(uncal)

    def write_ref(name, sci_ext, sci, companions, primary_header=None):
        hdus = [fits.PrimaryHDU(header=primary_header),
                fits.ImageHDU(sci, name=sci_ext)]
        hdus.extend(fits.ImageHDU(value, name=ext)
                    for ext, value in companions)
        path = tmp_path / f'{name}.fits'
        fits.HDUList(hdus).writeto(path)
        return str(path)

    dq = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
    paths = {
        'superbias': write_ref(
            'superbias', 'SCI', np.ones(shape, np.float32), [('DQ', dq)]),
        'linearity': write_ref(
            'linearity', 'COEFFS', np.ones((3,) + shape, np.float32),
            [('DQ', dq + 10),
             ('INV_COEFFS', np.full((3,) + shape, .5, np.float32))]),
        'dark': write_ref(
            'dark', 'SCI', np.ones((2,) + shape, np.float32),
            [('DQ', dq + 20),
             ('AVDRKCUR', np.full(shape, .2, np.float32))],
            fits.Header({'NFRAMES': 1, 'GROUPGAP': 0,
                         'AVDRKCUR': 9.})),
        'flat': write_ref(
            'flat', 'SCI', np.ones(shape, np.float32),
            [('ERR', np.full(shape, .02, np.float32)), ('DQ', dq + 30)]),
        'gain': write_ref(
            'gain', 'SCI', np.full(shape, 1.6, np.float32), [],
            fits.Header({'GAINFACT': 1.234})),
    }
    fake_crds = types.SimpleNamespace(
        getreferences=lambda *args, **kwargs: paths)
    monkeypatch.setitem(sys.modules, 'crds', fake_crds)
    out = refs.build_refpack(
        uncal, out=tmp_path / 'refpack.npz', reftypes=list(paths),
        include_inl=False)

    with np.load(out) as pack:
        np.testing.assert_array_equal(pack['superbias_dq'], dq)
        np.testing.assert_array_equal(pack['lin_dq'], dq + 10)
        np.testing.assert_array_equal(pack['lin_inv_coeffs'], .5)
        np.testing.assert_array_equal(pack['dark_dq'], dq + 20)
        assert pack['meta_dark_status'] == 'COMPLETE'
        np.testing.assert_allclose(pack['average_dark_current'], .2)
        np.testing.assert_array_equal(pack['flat_dq'], dq + 30)
        np.testing.assert_allclose(pack['flat_err'], .02)
        np.testing.assert_allclose(pack['gain_factor'], np.float32(1.234))
        assert 'gain' in str(pack['meta_gain_factor_source']).lower()

    paths['dark'] = write_ref(
        'dark_zero_average', 'SCI', np.ones((2,) + shape, np.float32),
        [('DQ', dq + 20),
         ('AVDRKCUR', np.zeros(shape, np.float32))],
        fits.Header({'NFRAMES': 1, 'GROUPGAP': 0, 'AVDRKCUR': .3}))
    out2 = refs.build_refpack(
        uncal, out=tmp_path / 'refpack_scalar_dark.npz',
        reftypes=list(paths), include_inl=False)
    with np.load(out2) as pack:
        np.testing.assert_allclose(pack['average_dark_current'], .3)


def test_build_refpack_extrapolates_short_dark(tmp_path,
                                                          monkeypatch):
    """Check build refpack extrapolates short dark."""
    from astropy.io import fits

    shape = (2, 3)
    sci_header = fits.Header({
        'INSTRUME': 'FGS', 'DETECTOR': 'GUIDER1', 'SUBARRAY': 'FULL',
        'READPATT': 'FGSRAPID', 'SUBSTRT1': 1, 'SUBSTRT2': 1,
        'SUBSIZE1': shape[1], 'SUBSIZE2': shape[0], 'NGROUPS': 3,
        'NFRAMES': 1, 'GROUPGAP': 0,
    })
    uncal = tmp_path / 'uncal.fits'
    fits.HDUList([
        fits.PrimaryHDU(header=sci_header),
        fits.ImageHDU(np.zeros((1, 3) + shape, np.float32), name='SCI'),
    ]).writeto(uncal)

    dark = tmp_path / 'dark.fits'
    dark_header = fits.Header({'NFRAMES': 1, 'GROUPGAP': 0})
    fits.HDUList([
        fits.PrimaryHDU(header=dark_header),
        fits.ImageHDU(np.ones((2,) + shape, np.float32), name='SCI'),
        fits.ImageHDU(np.full(shape, 2048, np.uint32), name='DQ'),
        fits.ImageHDU(np.full(shape, .2, np.float32), name='AVDRKCUR'),
    ]).writeto(dark)
    monkeypatch.setitem(
        sys.modules, 'crds',
        types.SimpleNamespace(getreferences=lambda *a, **k: {'dark': str(dark)}))

    out = refs.build_refpack(
        uncal, out=tmp_path / 'refpack.npz', reftypes=['dark'],
        include_inl=False)
    with np.load(out) as pack:
        assert pack['meta_dark_status'] == 'COMPLETE'
        np.testing.assert_array_equal(pack['dark'], 1.)
        np.testing.assert_array_equal(pack['dark_dq'], 2048)
        np.testing.assert_allclose(pack['average_dark_current'], .2)


def test_validate_refpack_rejects_missing_enabled_reference_early():
    """Check validate refpack rejects missing enabled reference early."""
    with pytest.raises(ValueError, match='dark'):
        refs.validate_refpack({}, _cube(),
                              _all_skipped(DarkCurrentStep='run'),
                              require_waves=False)


def test_soss_dqinit_requires_mask_but_not_crds_saturation_reference():
    """Check SOSS DQ initialization requires mask but not CRDS saturation reference."""
    opts = _all_skipped(DQInitStep='run')
    pack = {'mask_dq': np.zeros((4, 8), np.uint32)}
    assert refs.validate_refpack(
        pack, _cube(), opts, require_waves=False) is pack
    assert 'saturation' not in refs.REFTYPES
    with pytest.raises(ValueError, match='mask_dq'):
        refs.validate_refpack({}, _cube(), opts, require_waves=False)


def test_validate_refpack_checks_inl_and_dark_shapes():
    """Check validate refpack checks inverse linearity and dark shapes."""
    pack = {
        'inl_theta': np.arange(6, dtype=np.float32),
        'inl_periods': np.asarray([1, 2, 3], np.float32),
        'dark': np.zeros((3, 4, 8), np.float32),
        'average_dark_current': np.zeros((4, 8), np.float32),
    }
    opts = _all_skipped(INLCorrStep='run', DarkCurrentStep='run')
    assert refs.validate_refpack(pack, _cube(), opts,
                                 require_waves=False) is pack
    pack['dark'] = np.zeros((2, 4, 8), np.float32)
    with pytest.raises(ValueError, match='dark has shape'):
        refs.validate_refpack(pack, _cube(), opts, require_waves=False)

    pack['dark'] = np.zeros((3, 4, 8), np.float32)
    pack['inl_periods'] = np.asarray([1, 0, 3], np.float32)
    with pytest.raises(ValueError, match='invalid INL'):
        refs.validate_refpack(pack, _cube(), opts, require_waves=False)


def test_validate_refpack_enforces_neutral_skipped_dark():
    """Check validate refpack enforces neutral skipped dark."""
    opts = _all_skipped(DarkCurrentStep='run')
    pack = {
        'dark': np.zeros((3, 4, 8), np.float32),
        'dark_dq': np.zeros((4, 8), np.uint32),
        'average_dark_current': np.zeros((4, 8), np.float32),
        'meta_dark_status': 'SKIPPED',
    }
    assert refs.validate_refpack(
        pack, _cube(), opts, require_waves=False) is pack
    pack['dark_dq'][0, 0] = 2048
    with pytest.raises(ValueError, match='neutral zero'):
        refs.validate_refpack(pack, _cube(), opts, require_waves=False)


def test_step_controls_must_be_exact_lowercase():
    """Check step controls must be exact lowercase."""
    with pytest.raises(ValueError, match="exactly lowercase"):
        refs.validate_refpack({}, _cube(),
                              _all_skipped(DQInitStep='SKIP'),
                              require_waves=False)


def test_reference_dq_array_restores_o_bzero_and_maps_dq_def():
    # MIRI flat references preserve the unsigned FITS offset as O_BZERO.
    """Check reference DQ array restores o bzero and maps DQ def."""
    dq_hdu = fits.ImageHDU(
        np.asarray([[-2147483648, -2147483583]], dtype='>i4'),
        name='DQ')
    dq_hdu.header['O_BZERO'] = 2147483648
    definitions = np.asarray([
        (0, 1, 'DO_NOT_USE', 'bad'),
        (6, 64, 'NO_FLAT_FIELD', 'missing flat'),
    ], dtype=[
        ('Bit', np.int16), ('Value', np.uint32), ('Name', 'U40'),
        ('Description', 'U80'),
    ])
    dq_def_hdu = fits.BinTableHDU(definitions, name='DQ_DEF')

    mapped = refs._reference_dq_array(dq_hdu, dq_def_hdu)

    assert mapped.dtype == np.uint32
    np.testing.assert_array_equal(
        mapped, np.asarray([[0, 1 | 262144]], np.uint32))


def test_default_refpack_cache_is_visit_and_output_tag_specific(tmp_path):
    """Check default refpack cache is visit and output tag specific."""
    first = _cube()
    second_meta = dataclasses.replace(
        first.meta, filenames=('/data/jw00002_uncal.fits',))
    first = dataclasses.replace(
        first, meta=dataclasses.replace(
            first.meta, filenames=('/data/jw00001_uncal.fits',)))
    second = dataclasses.replace(first, meta=second_meta)
    opts = {
        'output_dir': str(tmp_path / 'pipeline_outputs_directory'),
        'output_tag': 'visit-a',
        'crds_context': 'jwst_1322.pmap',
    }

    path1 = refs.default_refpack_path(first, opts)
    path2 = refs.default_refpack_path(second, opts)
    assert path1 != path2
    assert path1.parent == (
        tmp_path / 'pipeline_outputs_directory_visit-a' / 'v2' / 'refpacks')
    assert f'refpack_v2s{refs.REFPACK_SCHEMA_VERSION}_' in path1.name
    assert 'jw00001_uncal' in path1.name
    assert 'jw00002_uncal' in path2.name


def test_explicit_stale_refpack_requires_rebuild(tmp_path):
    """Check explicit stale refpack requires rebuild."""
    stale = tmp_path / 'stale.npz'
    np.savez_compressed(stale, gain_factor=np.float32(9.))
    with pytest.raises(ValueError, match='meta_schema_version'):
        refs.resolve_refpack(_cube(), {}, refpack=stale)


def test_stale_auto_cache_is_rebuilt(tmp_path, monkeypatch):
    """Check stale auto cache is rebuilt."""
    cube = _cube()
    opts = {
        'output_dir': str(tmp_path / 'outputs'),
        'INLCorrStep': 'skip',
    }
    cache = refs.default_refpack_path(cube, opts)
    cache.parent.mkdir(parents=True)
    np.savez_compressed(cache, gain_factor=np.float32(9.))
    calls = []

    def fake_build(uncal_file, context=None, out=None, **kwargs):
        calls.append((uncal_file, pathlib.Path(out)))
        np.savez_compressed(
            out,
            meta_schema_version=np.int32(refs.REFPACK_SCHEMA_VERSION),
            gain_factor=np.float32(1.234))
        return str(out)

    monkeypatch.setattr(refs, 'build_refpack', fake_build)
    pack = refs.resolve_refpack(cube, opts)
    try:
        assert calls == [('x.fits', cache)]
        np.testing.assert_allclose(pack['gain_factor'], 1.234)
    finally:
        pack.close()


def test_validate_refpack_rejects_explicit_wrong_crds_context():
    """Check validate refpack rejects explicit wrong CRDS context."""
    opts = _all_skipped(crds_context='jwst_1322.pmap')
    matching = {'meta_context': 'jwst_1322.pmap'}
    assert refs.validate_refpack(
        matching, _cube(), opts, require_waves=False) is matching

    with pytest.raises(ValueError, match='meta_context'):
        refs.validate_refpack(
            {'meta_context': 'jwst_9999.pmap'}, _cube(), opts,
            require_waves=False)

    for legacy in ({}, {'meta_context': 'default'}):
        assert refs.validate_refpack(
            legacy, _cube(), opts, require_waves=False) is legacy


def test_ensure_tls_certificates_exports_first_existing_bundle(tmp_path):
    """Check ensure tls certificates exports first existing bundle."""
    missing = tmp_path / 'missing.pem'
    bundle = tmp_path / 'bundle.pem'
    bundle.write_text('dummy')
    env = {}
    paths = types.SimpleNamespace(cafile=None, capath=None)
    chosen = refs.ensure_tls_certificates(
        environ=env, verify_paths=paths, candidates=[missing, bundle])
    assert chosen == str(bundle)
    assert env['SSL_CERT_FILE'] == str(bundle)


def test_ensure_tls_certificates_respects_existing_configuration(tmp_path):
    """Check ensure tls certificates respects existing configuration."""
    bundle = tmp_path / 'bundle.pem'
    bundle.write_text('dummy')
    # An explicit user override is never touched.
    env = {'SSL_CERT_FILE': '/user/choice.pem'}
    paths = types.SimpleNamespace(cafile=None, capath=None)
    assert refs.ensure_tls_certificates(
        environ=env, verify_paths=paths, candidates=[bundle]) is None
    assert env['SSL_CERT_FILE'] == '/user/choice.pem'
    # A working OpenSSL default needs no help.
    env = {}
    paths = types.SimpleNamespace(cafile='/etc/ssl/cert.pem', capath=None)
    assert refs.ensure_tls_certificates(
        environ=env, verify_paths=paths, candidates=[bundle]) is None
    assert 'SSL_CERT_FILE' not in env
