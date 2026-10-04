"""Check io trace."""

import numpy as np
import pytest
from astropy.io import fits

from exotedrf.v2 import io, trace
from exotedrf.v2.core import ObsMeta, RateCube


def _write_uncal(path, nints, seg, ngroups=3, dimy=16, dimx=32):
    """Return write uncal."""
    rng = np.random.RandomState(seg)
    data = rng.uniform(0, 100, (nints, ngroups, dimy, dimx)).astype(np.uint16)
    ph = fits.PrimaryHDU()
    ph.header['INSTRUME'] = 'NIRISS'
    ph.header['DETECTOR'] = 'NIS'
    ph.header['SUBARRAY'] = 'SUBSTRIP256'
    ph.header['TGROUP'] = 5.494
    times = fits.BinTableHDU.from_columns(fits.ColDefs([fits.Column(
        name='int_mid_BJD_TDB', format='D',
        array=np.arange(nints, dtype=float) + 100 * seg)]), name='INT_TIMES')
    fits.HDUList([ph, fits.ImageHDU(data, name='SCI'), times]).writeto(path)
    return data


def test_load_ramp_cube_concatenates_segments(tmp_path):
    """Check load ramp cube concatenates segments."""
    d1 = _write_uncal(tmp_path / 'obs-seg001_uncal.fits', 4, 1)
    d2 = _write_uncal(tmp_path / 'obs-seg002_uncal.fits', 3, 2)
    files = io.find_segments(str(tmp_path))
    assert [f.split('seg')[1][:3] for f in files] == ['001', '002']
    cube = io.load_ramp_cube(files, baseline_ints=[2])
    assert not isinstance(cube.data, np.memmap)
    assert not isinstance(cube.groupdq, np.memmap)
    assert cube.meta.extra['host_storage'] == 'memory'
    assert cube.data.shape == (7, 3, 16, 32)
    np.testing.assert_allclose(cube.data[:4], d1.astype(np.float32))
    np.testing.assert_allclose(cube.data[4:], d2.astype(np.float32))
    assert cube.meta.segment_edges.tolist() == [4, 7]
    assert cube.meta.segment_int_starts.tolist() == [1, 5]
    assert cube.meta.frame_time == pytest.approx(5.494)
    assert cube.meta.int_times.shape == (7,)


@pytest.mark.parametrize('lazy', [False, True])
def test_empty_groupdq_and_single_integration_scalar_read(tmp_path, lazy):
    """Check empty GROUPDQ and single integration scalar read."""
    path = tmp_path / 'single_uncal.fits'
    data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    fits.HDUList([
        fits.PrimaryHDU(), fits.ImageHDU(data, name='SCI'),
        fits.ImageHDU(name='GROUPDQ'),
    ]).writeto(path)
    cube = io.load_ramp_cube([path], [1], lazy=lazy)
    assert cube.data.shape == (1, 2, 3, 4)
    assert cube.data[0, 1, 2, 3] == data[1, 2, 3]
    np.testing.assert_array_equal(cube.data[:], data[None].astype(np.float32))
    np.testing.assert_array_equal(cube.groupdq[:], np.zeros_like(data)[None])


def test_lazy_reader_keeps_explicit_budget_and_rejects_fancy_slices(tmp_path):
    """Check lazy reader keeps explicit budget and rejects fancy slices."""
    path = tmp_path / 'obs_uncal.fits'
    data = _write_uncal(path, 2, 1)
    scratch = tmp_path / 'scratch'
    cube = io.load_ramp_cube([path], [1], lazy=True, max_host_bytes=1,
                             scratch_dir=scratch)
    selected = cube.data[:]
    assert isinstance(selected, np.memmap)
    assert scratch.is_dir()
    np.testing.assert_array_equal(selected, data.astype(np.float32))
    with pytest.raises(TypeError, match='trailing selections'):
        cube.data[:, [0, 1], :, :]


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_lazy_fits_arrays_match_eager_scaled_data_without_scratch(
        tmp_path, monkeypatch, dtype):
    """Check lazy FITS arrays match eager scaled data without scratch."""
    files = [tmp_path / 'seg001_uncal.fits', tmp_path / 'seg002_uncal.fits']
    for index, path in enumerate(files):
        _write_uncal(path, 3 + index, index + 1)
    # Exercise nonzero group DQ in one segment and absent GROUPDQ in another.
    with fits.open(files[1], mode='append') as h:
        dq = np.zeros(h['SCI'].shape, np.uint8)
        dq[1, 2, 3, 4] = 5
        h.append(fits.ImageHDU(dq, name='GROUPDQ'))
    eager = io.load_ramp_cube(files, [2, -2], science_dtype=dtype)

    def forbidden(*args, **kwargs):
        raise AssertionError('lazy reader allocated a whole-visit array')

    monkeypatch.setattr(io, '_allocate_host_array', forbidden)
    lazy = io.load_ramp_cube(files, [2, -2], science_dtype=dtype, lazy=True)
    assert lazy.meta.extra['host_storage'] == 'fits'
    for key in ('data', 'groupdq'):
        left, right = getattr(eager, key), getattr(lazy, key)
        assert right.dtype == left.dtype
        for selection in (slice(0, 3), slice(2, 5), -1,
                          (slice(1, 6), -1, slice(2, 4), slice(3, 5)),
                          slice(0, 0)):
            np.testing.assert_array_equal(left[selection], right[selection])
        with pytest.raises(TypeError, match='bounded'):
            np.asarray(right)
    from exotedrf.v2.optimize import _first_segment_state
    from exotedrf.v2.pipeline import PipelineState
    first = _first_segment_state(PipelineState(lazy))
    np.testing.assert_array_equal(first.cube.data, eager.data[:3])


def test_load_ramp_cube_forced_spill_fills_segments_in_order(
        tmp_path, monkeypatch):
    """Check load ramp cube forced spill fills segments in order."""
    scratch = tmp_path / 'scratch'
    monkeypatch.setenv('EXOTEDRF_SCRATCH_DIR', str(scratch))
    d1 = _write_uncal(tmp_path / 'obs-seg001_uncal.fits', 2, 1,
                      ngroups=2, dimy=3, dimx=4)
    d2 = _write_uncal(tmp_path / 'obs-seg002_uncal.fits', 3, 2,
                      ngroups=2, dimy=3, dimx=4)
    files = io.find_segments(tmp_path)

    cube = io.load_ramp_cube(
        files, baseline_ints=[1], max_host_bytes=1)

    assert isinstance(cube.data, np.memmap)
    assert isinstance(cube.groupdq, np.memmap)
    assert cube.meta.extra['host_storage'] == 'memmap'
    assert cube.meta.segment_edges.tolist() == [2, 5]
    np.testing.assert_array_equal(cube.data[:2], d1.astype(np.float32))
    np.testing.assert_array_equal(cube.data[2:], d2.astype(np.float32))
    np.testing.assert_array_equal(cube.groupdq, 0)
    assert scratch.is_dir()
    assert list(scratch.iterdir()) == []


def test_estimate_ramp_storage_and_explicit_small_resident_path(tmp_path):
    """Check estimate ramp storage and explicit small resident path."""
    path = tmp_path / 'obs-seg001_uncal.fits'
    expected = _write_uncal(
        path, 2, 1, ngroups=3, dimy=4, dimx=5)
    samples = expected.size

    assert io.estimate_ramp_storage_bytes(
        [str(path)], np.float64) == samples * (8 + 1)
    cube = io.load_ramp_cube(
        [str(path)], baseline_ints=[1], max_host_bytes='1MiB')
    assert type(cube.data) is np.ndarray
    assert type(cube.groupdq) is np.ndarray
    np.testing.assert_array_equal(cube.data, expected.astype(np.float32))


def test_find_segments_filters_headers_and_sorts_exsegnum(tmp_path):
    """Check find segments filters headers and sorts exsegnum."""
    late = tmp_path / 'good-seg001_uncal.fits'
    early = tmp_path / 'good-seg099_uncal.fits'
    wrong = tmp_path / 'wrong-seg002_uncal.fits'
    _write_uncal(late, 1, 1)
    _write_uncal(early, 1, 2)
    _write_uncal(wrong, 1, 3)
    for path, exseg in ((late, 2), (early, 1)):
        fits.setval(path, 'EXP_TYPE', value='NIS_SOSS')
        fits.setval(path, 'FILTER', value='CLEAR')
        fits.setval(path, 'EXSEGNUM', value=exseg)
    fits.setval(wrong, 'INSTRUME', value='NIRSPEC')
    fits.setval(wrong, 'EXP_TYPE', value='NRS_BRIGHTOBJ')

    found = io.find_segments(tmp_path, mode='NIRISS/SOSS',
                             filter_detector='CLEAR')
    assert [str(early), str(late)] == found
    with pytest.raises(ValueError, match='INSTRUME'):
        io.load_ramp_cube([str(wrong)], [1], mode='NIRISS/SOSS',
                          filter_detector='CLEAR')


def test_nirspec_discovery_rejects_non_bright_object_exposures(tmp_path):
    """The v2 NIRSpec graph is the BOTS/NRS_BRIGHTOBJ workflow only."""
    path = tmp_path / 'nrs-seg001_uncal.fits'
    data = np.zeros((1, 3, 8, 16), np.uint16)
    ph = fits.PrimaryHDU()
    ph.header['INSTRUME'] = 'NIRSPEC'
    ph.header['DETECTOR'] = 'NRS1'
    ph.header['GRATING'] = 'G395H'
    ph.header['EXP_TYPE'] = 'NRS_FIXEDSLIT'
    fits.HDUList([ph, fits.ImageHDU(data, name='SCI')]).writeto(path)

    assert io.find_segments(
        tmp_path, mode='NIRSpec/G395H', filter_detector='NRS1') == []
    with pytest.raises(ValueError, match='NRS_BRIGHTOBJ'):
        io.load_ramp_cube(
            [str(path)], [1], mode='NIRSpec/G395H',
            filter_detector='NRS1')

    fits.delval(path, 'EXP_TYPE')
    assert io.find_segments(
        tmp_path, mode='NIRSpec/G395H', filter_detector='NRS1') == []
    with pytest.raises(ValueError, match='NRS_BRIGHTOBJ'):
        io.load_ramp_cube(
            [str(path)], [1], mode='NIRSpec/G395H',
            filter_detector='NRS1')

    fits.setval(path, 'EXP_TYPE', value='NRS_BRIGHTOBJ')
    assert io.find_segments(
        tmp_path, mode='NIRSpec/G395H', filter_detector='NRS1') == [str(path)]


def test_miri_discovery_rejects_non_lrs_slitless_exposures(tmp_path):
    """Check MIRI discovery rejects non lrs slitless exposures."""
    path = tmp_path / 'miri-seg001_uncal.fits'
    data = np.zeros((1, 3, 8, 16), np.uint16)
    ph = fits.PrimaryHDU()
    ph.header['INSTRUME'] = 'MIRI'
    ph.header['DETECTOR'] = 'MIRIMAGE'
    ph.header['SUBARRAY'] = 'SLITLESSPRISM'
    ph.header['EXP_TYPE'] = 'MIR_IMAGE'
    fits.HDUList([ph, fits.ImageHDU(data, name='SCI')]).writeto(path)

    assert io.find_segments(tmp_path, mode='MIRI/LRS') == []
    with pytest.raises(ValueError, match='MIR_LRS-SLITLESS'):
        io.load_ramp_cube([str(path)], [1], mode='MIRI/LRS')

    fits.setval(path, 'EXP_TYPE', value='MIR_LRS-SLITLESS')
    assert io.find_segments(tmp_path, mode='MIRI/LRS') == [str(path)]


def test_load_ramp_cube_preserves_dq_intstart_and_requested_dtype(tmp_path):
    """Check load ramp cube preserves DQ intstart and requested dtype."""
    path = tmp_path / 'obs-seg001_uncal.fits'
    sci = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
    gdq = np.zeros(sci.shape, np.uint8)
    gdq[1, 2, 3, 4] = 4
    pdq = np.zeros((4, 5), np.uint32)
    pdq[2, 3] = 2048
    ph = fits.PrimaryHDU()
    ph.header['INSTRUME'] = 'NIRISS'
    ph.header['DETECTOR'] = 'NIS'
    ph.header['SUBARRAY'] = 'SUBSTRIP96'
    ph.header['TGROUP'] = 2.214
    ph.header['INTSTART'] = 37
    ph.header['INTEND'] = 38
    fits.HDUList([
        ph, fits.ImageHDU(sci, name='SCI'),
        fits.ImageHDU(gdq, name='GROUPDQ'),
        fits.ImageHDU(pdq, name='PIXELDQ')]).writeto(path)

    cube = io.load_ramp_cube([str(path)], [1], science_dtype=np.float64)
    assert cube.data.dtype == np.float64
    np.testing.assert_array_equal(cube.groupdq, gdq)
    np.testing.assert_array_equal(cube.pixeldq, pdq)
    assert cube.meta.segment_int_starts.tolist() == [37]


def test_save_rate_cube_roundtrip(tmp_path):
    """Check save rate cube roundtrip."""
    meta = ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP256', 5.494, 3,
                   np.arange(5.), np.array([2]), np.array([5]), ('x.fits',))
    rate = RateCube(np.random.rand(5, 16, 32).astype(np.float32),
                    np.random.rand(5, 16, 32).astype(np.float32),
                    np.zeros((5, 16, 32), np.uint32), meta)
    p = tmp_path / 'rate.fits'
    io.save_rate_cube(rate, str(p))
    with fits.open(p) as h:
        np.testing.assert_allclose(h['SCI'].data, np.asarray(rate.data))
        assert h['DQ'].data.dtype == np.uint32 or h['DQ'].data.dtype == '>u4'


@pytest.mark.parametrize('dtype', ['float32', 'float64', '>f4'])
def test_save_rate_cube_streams_readonly_inputs_and_all_dq_bits(tmp_path, dtype):
    """Check save rate cube streams readonly inputs and all DQ bits."""
    shape = (5, 2, 3)
    science = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    science[0, 0, 0] = np.nan
    error = np.sqrt(science).astype(dtype)
    dq = np.resize(np.asarray([0, 1, 2**16, 2**31 - 1, 2**31, 2**32 - 1],
                              dtype=np.uint32), shape)
    expected = [a.copy() for a in (science, error, dq)]
    slices = []

    class SliceOnly:
        def __init__(self, values):
            self.values = values
            self.shape = values.shape
            values.flags.writeable = False

        def __array__(self, *args, **kwargs):
            raise AssertionError('writer materialized the whole input')

        def __getitem__(self, key):
            result = self.values[key]
            assert result.shape[0] <= 2
            slices.append(result.shape[0])
            return result

    meta = ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP256', 5.494, 3,
                   np.arange(5.), np.array([2]), np.array([5]), ('x.fits',))
    rate = RateCube(*(SliceOnly(a) for a in (science, error, dq)), meta)
    path = tmp_path / 'stream.fits'
    io.save_rate_cube(rate, path, extra_header={'OPTIMIZE': True},
                      chunk_bytes=2 * 2 * 3 * 4)
    assert slices == [2, 2, 1] * 3
    with fits.open(path, memmap=False) as hdus:
        hdus.verify('exception')
        assert [h.name for h in hdus] == ['PRIMARY', 'SCI', 'ERR', 'DQ', 'INT_TIMES']
        assert hdus[0].header['OPTIMIZE']
        for name, array in zip(('SCI', 'ERR', 'DQ'), expected):
            dtype_out = np.uint32 if name == 'DQ' else np.float32
            np.testing.assert_array_equal(hdus[name].data,
                                          array.astype(dtype_out))
        assert hdus['DQ'].header['BZERO'] == 2**31
        np.testing.assert_array_equal(hdus['INT_TIMES'].data['int_mid_BJD_TDB'],
                                      meta.int_times)
    for actual, original in zip((science, error, dq), expected):
        np.testing.assert_array_equal(actual, original)


def test_save_rate_cube_preserves_previous_product_on_write_failure(tmp_path,
                                                                   monkeypatch):
    """Check save rate cube preserves previous product on write failure."""
    meta = ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP256', 5.494, 3,
                   np.arange(2.), np.array([1]), np.array([2]), ('x.fits',))
    data = np.ones((2, 2, 3), np.float32)
    rate = RateCube(data, data, np.zeros(data.shape, np.uint32), meta)
    path = tmp_path / 'rate.fits'
    io.save_rate_cube(rate, path)
    previous = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError('simulated write failure')

    monkeypatch.setattr(io, '_stream_rate_image', fail)
    with pytest.raises(OSError, match='simulated write failure'):
        io.save_rate_cube(rate, path)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob('.rate-*.fits'))


def test_centroids_csv_roundtrip(tmp_path):
    """Check centroids csv roundtrip."""
    p = tmp_path / 'c_centroids.csv'
    cols = {'xpos': np.arange(10.), 'ypos o1': np.linspace(5, 6, 10)}
    trace.save_centroids_csv(str(p), cols)
    back = trace.load_centroids_csv(str(p))
    np.testing.assert_allclose(back['ypos o1'], cols['ypos o1'])


def test_centroids_csv_loads_v1_comments_and_blank_optional_cells(tmp_path):
    """Check centroids csv loads v1 comments and blank optional cells."""
    p = tmp_path / 'v1_centroids.csv'
    p.write_text(
        '# File Contents: Edgetrigger trace centroids\n'
        '# File Author: MCR\n'
        'xpos,ypos o1,ypos o2,ypos o3\n'
        '0,87.5,105.3,147.5\n'
        '# Optional orders end before order 1\n'
        '1,87.4,105.2,\n'
        '2,87.3,,\n')

    cols = trace.load_centroids_csv(p)

    np.testing.assert_array_equal(cols['xpos'], [0., 1., 2.])
    np.testing.assert_allclose(cols['ypos o1'], [87.5, 87.4, 87.3])
    np.testing.assert_allclose(cols['ypos o2'][:2], [105.3, 105.2])
    assert np.isnan(cols['ypos o2'][2])
    np.testing.assert_allclose(cols['ypos o3'][:1], [147.5])
    assert np.isnan(cols['ypos o3'][1:]).all()


@pytest.mark.parametrize(
    ('contents', 'message'),
    [
        ('xpos,ypos o1,ypos o2\n0,87.5\n', 'fields; expected'),
        ('xpos,ypos o1\n0,not-a-number\n', 'is not numeric'),
        ('xpos,ypos o1\n0,"unterminated\n', 'malformed centroids CSV'),
    ])
def test_centroids_csv_rejects_malformed_data(tmp_path, contents, message):
    """Check centroids csv rejects malformed data."""
    p = tmp_path / 'bad_centroids.csv'
    p.write_text(contents)
    with pytest.raises(ValueError, match=message):
        trace.load_centroids_csv(p)


def test_fallback_centroiding_recovers_trace():
    """Check fallback centroiding recovers trace."""
    dimy, dimx = 32, 64
    x = np.arange(dimx)
    true_y = 14 + 3 * np.sin(x / 30.0)
    yy = np.arange(dimy)[:, None]
    frame = 50 * np.exp(-0.5 * ((yy - true_y[None, :]) / 1.5) ** 2)
    frame += np.random.RandomState(0).normal(0, 0.05, frame.shape)
    xx, cen = trace.get_centroids_fallback(frame, poly_order=5)
    assert np.nanmax(np.abs(cen[3:-3] - true_y[3:-3])) < 1.0


def test_soss_group_centroiding_falls_back_from_latest_group(monkeypatch):
    """Check SOSS group centroiding falls back from latest group."""
    deepstack = np.stack([
        np.full((5, 4), group, dtype=float) for group in range(3)
    ])
    calls = []

    def fake_centroids(frame, mode, tracetable=None, subarray=None,
                       centroids_csv=None):
        group = int(frame[0, 0])
        calls.append((group, mode, tracetable, subarray))
        if group == 2:
            raise RuntimeError('final group saturated')
        return {
            'xpos': np.arange(frame.shape[-1], dtype=float),
            'ypos o1': np.full(frame.shape[-1], group + 0.5),
        }

    monkeypatch.setattr(trace, 'get_centroids', fake_centroids)
    result = trace.get_soss_centroids_with_group_fallback(
        deepstack, tracetable='spectrace.fits', subarray='SUBSTRIP96')

    assert [call[0] for call in calls] == [2, 1]
    assert all(call[1] == 'NIRISS/SOSS' for call in calls)
    assert calls[-1][2:] == ('spectrace.fits', 'SUBSTRIP96')
    np.testing.assert_array_equal(result['ypos o1'], np.full(4, 1.5))


def test_load_soss_wavemap_vectors_removes_reference_padding(tmp_path):
    """Check load SOSS wavemap vectors removes reference padding."""
    dimx, dimy = 8, 6
    wave1 = np.broadcast_to(np.arange(dimx, dtype=float), (dimy, dimx))
    wave2 = wave1 + 100
    padded1 = np.pad(wave1, ((20, 20), (20, 20)), constant_values=np.nan)
    padded2 = np.pad(wave2, ((20, 20), (20, 20)), constant_values=np.nan)
    path = tmp_path / 'wavemap.fits'
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(padded1),
                  fits.ImageHDU(padded2)]).writeto(path)
    waves = trace.load_soss_wavemap_vectors(path, dimx=dimx)
    np.testing.assert_allclose(waves[1], np.arange(dimx))
    np.testing.assert_allclose(waves[2], np.arange(dimx) + 100)


def test_lazy_read_spills_large_selected_range_without_full_conversion(
        tmp_path, monkeypatch):
    """Check lazy read spills large selected range without full conversion."""
    from exotedrf.v2 import core
    path = tmp_path / 'one_long_segment_uncal.fits'
    expected = _write_uncal(path, 17, 1, ngroups=2, dimy=4, dimx=8)
    cube = io.load_ramp_cube([path], [2, -2], lazy=True)
    with core.host_memory_budget(max_bytes=1, scratch_dir=tmp_path):
        selected = cube.data[2:15]
    assert isinstance(selected, np.memmap)
    np.testing.assert_array_equal(selected, expected[2:15].astype(np.float32))


def test_spectrum_writer_streams_flux_and_preserves_previous_file(tmp_path,
                                                                 monkeypatch):
    """Check spectrum writer streams flux and preserves previous file."""
    data = np.arange(21, dtype=np.float64).reshape(7, 3) / 3
    data[2, 1] = np.nan
    data.flags.writeable = False

    class SliceOnly:
        shape = data.shape

        def __array__(self, *a, **kw):
            raise AssertionError('whole-spectrum conversion')

        def __getitem__(self, key):
            part = data[key]
            assert part.shape[0] <= 2
            return part

    meta = ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP256', 1., 3,
                   np.arange(7.) + 60000, np.array([2]), np.array([7]), ())
    path = tmp_path / 'spectra.fits'
    orders = {'O1': {'wave': np.array([1., 2., 3.]),
                      'flux': SliceOnly(), 'ferr': SliceOnly()}}
    io.save_spectra(path, orders, meta, chunk_bytes=24)
    with fits.open(path) as hdus:
        hdus.verify('exception')
        np.testing.assert_array_equal(hdus['Flux O1'].data, data.astype(np.float32))
        np.testing.assert_array_equal(hdus['Time'].data, meta.int_times)
        assert hdus['Time'].data.dtype.itemsize == 8
        assert hdus['Flux O1'].header['UNITS'] == 'DN/s'
    previous = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError('simulated spectrum failure')

    monkeypatch.setattr(io, '_stream_image', fail)
    with pytest.raises(OSError, match='simulated spectrum failure'):
        io.save_spectra(path, orders, meta)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob('.spectra-*'))


def test_nirspec_trace_recovers_empty_polynomial_clipping_mask(monkeypatch):
    """Check NIRSpec trace recovers empty polynomial clipping mask."""
    utils = pytest.importorskip('exotedrf.utils')

    def empty_fit(*args, **kwargs):
        raise TypeError('expected non-empty vector for x')

    monkeypatch.setattr(utils, 'get_centroids_nirspec', empty_fit)
    rows = np.arange(32, dtype=float)
    profile = 100. + 20. * np.exp(-0.5 * ((rows - 15.) / 2.) ** 2)
    frame = np.broadcast_to(profile[:, None], (32, 48))
    actual = trace.get_centroids_nirspec(frame, xstart=7, xend=45)
    np.testing.assert_array_equal(actual['xpos'], np.arange(7, 45))
    np.testing.assert_allclose(actual['ypos'], 15., atol=1e-10)


def test_nirspec_trace_does_not_hide_unrelated_type_errors(monkeypatch):
    """Check NIRSpec trace does not hide unrelated type errors."""
    utils = pytest.importorskip('exotedrf.utils')

    def broken_trace(*args, **kwargs):
        raise TypeError('invalid trace input')

    monkeypatch.setattr(utils, 'get_centroids_nirspec', broken_trace)
    with pytest.raises(TypeError, match='invalid trace input'):
        trace.get_centroids_nirspec(np.ones((32, 48)))
