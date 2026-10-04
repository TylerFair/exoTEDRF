"""NIRISS/SOSS ATOCA extraction wrapper (exotedrf.v2.atoca)."""

import contextlib
import os
import types
import warnings

import numpy as np
import pytest
from astropy.io import fits

from exotedrf.v2 import atoca, config, stage2_models

datamodels = pytest.importorskip('stdatamodels.jwst.datamodels')


# Synthetic jwst products.

def _soss_header(**overrides):
    """Return SOSS header."""
    header = {
        'TELESCOP': 'JWST', 'INSTRUME': 'NIRISS', 'DETECTOR': 'NIS',
        'FILTER': 'CLEAR', 'PUPIL': 'GR700XD', 'EXP_TYPE': 'NIS_SOSS',
        'SUBARRAY': 'SUBSTRIP256', 'PWCPOS': 245.71756, 'READPATT': 'NISRAPID',
        'DATE-OBS': '2022-07-26', 'TIME-OBS': '20:45:11.094',
        'TARGNAME': 'WASP-39', 'NINTS': 6, 'INTSTART': 1, 'INTEND': 3,
        'EXSEGNUM': 1, 'EXSEGTOT': 2, 'TSOVISIT': True,
        'FILENAME': 'jw01366001001_04101_00001-seg001_nis_uncal.fits',
        'SUBSTRT1': 1, 'SUBSTRT2': 1793, 'SUBSIZE1': 2048, 'SUBSIZE2': 256,
    }
    header.update(overrides)
    return header


def _spec_dtype():
    # Stdatamodels >= 4 no longer materialises schema tables on access.
    """Return spec dtype."""
    model = datamodels.SpecModel()
    if hasattr(model, 'get_dtype'):
        return model.get_dtype('spec_table')
    return model.spec_table.dtype


def _multispec(nints, ncols=(12, 9), seed=0, int_start=1, types=None):
    """Order-1/2 MultiSpecModel in jwst ATOCA output layout."""
    rng = np.random.default_rng(seed)
    model = datamodels.MultiSpecModel()
    dtype = _spec_dtype()
    waves = {1: np.linspace(2.8, 0.85, ncols[0]),
             2: np.linspace(1.4, 0.6, ncols[1])}
    for i in range(nints):
        for order, n in zip((1, 2), ncols):
            table = np.zeros(n, dtype=dtype)
            table['WAVELENGTH'] = waves[order]
            table['FLUX'] = 1000 + rng.normal(0, 5, n)
            table['FLUX_ERROR'] = 5 + rng.random(n)
            spec = datamodels.SpecModel(spec_table=table)
            spec.spectral_order = order
            spec.int_num = int_start + i
            if types is not None:
                spec.meta.soss_extract1d.type = types[len(model.spec)]
            model.spec.append(spec)
    return model


def _tso_multispec(model):
    """Convert synthetic spectra to JWST 3's per-order table layout."""
    from jwst.datamodels.utils.tso_multispec import make_tso_specmodel
    result = datamodels.TSOMultiSpecModel()
    for order in (1, 2):
        result.spec.append(make_tso_specmodel(
            [spec for spec in model.spec if spec.spectral_order == order]))
    return result


# Pure helpers.

def test_validate_width_matches_v1_checks():
    """Check validate width matches v1 checks."""
    with pytest.raises(ValueError, match='Aperture optimization not '
                       'possible with ATOCA extraction.'):
        atoca.validate_width('optimize')
    with pytest.raises(ValueError):
        atoca.validate_width([30, 40])
    with pytest.raises(ValueError):
        atoca.validate_width(0)
    with pytest.warns(RuntimeWarning, match='Order 2 cannot use a '
                      'different width'):
        assert atoca.validate_width(40, extract_width_soss2=30) == 40.0


def test_v1_kwargs_are_the_stage3_keyword_set(tmp_path):
    """Check v1 keywords are the stage3 keyword set."""
    kwargs = atoca.v1_extract1d_kwargs(
        40, tmp_path / 'profile.fits', estimate=tmp_path / 'est.fits',
        references={'pastasoss': 'p.asdf', 'speckernel': 'k.fits',
                    'spectrace': 'ignored.fits'},
        crds_parameters={'soss_rtol': 1e-3, 'soss_bad_pix': 'masking'})
    assert kwargs['subtract_background'] is False
    assert kwargs['soss_bad_pix'] == 'model'
    assert kwargs['soss_width'] == 40
    assert kwargs['override_specprofile'] == os.fspath(
        tmp_path / 'profile.fits')
    assert kwargs['soss_estimate'] == os.fspath(tmp_path / 'est.fits')
    assert kwargs['soss_rtol'] == 1e-3
    assert kwargs['override_pastasoss'] == 'p.asdf'
    assert kwargs['override_speckernel'] == 'k.fits'
    assert 'override_spectrace' not in kwargs
    assert 'soss_tikfac' not in kwargs
    assert atoca.v1_extract1d_kwargs(40, 'p', tikfac=np.float64(2.5e-16))[
        'soss_tikfac'] == 2.5e-16


def test_tail_chunk_plan_covers_every_later_integration():
    """Check tail chunk plan covers every later integration."""
    assert atoca.plan_tail_chunks(1, 4) == []
    assert atoca.plan_tail_chunks(10, 4) == [(1, 5), (5, 9), (9, 10)]
    assert atoca.plan_tail_chunks(3, 1) == [(1, 2), (2, 3)]
    assert atoca.auto_chunk_ints(0, 4) == 1
    assert atoca.auto_chunk_ints(10, 4) == 1
    assert atoca.auto_chunk_ints(1000, 4) == 16
    assert atoca.auto_chunk_ints(100, 2) == 13


def test_resolve_parallelism_defaults_and_validation(monkeypatch):
    """Check resolve parallelism defaults and validation."""
    monkeypatch.setattr(atoca, 'available_cpus', lambda: 16)
    assert atoca.resolve_parallelism({}) == (4, 4)
    assert atoca.resolve_parallelism({}, n_tasks=2) == (2, 8)
    assert atoca.resolve_parallelism({'v2_atoca_workers': 8,
                                      'v2_atoca_solver_threads': 1}) == (8, 1)
    with pytest.raises(ValueError):
        atoca.resolve_parallelism({'v2_atoca_workers': 0})


def test_specprofile_name_and_subarray_rule():
    """Check specprofile name and subarray rule."""
    assert atoca.v1_specprofile_name('SUBSTRIP256') == \
        'APPLESOSS_ref_2D_profile_SUBSTRIP256_os1_pad20.fits'
    assert atoca.subarray_from_shape(96) == 'SUBSTRIP96'
    assert atoca.subarray_from_shape(256) == 'SUBSTRIP256'


def test_deepstack_matches_v1_make_deepstack():
    """Check deepstack matches v1 make deepstack."""
    utils = pytest.importorskip('exotedrf.utils')
    rng = np.random.default_rng(3)
    cube = rng.normal(size=(7, 20, 11)).astype(np.float32)
    cube[2, 3, 4] = np.nan
    cube[:, 5, 6] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected = utils.make_deepstack(cube)
        got = atoca.deepstack_nanmedian(cube, row_chunk=3)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == expected.dtype


# V1 oracles on synthetic jwst models.

def test_unpack_matches_v1_unpack_atoca_spectra():
    """Check unpack matches v1 unpack ATOCA spectra."""
    utils = pytest.importorskip('exotedrf.utils')
    model = _multispec(4)
    expected = utils.unpack_atoca_spectra(model)
    got = atoca.unpack_atoca_spectra(model)
    assert set(got) == {1, 2}
    for order in (1, 2):
        for key in ('WAVELENGTH', 'FLUX', 'FLUX_ERROR'):
            np.testing.assert_array_equal(got[order][key],
                                          expected[order][key])


def test_merge_then_format_matches_v1_format_soss_spectra(tmp_path):
    """Check merge then format matches v1 format SOSS spectra."""
    stage3 = pytest.importorskip('exotedrf.stage3')
    from exotedrf.v2 import stages

    seg1, seg2 = _multispec(5, seed=1), _multispec(4, seed=2, int_start=6)
    times = np.linspace(59786.9, 59787.1, 9)
    files = []
    for index, segment in enumerate((seg1, seg2)):
        path = tmp_path / f'segment{index}_extract1d.fits'
        _tso_multispec(segment).save(path)
        files.append(str(path))
    v1 = stage3.format_soss_spectra(
        files, times,
        {'extract_width': 40, 'method': 'atoca'},
        'WASP-39 b', output_dir=str(tmp_path) + '/', save_results=False,
        clip_thresh=10)
    unpacked = [atoca.unpack_atoca_spectra(seg) for seg in (seg1, seg2)]
    # Chunked unpacking (head + tails) must merge to the same arrays.
    chunked = atoca.merge_segment_orders(
        [{o: {k: v[:1] for k, v in d.items()} for o, d in unpacked[0].items()},
         {o: {k: v[1:] for k, v in d.items()} for o, d in unpacked[0].items()}])
    for order in (1, 2):
        for key in ('WAVELENGTH', 'FLUX', 'FLUX_ERROR'):
            np.testing.assert_array_equal(chunked[order][key],
                                          unpacked[0][order][key])
    spectra, products = atoca.format_atoca_products(
        [chunked, unpacked[1]], stages.sigma_clip_lightcurves)
    for order in (1, 2):
        np.testing.assert_array_equal(products[order]['wave'],
                                      v1[f'Wave O{order}'])
        np.testing.assert_array_equal(products[order]['flux'],
                                      v1[f'Flux O{order}'])
        np.testing.assert_array_equal(products[order]['ferr'],
                                      v1[f'Flux Err O{order}'])
        np.testing.assert_array_equal(spectra[order][0],
                                      v1[f'Flux O{order}'])


def test_estimate_table_matches_v1_get_soss_estimate(tmp_path):
    """Check estimate table matches v1 get SOSS estimate."""
    stage3 = pytest.importorskip('exotedrf.stage3')
    atoca_outputs = _multispec(
        2, seed=4, types=['TEST', 'TEST', 'OBSERVATION', 'OBSERVATION'])
    path = tmp_path / 'atoca_spectra.fits'
    atoca_outputs.save(path)
    v1_dir = tmp_path / 'v1'
    v1_dir.mkdir()
    v1_file = stage3.get_soss_estimate(str(path), str(v1_dir) + '/')
    table = atoca.estimate_spec_table(datamodels.open(path))
    v2_file = atoca.write_soss_estimate(table, tmp_path / 'soss_estimate.fits')
    with datamodels.open(v1_file) as v1, datamodels.open(v2_file) as v2:
        for column in ('WAVELENGTH', 'FLUX', 'FLUX_ERROR'):
            np.testing.assert_array_equal(v1.spec_table[column],
                                          v2.spec_table[column])


# Jwst transcription checks (no CRDS: run_extract1d is intercepted).

def _tiny_model():
    """Return tiny model."""
    data = np.ones((2, 256, 2048), np.float32)
    return stage2_models.stage2_datamodel(
        data, data, np.zeros(data.shape, np.uint32), _soss_header())


def test_extract_soss_in_memory_passes_jwst_soss_kwargs(monkeypatch, tmp_path):
    """Check extract SOSS in memory passes JWST SOSS keywords."""
    pytest.importorskip('jwst')
    from jwst.extract_1d import Extract1dStep
    from jwst.extract_1d.soss_extract import soss_extract

    calls = []

    def fake_run(model, pastasoss, specprofile, speckernel, subarray,
                 soss_filter, soss_kwargs):
        calls.append((pastasoss, specprofile, speckernel, subarray,
                      soss_filter, dict(soss_kwargs)))
        result = datamodels.MultiSpecModel()
        return result, datamodels.SossExtractModel(), \
            datamodels.MultiSpecModel()

    monkeypatch.setattr(soss_extract, 'run_extract1d', fake_run)
    kwargs = atoca.v1_extract1d_kwargs(
        33, tmp_path / 'profile.fits', estimate=tmp_path / 'est.fits',
        tikfac=1e-15, references={'pastasoss': str(tmp_path / 'p.asdf'),
                                  'speckernel': str(tmp_path / 'k.fits')})
    for name in ('profile.fits', 'est.fits', 'p.asdf', 'k.fits'):
        (tmp_path / name).write_text('')
    # Jwst's own method on the same configuration (soss_modelname as v1).
    step = Extract1dStep(**kwargs, soss_modelname=str(tmp_path / 'model'),
                         output_dir=str(tmp_path))
    step._extract_soss(_tiny_model())
    ours = atoca.make_step(kwargs)
    atoca.extract_soss_in_memory(ours, _tiny_model())
    assert len(calls) == 2
    assert calls[0] == calls[1]
    soss_kwargs = calls[1][-1]
    assert soss_kwargs['width'] == 33 and soss_kwargs['bad_pix'] == 'model'
    assert soss_kwargs['subtract_background'] is False
    assert soss_kwargs['tikfac'] == 1e-15 and soss_kwargs['model'] is True
    assert calls[1][3:5] == ('SUBSTRIP256', 'CLEAR')


def test_extract_soss_rejects_non_clear_filter():
    """Check extract SOSS rejects non clear filter."""
    pytest.importorskip('jwst')
    data = np.ones((1, 256, 2048), np.float32)
    model = stage2_models.stage2_datamodel(
        data, data, np.zeros(data.shape, np.uint32),
        _soss_header(FILTER='F277W'))
    with pytest.raises(ValueError, match='CLEAR filter only'):
        atoca.extract_soss_in_memory(types.SimpleNamespace(), model)


def test_model_image_hook_captures_and_injects(monkeypatch):
    """Check model image hook captures and injects."""
    pytest.importorskip('jwst')
    from jwst.extract_1d.soss_extract import soss_extract

    seen = []
    modern = not hasattr(soss_extract, 'model_image')
    name = '_process_one_integration' if modern else 'model_image'

    def fake_model_image(*args, wave_grid=None, tikfac=None, **kwargs):
        seen.append((wave_grid, tikfac))
        grid = (np.linspace(0.6, 2.8, 5) + 1e-13 if wave_grid is None
                else wave_grid)
        factor = np.float64(3.25e-16) if tikfac is None else tikfac
        if modern:
            return {}, [], [], {'Order 1': factor}, grid
        return {}, factor, 0., grid, []

    monkeypatch.setattr(soss_extract, name, fake_model_image)
    with atoca.model_image_hook() as head:
        getattr(soss_extract, name)(0, wave_grid=None, tikfac=None)
        getattr(soss_extract, name)(0, wave_grid='second', tikfac=7.)
    assert head['tikfac'] == 3.25e-16
    assert head['wave_grid'].dtype == np.float64
    injected = head['wave_grid']
    with atoca.model_image_hook(injected) as tail:
        getattr(soss_extract, name)(0, wave_grid=None, tikfac=head['tikfac'])
    assert seen[-1][0] is injected
    np.testing.assert_array_equal(tail['wave_grid'], injected)
    assert getattr(soss_extract, name) is fake_model_image


def test_complete_segment_task_needs_no_legacy_diagnostic_estimate(monkeypatch):
    """Check complete segment task needs no legacy diagnostic estimate."""
    model = _tiny_model()
    result = _multispec(2)
    diagnostics = datamodels.MultiSpecModel()
    references = datamodels.SossExtractModel()
    monkeypatch.setattr(stage2_models, 'stage2_datamodel', lambda *a, **k: model)
    monkeypatch.setattr(atoca, 'make_step', lambda kwargs: object())
    monkeypatch.setattr(atoca, 'extract_soss_in_memory',
                        lambda step, model: (result, references, diagnostics))

    @contextlib.contextmanager
    def hook(grid):
        yield {'tikfac': 1e-16, 'wave_grid': np.linspace(0.6, 2.8, 5)}

    monkeypatch.setattr(atoca, 'model_image_hook', hook)
    task = {'data': model.data, 'err': model.err, 'dq': model.dq,
            'header': fits.Header(_soss_header()).tostring(), 'int_start': 1,
            'kwargs': {}, 'head': True, 'make_estimate': False}
    output = atoca.run_task(task)
    assert 'error' not in output
    assert 'estimate' not in output
    assert output['orders'][1]['FLUX'].shape == (2, 12)


def test_threaded_tikhonov_tests_are_bit_identical():
    """Check threaded tikhonov tests are bit identical."""
    pytest.importorskip('jwst')
    import jwst
    if jwst.__version__ not in ('1.17.1', atoca.JWST_VERSION):
        pytest.skip('threaded Tikhonov patch requires a supported JWST release')
    import scipy.sparse as sp
    from jwst.extract_1d.soss_extract import atoca_utils

    rng = np.random.default_rng(5)
    n_pix, n_wave = 400, 120
    a_mat = sp.random(n_pix, n_wave, density=0.05, random_state=7,
                      format='csr') + sp.eye(n_pix, n_wave, format='csr')
    # Same container types as ExtractionEngine.get_detector_model.
    b_vec = sp.csr_matrix(rng.normal(size=(1, n_pix)))
    t_mat = atoca_utils.finite_first_d(np.linspace(1, 2, n_wave))
    tikho = atoca_utils.Tikhonov(a_mat, b_vec, t_mat)
    factors = np.logspace(-3, 1, 9)
    expected = tikho.test_factors(factors)
    with atoca.threaded_tikhonov_tests(4) as active:
        assert active
        got = tikho.test_factors(factors)
    assert atoca_utils.Tikhonov.test_factors is not None
    for key in ('factors', 'solution', 'error', 'reg'):
        np.testing.assert_array_equal(got[key], expected[key])
    with atoca.threaded_tikhonov_tests(1) as active:
        assert not active


# Scheduler (synthetic runner; inline and with a real spawn pool).

def synthetic_runner(task):
    """Deterministic stand-in for run_task: echoes the dependency inputs.

    Parameters
    ----------
    task : dict
        Extraction task specification.

    Returns
    -------
    result : dict
        Calculation results and associated metadata.
    """
    data = np.asarray(task['data'], dtype=np.float64)
    kwargs = task['kwargs']
    if task['head'] and kwargs.get('soss_estimate') is None and \
            float(data[0, 0, 0]) < 0:
        return {'error': atoca.ESTIMATE_FAILURE, 'error_type': 'error',
                'traceback': ''}
    nints = data.shape[0]
    if task['head']:
        tag = 0.0 if kwargs.get('soss_estimate') is None else 1000.0
        tikfac = 1e-16 * (1 + float(data[0, 0, 0]) + tag)
        wave_grid = np.linspace(0.6, 2.8, 4) + float(data[0, 0, 0])
    else:
        tikfac = float(kwargs['soss_tikfac'])
        wave_grid = task['wave_grid']
    flux = data[:, 0, :3] + tikfac * 1e16 + wave_grid[0]
    orders = {o: {'WAVELENGTH': np.tile(np.array([3., 2., 1.]), (nints, 1)),
                  'FLUX': flux * o, 'FLUX_ERROR': np.ones((nints, 3))}
              for o in (1, 2)}
    out = {'orders': orders, 'tikfac': tikfac, 'seconds': 0.0,
           'models': {i: (np.full((2, 3), task['lo'] + i, np.float32),
                          np.zeros((2, 3), np.float32))
                      for i in task['model_indices']}}
    if task['head']:
        out['wave_grid'] = wave_grid
        out['estimate'] = np.zeros(2, dtype=[('WAVELENGTH', 'f8'),
                                             ('FLUX', 'f8')])
    return out


def _segments(values):
    """Return segments."""
    segments = []
    for index, column in enumerate(values):
        column = np.asarray(column, dtype=np.float32)
        cube = np.broadcast_to(column[:, None, None], (column.size, 2, 3))

        def fetch(lo, hi, _cube=cube):
            return (np.array(_cube[lo:hi]), np.ones_like(_cube[lo:hi]),
                    np.zeros(_cube[lo:hi].shape, np.uint32))

        segments.append(atoca.AtocaSegment(
            index=index, nints=column.size, header='', int_start=1,
            int_times=None, kwargs={'soss_width': 40}, fetch=fetch))
    return segments


def _expected_flux(values, estimate_segment):
    """Return expected flux."""
    out = []
    for index, column in enumerate(values):
        column = np.asarray(column, dtype=np.float64)
        head = column[0]
        tag = 0.0 if index == estimate_segment else 1000.0
        flux = column + (1 + head) + tag + 0.6 + head
        out.append(np.repeat(flux[:, None], 3, axis=1))
    return out


def test_schedule_keeps_jwst3_segment_statistics(tmp_path):
    """Check schedule keeps jwst3 segment statistics."""
    values = [[0.5, 1, 2, 3, 4], [7, 8]]
    tasks = []

    def runner(task):
        tasks.append(task)
        return synthetic_runner(task)

    result = atoca.run_schedule(
        _segments(values), width=40, specprofile='p.fits', workers=1,
        chunk_ints=1, whole_segments=True, runner=runner,
        solver_threads=2, head_solver_threads=8)
    assert [task['data'].shape[0] for task in tasks] == [5, 2]
    assert all(task['kwargs']['soss_estimate'] is None for task in tasks)
    assert all(not task['make_estimate'] for task in tasks)
    assert all(task['solver_threads'] == 2 for task in tasks)
    assert result['estimate'] is None
    assert result['estimate_source'] is None
    assert [part[1]['FLUX'].shape[0] for part in result['segments']] == [5, 2]


@pytest.mark.parametrize('workers', [1, 2])
def test_schedule_reproduces_v1_segment_loop(tmp_path, monkeypatch, workers):
    """Check schedule reproduces v1 segment loop."""
    monkeypatch.setattr(atoca, 'write_soss_estimate',
                        lambda table, path: os.fspath(path))
    values = [[0.5, 1, 2, 3, 4], [7, 8], [9]]
    result = atoca.run_schedule(
        _segments(values), width=40, specprofile='p.fits',
        estimate_path=tmp_path / 'soss_estimate.fits', workers=workers,
        chunk_ints=2, model_indices={0: [3], 1: [0]},
        runner=synthetic_runner)
    assert result['estimate_source'] == 'segment 1'
    assert result['estimate'] == os.fspath(tmp_path / 'soss_estimate.fits')
    expected = _expected_flux(values, estimate_segment=0)
    for seg, want in zip(result['segments'], expected):
        np.testing.assert_array_equal(seg[1]['FLUX'], want)
        np.testing.assert_array_equal(seg[2]['FLUX'], 2 * want)
    np.testing.assert_allclose(result['tikfacs'],
                               [1e-16 * 1.5, 1e-16 * 1008, 1e-16 * 1010])
    assert sorted(result['models']) == [(0, 3), (1, 0)]
    assert result['models'][(0, 3)][0][0, 0] == 3
    kinds = [t['kind'] for t in result['timings']]
    assert kinds.count('head') == 3 and kinds.count('tail') == 3


def test_schedule_retries_estimate_failures_like_v1(tmp_path, monkeypatch):
    """Check schedule retries estimate failures like v1."""
    monkeypatch.setattr(atoca, 'write_soss_estimate',
                        lambda table, path: os.fspath(path))
    values = [[-1, 1, 2], [5, 6]]
    result = atoca.run_schedule(
        _segments(values), width=40, specprofile='p.fits',
        estimate_path=tmp_path / 'e.fits', workers=1, chunk_ints=8,
        runner=synthetic_runner)
    assert result['estimate_source'] == 'segment 2'
    # Segment 1 is re-extracted with segment 2's estimate.
    expected = _expected_flux(values, estimate_segment=1)
    for seg, want in zip(result['segments'], expected):
        np.testing.assert_array_equal(seg[1]['FLUX'], want)


def test_schedule_raises_when_no_segment_gives_an_estimate(tmp_path):
    """Check schedule raises when no segment gives an estimate."""
    with pytest.raises(RuntimeError, match='No segments could be properly'):
        atoca.run_schedule(
            _segments([[-1, 2], [-3]]), width=40, specprofile='p.fits',
            estimate_path=tmp_path / 'e.fits', workers=1,
            runner=synthetic_runner)


def test_schedule_with_user_estimate_runs_all_heads_with_it(tmp_path):
    """Check schedule with user estimate runs all heads with it."""
    values = [[-1, 1, 2], [5, 6]]
    result = atoca.run_schedule(
        _segments(values), width=40, specprofile='p.fits',
        estimate=tmp_path / 'user.fits', workers=1, chunk_ints=1,
        runner=synthetic_runner)
    assert result['estimate_source'] == 'user'
    expected = _expected_flux(values, estimate_segment=None)
    for seg, want in zip(result['segments'], expected):
        np.testing.assert_array_equal(seg[1]['FLUX'], want)


def failing_runner(task):
    """Raise an error when an extraction task is executed.

    Parameters
    ----------
    task : dict
        Extraction task specification.

    Returns
    -------
    result : dict
        Calculation results and associated metadata.
    """
    return {'error': 'boom', 'error_type': 'ValueError', 'traceback': 'tb'}


def test_schedule_propagates_task_errors(tmp_path):
    """Check schedule propagates task errors."""
    with pytest.raises(atoca.AtocaTaskError, match='boom'):
        atoca.run_schedule(
            _segments([[1, 2]]), width=40, specprofile='p.fits',
            estimate=tmp_path / 'e.fits', workers=1,
            runner=failing_runner)


# Config, step wiring, caching and products.

def _soss_cfg(**overrides):
    """Return SOSS cfg."""
    cfg = {'observing_mode': 'NIRISS/SOSS', 'filter_detector': 'CLEAR',
           'input_dir': '.', 'extract_method': 'atoca',
           'optimize_extract_width': True,
           'extract_width': [30, 35, 40]}
    cfg.update(overrides)
    return cfg


def _validate(cfg):
    """Return validate."""
    try:
        return config.validate_supported_config(cfg)
    except FileNotFoundError:  # pragma: no cover - uncal discovery
        pytest.skip('config validation requires input discovery')


def test_config_accepts_atoca_for_soss(monkeypatch):
    """Check config accepts ATOCA for SOSS."""
    monkeypatch.setattr(config, '_validate_uncal_input', lambda cfg: None)
    _validate(_soss_cfg())
    _validate(_soss_cfg(soss_specprofile='profile.fits',
                        v2_atoca_workers=8,
                        stage3_kwargs={'Extract1dStep': {'clip_thresh': 8}}))
    opts = config.fixed_options(_soss_cfg(
        soss_specprofile='profile.fits', v2_atoca_workers=8,
        stage3_kwargs={'Extract1dStep': {'clip_thresh': 8}}))
    assert opts['extract_method'] == 'atoca'
    assert opts['soss_specprofile'] == 'profile.fits'
    assert opts['soss_estimate'] is None
    assert opts['clip_thresh'] == 8.
    assert opts['v2_atoca_workers'] == 8


@pytest.mark.parametrize('overrides, error, match', [
    ({'extract_width': 'optimize', 'optimize_extract_width': False},
     ValueError, 'Aperture optimization not possible with ATOCA'),
    ({'filter_detector': 'F277W'}, NotImplementedError, 'CLEAR filter'),
    ({'v2_atoca_workers': 0}, ValueError, 'positive integer'),
    ({'stage3_kwargs': {'Extract1dStep': {'clip_thresh': 8,
                                          'soss_width': 3}}},
     NotImplementedError, 'stage3_kwargs'),
])
def test_config_rejects_invalid_atoca_requests(monkeypatch, overrides, error,
                                               match):
    """Check config rejects invalid ATOCA requests."""
    monkeypatch.setattr(config, '_validate_uncal_input', lambda cfg: None)
    with pytest.raises(error, match=match):
        _validate(_soss_cfg(**overrides))


def test_soss_estimate_rejected_like_v1_2_5_0(monkeypatch):
    """Check SOSS estimate rejected like v1 2 5 0."""
    monkeypatch.setattr(config, '_validate_uncal_input', lambda cfg: None)
    for method in ('box', 'atoca'):
        with pytest.raises(ValueError, match='soss_estimate'):
            _validate(_soss_cfg(extract_method=method, stage3_kwargs={
                'Extract1dStep': {'soss_estimate': 'e.fits'}}))


def test_soss_steps_use_atoca_extract_only_when_requested():
    """Check SOSS steps use ATOCA extract only when requested."""
    from exotedrf.v2 import stages
    assert stages.soss_steps({'extract_method': 'atoca'})[-1].fn is \
        stages.step_extract_atoca
    assert stages.soss_steps({})[-1].fn is stages.step_extract
    assert stages.soss_steps({'extract_method': 'atoca'})[-1].param_names \
        == ('extract_width',)


def _rate_state(nints=(3, 2), dimy=256, dimx=2048, seed=0):
    """Return rate state."""
    from exotedrf.v2 import core
    from exotedrf.v2.pipeline import PipelineState
    total = sum(nints)
    rng = np.random.default_rng(seed)
    data = rng.normal(100, 1, (total, dimy, dimx)).astype(np.float32)
    err = np.full(data.shape, 2.0, np.float32)
    dq = np.zeros(data.shape, np.uint32)
    edges = np.cumsum(nints)
    headers = (_soss_header(INTSTART=1, INTEND=nints[0], EXSEGNUM=1),
               _soss_header(INTSTART=nints[0] + 1, INTEND=total, EXSEGNUM=2,
                            FILENAME='jw01366001001_04101_00001-seg002_'
                                     'nis_uncal.fits'))
    meta = core.ObsMeta(
        mode='NIRISS/SOSS', detector='CLEAR', subarray='SUBSTRIP256',
        frame_time=5.494, ngroups=9,
        int_times=np.linspace(59786.9, 59787.0, total),
        baseline_ints=np.array([1, -1]), segment_edges=edges,
        filenames=('missing1.fits', 'missing2.fits'),
        extra={'header': dict(headers[0]), 'segment_headers': headers,
               'segment_int_starts': (1, nints[0] + 1),
               'segment_int_ends': (nints[0], total)})
    return PipelineState(cube=core.RateCube(data, err, dq, meta))


def array_runner(task):
    """Synthetic run_task: per-integration column sums as 'ATOCA' flux.

    Parameters
    ----------
    task : dict
        Extraction task specification.

    Returns
    -------
    result : dict
        Calculation results and associated metadata.
    """
    data = np.asarray(task['data'], dtype=np.float64)
    nints = data.shape[0]
    width = float(task['kwargs']['soss_width'])
    waves = {1: np.linspace(2.8, 0.85, 40), 2: np.linspace(1.4, 0.6, 30)}
    orders = {o: {'WAVELENGTH': np.tile(w, (nints, 1)),
                  'FLUX': data[:, 10 * o, :w.size] * width,
                  'FLUX_ERROR': np.ones((nints, w.size))}
              for o, w in waves.items()}
    out = {'orders': orders, 'tikfac': 1e-16, 'seconds': 0.0,
           'models': {i: (np.zeros(data.shape[1:], np.float32),
                          np.zeros(data.shape[1:], np.float32))
                      for i in task['model_indices']}}
    if task['head']:
        out['wave_grid'] = np.linspace(0.6, 2.8, 5)
        out['estimate'] = np.zeros(3, dtype=_spec_dtype())
    return out


def _fake_references(monkeypatch, tmp_path):
    """Return fake references."""
    profile = tmp_path / atoca.v1_specprofile_name('SUBSTRIP256')
    calls = {'profile': 0}

    def fake_build(deepstack, tracetable, wavemap, output_dir, empirical=True):
        calls['profile'] += 1
        assert deepstack.ndim == 2
        profile.write_text('profile')
        return os.fspath(profile)

    monkeypatch.setattr(atoca, 'resolve_references', lambda model, refs: {
        ref: os.fspath(tmp_path / f'{ref}.fits') for ref in refs})
    monkeypatch.setattr(atoca, 'crds_step_parameters', lambda model: {})
    monkeypatch.setattr(atoca, 'build_specprofile', fake_build)
    return calls


def test_extract_state_formats_caches_and_writes_products(monkeypatch,
                                                           tmp_path):
    """Check extract state formats caches and writes products."""
    from exotedrf.v2 import products, stages

    calls = _fake_references(monkeypatch, tmp_path)
    state = _rate_state()
    ctx = {'opts': {'extract_method': 'atoca', 'do_plots': True,
                    'v2_atoca_workers': 1, 'output_mode': 'standard'},
           'atoca_output_dir': tmp_path / 'Stage3'}
    runs = []

    def counting_runner(task):
        runs.append(task['head'])
        return array_runner(task)

    spectra, prods, info = atoca.extract_state(
        state, {'extract_width': 40}, ctx, log=lambda m: None,
        runner=counting_runner)
    assert calls['profile'] == 1
    assert info['specprofile_source'] == 'applesoss'
    assert info['soss_estimate_source'] == 'segment 1'
    assert (tmp_path / 'Stage3' / 'soss_estimate.fits').exists()
    data = np.asarray(state.cube.data, np.float64)
    for order, ncol in ((1, 40), (2, 30)):
        raw = data[:, 10 * order, :ncol] * 40
        np.testing.assert_array_equal(prods[order]['wave'],
                                      np.sort(prods[order]['wave']))
        np.testing.assert_array_equal(
            prods[order]['flux'],
            stages.sigma_clip_lightcurves(raw[:, ::-1]))
    assert info['decontam_panels'].shape == (9, 256, 2048)
    n_runs = len(runs)
    copy = state.snapshot()
    again = atoca.extract_state(copy, {'extract_width': 40}, ctx,
                                log=lambda m: None, runner=counting_runner)
    assert len(runs) == n_runs and again[2] is info
    # A new width re-extracts but reuses the specprofile.
    atoca.extract_state(state, {'extract_width': 30}, ctx,
                        log=lambda m: None, runner=counting_runner)
    assert len(runs) == 2 * n_runs and calls['profile'] == 1

    # Product writer: v1 naming/METHOD and the ATOCA sidecars.
    monkeypatch.setattr(stages, 'step_extract_atoca', stages.step_extract_atoca)
    aux = {'spectra': spectra, 'spectral_products': prods, 'atoca': info}
    from exotedrf.v2.pipeline import PipelineState
    final = PipelineState(cube=state.cube, aux=aux)
    paths = products.output_layout({'name_tag': 't', 'extract_method': 'atoca'},
                                   output_dir=tmp_path / 'out')
    written = products.write_final_products(
        final, {'extract_width': 40}, ctx, paths, output_mode='optimal')
    spectra_path = paths['stage3'] / 'WASP-39_atoca_spectra_fullres.fits'
    assert written['spectra'] == str(spectra_path)
    with fits.open(spectra_path) as hdul:
        assert hdul[0].header['METHOD'] == 'atoca'
        assert hdul[0].header['WIDTH'] == 40
        assert [h.name for h in hdul[1:]] == [
            'WAVE O1', 'WAVE ERR O1', 'FLUX O1', 'FLUX ERR O1', 'WAVE O2',
            'WAVE ERR O2', 'FLUX O2', 'FLUX ERR O2', 'TIME',
            'DQ REPORT O1', 'DQ REPORT O2']
        np.testing.assert_allclose(hdul['FLUX O1'].data, prods[1]['flux'],
                                   rtol=1e-6)
    assert written['atoca_soss_estimate'].endswith('soss_estimate.fits')
    assert written['atoca_specprofile'].endswith(
        'APPLESOSS_ref_2D_profile_SUBSTRIP256_os1_pad20.fits')
    plot = products._write_diagnostic_plots(
        final, {'extract_width': 40}, ctx, paths, [])
    assert os.path.exists(plot['atoca_decontamination_plot'])
    assert 'centroid_plot' not in plot


def test_step_extract_atoca_sets_aux(monkeypatch, tmp_path):
    """Check step extract ATOCA sets aux."""
    from exotedrf.v2 import stages
    _fake_references(monkeypatch, tmp_path)
    monkeypatch.setattr(atoca, 'run_task', array_runner)
    original = atoca.extract_state

    def via_runner(state, params, ctx, log=print):
        return original(state, params, ctx, log=lambda m: None,
                        runner=array_runner)

    monkeypatch.setattr(atoca, 'extract_state', via_runner)
    state = _rate_state()
    ctx = {'opts': {'extract_method': 'atoca', 'v2_atoca_workers': 1},
           'atoca_output_dir': tmp_path}
    out = stages.step_extract_atoca(state, {'extract_width': 35}, ctx)
    assert out.cube is state.cube
    assert set(out.aux['spectral_products']) == {1, 2}
    assert out.aux['atoca']['extract_width'] == 35
    cost, scatter = stages.evaluate_production_cost(
        out, {'extract_width': 35}, dict(ctx, opts=dict(
            ctx['opts'], wave_range=[1.0, 2.0], w1=0., w2=1.)))
    assert np.isfinite(cost)


# Optimizer end to end (synthetic uncal files; jwst ATOCA stubbed).

def test_optimizer_runs_atoca_in_phase2_only(monkeypatch, tmp_path):
    """Check optimizer runs ATOCA in phase2 only."""
    from exotedrf.v2 import trace
    from exotedrf.v2.optimize import run_optimizer
    from .test_yaml_niriss_e2e import (DIMX, SEGMENT_NINTS, Y_TRACE_O1,
                                       Y_TRACE_O2, _synthetic_refpack,
                                       _write_uncal_segment, _yaml_path)
    from .niriss_acceptance import apply_current_niriss_settings

    _fake_references(monkeypatch, tmp_path)
    calls = []

    def runner(task):
        calls.extend([float(task['kwargs']['soss_width'])] *
                     (task['hi'] - task['lo']))
        return array_runner(task)

    monkeypatch.setattr(atoca, 'run_task', runner)
    input_dir = tmp_path / 'inputs'
    input_dir.mkdir()
    rng = np.random.RandomState(1)
    int_start = 1
    for index, nints in enumerate(SEGMENT_NINTS, start=1):
        _write_uncal_segment(
            input_dir / f'jwsynth-seg{index:03d}_uncal.fits', nints,
            int_start, index, sum(SEGMENT_NINTS), rng)
        int_start += nints
    np.save(tmp_path / 'bkg.npy', np.ones((96, DIMX), np.float32))
    trace.save_centroids_csv(str(tmp_path / 'cen.csv'), {
        'xpos': np.arange(DIMX, dtype=float),
        'ypos o1': np.full(DIMX, Y_TRACE_O1),
        'ypos o2': np.full(DIMX, Y_TRACE_O2)})
    cfg = apply_current_niriss_settings(config.load_config(_yaml_path()))
    for key in list(cfg):
        if key.startswith('optimize_') and key not in (
                'optimize_extract_width', 'optimize_extract_width_only',
                'optimize_from_pca_only'):
            cfg[key] = False
            param = key[len('optimize_'):]
            if isinstance(cfg.get(param), list):
                cfg[param] = cfg[param][len(cfg[param]) // 2]
    cfg.update({
        'input_dir': str(input_dir),
        'pipeline_outputs_directory': str(tmp_path / 'out'),
        'crds_cache_path': str(tmp_path / 'crds'),
        'soss_background_file': str(tmp_path / 'bkg.npy'),
        'centroids': str(tmp_path / 'cen.csv'), 'f277w': None,
        'extract_method': 'atoca', 'optimize_extract_width': True,
        'extract_width': [20, 30], 'v2_atoca_workers': 1,
        'do_plots': False})
    result = run_optimizer(cfg, logger=None, refpack=_synthetic_refpack())
    phase2 = [t for t in result.trials if t.phase == 2]
    assert [t.value for t in phase2] == [20, 30]
    assert all(np.isfinite(t.cost) for t in phase2)
    assert sorted(set(calls)) == [20.0, 25.0, 30.0]
    assert len(calls) == 3 * sum(SEGMENT_NINTS)
    spectra = [p for k, p in result.products.items() if k == 'spectra']
    assert spectra and spectra[0].endswith('_atoca_spectra_fullres.fits')
    with fits.open(spectra[0]) as hdul:
        assert hdul[0].header['METHOD'] == 'atoca'
        assert hdul[0].header['WIDTH'] == result.winners['extract_width']
    import jwst
    if jwst.__version__ == atoca.JWST_VERSION:
        # V1 2.5.0 passes no estimate between segments under JWST 3.
        assert 'atoca_soss_estimate' not in result.products
    else:
        assert result.products['atoca_soss_estimate'].endswith(
            'soss_estimate.fits')


@pytest.mark.parametrize('extra', [
    {'st_teff': 5400, 'st_logg': 4.45, 'st_met': 0.0}])
def test_atoca_refuses_unwired_wavelength_options(extra):
    """ATOCA + stellar CCF is refused, never silently skipped."""
    from exotedrf.v2 import config
    import warnings
    cfg = {'observing_mode': 'NIRISS/SOSS', 'filter_detector': 'CLEAR',
           'extract_method': 'atoca', 'extract_width': 40, **extra}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(NotImplementedError, match='extract_method: atoca'):
            config.validate_supported_config(cfg)


def test_unpack_jwst3_tso_tables_preserves_integration_axes():
    """Check unpack jwst3 tso tables preserves integration axes."""
    model = _multispec(4)
    expected = atoca.unpack_atoca_spectra(model)
    actual = atoca.unpack_atoca_spectra(_tso_multispec(model))
    for order in (1, 2):
        for quantity in ('WAVELENGTH', 'FLUX', 'FLUX_ERROR'):
            np.testing.assert_array_equal(actual[order][quantity],
                                          expected[order][quantity])
