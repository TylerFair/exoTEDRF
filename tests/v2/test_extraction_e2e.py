"""End-to-end extraction options: YAML optimizer runs and instrument graphs."""

import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from exotedrf.v2 import config as v2config
from exotedrf.v2 import core, products, stages, trace
from exotedrf.v2.core import ObsMeta, RampCube
from exotedrf.v2.optimize import run_optimizer
from exotedrf.v2.pipeline import PipelineState

from . import extraction_v1_oracle as oracle
from .niriss_acceptance import apply_current_niriss_settings
from .test_yaml_niriss_e2e import (DIMX, DIMY, SEGMENT_NINTS, Y_TRACE_O1,
                                   Y_TRACE_O2, _synthetic_refpack,
                                   _write_uncal_segment, _yaml_path)


# SOSS: YAML -> run_optimizer -> products.

def _soss_config(base, **patch):
    """Return SOSS config."""
    input_dir = base / 'inputs'
    if not input_dir.exists():
        input_dir.mkdir()
        rng = np.random.RandomState(7)
        int_start = 1
        for index, nints in enumerate(SEGMENT_NINTS, start=1):
            _write_uncal_segment(
                input_dir / f'jwsynth-seg{index:03d}_uncal.fits', nints,
                int_start, index, sum(SEGMENT_NINTS), rng)
            int_start += nints
        np.save(base / 'background_synth.npy',
                np.full((DIMY, DIMX), 1., np.float32))
        trace.save_centroids_csv(str(base / 'synth_centroids.csv'), {
            'xpos': np.arange(DIMX, dtype=float),
            'ypos o1': np.full(DIMX, Y_TRACE_O1),
            'ypos o2': np.full(DIMX, Y_TRACE_O2),
        })
    cfg = apply_current_niriss_settings(v2config.load_config(_yaml_path()))
    # Detector sweeps are not under test: fix each at its v1 phase-1 value.
    for key in list(cfg):
        if key.startswith('optimize_') and cfg[key] is True and \
                key != 'optimize_extract_width':
            param = key[len('optimize_'):]
            cfg[key] = False
            cfg[param] = cfg[param][len(cfg[param]) // 2]
    cfg.update({
        'input_dir': str(input_dir),
        'crds_cache_path': str(base / 'crds_cache'),
        'f277w': None,
        'soss_background_file': str(base / 'background_synth.npy'),
        'centroids': str(base / 'synth_centroids.csv'),
        'do_plots': True,
    })
    cfg.update(patch)
    return cfg


@pytest.fixture(scope='module')
def soss_base(tmp_path_factory):
    """Create synthetic SOSS inputs for extraction tests.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for temporary test directories.

    Returns
    -------
    directory : pathlib.Path
        Directory containing synthetic inputs.
    """
    return tmp_path_factory.mktemp('extraction_soss')


def test_soss_two_sided_extract_width_sweep_end_to_end(soss_base):
    """Check SOSS two sided extract width sweep end to end."""
    candidates = [[8, 10], [10, 12], [12, 7]]
    cfg = _soss_config(
        soss_base, optimize_extract_width=True, extract_width=candidates,
        extract_width_soss2=None,
        pipeline_outputs_directory=str(soss_base / 'asym'))
    result = run_optimizer(cfg, logger=None, refpack=_synthetic_refpack())

    sweep = [t for t in result.trials if t.parameter == 'extract_width']
    assert [t.phase for t in sweep] == [2, 2, 2]
    assert [list(t.value) for t in sweep] == candidates
    winner = result.winners['extract_width']
    assert list(winner) in candidates
    assert np.isfinite(result.final_cost)
    costs = [t.cost for t in sweep]
    assert result.final_cost == pytest.approx(min(costs), rel=1e-6)

    lines = open(result.output_paths['cost']).read().splitlines()
    column = lines[0].split('\t').index('extract_width')
    assert [line.split('\t')[column] for line in lines[1:]] == [
        '[8, 10]', '[10, 12]', '[12, 7]']
    summary = json.loads(open(result.output_paths['summary']).read())
    assert summary['winners']['extract_width'] == list(winner)
    with fits.open(result.output_paths['spectra']) as hdul:
        assert hdul[0].header['WIDTH'] == \
            f'lower={winner[0]}, upper={winner[1]}'
        assert np.isfinite(hdul['FLUX O1'].data).all()
        assert np.isfinite(hdul['FLUX O2'].data).all()

    v1 = oracle.load()
    with fits.open(result.output_paths['rate']) as hdul:
        data = hdul['SCI'].data.astype(float)
        err = hdul['ERR'].data.astype(float)
        dq = hdul['DQ'].data
    data, err, _ = v1._mask_dq_pixels(
        data, err, dq, 'rate',
        mask_saturated_pixels=not cfg['saturation_rescue'],
        mask_do_not_use_pixels=cfg['mask_do_not_use_pixels'])
    ref, _, _ = v1.do_box_extraction(data, err, np.full(DIMX, Y_TRACE_O1),
                                  tuple(winner), progress=False)
    flux = result.final_state.aux['spectra'][1][0]
    # V1 clips after reversing to increasing wavelength (even window).
    clipped = _v1_clip(ref[:, ::-1])[:, ::-1]
    np.testing.assert_allclose(flux, clipped, rtol=1e-5, atol=1e-2)


def test_soss_optimize_width_and_optimal_fallback_end_to_end(soss_base):
    """Check SOSS optimize width and optimal fallback end to end."""
    cfg = _soss_config(
        soss_base, optimize_extract_width=False, extract_width='optimize',
        extract_method='optimal', name_tag='opt',
        pipeline_outputs_directory=str(soss_base / 'optimize'))
    with pytest.warns(UserWarning, match='Switching to box extraction'):
        result = run_optimizer(cfg, logger=None,
                               refpack=_synthetic_refpack())
    assert not [t for t in result.trials if t.parameter == 'extract_width']
    selected = result.final_state.aux['extract_width_selected']
    assert set(selected) == {1}
    assert 10 <= selected[1] <= 60
    assert result.summary['extract_width_selected'] == {'1': selected[1]}
    # The v1 name after the switch is the box product.
    assert result.output_paths['spectra'].endswith(
        'opt_box_spectra_fullres.fits')
    with fits.open(result.output_paths['spectra']) as hdul:
        assert hdul[0].header['METHOD'] == 'box'
        assert hdul[0].header['WIDTH'] == int(selected[1])
    assert result.output_paths['aperture_optimization_1'].endswith(
        'aperture_optimization_order1.png')

    # V1 box_extract_soss('optimize') on the same final rate cube.
    v1 = oracle.load(dimx=DIMX)
    with fits.open(result.output_paths['rate']) as hdul:
        seg = oracle.segment(hdul['SCI'].data, hdul['ERR'].data,
                             hdul['DQ'].data)
    centroids = oracle.centroid_frame(
        xpos=np.arange(DIMX), **{'ypos o1': np.full(DIMX, Y_TRACE_O1),
                                 'ypos o2': np.full(DIMX, Y_TRACE_O2)})
    reference = v1.box_extract_soss(
        [seg], centroids, 'optimize', soss_width_o2=None,
        mask_saturated_pixels=not cfg['saturation_rescue'])
    assert reference[-1] == selected[1]


# NIRSpec: optimal, two-sided box, custom deepframe.

def _nirspec_meta(nints, ngroups, edges, mode='NIRSpec/PRISM'):
    """Return NIRSpec meta."""
    header = {'EXP_TYPE': 'NRS_BRIGHTOBJ', 'GRATING': mode.split('/')[-1],
              'NFRAMES': 1, 'TFRAME': 0.902, 'TGROUP': 0.902, 'NINTS': nints}
    return ObsMeta(
        mode=mode, detector='NRS1', subarray='SUB512', frame_time=0.902,
        ngroups=ngroups, int_times=60000. + np.arange(nints) * 1e-4,
        baseline_ints=np.asarray([4]), segment_edges=np.asarray(edges),
        filenames=tuple(f'seg{i + 1:03d}_uncal.fits'
                        for i in range(len(edges))),
        extra={'header': header,
               'segment_headers': tuple(dict(header) for _ in edges),
               'segment_int_starts': tuple(
                   [1] + [int(edge) + 1 for edge in edges[:-1]])})


def _nirspec_graph(**opts_patch):
    """Return NIRSpec graph."""
    nints, ngroups, dimy, dimx = 24, 3, 16, 32
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[:dimy, :dimx]
    profile = np.exp(-0.5 * ((yy - 7.3 - 0.3 * np.sin(1.7 * xx)) / 1.4) ** 2)
    data = np.empty((nints, ngroups, dimy, dimx), np.float32)
    for integration in range(nints):
        slope = 25. + profile * (800. + 2. * integration)
        for group in range(ngroups):
            data[integration, group] = (100. + group * slope +
                                        rng.normal(0, 1., (dimy, dimx)))
    cube = RampCube(data, np.zeros_like(data, np.uint8),
                    np.zeros((dimy, dimx), np.uint32),
                    _nirspec_meta(nints, ngroups, [12, 24]))
    lin = np.zeros((2, dimy, dimx), np.float32)
    lin[1] = 1.
    refpack = {
        'mask_dq': np.zeros((dimy, dimx), np.uint32),
        'superbias': np.zeros((dimy, dimx), np.float32),
        'lin_coeffs': lin, 'lin_dq': np.zeros((dimy, dimx), np.uint32),
        'readnoise': np.full((dimy, dimx), 6., np.float32),
        'gain': np.ones((dimy, dimx), np.float32),
        'gain_factor': np.float32(1.),
        'wave_map': np.broadcast_to(np.linspace(5., 1., dimx),
                                    (dimy, dimx)).copy(),
    }
    opts = {name: 'skip' for name in stages._NIRSPEC_STEP_CONTROLS}
    for name in ('DQInitStep', 'SuperBiasStep', 'LinearityStep',
                 'RampFitStep', 'GainScaleStep', 'PCAReconstructStep'):
        opts[name] = 'run'
    opts.update({
        'Extract2DStep': 'skip', 'WaveCorrStep': 'skip',
        'INLCorrStep': 'skip', 'RefPixStep': 'skip', 'FlatFieldStep': 'skip',
        'BackgroundStep': 'skip', 'mode': 'NIRSpec/PRISM', 'input_dir': '.',
        'filter_detector': 'NRS1', 'oof_method': 'median',
        'superbias_method': 'crds', 'extract_method': 'box',
        'pca_components': 2, 'remove_components': None,
        'flag_up_ramp': False, 'flag_in_time': True, 'jump_threshold': 15,
        'saturation_threshold': 80, 'mask_do_not_use_pixels': True, 'mask_saturated_pixels': True,
        'wave_range': [1., 5.], 'w1': 0., 'w2': 1., 'centroids': None,
        'opt_max_iter': 10, 'opt_var_thresh': 25,
    })
    opts.update(opts_patch)
    ctx = stages.prepare_nirspec_context(cube, opts, refpack=refpack)
    pipeline = stages.build_pipeline('NIRSpec/PRISM', ctx)
    return cube, ctx, pipeline


def _run_to_extract(cube, pipeline, params):
    """Return run to extract."""
    extract = [step.name for step in pipeline.steps].index('Extract')
    before = pipeline.run(PipelineState(cube), params, stop=extract)
    after = pipeline.run(before, params, start=extract)
    return before, after


def test_nirspec_optimal_graph_matches_v1_on_its_own_rate_cube(tmp_path):
    """Check NIRSpec optimal graph matches v1 on its own rate cube."""
    cube, ctx, pipeline = _nirspec_graph(extract_method='optimal')
    assert pipeline.steps[-1].fn is stages.step_extract_nirspec
    before, after = _run_to_extract(cube, pipeline, {'extract_width': 5})
    centroids = after.aux['centroids_extract']
    deepframe = np.asarray(before.aux['stage3_deepframe'])

    v1 = oracle.load(trace_start=ctx['nirspec_xstart'])
    _, ref_flux, ref_err, _ = v1.optimal_extract_nirspec(
        [oracle.segment(np.asarray(before.cube.data).copy())],
        deepframe.copy(), oracle.centroid_frame(**centroids), 5,
        max_iter=10, var_thresh=25)
    product = after.aux['spectral_products'][1]
    np.testing.assert_allclose(
        product['flux'],
        _v1_clip(ref_flux),
        rtol=2e-5, atol=1e-4)
    np.testing.assert_allclose(product['ferr'], ref_err, rtol=2e-5,
                               atol=1e-5)
    assert after.aux['optimal_clipped_counts']

    paths = products.output_layout(
        {'name_tag': 'nrs', 'extract_method': 'optimal'},
        output_dir=tmp_path, create=True)
    written = products.write_final_products(after, {'extract_width': 5},
                                            ctx, paths)
    assert written['spectra'].endswith('nrs_optimal_spectra_fullres.fits')
    with fits.open(written['spectra']) as hdul:
        assert hdul[0].header['METHOD'] == 'optimal'
        assert hdul[0].header['WIDTH'] == 'N/A'


def _v1_clip(flux):
    # ExoTEDRF 2.5.0 final light-curve clip (thresh=10, window=10).
    """Return v1 clip."""
    from exotedrf import utils as v1_utils
    return v1_utils.sigma_clip_lightcurves(flux, thresh=10, window=10)


def test_nirspec_two_sided_box_graph_matches_v1(tmp_path):
    """Check NIRSpec two sided box graph matches v1."""
    cube, ctx, pipeline = _nirspec_graph()
    width = v2config._normalize_width_value(
        {'lower': 2.5, 'upper': 4}, 'extract_width', allow_pair_list=False)
    before, after = _run_to_extract(cube, pipeline, {'extract_width': width})
    centroids = after.aux['centroids_extract']
    v1 = oracle.load(detector='nrs1', grating='PRISM', subarray='SUB512')
    _, ref_flux, ref_err, _, _ = v1.box_extract_nirspec(
        [oracle.segment(before.cube.data, before.cube.err, before.cube.dq)],
        oracle.centroid_frame(**centroids), {'lower': 2.5, 'upper': 4},
        mask_saturated_pixels=True, mask_do_not_use_pixels=True)
    product = after.aux['spectral_products'][1]
    np.testing.assert_allclose(
        product['flux'],
        _v1_clip(ref_flux),
        rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(product['ferr'], ref_err, rtol=1e-5,
                               atol=1e-4)
    paths = products.output_layout({'name_tag': 'nrs'}, output_dir=tmp_path,
                                   create=True)
    written = products.write_final_products(after, {'extract_width': width},
                                            ctx, paths)
    with fits.open(written['spectra']) as hdul:
        assert hdul[0].header['WIDTH'] == 'lower=2.5, upper=4'


def test_nirspec_custom_deepframe_drives_stage3_trace_only(tmp_path):
    """Check NIRSpec custom deepframe drives stage3 trace only."""
    cube, ctx, pipeline = _nirspec_graph()
    params = {'extract_width': 4}
    before, after = _run_to_extract(cube, pipeline, params)
    computed = np.asarray(before.aux['stage3_deepframe'])
    shifted = np.roll(computed, 2, axis=0)
    path = tmp_path / 'custom_deepframe.fits'
    fits.PrimaryHDU(shifted).writeto(path)

    cube2, ctx2, pipeline2 = _nirspec_graph(deepframe=str(path))
    np.testing.assert_allclose(ctx2['custom_deepframe'], shifted)
    before2, after2 = _run_to_extract(cube2, pipeline2, params)
    # Stage 1/2 are untouched; only the Stage-3 trace moves with the file.
    np.testing.assert_array_equal(np.asarray(before2.cube.data),
                                  np.asarray(before.cube.data))
    expected = stages._trace_nirspec_deepframe(shifted, ctx2)
    np.testing.assert_allclose(after2.aux['centroids_extract']['ypos'],
                               expected['ypos'])
    assert np.median(after2.aux['centroids_extract']['ypos'] -
                     after.aux['centroids_extract']['ypos']) == \
        pytest.approx(2., abs=0.1)
    # The Stage-2 deepframe sidecar remains the computed stack, as in v1.
    paths = products.output_layout({'name_tag': 'nrs'}, output_dir=tmp_path,
                                   create=True)
    written = products.write_final_products(after2, params, ctx2, paths)
    np.testing.assert_allclose(fits.getdata(written['deepframe']), computed)

    with pytest.raises(ValueError, match='expected the 2-D detector shape'):
        bad = tmp_path / 'bad.fits'
        fits.PrimaryHDU(np.zeros((3, 3))).writeto(bad)
        _nirspec_graph(deepframe=str(bad))


# MIRI: optimal through the graph, allow_miri_slope.

def _miri_graph(**opts_patch):
    """Return MIRI graph."""
    from .test_miri_stages import _ramp
    nints, ngroups, dimy, dimx = 12, 8, 30, 48
    rng = np.random.default_rng(9)
    yy, xx = np.mgrid[:dimy, :dimx]
    trace_profile = 80. * np.exp(-0.5 * ((xx - 36.) / 1.2) ** 2)
    slope = 20. + trace_profile
    integration = np.arange(nints, dtype=np.float32)[:, None, None, None]
    group = np.arange(ngroups, dtype=np.float32)[None, :, None, None]
    data = (100. + .01 * integration + group * slope[None, None] +
            rng.normal(0, .5, (nints, ngroups, dimy, dimx))).astype(
                np.float32)
    cube = _ramp(data, edges=[6, 12], starts=[1, 7])
    coeffs = np.zeros((2, dimy, dimx), np.float32)
    coeffs[1] = 1.
    zeros_u32 = np.zeros((dimy, dimx), np.uint32)
    zeros_f32 = np.zeros((dimy, dimx), np.float32)
    refpack = {
        'mask_dq': zeros_u32,
        'reset_data': np.zeros((1, ngroups, dimy, dimx), np.float32),
        'reset_dq': zeros_u32, 'lin_coeffs': coeffs, 'lin_dq': zeros_u32,
        'dark': np.zeros((1, ngroups, dimy, dimx), np.float32),
        'dark_dq': zeros_u32, 'average_dark_current': zeros_f32,
        'readnoise': np.ones((dimy, dimx), np.float32),
        'gain': np.ones((dimy, dimx), np.float32),
        'gain_factor': np.float32(1.),
        'flat': np.ones((dimy, dimx), np.float32), 'flat_dq': zeros_u32,
        'flat_err': zeros_f32, 'wave_map': 5. + .1 * yy + .001 * xx,
    }
    opts = {name: 'skip' for name in stages._MIRI_STEP_CONTROLS}
    for name in ('DQInitStep', 'ResetStep', 'LinearityStep', 'RampFitStep',
                 'GainScaleStep', 'AssignWCSStep', 'SourceTypeStep',
                 'FlatFieldStep'):
        opts[name] = 'run'
    opts.update({
        'INLCorrStep': 'skip', 'SuperBiasStep': 'skip', 'RefPixStep': 'skip',
        'DarkCurrentStep': 'skip', 'OneOverFStep_grp': 'skip',
        'OneOverFStep_int': 'skip', 'mode': 'MIRI/LRS',
        'extract_method': 'optimal', 'opt_max_iter': 10,
        'opt_var_thresh': 25, 'miri_subtract_dark': True,
        'miri_drop_groups': 1, 'miri_background_method': 'median',
        'saturation_threshold': 100, 'mask_do_not_use_pixels': True, 'mask_saturated_pixels': True,
        'hot_pixel_map': None, 'outlier_maps': None,
        'centroids': {'xpos': np.full(dimy, 36.),
                      'ypos': np.arange(dimy, dtype=float)},
    })
    opts.update(opts_patch)
    ctx = stages.prepare_miri_context(cube, opts, refpack=refpack)
    return cube, ctx, stages.build_pipeline('MIRI/LRS', ctx)


@pytest.mark.parametrize('width', [6., None])
def test_miri_optimal_runs_through_prepare_and_build_pipeline(width,
                                                              tmp_path):
    """Check MIRI optimal runs through prepare and build pipeline."""
    cube, ctx, pipeline = _miri_graph()
    assert pipeline.steps[-1].fn is stages.step_extract_miri
    extract = len(pipeline.steps) - 1
    before = pipeline.run(PipelineState(cube), {'extract_width': width},
                          stop=extract)
    deep = np.nanmedian(np.asarray(before.cube.data), axis=0)
    ctx['custom_deepframe'] = deep
    after = pipeline.run(before, {'extract_width': width}, start=extract)
    product = after.aux['spectral_products'][1]
    # 2.5.0 clipping NaNs runs of >= 10 high-variance channels (v1 rule).
    assert after.aux['optimal_clipped_counts']

    v1 = oracle.load()
    centroids = ctx['centroids']
    _, ref_flux, ref_err, _, _ = v1.optimal_extract_miri(
        [oracle.segment(np.asarray(before.cube.data).copy(), None,
                        np.zeros(before.cube.data.shape, np.uint32))],
        deep.astype(np.float32).copy(), oracle.centroid_frame(**centroids),
        width, max_iter=10, var_thresh=25, dq_report=True)
    np.testing.assert_allclose(
        product['flux'],
        _v1_clip(ref_flux),
        rtol=2e-5, atol=1e-4)
    np.testing.assert_allclose(product['ferr'], ref_err, rtol=2e-5,
                               atol=1e-5)
    paths = products.output_layout(
        {'name_tag': 'miri', 'extract_method': 'optimal'},
        output_dir=tmp_path, create=True)
    written = products.write_final_products(
        after, {'extract_width': width}, ctx, paths)
    with fits.open(written['spectra']) as hdul:
        assert hdul[0].header['WIDTH'] == 'N/A'


def test_miri_allow_slope_reaches_only_the_stage3_trace():
    """Check MIRI allow slope reaches only the stage3 trace."""
    dimy, dimx = 300, 72
    yy, xx = np.mgrid[:dimy, :dimx]
    centre = 30. + 0.02 * (yy - 150.)
    deep = 50. + 2000. * np.exp(-0.5 * ((xx - centre) / 1.3) ** 2)
    meta = ObsMeta('MIRI/LRS', 'MIRIMAGE', 'SLITLESSPRISM', 1., 1,
                   np.arange(4, dtype=float), np.array([2]), np.array([4]),
                   ())
    data = np.broadcast_to(deep, (4, dimy, dimx)).astype(np.float32)
    state = PipelineState(core.RateCube(
        data, np.ones_like(data), np.zeros(data.shape, np.uint32), meta),
        {'stage3_deepframe': deep})
    flat_ctx = {'opts': {}, 'miri_ystart': trace.MIRI_TRACE_YSTART}
    sloped_ctx = {'opts': {'stage3_allow_miri_slope': True},
                  'miri_ystart': trace.MIRI_TRACE_YSTART}
    stage3 = stages._miri_centroids_for_extraction(state, sloped_ctx, 'sum')
    expected = trace.get_centroids_miri(
        deep, ystart=trace.MIRI_TRACE_YSTART, allow_slope=True)
    np.testing.assert_allclose(stage3['xpos'], expected['xpos'])
    assert np.ptp(stage3['xpos']) > 1.
    flat = stages._miri_centroids_for_extraction(state, flat_ctx, 'sum')
    assert np.ptp(flat['xpos']) < 1e-9
    stage2 = stages._miri_centroids_for_stage(
        PipelineState(state.cube, {}), sloped_ctx, 'centroids_bkg')
    assert np.ptp(stage2['xpos']) < 1e-9
