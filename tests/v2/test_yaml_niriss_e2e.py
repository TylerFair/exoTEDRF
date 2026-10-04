"""End-to-end executability of the v2 optimizer on the NIRISS YAML template."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from exotedrf.v2 import config as v2config
from exotedrf.v2 import stages, trace
from exotedrf.v2.optimize import run_optimizer

from .test_f277w import _v1_order0_mask
from .niriss_acceptance import apply_current_niriss_settings

YAML_PATH = 'tests/v2/data/run_optimize_niriss.yaml'

SEGMENT_NINTS = (64, 56)
NGROUPS = 2
DIMY, DIMX = 96, 128
Y_TRACE_O1, Y_TRACE_O2 = 20., 40.
F277W_BLOB = (slice(30, 50), slice(100, 103))


def _yaml_path():
    """Return YAML path."""
    import exotedrf.v2 as v2pkg
    import os
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(v2pkg.__file__))))
    return os.path.join(repo, YAML_PATH)


def _write_uncal_segment(path, nints, int_start, exsegnum, exposure_nints,
                         rng):
    """One realistic-layout SOSS uncal segment (uint16 SCI + INT_TIMES)."""
    yy = np.arange(DIMY, dtype=float)[:, None]
    profile = (600. * np.exp(-0.5 * ((yy - Y_TRACE_O1) / 3.) ** 2) +
               250. * np.exp(-0.5 * ((yy - Y_TRACE_O2) / 3.) ** 2) + 20.)
    column_shape = 1. + 0.25 * np.sin(np.arange(DIMX) / 13.)
    rate = profile * column_shape[None, :]
    group_scale = (1. + np.arange(NGROUPS))[:, None, None]
    data = np.empty((nints, NGROUPS, DIMY, DIMX), np.float64)
    for i in range(nints):
        data[i] = (1500. + rate[None] * group_scale +
                   rng.normal(0., 3., (NGROUPS, DIMY, DIMX)))
    sci = np.clip(np.round(data), 0, 65535).astype(np.uint16)

    ph = fits.PrimaryHDU()
    header = ph.header
    header['INSTRUME'] = 'NIRISS'
    header['DETECTOR'] = 'NIS'
    header['SUBARRAY'] = 'SUBSTRIP96'
    header['EXP_TYPE'] = 'NIS_SOSS'
    header['FILTER'] = 'CLEAR'
    header['PUPIL'] = 'GR700XD'
    header['TGROUP'] = 5.494
    header['TFRAME'] = 5.494
    header['NFRAMES'] = 1
    header['GROUPGAP'] = 0
    header['NGROUPS'] = NGROUPS
    header['NINTS'] = exposure_nints
    header['INTSTART'] = int_start
    header['INTEND'] = int_start + nints - 1
    header['EXSEGNUM'] = exsegnum
    times = fits.BinTableHDU.from_columns(fits.ColDefs([fits.Column(
        name='int_mid_BJD_TDB', format='D',
        array=60000. + (int_start - 1 + np.arange(nints)) * 1e-4)]),
        name='INT_TIMES')
    fits.HDUList([ph, fits.ImageHDU(sci, name='SCI'), times]).writeto(path)


def _synthetic_refpack():
    """Every reference required by the YAML's enabled steps."""
    lin_coeffs = np.zeros((2, DIMY, DIMX), np.float32)
    lin_coeffs[1] = 1.
    return {
        'mask_dq': np.zeros((DIMY, DIMX), np.uint32),
        'inl_theta': np.zeros(6, np.float32),
        'inl_periods': np.asarray([1024 / 3, 512, 1024], np.float32),
        'superbias': np.full((DIMY, DIMX), 1500., np.float32),
        'lin_coeffs': lin_coeffs,
        'lin_dq': np.zeros((DIMY, DIMX), np.uint32),
        'readnoise': np.full((DIMY, DIMX), 6., np.float32),
        'gain': np.ones((DIMY, DIMX), np.float32),
        'gain_factor': np.float32(1.),
        'flat': np.ones((DIMY, DIMX), np.float32),
        'wave_o1': np.linspace(2.75, 0.87, DIMX),
        'wave_o2': np.linspace(1.4, 0.6, DIMX),
    }


def _make_f277w(rng):
    """Create f277w for the test observation."""
    frame = rng.normal(0., 1., (DIMY, DIMX))
    frame[F277W_BLOB] += 40.
    return frame


@pytest.fixture(scope='module')
def yaml_run(tmp_path_factory):
    """Reduce synthetic SOSS segments using the optimizer YAML template.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for temporary test directories.

    Returns
    -------
    result : np.ndarray(float)
        Calculated reference array.
    """
    base = tmp_path_factory.mktemp('yaml_niriss_e2e')
    input_dir = base / 'inputs'
    input_dir.mkdir()
    outputs = base / 'outputs'
    rng = np.random.RandomState(42)

    exposure_nints = sum(SEGMENT_NINTS)
    int_start = 1
    for index, nints in enumerate(SEGMENT_NINTS, start=1):
        _write_uncal_segment(
            input_dir / f'jwsynth-seg{index:03d}_uncal.fits', nints,
            int_start, index, exposure_nints, rng)
        int_start += nints

    f277w = _make_f277w(rng)
    np.save(base / 'f277w_synth.npy', f277w)
    np.save(base / 'background_synth.npy',
            np.full((DIMY, DIMX), 1., np.float32))
    trace.save_centroids_csv(str(base / 'synth_centroids.csv'), {
        'xpos': np.arange(DIMX, dtype=float),
        'ypos o1': np.full(DIMX, Y_TRACE_O1),
        'ypos o2': np.full(DIMX, Y_TRACE_O2),
    })

    cfg = apply_current_niriss_settings(
        v2config.load_config(_yaml_path()))
    # Environment-bound overrides ONLY (see module docstring).
    cfg['input_dir'] = str(input_dir)
    cfg['pipeline_outputs_directory'] = str(outputs)
    cfg['crds_cache_path'] = str(base / 'crds_cache')
    cfg['f277w'] = str(base / 'f277w_synth.npy')
    cfg['soss_background_file'] = str(base / 'background_synth.npy')
    cfg['centroids'] = str(base / 'synth_centroids.csv')

    contexts = []

    def context_factory(cube, opts, refpack=None, phase=None, **_):
        ctx = stages.prepare_context(cube, opts, refpack=refpack)
        contexts.append((phase, ctx))
        return ctx

    result = run_optimizer(cfg, logger=None, refpack=_synthetic_refpack(),
                           context_factory=context_factory)
    return SimpleNamespace(cfg=cfg, result=result, contexts=contexts,
                           base=base, outputs=outputs, f277w=f277w)


def _enabled_sweeps(cfg):
    """{parameter: candidate list} for every optimize_*=True in the YAML."""
    return {param: list(cfg[param]) for param in v2config.SWEEP_PARAMETERS
            if cfg.get(f'optimize_{param}', False)}


def test_yaml_run_completes_with_finite_costs(yaml_run):
    """Check YAML run completes with finite costs."""
    result = yaml_run.result
    assert np.isfinite(result.final_cost)
    costs = np.asarray([trial.cost for trial in result.trials], float)
    assert costs.size > 0 and np.isfinite(costs).all()
    scatter = np.asarray(result.final_scatter, float)
    assert np.isfinite(scatter).any()
    assert not np.isinf(scatter).any()


def test_every_enabled_yaml_sweep_ran_with_winner_in_candidates(yaml_run):
    """Check every enabled YAML sweep ran with winner in candidates."""
    cfg, result = yaml_run.cfg, yaml_run.result
    sweeps = _enabled_sweeps(cfg)
    # The production YAML enables exactly these nine sweeps.
    assert sorted(sweeps) == sorted([
        'soss_inner_mask_width', 'soss_outer_mask_width',
        'time_jump_threshold', 'time_window', 'space_outlier_threshold',
        'time_outlier_threshold', 'box_size', 'window_size',
        'extract_width'])
    for param, candidates in sweeps.items():
        assert param in result.winners, param
        assert result.winners[param] in candidates, param


def test_trials_cover_every_candidate_in_yaml_order(yaml_run):
    """Check trials cover every candidate in YAML order."""
    cfg, result = yaml_run.cfg, yaml_run.result
    sweeps = _enabled_sweeps(cfg)
    for param, candidates in sweeps.items():
        tried = [trial.value for trial in result.trials
                 if trial.parameter == param]
        assert tried == candidates, param
    assert len(result.trials) == sum(len(c) for c in sweeps.values())


def test_extract_sweep_ran_in_phase_two(yaml_run):
    """Check extract sweep ran in phase two."""
    result = yaml_run.result
    extract_trials = [trial for trial in result.trials
                      if trial.parameter == 'extract_width']
    assert len(extract_trials) == len(yaml_run.cfg['extract_width'])
    assert all(trial.phase == 2 and trial.checkpoint == 'Extract'
               for trial in extract_trials)
    assert all(trial.phase == 1 for trial in result.trials
               if trial.parameter != 'extract_width')
    # Phase 1 scored the first sorted segment with the YAML's upper-middle extraction width.
    assert result.summary['phase1_segment'].endswith('seg001_uncal.fits')
    assert result.summary['phase1_extract_width'] == \
        v2config.phase1_extract_width(yaml_run.cfg)


def test_optimizer_logs_and_final_products_exist(yaml_run):
    """Check optimizer logs and final products exist."""
    import json
    import os

    result = yaml_run.result
    root = yaml_run.outputs / 'v2'
    for key in ('cost', 'scatter', 'summary', 'rate', 'lcestimate', 'spectra',
                'centroids', 'deepframe', 'hot_pixels', 'background', 'stability',
                'contaminant_mask'):
        assert key in result.output_paths, key
        path = result.output_paths[key]
        assert os.path.exists(path), path
        assert os.path.commonpath([path, str(root)]) == str(root)

    diagnostic_keys = {
        'cost_plot', 'scatter_plot', 'white_plot', 'flux_plot',
        'centroid_plot', 'dqinit_plot', 'inlcorrstep_plot_1',
        'inlcorrstep_plot_2', 'superbias_plot',
        'BackgroundStep_grp_plot_1', 'BackgroundStep_grp_plot_2',
        'OneOverFStep_grp_plot_1', 'OneOverFStep_grp_plot_2',
        'linearitystep_plot_1', 'linearitystep_plot_2', 'jump_plot',
        'contaminant_plot', 'BackgroundStep_plot_1',
        'BackgroundStep_plot_2', 'OneOverFStep_int_plot_1',
        'OneOverFStep_int_plot_2', 'badpix_plot', 'pca_plot',
    }
    assert diagnostic_keys <= result.output_paths.keys()
    for key in diagnostic_keys:
        path = result.output_paths[key]
        assert path.endswith('.png')
        assert os.path.exists(path), path
        assert os.path.commonpath([path, str(root)]) == str(root)

    cost_lines = open(result.output_paths['cost']).read().splitlines()
    assert len(cost_lines) == 1 + len(result.trials)
    # Cost columns follow the YAML's optimize_* declaration order.
    expected_columns = [key[len('optimize_'):] for key, value
                        in yaml_run.cfg.items()
                        if key.startswith('optimize_') and value is True]
    assert cost_lines[0].split('\t') == \
        expected_columns + ['duration_s', 'cost']

    summary = json.loads(open(result.output_paths['summary']).read())
    assert summary['final_cost'] == pytest.approx(result.final_cost)
    assert summary['winners'] == {
        key: value for key, value in result.summary['winners'].items()}

    with fits.open(result.output_paths['spectra']) as hdu:
        names = [h.name for h in hdu]
        for name in ('WAVE O1', 'FLUX O1', 'FLUX ERR O1',
                     'WAVE O2', 'FLUX O2'):
            assert name in names
        assert hdu['FLUX O1'].data.shape[0] == sum(SEGMENT_NINTS)
        assert np.isfinite(hdu['FLUX O1'].data).all()
    with fits.open(result.output_paths['rate']) as hdu:
        assert hdu['SCI'].data.shape == (sum(SEGMENT_NINTS), DIMY, DIMX)
    lcestimate = np.load(result.output_paths['lcestimate'], allow_pickle=False)
    assert lcestimate.shape == (sum(SEGMENT_NINTS),)
    assert np.isfinite(lcestimate).all()


def test_f277w_order0_mask_wiring_is_live(yaml_run):
    """Check f277w order0 mask wiring is live."""
    phases = sorted(phase for phase, _ in yaml_run.contexts)
    assert phases == [1, 2]
    for _, ctx in yaml_run.contexts:
        order0 = ctx['order0_mask']
        assert order0 is not None
        assert order0.shape == (DIMY, DIMX)
        assert order0.dtype == np.bool_
        assert not order0.any()


def test_f277w_blob_masks_nonempty_when_search_region_is_reachable(yaml_run):
    """Check f277w blob masks nonempty when search region is reachable."""
    f277w = np.load(yaml_run.base / 'f277w_synth.npy')
    mask = trace.make_order0_mask_from_f277w(f277w, start_col=0)
    assert mask.any()
    assert mask[30:50, 99:103].all()
    np.testing.assert_array_equal(
        mask, _v1_order0_mask(f277w, start_col=0).astype(bool))
    wide = np.zeros((DIMY, 2048))
    wide[F277W_BLOB[0], 800:803] = 40.
    wide_mask = trace.make_order0_mask_from_f277w(wide)
    assert wide_mask[30:50, 799:803].all()


def test_streamed_optimizer_matches_standard_winners_and_spectra(yaml_run):
    """Exercise lazy FITS -> both optimizer phases -> final products."""
    cfg = dict(yaml_run.cfg, v2_stream_stage1='auto', v2_output_mode='optimal',
               v2_max_host_bytes='8MiB', v2_scratch_dir=str(yaml_run.base),
               pipeline_outputs_directory=str(yaml_run.base / 'streamed'))
    streamed = run_optimizer(cfg, logger=None, refpack=_synthetic_refpack())
    standard = yaml_run.result
    assert streamed.summary['stream_stage1'] is True
    assert streamed.summary['diagnostic_policy']['mode'] == 'optimal'
    assert streamed.winners == standard.winners
    np.testing.assert_allclose(streamed.final_cost, standard.final_cost,
                               rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(streamed.final_scatter, standard.final_scatter,
                               rtol=1e-4, atol=1e-6, equal_nan=True)
    assert len(streamed.trials) == len(standard.trials)
    for a, b in zip(streamed.trials, standard.trials):
        assert (a.phase, a.parameter, a.value) == (b.phase, b.parameter, b.value)
        np.testing.assert_allclose(a.cost, b.cost, rtol=1e-4, atol=1e-6)
    np.testing.assert_array_equal(streamed.final_state.cube.dq,
                                  standard.final_state.cube.dq)
    for order in (1, 2):
        for key in ('wave', 'flux', 'ferr'):
            np.testing.assert_allclose(
                streamed.final_state.aux['spectral_products'][order][key],
                standard.final_state.aux['spectral_products'][order][key],
                rtol=1e-4, atol=1e-6, equal_nan=True)
    assert {'summary', 'spectra'} <= set(streamed.output_paths)
    assert 'rate' not in streamed.output_paths
    assert any(path.endswith('.png') for path in streamed.output_paths.values())


def test_streamed_standard_mode_matches_the_joined_standard_run(yaml_run):
    """Segment scheduling is a memory schedule, not a science variant."""
    import os

    cfg = dict(yaml_run.cfg, v2_stream_stage1=True,
               pipeline_outputs_directory=str(yaml_run.base / 'streamed_std'))
    streamed = run_optimizer(cfg, logger=None, refpack=_synthetic_refpack())
    standard = yaml_run.result

    assert streamed.summary['output_mode'] == 'standard'
    assert streamed.summary['stream_stage1'] is True
    assert streamed.winners == standard.winners
    assert len(streamed.trials) == len(standard.trials)
    for a, b in zip(streamed.trials, standard.trials):
        assert (a.phase, a.checkpoint, a.parameter, a.value) == \
               (b.phase, b.checkpoint, b.parameter, b.value)
        np.testing.assert_array_equal(np.float64(a.cost), np.float64(b.cost))
    np.testing.assert_array_equal(np.float64(streamed.final_cost),
                                  np.float64(standard.final_cost))
    np.testing.assert_array_equal(np.asarray(streamed.final_scatter),
                                  np.asarray(standard.final_scatter))

    for name in ('data', 'err', 'dq'):
        np.testing.assert_array_equal(
            np.asarray(getattr(streamed.final_state.cube, name)),
            np.asarray(getattr(standard.final_state.cube, name)))
    for order in (1, 2):
        for key in ('wave', 'flux', 'ferr'):
            np.testing.assert_array_equal(
                np.asarray(
                    streamed.final_state.aux['spectral_products'][order][key]),
                np.asarray(
                    standard.final_state.aux['spectral_products'][order][key]))

    # The standard product writers still run from the joined host state.
    for key in ('cost', 'scatter', 'summary', 'rate', 'lcestimate', 'spectra',
                'centroids', 'deepframe', 'hot_pixels', 'background',
                'stability', 'contaminant_mask'):
        assert key in streamed.output_paths, key
        assert os.path.exists(streamed.output_paths[key]), key
    with fits.open(streamed.output_paths['rate']) as hdu:
        assert hdu['SCI'].data.shape == (sum(SEGMENT_NINTS), DIMY, DIMX)
        np.testing.assert_array_equal(
            hdu['SCI'].data,
            np.asarray(standard.final_state.cube.data))
    for key in ('dqinit_plot', 'superbias_plot', 'jump_plot',
                'linearitystep_plot_1', 'badpix_plot', 'pca_plot'):
        assert key in streamed.output_paths, key
        assert os.path.exists(streamed.output_paths[key]), key


_PINNED = {'v2_max_host_bytes': '2GiB'}


def test_phase1_device_residency_and_automatic_streaming_change_nothing(
        yaml_run, monkeypatch, tmp_path):
    """Schedules, not science: three schedules, one set of numbers."""
    monkeypatch.setenv('EXOTEDRF_MAX_DEVICE_BYTES', str(1 << 31))
    assert 'v2_stream_stage1' not in yaml_run.cfg

    def run(name, **env):
        for key in ('EXOTEDRF_PHASE1_DEVICE', 'EXOTEDRF_STREAM_STAGE1'):
            monkeypatch.setenv(key, env.get(key, '0'))
        cfg = dict(yaml_run.cfg, **_PINNED,
                   pipeline_outputs_directory=str(tmp_path / name))
        return run_optimizer(cfg, logger=None, refpack=_synthetic_refpack())

    host = run('host')
    assert host.summary['stream_stage1'] is False
    resident = run('resident', EXOTEDRF_PHASE1_DEVICE='1')
    streamed = run('streamed', EXOTEDRF_PHASE1_DEVICE='1',
                   EXOTEDRF_STREAM_STAGE1='1')
    # The automatic policy streamed purely because of the backend test.
    assert streamed.summary['stream_stage1'] is True

    for other in (resident, streamed):
        assert other.winners == host.winners
        assert len(other.trials) == len(host.trials)
        for a, b in zip(other.trials, host.trials):
            assert (a.phase, a.checkpoint, a.parameter, a.value) == \
                   (b.phase, b.checkpoint, b.parameter, b.value)
            np.testing.assert_array_equal(np.float64(a.cost),
                                          np.float64(b.cost))
        np.testing.assert_array_equal(np.float64(other.final_cost),
                                      np.float64(host.final_cost))
        np.testing.assert_array_equal(np.asarray(other.final_scatter),
                                      np.asarray(host.final_scatter))
        for name in ('data', 'err', 'dq'):
            np.testing.assert_array_equal(
                np.asarray(getattr(other.final_state.cube, name)),
                np.asarray(getattr(host.final_state.cube, name)))
        for order in (1, 2):
            for key in ('wave', 'flux', 'ferr'):
                np.testing.assert_array_equal(
                    np.asarray(
                        other.final_state.aux['spectral_products'][order][key]),
                    np.asarray(
                        host.final_state.aux['spectral_products'][order][key]))
