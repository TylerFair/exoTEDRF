"""Restart from v1-format intermediate products (input_filetag != uncal)."""

import numpy as np
import pytest
from astropy.io import fits

from exotedrf.v2 import config as v2config
from exotedrf.v2 import io, restart, stages
from exotedrf.v2.optimize import run_optimizer

from .optimizer_modes_fixtures import (
    DIMX, DIMY, ROOT, SEGMENTS, base_cfg, soss_refpack, synthetic_rates,
    write_centroids, write_rate_products)


def test_find_products_uses_v1_substring_and_segment_order(tmp_path):
    """Check find products uses v1 substring and segment order."""
    paths = write_rate_products(tmp_path, 'badpixstep')
    write_rate_products(tmp_path, 'flatfieldstep')
    # A different instrument and a non-FITS sidecar must be ignored.
    fits.PrimaryHDU(header=fits.Header({'INSTRUME': 'NIRSPEC'})).writeto(
        tmp_path / 'other-seg001_nrs1_badpixstep.fits')
    (tmp_path / 'notes_badpixstep.txt').write_text('x')
    found = io.find_products(tmp_path, 'badpix', mode='NIRISS/SOSS',
                             filter_detector='CLEAR')
    assert found == paths
    # Ordering follows EXSEGNUM, not the directory listing.
    assert [fits.getheader(p)['EXSEGNUM'] for p in found] == [1, 2]
    assert io.find_products(tmp_path, 'pcareconstructstep',
                            mode='NIRISS/SOSS') == []


def test_load_rate_products_round_trips_v1_cubemodels(tmp_path):
    """Check load rate products round trips v1 cubemodels."""
    sci, err, dq = synthetic_rates()
    files = write_rate_products(tmp_path, 'gainscalestep', (sci, err, dq))
    cube = io.load_rate_products(files, [50, -50], mode='NIRISS/SOSS',
                                 filter_detector='CLEAR')
    np.testing.assert_array_equal(np.asarray(cube.data), sci)
    np.testing.assert_array_equal(np.asarray(cube.err), err)
    assert cube.dq.dtype == np.uint32
    np.testing.assert_array_equal(np.asarray(cube.dq), dq)
    assert int(cube.dq[3, 50, 60]) == 1 << 31
    meta = cube.meta
    assert tuple(meta.segment_edges) == (SEGMENTS[0], sum(SEGMENTS))
    assert tuple(meta.segment_int_starts) == (1, SEGMENTS[0] + 1)
    assert meta.ngroups == 2 and meta.subarray == 'SUBSTRIP96'
    np.testing.assert_allclose(meta.int_times,
                               60000. + np.arange(sum(SEGMENTS)) * 1e-3)
    assert meta.filenames == tuple(files)


def test_v1_fileroots_match_v1_utils(tmp_path):
    """Check v1 fileroots match v1 utils."""
    files = write_rate_products(tmp_path, 'badpixstep')
    roots = io.v1_fileroots(files)
    assert roots == [f'{ROOT}-seg001_nis_', f'{ROOT}-seg002_nis_']
    assert io.v1_fileroot_noseg(roots) == f'{ROOT}_nis_'
    from exotedrf import utils as v1utils
    assert roots == v1utils.get_filename_root(files)
    assert io.v1_fileroot_noseg(roots) == \
        v1utils.get_filename_root_noseg(roots)


@pytest.mark.parametrize('tag, ndim, step', [
    ('gainscalestep', 3, 'GainScaleStep'), ('jump', 4, 'JumpStep'),
    ('backgroundstep', 4, 'BackgroundStep_grp'),
    ('backgroundstep', 3, 'BackgroundStep'),
    ('oneoverfstep', 4, 'OneOverFStep_grp'),
    ('oneoverfstep', 3, 'OneOverFStep_int'),
    ('pcareconstructstep', 3, 'PCAReconstructStep'),
    ('rateints', 3, None), ('uncal', 4, None)])
def test_producing_step(tag, ndim, step):
    """Check producing step."""
    assert restart.producing_step(tag, ndim) == step


def test_restart_rejects_impossible_graphs():
    """Check restart rejects impossible graphs."""
    opts = v2config.fixed_options(base_cfg(BadPixStep='run',
                                           LinearityStep='run'))
    with pytest.raises(ValueError, match='post-RampFit'):
        restart.check_restart_steps(opts, 3, 'gainscalestep')
    opts = v2config.fixed_options(base_cfg(BadPixStep='run',
                                           JumpStep='run'))
    with pytest.raises(ValueError, match='need RampFitStep'):
        restart.check_restart_steps(opts, 4, 'linearitystep')


def test_restart_warns_about_reapplied_steps():
    """Check restart warns about reapplied steps."""
    messages = []
    opts = v2config.fixed_options(base_cfg(FlatFieldStep='run',
                                           BadPixStep='run'))
    restart.check_restart_steps(opts, 3, 'flatfieldstep',
                                logger=messages.append)
    assert any('FlatFieldStep' in m and 'applied again' in m
               for m in messages)


def test_required_reftypes_cover_only_enabled_steps():
    """Check required reftypes cover only enabled steps."""
    opts = v2config.fixed_options(base_cfg(FlatFieldStep='run',
                                           BadPixStep='run'))
    assert restart.required_reftypes(opts, 'NIRISS/SOSS') == ['flat']
    opts = v2config.fixed_options(base_cfg(LinearityStep='run',
                                           JumpStep='run',
                                           RampFitStep='run',
                                           GainScaleStep='run'))
    assert restart.required_reftypes(opts, 'NIRISS/SOSS') == [
        'linearity', 'readnoise', 'gain']


def test_cached_resume_point_follows_v1_expected_files(tmp_path):
    """Check cached resume point follows v1 expected files."""
    files = write_rate_products(tmp_path, 'flatfieldstep')
    roots = io.v1_fileroots(files)
    names = ['FlatFieldStep', 'BackgroundStep', 'BadPixStep',
             'PCAReconstructStep']
    assert restart.cached_resume_point(names, tmp_path, roots) == \
        (0, 'flatfieldstep')
    write_rate_products(tmp_path, 'badpixstep')
    assert restart.cached_resume_point(names, tmp_path, roots)[1] == \
        'flatfieldstep'
    np.save(tmp_path / f'{ROOT}_nis_hot_pixels.npy',
            np.zeros((DIMY, DIMX), bool))
    assert restart.cached_resume_point(names, tmp_path, roots)[1] == \
        'badpixstep'
    # ... and a single missing segment invalidates a step.
    (tmp_path / f'{ROOT}-seg002_nis_badpixstep.fits').unlink()
    assert restart.cached_resume_point(names, tmp_path, roots)[1] == \
        'flatfieldstep'


def _restart_cfg(tmp_path, **updates):
    """Return restart cfg."""
    stage = tmp_path / 'Stage2'
    write_rate_products(stage, 'flatfieldstep')
    cfg = base_cfg(
        input_dir=str(stage), input_filetag='flatfieldstep',
        pipeline_outputs_directory=str(tmp_path / 'out'),
        centroids=write_centroids(tmp_path / 'cen.csv'),
        BadPixStep='run', PCAReconstructStep='run',
        optimize_space_outlier_threshold=True,
        space_outlier_threshold=[5, 9], time_outlier_threshold=10,
        box_size=5, window_size=5, optimize_extract_width=True,
        extract_width=[8, 12])
    cfg.update(updates)
    return cfg


def _direct(cfg, params):
    """Reference: load the products and run the restarted graph directly."""
    opts = v2config.fixed_options(cfg)
    opts['dtype'] = np.float32
    state, files = restart.load_restart_state(opts)
    opts = restart.restart_opts(opts, state.cube)
    ctx = stages.prepare_context(state.cube, opts, refpack=soss_refpack())
    pipeline = stages.build_pipeline(opts['mode'], ctx)
    final = pipeline.run(state, params)
    cost, _ = stages.evaluate_production_cost(final, params, ctx)
    return final, float(cost), [step.name for step in pipeline.steps]


def test_optimizer_restarts_from_stage2_products(tmp_path):
    """Check optimizer restarts from stage2 products."""
    cfg = _restart_cfg(tmp_path)
    result = run_optimizer(cfg, logger=None, refpack=soss_refpack(),
                           output_dir=tmp_path / 'v2')
    swept = [t.parameter for t in result.trials]
    assert swept == ['space_outlier_threshold'] * 2 + ['extract_width'] * 2
    assert all(t.phase == 1 for t in result.trials[:2])
    assert result.summary['phase1_segment'].endswith(
        'seg001_nis_flatfieldstep.fits')
    final, cost, names = _direct(cfg, dict(result.winners))
    assert names == ['BadPixStep', 'PCAReconstructStep', 'Extract']
    np.testing.assert_allclose(result.final_cost, cost, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(result.final_state.cube.data),
                                  np.asarray(final.cube.data))


def test_debug_mode_scores_only_first_candidate(tmp_path):
    """Check debug mode scores only first candidate."""
    cfg = _restart_cfg(tmp_path, debug_mode=True)
    messages = []
    result = run_optimizer(cfg, logger=messages.append,
                           refpack=soss_refpack(),
                           output_dir=tmp_path / 'v2')
    phase1 = [t for t in result.trials if t.phase == 1]
    assert [t.value for t in phase1] == [5, 9]
    assert phase1[0].cost == phase1[1].cost and phase1[1].duration_s == 0.0
    assert result.winners['space_outlier_threshold'] == 5
    assert any('DEBUG MODE ENABLED' in m for m in messages)
    # The first candidate's cost is the ordinary (non-debug) trial cost.
    plain = run_optimizer(_restart_cfg(tmp_path / 'b'), logger=None,
                          refpack=soss_refpack(), output_dir=tmp_path / 'p')
    np.testing.assert_allclose(phase1[0].cost, plain.trials[0].cost,
                               rtol=1e-7)
    # Phase 2 (the aperture sweep) is unaffected, as in v1.
    widths = [t.cost for t in result.trials if t.phase == 2]
    assert len(set(widths)) == 2


def test_badpix_reuses_v1_hot_pixel_map(tmp_path):
    """Check bad-pixel correction reuses v1 hot pixel map."""
    cfg = base_cfg(BadPixStep='run')
    sci, err, dq = synthetic_rates()
    files = write_rate_products(tmp_path, 'backgroundstep', (sci, err, dq))
    opts = v2config.fixed_options(dict(cfg, input_dir=str(tmp_path),
                                       input_filetag='backgroundstep'))
    opts['dtype'] = np.float32
    state, _ = restart.load_restart_state(opts, files=files)
    opts = restart.restart_opts(opts, state.cube)
    ctx = stages.prepare_context(state.cube, opts, refpack=soss_refpack())
    params = {'space_outlier_threshold': 5, 'time_outlier_threshold': 10,
              'box_size': 5, 'window_size': 5}
    fresh = stages.step_badpix(state, params, ctx)
    reused_map = np.zeros((DIMY, DIMX), bool)
    reused_map[30:33, 50:60] = True
    ctx['badpix_reuse_map'] = reused_map
    reused = stages.step_badpix(state, params, ctx)
    np.testing.assert_array_equal(reused.aux['hot_pixel_map'], reused_map)
    assert not np.array_equal(np.asarray(fresh.cube.data),
                              np.asarray(reused.cube.data))
    ctx['badpix_reuse_map'] = fresh.aux['hot_pixel_map']
    again = stages.step_badpix(state, params, ctx)
    np.testing.assert_array_equal(np.asarray(again.cube.data),
                                  np.asarray(fresh.cube.data))
