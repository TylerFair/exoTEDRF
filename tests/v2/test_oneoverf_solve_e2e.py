"""Filesystem-to-products optimizer run with ``oof_method: solve``."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from exotedrf.v2 import config as v2config
from exotedrf.v2 import stages, trace
from exotedrf.v2.optimize import run_optimizer

from .niriss_acceptance import apply_current_niriss_settings
from .test_yaml_niriss_e2e import (DIMX, DIMY, NGROUPS, SEGMENT_NINTS,
                                   Y_TRACE_O1, Y_TRACE_O2, _synthetic_refpack,
                                   _write_uncal_segment, _yaml_path)

FIXED = {'time_jump_threshold': 10, 'time_window': 5,
         'space_outlier_threshold': 10, 'time_outlier_threshold': 10,
         'box_size': 5, 'window_size': 5}


@pytest.fixture(scope='module')
def solve_run(tmp_path_factory):
    """Reduce synthetic SOSS segments using 1/f solving.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for temporary test directories.

    Returns
    -------
    result : types.SimpleNamespace
        Loaded functions or synthetic observation records.
    """
    base = tmp_path_factory.mktemp('solve_e2e')
    input_dir = base / 'inputs'
    input_dir.mkdir()
    rng = np.random.RandomState(7)
    exposure_nints = sum(SEGMENT_NINTS)
    int_start = 1
    for index, nints in enumerate(SEGMENT_NINTS, start=1):
        _write_uncal_segment(
            input_dir / f'jwsynth-seg{index:03d}_uncal.fits', nints,
            int_start, index, exposure_nints, rng)
        int_start += nints
    np.save(base / 'background_synth.npy',
            np.full((DIMY, DIMX), 1., np.float32))
    trace.save_centroids_csv(str(base / 'synth_centroids.csv'), {
        'xpos': np.arange(DIMX, dtype=float),
        'ypos o1': np.full(DIMX, Y_TRACE_O1),
        'ypos o2': np.full(DIMX, Y_TRACE_O2),
    })
    cfg = apply_current_niriss_settings(v2config.load_config(_yaml_path()))
    cfg.update({
        'input_dir': str(input_dir),
        'pipeline_outputs_directory': str(base / 'outputs'),
        'crds_cache_path': str(base / 'crds_cache'),
        'f277w': None,
        'soss_background_file': str(base / 'background_synth.npy'),
        'centroids': str(base / 'synth_centroids.csv'),
        'oof_method': 'solve',
        'do_plots': True,
        'soss_inner_mask_width': [36, 40, 44],
        'soss_outer_mask_width': [20, 30, 40],
        'extract_width': [24, 30, 36],
    })
    for name, value in FIXED.items():
        cfg[f'optimize_{name}'] = False
        cfg[name] = value
    result = run_optimizer(cfg, logger=None, refpack=_synthetic_refpack())
    return SimpleNamespace(cfg=cfg, result=result, base=base)


def test_solve_run_completes_and_inner_width_is_inert(solve_run):
    """Check solve run completes and inner width is inert."""
    result = solve_run.result
    assert np.isfinite(result.final_cost)
    inner = [t for t in result.trials
             if t.parameter == 'soss_inner_mask_width']
    outer = [t for t in result.trials
             if t.parameter == 'soss_outer_mask_width']
    assert [t.value for t in inner] == [36, 40, 44]
    assert [t.value for t in outer] == [20, 30, 40]
    assert len({t.cost for t in inner}) == 1
    assert result.winners['soss_inner_mask_width'] == 36
    assert len({t.cost for t in outer}) > 1
    assert all(np.isfinite(t.cost) for t in outer)


def test_solve_sidecars_and_plots_follow_v1(solve_run):
    """Check solve sidecars and plots follow v1."""
    result = solve_run.result
    nints = sum(SEGMENT_NINTS)
    root = solve_run.base / 'outputs' / 'v2'
    for level, directory, shape in (
            ('grp', 'Stage1', (nints, NGROUPS, DIMX)),
            ('int', 'Stage2', (nints, DIMX))):
        for parity in ('even', 'odd'):
            key = f'oofscaling_{level}_{parity}_order1'
            path = result.output_paths[key]
            assert path == str(root / directory /
                               f'jwsynth_oofscaling_{parity}_order1.npy')
            saved = np.load(path)
            assert saved.dtype == np.float64 and saved.shape == shape
            # Trace columns carry real, near-unity light-curve scalings.
            assert np.nanmedian(saved) == pytest.approx(1., abs=0.05)
        assert f'oofscaling_{level}_even_order2' not in result.output_paths
        plot = result.output_paths[f'oneoverf_solve_{level}_o1_plot']
        assert plot == str(root / directory / 'oneoverfstep_o1_3.png')
        assert os.path.exists(plot)
    for key in ('OneOverFStep_grp_plot_1', 'OneOverFStep_grp_plot_2',
                'OneOverFStep_int_plot_1', 'OneOverFStep_int_plot_2'):
        assert os.path.exists(result.output_paths[key])
