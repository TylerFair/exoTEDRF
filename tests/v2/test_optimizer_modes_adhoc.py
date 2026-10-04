"""optimize_extract_width_only and from_pca_only on synthetic v1 products."""

import numpy as np
import pandas as pd
import pytest

from exotedrf.v2 import config as v2config
from exotedrf.v2 import restart, stages
from exotedrf.v2.optimize import run_optimizer

from .optimizer_modes_fixtures import (
    base_cfg, soss_refpack, synthetic_rates, v1_optimize_functions,
    write_centroids, write_deepframe, write_rate_products)

WIDTHS = [8, 10, 12, 14]


@pytest.fixture()
def products(tmp_path):
    """Write synthetic reduction products for restart tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.

    Returns
    -------
    result : dict
        Calculation results and associated metadata.
    """
    root = tmp_path / 'run'
    stage2, stage3 = root / 'Stage2', root / 'Stage3'
    cube = synthetic_rates()
    badpix = write_rate_products(stage2, 'badpixstep', cube)
    write_deepframe(stage2 / 'jwsynth_04102_00001_nis_deepframe.fits', cube[0])
    write_centroids(stage3 / 'jwsynth_04102_00001_nis_centroids.csv')
    return {'root': root, 'stage2': stage2, 'stage3': stage3,
            'badpix': badpix, 'cube': cube}


def _cfg(products, **updates):
    """Return cfg."""
    cfg = base_cfg(pipeline_outputs_directory=str(products['root']),
                   input_dir=str(products['stage2']),
                   optimize_extract_width=True, extract_width=list(WIDTHS),
                   # A first-pass YAML: lists behind disabled flags are fine.
                   optimize_box_size=False, box_size=[3, 5])
    cfg.update(updates)
    return cfg


def _direct_costs(files, cfg, remove=None):
    """Return direct costs."""
    opts = v2config.fixed_options(cfg)
    for name in ('PCAReconstructStep', 'BadPixStep'):
        opts[name] = 'skip'
    opts['remove_components'] = remove
    if remove is not None:
        opts['PCAReconstructStep'] = 'run'
    opts['dtype'] = np.float32
    opts['centroids'] = cfg['centroids']
    state, _ = restart.load_restart_state(opts, files=files)
    opts = restart.restart_opts(opts, state.cube)
    ctx = stages.prepare_context(state.cube, opts, refpack=soss_refpack())
    if remove is not None:
        state = stages.step_pca(state, {}, ctx)
    costs = []
    for width in WIDTHS:
        out = stages.step_extract(state, {'extract_width': width}, ctx)
        costs.append(float(stages.evaluate_production_cost(out, {}, ctx)[0]))
    return state, costs


def test_extract_width_only_matches_direct_extraction(products, tmp_path):
    """Check extract width only matches direct extraction."""
    cfg = _cfg(products, optimize_extract_width_only=True)
    result = run_optimizer(cfg, logger=None, refpack=soss_refpack(),
                           output_dir=tmp_path / 'v2')
    centroids = str(next(products['stage3'].glob('*centroids.csv')))
    _, costs = _direct_costs(products['badpix'],
                             dict(cfg, centroids=centroids))
    np.testing.assert_allclose([t.cost for t in result.trials], costs,
                               rtol=1e-6)
    assert result.winners['extract_width'] == WIDTHS[int(np.argmin(costs))]
    assert result.summary['input_products'] == products['badpix']
    log = (tmp_path / 'v2' / 'Files' / 'Cost_synth.txt').read_text()
    assert log.splitlines()[0] == \
        'ad_hoc_mode\tremove_components\textract_width\tduration_s\tcost'
    assert [line.split('\t')[:3] for line in log.splitlines()[1:]] == [
        ['extract_width_only', 'None', str(width)] for width in WIDTHS]
    assert not list((tmp_path / 'v2' / 'Stage3').glob('*centroids.csv'))
    assert 'spectra' in result.products


def test_extract_only_prefers_pca_products_when_components_removed(
        products, tmp_path):
    """Check extract only prefers PCA products when components removed."""
    pca_cube = tuple(np.copy(a) for a in products['cube'])
    pca_cube[0][...] *= 1.01
    pca = write_rate_products(products['stage2'], 'pcareconstructstep',
                              pca_cube)
    cfg = _cfg(products, optimize_extract_width_only=True,
               remove_components=[2], PCAReconstructStep='run')
    result = run_optimizer(cfg, logger=None, refpack=soss_refpack(),
                           output_dir=tmp_path / 'v2')
    assert result.summary['input_products'] == pca
    assert result.summary['timing_s']['pca'] == 0.
    cfg['remove_components'] = None
    result = run_optimizer(cfg, logger=None, refpack=soss_refpack(),
                           output_dir=tmp_path / 'v2b')
    assert result.summary['input_products'] == products['badpix']


def test_extract_only_requires_a_deepframe_like_v1(products, tmp_path):
    """Check extract only requires a deepframe like v1."""
    next(products['stage2'].glob('*deepframe.fits')).unlink()
    with pytest.raises(ValueError, match='Deepframe must be provided'):
        run_optimizer(_cfg(products, optimize_extract_width_only=True),
                      logger=None, refpack=soss_refpack(),
                      output_dir=tmp_path / 'v2')


def test_extract_only_reuses_first_pass_width(products, tmp_path):
    """Check extract only reuses first pass width."""
    first = run_optimizer(
        _cfg(products, optimize_extract_width_only=True,
             optimize_extract_width=False, extract_width=12),
        logger=None, refpack=soss_refpack(), output_dir=tmp_path / 'v2')
    assert first.winners['extract_width'] == 12
    again = run_optimizer(
        _cfg(products, optimize_extract_width_only=True,
             optimize_extract_width=False, extract_width=None,
             reuse_first_pass_extract_width=True),
        logger=None, refpack=soss_refpack(), output_dir=tmp_path / 'v2')
    assert again.winners['extract_width'] == 12
    np.testing.assert_allclose(again.final_cost, first.final_cost)


def test_from_pca_only_reruns_pca_and_matches_v1_log(products, tmp_path):
    """Check from PCA only reruns PCA and matches v1 log."""
    files = tmp_path / 'v2' / 'Files'
    files.mkdir(parents=True)
    seed = ('space_outlier_threshold\textract_width\tduration_s\tcost\n'
            '5\t10\t3.0\t0.002\n7\t12\t3.0\t0.001\n')
    (files / 'Cost_synth.txt').write_text(seed)
    cfg = _cfg(products, from_pca_only=True, remove_components=[2])
    result = run_optimizer(cfg, logger=None, refpack=soss_refpack(),
                           output_dir=tmp_path / 'v2')
    centroids = str(next(products['stage3'].glob('*centroids.csv')))
    expected_state, costs = _direct_costs(
        products['badpix'], dict(cfg, centroids=centroids), remove=[2])
    np.testing.assert_array_equal(np.asarray(result.final_state.cube.data),
                                  np.asarray(expected_state.cube.data))
    np.testing.assert_allclose([t.cost for t in result.trials], costs,
                               rtol=1e-6)
    # V1's own helpers, given the same rows, write the identical log.
    v1_dir = tmp_path / 'v1'
    v1_dir.mkdir()
    (v1_dir / 'Cost_synth.txt').write_text(seed)
    v1 = v1_optimize_functions('prepare_cost_log', 'append_cost_log_row',
                               'format_log_value', outdir_f=str(v1_dir))
    _, cols, _, best = v1['prepare_cost_log'](
        'synth', ['ad_hoc_mode', 'remove_components', 'extract_width'])
    merged = dict(best, ad_hoc_mode='from_pca_only', remove_components=[2])
    for trial in result.trials:
        v1['append_cost_log_row'](str(v1_dir / 'Cost_synth.txt'), cols,
                                  dict(merged, extract_width=trial.value),
                                  trial.duration_s, trial.cost)
    assert (files / 'Cost_synth.txt').read_text() == \
        (v1_dir / 'Cost_synth.txt').read_text()
    table = pd.read_csv(files / 'Cost_synth.txt', sep='\t',
                        keep_default_na=False)
    assert list(table['space_outlier_threshold'][-len(WIDTHS):]) == \
        [7] * len(WIDTHS)
    # V1 PCAReconstructStep side products.
    for key in ('deepframe', 'stability', 'lcestimate', 'spectra'):
        assert key in result.products, key


def test_from_pca_only_requires_badpix_products(products, tmp_path):
    """Check from PCA only requires bad-pixel correction products."""
    for path in products['badpix']:
        import os
        os.unlink(path)
    with pytest.raises(FileNotFoundError, match='No BadPix Step outputs'):
        run_optimizer(_cfg(products, from_pca_only=True,
                           remove_components=[2]),
                      logger=None, refpack=soss_refpack(),
                      output_dir=tmp_path / 'v2')
