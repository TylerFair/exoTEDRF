"""``python -m exotedrf.v2.run_dms``: the run_DMS.yaml single-pass pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

from exotedrf.v2 import config as v2config
from exotedrf.v2 import run_dms

from .optimizer_modes_fixtures import (
    DIMX, DIMY, REPO, ROOT, soss_refpack, synthetic_rates, write_centroids,
    write_deepframe, write_rate_products)
from .test_yaml_niriss_e2e import (
    SEGMENT_NINTS, _synthetic_refpack, _write_uncal_segment)


def _dms_cfg(tmp_path, **updates):
    """Return dms cfg."""
    with (REPO / 'exotedrf' / 'run_DMS.yaml').open() as stream:
        cfg = yaml.safe_load(stream)
    for key in list(cfg):
        if cfg[key] == 'None':
            cfg[key] = None
    np.save(tmp_path / 'bkg.npy', np.ones((DIMY, DIMX), np.float32))
    cfg.update(
        crds_cache_path=str(tmp_path / 'crds'),
        input_dir=str(tmp_path / 'inputs'),
        pipeline_outputs_directory=str(tmp_path / 'out' /
                                       'pipeline_outputs_directory'),
        soss_background_file=str(tmp_path / 'bkg.npy'),
        centroids=write_centroids(tmp_path / 'cen.csv'),
        extract_width=10, baseline_ints=[50, -50], do_plots=False,
        saturation_rescue=True)
    cfg.update(updates)
    path = tmp_path / 'run_DMS.yaml'
    with path.open('w') as stream:
        yaml.safe_dump(cfg, stream)
    return path, cfg


def test_normalize_handles_v1_quirks():
    """Check normalize handles v1 quirks."""
    messages = []
    cfg, stages, deep = run_dms.normalize_dms_config(
        {'inl_amplitudes_file': 'a.npy', 'run_stages': [3, 2],
         'deepframe': 'd.fits', 'BadPixStep': 'run'},
        logger=messages.append)
    assert cfg['inl_amplitude_file'] == 'a.npy' and stages == [2, 3]
    assert deep == 'd.fits' and 'deepframe' not in cfg
    assert all(cfg[name] == 'skip' for name in run_dms.DMS_STAGE1_STEPS)
    assert cfg['BadPixStep'] == 'run'
    assert any('misspelling' in message for message in messages)
    _, _, _ = run_dms.normalize_dms_config(
        {'inl_amplitude_file': 'b.npy'}, logger=messages.append)
    assert any('raises KeyError' in message for message in messages)
    for bad in ({'run_stages': [4]}, {'run_stages': []},
                {'optimize_box_size': True},
                {'time_jump_threshold': [5, 6]}):
        with pytest.raises((ValueError, TypeError)):
            run_dms.normalize_dms_config(bad, logger=None)
    with pytest.raises(NotImplementedError, match="'optimize'"):
        run_dms.normalize_dms_config({'extract_width': 'optimize'},
                                     logger=None)


def test_shipped_run_dms_yaml_is_accepted():
    """Check shipped run dms YAML is accepted."""
    raw = v2config.load_config(REPO / 'exotedrf' / 'run_DMS.yaml')
    cfg, stages, _ = run_dms.normalize_dms_config(raw, logger=None)
    assert stages == [1, 2, 3]
    v2config.validate_supported_config(cfg)


def test_save_config_numbers_copies(tmp_path):
    """Check save config numbers copies."""
    source = tmp_path / 'cfg.yaml'
    source.write_text('a: 1\n')
    first = run_dms.save_config(source, tmp_path / 'root')
    second = run_dms.save_config(source, tmp_path / 'root')
    assert first.name == 'cfg.yaml' and second.name == 'cfg_1.yaml'
    assert 'Run at' in second.read_text()


@pytest.fixture(scope='module')
def uncal_run(tmp_path_factory):
    """Create synthetic uncalibrated SOSS segments and a run configuration.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for temporary test directories.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    base = tmp_path_factory.mktemp('run_dms_uncal')
    inputs = base / 'inputs'
    inputs.mkdir()
    rng = np.random.RandomState(7)
    start = 1
    for index, nints in enumerate(SEGMENT_NINTS, start=1):
        _write_uncal_segment(inputs / f'jwsynth-seg{index:03d}_uncal.fits',
                             nints, start, index, sum(SEGMENT_NINTS), rng)
        start += nints
    path, cfg = _dms_cfg(base, f277w=None)
    result = run_dms.run_dms(str(path), logger=None,
                             refpack=_synthetic_refpack())
    return base, path, cfg, result


def test_run_dms_full_recipe_from_uncal(uncal_run):
    """Check run dms full recipe from uncal."""
    base, path, cfg, result = uncal_run
    root = Path(cfg['pipeline_outputs_directory'])
    steps = result['summary']['steps']
    assert steps[0] == 'DQInitStep' and steps[-1] == 'Extract'
    assert 'JumpStep' in steps and 'PCAReconstructStep' in steps
    assert (root / 'config_files' / 'run_DMS.yaml').exists()
    products = result['products']
    spectra = Path(products['spectra'])
    assert spectra.parent == root / 'Stage3'
    assert fits.getheader(spectra)['WIDTH'] == 10
    for key in ('rate', 'deepframe', 'hot_pixels', 'stability',
                'background', 'lcestimate'):
        assert Path(products[key]).exists(), key
    # V1 saves no centroid table when one was supplied.
    assert 'centroids' not in products
    summary = json.loads(Path(products['summary']).read_text())
    assert summary['run_stages'] == [1, 2, 3]


def test_run_dms_stage3_cache_and_restart(uncal_run, tmp_path):
    """Check run dms stage3 cache and restart."""
    base, path, cfg, result = uncal_run
    stage2 = tmp_path / 'stage2_products'
    cube = synthetic_rates()
    write_rate_products(stage2, 'pcareconstructstep', cube)
    deep = write_deepframe(tmp_path / 'deep.fits', cube[0])
    path3, cfg3 = _dms_cfg(tmp_path, run_stages=[3], deepframe=deep,
                           input_dir=str(stage2),
                           input_filetag='pcareconstructstep')
    first = run_dms.run_dms(str(path3), logger=None, refpack=soss_refpack())
    assert first['summary']['steps'] == ['Extract']
    messages = []
    again = run_dms.run_dms(str(path3), logger=messages.append,
                            refpack=soss_refpack())
    assert again['summary']['steps'] == []
    assert any('skipping Stage 3' in message for message in messages)
    assert again['products']['spectra'] == first['products']['spectra']


def test_run_dms_resumes_from_v1_cached_products(tmp_path):
    """Check run dms resumes from v1 cached products."""
    inputs = tmp_path / 'inputs'
    cube = synthetic_rates()
    write_rate_products(inputs, 'gainscalestep', cube)
    path, cfg = _dms_cfg(tmp_path, input_filetag='gainscalestep',
                         run_stages=[2, 3], OneOverFStep_int='skip',
                         BackgroundStep='skip', generate_lc=True)
    stage2 = Path(cfg['pipeline_outputs_directory']) / 'Stage2'
    for tag in ('assignwcsstep', 'sourcetypestep', 'flatfieldstep'):
        write_rate_products(stage2, tag, cube)
    hot = np.zeros((DIMY, DIMX), bool)
    hot[60:62, 70:90] = True
    np.save(stage2 / f'{ROOT}_nis_hot_pixels.npy', hot)
    messages = []
    result = run_dms.run_dms(str(path), logger=messages.append,
                             refpack=soss_refpack())
    assert result['summary']['resumed_after'] == 'FlatFieldStep'
    assert result['summary']['steps'] == ['BadPixStep',
                                          'PCAReconstructStep', 'Extract']
    assert any('reusing the cached hot_pixels.npy' in m for m in messages)
    np.testing.assert_array_equal(np.load(result['products']['hot_pixels']),
                                  hot)
    # Force_redo=True ignores every cache, as in v1.
    path2, _ = _dms_cfg(tmp_path, input_filetag='gainscalestep',
                        run_stages=[2, 3], OneOverFStep_int='skip',
                        BackgroundStep='skip', force_redo=True)
    redo = run_dms.run_dms(str(path2), logger=None, refpack=soss_refpack())
    assert redo['summary']['resumed_after'] is None
    assert redo['summary']['steps'][:3] == ['AssignWCSStep',
                                            'FlatFieldStep', 'BadPixStep']
