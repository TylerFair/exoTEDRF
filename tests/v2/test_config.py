"""Check reduction and optimizer configuration."""

from pathlib import Path

import numpy as np
import pytest

from exotedrf.v2 import config


def test_optimal_output_mode_keeps_diagnostics_and_validates():
    """Check optimal output mode keeps diagnostics and validates."""
    cfg = {'observing_mode': 'NIRISS/SOSS', 'do_plots': True}
    assert config.fixed_options(cfg)['do_plots']
    cfg['v2_output_mode'] = 'optimal'
    assert config.fixed_options(cfg)['output_mode'] == 'optimal'
    assert config.fixed_options(cfg)['do_plots']
    assert config.fixed_options({'v2_output_mode': 'optimal'})['do_plots']
    assert not config.fixed_options(dict(cfg, do_plots=False))['do_plots']
    assert config.fixed_options(cfg)['stream_stage1'] == 'auto'
    assert config.fixed_options(
        {'observing_mode': 'NIRISS/SOSS'})['stream_stage1'] == 'auto'
    cfg['v2_output_mode'] = 'typo'
    with pytest.raises(ValueError, match='v2_output_mode'):
        config.validate_supported_config(cfg)


def test_sweep_plan_from_shipped_yaml():
    """Check sweep plan from shipped YAML."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg = config.load_config(repo_root / 'exotedrf' / 'run_optimize.yaml')
    plan, initial = config.build_sweep_plan(cfg, 'NIRISS/SOSS')
    names = [cp for cp, _ in plan]
    assert names == ['OneOverFStep_grp', 'JumpStep', 'BadPixStep', 'Extract']
    grp = dict(plan)['OneOverFStep_grp']
    assert [p for p, _ in grp] == ['soss_inner_mask_width',
                                   'soss_outer_mask_width']
    for param in ('soss_inner_mask_width', 'soss_outer_mask_width',
                  'extract_width'):
        assert initial[param] == int(np.mean(cfg[param]))
    # Nirspec param not applicable to SOSS: excluded everywhere.
    assert not any(p == 'nirspec_mask_width' for _, g in plan for p, _ in g)
    assert 'nirspec_mask_width' not in initial


@pytest.mark.parametrize('name', ['run_optimize.yaml',
                                  'run_optimize_niriss.yaml'])
def test_shipped_soss_configs_are_inside_supported_boundary(name):
    """Check shipped SOSS configs are inside supported boundary."""
    repo_root = Path(__file__).resolve().parents[2]
    directory = 'tests/v2/data' if name.startswith('run_optimize_') else 'exotedrf'
    cfg = config.load_config(repo_root / directory / name)
    assert config.validate_supported_config(cfg) is cfg


@pytest.mark.parametrize(
    ('name', 'mode'),
    [('run_optimize_nirspec.yaml', 'NIRSpec/G395M'),
     ('run_optimize_miri.yaml', 'MIRI/LRS')])
def test_shipped_instrument_configs_are_supported(name, mode):
    """Check NIRSpec and MIRI optimizer configurations."""
    repo_root = Path(__file__).resolve().parents[2]
    directory = 'tests/v2/data' if name.startswith('run_optimize_') else 'exotedrf'
    cfg = config.load_config(repo_root / directory / name)
    assert cfg['observing_mode'] == mode
    assert cfg['input_filetag'] == 'uncal'
    assert config.validate_supported_config(cfg) is cfg
    # Disabled coordinates must be fixed scalars, not latent sweep lists.
    config.validate_optimizer_values(cfg)


def test_nirspec_sweep_plan_and_default_spectral_ranges():
    """Check NIRSpec sweep plan and default spectral ranges."""
    cfg = {
        'observing_mode': 'NIRSpec/G395H',
        'filter_detector': 'NRS1',
        'nirspec_mask_width': [10, 14, 18],
        'optimize_nirspec_mask_width': True,
        'OneOverFStep_grp': 'run',
        'extract_width': 7,
        'optimize_extract_width': False,
    }
    config.validate_supported_config(cfg)
    plan, initial = config.build_sweep_plan(cfg, cfg['observing_mode'])
    assert [checkpoint for checkpoint, _ in plan] == ['OneOverFStep_grp']
    assert initial['nirspec_mask_width'] == 14
    assert config.fixed_options(cfg)['wave_range'] == [3.0, 3.5]

    cfg['filter_detector'] = 'NRS2'
    assert config.fixed_options(cfg)['wave_range'] == [4.0, 4.5]


def test_miri_sweep_plan_and_default_spectral_range():
    """Check MIRI sweep plan and default spectral range."""
    cfg = {
        'observing_mode': 'MIRI/LRS',
        'miri_trace_width': [14, 18, 22],
        'optimize_miri_trace_width': True,
        'miri_background_width': 12,
        'optimize_miri_background_width': False,
        'BackgroundStep': 'run',
        'extract_width': 6,
        'optimize_extract_width': False,
        'w2': 1.0,
    }
    config.validate_supported_config(cfg)
    plan, initial = config.build_sweep_plan(cfg, cfg['observing_mode'])
    assert [checkpoint for checkpoint, _ in plan] == ['BackgroundStep']
    assert initial['miri_trace_width'] == 18
    assert config.fixed_options(cfg)['wave_range'] == [5.0, 10.0]


def test_miri_accepts_optimal_extraction_controls():
    """Check MIRI accepts optimal extraction controls."""
    cfg = {
        'observing_mode': 'MIRI/LRS',
        'extract_method': 'optimal',
        'extract_width': 10,
        'opt_max_iter': 7,
        'opt_var_thresh': 36,
    }
    config.validate_supported_config(cfg)
    opts = config.fixed_options(cfg)
    assert opts['extract_method'] == 'optimal'
    assert opts['opt_max_iter'] == 7
    assert opts['opt_var_thresh'] == 36


@pytest.mark.parametrize(
    'key, value, match',
    [('opt_max_iter', -1, 'non-negative'),
     ('opt_max_iter', True, 'non-negative'),
     ('opt_var_thresh', 0, 'positive'),
     ('extract_width', 'optimize', 'finite and positive')])
def test_miri_rejects_invalid_optimal_controls(key, value, match):
    """Check MIRI rejects invalid optimal controls."""
    cfg = {
        'observing_mode': 'MIRI/LRS', 'extract_method': 'optimal',
        'extract_width': 10, key: value,
    }
    with pytest.raises(ValueError, match=match):
        config.validate_supported_config(cfg)


def test_search_options_defaults_and_joint_forces_beam_width_one():
    """Check search options defaults and joint forces beam width one."""
    assert config.search_options({}) == {
        'strategy': 'greedy', 'beam_width': 3, 'final_candidates': 3,
        'group_max_evals': 4096, 'tree_tolerance': None,
    }
    joint = config.search_options(
        {'v2_search_strategy': 'joint', 'v2_beam_width': 5})
    assert joint['strategy'] == 'joint'
    assert joint['beam_width'] == 1
    assert joint['final_candidates'] == 1

    beam = config.search_options({
        'v2_search_strategy': 'BEAM', 'v2_beam_width': 4,
        'v2_final_candidates': 2, 'v2_group_max_evals': 128,
    })
    assert beam == {
        'strategy': 'beam', 'beam_width': 4, 'final_candidates': 2,
        'group_max_evals': 128, 'tree_tolerance': None,
    }

    tree = config.search_options({'v2_search_strategy': 'tree'})
    assert tree == {
        'strategy': 'tree', 'beam_width': 8, 'final_candidates': 8,
        'group_max_evals': 4096, 'tree_tolerance': 1e-3,
    }
    tree = config.search_options({
        'v2_search_strategy': 'tree', 'v2_tree_tolerance': 0.0,
        'v2_beam_width': 2})
    assert tree['tree_tolerance'] == 0.0 and tree['beam_width'] == 2


@pytest.mark.parametrize(
    'patch, match',
    [({'v2_search_strategy': 'bogus'}, 'v2_search_strategy'),
     ({'v2_beam_width': 0}, 'v2_beam_width'),
     ({'v2_beam_width': True}, 'v2_beam_width'),
     ({'v2_beam_width': 1.5}, 'v2_beam_width'),
     ({'v2_final_candidates': 0}, 'v2_final_candidates'),
     ({'v2_group_max_evals': -1}, 'v2_group_max_evals')])
def test_search_options_rejects_bad_values(patch, match):
    """Check search options rejects bad values."""
    with pytest.raises(ValueError, match=match):
        config.search_options(patch)

    cfg = {'observing_mode': 'NIRISS/SOSS', 'extract_method': 'box',
           'oof_method': 'scale-achromatic'}
    cfg.update(patch)
    with pytest.raises(ValueError, match=match):
        config.validate_supported_config(cfg)


def test_fixed_options_none_strings():
    """Check fixed options none strings."""
    cfg = {'centroids': 'None', 'observing_mode': 'NIRISS/SOSS'}
    import yaml, tempfile, os
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.dump(cfg, f)
        path = f.name
    loaded = config.load_config(path)
    os.unlink(path)
    assert loaded['centroids'] is None
    opts = config.fixed_options(loaded)
    assert opts['mode'] == 'NIRISS/SOSS'
    assert opts['w2'] == 1.0
    assert opts['wave_range'] == [1.0, 2.0]


def test_explicit_null_saturation_threshold_uses_v1_default():
    """Check explicit null saturation threshold uses v1 default."""
    assert config.fixed_options({'saturation_threshold': None})[
        'saturation_threshold'] == 80
    assert config.fixed_options({'saturation_threshold': 73})[
        'saturation_threshold'] == 73


def test_omitted_baseline_ints_uses_v1_optimizer_default():
    """Check omitted baseline ints uses v1 optimizer default."""
    assert config.fixed_options({})['baseline_ints'] == [100, -100]
    assert config.fixed_options({'baseline_ints': [12, -9]})[
        'baseline_ints'] == [12, -9]


def test_v2_host_memory_yaml_options_are_forwarded():
    """Check v2 host memory YAML options are forwarded."""
    opts = config.fixed_options({
        'v2_max_host_bytes': '12GiB',
        'v2_scratch_dir': '/path/to/exotedrf',
    })
    assert opts['max_host_bytes'] == '12GiB'
    assert opts['scratch_dir'] == '/path/to/exotedrf'


def test_soss_default_wave_range_only_when_spectral_cost_is_active():
    """Check SOSS default wave range only when spectral cost is active."""
    assert config.fixed_options({'wave_range': None, 'w2': 1.0})[
        'wave_range'] == [1.0, 2.0]
    assert config.fixed_options({'wave_range': None, 'w2': 0.0})[
        'wave_range'] is None
    assert config.fixed_options({'wave_range': [0.8, 1.4], 'w2': 1.0})[
        'wave_range'] == [0.8, 1.4]
    with pytest.raises(ValueError, match='within'):
        config.fixed_options({'wave_range': [0.5, 1.4], 'w2': 1.0})


def test_none_conversion_is_exact_and_top_level(tmp_path):
    """Check none conversion is exact and top level."""
    path = tmp_path / 'config.yaml'
    path.write_text(
        'exact: None\nlower: none\nnested:\n  value: None\n',
        encoding='utf-8')
    loaded = config.load_config(path)
    assert loaded['exact'] is None
    assert loaded['lower'] == 'none'
    assert loaded['nested']['value'] == 'None'


@pytest.mark.parametrize(
    ('flag', 'value', 'message'),
    [(True, 5, 'must be list'),
     (False, [5], 'must be single value'),
     (True, [], 'non-empty list')])
def test_optimizer_list_scalar_contract(flag, value, message):
    """Check optimizer list scalar contract."""
    cfg = {'optimize_box_size': flag, 'box_size': value}
    with pytest.raises(ValueError, match=message):
        config.validate_optimizer_values(cfg)


def test_unknown_enabled_optimizer_flag_fails_closed():
    """Check unknown enabled optimizer flag fails closed."""
    cfg = {
        'mystery_width': [1, 2],
        'optimize_mystery_width': True,
    }
    with pytest.raises(ValueError, match='unknown enabled optimizer flag'):
        config.validate_supported_config(cfg)

    config.validate_optimizer_values({
        'optimize_mystery_width': False,
        'optimize_extract_width_only': False,
        'optimize_from_pca_only': False,
    })


def test_candidate_order_duplicates_and_noop_sweep_rejection():
    """Check candidate order duplicates and noop sweep rejection."""
    cfg = {
        'soss_inner_mask_width': [9, 3, 3, 1],
        'optimize_soss_inner_mask_width': True,
        'OneOverFStep_grp': 'run',
    }
    plan, initial = config.build_sweep_plan(cfg, 'NIRISS/SOSS')
    assert dict(plan)['OneOverFStep_grp'][0][1].tolist() == [9, 3, 3, 1]
    assert initial['soss_inner_mask_width'] == 4

    cfg['OneOverFStep_grp'] = 'skip'
    with pytest.raises(ValueError, match='requires OneOverFStep_grp=run'):
        config.build_sweep_plan(cfg, 'NIRISS/SOSS')

    cfg.update(OneOverFStep_grp='run', time_window=[3, 5],
               optimize_time_window=True, flag_in_time=False,
               flag_up_ramp=True)
    with pytest.raises(ValueError, match='requires flag_in_time=True'):
        config.build_sweep_plan(cfg, 'NIRISS/SOSS')


def test_fixed_options_forwards_graph_controls_and_soss_inputs():
    """Check fixed options forwards graph controls and SOSS inputs."""
    cfg = {
        'DarkCurrentStep': 'run',
        'JumpStep': 'skip',
        'inl_amplitude_file': 'theta.npy',
        'inl_periods': [1, 2],
        'soss_timeseries': 'o1.npy',
        'soss_timeseries_o2': 'o2.npy',
        'hot_pixel_map': 'hot.npy',
        'outlier_maps': ['seg1.npy', 'seg2.npy'],
        'extract_width_soss2': 18,
        'saturation_rescue': True,
        'mask_do_not_use_pixels': False,
        'do_plots': True,
        'crds_context': 'jwst_test.pmap',
        'wavemap_file': 'detector_wavelength.npy',
    }
    opts = config.fixed_options(cfg)
    for key, value in cfg.items():
        assert opts[key] == value
    assert opts['DQInitStep'] == 'run'
    assert opts['PCAReconstructStep'] == 'run'


def test_missing_dark_control_uses_v1_run_default():
    """Check missing dark control uses v1 run default."""
    opts = config.fixed_options({})
    assert config.STEP_DEFAULTS['DarkCurrentStep'] == 'run'
    assert opts['DarkCurrentStep'] == 'run'


@pytest.mark.parametrize('patch, match', [
    ({'observing_mode': 'NIRCam/TSO'}, 'supports NIRISS/SOSS'),
    ({'extract_method': 'doublegauss'}, 'box only'),
    ({'oof_method': 'bogus'}, 'supported methods'),
    ({'stage1_kwargs': {'JumpStep': {'foo': 1}}}, 'stage1_kwargs'),
    ({'stage1_kwargs': {'RefPixStep': {'odd_even_rows': False}}}, 'jwst pass-through'),
])
def test_rejects_unsupported_configurations(patch, match):
    """Reject configurations without a supported reduction equivalent."""
    cfg = {'observing_mode': 'NIRISS/SOSS', 'extract_method': 'box',
           'oof_method': 'scale-achromatic'}
    cfg.update(patch)
    with pytest.raises(NotImplementedError, match=match):
        config.validate_supported_config(cfg)


def test_custom_outlier_maps_are_supported_in_normal_soss_mode():
    """Check custom outlier maps are supported in normal SOSS mode."""
    cfg = {'observing_mode': 'NIRISS/SOSS', 'extract_method': 'box',
           'oof_method': 'scale-achromatic',
           'outlier_maps': ['segment1.npy', 'segment2.fits']}
    assert config.validate_supported_config(cfg) is cfg
    assert config.fixed_options(cfg)['outlier_maps'] == cfg['outlier_maps']


def test_supported_oof_auxiliaries_reach_fixed_options():
    """Check supported 1/f auxiliaries reach fixed options."""
    cfg = {'observing_mode': 'NIRISS/SOSS', 'extract_method': 'box',
           'oof_method': 'scale-achromatic',
           'outlier_maps': ['mask.npy'], 'f277w': 'f277w.npy'}
    config.validate_supported_config(cfg)
    opts = config.fixed_options(cfg)
    assert opts['outlier_maps'] == ['mask.npy']
    assert opts['f277w'] == 'f277w.npy'


@pytest.mark.parametrize('mode, expected', [
    ('NIRISS/SOSS', 'run'),
    ('NIRSpec/G395H', 'skip'),
    ('MIRI/LRS', 'skip'),
])
def test_fixed_options_inl_defaults_follow_v1_instrument_gating(
        mode, expected):
    """Check fixed options inverse linearity defaults follow v1 instrument gating."""
    cfg = {'observing_mode': mode,
           'filter_detector': 'NRS1' if 'NIRS' in mode.upper() else ''}
    if mode.startswith('MIRI'):
        cfg['filter_detector'] = ''
    assert config.fixed_options(cfg)['INLCorrStep'] == expected
    cfg['INLCorrStep'] = 'run'
    if expected == 'run':
        assert config.fixed_options(cfg)['INLCorrStep'] == 'run'
    else:
        with pytest.warns(config.V1CompatibilityWarning, match='INLCorrStep'):
            assert config.fixed_options(cfg)['INLCorrStep'] == 'skip'


@pytest.mark.parametrize('override', [
    {'observing_mode': 'NIRCam/WFSS'},
    {'observing_mode': 'NIRSpec/G395H', 'filter_detector': 'NRS1',
     'superbias_method': 'custom-rescale'},
    {'RampFitStep': 'skip'},
    {'v2_stream_stage1': 'true'},
])
def test_stream_stage1_rejects_unsupported_workflows(override):
    """Check stream stage1 rejects unsupported workflows."""
    cfg = dict(observing_mode='NIRISS/SOSS', v2_output_mode='optimal',
               v2_stream_stage1=True)
    config.validate_supported_config(cfg)
    with pytest.raises(ValueError, match='v2_stream_stage1'):
        config.validate_supported_config(dict(cfg, **override))


@pytest.mark.parametrize('mode,extra', [
    ('NIRSpec/G395H', {'filter_detector': 'NRS1'}),
    ('NIRSpec/PRISM', {'filter_detector': 'NRS1'}),
    ('MIRI/LRS', {'filter_detector': ''}),
])
def test_stream_stage1_supports_nirspec_and_miri(mode, extra):
    """Check stream stage1 supports NIRSpec and MIRI."""
    cfg = dict(observing_mode=mode, v2_stream_stage1=True, **extra)
    assert config.supports_stage1_streaming(cfg)
    config.validate_supported_config(cfg)
    assert config.fixed_options(cfg)['stream_stage1'] is True
    # A visit-derived NIRSpec superbias has no per-segment equivalent.
    assert config.supports_stage1_streaming(
        dict(cfg, superbias_method='custom')) is mode.startswith('MIRI')


def test_stream_stage1_allows_standard_output_mode():
    """Check stream stage1 allows standard output mode."""
    cfg = dict(observing_mode='NIRISS/SOSS', v2_output_mode='standard',
               v2_stream_stage1=True)
    assert config.supports_stage1_streaming(cfg)
    config.validate_supported_config(cfg)
    assert config.fixed_options(cfg)['stream_stage1'] is True
    assert config.stream_stage1_mode(
        dict(cfg, v2_stream_stage1=False)) is False
