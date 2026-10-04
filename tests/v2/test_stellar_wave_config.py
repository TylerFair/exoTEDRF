"""YAML handling for v1's Stage-3 wavelength options."""

import warnings

import pytest

from exotedrf.v2 import config

SOSS = {'observing_mode': 'NIRISS/SOSS', 'extract_method': 'box',
        'oof_method': 'scale-achromatic'}
NIRSPEC = {'observing_mode': 'NIRSpec/G395H', 'filter_detector': 'NRS1',
           'extract_method': 'box', 'oof_method': 'median',
           'INLCorrStep': 'skip', 'RefPixStep': 'skip',
           'FlatFieldStep': 'skip', 'BackgroundStep': 'skip'}
MIRI = {'observing_mode': 'MIRI/LRS', 'extract_method': 'box',
        'SuperBiasStep': 'skip', 'RefPixStep': 'skip',
        'DarkCurrentStep': 'skip', 'OneOverFStep_grp': 'skip',
        'OneOverFStep_int': 'skip', 'INLCorrStep': 'skip'}
STELLAR = {'st_teff': 5400, 'st_logg': 4.45, 'st_met': -0.12}


@pytest.mark.parametrize('base', [SOSS, NIRSPEC, MIRI])
def test_stellar_parameters_are_accepted_for_every_mode(base):
    """Check stellar parameters are accepted for every mode."""
    cfg = dict(base, **STELLAR)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        config.validate_supported_config(cfg)
    opts = config.fixed_options(cfg)
    assert (opts['st_teff'], opts['st_logg'], opts['st_met']) == \
        (5400, 4.45, -0.12)
    assert opts['stellar_model_dir'] is None
    assert opts['use_pastasoss'] is False


def test_defaults_leave_the_wavelength_solution_untouched():
    """Check defaults leave the wavelength solution untouched."""
    opts = config.fixed_options(dict(SOSS))
    assert opts['st_teff'] is opts['st_logg'] is opts['st_met'] is None
    assert opts['use_pastasoss'] is False


def test_partial_stellar_parameters_warn_like_v1():
    """Check partial stellar parameters warn like v1."""
    cfg = dict(SOSS, st_teff=5400, st_logg=None, st_met=None)
    with pytest.warns(RuntimeWarning, match='default wavelength solution'):
        config.validate_supported_config(cfg)


@pytest.mark.parametrize('patch', [
    {'st_teff': '5400'}, {'st_logg': float('nan')}, {'st_met': True},
    {'st_teff': -10}])
def test_invalid_stellar_parameters_are_rejected(patch):
    """Check invalid stellar parameters are rejected."""
    cfg = dict(SOSS, **dict(STELLAR, **patch))
    with pytest.raises(ValueError):
        config.validate_supported_config(cfg)


def test_model_dir_option(tmp_path):
    """Check model dir option."""
    cfg = dict(SOSS, **STELLAR, v2_stellar_model_dir=str(tmp_path))
    config.validate_supported_config(cfg)
    assert config.fixed_options(cfg)['stellar_model_dir'] == str(tmp_path)
    with pytest.raises(ValueError, match='v2_stellar_model_dir'):
        config.validate_supported_config(
            dict(SOSS, **STELLAR, v2_stellar_model_dir=3))


def test_use_pastasoss_via_stage3_kwargs():
    """Check use pastasoss via stage3 keywords."""
    cfg = dict(SOSS, stage3_kwargs={'Extract1dStep': {'use_pastasoss': True}})
    config.validate_supported_config(cfg)
    assert config.fixed_options(cfg)['use_pastasoss'] is True
    cfg['stage3_kwargs'] = {'Extract1dStep': {'use_pastasoss': False}}
    config.validate_supported_config(cfg)
    assert config.fixed_options(cfg)['use_pastasoss'] is False


@pytest.mark.parametrize('stage3', [
    {'Extract1dStep': {'use_pastasoss': True, 'not_a_v1_kwarg': 1}},
    {'Extract1dStep': {'opt_max_iter': 5}}])
def test_other_stage3_kwargs_remain_rejected(stage3):
    """Check other stage3 keywords remain rejected."""
    with pytest.raises(NotImplementedError, match='stage3_kwargs'):
        config.validate_supported_config(dict(SOSS, stage3_kwargs=stage3))


def test_non_boolean_use_pastasoss_is_rejected():
    """Check non boolean use pastasoss is rejected."""
    cfg = dict(SOSS, stage3_kwargs={'Extract1dStep': {'use_pastasoss': 'yes'}})
    with pytest.raises(TypeError, match='use_pastasoss'):
        config.validate_supported_config(cfg)


@pytest.mark.parametrize('base', [NIRSPEC, MIRI])
def test_use_pastasoss_is_ignored_outside_soss_like_v1(base):
    """Check use pastasoss is ignored outside SOSS like v1."""
    cfg = dict(base, stage3_kwargs={'Extract1dStep': {'use_pastasoss': True}})
    with pytest.warns(UserWarning, match='NIRISS/SOSS only'):
        config.validate_supported_config(cfg)
    assert config.fixed_options(cfg)['use_pastasoss'] is False


def test_shipped_yamls_are_unchanged_by_the_new_options():
    """Check shipped yamls are unchanged by the new options."""
    import os
    import exotedrf
    root = os.path.join(os.path.dirname(os.path.dirname(exotedrf.__file__)),
                        'tests', 'v2', 'data')
    for name in ('run_optimize_niriss.yaml', 'run_optimize_nirspec.yaml',
                 'run_optimize_miri.yaml'):
        cfg = config.load_config(os.path.join(root, name))
        opts = config.fixed_options(cfg)
        assert opts['st_teff'] is opts['st_logg'] is opts['st_met'] is None
        assert opts['use_pastasoss'] is False
