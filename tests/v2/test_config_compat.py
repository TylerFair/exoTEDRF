"""v1-legal configurations that v1 ignores or coerces."""

from pathlib import Path
import warnings

import numpy as np
import pytest

from exotedrf.v2 import config, stage_kwargs

REPO = Path(__file__).resolve().parents[2]
NIRSPEC = {'observing_mode': 'NIRSpec/G395H', 'filter_detector': 'NRS1'}
MIRI = {'observing_mode': 'MIRI/LRS'}
SOSS = {'observing_mode': 'NIRISS/SOSS'}
W = config.V1CompatibilityWarning


def _messages(record):
    """Return messages."""
    return [str(item.message) for item in record
            if issubclass(item.category, W)]


def _normalized(cfg):
    """Return normalized."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        out = config.normalize_v1_config(cfg)
    return out, _messages(record)


# ---------------------------------------------------------------- item 1.

def test_nirspec_scale_achromatic_becomes_median_like_v1():
    """Check NIRSpec scale achromatic becomes median like v1."""
    cfg = dict(NIRSPEC, oof_method='scale-achromatic')
    with pytest.warns(W, match="uses 'median' for NIRSpec"):
        assert config.validate_supported_config(cfg) is cfg
    assert cfg['oof_method'] == 'scale-achromatic'
    assert config.fixed_options(cfg)['oof_method'] == 'median'
    # An omitted method resolves to median silently; slope is kept.
    out, messages = _normalized(NIRSPEC)
    assert out['oof_method'] == 'median' and not messages
    assert _normalized(dict(NIRSPEC, oof_method='slope'))[0][
        'oof_method'] == 'slope'
    with pytest.raises(NotImplementedError, match='median or slope'):
        config.validate_supported_config(
            dict(NIRSPEC, oof_method='scale-chromatic'))


# ---------------------------------------------------------------- item 2.

@pytest.mark.parametrize('base, instrument', [
    (SOSS, 'NIRISS'), (NIRSPEC, 'NIRSPEC'), (MIRI, 'MIRI')])
def test_v1_noop_step_switches_are_skipped_with_warning(base, instrument):
    """Check v1 noop step switches are skipped with warning."""
    noops = stage_kwargs.V1_NOOP_STEPS[instrument]
    cfg = dict(base, **{name: 'run' for name in noops})
    out, messages = _normalized(cfg)
    for name in noops:
        assert out[name] == 'skip'
        assert any(message.startswith(f"{name}='run' has no effect")
                   for message in messages)
    config.validate_supported_config(cfg)
    opts = config.fixed_options(cfg)
    assert all(opts[name] == 'skip' for name in noops if name in opts)
    # Steps v1 does run for the instrument are untouched.
    assert opts['DQInitStep'] == 'run' and opts['BadPixStep'] == 'run'


def test_omitted_noop_switches_skip_silently():
    """Check omitted noop switches skip silently."""
    out, messages = _normalized(MIRI)
    assert out['SuperBiasStep'] == 'skip' and out['DarkCurrentStep'] == 'skip'
    assert not messages
    assert config.fixed_options(MIRI)['SuperBiasStep'] == 'skip'
    # SOSS still runs its own steps by default.
    opts = config.fixed_options(SOSS)
    assert opts['INLCorrStep'] == 'run' and opts['RefPixStep'] == 'run'


@pytest.mark.parametrize('base', [NIRSPEC, MIRI])
@pytest.mark.parametrize('key, value', [
    ('soss_background_file', 'model_background256.npy'),
    ('soss_timeseries', 'o1.npy'), ('soss_timeseries_o2', 'o2.npy'),
    ('f277w', 'F277W.npy'), ('inl_amplitude_file', 'amps.npy'),
    ('inl_periods', [1.0, 2.0]), ('soss_specprofile', 'profile.fits')])
def test_soss_only_inputs_are_ignored_off_soss(base, key, value):
    """Check SOSS only inputs are ignored off SOSS."""
    cfg = dict(base, **{key: value})
    with pytest.warns(W, match=f'{key}=.* is ignored for'):
        config.validate_supported_config(cfg)
    assert config.fixed_options(cfg).get(key) is None


def test_soss_inputs_are_kept_for_soss():
    """Check SOSS inputs are kept for SOSS."""
    cfg = dict(SOSS, f277w='F277W.npy', inl_periods=[1.0])
    opts = config.fixed_options(cfg)
    assert opts['f277w'] == 'F277W.npy' and opts['inl_periods'] == [1.0]


@pytest.mark.parametrize('base', [NIRSPEC, MIRI])
def test_generate_order0_mask_is_a_dead_v1_key(base):
    """Check generate order0 mask is a dead v1 key."""
    cfg = dict(base, generate_order0_mask=True)
    with pytest.warns(W, match='generate_order0_mask=True is ignored'):
        config.validate_supported_config(cfg)


@pytest.mark.parametrize('base', [SOSS, NIRSPEC, MIRI])
@pytest.mark.parametrize('key', ['SaturationStep', 'TracingStep'])
def test_saturation_and_tracing_switches_are_dead_v1_keys(base, key):
    """Check saturation and tracing switches are dead v1 keys."""
    cfg = dict(base, **{key: 'skip'})
    with pytest.warns(W, match=f"{key}='skip' is ignored: v1 never reads"):
        assert config.validate_supported_config(cfg) is cfg
    out, messages = _normalized(dict(base, **{key: 'run'}))
    assert key not in out and not messages


@pytest.mark.parametrize('detector', ['CLEAR', 'NRS1', 'F277W'])
def test_miri_filter_detector_is_ignored_like_v1(detector):
    """Check MIRI filter detector is ignored like v1."""
    cfg = dict(MIRI, filter_detector=detector)
    with pytest.warns(W, match='filter_detector=.* is ignored for MIRI'):
        config.validate_supported_config(cfg)
    assert config.fixed_options(cfg)['filter_detector'] == ''


# ---------------------------------------------------------------- item 3.

@pytest.mark.parametrize('method', ['custom', 'custom-rescale'])
def test_niriss_custom_superbias_is_forced_to_crds(method):
    """Check NIRISS custom superbias is forced to CRDS."""
    cfg = dict(SOSS, superbias_method=method, v2_stream_stage1=True)
    with pytest.warns(W, match='changes the method to crds'):
        assert config.validate_supported_config(cfg) is cfg
    assert config.fixed_options(cfg)['superbias_method'] == 'crds'
    assert config.supports_stage1_streaming(config.coerce_config(cfg))
    # NIRSpec keeps its custom methods.
    assert config.fixed_options(dict(NIRSPEC, superbias_method=method))[
        'superbias_method'] == method


# ---------------------------------------------------------------- item 4.

def test_miri_generate_lc_is_accepted():
    """Check MIRI generate light curve is accepted."""
    cfg = dict(MIRI, generate_lc=True)
    assert config.validate_supported_config(cfg) is cfg
    assert config.fixed_options(cfg)['generate_lc'] is True


# ---------------------------------------------------------------- item 5.

@pytest.mark.parametrize('flag', [{'optimize_extract_width': False}, {}])
def test_unoptimized_extract_width_list_uses_v1_middle_then_first(flag):
    """Check unoptimized extract width list uses v1 middle then first."""
    cfg = dict(SOSS, extract_width=[24, 28, 32, 36], **flag)
    config.validate_supported_config(cfg)
    assert config.phase1_extract_width(cfg) == 32
    with pytest.warns(W, match='final extraction uses 24'):
        plan, initial = config.build_sweep_plan(cfg, 'NIRISS/SOSS')
    assert initial['extract_width'] == 24
    assert all(checkpoint != 'Extract' for checkpoint, _ in plan)
    with pytest.raises(ValueError, match='single value'):
        config.validate_optimizer_values(
            dict(cfg, optimize_extract_width=False, extract_width=[]))


# ---------------------------------------------------------------- templates.

@pytest.mark.parametrize('name', ['run_DMS.yaml', 'run_optimize.yaml',
                                  'run_optimize_niriss.yaml'])
@pytest.mark.parametrize('mode, detector', [
    ('NIRSpec/G395H', 'NRS1'), ('NIRSpec/PRISM', 'NRS2'),
    ('MIRI/LRS', 'CLEAR')])
def test_soss_templates_are_accepted_for_every_mode(name, mode, detector):
    """Check SOSS templates are accepted for every mode."""
    directory = 'tests/v2/data' if name.startswith('run_optimize_') else 'exotedrf'
    cfg = config.load_config(REPO / directory / name)
    cfg.update(observing_mode=mode, filter_detector=detector)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', W)
        assert config.validate_supported_config(cfg) is cfg
        opts = config.fixed_options(cfg)
        config.build_sweep_plan(cfg, mode)
    assert opts['oof_method'] == ('median' if mode.startswith('NIRSpec')
                                  else cfg['oof_method'])


@pytest.mark.parametrize('name', ['run_optimize.yaml',
                                  'run_optimize_niriss.yaml',
                                  'run_optimize_nirspec.yaml',
                                  'run_optimize_miri.yaml'])
def test_shipped_mode_templates_emit_no_compat_warnings(name):
    """Check shipped mode templates emit no compat warnings."""
    directory = 'tests/v2/data' if name.startswith('run_optimize_') else 'exotedrf'
    cfg = config.load_config(REPO / directory / name)
    out, messages = _normalized(cfg)
    assert messages == []
    assert config.normalize_v1_config(out) == out


# ---------------------------------------------------------------- item 7.

def _opts(cfg):
    """Return opts."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', W)
        config.validate_supported_config(cfg)
        return config.fixed_options(cfg)


def test_niriss_yaml_background_coords_example_is_supported():
    # Run_optimize_niriss.yaml:90 documents exactly this form.
    """Check NIRISS YAML background coords example is supported."""
    cfg = config.load_config(REPO / 'tests' / 'v2' / 'data' / 'run_optimize_niriss.yaml')
    cfg['stage1_kwargs'] = {
        'BackgroundStep': {'background_coords1': [190, 210, 350, 550]}}
    opts = _opts(cfg)
    assert opts['soss_background_grp'] == {
        'background_coords1': (190, 210, 350, 550)}
    assert 'soss_background_int' not in opts


def test_soss_background_keywords_translate_per_level():
    """Check SOSS background keywords translate per level."""
    cfg = dict(SOSS, stage1_kwargs={'BackgroundStep': {
        'scale1': [1.5, 2.5], 'differential': True,
        'background_coords2': [235.9, 250, 715, 750]}},
        stage2_kwargs={'BackgroundStep': {'scale1': 3.0, 'scale2': [4.0]}})
    opts = _opts(cfg)
    assert opts['soss_background_grp'] == {
        'scale1': (1.5, 2.5), 'differential': True,
        'background_coords2': (235, 250, 715, 750)}
    assert opts['soss_background_int'] == {'scale1': (3.0,), 'scale2': (4.0,)}
    with pytest.raises(ValueError, match='one value at the integration'):
        _opts(dict(SOSS, stage2_kwargs={
            'BackgroundStep': {'scale1': [1., 2.]}}))
    with pytest.raises(ValueError, match='asserts four values'):
        _opts(dict(SOSS, stage1_kwargs={
            'BackgroundStep': {'background_coords1': [1, 2, 3]}}))
    # MIRI's backgroundstep_miri accepts no such keyword: v1 TypeError.
    with pytest.raises(ValueError, match='backgroundstep_miri'):
        _opts(dict(MIRI, stage2_kwargs={
            'BackgroundStep': {'background_coords1': [1, 2, 3, 4]}}))
    # The group-level background only runs inside the SOSS 1/f branch.
    out, messages = _normalized(dict(SOSS, OneOverFStep_grp='skip',
                                     stage1_kwargs={'BackgroundStep': {
                                         'scale1': 1.0}}))
    assert out[config.STAGE_KWARGS_OPTIONS_KEY] == {}
    assert any('v1 does not run BackgroundStep' in m for m in messages)


def test_detector_and_pca_keywords_translate():
    """Check detector and PCA keywords translate."""
    opts = _opts(dict(MIRI, stage1_kwargs={
        'flag_neighbours': 2, 'miri_subtract_dark': False,
        'DQInitStep': {'flag_first_miri_frame': False,
                       'flag_last_miri_frame': True},
        'RampFitStep': {'suppress_one_group': False}},
        stage2_kwargs={'PCAReconstructStep': {'skip_pca': True}}))
    assert opts['flag_neighbours'] == 2
    assert opts['miri_subtract_dark'] is False
    assert opts['flag_first_miri_frame'] is False
    assert opts['flag_last_miri_frame'] is True
    assert opts['suppress_one_group'] is False
    assert opts['skip_pca'] is True
    # Negative half-width: v1's slice is empty, i.e. no dilation.
    with pytest.warns(W, match='empty box'):
        assert config.fixed_options(dict(SOSS, stage1_kwargs={
            'flag_neighbours': -1}))['flag_neighbours'] == 0
    # MIRI-only DQInit flags are ignored for SOSS, as in v1.
    with pytest.warns(W, match='applies it to MIRI only'):
        config.fixed_options(dict(SOSS, stage1_kwargs={
            'DQInitStep': {'flag_first_miri_frame': False}}))


def test_nirspec_superbias_keywords_need_a_custom_method():
    """Check NIRSpec superbias keywords need a custom method."""
    cfg = dict(NIRSPEC, superbias_method='custom-rescale', stage1_kwargs={
        'SuperBiasStep': {'mask_width': 14, 'override_centroids': True}})
    opts = _opts(cfg)
    assert opts['superbias_mask_width'] == 14.
    assert opts['superbias_override_centroids'] is True
    with pytest.raises(NotImplementedError, match='jwst pass-through'):
        _opts(dict(cfg, superbias_method='crds'))
    # NIRISS is forced to crds, so the keyword reaches the jwst step in v1.
    with pytest.raises(NotImplementedError, match='jwst pass-through'):
        _opts(dict(SOSS, superbias_method='custom', stage1_kwargs={
            'SuperBiasStep': {'mask_width': 14}}))


def test_badpix_kwargs_set_the_fixed_structural_value():
    """Check bad-pixel correction keywords set the fixed structural value."""
    cfg = dict(SOSS, box_size=7, window_size=5,
               stage2_kwargs={'BadPixStep': {'box_size': 3}})
    with pytest.warns(W, match='top-level box_size=7 is ignored'):
        plan, initial = config.build_sweep_plan(
            config.coerce_config(cfg), 'NIRISS/SOSS')
    assert initial['box_size'] == 3 and initial['window_size'] == 5
    swept = dict(cfg, box_size=[3, 5], optimize_box_size=True)
    with pytest.warns(W, match='superseded by the box_size sweep'):
        out = config.coerce_config(swept)
    assert out['box_size'] == [3, 5]


def test_even_odd_rows_targets_the_soss_scale_methods():
    """Check even odd rows targets the SOSS scale methods."""
    cfg = dict(SOSS, stage1_kwargs={'OneOverFStep': {'even_odd_rows': False}},
               stage2_kwargs={'OneOverFStep': {'even_odd_rows': True}})
    opts = _opts(cfg)
    assert opts['oof_even_odd_rows_grp'] is False
    assert opts['oof_even_odd_rows_int'] is True
    with pytest.warns(W, match='raises IndexError at the integration'):
        config.fixed_options(dict(SOSS, stage2_kwargs={
            'OneOverFStep': {'even_odd_rows': False}}))
    with pytest.warns(W, match='oneoverfstep_nirspec without'):
        # Ignored for NIRSpec: the option keeps its v1 default.
        assert config.fixed_options(dict(
            NIRSPEC, stage1_kwargs={'OneOverFStep': {'even_odd_rows': 0}})
        ).get('oof_even_odd_rows_grp', True) is True


@pytest.mark.parametrize('block, match, error', [
    ({'stage1_kwargs': {'JumpStep': {'time_window': 7}}},
     "JumpStep'\\]\\['time_window'\\] is not translated", NotImplementedError),
    ({'stage1_kwargs': {'LinearityStep': {'skip': True}}},
     'jwst pass-through', NotImplementedError),
    ({'stage1_kwargs': {'GainScaleStep': {'foo': 1}}},
     'jwst pass-through', NotImplementedError),
    ({'stage1_kwargs': {'DQInitStep': {'saturation_threshold': 60}}},
     'passed twice', ValueError),
    ({'stage1_kwargs': {'saturation_threshold': 60}},
     'passed twice', ValueError),
    ({'stage2_kwargs': {'BadPixStep': {'foo': 1}}},
     'does not accept it', ValueError),
    ({'stage1_kwargs': {'mystery': 1}},
     "stage1_kwargs\\['mystery'\\] is not a v1", NotImplementedError),
    ({'stage1_kwargs': {'BadPixStep': {'box_size': 3}}},
     'move it to stage2_kwargs', NotImplementedError),
    ({'stage2_kwargs': {'root_dir': '/tmp'}},
     'pipeline_outputs_directory', NotImplementedError),
    ({'stage1_kwargs': {'DQInitStep': 3}}, 'must be a mapping', ValueError),
])
def test_unsupported_stage_kwargs_raise_naming_the_keyword(
        block, match, error):
    """Check unsupported stage keywords raise naming the keyword."""
    cfg = dict(SOSS, **block)
    with pytest.raises(error, match=match):
        config.validate_supported_config(cfg)
    with pytest.raises(error, match=match):
        config.fixed_options(cfg)


def test_display_only_and_skipped_step_kwargs_warn():
    """Check display only and skipped step keywords warn."""
    cfg = dict(SOSS, JumpStep='skip', stage1_kwargs={
        'show_plot': True, 'RampFitStep': {'maximum_cores': 'all'},
        'INLCorrStep': {'npix_to_bin': 50},
        'JumpStep': {'time_window': 3}})
    out, messages = _normalized(cfg)
    assert out[config.STAGE_KWARGS_OPTIONS_KEY] == {}
    assert out['stage1_kwargs'] == {}
    joined = '\n'.join(messages)
    for key in ('show_plot', 'maximum_cores', 'npix_to_bin', 'time_window'):
        assert key in joined
    # NIRSpec/MIRI: a step that v1 never runs for the mode is ignored.
    out, messages = _normalized(dict(NIRSPEC, stage1_kwargs={
        'RefPixStep': {'odd_even_columns': False}}))
    assert any('v1 does not run RefPixStep' in m for m in messages)


def test_registry_hook_replaces_a_placeholder():
    """Check registry hook replaces a placeholder."""
    original = stage_kwargs.registered(1, 'JumpStep', '*')
    try:
        stage_kwargs.register(
            1, 'JumpStep', '*', lambda value, ctx: stage_kwargs.Translation(
                options={f'jump_{ctx.name}': value}))
        assert config.fixed_options(dict(SOSS, stage1_kwargs={
            'JumpStep': {'time_window': 9}}))['jump_time_window'] == 9
    finally:
        stage_kwargs.register(1, 'JumpStep', '*', original)
    # The upramp_jump translator is restored after the temporary hook.
    assert stage_kwargs.registered(1, 'JumpStep', '*') is original


def test_unported_stage_kwargs_drop_in_raises_specific_errors():
    """Check unported stage keywords drop in raises specific errors."""
    cfg = dict(SOSS, stage2_kwargs={'PCAReconstructStep': {'skip_pca': True}})
    assert stage_kwargs.unported_stage_kwargs(cfg, 'stage2_kwargs') == {}
    assert stage_kwargs.unported_stage_kwargs(cfg, 'stage1_kwargs') is None
    with pytest.raises(NotImplementedError, match='jwst pass-through'):
        stage_kwargs.unported_stage_kwargs(
            dict(SOSS, stage1_kwargs={'RefPixStep': {'x': 1}}),
            'stage1_kwargs')


def test_normalization_does_not_mutate_and_default_opts_unchanged():
    """Check normalization does not mutate and default opts unchanged."""
    cfg = dict(NIRSPEC, oof_method='scale-achromatic', INLCorrStep='run',
               stage1_kwargs={'flag_neighbours': 2})
    frozen = {key: (dict(value) if isinstance(value, dict) else value)
              for key, value in cfg.items()}
    _normalized(cfg)
    assert cfg == frozen
    # No compat keys appear in the options of a plain configuration.
    opts = config.fixed_options(SOSS)
    for key in ('flag_neighbours', 'soss_background_grp', 'skip_pca',
                'superbias_mask_width'):
        assert key not in opts
    # The OneOverFStep translator always records v1's defaults.
    assert opts.get('oof_even_odd_rows_grp', True) is True
    assert np.isclose(opts['w2'], 1.0)
