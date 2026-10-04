"""Translate v1 stage keyword arguments into v2 reduction options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


import numpy as np


# Use v1 step spellings and execution order for stage keywords.
STAGE_STEPS = {1: ('DQInitStep', 'INLCorrStep', 'EmiCorrStep',
        'SuperBiasStep', 'RefPixStep', 'DarkCurrentStep', 'BackgroundStep',
        'OneOverFStep', 'LinearityStep', 'JumpStep', 'RampFitStep', 'GainScaleStep'),
    2: ('AssignWCSStep', 'FlatFieldStep', 'BackgroundStep', 'OneOverFStep', 'BadPixStep',
        'PCAReconstructStep'), 3: ('SpeProfileStep', 'Extract1dStep')}

# Accept removed v1 steps with a warning.
REMOVED_V1_STEPS = {1: ('ResetStep',), 2: ('Extract2DStep', 'SourceTypeStep', 'WaveCorrStep')}

# Identify steps that v1 skips for each instrument.
V1_NOOP_STEPS = {'NIRISS': {'EmiCorrStep': 'stage1.py:3216-3227'}, 'NIRSPEC': {
        'INLCorrStep': 'stage1.py:3201-3214', 'EmiCorrStep': 'stage1.py:3216-3227',
        'RefPixStep': 'stage1.py:3257-3268', 'FlatFieldStep': 'stage2.py:1823-1834',
        'BackgroundStep': 'stage2.py:1836-1851'}, 'MIRI': {'INLCorrStep': 'stage1.py:3201-3214',
        'SuperBiasStep': 'stage1.py:3242-3255', 'RefPixStep': 'stage1.py:3257-3268',
        'DarkCurrentStep': 'stage1.py:3270-3279', 'OneOverFStep_grp': 'stage1.py:3281-3314',
        'OneOverFStep_int': 'stage2.py:1853-1871'}}

# Reject stage keywords already passed explicitly by the optimizer.
_STAGE_EXPLICIT = {1: {'results', 'mode', 'soss_background_model', 'baseline_ints',
        'oof_method', 'superbias_method', 'soss_timeseries',
        'soss_timeseries_o2', 'save_results', 'pixel_masks', 'force_redo',
        'flag_up_ramp', 'rejection_threshold', 'flag_in_time',
        'time_rejection_threshold', 'output_tag', 'skip_steps', 'do_plot',
        'soss_inner_mask_width', 'soss_outer_mask_width',
        'nirspec_mask_width', 'centroids', 'hot_pixel_map',
        'miri_drop_groups', 'saturation_threshold', 'f277w',
        'inl_amplitude_file', 'inl_periods', 'pipeline_outputs_directory'},
    2: {'results', 'mode', 'soss_background_model', 'baseline_ints',
        'save_results', 'force_redo', 'space_thresh', 'time_thresh',
        'remove_components', 'pca_components', 'soss_timeseries',
        'soss_timeseries_o2', 'oof_method', 'output_tag', 'skip_steps',
        'generate_lc', 'soss_inner_mask_width', 'soss_outer_mask_width',
        'nirspec_mask_width', 'pixel_masks', 'f277w', 'do_plot', 'centroids',
        'miri_trace_width', 'miri_background_width',
        'miri_background_method', 'pipeline_outputs_directory'},
    3: {'results', 'save_results', 'force_redo', 'extract_method',
        'soss_specprofile', 'centroids', 'extract_width',
        'extract_width_soss2', 'st_teff', 'st_logg', 'st_met',
        'planet_letter', 'output_tag', 'do_plot', 'deepframe',
        'saturation_rescue', 'mask_do_not_use_pixels', 'mask_saturated_pixels',
        'pipeline_outputs_directory'}}

# Reject step keywords already passed explicitly by the stage runner.
_STEP_EXPLICIT = {key: {'save_results', 'force_redo'} for key in (
    (1, 'EmiCorrStep'), (1, 'ResetStep'), (1, 'RefPixStep'),
    (1, 'DarkCurrentStep'), (1, 'RampFitStep'),
    (1, 'GainScaleStep'), (2, 'AssignWCSStep'), (2, 'Extract2DStep'), (2, 'SourceTypeStep'),
    (2, 'WaveCorrStep'), (2, 'FlatFieldStep'))}
_STEP_EXPLICIT.update({key: {'save_results', 'force_redo', *extra} for key, extra in {
    (1, 'DQInitStep'): ('do_plot', 'flag_neighbours', 'saturation_threshold', 'show_plot'),
    (1, 'INLCorrStep'): ('amplitude_file', 'do_plot', 'periods', 'show_plot'),
    (1, 'SuperBiasStep'): ('do_plot', 'show_plot'),
    (1, 'BackgroundStep'): ('background_model', 'datafile', 'deepstack', 'do_plot',
                           'fileroot', 'fileroot_noseg', 'output_dir', 'show_plot'),
    (1, 'OneOverFStep'): ('do_plot', 'nirspec_mask_width', 'show_plot',
                         'soss_inner_mask_width', 'soss_outer_mask_width'),
    (1, 'LinearityStep'): ('do_plot', 'miri_drop_groups', 'miri_subtract_dark', 'show_plot'),
    (2, 'BackgroundStep'): ('background_model', 'datafile', 'deepstack', 'do_plot',
                           'fileroot', 'fileroot_noseg', 'miri_background_width',
                           'miri_trace_width', 'output_dir', 'show_plot'),
    (2, 'OneOverFStep'): ('do_plot', 'nirspec_mask_width', 'show_plot',
                         'soss_inner_mask_width', 'soss_outer_mask_width'),
    (2, 'BadPixStep'): ('do_plot', 'show_plot', 'space_thresh', 'time_thresh'),
    (2, 'PCAReconstructStep'): ('do_plot', 'pca_components', 'remove_components', 'show_plot'),
}.items()})

# Identify steps that forward extra keywords to JWST.
_JWST_STEPS = {(1, 'DQInitStep'): 'calwebb_detector1.dq_init_step.DQInitStep '
                       '(stage1.py:125-127)',
    (1, 'EmiCorrStep'): 'calwebb_detector1.emicorr_step.EmiCorrStep (stage1.py:328-330)',
    (1, 'ResetStep'): 'calwebb_detector1.reset_step.ResetStep (stage1.py:411-413)',
    (1, 'RefPixStep'): 'calwebb_detector1.refpix_step.RefPixStep (stage1.py:656-658)',
    (1, 'DarkCurrentStep'): 'calwebb_detector1.dark_current_step.'
                            'DarkCurrentStep (stage1.py:739-741)',
    (1, 'LinearityStep'): 'calwebb_detector1.linearity_step.LinearityStep (stage1.py:1286-1288)',
    (1, 'RampFitStep'): 'calwebb_detector1.ramp_fit_step.RampFitStep (stage1.py:1539-1541)',
    (1, 'GainScaleStep'): 'calwebb_detector1.gain_scale_step.GainScaleStep '
                          '(stage1.py:1659-1661)',
    (2, 'AssignWCSStep'): 'calwebb_spec2.assign_wcs_step.AssignWcsStep (stage2.py:98-100)',
    (2, 'Extract2DStep'): 'calwebb_spec2.extract_2d_step.Extract2dStep (stage2.py:177-179)',
    (2, 'SourceTypeStep'): 'calwebb_spec2.srctype_step.SourceTypeStep (stage2.py:247-249)',
    (2, 'WaveCorrStep'): 'calwebb_spec2.wavecorr_step.WavecorrStep (stage2.py:326-328)',
    (2, 'FlatFieldStep'): 'calwebb_spec2.flat_field_step.FlatFieldStep (stage2.py:554-556)'}


class Translation:
    """Store translated configuration updates, options and warnings.

    Parameters
    ----------
    config, options : None, dict
        Top-level configuration and fixed reduction option updates.
    warnings : None, list[str]
        Messages to emit when applying the translation.
    """

    def __init__(self, config=None, options=None, warnings=None):
        """Initialize configuration updates, options and warnings."""
        self.config = dict(config or {})
        self.options = dict(options or {})
        self.warnings = list(warnings or [])

    def merge(self, other):
        """Merge translated updates, combining dictionary-valued options.

        Parameters
        ----------
        other : Translation
            Keyword translation to merge into this result.

        Returns
        -------
        translation : Translation
            This translation after merging the supplied updates.
        """
        self.config.update(other.config)
        for key, value in other.options.items():
            if isinstance(value, dict) and isinstance(self.options.get(key), dict):
                self.options[key] = dict(self.options[key], **value)
            else:
                self.options[key] = value
        self.warnings.extend(other.warnings)
        return self

    def __repr__(self):
        """Return the stored representation."""
        return (f'Translation(config={self.config!r}, '
                f'options={self.options!r}, warnings={self.warnings!r})')


@dataclass(frozen=True)
class KwargContext:
    """Describe a stage keyword and its reduction configuration.

    Parameters
    ----------
    stage : int
        Pipeline stage, from 1 to 3.
    step : None, str
        Step name, or None for a stage-level keyword.
    name, instrument, mode : str
        Keyword name, instrument and observing mode.
    cfg : dict
        Reduction configuration.
    """

    stage: int
    step: str | None
    name: str
    instrument: str
    mode: str
    cfg: Mapping = field(repr=False)

    @property
    def label(self):
        """Get the YAML path of the keyword."""
        if self.step is None:
            return f"stage{self.stage}_kwargs[{self.name!r}]"
        return f"stage{self.stage}_kwargs[{self.step!r}][{self.name!r}]"


_REGISTRY = {}


def register(stage, step, name, handler):
    """Register a translator for a stage keyword.

    A name of * handles keywords without a specific entry.

    Parameters
    ----------
    stage : int
        Pipeline stage, from 1 to 3.
    step : None, str
        Step name, or None for a stage-level keyword.
    name : str
        Keyword name; * selects the step fallback handler.
    handler : callable
        Keyword translator with signature handler(value, ctx).

    Returns
    -------
    handler : callable
        The registered translator.
    """
    if stage not in STAGE_STEPS:
        raise ValueError(f'stage must be 1, 2 or 3, got {stage!r}')
    if step is not None and step not in STAGE_STEPS[stage]:
        raise ValueError(f'{step!r} is not a v1 stage-{stage} Step')
    _REGISTRY[(stage, step, name)] = handler
    return handler


def registered(stage, step, name):
    """Get the translator registered for an exact keyword.

    Parameters
    ----------
    stage : int
        Pipeline stage, from 1 to 3.
    step : None, str
        Step name, or None for a stage-level keyword.
    name : str
        Keyword name; * selects the step fallback handler.

    Returns
    -------
    handler : None, callable
        Registered translator, if any.
    """
    return _REGISTRY.get((stage, step, name))


def placeholder(owner):
    """Create a handler that rejects an unsupported keyword.

    Parameters
    ----------
    owner : str
        Name to include in the unsupported-keyword error.

    Returns
    -------
    handler : callable
        Translator raising NotImplementedError.
    """
    def handler(value, ctx):
        """Translate or reject the keyword using its registered semantics."""
        raise NotImplementedError(f'{ctx.label} is not supported by v2 ({owner}); v2 refuses to '
            'ignore it silently')
    handler.placeholder_owner = owner
    return handler


def _is_true_note(value, ctx, v1_test):
    """Explain v1 identity tests for non-boolean values."""
    if isinstance(value, (bool, np.bool_)):
        return []
    return [f'{ctx.label}={value!r} is not a boolean; v1 tests '
            f'``{v1_test}``, so v2 uses the same result']


def _warn_only(reason):
    """Create a translator that reports an ignored keyword."""
    def handler(value, ctx):
        """Translate or reject the keyword using its registered semantics."""
        return Translation(warnings=[f'{ctx.label} is ignored: {reason}'])
    return handler


def _reject_v1_typeerror(ctx, why):
    """Reject a keyword that also causes a v1 TypeError."""
    raise ValueError(f'{ctx.label} is not valid in v1 either: {why}; v1 raises TypeError '
        'at run time')


def _reject_jwst(ctx):
    """Reject a JWST pass-through option without a v2 equivalent."""
    target = _JWST_STEPS.get((ctx.stage, ctx.step), 'a jwst Step')
    raise NotImplementedError(f'{ctx.label} is a jwst pass-through option: v1 forwards it to '
        f'{target}, which v2 does not run.  It has no faithful v2 '
        'equivalent, so v2 refuses it instead of silently ignoring it')


def _finite_number(value, ctx, where):
    """Validate a finite numeric keyword value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
        raise ValueError(f'{ctx.label} must be a finite number ({where}), got {value!r}')
    return value


def _flag_neighbours(value, ctx):
    """Translate the saturated-pixel flagging half-width."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{ctx.label} must be an integer pixel half-width (v1 slices '
            f'with it, stage1.py:1817-1825), got {value!r}')
    value = int(value)
    out = Translation(options={'flag_neighbours': max(value, 0)})
    if value < 0:
        out.warnings.append(f'{ctx.label}={value}: v1 expands an empty box for a negative '
            'half-width (stage1.py:1819-1824); v2 uses 0')
    return out


def _miri_subtract_dark(value, ctx):
    """Translate the MIRI dark-subtraction flag."""
    if ctx.instrument != 'MIRI':
        return Translation(warnings=[f'{ctx.label} is ignored: the MIRI dark inside LinearityStep '
            f'only runs for MIRI (stage1.py:1282-1299), not {ctx.mode}'])
    return Translation(options={'miri_subtract_dark': value is True},
                       warnings=_is_true_note(value, ctx, 'miri_subtract_dark is True'))


def _miri_frame_flag(option, line):
    """Create a translator for a MIRI first- or last-frame flag."""
    def handler(value, ctx):
        """Translate or reject the keyword using its registered semantics."""
        if ctx.instrument != 'MIRI':
            return Translation(warnings=[f'{ctx.label} is ignored: v1 applies it to MIRI only '
                f'(stage1.py:129-141), not {ctx.mode}'])
        return Translation(options={option: value is True},
            warnings=_is_true_note(value, ctx, f'{option} is True (stage1.py:{line})'))
    return handler


def _superbias_method(ctx):
    """Get the effective superbias method."""
    return str(ctx.cfg.get('superbias_method', 'crds')).lower()


def _superbias_keyword(option, convert):
    """Create a translator for a custom superbias keyword."""
    def handler(value, ctx):
        """Translate or reject the keyword using its registered semantics."""
        method = _superbias_method(ctx)
        if method == 'crds':
            _reject_jwst(ctx)
        out = Translation(options={option: convert(value, ctx)})
        if method == 'custom':
            out.warnings.append(f'{ctx.label} has no effect with superbias_method=custom: v1 '
                'accepts it but only custom-rescale masks the trace ' '(stage1.py:2961-3011)')
        return out
    return handler


def _mask_width(value, ctx):
    """Validate the full trace-mask width."""
    return float(_finite_number(value, ctx, 'trace-mask full width, stage1.py:2999-3000'))


def _override_centroids(value, ctx):
    """Preserve the v1 centroid override identity test."""
    return value is not False


def _superbias_fallback(value, ctx):
    """Reject an unsupported superbias keyword."""
    if _superbias_method(ctx) == 'crds':
        _reject_jwst(ctx)
    _reject_v1_typeerror(ctx, 'subtract_custom_superbias does not accept it '
             '(stage1.py:2898-2900)')


def _soss_background_keyword(name):
    """Create a translator for a SOSS background keyword."""
    def handler(value, ctx):
        """Translate or reject the keyword using its registered semantics."""
        if ctx.instrument != 'NIRISS':
            _reject_v1_typeerror(ctx, 'backgroundstep_miri does not accept it '
                     '(stage2.py:477-480, 901-902)')
        key = ('soss_background_grp' if ctx.stage == 1 else 'soss_background_int')
        out = Translation()
        if name in ('background_coords1', 'background_coords2'):
            if value is None:
                spec = None
            else:
                coords = np.atleast_1d(np.asarray(value))
                if coords.ndim != 1 or coords.size != 4 or not np.all(
                        np.isfinite(coords.astype(float))):
                    raise ValueError(f'{ctx.label} must be [row_low, row_high, col_low, '
                        'col_high] (v1 asserts four values, '
                        f'stage2.py:1080-1086), got {value!r}')
                # Convert background coordinates to integer row and column limits.
                spec = tuple(int(v) for v in coords.astype(int))
        elif name in ('scale1', 'scale2'):
            if value is None:
                spec = None
            else:
                scale = np.atleast_1d(np.asarray(value, dtype=float))
                if scale.ndim != 1 or not np.all(np.isfinite(scale)):
                    raise ValueError(f'{ctx.label} must be a finite number or one number '
                        f'per group (stage2.py:1059-1064), got {value!r}')
                if ctx.stage == 2 and scale.size != 1:
                    raise ValueError(f'{ctx.label} must hold one value at the '
                        'integration level: v1 asserts len(scale) == ngroup '
                        '== 1 (stage2.py:1059-1064)')
                spec = tuple(float(v) for v in scale)
        else:
            spec = value is True
            out.warnings.extend(_is_true_note(
                value, ctx, 'differential is True (stage2.py:1112, 1142)'))
        out.options[key] = {name: spec}
        return out
    return handler


def _background_fallback(value, ctx):
    """Reject an unsupported background keyword."""
    if ctx.instrument != 'NIRISS':
        _reject_v1_typeerror(ctx, 'backgroundstep_miri does not accept it (stage2.py:901-902)')
    _reject_v1_typeerror(ctx, 'backgroundstep_soss does not accept it (stage2.py:988-990)')


def _ignored_oneoverf(ctx):
    """Translate ignored 1/f options for non-SOSS and solve recipes."""
    if ctx.instrument != 'NIRISS':
        return Translation(warnings=[
            f'{ctx.label} is ignored: v1 calls oneoverfstep_nirspec without '
            'the OneOverFStep keywords (stage1.py:1069-1074)'])
    method = str(ctx.cfg.get('oof_method', 'scale-achromatic')).lower()
    if method == 'solve':
        return Translation(warnings=[f'{ctx.label} is ignored: v1 calls oneoverfstep_solve without '
            'the OneOverFStep keywords (stage1.py:1051-1057)'])
    return None


def _even_odd_rows(value, ctx):
    """Translate separate even and odd row 1/f levels."""
    ignored = _ignored_oneoverf(ctx)
    if ignored is not None:
        return ignored
    key = 'oof_even_odd_rows_grp' if ctx.stage == 1 else 'oof_even_odd_rows_int'
    effective = value is True
    out = Translation(options={key: effective}, warnings=_is_true_note(
                          value, ctx, 'even_odd_rows is True (stage1.py:2592)'))
    if ctx.stage == 2 and not effective:
        out.warnings.append(f'{ctx.label}=False: v1 raises IndexError at the integration '
            'level (the ``np.ndim(cube == 4)`` typo, stage1.py:2607); v2 '
            'applies the intended single 1/f level per column')
    return out


def _smoothing_scale(value, ctx):
    """Translate the SOSS 1/f light-curve smoothing scale."""
    ignored = _ignored_oneoverf(ctx)
    if ignored is not None:
        return ignored
    from exotedrf.v2.config import oneoverf_kwargs_from_stage_kwargs
    container = f'stage{ctx.stage}_kwargs'
    level = 'grp' if ctx.stage == 1 else 'int'
    opts = oneoverf_kwargs_from_stage_kwargs(
        {container: {'OneOverFStep': {'smoothing_scale': value}}})
    key = f'oof_smoothing_scale_{level}'
    return Translation(options={key: opts[key]})


def _suppress_one_group(value, ctx):
    """Translate the STCAL one-group suppression flag."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{ctx.label} must be a boolean (jwst RampFitStep '
                         f'spec), got {value!r}')
    return Translation(options={'suppress_one_group': bool(value)})


def _badpix_structural(param):
    """Create a translator for a BadPix box or window size."""
    def handler(value, ctx):
        """Translate or reject the keyword using its registered semantics."""
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)) or int(value) < 1:
            raise ValueError(f'{ctx.label} must be a positive integer, got {value!r}')
        if ctx.cfg.get(f'optimize_{param}', False):
            return Translation(warnings=[f'{ctx.label}={value} is superseded by the {param} sweep: '
                'v1 still used it outside that sweep, but v2 starts the '
                f'sweep from int(mean({param})) and propagates its winner '
                '(documented v2 correction)'])
        out = Translation(config={param: int(value)})
        top = ctx.cfg.get(param)
        if top is not None and not isinstance(top, list) and top != value:
            out.warnings.append(f'top-level {param}={top!r} is ignored in favour of '
                f'{ctx.label}={value!r}, as in v1 (optimize.py never passes '
                f'the top-level {param} outside a sweep)')
        return out
    return handler


def _strict_true_option(name):
    """Create a translator preserving the v1 True identity test."""
    def handler(value, ctx):
        """Translate or reject the keyword using its registered semantics."""
        return Translation(options={name: value is True})
    return handler


def _skip_pca(value, ctx):
    """Translate the v1 PCA skip identity test."""
    return Translation(options={'skip_pca': value is not False})


def _use_pastasoss(value, ctx):
    """Translate the instrument-specific PASTASOSS wavelength option."""
    from exotedrf.v2 import wavecal
    enabled = wavecal.use_pastasoss_option(
        {'stage3_kwargs': {'Extract1dStep': {'use_pastasoss': value}}})
    if ctx.instrument != 'NIRISS':
        return Translation(warnings=[
            f'{ctx.label} is ignored: PASTASOSS applies to NIRISS/SOSS only '
            f'and v1 ignores it for {ctx.mode} (stage3.py:350-357)'])
    return Translation(options={'use_pastasoss': enabled})


def _clip_thresh(value, ctx):
    """Translate the final light-curve clipping threshold."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)) or not value > 0:
        raise ValueError('stage3_kwargs.Extract1dStep.clip_thresh must be a positive number')
    return Translation(options={'clip_thresh': float(value)})


def _allow_miri_slope(value, ctx):
    """Translate the tilted MIRI Stage-3 trace flag."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('stage3_kwargs.Extract1dStep.allow_miri_slope must be a boolean')
    return Translation(options={'stage3_allow_miri_slope': bool(value)})


def _jumpstep(value, ctx):
    """Translate the complete JWST up-the-ramp jump keyword mapping."""
    from exotedrf.v2.config import jumpstep_kwargs_options
    block = (ctx.cfg.get('stage1_kwargs') or {}).get('JumpStep') or {}
    return Translation(options={'upramp_jump_kwargs': jumpstep_kwargs_options(block, ctx.cfg)})


def _custom_step_fallback(signature):
    """Create a translator rejecting an unsupported custom-step keyword."""
    def handler(value, ctx):
        """Translate or reject the keyword using its registered semantics."""
        _reject_v1_typeerror(ctx, f'{ctx.step}.run does not accept it ({signature})')
    return handler


def _root_dir(value, ctx):
    """Reject a stage-specific output relocation."""
    raise NotImplementedError(f'{ctx.label} relocates the v1 Stage{ctx.stage} directory only; v2 '
        'writes every product under pipeline_outputs_directory, so set that ' 'key instead')


def _install_defaults():
    """Register the supported stage keywords and rejection handlers."""
    show_plot = _warn_only('it only opens interactive matplotlib windows in '
                           'v1; v2 writes the same diagnostic PNG files')
    for stage in (1, 2, 3):
        register(stage, None, 'root_dir', _root_dir)
        register(stage, None, 'show_plot', show_plot)
    # Register Stage-1 arguments omitted by the optimizer.
    register(1, None, 'flag_neighbours', _flag_neighbours)
    register(1, None, 'miri_subtract_dark', _miri_subtract_dark)

    # Register the MIRI first- and last-frame flags.
    register(1, 'DQInitStep', 'flag_first_miri_frame',
             _miri_frame_flag('flag_first_miri_frame', 132))
    register(1, 'DQInitStep', 'flag_last_miri_frame', _miri_frame_flag('flag_last_miri_frame', 137))
    # Reject unsupported INL keywords.
    register(1, 'INLCorrStep', 'npix_to_bin', _warn_only(
        'it only bins the inlcorrstep diagnostic plot (stage1.py:251-255)'))
    register(1, 'INLCorrStep', '*', _custom_step_fallback('stage1.py:203-204'))
    # Register custom superbias options.
    register(1, 'SuperBiasStep', 'mask_width',
             _superbias_keyword('superbias_mask_width', _mask_width))
    register(1, 'SuperBiasStep', 'override_centroids',
             _superbias_keyword('superbias_override_centroids', _override_centroids))
    register(1, 'SuperBiasStep', '*', _superbias_fallback)
    # Register SOSS background keywords at both levels.
    for stage in (1, 2):
        for name in ('background_coords1', 'background_coords2', 'scale1',
                     'scale2', 'differential'):
            register(stage, 'BackgroundStep', name, _soss_background_keyword(name))
        register(stage, 'BackgroundStep', '*', _background_fallback)
    # Ignore MIRI widths in the SOSS group background step.
    for name in ('miri_trace_width', 'miri_background_width'):
        register(1, 'BackgroundStep', name, _warn_only(
            'BackgroundStep.run uses it for MIRI only (stage2.py:476-480)'))
    # Register group and integration 1/f options.
    for stage in (1, 2):
        register(stage, 'OneOverFStep', 'even_odd_rows', _even_odd_rows)
        register(stage, 'OneOverFStep', 'smoothing_scale', _smoothing_scale)
        register(stage, 'OneOverFStep', '*', _custom_step_fallback('stage1.py:888-891'))
    register(1, 'JumpStep', '*', _jumpstep)
    register(1, 'RampFitStep', 'maximum_cores', _warn_only(
        'it only sets jwst multiprocessing (stage1.py:1532-1541); v2 ramp '
        'fitting is a vectorized JAX kernel'))
    register(1, 'RampFitStep', 'suppress_one_group', _suppress_one_group)

    # Register BadPix and PCA options.
    for name in ('box_size', 'window_size'):
        register(2, 'BadPixStep', name, _badpix_structural(name))
    for name in ('median_high_variance', 'preserve_saturated', 'clear_interpolated_dq'):
        register(2, 'BadPixStep', name, _strict_true_option(name))
    register(2, 'BadPixStep', '*', _custom_step_fallback('stage2.py:693-694'))
    register(2, 'PCAReconstructStep', 'skip_pca', _skip_pca)
    register(2, 'PCAReconstructStep', '*', _custom_step_fallback('stage2.py:806-807'))

    # Register Stage-3 extraction options.
    register(3, 'Extract1dStep', '*', placeholder('Extract1dStep'))
    register(3, 'Extract1dStep', 'use_pastasoss', _use_pastasoss)
    # Reject the removed soss_estimate run keyword.
    register(3, 'Extract1dStep', 'soss_estimate', _custom_step_fallback(
        'stage3.py:152-155, soss_estimate removed in 2.5.0'))
    register(3, 'Extract1dStep', 'clip_thresh', _clip_thresh)
    register(3, 'SpeProfileStep', '*', placeholder('SpeProfileStep'))
    register(3, 'Extract1dStep', 'allow_miri_slope', _allow_miri_slope)
    for name in ('opt_max_iter', 'opt_var_thresh'):
        # Apply Stage-3 controls before top-level extraction defaults.
        register(3, None, name, lambda value, ctx, name=name: Translation(config={name: value}))


_install_defaults()

# Delegate duplicate-key checks to the step-specific translators.
_OWNED_STEPS = {(1, 'JumpStep'), (3, 'Extract1dStep'), (3, 'SpeProfileStep')}


def _switch_runs(cfg, key):
    """Check whether a step switch differs from skip."""
    return cfg.get(key, 'run') != 'skip'


def step_runs(stage, step, cfg, instrument):
    """Check whether v1 runs a step for this configuration.

    Parameters
    ----------
    stage : int
        Pipeline stage, from 1 to 3.
    step : None, str
        Step name, or None for a stage-level keyword.
    cfg : dict
        Reduction configuration.
    instrument : str
        Instrument name.

    Returns
    -------
    runs : bool
        Whether the step runs for the instrument and selected options.
    """
    if stage == 3:
        if step == 'SpeProfileStep':
            return (str(cfg.get('extract_method', 'box')).lower() == 'atoca'
                    and cfg.get('soss_specprofile') is None)
        return True
    if stage == 1 and step == 'BackgroundStep':
        # Apply the SOSS group background only inside group-level 1/f correction.
        return instrument == 'NIRISS' and _switch_runs(cfg, 'OneOverFStep_grp') and \
            'OneOverFStep_grp' not in V1_NOOP_STEPS.get(instrument, {})
    switch = {(1, 'OneOverFStep'): 'OneOverFStep_grp',
              (2, 'OneOverFStep'): 'OneOverFStep_int'}.get((stage, step), step)
    return _switch_runs(cfg, switch) and switch not in V1_NOOP_STEPS.get(instrument, {})


def _instrument(cfg):
    """Get the instrument name and observing mode."""
    mode = str(cfg.get('observing_mode', 'NIRISS/SOSS')).upper()
    return mode.split('/')[0], mode


def _translate_one(ctx, value):
    """Translate one stage keyword or raise its specific rejection error."""
    handler = _REGISTRY.get((ctx.stage, ctx.step, ctx.name))
    if handler is not None:
        return handler(value, ctx)
    if ctx.step is None:
        if ctx.name in _STAGE_EXPLICIT[ctx.stage]:
            _reject_v1_typeerror(ctx, f'run_stage{ctx.stage} already receives {ctx.name!r} '
                     'from the optimizer (keyword passed twice); set the '
                     'top-level YAML key instead')
        other = [s for s, steps in STAGE_STEPS.items() if s != ctx.stage and ctx.name in steps]
        hint = (f'; {ctx.name} is a stage-{other[0]} Step, so move it to '
                f'stage{other[0]}_kwargs' if other else '')
        raise NotImplementedError(f'{ctx.label} is not a v1 run_stage{ctx.stage} argument or Step '
            f'name: v1 would silently ignore it and v2 refuses to guess' f'{hint}')
    key = (ctx.stage, ctx.step)
    if key not in _OWNED_STEPS and ctx.name in _STEP_EXPLICIT.get(key, ()):
        _reject_v1_typeerror(ctx, f'run_stage{ctx.stage} already passes {ctx.name!r} to '
                 f'{ctx.step}.run (keyword passed twice)')
    handler = _REGISTRY.get((ctx.stage, ctx.step, '*'))
    if handler is not None:
        return handler(value, ctx)
    if key in _JWST_STEPS:
        _reject_jwst(ctx)
    _reject_v1_typeerror(ctx, f'{ctx.step}.run does not accept it')


def translate_stage_kwargs(cfg):
    """Translate configured stage keywords into reduction updates.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.

    Returns
    -------
    translation : Translation
        Configuration changes, fixed options and warning messages.
    """
    instrument, mode = _instrument(cfg)
    result = Translation()
    for stage in (1, 2, 3):
        name = f'stage{stage}_kwargs'
        block = cfg.get(name)
        if block is None:
            continue
        if not isinstance(block, Mapping):
            raise ValueError(f'{name} must be a mapping, got {block!r}')
        for key, value in block.items():
            if key in REMOVED_V1_STEPS.get(stage, ()):
                result.warnings.append(f'{name}[{key!r}] is ignored: {key} was removed from '
                    'v1 in 2.5.0')
                continue
            if key in STAGE_STEPS[stage]:
                if value is None:
                    value = {}
                if not isinstance(value, Mapping):
                    raise ValueError(f"{name}[{key!r}] must be a mapping of {key}.run "
                        f'keywords, got {value!r}')
                if not value:
                    continue
                runs = step_runs(stage, key, cfg, instrument)
                for kwarg, kwvalue in value.items():
                    ctx = KwargContext(stage, key, str(kwarg), instrument, mode, cfg)
                    if not runs:
                        result.warnings.append(f'{ctx.label} is ignored: v1 does not run {key} '
                            f'for this configuration ({mode})')
                        continue
                    result.merge(_translate_one(ctx, kwvalue))
            else:
                ctx = KwargContext(stage, None, str(key), instrument, mode, cfg)
                result.merge(_translate_one(ctx, value))
    return result


def unported_stage_kwargs(cfg, name):
    """Validate one stage keyword mapping through the registry.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    name : str
        Keyword or step name.

    Returns
    -------
    keywords : None, dict
        None for an absent mapping, otherwise an empty validated mapping.
    """
    value = cfg.get(name)
    if value is None:
        return None
    stage = int(str(name)[len('stage')])
    only = dict(cfg)
    for other in (1, 2, 3):
        if other != stage:
            only[f'stage{other}_kwargs'] = None
    translate_stage_kwargs(only)
    return {}


__all__ = ['KwargContext', 'REMOVED_V1_STEPS', 'STAGE_STEPS', 'Translation', 'V1_NOOP_STEPS',
    'placeholder', 'register', 'registered', 'step_runs',
    'translate_stage_kwargs', 'unported_stage_kwargs']
