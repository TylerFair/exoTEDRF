"""Read and validate YAML settings for exoTEDRF v2 reductions."""

from __future__ import annotations

import numbers
import os
import warnings
from collections.abc import Mapping

import numpy as np
import yaml


# Order sweeps by the first affected science step, as in v1.
SWEEP_TABLE = [('soss_inner_mask_width', 'OneOverFStep_grp', ('NIRISS',)),
    ('soss_outer_mask_width', 'OneOverFStep_grp', ('NIRISS',)),
    ('nirspec_mask_width', 'OneOverFStep_grp', ('NIRSPEC',)),
    ('time_jump_threshold', 'JumpStep', ()), ('time_window', 'JumpStep', ()),
    ('miri_trace_width', 'BackgroundStep', ('MIRI',)),
    ('miri_background_width', 'BackgroundStep', ('MIRI',)),
    ('space_outlier_threshold', 'BadPixStep', ()), ('time_outlier_threshold', 'BadPixStep', ()),
    ('box_size', 'BadPixStep', ()), ('window_size', 'BadPixStep', ()),
    ('extract_width', 'Extract', ())]

SWEEP_PARAMETERS = tuple(row[0] for row in SWEEP_TABLE)
_OPTIMIZER_CONTROL_FLAGS = {'optimize_extract_width_only', 'optimize_from_pca_only'}

# Use the v1 step names accepted by existing YAML files.
STEP_DEFAULTS = dict.fromkeys(('DQInitStep', 'INLCorrStep', 'SuperBiasStep', 'RefPixStep',
    'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep', 'JumpStep', 'RampFitStep',
    'GainScaleStep', 'AssignWCSStep', 'FlatFieldStep', 'BackgroundStep', 'OneOverFStep_int',
    'BadPixStep', 'PCAReconstructStep'), 'run')

_CHECKPOINT_CONTROL = {'OneOverFStep_grp': 'OneOverFStep_grp', 'JumpStep': 'JumpStep',
    'BackgroundStep': 'BackgroundStep', 'BadPixStep': 'BadPixStep'}


class _V1YamlLoader(yaml.SafeLoader):
    """Read safe YAML values and fixed apertures tagged as Python tuples."""


_V1YamlLoader.add_constructor('tag:yaml.org,2002:python/tuple',
    lambda loader, node: tuple(loader.construct_sequence(node)))


def load_config(path):
    """Read YAML settings, converting the literal None string to a missing value.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    cfg : dict
        Loaded configuration.
    """
    with open(os.fspath(path), encoding='utf-8') as stream:
        loaded = yaml.load(stream, Loader=_V1YamlLoader)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError('Optimizer config must contain a YAML mapping')
    for key in loaded:
        if loaded[key] == 'None':
            loaded[key] = None
    return loaded


def coerce_config(config_or_path):
    """Load and normalize a reduction configuration.

    Parameters
    ----------
    config_or_path : dict, str
        Reduction configuration or path to a YAML file.

    Returns
    -------
    cfg : dict
        Normalized reduction configuration.
    """
    if isinstance(config_or_path, Mapping):
        return normalize_v1_config(config_or_path)
    return normalize_v1_config(load_config(config_or_path))


def mode_applies(mode, applicable):
    """Check whether a setting applies to the observing mode.

    Parameters
    ----------
    mode : str
        Instrument and observing mode.
    applicable : tuple[str]
        Applicable mode prefixes. An empty tuple accepts every mode.

    Returns
    -------
    applies : bool
        Whether the mode matches an applicable prefix.
    """
    return not applicable or any(str(mode).upper().startswith(prefix) for prefix in applicable)


def validate_optimizer_values(cfg):
    """Validate fixed settings and optimizer candidate lists.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    """
    known_flags = {*(f'optimize_{param}' for param in SWEEP_PARAMETERS), *_OPTIMIZER_CONTROL_FLAGS}
    for flag_key, enabled in cfg.items():
        if not flag_key.startswith('optimize_'):
            continue
        if not isinstance(enabled, (bool, np.bool_)):
            raise TypeError(f'{flag_key} must be a boolean')
        if flag_key not in known_flags and enabled:
            raise ValueError(f'unknown enabled optimizer flag {flag_key!r}; v2 refuses '
                'to silently omit a requested sweep')

    for param in SWEEP_PARAMETERS:
        flag_key = f'optimize_{param}'
        if flag_key not in cfg:
            continue
        enabled = cfg[flag_key]
        if param not in cfg:
            raise ValueError(f'{param} is required when {flag_key} is set')
        value = cfg[param]
        if enabled:
            if not isinstance(value, list):
                raise ValueError(f'{param} must be list when {flag_key}=True')
            if not value:
                raise ValueError(f'{param} must be a non-empty list when {flag_key}=True')
            try:
                mean = np.mean(np.asarray(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{param} candidates must be numeric') from exc
            if not np.isfinite(mean):
                raise ValueError(f'{param} candidates must have a finite mean')
        elif isinstance(value, list):
            if param == 'extract_width' and value:
                # Accept a fixed list of extraction widths as in v1.
                continue
            raise ValueError(f'{param} must be single value when {flag_key}=False')


def _validate_sweep_step(cfg, param, checkpoint):
    """Reject sweeps whose consuming calibration cannot run."""
    control = _CHECKPOINT_CONTROL.get(checkpoint)
    if control is not None and cfg.get(control, STEP_DEFAULTS[control]) == 'skip':
        raise ValueError(f'optimize_{param}=True requires {control}=run; v2 refuses a '
            'no-op sweep because its winner/log semantics are ambiguous')
    # Reject time-domain sweeps disabled by an enabled up-the-ramp step.
    if param in ('time_jump_threshold', 'time_window') and not cfg.get('flag_in_time', True) and \
            cfg.get('flag_up_ramp', False):
        raise ValueError(f'optimize_{param}=True requires flag_in_time=True when '
            'flag_up_ramp=True; v2 refuses a no-op jump sweep (v1 forces '
            'time-domain flagging only for NGROUPS <= 2)')


def build_sweep_plan(cfg, mode):
    """Build parameter trials in v1 optimizer order.

    Candidate order and repeated values are preserved.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    mode : str
        Instrument and observing mode.

    Returns
    -------
    plan : list
        Checkpoint groups and their ordered parameter candidates.
    initial : dict
        Initial values for fixed and optimized parameters.
    """
    cfg = normalize_extract_widths(cfg, mode)
    validate_optimizer_values(cfg)
    grouped = {}
    initial = {}
    checkpoint_order = []

    for param, checkpoint, applicable in SWEEP_TABLE:
        if param not in cfg or not mode_applies(mode, applicable):
            continue
        value = cfg[param]
        swept = bool(cfg.get(f'optimize_{param}', False))
        if swept:
            values = np.asarray(value)
            # Initialize sweeps at the integer mean of their candidates.
            initial[param] = int(np.mean(values))
            if param == 'extract_width' and any(
                    isinstance(item, ExtractAperture) for item in value):
                # Retain asymmetric candidates as hashable aperture pairs.
                values = list(value)
            _validate_sweep_step(cfg, param, checkpoint)
            if checkpoint not in grouped:
                grouped[checkpoint] = []
                checkpoint_order.append(checkpoint)
            grouped[checkpoint].append((param, values))
        else:
            initial[param] = _v1_fixed_sweep_value(param, value)

    _check_phase1_extract_width(cfg, mode, checkpoint_order)
    return [(name, grouped[name]) for name in checkpoint_order], initial


def phase1_extract_width(cfg, initial_params=None):
    """Select the aperture for detector-setting trials.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    initial_params : None, dict
        Initial optimizer parameter values.

    Returns
    -------
    width : float, ExtractAperture
        Middle candidate, fixed aperture or initial extraction width.
    """
    value = normalize_extract_widths(cfg).get('extract_width')
    if isinstance(value, list):
        if not value:
            raise ValueError('extract_width candidate list cannot be empty')
        return value[len(value) // 2]
    if value is not None:
        return value
    if initial_params is not None and 'extract_width' in initial_params:
        return initial_params['extract_width']
    raise ValueError('extract_width is required for SOSS box extraction')


class V1CompatibilityWarning(UserWarning):
    """Warning for settings normalized to their effective v1 values."""


# Store translated stage options for the fixed recipe.
STAGE_KWARGS_OPTIONS_KEY = '_v2_stage_kwargs_options'

# Identify inputs read only for NIRISS SOSS observations.
_V1_SOSS_ONLY_KEYS = {'soss_background_file': 'the SOSS background model is read for NIRISS '
                            'only (stage1.py:847-856, stage2.py:378-392)',
    'soss_timeseries': 'OneOverFStep reads it for NIRISS only (stage1.py:813-822)',
    'soss_timeseries_o2': 'OneOverFStep reads it for NIRISS only (stage1.py:824-833)',
    'f277w': 'the order-0 mask is built for NIRISS only (stage1.py:858-866, 1000-1008)',
    'inl_amplitude_file': 'INLCorrStep runs for NIRISS/SOSS only (stage1.py:3201-3214)',
    'inl_periods': 'INLCorrStep runs for NIRISS/SOSS only (stage1.py:3201-3214)',
    'soss_specprofile': 'it is only used by SOSS ATOCA extraction (stage3.py:3188-3198)'}


def _log(logger, message):
    """Send a message to the configured logger."""
    if logger is not None:
        logger(message)



def _warn_v1(message):
    """Emit a warning for a setting normalized to its effective v1 value."""
    warnings.warn(message, V1CompatibilityWarning, stacklevel=2)


def _normalize_v1_steps(cfg, instrument, mode, messages):
    """Skip steps that v1 does not run for the instrument."""
    from exotedrf.v2.stage_kwargs import V1_NOOP_STEPS
    for name, where in V1_NOOP_STEPS.get(instrument, {}).items():
        if name in cfg and cfg[name] != 'skip':
            messages.append(f"{name}={cfg[name]!r} has no effect for {mode}: v1 does not "
                f"run it for this instrument ({where}); v2 treats it as " "'skip'")
        cfg[name] = 'skip'


def _normalize_v1_inputs(cfg, instrument, mode, messages):
    """Remove inputs and switches that v1 ignores for this configuration."""
    if instrument != 'NIRISS':
        for name, why in _V1_SOSS_ONLY_KEYS.items():
            if cfg.get(name) is not None:
                messages.append(f'{name}={cfg[name]!r} is ignored for {mode}: {why}')
                cfg[name] = None
        if cfg.get('generate_order0_mask', False):
            messages.append(f'generate_order0_mask={cfg["generate_order0_mask"]!r} is '
                'ignored: v1 never reads this key')
        cfg.pop('generate_order0_mask', None)
    for name in ('SaturationStep', 'TracingStep'):
        if cfg.get(name, 'run') != 'run':
            # Ignore switches folded into saturation flagging and trace preparation.
            messages.append(f'{name}={cfg[name]!r} is ignored: v1 never reads this '
                'switch, and v2 folds the step into ' + ('DQInitStep' if name == 'SaturationStep'
                   else 'its trace preparation'))
        cfg.pop(name, None)
    if instrument == 'MIRI':
        detector = cfg.get('filter_detector')
        if detector not in (None, '', 'MIRIMAGE'):
            messages.append(f'filter_detector={detector!r} is ignored for MIRI/LRS: v1 '
                'selects MIRI segments by EXP_TYPE only (utils.py:1788-1791)')
            cfg['filter_detector'] = ''


def _normalize_v1_methods(cfg, instrument, messages):
    """Apply the effective v1 superbias and 1/f methods."""
    if instrument == 'NIRSPEC':
        oof = cfg.get('oof_method')
        if oof is None or str(oof).lower() == 'scale-achromatic':
            if oof is not None:
                messages.append(f'oof_method={oof!r} is a SOSS method; v1 OneOverFStep '
                    "uses 'median' for NIRSpec instead (stage1.py:884-886)")
            cfg['oof_method'] = 'median'
    elif instrument == 'NIRISS':
        method = cfg.get('superbias_method', 'crds')
        if method != 'crds':
            if str(method).lower() != 'crds':
                messages.append(f'superbias_method={method!r}: v1 has no custom '
                    'superbias for NIRISS and changes the method to crds ' '(stage1.py:471-475)')
            cfg['superbias_method'] = 'crds'


def normalize_v1_config(cfg):
    """Normalize settings that v1 ignores or coerces, emitting V1CompatibilityWarning.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.

    Returns
    -------
    cfg : dict
        New configuration with translated stage keywords and effective v1 settings.
    """
    from exotedrf.v2 import stage_kwargs
    cfg = dict(cfg)
    mode = str(cfg.get('observing_mode', 'NIRISS/SOSS'))
    instrument = mode.upper().split('/')[0]
    messages = []
    _normalize_v1_steps(cfg, instrument, mode, messages)
    _normalize_v1_inputs(cfg, instrument, mode, messages)
    _normalize_v1_methods(cfg, instrument, messages)
    if any(cfg.get(f'stage{n}_kwargs') for n in (1, 2, 3)):
        translated = stage_kwargs.translate_stage_kwargs(cfg)
        cfg.update(translated.config)
        stored = dict(cfg.get(STAGE_KWARGS_OPTIONS_KEY) or {})
        stored.update(translated.options)
        cfg[STAGE_KWARGS_OPTIONS_KEY] = stored
        messages.extend(translated.warnings)
        for n in (1, 2, 3):
            if cfg.get(f'stage{n}_kwargs') is not None:
                cfg[f'stage{n}_kwargs'] = {}
    for message in messages:
        _warn_v1(message)
    return cfg


def _validates_v1_effective_config(function):
    """Validate normalized settings and retain in-place canonicalizations."""
    import functools
    @functools.wraps(function)
    def checked(cfg):
        """Validate effective settings and preserve canonicalized values."""
        normalized = normalize_v1_config(cfg)
        before = dict(normalized)
        function(normalized)
        # Copy validator canonicalizations back without applying instrument normalization.
        if isinstance(cfg, dict):
            for key, value in normalized.items():
                if key not in before or before[key] is not value:
                    cfg[key] = value
        return cfg
    return checked


def _v1_fixed_sweep_value(param, value):
    """Get the effective fixed value, using the first aperture of a fixed list."""
    if param == 'extract_width' and isinstance(value, list) and value:
        _warn_v1(f'extract_width={value!r} is a list but is not optimized: as in '
            f'v1, phase 1 uses {value[len(value) // 2]!r} (the middle entry) '
            f'and the final extraction uses {value[0]!r} (the first entry)')
        return value[0]
    return value


class ExtractAperture(tuple):
    """Store lower and upper aperture distances with the original log spelling.

    Parameters
    ----------
    values : array-like(float)
        Lower and upper centroid-to-edge distances.
    text : None, str
        Original spelling to preserve in the cost log.
    """

    def __new__(cls, values, text=None):
        """Construct a two-sided aperture and retain its original spelling."""
        values = tuple(values)
        if len(values) != 2:
            raise ValueError('an asymmetric aperture needs exactly two widths '
                '(lower_width, upper_width)')
        self = super().__new__(cls, values)
        self._v1_text = str(values) if text is None else str(text)
        return self

    @property
    def lower(self):
        """Get the lower aperture distance."""
        return self[0]

    @property
    def upper(self):
        """Get the upper aperture distance."""
        return self[1]

    def __repr__(self):
        """Return the stored representation."""
        return self._v1_text

    __str__ = __repr__


def _is_real_number(value):
    """Check for a real numeric value excluding booleans."""
    return (not isinstance(value, (bool, np.bool_)) and
            isinstance(value, (int, float, np.integer, np.floating)))


def is_asymmetric_width(width):
    """Check whether a width describes a two-sided aperture.

    Parameters
    ----------
    width : float, array-like(float), dict
        Full aperture width or lower and upper centroid-to-edge distances.

    Returns
    -------
    asymmetric : bool
        Whether the value describes lower and upper aperture distances.
    """
    if isinstance(width, (ExtractAperture, Mapping)):
        return True
    return (isinstance(width, (list, tuple, np.ndarray)) and
            np.ndim(width) == 1 and len(width) == 2)


def extract_width_halves(width):
    """Get the lower and upper aperture distances.

    A scalar full width is halved; two-sided distances are returned unchanged.

    Parameters
    ----------
    width : float, array-like(float), dict
        Full aperture width or lower and upper centroid-to-edge distances.

    Returns
    -------
    lower, upper : float
        Centroid-to-lower-edge and centroid-to-upper-edge distances.
    """
    if isinstance(width, str):
        raise ValueError('String widths are not supported by the low-level extraction helpers.')
    if isinstance(width, Mapping):
        if 'lower' not in width or 'upper' not in width:
            raise ValueError('Width dictionaries must contain "lower" and "upper" keys.')
        lower, upper = width['lower'], width['upper']
    elif _is_real_number(width):
        lower = upper = float(width) / 2
    else:
        try:
            lower, upper = width
        except (TypeError, ValueError):
            raise ValueError('width must be a scalar full width or a '
                             'two-element (lower_width, upper_width) pair.')
    if isinstance(lower, (bool, np.bool_)) or isinstance(upper, (bool, np.bool_)):
        raise ValueError('Extraction widths must be numbers.')
    lower, upper = float(lower), float(upper)
    if not (np.isfinite(lower) and np.isfinite(upper)) or lower <= 0 or upper <= 0:
        raise ValueError('Extraction widths must be strictly positive.')
    return lower, upper


def format_extract_width(width):
    """Format the FITS WIDTH value as in v1.

    Parameters
    ----------
    width : float, array-like(float), dict
        Full aperture width or lower and upper centroid-to-edge distances.

    Returns
    -------
    formatted : str, float
        Aperture metadata for the FITS header.
    """
    if isinstance(width, ExtractAperture):
        return 'lower={}, upper={}'.format(width.lower, width.upper)
    if isinstance(width, str) or np.isscalar(width):
        return width
    if isinstance(width, Mapping):
        if 'lower' in width and 'upper' in width:
            return 'lower={}, upper={}'.format(width['lower'], width['upper'])
        return str(width)
    try:
        lower_width, upper_width = width
    except (TypeError, ValueError):
        return str(width)
    return 'lower={}, upper={}'.format(lower_width, upper_width)


def _normalize_width_value(value, name, *, allow_pair_list):
    """Normalize a fixed width or asymmetric aperture."""
    if value is None or isinstance(value, ExtractAperture):
        return value
    if isinstance(value, str):
        if value == 'optimize':
            return value
        raise ValueError(f"{name} must be a number, 'optimize', or a two-sided "
            f"(lower_width, upper_width) aperture, got {value!r}")
    if isinstance(value, Mapping):
        extract_width_halves(value)
        return ExtractAperture((value['lower'], value['upper']), text=str(value))
    if isinstance(value, list) and not allow_pair_list:
        # Treat a fixed width list as separate aperture choices.
        if not value:
            raise ValueError(f'{name} list cannot be empty')
        return [_normalize_width_value(item, name, allow_pair_list=True) for item in value]
    if isinstance(value, (list, tuple, np.ndarray)):
        extract_width_halves(value)
        return ExtractAperture(tuple(value), text=str(value))
    extract_width_halves(value)
    return value


def _normalize_width_candidates(values, name='extract_width'):
    """Normalize an ordered list of numeric or asymmetric aperture candidates."""
    normalized = []
    for item in values:
        if isinstance(item, ExtractAperture):
            normalized.append(item)
        elif isinstance(item, (list, tuple)):
            extract_width_halves(item)
            normalized.append(ExtractAperture(tuple(item), text=str(item)))
        elif isinstance(item, Mapping):
            # Reject mappings that cannot be averaged to initialize a sweep.
            raise ValueError(f'{name} candidates cannot be {{lower:, upper:}} mappings '
                '(v1 cannot average them, optimize.py:1729); write each '
                'two-sided candidate as a [lower_width, upper_width] list')
        elif _is_real_number(item):
            extract_width_halves(item)
            normalized.append(item)
        else:
            raise ValueError(f'{name} candidates must be numbers or [lower_width, '
                f'upper_width] pairs, got {item!r}')
    kinds = {isinstance(item, ExtractAperture) for item in normalized}
    if len(kinds) > 1:
        raise ValueError(f'{name} candidates must be all full widths or all '
            '[lower_width, upper_width] pairs, not a mixture')
    return normalized


def normalize_extract_widths(cfg, mode=None):
    """Normalize extraction widths in a shallow configuration copy.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    mode : str
        Instrument and observing mode.

    Returns
    -------
    cfg : dict
        Configuration with numeric widths and hashable asymmetric apertures.
    """
    out = dict(cfg)
    if 'extract_width' in out:
        value = out['extract_width']
        swept = bool(out.get('optimize_extract_width', False))
        if swept:
            if isinstance(value, list) and value:
                out['extract_width'] = _normalize_width_candidates(value)
        else:
            out['extract_width'] = _normalize_width_value(
                value, 'extract_width', allow_pair_list=False)
    if out.get('extract_width_soss2') is not None and (
            mode is None or str(mode).upper().startswith('NIRISS')):
        # Treat the order-2 width list as a single asymmetric aperture.
        out['extract_width_soss2'] = _normalize_width_value(
            out['extract_width_soss2'], 'extract_width_soss2', allow_pair_list=True)
    return out


def _check_phase1_extract_width(cfg, mode, checkpoint_order):
    """Require numeric or asymmetric apertures for detector-setting trials."""
    if not any(name != 'Extract' for name in checkpoint_order):
        return
    names = ['extract_width']
    if str(mode).upper().startswith('NIRISS'):
        names.append('extract_width_soss2')
    for name in names:
        if name not in cfg:
            continue
        value = cfg[name]
        if name == 'extract_width' and isinstance(value, list):
            value = value[len(value) // 2] if value else None
        if (name == 'extract_width' and value is None) or value == 'optimize':
            raise ValueError(f'{name}={value!r} cannot score the phase-1 sweeps: v1 '
                'extract_at_step requires a numeric or two-sided aperture '
                '(optimize_helpers.py:20-35); use it only when no detector '
                'parameter is optimized')


def apply_v1_extract_method_fallback(cfg, instrument):
    """Apply the instrument-specific v1 extraction fallback in place.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    instrument : str
        Instrument name.

    Returns
    -------
    method : str
        Effective lower-case extraction method.
    """
    method = str(cfg.get('extract_method', 'box')).lower()
    instrument = str(instrument).upper()
    if instrument != 'NIRISS' and method == 'atoca':
        message = ('ATOCA extraction selected but observation does not use '
                   'NIRISS/SOSS. Switching to box extraction.')
    elif instrument == 'NIRISS' and method == 'optimal':
        message = ('Optimal extraction not available for NIRISS/SOSS. '
                   'Switching to box extraction.')
    else:
        return method
    warnings.warn(message, UserWarning, stacklevel=3)
    cfg['extract_method'] = 'box'
    return 'box'


# Accept supported Stage-3 extraction controls.
SUPPORTED_STAGE3_KWARGS = {'opt_max_iter', 'opt_var_thresh'}
SUPPORTED_EXTRACT1D_KWARGS = {'allow_miri_slope', 'use_pastasoss', 'clip_thresh'}


def stage3_kwargs_supported(cfg):
    """Check whether Stage-3 keywords have v2 implementations.

    Returns
    -------
    supported : bool
        Whether every configured Stage-3 keyword is supported.
    """
    value = cfg.get('stage3_kwargs')
    if value is None:
        return True
    if not isinstance(value, Mapping):
        return False
    for key, item in value.items():
        if key == 'Extract1dStep':
            if not isinstance(item, Mapping) or set(item) - SUPPORTED_EXTRACT1D_KWARGS:
                return False
            slope = item.get('allow_miri_slope', False)
            if not isinstance(slope, (bool, np.bool_)):
                raise ValueError('stage3_kwargs.Extract1dStep.'
                                 'allow_miri_slope must be a boolean')
        elif key not in SUPPORTED_STAGE3_KWARGS:
            return False
    return True


def _extract1d_option(cfg, name, default=None):
    """Read an Extract1dStep option from a stage keyword mapping."""
    stage = cfg.get('stage3_kwargs')
    step = stage.get('Extract1dStep') if isinstance(stage, Mapping) else None
    return step.get(name, default) if isinstance(step, Mapping) else default


def stage3_clip_thresh(cfg):
    """Get the final light-curve clipping threshold.

    Returns
    -------
    threshold : float
        Clipping threshold, defaulting to 10.
    """
    return float(_extract1d_option(cfg, 'clip_thresh', 10))


def stage3_allow_miri_slope(cfg):
    """Get the option for a tilted MIRI extraction trace.

    Returns
    -------
    enabled : bool
        Whether to allow a sloping Stage-3 trace.
    """
    return bool(_extract1d_option(cfg, 'allow_miri_slope', False))


def optimal_extraction_controls(cfg):
    """Get optimal-extraction iteration and rejection controls.

    Stage-3 keywords take precedence over top-level settings.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.

    Returns
    -------
    max_iter : int
        Maximum outlier-rejection iterations.
    var_thresh : float
        Variance threshold for outlier rejection.
    """
    stage3 = cfg.get('stage3_kwargs') or {}
    if not isinstance(stage3, Mapping):
        stage3 = {}
    max_iter = stage3.get('opt_max_iter', cfg.get('opt_max_iter', 25))
    var_thresh = stage3.get('opt_var_thresh', cfg.get('opt_var_thresh', 25))
    return max_iter, var_thresh


def _validate_optimal_extraction(cfg):
    """Validate optimal-extraction controls and scalar aperture widths."""
    max_iter, var_thresh = optimal_extraction_controls(cfg)
    if (isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 0):
        raise ValueError(f'opt_max_iter must be a non-negative integer, got {max_iter!r}')
    if (isinstance(var_thresh, bool) or not isinstance(var_thresh, (int, float, np.integer,
                                        np.floating)) or
            not np.isfinite(var_thresh) or var_thresh <= 0):
        raise ValueError(f'opt_var_thresh must be positive, got {var_thresh!r}')
    width = cfg.get('extract_width')
    widths = width if isinstance(width, list) else [width]
    for item in widths:
        if item is None and not isinstance(width, list):
            continue
        if (not _is_real_number(item) or not np.isfinite(item) or item <= 0):
            raise ValueError(f'extract_width must be finite and positive for optimal '
                f'extraction, got {item!r}')


def _validate_extraction_widths(cfg, method):
    """Validate apertures for the effective extraction method."""
    if method == 'optimal':
        _validate_optimal_extraction(cfg)
        return
    width = cfg.get('extract_width')
    widths = width if isinstance(width, list) else [width]
    if 'extract_width' in cfg and any(item is None for item in widths):
        raise ValueError('extract_width must be a scalar full width, '
                         "'optimize', or a two-element (lower_width, "
                         'upper_width) aperture for box extraction')


def resolve_deepframe_option(cfg):
    """Get the configured Stage-3 deepframe path.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.

    Returns
    -------
    deepframe : None, str
        Configured FITS path, or None for an unset value.
    """
    value = cfg.get('deepframe')
    if value in (None, 'None', 'null', ''):
        return None
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f'deepframe must be a FITS path or None, got {type(value)!r}')
    return value


NIRSPEC_GRATINGS = ('G395H', 'G395M', 'G235H', 'G235M', 'G140H', 'G140M', 'PRISM')


def _validate_supported_nirspec_config(cfg, mode):
    """Validate a supported NIRSpec reduction recipe."""
    grating = mode.split('/', 1)[1].upper() if '/' in mode else ''
    if grating not in NIRSPEC_GRATINGS:
        raise NotImplementedError(f'{mode}: NIRSpec observing_mode must be NIRSpec/<grating> with '
            f'grating in {sorted(NIRSPEC_GRATINGS)}')
    detector = str(cfg.get('filter_detector', '')).upper()
    if detector not in ('NRS1', 'NRS2'):
        raise ValueError(f'NIRSpec requires filter_detector NRS1 or NRS2, got '
            f'{cfg.get("filter_detector")!r}')
    validate_run_mode_options(cfg)
    method = apply_v1_extract_method_fallback(cfg, 'NIRSPEC')
    if method not in ('box', 'optimal'):
        raise NotImplementedError(f'extract_method={method!r}; NIRSpec supports box or optimal '
            'only')
    _validate_extraction_widths(cfg, method)
    sb_method = str(cfg.get('superbias_method', 'crds')).lower()
    if sb_method not in ('crds', 'custom', 'custom-rescale'):
        raise NotImplementedError(f'superbias_method={sb_method!r}; NIRSpec supports crds, custom, '
            'or custom-rescale (stage1.py:446-448)')
    oof = str(cfg.get('oof_method', 'median')).lower()
    if oof not in ('median', 'slope'):
        raise NotImplementedError(f'oof_method={oof!r}; NIRSpec supports median or slope only')

    warn_inert_smoothing_scale(cfg)
    validate_optimizer_values(cfg)
    return cfg


def _validate_supported_miri_config(cfg, mode):
    """Validate a supported MIRI LRS reduction recipe."""
    if mode.upper() != 'MIRI/LRS':
        raise NotImplementedError(f'{mode}: MIRI observing_mode must be exactly MIRI/LRS')
    detector = cfg.get('filter_detector')
    if detector not in (None, '', 'MIRIMAGE'):
        raise ValueError(f'MIRI/LRS does not select a detector via filter_detector, got '
            f'{detector!r}')
    validate_run_mode_options(cfg)
    method = apply_v1_extract_method_fallback(cfg, 'MIRI')
    if method not in ('box', 'optimal'):
        raise NotImplementedError(f'extract_method={method!r}; MIRI/LRS supports box or optimal '
            'only')
    _validate_extraction_widths(cfg, method)
    background = str(cfg.get('miri_background_method', 'median')).lower()
    if background not in ('median', 'slope'):
        raise NotImplementedError(f'miri_background_method={background!r}; MIRI supports median or '
            'slope only (stage2.py:901-950)')
    drop = cfg.get('miri_drop_groups', 12)
    if drop is not None and (isinstance(drop, bool) or not isinstance(drop, int) or drop < 0):
        raise ValueError(f'miri_drop_groups must be None or a non-negative integer, got '
            f'{drop!r}')

    warn_inert_smoothing_scale(cfg)
    validate_optimizer_values(cfg)
    return cfg


def _validate_uncal_input(cfg):
    """Require a non-empty product tag when a tag is supplied."""
    filetag = cfg.get('input_filetag', 'uncal')
    if filetag is not None and (not isinstance(filetag, str) or not filetag.strip()):
        raise ValueError(f'input_filetag must be a non-empty string, got {filetag!r}')


def _is_null_like(value):
    """Check for None or a recognized null string."""
    return value is None or (isinstance(value, str) and
                             value in ('None', 'none', 'null', 'NULL', ''))


def ad_hoc_mode(cfg):
    """Get the requested mode for rerunning saved products.

    Returns
    -------
    mode : None, str
        extract_width_only, from_pca_only or None.
    """
    if cfg.get('optimize_extract_width_only', False):
        return 'extract_width_only'
    if cfg.get('from_pca_only', cfg.get('optimize_from_pca_only', False)):
        return 'from_pca_only'
    return None


def validate_run_mode_options(cfg):
    """Validate restart, ad-hoc and archive options.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    """
    for key in ('optimize_extract_width_only', 'optimize_from_pca_only',
                'from_pca_only', 'reuse_first_pass_extract_width', 'debug_mode', 'v2_archive'):
        if key in cfg and not isinstance(cfg[key], (bool, np.bool_)):
            raise TypeError(f'{key} must be a boolean')
    method = cfg.get('first_pass_extract_method', 'box')
    if not isinstance(method, str) or not method.strip():
        raise ValueError('first_pass_extract_method must be a non-empty string such as box')
    archive = cfg.get('archive_to_longterm_storage')
    if not _is_null_like(archive) and not isinstance(archive, (str, os.PathLike)):
        raise TypeError('archive_to_longterm_storage must be a directory path or null')

    extract_only = bool(cfg.get('optimize_extract_width_only', False))
    from_pca = bool(cfg.get('from_pca_only', cfg.get('optimize_from_pca_only', False)))
    if extract_only and from_pca:
        raise ValueError('optimize_extract_width_only and from_pca_only cannot both be True.')
    restrictions = []
    if extract_only:
        restrictions.append('optimize_extract_width_only=True. Only the Stage 3 '
                            'extraction may be rerun in this mode.')
    if from_pca:
        restrictions.append('from_pca_only=True. Only '
                            'optimize_extract_width may be True in this mode.')
    for reason in restrictions:
        for flag, enabled in cfg.items():
            if (flag.startswith('optimize_') and enabled and flag not in
                    ('optimize_extract_width_only', 'optimize_from_pca_only',
                     'optimize_extract_width')):
                raise ValueError(f'{flag} must be False when {reason}')
    if from_pca:
        if _is_null_like(cfg.get('remove_components')):
            raise ValueError('remove_components must be set when from_pca_only=True')
    if cfg.get('debug_mode', False) and str(cfg.get(
            'v2_search_strategy', 'greedy')).lower() != 'greedy':
        raise ValueError('debug_mode reproduces v1 greedy cached-trial semantics and '
            'requires v2_search_strategy=greedy')


def _validate_stage3_wavelength_options(cfg, mode):
    """Validate stellar parameters and instrument-specific wavelength options."""
    from exotedrf.v2 import wavecal
    wavecal.validate_stellar_options(cfg)
    if wavecal.use_pastasoss_option(cfg) and not str(mode).upper().startswith('NIRISS'):
        warnings.warn('use_pastasoss applies to NIRISS/SOSS only; v1 ignores it for '
            f'{mode} and so does v2', RuntimeWarning, stacklevel=3)


_SEARCH_STRATEGIES = ('greedy', 'joint', 'beam', 'tree')


def _positive_int(cfg, name, default):
    """Read an integer setting greater than or equal to one."""
    value = cfg.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f'{name} must be an integer >= 1, got {value!r}')
    return int(value)


def _bounded_float(cfg, name, default, *, strict):
    """Read a finite positive or non-negative float setting."""
    value = cfg.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a float, got {value!r}')
    value = float(value)
    if not np.isfinite(value) or (strict and value <= 0) or (not strict and value < 0):
        bound = '> 0' if strict else '>= 0'
        raise ValueError(f'{name} must be a float {bound}, got {value!r}')
    return value


def search_options(cfg):
    """Validate and resolve the phase-1 search controls.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.

    Returns
    -------
    options : dict
        Search strategy, beam width, candidate limits and tree tolerance.
    """
    strategy = cfg.get('v2_search_strategy', 'greedy')
    if not isinstance(strategy, str) or strategy.lower() not in _SEARCH_STRATEGIES:
        raise ValueError(f'v2_search_strategy must be one of {_SEARCH_STRATEGIES}, got '
            f'{strategy!r}')
    strategy = strategy.lower()

    beam_width = _positive_int(cfg, 'v2_beam_width', 8 if strategy == 'tree' else 3)
    if strategy == 'joint':
        # Use one retained branch for a joint grid search.
        beam_width = 1
    group_max_evals = _positive_int(cfg, 'v2_group_max_evals', 4096)
    final_candidates = _positive_int(cfg, 'v2_final_candidates', beam_width)
    negligible = _bounded_float(cfg, 'v2_negligible_rel_cost', 1e-3, strict=True)
    _bounded_float(cfg, 'v2_leaf_tolerance', 0.0, strict=False)
    tree_tolerance = _bounded_float(cfg, 'v2_tree_tolerance', negligible, strict=False)
    return {'strategy': strategy, 'beam_width': beam_width, 'final_candidates': final_candidates,
        'group_max_evals': group_max_evals,
        'tree_tolerance': tree_tolerance if strategy == 'tree' else None}


def output_mode(cfg):
    """Get the reduction output policy.

    Returns
    -------
    mode : str
        standard or optimal.
    """
    mode = str(cfg.get('v2_output_mode', 'standard')).lower()
    if mode not in ('standard', 'optimal'):
        raise ValueError('v2_output_mode must be standard or optimal')
    return mode


def stream_stage1_mode(cfg):
    """Get the Stage-1 segment scheduling policy.

    Returns
    -------
    policy : bool, str
        Explicit boolean override or auto.
    """
    value = cfg.get('v2_stream_stage1', 'auto')
    if isinstance(value, bool) or value == 'auto':
        return value
    raise ValueError('v2_stream_stage1 must be true, false, or auto')


def supports_stage1_streaming(cfg):
    """Check whether the recipe supports exact segment scheduling.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.

    Returns
    -------
    supported : bool
        Whether the pre-RampFit steps can run by segment.
    """
    mode = str(cfg.get('observing_mode', 'NIRISS/SOSS')).upper()
    if str(cfg.get('RampFitStep', 'run')).lower() != 'run':
        return False
    if cfg.get('saturation_time_invariant', False):
        return False
    superbias = str(cfg.get('superbias_method', 'crds')).lower()
    if mode.startswith('NIRISS/SOSS') or mode.startswith('NIRSPEC'):
        return superbias == 'crds'
    return mode.startswith('MIRI/LRS')


def warn_inert_smoothing_scale(cfg):
    """Warn when a top-level smoothing_scale setting is ignored.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    """
    if cfg.get('smoothing_scale') is not None:
        warnings.warn('top-level smoothing_scale is ignored, as in v1 (optimize.py '
            'never forwards it); use stage1_kwargs/stage2_kwargs '
            "{'OneOverFStep': {'smoothing_scale': N}} to change the "
            'self-calibrated 1/f light-curve smoothing', UserWarning, stacklevel=3)


ONEOVERF_STEP_KWARGS = ('smoothing_scale', 'even_odd_rows')
_ONEOVERF_LEVELS = (('stage1_kwargs', 'grp'), ('stage2_kwargs', 'int'))


def oneoverf_kwargs_from_stage_kwargs(cfg):
    """Translate group and integration 1/f keywords.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.

    Returns
    -------
    options : dict
        Smoothing scales and even/odd row settings for both levels.
    """
    result = {}
    for container, level in _ONEOVERF_LEVELS:
        stage_kwargs = cfg.get(container)
        step = (stage_kwargs.get('OneOverFStep') if isinstance(stage_kwargs, Mapping) else None)
        if step is None:
            step = {}
        if not isinstance(step, Mapping):
            raise TypeError(f"{container}['OneOverFStep'] must be a mapping")
        unknown = sorted(set(step) - set(ONEOVERF_STEP_KWARGS))
        if unknown:
            raise NotImplementedError(f"{container}['OneOverFStep'] keys {unknown} are not "
                f'supported by v2; supported: {list(ONEOVERF_STEP_KWARGS)}')
        scale = step.get('smoothing_scale')
        if scale is not None:
            if isinstance(scale, (bool, np.bool_)) or not isinstance(
                    scale, numbers.Real) or not np.isfinite(scale) or int(scale) < 1:
                raise ValueError(f"{container}['OneOverFStep']['smoothing_scale'] must be "
                    f'None or a number >= 1 (v1 uses int(value) as the '
                    f'median-filter size), got {scale!r}')
            scale = int(scale)
        even_odd = step.get('even_odd_rows', True)
        if not isinstance(even_odd, (bool, np.bool_)):
            raise TypeError(f"{container}['OneOverFStep']['even_odd_rows'] must be a "
                f'boolean, got {even_odd!r}')
        result[f'oof_smoothing_scale_{level}'] = scale
        result[f'oof_even_odd_rows_{level}'] = bool(even_odd)
    return result


@_validates_v1_effective_config
def validate_supported_config(cfg):
    """Validate the effective reduction configuration.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.

    Returns
    -------
    cfg : dict
        Original configuration with canonicalized widths and extraction methods.
    """
    _validate_upramp_jump_config(cfg)
    _validate_uncal_input(cfg)
    search_options(cfg)
    output_mode(cfg)
    mode = str(cfg.get('observing_mode', 'NIRISS/SOSS'))
    _validate_stage3_wavelength_options(cfg, mode)
    # Use canonical apertures throughout optimization, products and logs.
    cfg.update(normalize_extract_widths(cfg, mode))
    resolve_deepframe_option(cfg)
    if stream_stage1_mode(cfg) is True and not supports_stage1_streaming(cfg):
        raise ValueError('v2_stream_stage1 requires NIRISS/SOSS, NIRSpec or MIRI/LRS '
            'with RampFitStep: run, and CRDS superbias for NIRISS/NIRSpec')
    if mode.upper().startswith('NIRSPEC'):
        return _validate_supported_nirspec_config(cfg, mode)
    if mode.upper().startswith('MIRI'):
        return _validate_supported_miri_config(cfg, mode)
    if not mode.upper().startswith('NIRISS/SOSS'):
        raise NotImplementedError(f'{mode}: v2 supports NIRISS/SOSS, NIRSpec, and MIRI/LRS only')
    validate_run_mode_options(cfg)
    apply_v1_extract_method_fallback(cfg, 'NIRISS')
    if str(cfg.get('extract_method', 'box')).lower() == 'atoca':
        from exotedrf.v2 import wavecal as _wavecal
        if _wavecal.stellar_refinement_enabled(cfg):
            # Reject wavelength refinements unsupported by ATOCA products.
            raise NotImplementedError('st_teff/st_logg/st_met wavelength refinement is not yet '
                'supported with extract_method: atoca in v2; use box '
                'extraction for refined wavelengths')
    method = str(cfg.get('extract_method', 'box')).lower()
    if method == 'atoca':
        _validate_atoca_config(cfg)
    elif method != 'box':
        raise NotImplementedError(f'extract_method={method!r}; v2 currently supports box only')
    if str(cfg.get('superbias_method', 'crds')).lower() != 'crds':
        raise NotImplementedError('NIRISS/SOSS v1 always uses superbias_method=crds; v2 rejects '
            'custom/rescale requests instead of silently changing them')
    oof = str(cfg.get('oof_method', 'scale-achromatic')).lower()
    supported_oof = {'scale-achromatic', 'scale-achromatic-window', 'scale-chromatic', 'solve'}
    if oof not in supported_oof:
        raise NotImplementedError(f'oof_method={oof!r}; supported methods are '
            f'{sorted(supported_oof)}')
    _validate_extraction_widths(cfg, 'box')

    warn_inert_smoothing_scale(cfg)
    validate_optimizer_values(cfg)
    return cfg


def atoca_soss_estimate(cfg):
    """Get the configured ATOCA initial estimate.

    Returns
    -------
    estimate : None, str
        Initial estimate path, if configured.
    """
    return _extract1d_option(cfg, 'soss_estimate')


def _validate_atoca_config(cfg):
    """Validate a NIRISS SOSS ATOCA extraction recipe."""
    widths = cfg.get('extract_width')
    for width in (widths if isinstance(widths, list) else [widths]):
        if width == 'optimize':
            raise ValueError('Aperture optimization not possible with ATOCA extraction.')
        if width is not None and (isinstance(width, (str, bool, dict, list))
                                  or not np.isscalar(width)):
            raise ValueError(f'ATOCA extract_width must be numeric, got {width!r}')
    detector = str(cfg.get('filter_detector', 'CLEAR') or 'CLEAR').upper()
    if detector != 'CLEAR':
        raise NotImplementedError('jwst 1.17.1 ATOCA supports the SOSS CLEAR filter only, got '
            f'filter_detector={detector!r}')
    for name in ('soss_specprofile', 'v2_atoca_workers',
                 'v2_atoca_solver_threads', 'v2_atoca_chunk_ints'):
        value = cfg.get(name)
        if value is None or value == 'auto':
            continue
        if name == 'soss_specprofile':
            if not isinstance(value, (str, os.PathLike)):
                raise ValueError('soss_specprofile must be a file path or None')
        elif isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer or auto')
    estimate = atoca_soss_estimate(cfg)
    if estimate is not None and not isinstance(estimate, (str, os.PathLike)):
        raise ValueError('Extract1dStep soss_estimate must be a file path')


def fixed_options(cfg):
    """Resolve the fixed recipe and instrument-specific defaults.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.

    Returns
    -------
    options : dict
        Reduction options shared by all optimizer trials.
    """
    from exotedrf.v2 import wavecal
    cfg = normalize_v1_config(cfg)
    w2 = float(cfg.get('w2', 1.0))
    wave_range = cfg.get('wave_range')
    mode_upper = str(cfg.get('observing_mode', 'NIRISS/SOSS')).upper()
    instrument = ('NIRSPEC' if mode_upper.startswith('NIRSPEC') else
                  'MIRI' if mode_upper.startswith('MIRI') else 'NIRISS')
    band_name, band_lo, band_hi, default_range = {'NIRSPEC': ('NIRSpec', 1.0, 5.0, None),
        'MIRI': ('MIRI', 5.0, 12.0, [5.0, 10.0]), 'NIRISS': ('SOSS', 0.6, 2.8, [1.0, 2.0]),
    }[instrument]
    if wave_range is None and w2 != 0.0:
        if instrument == 'NIRSPEC':
            detector = str(cfg.get('filter_detector', '')).upper()
            if detector not in ('NRS1', 'NRS2'):
                raise ValueError('NIRSpec spectral optimization requires '
                                 'filter_detector=NRS1 or NRS2')
            default_range = {'NRS1': [3.0, 3.5], 'NRS2': [4.0, 4.5]}[detector]
        wave_range = default_range
    if wave_range is not None:
        if not isinstance(wave_range, (list, tuple)) or len(wave_range) != 2:
            raise ValueError('wave_range must be None or a length-2 sequence')
        lo = band_lo if wave_range[0] is None else float(wave_range[0])
        hi = band_hi if wave_range[1] is None else float(wave_range[1])
        if lo > hi or lo < band_lo or hi > band_hi:
            raise ValueError(f'{band_name} wave_range must lie within '
                f'[{band_lo}, {band_hi}] microns')

    width_soss2 = normalize_extract_widths(cfg, mode_upper).get('extract_width_soss2')
    max_iter, var_thresh = optimal_extraction_controls(cfg)
    opts = {'input_dir': cfg.get('input_dir', './'),
        'input_filetag': cfg.get('input_filetag', 'uncal'),
        'mode': cfg.get('observing_mode', 'NIRISS/SOSS'), 'filter_detector': cfg.get(
            'filter_detector', '' if mode_upper.startswith('MIRI') else 'CLEAR'),
        'baseline_ints': cfg.get('baseline_ints', [100, -100]),
        'oof_method': cfg.get('oof_method', 'scale-achromatic'),
        'soss_background_file': cfg.get('soss_background_file'),
        'soss_timeseries': cfg.get('soss_timeseries'),
        'soss_timeseries_o2': cfg.get('soss_timeseries_o2'),
        'generate_lc': cfg.get('generate_lc', True), 'smoothing_scale': cfg.get('smoothing_scale'),
        'jump_threshold': cfg.get('jump_threshold', 15),
        'flag_up_ramp': cfg.get('flag_up_ramp', False),
        'flag_in_time': cfg.get('flag_in_time', True),
        'pca_components': cfg.get('pca_components', 10),
        'remove_components': cfg.get('remove_components'),
        'extract_method': cfg.get('extract_method', 'box'), 'extract_width_soss2': width_soss2,
        'opt_max_iter': max_iter, 'opt_var_thresh': var_thresh,
        'deepframe': resolve_deepframe_option(cfg),
        'stage3_allow_miri_slope': stage3_allow_miri_slope(cfg),
        # Refine wavelengths only when all three stellar parameters are set.
        'st_teff': cfg.get('st_teff'), 'st_logg': cfg.get('st_logg'), 'st_met': cfg.get('st_met'),
        'stellar_model_dir': cfg.get('v2_stellar_model_dir'),
        'use_pastasoss': (mode_upper.startswith('NIRISS') and wavecal.use_pastasoss_option(cfg)),
        'centroids': cfg.get('centroids'), 'wave_range': wave_range,
        'wave_range_plot': cfg.get('wave_range_plot'), 'ylim_plot': cfg.get('ylim_plot'),
        'w1': float(cfg.get('w1', 0.0)), 'w2': w2,
        'output_dir': cfg.get('pipeline_outputs_directory', 'pipeline_outputs_directory'),
        'pipeline_outputs_directory': cfg.get(
            'pipeline_outputs_directory', 'pipeline_outputs_directory'),
        'output_tag': cfg.get('output_tag', ''), 'name_tag': cfg.get('name_tag', ''),
        'output_mode': output_mode(cfg), 'stream_stage1': stream_stage1_mode(cfg),
        'do_plots': cfg.get('do_plots', output_mode(cfg) == 'optimal'), 'saturation_threshold': (
            80 if cfg.get('saturation_threshold') is None else cfg.get('saturation_threshold')),
        'saturation_rescue': cfg.get('saturation_rescue', False),
        'suppress_one_group': bool(cfg.get('suppress_one_group', True)),
        'saturation_time_invariant': bool(cfg.get('saturation_time_invariant', False)),
        'saturation_time_invariant_quantile': float(
            cfg.get('saturation_time_invariant_quantile', 0.01)),
        'hot_pixel_map': cfg.get('hot_pixel_map'), 'outlier_maps': cfg.get('outlier_maps'),
        'f277w': cfg.get('f277w'),
        'mask_do_not_use_pixels': cfg.get('mask_do_not_use_pixels', False),
        'mask_saturated_pixels': cfg.get('mask_saturated_pixels', False),
        'clip_thresh': stage3_clip_thresh(cfg),
        'miri_background_method': cfg.get('miri_background_method', 'median'),
        # Drop RSCD-affected MIRI groups and subtract dark during linearity correction.
        'miri_drop_groups': cfg.get('miri_drop_groups', 12),
        'miri_subtract_dark': cfg.get('miri_subtract_dark', True),
        'superbias_method': cfg.get('superbias_method', 'crds'),
        'inl_amplitude_file': cfg.get('inl_amplitude_file'), 'inl_periods': cfg.get('inl_periods'),
        'crds_context': cfg.get('crds_context', 'jwst_1322.pmap'), 'refpack': cfg.get('refpack'),
        'wavemap_file': cfg.get('wavemap_file'),
        # Use scratch storage when the host memory allowance is reached.
        'max_host_bytes': cfg.get('v2_max_host_bytes'), 'scratch_dir': cfg.get('v2_scratch_dir'),
        'soss_specprofile': cfg.get('soss_specprofile'), 'soss_estimate': atoca_soss_estimate(cfg),
        'v2_atoca_workers': cfg.get('v2_atoca_workers'),
        'v2_atoca_solver_threads': cfg.get('v2_atoca_solver_threads'),
        'v2_atoca_chunk_ints': cfg.get('v2_atoca_chunk_ints'),}
    opts.update(oneoverf_kwargs_from_stage_kwargs(cfg))
    opts.update({key: cfg.get(key, default) for key, default in STEP_DEFAULTS.items()})
    # Skip NIRISS-only INL calibration for other instruments.
    if 'INLCorrStep' not in cfg and not mode_upper.startswith('NIRISS'):
        opts['INLCorrStep'] = 'skip'
    opts['EmiCorrStep'] = cfg.get('EmiCorrStep', 'run')
    # Build NIRSpec wavelengths without the removed WaveCorr step.
    opts['WaveCorrStep'] = 'skip'
    opts.update(upramp_jump_options(cfg))
    # Apply translated stage options after resolving defaults.
    opts.update(cfg.get(STAGE_KWARGS_OPTIONS_KEY) or {})
    return opts


# Accept supported JWST JumpStep override names.
_JUMPSTEP_FLOAT_KWARGS = ('three_group_rejection_threshold', 'four_group_rejection_threshold',
    'max_jump_to_flag_neighbors', 'min_jump_to_flag_neighbors',
    'after_jump_flag_dn1', 'after_jump_flag_time1', 'after_jump_flag_dn2',
    'after_jump_flag_time2', 'min_sat_area', 'min_jump_area',
    'expand_factor', 'min_sat_radius_extend', 'max_shower_amplitude',
    'extend_snr_threshold', 'extend_inner_radius', 'extend_outer_radius',
    'extend_ellipse_expand_ratio', 'time_masked_after_shower')
_JUMPSTEP_BOOL_KWARGS = ('flag_4_neighbors', 'expand_large_events', 'use_ellipses',
    'sat_required_snowball', 'mask_snowball_core_next_int', 'find_showers', 'only_use_ints')
_JUMPSTEP_INT_KWARGS = ('sat_expand', 'edge_size', 'snowball_time_masked_next_int',
    'extend_min_area', 'min_diffs_single_pass', 'max_extended_radius', 'minimum_groups')
# Reject keywords that v1 already passes explicitly.
_JUMPSTEP_V1_DUPLICATES = {'rejection_threshold': 'use the top-level jump_threshold key',
    'flag_up_ramp': 'use the top-level flag_up_ramp key',
    'flag_in_time': 'use the top-level flag_in_time key',
    'time_rejection_threshold': 'use the top-level time_jump_threshold key',
    'save_results': 'v1 sets it itself', 'force_redo': 'v1 sets it itself',
    'do_plot': 'use the top-level do_plots key', 'show_plot': 'v1 sets it itself',
    'maximum_cores': "v1 hard-codes maximum_cores='quarter'",
    'minimum_sigclip_groups': 'v1 hard-codes minimum_sigclip_groups=1e6',
    'output_dir': 'v1 sets it itself'}


def _is_number(value):
    """Check for a finite real numeric value excluding booleans."""
    return _is_real_number(value) and np.isfinite(value)


def jump_kwargs_from_stage1_kwargs(cfg):
    """Validate JWST up-the-ramp JumpStep overrides.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.

    Returns
    -------
    options : dict
        Validated JWST jump parameters.
    """
    stage1 = cfg.get('stage1_kwargs')
    if stage1 is None or not isinstance(stage1, Mapping):
        return {}
    raw = stage1.get('JumpStep')
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("stage1_kwargs['JumpStep'] must be a mapping")
    out = {}
    for key, value in raw.items():
        if key in _JUMPSTEP_V1_DUPLICATES:
            raise ValueError(f"stage1_kwargs['JumpStep'][{key!r}] duplicates a keyword "
                f'v1 passes itself (TypeError in v1); ' f'{_JUMPSTEP_V1_DUPLICATES[key]}')
        if key == 'time_window':
            raise NotImplementedError("stage1_kwargs['JumpStep']['time_window'] is not translated: "
                'v2 reads the time-domain window from the top-level '
                'time_window key (v1 mutates this dict during a time_window '
                'sweep, so its final-run window is the last trial value)')
        if key in _JUMPSTEP_FLOAT_KWARGS:
            if not _is_number(value):
                raise ValueError(f"stage1_kwargs['JumpStep'][{key!r}] must be a finite "
                    f'number, got {value!r}')
            out[key] = float(value)
        elif key in _JUMPSTEP_BOOL_KWARGS:
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"stage1_kwargs['JumpStep'][{key!r}] must be a boolean, "
                    f'got {value!r}')
            out[key] = bool(value)
        elif key in _JUMPSTEP_INT_KWARGS:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"stage1_kwargs['JumpStep'][{key!r}] must be an integer, "
                    f'got {value!r}')
            out[key] = int(value)
        else:
            raise NotImplementedError(f"stage1_kwargs['JumpStep'][{key!r}] is not a supported "
                'JWST JumpStep jump parameter in v2')
    for key in ('three_group_rejection_threshold', 'four_group_rejection_threshold'):
        if key in out and out[key] < 0:
            raise ValueError(f"stage1_kwargs['JumpStep'][{key!r}] must be "
                             '>= 0 (jwst spec min=0)')
    if out.get('find_showers', False):
        raise NotImplementedError('JumpStep find_showers=True (MIRI shower flagging) is not ported '
            'to v2; no CRDS pars-jumpstep file used by v1 modes enables it')
    return out


def jumpstep_kwargs_options(step_kwargs, cfg):
    """Translate a JumpStep keyword mapping into v2 options.

    Parameters
    ----------
    step_kwargs, cfg : dict
        JWST JumpStep keyword arguments and reduction configuration.

    Returns
    -------
    options : dict
        Validated JWST jump parameters.
    """
    merged = dict(cfg) if isinstance(cfg, Mapping) else {}
    stage1 = dict(merged.get('stage1_kwargs') or {})
    stage1['JumpStep'] = step_kwargs
    merged['stage1_kwargs'] = stage1
    return jump_kwargs_from_stage1_kwargs(merged)


def _validate_upramp_jump_config(cfg):
    """Validate top-level JWST up-the-ramp jump settings."""
    if not cfg.get('flag_up_ramp', False):
        return
    threshold = cfg.get('jump_threshold', 15)
    if not _is_number(threshold) or threshold < 0:
        raise ValueError('flag_up_ramp=True requires jump_threshold to be a finite '
            f'number >= 0 (JumpStep rejection_threshold), got {threshold!r}')
    count = cfg.get('v2_upramp_cpu_count')
    if count is not None and (isinstance(count, bool) or not isinstance(count, (int, np.integer)) or
                              count < 1):
        raise ValueError(f'v2_upramp_cpu_count must be None or an integer >= 1, got {count!r}')


def upramp_jump_options(cfg):
    """Resolve up-the-ramp and time-domain jump options.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.

    Returns
    -------
    options : dict
        Time-domain flag, JWST overrides and CPU-count setting.
    """
    flag_in_time = cfg.get('flag_in_time', True)
    if not flag_in_time and not cfg.get('flag_up_ramp', False):
        warnings.warn('flag_in_time=False has no effect without flag_up_ramp=True: v1 '
            'forces time-domain jump flagging when the JWST up-the-ramp '
            'JumpStep does not run (stage1.py:1437-1440); running it')
        flag_in_time = True
    return {'flag_in_time': flag_in_time, 'upramp_jump_kwargs': jump_kwargs_from_stage1_kwargs(cfg),
        'upramp_cpu_count': cfg.get('v2_upramp_cpu_count')}


__all__ = ['ONEOVERF_STEP_KWARGS', 'STEP_DEFAULTS', 'SWEEP_PARAMETERS',
    'SWEEP_TABLE', 'build_sweep_plan',
    'coerce_config', 'fixed_options', 'jump_kwargs_from_stage1_kwargs',
    'load_config', 'mode_applies',
    'oneoverf_kwargs_from_stage_kwargs', 'warn_inert_smoothing_scale',
    'phase1_extract_width', 'search_options', 'validate_optimizer_values',
    'validate_supported_config', 'ad_hoc_mode', 'validate_run_mode_options']
