"""Compare calibration checkpoints, extracted spectra, and optimizer settings."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _datetime
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np

from exotedrf.v2.products import v1_output_root


REPORT_SCHEMA_VERSION = 1


V1_FITS_CHECKPOINTS = {'Stage1': (('DQInitStep', '_dqinitstep.fits'),
    ('INLCorrStep', '_inlcorrstep.fits'), ('EmiCorrStep', '_emicorrstep.fits'),
    ('ResetStep', '_resetstep.fits'), ('SuperBiasStep', '_superbiasstep.fits'),
    ('RefPixStep', '_refpixstep.fits'), ('DarkCurrentStep', '_darkcurrentstep.fits'),
    ('BackgroundStep_grp', '_backgroundstep.fits'), ('OneOverFStep_grp', '_oneoverfstep.fits'),
    ('LinearityStep', '_linearitystep.fits'), ('JumpStep', '_jump.fits'),
    ('RampFitStep', '_rampfitstep.fits'), ('GainScaleStep', '_gainscalestep.fits'),),
    'Stage2': (('AssignWCSStep', '_assignwcsstep.fits'), ('Extract2DStep', '_extract2dstep.fits'),
    ('SourceTypeStep', '_sourcetypestep.fits'), ('WaveCorrStep', '_wavecorrstep.fits'),
    ('FlatFieldStep', '_flatfieldstep.fits'), ('BackgroundStep', '_backgroundstep.fits'),
    ('OneOverFStep_int', '_oneoverfstep.fits'), ('BadPixStep', '_badpixstep.fits'),
    ('PCAReconstructStep', '_pcareconstructstep.fits'),)}

_V1_EXTENSION_FIELDS = {'SCI': 'data', 'ERR': 'err', 'DQ': 'dq', 'GROUPDQ': 'groupdq',
    'PIXELDQ': 'pixeldq'}


@dataclasses.dataclass(frozen=True)
class ArrayTolerance:
    """Define allowed numerical and finite-mask differences for array comparisons.

    Parameters
    ----------
    rtol : float
        Relative tolerance for finite numerical values.
    atol : float
        Absolute tolerance for finite numerical values.
    max_finite_mismatch_fraction : float
        Maximum fraction of pixels with different finite-value masks.
    """

    rtol: float = 1e-5
    atol: float = 1e-6
    max_finite_mismatch_fraction: float = 0.0


def _utc_now() -> str:
    """Return the current UTC time in ISO format."""
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _json_number(value: float) -> float | None:
    """Convert a finite number to float, or return None."""
    value = float(value)
    return value if np.isfinite(value) else None


def _coerce_tolerance(value: ArrayTolerance | Mapping[str, Any] | None,
        default: ArrayTolerance) -> ArrayTolerance:
    """Resolve a tolerance instance, mapping, or default."""
    if value is None:
        return default
    if isinstance(value, ArrayTolerance):
        return value
    return ArrayTolerance(**dict(value))


def compare_array(reference: Any, candidate: Any,
        tolerance: ArrayTolerance | None = None) -> dict[str, Any]:
    """Compare one v1 array with its v2 counterpart and summarize differences.

    Parameters
    ----------
    reference, candidate : array-like
        Reference array and candidate array to compare.
    tolerance : None, ArrayTolerance
        Allowed numerical and finite-mask differences.

    Returns
    -------
    result : dict
        Shape, mask, residual, and pass/fail metrics.
    """
    tol = tolerance or ArrayTolerance()
    ref = np.asarray(reference)
    got = np.asarray(candidate)
    result: dict[str, Any] = {'reference_shape': list(ref.shape),
        'candidate_shape': list(got.shape), 'reference_dtype': str(ref.dtype),
        'candidate_dtype': str(got.dtype), 'rtol': float(tol.rtol), 'atol': float(tol.atol),
        'max_finite_mismatch_fraction': float(tol.max_finite_mismatch_fraction)}
    if ref.shape != got.shape:
        result.update({'passed': False, 'reason': 'shape_mismatch', 'element_count': int(ref.size),
            })
        return result

    # Compare non-numeric values exactly.
    if ref.dtype.kind not in 'biufc' or got.dtype.kind not in 'biufc':
        equal = np.array_equal(ref, got)
        result.update({'passed': bool(equal), 'reason': 'exact_non_numeric_comparison',
            'element_count': int(ref.size), 'exact_match_count': int(np.count_nonzero(ref == got)),
            })
        return result

    # Compare integer DQ flags and Boolean masks exactly.
    if ref.dtype.kind in 'biu' and got.dtype.kind in 'biu':
        equal = ref == got
        exact_count = int(np.count_nonzero(equal))
        mismatch_count = int(ref.size - exact_count)
        result.update({'passed': mismatch_count == 0,
            'reason': ('exact_integer_match' if mismatch_count == 0 else 'integer_mismatch'),
            'element_count': int(ref.size), 'exact_match_count': exact_count,
            'mismatch_count': mismatch_count,
            'mismatch_fraction': float(mismatch_count / max(1, ref.size))})
        return result
    ref_finite = np.isfinite(ref)
    got_finite = np.isfinite(got)
    common = ref_finite & got_finite
    finite_mismatch = ref_finite ^ got_finite
    total = int(ref.size)
    mismatch_count = int(np.count_nonzero(finite_mismatch))
    mismatch_fraction = mismatch_count / max(1, total)
    both_nonfinite = ~ref_finite & ~got_finite
    same_nonfinite = ((np.isnan(ref) & np.isnan(got)) | (np.isposinf(ref) & np.isposinf(got))
        | (np.isneginf(ref) & np.isneginf(got)))
    nonfinite_kind_mismatch = int(np.count_nonzero(both_nonfinite & ~same_nonfinite))
    common_count = int(np.count_nonzero(common))
    if common_count:
        ref_common = ref[common]
        got_common = got[common]
        close = np.isclose(ref_common, got_common, rtol=tol.rtol, atol=tol.atol, equal_nan=False)
        close_count = int(np.count_nonzero(close))
        calculation_dtype = (np.complex128 if (np.iscomplexobj(ref_common)
            or np.iscomplexobj(got_common)) else np.float64)
        ref_calculation = ref_common.astype(calculation_dtype, copy=False)
        got_calculation = got_common.astype(calculation_dtype, copy=False)
        absolute = np.abs(got_calculation - ref_calculation)
        denominator = np.maximum(np.abs(ref_calculation), np.finfo(np.float64).tiny)
        relative = absolute / denominator
        max_abs = _json_number(np.max(absolute))
        mean_abs = _json_number(np.mean(absolute))
        max_rel = _json_number(np.max(relative))
    else:
        close_count = 0
        max_abs = mean_abs = max_rel = None
    mask_pass = (mismatch_fraction <= tol.max_finite_mismatch_fraction
        and nonfinite_kind_mismatch == 0)
    values_pass = close_count == common_count
    passed = bool(mask_pass and values_pass)
    result.update({'passed': passed,
        'reason': 'within_tolerance' if passed else 'comparison_failed', 'element_count': total,
        'reference_finite_count': int(np.count_nonzero(ref_finite)),
        'candidate_finite_count': int(np.count_nonzero(got_finite)),
        'common_finite_count': common_count, 'finite_mask_mismatch_count': mismatch_count,
        'finite_mask_mismatch_fraction': float(mismatch_fraction),
        'nonfinite_kind_mismatch_count': nonfinite_kind_mismatch,
        'within_tolerance_count': close_count, 'within_tolerance_fraction': (
        float(close_count / common_count) if common_count else 1.0),
        'max_absolute_error': max_abs, 'mean_absolute_error': mean_abs,
        'max_relative_error': max_rel})
    return result


def _array_leaves(value, prefix, *, selectors=None, dataclass_fields=False, root=''):
    """Yield selected dotted array fields without transferring unrelated leaves."""
    if selectors is not None and not _selector_reaches(prefix, selectors):
        return
    if dataclass_fields and dataclasses.is_dataclass(value):
        value = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping) or isinstance(value, (tuple, list)):
        children = value.items() if isinstance(value, Mapping) else enumerate(value)
        for name, item in children:
            child = str(name) if prefix == root else f'{prefix}.{name}'
            yield from _array_leaves(item, child, selectors=selectors,
                dataclass_fields=dataclass_fields, root=root)
    elif hasattr(value, 'shape') or np.isscalar(value):
        yield prefix, value


def _flatten_arrays(value: Any, prefix: str = 'value') -> dict[str, Any]:
    """Give every numerical array in a nested result a readable field name."""
    return dict(_array_leaves(value, prefix, dataclass_fields=True, root='value'))


def _tolerance_for(tolerances: Mapping[str, Any], step: str, field: str,
        default: ArrayTolerance) -> ArrayTolerance:
    """Select a field-specific, step-specific, or default tolerance."""
    specific = f'{step}::{field}'
    value = tolerances.get(specific, tolerances.get(step))
    return _coerce_tolerance(value, default)


def compare_named_steps(reference_steps: Mapping[str, Any], candidate_steps: Mapping[str, Any], *,
        tolerances: Mapping[str, ArrayTolerance | Mapping[str, Any]] | None = None,
        default_tolerance: ArrayTolerance | None = None, kind: str = 'stepwise',
        parity_scope: str = 'provided_checkpoints',) -> dict[str, Any]:
    """Compare every requested v1 and v2 reduction checkpoint.

    Parameters
    ----------
    reference_steps, candidate_steps : dict
        Reference and candidate arrays grouped by step name.
    tolerances : None, dict
        Tolerances keyed by step name or "step::field".
    default_tolerance : None, ArrayTolerance
        Tolerance for fields without an override.
    kind : str
        Comparison type recorded in the report.
    parity_scope : str
        Scope of the comparison recorded in the report.

    Returns
    -------
    report : dict
        Comparison results for each step and field.
    """
    tolerances = tolerances or {}
    default = default_tolerance or ArrayTolerance()
    all_steps = list(reference_steps)
    all_steps.extend(name for name in candidate_steps if name not in reference_steps)
    steps: dict[str, Any] = {}
    arrays_compared = 0
    arrays_passed = 0

    for step_name in all_steps:
        if step_name not in reference_steps or step_name not in candidate_steps:
            missing = ('reference' if step_name not in reference_steps else 'candidate')
            steps[step_name] = {'passed': False, 'reason': f'missing_{missing}_step', 'arrays': {}}
            continue
        ref_fields = _flatten_arrays(reference_steps[step_name])
        got_fields = _flatten_arrays(candidate_steps[step_name])
        all_fields = list(ref_fields)
        all_fields.extend(field for field in got_fields if field not in ref_fields)
        field_reports: dict[str, Any] = {}
        for field in all_fields:
            if field not in ref_fields or field not in got_fields:
                missing = 'reference' if field not in ref_fields else 'candidate'
                field_reports[field] = {'passed': False, 'reason': f'missing_{missing}_field'}
            else:
                metric = compare_array(ref_fields[field], got_fields[field],
                    _tolerance_for(tolerances, step_name, field, default))
                field_reports[field] = metric
                arrays_compared += 1
                arrays_passed += int(metric['passed'])
        step_passed = bool(field_reports) and all(
            metric['passed'] for metric in field_reports.values())
        steps[step_name] = {'passed': step_passed,
            'reason': 'within_tolerance' if step_passed else 'comparison_failed',
            'arrays': field_reports}
    passed = bool(steps) and all(step['passed'] for step in steps.values())
    return {'schema_version': REPORT_SCHEMA_VERSION, 'created_at_utc': _utc_now(), 'kind': kind,
        'status': 'passed' if passed else 'failed', 'passed': passed, 'parity_claimed': passed,
        'parity_scope': parity_scope, 'summary': {'steps_compared': len(steps),
        'steps_passed': sum(int(step['passed']) for step in steps.values()),
        'arrays_compared': arrays_compared, 'arrays_passed': arrays_passed}, 'steps': steps}


_WINNER_UNSET = object()


def _winner_equal(left: Any, right: Any) -> bool:
    """Compare optimizer settings recursively, treating matching NaNs as equal."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (set(left) == set(right)
            and all(_winner_equal(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return (len(left) == len(right) and all(_winner_equal(a, b) for a, b in zip(left, right)))
    if hasattr(left, 'shape') or hasattr(right, 'shape'):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))
        except (TypeError, ValueError):
            return False
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _json_value(value: Any) -> Any:
    """Convert report values to JSON values, representing nonfinite floats as strings."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def compare_precision_runs(float32_steps: Mapping[str, Any], float64_steps: Mapping[str, Any], *,
        tolerances: Mapping[str, ArrayTolerance | Mapping[str, Any]] | None = None,
        default_tolerance: ArrayTolerance | None = None, float32_winner: Any = _WINNER_UNSET,
        float64_winner: Any = _WINNER_UNSET,) -> dict[str, Any]:
    """Compare ordinary float32 and diagnostic float64 v2 reductions.

    Parameters
    ----------
    float32_steps, float64_steps : dict
        Production float32 and reference float64 checkpoint arrays from separate runs.
    tolerances : None, dict
        Tolerances keyed by step name or "step::field".
    default_tolerance : None, ArrayTolerance
        Tolerance for fields without an override.
    float32_winner : object
        Optional optimizer settings selected by the float32 run.
    float64_winner : object
        Optional optimizer settings selected by the float64 run.

    Returns
    -------
    report : dict
        Array agreement and optional optimizer-winner agreement.
    """
    report = compare_named_steps(float64_steps, float32_steps, tolerances=tolerances,
        default_tolerance=default_tolerance, kind='precision-float32-vs-float64',
        parity_scope='same-input_separate_precision_runs')
    have32 = float32_winner is not _WINNER_UNSET
    have64 = float64_winner is not _WINNER_UNSET
    winner_checked = have32 and have64
    winner_complete = have32 == have64
    winner_match = (_winner_equal(float32_winner, float64_winner) if winner_checked else None)
    arrays_passed = bool(report['passed'])
    passed = arrays_passed and winner_complete and (winner_match is not False)
    report.update({'status': 'passed' if passed else 'failed', 'passed': passed,
        'parity_claimed': False, 'precision_agreement_claimed': passed, 'precision': {
        'production_dtype': 'float32', 'validation_dtype': 'float64',
        'arrays_passed': arrays_passed, 'winner_check_supplied': winner_checked,
        'winner_inputs_complete': winner_complete, 'winner_match': winner_match,
        'float32_winner': (_json_value(float32_winner) if have32 else None),
        'float64_winner': (_json_value(float64_winner) if have64 else None),
        'recommended_named_fields': ['RampFitStep::data', 'RampFitStep::err', 'Cost::value'],
        'execution_note': ('Inputs must come from separate processes; this helper does '
        'not toggle JAX x64 mode.')}})
    report['summary']['winner_checked'] = winner_checked
    report['summary']['winner_match'] = winner_match
    return report


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write validation results as portable, human-readable JSON.

    Parameters
    ----------
    report : dict
        Validation results to save.
    path : str, Path
        Path to the report or checkpoint file.

    Returns
    -------
    destination : Path
        Path to the saved report.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as handle:
        json.dump(dict(report), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    return destination


def save_checkpoint_npz(checkpoints: Mapping[str, Any], path: str | Path,
        *, compressed: bool = False) -> Path:
    """Save named calibration checkpoints for a later v1/v2 comparison.

    Parameters
    ----------
    checkpoints : dict
        Checkpoint arrays grouped by step name.
    path : str, Path
        Path to the report or checkpoint file.
    compressed : bool
        If True, compress the checkpoint archive.

    Returns
    -------
    destination : Path
        Path to the saved checkpoint archive.
    """
    destination = Path(path)
    if destination.suffix.lower() == '.npz':
        destination = destination.with_suffix('.npz')
    else:
        destination = Path(str(destination) + '.npz')
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, np.ndarray] = {}
    for step, payload in checkpoints.items():
        if isinstance(payload, Mapping):
            for field, value in _array_leaves(payload, ''):
                array = np.asarray(value)
                if array.dtype.kind == 'O':
                    raise TypeError(f'checkpoint field {field!r} has object dtype and cannot '
                        'be loaded safely with allow_pickle=False')
                bundle[f'{step}::{field}'] = array
        else:
            array = np.asarray(payload)
            if array.dtype.kind == 'O':
                raise TypeError(f'checkpoint step {step!r} has object dtype')
            bundle[str(step)] = array
    writer = np.savez_compressed if compressed else np.savez
    writer(destination, **bundle)
    return destination


def load_checkpoint_npz(path: str | Path, *,
        steps: Iterable[str] | str | None = None) -> dict[str, Any]:
    """Load selected steps without materializing unrelated checkpoint arrays.

    Parameters
    ----------
    path : str, Path
        Path to the report or checkpoint file.
    steps : None, str, list[str]
        Step names to include. If None, include all available steps.

    Returns
    -------
    checkpoints : dict
        Selected checkpoint arrays grouped by step name.
    """
    result: dict[str, Any] = {}
    selected = _names(steps)
    with np.load(path, allow_pickle=False) as bundle:
        for key in bundle.files:
            if selected is not None and key.split('::', 1)[0] not in selected:
                continue
            if '::' in key:
                step, field = key.split('::', 1)
                existing = result.setdefault(step, {})
                if not isinstance(existing, dict):
                    raise ValueError(f'{step!r} has both scalar and field keys')
                existing[field] = np.asarray(bundle[key])
            else:
                if key in result:
                    raise ValueError(f'duplicate checkpoint step {key!r}')
                result[key] = np.asarray(bundle[key])
    return result


@dataclasses.dataclass(frozen=True)
class V1OutputLocations:
    """Store stage and log locations for an existing v1 reduction.

    Parameters
    ----------
    stage_root : Path
        Directory containing the Stage1, Stage2, and Stage3 products.
    log_directory : Path
        Directory containing the Cost and Scatter logs.
    log_name : None, str
        Name used in the optimizer log filenames.
    config : None, dict
        Configuration used for cost and winner reconstruction.
    """

    stage_root: Path
    log_directory: Path
    log_name: str | None = None
    config: Mapping[str, Any] | None = None


def resolve_v1_output_locations(*, config_path: str | Path | None = None,
        outputs_directory: str | Path | None = None,
        log_name: str | None = None) -> V1OutputLocations:
    """Find products from an already completed v1 reduction.

    Parameters
    ----------
    config_path : None, str, Path
        Path to the v1 YAML configuration.
    outputs_directory : None, str, Path
        Existing v1 output directory, supplied instead of config_path.
    log_name : None, str
        Name used in the Cost and Scatter log filenames.

    Returns
    -------
    locations : V1OutputLocations
        Resolved stage and optimizer-log locations.
    """
    if (config_path is None) == (outputs_directory is None):
        raise ValueError('provide exactly one of config_path or outputs_directory')

    if outputs_directory is not None:
        root = Path(outputs_directory).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        locations = V1OutputLocations(root.resolve(), (root / 'Files').resolve(), log_name)
    else:
        import yaml
        config = Path(config_path).expanduser()
        if not config.is_file():
            raise FileNotFoundError(f'v1 config file does not exist: {config}')
        with config.open('r', encoding='utf-8') as handle:
            options = yaml.safe_load(handle)
        if not isinstance(options, Mapping):
            raise ValueError(f'v1 config must contain a YAML mapping: {config}')
        configured = options.get('pipeline_outputs_directory', 'pipeline_outputs_directory')
        if not isinstance(configured, (str, Path)) or not str(configured):
            raise ValueError('pipeline_outputs_directory must be a non-empty path')
        # Preserve trailing slashes until after output-tag concatenation.
        base = v1_output_root(configured, cwd=Path.cwd())
        tag = options.get('output_tag')
        if tag not in (None, '') and not isinstance(tag, (str, int, float)):
            raise ValueError('output_tag must be scalar when packing v1 outputs')
        stage_root = v1_output_root(configured, tag, cwd=Path.cwd())
        selected_log = (log_name if log_name is not None
            else options.get('name_tag', 'default_run'))
        locations = V1OutputLocations(stage_root.resolve(), (base / 'Files').resolve(),
            None if selected_log is None else str(selected_log), dict(options))

    if not locations.stage_root.is_dir():
        raise FileNotFoundError(f'v1 pipeline output directory does not exist: '
            f'{locations.stage_root}')
    return locations


def _natural_key(path: Path) -> tuple[Any, ...]:
    """Return a filename sort key with numeric tokens ordered numerically."""
    return tuple(int(token) if token.isdigit() else token.lower()
        for token in re.split(r'(\d+)', path.name))


def _v1_series_key(path: Path, suffix: str) -> str:
    """Return the exposure identifier with the segment number removed."""
    name = path.name[:-len(suffix)]
    # Remove only the segment number from the exposure identifier.
    return re.sub(r'(?i)(?<![a-z0-9])seg\d+(?![a-z0-9])', 'seg#', name)


def _primary_segment_metadata(path: Path) -> tuple[int | None, int | None, int | None]:
    """Read integration bounds and segment number from the primary FITS header."""
    from astropy.io import fits
    header = fits.getheader(path, 0)
    start = header.get('INTSTART')
    end = header.get('INTEND')
    segment = header.get('EXSEGNUM')
    return (None if start is None else int(start), None if end is None else int(end),
        None if segment is None else int(segment))


def _ordered_v1_segments(paths: Iterable[Path]) -> tuple[list[Path], list[tuple[int | None,
        int | None, int | None]]]:
    """Order segments by INTSTART, EXSEGNUM, or filename."""
    files = list(paths)
    metadata = [_primary_segment_metadata(path) for path in files]
    starts = [item[0] for item in metadata]
    segments = [item[2] for item in metadata]
    if any(value is not None for value in starts):
        if not all(value is not None for value in starts):
            raise ValueError('only some v1 segments define INTSTART; refusing mixed '
                'header/filename ordering')
        if len(set(starts)) != len(starts):
            raise ValueError('duplicate INTSTART values in v1 segments')
        order = sorted(range(len(files)), key=lambda index:
            (starts[index], _natural_key(files[index])))
    elif all(value is not None for value in segments):
        if len(set(segments)) != len(segments):
            raise ValueError('duplicate EXSEGNUM values in v1 segments')
        order = sorted(range(len(files)), key=lambda index:
            (segments[index], _natural_key(files[index])))
    else:
        order = sorted(range(len(files)), key=lambda index: _natural_key(files[index]))
    return ([files[index] for index in order], [metadata[index] for index in order])


def _v1_segment_signature(paths: Iterable[Path]) -> tuple[tuple[str, int], ...]:
    """Return ordered identifiers for comparing segment sets across steps."""
    files, metadata = _ordered_v1_segments(paths)
    signature = []
    for index, (path, (start, _, segment)) in enumerate(zip(files, metadata)):
        if start is not None:
            signature.append(('INTSTART', start))
        elif segment is not None:
            signature.append(('EXSEGNUM', segment))
        else:
            match = re.search(r'(?i)(?<![a-z0-9])seg(\d+)(?![a-z0-9])', path.name)
            signature.append(('filename-segment', int(match.group(1)))
                if match else ('ordered-index', index))
    return tuple(signature)


def _read_v1_fits_fields(path: Path) -> dict[str, np.ndarray]:
    """Read science, error, and DQ extensions from a v1 FITS product."""
    from astropy.io import fits
    result: dict[str, np.ndarray] = {}
    with fits.open(path, memmap=False) as hdus:
        for hdu in hdus[1:]:
            extension = str(hdu.header.get('EXTNAME', '') or '').upper()
            field = _V1_EXTENSION_FIELDS.get(extension)
            if field is None or hdu.data is None:
                continue
            if field in result:
                raise ValueError(f'{path} contains duplicate {extension} extensions')
            result[field] = np.array(hdu.data, copy=True)
    if 'data' not in result:
        raise ValueError(f'{path} has no SCI extension')
    return result


def _combine_v1_fits_step(step: str, paths: Iterable[Path]) -> dict[str, np.ndarray]:
    """Concatenate integration arrays and check shared detector planes across segments."""
    files, metadata = _ordered_v1_segments(paths)
    fields_by_file = [_read_v1_fits_fields(path) for path in files]
    expected = set(fields_by_file[0])
    for path, fields in zip(files[1:], fields_by_file[1:]):
        if set(fields) != expected:
            raise ValueError(f'{step} extension mismatch in {path}: '
                f'{sorted(fields)} != {sorted(expected)}')
    starts = [item[0] for item in metadata]
    if starts and starts[0] is not None:
        previous_end = None
        for path, info, fields in zip(files, metadata, fields_by_file):
            start, end, _ = info
            data = fields['data']
            if data.ndim < 3:
                raise ValueError(f'{step} SCI in {path} is {data.ndim}D; cannot validate '
                    'INTSTART against an integration axis')
            inferred_end = start + data.shape[0] - 1
            if end is not None and end != inferred_end:
                raise ValueError(f'{path} INTSTART/INTEND describe {end-start+1} '
                    f'integrations but SCI contains {data.shape[0]}')
            end = inferred_end
            if previous_end is not None and start <= previous_end:
                raise ValueError(f'overlapping or out-of-order integration ranges near '
                    f'{path}: INTSTART={start}, previous INTEND={previous_end}')
            previous_end = end
    combined: dict[str, np.ndarray] = {}
    for field in sorted(expected):
        arrays = [fields[field] for fields in fields_by_file]
        if field == 'pixeldq':
            if any(array.ndim != 2 for array in arrays):
                raise ValueError(f'{step} PIXELDQ must be a detector plane')
            if any(not np.array_equal(arrays[0], array) for array in arrays[1:]):
                raise ValueError(f'{step} PIXELDQ differs between segments; refusing to '
                    'discard segment-specific flags')
            combined[field] = arrays[0]
            continue
        if arrays[0].ndim >= 3:
            trailing = arrays[0].shape[1:]
            if any(array.ndim != arrays[0].ndim or
                    array.shape[1:] != trailing for array in arrays[1:]):
                raise ValueError(f'{step} {field} shapes cannot be concatenated along '
                    'the integration axis')
            combined[field] = np.concatenate(arrays, axis=0)
        elif len(arrays) == 1:
            combined[field] = arrays[0]
        elif all(np.array_equal(arrays[0], array) for array in arrays[1:]):
            # Collapse identical detector planes across segments.
            combined[field] = arrays[0]
        else:
            raise ValueError(f'{step} {field} is {arrays[0].ndim}D with multiple '
                'non-identical files; no integration axis can be inferred')
    return combined


def _select_v1_spectrum(output_root: Path, spectrum_path: str | Path | None) -> Path | None:
    """Select an explicit or unique Stage 3 spectrum."""
    if spectrum_path is not None:
        selected = Path(spectrum_path).expanduser()
        if not selected.is_absolute():
            selected = Path.cwd() / selected
        if not selected.is_file():
            raise FileNotFoundError(f'v1 Stage3 spectrum does not exist: {selected}')
        return selected
    stage3 = output_root / 'Stage3'
    if not stage3.is_dir():
        return None
    candidates = sorted(stage3.glob('*_spectra_fullres.fits'), key=_natural_key)
    if len(candidates) > 1:
        raise ValueError(f'multiple v1 Stage3 spectra found in {stage3}; select one '
            'with --spectra or disable spectrum packing with --no-spectra: '
            + ', '.join(path.name for path in candidates))
    return candidates[0] if candidates else None


def _read_v1_spectrum(path: Path) -> dict[str, Any]:
    """Normalize one v1 ``save_extracted_spectra`` product to v2 form."""
    from astropy.io import fits
    extensions: dict[str, np.ndarray] = {}
    with fits.open(path, memmap=False) as hdus:
        for hdu in hdus[1:]:
            name = str(hdu.header.get('EXTNAME', '') or '').upper()
            if not name or hdu.data is None:
                continue
            if name in extensions:
                raise ValueError(f'{path} contains duplicate {name} extensions')
            extensions[name] = np.array(hdu.data, copy=True)

    if 'WAVE O1' in extensions:
        layout = [(1, ('WAVE O1', 'FLUX O1', 'FLUX ERR O1')),
            (2, ('WAVE O2', 'FLUX O2', 'FLUX ERR O2'))]
        description = 'v1 SOSS'
    elif 'WAVE' in extensions:
        layout = [(1, ('WAVE', 'FLUX', 'FLUX ERR'))]
        description = 'v1 single-order (NIRSpec/MIRI)'
    else:
        raise ValueError(f'{path} has neither SOSS per-order nor bare single-order '
            f'spectrum extensions (found: {sorted(extensions)})')
    products = {}
    for order, names in layout:
        missing = [name for name in names if name not in extensions]
        if missing:
            raise ValueError(f'{path} is not a complete {description} full-resolution '
                f'spectrum; missing extensions: {missing}')
        wave, flux, ferr = (extensions[name] for name in names)
        if wave.ndim == 2:
            # Check that each integration uses the same wavelength grid.
            if not np.allclose(wave, wave[0], equal_nan=True):
                raise ValueError(f'{path} {names[0]} varies between integrations')
            wave = wave[0]
        if wave.ndim != 1:
            raise ValueError(f'{path} {names[0]} must be 1D, got {wave.shape}')
        if flux.ndim != 2:
            raise ValueError(f'{path} {names[1]} must be 2D, got {flux.shape}')
        if ferr.shape != flux.shape:
            raise ValueError(f'{path} {names[2]} shape {ferr.shape} does not match '
                f'{names[1]} {flux.shape}')
        if flux.shape[-1] != wave.shape[0]:
            raise ValueError(f'{path} order {order} wavelength length {wave.shape[0]} '
                f'does not match flux axis {flux.shape[-1]}')
        columns = np.flatnonzero(np.isfinite(wave))
        columns = columns[np.argsort(wave[columns], kind='stable')]
        products[order] = {'wave': wave[columns], 'flux': flux[..., columns],
            'ferr': ferr[..., columns]}
    return {'aux': {'spectral_products': products}}


def _read_single_trace_spectrum(path: str | Path, *, order=None) -> dict[str, np.ndarray]:
    """Read one trace and its time axis from a Stage-3 product."""
    from astropy.io import fits
    path = Path(path)
    with fits.open(path, memmap=False) as hdus:
        values = {}
        for name in ('WAVE', 'WAVE ERR', 'FLUX', 'FLUX ERR', 'TIME'):
            extension = (f'{name} O{order}' if order is not None and name != 'TIME' else name)
            try:
                values[name] = np.array(hdus[extension].data, copy=True)
            except KeyError:
                if name in ('WAVE', 'FLUX', 'TIME'):
                    raise ValueError(f'{path} is missing required {extension} extension') from None
        wave = np.asarray(values['WAVE'], dtype=float)
        if wave.ndim == 2:
            if not np.allclose(wave, wave[0], equal_nan=True):
                raise ValueError(f'{path} WAVE varies by integration')
            wave = wave[0]
        flux = np.asarray(values['FLUX'], dtype=float)
        time = np.asarray(values['TIME'], dtype=float).reshape(-1)
    return {'wave': wave, 'flux': flux, 'time': time}


def _summary(values, *, absolute=False):
    """Summarize finite values by count, median, 95th percentile, and maximum."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if absolute:
        values = np.abs(values)
    if not values.size:
        return {'count': 0, 'median': None, 'p95': None, 'maximum': None}
    return {'count': int(values.size), 'median': float(np.median(values)),
        'p95': float(np.percentile(values, 95)), 'maximum': float(np.max(values))}


def _baseline_normalize(flux, baseline_ints):
    """Normalize each flux column by its baseline median."""
    flux = np.asarray(flux, dtype=float)
    if baseline_ints is None:
        sample = flux
    else:
        baseline = np.atleast_1d(np.asarray(baseline_ints, dtype=int))
        if baseline.size == 1:
            sample = flux[:int(baseline[0])]
        elif baseline.size == 2:
            sample = np.concatenate((flux[:int(baseline[0])], flux[int(baseline[1]):]), axis=0)
        else:
            raise ValueError('baseline_ints must have length 1 or 2')
    scale = np.nanmedian(sample, axis=0)
    return np.divide(flux, scale[None], out=np.full_like(flux, np.nan),
        where=np.isfinite(scale[None]) & (scale[None] != 0))


def _column_ptp_scatter(normalized, baseline_ints):
    """Calculate per-column point-to-point scatter over the baseline integrations."""
    d2 = (0.5 * (normalized[:-2] + normalized[2:]) - normalized[1:-1])
    values = np.abs(d2)
    if baseline_ints is None:
        return np.nanmedian(values, axis=0)
    baseline = np.atleast_1d(np.asarray(baseline_ints, dtype=int))
    if baseline.size == 1:
        return np.nanmedian(values[:int(baseline[0])], axis=0)
    if baseline.size == 2:
        low = np.nanmedian(values[:int(baseline[0])], axis=0)
        high = np.nanmedian(values[int(baseline[1]):], axis=0)
        return 0.5 * (low + high)
    raise ValueError('baseline_ints must have length 1 or 2')


def compare_single_trace_spectra(reference_path: str | Path, candidate_path: str | Path, *,
        baseline_ints=None, wave_range=None, max_flux_p95_ppm=None,
        max_scatter_ratio_deviation=None) -> dict[str, Any]:
    """Compare final NIRSpec/MIRI spectra with scale-robust science metrics.

    Parameters
    ----------
    reference_path, candidate_path : str, Path
        Reference and candidate spectrum FITS files.
    baseline_ints : None, array-like(int)
        One or two integration bounds for baseline statistics.
    wave_range : None, array-like(float)
        Inclusive wavelength bounds in microns.
    max_flux_p95_ppm : None, float
        Maximum 95th percentile of absolute normalized flux residuals in ppm.
    max_scatter_ratio_deviation : None, float
        Maximum deviation of the median scatter ratio from one.

    Returns
    -------
    report : dict
        Wavelength, time, normalized flux, and scatter metrics.
    """
    return _compare_spectrum_arrays(_read_single_trace_spectrum(reference_path),
        _read_single_trace_spectrum(candidate_path),
        reference_path=reference_path, candidate_path=candidate_path,
        baseline_ints=baseline_ints, wave_range=wave_range, max_flux_p95_ppm=max_flux_p95_ppm,
        max_scatter_ratio_deviation=max_scatter_ratio_deviation)


def _compare_spectrum_arrays(reference, candidate, *, reference_path, candidate_path,
        baseline_ints=None, wave_range=None, max_flux_p95_ppm=None,
        max_scatter_ratio_deviation=None, max_flux_noise_fraction=None):
    """Compare wavelength support, times, normalized flux, and scatter."""
    for limit in (max_flux_p95_ppm, max_scatter_ratio_deviation, max_flux_noise_fraction):
        if limit is not None and (not np.isfinite(limit) or limit < 0):
            raise ValueError('comparison limits must be finite and nonnegative')
    if wave_range is not None:
        wave_range = np.atleast_1d(np.asarray(wave_range, dtype=float))
        if (wave_range.size != 2 or not np.isfinite(wave_range).all() or
                wave_range[0] > wave_range[1]):
            raise ValueError('wave_range must be two finite ascending micron bounds')
    shape_pass = (reference['wave'].shape == candidate['wave'].shape and
        reference['flux'].shape == candidate['flux'].shape and
        reference['time'].shape == candidate['time'].shape and reference['flux'].ndim == 2 and
        reference['flux'].shape[-1] == reference['wave'].size and
        reference['flux'].shape[0] == reference['time'].size)
    checks: dict[str, Any] = {'shapes': {'passed': bool(shape_pass),
        'reference': {key: list(value.shape) for key, value in reference.items()},
        'candidate': {key: list(value.shape) for key, value in candidate.items()}}}
    if not shape_pass:
        report = {'schema_version': REPORT_SCHEMA_VERSION, 'created_at_utc': _utc_now(),
            'kind': 'single-trace-spectrum', 'status': 'failed', 'passed': False,
            'parity_claimed': False, 'baseline_ints': (None if baseline_ints is None else
            np.atleast_1d(baseline_ints).astype(int).tolist()),
            'wave_range_micron': (None if wave_range is None else wave_range.tolist()),
            'checks': checks, 'metrics': {}}
        return report
    ref_wave_finite = np.isfinite(reference['wave'])
    cand_wave_finite = np.isfinite(candidate['wave'])
    wave_mismatch = ref_wave_finite ^ cand_wave_finite
    checks['wavelength_finite_support'] = {'passed': not bool(wave_mismatch.any()),
        'mismatch_count': int(wave_mismatch.sum()),
        'mismatch_fraction': float(wave_mismatch.mean())}
    ref_nonzero = np.any(np.isfinite(reference['flux']) & (reference['flux'] != 0), axis=0)
    cand_nonzero = np.any(np.isfinite(candidate['flux']) & (candidate['flux'] != 0), axis=0)
    flux_mismatch = ref_nonzero ^ cand_nonzero
    checks['nonzero_flux_column_support'] = {'passed': not bool(flux_mismatch.any()),
        'mismatch_count': int(flux_mismatch.sum()),
        'mismatch_fraction': float(flux_mismatch.mean())}
    times_equal = np.array_equal(reference['time'], candidate['time'])
    checks['integration_times'] = {'passed': bool(times_equal), 'bitwise_equal': bool(times_equal),
        'maximum_absolute_difference_days': float(np.max(np.abs(
        reference['time'] - candidate['time']))) if reference['time'].size else 0.}
    common = (ref_wave_finite & cand_wave_finite & ref_nonzero & cand_nonzero)
    if wave_range is not None:
        common &= ((reference['wave'] >= wave_range[0]) & (reference['wave'] <= wave_range[1]))
    ref_flux = reference['flux'][:, common]
    cand_flux = candidate['flux'][:, common]
    ref_norm = _baseline_normalize(ref_flux, baseline_ints)
    cand_norm = _baseline_normalize(cand_flux, baseline_ints)
    checks['science_support'] = {'passed': bool(common.any() and ref_norm.shape[0] >= 3 and
        np.isfinite(ref_norm).all() and np.isfinite(cand_norm).all())}
    residual_ppm = (cand_norm - ref_norm) * 1e6
    ref_scatter = _column_ptp_scatter(ref_norm, baseline_ints)
    cand_scatter = _column_ptp_scatter(cand_norm, baseline_ints)
    ratio = np.divide(cand_scatter, ref_scatter, out=np.full_like(cand_scatter, np.nan),
        where=np.isfinite(ref_scatter) & (ref_scatter != 0))
    finite_ratio = ratio[np.isfinite(ratio)]
    scatter_metrics = {'reference': _summary(ref_scatter), 'candidate': _summary(cand_scatter),
        'ratio_candidate_over_reference': _summary(ratio), 'fraction_candidate_lower_or_equal': (
        float(np.mean(finite_ratio <= 1.)) if finite_ratio.size else None)}
    wavelength_difference = (candidate['wave'][common] - reference['wave'][common])
    metrics = {'common_science_column_count': int(common.sum()),
        'wavelength_absolute_difference_micron': _summary(wavelength_difference, absolute=True),
        'normalized_flux_residual_ppm': {'signed': _summary(residual_ppm),
        'absolute': _summary(residual_ppm, absolute=True)},
        'per_column_ptp_scatter': scatter_metrics}
    noise_fraction = np.divide(np.abs(cand_norm - ref_norm), ref_scatter[None],
        out=np.full_like(cand_norm, np.nan),
        where=np.isfinite(ref_scatter[None]) & (ref_scatter[None] > 0))
    metrics['flux_residual_reference_noise_fraction'] = _summary(noise_fraction)
    noise_p95 = metrics['flux_residual_reference_noise_fraction']['p95']
    checks['flux_noise_fraction_p95'] = {'passed': bool(max_flux_noise_fraction is None or
        (noise_p95 is not None and np.isfinite(noise_fraction).all() and
        noise_p95 <= max_flux_noise_fraction)), 'observed_p95_fraction': noise_p95,
        'maximum_p95_fraction': max_flux_noise_fraction}
    flux_p95 = metrics['normalized_flux_residual_ppm']['absolute']['p95']
    ratio_median = scatter_metrics['ratio_candidate_over_reference']['median']
    flux_gate = (max_flux_p95_ppm is None or
        (flux_p95 is not None and flux_p95 <= max_flux_p95_ppm))
    scatter_gate = (max_scatter_ratio_deviation is None or (ratio_median is not None and
        abs(ratio_median - 1.) <= max_scatter_ratio_deviation))
    checks['normalized_flux_p95'] = {'passed': bool(flux_gate), 'observed_p95_ppm': flux_p95,
        'maximum_p95_ppm': max_flux_p95_ppm}
    checks['median_scatter_ratio'] = {'passed': bool(scatter_gate),
        'observed_median_ratio': ratio_median,
        'maximum_deviation_from_one': max_scatter_ratio_deviation}
    passed = all(check['passed'] for check in checks.values())
    parity_claimed = ((max_flux_p95_ppm is not None or max_flux_noise_fraction is not None) and
        max_scatter_ratio_deviation is not None and passed)
    return {'schema_version': REPORT_SCHEMA_VERSION, 'created_at_utc': _utc_now(),
        'kind': 'single-trace-spectrum', 'status': 'passed' if passed else 'failed',
        'passed': bool(passed), 'parity_claimed': bool(parity_claimed),
        'reference': str(reference_path), 'candidate': str(candidate_path),
        'baseline_ints': (None if baseline_ints is None else
        np.atleast_1d(baseline_ints).astype(int).tolist()), 'wave_range_micron': (
        None if wave_range is None else wave_range.tolist()), 'checks': checks,
        'metrics': metrics}


def compare_soss_spectra(reference_path: str | Path, candidate_path: str | Path, *,
        baseline_ints=None, o1_wave_range=None, o2_wave_range=None,
        o1_max_flux_p95_ppm=None, o2_max_flux_p95_ppm=None,
        o1_max_flux_noise_fraction=None, o2_max_flux_noise_fraction=None,
        o1_max_scatter_ratio_deviation=None, o2_max_scatter_ratio_deviation=None) -> dict[str, Any]:
    """Compare both SOSS orders after removing nonfinite wavelength padding.

    Finite wavelengths must match exactly after sorting. Each order needs explicit flux and
    scatter limits to claim parity.

    Parameters
    ----------
    reference_path, candidate_path : str, Path
        Reference and candidate spectrum FITS files.
    baseline_ints : None, array-like(int)
        One or two integration bounds for baseline statistics.
    o1_wave_range : None, array-like(float)
        Inclusive wavelength bounds for order 1 in microns.
    o2_wave_range : None, array-like(float)
        Inclusive wavelength bounds for order 2 in microns.
    o1_max_flux_p95_ppm : None, float
        Maximum 95th percentile of absolute flux residuals for order 1 in ppm.
    o2_max_flux_p95_ppm : None, float
        Maximum 95th percentile of absolute flux residuals for order 2 in ppm.
    o1_max_flux_noise_fraction : None, float
        Maximum 95th percentile of flux residuals relative to reference noise for order 1.
    o2_max_flux_noise_fraction : None, float
        Maximum 95th percentile of flux residuals relative to reference noise for order 2.
    o1_max_scatter_ratio_deviation : None, float
        Maximum deviation of the median scatter ratio from one for order 1.
    o2_max_scatter_ratio_deviation : None, float
        Maximum deviation of the median scatter ratio from one for order 2.

    Returns
    -------
    report : dict
        Comparison metrics and wavelength alignment for both SOSS orders.
    """
    orders = {}
    for order, band, flux_limit, noise_limit, scatter_limit in (
        (1, o1_wave_range, o1_max_flux_p95_ppm,
        o1_max_flux_noise_fraction, o1_max_scatter_ratio_deviation),
        (2, o2_wave_range, o2_max_flux_p95_ppm,
        o2_max_flux_noise_fraction, o2_max_scatter_ratio_deviation)):
        traces = []
        padding = []
        for path in (reference_path, candidate_path):
            trace = _read_single_trace_spectrum(path, order=order)
            wave, flux = trace['wave'], trace['flux']
            if (wave.ndim != 1 or flux.ndim != 2 or flux.shape != (trace['time'].size, wave.size)):
                raise ValueError(f'{path} order {order} has invalid trace shapes')
            columns = np.flatnonzero(np.isfinite(wave))
            padding.append(int(wave.size - columns.size))
            columns = columns[np.argsort(wave[columns], kind='stable')]
            traces.append(dict(wave=wave[columns], flux=flux[:, columns], time=trace['time']))
        report = _compare_spectrum_arrays(*traces, reference_path=reference_path,
            candidate_path=candidate_path, baseline_ints=baseline_ints,
            wave_range=band, max_flux_p95_ppm=flux_limit, max_flux_noise_fraction=noise_limit,
            max_scatter_ratio_deviation=scatter_limit)
        aligned = np.array_equal(traces[0]['wave'], traces[1]['wave'])
        report['checks']['finite_wavelength_alignment'] = {'passed': bool(aligned),
            'reference_dropped_padding_columns': padding[0],
            'candidate_dropped_padding_columns': padding[1]}
        report['kind'] = 'soss-order-spectrum'
        report['passed'] = bool(report['passed'] and aligned)
        report['parity_claimed'] &= report['passed']
        report['status'] = 'passed' if report['passed'] else 'failed'
        orders[f'O{order}'] = report
    passed = all(order['passed'] for order in orders.values())
    return {'schema_version': REPORT_SCHEMA_VERSION,
        'created_at_utc': _utc_now(), 'kind': 'soss-spectrum',
        'status': 'passed' if passed else 'failed', 'passed': passed,
        'parity_claimed': all(order['parity_claimed'] for order in orders.values()),
        'reference': str(reference_path), 'candidate': str(candidate_path),
        'baseline_ints': (None if baseline_ints is None else
        np.atleast_1d(baseline_ints).astype(int).tolist()),
        'checks': {'integration_times': orders['O1']['checks'].get(
        'integration_times', {'passed': False, 'bitwise_equal': False})}, 'orders': orders}


def _numeric_column(values: list[str]) -> np.ndarray:
    """Parse a log column as floats when every nonempty value is numeric."""
    converted = []
    for value in values:
        text = value.strip()
        if not text:
            converted.append(np.nan)
            continue
        try:
            converted.append(float(text))
        except ValueError:
            return np.asarray(values, dtype=np.str_)
    return np.asarray(converted, dtype=np.float64)


def _read_v1_cost_log(path: Path) -> dict[str, Any]:
    """Read a v1 mixed-phase trial log without assigning it a final Cost."""
    with path.open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.reader(handle, delimiter='\t'))
    if not rows or not rows[0]:
        raise ValueError(f'empty v1 cost log: {path}')
    header = [name.strip() for name in rows[0]]
    if any(not name for name in header) or len(set(header)) != len(header):
        raise ValueError(f'invalid or duplicate columns in v1 cost log: {path}')
    if any(len(row) != len(header) for row in rows[1:]):
        raise ValueError(f'ragged v1 cost log: {path}')
    if not rows[1:]:
        raise ValueError(f'v1 cost log contains no trials: {path}')
    columns = {name: _numeric_column([row[index] for row in rows[1:]])
        for index, name in enumerate(header)}
    cost = columns.get('cost')
    if cost is None or cost.dtype.kind not in 'fiu':
        raise ValueError(f'v1 cost log has no numeric cost column: {path}')
    return {'header': tuple(header), 'columns': columns, 'trial_count': int(cost.shape[0])}


_V1_SWEEP_ORDER = ('soss_inner_mask_width', 'soss_outer_mask_width', 'nirspec_mask_width',
    'time_jump_threshold', 'time_window', 'miri_trace_width',
    'miri_background_width', 'space_outlier_threshold',
    'time_outlier_threshold', 'box_size', 'window_size', 'extract_width')
_V1_OPTIMIZER_CONTROL_FLAGS = {'optimize_extract_width_only', 'optimize_from_pca_only'}


def _normalize_pack_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any] | None:
    """Load a v1 configuration and convert literal None strings."""
    if config is None:
        return None
    if isinstance(config, Mapping):
        result = dict(config)
    else:
        import yaml
        path = Path(config).expanduser()
        with path.open('r', encoding='utf-8') as handle:
            result = yaml.safe_load(handle)
        if not isinstance(result, Mapping):
            raise ValueError(f'v1 config must contain a YAML mapping: {path}')
        result = dict(result)
    for key, value in tuple(result.items()):
        if value == 'None':
            result[key] = None
    return result


def _v1_sweep_blocks(config: Mapping[str, Any]
        ) -> tuple[list[tuple[str, list[Any]]], list[str]] | None:
    """Return strict normal-mode v1 row blocks and expected log columns."""
    if config.get('optimize_extract_width_only', False) or config.get(
            'from_pca_only', config.get('optimize_from_pca_only', False)):
        return None
    known = set(_V1_SWEEP_ORDER)
    columns = []
    for key, enabled in config.items():
        if not key.startswith('optimize_') or key in _V1_OPTIMIZER_CONTROL_FLAGS:
            continue
        parameter = key[len('optimize_'):]
        if not enabled:
            continue
        # Require a known sweep block for every enabled optimizer column.
        if parameter not in known or parameter not in config:
            return None
        candidates = config[parameter]
        if not isinstance(candidates, list) or not candidates:
            return None
        columns.append(parameter)
    blocks = []
    for parameter in _V1_SWEEP_ORDER:
        if not config.get(f'optimize_{parameter}', False):
            continue
        candidates = config.get(parameter)
        if not isinstance(candidates, list) or not candidates:
            return None
        blocks.append((parameter, list(candidates)))
    return blocks, columns


def _numeric_candidates_match(observed: np.ndarray, expected: list[Any]) -> bool:
    """Check that a numeric log column matches the configured candidates."""
    if observed.dtype.kind not in 'fiu' or len(observed) != len(expected):
        return False
    try:
        wanted = np.asarray(expected, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.array_equal(np.asarray(observed, dtype=np.float64), wanted, equal_nan=True))


def _unambiguous_logged_argmin(costs: np.ndarray) -> int | None:
    """Reproduce np.argmin only when 12-decimal logging preserves the winner."""
    costs = np.asarray(costs, dtype=np.float64)
    if not costs.size:
        return None
    index = int(np.argmin(costs))
    selected = costs[index]
    if np.isnan(selected) or np.isinf(selected):
        return index
    competitors = np.delete(costs, index)
    competitors = competitors[np.isfinite(competitors)]
    # Reject winners separated by no more than the log rounding precision.
    if competitors.size and np.any(np.abs(competitors - selected) <= 1e-12):
        return None
    return index


def _reconstruct_v1_winners(trials: Mapping[str, Any], config: Mapping[str, Any]
        ) -> dict[str, np.ndarray] | None:
    """Recover winners only when logged sweeps match the configured layout."""
    layout = _v1_sweep_blocks(config)
    if layout is None:
        return None
    blocks, expected_columns = layout
    if not blocks:
        return None
    header = tuple(trials['header'])
    if header != tuple([*expected_columns, 'duration_s', 'cost']):
        return None
    expected_rows = sum(len(candidates) for _, candidates in blocks)
    if int(trials['trial_count']) != expected_rows:
        return None
    columns = trials['columns']
    winners = {}
    offset = 0
    for parameter, candidates in blocks:
        end = offset + len(candidates)
        if not _numeric_candidates_match(columns[parameter][offset:end], candidates):
            return None
        winner_index = _unambiguous_logged_argmin(columns['cost'][offset:end])
        if winner_index is None:
            return None
        winners[f'winner.{parameter}'] = np.asarray(candidates[winner_index])
        offset = end
    return winners


def _v1_default_wave_range(config: Mapping[str, Any], w2: float):
    """Select the configured or default v1 cost wavelength range."""
    configured = config.get('wave_range')
    if configured not in (None, 'None', 'none', 'null', 'NULL', ''):
        return configured
    return None if w2 == 0 else [1.0, 2.0]


def _v1_numpy_final_cost(products: Mapping[int, Mapping[str, Any]], config: Mapping[str, Any]
        ) -> tuple[float, np.ndarray]:
    """Calculate the packed SOSS cost with NumPy as in v1 optimize.cost_function."""
    try:
        order1, order2 = products[1], products[2]
    except KeyError as error:
        raise ValueError('final v1 SOSS cost requires orders 1 and 2') from error
    wave1 = np.asarray(order1['wave'], dtype=float)
    wave2 = np.asarray(order2['wave'], dtype=float)
    flux1 = np.asarray(order1['flux'], dtype=float)
    flux2 = np.asarray(order2['flux'], dtype=float)
    i2 = np.where(wave2 <= 0.85)[0]
    i1 = np.where(wave1 > 0.85)[0]
    if i2.size == 0 or i1.size == 0:
        raise ValueError(f'cutoff produces empty packed SOSS segment: O2={i2.size}, '
            f'O1={i1.size}')
    wave = np.concatenate([wave2[:i2[-1] + 1], wave1[i1[0]:]])
    flux = np.concatenate([flux2[:, :i2[-1] + 1], flux1[:, i1[0]:]], axis=1)
    order = np.argsort(wave)
    wave, flux = wave[order], flux[:, order]
    white = np.nansum(flux, axis=1)
    white = white[~np.isnan(white)]
    norm_white = white / np.median(white)
    d2_white = 0.5 * (norm_white[:-2] + norm_white[2:]) - norm_white[1:-1]
    ptp2_white = np.nanmedian(np.abs(d2_white))
    medians = np.nanmedian(flux, axis=0, keepdims=True)
    normalized = flux / medians
    d2_spec = (0.5 * (normalized[:-2] + normalized[2:]) - normalized[1:-1])
    baseline = np.atleast_1d(np.asarray(config.get('baseline_ints', [100, -100]), dtype=int))
    if len(baseline) == 1:
        scatter = np.nanmedian(np.abs(d2_spec[:int(baseline[0])]), axis=0)
    elif len(baseline) == 2:
        low = np.nanmedian(np.abs(d2_spec[:int(baseline[0])]), axis=0)
        high = np.nanmedian(np.abs(d2_spec[int(baseline[1]):]), axis=0)
        scatter = 0.5 * (low + high)
    else:
        raise ValueError('baseline_ints must have length 1 or 2')
    w1 = float(config.get('w1', 0.0))
    w2 = float(config.get('w2', 1.0))
    wave_range = _v1_default_wave_range(config, w2)
    if wave_range is None:
        ptp2_spec = np.nanmedian(scatter)
    elif isinstance(wave_range, (list, tuple)) and len(wave_range) == 2:
        finite = np.isfinite(wave)
        if not finite.any():
            raise ValueError('all packed v1 wavelengths are non-finite')
        lo = np.nanmin(wave[finite]) if wave_range[0] is None else wave_range[0]
        hi = np.nanmax(wave[finite]) if wave_range[1] is None else wave_range[1]
        dist_lo = np.abs(wave - lo)
        dist_hi = np.abs(wave - hi)
        dist_lo[~finite] = np.inf
        dist_hi[~finite] = np.inf
        a, b = sorted((int(np.argmin(dist_lo)), int(np.argmin(dist_hi))))
        selected = scatter[a:b + 1]
        if np.all(np.isnan(selected)):
            raise ValueError(f'no valid scatter in wave_range {wave_range}')
        ptp2_spec = np.nanmedian(selected)
    else:
        raise ValueError('wave_range must be None or a length-2 sequence')
    cost = 0.0
    if w1 != 0:
        cost += w1 * ptp2_white
    if w2 != 0:
        cost += w2 * ptp2_spec
    return float(cost), np.asarray(scatter, dtype=np.float64)


def _select_v1_log_pair(log_directory: Path, log_name: str | None
        ) -> tuple[Path, Path | None] | None:
    """Select the Cost log and its optional Scatter log."""
    if not log_directory.is_dir():
        return None
    if log_name is not None:
        cost = log_directory / f'Cost_{log_name}.txt'
        if not cost.is_file():
            alternatives = sorted(log_directory.glob('Cost_*.txt'), key=_natural_key)
            if alternatives:
                raise FileNotFoundError(f'selected v1 Cost log does not exist: {cost}; found: '
                    + ', '.join(path.name for path in alternatives))
            return None
    else:
        costs = sorted(log_directory.glob('Cost_*.txt'), key=_natural_key)
        if not costs:
            return None
        if len(costs) != 1:
            raise ValueError(f'multiple v1 Cost logs found in {log_directory}; select '
                'one with --log-name')
        cost = costs[0]
        log_name = cost.stem[len('Cost_'):]
    scatter = log_directory / f'Scatter_{log_name}.txt'
    return cost, scatter if scatter.is_file() else None


def pack_v1_outputs(output_root: str | Path, destination: str | Path, *,
        log_directory: str | Path | None = None,
        log_name: str | None = None, include_logs: bool = True,
        spectrum_path: str | Path | None = None,
        include_spectrum: bool = True, compressed: bool = False,
        config: Mapping[str, Any] | str | Path | None = None,
        steps: Iterable[str] | str | None = None) -> Path:
    """Pack existing v1 calibration products into a checkpoint NPZ archive.

    Final cost is recomputed from spectra when a configuration is supplied. Optimizer
    winners are included only when the logged sweep layout is unambiguous.

    Parameters
    ----------
    output_root : str, Path
        Existing v1 pipeline output directory.
    destination : str, Path
        Path to the checkpoint NPZ archive to write.
    log_directory : None, str, Path
        Directory containing optimizer logs. If None, use Files below output_root.
    log_name : None, str
        Name used in the Cost and Scatter log filenames.
    include_logs : bool
        If True, reconstruct optimizer winners from unambiguous logs.
    spectrum_path : None, str, Path
        Spectrum to include. If None, select a unique Stage 3 spectrum.
    include_spectrum : bool
        If True, include final extracted spectra.
    compressed : bool
        If True, compress the checkpoint archive.
    config : None, dict, str, Path
        Configuration used to reconstruct the final cost and optimizer winners.
    steps : None, str, list[str]
        Step names to include. If None, include all available steps.

    Returns
    -------
    destination : Path
        Path to the saved checkpoint archive.
    """
    root = Path(output_root).expanduser()
    selected = _names(steps)
    checkpoints: dict[str, Any] = {}
    series_keys: set[str] = set()
    segment_signature: tuple[tuple[str, int], ...] | None = None
    signature_step = None
    for stage, specs in V1_FITS_CHECKPOINTS.items():
        if selected is not None:
            specs = [(step, suffix) for step, suffix in specs if step in selected]
            if not specs:
                continue
        directory = root / stage
        if not directory.is_dir():
            raise FileNotFoundError(f'missing v1 {stage} directory: {directory}')
        discovered = 0
        for step, suffix in specs:
            paths = sorted(directory.glob(f'*{suffix}'), key=_natural_key)
            if not paths:
                continue
            discovered += len(paths)
            keys = {_v1_series_key(path, suffix) for path in paths}
            if len(keys) != 1:
                raise ValueError(f'ambiguous {step} artifacts in {directory}: ' f'{sorted(keys)}')
            series_keys.update(keys)
            this_signature = _v1_segment_signature(paths)
            if segment_signature is None:
                segment_signature = this_signature
                signature_step = step
            elif this_signature != segment_signature:
                raise ValueError(f'{step} segment set {this_signature} does not match '
                    f'{signature_step} {segment_signature}; one or more v1 '
                    'artifacts may be missing')
            checkpoints[step] = _combine_v1_fits_step(step, paths)
        if discovered == 0:
            known = ', '.join(suffix for _, suffix in specs)
            raise FileNotFoundError(f'no known v1 {stage} FITS artifacts in {directory}; '
                f'expected suffixes such as {known}')
    if len(series_keys) > 1 or (selected is None and not series_keys):
        raise ValueError('Stage1/Stage2 contain artifacts from multiple datasets: '
            f'{sorted(series_keys)}')
    normalized_config = _normalize_pack_config(config)
    packed_products = None
    if include_spectrum and (selected is None or
            any(step in selected for step in ('Extract', 'Cost'))):
        spectrum = _select_v1_spectrum(root, spectrum_path)
        if spectrum is not None:
            extract = _read_v1_spectrum(spectrum)
            checkpoints['Extract'] = extract
            packed_products = extract['aux']['spectral_products']

    if normalized_config is not None and packed_products is not None:
        value, scatter = _v1_numpy_final_cost(packed_products, normalized_config)
        checkpoints['Cost'] = {'value': np.asarray(value), 'scatter': np.asarray(scatter)}

    if include_logs and (selected is None or 'Optimizer' in selected):
        logs = (Path(log_directory).expanduser() if log_directory is not None else root / 'Files')
        pair = _select_v1_log_pair(logs, log_name)
        if pair is not None and normalized_config is not None:
            cost_path, _ = pair
            try:
                trials = _read_v1_cost_log(cost_path)
                winners = _reconstruct_v1_winners(trials, normalized_config)
            except (KeyError, TypeError, ValueError):
                winners = None
            if winners is not None:
                checkpoints['Optimizer'] = winners

    if selected is not None:
        missing = set(selected) - set(checkpoints)
        if missing:
            raise ValueError(f'requested v1 checkpoint steps missing: {sorted(missing)}')
        checkpoints = {name: checkpoints[name] for name in selected}
    return save_checkpoint_npz(checkpoints, destination, compressed=compressed)


def _selector_reaches(name: str, selectors: tuple[str, ...] | None) -> bool:
    """Check whether a field and any selected dotted prefix overlap."""
    return (selectors is None or any(name == selected or name.startswith(f'{selected}.') or
        selected.startswith(f'{name}.') for selected in selectors))


def _state_arrays(state: Any, selectors: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Collect selected science, error, DQ, and auxiliary arrays from a state."""
    cube = getattr(state, 'cube', state)
    result: dict[str, Any] = {}
    for name in ('data', 'err', 'dq', 'groupdq', 'pixeldq'):
        if not _selector_reaches(name, selectors):
            continue
        value = getattr(cube, name, None)
        if value is not None:
            result[name] = np.asarray(value)
    aux = getattr(state, 'aux', {})
    for name, value in aux.items():
        for field, leaf in _array_leaves(value, f'aux.{name}', selectors=selectors):
            result[field] = np.asarray(leaf)
    return result


def _names(value: Iterable[str] | str | None) -> tuple[str, ...] | None:
    """Convert one or more selection names to a tuple."""
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _selected_state_arrays(state: Any, *, step: str, fields: Iterable[str] | str |
        Mapping[str, Iterable[str] | str] | None) -> dict[str, Any]:
    """Collect state arrays using global or step-specific field selections."""
    if fields is None:
        requested = None
    elif isinstance(fields, Mapping):
        requested = _names(fields.get(step, fields.get('*')))
    else:
        requested = _names(fields)
    arrays = _state_arrays(state, requested)
    if requested is not None and not arrays:
        raise ValueError(f'none of checkpoint fields {requested!r} exist after {step}; '
            'the selected state contains no matching numerical leaves')
    return arrays


def capture_pipeline_steps(pipeline: Any, initial_state: Any, params: Mapping[str, Any], *,
        steps: Iterable[str] | str | None = None, fields: Iterable[str] | str |
        Mapping[str, Iterable[str] | str] | None = None,) -> dict[str, Any]:
    """Run a pipeline and capture selected state arrays after each step.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline to run.
    initial_state : State
        Input pipeline state.
    params : dict
        Calibration and extraction parameters.
    steps : None, str, list[str]
        Step names to include. If None, include all available steps.
    fields : None, str, list[str], dict
        Fields or dotted prefixes to capture, globally or by step name.

    Returns
    -------
    captured : dict
        Selected state arrays grouped by step name.
    """
    available = ('input', *(step.name for step in pipeline.steps))
    requested_steps = _names(steps)
    if requested_steps is None:
        selected_steps = set(available)
    else:
        selected_steps = set(requested_steps)
        unknown = selected_steps.difference(available)
        if unknown:
            raise ValueError(f'unknown checkpoint step(s): {sorted(unknown)}; '
                f'available: {list(available)}')
    state = initial_state
    captured = {}
    if 'input' in selected_steps:
        captured['input'] = _selected_state_arrays(state, step='input', fields=fields)
    for index, step in enumerate(pipeline.steps):
        state = pipeline.run(state, params, start=index, stop=index + 1)
        if step.name in selected_steps:
            captured[step.name] = _selected_state_arrays(state, step=step.name, fields=fields)
    return captured


def synthetic_checkpoints() -> tuple[dict[str, Any], dict[str, Any]]:
    """Create independent deterministic reference and candidate checkpoints.

    Returns
    -------
    reference, candidate : dict
        Independent copies of deterministic checkpoint arrays.
    """
    base = np.linspace(1.0, 4.0, 2 * 3 * 4 * 5, dtype=np.float32)
    ramp = base.reshape(2, 3, 4, 5)
    dq = np.zeros(ramp.shape, dtype=np.uint32)
    dq[0, 0, 0, 0] = 4
    calibrated = ramp + np.float32(2e-4) * ramp * ramp
    rate = np.mean(np.diff(calibrated, axis=1), axis=1)
    flux = np.sum(rate[:, 1:3, :], axis=1)
    reference = {'DQInit': {'data': ramp, 'groupdq': dq},
        'Linearity': {'data': calibrated, 'groupdq': dq},
        'RampFit': {'data': rate, 'dq': np.bitwise_or.reduce(dq, axis=1)},
        'Extract': {'flux': flux}}
    # Copy each array for an independent comparison.
    candidate = {step: {name: np.array(value, copy=True) for name, value in fields.items()}
        for step, fields in reference.items()}
    return reference, candidate


def run_synthetic_validation(report_path: str | Path | None = None) -> dict[str, Any]:
    """Compare synthetic checkpoints without claiming real-data parity.

    Parameters
    ----------
    report_path : None, str, Path
        Path to which to save the report.

    Returns
    -------
    report : dict
        Synthetic comparison results without a real-data parity claim.
    """
    reference, candidate = synthetic_checkpoints()
    report = compare_named_steps(reference, candidate, kind='synthetic-stepwise',
        parity_scope='synthetic_fixture_only')
    report['synthetic_agreement'] = report['passed']
    report['parity_claimed'] = False
    report['note'] = ('This verifies the comparison harness; it is not a real-data parity '
        'claim for v1 versus v2.')
    if report_path is not None:
        write_report(report, report_path)
    return report


def deferred_real_report(*, config: str | None = None, dataset: str | None = None,
        status: str = 'deferred', reason: str | None = None) -> dict[str, Any]:
    """Create a report for deferred or unavailable real-data validation.

    Parameters
    ----------
    config : None, str
        Configuration path recorded in the report.
    dataset : None, str
        Dataset identifier recorded in the report.
    status : str
        Validation status to record.
    reason : None, str
        Reason real-data validation is unavailable.

    Returns
    -------
    report : dict
        Report recording why real-data validation was not performed.
    """
    if reason is None:
        reason = ('Real-data validation needs v1 and v2 checkpoint NPZ bundles '
            'produced on a machine with the dataset and JWST/CRDS setup.')
    return {'schema_version': REPORT_SCHEMA_VERSION, 'created_at_utc': _utc_now(),
        'kind': 'real-data-stepwise', 'status': status, 'passed': False, 'parity_claimed': False,
        'parity_scope': 'none', 'config': config, 'dataset': dataset, 'reason': reason,
        'summary': {'steps_compared': 0, 'arrays_compared': 0}, 'steps': {}}


def _parse_step_groups(groups: Iterable[str] | None) -> tuple[str, ...] | None:
    """Parse repeated comma-separated CLI step selections."""
    if groups is None:
        return None
    selected = []
    for group in groups:
        for name in group.split(','):
            name = name.strip()
            if name and name not in selected:
                selected.append(name)
    if not selected:
        raise ValueError('--steps did not contain a step name')
    return tuple(selected)


def _compare_files(reference_path: str, candidate_path: str, *, kind: str, scope: str,
        selected_steps: Iterable[str] | None = None,
        default_tolerance: ArrayTolerance | None = None,) -> dict[str, Any]:
    """Compare selected steps from two checkpoint archives."""
    selected = _names(selected_steps)
    reference = load_checkpoint_npz(reference_path, steps=selected)
    candidate = load_checkpoint_npz(candidate_path, steps=selected)
    if selected is not None:
        missing_reference = [name for name in selected if name not in reference]
        missing_candidate = [name for name in selected if name not in candidate]
        if missing_reference or missing_candidate:
            details = []
            if missing_reference:
                details.append(f'reference missing {missing_reference}')
            if missing_candidate:
                details.append(f'candidate missing {missing_candidate}')
            raise ValueError('requested checkpoint step selection is incomplete: '
                + '; '.join(details))
        reference = {name: reference[name] for name in selected}
        candidate = {name: candidate[name] for name in selected}
    report = compare_named_steps(reference, candidate, kind=kind, parity_scope=scope,
        default_tolerance=default_tolerance)
    report['selected_steps'] = list(selected) if selected is not None else None
    return report


def _nonnegative_finite(text: str) -> float:
    """Parse a finite nonnegative command-line number."""
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError('must be a number') from error
    if not np.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('must be finite and non-negative')
    return value


def _unit_interval(text: str) -> float:
    """Parse a command-line number between zero and one."""
    value = _nonnegative_finite(text)
    if value > 1:
        raise argparse.ArgumentTypeError('must be no greater than 1')
    return value


def _add_tolerance_arguments(parser: argparse.ArgumentParser) -> None:
    """Add numerical and finite-mask tolerance arguments to a parser."""
    defaults = ArrayTolerance()
    parser.add_argument('--rtol', type=_nonnegative_finite, default=defaults.rtol,
        help=f'relative array tolerance (default: {defaults.rtol:g})')
    parser.add_argument('--atol', type=_nonnegative_finite, default=defaults.atol,
        help=f'absolute array tolerance (default: {defaults.atol:g})')
    parser.add_argument('--max-finite-mismatch-fraction', type=_unit_interval,
        default=defaults.max_finite_mismatch_fraction,
        help='maximum fraction of finite-mask disagreements '
        f'(default: {defaults.max_finite_mismatch_fraction:g})')


def _cli_tolerance(args: argparse.Namespace) -> ArrayTolerance:
    """Build array tolerances from command-line arguments."""
    return ArrayTolerance(rtol=float(args.rtol), atol=float(args.atol),
        max_finite_mismatch_fraction=float(args.max_finite_mismatch_fraction))


def build_parser() -> argparse.ArgumentParser:
    """Define commands for packing, comparing, and documenting validation runs.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser for the validation commands.
    """
    parser = argparse.ArgumentParser(description='Compare named exoTEDRF v1/v2 checkpoint arrays')
    sub = parser.add_subparsers(dest='command', required=True)
    synthetic = sub.add_parser('synthetic', help='smoke-test the stepwise comparison harness')
    synthetic.add_argument('--report', default='validation_synthetic.json')
    synthetic.add_argument('--x64', action='store_true',
        help='enable JAX x64 before any calculation (validation processes only)')
    compare = sub.add_parser('compare', help='compare two checkpoint NPZ files')
    compare.add_argument('--reference', required=True)
    compare.add_argument('--candidate', required=True)
    compare.add_argument('--report', default='validation_report.json')
    compare.add_argument('--steps', action='append',
        help='compare only these steps; repeat or use comma-separated names')
    compare.add_argument('--x64', action='store_true',
        help='enable JAX x64 before loading checkpoint inputs')
    _add_tolerance_arguments(compare)
    precision = sub.add_parser('precision', help='compare separate float32 and float64 NPZ runs')
    precision.add_argument('--float32', required=True,
        help='checkpoint NPZ from the production float32 run')
    precision.add_argument('--float64', required=True,
        help='checkpoint NPZ from a separate x64-enabled run')
    precision.add_argument('--float32-winner-json',
        help='optional JSON optimizer-winner mapping from the float32 run')
    precision.add_argument('--float64-winner-json',
        help='optional JSON optimizer-winner mapping from the float64 run')
    precision.add_argument('--report', default='validation_precision.json')
    precision.add_argument('--x64', action='store_true',
        help='enable JAX x64 before loading checkpoint inputs')
    _add_tolerance_arguments(precision)
    pack = sub.add_parser('pack-v1',
        help='read existing v1 Stage1/Stage2 artifacts into checkpoint NPZ')
    source = pack.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', help='v1 optimizer YAML containing pipeline_outputs_directory')
    source.add_argument('--outputs-dir', help='explicit existing v1 pipeline_outputs_directory')
    pack.add_argument('--output', default='v1_checkpoints.npz')
    pack.add_argument('--log-name', help='name in Cost_<name>.txt (defaults to config name_tag)')
    pack.add_argument('--no-logs', action='store_true',
        help='do not read Cost/Scatter optimizer logs')
    pack.add_argument('--compressed', action='store_true',
        help='compress the NPZ (smaller but slower and more CPU-intensive)')
    spectra = pack.add_mutually_exclusive_group()
    spectra.add_argument('--spectra', help='explicit v1 Stage3 *_spectra_fullres.fits product')
    spectra.add_argument('--no-spectra', action='store_true',
        help='do not pack a unique Stage3 full-resolution spectrum')
    spectrum_compare = sub.add_parser('spectra',
        help='compare final single-trace NIRSpec/MIRI spectrum FITS files')
    spectrum_compare.add_argument('--reference', required=True)
    spectrum_compare.add_argument('--candidate', required=True)
    spectrum_compare.add_argument('--baseline-ints',
        help='one or two comma-separated v1 baseline bounds')
    spectrum_compare.add_argument('--wave-range',
        help='two comma-separated inclusive wavelength bounds in microns')
    spectrum_compare.add_argument('--max-flux-p95-ppm', type=_nonnegative_finite)
    spectrum_compare.add_argument('--max-scatter-ratio-deviation', type=_nonnegative_finite)
    spectrum_compare.add_argument('--report', default='validation_spectra.json')
    soss_compare = sub.add_parser(
        'soss-spectra', help='compare both orders of final SOSS spectrum FITS')
    soss_compare.add_argument('--reference', required=True)
    soss_compare.add_argument('--candidate', required=True)
    soss_compare.add_argument('--baseline-ints')
    soss_compare.add_argument('--report', default='validation_soss_spectra.json')
    for order in ('o1', 'o2'):
        soss_compare.add_argument(f'--{order}-wave-range')
        for gate in ('max-flux-p95-ppm', 'max-flux-noise-fraction', 'max-scatter-ratio-deviation'):
            soss_compare.add_argument(f'--{order}-{gate}', type=_nonnegative_finite)
    real = sub.add_parser('real', help='compare real-data bundles or record validation as deferred')
    real.add_argument('--config')
    real.add_argument('--dataset')
    real.add_argument('--reference')
    real.add_argument('--candidate')
    real.add_argument('--steps', action='append',
        help='compare only these steps; repeat or use comma-separated names')
    real.add_argument('--report', default='validation_real.json')
    real.add_argument('--x64', action='store_true',
        help='enable JAX x64 before any real-data calculation')
    _add_tolerance_arguments(real)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested validation workflow and report whether it passed.

    Parameters
    ----------
    argv : None, list[str]
        Command-line arguments. If None, read sys.argv.

    Returns
    -------
    status : int
        Exit code: 0 for passed or deferred, 1 for failed, and 2 for unavailable.
    """
    args = build_parser().parse_args(argv)
    if getattr(args, 'x64', False):
        # Enable x64 before loading arrays or running calculations.
        from exotedrf.v2 import core
        core.enable_x64()
    if args.command == 'synthetic':
        report = run_synthetic_validation(args.report)
    elif args.command == 'compare':
        report = _compare_files(args.reference, args.candidate, kind='checkpoint-stepwise',
            scope='provided_checkpoint_files', selected_steps=_parse_step_groups(args.steps),
            default_tolerance=_cli_tolerance(args))
        write_report(report, args.report)
    elif args.command == 'precision':
        winner_args: dict[str, Any] = {}
        if args.float32_winner_json is not None:
            winner_args['float32_winner'] = json.loads(args.float32_winner_json)
        if args.float64_winner_json is not None:
            winner_args['float64_winner'] = json.loads(args.float64_winner_json)
        report = compare_precision_runs(load_checkpoint_npz(args.float32),
            load_checkpoint_npz(args.float64),
            default_tolerance=_cli_tolerance(args), **winner_args)
        write_report(report, args.report)
    elif args.command == 'pack-v1':
        locations = resolve_v1_output_locations(
            config_path=args.config, outputs_directory=args.outputs_dir, log_name=args.log_name)
        destination = pack_v1_outputs(locations.stage_root, args.output,
            log_directory=locations.log_directory,
            log_name=locations.log_name, include_logs=not args.no_logs, spectrum_path=args.spectra,
            include_spectrum=not args.no_spectra,
            compressed=args.compressed, config=locations.config)
        print(f'packed existing v1 artifacts: {destination}')
        return 0
    elif args.command in ('spectra', 'soss-spectra'):
        baseline = None
        if args.baseline_ints:
            try:
                baseline = [int(value.strip()) for value in args.baseline_ints.split(',')
                    if value.strip()]
            except ValueError as exc:
                raise ValueError('--baseline-ints must contain comma-separated integers') from exc
            if len(baseline) not in (1, 2):
                raise ValueError('--baseline-ints must contain one or two integers')
        def parse_wave_range(value):
            """Parse two comma-separated wavelength bounds."""
            if value is None:
                return None
            bounds = [float(item.strip()) for item in value.split(',')]
            if len(bounds) != 2:
                raise ValueError('wavelength ranges must contain two numbers')
            return bounds

        if args.command == 'spectra':
            report = compare_single_trace_spectra(
                args.reference, args.candidate, baseline_ints=baseline,
                wave_range=parse_wave_range(args.wave_range),
                max_flux_p95_ppm=args.max_flux_p95_ppm,
                max_scatter_ratio_deviation=args.max_scatter_ratio_deviation)
        else:
            options = {}
            for order in ('o1', 'o2'):
                options[f'{order}_wave_range'] = parse_wave_range(
                    getattr(args, f'{order}_wave_range'))
                for gate in ('max_flux_p95_ppm', 'max_flux_noise_fraction',
                    'max_scatter_ratio_deviation'):
                    key = f'{order}_{gate}'
                    options[key] = getattr(args, key)
            report = compare_soss_spectra(args.reference, args.candidate, baseline_ints=baseline,
                **options)
        write_report(report, args.report)
    else:
        supplied = (args.reference, args.candidate)
        if bool(args.reference) != bool(args.candidate):
            report = deferred_real_report(
                config=args.config, dataset=args.dataset, status='unavailable',
                reason=('Both --reference and --candidate are required when '
                'either checkpoint path is supplied.'))
        elif all(supplied) and all(Path(path).is_file() for path in supplied):
            report = _compare_files(args.reference, args.candidate, kind='real-data-stepwise',
                scope='provided_real_data_checkpoints',
                selected_steps=_parse_step_groups(args.steps),
                default_tolerance=_cli_tolerance(args))
            report['config'] = args.config
            report['dataset'] = args.dataset
        else:
            missing = [path for path in supplied if path and not Path(path).is_file()]
            if missing:
                report = deferred_real_report(
                    config=args.config, dataset=args.dataset, status='unavailable',
                    reason='Checkpoint file(s) unavailable: ' + ', '.join(missing))
            else:
                report = deferred_real_report(config=args.config, dataset=args.dataset)
        write_report(report, args.report)
    print(f'validation status: {report["status"]}; report: {args.report}')
    # Return separate exit codes for deferred and unavailable validation.
    if report['status'] == 'deferred':
        return 0
    if report['status'] == 'unavailable':
        return 2
    return 0 if report.get('passed', False) else 1


if __name__ == '__main__':
    raise SystemExit(main())
