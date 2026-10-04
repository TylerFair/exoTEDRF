"""Resolve JWST JumpStep parameters and flag up-the-ramp cosmic rays."""

from __future__ import annotations

import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from exotedrf.v2 import core
from exotedrf.v2.kernels import upramp_jump as k_up

JUMP_STEP_SPEC = {'rejection_threshold': (float, 4.0),
    'three_group_rejection_threshold': (float, 6.0), 'four_group_rejection_threshold': (float, 5.0),
    'maximum_cores': (str, '1'), 'flag_4_neighbors': (bool, True),
    'max_jump_to_flag_neighbors': (float, 1000.), 'min_jump_to_flag_neighbors': (float, 10.),
    'after_jump_flag_dn1': (float, 0.), 'after_jump_flag_time1': (float, 0.),
    'after_jump_flag_dn2': (float, 0.), 'after_jump_flag_time2': (float, 0.),
    'expand_large_events': (bool, False), 'min_sat_area': (float, 1.0),
    'min_jump_area': (float, 5.0), 'expand_factor': (float, 2.0), 'use_ellipses': (bool, False),
    'sat_required_snowball': (bool, True), 'min_sat_radius_extend': (float, 2.5),
    'sat_expand': (int, 2), 'edge_size': (int, 25), 'mask_snowball_core_next_int': (bool, True),
    'snowball_time_masked_next_int': (int, 4000), 'find_showers': (bool, False),
    'max_shower_amplitude': (float, 4.), 'extend_snr_threshold': (float, 1.2),
    'extend_min_area': (int, 90), 'extend_inner_radius': (float, 1.),
    'extend_outer_radius': (float, 2.6), 'extend_ellipse_expand_ratio': (float, 1.1),
    'time_masked_after_shower': (float, 15.), 'min_diffs_single_pass': (int, 10),
    'max_extended_radius': (int, 200), 'minimum_groups': (int, 3),
    'minimum_sigclip_groups': (int, 100), 'only_use_ints': (bool, True),}

# Apply the fixed call parameters used by v1 JumpStep.
V1_FORCED_CALL_PARS = {'maximum_cores': 'quarter', 'minimum_sigclip_groups': int(1e6)}

# Use stored context parameters when a live CRDS lookup is unavailable.
_BUILTIN_PARS_1322 = {'NIRISS': ('jwst_niriss_pars-jumpstep_0081.asdf', {
        'after_jump_flag_dn1': 1000, 'after_jump_flag_dn2': 0,
        'after_jump_flag_time1': 90, 'after_jump_flag_time2': 0,
        'edge_size': 20, 'expand_factor': 1.75, 'expand_large_events': False,
        'flag_4_neighbors': False, 'max_extended_radius': 100,
        'max_jump_to_flag_neighbors': 200.0, 'min_jump_area': 15.0,
        'min_jump_to_flag_neighbors': 10.0, 'min_sat_area': 5,
        'min_sat_radius_extend': 5.0, 'rejection_threshold': 6.0,
        'sat_expand': 0, 'sat_required_snowball': True, 'use_ellipses': False}),
    'NIRSPEC': ('jwst_nirspec_pars-jumpstep_0003.asdf', {
        'after_jump_flag_dn1': 0, 'after_jump_flag_dn2': 0,
        'after_jump_flag_time1': 0, 'after_jump_flag_time2': 0,
        'edge_size': 25, 'expand_factor': 2, 'expand_large_events': True,
        'extend_ellipse_expand_ratio': 0.0, 'extend_inner_radius': 0,
        'extend_min_area': 0, 'extend_outer_radius': 0.0,
        'extend_snr_threshold': 0.0, 'find_showers': False,
        'flag_4_neighbors': True, 'max_extended_radius': 200,
        'max_jump_to_flag_neighbors': 1000, 'min_jump_area': 5,
        'min_jump_to_flag_neighbors': 30, 'min_sat_area': 1,
        'min_sat_radius_extend': 2.5, 'minimum_groups': 3,
        'minimum_sigclip_groups': 100, 'only_use_ints': True,
        'sat_expand': 2, 'sat_required_snowball': True,
        'time_masked_after_shower': 0, 'use_ellipses': False}),
    'MIRI': ('jwst_miri_pars-jumpstep_0004.asdf', {
        'after_jump_flag_dn1': 500, 'after_jump_flag_dn2': 1000,
        'after_jump_flag_time1': 15, 'after_jump_flag_time2': 3000,
        'edge_size': 0, 'expand_factor': 0, 'expand_large_events': False,
        'extend_ellipse_expand_ratio': 1.1, 'extend_inner_radius': 1,
        'extend_min_area': 50, 'extend_outer_radius': 2.6,
        'extend_snr_threshold': 3.0, 'find_showers': False,
        'flag_4_neighbors': True, 'max_extended_radius': 200,
        'max_jump_to_flag_neighbors': 1000, 'min_jump_area': 0,
        'min_jump_to_flag_neighbors': 30, 'min_sat_area': 1,
        'min_sat_radius_extend': 0.0, 'minimum_groups': 3,
        'minimum_sigclip_groups': 100, 'only_use_ints': True,
        'sat_expand': 0, 'sat_required_snowball': False,
        'time_masked_after_shower': 30, 'use_ellipses': False}),}
_BUILTIN_MATCH = {'NIRISS': {'FILTER': 'CLEAR', 'PUPIL': 'GR700XD'}, 'NIRSPEC': {},
    'MIRI': {'DETECTOR': 'MIRIMAGE', 'FILTER': 'P750L'},}

_NO_GAIN_VALUE = np.uint32(524288)


_TRUE = ('true', 'on', 'yes', '1')
_FALSE = ('false', 'off', 'no', '0')


def _coerce(name, value):
    """Convert a JumpStep parameter using the stpipe validation rules."""
    kind = JUMP_STEP_SPEC[name][0]
    if kind is bool:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in _TRUE:
                return True
            if lowered in _FALSE:
                return False
            raise ValueError(f'{name}: {value!r} is not a boolean')
        if isinstance(value, (bool, np.bool_)) or value in (0, 1):
            return bool(value)
        raise ValueError(f'{name}: {value!r} is not a boolean')
    if kind is int:
        if isinstance(value, (float, np.floating)):
            raise ValueError(f'{name}: {value!r} is not an integer')
        return int(value)
    if kind is float:
        return float(value)
    return str(value)


def _crds_steppars_disabled():
    """Check whether the environment disables CRDS step parameters."""
    return os.environ.get('STPIPE_DISABLE_CRDS_STEPPARS', '') in ('true', 'True', 't', 'yes', 'y')


def _builtin_pars(header, context):
    """Select stored JumpStep parameters for the matching CRDS context."""
    instrument = str(header.get('INSTRUME', '')).upper()
    entry = _BUILTIN_PARS_1322.get(instrument)
    if entry is None or str(context or 'jwst_1322.pmap') != 'jwst_1322.pmap':
        return None
    for key, value in _BUILTIN_MATCH[instrument].items():
        if str(header.get(key, '')).upper() != value:
            return None
    name, pars = entry
    return dict(pars), f'builtin:{name}'


def resolve_crds_pars(header, context=None):
    """Return ``(parameters, source)`` from the CRDS ``pars-jumpstep`` file.

    Use stored jwst_1322.pmap parameters if the CRDS service cannot be reached.

    Parameters
    ----------
    header : dict, fits.Header
        Exposure primary header keywords.
    context : None, str
        CRDS context; CRDS_CONTEXT takes precedence if set.

    Returns
    -------
    parameters : dict
        CRDS JumpStep overrides or stored context parameters.
    source : str
        Reference provenance or lookup status.
    """
    if _crds_steppars_disabled():
        return {}, 'disabled:STPIPE_DISABLE_CRDS_STEPPARS'
    from exotedrf.v2 import refs
    params = refs._crds_parameters(header)
    if header.get('TSOVISIT') is not None:
        params['META.VISIT.TSOVISIT'] = str(header.get('TSOVISIT'))
    try:
        import crds
        with refs._crds_context(context) as selected:
            resolved = crds.getreferences(params, reftypes=['pars-jumpstep'], context=selected,
                observatory='jwst')
    except Exception as exc:  # noqa: BLE001
        lookup_error = type(exc).__name__ in ('CrdsLookupError',)
        if lookup_error:
            return {}, f'crds:no-match ({exc})'
        builtin = _builtin_pars(header, refs.effective_crds_context(context))
        if builtin is not None:
            warnings.warn(f'CRDS pars-jumpstep lookup failed ({exc}); using the '
                f'verbatim {builtin[1]} parameters for jwst_1322.pmap')
            return builtin
        raise
    path = resolved.get('pars-jumpstep')
    if not path or str(path).upper().startswith('N/A') or \
            not os.path.exists(str(path)):
        return {}, f'crds:{path}'
    import asdf
    with asdf.open(path) as handle:
        tree = dict(handle.tree.get('parameters', {}))
    tree.pop('class', None)
    tree.pop('name', None)
    return tree, f'crds:{os.path.basename(str(path))}'


def effective_parameters(crds_pars, jump_threshold, overrides=None):
    """Merge spec defaults, CRDS pars and v1's call keywords like stpipe.

    Parameters
    ----------
    crds_pars, overrides : None, dict
        CRDS step parameters and explicit JumpStep overrides, respectively.
    jump_threshold : float
        Cosmic-ray rejection threshold in standard deviations.

    Returns
    -------
    parameters : dict
        JumpStep defaults merged with CRDS values and explicit call parameters.
    """
    pars = {name: default for name, (_, default) in JUMP_STEP_SPEC.items()}
    for name, value in (crds_pars or {}).items():
        if name in JUMP_STEP_SPEC:
            pars[name] = _coerce(name, value)
    if jump_threshold is None:
        raise ValueError('flag_up_ramp=True requires a numeric jump_threshold')
    pars['rejection_threshold'] = _coerce('rejection_threshold', jump_threshold)
    pars.update(V1_FORCED_CALL_PARS)
    for name, value in (overrides or {}).items():
        if name in V1_FORCED_CALL_PARS or name == 'rejection_threshold':
            raise ValueError(f'{name} is fixed by v1 and cannot be overridden')
        pars[name] = _coerce(name, value)
    return pars


def stcal_slices(nrows, maximum_cores='quarter', cpu_count=None):
    """Determine the STCAL worker count and row-slice boundaries.

    Parameters
    ----------
    nrows : int
        Number of detector rows.
    maximum_cores : str
        STCAL row-slice CPU selection.
    cpu_count : None, int
        Available CPUs; use the host CPU count if omitted.

    Returns
    -------
    n_slices : int
        Number of STCAL worker row slices.
    slice_starts : tuple[int]
        Interior row boundaries of the worker slices.
    """
    available = int(cpu_count or os.cpu_count() or 1)
    if maximum_cores.isnumeric():
        n_slices = int(maximum_cores)
    elif maximum_cores.lower() in ('none', 'one'):
        n_slices = 1
    elif maximum_cores == 'quarter':
        n_slices = available // 4 or 1
    elif maximum_cores == 'half':
        n_slices = available // 2 or 1
    elif maximum_cores == 'all':
        n_slices = available
    else:
        n_slices = 1
    n_slices = min(int(nrows), n_slices, available)
    if n_slices <= 1:
        return 1, ()
    yinc = int(nrows) // n_slices
    return n_slices, tuple(k * yinc for k in range(1, n_slices))


def _count_all_dnu_planes(groupdq, bounds):
    """Count fully unusable groups by row slice and integration."""
    nints = int(groupdq.shape[0])
    counts = np.zeros((len(bounds) - 1, nints), dtype=np.int64)
    step = max(1, min(nints, 16))
    for lo in range(0, nints, step):
        part = groupdq[lo:lo + step]
        xp = jnp if core.is_device_array(part) else np
        dnu = (part & xp.uint8(1)) != 0
        flags = xp.stack([xp.all(dnu[:, :, a:b], axis=(2, 3))
                          for a, b in zip(bounds[:-1], bounds[1:])])
        flags = np.asarray(jax.device_get(flags))
        counts[:, lo:lo + flags.shape[1]] = flags.sum(axis=2)
    return counts


def slice_modes(groupdq, slice_starts, n_slices, pars):
    """Select the STCAL cosmic-ray detection method for each row slice.

    A single STCAL worker fixes minimum_groups at three; multiple workers use the parameter.

    Parameters
    ----------
    groupdq : array-like(int)
        Detector quality flags.
    slice_starts : tuple[int]
        Row indices separating STCAL detection slices.
    n_slices : int
        Number of STCAL row slices.
    pars : dict
        Resolved JumpStep parameters.

    Returns
    -------
    modes : tuple[str]
        Detection method for each slice: skip, single, or iterative.
    """
    nints, ngrps, nrows = (int(x) for x in groupdq.shape[:3])
    bounds = [0, *slice_starts, nrows]
    flagged = _count_all_dnu_planes(groupdq, bounds)
    min_groups = 3 if n_slices == 1 else int(pars['minimum_groups'])
    msg = int(pars['minimum_sigclip_groups'])
    only_ints = bool(pars['only_use_ints'])
    modes = []
    for per_int in flagged:
        total_groups = nints * ngrps - int(per_int.sum())
        min_usable_groups = ngrps - int(per_int.max())
        total_sigclip = nints if only_ints else total_groups
        sigclip_fails = ((only_ints and nints < msg) or (not only_ints and total_sigclip < msg))
        if sigclip_fails and min_usable_groups < min_groups:
            modes.append('skip')
        elif ((only_ints and nints >= msg) or (not only_ints and total_groups >= msg)):
            raise NotImplementedError('this exposure would reach stcal\'s across-integration '
                'sigma-clip jump path, which v1 disables with '
                'minimum_sigclip_groups=1e6 and v2 does not port')
        elif min_usable_groups - 1 >= int(pars['min_diffs_single_pass']):
            modes.append('single')
        else:
            modes.append('iterative')
    return tuple(modes)


def _round_up(value, dtype):
    """Round a threshold upward to preserve comparisons in the requested dtype."""
    dtype = np.dtype(dtype)
    value = float(value)
    if not np.isfinite(value) or dtype == np.float64:
        return dtype.type(value)
    rounded = dtype.type(value)
    if float(rounded) < value:
        rounded = np.nextafter(rounded, dtype.type(np.inf))
    return rounded


def stcal_arguments(pars, gain, group_time, nframes):
    """Convert JumpStep parameters to STCAL detector units and group counts.

    Parameters
    ----------
    pars : dict
        Resolved JumpStep parameters.
    gain : array-like(float)
        Detector gain in electrons per DN.
    group_time : float
        Time per detector group in seconds.
    nframes : int
        Number of frames averaged into a group.

    Returns
    -------
    arguments : dict
        JumpStep amplitudes in electrons and temporal masks in detector groups.
    """
    if bool(pars['find_showers']):
        raise NotImplementedError('JumpStep find_showers=True (MIRI shower flagging) is not '
            'ported; no CRDS pars file used by v1 TSO modes enables it')
    gtime = float(group_time)
    gain_median = np.nanmedian(np.asarray(gain, np.float32))
    return {'nframes': int(nframes),
        'after_jump_flag_e1': pars['after_jump_flag_dn1'] * gain_median,
        'after_jump_flag_n1': int(pars['after_jump_flag_time1'] // gtime),
        'after_jump_flag_e2': pars['after_jump_flag_dn2'] * gain_median,
        'after_jump_flag_n2': int(pars['after_jump_flag_time2'] // gtime),
        'persist_grps_flagged': int(pars['snowball_time_masked_next_int'] // gtime),
        # Double the expansion radii as in JWST JumpStep.
        'sat_expand': int(pars['sat_expand']) * 2,
        'max_extended_radius': int(pars['max_extended_radius']) * 2,}


def _chunk_cols(data, modes, chunk_cols):
    """Choose a column chunk width from the detector working-memory requirement."""
    if chunk_cols is not None:
        return int(chunk_cols)
    ncols = int(data.shape[-1])
    bytes_per_col = (int(np.dtype(data.dtype).itemsize) + 1) * int(np.prod(data.shape[:-1]))
    buffers = 40 if 'iterative' in modes else 24
    return core.auto_chunk(ncols, bytes_per_col, n_buffers=buffers, headroom=0.6)


def detect_jumps_segment(data, groupdq, pixeldq, gain, readnoise, pars, *,
                         group_time, nframes, cpu_count=None,
                         chunk_cols=None, snowball_workers=None):
    """Run v1's JWST up-the-ramp JumpStep on one segment.

    Parameters
    ----------
    data, gain, readnoise : array-like(float)
        Science ramps, gain (electrons/DN), and readnoise (DN), respectively.
    groupdq, pixeldq : array-like(int)
        Detector group and pixel quality flags, respectively.
    pars : dict
        Resolved JumpStep parameters.
    group_time : float
        Time per detector group in seconds.
    nframes : int
        Number of frames averaged into a group.
    cpu_count : None, int
        Available CPUs; use the host CPU count if omitted.
    chunk_cols : None, int
        Number of detector columns to process together.
    snowball_workers : None, int
        Number of workers for host snowball flagging.

    Returns
    -------
    groupdq, pixeldq : array-like(int)
        Detector group and pixel quality flags, respectively.
    info : dict
        STCAL slice selection, derived parameters, and snowball count.
    """
    nints, ngroups, nrows, ncols = (int(x) for x in data.shape)
    if ngroups <= 2:
        raise ValueError('JWST JumpStep is skipped for NGROUPS <= 2')
    dtype = np.dtype(data.dtype)
    gain = np.broadcast_to(np.asarray(gain, np.float32), (nrows, ncols))
    readnoise = np.broadcast_to(np.asarray(readnoise, np.float32), (nrows, ncols))
    args = stcal_arguments(pars, gain, group_time, nframes)
    n_slices, starts = stcal_slices(nrows, pars['maximum_cores'], cpu_count)
    modes = slice_modes(groupdq, starts, n_slices, pars)
    flag4 = bool(pars['flag_4_neighbors'])
    resident = core.is_device_array(data) and core.is_device_array(groupdq)
    gain_in = jnp.asarray(gain, dtype) if resident else gain.astype(dtype)
    rn_in = (jnp.asarray(readnoise, dtype) if resident else readnoise.astype(dtype))
    scalars = {target: jnp.asarray(pars[source], dtype) for target, source in (
        ('rejection_threshold', 'rejection_threshold'),
        ('three_group_threshold', 'three_group_rejection_threshold'),
        ('four_group_threshold', 'four_group_rejection_threshold'),
        ('max_jump_to_flag_neighbors', 'max_jump_to_flag_neighbors'),
        ('min_jump_to_flag_neighbors', 'min_jump_to_flag_neighbors'))}
    scalars['nframes'] = jnp.asarray(args['nframes'], dtype)
    for index in (1, 2):
        key = f'after_jump_flag_e{index}'
        scalars[key] = jnp.asarray(_round_up(args[key], dtype))
        key = f'after_jump_flag_n{index}'
        scalars[key] = jnp.asarray(args[key], jnp.int32)

    def chunk(d, g, ga, rn):
        """Detect cosmic rays in a detector-column chunk."""
        return k_up.find_crs(d, g, ga, rn, slice_starts=starts,
                             slice_modes=modes, flag_4_neighbors=flag4, **scalars)
    width = _chunk_cols(data, modes, chunk_cols)
    new_dq = core.map_over_cols_with_halo(chunk, (data, groupdq, gain_in, rn_in), ncols,
        halo=1 if flag4 else 0, chunk_size=width)
    n_events = 0
    if bool(pars['expand_large_events']):
        host = np.array(jax.device_get(new_dq), dtype=np.uint8, copy=True)
        host, n_events = k_up.flag_large_events_host(host, min_sat_area=pars['min_sat_area'],
            min_jump_area=pars['min_jump_area'], expand_factor=pars['expand_factor'],
            sat_required_snowball=pars['sat_required_snowball'],
            min_sat_radius_extend=pars['min_sat_radius_extend'],
            sat_expand=args['sat_expand'], edge_size=pars['edge_size'],
            max_extended_radius=args['max_extended_radius'],
            mask_persist_grps_next_int=pars['mask_snowball_core_next_int'],
            persist_grps_flagged=args['persist_grps_flagged'], max_workers=snowball_workers)
        new_dq = jnp.asarray(host) if resident else host
    bad_gain = (gain <= 0.) | np.isnan(gain)
    new_pdq = pixeldq
    if np.any(bad_gain):
        add = np.where(bad_gain, _NO_GAIN_VALUE | np.uint32(1), np.uint32(0)).astype(np.uint32)
        if core.is_device_array(pixeldq):
            new_pdq = pixeldq | jnp.asarray(add, pixeldq.dtype)
        else:
            new_pdq = np.bitwise_or(np.asarray(pixeldq), add).astype(np.asarray(pixeldq).dtype)
    info = {'n_slices': n_slices, 'slice_starts': starts,
            'slice_modes': modes, 'snowballs': n_events, **args}
    return new_dq, new_pdq, info


__all__ = ['JUMP_STEP_SPEC', 'V1_FORCED_CALL_PARS', 'detect_jumps_segment',
           'effective_parameters', 'resolve_crds_pars', 'slice_modes',
           'stcal_arguments', 'stcal_slices']
