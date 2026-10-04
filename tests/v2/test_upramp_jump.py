"""Check up-the-ramp jump detection against NumPy and stcal references."""

import warnings

import numpy as np
import pytest

from exotedrf.v2 import config, upramp
from exotedrf.v2.kernels import upramp_jump as k_up

DQ = {'DO_NOT_USE': 1, 'SATURATED': 2, 'JUMP_DET': 4,
      'NO_GAIN_VALUE': 524288, 'GOOD': 0, 'REFERENCE_PIXEL': 2147483648}
NIRISS = upramp._BUILTIN_PARS_1322['NIRISS'][1]
NIRSPEC = upramp._BUILTIN_PARS_1322['NIRSPEC'][1]
MIRI = upramp._BUILTIN_PARS_1322['MIRI'][1]


def synth(nints, ngroups, ny, nx, seed=0, dnu_groups=(), crs=30, blobs=4,
          cores=2, integer=False):
    """Create ramps with cosmic rays and saturated samples.

    Parameters
    ----------
    nints : int
        Number of integrations.
    ngroups : int
        Number of groups per integration.
    ny : int
        Number of detector rows.
    nx : int
        Number of detector columns.
    seed : int
        Random seed.
    dnu_groups : tuple[int]
        Dnu groups.
    crs : int
        Number of injected cosmic rays.
    blobs : int
        Number of extended cosmic-ray regions.
    cores : int
        Number of CPU slices.
    integer : bool
        Integer option.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    rng = np.random.default_rng(seed)
    rate = rng.uniform(5, 400, (ny, nx)).astype(np.float32)
    g = np.arange(ngroups, dtype=np.float32)
    data = (rate[None, None] * g[None, :, None, None] + 1000 +
            rng.normal(0, 8, (nints, ngroups, ny, nx))).astype(np.float32)
    gdq = np.zeros(data.shape, np.uint8)
    yy, xx = np.mgrid[:ny, :nx]
    for _ in range(crs):
        i, k = rng.integers(nints), rng.integers(1, ngroups)
        y, x = rng.integers(ny), rng.integers(nx)
        amp = rng.choice([150., 800., 3000., 20000.])
        data[i, k:, y, x] += amp
        if rng.random() < .3 and y + 1 < ny:
            data[i, k:, y + 1, x] += amp / 3
    for _ in range(blobs):
        i, k = rng.integers(nints), rng.integers(1, ngroups)
        y, x = rng.integers(3, ny - 3), rng.integers(3, nx - 3)
        disk = (yy - y) ** 2 + (xx - x) ** 2 <= rng.integers(4, 10)
        data[i, k:, disk] += 5000.
    for _ in range(cores if ngroups > 2 else 0):
        i, k = rng.integers(nints), rng.integers(1, ngroups - 1)
        y, x = rng.integers(2, ny - 7), rng.integers(2, nx - 7)
        gdq[i, k:, y:y + 5, x:x + 5] = 2
        gdq[i, k + 1:, y:y + 5, x:x + 5] = 3
        data[i, k:, max(0, y - 2):y + 7, max(0, x - 2):x + 7] += 8000.
    for _ in range(6):
        i, k = rng.integers(nints), rng.integers(1, ngroups)
        gdq[i, k:, rng.integers(ny), rng.integers(nx)] = 2
    for k in dnu_groups:
        gdq[:, k] |= 1
    gdq[:, :, 0, 0] |= 1
    gain = rng.uniform(1.5, 2.0, (ny, nx)).astype(np.float32)
    gain[1, 2] = np.nan
    gain[2, 3] = 0.
    rn = rng.uniform(8, 12, (ny, nx)).astype(np.float32)
    if integer:
        data = np.round(data).astype(np.float32)
        gain = np.round(gain).astype(np.float32)
        gain[1, 2] = np.nan
        gain[2, 3] = 0.
        rn = np.round(rn).astype(np.float32)
    return data, gdq, gain, rn


def _jump_data(gain, rn, pars, args, cores='1'):
    """jwst 3.0.0 ``JumpStep._setup_jump_data`` for array inputs."""
    from stcal.jump.jump_class import JumpData
    jd = JumpData(gain2d=gain.copy(), rnoise2d=rn.copy(), dqflags=DQ)
    jd.nframes = args['nframes']
    # JumpData(model) without a read pattern (every JWST exposure).
    jd.dt_group = np.ones(1)
    jd.n_reads_groupdiff = np.ones(1) * 2 * args['nframes']
    jd.set_detection_settings(
        pars['rejection_threshold'], pars['three_group_rejection_threshold'],
        pars['four_group_rejection_threshold'],
        pars['max_jump_to_flag_neighbors'],
        pars['min_jump_to_flag_neighbors'], pars['flag_4_neighbors'])
    jd.set_after_jump(pars['after_jump_flag_dn1'], args['after_jump_flag_n1'],
                      pars['after_jump_flag_dn2'], args['after_jump_flag_n2'])
    jd.set_snowball_info(
        pars['expand_large_events'], pars['min_jump_area'],
        pars['min_sat_area'], pars['expand_factor'],
        pars['sat_required_snowball'], pars['min_sat_radius_extend'],
        args['sat_expand'], pars['edge_size'])
    jd.set_shower_info(
        False, pars['extend_snr_threshold'], pars['extend_min_area'],
        pars['extend_inner_radius'], pars['extend_outer_radius'],
        pars['extend_ellipse_expand_ratio'], pars['min_diffs_single_pass'],
        args['max_extended_radius'])
    jd.set_sigma_clipping_info(pars['minimum_groups'],
                               pars['minimum_sigclip_groups'],
                               pars['only_use_ints'])
    jd.max_cores = cores
    jd.mask_persist_grps_next_int = pars['mask_snowball_core_next_int']
    jd.persist_grps_flagged = args['persist_grps_flagged']
    return jd


def stcal_detect(data, gdq, gain, rn, pars, args, cores):
    """Detect jumps using the installed stcal implementation.

    Parameters
    ----------
    data : array-like(float)
        Data array.
    gdq : array-like(int)
        Gdq array.
    gain : array-like(float)
        Gain array.
    rn : array-like(float)
        Rn array.
    pars : dict
        Jump-detection parameters.
    args : dict
        Jump-detection options.
    cores : int
        Number of CPU slices.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    from stcal.jump.jump import detect_jumps_data
    jd = _jump_data(gain, rn, pars, args, cores)
    jd.init_arrays_from_arrays(data.copy(), gdq.copy(),
                               np.zeros(data.shape[2:], np.uint32))
    out, pdq_out, *_ = detect_jumps_data(jd)
    return out, pdq_out


def _calc_med_first_diffs(fd):
    """Return calc med first diffs."""
    fd = fd.copy()
    n = fd.shape[0] * fd.shape[1] - np.sum(np.isnan(fd), axis=(0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        maxval = np.nanmax(fd, axis=(0, 1))
        fd[(n >= 4) & (fd == maxval)] = np.nan
        med = np.nanmedian(fd, axis=(0, 1))
        med = np.where(n == 2, np.nanmin(fd, axis=(0, 1)), med)
    return np.where(n < 2, np.nan, med)


def reference_find_crs(data, gdq, gain, rn, pars, args):
    """Detect cosmic rays with the NumPy reference calculation.

    Parameters
    ----------
    data : array-like(float)
        Data array.
    gdq : array-like(int)
        Gdq array.
    gain : array-like(float)
        Gain array.
    rn : array-like(float)
        Rn array.
    pars : dict
        Jump-detection parameters.
    args : dict
        Jump-detection options.

    Returns
    -------
    flags : np.ndarray(int)
        Updated group DQ flags.
    """
    gdq = gdq.copy()
    nints, ngroups, nrows, ncols = data.shape
    dat = data * gain
    # Stcal divides by the float64 ``0.5 * n_reads_groupdiff`` array.
    rn2 = ((rn * gain) ** 2).astype(np.float64)
    nframes = args['nframes']
    thr, thr3, thr4 = (pars['rejection_threshold'],
                       pars['three_group_rejection_threshold'],
                       pars['four_group_rejection_threshold'])
    per_int = [sum(bool(np.all(gdq[i, g] & 1)) for g in range(ngroups))
               for i in range(nints)]
    min_usable_groups = ngroups - max(per_int)
    if min_usable_groups >= 3:
        dat[(gdq & 3) != 0] = np.nan
        first = np.diff(dat, axis=1).astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            med_all = np.nanmedian(first, axis=(0, 1))
        sigma = np.sqrt(np.abs(med_all) + rn2 / float(nframes))
        sigma[sigma == 0.] = np.nan
        ratio_all = np.abs(first - med_all) / sigma
        if min_usable_groups - 1 >= pars['min_diffs_single_pass']:
            jump = (ratio_all > thr) & np.isfinite(first)
            gdq[:, 1:] |= (jump * 4).astype(np.uint8)
        else:
            for r in range(nrows):
                for c in range(ncols):
                    v = np.abs(first[:, :, r, c])
                    flagged = np.zeros(v.shape, bool)
                    rounds = 0
                    while True:
                        m = int(np.sum(np.isfinite(v)))
                        if m < 2:
                            break
                        med = _calc_med_first_diffs(v[:, :, None])[0]
                        if rounds:
                            med = np.float32(med)
                        sig = np.sqrt(np.abs(med) + rn2[r, c] / float(nframes))
                        if sig == 0:
                            break
                        e = np.abs((v - med).astype(np.float32))
                        rat = e / sig
                        if np.all(np.isnan(rat)):
                            break
                        rmax = np.nanmax(rat)
                        t = thr if m >= 4 else (thr4 if m == 3 else thr3)
                        if not rmax > t:
                            break
                        hit = rat == rmax
                        flagged |= hit
                        v[hit] = np.nan
                        rounds += 1
                    gdq[:, 1:, r, c] |= (flagged * 4).astype(np.uint8)
        if pars['flag_4_neighbors']:
            src = ((ratio_all < pars['max_jump_to_flag_neighbors']) &
                   (ratio_all > pars['min_jump_to_flag_neighbors']) &
                   ((gdq[:, 1:] & 4) != 0))
            nb = np.zeros_like(src)
            nb[:, :, 1:] |= src[:, :, :-1]
            nb[:, :, :-1] |= src[:, :, 1:]
            nb[..., 1:] |= src[..., :-1]
            nb[..., :-1] |= src[..., 1:]
            target = gdq[:, 1:]
            target[(nb | src) & ((target & 3) == 0)] |= 4
        gmed = np.nanmedian(gain)
        e_jump = first - med_all
        jump_set = (gdq & 4) != 0
        add = np.zeros(gdq.shape, bool)
        for dn, n in ((pars['after_jump_flag_dn1'], args['after_jump_flag_n1']),
                      (pars['after_jump_flag_dn2'], args['after_jump_flag_n2'])):
            if n > 0:
                big = np.zeros(gdq.shape, bool)
                big[:, 1:] = (e_jump >= dn * gmed) & jump_set[:, 1:]
                for i, g, r, c in zip(*np.where(big)):
                    add[i, g:g + n + 1, r, c] = True
        gdq[add & ((gdq & 3) == 0)] |= 4
    for bit in (1, 2):
        both = (gdq & (bit | 4)) == (bit | 4)
        gdq[both] ^= 4
    return gdq


def run_v2(data, gdq, gain, rn, pars, cpu=1, group_time=2.0, **kw):
    """Run the v2 calculation on the synthetic observation.

    Parameters
    ----------
    data : array-like(float)
        Data array.
    gdq : array-like(int)
        Gdq array.
    gain : array-like(float)
        Gain array.
    rn : array-like(float)
        Rn array.
    pars : dict
        Jump-detection parameters.
    cpu : int
        Number of CPU slices.
    group_time : float
        Time between groups.
    kw : dict
        Additional reduction options or results.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    out, pdq, info = upramp.detect_jumps_segment(
        data, gdq, np.zeros(data.shape[2:], np.uint32), gain, rn, pars,
        group_time=group_time, nframes=1, cpu_count=cpu, **kw)
    return np.asarray(out), np.asarray(pdq), info


def test_cpu_iterative_jump_with_persistent_compilation_cache(tmp_path):
    """Check CPU iterative jump with persistent compilation cache."""
    import jax
    import jax.numpy as jnp
    from jax.experimental.compilation_cache import compilation_cache

    previous = (jax.config.jax_compilation_cache_dir,
                jax.config.jax_persistent_cache_min_compile_time_secs,
                jax.config.jax_enable_compilation_cache)
    try:
        jax.config.update('jax_compilation_cache_dir', str(tmp_path))
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
        jax.config.update('jax_enable_compilation_cache', True)
        compilation_cache.reset_cache()
        # Initialize the persistent cache before compiling a callback kernel.
        jax.jit(lambda x: x + 1)(jnp.zeros(17)).block_until_ready()
        data, gdq, gain, rn = synth(5, 10, 11, 13, seed=87)
        pars = upramp.effective_parameters(NIRISS, 4.5)
        pars['maximum_cores'] = '1'
        args = upramp.stcal_arguments(pars, gain, 2.0, 1)
        expected, expected_pdq = stcal_detect(data, gdq, gain, rn, pars, args, '1')
        actual, pdq, info = run_v2(data, gdq, gain, rn, pars)
        assert info['slice_modes'] == ('iterative',)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(pdq, expected_pdq)
        assert jax.config.jax_enable_compilation_cache
    finally:
        for name, value in zip(('jax_compilation_cache_dir',
                                 'jax_persistent_cache_min_compile_time_secs',
                                 'jax_enable_compilation_cache'), previous):
            jax.config.update(name, value)
        compilation_cache.reset_cache()


CASES = {
    # Name: (CRDS pars, shape, DO_NOT_USE groups).
    'niriss_single': (NIRISS, (6, 12, 20, 36), ()),
    'niriss_iterative': (NIRISS, (10, 9, 20, 36), ()),
    'nirspec_single': (dict(NIRSPEC, expand_large_events=False),
                       (5, 14, 16, 40), ()),
    'miri_iterative': (MIRI, (6, 12, 18, 14), (0, 1, 2, 11)),
    'tiny_iterative': (NIRSPEC | {'expand_large_events': False},
                       (1, 6, 18, 24), ()),
}


@pytest.mark.parametrize('integer', [False, True], ids=['float', 'ties'])
@pytest.mark.parametrize('case', sorted(CASES))
def test_find_crs_matches_numpy_reference(case, integer):
    """Check find crs matches NumPy reference."""
    mode_pars, shape, dnu = CASES[case]
    data, gdq, gain, rn = synth(*shape, dnu_groups=dnu, integer=integer,
                                seed=len(case))
    pars = upramp.effective_parameters(mode_pars, 4.5)
    args = upramp.stcal_arguments(pars, gain, 2.0, 1)
    expected = reference_find_crs(data, gdq, gain, rn, pars, args)
    got, pdq, info = run_v2(data, gdq, gain, rn, pars)
    want_mode = 'iterative' if 'iterative' in case else 'single'
    assert info['slice_modes'] == (want_mode,)
    assert np.array_equal(got, expected)
    assert (got & 4).any()
    bad = (gain <= 0) | np.isnan(gain)
    assert np.array_equal(pdq != 0, bad)


@pytest.mark.parametrize('integer', [False, True], ids=['float', 'ties'])
@pytest.mark.parametrize('cores', ['1', '3'])
@pytest.mark.parametrize('case', sorted(CASES))
def test_detect_jumps_matches_stcal(case, cores, integer):
    """Check detect jumps matches stcal."""
    mode_pars, shape, dnu = CASES[case]
    data, gdq, gain, rn = synth(*shape, dnu_groups=dnu, integer=integer,
                                seed=3 + len(case))
    pars = upramp.effective_parameters(mode_pars, 4.5)
    pars['maximum_cores'] = cores
    args = upramp.stcal_arguments(pars, gain, 2.0, 1)
    expected, expected_pdq = stcal_detect(data, gdq, gain, rn, pars, args,
                                          cores)
    got, pdq, info = run_v2(data, gdq, gain, rn, pars, cpu=64)
    assert info['n_slices'] == int(cores)
    assert np.array_equal(got, expected)
    assert np.array_equal(pdq, expected_pdq)


@pytest.mark.parametrize('cores', ['1', '4'])
@pytest.mark.parametrize('pars_name', ['nirspec_edge', 'niriss_saturated'])
def test_snowballs_match_stcal(pars_name, cores):
    """Check snowballs match stcal."""
    if pars_name == 'nirspec_edge':
        # 32-row NIRSpec subarray: every jump ellipse is "near the edge".
        mode_pars, shape = NIRSPEC, (4, 12, 32, 64)
    else:
        # Interior events need a saturated core (sat_required_snowball).
        mode_pars = dict(NIRISS, expand_large_events=True, sat_expand=2,
                         min_sat_area=1, min_jump_area=5)
        shape = (4, 9, 64, 64)
    data, gdq, gain, rn = synth(*shape, blobs=10, cores=4, seed=11)
    pars = upramp.effective_parameters(mode_pars, 6.)
    pars['maximum_cores'] = cores
    args = upramp.stcal_arguments(pars, gain, 0.9, 1)
    expected, _ = stcal_detect(data, gdq, gain, rn, pars, args, cores)
    got, _, info = run_v2(data, gdq, gain, rn, pars, cpu=64, group_time=0.9)
    assert info['snowballs'] > 0
    assert np.array_equal(got, expected)


def test_snowball_host_pass_matches_stcal_on_constructed_dq():
    """Check snowball host pass matches stcal on constructed DQ."""
    from stcal.jump.jump import flag_large_events
    rng = np.random.default_rng(5)
    gdq = np.zeros((3, 8, 60, 70), np.uint8)
    yy, xx = np.mgrid[:60, :70]
    for i in range(3):
        for _ in range(6):
            g = rng.integers(1, 8)
            y, x = rng.integers(5, 55), rng.integers(5, 65)
            gdq[i, g][(yy - y) ** 2 + 2 * (xx - x) ** 2 <= 12] |= 4
        y, x = rng.integers(10, 50), rng.integers(10, 60)
        g = rng.integers(1, 7)
        gdq[i, g:, y - 3:y + 3, x - 3:x + 3] |= 2
        gdq[i, g][(yy - y) ** 2 + (xx - x) ** 2 <= 30] |= 4
    kw = dict(min_sat_area=1, min_jump_area=5, expand_factor=2.,
              sat_required_snowball=True, min_sat_radius_extend=2.5,
              sat_expand=4, edge_size=25, max_extended_radius=400,
              mask_persist_grps_next_int=True, persist_grps_flagged=5)
    pars = upramp.effective_parameters(
        {'expand_large_events': True, 'min_sat_area': 1, 'min_jump_area': 5,
         'expand_factor': 2., 'sat_required_snowball': True,
         'min_sat_radius_extend': 2.5, 'edge_size': 25}, 4.)
    args = {'nframes': 1, 'after_jump_flag_n1': 0, 'after_jump_flag_n2': 0,
            'sat_expand': 4, 'max_extended_radius': 400,
            'persist_grps_flagged': 5}
    jd = _jump_data(np.ones((60, 70), np.float32),
                    np.ones((60, 70), np.float32), pars, args)
    expected, n_expected = flag_large_events(gdq.copy(), 4, 2, jd)
    got, n_got = k_up.flag_large_events_host(gdq.copy(), **kw)
    assert n_got == n_expected > 0
    assert np.array_equal(got, expected)


def test_iterative_kernel_stops_on_count_and_flags_last_cr():
    """Check iterative kernel stops on count and flags last cr."""
    diffs = np.array([[10.], [5000.], [11.], [9000.]], np.float32)
    valid = np.ones_like(diffs, bool)
    got = np.asarray(k_up.iterative_crs(
        diffs, valid, np.full(1, 4., np.float32), 4., 5., 6.))
    assert got[:, 0].tolist() == [False, True, False, True]


def test_skip_and_sigclip_slice_modes():
    """Check skip and sigclip slice modes."""
    gdq = np.zeros((4, 4, 8, 5), np.uint8)
    pars = upramp.effective_parameters({}, 4.)
    # Stcal 1.20 compares the per-integration count (3), not the pooled 12.
    assert upramp.slice_modes(gdq, (), 1, pars) == ('iterative',)
    assert upramp.slice_modes(
        gdq, (), 1, dict(pars, min_diffs_single_pass=3)) == ('single',)
    # Minimum_groups is honoured only by stcal's multiprocessing path.
    big = dict(pars, minimum_groups=5)
    assert upramp.slice_modes(gdq, (), 1, big) == ('iterative',)
    assert upramp.slice_modes(gdq, (4,), 2, big) == ('skip', 'skip')
    # One integration with two all-DNU groups leaves 2 < 3 usable groups.
    dnu = gdq.copy()
    dnu[1, 2:] = 1
    assert upramp.slice_modes(dnu, (), 1, pars) == ('skip',)
    # Rows 0-3 only: the slice without DNU planes still runs.
    dnu2 = gdq.copy()
    dnu2[1, 2:, 4:] = 1
    assert upramp.slice_modes(dnu2, (4,), 2, pars) == ('iterative', 'skip')
    with pytest.raises(NotImplementedError, match='sigma-clip'):
        upramp.slice_modes(gdq, (), 1, dict(pars, only_use_ints=False,
                                            minimum_sigclip_groups=10))


def test_stcal_slices_follow_calc_num_slices():
    """Check stcal slices follow calc num slices."""
    assert upramp.stcal_slices(256, 'quarter', 48) == (
        12, tuple(21 * k for k in range(1, 12)))
    assert upramp.stcal_slices(32, 'quarter', 3) == (1, ())
    assert upramp.stcal_slices(5, 'all', 48)[0] == 5


def test_crds_string_booleans_follow_configobj():
    # Jwst_niriss_pars-jumpstep_0081.asdf stores these two as strings.
    """Check CRDS string booleans follow configobj."""
    pars = upramp.effective_parameters(
        {'expand_large_events': 'False', 'flag_4_neighbors': 'False',
         'min_sat_area': 5}, 15)
    assert pars['expand_large_events'] is False
    assert pars['flag_4_neighbors'] is False
    assert pars['min_sat_area'] == 5.
    with pytest.raises(ValueError):
        upramp.effective_parameters({'expand_large_events': 'maybe'}, 15)


def test_effective_parameters_layering():
    """Check effective parameters layering."""
    pars = upramp.effective_parameters(NIRISS, 15)
    assert pars['rejection_threshold'] == 15.
    assert pars['minimum_sigclip_groups'] == int(1e6)
    assert pars['maximum_cores'] == 'quarter'
    assert pars['flag_4_neighbors'] is False
    assert pars['after_jump_flag_dn1'] == 1000.
    assert pars['four_group_rejection_threshold'] == 5.
    over = upramp.effective_parameters(NIRISS, 15, {'flag_4_neighbors': True})
    assert over['flag_4_neighbors'] is True
    with pytest.raises(ValueError):
        upramp.effective_parameters(NIRISS, 15, {'maximum_cores': 'all'})
    args = upramp.stcal_arguments(pars, np.full((2, 2), 1.6, np.float32),
                                  5.494, 1)
    assert args['after_jump_flag_n1'] == 16
    assert args['sat_expand'] == 0 and args['max_extended_radius'] == 200
    with pytest.raises(NotImplementedError, match='find_showers'):
        upramp.stcal_arguments(dict(pars, find_showers=True),
                               np.ones((2, 2), np.float32), 1., 1)


def test_crds_pars_disabled_and_builtin(monkeypatch):
    """Check CRDS pars disabled and builtin."""
    monkeypatch.setenv('STPIPE_DISABLE_CRDS_STEPPARS', 'True')
    assert upramp.resolve_crds_pars({'INSTRUME': 'NIRISS'})[0] == {}
    monkeypatch.delenv('STPIPE_DISABLE_CRDS_STEPPARS')
    header = {'INSTRUME': 'MIRI', 'DETECTOR': 'MIRIMAGE', 'FILTER': 'P750L'}
    pars, source = upramp._builtin_pars(header, 'jwst_1322.pmap')
    assert 'miri_pars-jumpstep_0004' in source
    assert pars['after_jump_flag_time2'] == 3000
    assert upramp._builtin_pars(header, 'jwst_9999.pmap') is None


# Configuration.

BASE = {'observing_mode': 'NIRISS/SOSS', 'extract_method': 'box',
        'oof_method': 'scale-achromatic'}


@pytest.mark.parametrize('mode_cfg', [
    {},
    {'observing_mode': 'NIRSpec/G395H', 'filter_detector': 'NRS1',
     'oof_method': 'median', 'INLCorrStep': 'skip', 'RefPixStep': 'skip',
     'FlatFieldStep': 'skip', 'BackgroundStep': 'skip',
     'EmiCorrStep': 'skip', 'ResetStep': 'skip'},
    {'observing_mode': 'MIRI/LRS', 'INLCorrStep': 'skip',
     'SuperBiasStep': 'skip', 'RefPixStep': 'skip',
     'DarkCurrentStep': 'skip', 'OneOverFStep_grp': 'skip',
     'OneOverFStep_int': 'skip', 'Extract2DStep': 'skip',
     'WaveCorrStep': 'skip'}], ids=['soss', 'nirspec', 'miri'])
def test_flag_up_ramp_and_jumpstep_kwargs_accepted(mode_cfg):
    """Check flag up ramp and jumpstep keywords accepted."""
    cfg = dict(BASE, **mode_cfg, flag_up_ramp=True, jump_threshold=12,
               stage1_kwargs={'JumpStep': {
                   'flag_4_neighbors': True, 'after_jump_flag_dn1': 500,
                   'expand_large_events': False, 'sat_expand': 3}})
    assert config.validate_supported_config(cfg) is cfg
    opts = config.fixed_options(cfg)
    assert opts['flag_up_ramp'] is True
    assert opts['upramp_jump_kwargs'] == {
        'flag_4_neighbors': True, 'after_jump_flag_dn1': 500.,
        'expand_large_events': False, 'sat_expand': 3}


@pytest.mark.parametrize('kwargs, error, match', [
    ({'rejection_threshold': 5}, ValueError, 'jump_threshold'),
    ({'maximum_cores': 'all'}, ValueError, 'quarter'),
    ({'minimum_sigclip_groups': 5}, ValueError, '1e6'),
    ({'time_window': 7}, NotImplementedError, 'top-level time_window'),
    ({'find_showers': True}, NotImplementedError, 'shower'),
    ({'sat_expand': 2.5}, ValueError, 'integer'),
    ({'flag_4_neighbors': 1}, ValueError, 'boolean'),
    ({'skip': True}, NotImplementedError, 'not a supported'),
])
def test_jumpstep_kwargs_rejected(kwargs, error, match):
    """Check jumpstep keywords rejected."""
    cfg = dict(BASE, flag_up_ramp=True,
               stage1_kwargs={'JumpStep': kwargs})
    with pytest.raises(error, match=match):
        config.validate_supported_config(cfg)


def test_other_stage1_kwargs_still_rejected():
    """Check other stage1 keywords still rejected."""
    cfg = dict(BASE, stage1_kwargs={'JumpStep': {'sat_expand': 1},
                                    'RefPixStep': {'odd_even_rows': False}})
    with pytest.raises(NotImplementedError, match='stage1_kwargs'):
        config.validate_supported_config(cfg)


def test_jump_threshold_validated_for_up_ramp():
    """Check jump threshold validated for up ramp."""
    with pytest.raises(ValueError, match='jump_threshold'):
        config.validate_supported_config(
            dict(BASE, flag_up_ramp=True, jump_threshold=None))


def test_flag_in_time_false_coerced_without_up_ramp():
    """Check flag in time false coerced without up ramp."""
    with pytest.warns(UserWarning, match='forces time-domain'):
        opts = config.fixed_options({'flag_in_time': False})
    assert opts['flag_in_time'] is True
    opts = config.fixed_options({'flag_in_time': False, 'flag_up_ramp': True})
    assert opts['flag_in_time'] is False
    # A time-domain sweep is meaningful (v1 runs it) without up-ramp .
    cfg = {'flag_in_time': False, 'time_window': [3, 5],
           'optimize_time_window': True}
    plan, _ = config.build_sweep_plan(cfg, 'NIRISS/SOSS')
    assert dict(plan)['JumpStep'][0][0] == 'time_window'
    # ... and a no-op when up-ramp replaces it.
    with pytest.raises(ValueError, match='flag_up_ramp=True'):
        config.build_sweep_plan(dict(cfg, flag_up_ramp=True), 'NIRISS/SOSS')
