"""Tests for exotedrf.v2.kernels.jump against numpy references."""

import numpy as np
import pytest
from scipy.ndimage import median_filter

import jax.numpy as jnp

from exotedrf.v2 import core
from exotedrf.v2.kernels import jump

NINTS, NGROUPS, DIMY, DIMX = 8, 3, 16, 32


# Numpy references (v1 transcriptions).

def ref_scatter_normalize_cube(cube, window=5):
    """Normalize a cube by its time-domain scatter using NumPy.

    Parameters
    ----------
    cube : array-like(float)
        Input observation cube.
    window : int
        Temporal filter window.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    cube_filt = median_filter(cube, (window,) + (1,) * (cube.ndim - 1))
    cube_filt[-2:], cube_filt[:2] = cube_filt[-3], cube_filt[3]
    scatter = np.median(np.abs(0.5 * (cube[0:-2] + cube[2:]) - cube[1:-1]),
                        axis=0)
    scatter = np.where(scatter == 0, np.inf, scatter)
    scale = np.abs(cube - cube_filt) / scatter
    return scale, cube_filt


def ref_jumpstep_in_time(cube, dqcube, window, thresh, artifact,
                         drop_groups=()):
    """Flag time-domain jumps using the v1 equations.

    Parameters
    ----------
    cube : array-like(float)
        Input observation cube.
    dqcube : array-like(int)
        Dqcube array.
    window : int
        Temporal filter window.
    thresh : float
        Detection threshold.
    artifact : array-like(float)
        Artifact array.
    drop_groups : tuple[int]
        Drop groups.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    cube, dqcube = cube.copy(), dqcube.copy()
    nints, ngroups, dimy, dimx = cube.shape
    for g in range(ngroups):
        if g in drop_groups:
            continue
        scale, cube_filt = ref_scatter_normalize_cube(cube[:, g], window)
        ii = ((scale >= thresh)
              & (cube[:, g] > np.nanpercentile(cube[:, g], 10))
              & (artifact == 0))
        if ngroups <= 2:
            jj = (dqcube[:, g] == 0) | (dqcube[:, g] == 4)
            replace = ii & jj
            cube[:, g][replace] = cube_filt[replace]
            dqcube[:, g][replace] = 0
        else:
            already = (dqcube[:, g] & 4) != 0
            to_flag = ii & ~already
            dqcube[:, g][to_flag] += 4
    return cube, dqcube


def ref_mask_reset_artifact(nints, dimy, dimx, int_start, instrument,
                            max_reset_int):
    """Build the v1 reset-artifact mask.

    Parameters
    ----------
    nints : int
        Number of integrations.
    dimy : int
        Number of detector rows.
    dimx : int
        Number of detector columns.
    int_start : int
        First integration index.
    instrument : str
        Instrument name.
    max_reset_int : int
        Last integration affected by the reset artifact.

    Returns
    -------
    mask : np.ndarray(int)
        Reset-artifact mask.
    """
    artifact = np.zeros((nints, dimy, dimx), dtype=int)
    int_end = np.min([int_start + nints - 1, max_reset_int])
    if int_start < max_reset_int:
        for j, jj in enumerate(range(int_start, int_end)):
            if instrument == 'NIRISS':
                min_row = np.max([max_reset_int - (jj + 3), 0])
                max_row = np.min([(max_reset_int + 2) - jj, dimy])
            else:
                min_row = np.max([max_reset_int - (jj + 2), 0])
                max_row = np.min([max_reset_int - jj, dimy])
            artifact[j, min_row:max_row, :] = 1
    return artifact


def make_cube(seed=0, ngroups=NGROUPS, noise=0.05):
    """Smooth positive scene + per-integration gaussian noise, float32.

    Parameters
    ----------
    seed : int
        Random seed.
    ngroups : int
        Number of groups per integration.
    noise : float
        Standard deviation of the synthetic noise.

    Returns
    -------
    result : np.ndarray(float)
        Calculated reference array.
    """
    rng = np.random.default_rng(seed)
    base = 100. + 1. * rng.standard_normal((ngroups, DIMY, DIMX))
    cube = base[None] + noise * rng.standard_normal(
        (NINTS, ngroups, DIMY, DIMX))
    return cube.astype(np.float32)


# Scatter normalization.

@pytest.mark.parametrize('window', [3, 5, 7, 9, 11])
def test_scatter_normalize_matches_v1(window):
    """Check scatter normalize matches v1."""
    rng = np.random.default_rng(1)
    cube = (50. + 10. * rng.standard_normal((NINTS, DIMY, DIMX))
            ).astype(np.float32)
    ref_scale, ref_filt = ref_scatter_normalize_cube(cube.copy(), window)
    scale, filt = jump.scatter_normalize(jnp.asarray(cube), window=window)
    np.testing.assert_allclose(np.asarray(filt), ref_filt, rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_allclose(np.asarray(scale), ref_scale, rtol=1e-5,
                               atol=1e-5)


def test_scatter_normalize_full_4d_equals_per_group():
    """Vectorizing over the group axis must equal v1's per-group loop."""
    cube = make_cube(seed=2)
    scale4, filt4 = jump.scatter_normalize(jnp.asarray(cube), window=5)
    for g in range(NGROUPS):
        ref_scale, ref_filt = ref_scatter_normalize_cube(cube[:, g].copy(), 5)
        np.testing.assert_allclose(np.asarray(scale4[:, g]), ref_scale,
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(filt4[:, g]), ref_filt,
                                   rtol=1e-5, atol=1e-5)


def test_prepared_jump_invariants_reused_across_windows():
    """Only the running median/scale changes with the structural window."""
    cube = make_cube(seed=12)
    cube[1, 2, 7, 9] += 20.
    data = jnp.asarray(cube)
    invariants = jump.prepare_jump_invariants(data)

    expected_scatter = np.median(
        np.abs(0.5 * (cube[:-2] + cube[2:]) - cube[1:-1]), axis=0)
    expected_scatter = np.where(expected_scatter == 0, np.inf,
                                expected_scatter)
    expected_floor = np.nanpercentile(cube, 10., axis=(0, 2, 3))
    np.testing.assert_allclose(np.asarray(invariants.scatter),
                               expected_scatter, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(invariants.flux_floor),
                               expected_floor, rtol=1e-6, atol=1e-6)

    for window in (3, 5, 7):
        prepared = jump.prepare_jumps_in_time(
            data, window=window, invariants=invariants)
        scale, cube_filt = jump.scatter_normalize(data, window=window)
        np.testing.assert_array_equal(np.asarray(prepared.scale),
                                      np.asarray(scale))
        np.testing.assert_array_equal(np.asarray(prepared.cube_filt),
                                      np.asarray(cube_filt))
        np.testing.assert_array_equal(np.asarray(prepared.flux_floor),
                                      np.asarray(invariants.flux_floor))


# Flag_jumps_in_time parity with v1.

def test_flag_jumps_matches_v1_ngroups3():
    """Check flag jumps matches v1 ngroups3."""
    cube = make_cube(seed=3)
    rng = np.random.default_rng(4)
    # Inject spikes, including at a pixel already carrying JUMP_DET.
    spikes = [(0, 0, 3, 5), (1, 1, 10, 20), (6, 2, 8, 30), (7, 0, 2, 2)]
    for i, g, y, x in spikes:
        cube[i, g, y, x] += 25.
    dq = rng.choice(np.array([0, 0, 0, 1, 4, 5, 2], dtype=np.uint8),
                    size=cube.shape)
    dq[1, 1, 10, 20] = 4
    artifact = np.zeros((NINTS, DIMY, DIMX), dtype=int)

    ref_data, ref_dq = ref_jumpstep_in_time(cube, dq, 5, 10., artifact)
    data, dqo = jump.flag_jumps_in_time(jnp.asarray(cube), jnp.asarray(dq),
                                        10., window=5,
                                        artifact=jnp.asarray(artifact) != 0)
    np.testing.assert_allclose(np.asarray(data), ref_data, rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(dqo), ref_dq)
    dqo = np.asarray(dqo)
    assert (dqo[0, 0, 3, 5] & 4) and (dqo[6, 2, 8, 30] & 4)
    assert dqo[1, 1, 10, 20] == 4


def test_flag_jumps_matches_v1_ngroups2_replacement():
    """Check flag jumps matches v1 ngroups2 replacement."""
    cube = make_cube(seed=5, ngroups=2)
    # Edge integrations: mid-series spikes self-shield (see ngroups3 test).
    spikes = [(1, 0, 4, 7), (6, 1, 12, 25), (7, 0, 9, 9)]
    for i, g, y, x in spikes:
        cube[i, g, y, x] += 30.
    dq = np.zeros(cube.shape, dtype=np.uint8)
    dq[1, 0, 4, 7] = 4
    dq[7, 0, 9, 9] = 1
    artifact = np.zeros((NINTS, DIMY, DIMX), dtype=int)

    ref_data, ref_dq = ref_jumpstep_in_time(cube, dq, 5, 10., artifact)
    data, dqo = jump.flag_jumps_in_time(jnp.asarray(cube), jnp.asarray(dq),
                                        10., window=5,
                                        artifact=jnp.asarray(artifact) != 0)
    np.testing.assert_allclose(np.asarray(data), ref_data, rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_array_equal(np.asarray(dqo), ref_dq)
    # The spiked values really were replaced by the running median.
    assert abs(np.asarray(data)[6, 1, 12, 25] - 100.) < 20.
    assert abs(np.asarray(data)[1, 0, 4, 7] - 100.) < 20.
    assert np.asarray(dqo)[1, 0, 4, 7] == 0
    # DNU pixel kept its spike and its flag.
    assert np.asarray(data)[7, 0, 9, 9] == cube[7, 0, 9, 9]
    assert np.asarray(dqo)[7, 0, 9, 9] == 1


def test_injected_outliers_flagged_exactly():
    """Check injected outliers flagged exactly."""
    rng = np.random.default_rng(6)
    base = (100. + rng.standard_normal((NGROUPS, DIMY, DIMX))
            ).astype(np.float32)
    alt = (0.05 * (-1.) ** np.arange(NINTS)).astype(np.float32)
    cube = base[None] + alt[:, None, None, None]
    # Edge integrations: mid-series spikes self-shield (see ngroups3 test).
    spikes = [(0, 0, 3, 5), (1, 1, 10, 20), (6, 2, 8, 30), (7, 0, 2, 2)]
    for i, g, y, x in spikes:
        cube[i, g, y, x] += 5.
    dq = np.zeros(cube.shape, dtype=np.uint8)
    _, dqo = jump.flag_jumps_in_time(jnp.asarray(cube), jnp.asarray(dq),
                                     10., window=5)
    flagged = np.argwhere((np.asarray(dqo) & 4) != 0)
    assert sorted(map(tuple, flagged)) == sorted(spikes)


def test_nan_pixels_never_flagged():
    """Check NaN pixels never flagged."""
    cube = make_cube(seed=7)
    cube[:, 1, 5, 5] = np.nan
    cube[3, 0, 6, 6] = np.nan
    cube[1, 2, 7, 7] += 25.
    dq = np.zeros(cube.shape, dtype=np.uint8)
    data, dqo = jump.flag_jumps_in_time(jnp.asarray(cube), jnp.asarray(dq),
                                        10., window=5)
    dqo = np.asarray(dqo)
    assert (dqo[:, 1, 5, 5] == 0).all()
    assert dqo[3, 0, 6, 6] == 0
    assert dqo[1, 2, 7, 7] == 4
    # Finite input stays finite.
    assert np.isfinite(np.asarray(data)[np.isfinite(cube)]).all()


@pytest.mark.parametrize('ngroups', [2, 3])
def test_prepared_apply_matches_v1_and_compatibility_wrapper(ngroups):
    """Preparation changes scheduling only, never output values or DQ."""
    cube = make_cube(seed=13 + ngroups, ngroups=ngroups)
    cube[0, 0, 3, 5] += 25.
    cube[6, ngroups - 1, 8, 30] += 25.
    dq = np.zeros(cube.shape, dtype=np.uint8)
    dq[0, 0, 3, 5] = 4
    artifact = np.zeros((NINTS, DIMY, DIMX), dtype=bool)
    artifact[0, 2, :] = True

    ref_data, ref_dq = ref_jumpstep_in_time(
        cube, dq, 5, 10., artifact.astype(int))
    data = jnp.asarray(cube)
    groupdq = jnp.asarray(dq)
    invariants = jump.prepare_jump_invariants(data)
    prepared = jump.prepare_jumps_in_time(
        data, window=5, invariants=invariants)
    got_data, got_dq = jump.apply_jumps_in_time(
        data, groupdq, 10., prepared, artifact=jnp.asarray(artifact))
    wrapper_data, wrapper_dq = jump.flag_jumps_in_time(
        data, groupdq, 10., window=5, artifact=jnp.asarray(artifact))

    np.testing.assert_array_equal(np.asarray(got_data),
                                  np.asarray(wrapper_data))
    np.testing.assert_array_equal(np.asarray(got_dq), np.asarray(wrapper_dq))
    np.testing.assert_allclose(np.asarray(got_data), ref_data, rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_array_equal(np.asarray(got_dq), ref_dq)


# Reset artifact & MIRI dropped groups.

@pytest.mark.parametrize('instrument,int_start,max_reset', [
    ('NIRISS', 253, 256),
    ('NIRISS', 1, 256),
    ('NIRSPEC', 58, 62),
    ('NIRSPEC', 300, 58),
])
def test_reset_artifact_mask_matches_v1(instrument, int_start, max_reset):
    """Check reset artifact mask matches v1."""
    got = jump.reset_artifact_mask(NINTS, DIMY, DIMX, int_start=int_start,
                                   instrument=instrument,
                                   max_reset_int=max_reset)
    ref = ref_mask_reset_artifact(NINTS, DIMY, DIMX, int_start, instrument,
                                  max_reset)
    np.testing.assert_array_equal(got.astype(int), ref)


def test_reset_artifact_defaults():
    """Check reset artifact defaults."""
    assert jump.default_max_reset_int('NIRISS') == 256
    assert jump.default_max_reset_int('NIRSPEC', 'G395H', 'NRS1') == 62
    assert jump.default_max_reset_int('NIRSPEC', 'G395H', 'NRS2') == 58
    assert jump.default_max_reset_int('NIRSPEC', 'G395M') == 81
    assert jump.default_max_reset_int('NIRSPEC', 'PRISM') == 68


def test_artifact_blocks_flagging():
    """Check artifact blocks flagging."""
    cube = make_cube(seed=8)
    artifact = jump.reset_artifact_mask(NINTS, DIMY, DIMX, int_start=253,
                                        instrument='NIRISS')
    assert artifact[0, 2, 2]
    assert not artifact[6, 2, 2]
    cube[0, 1, 2, 2] += 25.
    cube[6, 1, 12, 12] += 25.
    dq = np.zeros(cube.shape, dtype=np.uint8)
    _, dqo = jump.flag_jumps_in_time(jnp.asarray(cube), jnp.asarray(dq),
                                     10., window=5,
                                     artifact=jnp.asarray(artifact))
    dqo = np.asarray(dqo)
    assert dqo[0, 1, 2, 2] == 0
    assert dqo[6, 1, 12, 12] == 4


def test_miri_dropped_groups_skipped():
    """Check MIRI dropped groups skipped."""
    cube = make_cube(seed=9)
    dq = np.zeros(cube.shape, dtype=np.uint8)
    dq[:, 1] = 1
    dropped = jump.miri_dropped_groups(dq)
    np.testing.assert_array_equal(dropped, [False, True, False])
    cube[1, 1, 10, 20] += 25.
    cube[1, 0, 10, 20] += 25.
    _, dqo = jump.flag_jumps_in_time(jnp.asarray(cube), jnp.asarray(dq),
                                     10., window=5,
                                     group_ok=jnp.asarray(~dropped))
    dqo = np.asarray(dqo)
    assert (dqo[1, 1, 10, 20] & 4) == 0
    assert (dqo[1, 0, 10, 20] & 4) == 4


# Chunked driver & recompilation.

def test_chunked_over_columns_matches_unchunked():
    """Check chunked over columns matches unchunked."""
    cube = make_cube(seed=10)
    cube[6, 1, 10, 20] += 25.
    dq = np.zeros(cube.shape, dtype=np.uint8)
    d0, q0 = jump.flag_jumps_in_time(jnp.asarray(cube), jnp.asarray(dq), 10.,
                                     window=5)
    d1, q1 = jump.flag_jumps_in_time_chunked(cube, dq, 10., window=5,
                                             chunk_size=7)
    np.testing.assert_allclose(np.asarray(d1), np.asarray(d0), rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(q1), np.asarray(q0))


@pytest.mark.parametrize(
        ('specific', 'global_override', 'explicit', 'expected'),
        [
            (None, None, None, DIMX),
        (None, '41', None, 41),
        ('37', '41', None, 37),
        ('37', '41', 7, 7),
    ],
)
def test_chunked_driver_uses_bounded_jump_column_override(
        monkeypatch, specific, global_override, explicit, expected):
    """Check chunked driver uses bounded jump column override."""
    monkeypatch.delenv('EXOTEDRF_JUMP_CHUNK_COLS', raising=False)
    monkeypatch.delenv('EXOTEDRF_CHUNK_COLS', raising=False)
    if specific is not None:
        monkeypatch.setenv('EXOTEDRF_JUMP_CHUNK_COLS', specific)
    if global_override is not None:
        monkeypatch.setenv('EXOTEDRF_CHUNK_COLS', global_override)

    seen = {}

    def fake_map_over_cols(fn, arrays, n_cols, chunk_size=None):
        del fn
        seen['n_cols'] = n_cols
        seen['chunk_size'] = chunk_size
        return arrays[0], arrays[1]

    monkeypatch.setattr(jump.core, 'map_over_cols', fake_map_over_cols)
    cube = make_cube(seed=17)
    dq = np.zeros(cube.shape, dtype=np.uint8)
    jump.flag_jumps_in_time_chunked(
        cube, dq, 10., window=5, chunk_size=explicit)

    assert seen == {'n_cols': DIMX, 'chunk_size': expected}


def test_threshold_sweep_does_not_recompile():
    """Check threshold sweep does not recompile."""
    if not hasattr(jump.flag_jumps_in_time, '_cache_size'):
        pytest.skip('jit cache introspection unavailable in this jax')
    cube = jnp.asarray(make_cube(seed=11))
    dq = jnp.zeros(cube.shape, dtype=jnp.uint8)
    jump.flag_jumps_in_time(cube, dq, 5.0, window=5)
    n0 = jump.flag_jumps_in_time._cache_size()
    jump.flag_jumps_in_time(cube, dq, 12.5, window=5)
    assert jump.flag_jumps_in_time._cache_size() == n0


def test_prepared_threshold_sweep_does_not_reprepare_or_recompile():
    """Check prepared threshold sweep does not reprepare or recompile."""
    if not hasattr(jump.apply_jumps_in_time, '_cache_size'):
        pytest.skip('jit cache introspection unavailable in this jax')
    cube = jnp.asarray(make_cube(seed=16))
    dq = jnp.zeros(cube.shape, dtype=jnp.uint8)
    prepared = jump.prepare_jumps_in_time(cube, window=5)
    jump.apply_jumps_in_time(cube, dq, 5.0, prepared)
    n0 = jump.apply_jumps_in_time._cache_size()
    jump.apply_jumps_in_time(cube, dq, 12.5, prepared)
    assert jump.apply_jumps_in_time._cache_size() == n0
