"""Stage-1 steps must accept device-resident cubes and change nothing."""

import numpy as np
import jax.numpy as jnp
import pytest

from exotedrf.v2 import core, hoststats, stages
from exotedrf.v2.core import ObsMeta, RampCube, RateCube
from exotedrf.v2.pipeline import PipelineState


NINTS, NGROUPS, DIMY, DIMX = 8, 3, 8, 12


def meta(nints=NINTS, ngroups=NGROUPS, edges=None, starts=None):
    """Create metadata for a synthetic observation.

    Parameters
    ----------
    nints : int
        Number of integrations.
    ngroups : int
        Number of groups per integration.
    edges : None, array-like(int)
        Integration boundaries between segments.
    starts : None, array-like(int)
        First integration index of each segment.

    Returns
    -------
    meta : ObsMeta
        Synthetic observation metadata.
    """
    edges = np.asarray(edges if edges is not None else [nints])
    extra = {} if starts is None else {'segment_int_starts': tuple(starts)}
    return ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP96', 2.214, ngroups,
                   np.arange(nints, dtype=float), np.array([nints]), edges,
                   tuple(f'seg{i}.fits' for i in range(len(edges))), extra)


def ramp_arrays(seed=0x135283a):
    """A small, structured ramp with a few flagged and saturated samples.

    Parameters
    ----------
    seed : int
        Random seed.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    rng = np.random.default_rng(seed)
    ramp = np.arange(1, NGROUPS + 1, dtype=np.float32)[None, :, None, None]
    data = (5000. * ramp + rng.normal(0., 3., (NINTS, NGROUPS, DIMY, DIMX))
            ).astype(np.float32)
    # A bright trace so the 1/f and jump kernels have real structure.
    data[:, :, 3:5, :] += 900.
    data[2, -1, 6, 7] += 4000.
    groupdq = np.zeros(data.shape, np.uint8)
    groupdq[1, -1, 2, 2] = np.uint8(core.DQ_DO_NOT_USE)
    pixeldq = np.zeros((DIMY, DIMX), np.uint32)
    pixeldq[0, 1] = core.DQ_HOT
    return data, groupdq, pixeldq


def rate_arrays(seed=7):
    """Create synthetic rate, error and DQ arrays.

    Parameters
    ----------
    seed : int
        Random seed.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(100., 1., (NINTS, DIMY, DIMX)).astype(np.float32)
    err = np.full_like(data, 2.)
    dq = np.zeros(data.shape, np.uint32)
    dq[0, 1, 1] = core.DQ_DO_NOT_USE
    return data, err, dq


def ramp_state(edges=None, starts=None, aux=None):
    """Create a synthetic ramp state with optional segment boundaries.

    Parameters
    ----------
    edges : None, array-like(int)
        Integration boundaries between segments.
    starts : None, array-like(int)
        First integration index of each segment.
    aux : dict
        Additional reduction options or results.

    Returns
    -------
    state : PipelineState
        Synthetic observation state.
    """
    data, groupdq, pixeldq = ramp_arrays()
    cube = RampCube(data, groupdq, pixeldq,
                    meta(edges=edges, starts=starts))
    return PipelineState(cube, dict(aux or {}))


def device_state(state):
    """The same observation with its science arrays uploaded.

    Parameters
    ----------
    state : PipelineState
        Input observation state.

    Returns
    -------
    state : PipelineState
        Synthetic observation state.
    """
    cube = state.cube
    if isinstance(cube, RampCube):
        moved = RampCube(jnp.asarray(cube.data), jnp.asarray(cube.groupdq),
                         cube.pixeldq, cube.meta)
    else:
        moved = RateCube(jnp.asarray(cube.data), jnp.asarray(cube.err),
                         jnp.asarray(cube.dq), cube.meta)
    return PipelineState(moved, dict(state.aux))


def cube_arrays(cube):
    """Return the science, error and DQ arrays from a cube.

    Parameters
    ----------
    cube : RampCube, RateCube
        Input observation cube.

    Returns
    -------
    arrays : dict
        Named science and DQ arrays.
    """
    if isinstance(cube, RampCube):
        return {'data': cube.data, 'groupdq': cube.groupdq,
                'pixeldq': cube.pixeldq}
    return {'data': cube.data, 'err': cube.err, 'dq': cube.dq}


def assert_same_result(host_state, device_out, *, aux_keys=()):
    """Compare host and device science arrays and auxiliary results.

    Parameters
    ----------
    host_state : PipelineState
        Expected state from host inputs.
    device_out : PipelineState
        Calculated state from device inputs.
    aux_keys : tuple[str]
        Auxiliary results to compare.
    """
    expected = cube_arrays(host_state.cube)
    actual = cube_arrays(device_out.cube)
    assert set(expected) == set(actual)
    for name, reference in expected.items():
        np.testing.assert_array_equal(
            np.asarray(actual[name]), np.asarray(reference),
            err_msg=f'{name} differs between host and device inputs')
        assert np.asarray(actual[name]).dtype == np.asarray(reference).dtype
    for key in aux_keys:
        assert key in device_out.aux
        reference, produced = host_state.aux[key], device_out.aux[key]
        if isinstance(reference, dict):
            assert set(reference) == set(produced)
            for name, value in reference.items():
                if value is None:
                    assert produced[name] is None
                else:
                    np.testing.assert_array_equal(
                        np.asarray(produced[name]), np.asarray(value))
        else:
            np.testing.assert_array_equal(
                np.asarray(produced), np.asarray(reference))


def run_both(step, state, params, ctx, *, aux_keys=()):
    """Run one step on host and on device inputs and compare everything.

    Parameters
    ----------
    step : callable
        Reduction step to execute.
    state : PipelineState
        Input observation state.
    params : dict
        Calculation parameters.
    ctx : dict
        Reduction context.
    aux_keys : tuple[str]
        Auxiliary results to compare.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    host_out = step(state, params, ctx)
    device_out = step(device_state(state), params, ctx)
    assert_same_result(host_out, device_out, aux_keys=aux_keys)
    return host_out, device_out


def assert_resident(cube, *names):
    """Check that the named arrays remain on the device.

    Parameters
    ----------
    cube : RampCube, RateCube
        Input observation cube.
    names : list[str]
        Names of functions or arrays to select.
    """
    for name in names:
        assert core.is_device_array(getattr(cube, name)), (
            f'{name} left the accelerator')


# Detector steps.

def test_dq_init_accepts_device_inputs():
    """Check DQ init accepts device inputs."""
    state = ramp_state()
    ctx = {'opts': {'saturation_threshold': 80},
           'refpack': {'mask_dq': np.where(
               np.eye(DIMY, DIMX, dtype=bool), np.uint32(3), np.uint32(0))}}
    _, device_out = run_both(stages.step_dq_init, state, {}, ctx,
                             aux_keys=('hot_pixels',))
    assert_resident(device_out.cube, 'data', 'groupdq')
    assert isinstance(device_out.cube.pixeldq, np.ndarray)


def test_superbias_accepts_device_inputs_for_both_methods():
    """Check superbias accepts device inputs for both methods."""
    state = ramp_state()
    superbias = np.full((DIMY, DIMX), 4900., np.float32)
    superbias[0, 0] = np.nan
    crds = {'opts': {'superbias_method': 'crds'},
            'refpack': {'superbias': superbias,
                        'superbias_dq': np.full((DIMY, DIMX), 2048,
                                                np.uint32)}}
    _, device_out = run_both(stages.step_superbias, state, {}, crds)
    assert_resident(device_out.cube, 'data', 'groupdq')

    visit = {'opts': {'superbias_method': 'visit'}}
    host_out, device_out = run_both(stages.step_superbias, state, {}, visit)
    assert_resident(device_out.cube, 'data')
    baseline = stages.baseline_bool_for_meta(state.cube.meta, NINTS)
    reference = np.nanmedian(
        np.asarray(state.cube.data)[np.asarray(baseline), 0], axis=0)
    np.testing.assert_array_equal(
        np.asarray(host_out.cube.data),
        np.asarray(state.cube.data) - reference[None, None])


def test_refpix_accepts_device_inputs():
    """Check reference-pixel correction accepts device inputs."""
    data, groupdq, pixeldq = ramp_arrays()
    obs = meta()
    obs.subarray = 'SUBSTRIP256'
    state = PipelineState(RampCube(data, groupdq, pixeldq, obs))
    _, device_out = run_both(stages.step_refpix, state, {}, {'opts': {}})
    assert_resident(device_out.cube, 'data', 'groupdq')


def test_linearity_accepts_device_inputs():
    """Check linearity accepts device inputs."""
    state = ramp_state()
    coeffs = np.zeros((3, DIMY, DIMX), np.float32)
    coeffs[1] = 1.0001
    coeffs[2] = 1e-9
    lin_dq = np.zeros((DIMY, DIMX), np.uint32)
    lin_dq[1, 1] = np.uint32(1 << 20)
    ctx = {'opts': {}, 'refpack': {'lin_coeffs': coeffs, 'lin_dq': lin_dq}}
    _, device_out = run_both(stages.step_linearity, state, {}, ctx)
    assert_resident(device_out.cube, 'data', 'groupdq')


# Background and 1/f.

def soss_ctx(**updates):
    """Create the SOSS reduction context with the supplied overrides.

    Parameters
    ----------
    updates : dict
        Overrides for the returned configuration.

    Returns
    -------
    context : dict
        Reduction options and reference arrays.
    """
    ctx = {
        'opts': {'oof_method': 'scale-achromatic'},
        'background_model': np.full((DIMY, DIMX), .5, np.float32),
        'centroids': {'ypos o1': np.full(DIMX, 3.5, np.float32)},
        'soss_timeseries': np.linspace(.99, 1.01, NINTS).astype(np.float32),
        'soss_timeseries_o2': None,
        'outlier_mask': None,
        'order0_mask': None,
    }
    ctx.update(updates)
    return ctx


def test_background_grp_accepts_device_inputs():
    """Check background grp accepts device inputs."""
    state = ramp_state()
    _, device_out = run_both(
        stages.step_background_grp, state, {}, soss_ctx(),
        aux_keys=('bkg_grp',))
    assert_resident(device_out.cube, 'data', 'groupdq')
    # The prepared 1/f record is an exactly reproduced host artefact.
    host_prepared = stages.step_background_grp(
        state, {}, soss_ctx()).aux[stages._OOF_GRP_PREP_KEY]
    prepared = device_out.aux[stages._OOF_GRP_PREP_KEY]
    for key, value in host_prepared.items():
        if value is None:
            assert prepared[key] is None
            continue
        assert not core.is_device_array(prepared[key])
        np.testing.assert_array_equal(np.asarray(prepared[key]),
                                      np.asarray(value))


def test_host_deepstack_matches_for_device_and_host_cubes():
    """Check host deepstack matches for device and host cubes."""
    data, _, _ = ramp_arrays()
    baseline = np.zeros(NINTS, bool)
    baseline[:3] = True
    reference = stages._host_deepstack(data, baseline)
    resident = stages._host_deepstack(jnp.asarray(data), baseline)
    assert isinstance(resident, np.ndarray)
    np.testing.assert_array_equal(resident, reference)
    # The three-dimensional (post-RampFit) form takes the same route.
    rate = data[:, -1]
    np.testing.assert_array_equal(
        stages._host_deepstack(jnp.asarray(rate), baseline),
        stages._host_deepstack(rate, baseline))


@pytest.mark.parametrize('ndim', [3, 4])
@pytest.mark.parametrize('width', [1, 5, 12])
def test_device_deepstack_columns_slices_a_resident_cube(monkeypatch, ndim,
                                                         width):
    """The streamed column route must not gather a resident cube on host."""
    rng = np.random.default_rng(11)
    shape = (NINTS, 2, DIMY, DIMX) if ndim == 4 else (NINTS, DIMY, DIMX)
    data = rng.normal(size=shape).astype(np.float32)
    data[rng.random(shape) < .05] = np.nan
    baseline = rng.random(NINTS) < .6
    baseline[:3] = True
    reference = stages._device_deepstack_columns(data, baseline, width)
    resident = stages._device_deepstack_columns(
        jnp.asarray(data), baseline, width)
    np.testing.assert_array_equal(resident, reference)

    def forbidden(*args, **kwargs):
        raise AssertionError('a resident cube was gathered on the host')

    monkeypatch.setattr(stages, '_device_fast_path_fits',
                        lambda *a, **k: False)
    monkeypatch.setattr(stages, '_device_deepstack_column_width',
                        lambda *a, **k: width)
    monkeypatch.setattr(np, 'ascontiguousarray', forbidden)
    np.testing.assert_array_equal(
        stages._host_deepstack(jnp.asarray(data), baseline), reference)


@pytest.mark.parametrize('fast_path', ['true', 'false'])
@pytest.mark.parametrize('edges,starts', [(None, None), ([4, 8], [1, 5])])
def test_oneoverf_grp_accepts_device_inputs(monkeypatch, fast_path, edges,
                                            starts):
    """Check 1/f correction grp accepts device inputs."""
    monkeypatch.setenv('EXOTEDRF_DEVICE_FAST_PATH', fast_path)
    state = ramp_state(edges=edges, starts=starts)
    params = {'soss_inner_mask_width': 5., 'soss_outer_mask_width': 9.}
    _, device_out = run_both(
        stages.step_oneoverf_grp, state, params, soss_ctx())
    assert_resident(device_out.cube, 'data', 'groupdq')


def test_oneoverf_grp_window_method_accepts_device_inputs(monkeypatch):
    """Check 1/f correction grp window method accepts device inputs."""
    monkeypatch.setenv('EXOTEDRF_DEVICE_FAST_PATH', 'false')
    state = ramp_state()
    ctx = soss_ctx(opts={'oof_method': 'scale-achromatic-window'})
    params = {'soss_inner_mask_width': 5., 'soss_outer_mask_width': 9.}
    _, device_out = run_both(stages.step_oneoverf_grp, state, params, ctx)
    assert_resident(device_out.cube, 'data')


# Jump.

@pytest.mark.parametrize('edges,starts', [(None, None), ([4, 8], [1, 5])])
def test_jump_accepts_device_inputs(edges, starts):
    """Check jump accepts device inputs."""
    state = ramp_state(edges=edges, starts=starts)
    params = {'time_jump_threshold': 5., 'time_window': 3}
    ctx = {'opts': {'flag_up_ramp': False, 'flag_in_time': True}}
    host_out, device_out = run_both(stages.step_jump, state, params, ctx)
    assert_resident(device_out.cube, 'data', 'groupdq')
    # The detector really is flagging something, so the comparison has bite.
    assert np.any(np.asarray(host_out.cube.groupdq) != state.cube.groupdq)


def test_two_group_jump_accepts_device_inputs():
    """Check two group jump accepts device inputs."""
    data, groupdq, pixeldq = ramp_arrays()
    data, groupdq = data[:, :2].copy(), groupdq[:, :2].copy()
    data[3, 1, 5, 5] += 8000.
    state = PipelineState(RampCube(data, groupdq, pixeldq,
                                   meta(ngroups=2)))
    params = {'time_jump_threshold': 5., 'time_window': 3}
    ctx = {'opts': {'flag_up_ramp': False, 'flag_in_time': True}}
    _, device_out = run_both(stages.step_jump, state, params, ctx)
    # NGROUPS <= 2 replaces science values, so the joined cube is resident.
    assert_resident(device_out.cube, 'data', 'groupdq')


def test_upramp_jump_accepts_device_inputs():
    """Check upramp jump accepts device inputs."""
    state = ramp_state()
    params = {'time_jump_threshold': 5., 'time_window': 3}
    ctx = {'opts': {'flag_up_ramp': True, 'flag_in_time': False,
                    'jump_threshold': 4},
           'refpack': {'readnoise': np.full((DIMY, DIMX), 6., np.float32),
                       'gain': np.full((DIMY, DIMX), 1.6, np.float32)},
           # JWST JumpStep.spec defaults (no CRDS pars layer offline).
           'upramp_jump_crds_pars': {}}
    _, device_out = run_both(stages.step_jump, state, params, ctx)
    assert_resident(device_out.cube, 'groupdq')


def test_nanpercentile_fast_on_device_array_matches_numpy():
    """Check nanpercentile fast on device array matches NumPy."""
    rng = np.random.default_rng(4242)
    plane = rng.normal(50., 4., (NINTS, DIMY, DIMX)).astype(np.float32)
    plane[0, 0, 0] = np.nan
    plane[2, 3, 4] = np.nan
    resident = jnp.asarray(plane)
    for q in (0., 10., 25., 33.33, 50., 90., 100.):
        actual = hoststats.nanpercentile_fast(resident, q)
        assert np.array_equal(actual, np.nanpercentile(plane, q))
    np.testing.assert_array_equal(hoststats.nanmedian(resident),
                                  np.nanmedian(plane))


def test_all_nan_device_plane_warns_and_returns_nan():
    """Check all NaN device plane warns and returns NaN."""
    with pytest.warns(RuntimeWarning, match='All-NaN'):
        assert np.isnan(hoststats.nanpercentile_fast(
            jnp.full((3, 4), np.nan, np.float32), 10.))


# RampFit and the post-RampFit steps.

def rampfit_ctx():
    """Create reference arrays and options for ramp-fitting tests.

    Returns
    -------
    context : dict
        Reduction options and reference arrays.
    """
    return {'opts': {}, 'refpack': {
        'readnoise': np.full((DIMY, DIMX), 6., np.float32),
        'gain': np.full((DIMY, DIMX), 1.6, np.float32),
        'gain_factor': 1.05}}


@pytest.mark.parametrize('edges,starts', [(None, None), ([4, 8], [1, 5])])
def test_rampfit_accepts_device_inputs_and_returns_host_rates(edges, starts):
    """Check rampfit accepts device inputs and returns host rates."""
    state = ramp_state(edges=edges, starts=starts)
    # One pixel saturates late so the nearest-neighbour fill has work to do.
    state.cube.groupdq[0, 1:, 4, 4] = np.uint8(core.DQ_SATURATED)
    host_out, device_out = run_both(
        stages.step_rampfit, state, {}, rampfit_ctx())
    cube = device_out.cube
    assert isinstance(cube, RateCube)
    for name in ('data', 'err', 'dq'):
        array = getattr(cube, name)
        assert isinstance(array, np.ndarray), f'{name} must return to host'
    assert np.isfinite(np.asarray(host_out.cube.data)).all()


def test_rampfit_jump_segments_accept_device_inputs():
    """A mid-ramp jump exercises the multi-segment/sparse fitting route."""
    state = ramp_state()
    state.cube.groupdq[1, 1, 5, 5] = np.uint8(core.DQ_JUMP_DET)
    state.cube.groupdq[4, 1, 2, 9] = np.uint8(core.DQ_JUMP_DET)
    run_both(stages.step_rampfit, state, {}, rampfit_ctx())


def test_sparse_rampfit_gathers_identically_from_device_segments():
    """Check sparse rampfit gathers identically from device segments."""
    data, groupdq, pixeldq = ramp_arrays()
    groupdq[1, 1, 5, 5] = np.uint8(core.DQ_JUMP_DET)
    groupdq[4, 1, 2, 9] = np.uint8(core.DQ_JUMP_DET)
    readnoise = np.full((DIMY, DIMX), 6., np.float32)
    gain = np.full((DIMY, DIMX), 1.6, np.float32)
    median_rate = np.full((DIMY, DIMX), 5000., np.float32)
    exceptions = np.zeros((NINTS, DIMY, DIMX), bool)
    exceptions[1, 5, 5] = True
    exceptions[4, 2, 9] = True
    base = tuple(np.zeros((NINTS, DIMY, DIMX), dtype)
                 for dtype in (np.float32, np.float32, np.uint32))

    def run(science, dq):
        return stages._apply_sparse_rampfit(
            *(array.copy() for array in base), science, dq,
            readnoise, gain, pixeldq, median_rate, exceptions,
            np.float32(2.214), np.float32(1.), np.float32(0.), None, 2,
            chunk_size=1)

    expected = run(data, groupdq)
    actual = run(jnp.asarray(data), jnp.asarray(groupdq))
    for reference, produced in zip(expected, actual):
        np.testing.assert_array_equal(np.asarray(produced),
                                      np.asarray(reference))


def test_gain_scale_and_flat_accept_device_inputs():
    """Check gain scale and flat accept device inputs."""
    data, err, dq = rate_arrays()
    state = PipelineState(RateCube(data, err, dq, meta()))
    _, device_out = run_both(stages.step_gain_scale, state, {},
                             rampfit_ctx())
    assert_resident(device_out.cube, 'data', 'err')

    flat = np.full((DIMY, DIMX), 1.05, np.float32)
    flat[2, 2] = np.nan
    ctx = {'opts': {}, 'refpack': {
        'flat': flat,
        'flat_err': np.full((DIMY, DIMX), .01, np.float32),
        'flat_dq': np.zeros((DIMY, DIMX), np.uint32)}}
    _, device_out = run_both(stages.step_flat, state, {}, ctx)
    assert_resident(device_out.cube, 'data', 'err', 'dq')


def test_stage1_chain_stays_resident_until_rampfit():
    """The whole supported prefix runs once with a single upload."""
    state = ramp_state()
    ctx = soss_ctx()
    ctx['opts'] = {'oof_method': 'scale-achromatic',
                   'saturation_threshold': 80,
                   'flag_up_ramp': False, 'flag_in_time': True}
    ctx['refpack'] = {
        'mask_dq': np.zeros((DIMY, DIMX), np.uint32),
        'superbias': np.full((DIMY, DIMX), 4900., np.float32),
        'lin_coeffs': np.stack([
            np.zeros((DIMY, DIMX), np.float32),
            np.full((DIMY, DIMX), 1.0001, np.float32),
            np.full((DIMY, DIMX), 1e-9, np.float32)]),
        'lin_dq': np.zeros((DIMY, DIMX), np.uint32),
        'readnoise': np.full((DIMY, DIMX), 6., np.float32),
        'gain': np.full((DIMY, DIMX), 1.6, np.float32),
        'gain_factor': 1.05,
    }
    params = {'soss_inner_mask_width': 5., 'soss_outer_mask_width': 9.,
              'time_jump_threshold': 5., 'time_window': 3}
    prefix = (stages.step_dq_init, stages.step_superbias,
              stages.step_background_grp, stages.step_oneoverf_grp,
              stages.step_linearity, stages.step_jump)

    host, resident = state, device_state(state)
    for step in prefix:
        host = step(host, params, ctx)
        resident = step(resident, params, ctx)
        assert_resident(resident.cube, 'data', 'groupdq')
        assert_same_result(host, resident)

    host = stages.step_rampfit(host, params, ctx)
    resident = stages.step_rampfit(resident, params, ctx)
    assert_same_result(host, resident)
    host = stages.step_gain_scale(host, params, ctx)
    resident = stages.step_gain_scale(resident, params, ctx)
    assert_same_result(host, resident)
