"""Stage-2 (and NIRSpec/MIRI) steps must accept device-resident cubes."""

import numpy as np
import jax.numpy as jnp
import pytest

from exotedrf.v2 import core, stages
from exotedrf.v2.core import ObsMeta, RampCube, RateCube
from exotedrf.v2.kernels import pca as k_pca
from exotedrf.v2.pipeline import PipelineState


# Shared helpers (same shape as tests/v2/test_stage1_device_inputs.py).

@pytest.fixture(autouse=True)
def pinned_memory_budgets(monkeypatch):
    """Make every chunk-size decision independent of the shared machine.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for temporary replacements.
    """
    monkeypatch.setenv('EXOTEDRF_MAX_DEVICE_BYTES', str(8 << 30))
    monkeypatch.setattr(core, 'host_allocation_budget_bytes',
                        lambda explicit=None: 8 << 30)


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


def assert_equal_tree(produced, reference, label):
    """Bit-for-bit equality of an array, or of a record containing arrays.

    Parameters
    ----------
    produced : object
        Calculated array or nested record.
    reference : object
        Expected array or nested record.
    label : str
        Label used in assertion messages.
    """
    if reference is None:
        assert produced is None, label
        return
    if isinstance(reference, dict):
        assert set(reference) == set(produced), label
        for key, value in reference.items():
            assert_equal_tree(produced[key], value, f'{label}[{key!r}]')
        return
    if isinstance(reference, (list, tuple)):
        assert len(produced) == len(reference), label
        for index, value in enumerate(reference):
            assert_equal_tree(produced[index], value, f'{label}[{index}]')
        return
    if isinstance(reference, (str, bool, int, float, np.generic)) and \
            not isinstance(reference, np.ndarray):
        assert np.array_equal(np.asarray(produced), np.asarray(reference)), \
            label
        return
    actual, expected = np.asarray(produced), np.asarray(reference)
    assert actual.dtype == expected.dtype, f'{label} dtype'
    assert np.array_equal(actual, expected, equal_nan=
                          np.issubdtype(expected.dtype, np.inexact)), label


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
        assert_equal_tree(actual[name], reference, name)
    for key in aux_keys:
        assert key in device_out.aux, key
        assert_equal_tree(device_out.aux[key], host_state.aux[key], key)


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


def assert_host(cube, *names):
    """Check that the named arrays remain in host memory.

    Parameters
    ----------
    cube : RampCube, RateCube
        Input observation cube.
    names : list[str]
        Names of functions or arrays to select.
    """
    for name in names:
        assert isinstance(getattr(cube, name), np.ndarray), (
            f'{name} should have returned to system memory')


def record_host_allocations(monkeypatch):
    """Capture every host buffer a step asks ``core`` to allocate.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for temporary replacements.

    Returns
    -------
    allocations : list[tuple]
        Recorded buffer names, shapes and data types.
    """
    allocations = []
    original = core.empty_host_array

    def recording(shape, dtype, *args, **kwargs):
        allocations.append((kwargs.get('name', 'exotedrf'), tuple(shape),
                            np.dtype(dtype)))
        return original(shape, dtype, *args, **kwargs)

    monkeypatch.setattr(core, 'empty_host_array', recording)
    return allocations


def assert_no_observation_buffers(allocations, shape):
    """No observation-sized SCIENCE buffer may be allocated for a resident cube.

    Parameters
    ----------
    allocations : list
        Recorded host allocations.
    shape : tuple[int]
        Observation array shape.
    """
    leaked = sorted({name for name, allocated, dtype in allocations
                     if allocated == tuple(shape) and dtype.itemsize > 1})
    assert not leaked, f'a resident cube round-tripped through {leaked}'


# SOSS stage 2: fixtures.

NINTS, DIMY, DIMX = 14, 10, 12


def soss_meta(nints=NINTS, edges=None, starts=None, dimx=DIMX):
    """Create metadata for a synthetic SOSS observation.

    Parameters
    ----------
    nints : int
        Number of integrations.
    edges : None, array-like(int)
        Integration boundaries between segments.
    starts : None, array-like(int)
        First integration index of each segment.
    dimx : int
        Number of detector columns.

    Returns
    -------
    meta : ObsMeta
        Synthetic SOSS metadata.
    """
    edges = np.asarray(edges if edges is not None else [nints])
    extra = {} if starts is None else {'segment_int_starts': tuple(starts)}
    return ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP96', 2.214, 3,
                   np.arange(nints, dtype=float), np.array([nints]), edges,
                   tuple(f'seg{i}.fits' for i in range(len(edges))), extra)


def soss_rate_state(edges=None, starts=None, err_nan=False, aux=None,
                    seed=0x135283a):
    """A small rate cube with a trace, defects and a transit-like series.

    Parameters
    ----------
    edges : None, array-like(int)
        Integration boundaries between segments.
    starts : None, array-like(int)
        First integration index of each segment.
    err_nan : bool
        Err nan option.
    aux : dict
        Additional reduction options or results.
    seed : int
        Random seed.

    Returns
    -------
    state : PipelineState
        Synthetic observation state.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(100., .5, (NINTS, DIMY, DIMX)).astype(np.float32)
    data[:, 4:6] += 800.
    data *= np.linspace(1., 1.004, NINTS, dtype=np.float32)[:, None, None]
    data[7, 2, 3] += 400.
    data[:, 8, 9] += 300.
    err = np.full_like(data, 2.)
    if err_nan:
        err[3, 1, 1] = np.nan
    dq = np.zeros(data.shape, np.uint32)
    dq[:, 0, 1] = core.DQ_HOT
    dq[5, 6, 7] = core.DQ_SATURATED
    cube = RateCube(data, err, dq, soss_meta(edges=edges, starts=starts))
    return PipelineState(cube, dict(aux or {}))


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
        'background_model': np.full((DIMY, DIMX), .25, np.float32),
        'centroids': {'ypos o1': np.full(DIMX, 4.5, np.float32)},
        'soss_timeseries': np.linspace(.999, 1.001, NINTS).astype(np.float32),
        'soss_timeseries_o2': None,
        'outlier_mask': None,
        'order0_mask': None,
        'waves': {1: np.linspace(.8, 2.6, DIMX)},
    }
    ctx.update(updates)
    return ctx


# SOSS stage 2: background, 1/f, bad pixels, PCA.

def test_background_int_accepts_device_inputs():
    """Check background int accepts device inputs."""
    state = soss_rate_state()
    _, device_out = run_both(
        stages.step_background_int, state, {}, soss_ctx(),
        aux_keys=('bkg_int', stages._OOF_INT_DEEP_KEY))
    assert_resident(device_out.cube, 'data')
    # The scaled model and the tracing deep stack stay host artefacts.
    assert isinstance(device_out.aux['bkg_int'], np.ndarray)
    assert isinstance(device_out.aux[stages._OOF_INT_DEEP_KEY], np.ndarray)


@pytest.mark.parametrize('edges,starts', [(None, None), ([6, 14], [1, 7])])
def test_oneoverf_int_accepts_device_inputs(edges, starts):
    """Check 1/f correction int accepts device inputs."""
    state = soss_rate_state(edges=edges, starts=starts)
    params = {'soss_inner_mask_width': 5., 'soss_outer_mask_width': 9.}
    host_out, device_out = run_both(
        stages.step_oneoverf_int, state, params, soss_ctx())
    assert_resident(device_out.cube, 'data', 'err', 'dq')
    # The correction really does something, so the comparison has bite.
    assert not np.array_equal(np.asarray(host_out.cube.data),
                              np.asarray(state.cube.data))


def test_oneoverf_int_window_method_accepts_device_inputs():
    """Check 1/f correction int window method accepts device inputs."""
    state = soss_rate_state()
    ctx = soss_ctx(opts={'oof_method': 'scale-achromatic-window'},
                   centroids={'ypos o1': np.full(DIMX, 4.5, np.float32),
                              'ypos o2': np.full(DIMX, 8., np.float32)})
    params = {'soss_inner_mask_width': 5., 'soss_outer_mask_width': 9.}
    _, device_out = run_both(stages.step_oneoverf_int, state, params, ctx)
    assert_resident(device_out.cube, 'data')


def test_oneoverf_int_chromatic_method_accepts_device_inputs():
    """Check 1/f correction int chromatic method accepts device inputs."""
    state = soss_rate_state()
    series = np.broadcast_to(
        np.linspace(.999, 1.001, NINTS, dtype=np.float32)[:, None],
        (NINTS, DIMX)).copy()
    ctx = soss_ctx(opts={'oof_method': 'scale-chromatic'},
                   centroids={'ypos o1': np.full(DIMX, 4.5, np.float32),
                              'ypos o2': np.full(DIMX, 8., np.float32)},
                   soss_timeseries=series, soss_timeseries_o2=series)
    params = {'soss_inner_mask_width': 5., 'soss_outer_mask_width': 9.}
    _, device_out = run_both(stages.step_oneoverf_int, state, params, ctx)
    assert_resident(device_out.cube, 'data')


BADPIX_PARAMS = {'space_outlier_threshold': 10., 'time_outlier_threshold': 5.,
                 'box_size': 2, 'window_size': 3}


@pytest.mark.parametrize('edges,starts', [(None, None), ([6, 14], [1, 7])])
def test_badpix_accepts_device_inputs(edges, starts):
    """Check bad-pixel correction accepts device inputs."""
    state = soss_rate_state(edges=edges, starts=starts)
    host_out, device_out = run_both(
        stages.step_badpix, state, BADPIX_PARAMS, {'opts': {}},
        aux_keys=('hot_pixel_map', 'deepframe'))
    assert_resident(device_out.cube, 'data', 'err', 'dq')
    assert isinstance(host_out.cube.data, np.ndarray)
    # Something was actually replaced, so equality is not vacuous.
    assert not np.array_equal(np.asarray(host_out.cube.data),
                              np.asarray(state.cube.data))
    # BadPix never rewrites finite errors: the input cube is reused.
    assert device_out.cube.err is device_state(state).cube.err or \
        np.array_equal(np.asarray(device_out.cube.err),
                       np.asarray(state.cube.err))


def test_badpix_nan_error_fill_matches_for_both_residencies():
    """v1's global NaN-error median must not depend on where err lives."""
    state = soss_rate_state(edges=[6, 14], starts=[1, 7], err_nan=True)
    host_out, device_out = run_both(
        stages.step_badpix, state, BADPIX_PARAMS, {'opts': {}})
    assert_resident(device_out.cube, 'err')
    assert np.isfinite(np.asarray(host_out.cube.err)).all()


def test_badpix_plot_categories_accept_device_inputs():
    """Check bad-pixel correction plot categories accept device inputs."""
    state = soss_rate_state()
    run_both(stages.step_badpix, state, BADPIX_PARAMS,
             {'opts': {'do_plots': True}},
             aux_keys=('badpix_plot_hot', 'badpix_plot_nan',
                       'badpix_plot_other'))


PCA_AUX = ('pca_components', 'pca_eigvals', 'pca_wlc',
           'pca_components_reconstructed', 'pca_eigvals_reconstructed',
           'stage3_deepframe')


@pytest.mark.parametrize('remove', [None, [1]])
def test_pca_accepts_device_inputs(remove):
    """Check PCA accepts device inputs."""
    state = soss_rate_state()
    ctx = {'opts': {'pca_components': 3, 'remove_components': remove}}
    host_out, device_out = run_both(stages.step_pca, state, {}, ctx,
                                    aux_keys=PCA_AUX)
    assert_resident(device_out.cube, 'data')
    assert isinstance(host_out.cube.data, np.ndarray)
    if remove is not None:
        assert not np.array_equal(np.asarray(host_out.cube.data),
                                  np.asarray(state.cube.data))


def test_pca_plot_products_accept_device_inputs():
    """Check PCA plot products accept device inputs."""
    state = soss_rate_state()
    ctx = {'opts': {'pca_components': 3, 'remove_components': [1],
                    'do_plots': True}}
    run_both(stages.step_pca, state, {}, ctx,
             aux_keys=PCA_AUX + ('pca_projections',
                                 'pca_projections_reconstructed'))


@pytest.mark.parametrize('solver', ['exact', 'randomized'])
def test_pca_kernel_fit_is_identical_for_resident_cubes(solver):
    """Check PCA kernel fit is identical for resident cubes."""
    rng = np.random.default_rng(4)
    cube = rng.normal(50., 1., (12, 6, 9)).astype(np.float32)
    cube[3, 2, 2] = np.nan
    host_cube, host_fit = k_pca._fit_streamed(cube, 3, solver=solver)
    resident_cube, resident_fit = k_pca._fit_streamed(
        jnp.asarray(cube), 3, solver=solver)
    assert core.is_device_array(resident_cube)
    assert np.array_equal(np.asarray(resident_cube), host_cube,
                          equal_nan=True)
    for name in ('vectors', 'variance_ratio', 'mean'):
        assert np.array_equal(getattr(resident_fit, name),
                              getattr(host_fit, name)), name
    assert np.array_equal(resident_fit.fill_value, host_fit.fill_value)


def test_pca_kernel_reconstruction_is_identical_for_resident_cubes(
        monkeypatch):
    """Every column chunk is joined on the device, including several."""
    monkeypatch.setenv('EXOTEDRF_V2_PCA_COLUMN_CHUNK', '3')
    monkeypatch.setattr(k_pca, '_COLUMN_CHUNK_SIZE', 3)
    rng = np.random.default_rng(5)
    cube = rng.normal(50., 1., (12, 6, 9)).astype(np.float32)
    remove = np.array([True, False, False])
    baseline = np.zeros(12, bool)
    baseline[:4] = True
    host = k_pca.pca_reconstruction(cube, remove, baseline, n_components=3)
    resident = k_pca.pca_reconstruction(
        jnp.asarray(cube), remove, baseline, n_components=3)
    assert core.is_device_array(resident[0])
    for index, reference in enumerate(host):
        assert np.array_equal(np.asarray(resident[index]),
                              np.asarray(reference)), index


# SOSS stage 3: extraction.

def test_extract_orders_accepts_device_inputs():
    """Check extract orders accepts device inputs."""
    state = soss_rate_state()
    ctx = soss_ctx()
    params = {'extract_width': 4.}
    host = stages.extract_orders(state, params, ctx, mode='sum')
    resident = stages.extract_orders(device_state(state), params, ctx,
                                     mode='sum')
    assert core.is_device_array(resident[1][0])
    for order, (flux, ferr) in host.items():
        assert_equal_tree(resident[order][0], flux, f'order {order} flux')
        assert_equal_tree(resident[order][1], ferr, f'order {order} ferr')


def test_step_extract_products_accept_device_inputs():
    """Check step extract products accept device inputs."""
    state = soss_rate_state()
    ctx = soss_ctx()
    params = {'extract_width': 4.}
    _, device_out = run_both(stages.step_extract, state, params, ctx,
                             aux_keys=('spectra', 'spectral_products'))
    # Final spectra are small host products by contract.
    flux, _ = device_out.aux['spectra'][1]
    assert isinstance(flux, np.ndarray)
    assert_resident(device_out.cube, 'data')


def test_prepared_extract_results_accept_device_inputs():
    """Check prepared extract results accept device inputs."""
    state = soss_rate_state()
    ctx = soss_ctx()
    params = {'extract_width': 4.}
    candidates = [3., 4., 5.]
    host = stages.prepared_extract_results(state, params, ctx, candidates)
    resident = stages.prepared_extract_results(
        device_state(state), params, ctx, candidates)
    assert len(host) == len(resident)
    for index, (cost, scatter, _) in enumerate(host):
        assert np.array_equal(resident[index][0], cost), index
        assert_equal_tree(resident[index][1], scatter, f'scatter {index}')


def test_resident_soss_stage2_chain_never_lands_in_host_buffers(monkeypatch):
    """The whole two-segment stage-2 chain runs with one upload."""
    state = device_state(soss_rate_state(edges=[6, 14], starts=[1, 7]))
    ctx = soss_ctx()
    params = {'soss_inner_mask_width': 5., 'soss_outer_mask_width': 9.,
              'extract_width': 4., **BADPIX_PARAMS}
    allocations = record_host_allocations(monkeypatch)
    out = stages.step_background_int(state, params, ctx)
    out = stages.step_oneoverf_int(out, params, ctx)
    out = stages.step_badpix(out, params, {'opts': {}})
    out = stages.step_pca(out, params, {
        'opts': {'pca_components': 3, 'remove_components': [1]}})
    assert_resident(out.cube, 'data', 'err', 'dq')
    out = stages.step_extract(out, params, ctx)
    assert_no_observation_buffers(allocations, state.cube.data.shape)


# NIRSpec.

NRS_NINTS, NRS_NGROUPS, NRS_DIMY, NRS_DIMX = 12, 3, 10, 16


def nirspec_meta(nints=NRS_NINTS, edges=None, starts=None):
    """Create metadata for a synthetic NIRSpec observation.

    Parameters
    ----------
    nints : int
        Number of integrations.
    edges : None, array-like(int)
        Integration boundaries between segments.
    starts : None, array-like(int)
        First integration index of each segment.

    Returns
    -------
    meta : ObsMeta
        Synthetic NIRSpec metadata.
    """
    edges = np.asarray(edges if edges is not None else [nints])
    if starts is None:
        starts = tuple([1] + [int(edge) + 1 for edge in edges[:-1]])
    header = {'EXP_TYPE': 'NRS_BRIGHTOBJ', 'GRATING': 'PRISM',
              'NFRAMES': 1, 'TFRAME': 0.902, 'TGROUP': 0.902,
              'NINTS': nints}
    return ObsMeta(
        'NIRSpec/PRISM', 'NRS1', 'SUB512', 0.902, NRS_NGROUPS,
        60000. + np.arange(nints) * 1e-4, np.asarray([4]), edges,
        tuple(f'seg{i}.fits' for i in range(len(edges))),
        {'header': header,
         'segment_headers': tuple(header for _ in edges),
         'segment_int_starts': tuple(starts)})


def nirspec_ramp_state(edges=None, starts=None):
    """Create a synthetic NIRSpec ramp state.

    Parameters
    ----------
    edges : None, array-like(int)
        Integration boundaries between segments.
    starts : None, array-like(int)
        First integration index of each segment.

    Returns
    -------
    state : PipelineState
        Synthetic observation state.
    """
    rng = np.random.default_rng(99)
    yy, xx = np.mgrid[:NRS_DIMY, :NRS_DIMX]
    profile = np.exp(-0.5 * ((yy - 4.) / 1.5) ** 2)
    data = np.empty((NRS_NINTS, NRS_NGROUPS, NRS_DIMY, NRS_DIMX), np.float32)
    for integration in range(NRS_NINTS):
        for group in range(NRS_NGROUPS):
            data[integration, group] = (
                100. + group * (25. + profile * 80.) + .01 * xx +
                rng.normal(0., .2, (NRS_DIMY, NRS_DIMX)))
    groupdq = np.zeros(data.shape, np.uint8)
    groupdq[2, -1, 3, 3] = np.uint8(core.DQ_DO_NOT_USE)
    pixeldq = np.zeros((NRS_DIMY, NRS_DIMX), np.uint32)
    pixeldq[0, 0] = core.DQ_HOT
    cube = RampCube(data, groupdq, pixeldq,
                    nirspec_meta(edges=edges, starts=starts))
    return PipelineState(cube)


def nirspec_rate_state(edges=None, starts=None):
    """Create a synthetic NIRSpec rate state.

    Parameters
    ----------
    edges : None, array-like(int)
        Integration boundaries between segments.
    starts : None, array-like(int)
        First integration index of each segment.

    Returns
    -------
    state : PipelineState
        Synthetic observation state.
    """
    ramp = nirspec_ramp_state(edges=edges, starts=starts).cube
    data = np.ascontiguousarray(ramp.data[:, -1])
    cube = RateCube(data, np.full_like(data, 2.),
                    np.zeros(data.shape, np.uint32), ramp.meta)
    return PipelineState(cube)


def nirspec_centroids():
    """Return synthetic NIRSpec trace coordinates.

    Returns
    -------
    result : dict
        Calculation results and associated metadata.
    """
    return {'xpos': np.arange(NRS_DIMX, dtype=float),
            'ypos': np.full(NRS_DIMX, 4., dtype=float)}


def nirspec_ctx(**updates):
    """Create the NIRSpec reduction context with the supplied overrides.

    Parameters
    ----------
    updates : dict
        Overrides for the returned configuration.

    Returns
    -------
    context : dict
        Reduction options and reference arrays.
    """
    ctx = {'opts': {'oof_method': 'median'},
           'centroids': nirspec_centroids(),
           'nirspec_xstart': 0,
           'outlier_mask': None,
           'order0_mask': None,
           'nirspec_wave_map': np.broadcast_to(
               np.linspace(3., 5., NRS_DIMX)[None, :],
               (NRS_DIMY, NRS_DIMX)).copy()}
    ctx.update(updates)
    return ctx


@pytest.mark.parametrize('method', ['crds', 'custom', 'custom-rescale'])
def test_nirspec_superbias_accepts_device_inputs(method):
    """Check NIRSpec superbias accepts device inputs."""
    state = nirspec_ramp_state(edges=[6, 12], starts=[1, 7])
    ctx = nirspec_ctx(opts={'superbias_method': method},
                      refpack={'superbias': np.full(
                          (NRS_DIMY, NRS_DIMX), 95., np.float32),
                          'superbias_dq': np.zeros(
                              (NRS_DIMY, NRS_DIMX), np.uint32)})
    aux_keys = ()
    if method != 'crds':
        aux_keys = ('superbias_custom',)
    if method == 'custom-rescale':
        aux_keys += ('superbias_scale_factors',)
    _, device_out = run_both(stages.step_superbias_nirspec, state, {}, ctx,
                             aux_keys=aux_keys)
    assert_resident(device_out.cube, 'data')


def test_dark_accepts_device_inputs():
    """Check dark accepts device inputs."""
    state = nirspec_ramp_state()
    ctx = {'opts': {}, 'refpack': {
        'dark': np.full((NRS_NGROUPS, NRS_DIMY, NRS_DIMX), .5, np.float32),
        'dark_dq': np.zeros((NRS_DIMY, NRS_DIMX), np.uint32)}}
    _, device_out = run_both(stages.step_dark, state, {}, ctx)
    assert_resident(device_out.cube, 'data', 'groupdq')


@pytest.mark.parametrize('edges,starts', [(None, None), ([6, 12], [1, 7])])
def test_nirspec_oneoverf_grp_accepts_device_inputs(edges, starts):
    """Check NIRSpec 1/f correction grp accepts device inputs."""
    state = nirspec_ramp_state(edges=edges, starts=starts)
    _, device_out = run_both(
        stages.step_oneoverf_grp_nirspec, state,
        {'nirspec_mask_width': 6.}, nirspec_ctx())
    assert_resident(device_out.cube, 'data', 'groupdq')


def test_nirspec_oneoverf_grp_retraces_each_segment_on_device():
    """The short-NRS1 lifecycle joins its per-file results on the device."""
    state = nirspec_ramp_state(edges=[6, 12], starts=[1, 7])
    short = {'xpos': np.arange(1, NRS_DIMX, dtype=float),
             'ypos': np.full(NRS_DIMX - 1, 4., dtype=float)}
    ctx = nirspec_ctx(centroids=short)
    _, device_out = run_both(
        stages.step_oneoverf_grp_nirspec, state,
        {'nirspec_mask_width': 6.}, ctx)
    assert_resident(device_out.cube, 'data')


@pytest.mark.parametrize('edges,starts', [(None, None), ([6, 12], [1, 7])])
def test_nirspec_oneoverf_int_accepts_device_inputs(edges, starts):
    """Check NIRSpec 1/f correction int accepts device inputs."""
    state = nirspec_rate_state(edges=edges, starts=starts)
    _, device_out = run_both(
        stages.step_oneoverf_int_nirspec, state,
        {'nirspec_mask_width': 6.}, nirspec_ctx())
    assert_resident(device_out.cube, 'data', 'err', 'dq')


def test_nirspec_header_only_steps_do_not_touch_the_cube():
    """Check NIRSpec header only steps do not touch the cube."""
    state = device_state(nirspec_rate_state())
    ctx = nirspec_ctx(nirspec_wave_map=np.ones((NRS_DIMY, NRS_DIMX)),
                      wavemap_provenance='test')
    for step in (stages.step_assign_wcs_nirspec,
                 stages.step_extract2d_nirspec,
                 stages.step_wavecorr_nirspec):
        out = step(state, {}, ctx)
        assert out.cube.data is state.cube.data
        assert_resident(out.cube, 'data', 'err', 'dq')


def test_nirspec_badpix_accepts_device_inputs():
    """Check NIRSpec bad-pixel correction accepts device inputs."""
    state = nirspec_rate_state()
    _, device_out = run_both(stages.step_badpix, state, BADPIX_PARAMS,
                             {'opts': {}}, aux_keys=('hot_pixel_map',))
    assert_resident(device_out.cube, 'data', 'err', 'dq')


def test_nirspec_extract_accepts_device_inputs():
    """Check NIRSpec extract accepts device inputs."""
    state = nirspec_rate_state()
    ctx = nirspec_ctx(opts={'mask_do_not_use_pixels': True})
    _, device_out = run_both(
        stages.step_extract_nirspec, state, {'extract_width': 4.}, ctx,
        aux_keys=('spectra', 'spectral_products'))
    assert_resident(device_out.cube, 'data')


def test_nirspec_prepared_oneoverf_scorer_accepts_device_inputs():
    """Check NIRSpec prepared 1/f correction scorer accepts device inputs."""
    state = nirspec_ramp_state()
    params = {'nirspec_mask_width': 6., 'extract_width': 4.}
    ctx = nirspec_ctx(opts={'oof_method': 'median', 'wave_range': None,
                            'w1': 1., 'w2': 1.})
    host = stages.prepare_nirspec_oneoverf_scorer(state, params, ctx)
    resident = stages.prepare_nirspec_oneoverf_scorer(
        device_state(state), params, ctx)
    if host is None or resident is None:
        pytest.skip('prepared scorer unavailable on this host')
    candidates = [4., 6.]
    expected = host.evaluate_candidates(
        'nirspec_mask_width', candidates, params)
    produced = resident.evaluate_candidates(
        'nirspec_mask_width', candidates, params)
    for index, (cost, scatter, _) in enumerate(expected):
        assert np.array_equal(produced[index][0], cost), index
        assert_equal_tree(produced[index][1], scatter, f'scatter {index}')


def test_resident_nirspec_chain_never_lands_in_host_buffers(monkeypatch):
    """Superbias, both 1/f levels and BadPix keep one resident observation."""
    state = device_state(nirspec_ramp_state())
    ctx = nirspec_ctx(opts={'oof_method': 'median',
                            'superbias_method': 'custom-rescale'})
    params = {'nirspec_mask_width': 6., **BADPIX_PARAMS}
    allocations = record_host_allocations(monkeypatch)
    out = stages.step_superbias_nirspec(state, params, ctx)
    out = stages.step_oneoverf_grp_nirspec(out, params, ctx)
    assert_resident(out.cube, 'data', 'groupdq')
    assert_no_observation_buffers(allocations, state.cube.data.shape)

    rate = device_state(nirspec_rate_state())
    allocations = record_host_allocations(monkeypatch)
    out = stages.step_oneoverf_int_nirspec(rate, params, ctx)
    out = stages.step_badpix(out, params, {'opts': {}})
    assert_resident(out.cube, 'data', 'err', 'dq')
    assert_no_observation_buffers(allocations, rate.cube.data.shape)


# MIRI.

MIRI_NINTS, MIRI_NGROUPS, MIRI_DIMY, MIRI_DIMX = 12, 7, 12, 16


def miri_meta(nints=MIRI_NINTS, edges=None, starts=None):
    """Create metadata for a synthetic MIRI observation.

    Parameters
    ----------
    nints : int
        Number of integrations.
    edges : None, array-like(int)
        Integration boundaries between segments.
    starts : None, array-like(int)
        First integration index of each segment.

    Returns
    -------
    meta : ObsMeta
        Synthetic MIRI metadata.
    """
    edges = np.asarray([nints] if edges is None else edges, dtype=int)
    if starts is None:
        starts = tuple([1] + [int(edge) + 1 for edge in edges[:-1]])
    header = {'TGROUP': 2., 'TFRAME': 2., 'NFRAMES': 1, 'GROUPGAP': 0,
              'NINTS': nints, 'READPATT': 'FASTR1', 'NSAMPLES': 1,
              'SUBSTRT1': 1}
    return ObsMeta(
        'MIRI/LRS', 'MIRIMAGE', 'SLITLESSPRISM', 2., MIRI_NGROUPS,
        60000. + np.arange(nints) * 1e-4, np.asarray([nints]), edges,
        tuple(f'seg{i}.fits' for i in range(len(edges))),
        {'header': header, 'segment_headers': tuple(header for _ in edges),
         'segment_int_starts': tuple(starts), 'exposure_nints': nints})


def miri_ramp_state(edges=None, starts=None):
    """Create a synthetic MIRI ramp state.

    Parameters
    ----------
    edges : None, array-like(int)
        Integration boundaries between segments.
    starts : None, array-like(int)
        First integration index of each segment.

    Returns
    -------
    state : PipelineState
        Synthetic observation state.
    """
    rng = np.random.default_rng(1234)
    yy, xx = np.mgrid[:MIRI_DIMY, :MIRI_DIMX]
    slope = 20. + 60. * np.exp(-0.5 * ((xx - 8.) / 1.2) ** 2)
    group = np.arange(MIRI_NGROUPS, dtype=np.float32)[None, :, None, None]
    data = (100. + group * slope[None, None] +
            rng.normal(0., .2, (MIRI_NINTS, MIRI_NGROUPS, MIRI_DIMY,
                                MIRI_DIMX))).astype(np.float32)
    groupdq = np.zeros(data.shape, np.uint8)
    groupdq[:, 0] = np.uint8(core.DQ_DO_NOT_USE)
    groupdq[:, -1] = np.uint8(core.DQ_DO_NOT_USE)
    pixeldq = np.zeros((MIRI_DIMY, MIRI_DIMX), np.uint32)
    cube = RampCube(data, groupdq, pixeldq,
                    miri_meta(edges=edges, starts=starts))
    return PipelineState(cube)


def miri_rate_state(edges=None, starts=None):
    """Create a synthetic MIRI rate state.

    Parameters
    ----------
    edges : None, array-like(int)
        Integration boundaries between segments.
    starts : None, array-like(int)
        First integration index of each segment.

    Returns
    -------
    state : PipelineState
        Synthetic observation state.
    """
    ramp = miri_ramp_state(edges=edges, starts=starts).cube
    data = np.ascontiguousarray(ramp.data[:, -2])
    cube = RateCube(data, np.full_like(data, 2.),
                    np.zeros(data.shape, np.uint32), ramp.meta)
    return PipelineState(cube)


def miri_centroids():
    """Return synthetic MIRI trace coordinates.

    Returns
    -------
    result : dict
        Calculation results and associated metadata.
    """
    return {'xpos': np.full(8, 8., dtype=float),
            'ypos': np.arange(2., 10.)}


def miri_ctx(**updates):
    """Create the MIRI reduction context with the supplied overrides.

    Parameters
    ----------
    updates : dict
        Overrides for the returned configuration.

    Returns
    -------
    context : dict
        Reduction options and reference arrays.
    """
    ctx = {'opts': {'mask_do_not_use_pixels': True},
           'centroids': miri_centroids(),
           'miri_wave_map': np.broadcast_to(
               np.linspace(5., 12., MIRI_DIMY)[:, None],
               (MIRI_DIMY, MIRI_DIMX)).copy(),
           'outlier_mask': None, 'order0_mask': None}
    ctx.update(updates)
    return ctx


def test_miri_emicorr_returns_the_residency_it_was_given():
    """Read one segment at a time for NumPy EMI correction."""
    state = miri_ramp_state(edges=[6, 12], starts=[1, 7])
    ctx = {'refpack': {
        'emicorr_frequencies': np.array([390.625]),
        'emicorr_reference_waves': np.array([[0., 1., 0., -1.]]),
        'emicorr_reference_wave_lengths': np.array([4]),
        'emicorr_rowclocks': np.int64(28),
        'emicorr_frameclocks': np.int64(15904)}}
    host_out, device_out = run_both(
        stages.step_emicorr_miri, state, {}, ctx, aux_keys=('emicorr',))
    assert_resident(device_out.cube, 'data')
    assert_host(host_out.cube, 'data')


@pytest.mark.parametrize('edges,starts', [(None, None), ([6, 12], [1, 7])])
def test_miri_reset_accepts_device_inputs(edges, starts):
    """Check MIRI reset accepts device inputs."""
    state = miri_ramp_state(edges=edges, starts=starts)
    reset_data = np.stack([
        np.full((MIRI_NGROUPS, MIRI_DIMY, MIRI_DIMX), value, np.float32)
        for value in (10., 20., 30.)])
    ctx = {'refpack': {'reset_data': reset_data,
                       'reset_dq': np.full((MIRI_DIMY, MIRI_DIMX),
                                           core.DQ_PERSISTENCE, np.uint32)}}
    _, device_out = run_both(stages.step_reset_miri, state, {}, ctx)
    assert_resident(device_out.cube, 'data')
    assert isinstance(device_out.cube.pixeldq, np.ndarray)


def test_miri_linearity_and_dark_accept_device_inputs():
    """Check MIRI linearity and dark accept device inputs."""
    state = miri_ramp_state(edges=[6, 12], starts=[1, 7])
    coeffs = np.zeros((3, MIRI_DIMY, MIRI_DIMX), np.float32)
    coeffs[1] = 1.0001
    coeffs[2] = 1e-9
    dark = np.stack([
        np.full((MIRI_NGROUPS, MIRI_DIMY, MIRI_DIMX), value, np.float32)
        for value in (1., 2.)])
    ctx = {'opts': {'miri_subtract_dark': True, 'miri_drop_groups': 2},
           'refpack': {'lin_coeffs': coeffs,
                       'lin_dq': np.zeros((MIRI_DIMY, MIRI_DIMX), np.uint32),
                       'dark': dark,
                       'dark_dq': np.zeros((MIRI_DIMY, MIRI_DIMX),
                                           np.uint32)}}
    _, device_out = run_both(stages.step_linearity_miri, state, {}, ctx,
                             aux_keys=('miri_dark_subtracted',))
    assert_resident(device_out.cube, 'data', 'groupdq')


def test_miri_jump_accepts_device_inputs():
    """Check MIRI jump accepts device inputs."""
    state = miri_ramp_state(edges=[6, 12], starts=[1, 7])
    params = {'time_jump_threshold': 5., 'time_window': 3}
    ctx = {'opts': {'flag_up_ramp': False, 'flag_in_time': True}}
    _, device_out = run_both(stages.step_jump, state, params, ctx)
    assert_resident(device_out.cube, 'data', 'groupdq')


def test_miri_score_group_selects_the_same_group_for_both_residencies():
    """Check MIRI score group selects the same group for both residencies."""
    state = miri_ramp_state()
    cube = state.cube
    ctx = {'opts': {'wave_range': None, 'w1': 0., 'w2': 1.},
           'centroids': miri_centroids()}
    params = {'extract_width': 4.}
    expected = stages.miri_score_group(
        cube.data, cube.groupdq, params, ctx, cube.meta, ctx['centroids'])
    produced = stages.miri_score_group(
        jnp.asarray(cube.data), jnp.asarray(cube.groupdq), params, ctx,
        cube.meta, ctx['centroids'])
    assert produced == expected == MIRI_NGROUPS - 2


def test_miri_background_accepts_device_inputs():
    """Check MIRI background accepts device inputs."""
    state = miri_rate_state()
    ctx = miri_ctx(opts={'miri_background_method': 'median'},
                   miri_trace_center=8)
    params = {'miri_trace_width': 6., 'miri_background_width': 10.}
    _, device_out = run_both(stages.step_background_miri, state, params, ctx)
    assert_resident(device_out.cube, 'data', 'err', 'dq')


def test_miri_badpix_and_pca_accept_device_inputs():
    """Check MIRI bad-pixel correction and PCA accept device inputs."""
    state = miri_rate_state()
    _, device_out = run_both(stages.step_badpix, state, BADPIX_PARAMS,
                             {'opts': {}}, aux_keys=('hot_pixel_map',))
    assert_resident(device_out.cube, 'data', 'err', 'dq')

    pca_state = miri_rate_state()
    ctx = {'opts': {'pca_components': 3, 'remove_components': [1]}}
    _, device_pca = run_both(stages.step_pca, pca_state, {}, ctx,
                             aux_keys=PCA_AUX)
    assert_resident(device_pca.cube, 'data')


def test_miri_pca_column_trim_accepts_device_inputs():
    """The MIRI 12:61 window is re-inserted without leaving the device."""
    rng = np.random.default_rng(7)
    dimx = 72
    data = rng.normal(80., 1., (MIRI_NINTS, MIRI_DIMY, dimx)).astype(
        np.float32)
    meta = miri_meta()
    cube = RateCube(data, np.full_like(data, 2.),
                    np.zeros(data.shape, np.uint32), meta)
    state = PipelineState(cube)
    ctx = {'opts': {'pca_components': 3, 'remove_components': [1]}}
    host_out, device_out = run_both(stages.step_pca, state, {}, ctx,
                                    aux_keys=PCA_AUX)
    assert_resident(device_out.cube, 'data')
    untouched = np.asarray(host_out.cube.data)[:, :, :12]
    assert np.array_equal(untouched, data[:, :, :12])


@pytest.mark.parametrize('method', ['box', 'optimal'])
def test_miri_extract_accepts_device_inputs(method):
    """Check MIRI extract accepts device inputs."""
    state = miri_rate_state()
    deepframe = np.nanmedian(np.asarray(state.cube.data), axis=0)
    state = PipelineState(state.cube, {'stage3_deepframe': deepframe})
    ctx = miri_ctx(opts={'mask_do_not_use_pixels': True,
                         'extract_method': method,
                         'opt_max_iter': 2, 'opt_var_thresh': 25})
    _, device_out = run_both(
        stages.step_extract_miri, state, {'extract_width': 4.}, ctx,
        aux_keys=('spectra', 'spectral_products'))
    assert_resident(device_out.cube, 'data')


def test_resident_miri_chain_never_lands_in_host_buffers(monkeypatch):
    """Reset, linearity+dark and the stage-2 steps keep one resident ramp."""
    state = device_state(miri_ramp_state(edges=[6, 12], starts=[1, 7]))
    coeffs = np.zeros((3, MIRI_DIMY, MIRI_DIMX), np.float32)
    coeffs[1] = 1.0001
    coeffs[2] = 1e-9
    ctx = {'opts': {'miri_subtract_dark': True, 'miri_drop_groups': 2,
                    'flag_up_ramp': False, 'flag_in_time': True},
           'refpack': {
               'reset_data': np.stack([
                   np.full((MIRI_NGROUPS, MIRI_DIMY, MIRI_DIMX), value,
                           np.float32) for value in (10., 20.)]),
               'reset_dq': np.zeros((MIRI_DIMY, MIRI_DIMX), np.uint32),
               'lin_coeffs': coeffs,
               'lin_dq': np.zeros((MIRI_DIMY, MIRI_DIMX), np.uint32),
               'dark': np.stack([
                   np.full((MIRI_NGROUPS, MIRI_DIMY, MIRI_DIMX), value,
                           np.float32) for value in (1., 2.)]),
               'dark_dq': np.zeros((MIRI_DIMY, MIRI_DIMX), np.uint32)}}
    params = {'time_jump_threshold': 5., 'time_window': 3}
    allocations = record_host_allocations(monkeypatch)
    out = stages.step_reset_miri(state, params, ctx)
    out = stages.step_linearity_miri(out, params, ctx)
    out = stages.step_jump(out, params, ctx)
    assert_resident(out.cube, 'data', 'groupdq')
    assert_no_observation_buffers(allocations, state.cube.data.shape)

    rate = device_state(miri_rate_state())
    ctx = miri_ctx(opts={'miri_background_method': 'median',
                         'pca_components': 3, 'remove_components': [1]},
                   miri_trace_center=8)
    params = {'miri_trace_width': 6., 'miri_background_width': 10.,
              **BADPIX_PARAMS}
    allocations = record_host_allocations(monkeypatch)
    out = stages.step_background_miri(rate, params, ctx)
    out = stages.step_badpix(out, params, ctx)
    out = stages.step_pca(out, params, ctx)
    assert_resident(out.cube, 'data', 'err', 'dq')
    assert_no_observation_buffers(allocations, rate.cube.data.shape)


def test_miri_rampfit_and_flat_accept_device_inputs():
    """The shared stage-1 tail is exercised in MIRI's own geometry too."""
    state = miri_ramp_state()
    ctx = {'opts': {}, 'refpack': {
        'readnoise': np.full((MIRI_DIMY, MIRI_DIMX), 6., np.float32),
        'gain': np.full((MIRI_DIMY, MIRI_DIMX), 1.6, np.float32),
        'gain_factor': 1.05,
        'flat': np.full((MIRI_DIMY, MIRI_DIMX), 1.05, np.float32),
        'flat_err': np.full((MIRI_DIMY, MIRI_DIMX), .01, np.float32),
        'flat_dq': np.zeros((MIRI_DIMY, MIRI_DIMX), np.uint32)}}
    host_out, device_out = run_both(stages.step_rampfit, state, {}, ctx)
    assert_host(device_out.cube, 'data', 'err', 'dq')
    run_both(stages.step_gain_scale, host_out, {}, ctx)
    run_both(stages.step_flat, host_out, {}, ctx)
