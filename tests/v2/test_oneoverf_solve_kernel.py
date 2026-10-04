"""Check SOSS 1/f solving against v1 reference calculations."""
import ast
import copy
import warnings
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits

from exotedrf.v2.kernels import oneoverf

REPO = Path(__file__).resolve().parents[2]


# Actual v1 source, loaded without jwst.

def _load_functions(relative, names, namespace):
    """Return load functions."""
    path = REPO / relative
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'),
         namespace)
    return namespace


class Model(SimpleNamespace):
    """Minimal datamodel stand-in accepted by v1's open_filetype paths."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _v1(subarray):
    """Return v1."""
    helpers = {
        'np': np, 'warnings': warnings, 'fits': fits,
        'fancyprint': lambda *a, **k: None,
        'open_filetype': lambda value: value,
        'get_instrument_name': lambda value: 'NIRISS',
        'get_soss_subarray': lambda value: subarray,
    }
    _load_functions('exotedrf/utils.py',
                    ['line_mle', 'make_soss_tracemask', 'mask_reset_artifact'],
                    helpers)
    utils = SimpleNamespace(**helpers)
    namespace = _load_functions(
        'exotedrf/stage1.py', ['oneoverfstep_solve'],
        {'np': np, 'copy': copy, 'utils': utils, 'tqdm': lambda x: x,
         'fancyprint': lambda *a, **k: None})
    return namespace['oneoverfstep_solve'], utils


# NumPy reference transcribed from v1.

def ref_line_mle(x, y, e):
    """Fit a weighted straight line with the v1 equations.

    Parameters
    ----------
    x : array-like(float)
        X array.
    y : array-like(float)
        Y array.
    e : array-like(float)
        E array.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        sx_e = np.nansum(x[::2] / e[::2]**2, axis=0)
        sxx_e = np.nansum((x[::2] / e[::2])**2, axis=0)
        sy_e = np.nansum(y[::2] / e[::2]**2, axis=0)
        sxy_e = np.nansum(x[::2] * y[::2] / e[::2]**2, axis=0)
        s_e = np.nansum(1 / e[::2]**2, axis=0)
        m_e = (s_e * sxy_e - sx_e * sy_e) / (s_e * sxx_e - sx_e**2)
        b_e = (sy_e - m_e * sx_e) / s_e
        sx_o = np.nansum(x[1::2] / e[1::2]**2, axis=0)
        sxx_o = np.nansum((x[1::2] / e[1::2])**2, axis=0)
        sy_o = np.nansum(y[1::2] / e[1::2]**2, axis=0)
        sxy_o = np.nansum(x[1::2] * y[1::2] / e[1::2]**2, axis=0)
        s_o = np.nansum(1 / e[1::2]**2, axis=0)
        m_o = (s_o * sxy_o - sx_o * sy_o) / (s_o * sxx_o - sx_o**2)
        b_o = (sy_o - m_o * sx_o) / s_o
    return m_e, b_e, m_o, b_o


def ref_tracemask(ypix, width, dimy, dimx):
    """Build a trace mask with the v1 aperture convention.

    Parameters
    ----------
    ypix : array-like(float)
        Ypix array.
    width : int, float
        Trace-mask width.
    dimy : int
        Number of detector rows.
    dimx : int
        Number of detector columns.

    Returns
    -------
    mask : np.ndarray(float)
        Trace mask in the reference convention.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        low = np.max([np.zeros_like(ypix), ypix - width / 2], axis=0).astype(int)
        up = np.min([dimy * np.ones_like(ypix), ypix + width / 2], axis=0).astype(int)
    mask = np.zeros((dimy, dimx))
    for i, x in enumerate(np.arange(len(ypix))):
        mask[low[i]:up[i], int(x)] = 1
    return mask


def ref_solve(cube, dqcube, deepstack, y1, y2, width, outliers=None,
              artifact=None, background=None, dimy96=False):
    """Solve the SOSS 1/f correction with the v1 equations.

    Parameters
    ----------
    cube : array-like(float)
        Input observation cube.
    dqcube : array-like(int)
        Dqcube array.
    deepstack : array-like(float)
        Deepstack array.
    y1 : array-like(float)
        Y1 array.
    y2 : array-like(float)
        Y2 array.
    width : int, float
        Trace-mask width.
    outliers : array-like(float)
        Outliers array.
    artifact : array-like(float)
        Artifact array.
    background : array-like(float)
        Background array.
    dimy96 : bool
        Dimy96 option.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    cube = np.array(cube, copy=True)
    err1 = np.nanstd(cube, axis=0)
    err1 = np.repeat(err1[np.newaxis], cube.shape[0], axis=0)
    err2 = copy.deepcopy(err1)
    if cube.ndim == 4:
        nint, ngroup, dimy, dimx = cube.shape
    else:
        nint, dimy, dimx = cube.shape
        ngroup = 0
    outliers1 = (np.zeros((nint, dimy, dimx), bool) if outliers is None
                 else np.asarray(outliers, bool))
    outliers2 = np.copy(outliers1)
    mask1 = (~ref_tracemask(y1, width, dimy, dimx).astype(bool)).astype(int)
    trace1 = np.where(mask1 == 1, 0, 1)
    if not dimy96:
        mask2 = (~ref_tracemask(y2, width, dimy, dimx).astype(bool)).astype(int)
    else:
        mask2 = np.ones_like(mask1)
    trace2 = np.where(mask2 == 1, 0, 1)
    outliers1 = (outliers1 | mask1.astype(bool)).astype(int)
    outliers2 = (outliers2 | mask2.astype(bool)).astype(int)
    outliers2[:, :, :1100] = 1
    ii = np.where(dqcube != 0)
    err1[ii] = np.inf
    err2[ii] = np.inf
    err1[err1 == 0] = np.inf
    err2[err2 == 0] = np.inf
    ii = np.where(outliers1 != 0)
    ii2 = np.where(outliers2 != 0)
    if ngroup == 0:
        err1[ii] = np.inf
        err2[ii2] = np.inf
    else:
        for g in range(ngroup):
            err1[:, g][ii] = np.inf
            err2[:, g][ii2] = np.inf
    if artifact is not None:
        ii = np.where(artifact == 1)
        for g in range(ngroup):
            err1[:, g][ii] = np.inf
            err2[:, g][ii] = np.inf
    calc = {}
    for order, err, trace in zip([1, 2], [err1, err2], [trace1, trace2]):
        if order == 2 and dimy96:
            continue
        shape = (nint, dimx) if ngroup == 0 else (nint, ngroup, dimx)
        se, so = np.zeros(shape), np.zeros(shape)
        oof = np.zeros_like(cube)
        for i in range(nint):
            for g in (range(ngroup) if ngroup else [None]):
                ix = (i,) if g is None else (i, g)
                ds = deepstack if g is None else deepstack[g]
                m_e, b_e, m_o, b_o = ref_line_mle(ds, cube[ix], err[ix])
                oof[ix][::2] = b_e[None, :]
                oof[ix][1::2] = b_o[None, :]
                se[ix], so[ix] = m_e, m_o
                oof[np.isnan(oof)] = 0
                oof[np.isinf(oof)] = 0
                cube[ix] -= oof[ix] * trace
        calc[f'o{order}'] = {'oof': oof, 'scale_e': se, 'scale_o': so}
    if background is not None:
        cube += background
    return cube, calc


# Synthetic SOSS-like data.

DIMX = 1152


def make_case(seed=3, nints=7, ngroups=3, dimy=32, dimx=DIMX, ramp=True,
              dtype=np.float32):
    """Create synthetic SOSS inputs and reference arrays.

    Parameters
    ----------
    seed : int
        Random seed.
    nints : int
        Number of integrations.
    ngroups : int
        Number of groups per integration.
    dimy : int
        Number of detector rows.
    dimx : int
        Number of detector columns.
    ramp : bool
        Ramp option.
    dtype : np.dtype
        Data type of the synthetic observation.

    Returns
    -------
    result : types.SimpleNamespace
        Loaded functions or synthetic observation records.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(dimx)
    y = np.arange(dimy)[:, None]
    y1 = 7.3 + 3.0 * np.sin(x / 300.)
    y2 = 23.6 - 0.002 * x
    y2[1130:] = np.nan
    y3 = np.full(dimx, np.nan)
    prof = (900. * np.exp(-0.5 * ((y - y1) / 1.6)**2) +
            250. * np.exp(-0.5 * ((y - y2) / 1.4)**2) + 3.)
    lc = 1. - 0.02 * (np.arange(nints) >= nints // 2)
    chrom = 1. + 0.01 * np.cos(x / 150.)
    if ramp:
        deep = prof[None] * np.arange(1, ngroups + 1)[:, None, None]
        shape = (nints, ngroups, dimy, dimx)
        signal = lc[:, None, None, None] * chrom * deep[None]
        oof = rng.normal(0., 4., (nints, ngroups, 2, dimx))
        rows = np.where(np.arange(dimy) % 2, 1, 0)
        signal = signal + oof[:, :, rows, :]
    else:
        deep = prof
        shape = (nints, dimy, dimx)
        signal = lc[:, None, None] * chrom * deep[None]
        oof = rng.normal(0., 4., (nints, 2, dimx))
        rows = np.where(np.arange(dimy) % 2, 1, 0)
        signal = signal + oof[:, rows, :]
    cube = (signal + rng.normal(0., 2., shape)).astype(dtype)
    # Unflagged NaNs exercise v1's per-sum nansum asymmetry.
    cube[(rng.random(shape) < 0.002)] = np.nan
    dq = np.zeros(shape, np.uint8 if ramp else np.uint32)
    dq[rng.random(shape) < 0.01] = 4
    pixeldq = np.zeros((dimy, dimx), np.uint32)
    pixeldq[rng.random((dimy, dimx)) < 0.01] = 1
    pixeldq[:, 37] = 1
    if ramp:
        pixeldq[:, 1120] = 1
    deepstack = np.nanmedian(cube, axis=0).astype(dtype)
    deepstack[rng.random(deepstack.shape) < 0.002] = np.nan
    pmask = rng.random((dimy, dimx)) < 0.01
    background = (rng.normal(5., 1., deepstack.shape)).astype(dtype)
    return SimpleNamespace(cube=cube, dq=dq, pixeldq=pixeldq, deep=deepstack,
                           y1=y1, y2=y2, y3=y3, pmask=pmask, bkg=background,
                           ramp=ramp)


def v1_model(case, int_start=1):
    """Wrap synthetic arrays in a v1-compatible data model.

    Parameters
    ----------
    case : types.SimpleNamespace
        Synthetic observation and reference arrays.
    int_start : int
        First integration index.

    Returns
    -------
    result : types.SimpleNamespace
        Loaded functions or synthetic observation records.
    """
    nints = case.cube.shape[0]
    meta = SimpleNamespace(filename='synthetic', exposure=SimpleNamespace(
        integration_start=int_start, integration_end=int_start + nints - 1))
    if case.ramp:
        return Model(data=case.cube.copy(), groupdq=case.dq.copy(),
                     pixeldq=case.pixeldq.copy(), meta=meta)
    return Model(data=case.cube.copy(), dq=case.dq.copy(), meta=meta)


def v1_centroids(case):
    """Create a v1-compatible centroid table.

    Parameters
    ----------
    case : types.SimpleNamespace
        Synthetic observation and reference arrays.

    Returns
    -------
    centroids : dict
        Trace coordinates for each SOSS order.
    """
    return {'xpos': np.arange(case.cube.shape[-1]), 'ypos o1': case.y1,
            'ypos o2': case.y2, 'ypos o3': case.y3}


def run_v2(case, width, excluded=None, background=None, order2=True):
    """Run the v2 calculation on the synthetic observation.

    Parameters
    ----------
    case : types.SimpleNamespace
        Synthetic observation and reference arrays.
    width : int, float
        Trace-mask width.
    excluded : None, array-like(int)
        Detector columns excluded from the solve.
    background : array-like(float)
        Background array.
    order2 : bool
        Order2 option.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    dimy = case.cube.shape[-2]
    std = oneoverf.segment_nanstd(jnp.asarray(case.cube))
    b1 = oneoverf.solve_trace_bounds(case.y1, width, dimy)
    if order2:
        b2 = oneoverf.solve_trace_bounds(case.y2, width, dimy)
    else:
        b2 = (np.zeros(case.cube.shape[-1], np.int32),) * 2
    out, diag = oneoverf.oneoverf_solve(
        jnp.asarray(case.cube), jnp.asarray(case.deep), std,
        jnp.asarray(case.dq), b1, b2,
        excluded=None if excluded is None else jnp.asarray(excluded),
        pixeldq=jnp.asarray(case.pixeldq) if case.ramp else None,
        background=None if background is None else jnp.asarray(background),
        order2=order2)
    return np.asarray(out), jax.tree.map(np.asarray, diag)


def dq_cube(case):
    """Create the reference DQ cube for a synthetic observation.

    Parameters
    ----------
    case : types.SimpleNamespace
        Synthetic observation and reference arrays.

    Returns
    -------
    flags : np.ndarray(int)
        Group and pixel DQ flags combined.
    """
    if case.ramp:
        return case.dq.astype(np.uint32) + case.pixeldq[None, None]
    return case.dq


def assert_matches(out, diag, ref_out, ref_calc, ramp, *, cube_tol=2e-3,
                   b_tol=2e-3, m_tol=2e-6):
    """Compare corrected cubes and fit coefficients at float32 precision.

    Parameters
    ----------
    out : array-like(float)
        Out array.
    diag : array-like(float)
        Diag array.
    ref_out : array-like(float)
        Ref out array.
    ref_calc : array-like(float)
        Ref calc array.
    ramp : bool
        Ramp option.
    cube_tol : float
        Absolute tolerance for the corrected cube.
    b_tol : float
        Absolute tolerance for fitted intercepts.
    m_tol : float
        Absolute tolerance for fitted slopes.
    """
    np.testing.assert_array_equal(np.isnan(out), np.isnan(ref_out))
    np.testing.assert_allclose(out, ref_out, rtol=1e-6, atol=cube_tol)
    assert set(diag) == set(ref_calc)
    for order, calc in ref_calc.items():
        oof = calc['oof']
        row_e = oof[:, :, 10] if ramp else oof[:, 10]
        row_o = oof[:, :, 11] if ramp else oof[:, 11]
        np.testing.assert_allclose(diag[order]['oof_e'], row_e, atol=b_tol)
        np.testing.assert_allclose(diag[order]['oof_o'], row_o, atol=b_tol)
        for key in ('scale_e', 'scale_o'):
            np.testing.assert_array_equal(np.isnan(diag[order][key]),
                                          np.isnan(calc[key]))
            np.testing.assert_allclose(diag[order][key], calc[key],
                                       rtol=1e-5, atol=m_tol)


# Tests.

def test_line_mle_matches_v1_nan_inf_semantics():
    """Check line mle matches v1 NaN inf semantics."""
    rng = np.random.default_rng(0)
    x = rng.normal(100., 30., (4, 12, 9)).astype(np.float32)
    y = (1.7 * x + 3. + rng.normal(0., 1., x.shape)).astype(np.float32)
    e = rng.uniform(0.5, 2., x.shape).astype(np.float32)
    x[0, 2, 1] = np.nan
    y[1, 5, 4] = np.nan
    e[2, :, 6] = np.inf
    e[3, 3, 7] = np.nan
    e[0, ::2, 8] = np.inf
    got = oneoverf.line_mle(jnp.asarray(x), jnp.asarray(y), jnp.asarray(e))
    for g, r in zip(got, zip(*[ref_line_mle(x[i], y[i], e[i])
                               for i in range(4)])):
        r = np.stack(r)
        np.testing.assert_array_equal(np.isnan(np.asarray(g)), np.isnan(r))
        np.testing.assert_allclose(np.asarray(g), r, rtol=2e-5, atol=1e-4)


def test_line_mle_reference_is_actual_v1():
    """Check line mle reference is actual v1."""
    _, utils = _v1('SUBSTRIP256')
    rng = np.random.default_rng(1)
    x = rng.normal(50., 10., (10, 5))
    y = 2. * x + rng.normal(0., 1., x.shape)
    e = rng.uniform(1., 2., x.shape)
    e[:, 3] = np.inf
    x[4, 1] = np.nan
    for a, b in zip(utils.line_mle(x, y, e), ref_line_mle(x, y, e)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('width', [3., 6., 7.5, 40.])
def test_trace_bounds_match_v1_tracemask(width):
    """Check trace bounds match v1 tracemask."""
    _, utils = _v1('SUBSTRIP256')
    rng = np.random.default_rng(int(width * 10))
    dimy, dimx = 16, 64
    ypos = rng.uniform(-30., 45., dimx)
    ypos[::7] = np.nan
    ypos[3] = -width / 2. - 0.3
    ypos[5] = -width / 2. - 2.6
    ypos[9] = 100.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        v1_mask = utils.make_soss_tracemask(np.arange(dimx), ypos, width,
                                            dimy, dimx).astype(bool)
    low, up = oneoverf.solve_trace_bounds(ypos, width, dimy)
    rows = np.arange(dimy)[:, None]
    np.testing.assert_array_equal((rows >= low) & (rows < up), v1_mask)


def test_group_level_matches_actual_v1_with_pixel_mask():
    """Check group level matches actual v1 with pixel mask."""
    case = make_case()
    v1_solve, _ = _v1('SUBSTRIP256')
    width = 13.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        result, calc, outliers = v1_solve(
            v1_model(case), case.deep.copy(), trace_width=width,
            background=case.bkg, save_results=False, pixel_mask=case.pmask,
            centroids=v1_centroids(case))
    v1_out = result.data
    ref_out, ref_calc = ref_solve(
        case.cube, dq_cube(case), case.deep, case.y1, case.y2, width,
        outliers=np.broadcast_to(case.pmask, (case.cube.shape[0],
                                              *case.pmask.shape)),
        background=case.bkg)
    # The transcription is the v1 algorithm itself (bit-identical).
    np.testing.assert_array_equal(ref_out, v1_out)
    for order in ('o1', 'o2'):
        for key in ('oof', 'scale_e', 'scale_o'):
            np.testing.assert_array_equal(ref_calc[order][key],
                                          calc[order][key])
    excluded = np.broadcast_to(case.pmask, (case.cube.shape[0],
                                            *case.pmask.shape))
    out, diag = run_v2(case, width, excluded=excluded, background=case.bkg)
    assert_matches(out, diag, v1_out, calc, ramp=True)
    # The correction does real work in both orders.
    assert np.nanmax(np.abs(out - case.cube - case.bkg)) > 1.
    assert np.count_nonzero(diag['o2']['oof_e']) > 0


def test_group_level_without_mask_after_reset_window_matches_v1():
    """Check group level without mask after reset window matches v1."""
    case = make_case(seed=8)
    v1_solve, _ = _v1('SUBSTRIP256')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        result, calc, _ = v1_solve(
            v1_model(case, int_start=300), case.deep.copy(), trace_width=14.,
            background=None, save_results=False, pixel_mask=None,
            centroids=v1_centroids(case))
    out, diag = run_v2(case, 14.)
    assert_matches(out, diag, result.data, calc, ramp=True)


def test_v1_group_reset_artifact_crashes_and_v2_masks_every_group():
    """Check v1 group reset artifact crashes and v2 masks every group."""
    case = make_case(seed=11, nints=6)
    v1_solve, utils = _v1('SUBSTRIP256')
    model = v1_model(case, int_start=224)
    with pytest.raises(IndexError):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            v1_solve(model, case.deep.copy(), trace_width=12.,
                     save_results=False, pixel_mask=None,
                     centroids=v1_centroids(case))
    artifact = utils.mask_reset_artifact(v1_model(case, int_start=224))
    assert artifact.any()
    ref_out, ref_calc = ref_solve(case.cube, dq_cube(case), case.deep,
                                  case.y1, case.y2, 12., artifact=artifact)
    out, diag = run_v2(case, 12., excluded=artifact.astype(bool))
    assert_matches(out, diag, ref_out, ref_calc, ramp=True)


def test_integration_level_matches_actual_v1():
    """Check integration level matches actual v1."""
    case = make_case(seed=5, ramp=False, nints=9)
    v1_solve, _ = _v1('SUBSTRIP256')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        result, calc, _ = v1_solve(
            v1_model(case), case.deep.copy(), trace_width=12.5,
            background=None, save_results=False, pixel_mask=None,
            centroids=v1_centroids(case))
    out, diag = run_v2(case, 12.5)
    assert_matches(out, diag, result.data, calc, ramp=False)


def test_integration_level_with_3d_pixel_mask_matches_reference():
    """Check integration level with 3d pixel mask matches reference."""
    case = make_case(seed=6, ramp=False)
    rng = np.random.default_rng(66)
    pmask = rng.random(case.cube.shape) < 0.02
    ref_out, ref_calc = ref_solve(case.cube, case.dq, case.deep, case.y1,
                                  case.y2, 13., outliers=pmask)
    out, diag = run_v2(case, 13., excluded=pmask)
    assert_matches(out, diag, ref_out, ref_calc, ramp=False)


def test_substrip96_skips_order2_like_v1():
    """Check substrip96 skips order2 like v1."""
    case = make_case(seed=9, dimy=96, nints=5, ngroups=2)
    v1_solve, _ = _v1('SUBSTRIP96')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        result, calc, _ = v1_solve(
            v1_model(case), case.deep.copy(), trace_width=16.,
            save_results=False, pixel_mask=case.pmask,
            centroids=v1_centroids(case))
    assert set(calc) == {'o1'}
    excluded = np.broadcast_to(case.pmask, (case.cube.shape[0],
                                            *case.pmask.shape))
    out, diag = run_v2(case, 16., excluded=excluded, order2=False)
    assert_matches(out, diag, result.data, calc, ramp=True)


def test_float64_matches_actual_v1_tightly():
    """Check float64 matches actual v1 tightly."""
    case = make_case(seed=21, dtype=np.float64)
    v1_solve, _ = _v1('SUBSTRIP256')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        result, calc, _ = v1_solve(
            v1_model(case), case.deep.copy(), trace_width=12.,
            background=case.bkg, save_results=False, pixel_mask=case.pmask,
            centroids=v1_centroids(case))
    old = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        excluded = np.broadcast_to(case.pmask, (case.cube.shape[0],
                                                *case.pmask.shape))
        out, diag = run_v2(case, 12., excluded=excluded, background=case.bkg)
    finally:
        jax.config.update('jax_enable_x64', old)
    assert out.dtype == np.float64
    assert_matches(out, diag, result.data, calc, ramp=True, cube_tol=1e-9,
                   b_tol=1e-8, m_tol=1e-12)


def test_trace_width_sweep_does_not_recompile():
    """Check trace width sweep does not recompile."""
    case = make_case(seed=2, nints=4, ngroups=2)
    run_v2(case, 12.)
    fn = oneoverf.oneoverf_solve
    if not hasattr(fn, '_cache_size'):
        pytest.skip('jit cache introspection unavailable in this jax')
    before = fn._cache_size()
    for width in (10., 13., 15.5, 20.):
        run_v2(case, width)
    assert fn._cache_size() == before
