"""Check box and optimal extraction against v1 functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exotedrf.v2 import config, core, stages
from exotedrf.v2.kernels import extract
from exotedrf.v2.pipeline import PipelineState

from . import extraction_v1_oracle as oracle

RTOL = 1e-5


def _meta(mode, nints, detector='CLEAR', subarray='SUBSTRIP256'):
    """Return meta."""
    return core.ObsMeta(mode, detector, subarray, 1., 1,
                        np.arange(nints, dtype=float), np.array([5, -5]),
                        np.array([nints]), ())


def _cube(seed, nints=6, dimy=24, dimx=40, nan_fraction=0.02):
    """Return cube."""
    rng = np.random.default_rng(seed)
    data = rng.normal(100., 10., (nints, dimy, dimx)).astype(np.float32)
    err = rng.uniform(1., 3., (nints, dimy, dimx)).astype(np.float32)
    bad = rng.random(data.shape) < nan_fraction
    data[bad] = np.nan
    err[bad] = np.nan
    return data, err


def _centroid(dimx, start=9.37, slope=0.0713):
    """Return centroid."""
    return (start + slope * np.arange(dimx)).astype(np.float32)


# Asymmetric production box extraction (do_box_extraction).

@pytest.mark.parametrize('width', [
    (2.3, 4.6), [4.0, 1.5], {'lower': 3.7, 'upper': 3.2},
    (0.4, 0.3),
    (11.0, 2.5),
    (1.5, 30.0),
])
def test_asymmetric_sum_matches_v1_do_box_extraction(width):
    """Check asymmetric sum matches v1 do box extraction."""
    v1 = oracle.load()
    data, err = _cube(1)
    cen = _centroid(data.shape[-1])
    ref_f, ref_e, _ = v1.do_box_extraction(
        data.astype(float), err.astype(float), cen.astype(float), width,
        progress=False)
    halves = config.extract_width_halves(width)
    flux, ferr = extract.box_extract(
        data, err, cen, jnp.asarray(halves, jnp.float32), mode='sum')
    np.testing.assert_allclose(np.asarray(flux), ref_f, rtol=RTOL, atol=1e-3)
    np.testing.assert_allclose(np.asarray(ferr), ref_e, rtol=RTOL, atol=1e-4)


def test_asymmetric_sum_matches_v1_with_start_end_and_lower_upper_kwargs():
    """Check asymmetric sum matches v1 with start end and lower upper keywords."""
    v1 = oracle.load()
    data, err = _cube(2)
    cen = _centroid(30)
    ref_f, ref_e, _ = v1.do_box_extraction(
        data.astype(float), err.astype(float), cen.astype(float), 99,
        extract_start=5, extract_end=33, progress=False,
        lower_width=2.2, upper_width=5.1)
    flux, ferr = extract.box_extract(
        data, err, cen, jnp.asarray([2.2, 5.1], jnp.float32),
        extract_start=5, extract_end=33, mode='sum')
    np.testing.assert_allclose(np.asarray(flux), ref_f, rtol=RTOL, atol=1e-3)
    np.testing.assert_allclose(np.asarray(ferr), ref_e, rtol=RTOL, atol=1e-4)


@pytest.mark.parametrize('width', [(2.3, 4.6), (3.0, 3.0), (5.0, 2.0),
                                   (0.6, 0.7)])
def test_asymmetric_nanaware_matches_v1_optimizer_extraction(width):
    """Check asymmetric nanaware matches v1 optimizer extraction."""
    v1 = oracle.load()
    data, _ = _cube(3)
    cen = _centroid(data.shape[-1])
    ref = v1.do_box_extraction_nanaware(
        data.astype(float), cen.astype(float), list(width), progress=False)
    flux, _ = extract.box_extract(
        data, None, cen, jnp.asarray(width, jnp.float32), mode='nanaware')
    np.testing.assert_allclose(np.asarray(flux), ref, rtol=RTOL, atol=1e-3,
                               equal_nan=True)


@pytest.mark.parametrize('mode', ['sum', 'nanaware'])
def test_symmetric_pair_is_bitwise_identical_to_scalar(mode):
    """Check symmetric pair is bitwise identical to scalar."""
    data, err = _cube(4)
    cen = _centroid(data.shape[-1])
    for half in (1.25, 2.5, 3.7):
        scalar = extract.box_extract(data, err, cen, np.float32(half),
                                     mode=mode)
        pair = extract.box_extract(
            data, err, cen, jnp.asarray([half, half], jnp.float32), mode=mode)
        for a, b in zip(scalar, pair):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_asymmetric_values_are_traced_not_recompiled():
    """Check asymmetric values are traced not recompiled."""
    data, err = _cube(5)
    cen = _centroid(data.shape[-1])
    extract.box_extract(data, err, cen, jnp.asarray([1., 2.]), mode='sum')
    before = extract.box_extract._cache_size()
    for pair in ([1.5, 2.5], [3., 1.], [2.2, 2.9]):
        extract.box_extract(data, err, cen, jnp.asarray(pair), mode='sum')
    assert extract.box_extract._cache_size() == before


def test_sweep_over_asymmetric_pairs_matches_individual_calls():
    """Check sweep over asymmetric pairs matches individual calls."""
    data, err = _cube(6)
    cen = _centroid(data.shape[-1])
    pairs = np.array([[1., 2.], [2.5, 1.5], [3., 4.]], np.float32)
    swept, swept_err = extract.box_extract_sweep(data, err, cen, pairs,
                                                 mode='sum')
    for index, pair in enumerate(pairs):
        flux, ferr = extract.box_extract(data, err, cen, pair, mode='sum')
        np.testing.assert_array_equal(np.asarray(swept[index]),
                                      np.asarray(flux))
        np.testing.assert_array_equal(np.asarray(swept_err[index]),
                                      np.asarray(ferr))


def test_v1_format_and_metadata_parser_round_trip():
    """Check v1 format and metadata parser round trip."""
    v1 = oracle.load()
    for raw in ((10, 12), [10.5, 3], {'lower': 4, 'upper': 6}):
        aperture = config._normalize_width_value(
            raw, 'extract_width_soss2', allow_pair_list=True)
        assert config.format_extract_width(aperture) == \
            v1._format_extract_width(raw)
        parsed = v1.parse_extract_width_metadata(
            config.format_extract_width(aperture))
        assert (parsed['lower'], parsed['upper']) == \
            config.extract_width_halves(raw)
    for scalar in (30, 7.5, 'N/A'):
        assert config.format_extract_width(scalar) == \
            v1._format_extract_width(scalar)
    for bad in ((0, 3), (2, -1), 'thirty'):
        with pytest.raises(ValueError):
            config.extract_width_halves(bad)
        with pytest.raises(ValueError):
            v1._parse_extraction_width(bad)


# Extract_width: 'optimize' (v1 box_extract_* aperture search).

def _soss_like(seed=11, nints=40, dimy=96, dimx=48):
    """Return SOSS like."""
    rng = np.random.default_rng(seed)
    yy = np.arange(dimy)[None, :, None]
    xx = np.arange(dimx)
    centre1 = 40. + 0.05 * xx
    centre2 = 70. - 0.1 * xx
    jitter = rng.normal(0, 0.4, nints)[:, None, None]
    trace = (4000. * np.exp(-0.5 * ((yy - centre1 - jitter) / 4.) ** 2) +
             1500. * np.exp(-0.5 * ((yy - centre2 - jitter) / 3.) ** 2))
    data = (trace + rng.normal(0., 25., (nints, dimy, dimx))).astype(
        np.float32)
    err = np.full(data.shape, 25., np.float32)
    dq = np.zeros(data.shape, np.uint32)
    dq[3, 40, 7] = core.DQ_DO_NOT_USE
    dq[8, 41, 20] = core.DQ_SATURATED
    dq[:, 70, 4] = core.DQ_DO_NOT_USE | core.DQ_SATURATED
    ypos2 = centre2.astype(float).copy()
    ypos2[30:] = np.nan
    centroids = {'xpos': xx.astype(float), 'ypos o1': centre1.astype(float),
                 'ypos o2': ypos2}
    return data, err, dq, centroids


@pytest.mark.parametrize('o1, o2', [('optimize', None),
                                    ('optimize', 'optimize'),
                                    (30, 'optimize'),
                                    ('optimize', (6, 9))])
def test_soss_optimize_selects_v1_width_and_flux(o1, o2):
    """Check SOSS optimize selects v1 width and flux."""
    data, err, dq, centroids = _soss_like()
    nints, dimy, dimx = data.shape
    v1 = oracle.load(dimx=dimx)
    seg = oracle.segment(data, err, dq)
    ref = v1.box_extract_soss([seg], oracle.centroid_frame(**centroids), o1,
                              soss_width_o2=o2, mask_saturated_pixels=True,
                              mask_do_not_use_pixels=True)
    _, ref_f1, ref_e1, _, _, ref_f2, ref_e2, _, ref_width = ref

    opts = {'mask_do_not_use_pixels': True, 'mask_saturated_pixels': True,
            'saturation_rescue': False,
            'extract_width_soss2': (
                config._normalize_width_value(
                    o2, 'extract_width_soss2', allow_pair_list=True))}
    ctx = {'opts': opts, 'centroids': centroids}
    state = PipelineState(core.RateCube(data, err, dq,
                                        _meta('NIRISS/SOSS', nints)), {})
    params, ctx2, selection = stages._resolve_optimize_widths(
        state, {'extract_width': o1}, ctx)
    if o1 == 'optimize':
        assert params['extract_width'] == ref_width
        assert 1 < selection[1]['index'] < 50
    spectra = stages.extract_orders(state, params, ctx2, mode='sum')
    np.testing.assert_allclose(np.asarray(spectra[1][0]), ref_f1,
                               rtol=RTOL, atol=1e-2)
    np.testing.assert_allclose(np.asarray(spectra[1][1]), ref_e1,
                               rtol=RTOL, atol=1e-3)
    width2 = ref_f2.shape[-1]
    np.testing.assert_allclose(np.asarray(spectra[2][0])[:, :width2],
                               ref_f2, rtol=RTOL, atol=1e-2)
    np.testing.assert_allclose(np.asarray(spectra[2][1])[:, :width2],
                               ref_e2, rtol=RTOL, atol=1e-3)


@pytest.mark.parametrize('detector, grating, subarray, trace_start', [
    ('nrs1', 'PRISM', 'SUB512', 14),
    ('nrs2', 'G395H', 'SUB2048', 0),
])
def test_nirspec_optimize_selects_v1_width_and_flux(detector, grating,
                                                    subarray, trace_start):
    """Check NIRSpec optimize selects v1 width and flux."""
    rng = np.random.default_rng(21)
    nints, dimy, dimx = 40, 20, 48
    yy = np.arange(dimy)[None, :, None]
    xx = np.arange(dimx)
    centre = 9.3 + 0.02 * xx
    jitter = rng.normal(0, 0.15, nints)[:, None, None]
    data = (3000. * np.exp(-0.5 * ((yy - centre - jitter) / 1.1) ** 2) +
            rng.normal(0., 12., (nints, dimy, dimx))).astype(np.float32)
    err = np.full(data.shape, 12., np.float32)
    dq = np.zeros(data.shape, np.uint32)
    dq[4, 9, 30] = core.DQ_SATURATED
    centroids = {'xpos': np.arange(trace_start, dimx, dtype=float),
                 'ypos': centre[trace_start:].astype(float)}
    v1 = oracle.load(detector=detector, grating=grating, subarray=subarray)
    _, ref_flux, ref_err, _, ref_width = v1.box_extract_nirspec(
        [oracle.segment(data, err, dq)], oracle.centroid_frame(**centroids),
        'optimize', mask_saturated_pixels=True, mask_do_not_use_pixels=True)

    ctx = {'opts': {'mask_do_not_use_pixels': True,
                    'mask_saturated_pixels': True},
           'centroids': centroids, 'nirspec_detector': detector.upper(),
           'nirspec_grating': grating, 'nirspec_xstart': trace_start}
    meta = _meta(f'NIRSpec/{grating}', nints, detector=detector.upper(),
                 subarray=subarray)
    state = PipelineState(core.RateCube(data, err, dq, meta), {})
    params, ctx, selection = stages._resolve_optimize_widths(
        state, {'extract_width': 'optimize'}, ctx, centroids)
    assert params['extract_width'] == ref_width
    spectra = stages._extract_orders_nirspec(
        state, params, ctx, mode='sum', centroids=centroids)
    np.testing.assert_allclose(np.asarray(spectra[1][0]), ref_flux,
                               rtol=RTOL, atol=1e-2)
    np.testing.assert_allclose(np.asarray(spectra[1][1]), ref_err,
                               rtol=RTOL, atol=1e-3)


def test_miri_optimize_selects_v1_width_and_flux():
    """Check MIRI optimize selects v1 width and flux."""
    rng = np.random.default_rng(31)
    nints, dimy, dimx = 40, 44, 30
    xx = np.arange(dimx)[None, None, :]
    jitter = rng.normal(0, 0.2, nints)[:, None, None]
    data = (2500. * np.exp(-0.5 * ((xx - 15.4 - jitter) / 1.3) ** 2) +
            rng.normal(0., 10., (nints, dimy, dimx))).astype(np.float32)
    err = np.full(data.shape, 10., np.float32)
    dq = np.zeros(data.shape, np.uint32)
    dq[2, 20, 15] = core.DQ_DO_NOT_USE
    centroids = {'xpos': np.full(dimy - 6, 15.4),
                 'ypos': np.arange(6, dimy, dtype=float)}
    v1 = oracle.load()
    _, ref_flux, ref_err, _, ref_width = v1.box_extract_miri(
        [oracle.segment(data, err, dq)], oracle.centroid_frame(**centroids),
        'optimize', mask_saturated_pixels=True, mask_do_not_use_pixels=True)

    ctx = {'opts': {'mask_do_not_use_pixels': True,
                    'mask_saturated_pixels': True}, 'centroids': centroids}
    meta = _meta('MIRI/LRS', nints, detector='MIRIMAGE',
                 subarray='SLITLESSPRISM')
    state = PipelineState(core.RateCube(data, err, dq, meta), {})
    params, ctx, _ = stages._resolve_optimize_widths(
        state, {'extract_width': 'optimize'}, ctx, centroids)
    assert params['extract_width'] == ref_width
    spectra = stages._extract_orders_miri(
        state, params, ctx, mode='sum', centroids=centroids)
    np.testing.assert_allclose(np.asarray(spectra[1][0]), ref_flux,
                               rtol=RTOL, atol=1e-2)
    np.testing.assert_allclose(np.asarray(spectra[1][1]), ref_err,
                               rtol=RTOL, atol=1e-3)


def test_white_light_selection_rule_is_v1_first_minimum():
    """Check white light selection rule is v1 first minimum."""
    wlc = np.array([[1., 1., 1.], [2., 3., 2.], [1., 1., 1.], [2., 3., 2.]])
    width, index, scatter = extract.select_width_min_white_scatter(
        wlc, [4., 5., 6.])
    # Candidates 0 and 2 tie; np.argmin keeps the first.
    assert (width, index) == (4., 0)
    np.testing.assert_allclose(scatter, [1., 2., 1.])


# NIRSpec optimal (Horne) extraction.

def _nirspec_horne_dataset(dtype):
    """Return NIRSpec horne dataset."""
    rng = np.random.default_rng(7)
    nints, dimy, dimx, xstart = 41, 14, 36, 6
    yy = np.arange(dimy)[:, None]
    xx = np.arange(dimx)[None, :]
    centre = 6.2 + 0.03 * xx
    profile = np.exp(-0.5 * ((yy - centre) / 1.3) ** 2) + 0.01
    deep = (500. + 6. * xx) * profile
    data = (deep[None] * (1 + rng.normal(0, .003, (nints, 1, 1))) +
            rng.normal(0, 2., (nints, dimy, dimx))).astype(dtype)
    data[5, 1, 12] += 600.
    data[17, 6, 20] += 250.
    data[9, 7, 3] += 900.
    centroids = {'xpos': np.arange(xstart, dimx, dtype=float),
                 'ypos': centre[0, xstart:].astype(float)}
    return data, deep.astype(dtype), centroids, xstart


@pytest.mark.parametrize('width', [5, 7.5, None])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_nirspec_optimal_matches_v1_optimal_extract_nirspec(width, dtype):
    """Check NIRSpec optimal matches v1 optimal extract NIRSpec."""
    data, deep, centroids, xstart = _nirspec_horne_dataset(dtype)
    nints, dimy, dimx = data.shape
    v1 = oracle.load(trace_start=xstart)
    _, ref_flux, ref_err, _ = v1.optimal_extract_nirspec(
        [oracle.segment(data.copy())], deep.copy(),
        oracle.centroid_frame(**centroids), width, max_iter=10,
        var_thresh=25)

    meta = _meta('NIRSpec/G395H', nints, detector='NRS1', subarray='SUB2048')
    cube = core.RateCube(data, np.ones_like(data),
                         np.zeros(data.shape, np.uint32), meta)
    state = PipelineState(cube, {'stage3_deepframe': deep})
    ctx = {'opts': {'opt_max_iter': 10, 'opt_var_thresh': 25},
           'nirspec_xstart': xstart}
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', dtype == np.float64)
    try:
        flux, ferr, counts = stages._extract_nirspec_optimal(
            state, {'extract_width': width}, ctx, centroids)
    finally:
        jax.config.update('jax_enable_x64', previous)
    assert counts and counts[0] > 0
    tolerance = 2e-5 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(flux, ref_flux, rtol=tolerance, atol=1e-6)
    np.testing.assert_allclose(ferr, ref_err, rtol=tolerance, atol=1e-6)
    assert np.all(flux[:, :xstart] == 0.) and np.all(ferr[:, :xstart] == 0.)


def test_miri_optimal_whole_frame_width_none_matches_v1():
    """Check MIRI optimal whole frame width none matches v1."""
    rng = np.random.default_rng(3)
    nints, dimy, dimx = 33, 20, 10
    xx = np.arange(dimx)[None, :]
    yy = np.arange(dimy)[:, None]
    deep = (300. + 4. * yy) * (np.exp(-0.5 * ((xx - 5.) / 1.2) ** 2) + .02)
    data = (deep[None] * (1 + rng.normal(0, .002, (nints, 1, 1))) +
            rng.normal(0, 2, (nints, dimy, dimx))).astype(np.float32)
    data[4, 12, 1] += 400.
    centroids = {'xpos': np.full(dimy - 3, 5.), 'ypos': np.arange(3, dimy,
                                                                  dtype=float)}
    v1 = oracle.load()
    # V1 2.5.0 transposes the DQ cube unconditionally, so a DQ plane is needed.
    _, ref_flux, ref_err, _, _ = v1.optimal_extract_miri(
        [oracle.segment(data.copy(), None, np.zeros(data.shape, np.uint32))],
        deep.astype(np.float32).copy(),
        oracle.centroid_frame(**centroids), None, max_iter=10, var_thresh=25,
        dq_report=True)
    meta = _meta('MIRI/LRS', nints, detector='MIRIMAGE',
                 subarray='SLITLESSPRISM')
    state = PipelineState(core.RateCube(
        data, np.ones_like(data), np.zeros(data.shape, np.uint32), meta),
        {'stage3_deepframe': deep.astype(np.float32)})
    flux, ferr, _ = stages._extract_miri_optimal(
        state, {'extract_width': None},
        {'opts': {'opt_max_iter': 10, 'opt_var_thresh': 25}}, centroids)
    np.testing.assert_allclose(flux, ref_flux, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(ferr, ref_err, rtol=2e-5, atol=1e-6)


def test_custom_deepframe_replaces_profile_and_stage3_trace():
    """Check custom deepframe replaces profile and stage3 trace."""
    data, deep, centroids, xstart = _nirspec_horne_dataset(np.float32)
    nints = data.shape[0]
    custom = np.roll(deep, 1, axis=0)
    meta = _meta('NIRSpec/G395H', nints, detector='NRS1', subarray='SUB2048')
    state = PipelineState(core.RateCube(
        data, np.ones_like(data), np.zeros(data.shape, np.uint32), meta),
        {'stage3_deepframe': deep})
    ctx = {'opts': {'opt_max_iter': 3, 'opt_var_thresh': 25},
           'nirspec_xstart': xstart}
    with_custom = dict(ctx, custom_deepframe=custom)
    assert stages._stage3_deepframe(state, ctx) is deep
    assert stages._stage3_deepframe(state, with_custom) is custom
    computed = stages._extract_nirspec_optimal(
        state, {'extract_width': 5}, ctx, centroids)[0]
    swapped = stages._extract_nirspec_optimal(
        PipelineState(state.cube, {'stage3_deepframe': custom}),
        {'extract_width': 5}, ctx, centroids)[0]
    overridden = stages._extract_nirspec_optimal(
        state, {'extract_width': 5}, with_custom, centroids)[0]
    np.testing.assert_array_equal(overridden, swapped)
    assert not np.allclose(overridden, computed)
