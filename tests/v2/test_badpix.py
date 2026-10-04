"""Check bad-pixel kernels against v1 reference calculations."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.ndimage import median_filter

from exotedrf.v2 import core
from exotedrf.v2.kernels import badpix

DNU = np.uint32(1)
SAT = np.uint32(2)
HOT = np.uint32(2048)
WARM = np.uint32(4096)


# Numpy reference implementation.

def _box_stats(frame, ybox, xbox, j, i):
    """Literal v1 get_interp_box sample multiset and MAD sigma."""
    dimy, dimx = frame.shape
    low_x = max(i - xbox, 0)
    up_x = min(i + xbox + 1, dimx - 1)
    low_y = max(j - ybox, 0)
    up_y = min(j + ybox + 1, dimy - 1)
    box = np.concatenate([
        frame[j, low_x:i], frame[j, i + 1:up_x],
        frame[low_y, low_x:up_x], frame[up_y, low_x:up_x],
    ]).astype(np.float64)
    med = np.nanmedian(box)
    std = np.nanmedian(np.abs(box - med)) / 0.6745
    return med, std


def _box_median(frame, ybox, xbox, j, i):
    """Return box median."""
    return _box_stats(frame, ybox, xbox, j, i)[0]


def _running_median(cube, window):
    """Return running median."""
    return median_filter(cube, (window, 1, 1))


def _ref_noise_plane(std_dev, space_thresh, ymax, ybox, xbox, miri):
    """Calculate noise plane with the reference equations."""
    dimy, dimx = std_dev.shape
    std_dev = np.where(std_dev == 0, np.nanmedian(std_dev), std_dev)
    thresh = 1e6 if miri else space_thresh
    flags = np.zeros((dimy, dimx), bool)
    for i in range(5, dimx - 5):
        for j in range(ymax):
            med, std = _box_stats(std_dev, ybox, xbox, j, i)
            if np.abs(std_dev[j, i] - med) >= thresh * std:
                flags[j, i] = True
    out = std_dev.copy()
    for j, i in zip(*np.nonzero(flags)):
        out[j, i] = _box_median(std_dev, ybox, xbox, j, i)
    return flags, out


def ref_badpixstep(cube, err, dq, deepframe, space_thresh, time_thresh,
                   box_size, window_size, instrument,
                   median_high_variance=False, preserve_saturated=False,
                   clear_interpolated_dq=False):
    """Correct bad pixels using the v1 reference equations.

    Parameters
    ----------
    cube : array-like(float)
        Input observation cube.
    err : array-like(float)
        Err array.
    dq : array-like(int)
        Dq array.
    deepframe : array-like(float)
        Deepframe array.
    space_thresh : float
        Spatial outlier threshold.
    time_thresh : float
        Temporal outlier threshold.
    box_size : int
        Spatial interpolation box size.
    window_size : int
        Temporal filter window.
    instrument : str
        Instrument name.
    median_high_variance : bool
        Median high variance option.
    preserve_saturated : bool
        Preserve saturated option.
    clear_interpolated_dq : bool
        Clear interpolated dq option.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    cube = np.asarray(cube, np.float64).copy()
    err = np.asarray(err, np.float64).copy()
    dq = np.asarray(dq, np.uint32)
    deepframe = np.asarray(deepframe, np.float64)
    nints, dimy, dimx = cube.shape

    saturated = (dq & SAT) != 0
    sat_any = saturated.any(axis=0)

    if instrument == 'NIRISS':
        ymax, ybox, xbox = dimy - 5, 0, box_size
    elif instrument == 'NIRSPEC':
        ymax, ybox, xbox = dimy, 0, box_size
    else:
        ymax, ybox, xbox = dimy, box_size, 0

    newdata = cube.copy()
    newdata[newdata < 0] = 0
    ref_int = min(10, nints - 1)
    hot = (dq[ref_int] & (DNU | HOT | WARM)) != 0

    hotpix = np.zeros((dimy, dimx), bool)
    nanpix = np.zeros((dimy, dimx), bool)
    otherpix = np.zeros((dimy, dimx), bool)
    for i in range(5, dimx - 5):
        for j in range(ymax):
            if hot[j, i]:
                hotpix[j, i] = True
            else:
                med, std = _box_stats(deepframe, ybox, xbox, j, i)
                if np.isnan(deepframe[j, i]):
                    nanpix[j, i] = True
                elif (instrument != 'MIRI' and
                        np.abs(deepframe[j, i] - med) >= space_thresh * std):
                    otherpix[j, i] = True
    badpix = hotpix | nanpix | otherpix
    if preserve_saturated:
        badpix = badpix & ~sat_any

    newdq = dq.astype(np.uint64)
    for n in range(nints):
        frame = newdata[n].copy()
        for j, i in zip(*np.nonzero(badpix)):
            newdata[n, j, i] = _box_median(frame, ybox, xbox, j, i)
    if clear_interpolated_dq:
        newdq[:, badpix] = 0

    # Two temporal passes.
    for niter in range(2):
        cube_filt = _running_median(newdata, window_size)
        if instrument == 'NIRISS':
            cube_filt[:2] = np.median(cube_filt[2:7], axis=0)
            cube_filt[-2:] = np.median(cube_filt[-8:-3], axis=0)
        else:
            cube_filt[:5] = np.median(cube_filt[5:15], axis=0)
            cube_filt[-5:] = np.median(cube_filt[-16:-6], axis=0)
        if niter == 0:
            std_dev = np.nanmedian(np.abs(
                0.5 * (newdata[0:-2] + newdata[2:]) - newdata[1:-1]), axis=0)
        else:
            std_dev = np.nanstd(newdata, axis=0)
        std_flags, std_dev = _ref_noise_plane(
            std_dev, space_thresh, ymax, ybox, xbox, instrument == 'MIRI')
        if niter == 0:
            scale = np.abs(newdata - cube_filt) / std_dev
            tflag = scale > time_thresh
            if preserve_saturated:
                tflag &= ~saturated
            newdata[tflag] = cube_filt[tflag]
            if clear_interpolated_dq:
                newdq[tflag] = 0
        else:
            high_variance = std_flags
            stack = np.nanmedian(newdata, axis=0)
            if median_high_variance:
                replace = high_variance.copy()
                if preserve_saturated:
                    replace &= ~sat_any
                newdata[:, replace] = stack[replace]

    nanmask = np.isnan(newdata)
    newdata[nanmask] = cube_filt[nanmask]
    err[np.isnan(err)] = np.nanmedian(err)
    newdata[newdata < 0] = 0
    newdata[np.isnan(newdata)] = 0

    if instrument == 'NIRISS':
        newdata[:, :, :5] = 0
        newdata[:, :, -5:] = 0
        newdata[:, -5:] = 0

    if clear_interpolated_dq:
        newdq = newdq | (2 * saturated.astype(np.uint64))
    return (newdata, err, newdq.astype(np.uint32), badpix, high_variance)


# Synthetic dataset.

def make_dataset(seed=42):
    """Create a synthetic cube with structure and noise.

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
    nints, dimy, dimx = 10, 16, 32
    yy, xx = np.mgrid[:dimy, :dimx]
    base = 100. + 0.05 * yy + 0.08 * xx
    cube = base[None] + rng.uniform(-1., 1., (nints, dimy, dimx))
    cube = cube.astype(np.float32)
    err = np.abs(rng.uniform(0.5, 1.5, cube.shape)).astype(np.float32)
    dq = np.zeros(cube.shape, np.uint32)

    # Hot / DO_NOT_USE pixels via DQ.
    dq[:, 7, 10] |= HOT
    dq[:, 4, 20] |= DNU
    # Spatial outlier present in all integrations (and the deepframe).
    cube[:, 9, 14] += 300.
    # Saturated pixel, also deviant -- must NOT be flagged or interpolated.
    cube[:, 9, 22] += 500.
    dq[:, 9, 22] |= SAT
    # Temporal spike in a single integration.
    cube[4, 8, 18] += 40.
    # A NaN error value.
    err[3, 5, 5] = np.nan

    deepframe = np.median(cube[:8], axis=0).astype(np.float32)
    # Deepframe-only NaN pixel.
    deepframe[6, 25] = np.nan
    return cube, err, dq, deepframe


PARAMS = dict(space_thresh=15., time_thresh=10., box_size=5, window_size=5)


def run_prepared_badpix(cube, err, dq, deepframe, *, space_thresh,
                        time_thresh, box_size, window_size, instrument,
                        median_high_variance=False, preserve_saturated=False,
                        clear_interpolated_dq=False):
    """Compose the public prepared primitives exactly as the wrapper does.

    Parameters
    ----------
    cube : array-like(float)
        Input observation cube.
    err : array-like(float)
        Err array.
    dq : array-like(int)
        Dq array.
    deepframe : array-like(float)
        Deepframe array.
    space_thresh : float
        Spatial outlier threshold.
    time_thresh : float
        Temporal outlier threshold.
    box_size : int
        Spatial interpolation box size.
    window_size : int
        Temporal filter window.
    instrument : str
        Instrument name.
    median_high_variance : bool
        Median high variance option.
    preserve_saturated : bool
        Preserve saturated option.
    clear_interpolated_dq : bool
        Clear interpolated dq option.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    err_out, saturated, saturated_any = badpix.prepare_badpix_common(err, dq)
    spatial_prep = badpix.prepare_spatial_badpix(
        deepframe, dq, saturated_any, box_size=box_size,
        instrument=instrument, preserve_saturated=preserve_saturated)
    spatial_map = badpix.spatial_badpix_from_prepared(
        *spatial_prep, space_thresh=space_thresh)
    spatial_data, spatial_dq = badpix.apply_spatial_badpix(
        jnp.where(jnp.asarray(cube) < 0, 0., cube), dq, spatial_map,
        box_size=box_size, instrument=instrument,
        clear_interpolated_dq=clear_interpolated_dq)
    cube_filt, std_dev = badpix.prepare_temporal_badpix(
        spatial_data, window_size=window_size, instrument=instrument,
        space_thresh=space_thresh, box_size=box_size)
    data0, dq0 = badpix.apply_temporal_badpix(
        spatial_data, spatial_dq, saturated, cube_filt, std_dev,
        time_thresh=time_thresh, preserve_saturated=preserve_saturated,
        clear_interpolated_dq=clear_interpolated_dq)
    filt1 = badpix.temporal_filter(
        data0, window_size=window_size, instrument=instrument)
    high_variance, _, _ = badpix.noise_plane_spatial(
        badpix.temporal_std(data0), space_thresh, box_size=box_size,
        instrument=instrument)
    data1 = badpix.apply_high_variance(
        data0, high_variance, saturated_any,
        median_high_variance=median_high_variance,
        preserve_saturated=preserve_saturated)
    data_out, err_out, dq_out = badpix.finalize_badpix(
        data1, err_out, dq0, filt1, saturated, instrument=instrument,
        clear_interpolated_dq=clear_interpolated_dq)
    return data_out, err_out, dq_out, spatial_map, high_variance


def _check(got, expected):
    """Return check."""
    for actual, reference in zip(got, expected):
        if np.asarray(reference).dtype.kind in 'fc':
            np.testing.assert_allclose(actual, reference, rtol=1e-5,
                                       atol=1e-4)
        else:
            np.testing.assert_array_equal(actual, reference)


OPTION_SETS = [
    dict(),
    dict(median_high_variance=True),
    dict(preserve_saturated=True),
    dict(clear_interpolated_dq=True),
    dict(median_high_variance=True, preserve_saturated=True,
         clear_interpolated_dq=True),
]


@pytest.mark.parametrize('instrument', ['NIRISS', 'NIRSPEC', 'MIRI'])
@pytest.mark.parametrize('opts', OPTION_SETS)
def test_matches_numpy_reference(instrument, opts):
    """Check matches NumPy reference."""
    cube, err, dq, deepframe = make_dataset()
    got = badpix.badpixstep(
        cube, err, dq, deepframe, instrument=instrument,
        return_high_variance=True, **PARAMS, **opts)
    expected = ref_badpixstep(cube, err, dq, deepframe, instrument=instrument,
                              **PARAMS, **opts)
    _check(got, expected)


@pytest.mark.parametrize('instrument', ['NIRISS', 'NIRSPEC', 'MIRI'])
@pytest.mark.parametrize('opts', OPTION_SETS)
def test_prepared_primitive_composition_matches_wrapper(instrument, opts):
    """Check prepared primitive composition matches wrapper."""
    cube, err, dq, deepframe = make_dataset()
    prepared = run_prepared_badpix(
        cube, err, dq, deepframe, instrument=instrument, **PARAMS, **opts)
    wrapped = badpix.badpixstep(
        cube, err, dq, deepframe, instrument=instrument,
        return_high_variance=True, **PARAMS, **opts)
    for actual, expected in zip(prepared, wrapped):
        if np.asarray(expected).dtype.kind in 'fc':
            np.testing.assert_allclose(actual, expected, rtol=1e-6,
                                       atol=1e-6)
        else:
            np.testing.assert_array_equal(actual, expected)


def test_spatial_preparation_is_reusable_across_thresholds_and_vmappable():
    """Check spatial preparation is reusable across thresholds and vmappable."""
    cube, err, dq, deepframe = make_dataset()
    _, _, saturated_any = badpix.prepare_badpix_common(err, dq)
    spatial_prep = badpix.prepare_spatial_badpix(
        deepframe, dq, saturated_any, box_size=5, instrument='NIRISS')
    thresholds = jnp.asarray([5., 10., 15.], dtype=jnp.float32)
    maps = jax.vmap(
        lambda threshold: badpix.spatial_badpix_from_prepared(
            *spatial_prep, space_thresh=threshold))(thresholds)
    for threshold, spatial_map in zip(np.asarray(thresholds), maps):
        expected = badpix.badpixstep(
            cube, err, dq, deepframe, space_thresh=threshold,
            time_thresh=10., box_size=5, window_size=5,
            instrument='NIRISS')
        np.testing.assert_array_equal(spatial_map, expected[3])


def test_temporal_preparation_reuses_one_cube_and_one_detector_plane():
    """Check temporal preparation reuses one cube and one detector plane."""
    cube, err, dq, deepframe = make_dataset()
    err_out, saturated, saturated_any = badpix.prepare_badpix_common(err, dq)
    spatial_prep = badpix.prepare_spatial_badpix(
        deepframe, dq, saturated_any, box_size=5, instrument='NIRISS')
    spatial_map = badpix.spatial_badpix_from_prepared(
        *spatial_prep, space_thresh=15.)
    spatial_data, spatial_dq = badpix.apply_spatial_badpix(
        jnp.where(jnp.asarray(cube) < 0, 0., cube), dq, spatial_map,
        box_size=5, instrument='NIRISS')
    cube_filt, std_dev = badpix.prepare_temporal_badpix(
        spatial_data, window_size=5, instrument='NIRISS',
        space_thresh=15., box_size=5)
    assert cube_filt.shape == cube.shape
    assert std_dev.shape == deepframe.shape
    # A caller-supplied (chunk-assembled) plane is used verbatim.
    _, supplied = badpix.prepare_temporal_badpix(
        spatial_data, window_size=5, instrument='NIRISS', scatter=std_dev)
    np.testing.assert_array_equal(supplied, std_dev)

    for threshold in (5., 8., 11.):
        got = run_prepared_badpix(
            cube, err, dq, deepframe, space_thresh=15.,
            time_thresh=threshold, box_size=5, window_size=5,
            instrument='NIRISS')
        expected = badpix.badpixstep(
            cube, err, dq, deepframe, space_thresh=15.,
            time_thresh=threshold, box_size=5, window_size=5,
            instrument='NIRISS', return_high_variance=True)
        for actual, reference in zip(got, expected):
            np.testing.assert_allclose(actual, reference, rtol=1e-6,
                                       atol=1e-6)


@pytest.mark.parametrize('box_size,window_size', [(3, 3), (4, 7), (8, 11)])
def test_soss_shipped_box_and_window_sweeps_match_v1_reference(
        box_size, window_size):
    """Check SOSS shipped box and window sweeps match v1 reference."""
    cube, err, dq, deepframe = make_dataset()
    kwargs = dict(space_thresh=15., time_thresh=10.,
                  box_size=box_size, window_size=window_size,
                  instrument='NIRISS')
    got = badpix.badpixstep(cube, err, dq, deepframe,
                            return_high_variance=True, **kwargs)
    _check(got, ref_badpixstep(cube, err, dq, deepframe, **kwargs))


def test_flagging_and_replacement():
    """Check flagging and replacement."""
    cube, err, dq, deepframe = make_dataset()
    data, err_o, dq_o, bp = badpix.badpixstep(
        cube, err, dq, deepframe, instrument='NIRISS', **PARAMS)
    data, dq_o, bp = np.asarray(data), np.asarray(dq_o), np.asarray(bp)

    assert bp[7, 10] and bp[4, 20] and bp[9, 14] and bp[6, 25]
    # 2.5.0 default: saturated pixels are interpolated like any other.
    assert bp[9, 22]
    assert not bp[3, 8]
    assert np.all(np.abs(data[:, 9, 14] - 100.) < 5.)
    for (j, i) in [(7, 10), (4, 20), (6, 25)]:
        exp = [_box_median(np.clip(cube[n], 0, None), 0, 5, j, i)
               for n in range(len(cube))]
        np.testing.assert_allclose(data[:, j, i], exp, rtol=1e-5)

    # Preserve_saturated=True: never flagged, data untouched.
    data, _, _, bp = map(np.asarray, badpix.badpixstep(
        cube, err, dq, deepframe, instrument='NIRISS',
        preserve_saturated=True, **PARAMS))
    assert not bp[9, 22]
    np.testing.assert_array_equal(data[:, 9, 22], cube[:, 9, 22])


def test_temporal_spike():
    """Check temporal spike."""
    cube, err, dq, deepframe = make_dataset()
    data = np.asarray(badpix.badpixstep(cube, err, dq, deepframe,
                                        instrument='NIRISS', **PARAMS)[0])
    assert abs(data[4, 8, 18] - cube[4, 8, 18]) > 30.
    assert abs(data[4, 8, 18] - 100.) < 5.
    for n in [0, 1, 2, 3, 5, 6, 7, 8, 9]:
        assert data[n, 8, 18] == cube[n, 8, 18]


def test_dq_kept_by_default_and_cleared_on_request():
    """Check DQ kept by default and cleared on request."""
    cube, err, dq, deepframe = make_dataset()
    data, err_o, dq_o, _ = badpix.badpixstep(cube, err, dq, deepframe,
                                             instrument='NIRISS', **PARAMS)
    data, err_o, dq_o = np.asarray(data), np.asarray(err_o), np.asarray(dq_o)
    sl = np.s_[:, 2:4, 6:9]
    np.testing.assert_array_equal(data[sl], cube[sl])
    # 2.5.0: interpolated pixels keep their DQ record, DQ is untouched.
    np.testing.assert_array_equal(dq_o, dq)
    assert np.isfinite(err_o).all()
    assert np.all(data[:, :, :5] == 0)
    assert np.all(data[:, :, -5:] == 0)
    assert np.all(data[:, -5:] == 0)

    # Legacy option: cleared at interpolated pixels, SATURATED re-OR'd.
    _, _, dq_c, _ = map(np.asarray, badpix.badpixstep(
        cube, err, dq, deepframe, instrument='NIRISS',
        clear_interpolated_dq=True, **PARAMS))
    assert np.all(dq_c[:, 7, 10] == 0)
    assert np.all(dq_c[:, 4, 20] == 0)
    assert np.all(dq_c[:, 9, 22] & SAT)


def test_high_variance_flag_plane_and_median_replacement():
    """Check high variance flag plane and median replacement."""
    cube, err, dq, deepframe = make_dataset()
    cube[:, 3, 16] += np.linspace(0., 60., cube.shape[0]).astype(
        np.float32)
    kwargs = dict(instrument='NIRISS', return_high_variance=True, **PARAMS)
    _, _, _, _, hv = badpix.badpixstep(cube, err, dq, deepframe, **kwargs)
    assert np.asarray(hv)[3, 16]
    d_plain = np.asarray(badpix.badpixstep(
        cube, err, dq, deepframe, **kwargs)[0])
    d_med = np.asarray(badpix.badpixstep(
        cube, err, dq, deepframe, median_high_variance=True, **kwargs)[0])
    col = d_med[:, 3, 16]
    assert np.ptp(col) == 0.
    assert np.ptp(d_plain[:, 3, 16]) > 0.
    # DQ never carries the 2**32 bit (uint32).
    assert badpix.badpixstep(cube, err, dq, deepframe, **kwargs)[2].dtype == \
        np.uint32


def test_supplied_spatial_map_is_reused_without_redetection():
    """Check supplied spatial map is reused without redetection."""
    cube, err, dq, deepframe = make_dataset()
    supplied = np.zeros(deepframe.shape, dtype=bool)
    supplied[3, 12] = True
    dq[-1, 4, 20] |= HOT

    data, _, dq_out, returned = badpix.badpixstep(
        cube, err, dq, deepframe, instrument='NIRISS',
        spatial_badpix=supplied, space_thresh=1e6, time_thresh=1e6,
        box_size=3, window_size=5, clear_interpolated_dq=True)
    data, dq_out, returned = map(np.asarray, (data, dq_out, returned))
    np.testing.assert_array_equal(returned, supplied)
    assert np.all(dq_out[:, 3, 12] == 0)
    assert dq_out[-1, 4, 20] & HOT
    assert np.all(np.isfinite(data))


def test_column_halo_driver_matches_whole_spatial_kernel():
    """Check column halo driver matches whole spatial kernel."""
    cube, err, dq, deepframe = make_dataset()
    badmap = np.zeros(deepframe.shape, bool)
    badmap[9, 14] = badmap[7, 10] = True
    whole = tuple(map(np.asarray, badpix.apply_spatial_badpix(
        cube, dq, badmap, box_size=5, instrument='NIRISS',
        clear_interpolated_dq=True)))
    chunked = core.map_over_cols_with_halo(
        lambda d, q, b: badpix.apply_spatial_badpix(
            d, q, b, box_size=5, instrument='NIRISS',
            clear_interpolated_dq=True),
        (cube, dq, badmap), n_cols=cube.shape[-1], halo=6, chunk_size=7)
    for expected, actual in zip(whole, chunked):
        np.testing.assert_array_equal(actual, expected)
