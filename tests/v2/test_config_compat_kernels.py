"""Check configuration-dependent kernels against v1 references."""

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from exotedrf.v2 import core, products, stages
from exotedrf.v2.core import ObsMeta, RampCube, RateCube
from exotedrf.v2.kernels import background as k_bkg
from exotedrf.v2.kernels import pca as k_pca
from exotedrf.v2.pipeline import PipelineState


# SOSS background: v1 reference (one group, i.e. ngroup == 1 deepstack).

def _ref_scale(ratio):
    """Calculate scale with the reference equations."""
    finite = np.isfinite(ratio)
    if not np.any(finite):
        return np.nan
    vals = ratio[finite]
    q1, q2 = np.nanpercentile(vals, [25, 50])
    use = finite & (ratio > q1) & (ratio < q2)
    if not np.any(use):
        use = finite
    return np.nanmedian(ratio[use])


def ref_backgroundstep_soss(deep, model, scale1=None, coords1=None,
                            scale2=None, coords2=None, differential=False):
    """Calculate SOSS background subtraction with the v1 equations.

    Parameters
    ----------
    deep : array-like(float)
        Deep array.
    model : array-like(float)
        Model array.
    scale1 : None, float
        Background scale for the first region.
    coords1 : None, tuple[int]
        Detector bounds for background scaling.
    scale2 : None, float
        Background scale for the second region.
    coords2 : None, tuple[int]
        Detector bounds for background scaling.
    differential : bool
        Differential option.

    Returns
    -------
    result : np.ndarray(float)
        Calculated reference array.
    """
    dimy, dimx = deep.shape
    deep = deep.astype(np.float64)
    model = model.astype(np.float64)
    shift = 0.
    if scale1 is None:
        if coords1 is None:
            xl, xu, yl, yu = 230, 250, 350, 550
        else:
            xl, xu, yl, yu = np.array(coords1).astype(int)
        s1 = -1000
        while s1 < 0:
            s1 = _ref_scale((deep[xl:xu, yl:yu] + shift) /
                            model[xl:xu, yl:yu])
            if not np.isfinite(s1):
                s1 = 0.
                break
            if s1 < 0:
                shift -= s1 * np.nanmedian(model[xl:xu, yl:yu])
    else:
        s1 = np.atleast_1d(scale1)[0]
    if scale2 is None and differential is True:
        if coords2 is None:
            xl, xu, yl, yu = 235, 250, 715, 750
        else:
            xl, xu, yl, yu = np.array(coords2).astype(int)
        s2 = _ref_scale((deep[xl:xu, yl:yu] + shift) / model[xl:xu, yl:yu])
        if not np.isfinite(s2):
            s2 = 0.
        elif s2 < 0:
            s2 = 0
    elif scale2 is not None and differential is True:
        s2 = np.atleast_1d(scale2)[0]
    else:
        s2 = s1
    out = np.zeros_like(deep)
    if differential is True:
        grad = np.gradient(model, axis=1)
        step = np.argmax(grad[:, 10:-10], axis=1) + 10 - 4
        for j in range(256):
            out[j, :step[j]] = model[j, :step[j]] * s1 - shift
            out[j, step[j]:] = model[j, step[j]:] * s2 - shift
    else:
        out = model * s1 - shift
    return out


def _soss_scene(seed=3, dimx=1024):
    """Return SOSS scene."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:256, :dimx]
    model = (1. + 0.3 * (xx > 700) + 0.001 * yy).astype(np.float32)
    deep = (2.2 * model + rng.normal(0, 0.05, model.shape)).astype(
        np.float32)
    deep[100:120] += 50.
    return deep, model


@pytest.mark.parametrize('options', [
    {},
    {'coords1': (200, 240, 300, 500)},
    {'scale1': 1.7},
    {'scale1': -0.3},
    {'differential': True},
    {'differential': True, 'coords2': (230, 250, 720, 760)},
    {'differential': True, 'scale1': 2.0, 'scale2': 2.6},
    {'differential': True, 'scale1': 2.0},
    {'scale2': 9.0},
])
def test_background_soss_user_scales_and_regions_match_v1(options):
    """Check background SOSS user scales and regions match v1."""
    deep, model = _soss_scene()
    cube = np.stack([deep, deep + 1.])
    kwargs = {'region': None}
    if 'coords1' in options:
        kwargs['region'] = options['coords1']
    if options.get('differential'):
        kwargs['differential'] = True
        if 'coords2' in options:
            kwargs['region2'] = options['coords2']
        if 'scale2' in options:
            kwargs['scale2'] = options['scale2']
    if 'scale1' in options:
        kwargs['scale1'] = options['scale1']
    region = kwargs.pop('region')
    corr, scaled, *_ = k_bkg.background_soss(
        jnp.asarray(cube), jnp.asarray(deep), jnp.asarray(model),
        region1=region, **kwargs)
    ref = ref_backgroundstep_soss(deep, model, **options)
    np.testing.assert_allclose(np.asarray(scaled), ref, rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_allclose(np.asarray(corr), cube - ref[None],
                               rtol=1e-5, atol=1e-4)


def _soss_meta(nints, ngroups, subarray='SUBSTRIP256'):
    """Return SOSS meta."""
    return ObsMeta('NIRISS/SOSS', 'NIS', subarray, 5.494, ngroups,
                   np.arange(nints, dtype=float), np.array([nints]),
                   np.array([nints]), ('seg001.fits',), {})


def test_soss_background_calls_defaults_and_v1_assertions():
    """Check SOSS background calls defaults and v1 assertions."""
    assert stages.soss_background_calls({}, 'soss_background_grp', 2, 256,
                                         None) == [{'region': None}] * 2
    calls = stages.soss_background_calls(
        {'soss_background_grp': {'scale1': (1., 2.), 'differential': True,
                                 'background_coords2': (1, 2, 3, 4)}},
        'soss_background_grp', 2, 256, None)
    assert calls[1] == {'region': None, 'differential': True,
                        'region2': (1, 2, 3, 4), 'scale1': 2.}
    with pytest.raises(ValueError, match='one value per group'):
        stages.soss_background_calls(
            {'soss_background_grp': {'scale1': (1.,)}},
            'soss_background_grp', 3, 256, None)
    with pytest.raises(NotImplementedError, match='SUBSTRIP256'):
        stages.soss_background_calls(
            {'soss_background_int': {'differential': True}},
            'soss_background_int', 1, 96, None)


def test_background_steps_apply_translated_stage_kwargs():
    """Check background steps apply translated stage keywords."""
    deep, model = _soss_scene(dimx=1024)
    nints = 4
    rate = np.stack([deep] * nints).astype(np.float32)
    cube = RateCube(rate, np.ones_like(rate), np.zeros_like(rate, np.uint32),
                    _soss_meta(nints, 1))
    coords = (200, 240, 300, 500)
    ctx = {'opts': {'soss_background_int': {'background_coords1': coords}},
           'background_model': model, 'centroids': {
               'ypos o1': np.full(1024, 60.)}}
    out = stages.step_background_int(PipelineState(cube), {}, ctx)
    ref = ref_backgroundstep_soss(deep, model, coords1=coords)
    np.testing.assert_allclose(out.aux['bkg_int'], ref, rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_allclose(out.cube.data, rate - ref[None], atol=1e-4)

    ramp = np.stack([np.stack([deep, deep * 2.])] * nints).astype(np.float32)
    rcube = RampCube(ramp, np.zeros_like(ramp, np.uint8),
                     np.zeros((256, 1024), np.uint32), _soss_meta(nints, 2))
    ctx = dict(ctx, opts={'soss_background_grp': {'scale1': (1.25, 3.5)},
                          'oof_method': 'scale-achromatic'})
    out = stages.step_background_grp(PipelineState(rcube), {}, ctx)
    np.testing.assert_allclose(out.aux['bkg_grp'][0], model * 1.25,
                               rtol=1e-6)
    np.testing.assert_allclose(out.aux['bkg_grp'][1], model * 3.5,
                               rtol=1e-6)


# PCA: removal beyond pca_components, skip_pca, lcestimate for MIRI/NIRSpec.

def _svd_flip(u, vt):
    """Return svd flip."""
    idx = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[idx, np.arange(u.shape[1])])
    signs[signs == 0] = 1.
    return u * signs, vt * signs[:, None]


def ref_stability_pca(cube, n):
    """Fit stability components using the v1 PCA calculation.

    Parameters
    ----------
    cube : array-like(float)
        Input observation cube.
    n : int
        Number of PCA components.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    nints, dimy, dimx = cube.shape
    x = cube.reshape(nints, dimy * dimx).astype(np.float64).copy()
    x[np.isnan(x)] = np.nanmedian(x)
    xt = x.T
    mean = xt.mean(axis=0)
    u, s, vt = np.linalg.svd(xt - mean, full_matrices=False)
    u, vt = _svd_flip(u, vt)
    recon = (u[:, :n] * s[:n]) @ vt[:n] + mean
    return vt[:n], recon.T.reshape(nints, dimy, dimx)


def ref_remove(cube, remove):
    """Remove the selected PCA components using the v1 calculation.

    Parameters
    ----------
    cube : array-like(float)
        Input observation cube.
    remove : list[int]
        PCA components to remove.

    Returns
    -------
    result : np.ndarray(float)
        Calculated reference array.
    """
    new = cube.astype(np.float64).copy()
    for pc in np.atleast_1d(remove):
        if pc != 1:
            this = ref_stability_pca(cube, pc)[1] - \
                ref_stability_pca(cube, pc - 1)[1]
        else:
            this = ref_stability_pca(cube, pc)[1]
        new -= this
    return new


def ref_format_out_frames(out_frames):
    """Format output frames using the v1 conventions.

    Parameters
    ----------
    out_frames : list
        Per-segment extraction results.

    Returns
    -------
    result : np.ndarray(int)
        Calculated reference array.
    """
    out_frames = np.atleast_1d(out_frames)
    if len(out_frames) == 1:
        if out_frames[0] > 0:
            return np.arange(out_frames[0])
        return np.arange(abs(out_frames[0])) - abs(out_frames[0])
    out_frames = np.abs(out_frames)
    return np.concatenate([np.arange(out_frames[0]),
                           np.arange(out_frames[1]) - out_frames[1]])


def _pca_cube(nints=12, dimy=10, dimx=16, seed=11):
    """Return PCA cube."""
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, nints)
    modes = [np.sin(3 * t) * 400., t * 90., np.cos(5 * t) * 30.,
             t ** 2 * 12., np.sin(9 * t) * 5.]
    cube = np.full((nints, dimy, dimx), 1000., np.float64)
    for k, mode in enumerate(modes):
        cube += mode[:, None, None] * rng.normal(size=(1, dimy, dimx))
    cube += rng.normal(0, 0.05, cube.shape)
    return cube.astype(np.float32)


def test_pca_removal_above_pca_components_refits_like_v1():
    """Check PCA removal above PCA components refits like v1."""
    cube = _pca_cube()
    n_comp = 2
    remove = np.zeros(4, bool)
    remove[[1, 3]] = True
    baseline = np.zeros(cube.shape[0], bool)
    baseline[:3] = baseline[-3:] = True
    new, comps, var, wlc, comps_rec, _ = k_pca.pca_reconstruction(
        cube, remove, baseline, n_components=n_comp)
    np.testing.assert_allclose(np.asarray(new), ref_remove(cube, [2, 4]),
                               rtol=1e-5, atol=2e-3)
    ref_pcs = ref_stability_pca(cube, n_comp)[0]
    assert comps.shape == (n_comp, cube.shape[0])
    np.testing.assert_allclose(comps, ref_pcs, rtol=1e-4, atol=1e-5)
    # Reported PCs of the reconstructed cube use pca_components too.
    np.testing.assert_allclose(
        comps_rec, ref_stability_pca(np.asarray(ref_remove(cube, [2, 4]),
                                                np.float32), n_comp)[0],
        rtol=1e-3, atol=1e-4)
    with pytest.raises(ValueError, match='remove_mask'):
        k_pca.pca_reconstruction(cube, np.zeros(1, bool), baseline,
                                 n_components=n_comp)


def _rate_meta(mode, nints, baseline, detector='NRS1'):
    """Return rate meta."""
    return ObsMeta(mode, detector, 'SUB2048', 1., 5,
                   np.arange(nints, dtype=float), np.asarray(baseline),
                   np.array([nints]),
                   (f'jw01_04101_00001-seg001_{detector.lower()}_uncal.fits',),
                   {})


def test_step_pca_accepts_large_indices_and_skip_pca():
    """Check step PCA accepts large indices and skip PCA."""
    cube = _pca_cube()
    meta = _rate_meta('NIRSpec/G395H', cube.shape[0], [3, -3])
    state = PipelineState(RateCube(cube, np.ones_like(cube),
                                   np.zeros_like(cube, np.uint32), meta))
    opts = {'pca_components': 2, 'remove_components': [3]}
    stages._validate_nirspec_context({'opts': dict(opts)})
    stages._validate_miri_context({'opts': dict(opts)})
    out = stages.step_pca(state, {}, {'opts': opts})
    np.testing.assert_allclose(out.cube.data, ref_remove(cube, [3]),
                               rtol=1e-5, atol=2e-3)
    assert out.aux['pca_components'].shape == (2, cube.shape[0])

    skipped = stages.step_pca(state, {}, {'opts': dict(opts, skip_pca=True)})
    assert skipped.cube.data is state.cube.data
    assert 'pca_components' not in skipped.aux
    assert 'pca_wlc' not in skipped.aux
    np.testing.assert_allclose(
        skipped.aux['stage3_deepframe'],
        np.median(cube[ref_format_out_frames([3, -3])], axis=0), rtol=1e-6)


@pytest.mark.parametrize('mode, detector, name', [
    ('MIRI/LRS', 'MIRIMAGE', 'jw01_04101_00001_mirimage_lcestimate.npy'),
    ('NIRSpec/G395H', 'NRS1', 'jw01_04101_00001_nrs1_lcestimate.npy')])
@pytest.mark.parametrize('baseline', [[4, -3], [5], [-4], [9, -9]])
def test_lcestimate_written_for_miri_and_nirspec_like_v1(
        tmp_path, monkeypatch, mode, detector, name, baseline):
    """v1 PCAReconstructStep writes lcestimate.npy for every instrument."""
    monkeypatch.setattr('exotedrf.v2.io.save_rate_cube',
                        lambda cube, path, extra_header=None: None)
    dimx = 72 if mode.startswith('MIRI') else 16
    cube = _pca_cube(nints=14, dimx=dimx, seed=5)
    meta = _rate_meta(mode, cube.shape[0], baseline, detector)
    state = PipelineState(RateCube(cube, np.ones_like(cube),
                                   np.zeros_like(cube, np.uint32), meta))
    ctx = {'opts': {'pca_components': 3, 'remove_components': None,
                    'generate_lc': True}}
    out = stages.step_pca(state, {}, ctx)
    # Centroid/deepframe sidecars need a real trace; only lcestimate is under test here.
    out.aux.pop('stage3_deepframe', None)
    paths = products.output_layout({'name_tag': 'visit'},
                                   output_dir=tmp_path, create=True)
    written = products.write_final_products(out, {}, ctx, paths)
    assert Path(written['lcestimate']).name == name
    clipped = cube[:, :, 12:61] if mode.startswith('MIRI') else cube
    pcs = ref_stability_pca(clipped, 3)[0]
    ref = pcs[0] / np.nanmedian(pcs[0][ref_format_out_frames(baseline)])
    rtol = 3e-5 if baseline == [9, -9] else 1e-5
    np.testing.assert_allclose(np.load(written['lcestimate']), ref,
                               rtol=rtol, atol=1e-6)


# DQInit: flag_neighbours and MIRI first/last-group flags.

def ref_saturation(data, threshold_adu, flag_neighbours):
    """Flag saturated pixels with optional neighbor dilation.

    Parameters
    ----------
    data : array-like(float)
        Data array.
    threshold_adu : float
        Saturation threshold in detector counts.
    flag_neighbours : bool
        Flag neighbours option.

    Returns
    -------
    result : np.ndarray(float)
        Calculated reference array.
    """
    inds = data >= threshold_adu
    nints, ngroups, ydim, xdim = data.shape
    expanded = np.copy(inds)
    for i, g, y, x in zip(*np.where(inds)):
        y0, y1 = max(0, y - flag_neighbours), min(ydim, y + flag_neighbours
                                                  + 1)
        x0, x1 = max(0, x - flag_neighbours), min(xdim, x + flag_neighbours
                                                  + 1)
        expanded[i, g, y0:y1, x0:x1] = True
    return expanded


@pytest.mark.parametrize('flag_neighbours', [0, 2])
@pytest.mark.parametrize('first, last', [(False, True), (True, False),
                                         (False, False)])
def test_dq_init_miri_frame_flags_and_neighbour_box(flag_neighbours, first,
                                                    last):
    """Check DQ init MIRI frame flags and neighbour box."""
    nints, ngroups, dimy, dimx = 2, 4, 9, 11
    data = np.full((nints, ngroups, dimy, dimx), 100., np.float32)
    data[1, 2, 4, 5] = 60000.
    data[0, 3, 0, 10] = 60000.
    meta = ObsMeta('MIRI/LRS', 'MIRIMAGE', 'SLITLESSPRISM', 0.159, ngroups,
                   np.arange(nints, dtype=float), np.array([1]),
                   np.array([nints]), ('seg001.fits',), {})
    cube = RampCube(data, np.zeros_like(data, np.uint8),
                    np.zeros((dimy, dimx), np.uint32), meta)
    opts = {'saturation_threshold': 80, 'flag_neighbours': flag_neighbours,
            'flag_first_miri_frame': first, 'flag_last_miri_frame': last}
    out = stages.step_dq_init(PipelineState(cube), {}, {
        'opts': opts, 'refpack': {'mask_dq': np.zeros((dimy, dimx),
                                                      np.uint32)}})
    groupdq = np.asarray(out.cube.groupdq)
    sat = ref_saturation(data, core.FULL_WELL_ADU[('MIRI', None)] * 0.8,
                         flag_neighbours)
    np.testing.assert_array_equal(
        (groupdq & np.uint8(core.DQ_SATURATED)) > 0, sat)
    dnu = (groupdq & np.uint8(core.DQ_DO_NOT_USE)) > 0
    assert bool(np.all(dnu[:, 0])) == first
    assert bool(np.all(dnu[:, -1])) == last
    assert not np.any(dnu[:, 1:-1])


# NIRSpec custom-rescale SuperBiasStep(mask_width, override_centroids).

def ref_rescale_factors(group0, superbias, dq, xpos, ypos, mask_width):
    """Calculate superbias scale factors with the v1 equations.

    Parameters
    ----------
    group0 : array-like(float)
        Group0 array.
    superbias : array-like(float)
        Superbias array.
    dq : array-like(int)
        Dq array.
    xpos : array-like(float)
        Xpos array.
    ypos : array-like(float)
        Ypos array.
    mask_width : int, float
        Trace-mask width.

    Returns
    -------
    result : np.ndarray(float)
        Calculated reference array.
    """
    nint, dimy, dimx = group0.shape
    low = np.max([np.zeros_like(ypos), ypos - mask_width / 2],
                 axis=0).astype(int)
    up = np.min([dimy * np.ones_like(ypos), ypos + mask_width / 2],
                axis=0).astype(int)
    tracemask = np.ones((dimy, dimx))
    for i, x in enumerate(xpos):
        tracemask[low[i]:up[i], int(x)] = 0
    mask = ~dq | tracemask.astype(bool)
    mask = np.where(mask == 0, np.nan, mask)
    return np.nanmedian(mask * group0 / superbias, axis=(1, 2))


@pytest.mark.parametrize('mask_width', [4, 9.5])
def test_nirspec_custom_rescale_mask_width_and_override(mask_width):
    """Check NIRSpec custom rescale mask width and override."""
    rng = np.random.default_rng(2)
    nints, ngroups, dimy, dimx = 4, 2, 12, 10
    yy = np.arange(dimy)[:, None]
    trace = 400. * np.exp(-0.5 * ((yy - 6.) / 1.2) ** 2)
    data = (rng.normal(50, 1, (nints, ngroups, dimy, dimx)) +
            trace[None, None]).astype(np.float32)
    data[:, 0] *= (1. + 0.01 * np.arange(nints))[:, None, None]
    dq = np.zeros((dimy, dimx), np.uint32)
    dq[5:8, :] = 1
    meta = ObsMeta('NIRSpec/G395H', 'NRS1', 'SUB2048', 0.902, ngroups,
                   np.arange(nints, dtype=float), np.array([2]),
                   np.array([nints]), ('seg001.fits',), {})
    cube = RampCube(data, np.zeros_like(data, np.uint8), dq, meta)
    # A short (trimmed-slit) trace is used only with override_centroids.
    xpos = np.arange(2, dimx, dtype=float)
    ypos = np.full(dimx - 2, 6.3)
    ctx = {'opts': {'superbias_method': 'custom-rescale',
                    'superbias_mask_width': mask_width,
                    'superbias_override_centroids': True},
           'centroids': {'xpos': xpos, 'ypos': ypos}}
    out = stages.step_superbias_nirspec(PipelineState(cube), {}, ctx)
    superbias = np.nanmedian(data[:, 0], axis=0)
    ref = ref_rescale_factors(data[:, 0].astype(np.float64), superbias,
                              dq.astype(bool), xpos, ypos, mask_width)
    np.testing.assert_allclose(out.aux['superbias_scale_factors'], ref,
                               rtol=1e-6)


# OneOverFStep(even_odd_rows) reaches the SOSS kernels.

def test_integration_oof_even_odd_rows_reaches_kernel(monkeypatch):
    """Check integration 1/f even odd rows reaches kernel."""
    from exotedrf.v2.kernels import oneoverf as k_oof
    rng = np.random.default_rng(4)
    nints, dimy, dimx = 6, 10, 12
    data = rng.normal(100., 1., (nints, dimy, dimx)).astype(np.float32)
    data[:, 1::2] += np.arange(dimx)[None, None] * 0.5
    cube = RateCube(data, np.ones_like(data), np.zeros_like(data, np.uint32),
                    _soss_meta(nints, 1, 'SUBSTRIP96'))
    seen = []
    original = k_oof.oneoverf_scale_achromatic

    def spy(*args, **kwargs):
        seen.append(kwargs.get('even_odd_rows', True))
        return original(*args, **kwargs)

    monkeypatch.setattr(stages.k_oof, 'oneoverf_scale_achromatic', spy)
    results = {}
    for flag in (True, False):
        ctx = {'opts': {'oof_method': 'scale-achromatic',
                        'oof_even_odd_rows_int': flag},
               'centroids': {'ypos o1': np.full(dimx, 5., np.float32)},
               'soss_timeseries': np.ones(nints, np.float32)}
        results[flag] = stages.step_oneoverf_int(
            PipelineState(cube), {'soss_inner_mask_width': 4.,
                                  'soss_outer_mask_width': 8.}, ctx)
    assert set(seen) == {True, False}
    assert not np.allclose(results[True].cube.data, results[False].cube.data)
