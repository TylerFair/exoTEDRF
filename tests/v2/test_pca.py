"""Check PCA fitting and reconstruction against v1 reference calculations."""

import inspect

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from exotedrf.v2.kernels import pca


# Numpy reference (float64).

def _svd_flip(u, vt):
    """Return svd flip."""
    idx = np.argmax(np.abs(vt), axis=1)
    signs = np.sign(vt[np.arange(vt.shape[0]), idx])
    signs[signs == 0] = 1.
    return u * signs, vt * signs[:, None]


def ref_stability_pca(cube, n_components):
    """Fit stability components using the v1 PCA calculation.

    Parameters
    ----------
    cube : array-like(float)
        Input observation cube.
    n_components : int
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
    pcs = vt[:n_components]
    ev = s ** 2 / (dimy * dimx - 1)
    var = ev[:n_components] / ev.sum()
    recon = (u[:, :n_components] * s[:n_components]) @ vt[:n_components] + mean
    return pcs, var, recon.T.reshape(nints, dimy, dimx)


def ref_remove_components(cube, remove_components):
    """Reconstruct a cube after removing the selected PCA components.

    Parameters
    ----------
    cube : array-like(float)
        Input observation cube.
    remove_components : list[int]
        PCA components to remove.

    Returns
    -------
    result : np.ndarray(float)
        Calculated reference array.
    """
    newcube = cube.astype(np.float64).copy()
    for pc in np.atleast_1d(remove_components):
        if pc != 1:
            out_ncmo = ref_stability_pca(cube, pc - 1)[2]
            out_nc = ref_stability_pca(cube, pc)[2]
            thiscomp = out_nc - out_ncmo
        else:
            thiscomp = ref_stability_pca(cube, pc)[2]
        newcube -= thiscomp
    return newcube


# Synthetic dataset: static frame + known temporal components.

def make_dataset(seed=7, with_nan=False):
    """Create a synthetic cube with structure and noise.

    Parameters
    ----------
    seed : int
        Random seed.
    with_nan : bool
        With nan option.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    rng = np.random.default_rng(seed)
    nints, dimy, dimx = 10, 16, 32
    yy, xx = np.mgrid[:dimy, :dimx]
    gauss = np.exp(-((yy - 8.) / 3.) ** 2)
    base = 10. + 200. * gauss
    # Component 1: transit-like white light curve (multiplies the trace).
    t1 = np.ones(nints)
    t1[4:7] -= 0.02
    # Component 2: linear drift of an asymmetric (odd-in-x) pattern.
    t2 = np.linspace(-1., 1., nints)
    p2 = 20. * (xx - dimx / 2.) / dimx * gauss
    # Component 3: breathing of an even-in-x pattern.
    t3 = ((np.arange(nints) - 4.5) ** 2 - 8.25) / 8.25
    p3 = 2. * np.cos(2. * np.pi * xx / dimx) * gauss
    cube = (base[None] * t1[:, None, None] +
            p2[None] * t2[:, None, None] +
            p3[None] * t3[:, None, None] +
            rng.normal(0., 0.002, (nints, dimy, dimx)))
    cube = cube.astype(np.float32)
    if with_nan:
        cube[2, 0, 0] = np.nan
    return cube, t1, t2, t3, p2


N_COMP = 3


def test_matches_numpy_reference():
    """Check matches NumPy reference."""
    cube = make_dataset()[0]
    pcs, var, recon = pca.soss_stability_pca(cube, n_components=N_COMP)
    ref_pcs, ref_var, ref_recon = ref_stability_pca(cube, N_COMP)
    np.testing.assert_allclose(np.asarray(pcs), ref_pcs,
                               rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(np.asarray(var), ref_var,
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(recon), ref_recon,
                               rtol=1e-5, atol=1e-2)


def test_temporal_decomposition_has_bounded_transfers(monkeypatch):
    """A full SUBSTRIP96 frame crosses the device boundary by columns."""
    nints, dimy, dimx, n_components = 12, 96, 2048, 3
    time = np.linspace(-1., 1., nints, dtype=np.float32)[:, None, None]
    yy = np.linspace(-1., 1., dimy, dtype=np.float32)[None, :, None]
    xx = np.linspace(-1., 1., dimx, dtype=np.float32)[None, None, :]
    cube = (20. + 3. * time * yy + (time ** 2) * xx).astype(np.float32)

    transfers = []
    original_chunks = pca._host_chunks

    def audited_chunks(c):
        for start, stop, chunk in original_chunks(c):
            transfers.append(chunk.shape)
            yield start, stop, chunk

    # The exact solver accumulates its float64 Gram matrix chunk by chunk.
    monkeypatch.setattr(pca, '_host_chunks', audited_chunks)
    host, fit = pca._fit_streamed(cube, n_components, solver='exact')
    assert host is cube
    assert fit.vectors.shape == (nints, n_components)
    assert fit.largest_solver_shape == (nints, nints)
    assert np.all(np.isfinite(fit.variance_ratio))
    assert transfers
    assert all(shape[:2] == (nints, dimy) for shape in transfers)
    assert max(shape[2] for shape in transfers) <= pca._COLUMN_CHUNK_SIZE
    assert max(shape[2] for shape in transfers) < dimx
    source = inspect.getsource(pca)
    assert 'linalg.svd' not in source


def test_forced_small_columns_match_unchunked(monkeypatch):
    """Check forced small columns match unchunked."""
    cube = make_dataset(with_nan=True)[0]
    monkeypatch.setattr(pca, '_COLUMN_CHUNK_SIZE', cube.shape[2])
    unchunked = pca.soss_stability_pca(cube, n_components=N_COMP)
    monkeypatch.setattr(pca, '_COLUMN_CHUNK_SIZE', 3)
    chunked = pca.soss_stability_pca(cube, n_components=N_COMP)
    np.testing.assert_allclose(chunked[0], unchunked[0],
                               rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(chunked[1], unchunked[1],
                               rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(chunked[2], unchunked[2],
                               rtol=1e-4, atol=2e-3)
    assert all(isinstance(value, np.ndarray) for value in chunked)


def test_randomized_solver_is_truncated_and_matches_reference(monkeypatch):
    """Check randomized solver is truncated and matches reference."""
    rng = np.random.default_rng(42)
    nints, dimy, dimx = 40, 8, 16
    npix = dimy * dimx
    temporal, _ = np.linalg.qr(rng.normal(size=(nints, 4)))
    spatial, _ = np.linalg.qr(rng.normal(size=(npix, 4)))
    centered = ((temporal * np.array([1000., 200., 50., 5.])) @
                spatial.T)
    cube = (centered + np.linspace(20., 22., nints)[:, None]) \
        .reshape(nints, dimy, dimx).astype(np.float32)
    monkeypatch.setattr(pca, '_COLUMN_CHUNK_SIZE', 3)

    transfers = []
    original_chunks = pca._host_chunks

    def audited_chunks(c):
        for start, stop, chunk in original_chunks(c):
            transfers.append(chunk.shape)
            yield start, stop, chunk

    # The exact solver accumulates its float64 Gram matrix chunk by chunk.
    monkeypatch.setattr(pca, '_host_chunks', audited_chunks)
    host, fit = pca._fit_streamed(cube, N_COMP, solver='randomized')
    signs = pca._scan_signs(fit)
    pcs = pca._signed_components(fit, signs)
    ref_pcs, ref_var, _ = ref_stability_pca(cube, N_COMP)
    assert fit.solver == 'randomized'
    assert fit.work_rank == N_COMP + pca._RANDOM_OVERSAMPLE
    assert fit.largest_solver_shape == (nints, fit.work_rank)
    assert fit.work_rank < nints
    assert all(shape[2] <= 3 for shape in transfers)
    np.testing.assert_allclose(pcs, ref_pcs, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(fit.variance_ratio, ref_var,
                               rtol=2e-4, atol=2e-6)


def test_float64_matches_reference_tightly():
    """Check float64 matches reference tightly."""
    cube = make_dataset()[0].astype(np.float64)
    old_x64 = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        pcs, var, recon = pca.soss_stability_pca(
            cube, n_components=N_COMP)
        assert pcs.dtype == jnp.float64
        assert var.dtype == jnp.float64
        assert recon.dtype == jnp.float64
        ref_pcs, ref_var, ref_recon = ref_stability_pca(cube, N_COMP)
        np.testing.assert_allclose(np.asarray(pcs), ref_pcs,
                                   rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.asarray(var), ref_var,
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(recon), ref_recon,
                                   rtol=1e-10, atol=1e-10)
    finally:
        jax.config.update('jax_enable_x64', old_x64)


def test_component_recovery_up_to_sign():
    """Check component recovery up to sign."""
    cube, t1, t2, t3, _ = make_dataset()
    pcs, var, _ = pca.soss_stability_pca(cube, n_components=N_COMP)
    pcs = np.asarray(pcs)

    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))

    assert abs(corr(pcs[0], t1)) > 0.99
    assert abs(corr(pcs[1], t2)) > 0.99
    assert abs(corr(pcs[2], t3)) > 0.99
    # Injected components dominate the variance budget in order.
    var = np.asarray(var)
    assert var[0] > 0.9
    assert var[1] > 5 * var[2]


def test_sign_deterministic():
    """Check sign deterministic."""
    cube = make_dataset()[0]
    pcs_a = np.asarray(pca.soss_stability_pca(cube, n_components=N_COMP)[0])
    pcs_b = np.asarray(pca.soss_stability_pca(cube, n_components=N_COMP)[0])
    np.testing.assert_array_equal(pcs_a, pcs_b)
    ref_pcs = ref_stability_pca(cube, N_COMP)[0]
    big = np.abs(ref_pcs) > 1e-3
    assert np.all(np.sign(pcs_a[big]) == np.sign(ref_pcs[big]))


def test_removal_matches_v1_incremental_reference():
    """Check removal matches v1 incremental reference."""
    cube = make_dataset(with_nan=True)[0]
    remove_mask = np.array([False, True, False])
    baseline_mask = np.zeros(len(cube), bool)
    baseline_mask[:3] = baseline_mask[-3:] = True

    newcube, pcs, var, wlc, pcs_r, var_r = pca.pca_reconstruction(
        cube, remove_mask, baseline_mask, n_components=N_COMP)
    newcube = np.asarray(newcube)
    ref_new = ref_remove_components(cube, [2])
    np.testing.assert_allclose(newcube, ref_new, rtol=1e-5, atol=1e-2,
                               equal_nan=True)
    assert np.isnan(newcube[2, 0, 0])

    ref_pcs = ref_stability_pca(cube, N_COMP)[0]
    ref_wlc = ref_pcs[0] / np.nanmedian(ref_pcs[0][baseline_mask])
    np.testing.assert_allclose(np.asarray(wlc), ref_wlc, rtol=1e-5,
                               atol=1e-5)

    ref_pcs_r = ref_stability_pca(ref_new.astype(np.float32), N_COMP)[0]
    np.testing.assert_allclose(np.asarray(pcs_r)[:2], ref_pcs_r[:2],
                               rtol=1e-3, atol=1e-3)


def test_plot_projection_capture_is_opt_in_and_matches_v1_transform():
    """Plot maps are signed sklearn transform scores, not reconstructions."""
    cube = make_dataset(with_nan=True)[0]
    remove_mask = np.array([False, True, False])
    baseline_mask = np.ones(len(cube), bool)

    plain = pca.pca_reconstruction(
        cube, remove_mask, baseline_mask, n_components=N_COMP)
    assert len(plain) == 6
    result = pca.pca_reconstruction(
        cube, remove_mask, baseline_mask, n_components=N_COMP,
        return_plot_data=True)
    assert len(result) == 8
    for plotted_value, plain_value in zip(result[:6], plain):
        np.testing.assert_array_equal(
            np.asarray(plotted_value), np.asarray(plain_value))
    newcube, pcs, _, _, pcs_r, _, projections, projections_r = result

    def transform_maps(source, components):
        host = pca._host_cube(source)
        fill = pca._fill_value(host)
        mean = pca._feature_mean(host, fill)
        flat = host.reshape(host.shape[0], -1)
        centered = np.where(np.isnan(flat), fill, flat) - mean[:, None]
        return np.asarray(components @ centered).reshape(
            len(components), *host.shape[1:])

    expected = transform_maps(cube, pcs)
    expected_r = transform_maps(newcube, pcs_r)
    assert projections.shape == (N_COMP,) + cube.shape[1:]
    assert projections_r.shape == (N_COMP,) + cube.shape[1:]
    np.testing.assert_allclose(projections, expected,
                               rtol=1e-5, atol=2e-4)
    np.testing.assert_allclose(projections_r, expected_r,
                               rtol=1e-5, atol=2e-4)


def test_removal_removes_component_from_reconstruction():
    """Check removal removes component from reconstruction."""
    cube, t1, t2, t3, p2 = make_dataset()
    remove_mask = np.array([False, True, False])
    baseline_mask = np.ones(len(cube), bool)
    newcube = np.asarray(pca.pca_reconstruction(
        cube, remove_mask, baseline_mask, n_components=N_COMP)[0])

    # Amplitude of the injected drift pattern in each integration.
    def amplitude(c):
        resid = c - c.mean(axis=0)
        p = p2 - p2.mean()
        return np.array([(f * p).sum() / (p * p).sum() for f in resid])

    amp_before = amplitude(cube.astype(np.float64))
    amp_after = amplitude(newcube)
    # The drift signal is essentially gone from the reconstruction.
    assert amp_after.std() < amp_before.std() / 10.
    unchanged = pca.pca_reconstruction(
        cube, np.zeros(N_COMP, bool), baseline_mask,
        n_components=N_COMP)
    same = np.asarray(unchanged[0])
    np.testing.assert_allclose(same, cube, rtol=1e-6, atol=1e-4)
    np.testing.assert_array_equal(np.asarray(unchanged[1]),
                                  np.asarray(unchanged[4]))
    np.testing.assert_array_equal(np.asarray(unchanged[2]),
                                  np.asarray(unchanged[5]))


def test_removal_of_component_one_includes_mean():
    """Check removal of component one includes mean."""
    cube = make_dataset()[0]
    remove_mask = np.array([True, False, False])
    baseline_mask = np.ones(len(cube), bool)
    newcube = np.asarray(pca.pca_reconstruction(
        cube, remove_mask, baseline_mask, n_components=N_COMP)[0])
    ref_new = ref_remove_components(cube, [1])
    np.testing.assert_allclose(newcube, ref_new, rtol=1e-5, atol=1e-2)


def test_remove_mask_values_reuse_chunk_executable():
    """Different selections reuse the compiled bounded-chunk kernel."""
    cube = make_dataset()[0]
    baseline_mask = np.ones(len(cube), bool)
    _ = pca.pca_reconstruction(
        cube, np.array([False, True, False]), baseline_mask,
        n_components=N_COMP)
    n = pca._chunk_projection._cache_size()
    _ = pca.pca_reconstruction(
        cube, np.array([True, False, True]), baseline_mask,
        n_components=N_COMP)
    assert pca._chunk_projection._cache_size() == n


def test_matches_sklearn_arpack_oracle():
    """v1 2.5.0 soss_stability_pca fits PCA(svd_solver='arpack')."""
    sklearn_pca = pytest.importorskip('sklearn.decomposition')
    cube = make_dataset(with_nan=True)[0]
    nints, dimy, dimx = cube.shape
    x = cube.reshape(nints, dimy * dimx).copy()
    x[np.isnan(x)] = np.nanmedian(x)
    model = sklearn_pca.PCA(n_components=N_COMP, svd_solver='arpack')
    model.fit(x.T)
    proj = model.transform(x.T)
    ref_recon = model.inverse_transform(proj).T.reshape(nints, dimy, dimx)
    pcs, var, recon = pca.soss_stability_pca(cube, n_components=N_COMP)
    np.testing.assert_allclose(np.asarray(var), model.explained_variance_ratio_,
                               rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(pcs), model.components_,
                               rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(np.asarray(recon), ref_recon,
                               rtol=1e-4, atol=1e-2)


@pytest.mark.parametrize('solver', ['exact', 'randomized'])
def test_weak_components_beside_static_image(solver):
    """Recover nearby weak components without forming a float64 cube."""
    rng = np.random.default_rng(31)
    nints, dimy, dimx = 96, 8, 64
    temporal = rng.normal(size=(nints, 4))
    temporal[:, 0] = 1. + .02 * np.sin(np.linspace(0., 6., nints))
    temporal, _ = np.linalg.qr(temporal)
    spatial = rng.normal(size=(dimy * dimx, 4))
    spatial -= spatial.mean(axis=0)
    spatial, _ = np.linalg.qr(spatial)
    flat = ((temporal * [1e7, 2e4, 1.999e4, 1.998e4]) @ spatial.T)
    cube = (flat + 1e6).reshape(nints, dimy, dimx).astype(np.float32)
    reference, variance, reconstruction = ref_stability_pca(cube, 4)
    host, fit = pca._fit_streamed(cube, 4, solver=solver)
    cosine = np.abs(reference @ fit.vectors)
    assert np.all(np.diag(cosine)[variance > 1e-6] >= .9999)
    result, _ = pca._project_to_host(
        host, fit, np.ones(4, dtype=fit.dtype), 1., False)
    assert np.max(np.abs(result - reconstruction)) / np.ptp(cube) < 1e-4


@pytest.mark.parametrize('budget, max_nints, expected', [
    (1e9, 16384, 'exact'),
    (1., 16384, 'randomized'),
    (1e9, 8, 'randomized'),
])
def test_solver_selection_limits_product_work(monkeypatch, budget,
                                              max_nints, expected):
    """Both product work and temporal matrix size constrain exact fits."""
    cube = np.random.default_rng(9).normal(size=(64, 8, 16)).astype(np.float32)
    monkeypatch.setattr(pca, '_EXACT_FLOP_BUDGET', budget)
    monkeypatch.setattr(pca, '_EXACT_MAX_NINTS', max_nints)
    _, fit = pca._fit_streamed(cube, N_COMP)
    assert fit.solver == expected
