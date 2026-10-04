"""Calculate temporal principal components using detector column chunks."""

from dataclasses import dataclass
import functools
import os

import jax
import jax.numpy as jnp
from jax import enable_x64 as _enable_x64
import numpy as np
from scipy.linalg import eigh

from exotedrf.v2 import core, hoststats


_MATMUL_PRECISION = jax.lax.Precision.HIGHEST
_COLUMN_CHUNK_SIZE = int(os.environ.get('EXOTEDRF_V2_PCA_COLUMN_CHUNK', '512'))
# Limit the temporal covariance matrix to 2 GiB.
_EXACT_MAX_NINTS = 16384
_EXACT_FLOP_BUDGET = 4e12
_RANDOM_OVERSAMPLE = 20
_RANDOM_POWER_ITERATIONS = 6
_RANDOM_SEED = 319482278


@dataclass(frozen=True)
class _PCAFit:
    """Store temporal components and projection metadata."""

    vectors: np.ndarray
    variance_ratio: np.ndarray
    mean: np.ndarray
    fill_value: np.generic
    dtype: np.dtype
    solver: str
    work_rank: int
    largest_solver_shape: tuple


def _validate_shape(cube, n_components):
    """Check the cube dimensions and requested component count."""
    if cube.ndim != 3:
        raise ValueError('PCA cube must have shape (nints, dimy, dimx)')
    nints, dimy, dimx = cube.shape
    max_components = min(nints, dimy * dimx)
    if not 1 <= n_components <= max_components:
        raise ValueError(f'n_components={n_components} must be between 1 and '
            f'min(nints, npix)={max_components}')
    if dimy * dimx < 2:
        raise ValueError('PCA requires at least two spatial pixels')


def _host_cube(cube):
    """Normalize the cube dtype while retaining its host or device location."""
    if not core.is_device_array(cube):
        cube = np.asarray(jax.device_get(cube))
    if cube.ndim != 3:
        raise ValueError('PCA cube must have shape (nints, dimy, dimx)')
    use_float64 = cube.dtype == np.float64 and jax.config.x64_enabled
    dtype = np.dtype(np.float64 if use_float64 else np.float32)
    return cube if cube.dtype == dtype else cube.astype(dtype)


def _chunk_width(dimx, bytes_per_column):
    """Choose a detector column width within host and device memory limits."""
    width = int(_COLUMN_CHUNK_SIZE)
    if width <= 0:
        raise ValueError('PCA detector-column chunk size must be positive')
    host_width = max(1, core.host_allocation_budget_bytes() // max(8 * bytes_per_column, 1))
    device_width = core.auto_chunk(dimx, bytes_per_column, n_buffers=8, headroom=.5)
    width = min(width, host_width, device_width)
    return min(width, dimx)


def _host_chunks(cube):
    """Yield contiguous detector column chunks with all integrations."""
    per_column = cube.shape[0] * cube.shape[1] * cube.dtype.itemsize
    width = _chunk_width(cube.shape[2], per_column)
    resident = core.is_device_array(cube)
    for start in range(0, cube.shape[2], width):
        stop = min(start + width, cube.shape[2])
        block = cube[:, :, start:stop]
        yield start, stop, (block if resident else np.ascontiguousarray(block))


def _device_get(value):
    """Transfer a device result to a NumPy array."""
    return np.asarray(jax.device_get(value))


def _filled_centered(block, mean, fill_value):
    """Fill NaNs and subtract per-integration feature means."""
    nints = block.shape[0]
    flat = block.reshape(nints, -1)
    filled = jnp.where(jnp.isnan(flat), fill_value, flat)
    return filled - mean[:, None]


@jax.jit
def _chunk_scores(block, mean, fill_value, vectors):
    """Calculate spatial scores for a detector column chunk."""
    centered = _filled_centered(block, mean, fill_value)
    scores = jnp.matmul(vectors.T, centered, precision=_MATMUL_PRECISION)
    return scores


@functools.partial(jax.jit, static_argnames=('return_scores',))
def _chunk_projection(block, mean, fill_value, vectors, coefficients,
                      mean_multiplier, return_scores=False):
    """Reconstruct a chunk from weighted scores; return_scores is static."""
    scores = _chunk_scores(block, mean, fill_value, vectors)
    projection = jnp.matmul(vectors, scores * coefficients[:, None], precision=_MATMUL_PRECISION)
    projection = projection + mean_multiplier * mean[:, None]
    return (projection, scores) if return_scores else projection


def _fill_value(cube):
    """Get the global median for NaN replacement, or zero for finite data."""
    isnan = jnp.isnan if core.is_device_array(cube) else np.isnan
    has_nan = any(bool(isnan(chunk).any()) for _, _, chunk in _host_chunks(cube))
    if not has_nan:
        return np.dtype(cube.dtype).type(0)
    return np.dtype(cube.dtype).type(hoststats.nanmedian(cube))


def _feature_mean(cube, fill_value):
    """Calculate per-integration feature means using host float64 sums."""
    totals = np.zeros(cube.shape[0], dtype=np.float64)
    for _, _, chunk in _host_chunks(cube):
        block = np.asarray(core.to_host(chunk))
        filled = np.where(np.isnan(block), fill_value, block)
        # Keep the feature means in host float64 accumulation order.
        totals += filled.sum(axis=(1, 2), dtype=np.float64)
    return (totals / (cube.shape[1] * cube.shape[2])).astype(cube.dtype)


def _put_fit_inputs(mean, fill_value, dtype):
    """Transfer feature means and the fill value to the device."""
    return (jax.device_put(np.asarray(mean, dtype=dtype)),
            jax.device_put(np.asarray(fill_value, dtype=dtype)))


def _canonical_qr(matrix, width):
    """Calculate a reduced QR basis with positive diagonal signs."""
    q, r = np.linalg.qr(np.asarray(matrix, dtype=np.float64), mode='reduced')
    q, r = q[:, :width], r[:width, :]
    diagonal = np.diag(r)
    signs = np.where(diagonal < 0, -1., 1.)
    q *= signs[None, :q.shape[1]]
    return q


def _split_chunk(block, mean, fill_value, g):
    """Separate the static image from centered temporal residuals."""
    nints = block.shape[0]
    flat = block.reshape(nints, -1)
    filled = jnp.where(jnp.isnan(flat), fill_value, flat).astype(jnp.float64)
    mean, g = mean.astype(jnp.float64), g.astype(jnp.float64)
    reference = jnp.mean(filled, axis=0)
    residual = (filled - reference[None, :]) - (mean - g)[:, None]
    return reference - g, residual


@jax.jit
def _chunk_gram(block, mean, fill_value, g):
    """Calculate the static, cross and residual covariance terms of a chunk."""
    static, residual = _split_chunk(block, mean, fill_value, g)
    residual_work = residual.astype(block.dtype)
    return (jnp.sum(static * static), jnp.matmul(residual, static, precision=_MATMUL_PRECISION),
            jnp.matmul(residual_work, residual_work.T, precision=_MATMUL_PRECISION))


@jax.jit
def _chunk_stats(block, mean, fill_value, g):
    """Calculate chunk energies and cross terms without forming a covariance matrix."""
    static, residual = _split_chunk(block, mean, fill_value, g)
    return (jnp.sum(static * static), jnp.matmul(residual, static, precision=_MATMUL_PRECISION),
            jnp.sum(residual * residual))


@jax.jit
def _chunk_residual_power(block, mean, fill_value, g, basis):
    """Apply the residual temporal covariance to a narrow basis."""
    _, residual = _split_chunk(block, mean, fill_value, g)
    residual = residual.astype(block.dtype)
    spatial = jnp.matmul(residual.T, basis, precision=_MATMUL_PRECISION)
    return jnp.matmul(residual, spatial, precision=_MATMUL_PRECISION)


def _split_inputs(cube, mean, fill_value):
    """Transfer feature means and centering scalars to the device."""
    g = np.dtype(cube.dtype).type(np.mean(mean, dtype=np.float64))
    mean_d, fill_d = _put_fit_inputs(mean, fill_value, cube.dtype)
    return mean_d, fill_d, jax.device_put(g)


def _covariance_terms(cube, mean, fill_value, exact):
    """Accumulate static energy, cross terms and residual covariance or energy."""
    nints = cube.shape[0]
    mean_d, fill_d, g_d = _split_inputs(cube, mean, fill_value)
    static_energy = 0.
    cross = np.zeros(nints, dtype=np.float64)
    residual = np.zeros((nints, nints), dtype=np.float64) if exact else 0.
    kernel = _chunk_gram if exact else _chunk_stats
    for _, _, chunk in _host_chunks(cube):
        block_d = jax.device_put(chunk)
        # Retain weak temporal signals beside the static image.
        with _enable_x64():
            a_c, u_c, d_c = kernel(block_d, mean_d, fill_d, g_d)
        static_energy += float(_device_get(a_c))
        cross += _device_get(u_c)
        residual += _device_get(d_c) if exact else float(_device_get(d_c))
        del block_d, a_c, u_c, d_c
    return static_energy, cross, residual


def _fit_exact(cube, mean, fill_value, n_components):
    """Fit the leading eigenvectors of the accumulated temporal covariance."""
    nints = cube.shape[0]
    static_energy, cross, covariance = _covariance_terms(cube, mean, fill_value, True)
    # Restore the static image and cross terms in double precision.
    covariance += static_energy
    covariance += cross[:, None]
    covariance += cross[None, :]
    covariance = (covariance + covariance.T) * 0.5
    total_energy = float(np.trace(covariance))
    # Solve only for the requested components.
    values, vectors = eigh(
        covariance, subset_by_index=(nints - n_components, nints - 1),
        overwrite_a=True, check_finite=False)
    return (vectors[:, ::-1], np.maximum(values[::-1], 0.),
            total_energy, n_components, (nints, nints))


def _fit_randomized(cube, mean, fill_value, n_components):
    """Fit temporal components using streamed subspace iteration."""
    nints = cube.shape[0]
    rank = min(nints, n_components + _RANDOM_OVERSAMPLE)
    mean_d, fill_d, g_d = _split_inputs(cube, mean, fill_value)
    static_energy, cross, residual_energy = _covariance_terms(cube, mean, fill_value, False)
    total_energy = (nints * static_energy + 2. * float(cross.sum()) + residual_energy)

    def apply(basis):
        """Apply the accumulated temporal covariance to a basis."""
        out = np.zeros_like(basis)
        basis_d = jax.device_put(np.asarray(basis, dtype=cube.dtype))
        for _, _, chunk in _host_chunks(cube):
            block_d = jax.device_put(chunk)
            # Retain weak temporal signals beside the static image.
            with _enable_x64():
                contribution = _chunk_residual_power(block_d, mean_d, fill_d, g_d, basis_d)
            out += _device_get(contribution)
            del block_d, contribution
        col = basis.sum(axis=0)
        return (out + static_energy * col[None, :] +
                cross[:, None] * col[None, :] + (cross @ basis)[None, :])

    rng = np.random.default_rng(_RANDOM_SEED)
    basis = _canonical_qr(rng.standard_normal((nints, rank)), rank)
    for _ in range(_RANDOM_POWER_ITERATIONS):
        basis = _canonical_qr(apply(basis), rank)
    ritz = basis.T @ apply(basis)
    values, rotation = np.linalg.eigh((ritz + ritz.T) * 0.5)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.)
    basis = basis @ rotation[:, order]
    return (basis[:, :n_components], values[:n_components], total_energy, rank, (nints, rank))


def _fit_streamed(cube, n_components, solver=None):
    """Fit temporal components using exact or randomized decomposition."""
    host = _host_cube(cube)
    _validate_shape(host, n_components)
    fill_value = _fill_value(host)
    mean = _feature_mean(host, fill_value)
    nints = host.shape[0]
    randomized_rank = min(nints, n_components + _RANDOM_OVERSAMPLE)
    if solver is None:
        pixels = host.shape[1] * host.shape[2]
        solver = ('exact' if (nints <= _EXACT_MAX_NINTS and (
            float(nints) ** 2 * pixels <= _EXACT_FLOP_BUDGET or
            randomized_rank * 2 >= nints)) else 'randomized')
    if solver == 'exact':
        vectors, values, total, rank, largest = _fit_exact(host, mean, fill_value, n_components)
    elif solver == 'randomized':
        vectors, values, total, rank, largest = _fit_randomized(
            host, mean, fill_value, n_components)
    else:
        raise ValueError("PCA solver must be 'exact' or 'randomized'")

    variance = values / total
    return host, _PCAFit(vectors=np.asarray(vectors, dtype=host.dtype),
        variance_ratio=np.asarray(variance, dtype=host.dtype),
        mean=np.asarray(mean, dtype=host.dtype),
        fill_value=host.dtype.type(fill_value), dtype=host.dtype,
        solver=solver, work_rank=rank, largest_solver_shape=largest)


def _scan_signs(fit):
    """Orient components by their largest temporal entry, as in sklearn."""
    vectors = np.asarray(fit.vectors, dtype=np.float64)
    pick = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[pick, np.arange(vectors.shape[1])])
    signs[signs == 0] = 1.
    return signs


def _scan_signs_and_projections(cube, fit):
    """Return component signs and signed spatial projections."""
    n_components = fit.vectors.shape[1]
    mean_d, fill_d = _put_fit_inputs(fit.mean, fit.fill_value, fit.dtype)
    vectors_d = jax.device_put(fit.vectors)
    dimy, dimx = cube.shape[1:]
    projections = np.empty((n_components, dimy, dimx), dtype=fit.dtype)
    for start, stop, chunk in _host_chunks(cube):
        block_d = jax.device_put(chunk)
        scores = _chunk_scores(block_d, mean_d, fill_d, vectors_d)
        projections[:, :, start:stop] = _device_get(scores).reshape(
            n_components, dimy, stop - start)
        del block_d, scores
    signs = _scan_signs(fit)
    projections *= np.asarray(signs, dtype=fit.dtype)[:, None, None]
    return signs, projections


def _project_to_host(cube, fit, coefficients, mean_multiplier,
                     subtract_from_input, return_projections=False):
    """Reconstruct or remove components while retaining the input location."""
    coefficients = np.asarray(coefficients, dtype=fit.dtype)
    resident = core.is_device_array(cube)
    blocks = [] if resident else None
    output = (None if resident else core.empty_host_array(cube.shape, cube.dtype,
                                    name='exotedrf-pca-reconstruction'))
    mean_d, fill_d = _put_fit_inputs(fit.mean, fit.fill_value, fit.dtype)
    vectors_d = jax.device_put(fit.vectors)
    coefficients_d = jax.device_put(coefficients)
    multiplier_d = jax.device_put(np.asarray(mean_multiplier, dtype=fit.dtype))
    dimy, dimx = cube.shape[1:]
    projections = (np.empty((len(coefficients), dimy, dimx), dtype=fit.dtype)
                   if return_projections else None)
    for start, stop, chunk in _host_chunks(cube):
        block_d = jax.device_put(chunk)
        projected = _chunk_projection(
            block_d, mean_d, fill_d, vectors_d, coefficients_d, multiplier_d,
            return_scores=return_projections)
        if return_projections:
            projection, scores = projected
            if resident:
                scores_h = jax.device_get(scores)
            else:
                projection, scores_h = jax.device_get((projection, scores))
            projections[:, :, start:stop] = np.asarray(scores_h).reshape(
                len(coefficients), dimy, stop - start)
        else:
            projection = projected if resident else _device_get(projected)
        local = projection.reshape(chunk.shape)
        local = chunk - local if subtract_from_input else local
        if resident:
            blocks.append(local)
        else:
            output[:, :, start:stop] = local
        del block_d, projection
        del projected
        if return_projections:
            del scores
    if resident:
        output = (blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=2))
        blocks.clear()
    signs = _scan_signs(fit)
    if return_projections:
        projections *= np.asarray(signs, dtype=fit.dtype)[:, None, None]
        return output, signs, projections
    return output, signs


def _signed_components(fit, signs):
    """Apply component signs and return temporal components."""
    return np.asarray(fit.vectors.T * signs[:, None], dtype=fit.dtype)


def soss_stability_pca(cube, n_components=10):
    """Calculate temporal principal components and reconstruct the data.

    Parameters
    ----------
    cube : array-like(float)
        Data cube with shape (nints, dimy, dimx).
    n_components : int
        Number of temporal principal components to retain.

    Returns
    -------
    components : np.ndarray(float)
        Signed components with shape (n_components, nints).
    variance_ratio : np.ndarray(float)
        Explained variance ratios in descending order.
    reconstruction : array-like(float)
        Reconstructed cube, retaining the input host or device location.
    """
    host, fit = _fit_streamed(cube, n_components)
    reconstruction, signs = _project_to_host(host, fit, np.ones(n_components, dtype=fit.dtype),
        mean_multiplier=1., subtract_from_input=False)
    return (_signed_components(fit, signs), fit.variance_ratio, reconstruction)


def pca_reconstruction(cube, remove_mask, baseline_mask, n_components=10, return_plot_data=False):
    """Remove selected temporal principal components from the data.

    Removing the first component also removes the per-integration feature mean.

    Parameters
    ----------
    cube : array-like(float)
        Data cube with shape (nints, dimy, dimx).
    remove_mask : array-like(bool)
        Components to remove; a longer mask fits additional components for removal.
    baseline_mask : array-like(bool)
        True for baseline integrations.
    n_components : int
        Number of temporal principal components to retain.
    return_plot_data : bool
        Append original and reconstructed spatial component maps.

    Returns
    -------
    newcube : array-like(float)
        Modified cube, retaining the input host or device location.
    components, components_reconstructed : np.ndarray(float)
        Original and reconstructed temporal components, respectively.
    variance_ratio, variance_reconstructed : np.ndarray(float)
        Original and reconstructed explained variance ratios, respectively.
    wlc : np.ndarray(float)
        First component normalized by its baseline median.
    projections, projections_reconstructed : np.ndarray(float)
        Spatial maps, appended only when return_plot_data is True.
    """
    host, fit = _fit_streamed(cube, n_components)
    remove = np.asarray(jax.device_get(remove_mask), dtype=bool)
    baseline = np.asarray(jax.device_get(baseline_mask), dtype=bool)
    nints = host.shape[0]
    if remove.ndim != 1 or remove.shape[0] < n_components:
        raise ValueError(f'remove_mask must have shape ({n_components},), got ' f'{remove.shape}')
    if baseline.shape != (nints,):
        raise ValueError(f'baseline_mask must have shape ({nints},), got ' f'{baseline.shape}')

    removing = np.any(remove)
    extra = removing and remove.shape[0] > n_components
    newcube = host
    if removing:
        removal_fit = fit
        # Fit extra components for removal while retaining the reported component count.
        if extra:
            _, removal_fit = _fit_streamed(host, remove.shape[0])
        projected = _project_to_host(host, removal_fit, remove.astype(removal_fit.dtype),
            mean_multiplier=float(remove[0]), subtract_from_input=True,
            return_projections=return_plot_data and not extra)
        if return_plot_data and not extra:
            newcube, signs, projections = projected
        else:
            newcube, signs = projected
    if not removing or extra:
        if return_plot_data:
            signs, projections = _scan_signs_and_projections(host, fit)
        else:
            signs = _scan_signs(fit)
    components = _signed_components(fit, signs)
    wlc_norm = np.nanmedian(np.where(baseline, components[0], np.nan))
    wlc = np.asarray(components[0] / wlc_norm, dtype=fit.dtype)

    if removing:
        reconstructed_host, reconstructed_fit = _fit_streamed(newcube, n_components)
        if return_plot_data:
            reconstructed_signs, projections_reconstructed = _scan_signs_and_projections(
                    reconstructed_host, reconstructed_fit)
        else:
            reconstructed_signs = _scan_signs(reconstructed_fit)
        components_reconstructed = _signed_components(reconstructed_fit, reconstructed_signs)
        variance_reconstructed = reconstructed_fit.variance_ratio
    else:
        components_reconstructed = components
        variance_reconstructed = fit.variance_ratio
        if return_plot_data:
            projections_reconstructed = projections

    result = (newcube, components, fit.variance_ratio, wlc,
              components_reconstructed, variance_reconstructed)
    if return_plot_data:
        return result + (projections, projections_reconstructed)
    return result
