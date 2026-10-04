"""White-light and spectral point-to-point scatter for optimization."""

import functools

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ['stitch_soss_orders', 'resolve_wave_slice', 'cost_function']


def stitch_soss_orders(flux_o1, wave_o1, flux_o2, wave_o2, cutoff=0.85):
    """Merge order 2 below the cutoff with order 1 above it.

    Parameters
    ----------
    flux_o1, flux_o2 : array-like(float)
        Order 1 and 2 flux arrays with shape (nints, nwave), respectively.
    wave_o1, wave_o2 : array-like(float)
        Order 1 and 2 wavelength grids in microns, respectively.
    cutoff : float
        Wavelength dividing the two orders in microns.

    Returns
    -------
    flux : array-like(float)
        Sorted merged flux with shape (nints, nwave).
    wave : np.ndarray(float)
        Sorted merged wavelength grid.
    """
    wave_o1 = np.asarray(wave_o1, float)
    wave_o2 = np.asarray(wave_o2, float)
    i2 = np.where(wave_o2 <= cutoff)[0]
    i1 = np.where(wave_o1 > cutoff)[0]
    if i2.size == 0 or i1.size == 0:
        raise ValueError('Cutoff produces empty segment: '
                         f'O2<= {cutoff}: {i2.size}, O1> {cutoff}: {i1.size}')
    idx2 = int(i2[-1])
    idx1 = int(i1[0])
    wave = np.concatenate([wave_o2[:idx2 + 1], wave_o1[idx1:]])
    flux = jnp.concatenate([jnp.asarray(flux_o2)[:, :idx2 + 1],
                            jnp.asarray(flux_o1)[:, idx1:]], axis=1)
    s = np.argsort(wave)
    return flux[:, s], wave[s]


def resolve_wave_slice(wave, wave_range, tol=0.05):
    """Resolve wavelength limits to inclusive nearest-index bounds.

    Parameters
    ----------
    wave : None, array-like(float)
        Wavelength grid; integration-dependent grids are reduced to their median.
    wave_range : None, list[float]
        Lower and upper wavelength limits; None entries use the available extrema.
    tol : float
        Compatibility tolerance; closest indices are used.

    Returns
    -------
    bounds : None, tuple[int]
        Lower and upper indices, or None when no range was supplied.
    """
    if wave_range is None:
        return None
    if not (isinstance(wave_range, (list, tuple)) and len(wave_range) == 2):
        raise ValueError('wave_range must be None or a length-2 list/tuple')

    wave = np.asarray(wave, float)
    if wave.ndim == 2:
        wave = np.nanmedian(wave, axis=0) if wave.shape[0] > 1 else np.ravel(wave)
    elif wave.ndim != 1:
        raise ValueError(f'Expected 1D wavelength axis, got ' f'wave.ndim={wave.ndim}')

    lo, hi = wave_range
    finite = np.isfinite(wave)
    if not finite.any():
        raise ValueError('All entries in wave are NaN!')

    if lo is None:
        lo = np.nanmin(wave[finite])
    if hi is None:
        hi = np.nanmax(wave[finite])

    dist_lo = np.abs(wave - lo)
    dist_lo[~finite] = np.inf
    dist_hi = np.abs(wave - hi)
    dist_hi[~finite] = np.inf
    idx_lo = int(np.argmin(dist_lo))
    idx_hi = int(np.argmin(dist_hi))
    return tuple(sorted((idx_lo, idx_hi)))


@functools.partial(jax.jit, static_argnames=('baseline_ints', 'spec_slice', 'w1', 'w2'))
def _cost_core(flux, baseline_ints, spec_slice, w1, w2):
    """Combine white-light and spectral point-to-point scatter."""
    # Calculate the white-light point-to-point scatter.
    white = jnp.nansum(flux, axis=1)
    norm_white = white / jnp.median(white)
    d2_white = 0.5 * (norm_white[:-2] + norm_white[2:]) - norm_white[1:-1]
    ptp2_white = jnp.nanmedian(jnp.abs(d2_white))
    # Calculate point-to-point scatter per wavelength.
    wave_meds = jnp.nanmedian(flux, axis=0, keepdims=True)
    norm_spec = flux / wave_meds
    d2_spec = 0.5 * (norm_spec[:-2] + norm_spec[2:]) - norm_spec[1:-1]

    if baseline_ints is None:
        ptp2_spec_wave = jnp.nanmedian(jnp.abs(d2_spec), axis=0)
    elif len(baseline_ints) == 1:
        n = int(baseline_ints[0])
        ptp2_spec_wave = jnp.nanmedian(jnp.abs(d2_spec[:n]), axis=0)
    else:
        nlow, nhigh = (int(b) for b in baseline_ints)
        low_term = jnp.nanmedian(jnp.abs(d2_spec[:nlow]), axis=0)
        high_term = jnp.nanmedian(jnp.abs(d2_spec[nhigh:]), axis=0)
        ptp2_spec_wave = 0.5 * (low_term + high_term)

    if spec_slice is None:
        ptp2_spec = jnp.nanmedian(ptp2_spec_wave)
    else:
        i0, i1 = spec_slice
        ptp2_spec = jnp.nanmedian(ptp2_spec_wave[i0:i1 + 1])

    # Exclude zero-weighted terms so their NaNs cannot affect the cost.
    cost = jnp.zeros((), dtype=flux.dtype)
    if w1 != 0:
        cost = cost + w1 * ptp2_white
    if w2 != 0:
        cost = cost + w2 * ptp2_spec
    return cost, ptp2_spec_wave


def cost_function(flux, wave=None, baseline_ints=None, wave_range=None, w1=0.0, w2=1.0, tol=0.05):
    """Combine white-light and spectral point-to-point scatter.

    Parameters
    ----------
    flux : array-like(float)
        Extracted flux with shape (nints, nwave); merge SOSS orders before calling.
    wave : None, array-like(float)
        Wavelength grid; integration-dependent grids are reduced to their median.
    baseline_ints : None, list[int]
        One or two slice bounds on second differences; negative bounds count from the end.
    wave_range : None, list[float]
        Lower and upper wavelength limits; None entries use the available extrema.
    w1, w2 : float
        White-light and spectral scatter weights. Static: changes trigger a recompile.
    tol : float
        Compatibility tolerance; closest indices are used.

    Returns
    -------
    cost : float
        Weighted scatter; zero-weighted NaN terms are excluded.
    ptp2_spec_wave : array-like(float)
        Point-to-point scatter per wavelength.
    """
    if baseline_ints is not None:
        baseline_ints = tuple(int(b) for b in np.atleast_1d(baseline_ints))
        if len(baseline_ints) not in (1, 2):
            raise ValueError('baseline_ints must be length 1 or 2, got ' f'{len(baseline_ints)}')
    spec_slice = resolve_wave_slice(wave, wave_range, tol=tol) if wave_range is not None else None
    return _cost_core(jnp.asarray(flux), baseline_ints, spec_slice, float(w1), float(w2))
