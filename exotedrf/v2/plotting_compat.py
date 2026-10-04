"""Capture and render v1 diagnostic plots from in-memory calibration products."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping
import warnings

import bottleneck as bn
import numpy as np


RandInt = Callable[..., Any]


def _lombscargle_psd_kernel(timestamps, values, valid, frequencies):
    """Calculate floating-mean Lomb--Scargle power for each detector sample."""
    import jax
    import jax.numpy as jnp
    weights = valid.astype(values.dtype)
    counts = jnp.sum(weights, axis=1)
    means = jnp.sum(jnp.where(valid, values, 0.), axis=1) / counts
    centered = jnp.where(valid, values - means[:, None], 0.)
    phase = (2. * jnp.pi * frequencies[:, None] * timestamps[None, :])
    sine = jnp.sin(phase)
    cosine = jnp.cos(phase)

    def weighted_design(design):
        """Sum weighted design terms with full multiplication precision."""
        return jnp.matmul(weights, design.T, precision=jax.lax.Precision.HIGHEST)
    ss = weighted_design(sine * sine)
    cc = weighted_design(cosine * cosine)
    sc = weighted_design(sine * cosine)
    one_s = weighted_design(sine)
    one_c = weighted_design(cosine)
    ones = jnp.broadcast_to(counts[:, None], one_s.shape)
    matrix = jnp.stack((jnp.stack((ones, one_s, one_c), axis=-1),
        jnp.stack((one_s, ss, sc), axis=-1), jnp.stack((one_c, sc, cc), axis=-1),), axis=-2)
    rhs = jnp.stack((jnp.broadcast_to(jnp.sum(centered, axis=1)[:, None], one_s.shape),
        jnp.matmul(centered, sine.T, precision=jax.lax.Precision.HIGHEST),
        jnp.matmul(centered, cosine.T, precision=jax.lax.Precision.HIGHEST),), axis=-1)
    coefficients = jnp.linalg.solve(matrix, rhs[..., None])[..., 0]
    return .5 * jnp.sum(rhs * coefficients, axis=-1)


def batched_lombscargle_psd(timestamps, values, valid, frequencies):
    """Measure detector noise power with v1's float64 Astropy PSD normalization.

    Parameters
    ----------
    timestamps : array-like(float)
        Time of each detector sample.
    values : array-like(float)
        Sample values with shape (pixel, sample).
    valid : array-like(bool)
        Mask of samples to include, with the same shape as values.
    frequencies : array-like(float)
        Frequencies at which to evaluate the periodogram.

    Returns
    -------
    power : np.ndarray(float)
        Lomb--Scargle power with shape (pixel, frequency).
    """
    import jax
    import jax.numpy as jnp
    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError('values and valid must have matching 2-D shapes')
    if timestamps.shape != (values.shape[1],):
        raise ValueError('timestamps length must match the sample axis')
    if frequencies.ndim != 1:
        raise ValueError('frequencies must be one-dimensional')
    usable = np.sum(valid, axis=1) >= 3
    safe_valid = valid.copy()
    safe_values = np.where(np.isfinite(values), values, 0.)
    if not np.all(usable):
        safe_valid[~usable, :min(3, values.shape[1])] = True

    with jax.enable_x64():
        compiled = jax.jit(_lombscargle_psd_kernel)
        result = jax.device_get(compiled(jnp.asarray(timestamps), jnp.asarray(safe_values),
            jnp.asarray(safe_valid), jnp.asarray(frequencies)))
    result = np.array(result, copy=True)
    result[~usable] = np.nan
    return result


def _first_segment_shape(data, first_segment_stop, *, name):
    """Validate the segment boundary without transferring any detector data."""
    stop = int(first_segment_stop)
    if stop <= 0:
        raise ValueError('first_segment_stop must be positive')
    if data.ndim != 4:
        raise ValueError(f'{name} must have shape (integration, group, y, x), got ' f'{data.shape}')
    if data.shape[0] < stop:
        raise ValueError(f'first_segment_stop={stop} exceeds {name} integration count '
            f'{data.shape[0]}')
    return (stop, *data.shape[1:])


def _paired_segment_shape(before, after, stop, label):
    """Validate matching first-segment ramp shapes for a correction diagnostic."""
    before_shape = _first_segment_shape(before, stop, name='before_data')
    after_shape = _first_segment_shape(after, stop, name='after_data')
    if before_shape != after_shape:
        raise ValueError(f'{label} before/after shapes differ: {before_shape} != {after_shape}')
    return after_shape


def last_group_median(data, stop, *, rows=slice(None), max_bytes=32 << 20):
    """Calculate the first-segment image median with bounded column transfers.

    Parameters
    ----------
    data : array-like(float)
        Ramp or rate cube.
    stop : int
        Exclusive integration stop for the first segment.
    rows : slice
        Detector rows to include.
    max_bytes : int
        Maximum transfer size, with a minimum of one detector column.

    Returns
    -------
    image : np.ndarray(float)
        First-segment median image for the selected detector rows.
    """
    shape = data.shape
    nrows = len(range(*rows.indices(shape[-2])))
    width = shape[-1]
    cols = max(1, min(width, int(max_bytes) // max(
        1, int(stop) * nrows * np.dtype(data.dtype).itemsize)))
    result = np.empty((nrows, width), dtype=data.dtype)
    for start in range(0, width, cols):
        end = min(width, start + cols)
        block = (data[:stop, -1, rows, start:end] if data.ndim == 4
            else data[:stop, rows, start:end])
        result[:, start:end] = bn.nanmedian(np.asarray(block), axis=0)
    return result


def _bin_n_pix(data, npix):
    """Reproduce v1 ``plot_inl_correction`` chunk-median binning."""
    data_flat = np.asarray(data).flatten()
    remainder = data_flat.size % npix
    nbin = int((data_flat.size - remainder) / npix)
    if nbin == 0:
        return np.empty(0, dtype=data_flat.dtype)
    data_reshape = np.reshape(data_flat[remainder:], (nbin, npix))
    return np.nanmedian(data_reshape, axis=1)


def _ramp_line_parameters(data):
    """Calculate centered least-squares ramp intercepts, slopes, and group positions."""
    ngroup = data.shape[1]
    groups = np.arange(ngroup, dtype=np.float32)
    centered_groups = groups - np.mean(groups)
    denom = np.sum(centered_groups * centered_groups)
    if denom == 0:
        raise ValueError('INL diagnostic requires at least two groups')
    slope = np.tensordot(data, centered_groups, axes=([1], [0])) / denom
    intercept = np.mean(data, axis=1) - slope * np.mean(groups)
    return intercept, slope, groups


def capture_inl(before_data, after_data, first_segment_stop, npix_to_bin=1000):
    """Capture the exact arrays drawn by v1's two INL plots.

    Parameters
    ----------
    before_data, after_data : array-like(float)
        Ramp cubes before and after correction.
    first_segment_stop : int
        Exclusive integration stop for the first segment.
    npix_to_bin : int
        Initial number of samples per median bin.

    Returns
    -------
    record : dict
        Binned signal and residual arrays plus the effective bin size.
    """
    _paired_segment_shape(before_data, after_data, first_segment_stop, 'INL')
    npix_to_bin = int(npix_to_bin)
    if npix_to_bin <= 0:
        raise ValueError('npix_to_bin must be positive')

    # Select integrations 10:20 as in v1.
    int_start, int_end = 10, 20
    # Transfer only the selected integrations to the host.
    int_end = min(int_end, int(first_segment_stop))
    pre_window = np.asarray(before_data[int_start:int_end])
    post_window = np.asarray(after_data[int_start:int_end])
    if pre_window.shape[0] == 0:
        raise ValueError('v1 INL diagnostic needs at least 11 integrations in the first '
            'segment (it plots integrations 10:20)')

    # Select and sort samples within the 10--20 kADU plotting window.
    pre_flat = pre_window.reshape(-1)
    retained = np.flatnonzero((pre_flat >= 10000) & (pre_flat <= 20000))
    retained = retained[np.argsort(pre_flat[retained])]
    data_pre = np.asarray(pre_flat[retained])
    data_post = np.asarray(post_window.reshape(-1)[retained])
    n_group, dimy, dimx = pre_window.shape[1:]
    xpos = retained % dimx
    quotient = retained // dimx
    ypos = quotient % dimy
    quotient //= dimy
    group_index = quotient % n_group
    integration = quotient // n_group
    pre_intercept, pre_slope, groups = _ramp_line_parameters(pre_window)
    post_intercept, post_slope, _ = _ramp_line_parameters(post_window)
    res_pre = data_pre - (pre_intercept[integration, ypos, xpos] +
        pre_slope[integration, ypos, xpos] * groups[group_index])
    res_post = data_post - (post_intercept[integration, ypos, xpos] +
        post_slope[integration, ypos, xpos] * groups[group_index])
    data_pre_bin = _bin_n_pix(data_pre, npix_to_bin)
    while data_pre_bin.size >= 1_000_000:
        warnings.warn('Too many INL plot samples; increasing npix_to_bin.',
            RuntimeWarning, stacklevel=2)
        npix_to_bin *= 2
        data_pre_bin = _bin_n_pix(data_pre, npix_to_bin)
    res_pre_bin = _bin_n_pix(res_pre, npix_to_bin)
    data_post_bin = _bin_n_pix(data_post, npix_to_bin)
    res_post_bin = _bin_n_pix(res_post, npix_to_bin)

    if res_pre_bin.size and (np.max(res_pre_bin) - np.min(res_pre_bin) >= 6):
        warnings.warn('INL residual min-max spread is >6; correction visualization '
            f'may not be optimal (npix_to_bin={npix_to_bin}).', RuntimeWarning, stacklevel=2)
    return {'data_pre': data_pre_bin, 'res_pre': res_pre_bin, 'data_post': data_post_bin,
        'res_post': res_post_bin, 'npix_to_bin': npix_to_bin}


def render_inl(record: Mapping[str, Any], path1, path2):
    """Render v1's INL figures from a compact :func:`capture_inl` record.

    Parameters
    ----------
    record : dict
        Captured diagnostic arrays.
    path1, path2 : str, Path
        Paths to which to save the first and second diagnostic plots.

    Returns
    -------
    path1, path2 : Path
        Paths to the saved diagnostic figures.
    """
    from exotedrf import plotting as v1_plotting
    path1 = Path(path1)
    path2 = Path(path2)
    path1.parent.mkdir(parents=True, exist_ok=True)
    path2.parent.mkdir(parents=True, exist_ok=True)
    v1_plotting.make_inl_plot(np.asarray(record['data_pre']), np.asarray(record['res_pre']),
        np.asarray(record['data_post']), outfile=str(path1), show_plot=False)
    data_pre = np.asarray(record['data_pre'])
    data_post = np.asarray(record['data_post'])
    if (data_pre.size >= 3 and data_post.size >= 3 and np.nanmax(data_pre) > np.nanmin(data_pre) and
            np.nanmax(data_post) > np.nanmin(data_post)):
        v1_plotting.make_inl_plot2(data_pre, np.asarray(record['res_pre']), data_post,
            np.asarray(record['res_post']), outfile=str(path2), show_plot=False)
    else:
        # Draw the v1 axes when there are too few samples for a periodogram.
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        ax.set_xlim(100, 2100)
        for period in (1024, 1024 / 2, 1024 / 3):
            ax.axvline(period, ls='--', c='grey')
        ax.set_xlabel('Periodicity [ADU]', fontsize=12)
        ax.set_ylabel('Power', fontsize=12)
        plt.savefig(path2, bbox_inches='tight')
        plt.close()
    return path1, path2


def _randint_callable(randint):
    """Select a random integer callable from a generator or use np.random.randint."""
    if randint is None:
        return np.random.randint
    if callable(randint):
        return randint
    if hasattr(randint, 'integers'):
        return randint.integers
    if hasattr(randint, 'randint'):
        return randint.randint
    raise TypeError('randint must be callable or a NumPy random generator')


def capture_linearity(before_data, after_data, first_segment_stop,
        instrument='NIRISS', randint=None, groupdq=None):
    """Capture v1's two linearity diagnostic summaries.

    Random draws follow v1 make_linearity_plot and make_linearity_plot2 call order.

    Parameters
    ----------
    before_data, after_data : array-like(float)
        Ramp cubes before and after correction.
    first_segment_stop : int
        Exclusive integration stop for the first segment.
    instrument : str
        Instrument name; NIRISS, NIRSPEC, or MIRI.
    randint : None, callable, np.random.Generator
        Random integer generator. If None, use np.random.randint.
    groupdq : None, array-like(int)
        Post-linearity group flags. Required for dropped-group selection with MIRI.

    Returns
    -------
    record : dict
        Group differences, ramp residuals, and tick positions for both plots.
    """
    after_shape = _paired_segment_shape(before_data, after_data, first_segment_stop, 'linearity')
    old_cube, cube = before_data, after_data
    instrument = str(instrument).upper()
    miri = instrument == 'MIRI'
    if miri and groupdq is None:
        raise ValueError('MIRI linearity plotting needs the post-linearity groupdq so the '
            'dropped RSCD groups can be excluded (plotting.py:620-624)')
    draw = _randint_callable(randint)
    nint, ngroup, _dimy, _dimx = after_shape
    if ngroup < 2:
        raise ValueError('linearity diagnostic requires at least two groups')
    good_groups = None
    if miri:
        from exotedrf.v2.kernels import jump as k_jump
        _first_segment_shape(groupdq, first_segment_stop, name='groupdq')
        # Inspect the first integration for dropped MIRI groups.
        dropped = np.flatnonzero(np.asarray(k_jump.miri_dropped_groups(groupdq[:1])))
        good_groups = np.delete(np.arange(ngroup), dropped)
        if good_groups.size < 1:
            raise ValueError('every MIRI group is dropped; nothing to plot')

    # Capture group differences for the first linearity plot.
    stack_ints = draw(0, nint, 25)
    stack = bn.nanmedian(np.asarray(cube[stack_ints, -1]), axis=0)
    if miri:
        # Restrict the bright-pixel search to the MIRI trace columns.
        stack = stack[:, 20:60]
    bright = np.where((stack >= np.nanpercentile(stack, 80)) &
        (stack < np.nanpercentile(stack, 99)))
    nbright = len(bright[0])
    if nbright == 0:
        raise ValueError('linearity diagnostic found no bright trace pixels')
    new_diffs = np.zeros((ngroup - 1, nbright))
    old_diffs = np.zeros((ngroup - 1, nbright))
    num_pix = min(10_000, nbright)
    # Draw integrations in v1 order before transferring the sampled ramps.
    integrations1 = np.asarray([draw(0, nint) for _ in range(num_pix)])
    ypos1 = bright[0][:num_pix]
    xpos1 = bright[1][:num_pix] + (20 if miri else 0)
    new_ramps1 = np.asarray(cube[integrations1, :, ypos1, xpos1])
    old_ramps1 = np.asarray(old_cube[integrations1, :, ypos1, xpos1])
    new_diffs[:, :num_pix] = np.diff(new_ramps1, axis=1).T
    old_diffs[:, :num_pix] = np.diff(old_ramps1, axis=1).T
    new_med = np.mean(new_diffs, axis=1)
    old_med = np.mean(old_diffs, axis=1)
    plot1_locs = np.arange(ngroup - 1).astype(int)
    plot1_labels = np.asarray([f'{index + 2}-{index + 1}' for index in range(ngroup - 1)])
    if miri:
        # Select surviving group differences before subtracting their mean.
        difference_index = good_groups - 1
        old_med = old_med[difference_index]
        new_med = new_med[difference_index]
        plot1_locs = plot1_locs[difference_index]
        plot1_labels = plot1_labels[difference_index]
    old_diff = old_med - np.mean(old_med)
    new_diff = new_med - np.mean(new_med)
    plot1_tick_locs = plot1_locs
    plot1_tick_labels = plot1_labels
    if len(plot1_locs) > 10:
        selected = np.linspace(0, len(plot1_locs) - 1, 10).astype(int)
        plot1_tick_locs = plot1_locs[selected]
        plot1_tick_labels = plot1_labels[selected]

    # Capture ramp residuals for the second linearity plot.
    stack = last_group_median(cube, nint)
    if miri:
        stack = stack[:, 20:60]
    bright2 = np.where((stack >= np.nanpercentile(stack, 80)) &
        (stack < np.nanpercentile(stack, 99)))
    if len(bright2[0]) == 0:
        raise ValueError('linearity diagnostic found no bright trace pixels')
    selected_pixels = draw(0, len(bright2[0]), 1000)
    ypix = bright2[0][selected_pixels]
    xpix = bright2[1][selected_pixels]
    if miri:
        xpix = xpix + 20
    integrations = draw(0, nint, 1000)

    # Normalize each ramp over the surviving groups.
    groups = np.arange(ngroup) if good_groups is None else good_groups
    old_residuals = np.zeros((1000, ngroup))
    new_residuals = np.zeros((1000, ngroup))
    old_ramps = np.asarray(old_cube[integrations, :, ypix, xpix])
    new_ramps = np.asarray(cube[integrations, :, ypix, xpix])
    for index in range(1000):
        for ramps, residuals in ((old_ramps, old_residuals), (new_ramps, new_residuals)):
            ramp = ramps[index, groups]
            line = np.linspace(np.min(ramp), np.max(ramp), groups.size)
            residuals[index, groups] = (ramp - line) / np.max(ramp) * 100
    plot2_before = np.nanmedian(old_residuals, axis=0)
    plot2_after = np.nanmedian(new_residuals, axis=0)
    plot2_x = np.arange(ngroup) + 1
    # Use zero-based tick positions with one-based group labels as in v1.
    plot2_tick_locs = np.arange(ngroup).astype(int)
    plot2_tick_labels = (np.arange(ngroup) + 1).astype(str)
    if miri:
        # Keep only the surviving MIRI groups.
        plot2_x = plot2_x[good_groups]
        plot2_before = plot2_before[good_groups]
        plot2_after = plot2_after[good_groups]
        plot2_tick_locs = plot2_tick_locs[good_groups]
        plot2_tick_labels = plot2_tick_labels[good_groups]
    if len(plot2_tick_locs) > 10:
        selected = np.linspace(0, len(plot2_tick_locs) - 1, 10).astype(int)
        plot2_tick_locs = plot2_tick_locs[selected]
        plot2_tick_labels = plot2_tick_labels[selected]
    return {'instrument': instrument, 'plot1': {'locs': plot1_locs, 'tick_locs': plot1_tick_locs,
        'tick_labels': plot1_tick_labels, 'after': new_diff, 'before': old_diff,
        'ylim': (1.1 * np.nanmin(old_diff), 1.1 * np.nanmax(old_diff))}, 'plot2': {
        'x': plot2_x, 'tick_locs': plot2_tick_locs, 'tick_labels': plot2_tick_labels,
        'before': plot2_before, 'after': plot2_after}}


def render_linearity(record: Mapping[str, Any], path1, path2):
    """Render the exact v1 linearity styles from a compact record.

    Parameters
    ----------
    record : dict
        Captured diagnostic arrays.
    path1, path2 : str, Path
        Paths to which to save the first and second diagnostic plots.

    Returns
    -------
    path1, path2 : Path
        Paths to the saved diagnostic figures.
    """
    import matplotlib.pyplot as plt
    path1 = Path(path1)
    path2 = Path(path2)
    path1.parent.mkdir(parents=True, exist_ok=True)
    path2.parent.mkdir(parents=True, exist_ok=True)

    for index, path in enumerate((path1, path2), 1):
        plot = record[f'plot{index}']
        first = index == 1
        axis = plot['locs' if first else 'x']
        series = ('after', 'before') if first else ('before', 'after')
        plt.figure(figsize=(5, 3))
        for key, color in zip(series, ('blue', 'red')):
            options = {'lw': 2} if first else {}
            plt.plot(axis, plot[key], label=f'{key.title()} Correction', c=color, **options)
        plt.axhline(0, ls='--', c='black', **({'zorder': 0} if first else {}))
        if first:
            plt.xlabel('Groups', fontsize=12)
        plt.xticks(plot['tick_locs'], plot['tick_labels'], **({'rotation': 45} if first else {}))
        if not first:
            plt.xlabel('Group Number', fontsize=12)
        plt.ylabel('Differences [DN]' if first else 'Residual [%]', fontsize=12)
        if first:
            plt.ylim(*plot['ylim'])
        plt.legend()
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    return path1, path2
