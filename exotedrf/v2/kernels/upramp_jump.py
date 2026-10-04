"""Detect up-the-ramp jumps and expand large cosmic ray events."""

import functools

import jax
import jax.numpy as jnp
import numpy as np

_DNU = 1
_SAT = 2
_JUMP = 4


def _take0(arr, index):
    """Gather one value per pixel along the first axis."""
    index = jnp.clip(index, 0, arr.shape[0] - 1)
    return jnp.take_along_axis(arr, index[None, :], axis=0)[0]


def _median_sorted_range(sorted_vals, lo, count):
    """Calculate the median of a sorted range per pixel."""
    half = count // 2
    odd = (count % 2) == 1
    upper = _take0(sorted_vals, lo + half)
    lower = _take0(sorted_vals, jnp.where(odd, lo + half, lo + half - 1))
    med = jnp.where(odd, upper, (lower + upper) / 2)
    return jnp.where(count > 0, med, jnp.nan)


_UMAX = 0xFFFFFFFF


def _order_keys(values, valid):
    """Convert float32 values to unsigned ordered keys, placing invalid values last."""
    bits = jax.lax.bitcast_convert_type(values.astype(jnp.float32), jnp.uint32)
    keys = jnp.where((bits >> 31) == 1, ~bits, bits | jnp.uint32(0x80000000))
    return jnp.where(valid, keys, jnp.uint32(_UMAX))


def _key_values(keys):
    """Convert unsigned ordered keys back to float32 values."""
    bits = jnp.where((keys >> 31) == 1, keys & jnp.uint32(0x7FFFFFFF), ~keys)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _kth_key(keys, k):
    """Find the zero-based order statistic using bitwise bisection."""
    lo = jnp.zeros(keys.shape[1:], jnp.uint32)
    hi = jnp.full(keys.shape[1:], _UMAX, jnp.uint32)

    def body(_, bounds):
        """Update the loop state."""
        lo, hi = bounds
        mid = lo + (hi - lo) // 2
        enough = jnp.sum(keys <= mid[None], axis=0) >= k + 1
        return jnp.where(enough, lo, mid + 1), jnp.where(enough, mid, hi)

    return jax.lax.fori_loop(0, 32, body, (lo, hi))[0]


def pooled_nanmedian(values, valid):
    """Calculate per-pixel medians of usable pooled samples.

    Parameters
    ----------
    values : array-like(float)
        Samples with the pooled sample axis first.
    valid : array-like(bool)
        True for usable samples, matching the values.

    Returns
    -------
    median : array-like(float)
        Median with the first axis removed; empty samples return NaN.
    """
    count = jnp.sum(valid, axis=0).astype(jnp.int32)
    if values.dtype != jnp.float32:
        key = jnp.where(valid, values, jnp.inf)
        sorted_vals = jnp.sort(key.reshape(values.shape[0], -1), axis=0)
        flat_count = count.reshape(-1)
        return _median_sorted_range(sorted_vals, jnp.zeros_like(flat_count),
            flat_count).reshape(values.shape[1:])
    return _median_of_smallest(_order_keys(values, valid), count).astype(values.dtype)


def _median_of_smallest(keys, count):
    """Calculate the median of the selected smallest keys per column."""
    k_lo = jnp.maximum(count - 1, 0) // 2
    lower = _kth_key(keys, k_lo)
    n_le = jnp.sum(keys <= lower[None], axis=0)
    above = jnp.min(jnp.where(keys > lower[None], keys, jnp.uint32(_UMAX)), axis=0)
    odd = (count % 2) == 1
    upper = jnp.where(odd | (n_le >= k_lo + 2), lower, above)
    a, b = _key_values(lower), _key_values(upper)
    return jnp.where(count > 0, jnp.where(odd, a, (a + b) / 2), jnp.nan)


def _host_stable_argsort(keys, mask):
    """Sort selected key columns on the host, retaining the order of ties."""
    keys = np.asarray(keys)
    mask = np.asarray(mask)
    order = np.broadcast_to(np.arange(keys.shape[0], dtype=np.int32)[:, None], keys.shape).copy()
    cols = np.flatnonzero(mask)
    if cols.size:
        sub = np.ascontiguousarray(keys[:, cols].T)
        order[:, cols] = np.argsort(sub, axis=-1, kind='stable').T
    return order


def _stable_argsort_axis0(keys, mask):
    """Sort key columns stably along the first axis."""
    if jax.default_backend() == 'cpu':
        return jax.pure_callback(_host_stable_argsort,
            jax.ShapeDtypeStruct(keys.shape, jnp.int32), keys, mask)
    flat_index = jnp.broadcast_to(jnp.arange(keys.shape[0], dtype=jnp.int32)[:, None], keys.shape)
    return jax.lax.sort((keys, flat_index), dimension=0, is_stable=True, num_keys=1)[1]


def iterative_crs(abs_diffs, valid, rn2_nf, normal_thresh, three_diff_thresh, two_diff_thresh):
    """Detect cosmic rays by iteratively rejecting pooled group differences.

    Reject all ties at the maximum ratio together and recalculate the median.

    Parameters
    ----------
    abs_diffs : array-like(float)
        Absolute first differences in electrons with shape (nsamples, npixels).
    valid : array-like(bool)
        True for usable samples, matching the values.
    rn2_nf : array-like(float)
        Squared electron read noise divided by frames per group.
    normal_thresh, three_diff_thresh, two_diff_thresh : float
        Thresholds for at least four, three and two remaining differences, respectively. Traced.

    Returns
    -------
    flagged : array-like(bool)
        Cosmic ray mask matching the pooled differences.
    """
    n_items, n_pix = abs_diffs.shape
    dtype = abs_diffs.dtype
    values = jnp.where(valid, abs_diffs, jnp.inf)
    n = jnp.sum(valid, axis=0).astype(jnp.int32)
    inf = jnp.asarray(jnp.inf, dtype)

    def threshold(m):
        """Select the rejection threshold for the remaining sample count."""
        return jnp.where(m >= 4, normal_thresh, jnp.where(m == 3, three_diff_thresh,
                                   jnp.where(m == 2, two_diff_thresh, inf)))

    def ratios(med, v_lo, v_hi):
        """Calculate rejection ratios for the smallest and largest differences."""
        sigma = jnp.sqrt(jnp.abs(med) + rn2_nf)
        sigma = jnp.where(sigma == 0, jnp.nan, sigma)
        return jnp.abs(v_lo - med) / sigma, jnp.abs(v_hi - med) / sigma

    if dtype == jnp.float32:
        # Evaluate the first rejection pass before sorting candidate pixels.
        keys = _order_keys(abs_diffs, valid)
        key_lo = jnp.min(keys, axis=0)
        key_hi = jnp.max(jnp.where(valid, keys, jnp.uint32(0)), axis=0)
        n_top = jnp.sum(valid & (keys == key_hi[None]), axis=0)
        kept = jnp.where(n >= 4, n - n_top, n)
        med0 = jnp.where(n >= 3, _median_of_smallest(keys, kept),
                         jnp.where(n == 2, _key_values(key_lo), jnp.nan)).astype(dtype)
        r_lo, r_hi = ratios(med0, _key_values(key_lo).astype(dtype),
                            _key_values(key_hi).astype(dtype))
        candidate = jnp.fmax(r_lo, r_hi) > threshold(n)
        idx = _stable_argsort_axis0(keys, candidate)
    else:
        candidate = n >= 2
        flat_index = jnp.broadcast_to(
            jnp.arange(n_items, dtype=jnp.int32)[:, None], (n_items, n_pix))
        idx = jax.lax.sort(((~valid).astype(jnp.int8), values, flat_index),
                           dimension=0, is_stable=True, num_keys=2)[2]
    s = jnp.take_along_axis(values, idx, axis=0)
    pos = jnp.arange(n_items, dtype=jnp.int32)[:, None]
    differs = s[1:] != s[:-1]
    first = jnp.concatenate([jnp.ones((1, n_pix), bool), differs], axis=0)
    last = jnp.concatenate([differs, jnp.ones((1, n_pix), bool)], axis=0)
    block_start = jax.lax.cummax(jnp.where(first, pos, 0), axis=0)
    block_end = jax.lax.cummin(jnp.where(last, pos, n_items - 1), axis=0, reverse=True)

    def cond(state):
        """Check whether the loop should continue."""
        return jnp.any(state[2])

    def body(state):
        """Update the loop state."""
        lo, hi, active = state
        m = hi - lo + 1
        lo_c = jnp.clip(lo, 0, n_items - 1)
        hi_c = jnp.clip(hi, 0, n_items - 1)
        top = jnp.maximum(_take0(block_start, hi_c), lo)
        bottom = jnp.minimum(_take0(block_end, lo_c), hi)
        v_lo, v_hi = _take0(s, lo_c), _take0(s, hi_c)
        med = jnp.where(m >= 4, _median_sorted_range(s, lo, top - lo),
            jnp.where(m == 3, _take0(s, lo_c + 1), jnp.where(m == 2, v_lo, jnp.nan)))
        rl, rh = ratios(med, v_lo, v_hi)
        rmax = jnp.fmax(rl, rh)
        found = active & (rmax > threshold(m))
        lo = jnp.where(found & (rl == rmax), bottom + 1, lo)
        hi = jnp.where(found & (rh == rmax), top - 1, hi)
        return lo, hi, found & ((hi - lo + 1) >= 2)

    zero = jnp.zeros_like(n)
    lo, hi, _ = jax.lax.while_loop(cond, body, (zero, n - 1, candidate))
    flagged_sorted = (pos < n) & ((pos < lo) | (pos > hi)) & candidate[None]
    cols = jnp.broadcast_to(jnp.arange(n_pix)[None, :], (n_items, n_pix))
    return jnp.zeros((n_items, n_pix), bool).at[idx, cols].set(flagged_sorted)


def _row_masks(nrows, slice_starts, slice_modes):
    """Build row masks for the detector slices and their boundaries."""
    bounds = [0, *slice_starts, nrows]
    mode_of_row = np.empty(nrows, dtype=object)
    for k, mode in enumerate(slice_modes):
        mode_of_row[bounds[k]:bounds[k + 1]] = mode
    starts = np.zeros(nrows, bool)
    starts[list(slice_starts)] = True
    from_next_cross = np.zeros(nrows, bool)
    from_next_cross[:-1] = starts[1:]
    return mode_of_row, starts, from_next_cross


@functools.partial(jax.jit, static_argnames=('slice_starts', 'slice_modes', 'flag_4_neighbors'))
def find_crs(data, groupdq, gain, readnoise, nframes, rejection_threshold,
             three_group_threshold, four_group_threshold,
             max_jump_to_flag_neighbors, min_jump_to_flag_neighbors,
             after_jump_flag_e1, after_jump_flag_n1, after_jump_flag_e2,
             after_jump_flag_n2, *, slice_starts=(), slice_modes=('single',),
             flag_4_neighbors=True):
    """Detect up-the-ramp jumps with neighbour and after-jump flagging.

    Keep every integration and detector row when chunking columns. Expand large events separately.

    Parameters
    ----------
    data, groupdq : array-like(float), array-like(int)
        Detector samples and group flags with shape (nints, ngroups, dimy, dimx).
    readnoise, gain : array-like(float), float
        Single-read noise in DN and gain in electrons per DN, respectively.
    nframes : int
        Reads averaged into each group. Traced.
    rejection_threshold : float
        Jump detection threshold in scatter units. Traced.
    three_group_threshold, four_group_threshold : float
        Rejection thresholds for three- and four-group ramps, respectively. Traced.
    min_jump_to_flag_neighbors, max_jump_to_flag_neighbors : float
        Exclusive lower and upper rejection ratios for neighbour flagging, respectively. Traced.
    after_jump_flag_e1, after_jump_flag_e2 : float
        Electron thresholds for the first and second after-jump passes, respectively. Traced.
    after_jump_flag_n1, after_jump_flag_n2 : int
        Groups flagged in each after-jump pass; zero disables the pass. Traced.
    slice_starts : tuple[int]
        First row of each detector slice after the first. Static: changes trigger a recompile.
    slice_modes : tuple[str]
        Mode per row slice: single, iterative or skip. Static: changes trigger a recompile.
    flag_4_neighbors : bool
        Flag the four neighbouring pixels of eligible jumps. Static: changes trigger a recompile.

    Returns
    -------
    groupdq : array-like(int)
        Updated group flags with the input shape.
    """
    nints, ngroups, nrows, ncols = data.shape
    dtype = data.dtype
    if ngroups < 2:
        raise ValueError('up-the-ramp jump detection needs >= 2 groups')
    ndiffs = ngroups - 1
    mode_of_row, starts_np, next_cross_np = _row_masks(nrows, slice_starts, slice_modes)
    modes = set(slice_modes)
    row_shape = (1, 1, nrows, 1)
    skip_rows = jnp.asarray(mode_of_row == 'skip').reshape(row_shape)
    run_rows = ~skip_rows
    groupdq = groupdq.astype(jnp.uint8)
    unusable = (groupdq & jnp.uint8(_SAT | _DNU)) != 0
    gain = gain.astype(dtype)
    electrons = jnp.where(unusable, jnp.nan, data * gain)
    rn_e = readnoise.astype(dtype) * gain
    rn2_nf = rn_e ** 2 / jnp.asarray(nframes, dtype)
    diffs = electrons[:, 1:] - electrons[:, :-1]
    valid = ~jnp.isnan(diffs)
    flat = diffs.reshape(nints * ndiffs, nrows, ncols)
    flat_valid = valid.reshape(flat.shape)
    median = pooled_nanmedian(flat, flat_valid)
    sigma = jnp.sqrt(jnp.abs(median) + rn2_nf)
    sigma = jnp.where(sigma == 0, jnp.nan, sigma)
    e_jump = diffs - median
    ratio = jnp.abs(e_jump) / sigma
    detected = jnp.zeros(diffs.shape, bool)
    if 'single' in modes:
        single = (ratio > rejection_threshold) & valid
        rows = jnp.asarray(mode_of_row == 'single').reshape(row_shape)
        detected = jnp.where(rows, single, detected)
    if 'iterative' in modes:
        pooled = iterative_crs(jnp.abs(flat).reshape(nints * ndiffs, nrows * ncols),
            flat_valid.reshape(nints * ndiffs, nrows * ncols),
            rn2_nf.reshape(-1), rejection_threshold, four_group_threshold,
            three_group_threshold).reshape(diffs.shape)
        rows = jnp.asarray(mode_of_row == 'iterative').reshape(row_shape)
        detected = jnp.where(rows, pooled, detected)

    no_group = jnp.zeros((nints, 1, nrows, ncols), bool)
    jump = ((groupdq & jnp.uint8(_JUMP)) != 0) | jnp.concatenate([no_group, detected], axis=1)
    usable_target = ~unusable
    no_diff = jnp.full((nints, 1, nrows, ncols), jnp.nan, dtype)
    deferred = jnp.zeros_like(jump)
    if flag_4_neighbors:
        ratio_prev = jnp.concatenate([no_diff, ratio], axis=1)
        src = (jump & (ratio_prev < max_jump_to_flag_neighbors) &
               (ratio_prev > min_jump_to_flag_neighbors) & run_rows)
        pad_r = jnp.zeros((nints, ngroups, 1, ncols), bool)
        pad_c = jnp.zeros((nints, ngroups, nrows, 1), bool)
        from_next_row = jnp.concatenate([src[:, :, 1:], pad_r], axis=2)
        from_prev_row = jnp.concatenate([pad_r, src[:, :, :-1]], axis=2)
        from_next_col = jnp.concatenate([src[..., 1:], pad_c], axis=3)
        from_prev_col = jnp.concatenate([pad_c, src[..., :-1]], axis=3)
        next_cross = jnp.asarray(next_cross_np).reshape(row_shape)
        prev_cross = jnp.asarray(starts_np).reshape(row_shape)
        in_slice = ((from_next_row & ~next_cross) | (from_prev_row & ~prev_cross) |
                    from_next_col | from_prev_col) & usable_target
        deferred = (from_next_row & next_cross) | (from_prev_row & prev_cross)
        jump = jump | in_slice

    # Start both after-jump passes from the same detected jump set.
    e_prev = jnp.concatenate([no_diff, e_jump], axis=1)
    group_index = jnp.arange(ngroups, dtype=jnp.int32).reshape(1, -1, 1, 1)
    after_any = jnp.zeros_like(jump)
    for threshold, count in ((after_jump_flag_e1, after_jump_flag_n1),
                             (after_jump_flag_e2, after_jump_flag_n2)):
        count = jnp.asarray(count, jnp.int32)
        src = jump & (e_prev >= threshold) & run_rows
        last_src = jax.lax.cummax(jnp.where(src, group_index, jnp.int32(-(1 << 30))), axis=1)
        after = (((group_index - last_src) <= count) & usable_target & run_rows & (count > 0))
        after_any = after_any | after
    jump = jump | after_any
    flags = jnp.where(jump | deferred, jnp.uint8(_JUMP), jnp.uint8(0))
    out = groupdq | flags
    # Remove JUMP_DET wherever DO_NOT_USE or SATURATED is set.
    for bit in (_DNU, _SAT):
        both = jnp.uint8(bit | _JUMP)
        out = jnp.where((out & both) == both, out ^ jnp.uint8(_JUMP), out)
    return out


def _find_ellipses(plane, bitmask, min_area):
    """Fit moment ellipses to flagged regions above the minimum area."""
    import skimage.measure
    pixels = np.bitwise_and(plane, bitmask)
    if not pixels.any():
        return []
    ellipses = []
    for region in skimage.measure.regionprops(skimage.measure.label(pixels)):
        if region.area_filled < min_area:
            continue
        w = region.axis_major_length - 1
        h = region.axis_minor_length - 1
        ellipses.append(((float(region.centroid[1]), float(region.centroid[0])),
                         (h, w), np.degrees(region.orientation)))
    return ellipses


def _ellipse_subim(ceny, cenx, axis1, axis2, alpha, value, shape):
    """Rasterize an ellipse within a clipped image region."""
    import skimage.draw
    yc, xc = round(ceny), round(cenx)
    dn_over_2 = max(round(axis1 / 2), round(axis2 / 2)) + 2
    ix1 = max(yc - dn_over_2, 0)
    ix2 = min(yc + dn_over_2 + 1, shape[1])
    iy1 = max(xc - dn_over_2, 0)
    iy2 = min(xc + dn_over_2 + 1, shape[0])
    image = np.zeros(shape=(iy2 - iy1, ix2 - ix1), dtype=np.uint8)
    axes = (round(axis1 / 2), round(axis2 / 2))
    if axes[0] != 0 and axes[1] != 0:
        saty, satx = skimage.draw.ellipse(
            xc - iy1, yc - ix1, axes[1], axes[0], (iy2 - iy1, ix2 - ix1), np.radians(alpha))
        image[saty, satx] = value
    return (iy1, iy2, ix1, ix2), image


def _point_inside_ellipse(point, ellipse):
    """Check whether a point lies inside a rotated ellipse."""
    angle = np.radians(180 - ellipse[2])
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    xc = point[0] - ellipse[0][0]
    yc = point[1] - ellipse[0][1]
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle
    semi_major_axis = max(ellipse[1]) * 0.5
    semi_minor_axis = min(ellipse[1]) * 0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        rad_cc = (xct ** 2 / semi_major_axis ** 2 + yct ** 2 / semi_minor_axis ** 2)
    return rad_cc <= 1


def _near_edge(jump, low, high):
    """Check whether an event center lies near a detector edge."""
    return (jump[0][0] < low or jump[0][1] < low or jump[0][0] > high or jump[0][1] > high)


def _expanded_axes(ellipse, expansion, max_width):
    """Expand ellipse axes by ratio and limit their size."""
    if ellipse[1][1] < ellipse[1][0]:
        axis1 = ellipse[1][0] + (expansion - 1.0) * ellipse[1][1]
        axis2 = ellipse[1][1] * expansion
    else:
        axis1 = ellipse[1][0] * expansion
        axis2 = ellipse[1][1] + (expansion - 1.0) * ellipse[1][0]
    axis1 = min(axis1, max_width)
    axis2 = min(axis2, max_width)
    return round(axis1 / 2), round(axis2 / 2)


def _snowballs_one_integration(cube, persist, *, sat_flag, jump_flag,
                               min_sat_area, min_jump_area, expand_factor,
                               sat_required_snowball, min_sat_radius_extend,
                               sat_expand, edge_size, max_extended_radius):
    """Expand large events and record persistence for one integration."""
    ngrps, nrows, ncols = cube.shape
    events = 0
    low, high = edge_size, max(0, nrows - edge_size)
    next_new_sat = None
    for group in range(1, ngrps):
        current_sat = np.bitwise_and(cube[group], sat_flag)
        prev_sat = np.bitwise_and(cube[group - 1], sat_flag)
        new_sat = current_sat * np.logical_not(prev_sat)
        if group < ngrps - 1:
            next_sat = np.bitwise_and(cube[group + 1], sat_flag)
            next_new_sat = next_sat * np.logical_not(current_sat)
        jump_ellipses = _find_ellipses(cube[group], jump_flag, min_jump_area)
        sat_ellipses = _find_ellipses(new_sat, sat_flag, min_sat_area)
        if sat_required_snowball:
            next_sat_ellipses = (_find_ellipses(next_new_sat, sat_flag, min_sat_area)
                if (jump_ellipses and group < ngrps - 1) else [])
            # Require saturated cores for events away from the detector edge.
            snowballs = []
            cores = sat_ellipses + next_sat_ellipses
            for jump in jump_ellipses:
                if (_near_edge(jump, low, high) or
                        (jump not in snowballs and
                         any(_point_inside_ellipse(sat[0], jump) for sat in cores))):
                    snowballs.append(jump)
            # Extend saturation and record the next-integration persistence mask.
            for ellipse in sat_ellipses:
                minor_axis = min(ellipse[1][1], ellipse[1][0])
                if minor_axis > min_sat_radius_extend:
                    axis1 = min(ellipse[1][0] + sat_expand, max_extended_radius)
                    axis2 = min(ellipse[1][1] + sat_expand, max_extended_radius)
                    (iy1, iy2, ix1, ix2), image = _ellipse_subim(
                        ellipse[0][0], ellipse[0][1], axis1, axis2, ellipse[2], 22, (nrows, ncols))
                    is_sat = image == 22
                    for i in range(group, ngrps):
                        cube[i, iy1:iy2, ix1:ix2][is_sat] = sat_flag
                    (iy1, iy2, ix1, ix2), image = _ellipse_subim(
                        ellipse[0][0], ellipse[0][1], ellipse[1][0],
                        ellipse[1][1], ellipse[2], 22, (nrows, ncols))
                    persist[iy1:iy2, ix1:ix2][image == 22] = jump_flag
        else:
            snowballs = jump_ellipses
        events += len(snowballs)
        # Expand each event without adding jump flags to saturated pixels.
        for ellipse in snowballs:
            axes = _expanded_axes(ellipse, expand_factor, max_extended_radius)
            (iy1, iy2, ix1, ix2), image = _ellipse_subim(
                ellipse[0][0], ellipse[0][1], axes[0] * 2, axes[1] * 2,
                ellipse[2], jump_flag, (nrows, ncols))
            sat_pix = cube[group, iy1:iy2, ix1:ix2] & sat_flag
            image[sat_pix == sat_flag] = 0
            cube[group, iy1:iy2, ix1:ix2] |= image
    return events


def flag_large_events_host(gdq, *, min_sat_area=1., min_jump_area=5.,
                           expand_factor=2., sat_required_snowball=True,
                           min_sat_radius_extend=2.5, sat_expand=2,
                           edge_size=25, max_extended_radius=200, mask_persist_grps_next_int=True,
                           persist_grps_flagged=5, sat_flag=_SAT,
                           jump_flag=_JUMP, max_workers=None):
    """Expand large cosmic ray events and flag next-integration persistence.

    Parameters
    ----------
    gdq : np.ndarray(int)
        Group flags with shape (nints, ngroups, dimy, dimx), modified in place.
    min_sat_area, min_jump_area : float
        Minimum saturated and jump region areas in pixels, respectively.
    expand_factor : float
        Ratio by which to expand snowball ellipses.
    sat_required_snowball : bool
        Require a saturated region inside events away from the detector edge.
    min_sat_radius_extend : float
        Minimum minor-axis size for extending saturation.
    sat_expand : float
        Saturation axis extension supplied to STCAL, already doubled by the step.
    edge_size : int
        Detector edge region width in pixels.
    max_extended_radius : float
        Maximum ellipse axis size supplied to STCAL, already doubled by the step.
    mask_persist_grps_next_int : bool
        Flag persistence in the next integration.
    persist_grps_flagged : int
        Exclusive final group to flag for next-integration persistence.
    sat_flag, jump_flag : int
        Saturation and jump detection flag bits, respectively.
    max_workers : None, int
        Thread pool size; None uses the executor default.

    Returns
    -------
    gdq : np.ndarray(int)
        Input group flags modified in place.
    number_of_snowballs : int
        Number of detected large events.
    """
    from concurrent.futures import ThreadPoolExecutor
    gdq = np.asarray(gdq)
    if gdq.dtype != np.uint8:
        raise TypeError('snowball flagging expects a uint8 GROUPDQ')
    nints, ngrps, nrows, ncols = gdq.shape
    persist = np.zeros((nints, nrows, ncols), np.uint8)
    kwargs = dict(sat_flag=sat_flag, jump_flag=jump_flag,
                  min_sat_area=min_sat_area, min_jump_area=min_jump_area,
                  expand_factor=expand_factor, sat_required_snowball=sat_required_snowball,
                  min_sat_radius_extend=min_sat_radius_extend,
                  sat_expand=sat_expand, edge_size=edge_size,
                  max_extended_radius=max_extended_radius)

    def run(integration):
        """Expand large events in one integration."""
        return _snowballs_one_integration(gdq[integration], persist[integration], **kwargs)

    if max_workers == 1 or nints == 1:
        total = sum(run(i) for i in range(nints))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            total = sum(pool.map(run, range(nints)))
    # Apply persistence after processing all integrations.
    if mask_persist_grps_next_int and persist_grps_flagged >= 1:
        last_grp = min(int(persist_grps_flagged), ngrps)
        for intg in range(1, nints):
            gdq[intg, 1:last_grp] |= persist[intg - 1][None]
    return gdq, int(total)


__all__ = ['find_crs', 'flag_large_events_host', 'iterative_crs', 'pooled_nanmedian']
