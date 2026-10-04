"""Sequential and joint electromagnetic interference correction for MIRI ramps."""

import numpy as np

__all__ = ['apply_emicorr', 'apply_emicorr_joint']

_R1_READPATTS = ('FASTR1', 'SLOWR1')


def _sloper(data):
    """Calculate a slope per pixel, accumulating groups in order."""
    ngroups = data.shape[0]
    sxy = np.zeros(data.shape[1:], dtype=np.float64)
    sy = np.zeros(data.shape[1:], dtype=np.float64)
    sx = 0.0
    sxx = 0.0
    for groupcount in range(ngroups):
        t = float(groupcount)
        sxy = sxy + t * data[groupcount]
        sx = sx + t
        sy = sy + data[groupcount]
        sxx = sxx + t * t
    return (ngroups * sxy - sx * sy) / (ngroups * sxx - sx * sx)


def _rebin(arr, newlen):
    """Resample a one-dimensional array using the nearest lower index."""
    slices = [slice(0, arr.shape[0], float(arr.shape[0]) / newlen)]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')
    return arr[tuple(indices)]


def _sigma_clipped_mean(values):
    """Calculate the mean after iterative median-centered sigma clipping."""
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    for _ in range(5):
        if vals.size == 0:
            return np.nan
        med = np.median(vals)
        std = np.std(vals)
        keep = np.abs(vals - med) <= 3.0 * std
        if np.all(keep):
            break
        vals = vals[keep]
    if vals.size == 0:
        return np.nan
    return np.mean(vals)


def apply_emicorr(data, frequencies, reference_waves, rowclocks, frameclocks,
                  readpatt='FASTR1', nsamples=1, xstart=1,
                  nints_to_phase=None, nbins=None, scale_reference=True, use_n_cycles=3):
    """Subtract sequential EMI waveform corrections from a MIRI segment.

    Apply to each segment separately; fewer than four groups return an unchanged copy.

    Parameters
    ----------
    data : array-like(float)
        Detector samples with shape (nints, ngroups, dimy, dimx).
    frequencies : array-like(float)
        EMI frequencies in Hz, in reference order.
    reference_waves : None, list[np.ndarray(float)]
        Waveforms per frequency; None self-corrects sequential EMI or skips joint EMI.
    rowclocks, frameclocks : int
        Row and frame timing in detector clock units, respectively.
    readpatt : str
        Readout pattern; FASTR1 and SLOWR1 include an extra reset frame.
    nsamples : int
        Samples per pixel read.
    xstart : int
        One-based starting detector column of the subarray.
    nints_to_phase : None, int
        Leading phase integrations; an inferred count is reused for later frequencies.
    nbins : None, int
        Phase bin count; None derives it from the waveform period.
    scale_reference : bool
        Scale the reference waveform amplitude to the measured waveform.
    use_n_cycles : None, int
        Waveform cycles used to infer the phase integration count.

    Returns
    -------
    corrected : np.ndarray(float)
        Corrected copy of the ramp cube with the input dtype.
    """
    data = np.asarray(data)
    nints, ngroups, ny, nx = data.shape
    if nx % 4 != 0 or nx // 4 < 4:
        raise ValueError('dimx must be a multiple of 4 with dimx/4 >= 4 '
                         '(MIRI amplifier interleaving); got dimx={}' .format(nx))
    out_dtype = data.dtype
    if ngroups < 4:
        return data.copy()

    # Keep per-frequency rounding in the science dtype.
    output = np.array(data, copy=True)
    nx4 = nx // 4
    readpatt = str(readpatt).upper()
    extra_rowclocks = int((1024. - ny) * (4 + 3.))
    frame_block = ny * int(rowclocks) + extra_rowclocks
    int_block = ngroups * frame_block
    if readpatt in _R1_READPATTS:
        int_block += int(frameclocks)
    clock_offsets = (np.arange(ngroups, dtype=np.uint64)[:, None, None] * np.uint64(frame_block)
        + np.arange(ny, dtype=np.uint64)[None, :, None] * np.uint64(rowclocks)
        + np.arange(nx4, dtype=np.uint64)[None, None, :] * np.uint64(nsamples))
    times_this_int = np.empty(clock_offsets.shape, dtype=np.uint64)
    colstop = int(nx / 4 + xstart - 1)

    for fi, frequency in enumerate(frequencies):
        period_in_pixels = (1. / float(frequency)) / 10.0e-6

        # Reuse the first inferred phase integration count for later frequencies.
        if nints_to_phase is None and use_n_cycles is None:
            nints_to_phase = nints
        elif nints_to_phase is None and use_n_cycles is not None:
            nints_to_phase = (use_n_cycles * period_in_pixels) / (frameclocks * ngroups)
            nints_to_phase = int(np.ceil(nints_to_phase))
        elif nints_to_phase is not None and use_n_cycles == 3:
            if nints_to_phase > nints:
                nints_to_phase = nints

        phase_count = min(nints, max(0, int(nints_to_phase)))
        dd_all = np.empty((phase_count, ngroups, ny, nx4), dtype=np.float64)
        phaseall = np.empty((nints, ngroups, ny, nx4), dtype=np.float64)
        start_time = 0
        for integration in range(nints):
            # Clean only integrations used for phase estimation.
            if integration < phase_count:
                work = output[integration].copy()
                s0 = _sloper(work[1:ngroups - 1])
                for group in range(ngroups):
                    work[group] = (output[integration, group] - s0 * group)
                m0 = np.nanmedian(work[1:ngroups - 1], axis=0)
                for group in range(ngroups):
                    work[group] = work[group] - m0
                    d0 = work[group, :, 0:nx:4]
                    d1 = work[group, :, 1:nx:4]
                    d2 = work[group, :, 2:nx:4]
                    d3 = work[group, :, 3:nx:4]
                    dd = (d0 + d1 + d2 + d3) / 4.
                    fix = (dd[:, 0] + dd[:, 3]) / 2
                    dd[:, 1] = fix
                    dd[:, 2] = fix
                    dd_all[integration, group] = dd - np.median(dd)

            # Count pixel clocks one integration at a time.
            np.add(clock_offsets, np.uint64(start_time), out=times_this_int)
            if colstop == 258:
                times_this_int[..., nx4 - 1] += np.uint64(3 + 2 ** 32)
            phase_this_int = times_this_int / period_in_pixels
            phaseall[integration] = (phase_this_int - phase_this_int.astype(np.uint64))
            start_time += int_block

        if nbins is None:
            nbins_f = int(period_in_pixels / 2.0)
        else:
            nbins_f = nbins
        if nbins_f > 501:
            nbins_f = 500
        phase_temp = phaseall[:phase_count]
        dd_temp = dd_all
        pa = np.zeros(nbins_f, dtype=np.float64)
        for nb in range(nbins_f):
            u = (phase_temp > nb / nbins_f) & (phase_temp <= (nb + 1) / nbins_f)
            pa[nb] = _sigma_clipped_mean(dd_temp[u])
        pa -= np.median(pa)
        lut = _rebin(pa, period_in_pixels)

        if reference_waves is not None:
            reference_wave = np.asarray(reference_waves[fi], dtype=np.float64)
            if not np.all(np.isfinite(reference_wave)) or np.std(reference_wave) == 0.:
                del dd_all, phaseall
                continue
            reference_wave_size = np.size(reference_wave)
            rebinned_pa = _rebin(pa, reference_wave_size)
            cc = np.zeros(reference_wave_size)
            for i in range(reference_wave_size):
                shifted_ref_wave = np.roll(reference_wave, i)
                pears_coeff = np.corrcoef(shifted_ref_wave, rebinned_pa)
                cc[i] = pears_coeff[0, 1]
            u = np.argmax(cc)
            lut_reference = _rebin(np.roll(reference_wave, u), period_in_pixels)
            m, _ = np.polyfit(lut_reference, lut, 1)
            if scale_reference:
                lut_reference = lut_reference * m
            lut = lut_reference

        for integration in range(nints):
            dd_noise = lut[(phaseall[integration] * period_in_pixels).astype(int)]
            dd_noise[~np.isfinite(dd_noise)] = 0.0
            for k in range(4):
                output[integration, ..., k::4] -= dd_noise
        del dd_all, phaseall

    return output.astype(out_dtype, copy=False)


def _best_phase(phases, chisq):
    """Refine the phase with a parabola through the chi-square minimum."""
    chisq = np.asarray(chisq, dtype=float)
    if np.all(~np.isfinite(chisq)):
        return phases[0]
    ibest = int(np.nanargmin(chisq))
    ibest_m1, ibest_p1 = ibest - 1, ibest + 1
    if ibest_p1 > len(chisq) - 1:
        ibest_p1 = 0
    chisq_opt = [chisq[ibest_m1], chisq[ibest], chisq[ibest_p1]]
    if np.any(~np.isfinite(chisq_opt)):
        return phases[ibest]
    x = [phases[ibest_m1], phases[ibest], phases[ibest_p1]]
    if np.nanstd(chisq_opt) == 0:
        return phases[ibest]
    a, b, _ = np.polyfit(x, chisq_opt, 2)
    return -b / (2 * a)


def _chisq_amplitudes(fit, phases):
    """Calculate chi-square values and amplitudes for the trial phases."""
    phaselist = fit['phaselist']
    nints = fit['all_y'].shape[0]
    ints = np.arange(nints)
    s_tt, s_t, ngroups, delta = (fit['s_tt'], fit['s_t'], fit['ngroups'], fit['delta'])
    # Calculate waveform products for all integrations and reference phases.
    w = fit['zflat'] @ fit['yflat'].T
    chisq, amplitudes = [], []
    for phase in phases:
        k = np.empty(nints, dtype=np.int64)
        for lo in range(0, nints, 64):
            sel = ints[lo:lo + 64]
            dist = np.abs((phaselist[None, :] - phase - (fit['dphase_frame'] * sel)[:, None]) % 1)
            k[lo:lo + 64] = np.argmin(dist, axis=1)
        n_ = fit['all_n']
        s_y, s_ty = fit['all_sy'], fit['all_sty']
        s_z, s_tz, s_zz = fit['sz'][k], fit['stz'][k], fit['szz'][k]
        a_terms = np.sum(n_ / delta * (-s_tt * s_z ** 2 + 2 * s_t * s_z * s_tz -
                                       ngroups * s_tz ** 2 + s_zz * delta), axis=1)
        b_terms = 2 / delta * np.sum(s_tt * s_z * s_y - s_t * s_tz * s_y - s_t * s_z * s_ty +
            ngroups * s_tz * s_ty, axis=1)
        yz = 2 * w[k, ints]
        a_ = np.cumsum(np.concatenate([[0.], a_terms]))[-1]
        b_ = np.cumsum(np.concatenate([[0.], np.stack([b_terms, -yz], axis=1).ravel()]))[-1]
        if np.isclose(a_, 0, atol=1e-08) or ~np.isfinite(a_) or ~np.isfinite(b_):
            chisq.append(np.nan)
            amplitudes.append(0.0)
        else:
            c = -b_ / (2 * a_)
            chisq.append(a_ * c ** 2 + b_ * c)
            amplitudes.append(c)
    return chisq, amplitudes


def _emicorr_refwave(data, pdq, refwave, nsamples, rowclocks, frameclocks,
                     period_in_pixels, nphases_opt=500):
    """Fit and subtract one reference waveform across all integrations."""
    from scipy import interpolate
    nints, ngroups, ny, nx = data.shape
    nx4 = nx // 4
    extra_rowclocks = (1024.0 - ny) * (4 + 3.0)
    frametime = ny * rowclocks + extra_rowclocks
    t0_arr = (np.arange(ny)[:, None] * rowclocks + np.arange(nx4)[None, :] * nsamples).astype(float)
    phase = t0_arr / period_in_pixels % 1
    dphase = frametime / period_in_pixels % 1
    dphase_frame = dphase * ngroups + frameclocks / period_in_pixels
    grouptimes = np.arange(ngroups)
    refwave = np.asarray(refwave)
    refwave_extended = np.array([refwave[-1]] + list(refwave) + [refwave[0]])
    phase_extended = (np.arange(len(refwave) + 2) - 0.5) / len(refwave)
    phasefunc = interpolate.interp1d(phase_extended, refwave_extended, kind='cubic')
    phases_template = (phase_extended[1:-1, np.newaxis] + grouptimes * dphase) % 1
    nphases = phases_template.shape[0]
    # Bin usable pixels by phase before fitting the waveform.
    pixel_std = np.std(data, axis=1)
    pixel_ok = (pixel_std < 2 * np.median(pixel_std)) & (pdq == 0)
    all_y = np.zeros((nints, nphases, ngroups))
    all_n = np.zeros((nints, nphases))
    cols = np.array([k for k in range(nx4) if k not in (1, 2)], dtype=int)
    for j in range(ny):
        for k in cols:
            indx = int(phase[j, k] * nphases)
            for lane in range(4):
                pixok = pixel_ok[:, j, k * 4 + lane]
                all_y[:, indx] += data[:, :, j, k * 4 + lane] * pixok[:, np.newaxis]
                all_n[:, indx] += pixok

    phaselist = np.arange(nphases_opt) * 1.0 / nphases_opt
    zlist = np.stack([phasefunc((phases_template.T + dphaseval) % 1).T for dphaseval in phaselist])
    s_tt = np.sum(grouptimes ** 2)
    s_t = np.sum(grouptimes)
    fit = {'all_y': all_y, 'all_n': all_n, 'phaselist': phaselist,
        'dphase_frame': dphase_frame, 'ngroups': ngroups, 's_tt': s_tt,
        's_t': s_t, 'delta': ngroups * s_tt - s_t ** 2, 'all_sy': np.sum(all_y, axis=2),
        'all_sty': np.sum(all_y * grouptimes, axis=2),
        'sz': np.sum(zlist, axis=2), 'stz': np.sum(grouptimes * zlist, axis=2),
        'szz': np.sum(zlist ** 2, axis=2), 'zflat': zlist.reshape(nphases_opt, -1),
        'yflat': all_y.reshape(nints, -1), }
    chisq, _ = _chisq_amplitudes(fit, phaselist)
    best_phase = _best_phase(phaselist, chisq)
    _, c = _chisq_amplitudes(fit, [best_phase])
    phases_to_correct = (best_phase + np.arange(nints) * dphase_frame) % 1
    amplitude = c[0]
    for i in range(nints):
        for j in range(ngroups):
            phased_emi = phasefunc((phase + dphase * j + phases_to_correct[i]) % 1)
            correction = amplitude * phased_emi
            for k in range(4):
                data[i, j, :, k::4] -= correction
    return data


def apply_emicorr_joint(data, pixeldq, frequencies, reference_waves,
                        rowclocks, frameclocks, readpatt='FASTR1', nsamples=1, nphases_opt=500):
    """Fit and subtract EMI waveforms jointly with linear MIRI ramps.

    Apply to each segment separately; fewer than three groups return an unchanged copy.

    Parameters
    ----------
    data : array-like(float)
        Detector samples with shape (nints, ngroups, dimy, dimx).
    pixeldq : array-like(int)
        Pixel flags with shape (dimy, dimx).
    frequencies : array-like(float)
        EMI frequencies in Hz, in reference order.
    reference_waves : None, list[np.ndarray(float)]
        Waveforms per frequency; None self-corrects sequential EMI or skips joint EMI.
    rowclocks, frameclocks : int
        Row and frame timing in detector clock units, respectively.
    readpatt : str
        Readout pattern; FASTR1 and SLOWR1 include an extra reset frame.
    nsamples : int
        Samples per pixel read.
    nphases_opt : int
        Number of trial phase offsets.

    Returns
    -------
    corrected : np.ndarray(float)
        Corrected copy of the ramp cube with the input dtype.
    """
    data = np.array(data, copy=True)
    if data.shape[1] < 3 or reference_waves is None:
        return data
    pixeldq = np.asarray(pixeldq)
    _frameclocks = (frameclocks if str(readpatt).upper() in _R1_READPATTS else 0)
    for freq, ref_wave in zip(frequencies, reference_waves, strict=True):
        period_in_pixels = 1.0 / freq / 1e-05
        data = _emicorr_refwave(data, pixeldq, np.array(ref_wave), nsamples,
                                rowclocks, _frameclocks, period_in_pixels, nphases_opt=nphases_opt)
    return data
