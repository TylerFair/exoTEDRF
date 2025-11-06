#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for the optimize.py script with handling of 
DQ flags and extraction.
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
import os

from exotedrf import utils
from exotedrf.utils import fancyprint
from exotedrf.stage3 import get_wave_nirspec, get_wave_soss, get_wave_miri


def apply_dq_flags(datafiles):
    """
    Load data and apply DQ flags by NaN-ing out bad pixels.
    Errors are NOT loaded/returned since they're not needed for optimization.

    Parameters
    datafiles

    Returns
    cube  : array
        Flux with bad pixels as NaN
    is_4d : bool
        True if pre-RampFit (4D), False if post (3D)
    """
    datafiles = np.atleast_1d(datafiles)

    # get flux and DQ (errors not needed for optimization)
    for i, file in enumerate(datafiles):
        fancyprint(f'Loading segment {i}: {file if isinstance(file, str) else "datamodel"}')

        if isinstance(file, str):
            data = fits.getdata(file, 1)
            dq = fits.getdata(file, 3)
            fancyprint(f'  Loaded from FITS: data.shape={data.shape}, dq.shape={dq.shape}')
        else:
            with utils.open_filetype(file) as datamodel:
                data = datamodel.data
                dq = datamodel.dq
                fancyprint(f'  Loaded from datamodel: data.shape={data.shape}, dq.shape={dq.shape if dq is not None else None}')

        if dq is not None:
            # for 4D data (pre-rampfit), take last group
            is_4d = data.ndim == 4
            fancyprint(f'  is_4d={is_4d}, data.ndim={data.ndim}')

            if is_4d:
                # data shape: (nint, ngroup, y, x)
                # dq shape: (nint, ngroup, y, x) or (x, y, ngroup, nint) - need to check
                fancyprint(f'  4D processing: dq.shape={dq.shape}, data.shape={data.shape}')

                if dq.ndim == 4 and dq.shape[0] != data.shape[0]:
                    fancyprint(f'  Transposing DQ from {dq.shape} to match data')
                    dq = np.transpose(dq, (3, 2, 1, 0))
                    fancyprint(f'  After transpose: dq.shape={dq.shape}')

                # take last group for mask
                dq_for_mask = dq[:, -1, :, :]
                fancyprint(f'  Took last group: dq_for_mask.shape={dq_for_mask.shape}')

                # boolean mask - anything non-zero flag is bad
                bad_pixels = (dq_for_mask > 0).astype(bool)
                fancyprint(f'  bad_pixels (before broadcast).shape={bad_pixels.shape}')

                # expand to all groups
                bad_pixels = bad_pixels[:, np.newaxis, :, :]
                fancyprint(f'  After newaxis: bad_pixels.shape={bad_pixels.shape}')

                bad_pixels = np.broadcast_to(bad_pixels, data.shape)
                fancyprint(f'  After broadcast to data.shape: bad_pixels.shape={bad_pixels.shape}')
            else:
                # 3D data (post-RampFit) has shape (nint, y, x)
                fancyprint(f'  3D processing: dq.ndim={dq.ndim}, dq.shape={dq.shape}')

                if dq.ndim == 4:
                    fancyprint(f'  DQ is 4D, taking last group')
                    dq_for_mask = dq[:, -1, :, :]
                    fancyprint(f'  dq_for_mask.shape={dq_for_mask.shape}')
                elif dq.ndim == 3:
                    fancyprint(f'  DQ is 3D, using as-is')
                    dq_for_mask = dq
                elif dq.ndim == 2:
                    fancyprint(f'  DQ is 2D (PIXELDQ), broadcasting to data shape')
                    bad_pixels = (dq > 0).astype(bool)
                    bad_pixels = bad_pixels[np.newaxis, :, :]
                    bad_pixels = np.broadcast_to(bad_pixels, data.shape)
                    fancyprint(f'  bad_pixels.shape={bad_pixels.shape}')
                    dq_for_mask = None

                if dq_for_mask is not None:
                    bad_pixels = (dq_for_mask > 0).astype(bool)
                    fancyprint(f'  bad_pixels.shape={bad_pixels.shape}')

            # Apply mask
            fancyprint(f'  Applying mask: data.shape={data.shape}, bad_pixels.shape={bad_pixels.shape}')

            data[bad_pixels] = np.nan

            n_bad = np.sum(bad_pixels)
            fancyprint(f'Segment {i}: Flagged {n_bad}/{bad_pixels.size} pixels ({100*n_bad/bad_pixels.size:.2f}%)')
        else:
            fancyprint(f'Segment {i}: No DQ found', msg_type='WARNING')
            is_4d = data.ndim == 4

        # concatenate segments
        if i == 0:
            cube = data
        else:
            cube = np.concatenate([cube, data])

    return cube, is_4d


def do_box_extraction_nanaware(cube, ypos, width, extract_start=0, extract_end=None, progress=True):
    """
    Box extraction with nansum. Modified from stage3.do_box_extraction.
    Note: Errors are NOT calculated since they're not needed for optimization.

    Parameters
    cube :  (nint, y, x)
    ypos
        Y positions
    width :
        extraction  width
    extract_start : int
    extract_end : int or None

    Returns
    f :  (nint, nx) - Extracted flux
    """
    assert cube.ndim == 3, f"Expected 3D, got {cube.ndim}D shape {cube.shape}"

    nint, dimy, dimx = np.shape(cube)

    if extract_end is None:
        extract_end = dimx

    f = np.zeros((nint, dimx))

    edge_up = np.min([ypos + width / 2, np.ones_like(ypos) * dimy], axis=0)
    edge_low = np.max([ypos - width / 2, np.zeros_like(ypos)], axis=0)

    for i in tqdm(range(nint), disable=not progress, desc='Extracting'):
        for x in range(extract_start, extract_end):
            xx = x - extract_start
            if xx >= len(ypos):
                xx = len(ypos) - 1

            up_whole = np.floor(edge_up[xx]).astype(int)
            low_whole = np.ceil(edge_low[xx]).astype(int)

            #  total flux and total valid pixel area
            box = cube[i, low_whole:up_whole, x]

            total_flux = np.nansum(box)
            total_area = np.sum(np.isfinite(box))  #   valid whole pixels

            # add partial pixels
            if edge_up[xx] < (dimy-1) and edge_low[xx] > 0:
                up_part = edge_up[xx] % 1
                low_part = 1 - edge_low[xx] % 1

                up_val = cube[i, up_whole, x]
                low_val = cube[i, low_whole-1, x]

                # add partial pixel flux if valid
                if np.isfinite(up_val):
                    total_flux += up_part * up_val
                    total_area += up_part

                if np.isfinite(low_val):
                    total_flux += low_part * low_val
                    total_area += low_part

            # normalize by total valid pixel area
            if total_area > 0:
                f[i, x] = total_flux / total_area
            else:
                f[i, x] = np.nan

    return f


def extract_at_step(datafile, instrument, extract_width, centroids, baseline_ints, output_dir):
    """
    Extract spectra from a datafile at any pipeline step.
    Note: Errors are NOT returned since they're not needed for optimization.

    Parameters
    datafile
         datafile to extract (should be first segment if all is well)
    instrument
        'NIRISS', 'NIRSPEC', or 'MIRI'
    extract_width
        Extraction width (dict with 'o1'/'o2' for SOSS)
    centroids
        Centroids (will generate/cache if None)
    baseline_ints
        Baseline integrations
    output_dir
        For caching centroids

    Returns
    spectral_dict
        Keys: 'Wave', 'Flux' (and O1/O2 versions for SOSS) - no errors
    centroids
        The centroids used (for caching)
    """
    fancyprint(f'=== Extracting {instrument} at current step ===')
    fancyprint(f'  datafile: {datafile if isinstance(datafile, str) else "datamodel"}')

    # load with flags applied
    fancyprint(f'  Loading data with DQ flags...')
    cube, is_4d = apply_dq_flags([datafile])
    fancyprint(f'  Loaded: cube.shape={cube.shape}, is_4d={is_4d}')

    # convert 4D to 3D if needed
    if is_4d:
        fancyprint(f'  4D data detected: {cube.shape} -> taking last group')
        cube = cube[:, -1, :, :]
        fancyprint(f'  Now 3D: cube.shape={cube.shape}')

    assert cube.ndim == 3, f"Expected 3D after conversion, got {cube.ndim}D with shape {cube.shape}"

    # get centroids
    if centroids is None:
        cache_file = os.path.join(output_dir, 'cached_centroids.csv')
        if os.path.exists(cache_file):
            fancyprint(f'Loading cached centroids: {cache_file}')
            centroids = pd.read_csv(cache_file, comment='#')
        else:
            fancyprint('Generating centroids from deep stack')
            deepstack = utils.make_baseline_stack_general(datafiles=[datafile], baseline_ints=baseline_ints)
            if np.ndim(deepstack) == 3:
                deepstack = deepstack[-1]

            if instrument == 'NIRISS':
                from jwst.pipeline import calwebb_spec2
                subarray = utils.get_soss_subarray(datafile)
                step = calwebb_spec2.extract_1d_step.Extract1dStep()
                tracetable = step.get_reference_file(datafile, 'spectrace')
                cens = utils.get_centroids_soss(deepstack, tracetable, subarray, save_results=False)
                centroids = pd.DataFrame({
                    'xpos': cens[0][0],
                    'ypos o1': cens[0][1],
                    'ypos o2': cens[1][1],
                    'ypos o3': cens[2][1]
                }) # copying logic from satge1 1/f 
            elif instrument == 'NIRSPEC':
                det = utils.get_nrs_detector_name(datafile)
                subarray = utils.get_soss_subarray(datafile)
                grating = utils.get_nrs_grating(datafile)
                xstart = utils.get_nrs_trace_start(det, subarray, grating)
                cens = utils.get_centroids_nirspec(deepstack, xstart=xstart, save_results=False)
                centroids = pd.DataFrame({'xpos': cens[0], 'ypos': cens[1]})
            elif instrument == 'MIRI':
                from exotedrf.stage2 import TracingStep
                tracer = TracingStep([datafile], deepframe=deepstack, output_dir=output_dir)
                cens = tracer.run(save_results=False, force_redo=False)
                centroids = pd.DataFrame({'xpos': cens[0], 'ypos': cens[1]})

            centroids.to_csv(cache_file, index=False)
            fancyprint(f'Cached centroids: {cache_file}')

    # extract by instrument
    if instrument == 'NIRSPEC':
        x1, y1 = centroids['xpos'].values, centroids['ypos'].values
        det = utils.get_nrs_detector_name(datafile)
        subarray = utils.get_soss_subarray(datafile)
        grating = utils.get_nrs_grating(datafile)
        xstart = utils.get_nrs_trace_start(det, subarray, grating)

        flux = do_box_extraction_nanaware(cube, y1, width=extract_width, extract_start=xstart)
        wave = get_wave_nirspec(datafile, centroids, cube.shape[0], cube.shape[2])

        return {'Wave': wave, 'Flux': flux}, centroids

    elif instrument == 'NIRISS':
        x1 = centroids['xpos'].values
        y1, y2 = centroids['ypos o1'].values, centroids['ypos o2'].values

        if isinstance(extract_width, dict):
            w1 = extract_width.get('o1', 40)
            w2 = extract_width.get('o2', w1)
        else:
            w1 = w2 = extract_width

        flux_o1 = do_box_extraction_nanaware(cube, y1, width=w1)

        ii = np.where(np.isfinite(y2))[0]
        y2_finite = y2[ii]
        flux_o2 = do_box_extraction_nanaware(cube, y2_finite, width=w2, extract_end=len(y2_finite))

        wave_o1, wave_o2 = get_wave_soss(datafile)

        return {
            'Wave O1': wave_o1, 'Flux O1': flux_o1,
            'Wave O2': wave_o2, 'Flux O2': flux_o2
        }, centroids

    elif instrument == 'MIRI':
        x1, y1 = centroids['xpos'].values, centroids['ypos'].values

        flux = do_box_extraction_nanaware(
            cube.transpose(0, 2, 1), x1,
            width=extract_width, extract_start=int(np.min(y1)), extract_end=int(np.max(y1))
        )

        wave = get_wave_miri(datafile, centroids, cube.shape[0], cube.shape[1])

        return {'Wave': wave, 'Flux': flux}, centroids

    else:
        raise ValueError(f"Unknown instrument: {instrument}")
