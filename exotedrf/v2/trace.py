"""Locate spectral traces and read or write centroid tables."""

from datetime import datetime, timezone

import numpy as np


def load_centroids_csv(path):
    """Load detector-column and trace-center positions from a v1 CSV table.

    Parameters
    ----------
    path : str
        Path to the input or output file.

    Returns
    -------
    centroids : dict
        Detector positions and trace centers as named coordinate arrays.
    """
    import csv
    # Retain CSV row numbers for malformed-entry errors.
    parsed_rows = []
    try:
        with open(path, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f, strict=True)
            for row in reader:
                if not row or (len(row) == 1 and not row[0].strip()):
                    continue
                if row[0].lstrip().startswith('#'):
                    continue
                parsed_rows.append((reader.line_num, row))
    except csv.Error as exc:
        raise ValueError(f'malformed centroids CSV {path}: {exc}') from exc
    if not parsed_rows:
        raise ValueError(f'centroids CSV {path} contains no header')
    _, raw_header = parsed_rows[0]
    header = [name.strip() for name in raw_header]
    if not header or any(not name for name in header):
        raise ValueError(f'centroids CSV {path} has an empty column name')
    if len(set(header)) != len(header):
        raise ValueError(f'centroids CSV {path} has duplicate column names')
    if len(parsed_rows) == 1:
        raise ValueError(f'centroids CSV {path} contains no data rows')
    values = []
    for line_number, row in parsed_rows[1:]:
        if len(row) != len(header):
            raise ValueError(f'centroids CSV {path} row {line_number} has {len(row)} '
                f'fields; expected {len(header)}')
        converted = []
        for name, value in zip(header, row):
            value = value.strip()
            if not value:
                converted.append(np.nan)
                continue
            try:
                converted.append(float(value))
            except ValueError as exc:
                raise ValueError(f'centroids CSV {path} row {line_number}, column '
                    f'{name!r} is not numeric: {value!r}') from exc
        values.append(converted)
    data = np.asarray(values, dtype=float)
    return {name: data[:, i] for i, name in enumerate(header)}


def save_centroids_csv(path, cols):
    """Save trace positions in the CSV format understood by v1 and v2.

    Parameters
    ----------
    path : str
        Path to the input or output file.
    cols : dict
        Named detector and centroid coordinate arrays.
    """
    names = list(cols)
    arr = np.column_stack([np.asarray(cols[n], dtype=float) for n in names])
    with open(path, 'w') as f:
        f.write('# File Contents: Edgetrigger trace centroids\n')
        f.write('# File Creation Date: {}\n'.format(datetime.now(timezone.utc).replace(
                microsecond=0, tzinfo=None).isoformat()))
        f.write('# File Author: exoTEDRF v2\n')
        f.write(','.join(names) + '\n')
        for row in arr:
            f.write(','.join(f'{float(v):.10g}' for v in row) + '\n')


def make_order0_mask_from_f277w(f277w, thresh_std=1, thresh_size=10, start_col=700):
    """Locate field-star order-0 contaminants in a processed F277W image.

    Parameters
    ----------
    f277w : array-like(float)
        Processed F277W detector image.
    thresh_std : float
        Column-scatter threshold for detecting contaminants.
    thresh_size : int
        Minimum length of a contiguous contaminant; runs must exceed it.
    start_col : int
        First detector column to search for contaminants.

    Returns
    -------
    mask : np.ndarray(bool)
        Pixels affected by field-star order-0 contamination.
    """
    f277w = np.asarray(f277w, dtype=float)
    dimy, dimx = f277w.shape
    mask = np.zeros((dimy, dimx), dtype=bool)
    for col in range(min(start_col, dimx), dimx):
        column = f277w[:, col]
        diff = column - np.nanmedian(column)
        dev = np.nanstd(diff)
        with np.errstate(invalid='ignore'):
            deviant = np.abs(diff) > thresh_std * dev
        vals = np.nonzero(deviant)[0]
        if vals.size == 0:
            continue
        # Treat each contiguous bright or dark feature as one contaminant.
        breaks = np.nonzero(np.diff(vals) > 1)[0] + 1
        for group in np.split(vals, breaks):
            if len(group) > thresh_size:
                min_g = max(0, int(group[0]) - 3)
                max_g = min(dimy - 1, int(group[-1]) + 3)
                # Clamp the contaminant padding at the detector boundary.
                mask[min_g:max_g, max(0, col - 3):(col + 3)] = True
    return mask


def load_soss_waves(spectrace_path, dimx=2048, orders=(1, 2)):
    """Interpolate SOSS spectrace wavelengths onto detector columns.

    Parameters
    ----------
    spectrace_path : str
        Path to the SOSS trace-position reference file.
    dimx : int
        Number of detector columns.
    orders : tuple[int]
        Spectral orders to read.

    Returns
    -------
    waves : dict
        Wavelength vectors in microns, with NaN outside each order's coverage.
    """
    from astropy.io import fits as _fits
    out = {}
    with _fits.open(spectrace_path) as hdu:
        tables = [h for h in hdu[1:] if getattr(h, 'columns', None) is not None
                  and 'WAVELENGTH' in h.columns.names]
        for m, h in enumerate(tables, start=1):
            if m not in orders:
                continue
            x = np.asarray(h.data['X'], float)
            w = np.asarray(h.data['WAVELENGTH'], float)
            order_sort = np.argsort(x)
            x, w = x[order_sort], w[order_sort]
            xx = np.arange(dimx, dtype=float)
            wave = np.interp(xx, x, w, left=np.nan, right=np.nan)
            wave[(xx < x.min()) | (xx > x.max())] = np.nan
            out[m] = wave
    return out


def soss_github_wave_vectors(files_dir, subarray):
    """Read SOSS wavelength vectors from repository calibration files.

    Parameters
    ----------
    files_dir : str
        Directory containing repository calibration files.
    subarray : str
        Science subarray identifier.

    Returns
    -------
    waves : None, dict
        Order wavelength vectors and reference filename, or None if absent.
    """
    import os
    from astropy.io import fits as _fits
    name = ('jwst_niriss_wavemap_0020.fits' if subarray == 'SUBSTRIP96'
            else 'jwst_niriss_wavemap_0022.fits')
    path = os.path.join(files_dir, name)
    if not os.path.exists(path):
        return None
    with _fits.open(path, memmap=False) as hdu:
        out = {order: np.mean(np.asarray(hdu[order].data, dtype=float) [20:-20, 20:-20], axis=0)
               for order in (1, 2)}
    out['name'] = name
    return out


def load_soss_wavemap_vectors(wavemap_path, dimx=2048, orders=(1, 2)):
    """Collapse a CRDS SOSS wavemap reference to final 1D order vectors.

    Parameters
    ----------
    wavemap_path : str
        Path to the wavelength reference file.
    dimx : int
        Number of detector columns.
    orders : tuple[int]
        Spectral orders to read.

    Returns
    -------
    waves : dict
        Wavelength vector in microns for each requested spectral order.
    """
    from astropy.io import fits as _fits
    out = {}
    with _fits.open(wavemap_path, memmap=False) as hdu:
        images = [h for h in hdu[1:] if h.data is not None and np.asarray(h.data).ndim >= 2]
        for order, image in zip(orders, images):
            arr = np.asarray(image.data, dtype=float)
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(
                    f'wavemap order {order} must be 2D after squeeze, got {arr.shape}')
            if arr.shape[-1] < dimx:
                raise ValueError(
                    f'wavemap order {order} has {arr.shape[-1]} columns; expected {dimx}')
            xpad = arr.shape[-1] - dimx
            x0 = xpad // 2
            arr = arr[:, x0:x0 + dimx]
            # Remove the wavelength-map border before averaging trace rows.
            if xpad >= 40 and arr.shape[0] > 40:
                ypad = min(20, (arr.shape[0] - 1) // 2)
                trimmed = arr[ypad:arr.shape[0] - ypad]
                if trimmed.size:
                    arr = trimmed
            out[int(order)] = np.nanmean(arr, axis=0)
    if any(order not in out for order in orders):
        raise ValueError(f'wavemap {wavemap_path} does not contain orders {tuple(orders)}')
    return out


def validate_centroids(centroids, dimx, require_order2=False):
    """Validate and normalize a SOSS centroid mapping.

    Parameters
    ----------
    centroids : dict
        Named detector and trace-center coordinate arrays.
    dimx : int
        Number of detector columns.
    require_order2 : bool
        If True, require finite order-2 trace positions.

    Returns
    -------
    centroids : dict
        Detector positions and trace centers as named coordinate arrays.
    """
    if not isinstance(centroids, dict):
        raise TypeError('centroids must be a mapping of named arrays')
    if 'ypos' in centroids and 'ypos o1' not in centroids:
        centroids = dict(centroids)
        centroids['ypos o1'] = centroids.pop('ypos')
    if 'ypos o1' not in centroids:
        raise ValueError("centroids are missing 'ypos o1'")
    out = {k: np.asarray(v, dtype=float) for k, v in centroids.items()}
    if out['ypos o1'].ndim != 1 or out['ypos o1'].shape[0] != dimx:
        raise ValueError(
            f"order-1 centroids must have shape ({dimx},), got {out['ypos o1'].shape}")
    if not np.isfinite(out['ypos o1']).all():
        raise ValueError('order-1 centroids must be finite in every column')
    y2 = out.get('ypos o2')
    if require_order2 and (y2 is None or not np.isfinite(y2).any()):
        raise ValueError('SOSS order-2 centroids are required for this workflow')
    if y2 is not None and (y2.ndim != 1 or y2.shape[0] > dimx):
        raise ValueError(f'order-2 centroids must be 1D with at most {dimx} entries')
    return out


NIRSPEC_H_GRATINGS = ('G395H', 'G235H', 'G140H')
NIRSPEC_M_GRATINGS = ('G395M', 'G235M', 'G140M')
NIRSPEC_GRATINGS = NIRSPEC_H_GRATINGS + NIRSPEC_M_GRATINGS + ('PRISM',)


def nirspec_trace_start(detector, subarray, grating):
    """Get the first NIRSpec trace column using v1 get_nrs_trace_start.

    Parameters
    ----------
    detector : None, str
        Detector identifier for the observation.
    subarray : str
        Science subarray identifier.
    grating : str
        NIRSpec disperser identifier.

    Returns
    -------
    xstart : int
        First detector column of the NIRSpec trace.
    """
    detector = str(detector).lower()
    subarray = str(subarray).upper()
    grating = str(grating).upper()
    if detector != 'nrs1':
        return 0
    if grating in NIRSPEC_H_GRATINGS:
        return 0 if subarray == 'SUB1024B' else 500
    if grating in NIRSPEC_M_GRATINGS:
        return 0 if subarray == 'SUB1024B' else 200
    if grating == 'PRISM':
        return 14
    raise ValueError(f'Unknown NIRSpec grating {grating!r}')


def _centroid_mapping(centroids, keys, label):
    """Normalize centroid coordinates after checking required named arrays."""
    if not isinstance(centroids, dict):
        raise TypeError('centroids must be a mapping of named arrays')
    for key in keys:
        if key not in centroids:
            raise ValueError(f'{label} centroids are missing {key!r}')
    return {k: np.asarray(v, dtype=float) for k, v in centroids.items()}


def _matching_centroids(xpos, ypos, label):
    """Check matching, finite detector coordinate vectors."""
    if xpos.ndim != 1 or ypos.ndim != 1 or xpos.shape != ypos.shape:
        raise ValueError(
            f'{label} centroids must be matching 1D arrays, got xpos {xpos.shape} '
            f'and ypos {ypos.shape}')
    if not np.isfinite(xpos).all() or not np.isfinite(ypos).all():
        raise ValueError(f'{label} centroids must be finite everywhere')


def validate_nirspec_centroids(centroids, dimx, xstart=0):
    """Validate and normalize a NIRSpec centroid mapping.

    Parameters
    ----------
    centroids : dict
        Named detector and trace-center coordinate arrays.
    dimx, xstart : int
        Detector width and first trace column, respectively.

    Returns
    -------
    centroids : dict
        Detector positions and trace centers as named coordinate arrays.
    """
    out = _centroid_mapping(centroids, ('ypos',), 'NIRSpec')
    ypos = out['ypos']
    xpos = out.get('xpos')
    if xpos is None:
        xpos = np.arange(xstart, xstart + ypos.shape[0], dtype=float)
        out['xpos'] = xpos
    _matching_centroids(xpos, ypos, 'NIRSpec')
    if not np.allclose(np.diff(xpos), 1.):
        raise ValueError('NIRSpec centroid xpos must be contiguous columns')
    if xpos[0] < 0 or xpos[-1] > dimx - 1:
        raise ValueError(f'NIRSpec centroid columns [{xpos[0]}, {xpos[-1]}] fall outside '
            f'the detector (dimx={dimx})')
    return out


def get_centroids_nirspec(deepframe, xstart=0, xend=None):
    """Locate the NIRSpec trace with the edgetrigger method.

    Parameters
    ----------
    deepframe : array-like(float)
        Median detector image used to locate the trace.
    xstart : int
        First detector column to include.
    xend : None, int
        Exclusive final detector column; defaults to the image width.

    Returns
    -------
    centroids : dict
        Detector positions and trace centers as named coordinate arrays.
    """
    deepframe = np.asarray(deepframe, dtype=float)
    dimy, dimx = deepframe.shape
    xend = dimx if xend is None else int(xend)
    xstart = int(xstart)
    if not 0 <= xstart < xend <= dimx:
        raise ValueError(f'invalid NIRSpec centroid window [{xstart}, {xend}) for dimx={dimx}')
    try:
        from exotedrf import utils as v1utils
        cens = v1utils.get_centroids_nirspec(
            deepframe, xstart=xstart, xend=xend, save_results=False)
        return {'xpos': np.asarray(cens[0], float), 'ypos': np.asarray(cens[1], float)}
    except ImportError:
        pass
    except TypeError as exc:
        if 'expected non-empty vector for x' not in str(exc):
            raise
    # Recover constant traces when the clipping fit has no valid columns.
    window = deepframe[:, xstart:xend]
    x_local = np.arange(window.shape[1], dtype=float)
    try:
        from applesoss import get_centroids_edgetrigger
    except ImportError:
        raw = np.full(window.shape[1], np.nan)
        for i in range(window.shape[1]):
            col = window[:, i]
            if np.isfinite(col).any() and np.nanmax(col) > np.nanmedian(col):
                raw[i] = edgetrigger_centroids_1d(col - np.nanmedian(col), halfwidth=3)
    else:
        # Measure the same edges without the failed clipping fit.
        raw = np.asarray(get_centroids_edgetrigger(
            window, mode='mean', poly_order=None, halfwidth=3)[1], float)
    good = np.isfinite(raw)
    if good.sum() < 3:
        raise ValueError('Too few columns with a detectable NIRSpec trace.')
    pp = np.polyfit(x_local[good], raw[good], 2)
    x1 = x_local + xstart
    y1 = np.polyval(pp, x_local)
    ii = np.where((x1 >= xstart) & (x1 <= xend - 1))
    xx1 = np.linspace(xstart, xend - 1, (xend - 1) - xstart + 1)
    yy1 = np.interp(xx1, x1[ii], y1[ii])
    return {'xpos': xx1, 'ypos': yy1}


MIRI_TRACE_CROP = (26, 250)
MIRI_TRACE_YSTART = 50


def validate_miri_centroids(centroids, dimy, ystart=0):
    """Validate and normalize a MIRI/LRS centroid mapping.

    Parameters
    ----------
    centroids : dict
        Named detector and trace-center coordinate arrays.
    dimy, ystart : int
        Detector height and first trace row, respectively.

    Returns
    -------
    centroids : dict
        Detector positions and trace centers as named coordinate arrays.
    """
    out = _centroid_mapping(centroids, ('xpos', 'ypos'), 'MIRI')
    xpos, ypos = out['xpos'], out['ypos']
    _matching_centroids(xpos, ypos, 'MIRI')
    if not np.allclose(np.diff(ypos), 1.):
        raise ValueError('MIRI centroid ypos must be contiguous detector rows')
    if ypos[0] < 0 or ypos[-1] > dimy - 1:
        raise ValueError(f'MIRI centroid rows [{ypos[0]}, {ypos[-1]}] fall outside the '
            f'detector (dimy={dimy})')
    if ypos[0] < ystart:
        raise ValueError(f'MIRI centroids start at row {ypos[0]}, before ystart={ystart}')
    return out


SOSS_TRACETABLE_URL = ('https://raw.githubusercontent.com/radicamc/exoTEDRF/main/files/')


def soss_tracetable_name(subarray):
    """Return the v1 SOSS spectrace filename for the supplied subarray."""
    if str(subarray).upper() == 'SUBSTRIP96':
        return 'jwst_niriss_spectrace_0022.fits'
    return 'jwst_niriss_spectrace_0023.fits'


def resolve_soss_tracetable(subarray, search_dirs=(), *, download=True, logger=None):
    """Locate the SOSS tracetable that v1 2.5.0 uses for auto-centroids.

    Parameters
    ----------
    subarray : str
        Science subarray identifier.
    search_dirs : None, list[str]
        Directories to search for the reference file.
    download : bool
        If True, download a missing reference into the CRDS cache.
    logger : None, callable
        Function to receive progress messages.

    Returns
    -------
    path : None, str
        Located file path, or None when no file was selected.
    """
    import os
    filename = soss_tracetable_name(subarray)
    crds_dir = None
    if os.environ.get('CRDS_PATH'):
        crds_dir = os.path.join(os.environ['CRDS_PATH'], 'references', 'jwst', 'niriss')
    for directory in list(search_dirs) + [crds_dir]:
        if directory and os.path.isfile(os.path.join(directory, filename)):
            return os.path.join(directory, filename)
    if not download or crds_dir is None:
        return None
    try:
        import urllib.request
        os.makedirs(crds_dir, exist_ok=True)
        target = os.path.join(crds_dir, filename)
        with urllib.request.urlopen(SOSS_TRACETABLE_URL + filename, timeout=60) as response:
            data = response.read()
        with open(target, 'wb') as handle:
            handle.write(data)
        return target
    except Exception as exc:
        if logger is not None:
            logger(f'could not download {filename}: {exc}')
        return None


def get_centroids_miri(deepframe, ystart=MIRI_TRACE_YSTART, yend=None, allow_slope=False):
    """Locate the MIRI/LRS trace with the edgetrigger method.

    Short images clamp the MIRI trace crop to the available detector rows.

    Parameters
    ----------
    deepframe : array-like(float)
        Median detector image used to locate the trace.
    ystart : int
        First detector row to include.
    yend : None, int
        Exclusive final detector row; defaults to the image height.
    allow_slope : bool
        If True, allow the MIRI trace centroid to vary with detector row.

    Returns
    -------
    centroids : dict
        Detector positions and trace centers as named coordinate arrays.
    """
    deepframe = np.asarray(deepframe, dtype=float)
    dimy, dimx = deepframe.shape
    yend = dimy if yend is None else int(yend)
    ystart = int(ystart)
    if not 0 <= ystart < yend <= dimy:
        raise ValueError(f'invalid MIRI centroid window [{ystart}, {yend}) for dimy={dimy}')
    try:
        from exotedrf import utils as v1utils
        if dimy > MIRI_TRACE_CROP[1]:
            cens = v1utils.get_centroids_miri(
                deepframe, ystart=ystart, yend=yend, save_results=False, allow_slope=allow_slope)
            return {'xpos': np.asarray(cens[0], float), 'ypos': np.asarray(cens[1], float)}
    except ImportError:
        pass
    # Trace the flipped MIRI geometry with the local edge trigger.
    crop_lo = min(MIRI_TRACE_CROP[0], max(dimy - 1, 0))
    crop_hi = min(MIRI_TRACE_CROP[1], dimy)
    if crop_hi - crop_lo < 2:
        crop_lo, crop_hi = 0, dimy
    window = deepframe[::-1].T[:, crop_lo:crop_hi]
    ncol = window.shape[1]
    raw = np.full(ncol, np.nan)
    noise = np.nanmedian(np.abs(window)) + 1e-12
    for i in range(ncol):
        col = window[:, i]
        if np.nanmax(col) > 5.0 * noise:
            raw[i] = edgetrigger_centroids_1d(col, halfwidth=2)
    good = np.isfinite(raw)
    poly_order = 1 if allow_slope else 0
    if good.sum() < poly_order + 1:
        raise ValueError('Too few rows with a detectable MIRI trace.')
    local = np.arange(ncol, dtype=float)
    pp = np.polyfit(local[good], raw[good], poly_order)
    # Restore native detector coordinates.
    x1 = np.polyval(pp, local)
    y1 = dimy - (crop_lo + local)
    yy1 = np.linspace(ystart, yend - 1, (yend - 1) - ystart + 1)
    if np.min(yy1) < (dimy - crop_hi) or np.max(yy1) > (dimy - crop_lo):
        pp = np.polyfit(y1, x1, 1)
        xx1 = np.polyval(pp, yy1)
    else:
        order = np.argsort(y1)
        xx1 = np.interp(yy1, y1[order], x1[order])
    return {'xpos': xx1, 'ypos': yy1}


def _smooth1d(y, half):
    """Smooth a profile with a uniform convolution kernel."""
    k = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(np.nan_to_num(y, nan=0.0), k, mode='same')


def edgetrigger_centroids_1d(profile, halfwidth=2):
    """Estimate a trace center between its two strongest smoothed edges.

    Parameters
    ----------
    profile : array-like(float)
        Cross-dispersion intensity profile.
    halfwidth : int
        Half-width of the profile smoothing window.

    Returns
    -------
    center : float
        Midpoint of the smoothed trace edges; NaN when unresolved.
    """
    sm = _smooth1d(profile, halfwidth)
    grad = np.gradient(sm)
    top, bottom = np.nanargmax(grad), np.nanargmin(grad)
    if bottom <= top:
        return np.nan
    return 0.5 * (top + bottom)


def get_centroids_fallback(deepframe, poly_order=4, min_snr=5.0):
    """Trace a horizontal spectrum and smooth its column centers with a polynomial.

    Parameters
    ----------
    deepframe : array-like(float)
        Median detector image used to locate the trace.
    poly_order : int
        Polynomial order used to smooth measured trace centers.
    min_snr : float
        Minimum trace amplitude relative to the image noise estimate.

    Returns
    -------
    xpos, ypos : np.ndarray(float)
        Detector column grid and smoothed trace centers, respectively.
    """
    dimy, dimx = deepframe.shape
    x = np.arange(dimx)
    raw = np.full(dimx, np.nan)
    noise = np.nanmedian(np.abs(deepframe)) + 1e-12
    for i in x:
        col = deepframe[:, i]
        if np.nanmax(col) > min_snr * noise:
            raw[i] = edgetrigger_centroids_1d(col)
    good = np.isfinite(raw)
    if good.sum() < poly_order + 1:
        raise ValueError('Too few columns with a detectable trace.')
    pp = np.polyfit(x[good], raw[good], poly_order)
    return x.astype(float), np.polyval(pp, x)


def get_centroids(deepframe, mode, tracetable=None, subarray=None, centroids_csv=None):
    """Load or measure trace centers for the observation.

    A supplied CSV takes precedence. Automatic SOSS tracing requires v1/applesoss routines.

    Parameters
    ----------
    deepframe : array-like(float)
        Median detector image used to locate the trace.
    mode : None, str
        Observing mode, such as NIRISS/SOSS.
    tracetable : None, str
        Path to the SOSS trace-position reference file.
    subarray : str
        Science subarray identifier.
    centroids_csv : None, str
        Path to a centroid table; takes precedence over automatic tracing.

    Returns
    -------
    centroids : dict
        Detector positions and trace centers as named coordinate arrays.
    """
    if centroids_csv is not None:
        cols = load_centroids_csv(centroids_csv)
        if mode.upper().startswith('MIRI'):
            return validate_miri_centroids(cols, deepframe.shape[-2])
        if mode.upper().startswith('NIRSPEC'):
            xstart = int(np.asarray(cols.get('xpos', [0]))[0])
            return validate_nirspec_centroids(cols, deepframe.shape[-1], xstart=xstart)
        if 'ypos' in cols and 'ypos o1' not in cols:
            cols['ypos o1'] = cols.pop('ypos')
        return validate_centroids(cols, deepframe.shape[-1])
    if mode.upper().startswith('NIRISS'):
        try:
            from exotedrf import utils as v1utils
            c1, c2, c3 = v1utils.get_centroids_soss(
                deepframe, tracetable, subarray, save_results=False)
            out = {'xpos': c1[0], 'ypos o1': c1[1]}
            y2 = np.full_like(c1[1], np.nan)
            y2[:len(c2[1])] = c2[1]
            y3 = np.full_like(c1[1], np.nan)
            y3[:len(c3[1])] = c3[1]
            out['ypos o2'], out['ypos o3'] = y2, y3
            return out
        except ImportError as exc:
            raise RuntimeError('SOSS automatic tracing requires the v1/applesoss tracing '
                'dependencies. Install them or supply an explicit centroid '
                'CSV; a single-order fallback cannot safely calibrate SOSS.') from exc
    if mode.upper().startswith('MIRI'):
        return get_centroids_miri(deepframe)
    xx, yy = get_centroids_fallback(deepframe)
    return {'xpos': xx, 'ypos o1': yy}


def get_soss_centroids_with_group_fallback(deepstack, *, tracetable=None, subarray=None):
    """Trace SOSS on the latest usable group of a baseline deepstack.

    Try groups from last to first, matching v1 _get_soss_centroids_with_group_fallback.

    Parameters
    ----------
    deepstack : array-like(float)
        Median image or stack of median images by detector group.
    tracetable : None, str
        Path to the SOSS trace-position reference file.
    subarray : str
        Science subarray identifier.

    Returns
    -------
    centroids : dict
        Detector positions and trace centers as named coordinate arrays.
    """
    deepstack = np.asarray(deepstack)
    if deepstack.ndim == 2:
        return validate_centroids(get_centroids(deepstack, 'NIRISS/SOSS',
                          tracetable=tracetable, subarray=subarray), deepstack.shape[-1])
    if deepstack.ndim != 3:
        raise ValueError(f'SOSS centroid deepstack must be 2D or 3D, got {deepstack.shape}')
    failures = []
    for group in range(deepstack.shape[0] - 1, -1, -1):
        try:
            centroids = get_centroids(deepstack[group], 'NIRISS/SOSS',
                tracetable=tracetable, subarray=subarray)
            return validate_centroids(centroids, deepstack.shape[-1])
        except (TypeError, ValueError, RuntimeError) as exc:
            failures.append(f'group {group}: {exc}')
    raise RuntimeError('Could not locate SOSS centroids in any group deepstack. Pass a '
        'known-good centroids CSV in the config. Failures: ' + '; '.join(failures))
