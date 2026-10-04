"""Read and write JWST observations and extracted spectra."""

import glob
import os
import re
import tempfile

import numpy as np
from astropy.io import fits

from exotedrf.v2.core import (FRAME_TIME_S, ObsMeta, RampCube, host_allocation_budget_bytes)


def _science_shapes(files):
    """Read the dimensions of each SCI extension without loading its pixels."""
    shapes = []
    for path in files:
        with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
            try:
                shape = tuple(hdul['SCI'].shape)
            except (KeyError, AttributeError):
                raise ValueError(f'No SCI extension in {path}') from None
        if len(shape) == 3:
            shape = (1,) + shape
        if len(shape) != 4:
            raise ValueError(f'SCI in {path} must be 4D (nint, ngroup, y, x), got {shape}')
        shapes.append(shape)
    return shapes


def estimate_ramp_storage_bytes(files, science_dtype=np.float32):
    """Estimate the memory needed for science ramps and group DQ arrays.

    Parameters
    ----------
    files : list[str]
        Paths to the input FITS segments, in exposure order.
    science_dtype : dtype
        Data type of the array.

    Returns
    -------
    nbytes : int
        Storage required for science ramps and uint8 group flags.
    """
    dtype = np.dtype(science_dtype)
    if dtype.kind != 'f':
        raise TypeError('science_dtype must be a floating dtype')
    return sum(int(np.prod(shape)) * (dtype.itemsize + np.dtype(np.uint8).itemsize)
        for shape in _science_shapes(files))


def _allocate_host_array(shape, dtype, *, spill, scratch_dir, max_bytes=None):
    """Allocate input storage within the host budget or on scratch."""
    from exotedrf.v2 import core
    return core.empty_host_array(shape, dtype, name='exotedrf-input',
                                 max_bytes=0 if spill else max_bytes, scratch_dir=scratch_dir)


def _requested_instrument(mode):
    """Extract the instrument identifier from an observing mode."""
    if mode is None:
        return None
    upper = str(mode).upper()
    for instrument in ('NIRISS', 'NIRSPEC', 'MIRI'):
        if upper.startswith(instrument):
            return instrument
    return upper.split('/')[0]


def _match_header_selection(header, key, wanted, filename):
    """Reject a populated FITS selection that differs from the requested value."""
    actual = str(header.get(key, '') or '').upper()
    if wanted and actual and wanted != actual:
        raise ValueError(f'{filename} has {key}={actual!r}, expected {wanted!r}')


def _validate_requested_header(header, *, mode=None, filter_detector=None, filename='FITS file'):
    """Confirm that a FITS segment belongs to the requested observing mode."""
    expected = _requested_instrument(mode)
    instrument = str(header.get('INSTRUME', '') or '').upper()
    if expected and instrument and instrument != expected:
        raise ValueError(f'{filename} has INSTRUME={instrument!r}, expected {expected!r}')
    if expected == 'NIRISS' and str(mode).upper().startswith('NIRISS/SOSS'):
        exposure = str(header.get('EXP_TYPE', '') or '').upper()
        if exposure and exposure != 'NIS_SOSS':
            raise ValueError(f'{filename} has EXP_TYPE={exposure!r}, expected NIS_SOSS')
        _match_header_selection(header, 'FILTER', str(filter_detector or '').upper(), filename)
    if expected == 'NIRSPEC':
        exposure = str(header.get('EXP_TYPE', '') or '').upper()
        if exposure != 'NRS_BRIGHTOBJ':
            raise ValueError(f'{filename} has EXP_TYPE={exposure!r}, expected NRS_BRIGHTOBJ')
        _match_header_selection(header, 'DETECTOR', str(filter_detector or '').upper(), filename)
        wanted_grating = str(mode).upper().split('/', 1)
        wanted_grating = (wanted_grating[1] if len(wanted_grating) == 2 else '')
        _match_header_selection(header, 'GRATING', wanted_grating, filename)
    if expected == 'MIRI':
        exposure = str(header.get('EXP_TYPE', '') or '').upper()
        if exposure and exposure != 'MIR_LRS-SLITLESS':
            raise ValueError(f'{filename} has EXP_TYPE={exposure!r}, expected MIR_LRS-SLITLESS')


def find_segments(input_dir, filetag='uncal', mode=None, filter_detector=None):
    """Find matching exposure segments and put them in exposure order.

    Parameters
    ----------
    input_dir : str
        Directory containing the input FITS files.
    filetag : str
        Product tag to match in input filenames.
    mode : None, str
        Observing mode, such as NIRISS/SOSS.
    filter_detector : None, str
        Filter or detector selection for the observation.

    Returns
    -------
    files : list[str]
        Matching FITS segment paths in exposure order.
    """
    candidates = glob.glob(os.path.join(input_dir, f'*_{filetag}.fits'))
    selected = []
    for path in candidates:
        header = fits.getheader(path, 0)
        try:
            _validate_requested_header(header, mode=mode, filter_detector=filter_detector,
                filename=path)
        except ValueError:
            if mode is None and filter_detector is None:
                raise
            continue
        match = re.search('seg(\\d+)', os.path.basename(path), re.IGNORECASE)
        fallback = int(match.group(1)) if match else 0
        segment = int(header.get('EXSEGNUM', fallback) or fallback)
        selected.append((segment, os.path.basename(path), path))
    return [path for _, _, path in sorted(selected)]


class FitsSegmentArray:
    """Read a bounded integration range directly from segmented raw FITS."""

    def __init__(self, files, shapes, extension, dtype, *, max_host_bytes=None, scratch_dir=None):
        """Initialize access to the segmented FITS arrays.

        Parameters
        ----------
        files : list[str]
            Paths to the input FITS segments, in exposure order.
        shapes : tuple[tuple]
            Dimensions of the SCI image in each input segment.
        extension : str
            FITS image extension to read.
        dtype : dtype
            Data type of the array.
        max_host_bytes : None, int, str
            Maximum memory allowed for managed host arrays.
        scratch_dir : None, str
            Directory for temporary arrays that exceed the host memory limit.
        """
        self.files = tuple(files)
        self.shapes = tuple(shapes)
        self.extension = extension
        self.dtype = np.dtype(dtype)
        self.max_host_bytes = max_host_bytes
        self.scratch_dir = scratch_dir
        self.edges = np.cumsum([shape[0] for shape in shapes])
        self.shape = (int(self.edges[-1]), *shapes[0][1:])
        self.ndim = len(self.shape)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize

    def __array__(self, dtype=None, copy=None):
        """Reject conversion of the complete segmented FITS observation."""
        raise TypeError('Read a bounded integration slice of FitsSegmentArray')

    def __getitem__(self, key):
        """Read the selected array or integration slice."""
        key = key if isinstance(key, tuple) else (key,)
        leading, trailing = key[0], key[1:]
        if any(item is not Ellipsis and (isinstance(item, (bool, np.bool_)) or
                not isinstance(item, (slice, int, np.integer))) for item in trailing):
            raise TypeError('FITS trailing selections require slices or integers')
        squeeze = isinstance(leading, (int, np.integer))
        if squeeze:
            index = int(leading)
            if index < 0:
                index += self.shape[0]
            if not 0 <= index < self.shape[0]:
                raise IndexError('integration index out of range')
            leading = slice(index, index + 1)
        if not isinstance(leading, slice):
            raise TypeError('FITS integration selection must be a slice or int')
        lo, hi, stride = leading.indices(self.shape[0])
        if stride != 1:
            raise ValueError('FITS integration slices require unit stride')
        # Read bounded FITS sections into storage within the host budget.
        from exotedrf.v2 import core
        count = max(0, hi - lo)
        proxy = np.broadcast_to(np.empty((), self.dtype), (count, *self.shape[1:]))
        shape = proxy[(slice(None), *trailing)].shape
        result = core.empty_host_array(shape, self.dtype, name='exotedrf-fits-section',
                                       max_bytes=self.max_host_bytes, scratch_dir=self.scratch_dir)
        self._copy_into(result, lo, hi, trailing)
        return result[0] if squeeze else result

    def _copy_into(self, result, lo, hi, trailing=()):
        """Fill a caller-owned destination without another full-size array."""
        plane_bytes = int(np.prod(result.shape[1:])) * self.dtype.itemsize
        chunk_ints = max(1, (32 << 20) // max(1, plane_bytes))
        start = 0
        for file, end in zip(self.files, self.edges):
            a, b = max(lo, start), min(hi, int(end))
            if a < b:
                with fits.open(file, memmap=False) as hdus:
                    if self.extension not in hdus:
                        if self.extension != 'GROUPDQ':
                            raise ValueError(f'No {self.extension} in {file}')
                        result[a - lo:b - lo] = 0
                    else:
                        hdu = hdus[self.extension]
                        if not hdu.shape and self.extension == 'GROUPDQ':
                            result[a - lo:b - lo] = 0
                            start = int(end)
                            continue
                        for first in range(a, b, chunk_ints):
                            last = min(first + chunk_ints, b)
                            selection = (slice(first - start, last - start), *trailing)
                            if len(hdu.shape) == 3:
                                piece = np.asarray(hdu.section[trailing or (...,)])[None]
                            else:
                                piece = hdu.section[selection]
                            result[first - lo:last - lo] = piece
            start = int(end)


def _segment_header(header, first, mode, detector, filter_detector, filename):
    """Validate segment selection and consistency with the first exposure header."""
    requested_filter = filter_detector
    if requested_filter is None and str(mode).upper().startswith(
            'NIRISS/SOSS') and str(detector).upper() in ('CLEAR', 'F277W'):
        requested_filter = detector
    _validate_requested_header(
        header, mode=mode, filter_detector=requested_filter, filename=filename)
    if first is None:
        return header
    for key in ('INSTRUME', 'DETECTOR', 'SUBARRAY'):
        expected = str(first.get(key, '') or '').upper()
        current = str(header.get(key, '') or '').upper()
        if expected and current and expected != current:
            raise ValueError(f'Inconsistent {key} across segments: '
                f'{expected!r} != {current!r} in {filename}')
    return first


def _segment_times(header, times, nints, total, path, *, exposure_table=False):
    """Validate segment integration bounds and retain or select its timing rows."""
    start = int(header.get('INTSTART', total + 1) or (total + 1))
    end = int(header.get('INTEND', start + nints - 1) or (start + nints - 1))
    if end - start + 1 != nints:
        raise ValueError(f'INTSTART/INTEND in {path} describe {end - start + 1} '
            f'integrations but SCI contains {nints}')
    if times is not None and times.shape[0] != nints:
        if exposure_table and times.shape[0] >= end:
            times = times[start - 1:end]
        else:
            raise ValueError(
                f'INT_TIMES length {times.shape[0]} does not match SCI '
                f'integrations {nints} in {path}')
    return start, end, times if times is not None else np.full(nints, np.nan, dtype=np.float64)


def _observation_meta(header, mode, detector, nints, ngroups, files, baseline_ints,
                      times_parts, edges, segment_headers, int_starts, int_ends, extra):
    """Build shared observing metadata after joining and validating segment times."""
    int_times = np.concatenate(times_parts)
    if not np.isfinite(int_times).any():
        int_times = np.arange(nints, dtype=np.float64)
    elif not np.isfinite(int_times).all():
        raise ValueError('INT_TIMES is present for only a subset of segments; refusing '
            'to fabricate or misalign BJD timestamps')
    instrume = (header.get('INSTRUME', '') or '').upper()
    subarray = header.get('SUBARRAY', '')
    if mode is None:
        mode = {'NIRISS': 'NIRISS/SOSS', 'NIRSPEC': 'NIRSpec',
                'MIRI': 'MIRI/LRS'}.get(instrume, instrume)
    if detector is None:
        detector = header.get('DETECTOR', '')
    frame_time = float(header.get('TGROUP', 0.) or
                       FRAME_TIME_S.get(f'{instrume}/SOSS-{subarray}', 1.0))
    return ObsMeta(mode=mode, detector=detector, subarray=subarray,
        frame_time=frame_time, ngroups=ngroups, int_times=int_times,
        baseline_ints=np.atleast_1d(np.asarray(baseline_ints)),
        segment_edges=np.asarray(edges), filenames=tuple(files),
        extra={'header': dict(header), 'segment_headers': tuple(segment_headers),
               'segment_int_starts': tuple(int_starts),
               'segment_int_ends': tuple(int_ends), **extra})


def load_ramp_cube(files, baseline_ints, mode=None, detector=None,
                   science_dtype=np.float32, filter_detector=None,
                   max_host_bytes=None, scratch_dir=None, lazy=False):
    """Load all uncalibrated segments as one up-the-ramp observation.

    Segment boundaries, identifying headers, and integration times are retained.

    Parameters
    ----------
    files : list[str]
        Paths to the input FITS segments, in exposure order.
    baseline_ints : array-like(int)
        Out-of-transit or out-of-eclipse integration selection.
    mode, detector, filter_detector : None, str
        Observing mode, detector identifier, and filter or detector selection.
    science_dtype : dtype
        Data type of the array.
    max_host_bytes : None, int, str
        Maximum memory allowed for managed host arrays.
    scratch_dir : None, str
        Directory for temporary arrays that exceed the host memory limit.
    lazy : bool
        If True, read only the requested FITS integration slices.

    Returns
    -------
    cube : RampCube
        Joined ramp observation with segment metadata.
    """
    science_dtype = np.dtype(science_dtype)
    if science_dtype.kind != 'f':
        raise TypeError('science_dtype must be a floating dtype')
    if not files:
        raise ValueError('No input FITS segments were provided')
    shapes = _science_shapes(files)
    trailing_shape = shapes[0][1:]
    for path, shape in zip(files, shapes):
        if shape[1:] != trailing_shape:
            raise ValueError(f'Inconsistent SCI shape in {path}: {shape[1:]} != '
                f'{trailing_shape}')
    full_shape = (sum(shape[0] for shape in shapes),) + trailing_shape
    limit = host_allocation_budget_bytes(max_host_bytes)
    required = sum(int(np.prod(shape)) * (science_dtype.itemsize + np.dtype(np.uint8).itemsize)
        for shape in shapes)
    spill = required > limit
    scratch_dir = (scratch_dir or os.environ.get('EXOTEDRF_SCRATCH_DIR') or tempfile.gettempdir())
    if lazy:
        data = FitsSegmentArray(files, shapes, 'SCI', science_dtype, max_host_bytes=max_host_bytes,
                                scratch_dir=scratch_dir)
        groupdq_all = FitsSegmentArray(files, shapes, 'GROUPDQ', np.uint8,
                                       max_host_bytes=max_host_bytes, scratch_dir=scratch_dir)
    else:
        data = _allocate_host_array(
            full_shape, science_dtype, spill=spill, scratch_dir=scratch_dir,
            max_bytes=max_host_bytes)
        groupdq_all = _allocate_host_array(
            full_shape, np.uint8, spill=spill, scratch_dir=scratch_dir, max_bytes=max_host_bytes)
    combined_pixeldq = None
    times_parts, edges, segment_headers, int_starts, int_ends = [], [], [], [], []
    hdr0 = None
    total = 0
    for f in files:
        # Read scaled detector integers at the requested calibration precision.
        with fits.open(f, memmap=False) as hdu:
            header = hdu[0].header.copy()
            hdr0 = _segment_header(header, hdr0, mode, detector, filter_detector, f)
            segment_headers.append(dict(header))
            sci = groupdq = pixeldq = None
            times = None
            for h in hdu[1:]:
                name = (h.header.get('EXTNAME', '') or '').upper()
                if name == 'SCI':
                    sci = np.broadcast_to(np.zeros((), science_dtype), h.shape)
                elif name == 'GROUPDQ' and h.shape:
                    groupdq = np.broadcast_to(np.zeros((), np.uint8), h.shape)
                elif name in ('PIXELDQ', 'DQ') and h.data is not None and \
                        np.asarray(h.data).ndim == 2:
                    pixeldq = np.asarray(h.data, dtype=np.uint32)
                elif name == 'INT_TIMES' and h.data is not None:
                    try:
                        times = np.asarray(h.data['int_mid_BJD_TDB'], dtype=np.float64)
                    except KeyError:
                        times = None
            if sci is None:
                raise ValueError(f'No SCI extension in {f}')
            if sci.ndim == 3:
                sci = sci[None]
            if sci.ndim != 4:
                raise ValueError(f'SCI in {f} must be 4D (nint, ngroup, y, x), got {sci.shape}')
            if groupdq is not None and groupdq.ndim == 3 and sci.shape[0] == 1:
                groupdq = groupdq[None]
            if groupdq is not None and groupdq.shape != sci.shape:
                raise ValueError(
                    f'GROUPDQ shape {groupdq.shape} does not match SCI {sci.shape} in {f}')
            spatial_shape = sci.shape[-2:]
            if pixeldq is None:
                pixeldq = np.zeros(spatial_shape, np.uint32)
            if pixeldq.shape != spatial_shape:
                raise ValueError(
                    f'PIXELDQ shape {pixeldq.shape} does not match SCI {spatial_shape} in {f}')
            if combined_pixeldq is None:
                combined_pixeldq = pixeldq.copy()
            else:
                np.bitwise_or(combined_pixeldq, pixeldq, out=combined_pixeldq)
            segment_nints = sci.shape[0]
            stop = total + segment_nints
            int_start, int_end, times = _segment_times(header, times, segment_nints, total, f)
            int_starts.append(int_start)
            int_ends.append(int_end)
            total = stop
            edges.append(total)
            times_parts.append(times)
    if not lazy:
        FitsSegmentArray(files, shapes, 'SCI', science_dtype)._copy_into(data, 0, full_shape[0])
        FitsSegmentArray(files, shapes, 'GROUPDQ', np.uint8)._copy_into(
            groupdq_all, 0, full_shape[0])
    meta = _observation_meta(
        hdr0, mode, detector, data.shape[0], data.shape[1], files, baseline_ints,
        times_parts, edges, segment_headers, int_starts, int_ends,
        {'host_storage': ('fits' if lazy else 'memmap' if any(isinstance(value, np.memmap)
                                          for value in (data, groupdq_all)) else 'memory')})
    return RampCube(data=data, groupdq=groupdq_all, pixeldq=combined_pixeldq, meta=meta)


def _stream_image(path, data, name, chunk_bytes, *, dtype, units=None):
    """Append one image with bounded conversion/byte-order buffers."""
    dtype = np.dtype(dtype)
    is_dq = dtype == np.dtype(np.uint32)
    shape = tuple(data.shape)
    header = fits.Header([('XTENSION', 'IMAGE'), ('BITPIX', 32 if is_dq else -8 * dtype.itemsize),
        ('NAXIS', len(shape)), *[(f'NAXIS{i}', length)
          for i, length in enumerate(reversed(shape), start=1)],
        ('PCOUNT', 0), ('GCOUNT', 1), ('EXTNAME', name.upper()),])
    if units is not None:
        header['UNITS'] = units
    if is_dq:
        # Encode unsigned DQ with the FITS signed-integer offset.
        header['BSCALE'] = 1
        header['BZERO'] = 1 << 31
    plane_bytes = int(np.prod(shape[1:])) * dtype.itemsize
    chunk_ints = max(1, int(chunk_bytes) // max(1, plane_bytes))
    with fits.StreamingHDU(path, header) as stream:
        if stream.writecomplete:
            return
        for start in range(0, shape[0], chunk_ints):
            piece = data[start:start + chunk_ints]
            if is_dq:
                # Flip the sign bit to preserve all unsigned DQ values.
                unsigned = np.array(piece, dtype=np.uint32, copy=True)
                unsigned ^= np.uint32(1 << 31)
                encoded = unsigned.view(np.int32).astype('>i4')
            else:
                encoded = np.asarray(piece, dtype=dtype.newbyteorder('>'))
            stream.write(encoded)


def _stream_rate_image(path, data, name, chunk_bytes):
    """Append one calibrated SCI, ERR, or DQ image to FITS."""
    _stream_image(path, data, name, chunk_bytes, dtype=np.uint32 if name == 'DQ' else np.float32)


def save_rate_cube(rate, path, extra_header=None, *, chunk_bytes=32 << 20):
    """Atomically write a rate FITS, converting at most one chunk at a time.

    Write through a temporary file and replace the destination only after completion.

    Parameters
    ----------
    rate : RateCube
        Calibrated count rates, uncertainties, and quality flags.
    path : str
        Path to the input or output file.
    extra_header : None, dict
        Additional primary header keywords to write.
    chunk_bytes : int
        Maximum bytes to convert or transfer at a time.
    """
    if chunk_bytes < 1:
        raise ValueError('chunk_bytes must be positive')
    shape = tuple(rate.data.shape)
    if (len(shape) != 3 or tuple(rate.err.shape) != shape or tuple(rate.dq.shape) != shape):
        raise ValueError('rate SCI, ERR and DQ must have matching 3D shapes')
    if np.shape(rate.meta.int_times) != (shape[0],):
        raise ValueError('rate integration times must match the image time axis')
    ph = fits.PrimaryHDU()
    ph.header['PIPELINE'] = 'exoTEDRF v2'
    ph.header['OBSMODE'] = rate.meta.mode
    for k, v in (extra_header or {}).items():
        ph.header[k] = v
    cols = fits.ColDefs([fits.Column(
        name='int_mid_BJD_TDB', format='D', array=rate.meta.int_times)])
    # Replace the output path only after the FITS file is complete.
    path = os.fspath(path)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(os.path.abspath(path)),
            prefix='.rate-', suffix='.fits', delete=False) as temporary:
        temporary_path = temporary.name
    try:
        ph.writeto(temporary_path, overwrite=True)
        for name, data in (('SCI', rate.data), ('ERR', rate.err), ('DQ', rate.dq)):
            _stream_rate_image(temporary_path, data, name, chunk_bytes)
        with fits.open(temporary_path, mode='append', memmap=True,
                       do_not_scale_image_data=True) as hdus:
            hdus.append(fits.BinTableHDU.from_columns(cols, name='INT_TIMES'))
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def save_spectra(path, orders, meta, extra_header=None, *, chunk_bytes=32 << 20):
    """Write extracted wavelength, flux, and uncertainty arrays to FITS.

    Parameters
    ----------
    path : str
        Path to the input or output file.
    orders : dict
        Wavelength, flux, uncertainty, and optional DQ arrays by spectral order.
    meta : ObsMeta
        Observing metadata and integration times.
    extra_header : None, dict
        Additional primary header keywords to write.
    chunk_bytes : int
        Maximum bytes to convert or transfer at a time.
    """
    if chunk_bytes < 1:
        raise ValueError('chunk_bytes must be positive')
    source_header = (getattr(meta, 'extra', {}) or {}).get('header', {})
    target = next((source_header.get(key) for key in ('TARGET', 'TARGNAME', 'OBJECT')
                   if source_header.get(key) not in (None, '')), '')
    instrument = (getattr(meta, 'mode', '') or source_header.get('INSTRUME', '') or '')
    ph = fits.PrimaryHDU()
    # Retain the v1 metadata consumed by downstream light-curve tools.
    ph.header['TARGET'] = str(target)
    ph.header['INST'] = str(instrument)
    ph.header['PIPELINE'] = 'exoTEDRF v2'
    ph.header['CONTENTS'] = 'Full resolution stellar spectra'
    ph.header['METHOD'] = ''
    ph.header['WIDTH'] = ''
    ph.header['OBSMODE'] = getattr(meta, 'mode', '')
    for k, v in (extra_header or {}).items():
        ph.header[k] = v
    images = []
    for label, d in orders.items():
        suf = str(label).replace(' ', '').upper()
        # Use suffix-free spectrum extension names for NIRSpec and MIRI.
        suffix = f' {suf}' if suf else ''
        wave = np.asarray(d['wave'], np.float64)
        if wave.size >= 2:
            edges = np.empty(wave.size + 1, dtype=np.float64)
            edges[0] = wave[0] - (wave[1] - wave[0]) / 2
            edges[-1] = wave[-1] + (wave[-1] - wave[-2]) / 2
            edges[1:-1] = (wave[1:] + wave[:-1]) / 2
            widths = np.empty_like(wave)
            widths[:-1] = edges[1:-1] - edges[:-2]
            widths[-1] = wave[-1] - wave[-2]
            wave_error = np.abs(widths) / 2
        else:
            wave_error = np.full_like(wave, np.nan)
        images.extend([(f'Wave{suffix}', wave, np.float64, 'Micron'),
            (f'Wave Err{suffix}', wave_error, np.float64, 'Micron'),
            (f'Flux{suffix}', d['flux'], np.float32, 'DN/s'),
            (f'Flux Err{suffix}', d['ferr'], np.float32, 'DN/s')])
    images.append(('Time', np.asarray(meta.int_times), np.float64, 'MJD_TDB'))
    # Write DQ reports after Time, using -1 when unavailable.
    for label, d in orders.items():
        suf = str(label).replace(' ', '').upper()
        suffix = f' {suf}' if suf else ''
        report = d.get('dq')
        if report is None:
            report = -np.ones(np.asarray(d['wave']).shape[0])
        images.append((f'DQ Report{suffix}', np.asarray(report), np.float32,
                       '1-DNU, 2-SAT, 4-HOT, 8-VAR'))
    path = os.fspath(path)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(os.path.abspath(path)),
            prefix='.spectra-', suffix='.fits', delete=False) as temporary:
        temporary_path = temporary.name
    try:
        ph.writeto(temporary_path, overwrite=True)
        for name, data, dtype, units in images:
            _stream_image(temporary_path, data, name, chunk_bytes, dtype=dtype, units=units)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def find_products(input_dir, filetag, mode=None, filter_detector=None):
    """Find v1-style segment products whose name contains ``filetag``.

    Match filetag against the filename and order matching products by EXSEGNUM.

    Parameters
    ----------
    input_dir : str
        Directory containing the input FITS files.
    filetag : str
        Product tag to match in input filenames.
    mode : None, str
        Observing mode, such as NIRISS/SOSS.
    filter_detector : None, str
        Filter or detector selection for the observation.

    Returns
    -------
    files : list[str]
        Matching FITS segment paths in exposure order.
    """
    selected = []
    for path in glob.glob(os.path.join(os.fspath(input_dir), '*')):
        if str(filetag) not in os.path.basename(path) or \
                not os.path.isfile(path):
            continue
        try:
            header = fits.getheader(path, 0)
        except (OSError, ValueError):
            continue
        try:
            _validate_requested_header(header, mode=mode, filter_detector=filter_detector,
                filename=path)
        except ValueError:
            continue
        if mode is not None and not header.get('INSTRUME'):
            continue
        match = re.search('seg(\\d+)', os.path.basename(path), re.IGNORECASE)
        fallback = int(match.group(1)) if match else 0
        segment = int(header.get('EXSEGNUM', fallback) or fallback)
        selected.append((segment, os.path.basename(path), path))
    return [path for _, _, path in sorted(selected)]


def product_ndim(path):
    """Read the dimension count of the FITS SCI product without loading its pixels."""
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
        try:
            return len(hdul['SCI'].shape)
        except KeyError:
            raise ValueError(f'No SCI extension in {path}') from None


def read_wavelength_plane(path):
    """Read the FITS WAVELENGTH plane as float64, or return None if it is absent."""
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul[1:]:
            name = (hdu.header.get('EXTNAME', '') or '').upper()
            if name == 'WAVELENGTH' and hdu.data is not None:
                wave = np.asarray(hdu.data, dtype=np.float64)
                if wave.ndim == 2 and np.isfinite(wave).any():
                    return wave
    return None


def _rate_product_shape(path):
    """Read and validate the shape of a calibrated SCI extension."""
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
        try:
            shape = tuple(hdul['SCI'].shape)
        except KeyError:
            raise ValueError(f'No SCI extension in {path}') from None
    if len(shape) == 2:
        shape = (1,) + shape
    if len(shape) != 3:
        raise ValueError(f'SCI in {path} must be 3D (nint, y, x) for a post-RampFit '
            f'product, got {shape}')
    return shape


def load_rate_products(files, baseline_ints, mode=None, detector=None,
                       science_dtype=np.float32, filter_detector=None,
                       max_host_bytes=None, scratch_dir=None):
    """Load post-RampFit per-segment products as one rate observation.

    Parameters
    ----------
    files : list[str]
        Paths to the input FITS segments, in exposure order.
    baseline_ints : array-like(int)
        Out-of-transit or out-of-eclipse integration selection.
    mode, detector, filter_detector : None, str
        Observing mode, detector identifier, and filter or detector selection.
    science_dtype : dtype
        Data type of the array.
    max_host_bytes : None, int, str
        Maximum memory allowed for managed host arrays.
    scratch_dir : None, str
        Directory for temporary arrays that exceed the host memory limit.

    Returns
    -------
    cube : RateCube
        Joined rate observation with segment metadata.
    """
    from exotedrf.v2.core import RateCube
    science_dtype = np.dtype(science_dtype)
    if science_dtype.kind != 'f':
        raise TypeError('science_dtype must be a floating dtype')
    if not files:
        raise ValueError('No input FITS products were provided')
    files = [os.fspath(path) for path in files]
    shapes = [_rate_product_shape(path) for path in files]
    spatial = shapes[0][1:]
    for path, shape in zip(files, shapes):
        if shape[1:] != spatial:
            raise ValueError(f'Inconsistent SCI shape in {path}: {shape[1:]} != {spatial}')
    full_shape = (sum(shape[0] for shape in shapes),) + spatial
    scratch_dir = (scratch_dir or os.environ.get('EXOTEDRF_SCRATCH_DIR') or tempfile.gettempdir())
    from exotedrf.v2 import core
    data, err, dq = (core.empty_host_array(full_shape, dtype, name=f'exotedrf-product-{name}',
                              max_bytes=max_host_bytes, scratch_dir=scratch_dir)
        for name, dtype in (('sci', science_dtype), ('err', science_dtype), ('dq', np.uint32)))
    hdr0 = None
    segment_headers, int_starts, int_ends, edges, times_parts = \
        [], [], [], [], []
    total = 0
    for path, shape in zip(files, shapes):
        # Read scaled FITS sections without loading complete images.
        with fits.open(path, memmap=False) as hdul:
            header = hdul[0].header.copy()
            hdr0 = _segment_header(header, hdr0, mode, detector, filter_detector, path)
            segment_headers.append(dict(header))
            nint = shape[0]
            names = {(hdu.header.get('EXTNAME', '') or '').upper(): hdu for hdu in hdul[1:]}
            chunk = max(1, (32 << 20) // max(1, int(np.prod(spatial)) * 4))

            def read(name, first, last):
                """Read the selected image extension with FITS scaling applied."""
                hdu = names.get(name)
                if hdu is None or not hdu.shape:
                    return None
                if len(hdu.shape) == 2:
                    return np.asarray(hdu.section[...])[None]
                return np.asarray(hdu.section[first:last])
            for first in range(0, nint, chunk):
                last = min(first + chunk, nint)
                out = slice(total + first, total + last)
                data[out] = read('SCI', first, last).astype(science_dtype, copy=False)
                piece = read('ERR', first, last)
                err[out] = 0 if piece is None else piece
                piece = read('DQ', first, last)
                dq[out] = 0 if piece is None else piece.astype(np.uint32)
            times = None
            table = names.get('INT_TIMES')
            if table is not None and table.data is not None:
                try:
                    times = np.asarray(table.data['int_mid_BJD_TDB'], dtype=np.float64)
                except KeyError:
                    times = None
        int_start, int_end, times = _segment_times(
            header, times, nint, total, path, exposure_table=True)
        times_parts.append(times)
        int_starts.append(int_start)
        int_ends.append(int_end)
        total += nint
        edges.append(total)
    meta = _observation_meta(hdr0, mode, detector, full_shape[0], int(hdr0.get('NGROUPS', 0) or 0),
        files, baseline_ints, times_parts, edges, segment_headers, int_starts, int_ends,
        {'input_products': tuple(files),
         'host_storage': ('memmap' if any(isinstance(value, np.memmap)
                                        for value in (data, err, dq)) else 'memory')})
    return RateCube(data=data, err=err, dq=dq, meta=meta)


def v1_fileroots(files):
    """Get segment filename roots using v1 get_filename_root.

    Parameters
    ----------
    files : list[str]
        Paths to the input FITS segments, in exposure order.

    Returns
    -------
    roots : list[str]
        Segment filename roots in exposure order.
    """
    files = [os.fspath(path) for path in files]
    if not files:
        return []
    header = fits.getheader(files[0], 0)
    filename = header.get('FILENAME') or os.path.basename(files[0])
    chunks = str(filename).split('/')[-1].split('_')
    root = ''.join(chunk + '_' for chunk in chunks[:-1])
    seg_start = header.get('EXSEGNUM')
    if seg_start is None or 'seg' not in root:
        roots = []
        for path in files:
            name = os.path.basename(path).split('_')
            roots.append(''.join(chunk + '_' for chunk in name[:-1]))
        return roots
    roots = [root]
    head, tail = root.split('seg', 1)
    for segment in range(int(seg_start) + 1, int(seg_start) + len(files)):
        roots.append(f'{head}seg{segment:03d}{tail[3:]}')
    return roots


def v1_fileroot_noseg(fileroots):
    """Remove the segment token using v1 get_filename_root_noseg."""
    working = fileroots[0]
    if 'seg' in working:
        part1, part2 = working.split('seg', 1)
        return part1[:-1] + part2[3:]
    return working
