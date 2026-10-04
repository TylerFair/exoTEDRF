"""Build JWST Stage 2 datamodels from calibrated observation arrays."""

from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

# Let Astropy regenerate FITS structure keywords for the new extensions.
_STRUCTURAL_KEYS = frozenset(('SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'NAXIS4',
    'EXTEND', 'PCOUNT', 'GCOUNT', 'XTENSION', 'EXTNAME', 'EXTVER', 'BZERO',
    'BSCALE', 'CHECKSUM', 'DATASUM'))
_COMMENTARY_KEYS = frozenset(('', 'COMMENT', 'HISTORY'))

INT_TIMES_COLUMNS = (('integration_number', '>i4'), ('int_start_MJD_UTC', '>f8'),
    ('int_mid_MJD_UTC', '>f8'), ('int_end_MJD_UTC', '>f8'), ('int_start_BJD_TDB', '>f8'),
    ('int_mid_BJD_TDB', '>f8'), ('int_end_BJD_TDB', '>f8'),)

def header_from_mapping(mapping):
    """Return a FITS header holding the keyword values of ``mapping``.

    Parameters
    ----------
    mapping : dict, fits.Header
        Original FITS primary header keywords.

    Returns
    -------
    header : fits.Header
        Retained exposure keywords without FITS structure or commentary cards.
    """
    header = fits.Header()
    if mapping is None:
        return header
    if isinstance(mapping, fits.Header):
        items = [(card.keyword, card.value, card.comment) for card in mapping.cards]
    else:
        items = [(key, value, None) for key, value in dict(mapping).items()]
    for key, value, comment in items:
        key = str(key)
        if key.upper() in _STRUCTURAL_KEYS or key.upper() in _COMMENTARY_KEYS:
            continue
        if value is None or isinstance(value, fits.card.Undefined):
            value = fits.card.UNDEFINED
        elif not isinstance(value, (str, bool, int, float, complex, np.generic)):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        try:
            header[key] = (value, comment) if comment else value
        except (ValueError, TypeError, fits.VerifyError):
            continue
    return header


def segment_header(meta, segment):
    """Return the retained primary header for a zero-based segment, falling back to the first."""
    extra = getattr(meta, 'extra', {}) or {}
    headers = tuple(extra.get('segment_headers') or ())
    source = (headers[segment] if segment < len(headers) else extra.get('header'))
    return header_from_mapping(source)


def segment_bounds(meta, nints):
    """Return core.segment_slices for the supplied metadata and integration count."""
    from exotedrf.v2 import core
    return core.segment_slices(meta, nints)


def _original_int_times(meta, segment):
    """Read the full INT_TIMES table of a segment's original FITS file."""
    filenames = tuple(getattr(meta, 'filenames', ()) or ())
    if segment >= len(filenames):
        return None
    path = os.fspath(filenames[segment])
    if not os.path.exists(path):
        return None
    try:
        with fits.open(path, memmap=True) as hdul:
            if 'INT_TIMES' not in hdul:
                return None
            return np.array(hdul['INT_TIMES'].data)
    except OSError:
        return None


def segment_int_times(meta, segment, seg_slice, int_start=None):
    """Return the segment's JWST ``INT_TIMES`` rows as a record array.

    Use the original timing table when available; otherwise retain BJD midpoints and mark other
    times NaN.

    Parameters
    ----------
    meta : ObsMeta
        Observing metadata and integration times.
    segment : int
        Zero-based segment index.
    seg_slice : slice
        Integration range of the segment in the joined observation.
    int_start : None, int
        One-based integration number of the first selected integration.

    Returns
    -------
    table : np.recarray
        Segment integration numbers and JWST timing columns.
    """
    times = np.asarray(meta.int_times, dtype=np.float64)[seg_slice]
    nints = int(times.shape[0])
    if int_start is None:
        starts = getattr(meta, 'segment_int_starts', None)
        int_start = int(np.asarray(starts)[segment]) if starts is not None \
            else int(seg_slice.start) + 1
    numbers = np.arange(int_start, int_start + nints, dtype=np.int32)
    original = _original_int_times(meta, segment)
    if original is not None and 'integration_number' in original.dtype.names:
        lookup = {int(n): i for i, n in enumerate(np.asarray(original['integration_number']))}
        rows = [lookup.get(int(n)) for n in numbers]
        if all(row is not None for row in rows):
            table = np.zeros(nints, dtype=list(INT_TIMES_COLUMNS))
            for name, _ in INT_TIMES_COLUMNS:
                if name in original.dtype.names:
                    table[name] = np.asarray(original[name])[rows]
                else:
                    table[name] = np.nan
            return table.view(np.recarray)
    table = np.zeros(nints, dtype=list(INT_TIMES_COLUMNS))
    for name, _ in INT_TIMES_COLUMNS[1:]:
        table[name] = np.nan
    table['integration_number'] = numbers
    table['int_mid_BJD_TDB'] = times
    return table.view(np.recarray)


def stage2_hdulist(data, err, dq, primary_header, *, int_times=None,
                   wavelength=None, int_start=None, bunit='DN/s'):
    """Assemble an in-memory Stage-2 HDUList with v1's extension layout.

    Parameters
    ----------
    data, err : array-like(float)
        Science rate cube and uncertainties with integration, row, and column axes.
    dq : array-like(int)
        Detector quality flags.
    primary_header : dict, fits.Header
        Original FITS primary header keywords.
    int_times : None, np.ndarray
        JWST integration timing record array.
    wavelength : None, array-like(float)
        Detector wavelength plane in microns.
    int_start : None, int
        One-based integration number of the first selected integration.
    bunit : str
        Units for science data and uncertainty extensions.

    Returns
    -------
    hdul : fits.HDUList
        Science, uncertainty, DQ, and optional wavelength and timing extensions.
    """
    data = np.asarray(data, dtype=np.float32)
    err = np.asarray(err, dtype=np.float32)
    dq = np.asarray(dq, dtype=np.uint32)
    if data.ndim != 3 or err.shape != data.shape or dq.shape != data.shape:
        raise ValueError('SCI, ERR and DQ must share one (nints, dimy, dimx) shape; got '
            f'{data.shape}, {err.shape}, {dq.shape}')
    primary = fits.PrimaryHDU(header=header_from_mapping(primary_header))
    if int_start is not None:
        primary.header['INTSTART'] = int(int_start)
        primary.header['INTEND'] = int(int_start) + data.shape[0] - 1
    hdus = [primary]
    for name, array in (('SCI', data), ('ERR', err), ('DQ', dq)):
        hdu = fits.ImageHDU(array, name=name)
        if name in ('SCI', 'ERR') and bunit:
            hdu.header['BUNIT'] = bunit
        hdus.append(hdu)
    if wavelength is not None:
        wavelength = np.asarray(wavelength, dtype=np.float32)
        if wavelength.shape != data.shape[1:]:
            raise ValueError(f'wavelength shape {wavelength.shape} does not match the '
                f'detector shape {data.shape[1:]}')
        hdus.append(fits.ImageHDU(wavelength, name='WAVELENGTH'))
    if int_times is not None:
        hdus.append(fits.BinTableHDU(np.asarray(int_times), name='INT_TIMES'))
    return fits.HDUList(hdus)


def model_class_name(instrument):
    """Select SlitModel for NIRSpec or CubeModel for other instruments."""
    return 'SlitModel' if str(instrument).upper() == 'NIRSPEC' else \
        'CubeModel'


def stage2_datamodel(data, err, dq, primary_header, *, int_times=None,
                     wavelength=None, int_start=None, model_type='CubeModel'):
    """Build the jwst datamodel v1 would hold for this segment (in memory).

    Parameters
    ----------
    data, err : array-like(float)
        Science rate cube and uncertainties with integration, row, and column axes.
    dq : array-like(int)
        Detector quality flags.
    primary_header : dict, fits.Header
        Original FITS primary header keywords.
    int_times : None, np.ndarray
        JWST integration timing record array.
    wavelength : None, array-like(float)
        Detector wavelength plane in microns.
    int_start : None, int
        One-based integration number of the first selected integration.
    model_type : str
        JWST datamodel class to construct.

    Returns
    -------
    model : datamodel
        JWST Stage 2 model for the selected arrays or segment.
    """
    from stdatamodels.jwst import datamodels
    hdul = stage2_hdulist(data, err, dq, primary_header, int_times=int_times,
                          wavelength=wavelength, int_start=int_start)
    return getattr(datamodels, model_type)(hdul)


def _host(piece):
    """Convert a NumPy or JAX array to a NumPy array."""
    try:
        import jax
        if isinstance(piece, jax.Array):
            return np.asarray(jax.device_get(piece))
    except ImportError:  # pragma: no cover
        pass
    return np.asarray(piece)


def segment_wavelength(ctx, instrument):
    """Return the supplied float32 NIRSpec/MIRI wavelength plane, or None if absent."""
    key = {'NIRSPEC': 'nirspec_wave_map', 'MIRI': 'miri_wave_map'}.get(str(instrument).upper())
    if key is None:
        return None
    wave = (ctx or {}).get(key)
    return None if wave is None else np.asarray(wave, dtype=np.float32)


def state_segment_model(state, segment, ctx=None, *, start=None, stop=None):
    """Return the jwst Stage-2 datamodel for one segment of a rate state.

    Parameters
    ----------
    state : PipelineState
        Observation arrays and accumulated calibration results.
    segment : int
        Zero-based segment index.
    ctx : dict
        Reduction context containing options and prepared calibration data.
    start, stop : None, int
        First and exclusive final integration within the segment; defaults to all.

    Returns
    -------
    model : datamodel
        JWST Stage 2 model for the selected arrays or segment.
    """
    cube = state.cube
    meta = cube.meta
    nints = int(cube.data.shape[0])
    seg_slice = segment_bounds(meta, nints)[segment]
    lo = seg_slice.start + (0 if start is None else int(start))
    hi = seg_slice.stop if stop is None else seg_slice.start + int(stop)
    if not seg_slice.start <= lo < hi <= seg_slice.stop:
        raise ValueError(f'invalid integration range [{lo}, {hi}) for '
                         f'segment {segment} {seg_slice}')
    starts = getattr(meta, 'segment_int_starts', None)
    seg_int_start = (int(np.asarray(starts)[segment]) if starts is not None
                     else seg_slice.start + 1)
    int_times = segment_int_times(meta, segment, seg_slice, int_start=seg_int_start)
    first = lo - seg_slice.start
    int_times = int_times[first:first + (hi - lo)]
    instrument = str(getattr(meta, 'mode', '')).split('/')[0].upper()
    return stage2_datamodel(_host(cube.data[lo:hi]), _host(cube.err[lo:hi]),
        _host(cube.dq[lo:hi]), segment_header(meta, segment),
        int_times=int_times, wavelength=segment_wavelength(ctx, instrument),
        int_start=seg_int_start + first, model_type=model_class_name(instrument))


__all__ = ['INT_TIMES_COLUMNS', 'header_from_mapping', 'model_class_name',
    'segment_bounds', 'segment_header', 'segment_int_times',
    'stage2_datamodel', 'stage2_hdulist', 'state_segment_model',]
