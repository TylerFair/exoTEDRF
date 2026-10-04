"""Build, load, and validate calibration reference packs."""

import argparse
from contextlib import contextmanager
import hashlib
import os
import pathlib
import re
import urllib.request
import warnings

import numpy as np
from astropy.io import fits

REFTYPES = ['superbias', 'linearity', 'mask', 'readnoise', 'gain', 'dark', 'flat']
NIRSPEC_REFTYPES = ['superbias', 'linearity', 'mask', 'readnoise', 'gain', 'dark']
MIRI_REFTYPES = ['linearity', 'mask', 'readnoise', 'gain', 'dark', 'flat', 'emicorr']
REFPACK_SCHEMA_VERSION = 7

DEFAULT_INL_PERIODS = np.asarray([1024 / 3, 1024 / 2, 1024], np.float32)
DEFAULT_INL_URL = ('https://raw.githubusercontent.com/shashankdholakia/'
    'niriss-cal-inl/main/fourier_series_amplitudes.npy')

_REF_EXT = {'superbias': 'SCI', 'linearity': 'COEFFS', 'mask': 'DQ',
            'saturation': 'SCI', 'readnoise': 'SCI', 'gain': 'SCI',
            'dark': 'SCI', 'flat': 'SCI', 'reset': 'SCI'}
_REF_DTYPE = {'mask': np.uint32}
_REF_COMPANIONS = {'superbias': (('DQ', 'superbias_dq', np.uint32),),
    'linearity': (('DQ', 'lin_dq', np.uint32),), 'dark': (('DQ', 'dark_dq', np.uint32),
             ('AVDRKCUR', 'average_dark_current', np.float32)),
    'flat': (('DQ', 'flat_dq', np.uint32), ('ERR', 'flat_err', np.float32)),
    'reset': (('DQ', 'reset_dq', np.uint32),),}


def _reference_dq_array(hdu, dq_def_hdu=None):
    """Translate a reference-file DQ image into standard JWST flag values."""
    raw = np.asarray(hdu.data)
    original_bzero = hdu.header.get('O_BZERO')
    if original_bzero is not None and 'BZERO' not in hdu.header:
        raw = (raw.astype(np.int64) + int(original_bzero)).astype(np.uint32)
    else:
        raw = raw.astype(np.uint32)
    if dq_def_hdu is None or dq_def_hdu.data is None:
        return raw
    definitions = np.asarray(dq_def_hdu.data)
    if definitions.ndim == 0 or definitions.size == 0:
        return raw
    from stdatamodels.jwst.datamodels.dqflags import pixel
    fields = {str(name).upper(): name for name in (definitions.dtype.names or ())}
    name_field = fields.get('NAME', fields.get('MNEMONIC'))
    value_field = fields.get('VALUE')
    if name_field is None or value_field is None:
        raise ValueError('reference DQ_DEF must contain NAME/MNEMONIC and VALUE columns')
    mapped = np.zeros(raw.shape, dtype=np.uint32)
    for record in definitions:
        name = record[name_field]
        if isinstance(name, bytes):
            name = name.decode('ascii', errors='ignore')
        standard_value = pixel.get(str(name).strip())
        if standard_value is None:
            continue
        local_value = np.uint32(record[value_field])
        mapped[(raw & local_value) != 0] |= np.uint32(standard_value)
    return mapped


def _reference_dq_from_hdul(hdul, ext_name='DQ'):
    """Read a reference DQ image and convert its flags to JWST meanings."""
    dq_def = hdul['DQ_DEF'] if 'DQ_DEF' in hdul else None
    return _reference_dq_array(hdul[ext_name], dq_def)


def _extrapolate_dark(arr, extra_groups):
    """Extend a dark using the difference between its final two groups."""
    rate = arr[..., -1, :, :] - arr[..., -2, :, :]
    offsets = np.arange(1, extra_groups + 1).reshape((-1, 1, 1))
    tail = offsets * rate[..., None, :, :] + arr[..., -1:, :, :]
    return np.concatenate((arr, tail), axis=-3)


def _dark_readout(exposure_header, dark_header, *, miri=False):
    """Validate and return science and reference group readout dimensions."""
    sci_ngroups = int(exposure_header.get('NGROUPS', 0) or 0)
    if not sci_ngroups:
        sci_ngroups = int(exposure_header.get('NAXIS3', 0) or 0)
    if sci_ngroups < 1:
        raise ValueError('science exposure must provide a positive NGROUPS')
    sci_nints = int(exposure_header.get('NINTS', 0) or 0) if miri else 0
    if (exposure_header.get('NFRAMES') is None or exposure_header.get('GROUPGAP') is None):
        raise ValueError('science exposure header must provide NFRAMES and GROUPGAP')
    sci_nframes = int(exposure_header['NFRAMES'])
    sci_groupgap = int(exposure_header['GROUPGAP'])
    if dark_header is None:
        raise ValueError('dark reference metadata/header is required')
    if dark_header.get('NFRAMES') is None or dark_header.get('GROUPGAP') is None:
        raise ValueError('dark reference header must provide NFRAMES and GROUPGAP')
    dark_nframes = int(dark_header['NFRAMES'])
    dark_groupgap = int(dark_header['GROUPGAP'])
    if (sci_nframes < 1 or sci_groupgap < 0 or dark_nframes < 1 or dark_groupgap < 0):
        raise ValueError('science/dark NFRAMES must be positive and GROUPGAP non-negative')
    return sci_ngroups, sci_nframes, sci_groupgap, dark_nframes, dark_groupgap, sci_nints


def _prepare_dark(arr, readout):
    """Extrapolate dark groups and return whether subtraction must be skipped."""
    ngroups, nframes, gap, dark_nframes, dark_gap, _ = readout
    sci_frames = ngroups * nframes + (ngroups - 1) * gap
    dark_groups = arr.shape[-3]
    dark_frames = dark_groups * dark_nframes + (dark_groups - 1) * dark_gap
    if sci_frames > dark_frames:
        extra = int(np.ceil((sci_frames - dark_frames + dark_gap) / (dark_nframes + dark_gap)))
        arr = _extrapolate_dark(arr, extra)
    skipped = dark_nframes > nframes or dark_gap > gap
    if not skipped:
        arr = arr.copy()
        arr[np.isnan(arr)] = 0.
    return arr, skipped


def _average_dark_groups(arr, readout):
    """Average a single dark integration with the science group pattern."""
    ngroups, nframes, gap, _, _, _ = readout
    groups = []
    for group in range(ngroups):
        start = group * (nframes + gap)
        groups.append(arr[start] if nframes == 1 else arr[start:start + nframes].mean(axis=0))
    return np.asarray(np.stack(groups, axis=0), np.float32)


def _adapt_dark_reference(arr, exposure_header, dark_header):
    """Prepare a NIR dark reference with the science exposure's group pattern."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f'NIRISS dark reference must be 3D (groups, y, x); got {arr.shape}')
    readout = _dark_readout(exposure_header, dark_header)
    ngroups, nframes, gap, dark_nframes, dark_gap, _ = readout
    neutral = np.zeros((ngroups,) + arr.shape[-2:], dtype=np.float32)
    arr, skipped = _prepare_dark(arr, readout)
    if skipped:
        return neutral, 'SKIPPED', 'dark_readout_exceeds_science'
    if nframes == dark_nframes and gap == dark_gap:
        return np.asarray(arr[:ngroups], np.float32), 'COMPLETE', ''
    return _average_dark_groups(arr, readout), 'COMPLETE', ''


def _collapse_miri_dark_dq(dq, spatial_shape):
    """Combine MIRI dark-reference flags into one detector pixel-DQ image."""
    if dq is None:
        return np.zeros(spatial_shape, np.uint32)
    dq = np.asarray(dq, dtype=np.uint32)
    if dq.ndim == 4:
        dq = dq[:, 0]
    if dq.ndim == 3:
        collapsed = dq[0].copy()
        for index in range(1, dq.shape[0]):
            collapsed = np.bitwise_or(collapsed, dq[index])
    elif dq.ndim == 2:
        collapsed = dq
    else:
        raise ValueError(f'MIRI dark DQ must be 2D, 3D, or 4D; got {dq.shape}')
    if collapsed.shape != tuple(spatial_shape):
        raise ValueError(f'MIRI dark DQ shape {collapsed.shape} does not match '
            f'{tuple(spatial_shape)}')
    return np.asarray(collapsed, dtype=np.uint32)


def _adapt_miri_dark_reference(arr, dq, exposure_header, dark_header):
    """Prepare an integration-dependent MIRI dark for this science exposure."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f'MIRI dark reference must be 4D (nints, groups, y, x); got {arr.shape}')
    readout = _dark_readout(exposure_header, dark_header, miri=True)
    ngroups, nframes, gap, dark_nframes, dark_gap, sci_nints = readout
    dq = _collapse_miri_dark_dq(dq, arr.shape[-2:])
    neutral = np.zeros((1, ngroups) + arr.shape[-2:], dtype=np.float32)
    neutral_dq = np.zeros(arr.shape[-2:], np.uint32)
    dark_nints = arr.shape[0]
    arr, skipped = _prepare_dark(arr, readout)
    if skipped:
        return neutral, neutral_dq, 'SKIPPED', 'dark_readout_exceeds_science'
    if nframes == dark_nframes and gap == dark_gap:
        return np.asarray(arr[:, :ngroups], np.float32), dq, 'COMPLETE', ''
    num_ints = dark_nints if sci_nints < 1 else min(dark_nints, sci_nints)
    averaged = np.zeros((dark_nints, ngroups) + arr.shape[-2:], dtype=np.float32)
    for integration in range(num_ints):
        averaged[integration] = _average_dark_groups(arr[integration], readout)
    return averaged, dq, 'COMPLETE', ''


def _resolve_emi_frequency_names(freqs, readpatt, detector, path, case_name):
    """Select the EMI frequencies appropriate to this MIRI readout."""
    from collections.abc import Mapping, Sequence

    def _candidate_keys(value):
        # Use the FAST or SLOW frequency family for the corresponding R1 readout.
        """List the specific and general EMI readout lookup keys."""
        base = str(value).upper()
        keys = [base]
        if base.endswith('R1'):
            keys.append(base[:-2])
        keys.append('ALL')
        return keys
    node = freqs
    for keys in (_candidate_keys(readpatt), _candidate_keys(detector)):
        if not isinstance(node, Mapping):
            break
        selected = next((node[key] for key in keys if key in node), None)
        if selected is None:
            if len(node) == 1:
                selected = next(iter(node.values()))
            else:
                raise ValueError(f'{path} EMI case {case_name!r} has no entry for any of '
                    f'{keys} (available: {sorted(node)})')
        node = selected
    if isinstance(node, str) or not isinstance(node, Sequence):
        raise ValueError(f'{path} EMI case {case_name!r} resolved to {type(node).__name__}'
            ', expected a list of frequency names')
    names = [str(name) for name in node]
    if not names:
        raise ValueError(f'{path} EMI case {case_name!r} lists no frequencies')
    return names


def _pack_emi_waves(frequencies, waves, rowclocks, frameclocks):
    """Pack variable-length EMI waveforms with their clock and frequency metadata."""
    lengths = np.asarray([wave.size for wave in waves], dtype=np.int64)
    padded = np.zeros((len(waves), int(lengths.max(initial=0))), dtype=np.float64)
    for index, wave in enumerate(waves):
        padded[index, :wave.size] = wave
    return {'emicorr_frequencies': np.asarray(frequencies, dtype=np.float64),
        'emicorr_reference_waves': padded, 'emicorr_reference_wave_lengths': lengths,
        'emicorr_rowclocks': np.int64(rowclocks), 'emicorr_frameclocks': np.int64(frameclocks),}


def _load_emicorr_reference(path, header):
    """Read MIRI interference frequencies and waveforms from an EMI reference."""
    import asdf
    subarray = str(header.get('SUBARRAY', '') or '').upper()
    readpatt = str(header.get('READPATT', '') or '').upper()
    detector = str(header.get('DETECTOR', '') or '').upper()
    with asdf.open(os.fspath(path)) as af:
        tree = af.tree
        cases = tree.get('subarray_cases') or {}
        frequencies_all = tree.get('frequencies') or {}
        case = None
        case_name = None
        # Prefer the specific subarray and readout reference entry.
        for candidate in (f'{subarray}_{readpatt}', subarray):
            if candidate in cases:
                case, case_name = cases[candidate], candidate
                break
        if case is None:
            raise ValueError(f'{path} has no EMI subarray case for SUBARRAY={subarray!r} '
                f'READPATT={readpatt!r} (available: {sorted(cases)})')
        names = _resolve_emi_frequency_names(
            case.get('freqs'), readpatt, detector, path, case_name)
        values, waves = [], []
        for name in names:
            entry = frequencies_all.get(name)
            if entry is None:
                raise ValueError(f'{path} EMI case {case_name!r} references unknown '
                    f'frequency {name!r}')
            values.append(float(entry['frequency']))
            waves.append(np.asarray(entry['phase_amplitudes'], dtype=np.float32))
        return dict(_pack_emi_waves(values, waves, case['rowclocks'], case['frameclocks']),
                    emicorr_frequency_names=np.asarray(names, dtype='U32'), emicorr_case=case_name)


def _extract_miri_emicorr_reference(path, header):
    """Use JWST's EMI reader to select this exposure's reference patterns."""
    try:
        from jwst import datamodels
        from jwst.emicorr.emicorr import (get_frequency_info, get_subarcase)
    except ImportError:
        # Read the ASDF reference directly when JWST is unavailable.
        return _load_emicorr_reference(path, header)
    subarray = str(header.get('SUBARRAY', '') or '').upper()
    readpatt = str(header.get('READPATT', '') or '').upper()
    detector = str(header.get('DETECTOR', '') or '').upper()
    with datamodels.EmiModel(os.fspath(path)) as model:
        _, rowclocks, frameclocks, names = get_subarcase(model, subarray, readpatt, detector)
        frequencies = []
        waves = []
        for name in names:
            frequency, wave = get_frequency_info(model, name)
            frequencies.append(float(frequency))
            waves.append(np.asarray(wave, dtype=np.float64).reshape(-1))
    return _pack_emi_waves(frequencies, waves, rowclocks, frameclocks)


def _compute_miri_wave_map(uncal_file, spatial_shape):
    """Calculate the wavelength at every MIRI/LRS detector pixel."""
    try:
        from jwst import datamodels
        from jwst.assign_wcs import AssignWcsStep
    except ImportError as exc:
        raise RuntimeError('building a MIRI refpack wavelength map requires jwst; either '
            'run the builder in the v1 CRDS/jwst environment or pass '
            '--wavemap-file with a precomputed wavelength plane') from exc
    with datamodels.open(uncal_file) as ramp:
        rate = datamodels.CubeModel(data=np.zeros((1,) + tuple(spatial_shape), dtype=np.float32))
        rate.update(ramp)
    rate.meta.exposure.type = rate.meta.exposure.type or 'MIR_LRS-SLITLESS'
    result = AssignWcsStep.call(rate, slit_y_low=-0.55, slit_y_high=0.55)
    wave = np.asarray(getattr(result, 'wavelength', None) if
                      getattr(result, 'wavelength', None) is not None else
                      np.zeros(0), dtype=np.float64)
    if wave.shape != tuple(spatial_shape) or not np.isfinite(wave).any():
        dimy, dimx = spatial_shape
        xx, yy = np.meshgrid(np.arange(dimx), np.arange(dimy))
        world = result.meta.wcs(xx, yy)
        wave = np.asarray(world[-1], dtype=np.float64)
    if wave.shape != tuple(spatial_shape):
        raise ValueError(f'jwst returned a {wave.shape} MIRI wavelength plane; expected '
            f'the {tuple(spatial_shape)} subarray. Supply --wavemap-file '
            'with a detector-frame wavelength plane instead')
    return wave


def _load_inl_calibration(amplitude_file, periods, cache_dir):
    """Load the calibrated Fourier terms for NIRISS count-rate nonlinearity."""
    periods = (DEFAULT_INL_PERIODS if periods is None else np.asarray(periods, dtype=np.float32))
    if (periods.ndim != 1 or periods.size == 0 or
            not np.isfinite(periods).all() or np.any(periods <= 0)):
        raise ValueError('INL periods must be a positive, finite, non-empty 1D array')
    if amplitude_file is None:
        cache_dir = pathlib.Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        amplitude_file = cache_dir / 'fourier_series_amplitudes.npy'
        if not amplitude_file.exists():
            try:
                urllib.request.urlretrieve(DEFAULT_INL_URL, amplitude_file)
            except Exception as exc:
                raise RuntimeError('Unable to obtain the default NIRISS INL amplitudes; '
                    'set inl_amplitude_file explicitly') from exc
    amplitude_file = pathlib.Path(amplitude_file).expanduser()
    if not amplitude_file.exists():
        raise FileNotFoundError(f'INL amplitude file not found: {amplitude_file}')
    theta = np.asarray(np.load(amplitude_file, allow_pickle=False), np.float32)
    if theta.ndim != 1 or theta.size != 2 * periods.size:
        raise ValueError(
            f'INL amplitudes must have shape ({2 * periods.size},), got {theta.shape}')
    if not np.isfinite(theta).all():
        raise ValueError('INL amplitudes must all be finite')
    return theta, periods, str(amplitude_file)


SOSS_FILES_URL = 'https://raw.githubusercontent.com/radicamc/exoTEDRF/main/files/'


def default_files_dirs():
    """Locate repository calibration files before installed copies."""
    import sys
    checkout = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), 'files')
    return checkout, os.path.join(sys.prefix, 'files')


def soss_reference_file(kind, subarray, search_dirs=None, download_dir=None):
    """Path of v1 2.5.0's SOSS ``'wavemap'`` or ``'spectrace'`` file.

    Parameters
    ----------
    kind : str
        Reference type: wavemap or spectrace.
    subarray : str
        Science subarray identifier.
    search_dirs : None, list[str]
        Directories to search for the reference file.
    download_dir : None, str
        Directory to which to download missing reference files.

    Returns
    -------
    path : None, str
        Located file path, or None when no file was selected.
    """
    small = str(subarray or '').upper() == 'SUBSTRIP96'
    name = {'wavemap': ('jwst_niriss_wavemap_0020.fits' if small else
                        'jwst_niriss_wavemap_0022.fits'),
            'spectrace': ('jwst_niriss_spectrace_0022.fits' if small else
                          'jwst_niriss_spectrace_0023.fits')}[kind]
    for directory in (search_dirs or default_files_dirs()):
        candidate = os.path.join(directory, name)
        if directory and os.path.exists(candidate):
            return candidate
    if download_dir is None:
        return None
    target = pathlib.Path(download_dir) / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(SOSS_FILES_URL + name, str(target))
        except OSError as exc:  # noqa: PERF203
            warnings.warn(f'could not download {name}: {exc}')
            return None
    return str(target)


def v1_soss_wave_vectors(wavemap_file):
    """Average SOSS wavelengths inside the padded border, as v1 does."""
    with fits.open(wavemap_file, memmap=False) as hdu:
        return {order: np.mean(hdu[order].data[20:-20, 20:-20], axis=0) for order in (1, 2)}


def _crds_parameters(header):
    """Collect the exposure properties CRDS uses to choose references."""
    mapping = {'META.INSTRUMENT.NAME': 'INSTRUME', 'META.INSTRUMENT.DETECTOR': 'DETECTOR',
        'META.INSTRUMENT.FILTER': 'FILTER', 'META.INSTRUMENT.PUPIL': 'PUPIL',
        'META.INSTRUMENT.GRATING': 'GRATING', 'META.EXPOSURE.TYPE': 'EXP_TYPE',
        'META.EXPOSURE.READPATT': 'READPATT', 'META.SUBARRAY.NAME': 'SUBARRAY',
        'META.SUBARRAY.XSTART': 'SUBSTRT1', 'META.SUBARRAY.YSTART': 'SUBSTRT2',
        'META.SUBARRAY.XSIZE': 'SUBSIZE1', 'META.SUBARRAY.YSIZE': 'SUBSIZE2',
        'META.OBSERVATION.DATE': 'DATE-OBS', 'META.OBSERVATION.TIME': 'TIME-OBS',}
    params = {}
    for meta_key, hdr_key in mapping.items():
        val = header.get(hdr_key)
        if val is not None:
            params[meta_key] = val
    return params


def _cut_to_subarray(arr, header, reference_header=None):
    """Trim a full-detector reference to the science exposure's subarray."""
    reference_header = reference_header or {}
    ys = (int(header.get('SUBSTRT2', 1)) - int(reference_header.get('SUBSTRT2', 1)))
    xs = (int(header.get('SUBSTRT1', 1)) - int(reference_header.get('SUBSTRT1', 1)))
    ny, nx = int(header.get('SUBSIZE2', 0)), int(header.get('SUBSIZE1', 0))
    if ny == 0 or nx == 0:
        return arr
    if arr.shape[-2] == ny and arr.shape[-1] == nx:
        return arr
    if (ys < 0 or xs < 0 or ys + ny > arr.shape[-2] or xs + nx > arr.shape[-1]):
        raise ValueError(f'science subarray ({ys}:{ys + ny}, {xs}:{xs + nx}) falls '
            f'outside reference shape {arr.shape[-2:]}')
    return arr[..., ys:ys + ny, xs:xs + nx]


def _load_wave_map(wavemap_file, spatial_shape, label='NIRSpec'):
    """Load a detector wavelength image supplied by the user."""
    path = os.fspath(wavemap_file)
    if path.endswith('.npy'):
        wave = np.load(path, allow_pickle=False)
    else:
        with fits.open(path, memmap=False) as hdu:
            wave = None
            for h in hdu:
                name = (h.header.get('EXTNAME', '') or '').upper()
                if name == 'WAVELENGTH' and h.data is not None:
                    wave = np.asarray(h.data)
                    break
            if wave is None:
                for h in hdu[1:]:
                    if h.data is not None and np.asarray(h.data).ndim == 2 \
                            and np.asarray(h.data).shape == spatial_shape:
                        wave = np.asarray(h.data)
                        break
            if wave is None:
                raise ValueError(f'{path} has no WAVELENGTH extension or matching 2D image')
    wave = np.squeeze(np.asarray(wave, dtype=np.float64))
    if wave.ndim != 2:
        raise ValueError(f'{label} wavelength map must be 2D, got shape {wave.shape}')
    if spatial_shape and wave.shape != tuple(spatial_shape):
        raise ValueError(f'{label} wavelength map shape {wave.shape} does not match the '
            f'exposure subarray {tuple(spatial_shape)}')
    return wave


def _load_miri_wave_map(wavemap_file, spatial_shape):
    """Load a two-dimensional MIRI detector wavelength image."""
    return _load_wave_map(wavemap_file, spatial_shape, label='MIRI')


def _compute_nirspec_wave_map(uncal_file, spatial_shape):
    """Calculate the wavelength at every NIRSpec/BOTS detector pixel."""
    try:
        from functools import partial
        from jwst import datamodels
        from jwst.assign_wcs import nirspec
        from jwst.assign_wcs import AssignWcsStep
    except ImportError as exc:
        raise RuntimeError('building a NIRSpec refpack wavelength map requires jwst; '
            'either run the builder in the v1 CRDS/jwst environment or '
            'pass --wavemap-file with a precomputed wavelength plane') from exc
    original = nirspec.generate_compound_bbox
    nirspec.generate_compound_bbox = partial(original, wavelength_range=[6e-08, 6e-06])
    try:
        with datamodels.open(uncal_file) as ramp:
            rate = datamodels.CubeModel(
                data=np.zeros((1,) + tuple(spatial_shape), dtype=np.float32))
            rate.update(ramp)
        rate.meta.exposure.type = rate.meta.exposure.type or 'NRS_BRIGHTOBJ'
        result = AssignWcsStep.call(rate, slit_y_low=-50, slit_y_high=50)
        slit_wcs = nirspec.nrs_wcs_set_input(result, 'S1600A1')
    finally:
        nirspec.generate_compound_bbox = original
    dimy, dimx = (int(n) for n in spatial_shape)
    x, y = np.meshgrid(np.arange(dimx), np.arange(dimy))
    wave = np.asarray(slit_wcs(x, y)[2], dtype=np.float64)
    if wave.shape != (dimy, dimx):
        raise ValueError(f'jwst returned a {wave.shape} wavelength plane; expected the '
            f'{(dimy, dimx)} subarray. Supply --wavemap-file with a '
            'detector-frame wavelength plane instead')
    return wave


def effective_crds_context(context=None):
    """Return CRDS_CONTEXT if set, otherwise the supplied context."""
    return os.environ.get('CRDS_CONTEXT') or context


@contextmanager
def _crds_context(context):
    """Give CRDS lookups and nested JWST steps the same temporary context."""
    previous = os.environ.get('CRDS_CONTEXT')
    selected = effective_crds_context(context)
    if selected is not None:
        os.environ['CRDS_CONTEXT'] = str(selected)
    try:
        yield selected
    finally:
        if previous is None:
            os.environ.pop('CRDS_CONTEXT', None)
        else:
            os.environ['CRDS_CONTEXT'] = previous


_TLS_BUNDLE_CANDIDATES = ('/etc/pki/tls/certs/ca-bundle.crt', '/etc/ssl/certs/ca-certificates.crt',
    '/etc/ssl/cert.pem', '/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem',)


def ensure_tls_certificates(environ=None, verify_paths=None, candidates=None):
    """Point OpenSSL at a system CA bundle when its compiled-in path is gone.

    Parameters
    ----------
    environ : None, dict
        Environment mapping; use the process environment if omitted.
    verify_paths : None, object
        OpenSSL default verification paths; queried if omitted.
    candidates : None, list[str]
        Certificate bundle paths to search in order.

    Returns
    -------
    path : None, str
        Selected certificate bundle, or None when no environment change was needed.
    """
    environ = os.environ if environ is None else environ
    if environ.get('SSL_CERT_FILE') or environ.get('SSL_CERT_DIR'):
        return None
    if verify_paths is None:
        import ssl
        verify_paths = ssl.get_default_verify_paths()
    if getattr(verify_paths, 'cafile', None) or \
            getattr(verify_paths, 'capath', None):
        return None
    if candidates is None:
        candidates = list(_TLS_BUNDLE_CANDIDATES)
        try:
            import certifi
            candidates.append(certifi.where())
        except ImportError:
            pass
    for candidate in candidates:
        path = pathlib.Path(candidate)
        if path.is_file():
            environ['SSL_CERT_FILE'] = str(path)
            return str(path)
    return None


def build_refpack(uncal_file, context=None, out=None, reftypes=None,
                  inl_amplitude_file=None, inl_periods=None,
                  wavemap_file=None, include_inl=None, apply_wavecorr=True):
    """Build references under one CRDS context, including JWST wavelengths.

    A configured CRDS cache is required when building, but not when loading a saved pack.

    Parameters
    ----------
    uncal_file : str
        Representative uncalibrated exposure supplying the observing headers.
    context : None, str
        CRDS context; CRDS_CONTEXT takes precedence if set.
    out : None, str
        Path to which to save the reference pack.
    reftypes : None, list[str]
        CRDS reference types to collect.
    inl_amplitude_file : None, str
        Path to calibrated INL Fourier amplitudes.
    inl_periods : None, array-like(float)
        Detector-count periods for the INL correction.
    wavemap_file : None, str
        Supplied wavelength reference; derive or locate one if omitted.
    include_inl : None, bool
        If True, include INL calibration; defaults to True for NIRISS.
    apply_wavecorr : bool
        Retained call option; the NIRSpec wavelength solution omits WaveCorr.

    Returns
    -------
    path : str
        Absolute path to the saved reference pack.
    """
    ensure_tls_certificates()
    with _crds_context(context) as selected:
        return _build_refpack(uncal_file, context=selected, out=out, reftypes=reftypes,
            inl_amplitude_file=inl_amplitude_file, inl_periods=inl_periods,
            wavemap_file=wavemap_file, include_inl=include_inl)


def _build_refpack(uncal_file, context=None, out=None, reftypes=None,
                  inl_amplitude_file=None, inl_periods=None,
                  wavemap_file=None, include_inl=None):
    """Build the complete reference pack for one representative uncal file."""
    import crds
    with fits.open(uncal_file, memmap=False) as hdu:
        header = hdu[0].header.copy()
        try:
            sci_shape = tuple(hdu['SCI'].data.shape)
        except (KeyError, TypeError):
            sci_shape = ()
    if out is None:
        provisional_out = pathlib.Path.cwd() / 'refpack_build.npz'
    else:
        provisional_out = pathlib.Path(out).expanduser()
    params = _crds_parameters(header)
    build_instrument = str(header.get('INSTRUME', '') or '').upper()
    if reftypes is None:
        reftypes = {'NIRSPEC': NIRSPEC_REFTYPES,
                    'MIRI': MIRI_REFTYPES}.get(build_instrument, REFTYPES)
    resolved = crds.getreferences(params, reftypes=reftypes, context=context, observatory='jwst')
    pack = {'meta_context': context or 'default',
            'meta_schema_version': np.int32(REFPACK_SCHEMA_VERSION),
            'meta_instrument': header.get('INSTRUME', ''),
            'meta_detector': header.get('DETECTOR', ''),
            'meta_subarray': header.get('SUBARRAY', ''),
            'meta_readpatt': header.get('READPATT', ''),
            'meta_ngroups': int(header.get('NGROUPS', 0) or
                                (sci_shape[-3] if len(sci_shape) >= 3 else 0))}
    gain_factor_from_ref = None
    gain_factor_source = None
    for reftype, path in resolved.items():
        if not isinstance(path, str) or not os.path.exists(path):
            continue
        if reftype == 'emicorr':
            pack.update(_extract_miri_emicorr_reference(path, header))
            pack['meta_file_emicorr'] = os.path.basename(path)
            continue
        with fits.open(path) as rhdu:
            ext = _REF_EXT.get(reftype, 'SCI')
            try:
                ref_hdu = rhdu[ext]
            except KeyError:
                ref_hdu = rhdu[1]
            reference_header = rhdu[0].header.copy()
            for key in ('SUBSTRT1', 'SUBSTRT2', 'SUBSIZE1', 'SUBSIZE2'):
                if key not in reference_header and key in ref_hdu.header:
                    reference_header[key] = ref_hdu.header[key]
            arr = (_reference_dq_from_hdul(rhdu, ext) if ext.upper() == 'DQ'
                   else np.asarray(ref_hdu.data))
            if reftype == 'gain':
                for location, candidate in (('PRIMARY', rhdu[0].header), (ext, ref_hdu.header)):
                    if 'GAINFACT' not in candidate:
                        continue
                    value = candidate.get('GAINFACT')
                    if value is None or not np.isfinite(value):
                        raise ValueError(f'gain reference {path} has invalid GAINFACT')
                    gain_factor_from_ref = np.float32(value)
                    gain_factor_source = (f'{os.path.basename(path)}:{location}:GAINFACT')
                    break
            arr = _cut_to_subarray(arr, header, reference_header)
            dark_status = None
            miri_dark_dq = None
            if reftype == 'dark':
                dark_header = rhdu[0].header.copy()
                # Read missing reference metadata from the science extension.
                for key in ('NFRAMES', 'NGROUPS', 'GROUPGAP'):
                    if key not in dark_header and key in rhdu[ext].header:
                        dark_header[key] = rhdu[ext].header[key]
                if build_instrument == 'MIRI':
                    try:
                        raw_dq = _cut_to_subarray(_reference_dq_from_hdul(rhdu), header,
                            reference_header)
                    except (KeyError, TypeError):
                        raw_dq = None
                    (arr, miri_dark_dq, dark_status, dark_reason) = _adapt_miri_dark_reference(
                        arr, raw_dq, header, dark_header)
                else:
                    arr, dark_status, dark_reason = _adapt_dark_reference(arr, header, dark_header)
            dtype = _REF_DTYPE.get(reftype, np.float32)
            pack_key = {'mask': 'mask_dq', 'linearity': 'lin_coeffs',
                        'reset': 'reset_data'}.get(reftype, reftype)
            pack[pack_key] = arr.astype(dtype)
            if reftype == 'linearity' and 'INV_COEFFS' in rhdu:
                pack['lin_inv_coeffs'] = _cut_to_subarray(
                    np.asarray(rhdu['INV_COEFFS'].data), header,
                    reference_header).astype(np.float32)
            for ext_name, key, comp_dtype in _REF_COMPANIONS.get(reftype, ()):
                try:
                    companion = (_reference_dq_from_hdul(rhdu, ext_name)
                        if ext_name.upper() == 'DQ' else np.asarray(rhdu[ext_name].data))
                    pack[key] = _cut_to_subarray(companion, header,
                        reference_header).astype(comp_dtype)
                except (KeyError, TypeError):
                    pass
            if reftype == 'reset':
                reset_dq = pack.get('reset_dq')
                if reset_dq is None:
                    pack['reset_dq'] = np.zeros(arr.shape[-2:], np.uint32)
                else:
                    reset_dq = np.asarray(reset_dq, np.uint32)
                    while reset_dq.ndim > 2:
                        collapsed = reset_dq[0]
                        for index in range(1, reset_dq.shape[0]):
                            collapsed = np.bitwise_or(collapsed, reset_dq[index])
                        reset_dq = collapsed
                    pack['reset_dq'] = np.asarray(reset_dq, np.uint32)
            if reftype == 'dark' and miri_dark_dq is not None:
                pack['dark_dq'] = miri_dark_dq
            if reftype == 'dark':
                pack['meta_dark_status'] = dark_status
                pack['meta_dark_skip_reason'] = dark_reason
                pack['meta_dark_reference_nframes'] = int(dark_header['NFRAMES'])
                pack['meta_dark_reference_groupgap'] = int(dark_header['GROUPGAP'])
                if dark_status == 'SKIPPED':
                    pack['dark_dq'] = np.zeros(arr.shape[-2:], np.uint32)
                # Retain average dark current for ramp fitting even when subtraction is skipped.
                average = pack.get('average_dark_current')
                if average is None or np.sum(average) == 0:
                    value = None
                    for candidate in (rhdu[0].header, ref_hdu.header):
                        if 'AVDRKCUR' in candidate:
                            value = candidate.get('AVDRKCUR')
                            break
                    if value is None:
                        value = 0.
                    if not np.isfinite(value):
                        raise ValueError(f'dark reference {path} has invalid AVDRKCUR')
                    pack['average_dark_current'] = np.full(arr.shape[-2:], value, np.float32)
            pack[f'meta_file_{reftype}'] = os.path.basename(path)
    # Use unity when the gain reference supplies no GainScale factor.
    gain_factor = (np.float32(1.) if gain_factor_from_ref is None else gain_factor_from_ref)
    pack['gain_factor'] = gain_factor
    pack['GAINFACT'] = gain_factor
    pack['meta_gain_factor_source'] = ('gain-reference:missing; GainScale equivalent skip'
        if gain_factor_source is None else gain_factor_source)
    instrument = str(pack['meta_instrument']).upper()
    if include_inl is None:
        include_inl = instrument == 'NIRISS'
    if include_inl:
        theta, periods, source = _load_inl_calibration(inl_amplitude_file, inl_periods,
            provisional_out.parent / 'calibration')
        pack['inl_theta'] = theta
        pack['inl_periods'] = periods
        pack['meta_file_inl'] = os.path.basename(source)
    if instrument == 'NIRISS':
        if wavemap_file is None:
            default_wavemap = soss_reference_file('wavemap', header.get('SUBARRAY'),
                download_dir=provisional_out.parent / 'calibration')
            if default_wavemap is not None:
                waves = v1_soss_wave_vectors(default_wavemap)
                pack['wave_o1'] = np.asarray(waves[1], np.float64)
                pack['wave_o2'] = np.asarray(waves[2], np.float64)
                pack['meta_file_wavemap'] = os.path.basename(default_wavemap)
        else:
            from exotedrf.v2.trace import load_soss_wavemap_vectors
            dimx = (int(header.get('SUBSIZE1', 0) or 0) or (sci_shape[-1] if sci_shape else 2048))
            waves = load_soss_wavemap_vectors(wavemap_file, dimx=dimx)
            pack['wave_o1'] = np.asarray(waves[1], np.float64)
            pack['wave_o2'] = np.asarray(waves[2], np.float64)
            pack['meta_file_wavemap'] = os.path.basename(str(wavemap_file))
    elif instrument in ('NIRSPEC', 'MIRI'):
        spatial_shape = ((int(header.get('SUBSIZE2', 0) or 0), int(header.get('SUBSIZE1', 0) or 0))
            if header.get('SUBSIZE1') else tuple(sci_shape[-2:]))
        if wavemap_file is not None:
            pack['wave_map'] = _load_wave_map(wavemap_file, spatial_shape, label=instrument)
            pack['meta_file_wavemap'] = os.path.basename(str(wavemap_file))
        elif instrument == 'NIRSPEC':
            pack['wave_map'] = _compute_nirspec_wave_map(uncal_file, spatial_shape)
            pack['meta_wavecorr'] = 'SKIPPED'
            pack['meta_file_wavemap'] = 'jwst:assign_wcs+nrs_wcs_set_input'
        else:
            pack['wave_map'] = _compute_miri_wave_map(uncal_file, spatial_shape)
            pack['meta_file_wavemap'] = 'jwst:assign_wcs'
    if out is None:
        out = 'refpack_{}_{}.npz'.format(pack['meta_instrument'], pack['meta_subarray']).lower()
    out = os.path.abspath(os.path.expanduser(str(out)))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **pack)
    return out


class RefPack:
    """Open a saved calibration reference pack for use during reduction."""

    def __init__(self, path):
        """Open a saved calibration reference pack.

        Parameters
        ----------
        path : str
            Path to the input or output file.
        """
        self._npz = np.load(path, allow_pickle=False)
        self.path = path

    def __contains__(self, key):
        """Check whether the reference pack contains a key."""
        return key in self._npz.files

    def get(self, key, default=None):
        """Read a reference array or return the supplied default."""
        return self._npz[key] if key in self._npz.files else default

    def __getitem__(self, key):
        """Read the selected array or integration slice."""
        return self._npz[key]

    @property
    def keys(self):
        """List the reference keys stored in the pack."""
        return list(self._npz.files)

    def close(self):
        """Close the reference pack file."""
        self._npz.close()

    def __enter__(self):
        """Return the open reference pack."""
        return self

    def __exit__(self, *exc):
        """Close the reference pack on leaving the context."""
        self.close()


def _get(pack, key, default=None):
    """Read a reference array, using the default for missing values."""
    if pack is None:
        return default
    if hasattr(pack, 'get'):
        value = pack.get(key)
        return default if value is None else value
    return pack[key] if key in pack else default


def _validate_schema(pack, *, required):
    """Ensure a saved pack follows the current reference layout."""
    raw = _get(pack, 'meta_schema_version')
    if raw is None:
        if required:
            raise ValueError('refpack has no meta_schema_version; rebuild it with the '
                'current exoTEDRF v2 implementation')
        return
    value = np.asarray(raw)
    if value.ndim != 0:
        raise ValueError('refpack meta_schema_version must be a scalar')
    try:
        version = int(value.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('refpack meta_schema_version is invalid') from exc
    if version != REFPACK_SCHEMA_VERSION:
        raise ValueError(f'refpack schema v{version} is stale; expected '
            f'v{REFPACK_SCHEMA_VERSION}, rebuild the refpack')


def _step_runs(opts, key, default='run'):
    """Interpret the exact lowercase ``run``/``skip`` controls used by v1."""
    value = opts.get(key, default)
    if value not in ('run', 'skip'):
        raise ValueError(f"{key} must be exactly lowercase 'run' or 'skip', got {value!r}")
    return value == 'run'


def validate_refpack(pack, cube, opts, require_waves=True):
    """Confirm that a reference pack is suitable for this observation.

    Parameters
    ----------
    pack : RefPack, dict
        Calibration reference arrays and provenance.
    cube : RampCube, RateCube
        Observation arrays and metadata.
    opts : dict
        Reduction configuration.
    require_waves : bool
        If True, validate the wavelength references required by extraction.

    Returns
    -------
    pack : RefPack, dict
        Validated, supplied, or automatically loaded reference pack.
    """
    if pack is None:
        raise ValueError('A SOSS refpack is required')
    _validate_schema(pack, required=isinstance(pack, RefPack))
    shape = tuple(int(x) for x in cube.data.shape[-2:])
    ngroups = int(cube.data.shape[1])
    required = []
    if _step_runs(opts, 'DQInitStep'):
        required += ['mask_dq']
    if _step_runs(opts, 'INLCorrStep'):
        required += ['inl_theta', 'inl_periods']
    if _step_runs(opts, 'SuperBiasStep') and \
            opts.get('superbias_method', 'crds') == 'crds':
        required += ['superbias']
    if _step_runs(opts, 'DarkCurrentStep'):
        required += ['dark', 'average_dark_current']
    if _step_runs(opts, 'LinearityStep'):
        required += ['lin_coeffs', 'lin_dq']
    if _step_runs(opts, 'RampFitStep'):
        required += ['readnoise', 'gain']
    elif (_step_runs(opts, 'JumpStep') and opts.get('flag_up_ramp', False)):
        required += ['readnoise', 'gain']
    if _step_runs(opts, 'GainScaleStep'):
        if _get(pack, 'gain_factor') is None and _get(pack, 'GAINFACT') is None:
            required += ['gain_factor']
    if _step_runs(opts, 'FlatFieldStep'):
        required += ['flat']
    instrument = str(cube.meta.mode).split('/')[0].upper()
    wavecorr_status = _get(pack, 'meta_wavecorr')
    if instrument == 'NIRSPEC' and wavecorr_status is not None:
        expected = 'SKIPPED'
        if str(np.asarray(wavecorr_status).item()).upper() != expected:
            raise ValueError('refpack WaveCorr status does not match WaveCorrStep; '
                'rebuild the reference pack with the requested setting')
    if instrument == 'MIRI':
        if _step_runs(opts, 'EmiCorrStep'):
            required += ['emicorr_frequencies', 'emicorr_reference_waves',
                         'emicorr_reference_wave_lengths',
                         'emicorr_rowclocks', 'emicorr_frameclocks']
        # Require the MIRI dark for the Linearity correction.
        if _step_runs(opts, 'LinearityStep') and \
                opts.get('miri_subtract_dark', True):
            required += ['dark', 'average_dark_current']
    if require_waves and (_step_runs(opts, 'AssignWCSStep') or
                          opts.get('extract_method', 'box') == 'box'):
        required += (['wave_map'] if instrument in ('NIRSPEC', 'MIRI') else ['wave_o1', 'wave_o2'])
    missing = sorted({key for key in required if _get(pack, key) is None})
    if missing:
        raise ValueError('refpack is missing references required by enabled steps: ' +
            ', '.join(missing))
    spatial_keys = ('superbias', 'superbias_dq', 'lin_dq', 'mask_dq',
                    'dark_dq', 'flat', 'flat_dq', 'flat_err')
    for key in spatial_keys:
        arr = _get(pack, key)
        if arr is not None and np.shape(arr) != shape:
            raise ValueError(f'refpack {key} has shape {np.shape(arr)}, expected {shape}')
    for key in ('readnoise', 'gain'):
        arr = _get(pack, key)
        if arr is not None and np.ndim(arr) != 0 and np.shape(arr) != shape:
            raise ValueError(
                f'refpack {key} has shape {np.shape(arr)}, expected scalar or {shape}')
    for key in ('lin_coeffs', 'lin_inv_coeffs'):
        coeffs = _get(pack, key)
        if coeffs is not None and (np.ndim(coeffs) != 3 or tuple(np.shape(coeffs)[-2:]) != shape):
            raise ValueError(f'refpack {key} has shape {np.shape(coeffs)}, expected '
                f'(ncoeff, {shape[0]}, {shape[1]})')
    dark = _get(pack, 'dark')
    if dark is not None:
        if instrument == 'MIRI':
            if (np.ndim(dark) != 4 or np.shape(dark)[1] < ngroups or
                    tuple(np.shape(dark)[-2:]) != shape):
                raise ValueError(f'refpack MIRI dark has shape {np.shape(dark)}, expected '
                    f'(dark_nints, >={ngroups}, {shape[0]}, {shape[1]})')
        elif np.shape(dark) != (ngroups,) + shape:
            raise ValueError(
                f'refpack dark has shape {np.shape(dark)}, expected {(ngroups,) + shape}')
    reset_data = _get(pack, 'reset_data')
    if reset_data is not None and (np.ndim(reset_data) != 4 or
            tuple(np.shape(reset_data)[-2:]) != shape):
        raise ValueError(f'refpack reset_data has shape {np.shape(reset_data)}, expected '
            f'(reset_nints, reset_ngroups, {shape[0]}, {shape[1]})')
    reset_dq = _get(pack, 'reset_dq')
    if reset_dq is not None and np.shape(reset_dq) != shape:
        raise ValueError(f'refpack reset_dq has shape {np.shape(reset_dq)}, expected {shape}')
    emi_waves = _get(pack, 'emicorr_reference_waves')
    emi_freqs = _get(pack, 'emicorr_frequencies')
    if emi_waves is not None or emi_freqs is not None:
        emi_waves = np.asarray(emi_waves) if emi_waves is not None else None
        emi_freqs = np.asarray(emi_freqs) if emi_freqs is not None else None
        if (emi_waves is None or emi_freqs is None or emi_waves.ndim != 2 or
                emi_freqs.ndim != 1 or not np.isfinite(emi_freqs).all() or np.any(emi_freqs <= 0)):
            raise ValueError('refpack EMI references must be a (nfreq,) positive frequency '
                'vector with a matching (nfreq, nphase) reference-wave array')
        if emi_waves.shape[0] != emi_freqs.shape[0]:
            raise ValueError('refpack emicorr_reference_waves must have one row per frequency')
        lengths = _get(pack, 'emicorr_reference_wave_lengths')
        if lengths is None:
            raise ValueError('refpack emicorr_reference_wave_lengths is required')
        lengths = np.asarray(lengths)
        if lengths.ndim != 1 or lengths.shape[0] != emi_freqs.shape[0]:
            raise ValueError('refpack emicorr_reference_wave_lengths must have one entry '
                'per frequency')
        if lengths.dtype.kind not in 'iu':
            raise ValueError('refpack emicorr_reference_wave_lengths must be an integer array')
        if np.any(lengths < 1) or np.any(lengths > emi_waves.shape[1]):
            raise ValueError('refpack emicorr_reference_wave_lengths exceed the padded '
                'wave width')
        for key in ('emicorr_rowclocks', 'emicorr_frameclocks'):
            value = np.asarray(_get(pack, key))
            if (value.ndim != 0 or value.dtype.kind not in 'iu' or int(value) <= 0):
                raise ValueError(f'refpack {key} must be a positive integer scalar')
    average_dark = _get(pack, 'average_dark_current')
    if (average_dark is not None and np.ndim(average_dark) != 0 and
            np.shape(average_dark) != shape):
        raise ValueError('refpack average_dark_current has shape '
            f'{np.shape(average_dark)}, expected scalar or {shape}')
    if (average_dark is not None and not np.isfinite(np.asarray(average_dark)).all()):
        raise ValueError('refpack average_dark_current must be finite')
    dark_status = _get(pack, 'meta_dark_status')
    if dark_status is not None:
        dark_status = str(np.asarray(dark_status).item()).upper()
        if dark_status not in ('COMPLETE', 'SKIPPED'):
            raise ValueError(f'refpack meta_dark_status must be COMPLETE or SKIPPED, got '
                f'{dark_status!r}')
        if dark_status == 'SKIPPED':
            dark_dq = _get(pack, 'dark_dq')
            if ((dark is not None and np.any(np.asarray(dark) != 0)) or (dark_dq is not None and
                     np.any(np.asarray(dark_dq) != 0))):
                raise ValueError('a SKIPPED dark must contain neutral zero dark/dark_dq arrays')
    theta, periods = _get(pack, 'inl_theta'), _get(pack, 'inl_periods')
    if theta is not None or periods is not None:
        theta, periods = np.asarray(theta), np.asarray(periods)
        if (theta.ndim != 1 or periods.ndim != 1 or theta.size != 2 * periods.size or
                not np.isfinite(theta).all() or not np.isfinite(periods).all() or
                np.any(periods <= 0)):
            raise ValueError('invalid INL theta/periods in refpack')
    for key in ('wave_o1', 'wave_o2'):
        wave = _get(pack, key)
        if wave is not None and (np.shape(wave) != (shape[1],) or
                                 not np.isfinite(np.asarray(wave)).any()):
            raise ValueError(f'refpack {key} must have shape ({shape[1]},) with finite values')
    wave_map = _get(pack, 'wave_map')
    if wave_map is not None and (np.shape(wave_map) != shape or
                                 not np.isfinite(np.asarray(wave_map)).any()):
        raise ValueError(f'refpack wave_map must have shape {shape} with finite values')
    gain_factor = _get(pack, 'gain_factor', _get(pack, 'GAINFACT'))
    if gain_factor is not None and (np.ndim(gain_factor) != 0 or not np.isfinite(gain_factor)):
        raise ValueError('refpack gain_factor/GAINFACT must be a finite scalar')
    requested_context = effective_crds_context(opts.get('crds_context'))
    if requested_context is not None:
        requested_context = str(requested_context).strip()
    if requested_context and requested_context.lower() != 'default':
        pack_context = _get(pack, 'meta_context')
        if pack_context is not None:
            pack_context = str(np.asarray(pack_context).item()).strip()
            # Compare CRDS contexts only when the pack records a pinned context.
            legacy = ('', 'default', 'legacy', 'none', 'n/a')
            if (pack_context.lower() not in legacy and
                    pack_context.lower() != requested_context.lower()):
                raise ValueError(f'refpack meta_context={pack_context!r} does not match '
                    f'opts crds_context={requested_context!r}')
    expected = {'meta_instrument': cube.meta.mode.split('/')[0].upper(),
        'meta_detector': str(cube.meta.detector).upper(),
        'meta_subarray': str(cube.meta.subarray).upper(),}
    for key, want in expected.items():
        got = _get(pack, key)
        if got is None:
            continue
        got = str(np.asarray(got).item()).upper()
        # Accept the NIRISS filter selection when the reference records detector NIS.
        if key == 'meta_detector' and expected['meta_instrument'] == 'NIRISS' \
                and want in ('CLEAR', 'F277W') and got == 'NIS':
            continue
        if want and got and got != want:
            raise ValueError(f'refpack {key}={got!r} does not match observation {want!r}')
    return pack


def _configured_file(opts, key):
    """Resolve a calibration file relative to the input directory or cwd."""
    value = opts.get(key)
    if value is None:
        return None
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        candidate = pathlib.Path(opts.get('input_dir') or '.') / path
        if candidate.is_file():
            path = candidate
    return path.resolve()


def _configured_wavemap(opts):
    """Resolve the configured wavelength reference path."""
    return _configured_file(opts, 'wavemap_file')


def _hash_file(digest, path):
    """Add a calibration path and its contents to a digest."""
    digest.update(os.fsencode(path))
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)


def default_refpack_path(cube, opts):
    """Choose a cache filename tied to this visit and CRDS context.

    Parameters
    ----------
    cube : RampCube, RateCube
        Observation arrays and metadata.
    opts : dict
        Reduction configuration.

    Returns
    -------
    path : Path
        Cache path including the observing setup and calibration identifiers.
    """
    output = os.path.expanduser(os.fspath(opts.get('output_dir', 'pipeline_outputs_directory')))
    output_tag = str(opts.get('output_tag', '') or '')
    if output_tag:
        output += f'_{output_tag}'
    output = pathlib.Path(output)

    def safe_component(value, fallback):
        """Replace unsafe cache filename characters and apply a fallback."""
        value = re.sub('[^A-Za-z0-9_.-]+', '-', str(value)).strip('._-')
        return value or fallback
    context = safe_component(
        effective_crds_context(opts.get('crds_context')) or 'default', 'default')
    if cube.meta.filenames:
        source = pathlib.Path(os.fspath(cube.meta.filenames[0])).name
        source = pathlib.Path(source).stem
    else:
        source = 'no-source'
    source = safe_component(source, 'no-source')
    stem = 'refpack_v2s{}_{}_{}_{}_{}_{}'.format(REFPACK_SCHEMA_VERSION,
        cube.meta.mode.split('/')[0], cube.meta.detector,
        cube.meta.subarray, source, context).lower()
    wavemap = _configured_wavemap(opts)
    if wavemap is not None:
        digest = hashlib.sha256()
        _hash_file(digest, wavemap)
        stem += '_wave-' + digest.hexdigest()[:16]
    instrument = cube.meta.mode.split('/')[0].upper()
    if instrument == 'NIRISS':
        if _step_runs(opts, 'INLCorrStep'):
            periods = opts.get('inl_periods')
            periods = np.asarray(DEFAULT_INL_PERIODS if periods is None else periods,
                dtype=np.float32)
            digest = hashlib.sha256(periods.tobytes())
            amplitude = _configured_file(opts, 'inl_amplitude_file')
            if amplitude is None:
                digest.update(DEFAULT_INL_URL.encode())
            else:
                _hash_file(digest, amplitude)
            stem += '_inl-' + digest.hexdigest()[:16]
        else:
            stem += '_inl-skip'
    if instrument == 'NIRSPEC' and wavemap is None:
        stem += '_wavecorr-' + ('run' if _step_runs(opts, 'WaveCorrStep') else 'skip')
    return output / 'v2' / 'refpacks' / f'{stem}.npz'


def resolve_refpack(cube, opts, refpack=None):
    """Use a supplied reference pack or build and cache one automatically.

    Explicit stale packs raise an error; automatic stale caches are rebuilt.

    Parameters
    ----------
    cube : RampCube, RateCube
        Observation arrays and metadata.
    opts : dict
        Reduction configuration.
    refpack : None, str, RefPack, dict
        Supplied reference pack or path; use configuration or automatic cache if omitted.

    Returns
    -------
    pack : RefPack, dict
        Validated, supplied, or automatically loaded reference pack.
    """
    refpack = refpack if refpack is not None else opts.get('refpack')
    if refpack is not None and not isinstance(refpack, (str, os.PathLike)):
        return refpack
    if refpack is None:
        path = default_refpack_path(cube, opts)
    else:
        path = pathlib.Path(refpack).expanduser()
    if path.exists():
        loaded = RefPack(str(path))
        try:
            _validate_schema(loaded, required=True)
        except ValueError:
            loaded.close()
            if refpack is not None:
                raise
            # Rebuild stale automatic caches from the original exposure.
        else:
            return loaded
    if refpack is not None:
        raise FileNotFoundError(f'refpack not found: {path}')
    if not cube.meta.filenames:
        raise ValueError('cannot build a refpack without a source uncal FITS file')
    path.parent.mkdir(parents=True, exist_ok=True)
    built = build_refpack(cube.meta.filenames[0],
        context=effective_crds_context(opts.get('crds_context')), out=path,
        inl_amplitude_file=_configured_file(opts, 'inl_amplitude_file'),
        inl_periods=opts.get('inl_periods'), wavemap_file=_configured_wavemap(opts),
        include_inl=_step_runs(opts, 'INLCorrStep'),
        apply_wavecorr=_step_runs(opts, 'WaveCorrStep'))
    return RefPack(str(built))


def main():
    """Build a reference pack from one ``uncal`` file on the command line."""
    p = argparse.ArgumentParser(description='Build an exoTEDRF v2 refpack.')
    p.add_argument('uncal_file')
    p.add_argument('--context', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--inl-amplitude-file', default=None)
    p.add_argument('--inl-periods', nargs='*', type=float, default=None)
    p.add_argument('--wavemap-file', default=None)
    p.add_argument('--skip-wavecorr', action='store_true',
                   help='Build NIRSpec wavelengths without WaveCorr')
    args = p.parse_args()
    out = build_refpack(args.uncal_file, context=args.context, out=args.out,
        inl_amplitude_file=args.inl_amplitude_file,
        inl_periods=args.inl_periods, wavemap_file=args.wavemap_file,
        apply_wavecorr=not args.skip_wavecorr)
    print(f'Refpack written to {out}')


if __name__ == '__main__':
    main()
