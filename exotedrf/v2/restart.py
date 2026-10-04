"""Resume reductions from v1-format intermediate products."""

from __future__ import annotations

import hashlib
import os
import pathlib

import numpy as np

# Map step names to v1 tags, distinguishing shared tags by product dimensions.
STAGE1_TAGS = (('DQInitStep', 'dqinitstep'), ('INLCorrStep', 'inlcorrstep'),
    ('EmiCorrStep', 'emicorrstep'), ('ResetStep', 'resetstep'), ('SuperBiasStep', 'superbiasstep'),
    ('RefPixStep', 'refpixstep'), ('DarkCurrentStep', 'darkcurrentstep'),
    ('BackgroundStep_grp', 'backgroundstep'), ('OneOverFStep_grp', 'oneoverfstep'),
    ('LinearityStep', 'linearitystep'), ('JumpStep', 'jump'), ('RampFitStep', 'rampfitstep'),
    ('GainScaleStep', 'gainscalestep'),)
STAGE2_TAGS = (('AssignWCSStep', 'assignwcsstep'), ('Extract2DStep', 'extract2dstep'),
    ('SourceTypeStep', 'sourcetypestep'), ('WaveCorrStep', 'wavecorrstep'),
    ('FlatFieldStep', 'flatfieldstep'), ('BackgroundStep', 'backgroundstep'),
    ('OneOverFStep_int', 'oneoverfstep'), ('BadPixStep', 'badpixstep'),
    ('PCAReconstructStep', 'pcareconstructstep'),)
STEP_TAGS = dict(STAGE1_TAGS + STAGE2_TAGS)
STAGE1_STEPS = tuple(name for name, _ in STAGE1_TAGS)
STAGE2_STEPS = tuple(name for name, _ in STAGE2_TAGS)

# Identify steps that require up-the-ramp inputs.
RAMP_STEPS = frozenset(STAGE1_STEPS[:STAGE1_STEPS.index('RampFitStep') + 1])

def is_restart(cfg):
    """Check whether the input tag selects intermediate products.

    Returns
    -------
    restart : bool
        Whether input_filetag differs from uncal.
    """
    tag = cfg.get('input_filetag', 'uncal')
    return tag is not None and str(tag).strip().lower() != 'uncal'


def producing_step(filetag, ndim):
    """Find the reduction step that produces an input tag.

    Parameters
    ----------
    filetag : str
        Input product tag.
    ndim : int
        Number of science-data dimensions.

    Returns
    -------
    step : None, str
        Producing step, or None for a generic JWST product.
    """
    tag = str(filetag).strip().lower()
    if tag.endswith('.fits'):
        tag = tag[:-5]
    tables = (STAGE1_TAGS, STAGE2_TAGS) if ndim == 4 else (STAGE2_TAGS, STAGE1_TAGS)
    for table in tables:
        for name, v1_tag in table:
            if tag == v1_tag:
                if ndim == 4 and name not in RAMP_STEPS:
                    continue
                if ndim != 4 and name in RAMP_STEPS:
                    continue
                return name
    return None


def _graph_step_names(mode, opts):
    """Get enabled step names for the instrument recipe."""
    from exotedrf.v2 import stages
    mode = str(mode).upper()
    if mode.startswith('NIRSPEC'):
        steps = stages.nirspec_steps(opts)
    elif mode.startswith('MIRI'):
        steps = stages.miri_steps(opts)
    else:
        steps = stages.soss_steps(opts)
    return [step.name for step in steps]


def check_restart_steps(opts, ndim, filetag, logger=None):
    """Validate enabled steps for the input product dimensions.

    Parameters
    ----------
    opts : dict
        Fixed reduction options.
    ndim : int
        Number of science-data dimensions.
    filetag : str
        Input product tag.
    logger : None, callable
        Function to receive progress and warning messages.

    Returns
    -------
    names : list[str]
        Enabled steps in execution order.
    """
    names = _graph_step_names(opts.get('mode', ''), opts)
    if ndim != 4:
        ramp = [name for name in names if name in RAMP_STEPS]
        if ramp:
            raise ValueError(f'input_filetag={filetag!r} products are post-RampFit (3-D); '
                f'set {ramp} to \'skip\' (v1 cannot run ramp steps on them ' 'either)')
    if ndim == 4 and 'RampFitStep' not in names and any(name in STAGE2_STEPS for name in names):
        raise ValueError(f'input_filetag={filetag!r} products are up-the-ramp (4-D); '
            'Stage-2 steps need RampFitStep to run first')
    producer = producing_step(filetag, ndim)
    if producer is None:
        return names
    order = list(STAGE1_STEPS + STAGE2_STEPS)
    position = order.index(producer)
    repeated = [name for name in names if name in order and order.index(name) <= position]
    if repeated and logger is not None:
        logger(f'[v2] WARNING: input_filetag={filetag!r} was written by '
               f'{producer}, but {repeated} are still enabled and will be '
               'applied again (v1 restart semantics); set them to \'skip\' '
               'to continue from the product instead')
    return names


def load_restart_state(opts, *, files=None, logger=None):
    """Read intermediate products into the corresponding pipeline state.

    Parameters
    ----------
    opts : dict
        Fixed reduction options.
    files : None, list[str]
        Input FITS paths. If None, find products from the configuration.
    logger : None, callable
        Function to receive progress and warning messages.

    Returns
    -------
    state : PipelineState
        Loaded ramp or rate observation.
    files : list[str]
        Input product paths.
    """
    from exotedrf.v2 import io
    from exotedrf.v2.pipeline import PipelineState
    tag = opts.get('input_filetag')
    if files is None:
        files = io.find_products(opts['input_dir'], tag, mode=opts.get('mode'),
            filter_detector=opts.get('filter_detector'))
    files = list(files)
    if not files:
        raise RuntimeError(f'No FITS found in {opts["input_dir"]} matching '
            f'input_filetag={tag!r}')
    ndims = {io.product_ndim(path) for path in files}
    if len(ndims) != 1:
        raise ValueError(f'input_filetag={tag!r} matches products of mixed '
            f'dimensionality {sorted(ndims)}')
    ndim = ndims.pop()
    check_restart_steps(opts, ndim, tag, logger=logger)
    common = dict(baseline_ints=opts['baseline_ints'], mode=opts['mode'],
                  filter_detector=opts.get('filter_detector'),
                  science_dtype=opts.get('dtype', np.float32),
                  max_host_bytes=opts.get('max_host_bytes'), scratch_dir=opts.get('scratch_dir'))
    if ndim == 4:
        cube = io.load_ramp_cube(files, **common)
    else:
        cube = io.load_rate_products(files, **common)
    cube.meta.extra['input_filetag'] = str(tag)
    if logger is not None:
        kind = 'ramp' if ndim == 4 else 'rate'
        logger(f'[v2] restart: loaded {len(files)} {tag!r} {kind} product(s) '
               f'{tuple(cube.data.shape)}')
    return PipelineState(cube=cube), files


def required_reftypes(opts, mode):
    """Get the CRDS reference types needed by enabled restart steps.

    Parameters
    ----------
    opts : dict
        Fixed reduction options.
    mode : str
        Instrument and observing mode.

    Returns
    -------
    reftypes : list[str]
        Required reference types in request order.
    """
    instrument = str(mode).split('/')[0].upper()
    names = set(_graph_step_names(mode, opts))
    requested = (('DQInitStep', ('mask',), True), ('SuperBiasStep', ('superbias',),
         str(opts.get('superbias_method', 'crds')).lower() == 'crds'),
        ('DarkCurrentStep', ('dark',), True), ('LinearityStep', ('linearity',), True),
        ('LinearityStep', ('dark',), instrument == 'MIRI' and opts.get('miri_subtract_dark', True)),
        ('JumpStep', ('readnoise', 'gain'), opts.get('flag_up_ramp', False)),
        ('RampFitStep', ('readnoise', 'gain'), True),
        ('GainScaleStep', ('gain',), True), ('FlatFieldStep', ('flat',), True),
        ('ResetStep', ('reset',), True), ('EmiCorrStep', ('emicorr',), True),)
    return list(dict.fromkeys(reftype for step, types, enabled in requested
                              if step in names and enabled for reftype in types))


def restart_refpack(cube, opts, files, *, logger=None):
    """Build or reuse a reference pack for the enabled restart steps.

    Parameters
    ----------
    cube : RampCube, RateCube
        Observation data and metadata.
    opts : dict
        Fixed reduction options.
    files : None, list[str]
        Input FITS paths. If None, find products from the configuration.
    logger : None, callable
        Function to receive progress and warning messages.

    Returns
    -------
    path : str
        Path to the configured or cached reference pack.
    """
    from exotedrf.v2 import io, refs
    configured = opts.get('refpack')
    if configured is not None:
        return configured
    mode = opts.get('mode', cube.meta.mode)
    instrument = str(mode).split('/')[0].upper()
    # Request gain when no references are needed to avoid fetching every CRDS type.
    reftypes = required_reftypes(opts, mode) or ['gain']
    build_opts = dict(opts)
    base = refs.default_refpack_path(cube, opts)
    if instrument in ('NIRSPEC', 'MIRI') and opts.get('wavemap_file') is None:
        wave = io.read_wavelength_plane(files[0])
        if wave is not None and wave.shape == tuple(cube.data.shape[-2:]):
            wave_path = base.with_name(base.stem + '_product_wavelength.npy')
            wave_path.parent.mkdir(parents=True, exist_ok=True)
            if not wave_path.exists() or not np.array_equal(
                    np.load(wave_path, allow_pickle=False), wave, equal_nan=True):
                np.save(wave_path, wave, allow_pickle=False)
            build_opts['wavemap_file'] = str(wave_path)
            if logger is not None:
                logger(f'[v2] restart: wavelengths from the WAVELENGTH plane '
                       f'of {os.path.basename(files[0])}')
    base = refs.default_refpack_path(cube, build_opts)
    digest = hashlib.sha256(','.join(reftypes).encode()).hexdigest()[:8]
    label = '-'.join(reftypes)
    path = base.with_name(f'{base.stem}_restart-{label}-{digest}.npz')
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if logger is not None:
        logger(f'[v2] restart: building refpack {path.name} (reftypes={reftypes})')
    built = refs.build_refpack(files[0], context=refs.effective_crds_context(
            opts.get('crds_context')), out=path, reftypes=reftypes,
        inl_amplitude_file=refs._configured_file(opts, 'inl_amplitude_file'),
        inl_periods=opts.get('inl_periods'), wavemap_file=refs._configured_wavemap(build_opts),
        include_inl=('INLCorrStep' in _graph_step_names(mode, opts)),
        apply_wavecorr=refs._step_runs(opts, 'WaveCorrStep'))
    return str(built)


def restart_opts(opts, cube):
    """Record the restart point and disable raw-ramp segment scheduling.

    Parameters
    ----------
    opts : dict
        Fixed reduction options.
    cube : RampCube, RateCube
        Observation data and metadata.

    Returns
    -------
    options : dict
        New option dictionary with restart metadata.
    """
    opts = dict(opts)
    opts['restart_input_ndim'] = int(cube.data.ndim)
    opts['restart_input_filetag'] = str(opts.get('input_filetag'))
    # Disable segment scheduling for intermediate-product inputs.
    opts['stream_stage1'] = False
    return opts


def cached_resume_point(step_names, stage_dir, fileroots, *, save_results=True,
                        instrument='NIRISS', remove_components=None):
    """Find the last step with complete cached products and sidecars.

    Parameters
    ----------
    step_names, fileroots : list[str]
        Enabled step names and input segment filename roots.
    stage_dir : str
        Directory containing cached stage products.
    save_results : bool
        If True, require sidecars saved with the cached step.
    instrument : str
        Instrument name.
    remove_components : None, list[int]
        Unused argument retained for existing callers.

    Returns
    -------
    resume : None, tuple
        Cached step index and product tag, or None.
    """
    from exotedrf.v2 import io
    stage_dir = pathlib.Path(stage_dir)
    if not stage_dir.is_dir() or not fileroots:
        return None
    noseg = io.v1_fileroot_noseg(fileroots)
    best = None
    for index, name in enumerate(step_names):
        tag = STEP_TAGS.get(name)
        if tag is None:
            continue
        if not all((stage_dir / f'{root}{tag}.fits').is_file() for root in fileroots):
            continue
        if name == 'BadPixStep' and save_results and \
                not (stage_dir / f'{noseg}hot_pixels.npy').is_file():
            continue
        if name in ('BackgroundStep', 'BackgroundStep_grp') and instrument == 'NIRISS' and \
                not (stage_dir / f'{noseg}background.npy').is_file():
            continue
        if name == 'PCAReconstructStep' and not (stage_dir / f'{noseg}deepframe.fits').is_file():
            continue
        best = (index, tag)
    return best


def reusable_hot_pixels(stage_dir, fileroots, *, save_results=True):
    """Load the cached BadPix hot-pixel map when results are saved.

    Parameters
    ----------
    stage_dir : str
        Directory containing cached stage products.
    fileroots : list[str]
        Filename roots for every input segment.
    save_results : bool
        If True, require sidecars saved with the cached step.

    Returns
    -------
    hot_pixels : None, np.ndarray(bool)
        Cached hot-pixel mask, if available.
    """
    from exotedrf.v2 import io
    if not save_results or not fileroots:
        return None
    path = pathlib.Path(stage_dir) / (f'{io.v1_fileroot_noseg(fileroots)}hot_pixels.npy')
    if not path.is_file():
        return None
    return np.asarray(np.load(path, allow_pickle=False)) != 0


SEGMENT_WRITER_NAMES = ('write_stage2_segments', 'write_v1_segment_products')


def segment_product_writer():
    """Get the available writer for v1-format segment products.

    Returns
    -------
    writer : None, callable
        Per-segment FITS writer, if available.
    """
    from exotedrf.v2 import io
    for name in SEGMENT_WRITER_NAMES:
        function = getattr(io, name, None)
        if callable(function):
            return function
    return None


def write_segment_products(cube, templates, out_dir, tag, *, logger=None):
    """Write calibrated segments with v1 product names.

    Parameters
    ----------
    cube : RampCube, RateCube
        Observation data and metadata.
    templates : list[str]
        Input FITS files to use as per-segment templates.
    out_dir, tag : str
        Output directory and product filename tag.
    logger : None, callable
        Function to receive progress and warning messages.

    Returns
    -------
    files : list[str]
        Written product paths, or an empty list if no writer is available.
    """
    from exotedrf.v2 import io
    writer = segment_product_writer()
    if writer is None:
        if logger is not None:
            logger(f'[v2] per-segment {tag!r} FITS products are not written: '
                   'no v1-format segment writer is installed (the combined '
                   'v2 rate cube is written instead)')
        return []
    fileroots = io.v1_fileroots(templates)
    return list(writer(cube, list(templates), os.fspath(out_dir), fileroots, tag))


__all__ = ['RAMP_STEPS', 'STAGE1_STEPS', 'STAGE1_TAGS', 'STAGE2_STEPS',
    'STAGE2_TAGS', 'STEP_TAGS', 'cached_resume_point', 'check_restart_steps',
    'is_restart', 'load_restart_state', 'producing_step', 'required_reftypes',
    'restart_opts', 'restart_refpack', 'reusable_hot_pixels',
    'segment_product_writer', 'write_segment_products']
