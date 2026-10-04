"""Extract NIRISS/SOSS spectra with the JWST ATOCA algorithm."""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import dataclasses
import functools
import hashlib
import math
import multiprocessing
import os
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

from exotedrf.v2 import stage2_models

JWST_VERSION = '3.0.0'
ESTIMATE_FAILURE = '(m>k) failed for hidden m: fpcurf0:m=0'
V1_STEP_KWARGS = {'subtract_background': False, 'soss_bad_pix': 'model'}
SPECPROFILE_PAD = 20
DECONTAM_PANELS = 9
_RESULT_CACHE_SIZE = 8


def v1_specprofile_name(subarray):
    """Get the APPLESOSS reference filename used by v1 SpecProfileStep."""
    return 'APPLESOSS_ref_2D_profile_{}_os1_pad{}.fits'.format(subarray, SPECPROFILE_PAD)


def subarray_from_shape(dimy):
    """Return SUBSTRIP96 for 96 detector rows, otherwise SUBSTRIP256."""
    return 'SUBSTRIP96' if int(dimy) == 96 else 'SUBSTRIP256'


def validate_width(width, extract_width_soss2=None, warn=True):
    """Validate the SOSS aperture width using the v1 ATOCA constraints.

    Parameters
    ----------
    width : float
        SOSS extraction aperture width in pixels.
    extract_width_soss2 : None, float
        Requested order-2 width; ATOCA uses the same width for both orders.
    warn : bool
        If True, warn when a separate order-2 width is supplied.

    Returns
    -------
    width : float
        Validated aperture width in pixels.
    """
    if isinstance(width, str):
        if width == 'optimize':
            raise ValueError('Aperture optimization not possible with ATOCA extraction.')
        raise ValueError(f'ATOCA extract_width must be numeric, got {width!r}')
    if isinstance(width, (dict, list, tuple)) or np.ndim(width) != 0:
        raise ValueError(f'ATOCA extract_width must be a scalar, got {width!r}')
    if warn and extract_width_soss2 is not None:
        warnings.warn('Order 2 cannot use a different width for ATOCA '
                      'extraction.', RuntimeWarning, stacklevel=2)
    value = float(np.asarray(width))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'ATOCA extract_width must be positive, got {width!r}')
    return value


def v1_extract1d_kwargs(width, specprofile, *, estimate=None, tikfac=None,
                        references=None, crds_parameters=None):
    """Build the keyword arguments used by v1 atoca_extract_soss.

    Parameters
    ----------
    width : float
        SOSS extraction aperture width in pixels.
    specprofile : str
        Path to the APPLESOSS or CRDS spatial profile reference.
    estimate : None, str
        Path to the initial SOSS flux estimate.
    tikfac : None, float
        Tikhonov regularization factor supplied to the extraction.
    references, crds_parameters : None, dict
        Resolved CRDS reference paths and step parameter overrides, respectively.

    Returns
    -------
    kwargs : dict
        Explicit extraction keywords and resolved CRDS parameter overrides.
    """
    kwargs = dict(crds_parameters or {})
    kwargs.update(V1_STEP_KWARGS)
    kwargs['soss_width'] = width
    kwargs['override_specprofile'] = os.fspath(specprofile)
    kwargs['soss_estimate'] = None if estimate is None else os.fspath(estimate)
    if tikfac is not None:
        kwargs['soss_tikfac'] = float(tikfac)
    for reftype, path in (references or {}).items():
        if reftype in ('pastasoss', 'speckernel'):
            kwargs[f'override_{reftype}'] = os.fspath(path)
    return kwargs


def unpack_atoca_spectra(multi_spec, quantities=('WAVELENGTH', 'FLUX', 'FLUX_ERROR')):
    """Unpack wavelength and flux arrays by spectral order.

    Parameters
    ----------
    multi_spec : datamodel
        JWST spectra containing wavelength, flux, and uncertainty tables.
    quantities : tuple[str]
        Spectrum table columns to unpack.

    Returns
    -------
    orders : dict
        Spectrum quantity arrays by order, with integration and wavelength axes.
    """
    all_spec = {order: {quantity: [] for quantity in quantities} for order in (1, 2, 3)}
    for spec in multi_spec.spec:
        order = int(spec.spectral_order)
        for quantity in quantities:
            values = np.array(spec.spec_table[quantity])
            if values.ndim == 2:
                all_spec[order][quantity].extend(values)
            else:
                all_spec[order][quantity].append(values)
    return {order: {key: np.array(value) for key, value in columns.items()}
            for order, columns in all_spec.items() if columns[quantities[0]]}


def estimate_spec_table(atoca_outputs):
    """Get the first OBSERVATION spectrum from the ATOCA diagnostics."""
    for spec in atoca_outputs.spec:
        if spec.meta.soss_extract1d.type == 'OBSERVATION':
            return np.array(spec.spec_table)
    raise ValueError('ATOCA outputs contain no OBSERVATION spectrum')


def write_soss_estimate(spec_table, path):
    """Save the supplied SOSS flux-estimate table as a SpecModel and return its path."""
    from stdatamodels.jwst import datamodels
    estimate = datamodels.SpecModel(spec_table=spec_table)
    try:
        return estimate.save(os.fspath(path))
    finally:
        estimate.close()


def merge_segment_orders(parts):
    """Concatenate ordered integration chunks of one segment as ``{order: {quantity: array}}``."""
    if not parts:
        raise ValueError('no ATOCA chunks to merge')
    return {order: {key: np.concatenate([part[order][key] for part in parts], axis=0)
                    for key in parts[0][order]} for order in parts[0]}


def format_atoca_products(segment_orders, sigma_clip):
    """Format ATOCA spectra with the v1 order and temporal clipping.

    Parameters
    ----------
    segment_orders : list[dict]
        Extracted order arrays for each segment, in exposure order.
    sigma_clip : callable
        Temporal clipping function applied to flux arrays.

    Returns
    -------
    spectra, products : dict
        Clipped order spectra and wavelength/flux/uncertainty output arrays, respectively.
    """
    spectra, products = {}, {}
    for order in (1, 2):
        wave2d, flux, ferr = (np.concatenate([seg[order][key] for seg in segment_orders])
                              for key in ('WAVELENGTH', 'FLUX', 'FLUX_ERROR'))
        wave1d = wave2d[0][::-1]
        flux = flux[:, ::-1]
        ferr = ferr[:, ::-1]
        flux_clip = sigma_clip(flux)
        spectra[order] = (flux_clip, ferr)
        products[order] = {'wave': wave1d, 'flux': flux_clip, 'ferr': ferr}
    return spectra, products


def plan_tail_chunks(nints, chunk_ints):
    """Return zero-based chunk bounds for integrations after the first calibration integration."""
    nints = int(nints)
    chunk_ints = max(1, int(chunk_ints))
    return [(lo, min(lo + chunk_ints, nints)) for lo in range(1, nints, chunk_ints)]


def auto_chunk_ints(tail_ints, workers):
    """Return a 1-16 integration chunk size giving about four tail tasks per worker."""
    if tail_ints <= 0:
        return 1
    return int(min(16, max(1, math.ceil(tail_ints / (4 * max(1, workers))))))


def available_cpus():
    """Count available CPUs using process affinity, falling back to the host CPU count."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # pragma: no cover
        return max(1, os.cpu_count() or 1)


def resolve_parallelism(opts, n_tasks=None):
    """Return ``(workers, solver_threads)`` from YAML options.

    Use v2_atoca_workers and v2_atoca_solver_threads, capped by available CPUs and tasks.

    Parameters
    ----------
    opts : dict
        Reduction configuration.
    n_tasks : None, int
        Number of extraction tasks used to cap the worker count.

    Returns
    -------
    workers, solver_threads : int
        Worker processes and Tikhonov solver threads per worker, respectively.
    """
    cpus = available_cpus()
    workers = opts.get('v2_atoca_workers')
    workers = min(4, cpus) if workers in (None, 'auto') else int(workers)
    if workers < 1:
        raise ValueError('v2_atoca_workers must be a positive integer')
    if n_tasks is not None:
        workers = max(1, min(workers, int(n_tasks)))
    threads = opts.get('v2_atoca_solver_threads')
    threads = (max(1, cpus // workers) if threads in (None, 'auto') else int(threads))
    if threads < 1:
        raise ValueError('v2_atoca_solver_threads must be a positive integer')
    return workers, threads


def _check_jwst_version():
    """Check the installed JWST release against the supported extraction version."""
    import jwst
    if jwst.__version__ != JWST_VERSION:
        warnings.warn(f'exotedrf.v2.atoca transcribes jwst {JWST_VERSION} '
            f'Extract1dStep._extract_soss; installed jwst is '
            f'{jwst.__version__}. Results may differ from v1 on the pinned '
            'environment.', RuntimeWarning, stacklevel=3)
        return False
    return True


def make_step(kwargs):
    """Instantiate Extract1dStep exactly as ``Step.call`` configures it."""
    from jwst.extract_1d import Extract1dStep
    config, _ = Extract1dStep.build_config(None, **kwargs)
    config.pop('class', None)
    return Extract1dStep.from_config_section(config)


def extract_soss_in_memory(step, model):
    """Run the installed ``Extract1dStep._extract_soss`` without file writes.

    Parameters
    ----------
    step : Extract1dStep
        Configured JWST SOSS extraction step.
    model : datamodel
        JWST observation model used for extraction or CRDS matching.

    Returns
    -------
    result, ref_outputs, atoca_outputs : datamodel
        Extracted spectra, detector order models, and ATOCA diagnostics, respectively.
    """
    from jwst.extract_1d.soss_extract import soss_extract
    if model.meta.instrument.filter != 'CLEAR':
        raise ValueError('The SOSS extraction is implemented for the CLEAR filter only; '
            f'requested filter is {model.meta.instrument.filter}.')
    subarray = model.meta.subarray.name
    if subarray not in ('SUBSTRIP256', 'SUBSTRIP96'):
        raise ValueError('The SOSS extraction is implemented for the SUBSTRIP256 and '
            f'SUBSTRIP96 subarrays only; subarray is {subarray}.')
    pastasoss_ref_name = step.get_reference_file(model, 'pastasoss')
    specprofile_ref_name = step.get_reference_file(model, 'specprofile')
    speckernel_ref_name = step.get_reference_file(model, 'speckernel')
    soss_kwargs = dict(order_3=step.soss_order_3, threshold=step.soss_threshold,
        n_os=step.soss_n_os, tikfac=step.soss_tikfac, width=step.soss_width,
        bad_pix=step.soss_bad_pix, subtract_background=step.subtract_background,
        rtol=step.soss_rtol, max_grid_size=step.soss_max_grid_size,
        wave_grid_in=step.soss_wave_grid_in, wave_grid_out=step.soss_wave_grid_out,
        estimate=step.soss_estimate, atoca=step.soss_atoca, model=True)
    return soss_extract.run_extract1d(
        model, pastasoss_ref_name, specprofile_ref_name, speckernel_ref_name,
        subarray, 'CLEAR', soss_kwargs)


@contextlib.contextmanager
def model_image_hook(wave_grid=None):
    """Capture the first integration factor and wavelength grid during extraction.

    Restore the original JWST integration function after the extraction call.

    Parameters
    ----------
    wave_grid : None, array-like(float)
        Wavelength grid to reuse if the extraction has no supplied grid.

    Yields
    ------
    captured : dict
        Tikhonov factor and wavelength grid from the first integration.
    """
    from jwst.extract_1d.soss_extract import soss_extract
    name = '_process_one_integration'
    original = getattr(soss_extract, name)
    captured = {}

    def hooked(*args, **kwargs):
        """Capture the first extraction factor and grid while reusing a supplied grid."""
        if wave_grid is not None and kwargs.get('wave_grid') is None:
            kwargs['wave_grid'] = wave_grid
        result = original(*args, **kwargs)
        if 'tikfac' not in captured:
            captured['tikfac'] = float(result[3]['Order 1'])
            captured['wave_grid'] = np.array(result[4], dtype=np.float64, copy=True)
        return result
    setattr(soss_extract, name, hooked)
    try:
        yield captured
    finally:
        setattr(soss_extract, name, original)


def _test_factors_threaded(self, factors, *, threads):
    """Run independent Tikhonov factor solves concurrently."""
    from jwst.extract_1d.soss_extract import atoca_utils
    atoca_utils.log.info('Testing factors...')
    b_vec = self.b_vec
    a_mat = self.a_mat
    t_mat = self.t_mat
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix='atoca-tikho') as pool:
        sln = list(pool.map(self.solve, list(factors)))
    err = [a_mat.dot(solution) - b_vec for solution in sln]
    err = [np.array(value).flatten() for value in err]
    reg = [t_mat.dot(solution) for solution in sln]
    atoca_utils.log.info('{}/{}'.format(len(factors), len(factors)))
    return atoca_utils.TikhoTests({'factors': factors, 'solution': np.array(sln),
                                   'error': np.array(err), 'reg': np.array(reg)})


@contextlib.contextmanager
def threaded_tikhonov_tests(threads):
    """Run Tikhonov factor tests concurrently on supported JWST releases.

    Parameters
    ----------
    threads : None, int
        Number of concurrent Tikhonov factor solves.

    Yields
    ------
    active : bool
        Whether the threaded solver patch is active.
    """
    if threads is None or int(threads) <= 1:
        yield False
        return
    import jwst
    if jwst.__version__ != JWST_VERSION:
        yield False
        return
    from jwst.extract_1d.soss_extract import atoca_utils
    original = atoca_utils.Tikhonov.test_factors

    def patched(self, factors):
        """Test Tikhonov factors with the requested worker threads."""
        return _test_factors_threaded(self, factors, threads=int(threads))
    atoca_utils.Tikhonov.test_factors = patched
    try:
        yield True
    finally:
        atoca_utils.Tikhonov.test_factors = original


def run_task(task):
    """Execute an ATOCA task and return its spectra and diagnostics.

    Worker exceptions are returned as text for the scheduler to classify.

    Parameters
    ----------
    task : dict
        Segment arrays, observing headers, and extraction options.

    Returns
    -------
    result : dict
        Extracted orders and diagnostics, or error text and traceback if the task failed.
    """
    started = time.perf_counter()
    try:
        _check_jwst_version()
        model = stage2_models.stage2_datamodel(task['data'], task['err'], task['dq'],
            fits.Header.fromstring(task['header']),
            int_times=task.get('int_times'), int_start=task['int_start'])
        step = make_step(task['kwargs'])
        with threaded_tikhonov_tests(task.get('solver_threads')), \
                model_image_hook(task.get('wave_grid')) as captured:
            result, ref_outputs, atoca_outputs = extract_soss_in_memory(step, model)
        out = {'orders': unpack_atoca_spectra(result), 'tikfac': captured.get('tikfac')}
        if task.get('head'):
            out['wave_grid'] = captured['wave_grid']
            if task.get('make_estimate', True):
                out['estimate'] = estimate_spec_table(atoca_outputs)
        models = {}
        for local in task.get('model_indices', ()):
            models[int(local)] = (np.array(ref_outputs.order1[int(local)], dtype=np.float32),
                np.array(ref_outputs.order2[int(local)], dtype=np.float32))
        out['models'] = models
        for item in (result, ref_outputs, atoca_outputs, model):
            with contextlib.suppress(Exception):
                item.close()
    except Exception as error:  # noqa: BLE001
        return {'error': str(error), 'error_type': type(error).__name__,
                'traceback': traceback.format_exc(), 'seconds': time.perf_counter() - started}
    out['seconds'] = time.perf_counter() - started
    return out


def parent_blas_threads():
    """Get the BLAS thread count to use in extraction workers."""
    with contextlib.suppress(Exception):
        import threadpoolctl
        counts = [int(info['num_threads']) for info in threadpoolctl.threadpool_info()
                  if info.get('user_api') == 'blas']
        if counts:
            return max(counts)
    value = os.environ.get('OMP_NUM_THREADS')
    return int(value) if value and value.isdigit() else None


def _worker_init(blas_threads):
    """Set the worker BLAS pool to the parent thread count."""
    if blas_threads is None:
        return
    with contextlib.suppress(Exception):
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=int(blas_threads), user_api='blas')


@dataclasses.dataclass
class AtocaSegment:
    """One original segment, with lazy access to its host arrays."""
    index: int
    nints: int
    header: str
    int_start: int
    int_times: object
    kwargs: dict
    fetch: object


class _InlineExecutor:
    """Serial executor with the ``submit`` contract (workers == 1)."""

    def submit(self, fn, *args):
        """Run the callable inline and return its completed future."""
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args))
        except BaseException as error:  # pragma: no cover
            future.set_exception(error)
        return future

    def shutdown(self, wait=True):
        """Finish the inline executor."""
        return None


@contextlib.contextmanager
def _worker_environment(blas_threads):
    """Temporarily set the environment inherited by extraction workers."""
    values = {'JAX_PLATFORMS': 'cpu', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'}
    if blas_threads is not None:
        values.update({key: str(blas_threads) for key in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def task_executor(workers, threads=1):
    """Yield a spawn-context process pool, or an inline executor.

    Parameters
    ----------
    workers : int
        Number of concurrent workers.
    threads : None, int
        Number of concurrent Tikhonov factor solves.

    Yields
    ------
    executor : object
        Inline or process executor for extraction tasks.
    """
    if workers <= 1:
        yield _InlineExecutor()
        return
    blas_threads = parent_blas_threads()
    with _worker_environment(blas_threads):
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=int(workers),
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_worker_init, initargs=(blas_threads,))
        try:
            yield executor
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


class AtocaTaskError(RuntimeError):
    """An ATOCA worker task failed."""

    def __init__(self, result, label):
        """Record the failed extraction task and its diagnostic text.

        Parameters
        ----------
        result : dict
            Failed extraction task result and diagnostic text.
        label : str
            Segment or integration label for the error message.
        """
        self.result = result
        super().__init__(f'ATOCA {label} failed: {result.get('error_type')}: '
            f'{result.get('error')}\n{result.get('traceback', '')}')


def run_schedule(segments, *, width, specprofile, estimate=None,
                 estimate_path=None, workers=1, solver_threads=1,
                 head_solver_threads=None, other_head_solver_threads=None, chunk_ints=None,
                 model_indices=None, runner=run_task, executor=None, whole_segments=False,
                 log=None):
    """Schedule per-segment ATOCA tasks with shared initial calibration state.

    Without a supplied estimate, the first successful task provides it for later tasks.

    Parameters
    ----------
    segments : list[AtocaSegment]
        Original segments in chronological order.
    width : float
        SOSS extraction aperture width in pixels.
    specprofile : str
        Path to the APPLESOSS or CRDS spatial profile reference.
    estimate, estimate_path : None, str
        Supplied flux-estimate path and destination for an automatically derived estimate.
    workers, solver_threads : int
        Worker processes and Tikhonov solver threads per worker, respectively.
    head_solver_threads : None, int
        Solver threads for the task that supplies the initial estimate.
    other_head_solver_threads : None, int
        Solver threads for the first task of each remaining segment.
    chunk_ints : None, int
        Number of integrations per extraction or transfer chunk.
    model_indices : None, dict
        Local integration indices whose detector models should be returned.
    executor : None, object
        Executor providing submit; create one if omitted.
    whole_segments : bool
        If True, retain each complete segment for JWST representative-integration statistics.
    log, runner : None, callable
        Progress callback and extraction task function, respectively.

    Returns
    -------
    result : dict
        Joined segment orders, factors, flux estimate, detector models, and task timings.
    """
    log = log or (lambda message: None)
    model_indices = {int(k): set(int(i) for i in v) for k, v in (model_indices or {}).items()}
    n_tail = sum(max(0, seg.nints - 1) for seg in segments)
    if chunk_ints in (None, 'auto'):
        chunk_ints = auto_chunk_ints(n_tail, workers)
    chunk_ints = int(chunk_ints)
    if chunk_ints < 1:
        raise ValueError('v2_atoca_chunk_ints must be positive')

    def make_task(seg, lo, hi, *, head, est, tikfac=None, wave_grid=None, threads=solver_threads):
        """Build an extraction task with segment arrays and dependency inputs."""
        data, err, dq = seg.fetch(lo, hi)
        kwargs = dict(seg.kwargs)
        kwargs['soss_estimate'] = None if est is None else os.fspath(est)
        if tikfac is not None:
            kwargs['soss_tikfac'] = float(tikfac)
        wanted = sorted(i - lo for i in model_indices.get(seg.index, ()) if lo <= i < hi)
        int_times = (None if seg.int_times is None else np.asarray(seg.int_times)[lo:hi])
        return {'data': data, 'err': err, 'dq': dq, 'header': seg.header,
                'int_times': int_times, 'int_start': seg.int_start + lo,
                'kwargs': kwargs, 'head': head, 'wave_grid': wave_grid,
                'make_estimate': not whole_segments,
                'solver_threads': threads, 'model_indices': wanted,
                'segment': seg.index, 'lo': lo, 'hi': hi}
    timings = []
    chunks = {seg.index: {} for seg in segments}
    heads = {}
    started = time.perf_counter()

    def record(task_kind, seg, lo, hi, result):
        """Record extraction task timing and integration bounds."""
        timings.append({'kind': task_kind, 'segment': seg.index, 'start': lo, 'stop': hi,
                        'seconds': float(result.get('seconds', np.nan)),
                        'finished_s': time.perf_counter() - started})
    owned = executor is None
    context = (task_executor(workers, solver_threads) if owned
               else contextlib.nullcontext(executor))
    estimate_table = None
    estimate_source = 'user' if estimate is not None else None
    if whole_segments:
        # Retain each segment's representative integration and shared masks.
        with context as pool:
            pending = {}
            for seg in segments:
                task = make_task(seg, 0, seg.nints, head=True, est=estimate, threads=solver_threads)
                pending[seg.index] = pool.submit(runner, task)
            results = []
            for seg in segments:
                result = pending[seg.index].result()
                record('segment', seg, 0, seg.nints, result)
                if 'error' in result:
                    raise AtocaTaskError(result, f'segment {seg.index + 1}')
                results.append(result)
        models = {(seg.index, int(local)): pair for seg, result in zip(segments, results)
                  for local, pair in result.get('models', {}).items()}
        return {'segments': [result['orders'] for result in results],
                'tikfacs': [result['tikfac'] for result in results],
                'estimate': None if estimate is None else os.fspath(estimate),
                'estimate_table': None, 'estimate_source': estimate_source,
                'models': models, 'timings': timings, 'chunk_ints': None,
                'workers': int(workers), 'solver_threads': int(solver_threads),
                'wall_s': time.perf_counter() - started}
    with context as pool:
        # Run initial tasks serially until one supplies a usable flux estimate.
        if estimate is None:
            for seg in segments:
                task = make_task(seg, 0, 1, head=True, est=None, threads=head_solver_threads or
                                 solver_threads)
                result = pool.submit(runner, task).result()
                record('head', seg, 0, 1, result)
                if 'error' in result:
                    if result['error'] == ESTIMATE_FAILURE:
                        log(f'[v2] ATOCA: initial flux estimate failed for '
                            f'segment {seg.index + 1}; trying the next one')
                        continue
                    raise AtocaTaskError(result, f'segment {seg.index + 1} integration 1')
                heads[seg.index] = result
                estimate_table = result['estimate']
                if estimate_path is None:
                    raise ValueError('estimate_path is required when no '
                                     'soss_estimate is supplied')
                estimate = write_soss_estimate(estimate_table, estimate_path)
                estimate_source = f'segment {seg.index + 1}'
                log(f'[v2] ATOCA: soss_estimate from segment {seg.index + 1} -> {estimate}')
                break
            else:
                raise RuntimeError('No segments could be properly extracted.')
        # Queue each segment's first integration before its dependent chunks.
        queue = collections.deque(('head', seg, 0, 1) for seg in segments
            if seg.index not in heads)
        for seg in segments:
            if seg.index in heads:
                queue.extend(('tail', seg, lo, hi) for lo, hi in plan_tail_chunks(seg.nints,
                                                            chunk_ints))
        limit = max(1, 2 * int(workers))
        pending = {}

        def submit(item):
            """Submit the next extraction task with its required calibration state."""
            kind, seg, lo, hi = item
            if kind == 'head':
                task = make_task(seg, lo, hi, head=True, est=estimate,
                                 threads=other_head_solver_threads or solver_threads)
            else:
                head = heads[seg.index]
                task = make_task(seg, lo, hi, head=False, est=estimate, tikfac=head['tikfac'],
                                 wave_grid=head['wave_grid'])
            pending[pool.submit(runner, task)] = item
        while queue or pending:
            while queue and len(pending) < limit:
                submit(queue.popleft())
            done, _ = concurrent.futures.wait(
                list(pending), return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                kind, seg, lo, hi = pending.pop(future)
                result = future.result()
                record(kind, seg, lo, hi, result)
                if 'error' in result:
                    raise AtocaTaskError(result, f'segment {seg.index + 1} integrations '
                        f'{lo + 1}-{hi}')
                if kind == 'head':
                    heads[seg.index] = result
                    # Prioritize chunks whose initial extraction has completed.
                    for chunk in reversed(plan_tail_chunks(seg.nints, chunk_ints)):
                        queue.appendleft(('tail', seg) + chunk)
                else:
                    chunks[seg.index][lo] = result
    merged, tikfacs, models = [], [], {}
    for seg in segments:
        head = heads[seg.index]
        parts = [head['orders']] + [chunks[seg.index][lo]['orders']
                                    for lo in sorted(chunks[seg.index])]
        merged.append(merge_segment_orders(parts))
        tikfacs.append(head['tikfac'])
        for result, lo in [(head, 0)] + [(chunks[seg.index][lo], lo) for lo in chunks[seg.index]]:
            for local, pair in result.get('models', {}).items():
                models[(seg.index, lo + int(local))] = pair
    return {'segments': merged, 'tikfacs': tikfacs,
            'estimate': None if estimate is None else os.fspath(estimate),
            'estimate_table': estimate_table, 'estimate_source': estimate_source, 'models': models,
            'timings': timings, 'chunk_ints': chunk_ints,
            'workers': int(workers), 'solver_threads': int(solver_threads),
            'wall_s': time.perf_counter() - started}


def deepstack_nanmedian(data, row_chunk=8):
    """Calculate the integration median in bounded detector-row chunks.

    Parameters
    ----------
    data : array-like(float)
        Rate cube with integration, detector row, and column axes.
    row_chunk : int
        Number of detector rows to median at a time.

    Returns
    -------
    deepstack : np.ndarray(float)
        Per-pixel median over all integrations.
    """
    import bottleneck as bn
    nints, dimy, dimx = (int(n) for n in data.shape)
    out = None
    for lo in range(0, dimy, int(row_chunk)):
        hi = min(dimy, lo + int(row_chunk))
        block = np.ascontiguousarray(stage2_models._host(data[:, lo:hi]))
        median = bn.nanmedian(block, axis=0)
        if out is None:
            out = np.empty((dimy, dimx), dtype=median.dtype)
        out[lo:hi] = median
    return out


def _github_profile_inputs(ctx, cube):
    """Find trace and wavelength references beside the centroid tracetable."""
    trace_path = ctx.get('centroid_tracetable')
    if not trace_path:
        return None
    directory = os.path.dirname(os.fspath(trace_path))
    small = cube.data.shape[-2] == 96
    wavemap = os.path.join(directory, 'jwst_niriss_wavemap_0020.fits' if small
                           else 'jwst_niriss_wavemap_0022.fits')
    if not (os.path.exists(trace_path) and os.path.exists(wavemap)):
        return None
    return os.fspath(trace_path), wavemap


def build_specprofile(deepstack, tracetable, wavemap, output_dir, empirical=True):
    """Build the APPLESOSS spatial profile from a median detector image.

    Parameters
    ----------
    deepstack : array-like(float)
        Median image or stack of median images by detector group.
    tracetable : None, str
        Path to the SOSS trace-position reference file.
    wavemap : str
        Path to the SOSS wavelength reference file.
    output_dir : str
        Directory for calibration or stellar model files.
    empirical : bool
        If True, build the APPLESOSS empirical spatial profile.

    Returns
    -------
    path : str
        Path to the saved SOSS calibration input.
    """
    from applesoss import applesoss
    output_dir = os.fspath(output_dir)
    if not output_dir.endswith(os.sep):
        output_dir += os.sep
    spat_prof = applesoss.EmpiricalProfile(deepstack, tracetable=tracetable, wavemap=wavemap,
                                           pad=SPECPROFILE_PAD)
    if empirical is False:
        raise NotImplementedError(
            'APPLESOSS empirical=False (WebbPSF wings) is not supported by v2')
    spat_prof.build_empirical_profile(verbose=0)
    subarray = subarray_from_shape(np.shape(deepstack)[0])
    filename = spat_prof.write_specprofile_reference(subarray, output_dir=output_dir)
    return os.path.join(output_dir, filename)


def resolve_references(model, reftypes):
    """Resolve CRDS reference files the way v1's Extract1dStep would."""
    from jwst.extract_1d import Extract1dStep
    step = Extract1dStep()
    return {reftype: step.get_reference_file(model, reftype) for reftype in reftypes}


def crds_step_parameters(model):
    """Read CRDS parameter overrides for the JWST Extract1dStep."""
    from jwst.extract_1d import Extract1dStep
    config = Extract1dStep.get_config_from_reference(model)
    return {key: value for key, value in dict(config).items() if key not in ('class', 'name')}


def state_fingerprint(cube, chunk_ints=32):
    """Content hash of the SCI/ERR/DQ cube and its segment metadata.

    Parameters
    ----------
    cube : RampCube, RateCube
        Observation arrays and metadata.
    chunk_ints : int
        Number of integrations to hash at a time.

    Returns
    -------
    fingerprint : str
        Content digest of the rate arrays and segment metadata.
    """
    digest = hashlib.blake2b(digest_size=20)
    meta = cube.meta
    digest.update(repr((tuple(cube.data.shape), str(cube.data.dtype),
                        tuple(np.asarray(meta.segment_edges).tolist()),
                        getattr(meta, 'subarray', ''))).encode())
    extra = getattr(meta, 'extra', {}) or {}
    for header in tuple(extra.get('segment_headers') or ()):
        digest.update(repr(sorted((str(k), repr(v)) for k, v in dict(header).items()
                                  if str(k) not in ('', 'COMMENT', 'HISTORY'))).encode())
    nints = int(cube.data.shape[0])
    for lo in range(0, nints, int(chunk_ints)):
        hi = min(nints, lo + int(chunk_ints))
        for array in (cube.data, cube.err, cube.dq):
            digest.update(np.ascontiguousarray(stage2_models._host(array[lo:hi])))
    return digest.hexdigest()


def _work_dir(ctx):
    """Get or create the ATOCA product directory."""
    directory = ctx.get('atoca_output_dir')
    if directory is None:
        directory = ctx.get('_atoca_tmpdir')
        if directory is None:
            base = (ctx.get('opts', {}).get('scratch_dir') or
                    os.environ.get('EXOTEDRF_SCRATCH_DIR') or None)
            directory = tempfile.mkdtemp(prefix='exotedrf-atoca-', dir=base)
            ctx['_atoca_tmpdir'] = directory
    Path(directory).mkdir(parents=True, exist_ok=True)
    return Path(directory)


def _resolve_option_path(value, ctx):
    """Resolve an extraction input relative to the configured input directory."""
    path = Path(os.path.expanduser(os.fspath(value)))
    if path.is_absolute() or path.exists():
        return path
    for base in (ctx.get('opts', {}).get('input_dir'), os.getcwd()):
        if base:
            candidate = Path(base) / path
            if candidate.exists():
                return candidate
    return path


def extract_state(state, params, ctx, *, log=print, runner=None):
    """Extract and cache ATOCA spectra from a calibrated rate observation.

    Parameters
    ----------
    state : PipelineState
        Observation arrays and accumulated calibration results.
    params : dict
        Calibration or extraction parameter values.
    ctx : dict
        Reduction context containing options and prepared calibration data.
    log, runner : None, callable
        Progress callback and extraction task function, respectively.

    Returns
    -------
    spectra, products, info : dict
        Extracted order spectra, output arrays, and extraction provenance, respectively.
    """
    from exotedrf.v2 import stages
    runner = run_task if runner is None else runner
    opts = ctx.get('opts', {})
    cube = state.cube
    if getattr(cube, 'data', None) is None or cube.data.ndim != 3:
        raise ValueError('ATOCA extraction requires the Stage-2 rate cube')
    meta = cube.meta
    if not str(getattr(meta, 'mode', 'NIRISS')).upper().startswith('NIRISS'):
        raise ValueError('ATOCA extraction is NIRISS/SOSS only')
    width = params.get('extract_width')
    validate_width(width, opts.get('extract_width_soss2'), warn=True)
    cache = ctx.setdefault('_atoca_cache', {
        'results': collections.OrderedDict(), 'specprofiles': {}, 'references': None})
    fingerprint = state_fingerprint(cube)
    workdir = _work_dir(ctx)
    nints = int(cube.data.shape[0])
    slices = stage2_models.segment_bounds(meta, nints)
    # Build one metadata model per segment for CRDS matching.
    if cache['references'] is None or cache['references'][0] != len(slices):
        per_segment = []
        for index in range(len(slices)):
            probe = stage2_models.state_segment_model(state, index, ctx, start=0, stop=1)
            try:
                refs = resolve_references(probe, ('pastasoss', 'speckernel'))
                pars = crds_step_parameters(probe)
                if index == 0 and _github_profile_inputs(ctx, cube) is None:
                    refs.update(resolve_references(probe, ('spectrace', 'wavemap')))
            finally:
                probe.close()
            per_segment.append((refs, pars))
        cache['references'] = (len(slices), per_segment)
    per_segment = cache['references'][1]
    user_profile = opts.get('soss_specprofile')
    if user_profile is not None:
        specprofile = _resolve_option_path(user_profile, ctx)
        if not specprofile.exists():
            raise FileNotFoundError(f'soss_specprofile not found: {specprofile}')
        profile_source = 'user'
    else:
        specprofile = cache['specprofiles'].get(fingerprint)
        if specprofile is None or not Path(specprofile).exists():
            log('[v2] ATOCA: building the APPLESOSS specprofile from the '
                'median of all integrations')
            refs0 = per_segment[0][0]
            deepstack = deepstack_nanmedian(cube.data)
            # Prefer repository trace and wavelength references for the APPLESOSS profile.
            github = _github_profile_inputs(ctx, cube)
            tracetable, wavemap = (github if github is not None
                else (refs0['spectrace'], refs0['wavemap']))
            specprofile = build_specprofile(deepstack, tracetable, wavemap, workdir)
            cache['specprofiles'][fingerprint] = specprofile
        profile_source = 'applesoss'
    specprofile = Path(specprofile)
    user_estimate = opts.get('soss_estimate')
    estimate = (None if user_estimate is None else _resolve_option_path(user_estimate, ctx))
    key = (fingerprint, float(np.asarray(width)), os.fspath(specprofile),
           None if estimate is None else os.fspath(estimate))
    if key in cache['results']:
        cache['results'].move_to_end(key)
        log(f'[v2] ATOCA: reusing extraction for extract_width={width}')
        return cache['results'][key]
    # Return detector models only for the nine diagnostic integrations.
    plot_ints = None
    model_indices = {}
    if opts.get('do_plots', False):
        plot_ints = np.random.randint(0, nints, DECONTAM_PANELS)
        for global_index in plot_ints:
            seg = next(i for i, sl in enumerate(slices) if sl.start <= int(global_index) < sl.stop)
            model_indices.setdefault(seg, set()).add(int(global_index) - slices[seg].start)
    starts = np.asarray(meta.segment_int_starts)
    segments = []
    for index, seg_slice in enumerate(slices):
        refs, pars = per_segment[index]
        header = stage2_models.segment_header(meta, index)
        int_times = stage2_models.segment_int_times(
            meta, index, seg_slice, int_start=int(starts[index]))

        def fetch(lo, hi, _slice=seg_slice):
            """Read calibrated arrays for the selected segment integrations."""
            a, b = _slice.start + lo, _slice.start + hi
            return (stage2_models._host(cube.data[a:b]), stage2_models._host(cube.err[a:b]),
                    stage2_models._host(cube.dq[a:b]))
        segments.append(AtocaSegment(index=index, nints=seg_slice.stop - seg_slice.start,
            header=header.tostring(), int_start=int(starts[index]), int_times=int_times,
            kwargs=v1_extract1d_kwargs(width, specprofile, references=refs, crds_parameters=pars),
            fetch=fetch))
    whole_segments = False
    if runner is run_task:
        import jwst
        whole_segments = jwst.__version__ == JWST_VERSION
    n_tasks = (len(segments) if whole_segments else
               sum(1 + len(plan_tail_chunks(seg.nints, 1)) for seg in segments))
    workers, threads = resolve_parallelism(opts, n_tasks=n_tasks)
    # Use available solver threads while the initial estimate task runs alone.
    head_threads = (threads if opts.get('v2_atoca_solver_threads') not in (None, 'auto') else
                    max(threads, min(available_cpus(), 20)))
    # Share solver threads among the remaining initial segment tasks.
    other_head_threads = (threads if opts.get('v2_atoca_solver_threads') not in (None, 'auto') else
                          max(threads, min(20, available_cpus() // max(1, len(segments) - 1))))
    log(f'[v2] ATOCA: extract_width={width}, {len(segments)} segment(s), '
        f'{nints} integrations, workers={workers}, solver ' f'threads={threads}')
    result = run_schedule(segments, width=width, specprofile=specprofile, estimate=estimate,
        estimate_path=workdir / 'soss_estimate.fits', workers=workers,
        solver_threads=threads, head_solver_threads=head_threads,
        other_head_solver_threads=other_head_threads, chunk_ints=opts.get('v2_atoca_chunk_ints'),
        model_indices=model_indices, runner=runner, log=log, whole_segments=whole_segments)
    log(f"[v2] ATOCA: done in {result['wall_s']:.1f}s")
    spectra, products = format_atoca_products(result['segments'], functools.partial(
            stages.sigma_clip_lightcurves, thresh=float(opts.get('clip_thresh', 10)), window=10))
    info = {'extract_width': width, 'specprofile': os.fspath(specprofile),
        'specprofile_source': profile_source, 'soss_estimate': result['estimate'],
        'soss_estimate_source': result['estimate_source'], 'tikhonov_factors': result['tikfacs'],
        'timings': result['timings'], 'wall_s': result['wall_s'], 'workers': result['workers'],
        'solver_threads': result['solver_threads'], 'chunk_ints': result['chunk_ints'],
        'work_dir': os.fspath(workdir),}
    if plot_ints is not None:
        panels = []
        for global_index in plot_ints:
            seg = next(i for i, sl in enumerate(slices) if sl.start <= int(global_index) < sl.stop)
            order1, order2 = result['models'][(seg, int(global_index) - slices[seg].start)]
            frame = stage2_models._host(cube.data[int(global_index):int(global_index) + 1])[0]
            error = stage2_models._host(cube.err[int(global_index):int(global_index) + 1])[0]
            panels.append((frame - order1 - order2) / error)
        info['decontam_ints'] = [int(i) for i in plot_ints]
        info['decontam_panels'] = np.asarray(panels, dtype=np.float32)
    output = (spectra, products, info)
    cache['results'][key] = output
    while len(cache['results']) > _RESULT_CACHE_SIZE:
        cache['results'].popitem(last=False)
    return output


def write_atoca_products(state, ctx, paths):
    """Place v1's reusable ATOCA inputs in Stage3 and report their paths.

    Parameters
    ----------
    state : PipelineState
        Observation arrays and accumulated calibration results.
    ctx : dict
        Reduction context containing options and prepared calibration data.
    paths : dict
        Output directories for pipeline products.

    Returns
    -------
    written : dict
        Product keys and saved calibration input paths.
    """
    import shutil
    info = state.aux.get('atoca') or {}
    stage3 = Path(paths['stage3'])
    stage3.mkdir(parents=True, exist_ok=True)
    written = {}
    for key, source_key in (('atoca_specprofile', 'specprofile'),
                            ('atoca_soss_estimate', 'soss_estimate')):
        source = info.get(source_key)
        if not source:
            continue
        if source_key == 'specprofile' and \
                info.get('specprofile_source') == 'user':
            continue
        source = Path(source)
        target = stage3 / source.name
        if source.exists() and source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        if target.exists():
            written[key] = str(target)
    return written


def plot_decontamination(info, path):
    """Plot the ATOCA decontamination diagnostic panels.

    Parameters
    ----------
    info : dict
        ATOCA extraction metadata and diagnostic arrays.
    path : str
        Path to the input or output file.

    Returns
    -------
    plotted : bool
        Whether diagnostic panels were available and plotted.
    """
    panels = info.get('decontam_panels')
    if panels is None:
        return False
    from exotedrf import plotting as v1_plotting
    import matplotlib.pyplot as plt
    labels = ['({0})'.format(i) for i in info['decontam_ints']]
    v1_plotting.nine_panel_plot(list(np.asarray(panels)), labels,
                                outfile=os.fspath(path), show_plot=False, vmin=-5, vmax=5)
    plt.close('all')
    return True


__all__ = ['AtocaSegment', 'AtocaTaskError', 'ESTIMATE_FAILURE', 'JWST_VERSION',
    'V1_STEP_KWARGS', 'auto_chunk_ints', 'parent_blas_threads', 'build_specprofile',
    'deepstack_nanmedian', 'estimate_spec_table', 'extract_soss_in_memory',
    'extract_state', 'format_atoca_products', 'make_step',
    'merge_segment_orders', 'model_image_hook', 'plan_tail_chunks',
    'plot_decontamination', 'resolve_parallelism', 'run_schedule',
    'run_task', 'threaded_tikhonov_tests', 'unpack_atoca_spectra',
    'v1_extract1d_kwargs', 'v1_specprofile_name', 'validate_width',
    'write_atoca_products', 'write_soss_estimate',]
