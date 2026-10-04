"""Measure reduction runtimes and memory use."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - exercised on Windows only
    resource = None


def sync(value: Any) -> Any:
    """Wait for array calculations in a returned result to finish.

    Parameters
    ----------
    value : object
        Result containing arrays or nested collections.

    Returns
    -------
    value : object
        Input result after all array calculations finish.
    """
    if dataclasses.is_dataclass(value):
        children = (getattr(value, field.name) for field in dataclasses.fields(value))
    elif isinstance(value, Mapping):
        children = value.values()
    elif isinstance(value, (tuple, list)):
        children = value
    elif hasattr(value, 'block_until_ready'):
        value.block_until_ready()
        return value
    else:
        for leaf in jax.tree.leaves(value):
            if hasattr(leaf, 'block_until_ready'):
                leaf.block_until_ready()
        return value
    for item in children:
        sync(item)
    return value


def time_fn(fn: Callable[..., Any], *args: Any, reps: int = 5, **kwargs: Any) -> dict[str, Any]:
    """Measure the first call and the typical time after JAX compilation.

    Parameters
    ----------
    fn : callable
        Function to time.
    reps : int
        Number of timed repetitions.
    args : tuple
        Positional arguments for fn.
    kwargs : dict
        Keyword arguments for fn.

    Returns
    -------
    timings : dict
        First-call and repeated runtimes in seconds.
    """
    if reps < 1:
        raise ValueError('reps must be at least 1')
    t0 = time.perf_counter()
    sync(fn(*args, **kwargs))
    first = time.perf_counter() - t0
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sync(fn(*args, **kwargs))
        times.append(time.perf_counter() - t0)
    return {'first_call_s': float(first), 'steady_s': float(np.median(times)),
        'steady_min_s': float(np.min(times)), 'steady_max_s': float(np.max(times)),
        'reps': int(reps)}


def _process_peak_rss_bytes() -> int | None:
    """Return peak process memory in bytes when resource is available."""
    if resource is None:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Convert Linux and BSD peak memory from KiB to bytes.
    return value if sys.platform == 'darwin' else value * 1024


def _device_memory() -> list[dict[str, Any]]:
    """Collect available memory statistics for each JAX device."""
    snapshots = []
    for device in jax.devices():
        try:
            stats = device.memory_stats()
        except (AttributeError, RuntimeError):
            stats = None
        selected = {}
        if stats:
            for key in ('bytes_in_use', 'peak_bytes_in_use', 'bytes_limit',
                'largest_free_block_bytes'):
                if key in stats and stats[key] is not None:
                    selected[key] = int(stats[key])
        snapshots.append({'device': str(device), 'stats': selected or None})
    return snapshots


def _json_summary(value: Any, *, depth: int = 0) -> Any:
    """Serialize useful optimizer results without embedding science cubes."""
    if depth > 5:
        return repr(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_summary(value.item(), depth=depth + 1)
    if isinstance(value, Mapping):
        return {str(key): _json_summary(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_summary(item, depth=depth + 1) for item in value]
    if dataclasses.is_dataclass(value):
        # Read dataclass fields without copying the science arrays.
        return {field.name: _json_summary(getattr(value, field.name), depth=depth + 1)
            for field in dataclasses.fields(value)}
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return {'shape': list(value.shape), 'dtype': str(value.dtype)}
    return repr(value)


def summarize_result(result: Any) -> Any:
    """Summarize optimizer results without copying science arrays.

    Parameters
    ----------
    result : object
        Optimizer result to summarize.

    Returns
    -------
    summary : object
        JSON-compatible result summary with arrays described by shape and dtype.
    """
    summary = getattr(result, 'summary', None)
    if callable(summary):
        summary = summary()
    if summary is not None:
        return _json_summary(summary)
    if isinstance(result, Mapping) or dataclasses.is_dataclass(result):
        return _json_summary(result)
    public = {}
    for name in ('winners', 'trials', 'cost', 'output_paths'):
        if hasattr(result, name):
            public[name] = _json_summary(getattr(result, name))
    return public or _json_summary(result)


def benchmark_callable(call: Callable[[], Any], *, warmups: int = 1,
        reps: int = 3) -> dict[str, Any]:
    """Measure runtimes and memory use for a zero-argument function.

    Parameters
    ----------
    call : callable
        Zero-argument function to benchmark.
    warmups : int
        Number of warmup runs before timed repetitions.
    reps : int
        Number of timed repetitions.

    Returns
    -------
    result : dict
        Timing, memory, and result summary for the benchmark.
    """
    if warmups < 0:
        raise ValueError('warmups cannot be negative')
    if reps < 1:
        raise ValueError('reps must be at least 1')
    rss_before = _process_peak_rss_bytes()
    device_before = _device_memory()
    warmup_times = []
    result = None
    for _ in range(warmups):
        t0 = time.perf_counter()
        result = sync(call())
        warmup_times.append(time.perf_counter() - t0)
    timed = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = sync(call())
        timed.append(time.perf_counter() - t0)
    rss_after = _process_peak_rss_bytes()
    rss_delta = None
    if rss_before is not None and rss_after is not None:
        rss_delta = max(0, rss_after - rss_before)
    return {'warmups': int(warmups), 'warmup_s': [float(value) for value in warmup_times],
        'reps': int(reps), 'timed_s': [float(value) for value in timed],
        'median_s': float(np.median(timed)), 'min_s': float(np.min(timed)),
        'max_s': float(np.max(timed)), 'process_peak_rss_before_bytes': rss_before,
        'process_peak_rss_after_bytes': rss_after, 'process_peak_rss_growth_bytes': rss_delta,
        'device_memory_before': device_before, 'device_memory_after': _device_memory(),
        'result_summary': summarize_result(result)}


def environment() -> dict[str, Any]:
    """Record the software, hardware, and source revision behind a benchmark.

    Returns
    -------
    context : dict
        Software versions, devices, platform, and source revision.
    """
    from exotedrf.v2 import __version__
    revision = os.environ.get('GITHUB_SHA')
    if not revision:
        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                cwd=Path(__file__).resolve().parents[2], text=True,
                stderr=subprocess.DEVNULL, timeout=2).strip()
        except (OSError, subprocess.SubprocessError):
            revision = None
    return {'backend': jax.default_backend(), 'devices': [str(device) for device in jax.devices()],
        'jax': jax.__version__, 'jaxlib': getattr(jax.lib, '__version__', None),
        'jax_enable_x64': bool(jax.config.jax_enable_x64), 'exotedrf_v2': __version__,
        'source_revision': revision, 'python': platform.python_version(),
        'platform': platform.platform(), 'pid': os.getpid()}


class Report:
    """Collect benchmark rows and save them as a JSON report."""

    def __init__(self, benchmark: str = 'micro', context=None):
        """Initialize the benchmark report."""
        self.benchmark = benchmark
        self.context = dict(context or {})
        self.rows: list[dict[str, Any]] = []

    def add(self, name: str, result: Mapping[str, Any], note: str = '') -> None:
        """Add a benchmark row and print its timing summary.

        Parameters
        ----------
        name : str
            Name of the benchmark row.
        result : dict
            Timing and memory measurements for the row.
        note : str
            Description printed and stored with the row.
        """
        self.rows.append({'name': name, **dict(result), 'note': note})
        if 'first_call_s' in result:
            print(f'  {name:<42s} first {result["first_call_s"]*1e3:9.1f} ms   '
                f'steady {result["steady_s"]*1e3:9.1f} ms  {note}')
        else:
            print(f'  {name:<42s} median {result["median_s"]:9.3f} s   '
                f'min {result["min_s"]:9.3f} s  {note}')

    def as_dict(self) -> dict[str, Any]:
        """Build the benchmark report dictionary.

        Returns
        -------
        report : dict
            Benchmark rows, context, and environment metadata.
        """
        return {'schema_version': 1, 'benchmark': self.benchmark, 'context': self.context,
            'environment': environment(), 'rows': self.rows}

    def save(self, path: str | Path = 'bench_results.json') -> Path:
        """Save the benchmark report as JSON.

        Parameters
        ----------
        path : str, Path
            Path to which to save the JSON report.

        Returns
        -------
        destination : Path
            Path to the saved report.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.as_dict()
        with destination.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
        print(f'\nEnvironment: {payload["environment"]}')
        print(f'Results written to {destination}')
        return destination


def bench_micro(nints: int, ngroups: int, dimy: int, dimx: int, reps: int,
        *, output_path: str | Path = 'bench_results.json') -> dict[str, Any]:
    """Measure core operations without loading a science dataset.

    Parameters
    ----------
    nints, ngroups, dimy, dimx : int
        Synthetic integration, ramp group, detector row, and detector column counts.
    reps : int
        Number of timed repetitions.
    output_path : str, Path
        Path to which to save the benchmark report.

    Returns
    -------
    report : dict
        Synthetic benchmark timings and environment metadata.
    """
    from exotedrf.v2 import core, mathutils as mu
    if min(nints, ngroups, dimy, dimx) < 1:
        raise ValueError('all synthetic dimensions must be positive')
    core.setup()
    rng = np.random.RandomState(0)
    report = Report('micro')
    print(f'\nSOSS-like cube: ({nints}, {ngroups}, {dimy}, {dimx}) float32 '
        f'= {nints*ngroups*dimy*dimx*4/1e9:.2f} GB\n')
    frame = jnp.asarray(rng.randn(dimy, dimx).astype(np.float32))
    cube3 = jnp.asarray(rng.randn(nints, dimy, dimx).astype(np.float32))
    report.add('deepstack: nanmedian over ints (3D)',
        time_fn(jax.jit(lambda cube: jnp.nanmedian(cube, axis=0)), cube3, reps=reps))
    report.add('column median (1/f core op, 3D)',
        time_fn(jax.jit(lambda cube: jnp.nanmedian(cube, axis=-2)), cube3, reps=reps))
    report.add('median filter 5x5 on deepframe',
        time_fn(jax.jit(lambda data: mu.nanmedian_filter_2d(data, 2)), frame, reps=reps))
    centroid = jnp.full((dimx,), dimy / 2, jnp.float32)
    @jax.jit
    def box_extract_like(cube, centroid_y, width):
        """Extract box-aperture flux from a synthetic rate cube."""
        weights = mu.box_aperture_weights(centroid_y, width, dimy)
        return jnp.einsum('nyx,yx->nx', jnp.nan_to_num(cube), weights)
    report.add('box extraction (3D cube -> lightcurves)',
        time_fn(box_extract_like, cube3, centroid, jnp.float32(4.0), reps=reps))
    report.add('box extraction, resweep new width (no recompile)',
        time_fn(box_extract_like, cube3, centroid, jnp.float32(5.0), reps=reps),
        note='aperture width can change without rebuilding the calculation')
    cube4 = rng.randn(nints, ngroups, dimy, dimx).astype(np.float32)
    colmed = jax.jit(lambda cube: cube - jnp.nanmedian(cube, axis=-2, keepdims=True))
    report.add('4D column-median subtract via map_over_ints',
        time_fn(lambda cube: core.map_over_ints(colmed, cube, nints),
        cube4, reps=max(1, reps // 2)))
    report.save(output_path)
    return report.as_dict()


def _call_runner(runner: Callable[..., Any], config_path: str | Path,
        kwargs: Mapping[str, Any]) -> Any:
    """Invoke either a zero-argument injected runner or config-path runner."""
    signature = inspect.signature(runner)
    try:
        signature.bind(**kwargs)
    except TypeError:
        args = (config_path,)
    else:
        args = ()
    if args:
        signature.bind(*args, **kwargs)
    return runner(*args, **kwargs)


def _print_progress(message: object) -> None:
    """Print optimizer progress and flush the output."""
    print(message, flush=True)


def bench_full(config_path: str | Path, reps: int = 3, *, warmups: int = 1,
        runner: Callable[..., Any] | None = None,
        runner_kwargs: Mapping[str, Any] | None = None,
        output_path: str | Path = 'bench_results.json') -> dict[str, Any]:
    """Run and time the full v2 optimizer.

    Parameters
    ----------
    config_path : str, Path
        Path to the optimizer YAML configuration.
    reps : int
        Number of timed repetitions.
    warmups : int
        Number of warmup runs before timed repetitions.
    runner : None, callable
        Optimizer runner accepting config_path or no positional arguments.
    runner_kwargs : None, dict
        Keyword arguments for the optimizer runner.
    output_path : str, Path
        Path to which to save the benchmark report.

    Returns
    -------
    report : dict
        Optimizer runtimes, memory use, and configuration metadata.
    """
    using_default_runner = runner is None
    if using_default_runner:
        from exotedrf.v2 import core
        from exotedrf.v2.optimize import run_optimizer
        core.setup()
        runner = run_optimizer
    kwargs = dict(runner_kwargs or {})
    if using_default_runner:
        # Disable product and log writes during repeated optimizer runs.
        kwargs.setdefault('write_products', False)
        kwargs.setdefault('write_logs', False)
        # Print per-candidate progress during optimizer runs.
        kwargs.setdefault('logger', _print_progress)

    def run_once():
        """Run the configured optimizer once."""
        return _call_runner(runner, config_path, kwargs)
    print(f'\nFull optimizer benchmark: {config_path}\n')
    result = benchmark_callable(run_once, warmups=warmups, reps=reps)
    config_file = Path(config_path)
    config_context = {'name': config_file.name, 'exists': config_file.is_file(), 'sha256': None}
    if config_file.is_file():
        config_context['sha256'] = hashlib.sha256(config_file.read_bytes()).hexdigest()
    report = Report('full-optimizer', context={'config': config_context})
    report.add('v2 optimizer end-to-end', result,
        note=f'{warmups} warmup run(s), {reps} timed run(s)')
    report.save(output_path)
    return report.as_dict()


def build_parser() -> argparse.ArgumentParser:
    """Define command-line choices for tiny, representative, and full benchmarks.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser for tiny, micro, and full benchmark commands.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    micro = sub.add_parser('micro')
    micro.add_argument('--nints', type=int, default=500)
    micro.add_argument('--ngroups', type=int, default=5)
    micro.add_argument('--dimy', type=int, default=256)
    micro.add_argument('--dimx', type=int, default=2048)
    micro.add_argument('--reps', type=int, default=5)
    micro.add_argument('--output', default='bench_results.json')
    tiny = sub.add_parser('tiny', help='CPU-safe microbenchmark smoke run')
    tiny.add_argument('--reps', type=int, default=1)
    tiny.add_argument('--output', default='bench_tiny.json')
    full = sub.add_parser('full')
    full.add_argument('--config', required=True)
    full.add_argument('--warmups', type=int, default=1)
    full.add_argument('--reps', type=int, default=3)
    full.add_argument('--output', default='bench_results.json')
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested benchmark and write its JSON report.

    Parameters
    ----------
    argv : None, list[str]
        Command-line arguments. If None, read sys.argv.

    Returns
    -------
    status : int
        Exit code, equal to zero when the benchmark completes.
    """
    args = build_parser().parse_args(argv)
    if args.command == 'tiny':
        bench_micro(2, 2, 8, 16, args.reps, output_path=args.output)
    elif args.command == 'micro':
        bench_micro(args.nints, args.ngroups, args.dimy, args.dimx, args.reps,
            output_path=args.output)
    else:
        bench_full(args.config, args.reps, warmups=args.warmups, output_path=args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
