"""Check bench."""

import json

import jax
import jax.numpy as jnp
import pytest

from exotedrf.v2 import bench


def test_benchmark_callable_warmup_timing_and_memory_fields():
    """Check benchmark callable warmup timing and memory fields."""
    fn = jax.jit(lambda value: value * 2 + 1)
    value = jnp.arange(8, dtype=jnp.float32)
    result = bench.benchmark_callable(
        lambda: {'array': fn(value)}, warmups=1, reps=2)

    assert result['warmups'] == 1
    assert len(result['warmup_s']) == 1
    assert len(result['timed_s']) == 2
    assert result['min_s'] <= result['median_s'] <= result['max_s']
    assert 'process_peak_rss_after_bytes' in result
    assert result['result_summary']['array'] == {
        'shape': [8], 'dtype': 'float32'}


def test_full_benchmark_accepts_zero_argument_runner(tmp_path):
    """Check full benchmark accepts zero argument runner."""
    output = tmp_path / 'full.json'
    calls = []

    def runner():
        calls.append(None)
        return {'cost': 0.5, 'winner': {'width': 20}}

    payload = bench.bench_full(
        'unused.yaml', reps=2, warmups=1, runner=runner,
        output_path=output)
    saved = json.loads(output.read_text())

    assert len(calls) == 3
    assert payload['benchmark'] == 'full-optimizer'
    assert payload['context']['config']['name'] == 'unused.yaml'
    assert payload['context']['config']['exists'] is False
    assert 'source_revision' in payload['environment']
    assert 'jax_enable_x64' in payload['environment']
    assert saved['rows'][0]['name'] == 'v2 optimizer end-to-end'
    assert len(saved['rows'][0]['timed_s']) == 2
    assert saved['rows'][0]['result_summary']['winner']['width'] == 20


def test_full_benchmark_passes_config_to_runner(tmp_path):
    """Check full benchmark passes config to runner."""
    seen = []

    def runner(config_path, *, token):
        seen.append((config_path, token))
        return {'summary': 'ok'}

    bench.bench_full(
        'config.yaml', reps=1, warmups=0, runner=runner,
        runner_kwargs={'token': 7}, output_path=tmp_path / 'bench.json')
    assert seen == [('config.yaml', 7)]


def test_path_runner_failure_is_not_chained_to_signature_probe():
    """Check path runner failure is not chained to signature probe."""
    def runner(config_path):
        raise ValueError(f'pipeline failed for {config_path}')

    with pytest.raises(ValueError) as caught:
        bench._call_runner(runner, 'config.yaml', {})
    assert caught.value.__context__ is None


def test_default_full_benchmark_suppresses_repeated_artifact_writes(
        tmp_path, monkeypatch):
    """Check default full benchmark suppresses repeated artifact writes."""
    seen = []

    def runner(config_path, **kwargs):
        seen.append((config_path, kwargs))
        return {'summary': 'ok'}

    monkeypatch.setattr('exotedrf.v2.optimize.run_optimizer', runner)
    monkeypatch.setattr('exotedrf.v2.core.setup', lambda: None)
    bench.bench_full(
        'config.yaml', reps=1, warmups=1,
        output_path=tmp_path / 'bench.json')

    assert len(seen) == 2
    assert all(config == 'config.yaml' for config, _ in seen)
    assert all(kwargs['write_products'] is False and
               kwargs['write_logs'] is False and
               kwargs['logger'] is bench._print_progress
               for _, kwargs in seen)


def test_tiny_microbenchmark_is_cpu_safe(tmp_path):
    """Check tiny microbenchmark is CPU safe."""
    output = tmp_path / 'tiny.json'
    payload = bench.bench_micro(
        1, 2, 4, 6, 1, output_path=output)
    assert output.is_file()
    assert payload['benchmark'] == 'micro'
    assert len(payload['rows']) == 6
    assert all(row['steady_s'] >= 0 for row in payload['rows'])
