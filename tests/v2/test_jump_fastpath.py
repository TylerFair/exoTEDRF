"""Exact regression tests for Jump's transfer-minimizing fast paths."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from exotedrf.v2 import core, products, stages
from exotedrf.v2.kernels import jump
from exotedrf.v2.pipeline import PipelineState


def _ramp(seed=0x1352770, *, nints=9, ngroups=4, dimy=6, dimx=23):
    """Return ramp."""
    rng = np.random.default_rng(seed)
    base = rng.normal(100., 0.5, (ngroups, dimy, dimx))
    alternate = 0.05 * (-1.) ** np.arange(nints)
    data = base[None] + alternate[:, None, None, None]
    data = data.astype(np.float32)
    last_group = min(2, ngroups - 1)
    last_x = dimx - 4
    data[0, 0, 2, 5] += 20.
    data[-1, last_group, 4, last_x] += 25.
    dq = rng.choice(np.asarray([0, 0, 0, 1, 2, 4, 5], np.uint8),
                    size=data.shape)
    dq[0, 0, 2, 5] = 0
    dq[-1, last_group, 4, last_x] = 0
    return data, dq


def _legacy_chunked(data, dq, threshold, *, window, artifact, group_ok,
                    chunk_size):
    """The pre-fast-path implementation, retained locally as an oracle."""
    floor = np.nanpercentile(
        data, 10., axis=(0, 2, 3)).astype(data.dtype, copy=False)

    def apply(data_chunk, dq_chunk, artifact_chunk):
        return jump.flag_jumps_in_time(
            data_chunk, dq_chunk, threshold, window=window,
            artifact=artifact_chunk, group_ok=group_ok,
            flux_floor=jnp.asarray(floor, dtype=data_chunk.dtype))

    return core.map_over_cols(
        apply, (data, dq, artifact), data.shape[-1],
        chunk_size=chunk_size)


def test_prepared_window_chunks_preserve_global_floor_and_time_edges(monkeypatch):
    """Check prepared window chunks preserve global floor and time edges."""
    data, _ = _ramp()
    data = jnp.asarray(data[:, -1:])
    invariants = jump.prepare_jump_invariants(data)
    expected = jump.prepare_jumps_in_time(data, window=7, invariants=invariants)
    original = jump.prepare_jumps_in_time
    widths = []

    def record(chunk, **kwargs):
        widths.append(chunk.shape[-1])
        assert chunk.shape[0] == data.shape[0]
        return original(chunk, **kwargs)

    monkeypatch.setenv('EXOTEDRF_JUMP_CHUNK_COLS', '5')
    monkeypatch.setattr(jump, 'prepare_jumps_in_time', record)
    scorer = SimpleNamespace(_prepared_window={})
    actual = stages.PreparedJumpScorer._prepared_for_window(
        scorer, 0, data, invariants, 7)
    assert widths == [5, 5, 5, 5, 3]
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=1e-6, equal_nan=True)


def test_dq_only_chunk_path_is_bit_exact_to_legacy_tuple_path():
    """Check DQ only chunk path is bit exact to legacy tuple path."""
    data, dq = _ramp()
    artifact = np.zeros((data.shape[0], data.shape[2], data.shape[3]), bool)
    artifact[0, 2, 5] = True
    artifact[3, 1, 7:11] = True
    group_ok = jnp.asarray([True, False, True, True])

    legacy_data, legacy_dq = _legacy_chunked(
        data, dq, np.float32(7.), window=5, artifact=artifact,
        group_ok=group_ok, chunk_size=7)
    fast_data, fast_dq = jump.flag_jumps_in_time_chunked(
        data, dq, np.float32(7.), window=5, artifact=artifact,
        group_ok=group_ok, chunk_size=7)

    # More-than-two-group Jump is DQ-only by definition.
    assert fast_data is data
    np.testing.assert_array_equal(fast_data, legacy_data)
    np.testing.assert_array_equal(fast_dq, legacy_dq)


def test_two_group_replacement_still_uses_legacy_science_path():
    """Check two group replacement still uses legacy science path."""
    data, dq = _ramp(7, ngroups=2)
    artifact = np.zeros((data.shape[0], data.shape[2], data.shape[3]), bool)
    legacy_data, legacy_dq = _legacy_chunked(
        data, dq, np.float32(6.), window=5, artifact=artifact,
        group_ok=None, chunk_size=7)
    fast_data, fast_dq = jump.flag_jumps_in_time_chunked(
        data, dq, np.float32(6.), window=5, artifact=artifact,
        chunk_size=7)

    np.testing.assert_array_equal(fast_data, legacy_data)
    np.testing.assert_array_equal(fast_dq, legacy_dq)


def test_step_jump_multisegment_dq_only_assembly_is_exact_and_keeps_sci(
        monkeypatch):
    """Check step jump multisegment DQ only assembly is exact and keeps sci."""
    data, dq = _ramp(nints=9)
    meta = core.ObsMeta(
        mode='NIRISS/SOSS', detector='NIS', subarray='SUBSTRIP96',
        frame_time=2.214, ngroups=data.shape[1],
        int_times=np.arange(data.shape[0], dtype=float),
        baseline_ints=np.arange(data.shape[0]),
        segment_edges=np.asarray([5, 9]),
        filenames=('seg1.fits', 'seg2.fits'),
        extra={'segment_int_starts': (250, 300)})
    cube = core.RampCube(
        data, dq, np.zeros(data.shape[-2:], np.uint32), meta)
    params = {'time_jump_threshold': 7., 'time_window': 5}
    ctx = {'opts': {'flag_up_ramp': False, 'flag_in_time': True}}

    expected_dq = []
    for segment, int_start in zip(core.segment_slices(meta, data.shape[0]),
                                  meta.segment_int_starts):
        artifact = jump.reset_artifact_mask(
            segment.stop - segment.start, data.shape[-2], data.shape[-1],
            int_start=int(int_start), instrument='NIRISS')
        _, part_dq = _legacy_chunked(
            data[segment], dq[segment], np.float32(7.), window=5,
            artifact=artifact, group_ok=None, chunk_size=7)
        expected_dq.append(part_dq)

    monkeypatch.setenv('EXOTEDRF_JUMP_CHUNK_COLS', '7')
    result = stages.step_jump(PipelineState(cube), params, ctx)

    assert result.cube.data is data
    np.testing.assert_array_equal(result.cube.data, data)
    np.testing.assert_array_equal(
        result.cube.groupdq, np.concatenate(expected_dq, axis=0))


def test_jump_diagnostic_chunk_reduction_matches_monolithic_uint32(tmp_path):
    """Check jump diagnostic chunk reduction matches monolithic uint32."""
    rng = np.random.default_rng(44)
    shape = (17, 4, 3, 5)
    old = rng.integers(0, 2**16, shape, dtype=np.uint32)
    new = old.copy()
    # Toggle JUMP_DET in both directions and retain unrelated high bits.
    new[::3, 1, 0, 2] ^= np.uint32(core.DQ_JUMP_DET)
    new[1::4, 3, 2, 4] ^= np.uint32(core.DQ_JUMP_DET)
    new[2::5, 0, 1, 1] ^= np.uint32(1 << 15)
    expected = np.sum(
        (((old ^ new) & np.uint32(core.DQ_JUMP_DET)) != 0),
        axis=(0, 1), dtype=np.uint32)

    paths = products.output_layout({'name_tag': ''}, output_dir=tmp_path)
    capture = products.CompatibilityCapture(
        paths, {'opts': {'do_plots': True}})
    before = SimpleNamespace(cube=SimpleNamespace(groupdq=old))
    after = SimpleNamespace(cube=SimpleNamespace(groupdq=new))
    capture(SimpleNamespace(name='JumpStep'), before, after)

    assert capture.records['JumpStep'].dtype == np.uint32
    np.testing.assert_array_equal(capture.records['JumpStep'], expected)


def test_prepared_scorer_transfers_only_final_group(monkeypatch):
    """Check prepared scorer transfers only final group."""
    data, dq = _ramp(nints=9, dimx=12)
    meta = core.ObsMeta(
        mode='NIRISS/SOSS', detector='NIS', subarray='SUBSTRIP96',
        frame_time=2.214, ngroups=data.shape[1],
        int_times=np.arange(data.shape[0], dtype=float),
        baseline_ints=np.arange(data.shape[0]),
        segment_edges=np.asarray([5, 9]),
        filenames=('seg1.fits', 'seg2.fits'),
        extra={'segment_int_starts': (250, 300)})
    state = PipelineState(core.RampCube(
        data, dq, np.zeros(data.shape[-2:], np.uint32), meta))
    ctx = {
        'opts': {'flag_up_ramp': False, 'flag_in_time': True,
                 'extract_width_soss2': None,
                 'wave_range': None, 'w1': 1., 'w2': 1.},
        'centroids': {'ypos o1': np.full(data.shape[-1], 3.)},
        'waves': {1: np.linspace(.9, 2., data.shape[-1])},
    }

    original = stages.jax.device_get
    transferred_shapes = []

    def record(value):
        transferred_shapes.append(value.shape)
        return original(value)

    monkeypatch.setattr(stages.jax, 'device_get', record)
    scorer = stages.prepare_jump_scorer(
        state, {'time_jump_threshold': 7., 'time_window': 5,
                'extract_width': 4.}, ctx)

    assert scorer is not None
    assert transferred_shapes == [
        (data.shape[0], data.shape[2], data.shape[3]),
        (data.shape[0], data.shape[2], data.shape[3]),
    ]
