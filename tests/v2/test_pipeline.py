"""Check pipeline."""

import jax
import numpy as np
import pytest
import weakref

from exotedrf.v2.pipeline import CheckpointStore, Pipeline, PipelineState, Step


def _mkstate(val):
    """Return mkstate."""
    return PipelineState(cube=np.full((3, 4), val, np.float32), aux={'n': 0})


def _add(amount_key):
    """Return add."""
    def fn(state, params, ctx):
        return PipelineState(cube=state.cube + params[amount_key],
                             aux=dict(state.aux, n=state.aux['n'] + 1))
    return fn


def _pipe():
    """Return pipe."""
    steps = [Step('a', _add('pa'), ('pa',)),
             Step('b', _add('pb'), ('pb',)),
             Step('c', _add('pc'), ('pc',))]
    return Pipeline(steps, 'NIRISS/SOSS', ctx={})


def test_run_and_snapshot_isolation():
    """Check run and snapshot isolation."""
    pipe = _pipe()
    store = CheckpointStore()
    out = pipe.run(_mkstate(0.), {'pa': 1., 'pb': 10., 'pc': 100.},
                   store=store)
    assert out.cube[0, 0] == 111.
    assert out.aux['n'] == 3
    assert store.get('b').cube[0, 0] == 1.
    out.cube[:] = -1
    assert store.get('c').cube[0, 0] == 11.


def test_run_observer_receives_each_before_and_after_state_once():
    """Check run observer receives each before and after state once."""
    pipe = _pipe()
    calls = []

    def observer(step, before, after):
        calls.append((step.name, float(before.cube[0, 0]),
                      float(after.cube[0, 0])))

    pipe.run(_mkstate(0.), {'pa': 1., 'pb': 10., 'pc': 100.},
             observer=observer)

    assert calls == [('a', 0., 1.), ('b', 1., 11.), ('c', 11., 111.)]


def test_snapshot_recursively_detaches_nested_aux_arrays():
    """Check snapshot recursively detaches nested aux arrays."""
    nested = np.arange(3, dtype=np.float32)
    state = PipelineState(
        cube=np.zeros(1, np.float32),
        aux={'orders': {1: {'flux': nested}}},
    )
    snapshot = state.snapshot()
    nested[:] = -1
    state.aux['orders'][1]['flux'] = np.zeros(3, np.float32)

    np.testing.assert_array_equal(
        snapshot.aux['orders'][1]['flux'], np.arange(3, dtype=np.float32))
    assert snapshot.aux['orders'] is not state.aux['orders']


def test_rerun_only_downstream():
    """Check rerun only downstream."""
    pipe = _pipe()
    store = CheckpointStore()
    params = {'pa': 1., 'pb': 10., 'pc': 100.}
    pipe.run(_mkstate(0.), params, store=store)
    # Sweep pb: rerun must start from snapshot-before-b (value 1.).
    trial = pipe.rerun_for(['pb'], {**params, 'pb': 20.}, store)
    assert trial.cube[0, 0] == 121.
    # Only steps b and c executed in the trial.
    assert trial.aux['n'] == 3
    # Store untouched by trials.
    assert store.get('b').cube[0, 0] == 1.


def test_commit_refreshes_downstream_snapshots():
    """Check commit refreshes downstream snapshots."""
    pipe = _pipe()
    store = CheckpointStore()
    params = {'pa': 1., 'pb': 10., 'pc': 100.}
    pipe.run(_mkstate(0.), params, store=store)
    pipe.commit_from(['pb'], {**params, 'pb': 20.}, store)
    assert store.get('c').cube[0, 0] == 21.
    assert store.get('b').cube[0, 0] == 1.


def test_mode_filtering_and_stop():
    """Check mode filtering and stop."""
    steps = [Step('a', _add('pa'), ('pa',)),
             Step('m', _add('pb'), ('pb',), modes=('MIRI',)),
             Step('c', _add('pc'), ('pc',))]
    pipe = Pipeline(steps, 'NIRISS/SOSS', ctx={})
    assert [s.name for s in pipe.steps] == ['a', 'c']
    out = pipe.run(_mkstate(0.), {'pa': 1., 'pc': 100.}, stop=1)
    assert out.cube[0, 0] == 1.


def test_opt_in_step_profile_waits_and_reports(monkeypatch, capsys):
    """Check opt in step profile waits and reports."""
    monkeypatch.setenv('EXOTEDRF_PROFILE_STEPS', '1')
    pipe = Pipeline([Step('profiled', _add('pa'), ('pa',))],
                    'NIRISS/SOSS', ctx={})

    out = pipe.run(_mkstate(0.), {'pa': 2.})

    assert out.cube[0, 0] == 2.
    line = capsys.readouterr().out.strip()
    assert line.startswith('[v2-profile] profiled: ')
    assert line.split(';')[0].endswith('s')
    assert 'filesystem read=' in line and 'write=' in line


def test_keep_policy_limits_snapshots():
    """Check keep policy limits snapshots."""
    pipe = _pipe()
    store = CheckpointStore(keep=['b'])
    pipe.run(_mkstate(0.), {'pa': 1., 'pb': 10., 'pc': 100.}, store=store)
    assert 'b' in store and 'a' not in store and 'c' not in store
    assert store.nbytes() > 0


def test_checkpoint_nbytes_counts_nested_aux_arrays():
    """Check checkpoint nbytes counts nested aux arrays."""
    state = PipelineState(
        cube=np.zeros(2, np.float32),
        aux={'nested': {'array': np.zeros(3, np.float64)}},
    )
    store = CheckpointStore()
    store.put('state', state)
    assert store.nbytes() == 2 * 4 + 3 * 8


def test_device_store_copies_nested_state_and_reports_residency():
    """Check device store copies nested state and reports residency."""
    cube = np.arange(6, dtype=np.float32).reshape(2, 3)
    aux_array = np.arange(2, dtype=np.int16)
    state = PipelineState(cube=cube, aux={'nested': {'a': aux_array}})
    store = CheckpointStore.device_resident(device=jax.devices()[0])
    store.put('state', state)

    saved = store.get('state')
    assert isinstance(saved.cube, jax.Array)
    assert isinstance(saved.aux['nested']['a'], jax.Array)
    assert store.location('state') == 'device'
    assert store.device_nbytes() == cube.nbytes + aux_array.nbytes
    assert store.host_nbytes() == 0

    cube[:] = -1
    aux_array[:] = -1
    np.testing.assert_array_equal(np.asarray(saved.cube),
                                  np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(np.asarray(saved.aux['nested']['a']),
                                  np.arange(2))


def test_device_if_fits_falls_back_per_checkpoint_and_reuses_drop_budget():
    """Check device if FITS falls back per checkpoint and reuses drop budget."""
    state = PipelineState(cube=np.arange(2, dtype=np.float32))
    store = CheckpointStore.device_if_fits(
        max_device_bytes=state.cube.nbytes, device=jax.devices()[0])

    store.put('first', state)
    store.put('second', state)
    assert store.location('first') == 'device'
    assert store.location('second') == 'host'
    assert isinstance(store.get('first').cube, jax.Array)
    assert isinstance(store.get('second').cube, np.ndarray)
    assert store.device_nbytes() == state.cube.nbytes
    assert store.host_nbytes() == state.cube.nbytes

    store.drop('first')
    store.put('third', state)
    assert store.location('third') == 'device'


def test_default_store_remains_host_resident():
    """Check default store remains host resident."""
    store = CheckpointStore()
    store.put('state', _mkstate(1.))
    assert store.location('state') == 'host'
    assert isinstance(store.get('state').cube, np.ndarray)
    assert store.device_nbytes() == 0


def test_snapshot_spills_large_arrays_without_losing_copy_isolation(
        monkeypatch, tmp_path):
    """Check snapshot spills large arrays without losing copy isolation."""
    monkeypatch.setenv('EXOTEDRF_MAX_HOST_BYTES', '1B')
    monkeypatch.setenv('EXOTEDRF_SCRATCH_DIR', str(tmp_path))
    data = np.arange(30, dtype=np.float64).reshape(5, 6)
    nested = np.arange(7, dtype=np.uint32)
    snapshot = PipelineState(data, {'nested': {'dq': nested}}).snapshot()
    assert isinstance(snapshot.cube, np.memmap)
    assert isinstance(snapshot.aux['nested']['dq'], np.memmap)
    data[:] = -1
    nested[:] = 0
    np.testing.assert_array_equal(snapshot.cube,
                                  np.arange(30).reshape(5, 6))
    np.testing.assert_array_equal(snapshot.aux['nested']['dq'], np.arange(7))
    assert snapshot.cube.dtype == np.float64
    assert snapshot.aux['nested']['dq'].dtype == np.uint32


def test_checkpoint_copies_and_accounts_for_shared_array_only_once():
    """Check checkpoint copies and accounts for shared array only once."""
    data = np.arange(10, dtype=np.float32)
    state = PipelineState(data, {'same_data': data})
    store = CheckpointStore()
    store.put('shared', state)
    snapshot = store.get('shared')
    assert store.nbytes() == data.nbytes
    assert snapshot.cube is snapshot.aux['same_data']
    data[:] = -1
    np.testing.assert_array_equal(snapshot.cube, np.arange(10))


def test_saving_existing_snapshot_does_not_copy_again(monkeypatch):
    """Check saving existing snapshot does not copy again."""
    store = CheckpointStore()
    store.put('restart', _mkstate(1.))
    saved = store.get('restart')

    def forbidden_copy(*args, **kwargs):
        raise AssertionError('the same saved checkpoint must not be recopied')

    monkeypatch.setattr(PipelineState, 'snapshot', forbidden_copy)
    store.put('restart', saved)
    assert store.get('restart') is saved


def test_borrowed_checkpoint_reuses_readonly_arrays_and_detaches_containers(
        monkeypatch):
    """Check borrowed checkpoint reuses readonly arrays and detaches containers."""
    data = np.arange(12, dtype=np.float64).reshape(3, 4)
    aux = np.arange(3, dtype=np.uint32)
    state = PipelineState(data, {'nested': {'dq': aux}})

    def forbidden_copy(*args, **kwargs):
        raise AssertionError('borrowed extraction must not snapshot the cube')

    monkeypatch.setattr(PipelineState, 'snapshot', forbidden_copy)
    store = CheckpointStore.borrowed_readonly(keep={'Extract'})
    store.put('Extract', state)
    saved = store.get('Extract')
    assert np.shares_memory(saved.cube, data)
    assert np.shares_memory(saved.aux['nested']['dq'], aux)
    assert saved.aux is not state.aux
    assert saved.aux['nested'] is not state.aux['nested']
    assert not saved.cube.flags.writeable
    assert not saved.aux['nested']['dq'].flags.writeable
    assert data.flags.writeable and aux.flags.writeable
    with pytest.raises(ValueError, match='read-only'):
        saved.cube[0, 0] = 100.
    assert store.location('Extract') == 'borrowed_readonly'
    assert store.borrowed_nbytes() == data.nbytes + aux.nbytes
    assert store.host_nbytes() == store.device_nbytes() == 0
    assert store.nbytes() == store.borrowed_nbytes()
    store.drop('Extract')
    assert store.borrowed_nbytes() == 0


def test_borrowed_checkpoint_requires_explicit_ownership_scope():
    """Check borrowed checkpoint requires explicit ownership scope."""
    with pytest.raises(ValueError, match='explicit step names'):
        CheckpointStore.borrowed_readonly(keep=None)


@pytest.mark.parametrize('mode', ['NIRISS/SOSS', 'NIRSPEC/G395H', 'MIRI/LRS'])
def test_builtin_extract_trials_use_borrowed_cube_without_source_mutations(mode):
    """Check builtin extract trials use borrowed cube without source mutations."""
    from exotedrf.v2 import core, stages

    rng = np.random.default_rng(23)
    shape = (12, 16, 8) if mode.startswith('MIRI') else (12, 8, 16)
    data = (100 + rng.normal(size=shape)).astype(np.float32)
    err = np.full(shape, 2., dtype=np.float32)
    dq = np.zeros(shape, dtype=np.uint32)
    dq[2, 3, 5] = core.DQ_DO_NOT_USE
    meta = core.ObsMeta(
        mode=mode, detector='TEST', subarray='TEST', frame_time=1.,
        ngroups=3, int_times=np.arange(shape[0], dtype=float),
        baseline_ints=np.asarray([3, -3]),
        segment_edges=np.asarray([shape[0]]), filenames=('test.fits',))
    state = PipelineState(core.RateCube(data, err, dq, meta))
    originals = [array.copy() for array in (data, err, dq)]
    ctx = {'opts': {'mask_do_not_use_pixels': True}}
    if mode.startswith('NIRISS'):
        ctx.update(centroids={'ypos o1': np.full(shape[-1], 3.5)},
                   waves={1: np.linspace(1., 2., shape[-1])})
        extract = stages.step_extract
    elif mode.startswith('NIRSPEC'):
        ctx.update(centroids={'xpos': np.arange(shape[-1]),
                              'ypos': np.full(shape[-1], 3.5)},
                   nirspec_xstart=0,
                   nirspec_wave_map=np.broadcast_to(
                       np.linspace(1., 2., shape[-1])[None], shape[-2:]))
        extract = stages.step_extract_nirspec
    else:
        ctx.update(centroids={'xpos': np.full(shape[-2], 3.5),
                              'ypos': np.arange(shape[-2])},
                   miri_wave_map=np.broadcast_to(
                       np.linspace(1., 2., shape[-2])[:, None], shape[-2:]))
        extract = stages.step_extract_miri
    pipe = Pipeline([Step('Extract', extract,
                          ('extract_width',))], meta.mode, ctx)
    store = CheckpointStore.borrowed_readonly(keep={'Extract'})
    pipe.run(state, {'extract_width': 4}, store=store)
    for width in (2, 4, 6):
        params = {'extract_width': width}
        trial = pipe.rerun_for(['extract_width'], params, store)
        expected = extract(state, params, ctx)
        np.testing.assert_array_equal(trial.aux['spectra'][1][0],
                                      expected.aux['spectra'][1][0])
        np.testing.assert_array_equal(trial.aux['spectra'][1][1],
                                      expected.aux['spectra'][1][1])
    for original, current in zip(originals, (data, err, dq)):
        np.testing.assert_array_equal(current, original)
        assert current.flags.writeable
    assert np.shares_memory(store.get('Extract').cube.data, data)


def test_pipeline_releases_previous_cube_before_next_checkpoint():
    """Check pipeline releases previous cube before next checkpoint."""
    previous = []

    def advance(state, params, ctx):
        previous.append(weakref.ref(state.cube))
        return PipelineState(state.cube + 1)

    class ObserveLifetime:
        def put(self, name, state):
            if name == 'second':
                assert previous[0]() is None

    pipe = Pipeline([Step('first', advance),
                     Step('second', lambda state, params, ctx: state)],
                    'NIRISS/SOSS', {})
    result = pipe.run(PipelineState(np.zeros(3, np.float32)), {},
                      store=ObserveLifetime())
    np.testing.assert_array_equal(result.cube, np.ones(3))
