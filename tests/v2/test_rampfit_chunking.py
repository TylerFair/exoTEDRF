"""Check rampfit chunking."""

import numpy as np
import pytest

from exotedrf.v2 import core, stages
from exotedrf.v2.core import ObsMeta, RampCube
from exotedrf.v2.pipeline import PipelineState


def _meta(nints, ngroups=3):
    """Return meta."""
    return ObsMeta(
        'NIRISS/SOSS', 'NIS', 'SUBSTRIP256', 2.214, ngroups,
        np.arange(nints, dtype=float), np.asarray([nints]),
        np.asarray([nints]), ('segment.fits',), {})


def test_rampfit_chunk_default_and_environment_precedence(monkeypatch):
    """Check rampfit chunk default and environment precedence."""
    monkeypatch.delenv('EXOTEDRF_RAMPFIT_CHUNK_INTS', raising=False)
    monkeypatch.delenv('EXOTEDRF_CHUNK_INTS', raising=False)
    assert stages._rampfit_integration_chunk_size(105) == 16
    assert stages._rampfit_integration_chunk_size(7) == 7

    monkeypatch.setenv('EXOTEDRF_CHUNK_INTS', '8')
    assert stages._rampfit_integration_chunk_size(105) == 8

    monkeypatch.setenv('EXOTEDRF_RAMPFIT_CHUNK_INTS', '24')
    assert stages._rampfit_integration_chunk_size(105) == 24
    assert stages._rampfit_integration_chunk_size(12) == 12


@pytest.mark.parametrize(
    ('name', 'value'),
    [('EXOTEDRF_RAMPFIT_CHUNK_INTS', '0'),
     ('EXOTEDRF_RAMPFIT_CHUNK_INTS', 'not-an-int'),
     ('EXOTEDRF_CHUNK_INTS', '-2')])
def test_rampfit_chunk_rejects_invalid_environment(monkeypatch, name, value):
    """Check rampfit chunk rejects invalid environment."""
    monkeypatch.delenv('EXOTEDRF_RAMPFIT_CHUNK_INTS', raising=False)
    monkeypatch.delenv('EXOTEDRF_CHUNK_INTS', raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=f'{name} must be a positive integer'):
        stages._rampfit_integration_chunk_size(105)


def test_segment_median_rate_uses_rampfit_specific_cap(monkeypatch):
    """Check segment median rate uses rampfit specific cap."""
    data = np.zeros((5, 3, 2, 4), np.float32)
    groupdq = np.zeros_like(data, np.uint8)
    calls = []

    def fake_median(chunk, dq_chunk, group_time, one_group_time=None):
        calls.append(int(chunk.shape[0]))
        return np.zeros((chunk.shape[0], 2, 4), np.float32)

    monkeypatch.setenv('EXOTEDRF_CHUNK_INTS', '5')
    monkeypatch.setenv('EXOTEDRF_RAMPFIT_CHUNK_INTS', '2')
    monkeypatch.setattr(
        stages.k_ramp, 'median_rates_per_integration', fake_median)
    result = stages._segment_median_rate(data, groupdq, 2.214)

    assert calls == [2, 2, 1]
    np.testing.assert_array_equal(np.asarray(result), np.zeros((2, 4)))


def test_step_rampfit_passes_specific_cap_to_fit_driver(monkeypatch):
    """Check step rampfit passes specific cap to fit driver."""
    nints, ngroups, dimy, dimx = 5, 3, 2, 4
    groups = np.arange(ngroups, dtype=np.float32)
    data = np.broadcast_to(
        groups[None, :, None, None],
        (nints, ngroups, dimy, dimx)).copy()
    cube = RampCube(
        data, np.zeros_like(data, np.uint8),
        np.zeros((dimy, dimx), np.uint32), _meta(nints, ngroups))
    observed = []

    def fake_map(fn, arrays, mapped_nints, chunk_size=None,
                 bytes_per_int=None):
        observed.append((mapped_nints, chunk_size))
        shape = (mapped_nints, dimy, dimx)
        return (np.zeros(shape, np.float32),
                np.zeros(shape, np.float32),
                np.zeros(shape, np.uint32))

    monkeypatch.setenv('EXOTEDRF_CHUNK_INTS', '5')
    monkeypatch.setenv('EXOTEDRF_RAMPFIT_CHUNK_INTS', '2')
    monkeypatch.setattr(
        stages, '_segment_median_rate',
        lambda *args, **kwargs: (
            np.zeros((dimy, dimx), np.float32), 1,
            np.zeros((nints, dimy, dimx), bool)))
    monkeypatch.setattr(core, 'map_over_ints', fake_map)

    out = stages.step_rampfit(
        PipelineState(cube), {},
        {'refpack': {
            'readnoise': np.ones((dimy, dimx), np.float32),
            'gain': np.ones((dimy, dimx), np.float32)}})

    assert observed == [(nints, 2)]
    assert out.cube.data.shape == (nints, dimy, dimx)


# A resident ramp sizes its batch from the device budget.

def _clear_chunk_environment(monkeypatch):
    """Return clear chunk environment."""
    for name in ('EXOTEDRF_RAMPFIT_CHUNK_INTS', 'EXOTEDRF_CHUNK_INTS',
                 'EXOTEDRF_MAX_DEVICE_BYTES',
                 'EXOTEDRF_RAMPFIT_DEVICE_CHUNK'):
        monkeypatch.delenv(name, raising=False)


def test_resident_ramp_may_exceed_the_default_batch(monkeypatch):
    """A large accelerator budget lifts the small-GPU batch of 16."""
    _clear_chunk_environment(monkeypatch)
    monkeypatch.setenv('EXOTEDRF_RAMPFIT_DEVICE_CHUNK', 'on')
    monkeypatch.setenv('EXOTEDRF_MAX_DEVICE_BYTES', str(1 << 30))

    # 105 integrations of 1000 bytes: the budget model allows all of them.
    assert stages._rampfit_integration_chunk_size(
        105, 1000, resident=True) == 105
    # A host-resident ramp keeps the measured-safe default.
    assert stages._rampfit_integration_chunk_size(
        105, 1000, resident=False) == 16


def test_resident_ramp_never_drops_below_the_default_batch(monkeypatch):
    """16 stays the floor when the reported budget is pessimistic."""
    _clear_chunk_environment(monkeypatch)
    monkeypatch.setenv('EXOTEDRF_RAMPFIT_DEVICE_CHUNK', 'on')
    monkeypatch.setenv('EXOTEDRF_MAX_DEVICE_BYTES', str(128000))

    assert core.auto_chunk(105, 1000, n_buffers=32, headroom=.5) == 2
    assert stages._rampfit_integration_chunk_size(
        105, 1000, resident=True) == 16
    # Without residency the pessimistic budget still wins.
    assert stages._rampfit_integration_chunk_size(
        105, 1000, resident=False) == 2


def test_explicit_batch_overrides_beat_the_device_budget(monkeypatch):
    """Check explicit batch overrides beat the device budget."""
    _clear_chunk_environment(monkeypatch)
    monkeypatch.setenv('EXOTEDRF_RAMPFIT_DEVICE_CHUNK', 'on')
    monkeypatch.setenv('EXOTEDRF_MAX_DEVICE_BYTES', str(1 << 30))
    monkeypatch.setenv('EXOTEDRF_RAMPFIT_CHUNK_INTS', '7')
    assert stages._rampfit_integration_chunk_size(
        105, 1000, resident=True) == 7


def test_device_chunk_policy_rejects_unknown_values(monkeypatch):
    """Check device chunk policy rejects unknown values."""
    _clear_chunk_environment(monkeypatch)
    monkeypatch.setenv('EXOTEDRF_RAMPFIT_DEVICE_CHUNK', 'maybe')
    with pytest.raises(ValueError,
                       match='EXOTEDRF_RAMPFIT_DEVICE_CHUNK must be'):
        stages._rampfit_integration_chunk_size(105, 1000, resident=True)


def test_resident_policy_follows_the_backend(monkeypatch):
    """``auto`` only lifts the batch on an accelerator backend."""
    _clear_chunk_environment(monkeypatch)
    monkeypatch.setenv('EXOTEDRF_MAX_DEVICE_BYTES', str(1 << 30))
    monkeypatch.setattr(stages.jax, 'default_backend', lambda: 'cpu')
    assert stages._rampfit_integration_chunk_size(
        105, 1000, resident=True) == 16
    monkeypatch.setattr(stages.jax, 'default_backend', lambda: 'gpu')
    assert stages._rampfit_integration_chunk_size(
        105, 1000, resident=True) == 105


# The fitted planes do not depend on the batch size.

def _difficult_ramp_case():
    """A segment with jumps, saturation and multi-piece exception ramps."""
    nints, ngroups, dimy, dimx = 11, 6, 5, 7
    yy, xx = np.mgrid[:dimy, :dimx]
    slope = 20. + 3. * yy + 1.5 * xx
    groups = np.arange(ngroups, dtype=np.float32)[:, None, None]
    data = (500. + groups * slope[None] +
            np.arange(nints, dtype=np.float32)[:, None, None, None] * .25)
    data = np.ascontiguousarray(data, dtype=np.float32)
    groupdq = np.zeros(data.shape, np.uint8)

    for integration, group, y, x in ((1, 3, 1, 2), (4, 2, 3, 5),
                                     (7, 4, 0, 6), (9, 3, 4, 1)):
        groupdq[integration, group, y, x] |= np.uint8(core.DQ_JUMP_DET)
        data[integration, group:, y, x] += 900.

    groupdq[2, 4:, 2, 3] |= np.uint8(core.DQ_SATURATED)
    groupdq[5, :, 1, 4] |= np.uint8(core.DQ_SATURATED)
    groupdq[8, 5:, 4, 6] |= np.uint8(core.DQ_SATURATED)

    cube = RampCube(data, groupdq, np.zeros((dimy, dimx), np.uint32),
                    _meta(nints, ngroups))
    ctx = {'opts': {'DarkCurrentStep': 'skip'},
           'refpack': {'readnoise': np.full((dimy, dimx), 5.5, np.float32),
                       'gain': np.full((dimy, dimx), 1.3, np.float32)}}
    return cube, ctx


@pytest.mark.parametrize('sparse', ['on', 'off'])
def test_rampfit_planes_are_independent_of_the_integration_batch(
        monkeypatch, sparse):
    """Check rampfit planes are independent of the integration batch."""
    cube, ctx = _difficult_ramp_case()
    monkeypatch.delenv('EXOTEDRF_CHUNK_INTS', raising=False)
    monkeypatch.setenv('EXOTEDRF_RAMPFIT_SPARSE', sparse)

    reference = None
    for chunk in (1, 2, 3, 5, 16):
        monkeypatch.setenv('EXOTEDRF_RAMPFIT_CHUNK_INTS', str(chunk))
        out = stages.step_rampfit(PipelineState(cube), {}, ctx).cube
        planes = tuple(np.asarray(getattr(out, name))
                       for name in ('data', 'err', 'dq'))
        if reference is None:
            reference = planes
            # The fixture must actually exercise the interesting branches.
            assert np.isfinite(planes[0]).all()
            assert planes[2].any()
            continue
        for name, got, want in zip(('data', 'err', 'dq'), planes, reference):
            assert np.array_equal(got, want, equal_nan=True), name


def test_resident_and_host_ramps_fit_identically(monkeypatch):
    """The same segment gives the same planes wherever it lives."""
    import jax.numpy as jnp

    cube, ctx = _difficult_ramp_case()
    monkeypatch.delenv('EXOTEDRF_CHUNK_INTS', raising=False)
    monkeypatch.delenv('EXOTEDRF_RAMPFIT_CHUNK_INTS', raising=False)
    monkeypatch.setenv('EXOTEDRF_RAMPFIT_DEVICE_CHUNK', 'on')
    monkeypatch.setenv('EXOTEDRF_MAX_DEVICE_BYTES', str(1 << 32))

    host = stages.step_rampfit(PipelineState(cube), {}, ctx).cube
    resident = RampCube(jnp.asarray(cube.data), jnp.asarray(cube.groupdq),
                        cube.pixeldq, cube.meta)
    device = stages.step_rampfit(PipelineState(resident), {}, ctx).cube
    for name in ('data', 'err', 'dq'):
        assert np.array_equal(np.asarray(getattr(device, name)),
                              np.asarray(getattr(host, name)),
                              equal_nan=True), name
