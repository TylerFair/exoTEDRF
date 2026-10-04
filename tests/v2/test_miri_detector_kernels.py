"""Check miri detector kernels."""

import numpy as np
import pytest
from types import SimpleNamespace

from exotedrf.v2.kernels import emicorr as emicorr_kernel
from exotedrf.v2.kernels.emicorr import apply_emicorr
from exotedrf.v2.kernels.reset import reset_correct


def test_reset_correction_matches_pinned_integration_and_group_selection():
    """Check reset correction matches pinned integration and group selection."""
    science = np.full((3, 4, 2, 3), 100., np.float32)
    reset = np.arange(2 * 2 * 2 * 3, dtype=np.float32).reshape(2, 2, 2, 3)
    reset[1, 0, 0, 0] = np.nan
    pixeldq = np.asarray([[0, 1, 0], [0, 0, 0]], np.uint32)
    reset_dq = np.asarray([[4, 0, 0], [0, 8, 0]], np.uint32)

    corrected, dq = reset_correct(
        science, pixeldq, reset, reset_dq, int_start=np.int32(1))
    expected = science.copy()
    safe_last = np.nan_to_num(reset[1], nan=0.)
    expected[:, :2] -= safe_last[None]
    np.testing.assert_array_equal(np.asarray(corrected), expected)
    np.testing.assert_array_equal(np.asarray(dq), pixeldq | reset_dq)


def test_emicorr_short_ramp_is_a_clean_noop_and_validates_geometry():
    """Check emicorr short ramp is a clean noop and validates geometry."""
    data = np.arange(2 * 3 * 2 * 16, dtype=np.float32).reshape(2, 3, 2, 16)
    wave = np.sin(np.linspace(0., 2. * np.pi, 64, endpoint=False))[None]
    corrected = apply_emicorr(
        data, [390.625], wave, rowclocks=28, frameclocks=15904)
    np.testing.assert_array_equal(corrected, data)
    assert corrected is not data

    with pytest.raises(ValueError, match='multiple of 4'):
        apply_emicorr(
            np.zeros((1, 3, 2, 12), np.float32), [390.625], wave,
            rowclocks=28, frameclocks=15904)


def test_emicorr_matches_pinned_jwst_reference_loop(monkeypatch):
    """Check emicorr matches pinned JWST reference loop."""
    official = pytest.importorskip('jwst.emicorr.emicorr')
    rng = np.random.default_rng(1771)
    data = rng.normal(1000., 4., (4, 5, 4, 16)).astype(np.float32)
    frequencies = (390.625, 10.039216)
    waves = (
        np.sin(np.linspace(0., 2. * np.pi, 32, endpoint=False)),
        np.cos(np.linspace(0., 2. * np.pi, 45, endpoint=False)),
    )
    names = ('Hz390', 'Hz10')
    info = dict(zip(names, zip(frequencies, waves)))
    monkeypatch.setattr(
        official, 'get_subarcase',
        lambda *args: ('SLITLESSPRISM', 28, 15904, names))
    monkeypatch.setattr(
        official, 'get_frequency_info', lambda model, name: info[name])
    model = SimpleNamespace(
        data=data.copy(),
        meta=SimpleNamespace(
            instrument=SimpleNamespace(detector='MIRIMAGE'),
            subarray=SimpleNamespace(
                name='SLITLESSPRISM', xsize=16, xstart=1),
            exposure=SimpleNamespace(readpatt='FASTR1', nsamples=1)))

    import inspect
    if 'algorithm' in inspect.signature(official.apply_emicorr).parameters:
        expected = official.apply_emicorr(
            model, object(), algorithm='sequential', nints_to_phase=2,
            nbins=8, scale_reference=True, use_n_cycles=3).data
    else:
        expected = official.apply_emicorr(
            model, object(), None, None, nints_to_phase=2, nbins_all=8,
            scale_reference=True, use_n_cycles=3).data
    actual = apply_emicorr(
        data, frequencies, waves, rowclocks=28, frameclocks=15904,
        readpatt='FASTR1', nsamples=1, xstart=1, nints_to_phase=2,
        nbins=8, scale_reference=True, use_n_cycles=3)

    np.testing.assert_array_equal(actual, expected)


def test_emicorr_joint_matches_jwst3_joint_algorithm(monkeypatch):
    """jwst 3.0 EmiCorrStep defaults to ``algorithm='joint'``."""
    official = pytest.importorskip('jwst.emicorr.emicorr')
    if not hasattr(official, '_run_joint_algorithm'):
        pytest.skip('jwst < 3.0 has no joint EMI algorithm')
    rng = np.random.default_rng(30)
    nints, ngroups, ny, nx = 6, 8, 12, 32
    frequencies = (390.625, 10.039216)
    waves = (
        np.sin(np.linspace(0., 2. * np.pi, 256, endpoint=False)),
        np.cos(np.linspace(0., 2. * np.pi, 500, endpoint=False)) ** 3,
    )
    # Ramps plus an injected 390 Hz EMI pattern in pixel-time order.
    t = (np.arange(ny)[:, None] * 28 + np.arange(nx)[None, :] // 4) * 1e-5
    group_t = np.arange(ngroups)[:, None, None] * 0.02
    emi = 5. * np.sin(2. * np.pi * 390.625 * (t[None] + group_t) + 0.4)
    data = (1000. + 40. * np.arange(ngroups)[None, :, None, None] +
            emi[None] + rng.normal(0., 2., (nints, ngroups, ny, nx))
            ).astype(np.float32)
    pixeldq = np.zeros((ny, nx), np.uint32)
    pixeldq[3, 5] = 1
    names = ('Hz390', 'Hz10')
    info = dict(zip(names, zip(frequencies, waves)))
    monkeypatch.setattr(
        official, 'get_subarcase',
        lambda *args: ('SLITLESSPRISM', 28, 15904, names))
    monkeypatch.setattr(
        official, 'get_frequency_info', lambda model, name: info[name])
    model = SimpleNamespace(
        data=data.copy(), pixeldq=pixeldq.copy(),
        meta=SimpleNamespace(
            instrument=SimpleNamespace(detector='MIRIMAGE'),
            subarray=SimpleNamespace(name='SLITLESSPRISM'),
            exposure=SimpleNamespace(readpatt='FASTR1', nsamples=1)))
    expected = official.apply_emicorr(model, object(),
                                      algorithm='joint').data
    actual = emicorr_kernel.apply_emicorr_joint(
        data, pixeldq, frequencies, waves, 28, 15904, readpatt='FASTR1',
        nsamples=1)
    assert actual.dtype == np.float32
    assert not np.array_equal(actual, data)
    # Only sum(z * y) is reduced in a different float64 order.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-4)


def test_emicorr_has_no_observation_sized_float64_allocation(monkeypatch):
    """Check emicorr has no observation sized float64 allocation."""
    allocations = []
    real_empty = np.empty
    real_array = np.array

    def record(result):
        allocations.append((result.shape, result.dtype, result.size))
        return result

    def tracked_empty(*args, **kwargs):
        return record(real_empty(*args, **kwargs))

    def tracked_array(*args, **kwargs):
        return record(real_array(*args, **kwargs))

    monkeypatch.setattr(emicorr_kernel.np, 'empty', tracked_empty)
    monkeypatch.setattr(emicorr_kernel.np, 'array', tracked_array)
    data = np.arange(4 * 5 * 4 * 16, dtype=np.float32).reshape(4, 5, 4, 16)
    wave = np.sin(np.linspace(0., 2. * np.pi, 32, endpoint=False))[None]

    result = emicorr_kernel.apply_emicorr(
        data, [390.625], wave, rowclocks=28, frameclocks=15904,
        nints_to_phase=2, nbins=4)

    assert result.dtype == np.float32
    assert max(size for _, dtype, size in allocations
               if dtype == np.dtype(np.float64)) <= data.size // 4
