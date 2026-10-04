"""Science-preserving memory bounds for full-visit stage orchestration."""

import numpy as np
import pytest

from exotedrf.v2 import core, stages
from exotedrf.v2.kernels import pca
from exotedrf.v2.pipeline import PipelineState


@pytest.mark.parametrize('groups', [None, 3])
def test_baseline_median_bounds_host_selection_and_matches_full_numpy(
        monkeypatch, groups):
    """Check baseline median bounds host selection and matches full NumPy."""
    rng = np.random.default_rng(6548)
    shape = (17, 5, 23) if groups is None else (17, groups, 5, 23)
    data = rng.normal(size=shape).astype(np.float32)
    data[..., 2, 7] = np.nan
    baseline = np.arange(17) % 3 != 1
    reference = np.nanmedian(data[baseline], axis=0)
    original = data.copy()

    class GuardedCube:
        def __init__(self):
            self.shape, self.ndim, self.dtype = data.shape, data.ndim, data.dtype

        def __array__(self, *args, **kwargs):
            raise AssertionError('Whole observation was converted')

        def __getitem__(self, key):
            assert isinstance(key, tuple)
            columns = key[-1]
            assert isinstance(columns, slice)
            assert columns.stop - columns.start <= 2
            return data[key]

    monkeypatch.setenv('EXOTEDRF_DEVICE_FAST_PATH', 'false')
    monkeypatch.setattr(core, 'host_allocation_budget_bytes', lambda: 7680)
    actual = stages._host_deepstack(GuardedCube(), baseline)
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize('ngroups', [2, 3])
def test_jump_output_spills_and_matches_segmentwise_kernel(monkeypatch, tmp_path,
                                                         ngroups):
    """Check jump output spills and matches segmentwise kernel."""
    rng = np.random.default_rng(376)
    shape = (27, ngroups, 6, 15)
    data = rng.normal(100, .3, shape).astype(np.float32)
    data[8, -1, 4, 8] += 10
    data[23, -1, 4, 8] += 20
    dq = np.zeros(shape, np.uint8)
    meta = core.ObsMeta(
        'NIRISS/SOSS', 'NIS', 'SUBSTRIP96', 1., ngroups,
        np.arange(27.), np.array([5, -5]), np.array([14, 27]), (),
        {'segment_int_starts': (300, 314)})
    cube = core.RampCube(data, dq, np.zeros(shape[-2:], np.uint32), meta)
    reference_data, reference_dq = [], []
    for seg in core.segment_slices(meta, 27):
        out = stages.k_jump.flag_jumps_in_time_chunked(
            data[seg], dq[seg], 5., window=3,
            artifact=np.zeros(data[seg].shape[:1] + shape[-2:], bool),
            chunk_size=5)
        reference_data.append(np.asarray(out[0]))
        reference_dq.append(np.asarray(out[1]))
    monkeypatch.setenv('EXOTEDRF_MAX_HOST_BYTES', '1')
    monkeypatch.setenv('EXOTEDRF_SCRATCH_DIR', str(tmp_path))
    actual = stages.step_jump(PipelineState(cube),
        {'time_jump_threshold': 5., 'time_window': 3}, {'opts': {}})
    assert isinstance(actual.cube.groupdq, np.memmap)
    if ngroups == 2:
        assert isinstance(actual.cube.data, np.memmap)
    else:
        assert actual.cube.data is data
    np.testing.assert_array_equal(actual.cube.data, np.concatenate(reference_data))
    np.testing.assert_array_equal(actual.cube.groupdq, np.concatenate(reference_dq))


def test_pca_memory_limits_adjust_spatial_chunks_and_preserve_reconstruction(
        monkeypatch, tmp_path):
    """Check PCA memory limits adjust spatial chunks and preserve reconstruction."""
    rng = np.random.default_rng(362)
    times = np.linspace(-1., 1., 18, dtype=np.float32)
    image = rng.normal(10., 1., (5, 12)).astype(np.float32)
    cube = image[None] * (1 + .01 * times[:, None, None])
    cube += (.2 * times ** 2)[:, None, None] * np.linspace(
        -1., 1., 12, dtype=np.float32)[None, None, :]
    cube += .0001 * rng.normal(size=cube.shape).astype(np.float32)
    reference = pca.soss_stability_pca(cube, n_components=2)
    monkeypatch.setenv('EXOTEDRF_MAX_HOST_BYTES', '4096')
    monkeypatch.setenv('EXOTEDRF_SCRATCH_DIR', str(tmp_path))
    assert all(chunk.shape[-1] == 1 for _, _, chunk in pca._host_chunks(cube))
    actual = pca.soss_stability_pca(cube, n_components=2)
    assert isinstance(actual[2], np.memmap)
    np.testing.assert_allclose(actual[1], reference[1], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(actual[2], reference[2], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('mode,starts', [
    ('NIRISS/SOSS', (230, 245)), ('NIRSPEC/G395H', (48, 63))])
def test_streamed_oof_mask_preserves_segment_reset_and_full_detector_flags(
        monkeypatch, tmp_path, mode, starts):
    """Check streamed 1/f mask preserves segment reset and full detector flags."""
    rng = np.random.default_rng(6891)
    shape = (31, 3, 12, 18)
    data = rng.normal(100, 1, shape).astype(np.float32)
    data[6, -1, 8, 9] += 200
    dq = np.zeros(shape, np.uint8)
    dq[7, -1, 2, 8] = 4
    pixeldq = np.zeros(shape[-2:], np.uint32)
    pixeldq[3, 7] = 2048
    custom = np.zeros((31, *shape[-2:]), bool)
    custom[19, 4, 8] = True
    meta = core.ObsMeta(mode, 'NRS2', 'SUBSTRIP96', 1., 3,
        np.arange(31.), np.array([5, -5]), np.array([15, 31]), (),
        {'segment_int_starts': starts})
    cube = core.RampCube(data, dq, pixeldq, meta)
    expected = (dq[:, -1] != 0) | (pixeldq[None] != 0) | custom
    for seg, first in zip(core.segment_slices(meta, 31), starts):
        expected[seg] |= np.asarray(stages.k_oof.flag_temporal_outliers(
            data[seg, -1]))
        expected[seg] |= stages.k_jump.reset_artifact_mask(
            seg.stop - seg.start, *shape[-2:], int_start=first,
            instrument=mode.split('/')[0],
            max_reset_int=stages._reset_max_int(meta))
    monkeypatch.setenv('EXOTEDRF_MAX_HOST_BYTES', '1024')
    monkeypatch.setenv('EXOTEDRF_SCRATCH_DIR', str(tmp_path))
    actual = stages._oneoverf_base_mask(cube, {'outlier_mask': custom}, reset=True)
    assert isinstance(actual, np.memmap)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('nan_error', [False, True])
def test_badpix_reuses_finite_errors_and_bounds_spatial_dq_transfer(
        monkeypatch, tmp_path, nan_error):
    """Check bad-pixel correction reuses finite errors and bounds spatial DQ transfer."""
    rng = np.random.default_rng(71)
    shape = (27, 10, 23)
    data = rng.normal(100, 1, shape).astype(np.float32)
    err = np.ones(shape, np.float32)
    err[14:] *= 3
    if nan_error:
        err[17, 2, 4] = np.nan
    original = err.copy()
    dq = np.zeros(shape, np.uint32)
    dq[10, 2, 8] = core.DQ_HOT
    meta = core.ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP96', 1., 3,
        np.arange(27.), np.array([5, -5]), np.array([14, 27]), ())
    cube = core.RateCube(data, err, dq, meta)
    prepare = stages.k_badpix.prepare_spatial_badpix

    def bounded_dq(deep, reference, saturation, **kwargs):
        assert reference.shape == (1, *shape[-2:])
        np.testing.assert_array_equal(reference[0], dq[10])
        return prepare(deep, reference, saturation, **kwargs)

    monkeypatch.setattr(stages.k_badpix, 'prepare_spatial_badpix', bounded_dq)
    monkeypatch.setenv('EXOTEDRF_MAX_HOST_BYTES', '4096')
    monkeypatch.setenv('EXOTEDRF_SCRATCH_DIR', str(tmp_path))
    actual = stages.step_badpix(PipelineState(cube),
        dict(box_size=2, window_size=3, space_outlier_threshold=8.,
             time_outlier_threshold=8.), {'opts': {}})
    expected = np.nan_to_num(original, nan=3.)
    np.testing.assert_array_equal(actual.cube.err, expected)
    np.testing.assert_array_equal(err, original)
    if nan_error:
        assert isinstance(actual.cube.err, np.memmap)
    else:
        assert actual.cube.err is err


def test_oversized_prepared_scorers_fall_back_before_device_conversion(monkeypatch):
    """Check oversized prepared scorers fall back before device conversion."""
    shape = (50, 3, 16, 32)
    meta = core.ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP96', 1., 3,
        np.arange(50.), np.array([10, -10]), np.array([50]), ())
    ramp = core.RampCube(np.zeros(shape, np.float32),
        np.zeros(shape, np.uint8), np.zeros(shape[-2:], np.uint32), meta)
    rate = core.RateCube(ramp.data[:, 0], ramp.data[:, 0],
        np.zeros((50, *shape[-2:]), np.uint32), meta)
    monkeypatch.setattr(core, 'device_memory_bytes', lambda: 1024)
    assert stages.prepare_oneoverf_grp_scorer(PipelineState(ramp), {}, {}) is None
    assert stages.prepare_jump_scorer(PipelineState(ramp), {}, {}) is None
    assert stages.prepare_badpix_scorer(PipelineState(rate), {}, {}) is None
    assert stages.prepare_nirspec_oneoverf_scorer(PipelineState(ramp), {}, {}) is None
    assert stages.prepare_miri_background_scorer(PipelineState(rate), {}, {}) is None


def test_rampfit_default_batch_accounts_for_large_group_count(monkeypatch):
    """Check rampfit default batch accounts for large group count."""
    monkeypatch.delenv('EXOTEDRF_RAMPFIT_CHUNK_INTS', raising=False)
    monkeypatch.delenv('EXOTEDRF_CHUNK_INTS', raising=False)
    monkeypatch.setattr(core, 'device_memory_bytes', lambda: 1 << 30)
    assert stages._rampfit_integration_chunk_size(100, 32 << 20) == 1
    assert stages._rampfit_integration_chunk_size(100, 1 << 20) == 16


def test_joint_jump_window_change_releases_previous_detector_caches():
    """Check joint jump window change releases previous detector caches."""
    rng = np.random.default_rng(1920)
    data = rng.normal(100, 1, (14, 1, 4, 9)).astype(np.float32)
    invariants = stages.k_jump.prepare_jump_invariants(data)
    scorer = object.__new__(stages.PreparedJumpScorer)
    scorer._prepared_window = {}
    scorer._prepared_window_value = None
    scorer._prepared_for_window(0, data, invariants, 3)
    scorer._prepared_for_window(1, data, invariants, 3)
    assert set(scorer._prepared_window) == {(0, 3), (1, 3)}
    actual = scorer._prepared_for_window(0, data, invariants, 5)
    assert set(scorer._prepared_window) == {(0, 5)}
    expected = stages.k_jump.prepare_jumps_in_time(
        data, window=5, invariants=invariants)
    np.testing.assert_array_equal(actual.scale, expected.scale)
    np.testing.assert_array_equal(actual.cube_filt, expected.cube_filt)
