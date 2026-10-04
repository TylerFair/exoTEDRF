"""Check sparse ramp fitting against complete semiramp calculations."""

import itertools

import numpy as np
import pytest

from exotedrf.v2 import core, stages
from exotedrf.v2.kernels import ramp


def _assert_float_bits_equal(actual, expected):
    """Compare values, signed zero, and NaN payloads exactly."""
    actual = np.ascontiguousarray(np.asarray(actual))
    expected = np.ascontiguousarray(np.asarray(expected))
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(
        actual.view(np.dtype(f'u{actual.dtype.itemsize}')),
        expected.view(np.dtype(f'u{expected.dtype.itemsize}')))


def _reference_topology(data, groupdq):
    """Independent scalar state machine for useful later semiramps."""
    data = np.asarray(data)
    groupdq = np.asarray(groupdq, dtype=np.uint8)
    nints, ngroups, dimy, dimx = groupdq.shape
    exception = np.zeros((nints, dimy, dimx), dtype=bool)
    maximum = 1

    for integration in range(nints):
        for row in range(dimy):
            for column in range(dimx):
                dq = groupdq[integration, :, row, column].copy()
                if np.count_nonzero(dq == 0) == 1 and ngroups != 1:
                    dq |= np.uint8(core.DQ_DO_NOT_USE)

                pure_jump = dq == np.uint8(core.DQ_JUMP_DET)
                exact_good = dq == 0
                jump_start = np.zeros(ngroups, dtype=bool)
                if ngroups > 1:
                    jump_start[:-1] = pure_jump[:-1] & exact_good[1:]
                usable = ((exact_good | jump_start) &
                          np.isfinite(data[integration, :, row, column]))
                if dq[0] & np.uint8(core.DQ_SATURATED):
                    usable[:] = False

                counts = [0] * ngroups
                segment = 0
                previous_usable = True
                for group in range(ngroups):
                    if (group > 0 and usable[group] and
                            (pure_jump[group] or not previous_usable)):
                        segment += 1
                    if usable[group]:
                        counts[segment] += 1
                    previous_usable = bool(usable[group])

                useful_later = [
                    index for index, count in enumerate(counts[1:], start=1)
                    if count >= 2
                ]
                if useful_later:
                    exception[integration, row, column] = True
                    maximum = max(maximum, useful_later[-1] + 1)

    return exception, maximum


def _adversarial_ramps():
    """Return adversarial ramps."""
    rng = np.random.default_rng(0x1352770)
    nints, ngroups, dimy, dimx = 6, 9, 4, 7
    increments = rng.normal(
        25., 4., size=(nints, ngroups, dimy, dimx)).astype(np.float32)
    data = np.cumsum(increments, axis=1, dtype=np.float32)
    groupdq = np.zeros(data.shape, dtype=np.uint8)

    # Multiple useful semiramps.
    groupdq[0, 2, 0, 0] = np.uint8(core.DQ_JUMP_DET)
    groupdq[0, 5, 0, 0] = np.uint8(core.DQ_JUMP_DET)
    # A leading gap makes the first useful run segment one.
    groupdq[1, :2, 0, 1] = np.uint8(core.DQ_DO_NOT_USE)
    # An ordinary later jump and a gap-recovery segment.
    groupdq[2, 4, 1, 2] = np.uint8(core.DQ_JUMP_DET)
    groupdq[3, 3, 2, 3] = np.uint8(core.DQ_DROPOUT)
    # A compound JUMP flag is a gap, not a pure-JUMP segment start.
    groupdq[4, 3, 2, 4] = np.uint8(
        core.DQ_JUMP_DET | core.DQ_DO_NOT_USE)
    # Masked non-finite science participates in neither topology nor fitting.
    data[5, 4, 3, 5] = np.nan
    groupdq[5, 4, 3, 5] = np.uint8(core.DQ_SATURATED)

    groupdq[0, 2::2, 3, 6] = np.uint8(core.DQ_DO_NOT_USE)
    groupdq[1, :, 3, 0] = np.uint8(core.DQ_DO_NOT_USE)
    groupdq[1, 4, 3, 0] = 0
    groupdq[2, 0, 3, 1] = np.uint8(core.DQ_SATURATED)

    readnoise = rng.uniform(4., 8., (dimy, dimx)).astype(np.float32)
    gain = rng.uniform(1.2, 1.8, (dimy, dimx)).astype(np.float32)
    pixeldq = np.zeros((dimy, dimx), dtype=np.uint32)
    pixeldq[0, 0] = core.DQ_HOT
    pixeldq[0, 1] = core.DQ_DO_NOT_USE | core.DQ_HOT
    pixeldq[1, 2] = core.DQ_SATURATED
    median_rate = rng.uniform(5., 30., (dimy, dimx)).astype(np.float32)
    return data, groupdq, readnoise, gain, pixeldq, median_rate


def test_exception_mask_exhaustive_supported_groupdq_topologies():
    """The device predicate selects exactly the ramps needing later loops."""
    first_codes = (
        0, core.DQ_JUMP_DET, core.DQ_DO_NOT_USE, core.DQ_SATURATED)
    later_codes = (0, core.DQ_JUMP_DET, core.DQ_DO_NOT_USE)
    sequences = np.asarray([
        (first,) + tail
        for first in first_codes
        for tail in itertools.product(later_codes, repeat=8)
    ], dtype=np.uint8)
    groupdq = sequences[:, :, None, None]
    data = np.zeros(groupdq.shape, dtype=np.float32)

    expected_mask, expected_maximum = _reference_topology(data, groupdq)
    _, invalid, maximum, actual_mask = ramp.rampfit_chunk_statistics(
        data, groupdq, np.float32(2.214))

    assert not bool(np.asarray(invalid))
    assert int(np.asarray(maximum)) == expected_maximum
    np.testing.assert_array_equal(np.asarray(actual_mask), expected_mask)


def test_sparse_main_plus_packed_patch_is_bitwise_full_fit():
    """Check sparse main plus packed patch is bitwise full fit."""
    data, groupdq, readnoise, gain, pixeldq, median_rate = \
        _adversarial_ramps()
    group_time = np.float32(2.214)
    nframes = np.float32(2.)
    average_dark = np.float32(0.13)

    _, invalid, maximum, exception_mask = ramp.rampfit_chunk_statistics(
        data, groupdq, group_time)
    assert not bool(np.asarray(invalid))
    exception_mask = np.asarray(exception_mask)
    expected_mask, expected_maximum = _reference_topology(data, groupdq)
    np.testing.assert_array_equal(exception_mask, expected_mask)
    assert int(np.asarray(maximum)) == expected_maximum
    assert exception_mask.any()

    kwargs = dict(
        median_rate=median_rate,
        nframes=nframes,
        average_dark_current=average_dark,
        max_segments=int(np.asarray(maximum)))
    full = ramp.fit_ramps_stage(
        data, groupdq, readnoise, gain, group_time, pixeldq, **kwargs)
    main = ramp.fit_ramps_stage(
        data, groupdq, readnoise, gain, group_time, pixeldq,
        median_rate=median_rate, nframes=nframes,
        average_dark_current=average_dark, max_segments=1)

    sparse = stages._apply_sparse_rampfit(
        *(np.asarray(value).copy() for value in main),
        data, groupdq, readnoise, gain, pixeldq, median_rate,
        exception_mask, group_time, nframes, average_dark, None,
        int(np.asarray(maximum)), chunk_size=3)

    _assert_float_bits_equal(sparse[0], full[0])
    _assert_float_bits_equal(sparse[1], full[1])
    np.testing.assert_array_equal(np.asarray(sparse[2]), np.asarray(full[2]))


def test_packed_kernel_matches_full_fit_at_exception_coordinates():
    """Check packed kernel matches full fit at exception coordinates."""
    data, groupdq, readnoise, gain, pixeldq, median_rate = \
        _adversarial_ramps()
    group_time = np.float32(2.214)
    nframes = np.float32(2.)
    average_dark = np.float32(0.13)
    _, _, maximum, exception_mask = ramp.rampfit_chunk_statistics(
        data, groupdq, group_time)
    maximum = int(np.asarray(maximum))
    integrations, rows, columns = np.nonzero(np.asarray(exception_mask))

    packed_data = data[integrations, :, rows, columns][:, :, None, None]
    packed_dq = groupdq[integrations, :, rows, columns][:, :, None, None]
    packed_rn = readnoise[rows, columns][:, None, None]
    packed_gain = gain[rows, columns][:, None, None]
    packed_pixeldq = pixeldq[rows, columns][:, None, None]
    packed_median = median_rate[rows, columns][:, None, None]

    packed = ramp.fit_ramps_packed_stage(
        packed_data, packed_dq, packed_rn, packed_gain, group_time,
        packed_pixeldq, median_rate=packed_median, nframes=nframes,
        average_dark_current=average_dark, max_segments=maximum)
    full = ramp.fit_ramps_stage(
        data, groupdq, readnoise, gain, group_time, pixeldq,
        median_rate=median_rate, nframes=nframes,
        average_dark_current=average_dark, max_segments=maximum)

    _assert_float_bits_equal(
        np.asarray(packed[0])[:, 0, 0],
        np.asarray(full[0])[integrations, rows, columns])
    _assert_float_bits_equal(
        np.asarray(packed[1])[:, 0, 0],
        np.asarray(full[1])[integrations, rows, columns])
    np.testing.assert_array_equal(
        np.asarray(packed[2])[:, 0, 0],
        np.asarray(full[2])[integrations, rows, columns])


@pytest.mark.parametrize('value', ['0', 'false', 'off'])
def test_sparse_policy_can_force_dense_fallback(monkeypatch, value):
    """Check sparse policy can force dense fallback."""
    monkeypatch.setenv('EXOTEDRF_RAMPFIT_SPARSE', value)
    assert not stages._use_sparse_rampfit(1, 1_000_000)


@pytest.mark.parametrize('value', ['1', 'true', 'on'])
def test_sparse_policy_can_force_sparse_path(monkeypatch, value):
    """Check sparse policy can force sparse path."""
    monkeypatch.setenv('EXOTEDRF_RAMPFIT_SPARSE', value)
    assert stages._use_sparse_rampfit(900_000, 1_000_000)


def test_sparse_policy_auto_caps_work(monkeypatch):
    """Check sparse policy auto caps work."""
    monkeypatch.delenv('EXOTEDRF_RAMPFIT_SPARSE', raising=False)
    assert stages._use_sparse_rampfit(10_000, 1_000_000)
    assert not stages._use_sparse_rampfit(10_001, 1_000_000)
    assert stages._use_sparse_rampfit(65_536, 100_000_000)
    assert not stages._use_sparse_rampfit(65_537, 100_000_000)


def test_sparse_policy_rejects_unknown_value(monkeypatch):
    """Check sparse policy rejects unknown value."""
    monkeypatch.setenv('EXOTEDRF_RAMPFIT_SPARSE', 'sometimes')
    with pytest.raises(ValueError, match='EXOTEDRF_RAMPFIT_SPARSE'):
        stages._use_sparse_rampfit(1, 1_000_000)
