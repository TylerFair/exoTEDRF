"""Custom 1/f outlier maps: v1-compatible loading and mask placement."""

import numpy as np
import pytest
from astropy.io import fits

from exotedrf.v2 import stages
from exotedrf.v2.core import RampCube, RateCube
from exotedrf.v2.optimize import _first_segment_state, _phase1_options
from exotedrf.v2.pipeline import PipelineState

from .test_stages import meta


def _ramp(nints=5, edges=None, starts=None, dimy=4, dimx=6):
    """Return ramp."""
    data = np.zeros((nints, 2, dimy, dimx), np.float32)
    return RampCube(
        data, np.zeros_like(data, np.uint8),
        np.zeros((dimy, dimx), np.uint32),
        meta(nints, edges=edges, starts=starts, ngroups=2, dimx=dimx))


def test_static_npy_and_fits_maps_match_v1_bool_broadcast(tmp_path):
    # V1 casts the supplied numeric map to bool: zero is good; nonzero and NaN are masked.
    """Check static npy and FITS maps match v1 bool broadcast."""
    raw = np.zeros((4, 6), np.float32)
    raw[1, 2] = -3
    raw[2, 3] = np.nan
    np.save(tmp_path / 'mask.npy', raw)
    fits.PrimaryHDU(raw).writeto(tmp_path / 'mask.fits')
    cube = _ramp()
    expected = np.broadcast_to(raw.astype(bool), (5, 4, 6))

    for filename in ('mask.npy', 'mask.fits'):
        got = stages._load_outlier_maps(filename, cube, (tmp_path,))
        assert got.dtype == bool
        np.testing.assert_array_equal(got, expected)


def test_one_map_per_segment_aligns_2d_and_3d_inputs(tmp_path):
    """Check one map per segment aligns 2d and 3d inputs."""
    first = np.zeros((4, 6), np.uint8)
    first[0, 1] = 1
    second = np.zeros((3, 4, 6), np.uint8)
    second[0, 1, 2] = 1
    second[2, 3, 4] = 7
    np.save(tmp_path / 'seg1.npy', first)
    np.save(tmp_path / 'seg2.npy', second)
    cube = _ramp(nints=5, edges=[2, 5], starts=[10, 30])

    got = stages._load_outlier_maps(
        ['seg1.npy', 'seg2.npy'], cube, (tmp_path,))
    expected = np.concatenate([
        np.broadcast_to(first.astype(bool), (2, 4, 6)),
        second.astype(bool),
    ])
    np.testing.assert_array_equal(got, expected)


def test_single_3d_map_supports_concatenated_global_and_reused_forms():
    """Check single 3d map supports concatenated global and reused forms."""
    cube = _ramp(nints=5, edges=[2, 5], starts=[10, 30])

    concatenated = np.zeros((5, 4, 6), np.uint8)
    concatenated[0, 0, 0] = 1
    concatenated[4, 3, 5] = 1
    np.testing.assert_array_equal(
        stages._load_outlier_maps(concatenated, cube),
        concatenated.astype(bool))

    exposure_global = np.zeros((40, 4, 6), np.uint8)
    exposure_global[9:11, 0, 1] = 1
    exposure_global[29:32, 2, 3] = 1
    expected = np.concatenate([exposure_global[9:11],
                               exposure_global[29:32]]).astype(bool)
    np.testing.assert_array_equal(
        stages._load_outlier_maps(exposure_global, cube), expected)

    equal_segments = _ramp(nints=4, edges=[2, 4], starts=[10, 30])
    reusable = np.zeros((2, 4, 6), np.uint8)
    reusable[1, 2, 2] = 1
    np.testing.assert_array_equal(
        stages._load_outlier_maps(reusable, equal_segments),
        np.concatenate([reusable, reusable]).astype(bool))


@pytest.mark.parametrize(
    ('value', 'match'),
    [([np.zeros((4, 6)), np.zeros((4, 6)), np.zeros((4, 6))],
      'exactly one per input segment'),
     ([np.zeros((2, 4, 6)), np.zeros((2, 4, 6))],
      r'outlier_maps\[1\].*expected 3'),
     (np.zeros((5, 3, 6)), 'detector shape'),
     (np.zeros((4, 4, 6)), 'one 3D outlier map')])
def test_outlier_map_shape_mismatches_fail_closed(value, match):
    """Check outlier map shape mismatches fail closed."""
    cube = _ramp(nints=5, edges=[2, 5], starts=[10, 30])
    with pytest.raises(ValueError, match=match):
        stages._load_outlier_maps(value, cube)


def test_phase1_uses_first_segment_map_and_slices_concatenated_cube(tmp_path):
    """Check phase1 uses first segment map and slices concatenated cube."""
    full = PipelineState(_ramp(nints=5, edges=[2, 5], starts=[10, 30]))
    first = _first_segment_state(full)
    paths = ['seg1.npy', 'seg2.npy']
    assert _phase1_options(
        {'outlier_maps': paths}, first, full)['outlier_maps'] == ['seg1.npy']

    concatenated = np.arange(5 * 4 * 6).reshape(5, 4, 6)
    got = _phase1_options(
        {'outlier_maps': concatenated}, first, full)['outlier_maps']
    np.testing.assert_array_equal(got, concatenated[:2])

    np.save(tmp_path / 'all.npy', concatenated)
    got = _phase1_options(
        {'outlier_maps': [str(tmp_path / 'all.npy')]}, first, full)[
            'outlier_maps']
    np.testing.assert_array_equal(got, concatenated[:2])


def test_group_and_integration_oof_receive_same_custom_mask(monkeypatch):
    """Check group and integration 1/f receive same custom mask."""
    nints, ngroups, dimy, dimx = 6, 2, 16, 8
    data4 = np.zeros((nints, ngroups, dimy, dimx), np.float32)
    custom = np.zeros((nints, dimy, dimx), bool)
    custom[2, 7, 4] = True
    seen = []

    def capture_masks(base, o1, o2, o3, inner, outer, method):
        seen.append(np.asarray(base).copy())
        zeros = np.zeros((dimy, dimx), bool)
        return np.asarray(base), np.asarray(base), zeros, zeros

    monkeypatch.setattr(stages, '_build_soss_masks_chunked', capture_masks)
    common_ctx = {
        'opts': {'oof_method': 'scale-achromatic'},
        'centroids': {'ypos o1': np.full(dimx, 2.)},
        'soss_timeseries': np.ones(nints, np.float32),
        'outlier_mask': custom,
    }
    params = {'soss_inner_mask_width': 3, 'soss_outer_mask_width': 5}

    ramp = RampCube(
        data4, np.zeros_like(data4, np.uint8),
        np.zeros((dimy, dimx), np.uint32),
        meta(nints, starts=[1], ngroups=ngroups, dimx=dimx))
    out4 = stages.step_oneoverf_grp(
        PipelineState(ramp), params, common_ctx)

    data3 = np.zeros((nints, dimy, dimx), np.float32)
    rate = RateCube(
        data3, np.ones_like(data3), np.zeros_like(data3, np.uint32),
        meta(nints, starts=[1], dimx=dimx))
    out3 = stages.step_oneoverf_int(
        PipelineState(rate), params, common_ctx)

    assert len(seen) == 2
    assert seen[0][2, 7, 4] and seen[1][2, 7, 4]
    # Custom masks influence the estimator only; they are not promoted to DQ.
    assert not np.asarray(out4.cube.groupdq).any()
    assert not np.asarray(out4.cube.pixeldq).any()
    assert not np.asarray(out3.cube.dq).any()
