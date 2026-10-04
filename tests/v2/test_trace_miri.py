"""Check trace miri."""

import builtins
from types import SimpleNamespace

import numpy as np
import pytest

import exotedrf
from exotedrf.v2 import trace


def _force_local_fallback(monkeypatch):
    """Make the optional v1 helper unavailable for one test."""
    real_import = builtins.__import__

    def without_v1_utils(name, globals=None, locals=None, fromlist=(),
                         level=0):
        if name == 'exotedrf' and 'utils' in fromlist:
            raise ImportError('forced local centroid fallback')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', without_v1_utils)


def test_validate_miri_centroids_accepts_contiguous_detector_rows():
    """Check validate MIRI centroids accepts contiguous detector rows."""
    result = trace.validate_miri_centroids(
        {'xpos': [35.1, 35.2, 35.3], 'ypos': [50, 51, 52]}, dimy=64)

    assert set(result) == {'xpos', 'ypos'}
    assert result['xpos'].dtype == np.float64
    np.testing.assert_array_equal(result['ypos'], [50., 51., 52.])


@pytest.mark.parametrize(
    ('centroids', 'message'),
    [
        ({'ypos': [1, 2]}, "missing 'xpos'"),
        ({'xpos': [3, 3]}, "missing 'ypos'"),
        ({'xpos': [3], 'ypos': [1, 2]}, 'matching 1D arrays'),
        ({'xpos': [3, np.nan], 'ypos': [1, 2]}, 'finite everywhere'),
        ({'xpos': [3, 3], 'ypos': [1, 3]}, 'contiguous detector rows'),
        ({'xpos': [3, 3], 'ypos': [7, 8]}, 'outside the detector'),
    ])
def test_validate_miri_centroids_rejects_malformed_traces(centroids,
                                                           message):
    """Check validate MIRI centroids rejects malformed traces."""
    with pytest.raises(ValueError, match=message):
        trace.validate_miri_centroids(centroids, dimy=8)


def test_get_centroids_miri_prefers_v1_helper_and_forwards_options(
        monkeypatch):
    """Check get centroids MIRI prefers v1 helper and forwards options."""
    calls = []

    def v1_centroids(frame, **kwargs):
        calls.append((frame, kwargs))
        ypos = np.arange(kwargs['ystart'], kwargs['yend'], dtype=float)
        return np.vstack((np.full(ypos.size, 17.25), ypos))

    monkeypatch.setattr(
        exotedrf, 'utils',
        SimpleNamespace(get_centroids_miri=v1_centroids), raising=False)
    frame = np.zeros((300, 72), dtype=np.float32)

    result = trace.get_centroids_miri(
        frame, ystart=50, yend=275, allow_slope=True)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], frame)
    assert calls[0][1] == {
        'ystart': 50,
        'yend': 275,
        'save_results': False,
        'allow_slope': True,
    }
    np.testing.assert_array_equal(result['ypos'], np.arange(50., 275.))
    np.testing.assert_array_equal(result['xpos'], np.full(225, 17.25))


def test_get_centroids_miri_local_fallback_matches_vertical_trace(
        monkeypatch):
    """Check get centroids MIRI local fallback matches vertical trace."""
    _force_local_fallback(monkeypatch)
    dimy, dimx = 320, 80
    ypos = np.arange(dimy, dtype=float)[:, None]
    xgrid = np.arange(dimx, dtype=float)[None, :]
    true_xpos = 35. + 0.02 * (ypos - 160.)
    frame = 100. * np.exp(-0.5 * ((xgrid - true_xpos) / 2.) ** 2)

    flat = trace.get_centroids_miri(
        frame, ystart=50, yend=300, allow_slope=False)
    sloped = trace.get_centroids_miri(
        frame, ystart=50, yend=300, allow_slope=True)

    expected_ypos = np.arange(50., 300.)
    np.testing.assert_array_equal(flat['ypos'], expected_ypos)
    np.testing.assert_array_equal(sloped['ypos'], expected_ypos)
    assert np.ptp(flat['xpos']) < 1e-10
    expected_xpos = 35. + 0.02 * (expected_ypos - 160.)
    np.testing.assert_allclose(sloped['xpos'], expected_xpos, atol=0.08)


def test_get_centroids_routes_miri_csv_without_soss_key_renaming(tmp_path):
    """Check get centroids routes MIRI csv without SOSS key renaming."""
    path = tmp_path / 'miri_centroids.csv'
    trace.save_centroids_csv(path, {
        'xpos': np.array([34.5, 34.6, 34.7]),
        'ypos': np.array([2., 3., 4.]),
    })

    result = trace.get_centroids(
        np.zeros((8, 6)), 'MIRI/LRS', centroids_csv=path)

    assert set(result) == {'xpos', 'ypos'}
    np.testing.assert_array_equal(result['ypos'], [2., 3., 4.])
