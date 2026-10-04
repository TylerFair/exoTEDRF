"""Check bounded host statistics against NumPy."""

import numpy as np
import pytest

from exotedrf.v2 import hoststats


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('nints', [16, 17])
def test_bounded_order_statistics_match_numpy_and_preserve_input(
        monkeypatch, tmp_path, dtype, nints):
    """Check bounded order statistics match NumPy and preserve input."""
    rng = np.random.default_rng(936)
    original = rng.normal(size=(nints, 4, 19)).astype(dtype)
    original[1, 1, 1] = np.nan
    original[2, 2, 2] = np.nan
    data = original[:, :, ::2]
    reference = data.copy()
    monkeypatch.setenv('EXOTEDRF_MAX_HOST_BYTES', '256')
    monkeypatch.setenv('EXOTEDRF_SCRATCH_DIR', str(tmp_path))
    for q in (0., 10., 33.33, 50., 90., 100.):
        actual = hoststats.nanpercentile(data, q)
        np.testing.assert_array_equal(actual, np.nanpercentile(data, q))
    np.testing.assert_array_equal(hoststats.nanmedian(data), np.nanmedian(data))
    np.testing.assert_array_equal(data, reference)


def test_bounded_all_nan_statistic(monkeypatch, tmp_path):
    """Check bounded all NaN statistic."""
    monkeypatch.setenv('EXOTEDRF_MAX_HOST_BYTES', '32')
    monkeypatch.setenv('EXOTEDRF_SCRATCH_DIR', str(tmp_path))
    with pytest.warns(RuntimeWarning, match='All-NaN'):
        assert np.isnan(hoststats.nanpercentile(np.full((7, 11), np.nan), 10))
