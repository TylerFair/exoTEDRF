"""Check Stage-3 DQ reports, clipping, wavelengths and extraction masks."""
import numpy as np
import pytest

from exotedrf.v2 import stages, wavecal
from exotedrf.v2.kernels import extract as kx


def _v1_box_report(dq_cube, ypos, width, start, end):
    """Loop transcription of v1 2.5.0 do_box_extraction's DQ report."""
    frame = dq_cube[min(10, len(dq_cube) - 1)].astype(np.uint64)
    dimy, dimx = frame.shape
    out = np.zeros(dimx)
    for x in range(start, end):
        xx = x - start
        lo = max(ypos[xx] - width / 2, 0)
        up = min(ypos[xx] + width / 2, dimy)
        lw, uw = int(np.ceil(lo)), int(np.floor(up))
        col = frame[lw:uw, x]
        for value, bits in ((1, 1), (2, 2), (4, 2048 | 4096), (8, 1 << 32)):
            if np.any(col & np.uint64(bits)):
                out[x] += value
    return out


@pytest.mark.parametrize('nint', [20, 6])
def test_box_dq_report_matches_v1_loop(nint):
    """Compare the box-aperture DQ report with the v1 reference."""
    rng = np.random.default_rng(1)
    dimy, dimx = 32, 64
    dq = np.zeros((nint, dimy, dimx), np.uint64)
    pick = min(10, nint - 1)
    for bit in (1, 2, 2048, 4096, 1 << 32):
        ys = rng.integers(0, dimy, 40)
        xs = rng.integers(0, dimx, 40)
        dq[pick, ys, xs] |= np.uint64(bit)
    dq[0, :, :] |= np.uint64(1)
    ypos = 15 + 4 * np.sin(np.arange(40) / 8)
    for width in (3.0, 6.5):
        want = _v1_box_report(dq, ypos, width, 10, 50)
        got = kx.box_dq_report(dq[pick], ypos, width / 2, 10, 50)
        np.testing.assert_array_equal(got, want)
    assert kx.dq_report_frame_index(5) == 4 and kx.dq_report_frame_index(500) == 10


def test_uint32_dq_cannot_report_high_variance():
    """Check uint32 DQ cannot report high variance."""
    dq = np.zeros((32, 8), np.uint32)
    dq[10, 3] = 2 ** 31
    rep = kx.box_dq_report(dq, np.full(8, 10.0), 8.0)
    assert not np.any(rep >= 8)


def test_dq_report_from_mask():
    """Check DQ report from mask."""
    dq = np.zeros((6, 10), np.uint32)
    dq[2, 4] = 2
    dq[3, 9] = 1
    mask = np.zeros((4, 10), bool)
    mask[:, 3:6] = True
    rep = kx.dq_report_from_mask(dq, mask, start=1)
    np.testing.assert_array_equal(rep, [0, 0, 2, 0, 0, 0])


def _v1_clip(flux, thresh, window):
    """Clip light curves with the v1 implementation."""
    from exotedrf import utils
    return utils.sigma_clip_lightcurves(flux, thresh=thresh, window=window)


def test_sigma_clip_matches_v1_2_5_0():
    """Check light-curve clipping against exoTEDRF 2.5.0."""
    rng = np.random.default_rng(2)
    flux = 1 + 1e-3 * rng.standard_normal((60, 200))
    flux[:, 100] += 0.05 * rng.standard_normal(60)
    flux[30, 50] += 0.5
    flux[:, 150:165] += 0.05 * rng.standard_normal((60, 15))
    for thresh in (10, 5):
        got = stages.sigma_clip_lightcurves(flux, thresh=thresh, window=10)
        want = _v1_clip(flux, thresh, 10)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
        np.testing.assert_allclose(got, want, rtol=1e-12, equal_nan=True)
    # Check the default threshold and window.
    np.testing.assert_allclose(stages.sigma_clip_lightcurves(flux),
                               _v1_clip(flux, 10, 10), equal_nan=True)


def test_sigma_clip_quiet_light_curve_no_flagged_channels():
    """Check that clipping leaves quiet light curves unchanged."""
    flux = 1 + 1e-4 * np.random.default_rng(3).standard_normal((40, 30))
    got = stages.sigma_clip_lightcurves(flux, thresh=100.)
    np.testing.assert_allclose(got, flux)


def test_do_ccf_returns_pair_and_recovers_shift():
    """Check the wavelength shift and oversampling returned by cross-correlation."""
    wave = np.linspace(3.0, 5.0, 800)
    model = 1 + 0.3 * np.sin(wave * 60) + 0.1 * np.sin(wave * 17) + 0.2 * (wave - 4)
    shifted = np.interp(wave - 0.004, wave, model)
    shift, steps = wavecal.do_ccf(wave, shifted, model, oversample=1)
    assert abs(abs(shift) - 0.004) < 0.0013 and steps == steps
    from exotedrf.stage3 import do_ccf as v1_ccf
    v1 = v1_ccf(wave, shifted, model, oversample=1)
    assert shift == pytest.approx(v1[0]) and steps == v1[1]


def test_box_bitmask_defaults_and_rescue():
    """Check default extraction masks and saturated-pixel rescue."""
    f = stages._box_bitmask
    assert int(f({}, 'NIRISS')) == 0 and int(f({}, 'MIRI')) == 0
    assert int(f({'mask_do_not_use_pixels': True,
                  'mask_saturated_pixels': True}, 'NIRSPEC')) == 3
    # Saturation_rescue only overrides the saturated mask, NIRISS only.
    assert int(f({'mask_saturated_pixels': True, 'saturation_rescue': True},
                 'NIRISS')) == 0
    assert int(f({'mask_saturated_pixels': True, 'saturation_rescue': True},
                 'NIRSPEC')) == 2


def test_save_spectra_writes_dq_report_columns(tmp_path):
    """Check that saved spectra include the expected DQ reports."""
    from astropy.io import fits
    from exotedrf.v2 import io
    from exotedrf.v2.core import ObsMeta
    meta = ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP256', 1., 3,
                   np.arange(2.) + 60000, np.array([2]), np.array([2]), ())
    wave = np.array([1., 2., 3.])
    flux = np.ones((2, 3), np.float32)
    orders = {'O1': {'wave': wave, 'flux': flux, 'ferr': flux,
                     'dq': np.array([0., 3., 8.])},
              'O2': {'wave': wave, 'flux': flux, 'ferr': flux}}
    path = tmp_path / 's.fits'
    io.save_spectra(path, orders, meta)
    with fits.open(path) as hdus:
        np.testing.assert_array_equal(hdus['DQ Report O1'].data, [0, 3, 8])
        np.testing.assert_array_equal(hdus['DQ Report O2'].data, [-1, -1, -1])
        assert hdus['DQ Report O1'].header['UNITS'] == '1-DNU, 2-SAT, 4-HOT, 8-VAR'


def test_dq_report_frame_reads_high_variance_aux():
    """Check that the DQ report includes the high-variance map."""
    import types
    from exotedrf.v2.core import ObsMeta
    meta = ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP256', 1., 3,
                   np.arange(12.), np.array([2]), np.array([12]), ())
    dq = np.zeros((12, 4, 5), np.uint32)
    var = np.zeros((12, 4, 5), bool)
    var[10, 2, 3] = True
    state = types.SimpleNamespace(cube=types.SimpleNamespace(dq=dq, meta=meta),
                                  aux={'high_variance_map': var})
    frame = stages._dq_report_frame(state)
    rep = kx.dq_report_from_bounds(frame, 0, 4, 0, 5)
    np.testing.assert_array_equal(rep, [0, 0, 0, 8, 0])
    state.aux = {}
    assert not stages._dq_report_frame(state).any()
