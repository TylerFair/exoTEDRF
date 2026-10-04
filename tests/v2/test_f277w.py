"""F277W order-0 contaminant masking."""

from pathlib import Path

import numpy as np

from exotedrf.v2 import config, stages, trace
from exotedrf.v2.core import RampCube
from exotedrf.v2.pipeline import PipelineState

from .niriss_acceptance import apply_current_niriss_settings
from .test_stages import controls, meta


def _consecutive_groups(iterable):
    """Return consecutive groups."""
    import itertools
    for _, group in itertools.groupby(enumerate(iterable),
                                      key=lambda pair: pair[1] - pair[0]):
        yield [value for _, value in group]


def _v1_order0_mask(f277w, thresh_std=1, thresh_size=10, start_col=700):
    """Return v1 order0 mask."""
    dimy, dimx = np.shape(f277w)
    mask = np.zeros_like(f277w)
    for col in range(start_col, dimx):
        diff = f277w[:, col] - np.nanmedian(f277w[:, col])
        dev = np.nanstd(diff)
        vals = np.where(np.abs(diff) > thresh_std * dev)[0]
        for group in _consecutive_groups(vals):
            if len(group) > thresh_size:
                min_g = np.max([0, np.min(group) - 3])
                max_g = np.min([dimy - 1, np.max(group) + 3])
                mask[min_g:max_g, (col - 3):(col + 3)] = 1
    return mask.astype(int)


def test_order0_mask_matches_v1_reference_at_default_start_col():
    """Check order0 mask matches v1 reference at default start col."""
    rng = np.random.RandomState(7)
    dimy, dimx = 64, 760
    frame = rng.normal(0., 1., (dimy, dimx))
    # Order-0-like blobs: bright vertical runs beyond column 700.
    frame[20:35, 705:708] += 40.
    frame[5:18, 731] += 25.
    frame[40:44, 715] += 30.
    frame[10:30, 300] += 50.
    frame[2, 741] = np.nan
    got = trace.make_order0_mask_from_f277w(frame)
    ref = _v1_order0_mask(frame)
    np.testing.assert_array_equal(got, ref.astype(bool))
    assert got.any()
    # V1 slice quirks hold: row dimy-1 never masked; nothing before col 697.
    assert not got[dimy - 1].any()
    assert not got[:, :697].any()


def test_order0_mask_run_length_strictly_greater_and_padding():
    """Check order0 mask run length strictly greater and padding."""
    dimy, dimx = 32, 24
    frame = np.zeros((dimy, dimx))
    frame[8:19, 10] = 100.
    frame[4:14, 16] = 100.
    got = trace.make_order0_mask_from_f277w(frame, start_col=0)
    ref = _v1_order0_mask(frame, start_col=0)
    np.testing.assert_array_equal(got, ref.astype(bool))
    # Padded patch: rows [8-3, 18+3) and columns [10-3, 10+3).
    assert got[5:21, 7:13].all()
    assert not got[:, 13:].any()


def test_oneoverf_grp_excludes_masked_order0_contaminant():
    """Check 1/f correction grp excludes masked order0 contaminant."""
    nints, ngroups, dimy, dimx = 6, 2, 16, 8
    rng = np.random.RandomState(3)
    base = np.tile(rng.uniform(50., 55., (dimy, dimx)).astype(np.float32),
                   (nints, ngroups, 1, 1))

    clean = base.copy()
    contaminated = base.copy()
    contaminated[3, :, 6:14, 5] += 500.
    order0 = np.zeros((dimy, dimx), dtype=bool)
    order0[6:14, 5] = True

    def run(data, order0_mask):
        cube = RampCube(data.copy(),
                        np.zeros_like(data, np.uint8),
                        np.zeros((dimy, dimx), np.uint32),
                        meta(nints, starts=[1], ngroups=ngroups, dimx=dimx))
        ctx = {'opts': {'oof_method': 'scale-achromatic'},
               'centroids': {'ypos o1': np.full(dimx, 2.0)},
               'soss_timeseries': np.ones(nints, np.float32),
               'order0_mask': order0_mask}
        out = stages.step_oneoverf_grp(
            PipelineState(cube),
            {'soss_inner_mask_width': 3, 'soss_outer_mask_width': 5}, ctx)
        return np.asarray(out.cube.data)

    out_clean = run(clean, None)
    out_bad = run(contaminated, None)
    out_masked = run(contaminated, order0)

    untouched = ~order0
    # Unmasked contaminant skews column 5's 1/f estimate at integration 3.
    assert not np.allclose(out_bad[3][:, untouched],
                           out_clean[3][:, untouched], atol=1e-4)
    np.testing.assert_allclose(out_masked[:, :, untouched],
                               out_clean[:, :, untouched], atol=1e-4)
    assert (out_masked[3, :, 6:14, 5] > 400.).all()


def test_oneoverf_int_applies_order0_mask():
    """Check 1/f correction int applies order0 mask."""
    nints, dimy, dimx = 6, 16, 8
    rng = np.random.RandomState(5)
    base = np.tile(rng.uniform(50., 55., (dimy, dimx)).astype(np.float32),
                   (nints, 1, 1))
    contaminated = base.copy()
    contaminated[2, 6:14, 4] += 300.
    order0 = np.zeros((dimy, dimx), dtype=bool)
    order0[6:14, 4] = True

    def run(data, order0_mask):
        from exotedrf.v2.core import RateCube
        cube = RateCube(data.copy(), np.ones_like(data),
                        np.zeros(data.shape, np.uint32),
                        meta(nints, starts=[1], dimx=dimx))
        ctx = {'opts': {'oof_method': 'scale-achromatic'},
               'centroids': {'ypos o1': np.full(dimx, 2.0)},
               'soss_timeseries': np.ones(nints, np.float32),
               'order0_mask': order0_mask}
        out = stages.step_oneoverf_int(
            PipelineState(cube),
            {'soss_inner_mask_width': 3, 'soss_outer_mask_width': 5}, ctx)
        return np.asarray(out.cube.data)

    out_clean = run(base, None)
    out_masked = run(contaminated, order0)
    untouched = ~order0
    np.testing.assert_allclose(out_masked[:, untouched],
                               out_clean[:, untouched], atol=1e-4)


def test_niriss_yaml_with_f277w_passes_config_validation():
    """Check NIRISS YAML with f277w passes config validation."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg = apply_current_niriss_settings(config.load_config(
        repo_root / 'tests' / 'v2' / 'data' / 'run_optimize_niriss.yaml'))
    assert cfg['f277w'] is not None
    config.validate_supported_config(cfg)
    opts = config.fixed_options(cfg)
    assert opts['f277w'] == cfg['f277w']
    plan, initial = config.build_sweep_plan(cfg, opts['mode'])
    assert [name for name, _ in plan] == [
        'OneOverFStep_grp', 'JumpStep', 'BadPixStep', 'Extract']
