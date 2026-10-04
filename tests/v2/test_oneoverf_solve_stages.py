"""SOSS ``oof_method: solve`` stage wiring, products, scorer and config."""
import warnings
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from exotedrf.v2 import config, core, products, stages
from exotedrf.v2.core import ObsMeta, RampCube, RateCube
from exotedrf.v2.kernels import oneoverf
from exotedrf.v2.pipeline import PipelineState

from .test_oneoverf_solve_kernel import (_v1, make_case, ref_solve,
                                         v1_centroids)


def _meta(nints, ngroups, edges, starts, subarray='SUBSTRIP256'):
    """Return meta."""
    return ObsMeta('NIRISS/SOSS', 'NIS', subarray, 5.494, ngroups,
                   np.arange(nints, dtype=float), np.array([2, -2]),
                   np.asarray(edges),
                   tuple(f'jwtest_00001-seg{i + 1:03d}_nis_uncal.fits'
                         for i in range(len(edges))),
                   {'segment_int_starts': tuple(starts)})


def _ctx(case, outlier_mask=None, **opts):
    """Return ctx."""
    return {
        'opts': {'oof_method': 'solve', **opts},
        'centroids': {'ypos o1': case.y1, 'ypos o2': case.y2,
                      'ypos o3': case.y3},
        'outlier_mask': outlier_mask,
        'order0_mask': None,
        'soss_timeseries': None,
        'soss_timeseries_o2': None,
    }


class _Model(SimpleNamespace):
    """Represent a minimal ramp data model."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _v1_per_segment(case, deep, edges, starts, width, pixel_mask=None,
                    background=None):
    """Actual v1 oneoverfstep_solve, one call per segment, concatenated."""
    v1_solve, _ = _v1('SUBSTRIP256')
    outs, calcs = [], []
    lo = 0
    for hi, start in zip(edges, starts):
        sl = slice(lo, hi)
        meta = SimpleNamespace(filename='x', exposure=SimpleNamespace(
            integration_start=start, integration_end=start + hi - lo - 1))
        if case.ramp:
            model = _Model(data=case.cube[sl].copy(),
                           groupdq=case.dq[sl].copy(),
                           pixeldq=case.pixeldq.copy(), meta=meta)
        else:
            model = _Model(data=case.cube[sl].copy(), dq=case.dq[sl].copy(),
                           meta=meta)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            result, calc, _ = v1_solve(
                model, deep.copy(), trace_width=width, background=background,
                save_results=False, pixel_mask=pixel_mask,
                centroids=v1_centroids(case))
        outs.append(result.data)
        calcs.append(calc)
        lo = hi
    calc = {order: {key: np.concatenate([c[order][key] for c in calcs])
                    for key in calcs[0][order]} for order in calcs[0]}
    return np.concatenate(outs), calc


def _ramp_state(case, edges, starts, background=None):
    """Return ramp state."""
    nints, ngroups = case.cube.shape[:2]
    cube = RampCube(case.cube.copy(), case.dq.copy(), case.pixeldq.copy(),
                    _meta(nints, ngroups, edges, starts))
    aux = {}
    deep = np.asarray(case.deep)
    aux[stages._OOF_GRP_PREP_KEY] = {
        'deep': deep, 'base_mask': None, 'timeseries': None,
        'centroid_o1': np.asarray(case.y1, np.float32),
        'centroid_o2_padded': np.asarray(case.y2, np.float32),
        'centroid_o3_padded': np.full(case.cube.shape[-1], np.nan,
                                      np.float32),
        'centroid_o2_extract': np.asarray(
            case.y2[np.isfinite(case.y2)], np.float32),
        'background': background,
    }
    return PipelineState(cube, aux)


def _compare(out, diag, ref_out, ref_calc, ramp):
    """Return compare."""
    scale = float(np.nanmax(np.abs(ref_out)))
    tol = 5e-6 * scale
    np.testing.assert_array_equal(np.isnan(out), np.isnan(ref_out))
    np.testing.assert_allclose(out, ref_out, rtol=0, atol=tol)
    assert set(diag) == set(ref_calc)
    for order, calc in ref_calc.items():
        row_e = calc['oof'][:, :, 10] if ramp else calc['oof'][:, 10]
        row_o = calc['oof'][:, :, 11] if ramp else calc['oof'][:, 11]
        np.testing.assert_allclose(diag[order]['oof_e'], row_e, atol=tol)
        np.testing.assert_allclose(diag[order]['oof_o'], row_o, atol=tol)
        for key in ('scale_e', 'scale_o'):
            np.testing.assert_array_equal(np.isnan(diag[order][key]),
                                          np.isnan(calc[key]))
            np.testing.assert_allclose(diag[order][key], calc[key],
                                       rtol=1e-4, atol=1e-5)


def test_group_step_with_outlier_map_matches_v1_per_segment():
    """Check group step with outlier map matches v1 per segment."""
    case = make_case(seed=31, nints=15)
    edges, starts = [8, 15], [1, 9]
    state = _ramp_state(case, edges, starts, background=case.bkg)
    pmask = np.broadcast_to(case.pmask, case.cube.shape[:1] +
                            case.pmask.shape)
    out = stages.step_oneoverf_grp(state, {'soss_outer_mask_width': 13.,
                                           'soss_inner_mask_width': 999.},
                                   _ctx(case, outlier_mask=pmask))
    ref_out, ref_calc = _v1_per_segment(case, case.deep, edges, starts, 13.,
                                        pixel_mask=case.pmask,
                                        background=case.bkg)
    diag = out.aux[stages.OOF_SOLVE_AUX_KEYS['grp']]
    _compare(np.asarray(out.cube.data), diag, ref_out, ref_calc, ramp=True)
    assert stages._OOF_GRP_PREP_KEY not in out.aux
    assert diag['o1']['scale_e'].shape == (15, 3, case.cube.shape[-1])


def test_group_step_without_map_after_reset_window_matches_v1():
    """Check group step without map after reset window matches v1."""
    case = make_case(seed=32, nints=16)
    edges, starts = [7, 16], [300, 307]
    state = _ramp_state(case, edges, starts)
    out = stages.step_oneoverf_grp(state, {'soss_outer_mask_width': 12.},
                                   _ctx(case))
    ref_out, ref_calc = _v1_per_segment(case, case.deep, edges, starts, 12.)
    _compare(np.asarray(out.cube.data),
             out.aux[stages.OOF_SOLVE_AUX_KEYS['grp']], ref_out, ref_calc,
             ramp=True)


def test_group_step_masks_reset_artifact_in_every_group():
    """v1 crashes here (IndexError); v2 implements the documented intent."""
    case = make_case(seed=33, nints=6)
    edges, starts = [6], [222]
    state = _ramp_state(case, edges, starts)
    out = stages.step_oneoverf_grp(state, {'soss_outer_mask_width': 12.},
                                   _ctx(case))
    _, utils = _v1('SUBSTRIP256')
    model = SimpleNamespace(data=case.cube, meta=SimpleNamespace(
        exposure=SimpleNamespace(integration_start=222, integration_end=227)))
    artifact = utils.mask_reset_artifact(model)
    assert artifact.any()
    ref_out, ref_calc = ref_solve(
        case.cube, case.dq.astype(np.uint32) + case.pixeldq[None, None],
        case.deep, case.y1, case.y2, 12., artifact=artifact)
    _compare(np.asarray(out.cube.data),
             out.aux[stages.OOF_SOLVE_AUX_KEYS['grp']], ref_out, ref_calc,
             ramp=True)


def test_integration_step_matches_v1_per_segment_and_ignores_order0():
    """Check integration step matches v1 per segment and ignores order0."""
    case = make_case(seed=34, ramp=False, nints=16)
    edges, starts = [9, 16], [1, 10]
    cube = RateCube(case.cube.copy(), np.ones_like(case.cube), case.dq.copy(),
                    _meta(16, 1, edges, starts))
    ctx = _ctx(case)
    ctx['order0_mask'] = np.ones(case.cube.shape[-2:], bool)
    state = PipelineState(cube, {stages._OOF_INT_DEEP_KEY: case.deep})
    out = stages.step_oneoverf_int(state, {'soss_outer_mask_width': 12.5},
                                   ctx)
    ref_out, ref_calc = _v1_per_segment(case, case.deep, edges, starts, 12.5)
    _compare(np.asarray(out.cube.data),
             out.aux[stages.OOF_SOLVE_AUX_KEYS['int']], ref_out, ref_calc,
             ramp=False)
    assert stages._OOF_INT_DEEP_KEY not in out.aux
    np.testing.assert_array_equal(out.cube.err, cube.err)


def test_device_resident_inputs_give_identical_results():
    """Check device resident inputs give identical results."""
    case = make_case(seed=35, nints=6)
    state = _ramp_state(case, [3, 6], [300, 303])
    params = {'soss_outer_mask_width': 12.}
    host = stages.step_oneoverf_grp(state, params, _ctx(case))
    moved = PipelineState(RampCube(jnp.asarray(state.cube.data),
                                   jnp.asarray(state.cube.groupdq),
                                   state.cube.pixeldq, state.cube.meta),
                          dict(state.aux))
    device = stages.step_oneoverf_grp(moved, params, _ctx(case))
    assert core.is_device_array(device.cube.data)
    np.testing.assert_array_equal(np.asarray(device.cube.data),
                                  np.asarray(host.cube.data))
    for order, values in host.aux[stages.OOF_SOLVE_AUX_KEYS['grp']].items():
        for key, value in values.items():
            np.testing.assert_array_equal(
                device.aux[stages.OOF_SOLVE_AUX_KEYS['grp']][order][key],
                value)


def test_sidecars_follow_v1_names_dtype_and_concatenation(tmp_path):
    """Check sidecars follow v1 names dtype and concatenation."""
    case = make_case(seed=36, nints=14)
    state = _ramp_state(case, [7, 14], [300, 307])
    out = stages.step_oneoverf_grp(state, {'soss_outer_mask_width': 12.},
                                   _ctx(case))
    paths = {'stage1': tmp_path / 'Stage1', 'stage2': tmp_path / 'Stage2'}
    for path in paths.values():
        path.mkdir()
    written = products._write_oneoverf_solve_sidecars(out, paths)
    assert len(written) == 4
    _, calc = _v1_per_segment(case, case.deep, [7, 14], [300, 307], 12.)
    for order in (1, 2):
        for parity, key in (('even', 'scale_e'), ('odd', 'scale_o')):
            # V1: fileroots[0][:-12] + '_nis_oofscaling_<parity>_order<n>'.
            path = (paths['stage1'] /
                    f'jwtest_00001_nis_oofscaling_{parity}_order{order}.npy')
            saved = np.load(path)
            assert saved.dtype == np.float64
            assert saved.shape == calc[f'o{order}'][key].shape
            np.testing.assert_allclose(saved, calc[f'o{order}'][key],
                                       rtol=1e-4, atol=1e-5)
    assert not list(paths['stage2'].iterdir())


def test_solve_plots_use_v1_chromatic_plot(tmp_path, monkeypatch):
    """Check solve plots use v1 chromatic plot."""
    case = make_case(seed=37, nints=4, ngroups=2)
    out = stages.step_oneoverf_grp(
        _ramp_state(case, [4], [300]), {'soss_outer_mask_width': 12.},
        _ctx(case))
    calls = []
    from exotedrf import plotting as v1_plotting
    monkeypatch.setattr(v1_plotting, 'make_oneoverf_chromatic_plot',
                        lambda *a, **k: calls.append((a, k)))
    capture = products.CompatibilityCapture(
        {'stage1': tmp_path, 'stage2': tmp_path}, {'opts': {}})
    capture._render_oneoverf_solve(out, Path(tmp_path), Path(tmp_path))
    assert [Path(k['outfile']).name for _, k in calls] == [
        'oneoverfstep_o1_3.png', 'oneoverfstep_o2_3.png']
    args = calls[0][0]
    assert args[4] == 2 and args[0].shape == (4, 2, case.cube.shape[-1])


def test_solve_scorer_matches_generic_rerun_and_inner_width_is_inert():
    """Check solve scorer matches generic rerun and inner width is inert."""
    case = make_case(seed=38, nints=8)
    state = _ramp_state(case, [8], [300])
    state.aux['centroids_group'] = _ctx(case)['centroids']
    ctx = _ctx(case, extract_method='box')
    ctx['waves'] = {1: np.linspace(2.8, 0.85, case.cube.shape[-1]),
                    2: np.linspace(1.4, 0.6, case.cube.shape[-1])}
    ctx['opts'].update({'w1': 0.0, 'w2': 1.0, 'wave_range': None,
                        'mask_do_not_use_pixels': True,
                        'saturation_rescue': False})
    params = {'soss_inner_mask_width': 40, 'soss_outer_mask_width': 12.,
              'extract_width': 6}
    scorer = stages.prepare_oneoverf_grp_scorer(state, params, ctx)
    assert isinstance(scorer, stages.PreparedOneOverFSolveScorer)
    outer = scorer.evaluate_candidates('soss_outer_mask_width',
                                       [10., 12., 16.], params)
    for width, (cost, _, _) in zip([10., 12., 16.], outer):
        trial = stages.step_oneoverf_grp(
            state, dict(params, soss_outer_mask_width=width), ctx)
        expected, _ = stages.evaluate_optimizer_cost(
            trial, dict(params, soss_outer_mask_width=width), ctx)
        assert cost == pytest.approx(expected, rel=1e-6)
    inner = scorer.evaluate_candidates('soss_inner_mask_width',
                                       [20, 40, 60], params)
    assert len({c for c, _, _ in inner}) == 1
    assert inner[0][0] == outer[1][0]


# Configuration.

BASE = {'observing_mode': 'NIRISS/SOSS', 'extract_method': 'box'}


def test_config_accepts_solve_and_validates_context():
    """Check config accepts solve and validates context."""
    config.validate_supported_config(dict(BASE, oof_method='solve'))
    ctx = {'opts': {'oof_method': 'solve', 'extract_method': 'box'},
           'background_model': np.zeros((4, 4), np.float32),
           'soss_timeseries': None,
           'soss_timeseries_o2': None}
    assert stages._validate_soss_context(ctx) is ctx


def test_top_level_smoothing_scale_is_inert_like_v1():
    """Check top level smoothing scale is inert like v1."""
    with pytest.warns(UserWarning, match='ignored, as in v1'):
        config.validate_supported_config(dict(BASE, smoothing_scale=7))
    opts = config.fixed_options(dict(BASE, smoothing_scale=7))
    assert opts['oof_smoothing_scale_grp'] is None
    assert opts['oof_smoothing_scale_int'] is None
    for mode, detector in (('NIRSpec/G395H', 'NRS1'), ('MIRI/LRS', '')):
        cfg = {'observing_mode': mode, 'filter_detector': detector,
               'smoothing_scale': 3}
        with pytest.warns(UserWarning, match='ignored, as in v1'):
            config.validate_supported_config(cfg)


def test_oneoverf_stage_kwargs_translate_per_level():
    """Check 1/f correction stage keywords translate per level."""
    cfg = dict(BASE, stage1_kwargs={'OneOverFStep': {'smoothing_scale': 4.7,
                                                     'even_odd_rows': False}},
               stage2_kwargs={'OneOverFStep': {'smoothing_scale': 9}})
    config.validate_supported_config(cfg)
    opts = config.fixed_options(cfg)
    assert opts['oof_smoothing_scale_grp'] == 4
    assert opts['oof_even_odd_rows_grp'] is False
    assert opts['oof_smoothing_scale_int'] == 9
    assert opts['oof_even_odd_rows_int'] is True


@pytest.mark.parametrize('kwargs, error', [
    # V1 raises TypeError at run time; v2 rejects it at config load.
    ({'OneOverFStep': {'inner_mask_width': 3}}, ValueError),
    ({'OneOverFStep': {'smoothing_scale': 0.5}}, ValueError),
    ({'OneOverFStep': {'smoothing_scale': True}}, ValueError),
    ({'OneOverFStep': {}, 'JumpStep': {'x': 1}}, NotImplementedError),
])
def test_oneoverf_stage_kwargs_reject_unsupported(kwargs, error):
    """Check 1/f correction stage keywords reject unsupported."""
    with pytest.raises(error):
        config.validate_supported_config(dict(BASE, stage1_kwargs=kwargs))


def test_solve_warns_that_oneoverf_kwargs_are_unused():
    """Check solve warns that 1/f correction keywords are unused."""
    cfg = dict(BASE, oof_method='solve',
               stage2_kwargs={'OneOverFStep': {'smoothing_scale': 3}})
    with pytest.warns(UserWarning, match='oneoverfstep_solve without'):
        config.validate_supported_config(cfg)


def test_default_options_keep_v1_oneoverf_defaults():
    """Check default options keep v1 1/f correction defaults."""
    opts = config.fixed_options(dict(BASE))
    assert opts['oof_smoothing_scale_grp'] is None
    assert opts['oof_smoothing_scale_int'] is None
    assert opts['oof_even_odd_rows_grp'] is True
    assert opts['oof_even_odd_rows_int'] is True


def test_full_graph_streamed_and_joined_solve_runs_agree():
    """Check full graph streamed and joined solve runs agree."""
    from exotedrf.v2.pipeline import CheckpointStore
    from exotedrf.v2.streaming import run_soss_segmented

    from .test_soss_e2e import build_synthetic_soss_case

    cube, ctx, params = build_synthetic_soss_case()
    ctx['opts']['oof_method'] = 'solve'
    params = dict(params, soss_outer_mask_width=8)
    pipeline = stages.build_pipeline('NIRISS/SOSS', ctx)
    joined = pipeline.run(PipelineState(cube), params)
    streamed = run_soss_segmented(pipeline, PipelineState(cube), params,
                                  store=CheckpointStore(keep={'Extract'}))
    for name in ('data', 'err', 'dq'):
        np.testing.assert_array_equal(getattr(streamed.cube, name),
                                      getattr(joined.cube, name))
    nints, dimx = cube.data.shape[0], cube.data.shape[-1]
    for level in ('grp', 'int'):
        key = stages.OOF_SOLVE_AUX_KEYS[level]
        assert set(joined.aux[key]) == set(streamed.aux[key])
        for order, values in joined.aux[key].items():
            assert values['scale_e'].shape[0] == nints
            assert values['scale_e'].shape[-1] == dimx
            for name, value in values.items():
                np.testing.assert_array_equal(
                    streamed.aux[key][order][name], value)
    assert np.isfinite(joined.aux['oneoverf_solve_grp']['o1']['scale_e']).any()


def test_without_plots_only_the_sidecar_scalings_are_retained():
    """Check without plots only the sidecar scalings are retained."""
    case = make_case(seed=39, nints=6)
    out = stages.step_oneoverf_grp(
        _ramp_state(case, [6], [300]), {'soss_outer_mask_width': 12.},
        _ctx(case, do_plots=False))
    diag = out.aux[stages.OOF_SOLVE_AUX_KEYS['grp']]
    assert {key for values in diag.values() for key in values} == {
        'scale_e', 'scale_o'}
