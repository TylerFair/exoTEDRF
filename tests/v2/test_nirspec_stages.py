"""Focused NIRSpec graph and v1 lifecycle tests."""

import numpy as np
import pytest

from exotedrf.v2 import stages
from exotedrf.v2.core import ObsMeta, RampCube, RateCube
from exotedrf.v2.pipeline import PipelineState


def _meta(nints, ngroups, edges, *, mode='NIRSpec/G395H',
          subarray='SUB2048'):
    """Return meta."""
    return ObsMeta(
        mode=mode, detector='NRS1', subarray=subarray,
        frame_time=0.902, ngroups=ngroups,
        int_times=60000. + np.arange(nints) * 1e-4,
        baseline_ints=np.asarray([4]),
        segment_edges=np.asarray(edges),
        filenames=tuple(f'seg{i + 1:03d}_uncal.fits'
                        for i in range(len(edges))),
        extra={
            'header': {
                'EXP_TYPE': 'NRS_BRIGHTOBJ', 'GRATING': mode.split('/')[-1],
                'NFRAMES': 1, 'TFRAME': 0.902, 'TGROUP': 0.902,
                'NINTS': nints,
            },
            'segment_headers': tuple({
                'EXP_TYPE': 'NRS_BRIGHTOBJ',
                'GRATING': mode.split('/')[-1],
                'NFRAMES': 1, 'TFRAME': 0.902, 'TGROUP': 0.902,
                'NINTS': nints,
            } for _ in edges),
            'segment_int_starts': tuple(
                [1] + [int(edge) + 1 for edge in edges[:-1]]),
        },
    )


@pytest.mark.parametrize('level', ['group', 'integration'])
def test_short_nrs1_oof_trace_is_recomputed_per_segment(monkeypatch, level):
    """v1 rejects its shared short NRS1 trace inside every segment call."""
    nints, ngroups, dimy, dimx = 4, 2, 5, 4
    markers = np.asarray([10., 10., 20., 20.], np.float32)
    if level == 'group':
        data = np.broadcast_to(
            markers[:, None, None, None],
            (nints, ngroups, dimy, dimx)).copy()
        cube = RampCube(
            data, np.zeros_like(data, np.uint8),
            np.zeros((dimy, dimx), np.uint32),
            _meta(nints, ngroups, [2, 4]))
        step = stages.step_oneoverf_grp_nirspec
        aux_key = 'centroids_group'
    else:
        data = np.broadcast_to(
            markers[:, None, None], (nints, dimy, dimx)).copy()
        cube = RateCube(
            data, np.ones_like(data), np.zeros_like(data, np.uint32),
            _meta(nints, ngroups, [2, 4]))
        step = stages.step_oneoverf_int_nirspec
        aux_key = 'centroids_int'

    traced_markers = []

    def fake_trace(deepframe, ctx):
        marker = float(np.asarray(deepframe)[0, 0])
        traced_markers.append(marker)
        return {
            'xpos': np.arange(1, dimx, dtype=float),
            'ypos': np.full(dimx - 1, marker, dtype=float),
        }

    applied_traces = []

    def fake_apply(data, base_mask, ypos, width, method):
        del base_mask, width, method
        applied_traces.append(np.array(ypos, copy=True))
        return np.asarray(data)

    monkeypatch.setattr(stages, '_trace_nirspec_deepframe', fake_trace)
    monkeypatch.setattr(
        stages, '_oneoverf_base_mask',
        lambda c, ctx, *, reset=False: np.zeros((nints, dimy, dimx), dtype=bool))
    monkeypatch.setattr(stages, '_apply_oneoverf_nirspec', fake_apply)

    ctx = {
        'centroids': None,
        'nirspec_xstart': 1,
        'opts': {'oof_method': 'median'},
    }
    out = step(
        PipelineState(cube), {'nirspec_mask_width': 2}, ctx)

    # Shared all-file trace first, then the trace on each current segment.
    np.testing.assert_allclose(traced_markers, [15., 10., 20.])
    assert len(applied_traces) == 2
    assert np.isnan(applied_traces[0][0])
    assert np.isnan(applied_traces[1][0])
    np.testing.assert_allclose(applied_traces[0][1:], 10.)
    np.testing.assert_allclose(applied_traces[1][1:], 20.)
    np.testing.assert_allclose(out.aux[aux_key]['ypos'], 15.)


def test_custom_rescale_retraces_each_segment_for_short_explicit_trace(
        monkeypatch):
    """Check custom rescale retraces each segment for short explicit trace."""
    nints, ngroups, dimy, dimx = 4, 2, 12, 4
    data = np.empty((nints, ngroups, dimy, dimx), np.float32)
    data[:2] = 10.
    data[2:] = 20.
    cube = RampCube(
        data, np.zeros_like(data, np.uint8),
        np.zeros((dimy, dimx), np.uint32),
        _meta(nints, ngroups, [2, 4]))
    calls = []

    def fake_trace(deepframe, ctx):
        calls.append(float(np.asarray(deepframe)[0, 0]))
        return {
            'xpos': np.arange(1, dimx, dtype=float),
            'ypos': np.full(dimx - 1, 6., dtype=float),
        }

    monkeypatch.setattr(stages, '_trace_nirspec_deepframe', fake_trace)
    ctx = {
        'opts': {'superbias_method': 'custom-rescale'},
        'centroids': {
            'xpos': np.arange(1, dimx, dtype=float),
            'ypos': np.full(dimx - 1, 6., dtype=float),
        },
    }
    out = stages.step_superbias_nirspec(PipelineState(cube), {}, ctx)

    np.testing.assert_allclose(calls, [10., 20.])
    assert out.aux['superbias_scale_factors'].shape == (nints,)


def test_nirspec_resident_oof_scorer_falls_back_for_multiple_segments():
    """A single resident trace cannot represent v1's per-file retracing."""
    nints, ngroups, dimy, dimx = 4, 2, 5, 4
    data = np.zeros((nints, ngroups, dimy, dimx), np.float32)
    cube = RampCube(
        data, np.zeros_like(data, np.uint8),
        np.zeros((dimy, dimx), np.uint32),
        _meta(nints, ngroups, [2, 4]))

    scorer = stages.prepare_nirspec_oneoverf_scorer(
        PipelineState(cube), {}, {'opts': {'oof_method': 'median'}})

    assert scorer is None


def test_all_enabled_nirspec_graph_preserves_native_detector_axis():
    """Run every supported NIRSpec step with real kernels on two segments."""
    nints, ngroups, dimy, dimx = 24, 3, 16, 32
    yy, xx = np.mgrid[:dimy, :dimx]
    profile = np.exp(-0.5 * ((yy - 7. - 0.3 * np.sin(1.7 * xx)) / 1.5) ** 2)
    data = np.empty((nints, ngroups, dimy, dimx), np.float32)
    for integration in range(nints):
        for group in range(ngroups):
            slope = 25. + profile * (80. + 0.2 * integration)
            data[integration, group] = (
                100. + group * slope + 0.01 * xx + 0.02 * yy)

    mode = 'NIRSpec/PRISM'
    meta = _meta(
        nints, ngroups, [12, 24], mode=mode, subarray='SUB512')
    cube = RampCube(
        data, np.zeros_like(data, np.uint8),
        np.zeros((dimy, dimx), np.uint32), meta)

    lin_coeffs = np.zeros((2, dimy, dimx), np.float32)
    lin_coeffs[1] = 1.
    wave_map = np.broadcast_to(
        np.linspace(5., 1., dimx), (dimy, dimx)).copy()
    refpack = {
        'mask_dq': np.zeros((dimy, dimx), np.uint32),
        'superbias': np.zeros((dimy, dimx), np.float32),
        'dark': np.zeros((ngroups, dimy, dimx), np.float32),
        'average_dark_current': np.zeros((dimy, dimx), np.float32),
        'lin_coeffs': lin_coeffs,
        'lin_dq': np.zeros((dimy, dimx), np.uint32),
        'readnoise': np.full((dimy, dimx), 6., np.float32),
        'gain': np.ones((dimy, dimx), np.float32),
        'gain_factor': np.float32(1.),
        'wave_map': wave_map,
    }
    controls = {name: 'run' for name in stages._NIRSPEC_STEP_CONTROLS}
    opts = {
        **controls,
        'Extract2DStep': 'run',
        'WaveCorrStep': 'run',
        # Instrument-inapplicable reference consumers must remain skipped.
        'INLCorrStep': 'skip',
        'RefPixStep': 'skip',
        'FlatFieldStep': 'skip',
        'BackgroundStep': 'skip',
        'mode': mode,
        'input_dir': '.',
        'filter_detector': 'NRS1',
        'oof_method': 'median',
        'superbias_method': 'crds',
        'extract_method': 'box',
        'pca_components': 2,
        'remove_components': None,
        'flag_up_ramp': False,
        'flag_in_time': True,
        'jump_threshold': 15,
        'saturation_threshold': 80,
        'mask_do_not_use_pixels': True, 'mask_saturated_pixels': True,
        'wave_range': [1., 5.],
        'w1': 0.,
        'w2': 1.,
        'centroids': {
            'xpos': np.arange(14, dimx, dtype=float),
            'ypos': np.full(dimx - 14, 7., dtype=float),
        },
    }
    ctx = stages.prepare_nirspec_context(cube, opts, refpack=refpack)
    params = {
        'nirspec_mask_width': 4,
        'time_jump_threshold': 1e6,
        'time_window': 3,
        'space_outlier_threshold': 1e6,
        'time_outlier_threshold': 1e6,
        'box_size': 2,
        'window_size': 3,
        'extract_width': 4,
    }

    pipeline = stages.build_pipeline(mode, ctx)
    out = pipeline.run(PipelineState(cube), params)

    assert [step.name for step in pipeline.steps] == [
        'DQInitStep', 'SuperBiasStep', 'DarkCurrentStep',
        'OneOverFStep_grp', 'LinearityStep', 'JumpStep', 'RampFitStep',
        'GainScaleStep', 'AssignWCSStep',
        'OneOverFStep_int', 'BadPixStep',
        'PCAReconstructStep', 'Extract',
    ]
    assert isinstance(out.cube, RateCube)
    assert np.isfinite(np.asarray(out.cube.data)).all()
    product = out.aux['spectral_products'][1]
    assert product['wave'].shape == (dimx,)
    assert product['flux'].shape == (nints, dimx)
    assert product['ferr'].shape == (nints, dimx)
    assert np.isnan(product['wave'][:14]).all()
    np.testing.assert_allclose(product['wave'][14:], wave_map[7, 14:])
    assert np.all(np.diff(product['wave'][14:]) < 0.)
    assert np.all(product['flux'][:, :14] == 0.)
    cost, scatter = stages.evaluate_production_cost(out, params, ctx)
    assert np.isfinite(float(cost))
    assert scatter.shape == (dimx,)
