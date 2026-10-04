"""Check joined and segmented NIRSpec and MIRI reductions."""

import numpy as np
import pytest

from exotedrf.v2 import core, stages, streaming
from exotedrf.v2.core import ObsMeta, RampCube
from exotedrf.v2.pipeline import CheckpointStore, PipelineState


# NIRSpec fixture (mirrors tests/v2/test_nirspec_stages.py's all-enabled run).

def _nirspec_meta(nints, ngroups, edges, *, mode='NIRSpec/PRISM',
                  subarray='SUB512'):
    """Return NIRSpec meta."""
    header = {
        'EXP_TYPE': 'NRS_BRIGHTOBJ', 'GRATING': mode.split('/')[-1],
        'NFRAMES': 1, 'TFRAME': 0.902, 'TGROUP': 0.902, 'NINTS': nints,
    }
    return ObsMeta(
        mode=mode, detector='NRS1', subarray=subarray,
        frame_time=0.902, ngroups=ngroups,
        int_times=60000. + np.arange(nints) * 1e-4,
        baseline_ints=np.asarray([4]),
        segment_edges=np.asarray(edges),
        filenames=tuple(f'seg{i + 1:03d}_uncal.fits'
                        for i in range(len(edges))),
        extra={
            'header': dict(header),
            'segment_headers': tuple(dict(header) for _ in edges),
            'segment_int_starts': tuple(
                [1] + [int(edge) + 1 for edge in edges[:-1]]),
            'exposure_nints': nints,
        },
    )


def build_nirspec_case(centroids='explicit'):
    """Two-segment NIRSpec graph with every supported step enabled.

    Parameters
    ----------
    centroids : str
        Trace-coordinate source selection.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
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
    meta = _nirspec_meta(nints, ngroups, [12, 24], mode=mode)
    cube = RampCube(data, np.zeros_like(data, np.uint8),
                    np.zeros((dimy, dimx), np.uint32), meta)

    lin_coeffs = np.zeros((2, dimy, dimx), np.float32)
    lin_coeffs[1] = 1.
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
        'wave_map': np.broadcast_to(
            np.linspace(5., 1., dimx), (dimy, dimx)).copy(),
    }
    opts = {name: 'run' for name in stages._NIRSPEC_STEP_CONTROLS}
    opts.update({
        'Extract2DStep': 'run', 'WaveCorrStep': 'run',
        'INLCorrStep': 'skip', 'RefPixStep': 'skip', 'FlatFieldStep': 'skip',
        'BackgroundStep': 'skip',
        'mode': mode, 'input_dir': '.', 'filter_detector': 'NRS1',
        'oof_method': 'median', 'superbias_method': 'crds',
        'extract_method': 'box', 'pca_components': 2,
        'remove_components': None, 'flag_up_ramp': False,
        'flag_in_time': True, 'jump_threshold': 15,
        'saturation_threshold': 80, 'mask_do_not_use_pixels': True, 'mask_saturated_pixels': True,
        'wave_range': [1., 5.], 'w1': 0., 'w2': 1.,
        'centroids': (None if centroids is None else {
            'xpos': np.arange(14, dimx, dtype=float),
            'ypos': np.full(dimx - 14, 7., dtype=float),
        }),
    })
    ctx = stages.prepare_nirspec_context(cube, opts, refpack=refpack)
    params = {
        'nirspec_mask_width': 4, 'time_jump_threshold': 1e6, 'time_window': 3,
        'space_outlier_threshold': 1e6, 'time_outlier_threshold': 1e6,
        'box_size': 2, 'window_size': 3, 'extract_width': 4,
    }
    return cube, ctx, params


# MIRI fixture (mirrors tests/v2/test_miri_stages.py's two-segment run).

def _miri_meta(nints, ngroups, edges, starts):
    """Return MIRI meta."""
    header = {
        'TGROUP': 2., 'TFRAME': 2., 'NFRAMES': 1, 'GROUPGAP': 0,
        'NINTS': nints, 'READPATT': 'FASTR1', 'NSAMPLES': 1, 'SUBSTRT1': 1,
    }
    return ObsMeta(
        mode='MIRI/LRS', detector='MIRIMAGE', subarray='SLITLESSPRISM',
        frame_time=2., ngroups=ngroups,
        int_times=60000. + np.arange(nints) * 1e-4,
        baseline_ints=np.asarray([nints]),
        segment_edges=np.asarray(edges, dtype=int),
        filenames=tuple(f'seg{i + 1:03d}_uncal.fits'
                        for i in range(len(edges))),
        extra={
            'header': dict(header),
            'segment_headers': tuple(dict(header) for _ in edges),
            'segment_int_starts': tuple(starts),
            'exposure_nints': nints,
        },
    )


def build_miri_case(emicorr=False):
    # V1's temporal bad-pixel statistic uses integrations [-16:-6].
    """Create a two-segment MIRI observation and reduction graph.

    Parameters
    ----------
    emicorr : bool
        Emicorr option.

    Returns
    -------
    result : tuple
        Calculated arrays and auxiliary results.
    """
    nints, ngroups, dimy, dimx = 24, 8, 8, 48
    yy, xx = np.mgrid[:dimy, :dimx]
    trace_profile = 80. * np.exp(-0.5 * ((xx - 36.) / 1.2) ** 2)
    slope = 20. + trace_profile
    integration = np.arange(nints, dtype=np.float32)[:, None, None, None]
    group = np.arange(ngroups, dtype=np.float32)[None, :, None, None]
    data = (100. + .01 * integration +
            group * slope[None, None]).astype(np.float32)
    cube = RampCube(data, np.zeros(data.shape, np.uint8),
                    np.zeros((dimy, dimx), np.uint32),
                    _miri_meta(nints, ngroups, [12, 24], [1, 13]))

    coeffs = np.zeros((2, dimy, dimx), np.float32)
    coeffs[1] = 1.
    zeros_u32 = np.zeros((dimy, dimx), np.uint32)
    zeros_f32 = np.zeros((dimy, dimx), np.float32)
    refpack = {
        'mask_dq': zeros_u32,
        'reset_data': np.zeros((1, ngroups, dimy, dimx), np.float32),
        'reset_dq': zeros_u32,
        'lin_coeffs': coeffs,
        'lin_dq': zeros_u32,
        'dark': np.zeros((1, ngroups, dimy, dimx), np.float32),
        'dark_dq': zeros_u32,
        'average_dark_current': zeros_f32,
        'readnoise': np.ones((dimy, dimx), np.float32),
        'gain': np.ones((dimy, dimx), np.float32),
        'gain_factor': np.float32(1.),
        'flat': np.ones((dimy, dimx), np.float32),
        'flat_dq': zeros_u32,
        'flat_err': zeros_f32,
        'wave_map': 5. + .1 * yy + .001 * xx,
    }
    if emicorr:
        refpack.update({
            'emicorr_frequencies': np.asarray([390.625]),
            'emicorr_reference_waves': np.asarray(
                [np.linspace(-1., 1., 20)]),
            'emicorr_reference_wave_lengths': np.asarray([20]),
            'emicorr_rowclocks': np.asarray(82),
            'emicorr_frameclocks': np.asarray(23968),
        })

    opts = {name: 'skip' for name in stages._MIRI_STEP_CONTROLS}
    for name in ('DQInitStep', 'ResetStep', 'LinearityStep', 'JumpStep',
                 'RampFitStep', 'GainScaleStep', 'AssignWCSStep',
                 'SourceTypeStep', 'FlatFieldStep', 'BackgroundStep',
                 'BadPixStep'):
        opts[name] = 'run'
    opts['EmiCorrStep'] = 'run' if emicorr else 'skip'
    opts.update({
        'INLCorrStep': 'skip', 'SuperBiasStep': 'skip', 'RefPixStep': 'skip',
        'DarkCurrentStep': 'skip', 'OneOverFStep_grp': 'skip',
        'OneOverFStep_int': 'skip',
        'mode': 'MIRI/LRS', 'extract_method': 'box',
        'miri_subtract_dark': True, 'miri_drop_groups': 1,
        'miri_background_method': 'median', 'flag_in_time': True,
        'saturation_threshold': 100, 'mask_do_not_use_pixels': True, 'mask_saturated_pixels': True,
        'hot_pixel_map': None, 'outlier_maps': None, 'pca_components': 2,
        'remove_components': None, 'w1': 0., 'w2': 1.,
        'centroids': {
            'xpos': np.full(dimy, 36., dtype=float),
            'ypos': np.arange(dimy, dtype=float),
        },
    })
    ctx = stages.prepare_miri_context(cube, opts, refpack=refpack)
    params = {
        'miri_trace_width': 4., 'miri_background_width': 4.,
        'extract_width': 4., 'time_jump_threshold': 1e6, 'time_window': 3,
        'space_outlier_threshold': 1e6, 'time_outlier_threshold': 1e6,
        'box_size': 2, 'window_size': 3,
    }
    return cube, ctx, params


def _assert_same_reduction(streamed, joined, orders=(1,), host=True):
    """Check same reduction."""
    for name in ('data', 'err', 'dq'):
        streamed_array = getattr(streamed.cube, name)
        joined_array = getattr(joined.cube, name)
        assert core.is_device_array(streamed_array) is not host
        assert streamed_array.dtype == joined_array.dtype
        np.testing.assert_array_equal(core.to_host(streamed_array),
                                      np.asarray(joined_array))
    for order in orders:
        for key in ('wave', 'flux', 'ferr'):
            np.testing.assert_array_equal(
                np.asarray(streamed.aux['spectral_products'][order][key]),
                np.asarray(joined.aux['spectral_products'][order][key]))


# NIRSpec.

@pytest.mark.parametrize('centroids', ['explicit', None])
def test_run_segmented_nirspec_equals_joined_pipeline_run(centroids,
                                                          monkeypatch):
    """Check run segmented NIRSpec equals joined pipeline run."""
    monkeypatch.setenv(streaming.DEVICE_RESIDENT_ENV, 'off')
    cube, ctx, params = build_nirspec_case(centroids=centroids)
    pipeline = stages.build_pipeline('NIRSpec/PRISM', ctx)

    joined = pipeline.run(PipelineState(cube), params)
    streamed = streaming.run_segmented(
        pipeline, PipelineState(cube), params,
        store=CheckpointStore(keep={'Extract'}))

    _assert_same_reduction(streamed, joined)
    if centroids is None:
        # The prepass must reproduce the step's own all-integration stack.
        np.testing.assert_array_equal(
            np.asarray(streamed.aux['centroids_group']['ypos']),
            np.asarray(joined.aux['centroids_group']['ypos']))


def test_nirspec_prepass_traces_the_all_integration_last_group_stack():
    """The prepass value is the joined step's own visit-wide trace."""
    cube, ctx, params = build_nirspec_case(centroids=None)
    pipeline = stages.build_pipeline('NIRSpec/PRISM', ctx)
    names = [step.name for step in pipeline.steps]
    index = names.index('OneOverFStep_grp')

    shared = streaming._nirspec_group_centroids(
        pipeline, PipelineState(cube), params, index, None, 0.)

    # Same prefix, but joined over the whole visit rather than per segment.
    prefix = pipeline.run(PipelineState(cube), params, stop=index)
    expected = stages._trace_nirspec_deepframe(
        stages._nirspec_deepstack_all_ints(prefix.cube.data), ctx)
    np.testing.assert_array_equal(shared['centroids_group']['ypos'],
                                  expected['ypos'])
    np.testing.assert_array_equal(shared['centroids_group']['xpos'],
                                  expected['xpos'])


def test_nirspec_single_segment_needs_no_prepass(monkeypatch):
    """One segment already IS the visit, so nothing is recomputed."""
    cube, ctx, params = build_nirspec_case(centroids=None)
    meta = cube.meta.__class__(**{
        **{f.name: getattr(cube.meta, f.name)
           for f in cube.meta.__dataclass_fields__.values()},
        'segment_edges': np.asarray([cube.data.shape[0]]),
        'filenames': cube.meta.filenames[:1],
        'extra': dict(cube.meta.extra,
                      segment_headers=cube.meta.extra['segment_headers'][:1],
                      segment_int_starts=(1,)),
    })
    single = RampCube(cube.data, cube.groupdq, cube.pixeldq, meta)
    pipeline = stages.build_pipeline('NIRSpec/PRISM', ctx)
    names = [step.name for step in pipeline.steps]

    calls = []
    original = stages._trace_nirspec_deepframe
    monkeypatch.setattr(
        stages, '_trace_nirspec_deepframe',
        lambda frame, context: calls.append(1) or original(frame, context))
    shared = streaming._nirspec_group_centroids(
        pipeline, PipelineState(single), params,
        names.index('OneOverFStep_grp'), None, 0.)
    assert shared == {} and calls == []


def test_nirspec_streaming_refuses_a_visit_derived_superbias():
    """Check NIRSpec streaming refuses a visit derived superbias."""
    cube, ctx, params = build_nirspec_case()
    ctx['opts']['superbias_method'] = 'custom'
    pipeline = stages.build_pipeline('NIRSpec/PRISM', ctx)
    with pytest.raises(ValueError, match='CRDS superbias'):
        streaming.run_segmented(pipeline, PipelineState(cube), params)


# MIRI.

@pytest.mark.parametrize('emicorr', [False, True])
def test_run_segmented_miri_equals_joined_pipeline_run(emicorr, monkeypatch):
    """Check run segmented MIRI equals joined pipeline run."""
    monkeypatch.setenv(streaming.DEVICE_RESIDENT_ENV, 'off')
    cube, ctx, params = build_miri_case(emicorr=emicorr)
    pipeline = stages.build_pipeline('MIRI/LRS', ctx)

    joined = pipeline.run(PipelineState(cube), params)
    streamed = streaming.run_segmented(
        pipeline, PipelineState(cube), params,
        store=CheckpointStore(keep={'Extract'}))

    _assert_same_reduction(streamed, joined)


def test_miri_segment_state_keeps_absolute_integration_numbers():
    """Reset and the Jump reset-artifact mask are absolute-index kernels."""
    cube, _ctx, _params = build_miri_case()
    for index, start in enumerate((1, 13)):
        part = streaming.segment_state(PipelineState(cube), index)
        assert part.cube.meta.extra['segment_int_starts'] == (start,)
        assert part.cube.meta.extra['exposure_nints'] == 24
        assert core.is_device_array(part.cube.data)


# Dispatch and guard rails.

def test_run_segmented_rejects_unsupported_modes():
    """Check run segmented rejects unsupported modes."""
    cube, ctx, params = build_miri_case()
    pipeline = stages.build_pipeline('MIRI/LRS', ctx)
    pipeline.mode = 'NIRCam/WFSS'
    with pytest.raises(ValueError, match='does not support'):
        streaming.run_segmented(pipeline, PipelineState(cube), params)


@pytest.mark.parametrize('builder,mode', [
    (build_nirspec_case, 'NIRSpec/PRISM'),
    (build_miri_case, 'MIRI/LRS'),
])
def test_run_segmented_refuses_ramp_checkpoints(builder, mode):
    """Check run segmented refuses ramp checkpoints."""
    cube, ctx, params = builder()
    pipeline = stages.build_pipeline(mode, ctx)
    with pytest.raises(ValueError, match='ramp checkpoints'):
        streaming.run_segmented(pipeline, PipelineState(cube), params,
                                store=CheckpointStore(keep={'JumpStep'}))


@pytest.mark.parametrize('builder,mode', [
    (build_nirspec_case, 'NIRSpec/PRISM'),
    (build_miri_case, 'MIRI/LRS'),
])
def test_run_segmented_requires_the_builtin_graph(builder, mode):
    """Check run segmented requires the builtin graph."""
    cube, ctx, params = builder()
    pipeline = stages.build_pipeline(mode, ctx)
    pipeline.steps = list(pipeline.steps[1:])
    with pytest.raises(ValueError, match='built-in'):
        streaming.run_segmented(pipeline, PipelineState(cube), params)


# The accelerator-resident join applies to every streamed mode.

@pytest.mark.parametrize(('mode', 'builder'), [
    ('NIRSpec/PRISM', build_nirspec_case), ('MIRI/LRS', build_miri_case)])
def test_device_resident_join_is_bit_exact(monkeypatch, mode, builder):
    """Check device resident join is bit exact."""
    cube, ctx, params = builder()
    pipeline = stages.build_pipeline(mode, ctx)
    joined = pipeline.run(PipelineState(cube), params)

    monkeypatch.setenv(streaming.DEVICE_RESIDENT_ENV, 'on')
    streamed = streaming.run_segmented(
        pipeline, PipelineState(cube), params,
        store=CheckpointStore(keep={'Extract'}))
    _assert_same_reduction(streamed, joined, host=False)
