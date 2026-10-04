"""Check Stage-1 and Stage-2 configuration and reference compatibility."""

import os

from exotedrf.v2 import run_dms, stage_kwargs, stages, trace

FILES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'files')


def test_dms_step_lists_drop_removed_steps():
    """Check that pipeline step lists exclude removed calibration steps."""
    removed = {'ResetStep', 'Extract2DStep', 'SourceTypeStep',
               'WaveCorrStep'}
    assert not removed & set(run_dms.DMS_STAGE1_STEPS)
    assert not removed & set(run_dms.DMS_STAGE2_STEPS)
    assert 'AssignWCSStep' in run_dms.DMS_STAGE2_STEPS


def test_graphs_do_not_run_removed_steps():
    """Check that instrument graphs ignore removed calibration steps."""
    removed = {'ResetStep', 'Extract2DStep', 'SourceTypeStep',
               'WaveCorrStep'}
    # Stale 'run' switches in an old YAML must be ignored, not executed.
    opts = {name: 'run' for name in removed}
    for build in (stages.miri_steps, stages.nirspec_steps):
        assert not removed & {step.name for step in build(opts)}


def test_removed_step_kwargs_are_accepted_and_ignored():
    """Check removed step keywords are accepted and ignored."""
    cfg = {'observing_mode': 'NIRISS/SOSS',
           'stage1_kwargs': {'ResetStep': {'whatever': 1}},
           'stage2_kwargs': {'SourceTypeStep': {'a': 1},
                             'WaveCorrStep': {}, 'Extract2DStep': None}}
    result = stage_kwargs.translate_stage_kwargs(cfg)
    assert any('ResetStep' in w for w in result.warnings)
    assert any('SourceTypeStep' in w for w in result.warnings)


def test_badpix_new_kwargs_are_translated_strictly():
    """Check boolean handling for bad-pixel correction keywords."""
    cfg = {'observing_mode': 'NIRISS/SOSS',
           'stage2_kwargs': {'BadPixStep': {
               'median_high_variance': True, 'preserve_saturated': True,
               'clear_interpolated_dq': 1}}}
    result = stage_kwargs.translate_stage_kwargs(cfg)
    assert result.options['median_high_variance'] is True
    assert result.options['preserve_saturated'] is True
    # V1 tests ``is True``: a truthy non-bool does not enable the option.
    assert result.options['clear_interpolated_dq'] is False


def test_soss_tracetable_choice_and_resolution():
    """Check SOSS trace-reference selection for both subarrays."""
    assert trace.soss_tracetable_name('SUBSTRIP96').endswith('0022.fits')
    assert trace.soss_tracetable_name('SUBSTRIP256').endswith('0023.fits')
    for subarray in ('SUBSTRIP96', 'SUBSTRIP256'):
        path = trace.resolve_soss_tracetable(subarray, (FILES,),
                                             download=False)
        assert path == os.path.join(FILES, trace.soss_tracetable_name(
            subarray))
    assert trace.resolve_soss_tracetable(
        'SUBSTRIP96', ('/nonexistent',), download=False) is None or \
        os.environ.get('CRDS_PATH')
