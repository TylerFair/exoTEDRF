"""Check extraction configuration, logs and headers against v1."""

import warnings

import numpy as np
import pytest

from exotedrf.v2 import config, products
from exotedrf.v2.optimize import TrialResult


SOSS = {'observing_mode': 'NIRISS/SOSS', 'extract_method': 'box',
        'oof_method': 'scale-achromatic'}
NIRSPEC = {'observing_mode': 'NIRSpec/G395H', 'filter_detector': 'NRS1',
           'oof_method': 'median'}
MIRI = {'observing_mode': 'MIRI/LRS'}


def _validated(base, **patch):
    """Return validated."""
    cfg = dict(base, **patch)
    config.validate_supported_config(cfg)
    return cfg


# Asymmetric widths.

@pytest.mark.parametrize('base', [SOSS, NIRSPEC, MIRI])
def test_fixed_two_sided_apertures_are_accepted_in_every_mode(base):
    """Check fixed two sided apertures are accepted in every mode."""
    cfg = _validated(base, extract_width={'lower': 10, 'upper': 12})
    width = cfg['extract_width']
    assert isinstance(width, config.ExtractAperture)
    assert config.extract_width_halves(width) == (10., 12.)
    assert config.format_extract_width(width) == 'lower=10, upper=12'
    _, initial = config.build_sweep_plan(cfg, cfg['observing_mode'])
    assert initial['extract_width'] == (10, 12)


def test_python_tuple_yaml_tag_is_read_like_v1_full_loader(tmp_path):
    """Check python tuple YAML tag is read like v1 full loader."""
    path = tmp_path / 'run.yaml'
    path.write_text(
        "observing_mode: NIRISS/SOSS\n"
        "extract_width: !!python/tuple [9, 13.5]\n"
        "extract_width_soss2: [7, 8]\n"
        "deepframe: None\n", encoding='utf-8')
    cfg = config.load_config(path)
    assert cfg['extract_width'] == (9, 13.5)
    assert cfg['deepframe'] is None
    config.validate_supported_config(cfg)
    assert config.format_extract_width(cfg['extract_width']) == \
        'lower=9, upper=13.5'
    # Extract_width_soss2 is never indexed by v1, so a list is an aperture.
    assert config.extract_width_halves(cfg['extract_width_soss2']) == (7., 8.)
    opts = config.fixed_options(cfg)
    assert isinstance(opts['extract_width_soss2'], config.ExtractAperture)


def test_arbitrary_python_yaml_tags_stay_unconstructable(tmp_path):
    """Check arbitrary python YAML tags stay unconstructable."""
    path = tmp_path / 'run.yaml'
    path.write_text("extract_width: !!python/object/apply:os.getcwd []\n",
                    encoding='utf-8')
    with pytest.raises(Exception):
        config.load_config(path)


def test_two_sided_sweep_keeps_v1_order_seed_middle_and_log_text():
    """Check two sided sweep keeps v1 order seed middle and log text."""
    cfg = _validated(SOSS, optimize_extract_width=True,
                     extract_width=[[10, 12], [12, 14], [14, 18], [9, 9]])
    plan, initial = config.build_sweep_plan(cfg, 'NIRISS/SOSS')
    (checkpoint, [(name, candidates)]), = plan
    assert (checkpoint, name) == ('Extract', 'extract_width')
    assert [tuple(c) for c in candidates] == [(10, 12), (12, 14), (14, 18),
                                              (9, 9)]
    # V1 seeds with int(np.mean(all numbers)) and extracts phase 1 with the middle candidate.
    assert initial['extract_width'] == int(np.mean([10, 12, 12, 14, 14, 18,
                                                    9, 9]))
    assert tuple(config.phase1_extract_width(cfg)) == (14, 18)
    assert [products.format_log_value(c) for c in candidates] == [
        '[10, 12]', '[12, 14]', '[14, 18]', '[9, 9]']


@pytest.mark.parametrize('values, match', [
    ([{'lower': 1, 'upper': 2}, {'lower': 2, 'upper': 3}], 'mappings'),
    ([24, [10, 12]], 'mixture'),
    ([[10, 12, 14]], 'two-element'),
    ([[0, 12]], 'strictly positive'),
])
def test_invalid_two_sided_sweeps_fail_like_v1(values, match):
    """Check invalid two sided sweeps fail like v1."""
    with pytest.raises(ValueError, match=match):
        _validated(SOSS, optimize_extract_width=True, extract_width=values)


def test_fixed_extract_width_list_follows_v1_middle_then_first():
    """Check fixed extract width list follows v1 middle then first."""
    cfg = dict(SOSS, optimize_extract_width=False, extract_width=[20, 30, 40],
               optimize_box_size=True, box_size=[3, 5])
    config.validate_supported_config(cfg)
    with pytest.warns(UserWarning, match='final extraction uses 20'):
        plan, initial = config.build_sweep_plan(cfg, 'NIRISS/SOSS')
    assert [name for name, _ in plan] == ['BadPixStep']
    assert initial['extract_width'] == 20
    assert config.phase1_extract_width(cfg) == 30


# 'optimize', None.

@pytest.mark.parametrize('base', [SOSS, NIRSPEC, MIRI])
def test_optimize_width_is_accepted_without_phase1_sweeps(base):
    """Check optimize width is accepted without phase1 sweeps."""
    cfg = _validated(base, extract_width='optimize')
    plan, initial = config.build_sweep_plan(cfg, cfg['observing_mode'])
    assert plan == [] and initial['extract_width'] == 'optimize'


@pytest.mark.parametrize('key', ['extract_width', 'extract_width_soss2'])
def test_optimize_width_with_phase1_sweeps_fails_like_v1(key):
    """Check optimize width with phase1 sweeps fails like v1."""
    cfg = dict(SOSS, extract_width=30, optimize_box_size=True,
               box_size=[3, 5])
    cfg[key] = 'optimize'
    config.validate_supported_config(cfg)
    with pytest.raises(ValueError, match='phase-1'):
        config.build_sweep_plan(cfg, 'NIRISS/SOSS')


def test_box_extraction_rejects_unknown_width_strings_and_none():
    """Check box extraction rejects unknown width strings and none."""
    with pytest.raises(ValueError, match="'optimize'"):
        _validated(SOSS, extract_width='wide')
    with pytest.raises(ValueError, match='box extraction'):
        _validated(NIRSPEC, extract_width=None)


# Optimal extraction and method fallbacks.

@pytest.mark.parametrize('base', [NIRSPEC, MIRI])
def test_optimal_extraction_accepts_v1_widths(base):
    """Check optimal extraction accepts v1 widths."""
    for width in (8, 7.5, None):
        cfg = _validated(base, extract_method='optimal', extract_width=width)
        assert config.fixed_options(cfg)['extract_method'] == 'optimal'
    cfg = _validated(base, extract_method='optimal',
                     optimize_extract_width=True, extract_width=[4, 6, 8])
    plan, _ = config.build_sweep_plan(cfg, cfg['observing_mode'])
    assert plan[0][0] == 'Extract'


@pytest.mark.parametrize('base', [NIRSPEC, MIRI])
@pytest.mark.parametrize('width', ['optimize', {'lower': 2, 'upper': 3}])
def test_optimal_extraction_rejects_widths_v1_cannot_halve(base, width):
    """Check optimal extraction rejects widths v1 cannot halve."""
    with pytest.raises(ValueError, match='finite and positive'):
        _validated(base, extract_method='optimal', extract_width=width)


@pytest.mark.parametrize('base, method, message', [
    (SOSS, 'optimal', 'Optimal extraction not available for NIRISS/SOSS'),
    (NIRSPEC, 'atoca', 'ATOCA extraction selected but observation does not'),
    (MIRI, 'atoca', 'ATOCA extraction selected but observation does not'),
])
def test_v1_method_fallbacks_warn_and_switch_to_box(tmp_path, base, method,
                                                    message):
    """Check v1 method fallbacks warn and switch to box."""
    cfg = dict(base, extract_method=method, extract_width=8,
               name_tag='run')
    with pytest.warns(UserWarning, match=message):
        config.validate_supported_config(cfg)
    assert cfg['extract_method'] == 'box'
    assert config.fixed_options(cfg)['extract_method'] == 'box'
    layout = products.output_layout(cfg, output_dir=tmp_path)
    assert layout['spectra'].name == 'run_box_spectra_fullres.fits'


def test_supported_methods_do_not_warn():
    """Check supported methods do not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _validated(SOSS, extract_width=30)
        _validated(NIRSPEC, extract_method='optimal', extract_width=6)


# Stage3_kwargs, deepframe.

def test_stage3_kwargs_whitelist_and_optimal_control_precedence():
    """Check stage3 keywords whitelist and optimal control precedence."""
    cfg = _validated(MIRI, extract_method='optimal', extract_width=8,
                     opt_max_iter=5, stage3_kwargs={
                         'opt_max_iter': 12, 'opt_var_thresh': 16,
                         'Extract1dStep': {'allow_miri_slope': True}})
    opts = config.fixed_options(cfg)
    # V1's optimizer reaches run_stage3 only through stage3_kwargs.
    assert (opts['opt_max_iter'], opts['opt_var_thresh']) == (12, 16)
    assert opts['stage3_allow_miri_slope'] is True
    assert config.fixed_options(
        dict(MIRI, opt_max_iter=5))['opt_max_iter'] == 5
    assert config.fixed_options(MIRI)['stage3_allow_miri_slope'] is False
    # V1 ignores use_pastasoss outside NIRISS/SOSS.
    with pytest.warns(UserWarning, match='NIRISS/SOSS only'):
        _validated(MIRI, stage3_kwargs={'Extract1dStep': {
            'use_pastasoss': True}})
    # An empty step mapping is a no-op in v1 (config_compat registry).
    _validated(SOSS, stage3_kwargs={'SpeProfileStep': {}})
    with pytest.raises(ValueError, match='allow_miri_slope'):
        _validated(MIRI, stage3_kwargs={'Extract1dStep': {
            'allow_miri_slope': 'yes'}})


@pytest.mark.parametrize('base', [SOSS, NIRSPEC, MIRI])
def test_deepframe_path_is_accepted_and_null_spellings_are_none(base):
    """Check deepframe path is accepted and null spellings are none."""
    cfg = _validated(base, extract_width=8, deepframe='deep.fits')
    assert config.fixed_options(cfg)['deepframe'] == 'deep.fits'
    for spelling in ('null', '', None):
        cfg = _validated(base, extract_width=8, deepframe=spelling)
        assert config.fixed_options(cfg)['deepframe'] is None
    with pytest.raises(ValueError, match='FITS path'):
        _validated(base, extract_width=8, deepframe=[1, 2])


# Products metadata and Cost logs.

class _State:
    """Represent a reduction state with extraction results."""
    def __init__(self, aux=None):
        self.aux = aux or {}


def test_spectrum_width_header_matches_v1_format_extract_width():
    """Check spectrum width header matches v1 format extract width."""
    aperture = config._normalize_width_value(
        {'lower': 10, 'upper': 12.5}, 'extract_width', allow_pair_list=False)
    assert products._v1_width_header(aperture, 'box', _State()) == \
        'lower=10, upper=12.5'
    assert products._v1_width_header(30, 'box', _State()) == 30
    assert products._v1_width_header(np.float32(7.5), 'box', _State()) == 7.5
    assert products._v1_width_header(8, 'optimal', _State()) == 'N/A'
    assert products._v1_width_header(None, 'optimal', _State()) == 'N/A'
    selected = _State({'extract_width_selected': {1: 27.0, 2: 15.0}})
    assert products._v1_width_header('optimize', 'box', selected) == 27
    assert isinstance(
        products._v1_width_header('optimize', 'box', selected), int)


def _trial(phase, parameter, value, params, cost):
    """Return trial."""
    return TrialResult(phase=phase, checkpoint='x', parameter=parameter,
                       candidate_index=0, value=value, params=params,
                       duration_s=1.0, cost=cost, scatter=np.zeros(2))


def test_cost_log_matches_v1_phase1_seed_and_phase2_aperture_text(tmp_path):
    """Check cost log matches v1 phase1 seed and phase2 aperture text."""
    cfg = _validated(SOSS, optimize_extract_width=True,
                     extract_width=[[10, 12], [12, 14], [14, 18]],
                     optimize_box_size=True, box_size=[3, 5])
    plan, initial = config.build_sweep_plan(cfg, 'NIRISS/SOSS')
    middle = config.phase1_extract_width(cfg)
    candidates = plan[-1][1][0][1]
    trials = [
        _trial(1, 'box_size', 3, {'box_size': 3, 'extract_width': middle},
               .5),
        _trial(1, 'box_size', 5, {'box_size': 5, 'extract_width': middle},
               .4),
        *[_trial(2, 'extract_width', c, {'box_size': 5, 'extract_width': c},
                 .3 - 0.01 * i) for i, c in enumerate(candidates)],
    ]
    paths = products.output_layout(cfg, output_dir=tmp_path, create=True)
    products.write_optimizer_logs(
        paths, trials, ['box_size', 'extract_width'],
        products.jsonable({'initial_params': initial}))
    rows = [line.split('\t') for line in
            paths['cost'].read_text().splitlines()]
    assert rows[0] == ['box_size', 'extract_width', 'duration_s', 'cost']
    # V1 phase-1 rows log current_best's int(mean) seed (13 here) .
    assert [row[1] for row in rows[1:3]] == ['13', '13']
    # ... and phase-2 rows log str(width) of each YAML candidate.
    assert [row[1] for row in rows[3:]] == ['[10, 12]', '[12, 14]',
                                           '[14, 18]']
