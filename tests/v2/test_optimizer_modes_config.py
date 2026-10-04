"""Run-mode YAML keys: v1 validation rules and v1 helper parity."""

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from exotedrf.v2 import adhoc
from exotedrf.v2 import config as v2config

from .optimizer_modes_fixtures import v1_optimize_functions


def _soss(**updates):
    """Return SOSS."""
    cfg = {'observing_mode': 'NIRISS/SOSS', 'extract_method': 'box',
           'oof_method': 'scale-achromatic'}
    cfg.update(updates)
    return cfg


@pytest.mark.parametrize('mode', ['NIRISS/SOSS', 'NIRSpec/G395H',
                                  'MIRI/LRS'])
@pytest.mark.parametrize('flag', ['optimize_extract_width_only',
                                  'from_pca_only', 'optimize_from_pca_only'])
def test_ad_hoc_modes_are_no_longer_rejected(mode, flag):
    """Check ad hoc modes are no longer rejected."""
    cfg = {'observing_mode': mode, 'extract_method': 'box', flag: True,
           'remove_components': [2]}
    if mode.startswith('NIRSpec'):
        cfg.update(filter_detector='NRS1', oof_method='median')
    if mode.startswith('NIRISS'):
        cfg.update(oof_method='scale-achromatic')
    v2config.validate_supported_config(cfg)
    expected = ('extract_width_only' if 'extract' in flag
                else 'from_pca_only')
    assert v2config.ad_hoc_mode(cfg) == expected


def test_restart_filetags_are_accepted():
    """Check restart filetags are accepted."""
    for tag in ('uncal', 'gainscalestep', 'badpixstep', 'rateints', 'jump'):
        v2config.validate_supported_config(_soss(input_filetag=tag))
    with pytest.raises(ValueError, match='non-empty string'):
        v2config.validate_supported_config(_soss(input_filetag=''))


def test_saturation_flag_is_irrelevant_once_dqinit_is_skipped():
    # V1 never reads SaturationStep : warn, not raise.
    """Check saturation flag is irrelevant once DQ initialization is skipped."""
    with pytest.warns(UserWarning, match='SaturationStep'):
        v2config.validate_supported_config(_soss(SaturationStep='skip'))
    v2config.validate_supported_config(
        _soss(SaturationStep='skip', DQInitStep='skip'))


@pytest.mark.parametrize('patch, match', [
    ({'optimize_extract_width_only': True, 'from_pca_only': True},
     'cannot both be True'),
    ({'optimize_extract_width_only': True, 'optimize_box_size': True,
      'box_size': [3, 5]}, 'optimize_box_size must be False when '
                           'optimize_extract_width_only=True'),
    ({'from_pca_only': True, 'remove_components': [1],
      'optimize_time_window': True, 'time_window': [3, 5]},
     'optimize_time_window must be False when from_pca_only=True'),
    ({'from_pca_only': True, 'remove_components': None},
     'remove_components must be set when from_pca_only=True'),
    ({'optimize_from_pca_only': True, 'remove_components': 'None'},
     'remove_components must be set'),
    ({'debug_mode': True, 'v2_search_strategy': 'beam'},
     'requires v2_search_strategy=greedy'),
])
def test_run_mode_rules_follow_v1(patch, match):
    """Check run mode rules follow v1."""
    with pytest.raises(ValueError, match=match):
        v2config.validate_supported_config(_soss(**patch))


def test_extract_only_allows_its_own_width_sweep():
    """Check extract only allows its own width sweep."""
    v2config.validate_supported_config(_soss(
        optimize_extract_width_only=True, optimize_extract_width=True,
        extract_width=[20, 30]))
    v2config.validate_supported_config(_soss(
        from_pca_only=True, remove_components=[2],
        optimize_extract_width=True, extract_width=[20, 30]))


@pytest.mark.parametrize('key, value', [
    ('debug_mode', 'yes'), ('reuse_first_pass_extract_width', 1),
    ('from_pca_only', 'True'), ('archive_to_longterm_storage', 3),
    ('first_pass_extract_method', '')])
def test_run_mode_key_types(key, value):
    """Check run mode key types."""
    with pytest.raises((TypeError, ValueError)):
        v2config.validate_supported_config(_soss(**{key: value}))


def test_archive_destination_null_spellings_are_accepted():
    """Check archive destination null spellings are accepted."""
    for value in (None, 'None', 'null', '', '/tmp/archive'):
        v2config.validate_supported_config(
            _soss(archive_to_longterm_storage=value))


WIDTH_CASES = [30, 30.5, '30', '30.0', '30.5', 'lower=3, upper=4.5',
               'lower = 2,upper=3', '[3, 4]', '[1, 2, 3]', [3, 4], (5, 6),
               [1, 2, 3], {'lower': 1, 'upper': 2},
               "{'lower': 2, 'upper': 3}", None, 'None', 'null', '',
               'abc', np.float64(7.0)]


@pytest.mark.parametrize('value', WIDTH_CASES)
def test_parse_extract_width_metadata_matches_v1(value):
    """Check parse extract width metadata matches v1."""
    v1 = v1_optimize_functions('parse_extract_width_metadata')
    expected = v1['parse_extract_width_metadata'](value)
    got = adhoc.parse_extract_width_metadata(value)
    assert got == expected and type(got) is type(expected)


@pytest.mark.parametrize('value', [None, 3, 2.5, [5, 7], (1,), np.array([4]),
                                   'x', np.nan, {'lower': 1, 'upper': 2}])
def test_format_log_value_matches_v1(value):
    """Check format log value matches v1."""
    v1 = v1_optimize_functions('format_log_value')
    assert adhoc.format_log_value(value) == v1['format_log_value'](value)


def _seed_log(path):
    """Return seed log."""
    path.write_text('box_size\textract_width\tduration_s\tcost\n'
                    '3\t24\t10.0\t0.5\n5\t26\t11.0\t0.25\n'
                    '7\t28\t12.0\tnan\n')


def test_prepare_cost_log_and_rows_match_v1(tmp_path):
    """Check prepare cost log and rows match v1."""
    v1_dir, v2_dir = tmp_path / 'v1', tmp_path / 'v2'
    v1_dir.mkdir()
    v2_dir.mkdir()
    _seed_log(v1_dir / 'Cost_run.txt')
    _seed_log(v2_dir / 'Cost_run.txt')
    v1 = v1_optimize_functions('prepare_cost_log', 'append_cost_log_row',
                               'format_log_value', outdir_f=str(v1_dir))
    required = ['ad_hoc_mode', 'remove_components', 'extract_width']
    _, v1_cols, v1_offset, v1_best = v1['prepare_cost_log']('run', required)
    cols, offset, best = adhoc.prepare_cost_log(v2_dir / 'Cost_run.txt',
                                                required)
    assert (cols, offset, best) == (v1_cols, v1_offset, v1_best)
    base = dict(best, ad_hoc_mode='from_pca_only', remove_components=[5, 7])
    for width, cost in ((24, 0.1), (26, 0.05)):
        row = dict(base, extract_width=width)
        v1['append_cost_log_row'](str(v1_dir / 'Cost_run.txt'), v1_cols,
                                  row, 1.25, cost)
        adhoc.append_cost_log_row(v2_dir / 'Cost_run.txt', cols, row, 1.25,
                                  cost)
    assert (v2_dir / 'Cost_run.txt').read_text() == \
        (v1_dir / 'Cost_run.txt').read_text()


def test_new_cost_log_layout_matches_archived_v1_header(tmp_path):
    """Check new cost log layout matches archived v1 header."""
    cols, offset, best = adhoc.prepare_cost_log(
        tmp_path / 'Cost_new.txt',
        ['ad_hoc_mode', 'remove_components', 'extract_width'])
    assert offset == 0 and best == {}
    # Archived v1 TOI-674 MIRI from_pca_only log header.
    assert (tmp_path / 'Cost_new.txt').read_text().splitlines()[0] == \
        'ad_hoc_mode\tremove_components\textract_width\tduration_s\tcost'


def test_best_logged_width_matches_v1(tmp_path):
    """Check best logged width matches v1."""
    (tmp_path / 'Cost_run.txt').write_text(
        'extract_width\tduration_s\tcost\n24\t1\t0.3\n'
        '"lower=3, upper=4"\t1\t0.1\n\t1\t0.01\n30\t1\tnan\n')
    v1 = v1_optimize_functions('find_best_logged_extract_width',
                               'parse_extract_width_metadata',
                               outdir_f=str(tmp_path))
    with pytest.raises(ValueError, match='any valid logged'):
        v1['find_best_logged_extract_width']('run')
    # V2 returns the intended lowest-cost row with a non-empty width.
    assert adhoc.find_best_logged_extract_width(
        tmp_path / 'Cost_run.txt') == {'lower': 3.0, 'upper': 4.0}
    (tmp_path / 'Cost_bad.txt').write_text(
        'extract_width\tduration_s\tcost\n\t1\t0.1\n24\t1\tnan\n')
    with pytest.raises(ValueError, match='any valid logged'):
        adhoc.find_best_logged_extract_width(tmp_path / 'Cost_bad.txt')


def _spectrum(path, width):
    """Return spectrum."""
    hdu = fits.PrimaryHDU()
    hdu.header['WIDTH'] = width
    hdu.writeto(path)


@pytest.mark.parametrize('cfg, spectrum, log, expected', [
    ({'optimize_extract_width': True, 'extract_width': [1, 2]}, None, None,
     [1, 2]),
    ({'extract_width': 22}, 30, None, 22),
    ({'extract_width': None, 'reuse_first_pass_extract_width': True}, 34,
     None, 34),
    ({'extract_width': None, 'reuse_first_pass_extract_width': True},
     'lower=3, upper=5', None, {'lower': 3.0, 'upper': 5.0}),
    ({'extract_width': None, 'reuse_first_pass_extract_width': True}, None,
     'extract_width\tduration_s\tcost\n26\t1\t0.2\n28\t1\t0.1\n', 28),
])
def test_resolve_ad_hoc_width_matches_v1(tmp_path, cfg, spectrum, log,
                                         expected):
    """Check resolve ad hoc width matches v1."""
    stage3, files = tmp_path / 'Stage3', tmp_path / 'Files'
    stage3.mkdir()
    files.mkdir()
    if spectrum is not None:
        _spectrum(stage3 / 'T_box_spectra_fullres.fits', spectrum)
    if log is not None:
        (files / 'Cost_run.txt').write_text(log)
    cfg = dict(cfg, name_tag='run')
    v1 = v1_optimize_functions(
        'resolve_ad_hoc_extract_width', 'find_stage3_spectrum_file',
        'parse_extract_width_metadata', 'find_best_logged_extract_width',
        outdir_s3=str(stage3) + '/', outdir_f=str(files))
    if log is not None:
        # The cost-log fallback hits v1's always-raising ``is not True``.
        with pytest.raises(ValueError, match='any valid logged'):
            v1['resolve_ad_hoc_extract_width'](cfg)
    else:
        assert v1['resolve_ad_hoc_extract_width'](cfg) == expected
    assert adhoc.resolve_ad_hoc_extract_width(
        cfg, stage3_dirs=[stage3], cost_paths=[files / 'Cost_run.txt']) == \
        expected


def test_resolve_ad_hoc_width_errors_like_v1(tmp_path):
    """Check resolve ad hoc width errors like v1."""
    with pytest.raises(ValueError, match='No extract_width specified'):
        adhoc.resolve_ad_hoc_extract_width(
            {'extract_width': None}, stage3_dirs=[tmp_path],
            cost_paths=[tmp_path / 'Cost_.txt'])
    with pytest.raises(FileNotFoundError, match='No optimizer cost log'):
        adhoc.resolve_ad_hoc_extract_width(
            {'extract_width': None, 'reuse_first_pass_extract_width': True},
            stage3_dirs=[tmp_path], cost_paths=[tmp_path / 'Cost_.txt'])


def test_ad_hoc_width_list_rules():
    """Check ad hoc width list rules."""
    assert adhoc.ad_hoc_widths({'extract_width': [3, 4]}) == [3]
    assert adhoc.ad_hoc_widths({'extract_width': 5}) == [5]
    assert adhoc.ad_hoc_widths({'optimize_extract_width': True,
                                'extract_width': [3, 4]}) == [3, 4]
    with pytest.raises(ValueError, match='must be a list'):
        adhoc.ad_hoc_widths({'optimize_extract_width': True,
                             'extract_width': 4})


def test_stage2_pattern_preference_matches_v1(tmp_path):
    """Check stage2 pattern preference matches v1."""
    pca_first = adhoc.stage2_patterns({'remove_components': [2]}, tmp_path)
    assert pca_first[0].endswith('*_pcareconstructstep.fits')
    for cfg in ({'remove_components': None}, {'remove_components': []},
                {'remove_components': [2], 'PCAReconstructStep': 'skip'}):
        assert adhoc.stage2_patterns(cfg, tmp_path)[0].endswith(
            '*_badpixstep.fits')
    assert pd is not None
