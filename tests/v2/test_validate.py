"""Check validate."""

import json
from dataclasses import dataclass

import numpy as np
import pytest
from astropy.io import fits

from exotedrf.v2 import core, validate
from exotedrf.v2.pipeline import Pipeline, PipelineState, Step


def _write_v1_artifact(path, data, *, int_start=None, exsegnum=None,
                       groupdq=None, pixeldq=None, err=None, dq=None):
    """Return write v1 artifact."""
    primary = fits.PrimaryHDU()
    primary.header['FILENAME'] = path.name
    if int_start is not None:
        primary.header['INTSTART'] = int_start
        primary.header['INTEND'] = int_start + data.shape[0] - 1
    if exsegnum is not None:
        primary.header['EXSEGNUM'] = exsegnum
    hdus = [primary, fits.ImageHDU(data, name='SCI')]
    for name, value in (
            ('ERR', err), ('DQ', dq), ('GROUPDQ', groupdq),
            ('PIXELDQ', pixeldq)):
        if value is not None:
            hdus.append(fits.ImageHDU(value, name=name))
    fits.HDUList(hdus).writeto(path)


def _make_v1_output_tree(tmp_path, *, mismatched_pixeldq=False,
                         root_name='pipeline_outputs_directory'):
    """Create v1 output tree for the test observation."""
    root = tmp_path / root_name
    stage1 = root / 'Stage1'
    stage2 = root / 'Stage2'
    files = root / 'Files'
    stage1.mkdir(parents=True)
    stage2.mkdir()
    files.mkdir()
    spatial = (2, 3)
    pixel_a = np.arange(6, dtype=np.uint32).reshape(spatial)
    pixel_b = pixel_a + np.uint32(mismatched_pixeldq)

    # Deliberately make filename order disagree with INTSTART order.
    for segment, start, value, pixel in (
            (1, 3, 20.0, pixel_b), (2, 1, 10.0, pixel_a)):
        ramp = np.full((2, 2) + spatial, value, np.float32)
        groupdq = np.full(ramp.shape, segment, np.uint32)
        _write_v1_artifact(
            stage1 / f'visit_seg{segment:03d}_nis_dqinitstep.fits', ramp,
            int_start=start, exsegnum=segment, groupdq=groupdq,
            pixeldq=pixel)
        rate = np.full((2,) + spatial, value / 10, np.float32)
        _write_v1_artifact(
            stage2 / f'visit_seg{segment:03d}_nis_pcareconstructstep.fits',
            rate, int_start=start, exsegnum=segment,
            err=np.ones_like(rate), dq=np.full(rate.shape, segment, np.uint32))

    (files / 'Cost_demo.txt').write_text(
        'extract_width\tduration_s\tcost\n'
        '20\t1.0\t3.0\n22\t2.0\t1.5\n', encoding='utf-8')
    (files / 'Scatter_demo.txt').write_text(
        '10 11 12\n20 21 22\n', encoding='utf-8')
    return root


def _write_v1_soss_spectrum(path):
    """Return write v1 SOSS spectrum."""
    path.parent.mkdir(exist_ok=True)
    hdus = [fits.PrimaryHDU()]
    for order in (1, 2):
        wave = np.asarray([2.0, np.nan, 1.0]) + (order - 1) * 2
        flux = np.asarray([[20.0, 99.0, 10.0],
                           [21.0, 99.0, 11.0]]) * order
        ferr = flux / 10
        hdus.extend([
            fits.ImageHDU(wave, name=f'Wave O{order}'),
            fits.ImageHDU(flux, name=f'Flux O{order}'),
            fits.ImageHDU(ferr, name=f'Flux Err O{order}'),
        ])
    fits.HDUList(hdus).writeto(path)


def _write_costable_v1_soss_spectrum(path):
    """Write an already-clipped final spectrum with an analytic cost of 0.5."""
    path.parent.mkdir(exist_ok=True)
    curve = np.asarray([1., 1., 2., 1., 1., 2., 1., 1.])
    hdus = [fits.PrimaryHDU()]
    for order, wave, scales in (
            (1, np.asarray([0.9, 1.0, 1.1]), np.asarray([10., 20., 30.])),
            (2, np.asarray([0.7, 0.8]), np.asarray([40., 50.]))):
        flux = curve[:, None] * scales[None]
        hdus.extend([
            fits.ImageHDU(wave, name=f'Wave O{order}'),
            fits.ImageHDU(flux, name=f'Flux O{order}'),
            fits.ImageHDU(np.ones_like(flux), name=f'Flux Err O{order}'),
        ])
    fits.HDUList(hdus).writeto(path)


def _write_single_trace_spectrum(path, wave, flux, time, *,
                                 wave_error=None, flux_error=None):
    """Return write single trace spectrum."""
    path.parent.mkdir(exist_ok=True)
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    time = np.asarray(time, dtype=float)
    if wave_error is None:
        wave_error = np.full_like(wave, 0.01)
        wave_error[~np.isfinite(wave)] = np.nan
    if flux_error is None:
        flux_error = np.sqrt(np.abs(flux))
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(wave, name='Wave'),
        fits.ImageHDU(np.asarray(wave_error), name='Wave Err'),
        fits.ImageHDU(flux, name='Flux'),
        fits.ImageHDU(np.asarray(flux_error), name='Flux Err'),
        fits.ImageHDU(time, name='Time'),
    ]).writeto(path)


def _write_soss_spectrum(path, orders, time):
    """Return write SOSS spectrum."""
    path.parent.mkdir(exist_ok=True)
    hdus = [fits.PrimaryHDU()]
    for order in (1, 2):
        wave, flux = orders[order]
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        hdus.extend([
            fits.ImageHDU(wave, name=f'Wave O{order}'),
            fits.ImageHDU(np.full_like(wave, .01),
                          name=f'Wave Err O{order}'),
            fits.ImageHDU(flux, name=f'Flux O{order}'),
            fits.ImageHDU(np.sqrt(np.abs(flux)),
                          name=f'Flux Err O{order}'),
        ])
    hdus.append(fits.ImageHDU(np.asarray(time, dtype=float), name='Time'))
    fits.HDUList(hdus).writeto(path)


def _cost_config(*, optimize=False, candidates=None):
    """Return cost config."""
    config = {
        'observing_mode': 'NIRISS/SOSS',
        'baseline_ints': [8],
        'wave_range': [0.7, 1.1],
        'w1': 0.0,
        'w2': 1.0,
        'optimize_extract_width': bool(optimize),
        'extract_width': candidates if optimize else 20,
    }
    return config


def test_compare_array_reports_finite_mask_and_nonfinite_kind():
    """Check compare array reports finite mask and nonfinite kind."""
    reference = np.array([1.0, np.nan, np.inf, 4.0])
    candidate = np.array([1.0 + 1e-7, np.nan, -np.inf, np.nan])
    metrics = validate.compare_array(reference, candidate)

    assert not metrics['passed']
    assert metrics['finite_mask_mismatch_count'] == 1
    assert metrics['nonfinite_kind_mismatch_count'] == 1
    assert metrics['common_finite_count'] == 1
    assert metrics['within_tolerance_count'] == 1


def test_compare_array_requires_exact_integer_dq_bits():
    """Check compare array requires exact integer DQ bits."""
    reference = np.asarray([1 << 28 | 4], dtype=np.uint32)
    candidate = np.asarray([1 << 28], dtype=np.uint32)
    metrics = validate.compare_array(
        reference, candidate,
        validate.ArrayTolerance(rtol=1.0, atol=1.0))
    assert not metrics['passed']
    assert metrics['reason'] == 'integer_mismatch'
    assert metrics['mismatch_count'] == 1


def test_named_stepwise_tolerances_and_machine_readable_report(tmp_path):
    """Check named stepwise tolerances and machine readable report."""
    reference = {
        'RampFit': {'slope': np.array([1.0, 2.0]),
                    'dq': np.array([0, 4], dtype=np.uint32)},
        'Cost': {'value': np.array(0.25)},
    }
    candidate = {
        'RampFit': {'slope': np.array([1.0, 2.002]),
                    'dq': np.array([0, 4], dtype=np.uint32)},
        'Cost': {'value': np.array(0.25001)},
    }
    report = validate.compare_named_steps(
        reference, candidate,
        tolerances={
            'RampFit::slope': {'rtol': 2e-3, 'atol': 0.0},
            'Cost': validate.ArrayTolerance(rtol=1e-3, atol=0.0),
        })
    path = validate.write_report(report, tmp_path / 'nested' / 'report.json')
    loaded = json.loads(path.read_text())

    assert loaded['status'] == 'passed'
    assert loaded['summary'] == {
        'arrays_compared': 3,
        'arrays_passed': 3,
        'steps_compared': 2,
        'steps_passed': 2,
    }
    assert loaded['steps']['RampFit']['arrays']['slope']['max_absolute_error']


def test_missing_named_step_is_a_failure():
    """Check missing named step is a failure."""
    report = validate.compare_named_steps(
        {'DQInit': np.ones(2), 'RampFit': np.ones(2)},
        {'DQInit': np.ones(2)})
    assert report['status'] == 'failed'
    assert report['steps']['RampFit']['reason'] == 'missing_candidate_step'


def test_checkpoint_file_comparison_can_select_strict_shared_steps(tmp_path):
    """Check checkpoint file comparison can select strict shared steps."""
    reference = validate.save_checkpoint_npz({
        'RampFitStep': {'data': np.ones(2)},
        'Cost': {'value': np.asarray(1.0)},
        'V1Only': np.asarray(4),
    }, tmp_path / 'reference.npz')
    candidate = validate.save_checkpoint_npz({
        'RampFitStep': {'data': np.ones(2)},
        'Cost': {'value': np.asarray(1.0)},
        'Optimizer': {'winner.width': np.asarray(20)},
    }, tmp_path / 'candidate.npz')

    report = validate._compare_files(
        str(reference), str(candidate), kind='real-data-stepwise',
        scope='test', selected_steps=('RampFitStep', 'Cost'))
    assert report['passed']
    assert report['selected_steps'] == ['RampFitStep', 'Cost']
    assert set(report['steps']) == {'RampFitStep', 'Cost'}

    with pytest.raises(ValueError, match='candidate missing'):
        validate._compare_files(
            str(reference), str(candidate), kind='real-data-stepwise',
            scope='test', selected_steps=('V1Only',))


def test_npz_checkpoint_convention(tmp_path):
    """Check npz checkpoint convention."""
    path = tmp_path / 'checkpoints.npz'
    np.savez(path, **{
        'DQInit::data': np.ones((2, 3)),
        'DQInit::dq': np.zeros((2, 3), dtype=np.uint32),
        'Cost': np.array(1.5),
    })
    loaded = validate.load_checkpoint_npz(path)
    assert set(loaded) == {'DQInit', 'Cost'}
    np.testing.assert_array_equal(loaded['DQInit']['dq'], 0)
    assert loaded['Cost'] == 1.5
    selected = validate.load_checkpoint_npz(path, steps='Cost')
    assert set(selected) == {'Cost'}
    assert selected['Cost'] == 1.5


def test_save_checkpoint_normalizes_npz_suffix_and_returns_existing_path(
        tmp_path):
    """Check save checkpoint normalizes npz suffix and returns existing path."""
    path = validate.save_checkpoint_npz(
        {'Cost': np.array(1.0)}, tmp_path / 'without_suffix')
    assert path == tmp_path / 'without_suffix.npz'
    assert path.is_file()


def test_selected_checkpoint_does_not_load_unrequested_payload(tmp_path):
    """Check selected checkpoint does not load unrequested payload."""
    path = tmp_path / 'selective.npz'
    # Accessing this unrelated payload with allow_pickle=False would fail.
    np.savez(path, Cost=np.array(1.5), **{
        'Unrequested::data': np.array([object()], dtype=object)})
    loaded = validate.load_checkpoint_npz(path, steps='Cost')
    assert set(loaded) == {'Cost'}
    assert loaded['Cost'] == 1.5


def test_pack_existing_v1_outputs_orders_segments_and_preserves_pixeldq(
        tmp_path):
    """Check pack existing v1 outputs orders segments and preserves PIXELDQ."""
    root = _make_v1_output_tree(tmp_path)
    path = validate.pack_v1_outputs(
        root, tmp_path / 'v1_bundle', log_name='demo')
    packed = validate.load_checkpoint_npz(path)

    assert path == tmp_path / 'v1_bundle.npz'
    assert set(packed) == {'DQInitStep', 'PCAReconstructStep'}
    selected_path = validate.pack_v1_outputs(
        root, tmp_path / 'selected_bundle', include_logs=False,
        include_spectrum=False, steps='DQInitStep')
    selected = validate.load_checkpoint_npz(selected_path)
    assert set(selected) == {'DQInitStep'}
    # INTSTART=1 (seg002) wins over natural filename order.
    np.testing.assert_array_equal(
        packed['DQInitStep']['data'][:, 0, 0, 0], [10, 10, 20, 20])
    assert packed['DQInitStep']['pixeldq'].shape == (2, 3)
    np.testing.assert_array_equal(
        packed['PCAReconstructStep']['data'][:, 0, 0], [1, 1, 2, 2])
    assert 'Optimizer' not in packed


def test_pack_v1_fixed_width_recomputes_final_cost_from_spectrum_not_log(
        tmp_path):
    """Check pack v1 fixed width recomputes final cost from spectrum not log."""
    root = _make_v1_output_tree(tmp_path)
    _write_costable_v1_soss_spectrum(
        root / 'Stage3' / 'visit_box_spectra_fullres.fits')
    packed = validate.load_checkpoint_npz(validate.pack_v1_outputs(
        root, tmp_path / 'fixed.npz', log_name='demo',
        config=_cost_config()))

    # The mixed trial log's global minimum is 1.5.
    assert packed['Cost']['value'] == pytest.approx(0.5)
    np.testing.assert_allclose(packed['Cost']['scatter'], 0.5)
    assert 'Optimizer' not in packed


def test_pack_v1_reconstructs_winner_only_for_proven_trial_blocks(tmp_path):
    """Check pack v1 reconstructs winner only for proven trial blocks."""
    root = _make_v1_output_tree(tmp_path)
    _write_costable_v1_soss_spectrum(
        root / 'Stage3' / 'visit_box_spectra_fullres.fits')
    config = _cost_config(optimize=True, candidates=[20, 22])
    packed = validate.load_checkpoint_npz(validate.pack_v1_outputs(
        root, tmp_path / 'winner.npz', log_name='demo', config=config))

    assert packed['Optimizer']['winner.extract_width'] == 22
    assert packed['Cost']['value'] == pytest.approx(0.5)

    # A candidate-count mismatch makes the block layout unprovable.
    malformed = _cost_config(optimize=True, candidates=[20, 22, 24])
    packed_bad = validate.load_checkpoint_npz(validate.pack_v1_outputs(
        root, tmp_path / 'no_winner.npz', log_name='demo', config=malformed))
    assert 'Optimizer' not in packed_bad
    assert packed_bad['Cost']['value'] == pytest.approx(0.5)


def test_pack_v1_malformed_log_never_displaces_spectrum_cost(tmp_path):
    """Check pack v1 malformed log never displaces spectrum cost."""
    root = _make_v1_output_tree(tmp_path)
    _write_costable_v1_soss_spectrum(
        root / 'Stage3' / 'visit_box_spectra_fullres.fits')
    (root / 'Files' / 'Cost_demo.txt').write_text(
        'extract_width\tduration_s\tcost\n20\t1.0\t3.0\n22\tbroken\n',
        encoding='utf-8')

    packed = validate.load_checkpoint_npz(validate.pack_v1_outputs(
        root, tmp_path / 'malformed.npz', log_name='demo',
        config=_cost_config(optimize=True, candidates=[20, 22])))
    assert packed['Cost']['value'] == pytest.approx(0.5)
    assert 'Optimizer' not in packed


def test_pack_v1_normalizes_unique_stage3_soss_spectrum(tmp_path):
    """Check pack v1 normalizes unique stage3 SOSS spectrum."""
    root = _make_v1_output_tree(tmp_path)
    _write_v1_soss_spectrum(
        root / 'Stage3' / 'visit_box_spectra_fullres.fits')
    packed = validate.load_checkpoint_npz(validate.pack_v1_outputs(
        root, tmp_path / 'with_spectra.npz', log_name='demo'))

    extract = packed['Extract']
    np.testing.assert_array_equal(
        extract['aux.spectral_products.1.wave'], [1.0, 2.0])
    np.testing.assert_array_equal(
        extract['aux.spectral_products.1.flux'], [[10, 20], [11, 21]])
    np.testing.assert_array_equal(
        extract['aux.spectral_products.2.wave'], [3.0, 4.0])
    assert 'Cost' not in packed
    assert 'Optimizer' not in packed


def test_single_trace_spectrum_report_has_structural_and_robust_metrics(
        tmp_path):
    """Check single trace spectrum report has structural and robust metrics."""
    reference_path = tmp_path / 'v1_nrs1_spectra_fullres.fits'
    candidate_path = tmp_path / 'v2_nrs1_spectra_fullres.fits'
    wave = np.asarray([np.nan, 3.0, 3.5, 4.0])
    curve = np.asarray([1., 1.001, .999, 1.002, 1., .998, 1.001, 1.])
    flux = curve[:, None] * np.asarray([0., 100., 200., 300.])[None]
    time = 60000. + np.arange(curve.size) / 86400.
    _write_single_trace_spectrum(reference_path, wave, flux, time)

    candidate_flux = flux * np.asarray([1., 1.2, .8, 1.1])[None]
    candidate_flux[3, 2] *= 1.0002
    _write_single_trace_spectrum(
        candidate_path,
        np.where(np.isfinite(wave), wave + 1e-5, wave),
        candidate_flux, time,
        flux_error=np.sqrt(np.abs(candidate_flux)))

    report = validate.compare_single_trace_spectra(
        reference_path, candidate_path, baseline_ints=[3, -3])

    assert report['passed']
    assert not report['parity_claimed']
    assert report['checks']['shapes']['passed']
    assert report['checks']['wavelength_finite_support']['passed']
    assert report['checks']['nonzero_flux_column_support']['passed']
    assert report['checks']['integration_times']['bitwise_equal']
    assert report['metrics']['common_science_column_count'] == 3
    assert report['metrics'][
        'wavelength_absolute_difference_micron']['maximum'] == \
        pytest.approx(1e-5)
    assert report['metrics'][
        'normalized_flux_residual_ppm']['absolute']['p95'] > 0
    scatter = report['metrics']['per_column_ptp_scatter']
    assert scatter['ratio_candidate_over_reference']['count'] == 3
    assert 0 <= scatter['fraction_candidate_lower_or_equal'] <= 1

    gated = validate.compare_single_trace_spectra(
        reference_path, candidate_path, baseline_ints=[3, -3],
        max_flux_p95_ppm=0., max_scatter_ratio_deviation=10.)
    assert not gated['passed']
    assert not gated['checks']['normalized_flux_p95']['passed']

    ranged = validate.compare_single_trace_spectra(
        reference_path, candidate_path, baseline_ints=[3, -3],
        wave_range=[3., 3.5])
    assert ranged['wave_range_micron'] == [3., 3.5]
    assert ranged['metrics']['common_science_column_count'] == 2


def test_single_trace_spectrum_support_mismatch_is_a_hard_failure(tmp_path):
    """Check single trace spectrum support mismatch is a hard failure."""
    reference_path = tmp_path / 'v1_miri_spectra_fullres.fits'
    candidate_path = tmp_path / 'v2_miri_spectra_fullres.fits'
    wave = np.asarray([np.nan, 12., 10., 5.])
    flux = np.arange(20., dtype=float).reshape(5, 4)
    time = 61000. + np.arange(5) / 86400.
    _write_single_trace_spectrum(reference_path, wave, flux, time)
    candidate_wave = np.asarray([13., 12., 10., 5.])
    _write_single_trace_spectrum(candidate_path, candidate_wave, flux, time)

    report = validate.compare_single_trace_spectra(
        reference_path, candidate_path)

    assert not report['passed']
    support = report['checks']['wavelength_finite_support']
    assert support['mismatch_count'] == 1
    assert support['mismatch_fraction'] == pytest.approx(.25)


def test_single_trace_spectrum_cli_writes_gated_report(tmp_path):
    """Check single trace spectrum cli writes gated report."""
    reference_path = tmp_path / 'v1_spectra.fits'
    candidate_path = tmp_path / 'v2_spectra.fits'
    report_path = tmp_path / 'single_trace_report.json'
    wave = np.asarray([np.nan, 3., 3.5, 4.])
    curve = np.asarray([1., 1.01, .99, 1.02, 1., .98, 1.01, 1.])
    flux = curve[:, None] * np.asarray([0., 10., 20., 30.])[None]
    time = 60000. + np.arange(curve.size) / 86400.
    _write_single_trace_spectrum(reference_path, wave, flux, time)
    _write_single_trace_spectrum(candidate_path, wave, flux, time)

    assert validate.main([
        'spectra', '--reference', str(reference_path),
        '--candidate', str(candidate_path), '--baseline-ints', '3,-3',
        '--wave-range', '3,3.5',
        '--max-flux-p95-ppm', '1',
        '--max-scatter-ratio-deviation', '.01',
        '--report', str(report_path)]) == 0

    report = json.loads(report_path.read_text())
    assert report['passed']
    assert report['parity_claimed']
    assert report['baseline_ints'] == [3, -3]
    assert report['wave_range_micron'] == [3., 3.5]
    assert report['checks']['normalized_flux_p95']['observed_p95_ppm'] == 0
    assert report['checks']['median_scatter_ratio'][
        'observed_median_ratio'] == pytest.approx(1.)


def test_soss_spectrum_comparison_aligns_finite_wavelength_padding(tmp_path):
    """Check SOSS spectrum comparison aligns finite wavelength padding."""
    reference_path = tmp_path / 'v1_soss_spectra.fits'
    candidate_path = tmp_path / 'v2_soss_spectra.fits'
    time = 60000. + np.arange(8) / 86400.
    curve = np.asarray([1., 1.01, .99, 1.02, 1., .98, 1.01, 1.])
    ref_orders = {
        1: (
            [np.nan, 1., 1.5, 2.],
            curve[:, None] * np.asarray([0., 10., 20., 30.])[None]),
        2: (
            [np.nan, np.nan, .6, .7, .8],
            curve[:, None] * np.asarray([0., 0., 40., 50., 60.])[None]),
    }
    got_orders = {
        1: (
            [1., 1.5, 2.],
            curve[:, None] * np.asarray([10., 20., 30.])[None]),
        2: (
            [.6, .7, .8],
            curve[:, None] * np.asarray([40., 50., 60.])[None]),
    }
    _write_soss_spectrum(reference_path, ref_orders, time)
    _write_soss_spectrum(candidate_path, got_orders, time)

    report = validate.compare_soss_spectra(
        reference_path, candidate_path, baseline_ints=[3, -3],
        o1_wave_range=[1., 2.], o2_wave_range=[.6, .8],
        o1_max_flux_p95_ppm=1., o2_max_flux_p95_ppm=1.,
        o1_max_scatter_ratio_deviation=.01,
        o2_max_scatter_ratio_deviation=.01)

    assert report['passed']
    assert report['parity_claimed']
    assert report['checks']['integration_times']['bitwise_equal']
    assert report['orders']['O1']['checks'][
        'finite_wavelength_alignment']['reference_dropped_padding_columns'] == 1
    assert report['orders']['O2']['checks'][
        'finite_wavelength_alignment']['reference_dropped_padding_columns'] == 2
    assert report['orders']['O2']['metrics'][
        'common_science_column_count'] == 3
    assert report['orders']['O1']['checks'][
        'normalized_flux_p95']['observed_p95_ppm'] == 0
    assert report['orders']['O2']['checks'][
        'normalized_flux_p95']['observed_p95_ppm'] == 0


def test_soss_spectrum_cli_requires_both_orders_to_pass(tmp_path):
    """Check SOSS spectrum cli requires both orders to pass."""
    reference_path = tmp_path / 'v1_soss_spectra.fits'
    candidate_path = tmp_path / 'v2_soss_spectra.fits'
    report_path = tmp_path / 'soss_report.json'
    time = 60000. + np.arange(8) / 86400.
    curve = np.asarray([1., 1.01, .99, 1.02, 1., .98, 1.01, 1.])
    reference_orders = {
        1: ([1., 1.5], curve[:, None] * np.asarray([10., 20.])[None]),
        2: ([np.nan, .6, .8],
            curve[:, None] * np.asarray([0., 30., 40.])[None]),
    }
    candidate_orders = {
        1: ([1., 1.5], curve[:, None] * np.asarray([10., 20.])[None]),
        2: ([.6, .8], curve[:, None] * np.asarray([30., 40.])[None]),
    }
    candidate_orders[2][1][3, 0] *= 1.01
    _write_soss_spectrum(reference_path, reference_orders, time)
    _write_soss_spectrum(candidate_path, candidate_orders, time)

    assert validate.main([
        'soss-spectra', '--reference', str(reference_path),
        '--candidate', str(candidate_path), '--baseline-ints', '3,-3',
        '--o1-wave-range', '1,1.5', '--o2-wave-range', '.6,.8',
        '--o1-max-flux-p95-ppm', '1',
        '--o2-max-flux-p95-ppm', '1',
        '--o1-max-scatter-ratio-deviation', '.01',
        '--o2-max-scatter-ratio-deviation', '10',
        '--report', str(report_path)]) == 1

    report = json.loads(report_path.read_text())
    assert not report['passed']
    assert not report['parity_claimed']
    assert report['orders']['O1']['passed']
    assert not report['orders']['O2']['passed']
    assert report['orders']['O1']['parity_claimed']
    assert not report['orders']['O2']['parity_claimed']
    assert not report['orders']['O2']['checks'][
        'normalized_flux_p95']['passed']


def test_soss_noise_relative_gate_accepts_residual_below_reference_noise(
        tmp_path):
    """Check SOSS noise relative gate accepts residual below reference noise."""
    reference_path = tmp_path / 'v1_noisy_soss.fits'
    candidate_path = tmp_path / 'v2_noisy_soss.fits'
    report_path = tmp_path / 'relative_soss_report.json'
    time = 60000. + np.arange(8) / 86400.
    noisy_curve = np.asarray([1., 1.4, .6, 1.3, .7, 1.2, .8, 1.])
    residual = np.asarray([0., .002, -.002, .002, -.002, .002, -.002, 0.])
    candidate_curve = noisy_curve + residual
    reference_orders = {
        1: ([1., 1.5], noisy_curve[:, None] * np.asarray([10., 20.])[None]),
        2: ([.6, .8], noisy_curve[:, None] * np.asarray([30., 40.])[None]),
    }
    candidate_orders = {
        1: ([1., 1.5],
            candidate_curve[:, None] * np.asarray([10., 20.])[None]),
        2: ([.6, .8],
            candidate_curve[:, None] * np.asarray([30., 40.])[None]),
    }
    _write_soss_spectrum(reference_path, reference_orders, time)
    _write_soss_spectrum(candidate_path, candidate_orders, time)

    assert validate.main([
        'soss-spectra', '--reference', str(reference_path),
        '--candidate', str(candidate_path), '--baseline-ints', '3,-3',
        '--o1-wave-range', '1,1.5', '--o2-wave-range', '.6,.8',
        '--o1-max-flux-noise-fraction', '.5',
        '--o2-max-flux-noise-fraction', '.5',
        '--o1-max-scatter-ratio-deviation', '.01',
        '--o2-max-scatter-ratio-deviation', '.01',
        '--report', str(report_path)]) == 0

    relative = json.loads(report_path.read_text())
    assert relative['passed']
    assert relative['parity_claimed']
    for order in ('O1', 'O2'):
        assert relative['orders'][order]['metrics'][
            'normalized_flux_residual_ppm']['absolute']['p95'] > 1000
        noise_gate = relative['orders'][order]['checks'][
            'flux_noise_fraction_p95']
        assert noise_gate['observed_p95_fraction'] < .5
        assert noise_gate['passed']
        assert relative['orders'][order]['checks'][
            'normalized_flux_p95']['maximum_p95_ppm'] is None

    both = validate.compare_soss_spectra(
        reference_path, candidate_path, baseline_ints=[3, -3],
        o1_wave_range=[1., 1.5], o2_wave_range=[.6, .8],
        o1_max_flux_p95_ppm=1000., o2_max_flux_p95_ppm=1000.,
        o1_max_flux_noise_fraction=.5, o2_max_flux_noise_fraction=.5,
        o1_max_scatter_ratio_deviation=.01,
        o2_max_scatter_ratio_deviation=.01)
    assert not both['passed']
    assert not both['parity_claimed']
    for order in ('O1', 'O2'):
        assert not both['orders'][order]['checks'][
            'normalized_flux_p95']['passed']
        assert both['orders'][order]['checks'][
            'flux_noise_fraction_p95']['passed']

    diagnostic_only = validate.compare_soss_spectra(
        reference_path, candidate_path, baseline_ints=[3, -3])
    assert diagnostic_only['passed']
    assert not diagnostic_only['parity_claimed']


@pytest.mark.parametrize('defect', ['wavelength', 'time', 'nan_flux', 'empty_band'])
def test_soss_comparison_rejects_invalid_science_support(tmp_path, defect):
    """Check SOSS comparison rejects invalid science support."""
    reference_path = tmp_path / 'reference.fits'
    candidate_path = tmp_path / 'candidate.fits'
    time = 60000. + np.arange(8) / 86400.
    curve = np.asarray([1., 1.01, .99, 1.02, 1., .98, 1.01, 1.])
    orders = {
        1: (np.asarray([1., 1.5]), curve[:, None] * [[10., 20.]]),
        2: (np.asarray([.6, .8]), curve[:, None] * [[30., 40.]]),
    }
    _write_soss_spectrum(reference_path, orders, time)
    if defect == 'wavelength':
        orders[2][0][0] += .001
    elif defect == 'time':
        time[0] += 1. / 86400.
    elif defect == 'nan_flux':
        orders[2][1][3, 0] = np.nan
    _write_soss_spectrum(candidate_path, orders, time)
    report = validate.compare_soss_spectra(
        reference_path, candidate_path, baseline_ints=[3, -3],
        o2_wave_range=[3., 4.] if defect == 'empty_band' else None,
        o1_max_flux_p95_ppm=1., o2_max_flux_p95_ppm=1.,
        o1_max_scatter_ratio_deviation=.01,
        o2_max_scatter_ratio_deviation=.01)
    assert not report['passed']
    assert not report['parity_claimed']


def test_outputs_dir_location_intentionally_has_no_config(tmp_path):
    """Check outputs dir location intentionally has no config."""
    root = _make_v1_output_tree(tmp_path)
    locations = validate.resolve_v1_output_locations(
        outputs_directory=root, log_name='demo')
    assert locations.config is None


def test_pack_v1_rejects_ambiguous_stage3_spectra(tmp_path):
    """Check pack v1 rejects ambiguous stage3 spectra."""
    root = _make_v1_output_tree(tmp_path)
    _write_v1_soss_spectrum(root / 'Stage3' / 'a_spectra_fullres.fits')
    _write_v1_soss_spectrum(root / 'Stage3' / 'b_spectra_fullres.fits')
    with pytest.raises(ValueError, match='multiple v1 Stage3 spectra'):
        validate.pack_v1_outputs(
            root, tmp_path / 'ambiguous_spectra.npz', log_name='demo')


def test_pack_v1_rejects_segment_specific_pixeldq(tmp_path):
    """Check pack v1 rejects segment specific PIXELDQ."""
    root = _make_v1_output_tree(tmp_path, mismatched_pixeldq=True)
    with pytest.raises(ValueError, match='PIXELDQ differs between segments'):
        validate.pack_v1_outputs(
            root, tmp_path / 'unsafe.npz', include_logs=False)


def test_pack_v1_rejects_missing_segment_artifact(tmp_path):
    """Check pack v1 rejects missing segment artifact."""
    root = _make_v1_output_tree(tmp_path)
    (root / 'Stage2' /
     'visit_seg001_nis_pcareconstructstep.fits').unlink()
    with pytest.raises(ValueError, match='one or more v1 artifacts may be missing'):
        validate.pack_v1_outputs(
            root, tmp_path / 'incomplete.npz', include_logs=False)


def test_pack_v1_rejects_ambiguous_datasets(tmp_path):
    """Check pack v1 rejects ambiguous datasets."""
    root = _make_v1_output_tree(tmp_path)
    data = np.ones((1, 2, 2, 3), np.float32)
    _write_v1_artifact(
        root / 'Stage1' / 'other_seg001_nis_dqinitstep.fits', data,
        int_start=1, exsegnum=1, groupdq=np.zeros_like(data, np.uint32),
        pixeldq=np.zeros((2, 3), np.uint32))
    with pytest.raises(ValueError, match='ambiguous DQInitStep artifacts'):
        validate.pack_v1_outputs(
            root, tmp_path / 'ambiguous.npz', include_logs=False)


def test_v1_segment_order_falls_back_to_natural_filenames(tmp_path):
    """Check v1 segment order falls back to natural filenames."""
    segment10 = tmp_path / 'visit_seg10_nis_dqinitstep.fits'
    segment2 = tmp_path / 'visit_seg2_nis_dqinitstep.fits'
    _write_v1_artifact(segment10, np.ones((1, 2, 2, 3)))
    _write_v1_artifact(segment2, np.ones((1, 2, 2, 3)))
    ordered, _ = validate._ordered_v1_segments([segment10, segment2])
    assert ordered == [segment2, segment10]


def test_pack_v1_cli_resolves_configured_output_directory(monkeypatch,
                                                          tmp_path):
    """Check pack v1 cli resolves configured output directory."""
    _make_v1_output_tree(tmp_path)
    config = tmp_path / 'run_optimize.yaml'
    config.write_text(
        'pipeline_outputs_directory: pipeline_outputs_directory\n'
        'name_tag: demo\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    assert validate.main([
        'pack-v1', '--config', str(config), '--output', 'oracle']) == 0
    assert (tmp_path / 'oracle.npz').is_file()


def test_config_location_uses_exact_v1_trailing_slash_tag_root(tmp_path):
    """Check config location uses exact v1 trailing slash tag root."""
    base = _make_v1_output_tree(tmp_path)
    tagged = _make_v1_output_tree(base, root_name='_visit')
    config = tmp_path / 'tagged.yaml'
    config.write_text(
        f"pipeline_outputs_directory: '{base}/'\n"
        'output_tag: visit\nname_tag: demo\n', encoding='utf-8')

    locations = validate.resolve_v1_output_locations(config_path=config)
    assert locations.stage_root == tagged.resolve()
    assert locations.log_directory == (base / 'Files').resolve()

    config.write_text(
        f"pipeline_outputs_directory: '{base}/'\n"
        'output_tag: missing\nname_tag: demo\n', encoding='utf-8')
    with pytest.raises(FileNotFoundError, match='_missing'):
        validate.resolve_v1_output_locations(config_path=config)


def test_capture_and_save_nested_spectral_products(tmp_path):
    """Check capture and save nested spectral products."""
    @dataclass
    class Cube:
        data: np.ndarray

    def extract(state, params, _ctx):
        flux = state.cube.data[:, :2]
        return PipelineState(
            state.cube,
            {'spectral_products': {
                1: {'wave': np.array([1., 2.]), 'flux': flux,
                    'ferr': np.ones_like(flux)},
            }})

    pipeline = Pipeline([Step('Extract', extract)], mode='SOSS', ctx={})
    captured = validate.capture_pipeline_steps(
        pipeline, PipelineState(Cube(np.arange(6.).reshape(2, 3))), {},
        steps='Extract', fields={'Extract': 'aux.spectral_products'})
    path = validate.save_checkpoint_npz(captured, tmp_path / 'captured.npz')
    loaded = validate.load_checkpoint_npz(path)

    assert set(loaded) == {'Extract'}
    np.testing.assert_array_equal(
        loaded['Extract']['aux.spectral_products.1.wave'], [1., 2.])
    np.testing.assert_array_equal(
        loaded['Extract']['aux.spectral_products.1.flux'], [[0., 1.], [3., 4.]])


def test_synthetic_and_real_deferred_cli_reports(tmp_path):
    """Check synthetic and real deferred cli reports."""
    synthetic_path = tmp_path / 'synthetic.json'
    assert validate.main(['synthetic', '--report', str(synthetic_path)]) == 0
    synthetic = json.loads(synthetic_path.read_text())
    assert synthetic['passed']
    assert synthetic['parity_scope'] == 'synthetic_fixture_only'
    assert 'not a real-data parity claim' in synthetic['note']

    real_path = tmp_path / 'real.json'
    assert validate.main([
        'real', '--config', 'run_optimize.yaml', '--dataset', 'gpu-only',
        '--report', str(real_path)]) == 0
    real = json.loads(real_path.read_text())
    assert real['status'] == 'deferred'
    assert real['parity_claimed'] is False
    assert real['summary']['steps_compared'] == 0

    unavailable_path = tmp_path / 'unavailable.json'
    assert validate.main([
        'real', '--reference', str(tmp_path / 'missing-v1.npz'),
        '--candidate', str(tmp_path / 'missing-v2.npz'),
        '--report', str(unavailable_path)]) == 2
    unavailable = json.loads(unavailable_path.read_text())
    assert unavailable['status'] == 'unavailable'
    assert unavailable['passed'] is False

    one_sided_path = tmp_path / 'one-sided.json'
    existing = tmp_path / 'existing.npz'
    np.savez(existing, Cost=np.asarray(1.0))
    assert validate.main([
        'real', '--reference', str(existing),
        '--report', str(one_sided_path)]) == 2
    one_sided = json.loads(one_sided_path.read_text())
    assert one_sided['status'] == 'unavailable'
    assert 'Both --reference and --candidate' in one_sided['reason']


def test_compare_precision_and_real_cli_honor_custom_tolerances(tmp_path):
    """Check compare precision and real cli honor custom tolerances."""
    reference = validate.save_checkpoint_npz(
        {'Cost': {'value': np.asarray([1.0, np.nan])}},
        tmp_path / 'reference.npz')
    candidate = validate.save_checkpoint_npz(
        {'Cost': {'value': np.asarray([1.1, 5.0])}},
        tmp_path / 'candidate.npz')
    assert not validate._compare_files(
        str(reference), str(candidate), kind='default', scope='test')['passed']

    commands = (
        ('compare', ['--reference', str(reference),
                     '--candidate', str(candidate)]),
        ('precision', ['--float32', str(candidate),
                       '--float64', str(reference)]),
        ('real', ['--reference', str(reference),
                  '--candidate', str(candidate)]),
    )
    for command, paths in commands:
        report_path = tmp_path / f'{command}.json'
        assert validate.main([
            command, *paths, '--rtol', '0', '--atol', '0.2',
            '--max-finite-mismatch-fraction', '0.5',
            '--report', str(report_path)]) == 0
        report = json.loads(report_path.read_text())
        assert report['passed']
        metrics = report['steps']['Cost']['arrays']['value']
        assert metrics['rtol'] == 0.
        assert metrics['atol'] == .2
        assert metrics['max_finite_mismatch_fraction'] == .5


def test_all_tolerance_clis_reject_invalid_thresholds():
    """Check all tolerance clis reject invalid thresholds."""
    parser = validate.build_parser()
    commands = (
        ['compare', '--reference', 'reference.npz',
         '--candidate', 'candidate.npz'],
        ['precision', '--float32', 'float32.npz',
         '--float64', 'float64.npz'],
        ['real'],
    )
    invalid = (
        ('--rtol', '-1'), ('--rtol', 'nan'), ('--rtol', 'inf'),
        ('--atol', '-1'), ('--atol', 'nan'), ('--atol', 'inf'),
        ('--max-finite-mismatch-fraction', '-0.1'),
        ('--max-finite-mismatch-fraction', 'nan'),
        ('--max-finite-mismatch-fraction', 'inf'),
        ('--max-finite-mismatch-fraction', '1.0001'),
    )
    for command in commands:
        for flag, value in invalid:
            with pytest.raises(SystemExit):
                parser.parse_args([*command, flag, value])


def test_x64_cli_flag_is_applied_before_validation(monkeypatch, tmp_path):
    """Check x64 cli flag is applied before validation."""
    calls = []
    monkeypatch.setattr(core, 'enable_x64', lambda: calls.append('enabled'))
    assert validate.main([
        'synthetic', '--x64', '--report', str(tmp_path / 'x64.json')]) == 0
    assert calls == ['enabled']


def test_precision_report_has_drift_metrics_and_winner_match():
    """Check precision report has drift metrics and winner match."""
    float64 = {
        'RampFit': {'slope': np.array([1.0, 2.0], dtype=np.float64),
                    'error': np.array([0.1, 0.2], dtype=np.float64)},
        'Cost': {'value': np.array(0.125, dtype=np.float64)},
    }
    float32 = {
        step: {name: value.astype(np.float32) for name, value in fields.items()}
        for step, fields in float64.items()
    }
    winner = {'time_sigma': 7, 'extract_width': 24}
    report = validate.compare_precision_runs(
        float32, float64, float32_winner=winner, float64_winner=dict(winner))

    assert report['passed']
    assert report['parity_claimed'] is False
    assert report['precision_agreement_claimed']
    assert report['precision']['winner_match'] is True
    slope = report['steps']['RampFit']['arrays']['slope']
    assert slope['candidate_dtype'] == 'float32'
    assert slope['reference_dtype'] == 'float64'
    assert slope['max_absolute_error'] is not None
    assert slope['max_relative_error'] is not None


def test_precision_winner_mismatch_fails_even_when_arrays_match():
    """Check precision winner mismatch fails even when arrays match."""
    arrays = {'Cost': {'value': np.array(1.0)}}
    report = validate.compare_precision_runs(
        arrays, arrays, float32_winner={'width': 20},
        float64_winner={'width': 22})
    assert not report['passed']
    assert report['precision']['winner_match'] is False


def test_capture_pipeline_steps_names_outputs():
    """Check capture pipeline steps names outputs."""
    @dataclass
    class Cube:
        data: np.ndarray

    def add(state, params, _ctx):
        return PipelineState(Cube(state.cube.data + params['amount']))

    pipeline = Pipeline([
        Step('first', add), Step('second', add)], mode='SOSS', ctx={})
    captured = validate.capture_pipeline_steps(
        pipeline, PipelineState(Cube(np.array([1.0, 2.0]))), {'amount': 3.0})
    assert list(captured) == ['input', 'first', 'second']
    np.testing.assert_allclose(captured['second']['data'], [7.0, 8.0])


def test_capture_pipeline_steps_selects_steps_and_fields_without_materializing():
    """Check capture pipeline steps selects steps and fields without materializing."""
    class Cube:
        def __init__(self, data):
            self.data = data

        @property
        def err(self):
            raise AssertionError('unselected ERR must not be materialized')

    def add(state, params, _ctx):
        return PipelineState(Cube(state.cube.data + params['amount']))

    pipeline = Pipeline([
        Step('first', add), Step('second', add)], mode='SOSS', ctx={})
    captured = validate.capture_pipeline_steps(
        pipeline, PipelineState(Cube(np.array([1.0, 2.0]))), {'amount': 3.0},
        steps='second', fields='data')

    assert list(captured) == ['second']
    np.testing.assert_allclose(captured['second']['data'], [7.0, 8.0])
    with pytest.raises(ValueError, match='unknown checkpoint step'):
        validate.capture_pipeline_steps(
            pipeline, PipelineState(Cube(np.ones(2))), {'amount': 1.0},
            steps='not-a-step')
