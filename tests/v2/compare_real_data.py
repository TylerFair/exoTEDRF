"""Compare reference-pixel correction and ATOCA on real integrations."""

import argparse
import ast
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits


def compare_refpix(source, output):
    """Compare the reference-pixel kernel with the installed JWST step.

    Parameters
    ----------
    source : pathlib.Path
        Superbias-corrected ramp subset.
    output : pathlib.Path
        Directory for comparison arrays and statistics.
    """
    from jwst.refpix import RefPixStep
    from stdatamodels.jwst import datamodels
    from exotedrf.v2.kernels import detector

    with datamodels.RampModel(source) as model:
        data = np.array(model.data, dtype=np.float32)
        flags = np.array(model.pixeldq, dtype=np.uint32)
        result = RefPixStep.call(model, save_results=False)
        expected = np.array(result.data)
        result.close()
    actual = np.asarray(detector.refpix_correct(
        data, flags, nref_side=4, detector_column_axis=-2))
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    stats = {'nints': len(data), 'max_abs': float(np.nanmax(
        np.abs(actual - expected))), 'mean_abs': float(np.nanmean(
        np.abs(actual - expected)))}
    np.savez(output / 'refpix_arrays.npz', expected=expected[:, :, -4:],
             actual=actual[:, :, -4:])
    (output / 'refpix.json').write_text(json.dumps(stats, indent=2) + '\n')
    print('REFPIX', stats, flush=True)


def compare_atoca(source, output):
    """Compare the Stage-3 oracle and in-memory ATOCA on one segment.

    Parameters
    ----------
    source : pathlib.Path
        Bad-pixel-corrected rate subset.
    output : pathlib.Path
        Directory for reference files, spectra and statistics.
    """
    from jwst.extract_1d import Extract1dStep
    from stdatamodels.jwst import datamodels
    from exotedrf.v2 import atoca, core, refs
    from exotedrf.v2.pipeline import PipelineState

    with fits.open(source) as hdul:
        header = hdul[0].header.copy()
        data = np.array(hdul['SCI'].data, dtype=np.float32)
        err = np.array(hdul['ERR'].data, dtype=np.float32)
        dq = np.array(hdul['DQ'].data, dtype=np.uint32)
        times = np.array(hdul['INT_TIMES'].data['int_mid_BJD_TDB'])
    search = list(refs.default_files_dirs())
    if os.environ.get('CRDS_PATH'):
        search.append(os.path.join(os.environ['CRDS_PATH'],
                                   'references', 'jwst', 'niriss'))
    trace = refs.soss_reference_file('spectrace', header['SUBARRAY'],
                                    search_dirs=search, download_dir=output)
    wave = refs.soss_reference_file('wavemap', header['SUBARRAY'],
                                   search_dirs=search, download_dir=output)
    profile = output / atoca.v1_specprofile_name(header['SUBARRAY'])
    if not profile.exists():
        atoca.build_specprofile(atoca.deepstack_nanmedian(data), trace,
                                wave, output)

    # Load the unmodified Stage-3 oracle without unrelated imports.
    path = Path(__file__).resolve().parents[2] / 'exotedrf' / 'stage3.py'
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef)
                and node.name == 'atoca_extract_soss')
    namespace = {'calwebb_spec2': SimpleNamespace(
        extract_1d_step=SimpleNamespace(Extract1dStep=Extract1dStep))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path),
                 'exec'), namespace)
    started = time.perf_counter()
    with datamodels.CubeModel(source) as model:
        result = namespace['atoca_extract_soss'](
            [model], str(profile), ['parity_'],
            output_dir=str(output) + os.sep, save_results=False,
            extract_width=40)[0]
        expected = atoca.unpack_atoca_spectra(result)
        result.close()
    v1_seconds = time.perf_counter() - started
    meta = core.ObsMeta(
        'NIRISS/SOSS', 'NIS', header['SUBARRAY'], header['TFRAME'],
        header['NGROUPS'], times, np.array([2, -2]),
        np.array([len(data)]), (str(source),),
        extra={'header': dict(header), 'segment_headers': (dict(header),),
               'segment_int_starts': (header['INTSTART'],)})
    state = PipelineState(core.RateCube(data, err, dq, meta))
    ctx = {'opts': {'soss_specprofile': str(profile),
                    'v2_atoca_workers': 1, 'v2_atoca_solver_threads': 4},
           'centroid_tracetable': trace,
           'atoca_output_dir': str(output / 'v2')}
    captured = []
    formatter = atoca.format_atoca_products

    def record_orders(segments, sigma_clip):
        captured.extend(segments)
        return formatter(segments, sigma_clip)

    started = time.perf_counter()
    atoca.format_atoca_products = record_orders
    try:
        _, _, info = atoca.extract_state(state, {'extract_width': 40}, ctx)
    finally:
        atoca.format_atoca_products = formatter
    stats = {'nints': len(data), 'v1_seconds': v1_seconds,
             'v2_seconds': time.perf_counter() - started, 'orders': {}}
    arrays = {}
    for order in (1, 2):
        stats['orders'][order] = {}
        for quantity, key in [('WAVELENGTH', 'wave'), ('FLUX', 'flux'),
                              ('FLUX_ERROR', 'ferr')]:
            reference = expected[order][quantity]
            actual = captured[0][order][quantity]
            np.testing.assert_array_equal(np.isfinite(actual),
                                          np.isfinite(reference))
            diff = np.abs(actual - reference)
            stats['orders'][order][key] = {
                'max_abs': float(np.nanmax(diff)),
                'max_rel': float(np.nanmax(diff / np.maximum(
                    np.abs(reference), np.finfo(float).tiny)))}
            arrays[f'v1_{key}{order}'] = reference
            arrays[f'v2_{key}{order}'] = actual
    stats['tikhonov_factors'] = info['tikhonov_factors']
    np.savez(output / 'atoca_arrays.npz', **arrays)
    (output / 'atoca.json').write_text(json.dumps(stats, indent=2) + '\n')
    print('ATOCA', stats, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('refpix', 'atoca'))
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'refpix':
        compare_refpix(args.source, args.output)
    else:
        compare_atoca(args.source, args.output)
