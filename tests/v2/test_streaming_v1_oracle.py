"""Parity against the actual local v1 source, loaded without its I/O stack."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import bottleneck as bn
import numpy as np
import pytest
from scipy.ndimage import median_filter

from exotedrf.v2 import core, stages
from exotedrf.v2.pipeline import PipelineState

REPO = Path(__file__).resolve().parents[2]


def load_functions(relative, names, namespace):
    """Load selected functions from the read-only v1 source.

    Parameters
    ----------
    relative : str
        Source path relative to the repository root.
    names : list[str]
        Names of functions or arrays to select.
    namespace : dict
        Globals for the loaded functions.

    Returns
    -------
    namespace : dict
        Loaded v1 functions and their globals.
    """
    path = REPO / relative
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class Model(SimpleNamespace):
    """Represent a minimal v1 data model."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.mark.parametrize('chunk', [1, 11, 48])
def test_segmented_badpix_matches_actual_v1_source(monkeypatch, chunk):
    """Check segmented bad-pixel correction matches actual v1 source."""
    namespace = load_functions('exotedrf/utils.py',
        ['get_interp_box', 'outlier_resistant_variance', 'do_replacement',
         'get_dq_flag_metrics'], {'np': np, 'bn': bn})
    helpers = SimpleNamespace(**{k: v for k, v in namespace.items()
                                 if callable(v)})
    helpers.open_filetype = lambda value: value
    helpers.get_instrument_name = lambda value: 'NIRISS'
    oracle = load_functions('exotedrf/stage2.py', ['badpixstep'],
        {'np': np, 'bn': bn, 'copy': copy, 'utils': helpers,
         'median_filter': median_filter, 'tqdm': lambda x: x,
         'fancyprint': lambda *a, **kw: None})['badpixstep']
    rng = np.random.default_rng(192)
    data = rng.normal(100., 2., (33, 12, 48)).astype(np.float32)
    data[:, 2, 17] = 1000.
    data[7, 3, 21] = 1000.
    data[22, 4, 29] = -1.
    err = np.full_like(data, 2.)
    err[2, 4, 21] = np.nan
    dq = np.zeros(data.shape, np.uint32)
    dq[:, 2, 17] = core.DQ_HOT
    dq[4, 3, 21] = core.DQ_SATURATED
    dq[:, 5, 27] = core.DQ_DO_NOT_USE
    meta = core.ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP96', 1., 3,
                        np.arange(33.), np.array([5, -5]),
                        np.array([17, 33]), ('first', 'last'))
    cube = core.RateCube(data, err, dq, meta)
    baseline = stages.baseline_bool_for_meta(meta, 33)
    deep = np.nanmedian(data[baseline], axis=0)
    expected = []
    badpix = None
    for seg in core.segment_slices(meta, 33):
        model = Model(data=data[seg].copy(), err=err[seg].copy(),
                      dq=dq[seg].copy(), meta=SimpleNamespace(filename='fixture'))
        corrected, badpix = oracle(
            model, deep.copy(), space_thresh=5, time_thresh=5,
            box_size=7, window_size=3, to_flag=badpix, save_results=False)
        expected.append(corrected)
    monkeypatch.setenv('EXOTEDRF_V2_BADPIX_COLUMN_CHUNK', str(chunk))
    actual = stages.step_badpix(PipelineState(cube),
        dict(space_outlier_threshold=5, time_outlier_threshold=5,
             box_size=7, window_size=3), {'opts': {}})
    for key in ('data', 'err', 'dq'):
        reference = np.concatenate([getattr(x, key) for x in expected])
        if key == 'dq':
            # V2 stores v1 2.5.0's bit 32 in a separate variance plane.
            reference = reference.astype(np.uint32)
        np.testing.assert_array_equal(getattr(actual.cube, key), reference)
    variance = [np.any((x.dq & np.uint64(1 << 32)) != 0, axis=0)
                for x in expected]
    for actual_map, reference in zip(actual.aux['high_variance_maps'], variance):
        np.testing.assert_array_equal(actual_map, reference)
    np.testing.assert_array_equal(actual.aux['high_variance_map'],
                                  np.logical_or.reduce(variance))
    np.testing.assert_array_equal(actual.aux['hot_pixel_map'], badpix)


def test_production_cost_and_winner_match_actual_v1_optimizer():
    """Check production cost and winner match actual v1 optimizer."""
    from exotedrf.v2.optimize import select_first_finite_minimum
    oracle = load_functions('exotedrf/optimize.py', ['cost_function'],
        {'np': np, 'obs_early': 'niriss',
         'fancyprint': lambda *a, **kw: None})['cost_function']
    rng = np.random.default_rng(741)
    wave1, wave2 = np.linspace(.8, 2.8, 25), np.linspace(.6, 1.4, 20)
    noise1, noise2 = rng.normal(size=(40, 25)), rng.normal(size=(40, 20))
    meta = core.ObsMeta('NIRISS/SOSS', 'NIS', 'SUBSTRIP96', 1., 3,
                        np.arange(40.), np.array([10, -10]),
                        np.array([20, 40]), ('first', 'last'))
    cube = core.RateCube(np.zeros((40, 1, 1)), np.ones((40, 1, 1)),
                         np.zeros((40, 1, 1), np.uint32), meta)
    expected_costs, actual_costs = [], []
    for amplitude in (.04, .01, .03):
        f1 = (1000 * (1 + amplitude * noise1)).astype(np.float32)
        f2 = (500 * (1 + amplitude * noise2)).astype(np.float32)
        products = {1: {'wave': wave1, 'flux': f1},
                    2: {'wave': wave2, 'flux': f2}}
        actual, scatter = stages.evaluate_production_cost(
            PipelineState(cube, {'spectral_products': products}), {},
            {'opts': {'w1': .3, 'w2': .7}})
        expected, ref_scatter = oracle(
            {'Flux O1': f1, 'Flux O2': f2, 'Wave O1': wave1, 'Wave O2': wave2},
            baseline_ints=[10, -10], w1=.3, w2=.7)
        np.testing.assert_allclose(scatter, ref_scatter, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
        expected_costs.append(expected)
        actual_costs.append(actual)
    assert select_first_finite_minimum(actual_costs)[0] == np.argmin(expected_costs)
