"""MIRI Horne extraction against the actual local, unmodified v1 source."""

import ast
from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.ndimage import median_filter

from exotedrf.v2 import core, stages
from exotedrf.v2.kernels import extract
from exotedrf.v2.pipeline import PipelineState


def _v1_horne():
    """Return v1 horne."""
    path = Path(__file__).resolve().parents[2] / 'exotedrf/stage3.py'
    names = {'get_spatial_prof_opt', 'extract_optimal', 'do_optimal_extraction'}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == 3
    for node in nodes:
        if node.name == 'do_optimal_extraction':
            node.body[-1].value.elts.append(
                ast.Name(id='num_clipped', ctx=ast.Load()))
            # V1 2.5.0 leaves ``dq`` unbound when max_iter == 0; bind it.
            node.body.insert(0, ast.parse('dq = None').body[0])
    namespace = {'np': np, 'median_filter': median_filter,
                 'fancyprint': lambda *args, **kwargs: None,
                 'tqdm': lambda values: values}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    v1_full = namespace['do_optimal_extraction']

    def v1_three(*args, **kwargs):
        # 2.5.0 returns (flux, error, dq_report[, counts injected above]).
        flux, error, _, counts = v1_full(*args, **kwargs)
        return flux, error, counts
    return v1_three


def _dataset(dtype=np.float32):
    """Return dataset."""
    rng = np.random.default_rng(0)
    nints, naperture, nwave = 61, 12, 20
    spatial = np.exp(-.5 * ((np.arange(naperture) - 6.) / 1.2) ** 2) + .02
    deep = (300. + np.arange(nwave)[None, :] * 8) * spatial[:, None]
    source = (deep[None] * (1 + rng.normal(0, .002, (nints, 1, 1))) +
              rng.normal(0, 2, (nints, naperture, nwave))).astype(dtype)
    source[5, 1, 10] += 500.
    source[14, 6, 9] += 200.
    return source.transpose(0, 2, 1), deep.T.astype(dtype)


@pytest.mark.parametrize('chunk', [None, 1, 7])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_miri_horne_preserves_full_frame_rejection_and_global_v1_stop(
        monkeypatch, chunk, dtype):
    """Check MIRI horne preserves full frame rejection and global v1 stop."""
    data, deep = _dataset(dtype)
    original = data.copy()
    nints, nwave, naperture = data.shape
    expected_flux, expected_error, expected_counts = _v1_horne()(
        data.transpose(0, 2, 1).copy(), deep.T.copy(),
        ymin=np.full(nwave, 4), ymax=np.full(nwave, 9),
        xmin=0, xmax=nwave, max_iter=10, var_thresh=25)
    assert expected_counts == [1, 1]
    if chunk is not None:
        monkeypatch.setenv('EXOTEDRF_HORNE_CHUNK_WAVES', str(chunk))
    meta = core.ObsMeta('MIRI/LRS', 'MIRIMAGE', 'SLITLESSPRISM', 1., 1,
        np.arange(nints, dtype=float), np.array([10, -10]),
        np.array([nints]), ())
    cube = core.RateCube(data, np.ones_like(data),
                         np.zeros(data.shape, np.uint32), meta)
    state = PipelineState(cube, {'stage3_deepframe': deep})
    centroids = {'xpos': np.full(nwave, 6.5), 'ypos': np.arange(nwave)}
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', dtype == np.float64)
    try:
        flux, error, counts = stages._extract_miri_optimal(
            state, {'extract_width': 5.},
            {'opts': {'opt_max_iter': 10, 'opt_var_thresh': 25}}, centroids)
    finally:
        jax.config.update('jax_enable_x64', previous)
    assert counts == tuple(expected_counts)
    tolerance = 2e-5 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(flux, expected_flux, rtol=tolerance)
    np.testing.assert_allclose(error, expected_error, rtol=tolerance)
    np.testing.assert_array_equal(data, original)


def test_horne_streams_spillable_outputs_without_materializing_input(
        monkeypatch, tmp_path):
    """Check horne streams spillable outputs without materializing input."""
    data, deep = _dataset()
    nwave, naperture = data.shape[1:]
    mask = extract.miri_optimal_aperture_mask(
        np.full(nwave, 6.5, np.float32), 2.5, naperture)
    profile = extract.miri_optimal_profile(
        extract.miri_optimal_deepframe_prep(deep), mask, mask_output=False)
    reference = extract.horne_optimal_extract(
        data, profile, aperture_mask=mask, max_iter=10)

    class SliceOnly:
        shape, ndim, dtype = data.shape, data.ndim, data.dtype

        def __array__(self, *args, **kwargs):
            raise AssertionError('Whole Horne cube was converted')

        def __getitem__(self, key):
            assert isinstance(key, tuple)
            assert key[1].stop - key[1].start <= 2
            return data[key]

    monkeypatch.setenv('EXOTEDRF_MAX_HOST_BYTES', '4096')
    monkeypatch.setenv('EXOTEDRF_SCRATCH_DIR', str(tmp_path))
    actual = extract.horne_optimal_extract(
        SliceOnly(), profile, aperture_mask=mask, max_iter=10, chunk_size=2)
    assert isinstance(actual[0], np.memmap)
    assert isinstance(actual[1], np.memmap)
    assert actual[2] == reference[2]
    np.testing.assert_allclose(actual[0], reference[0], rtol=2e-5)
    np.testing.assert_allclose(actual[1], reference[1], rtol=2e-5)


@pytest.mark.parametrize('max_iter', [0, 1, 2, 10])
def test_horne_chunking_obeys_global_iteration_budget(max_iter):
    """Check horne chunking obeys global iteration budget."""
    data, deep = _dataset()
    mask = extract.miri_optimal_aperture_mask(
        np.full(data.shape[1], 6.5, np.float32), 2.5, data.shape[2])
    profile = extract.miri_optimal_profile(
        extract.miri_optimal_deepframe_prep(deep), mask, mask_output=False)
    actual = extract.horne_optimal_extract(
        data, profile, aperture_mask=mask, max_iter=max_iter, chunk_size=3)
    expected = _v1_horne()(data.transpose(0, 2, 1).copy(), deep.T.copy(),
        ymin=np.full(data.shape[1], 4), ymax=np.full(data.shape[1], 9),
        xmin=0, xmax=data.shape[1], max_iter=max_iter)
    assert actual[2] == tuple(expected[2])
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5)


@pytest.mark.parametrize('case', ['negative_profile', 'nan_inside', 'nan_outside'])
@pytest.mark.parametrize('chunk', [None, 3])
def test_horne_edge_values_preserve_actual_v1_median_semantics(
        monkeypatch, case, chunk):
    """Check horne edge values preserve actual v1 median semantics."""
    import scipy.ndimage

    data, deep = _dataset()
    if case == 'negative_profile':
        deep[9, 1] = -10.
        deep[10, 5] = -3.
    elif case == 'nan_inside':
        data[10, 9, 6] = np.nan
    else:
        data[2, 10, 1] = np.nan
    nwave = data.shape[1]
    expected = _v1_horne()(
        data.transpose(0, 2, 1).copy(), deep.T.copy(),
        ymin=np.full(nwave, 4), ymax=np.full(nwave, 9),
        xmin=0, xmax=nwave, max_iter=10)
    filtered_shapes = []
    original_filter = scipy.ndimage.median_filter

    def record_filter(values, size, **kwargs):
        filtered_shapes.append(values.shape)
        return original_filter(values, size, **kwargs)

    monkeypatch.setattr(scipy.ndimage, 'median_filter', record_filter)
    mask = extract.miri_optimal_aperture_mask(
        np.full(nwave, 6.5, np.float32), 2.5, data.shape[2])
    profile = extract.miri_optimal_profile(
        extract.miri_optimal_deepframe_prep(deep), mask, mask_output=False)
    actual = extract.horne_optimal_extract(
        data, profile, aperture_mask=mask, max_iter=10, chunk_size=chunk)
    assert actual[2] == tuple(expected[2])
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5)
    if case.startswith('nan'):
        assert filtered_shapes
        assert all(shape == (data.shape[0], 1) for shape in filtered_shapes)
    else:
        assert not filtered_shapes
