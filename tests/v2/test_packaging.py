"""Check package discovery, optional dependencies and reference files."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_setup_discovers_v2_subpackages_and_exposes_jax_cpu_extra():
    """Check setup discovers v2 subpackages and exposes jax CPU extra."""
    source = (ROOT / 'setup.py').read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == 'find_packages']
    assert calls, 'setup.py must discover exotedrf.v2 and kernels'
    assert "'v2-cpu': ['jax[cpu]>=0.4.30']" in source
    assert "'v2':" not in source
    assert (ROOT / 'exotedrf/v2/__init__.py').is_file()
    assert (ROOT / 'exotedrf/v2/kernels/__init__.py').is_file()
    for asset in ('jwst_niriss_spectrace_0023.fits',
                  'model_background256.npy', 'model_background96.npy'):
        assert f"'files/{asset}'" in source
        assert (ROOT / 'files' / asset).is_file()
