"""Bound compiled executable storage across independent test modules."""

import gc

import jax
import pytest


@pytest.fixture(scope='module', autouse=True)
def release_compiled_test_shapes():
    """Release compiled executables after each test module."""
    yield
    # Release completed test shapes before compiling the next module.
    jax.clear_caches()
    gc.collect()
