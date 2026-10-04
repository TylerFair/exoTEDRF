"""Check that the optimizer releases raw ramps during the final reduction."""

import weakref

from exotedrf.v2 import io, pipeline
from exotedrf.v2.optimize import run_optimizer
from .test_yaml_niriss_e2e import _synthetic_refpack, SEGMENT_NINTS, yaml_run  # noqa: F401


def test_raw_visit_released_during_final_reduction(yaml_run, monkeypatch):
    """Release the raw science array after reference-pixel correction."""
    cfg = dict(yaml_run.cfg)
    cfg['pipeline_outputs_directory'] = str(yaml_run.base / 'diag_out')
    raw = {}
    load_ramp = io.load_ramp_cube

    def load(*args, **kwargs):
        """Record a weak reference to the input science array."""
        cube = load_ramp(*args, **kwargs)
        raw['science'] = weakref.ref(cube.data)
        return cube

    monkeypatch.setattr(io, 'load_ramp_cube', load)
    run_pipeline = pipeline.Pipeline.run
    exposure = sum(SEGMENT_NINTS)
    alive = []

    def run(self, state, params, start=0, stop=None, store=None,
            verbose=False, observer=None):
        """Observe raw-array lifetime during the full reduction."""
        if state.cube.data.shape[0] == exposure and start == 0 and stop is None:
            parent_observer = observer

            def observe(step, before, after):
                """Record whether the raw science array is still resident."""
                if parent_observer is not None:
                    parent_observer(step, before, after)
                if step.name == 'RefPixStep':
                    alive.append(raw['science']() is not None)
            observer = observe
        # Release the wrapper's reference before running the graph.
        holder = [state]
        del state
        return run_pipeline(self, holder.pop(), params, start, stop, store,
                            verbose, observer)

    monkeypatch.setattr(pipeline.Pipeline, 'run', run)
    run_optimizer(cfg, logger=None, refpack=_synthetic_refpack(), write_products=False)
    assert alive and alive[0] is False
