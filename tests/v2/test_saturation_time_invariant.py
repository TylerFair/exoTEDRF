"""Opt-in time-invariant saturation mask (stages.time_invariant_saturation)."""
import numpy as np

from exotedrf.v2 import config as v2config
from exotedrf.v2 import core, stages

SAT = np.uint8(core.DQ_SATURATED)


def _groupdq(first_sat, ngroups=5):
    """GROUPDQ (nints, ngroups, 1, 1) saturated from ``first_sat[i]`` on."""
    g = np.zeros((len(first_sat), ngroups, 1, 1), np.uint8)
    for i, k in enumerate(first_sat):
        g[i, k:] |= SAT
    return g


def test_pixel_flagged_from_one_group_for_the_whole_visit():
    # Brighter (out-of-transit) integrations saturate at group 2, dimmer ones at 3.
    """Check pixel flagged from one group for the whole visit."""
    g = _groupdq([2] * 6 + [3] * 4)
    out, limit = stages.time_invariant_saturation(g, quantile=0.2)
    assert int(limit[0, 0]) == 2
    assert np.array_equal(out, _groupdq([2] * 10))


def test_rare_saturation_is_not_forced_but_stays_flagged():
    """Check rare saturation is not forced but stays flagged."""
    g = _groupdq([5] * 9 + [1])
    out, limit = stages.time_invariant_saturation(g, quantile=0.2)
    assert int(limit[0, 0]) == 5
    assert np.array_equal(out, g)


def test_option_disables_segment_streaming():
    """Check option disables segment streaming."""
    cfg = {'observing_mode': 'NIRSpec/PRISM', 'RampFitStep': 'run'}
    assert v2config.supports_stage1_streaming(cfg)
    assert not v2config.supports_stage1_streaming(dict(cfg, saturation_time_invariant=True))
