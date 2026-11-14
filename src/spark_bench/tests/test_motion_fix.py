"""SPARK_MOTION_FIX pure decisions: reach check, wrist-turned rim target, contact stall, and the failed-pick leaf."""
import numpy as np
import pytest

from spark_bench.libero_pro.binding import keep_pick_binding
from spark_bench.libero_pro.motion import blocked_above, contact_stall, reach_short
from spark_bench.libero_pro.primitives._common import fail_pick
from spark_bench.libero_pro.primitives.keypoint import turned_rim


def test_reach_short():
    aim = np.array([0.055, 0.058, 0.939])
    assert not reach_short(aim + [0.005, -0.005, 0.01], aim)       # at the rim
    assert not reach_short(aim + [0.004, 0.0, -0.035], aim)        # top-down crawl ends below its aim
    assert reach_short(aim + [0.0, 0.0, 0.137], aim)               # stopped on the cabinet edge
    assert reach_short(aim + [-0.26, 0.0, 0.0], aim)               # swung away
    assert reach_short(aim + [0.0, 0.0, 0.045], aim)               # 4.5 cm above the rim point


def test_turned_rim():
    rim = np.array([0.055, 0.058, 0.939])
    out = turned_rim(rim, centre_y=0.033)
    assert np.allclose(out, [0.030, 0.033, 0.939])                  # same 2.5 cm offset, on the -x side
    assert rim[1] == 0.058                                          # input untouched
    assert np.allclose(turned_rim(np.array([0.0, -0.045, 0.9]), 0.0), [-0.045, 0.0, 0.9])


def test_contact_stall():
    tgt = np.array([0.055, 0.058, 0.939])
    assert contact_stall(tgt + [0.01, 0.0, 0.137], tgt)            # over the target, blocked in z
    assert contact_stall(tgt + [0.08, 0.0, 0.05], tgt)             # stalled while descending
    assert not contact_stall(tgt + [-0.20, 0.10, -0.05], tgt)      # far in xy, target above: reach stall, IK helps


def test_blocked_above():
    high = np.array([0.055, 0.058, 1.039])
    assert blocked_above(high + [0.008, 0.0, 0.063], high)         # over the rim, 6 cm above the high point
    assert blocked_above(high + [0.01, -0.04, 0.096], high)        # held 4 cm to the side by the cabinet
    assert not blocked_above(high + [0.0, 0.0, 0.004], high)       # arrived
    assert not blocked_above(high + [0.12, 0.05, 0.08], high)      # still in transit


def test_keep_pick_binding():
    from types import SimpleNamespace as D
    a, twin, plate = D(position_3d=[0.055, 0.033, 0.97]), D(position_3d=[0.01, 0.21, 1.13]), D(position_3d=[0.2, 0.2, 0.9])
    old = {'bowl': a, 'round black container': a, 'plate': plate}
    fresh = {'bowl': twin, 'round black container': D(position_3d=[0.06, 0.03, 0.97]), 'plate': twin, 'box': twin}
    out = keep_pick_binding(fresh, old, anchor=a.position_3d)
    assert out['bowl'] is a                                         # not moved to the bowl on the cabinet
    assert out['round black container'] is fresh['round black container']   # same instance, fresher position
    assert out['plate'] is twin and out['box'] is twin              # other labels are free
    assert keep_pick_binding(fresh, old, None) is fresh
    # with the scene in view: the instance still stands at the anchor -> held back
    assert keep_pick_binding(fresh, old, a.position_3d, present=lambda: [a, twin])['bowl'] is a
    # nothing at the anchor any more (the object was displaced 10 cm): the fresh detection is the object
    moved = {'bowl': D(position_3d=[0.155, 0.033, 0.97])}
    assert keep_pick_binding(moved, old, a.position_3d, present=lambda: [moved['bowl'], plate])['bowl'] is moved['bowl']
    assert keep_pick_binding(moved, old, a.position_3d, present=lambda: None)['bowl'] is a   # no view: unchanged rule
    assert keep_pick_binding(moved, old, a.position_3d)['bowl'] is a                          # callers without a view


def test_drop_other_objects():
    from types import SimpleNamespace as D
    pytest.importorskip('robosuite')   # the perception package imports the sim stack; bench venv
    from spark_bench.libero_pro.perception.relations import _PLACE_CLAUSE, drop_other_objects

    def blob(r0, r1, c0, c1):
        m = np.zeros((40, 40), bool)
        m[r0:r1, c0:c1] = True
        return D(mask=m)
    bowl1, bowl2, plate_as_bowl, ramekin = blob(0, 8, 0, 8), blob(20, 28, 0, 8), blob(10, 20, 20, 32), blob(30, 36, 30, 36)
    plate, cabinet = blob(10, 20, 21, 32), blob(15, 40, 0, 12)      # the cabinet mask holds bowl2, it is not bowl2
    out = drop_other_objects([bowl1, bowl2, ramekin, plate_as_bowl], [plate, cabinet])
    assert out == [bowl1, bowl2, ramekin]                           # the plate is no bowl
    two = [bowl1, plate_as_bowl]
    assert drop_other_objects(two, [plate]) is two                  # keep at least two
    assert _PLACE_CLAUSE.search('pick the akita black bowl from table center and place it on the plate').group(1) == 'plate'


class _Ex:
    def __init__(self):
        self.holding, self.pick_failed, self.last_pick_label, self.trial_meta = True, None, 'bowl', {}


def test_fail_pick_marks_the_attempt():
    ex = _Ex()
    fail_pick(ex, 'not_reached')
    assert ex.pick_failed == 'not_reached' and ex.holding is False
    assert ex.trial_meta['grasp_outcomes'][-1]['outcome'] == 'not_reached'


def test_failed_pick_fails_its_leaf():
    """A handler leaf whose pick failed returns False (the retry / fallback node above acts on it), once."""
    pytest.importorskip('robosuite')   # the tree walker imports the sim stack; run this one in the bench venv
    from spark_bench.libero_pro import conditions

    class Ex(_Ex, conditions.TreeMixin):
        abort = None
        cfg = type('C', (), {'_trace': None})()
        _tree_stats = {'no_handler': []}
        _leaf_idx = 0

        def _goal_satisfied(self):
            return False

        def dispatch_action(self, idx, node):
            fail_pick(self, 'empty_close')
            return True
    ex = Ex()
    assert ex._leaf({'type': 'grasp'}) is False
    assert ex.pick_failed is None
    ex.dispatch_action = lambda idx, node: True
    assert ex._leaf({'type': 'grasp'}) is True


def test_extent_centre():
    pytest.importorskip('robosuite')
    from spark_bench.libero_pro.perception.relations import extent_centre
    # a table seen with one quarter hidden: the centroid moves away from the hidden part, the extent does not
    g = np.array([[x, y, 0.9] for x in np.linspace(-0.4, 0.4, 41) for y in np.linspace(-0.6, 0.6, 61)
                  if not (x > 0 and y < 0)])
    assert np.linalg.norm(g[:, :2].mean(0)) > 0.1
    assert np.allclose(extent_centre(g), [0.0, 0.0, 0.9], atol=0.02)
