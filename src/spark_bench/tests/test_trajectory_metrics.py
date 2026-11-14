"""DTW, roles and labels of the trajectory analysis, and a re-aim that costs nothing in the object frame."""
import numpy as np

from spark_bench.dyn.trajectory_metrics import (action_cosine, attempt_features, dtw_distance, failure_label,
                                                path_distance, resample, roles_of)

NOTES = [{'attempt': 0, 'idx': -1, 'type': 'attempt_start', 'label': None},
         {'attempt': 0, 'idx': 0, 'type': 'move_to_keypoint', 'label': 'can'},
         {'attempt': 0, 'idx': 1, 'type': 'grasp', 'label': None},
         {'attempt': 0, 'idx': 2, 'type': 'move_relative', 'label': None},
         {'attempt': 0, 'idx': 3, 'type': 'move_to_keypoint', 'label': 'basket'},
         {'attempt': 0, 'idx': 4, 'type': 'release', 'label': None}]


def _rows(obj):
    """A pick at ``obj`` and a carry to a basket at (0, .3, 0); columns follow dyn/trace.py."""
    obj = np.asarray(obj, float)
    goal = np.array([0, .3, .1])
    ee = np.vstack([np.linspace([0, 0, .3], obj, 8), np.tile(obj, (4, 1)), np.linspace(obj, obj + [0, 0, .2], 4),
                    np.linspace(obj + [0, 0, .2], goal, 8), np.tile(goal, (4, 1))])
    note = np.repeat([1, 2, 3, 4, 5], [8, 4, 4, 8, 4])
    n = len(ee)
    pick = np.tile(obj, (n, 1))
    pick[12:] = ee[12:]
    rows = np.zeros((n, 12 + 6))
    rows[:, 0] = np.arange(n) * 5
    rows[:, 1:4] = ee
    rows[:, 4] = .05
    rows[:, 5] = 1
    rows[:, 6] = note
    rows[:, 9:12] = obj
    rows[:, 12:15] = pick
    rows[:, 15:18] = [0, .3, 0]
    info = {'notes': NOTES, 'body_names': ['can', 'basket'], 'pick_body': 'can', 'place_body': 'basket',
            'initial_bodies': [obj.tolist(), [0, .3, 0]]}
    return rows, info


def test_dtw_matches_the_paper_convention():
    a = np.zeros((4, 3))
    assert dtw_distance(a, a) == 0
    b = a + [0, 0, .1]
    assert abs(dtw_distance(a, b) - 4 * .1 / 8) < 1e-9
    assert dtw_distance(a, np.zeros((0, 3))) == float('inf')
    assert abs(action_cosine([[1, 0]], [[2, 0]]) - 1) < 1e-9
    assert len(resample(np.zeros((3, 3)))) == 20


def test_roles_come_from_order_not_labels():
    r = roles_of({'notes': NOTES}, 0)
    assert r == {'approach': [1], 'grasp': [2], 'lift': [3], 'carry': [4], 'release': [5]}


def test_correct_reaim_is_cheap_in_the_object_frame_only():
    rows, info = _rows([.2, 0, .05])
    f, p = attempt_features(rows, info, 0)
    rows2, info2 = _rows([.3, 0, .05])
    _, q = attempt_features(rows2, info2, 0)
    assert f['gt_pick_lifted'] and f['gt_cos_obj'] > .99
    world, _ = path_distance(p['world'], q['world'])
    obj, _ = path_distance(p['obj'], q['obj'])
    assert obj < world / 2


def test_wrong_object_label_and_success_is_not_read():
    a = {'lifted': ['juice'], 'gt_pick_lifted': False, 'gt_goal_xy': .5, 'gt_held_end': False}
    t = {'timed_out': False, 'mag': 5.0, 'attempts': [a], 'success': True}
    assert failure_label(t) == 'wrong_object'
    a2 = {**a, 'lifted': [], 'gt_close_xy': .01, 'gt_stale_xy': .05}
    assert failure_label({**t, 'attempts': [a2]}) == 'reaimed_missed'
    a3 = {**a2, 'gt_close_xy': .05, 'gt_stale_xy': .01}
    assert failure_label({**t, 'attempts': [a3]}) == 'never_reaimed'
    assert failure_label({**t, 'attempts': [{**a, 'gt_goal_xy': .01, 'gt_held_end': True}]}) == 'held_over_goal'
