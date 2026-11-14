"""The switchable recovery options: fresh re-bind, unknown diff status, object snap, step-based freshness gate."""
from types import SimpleNamespace

import numpy as np
import yaml

from spark_bench.fair.execution import fresh_pick_swap
from spark_bench.fair.planning import _plan, _run_label_validator, snap_to_object_phrase
from spark_bench.libero_pro.executor import LiberoExecutor
from spark_core.binding import STATUS_OK, STATUS_UNKNOWN, compute_scene_diff, mark_no_evidence
from spark_core.bt_validator import fuzzy_snap_bt_labels

PLAN = {'tree': {'type': 'sequence', 'children': [
    {'type': 'move_to_keypoint', 'params': {'keypoint_label': 'cream cheese box'}},
    {'type': 'grasp', 'params': {}},
    {'type': 'move_to_keypoint', 'params': {'keypoint_label': 'basket'}}]}}


def _labels(score):
    return [c['params'].get('keypoint_label') for c in score['tree']['children']]


def test_swap_only_when_the_phrase_is_not_fresh():
    fresh = {'small white rectangular box': 0.6, 'basket': 0.8}
    new, label = fresh_pick_swap(PLAN, 'cream cheese box', fresh)
    assert label == 'small white rectangular box'
    assert _labels(new) == ['small white rectangular box', None, 'basket']
    assert _labels(PLAN)[0] == 'cream cheese box'  # the plan itself is left alone
    assert fresh_pick_swap(PLAN, 'cream cheese box', {'cream cheese box': 0.1, **fresh}) is None
    assert fresh_pick_swap(PLAN, 'cream cheese box', {'basket': 0.8}) is None  # no phrase of this object is fresh


def test_no_fresh_evidence_reads_unknown():
    sd = compute_scene_diff({'a': np.zeros(3), 'b': np.zeros(3)}, {'a': np.zeros(3), 'b': np.zeros(3)}, target_label='a')
    assert sd.target_status == STATUS_OK
    mark_no_evidence(sd, {'b'})
    assert sd.target_status == STATUS_UNKNOWN and sd.labels['a'].status == STATUS_UNKNOWN
    assert sd.labels['b'].status == STATUS_OK and sd.ok


def _plan_with(label):
    return {'tree': {'type': 'sequence', 'children': [{'type': 'move_to_keypoint', 'params': {'keypoint_label': label}}]}}


def test_validator_does_not_snap_a_pick_onto_the_basket():
    det_map = {k: SimpleNamespace(confidence=c) for k, c in
               (('small white rectangular box', 0.6), ('basket', 0.8), ('wire basket', 0.2))}
    allowed = set(det_map)
    old = _plan_with('cream cheese')
    fuzzy_snap_bt_labels(old, allowed)
    assert _labels(old) == ['wire basket']  # default behaviour, kept
    related = _plan_with('cream cheese')
    assert fuzzy_snap_bt_labels(related, allowed, require_related=True)[1] == ['cream cheese']
    new = _plan_with('cream cheese')
    assert snap_to_object_phrase(new, det_map)[0]['to'] == 'small white rectangular box'
    cfg = SimpleNamespace(verbose=False, no_gemini=True, validator_object_snap=True, _replay_plans=None)
    assert _labels(_run_label_validator(cfg, _plan_with('cream cheese'), det_map, 'i', None, 'p', 'q')) == [
        'small white rectangular box']


def test_plan_replay_returns_recorded_plans_in_order():
    cfg = SimpleNamespace(_replay_plans=[yaml.safe_dump(PLAN)], no_gemini=False)
    assert _labels(_plan(cfg, 'i', {}, None, 'p', 'q')) == _labels(PLAN)
    assert _plan(cfg, 'i', {}, None, 'p', 'q') is None


def test_step_gate_reads_the_env_clock():
    ex = LiberoExecutor.__new__(LiberoExecutor)
    ex.env = SimpleNamespace(env=SimpleNamespace(timestep=7))
    ex.cfg = SimpleNamespace(freshness_steps=3)
    assert ex._perception_clock() == 7.0
    ex.cfg = SimpleNamespace(freshness_steps=0)
    assert ex._perception_clock(wall=5.0) == 5.0
