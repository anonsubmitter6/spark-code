"""The plan-as-tree walker and the post-condition leaves, on a stub executor (no simulator)."""
from types import SimpleNamespace

import numpy as np

from spark_bench.fair.execution import _attempt_ok
from spark_bench.libero_pro import conditions as C


class Ex(C.TreeMixin):
    """Leaves log their type; verify leaves answer from a script."""

    def __init__(self, verdicts=()):
        self.log = []
        self.verdicts = list(verdicts)
        self.abort = None
        self.holding = False
        self.trial_meta = {}
        self.cfg = SimpleNamespace(verbose=False)

    def _goal_satisfied(self):
        return False

    def _ee(self):
        return np.zeros(3)

    def _move_to(self, *a, **k):
        self.log.append('lift')

    def dispatch_action(self, idx, node):
        self.log.append(node['type'])
        return True


def _leaf(t):
    return {'type': t, 'params': {}}


def _seq(*ts):
    return {'type': 'sequence', 'children': [_leaf(t) if isinstance(t, str) else t for t in ts]}


def _scripted(monkeypatch, verdicts):
    ex = Ex()
    answers = list(verdicts)

    def verdict(name):
        def fn(e, params):
            e.log.append(name)
            return answers.pop(0)
        return fn
    monkeypatch.setattr(C, 'CONDITIONS', {'verify_grasp': verdict('verify_grasp'),
                                          'verify_placed': verdict('verify_placed')})
    monkeypatch.setattr(C, 'aperture', lambda e: 0.04)
    return ex


def _run(ex, tree):
    ex._run_tree({'tree': tree})
    return ex.log, ex.trial_meta


def test_retry_stops_at_first_pass_and_success_is_not_rerun(monkeypatch):
    tree = _seq({'type': 'retry', 'params': {'max_attempts': 3}, 'children': [_seq('grasp', 'verify_grasp')]}, 'wait')
    log, meta = _run(_scripted(monkeypatch, [False, True]), tree)
    assert log == ['grasp', 'verify_grasp', 'lift', 'grasp', 'verify_grasp', 'wait']
    assert meta['tree_ok'] is True and meta['tree_runs'][0]['reattempts'] == 1


def test_fallback_runs_second_branch_only_on_failure(monkeypatch):
    tree = {'type': 'fallback', 'children': [_seq('release', 'verify_placed'), _seq('grasp', 'release')]}
    assert _run(_scripted(monkeypatch, [True]), tree)[0] == ['release', 'verify_placed']
    assert _run(_scripted(monkeypatch, [False]), tree)[0] == ['release', 'verify_placed', 'lift', 'grasp', 'release']


def test_exhausted_retry_fails_the_tree_and_skips_the_rest(monkeypatch):
    tree = _seq({'type': 'retry', 'params': {'max_attempts': 2}, 'children': [_seq('grasp', 'verify_grasp')]}, 'release')
    log, meta = _run(_scripted(monkeypatch, [False, False]), tree)
    assert 'release' not in log and meta['tree_ok'] is False


def test_budget_caps_nested_reattempts(monkeypatch):
    inner = {'type': 'retry', 'params': {'max_attempts': 3}, 'children': [_seq('grasp', 'verify_grasp')]}
    tree = {'type': 'fallback', 'children': [inner, inner, inner]}
    log, meta = _run(_scripted(monkeypatch, [False] * 20), tree)
    assert meta['tree_runs'][0]['reattempts'] == 6 and log.count('lift') == 6


def test_plan_without_verify_leaves_the_runner_alone(monkeypatch):
    log, meta = _run(_scripted(monkeypatch, []), _seq('grasp', 'release'))
    assert log == ['grasp', 'release'] and 'tree_ok' not in meta


def test_attempt_ok_prefers_the_tree_verdict(monkeypatch):
    monkeypatch.setenv('SPARK_NO_ORACLE', '1')
    monkeypatch.delenv('SPARK_ORACLE_RETRY', raising=False)
    slip = [{'outcome': 'slip'}]
    assert _attempt_ok(None, {'tree_ok': True, 'grasp_outcomes': slip}) is True    # a passed verify is not redone
    assert _attempt_ok(None, {'tree_ok': False, 'grasp_outcomes': []}) is False
    assert _attempt_ok(None, {'grasp_outcomes': slip}) is False                    # no tree verdict


def _placed_ex(monkeypatch, seen):
    ex = Ex()
    ex.last_pick_label = 'box'
    ex._aperture_at_release = 0.04
    ex.det_map = {'box': SimpleNamespace(position_3d=np.zeros(3), mask=None),
                  'basket': SimpleNamespace(position_3d=np.array([0.3, 0.3, 0.1]), mask=None)}
    ex.sam3, ex.prompts_for_refresh = object(), ['box', 'basket']
    monkeypatch.setattr(C, 'aperture', lambda e: 0.08)
    monkeypatch.setattr(C, 'simulator_truth', lambda e: {})
    monkeypatch.setattr(C, 'fresh_instances', lambda e, labels, **k: seen)
    monkeypatch.setattr(C, 'mask_points_3d', lambda e, d: None)
    return ex


def _det(label, xy):
    return SimpleNamespace(label=label, confidence=0.9, position_3d=np.array([*xy, 0.1]))


def test_verify_placed_needs_positive_evidence_to_fail(monkeypatch):
    params = {'obj': 'box', 'container': 'basket'}
    assert C.verify_placed(_placed_ex(monkeypatch, [('agentview', _det('box', (0.3, 0.3)))]), params) is True
    assert C.verify_placed(_placed_ex(monkeypatch, []), params) is True              # unseen: abstain
    ex = _placed_ex(monkeypatch, [('agentview', _det('box', (0.0, 0.0)))])
    assert C.verify_placed(ex, params) is False                                      # seen only outside
    assert ex.trial_meta['pc_events'][0]['reason'] == 'seen only outside target'
    ex = _placed_ex(monkeypatch, [])
    ex._aperture_at_release = 0.001
    assert C.verify_placed(ex, params) is False                                      # jaws held nothing
    monkeypatch.setattr(C, 'aperture', lambda e: 0.012)
    ex = _placed_ex(monkeypatch, [('agentview', _det('box', (0.3, 0.3)))])
    ex._aperture_at_release = 0.005
    monkeypatch.setattr(C, 'aperture', lambda e: 0.012)
    assert C.verify_placed(ex, params) is True                                       # slow opening is opening
    monkeypatch.setattr(C, 'aperture', lambda e: 0.0055)
    assert C.verify_placed(ex, params) is False                                      # no rise: still gripping


def test_verify_grasp_reads_the_fingers_and_drops_the_hold(monkeypatch):
    ex = Ex()
    ex.holding = True
    monkeypatch.setattr(C, 'simulator_truth', lambda e: {})
    monkeypatch.setattr(C, 'aperture', lambda e: 0.001)
    assert C.verify_grasp(ex, {}) is False and ex.holding is False
    monkeypatch.setattr(C, 'aperture', lambda e: 0.04)
    assert C.verify_grasp(ex, {}) is True
