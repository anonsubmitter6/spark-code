"""A mid-motion retarget shifts the hover point; the descent that follows must use the same shift."""
from types import SimpleNamespace

import numpy as np
import pytest

from spark_bench.libero_pro import binding
from spark_bench.libero_pro.primitives import keypoint


class _Stop(Exception):
    pass


class _Exec:
    last_pick_label = 'bowl'

    def __init__(self):
        self.moves = []

    def _move_to_monitored(self, label, target, gripper_open, steps=200):
        self.monitor_shift = np.array([0.05, -0.02, 0.0])

    def _move_to(self, target, gripper_open, steps=200):
        self.moves.append(np.asarray(target, float).copy())
        raise _Stop   # the descent is the first plain move; stop before the contact crawl


def test_pick_descent_follows_retarget():
    ex = _Exec()
    try:
        keypoint._do_top_down_pick(ex, np.array([0.5, 0.1, 0.9]))
    except _Stop:
        pass
    assert np.allclose(ex.moves[0], [0.55, 0.08, 0.9])


class _Monitored(binding.BindingMixin):
    cfg = SimpleNamespace(scene_diff=True)
    env = model = data = joint_ids = ee_site = arm_actuator_ids = gripper_actuator_ids = None

    def __init__(self):
        self.plan_binding = {'bowl': np.zeros(3)}

    def _run_scene_diff(self, label, context=''):
        return SimpleNamespace(target_key='bowl')

    def _maybe_refresh_perception(self, act):
        self.plan_binding['bowl'] = np.array([0.05, -0.02, 0.0])

    def _handle_target_diff(self, sd, context=''):
        return True


@pytest.mark.parametrize('follow, shift', [(False, [0, 0, 0]), (True, [0.05, -0.02, 0])])
def test_descent_shift_only_with_the_switch(monkeypatch, follow, shift):
    monkeypatch.setattr(binding, 'move_to', lambda *a, **k: 0.0)
    monkeypatch.setattr(binding, 'FOLLOW_RETARGET', follow)
    ex = _Monitored()
    ex._move_to_monitored('bowl', np.array([0.5, 0.1, 0.9]), True, steps=200)
    assert np.allclose(ex.monitor_shift, shift)
