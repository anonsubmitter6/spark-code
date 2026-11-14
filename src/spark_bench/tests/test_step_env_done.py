import numpy as np

from spark_bench.libero_pro.motion import step_env


class Env:
    def __init__(self, fail=False):
        self.fail = fail

    def step(self, action):
        if self.fail:
            raise ValueError('terminated')
        return None, 0.0, True, {}


def test_done_stops_motion_by_default(monkeypatch):
    monkeypatch.delenv('SPARK_IGNORE_DONE', raising=False)
    assert step_env(Env(), np.zeros(7)) is True


def test_flag_hides_done_but_not_a_terminated_episode(monkeypatch):
    monkeypatch.setenv('SPARK_IGNORE_DONE', '1')
    assert step_env(Env(), np.zeros(7)) is False
    assert step_env(Env(fail=True), np.zeros(7)) is True
