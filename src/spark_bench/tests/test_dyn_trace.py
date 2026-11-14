"""Tracing reads only (same physics, no extra steps, no random draws), and the failure-mode rules."""
import json

import mujoco
import numpy as np

from spark_bench.dyn.failure_modes import classify, trial_metrics
from spark_bench.dyn.trace import TrialTrace

XML = """<mujoco><worldbody>
  <body name="gripper0_hand" pos="0 0 .5"><joint name="gripper0_finger_joint1" type="slide" axis="1 0 0"/>
    <joint name="gripper0_finger_joint2" type="slide" axis="0 1 0"/>
    <geom size=".01"/><site name="gripper0_grip_site"/></body>
  <body name="soup_1_main" pos=".2 0 .1"><freejoint/><geom size=".02"/></body>
  <body name="basket_1_main" pos="-.2 0 .1"><freejoint/><geom size=".05"/></body>
</worldbody></mujoco>"""


class _Env:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.sim = type('S', (), {})()
        self.sim.model = type('M', (), {'_model': self.model})()
        self.sim.data = type('D', (), {'_data': self.data})()
        self.calls = 0
        self.horizon, self.timestep = 100, 0

    def step(self, action):
        self.calls += 1
        self.timestep += 1
        mujoco.mj_step(self.model, self.data)
        return {}, 0.0, False, {}


def _roll(env, n=12):
    for _ in range(n):
        env.step(np.zeros(7))
    return env.data.qpos.copy(), env.data.qvel.copy()


def test_trace_changes_nothing_and_saves_rows(tmp_path):
    bare = _Env()
    q0, v0 = _roll(bare)
    env = _Env()
    meta = {}
    tr = TrialTrace(env, tmp_path / 't.npz', 'soup_1', 'basket_1', meta, stride=5)
    tr.install()
    rng = np.random.get_state()[1].copy()
    q1, v1 = _roll(env)
    tr.uninstall()
    tr.save()
    np.testing.assert_array_equal(q0, q1)
    np.testing.assert_array_equal(v0, v1)
    np.testing.assert_array_equal(rng, np.random.get_state()[1])
    assert env.calls == 12
    z = np.load(tmp_path / 't.npz')
    info = json.loads(str(z['info']))
    assert info['pick_body'] == 'soup_1_main' and info['place_body'] == 'basket_1_main'
    # First row forced, then every 5th step.
    assert list(z['rows'][:, 0]) == [1, 5, 10]


def _synthetic(ee_xy, lift=False, aperture=0.0):
    """A trial with one grasp note: arm stops at ee_xy above an object that slid 10 cm in +x before it."""
    n = 40
    rows = np.zeros((n, 12 + 6))
    rows[:, 0] = np.arange(n) * 5
    rows[:, 1:3] = ee_xy
    rows[:, 3] = 0.05
    rows[:, 4], rows[:, 5] = 0.08, -1.0
    rows[:, 6] = 0                         # note 0: move_to_keypoint
    rows[20:, 6] = 1                       # note 1: grasp
    rows[20:, 4], rows[20:, 5] = aperture, 1.0
    rows[:, 12:15] = [0.1, 0, 0.02]        # true object
    if lift:
        rows[30:, 14] = 0.15
        rows[30:, 3] = 0.16
    rows[:, 15:18] = [0.5, 0.5, 0.02]      # basket
    info = {'body_names': ['soup', 'basket'], 'pick_body': 'soup', 'place_body': 'basket',
            'initial_bodies': [[0, 0, 0.02], [0.5, 0.5, 0.02]],
            'notes': [{'attempt': 0, 'idx': 0, 'type': 'move_to_keypoint', 'label': 'soup'},
                      {'attempt': 0, 'idx': 1, 'type': 'grasp', 'label': None}],
            'perturb_row': 0, 'perturb_t': 0, 'stride': 5, 'n_env_steps': 200, 'final_timestep': 200, 'horizon': 6000}
    return trial_metrics(rows, info, 10.0, (0.1, 0.0))


def test_failure_rules():
    fail = {'success': False}
    assert classify(_synthetic((0, 0)), fail) == 'never_reaimed'             # stayed at the old spot
    assert classify(_synthetic((0.09, 0)), fail) == 'reaimed_missed'         # at the object, closed on nothing
    assert classify(_synthetic((0.09, 0), aperture=0.04), fail) == 'grasped_wrong_place'
    assert classify(_synthetic((0.1, 0), lift=True, aperture=0.04), fail) == 'grasped_dropped'
    assert classify(None, {'success': False, 'grasp_outcomes': [{'outcome': 'empty_close'}]}) == 'json:only_empty_closes'
