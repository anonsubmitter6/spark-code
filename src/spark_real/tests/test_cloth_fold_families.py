"""
Single-arm cloth folds on a Franka-like and a UR10e-like profile, no hardware.

A fake executor built from the packaged family YAML records every pose the
skill commands (before any workspace clipping, so the skill itself is what is
checked) and every gripper call. The Franka poses are compared with a record
taken from the skill before it was made family-agnostic.

    PYTHONPATH=src python -m pytest src/spark_real/tests/test_cloth_fold_families.py -q
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import spark_real.control.recovery as redetect
from spark_real.config import family_block
from spark_real.control.grasp_strategy import measured_yaw_offset
from spark_real.control.motion import KeypointMixin
from spark_real.skills import cloth
from spark_real.skills import task_motion as tm


class FakeRobot:
    """Jaw double: a close on cloth stops at `closed_width` with the object flag set."""

    def __init__(self, events, closed_width=0.002, grips=True):
        self.events, self.closed_width, self.grips, self._open = events, closed_width, grips, True

    def open_gripper(self, **kw):
        self._open = True
        self.events.append(("open",))

    def close_gripper(self, force=None, **kw):
        self._open = False
        self.events.append(("close", force))

    def get_gripper_width(self):
        return 0.08 if self._open else (self.closed_width if self.grips else 0.0)

    def is_object_detected(self):
        return (not self._open) and self.grips


class FakeExecutor:
    """The executor surface the cloth skills use, bounds from the family YAML."""

    _oriented_grasp = KeypointMixin._oriented_grasp

    def __init__(self, family, scene, grips=True):
        raw = family_block(None, family)
        control = raw.get("control") or {}
        if family == "ur10e":
            self.WORKSPACE_MIN = np.array(control["workspace_min"], dtype=float)
            self.WORKSPACE_MAX = np.array(control["workspace_max"], dtype=float)
            self.MAX_REACH = float(control["max_reach"])
            self.TABLE_Z_FLOOR = float(control["table_z_floor"])
            self.MAX_YAW_OFFSET = math.radians(raw["grasp"]["max_yaw_offset_deg"])
            self.velocity = 0.25
            tcp0 = [-0.7, 0.0, 0.1]
        else:
            ws = raw["workspace"]
            self.WORKSPACE_MIN = np.array([ws["x_min"], ws["y_min"], ws["z_min"]], dtype=float)
            self.WORKSPACE_MAX = np.array([ws["x_max"], ws["y_max"], ws["z_max"]], dtype=float)
            self.MAX_REACH = float(ws["reach_max"])
            self.TABLE_Z_FLOOR = float(raw["table_z_floor"])
            self.MAX_YAW_OFFSET = math.radians(100)
            self.velocity = 0.10
            tcp0 = [0.4, 0.0, 0.35]
        self.GRASP_ORIENTATION = [float(v) for v in control["grasp_orientation"]]
        self._pipeline = SimpleNamespace(config=SimpleNamespace(robot_family=family))
        self._profile = None
        self._holding = False
        self.events = []
        self.robot = FakeRobot(self.events, grips=grips)
        self.detection_map = {k: {"position_3d": list(v)} for k, v in scene.items()}
        self._tcp = np.array(tcp0, dtype=float)

    def _check_abort(self):
        pass

    def _check_workspace(self, pos):
        return bool(np.linalg.norm(pos) <= self.MAX_REACH and np.all(pos >= self.WORKSPACE_MIN)
                    and np.all(pos <= self.WORKSPACE_MAX))

    def _get_current_position(self):
        return self._tcp.copy()

    def _grip_intact(self):
        self.events.append(("grip_intact",))
        return self.robot.is_object_detected()

    def _move_to(self, position, orientation, velocity=None):
        self._tcp = np.array(position, dtype=float)
        self.events.append(("move", [round(float(v), 6) for v in position],
                            [round(float(v), 6) for v in orientation], velocity))

    def moves(self):
        return [e for e in self.events if e[0] == "move"]


SCENES = {
    # Base frame of each rig: FR3 table near z = -0.02, UR10e table near z = -0.27.
    "franka": {"left sleeve": [0.45, 0.22, -0.02], "right sleeve": [0.45, -0.22, -0.02],
               "shirt": [0.42, 0.0, -0.02], "hem": [0.30, 0.0, -0.02]},
    "ur10e": {"shirt left sleeve": [-0.80, 0.27, -0.268], "shirt right sleeve": [-0.80, -0.17, -0.268],
              "shirt body": [-0.80, 0.05, -0.268], "shirt hem": [-0.62, 0.05, -0.268]},
}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(tm.time, "sleep", lambda s: None)


def _kinds(events):
    return [e[0] for e in events]


def _assert_in_bounds(ex):
    for _, pos, orient, _ in ex.moves():
        p = np.array(pos)
        assert p[2] >= ex.TABLE_Z_FLOOR - 1e-9, f"{pos} is below the table floor {ex.TABLE_Z_FLOOR}"
        assert ex._check_workspace(p), f"{pos} is outside the workspace"
        yaw = measured_yaw_offset(orient, ex.GRASP_ORIENTATION)
        assert abs(yaw) <= ex.MAX_YAW_OFFSET + 1e-6, f"yaw {math.degrees(yaw):.1f} deg past the limit"


@pytest.mark.parametrize("family", ["franka", "ur10e"])
def test_single_fold_sequence_bounds_and_mirror(family):
    scene = SCENES[family]
    corner_label = next(k for k in scene if "left sleeve" in k)
    corner = np.array(scene[corner_label])
    anchor = corner + [0.02, -0.12, 0.0]  # fold-axis midpoint
    ex = FakeExecutor(family, scene)
    r = cloth.cloth_fold_single(ex, {"corner_label": corner_label, "anchor_midpoint": anchor.tolist(),
                                     "yaw_deg": 90.0})
    assert r.success, r.message

    kinds = _kinds(e for e in ex.events if e[0] in ("move", "open", "close"))
    n = int(cloth._f(ex, {}, "arc_steps"))
    # open, hover, descend, pinch, lift, arc (n waypoints, the last lays), release, retract
    assert kinds == ["open", "move", "move", "close", "move"] + ["move"] * n + ["open", "move"]
    moves = ex.moves()
    hover, pinch, lift, arc, retract = moves[0], moves[1], moves[2], moves[3:3 + n], moves[-1]
    assert hover[1][2] > pinch[1][2] and lift[1][2] > pinch[1][2]
    assert np.allclose(pinch[1][:2], corner[:2])
    assert max(m[1][2] for m in arc) > lift[1][2]  # the swing rises above the lift
    mirror = 2 * anchor - corner
    assert np.allclose(arc[-1][1][:2], mirror[:2], atol=1e-6)  # lay at the mirror point
    assert arc[-1][1][2] < lift[1][2]
    assert retract[1][2] > arc[-1][1][2] and np.allclose(retract[1][:2], mirror[:2], atol=1e-6)
    _assert_in_bounds(ex)


@pytest.mark.parametrize("family", ["franka", "ur10e"])
def test_yaw_past_the_wrist_limit_uses_the_equivalent_pinch(family):
    scene = SCENES[family]
    corner_label = next(k for k in scene if "left sleeve" in k)
    land = (np.array(scene[corner_label]) + [0.0, -0.2, 0.0]).tolist()
    ex = FakeExecutor(family, scene)
    assert cloth.cloth_fold_single(ex, {"corner_label": corner_label, "land_point": land, "yaw_deg": 170.0}).success
    yaws = {round(math.degrees(measured_yaw_offset(m[2], ex.GRASP_ORIENTATION)), 3) for m in ex.moves()}
    assert yaws == {-10.0}  # 170 deg is past both limits; the jaws are symmetric under 180 deg
    _assert_in_bounds(ex)


@pytest.mark.parametrize("family", ["franka", "ur10e"])
def test_tshirt_is_three_folds_with_a_fresh_look_before_the_hem(family, monkeypatch):
    scene = SCENES[family]
    ex = FakeExecutor(family, scene)
    hem_label = next(k for k in scene if "hem" in k)
    new_hem = (np.array(scene[hem_label]) + [0.02, 0.01, 0.0]).tolist()

    def fake_redetect(executor, label):
        executor.events.append(("redetect", label))
        if label == hem_label:
            executor.detection_map[label] = {"position_3d": new_hem}

    monkeypatch.setattr(redetect, "redetect_single", fake_redetect)
    r = cloth.cloth_fold(ex, {"instruction": "fold the shirt"})
    assert r.success, r.message

    closes = [i for i, e in enumerate(ex.events) if e[0] == "close"]
    looks = [i for i, e in enumerate(ex.events) if e[0] == "redetect"]
    assert len(closes) == 3  # two sleeves and the hem, one pinch each
    assert looks and closes[1] < min(looks) and max(looks) < closes[2]
    assert hem_label in [ex.events[i][1] for i in looks]
    hem_pinch = [e for e in ex.events[:closes[2]] if e[0] == "move"][-1]
    assert np.allclose(hem_pinch[1][:2], new_hem[:2])  # the hem pinch uses the fresh detection
    _assert_in_bounds(ex)


def test_empty_pinch_fails_and_never_goes_below_the_floor():
    ex = FakeExecutor("ur10e", SCENES["ur10e"], grips=False)
    r = cloth.cloth_fold_single(ex, {"corner_label": "shirt left sleeve", "land_point": [-0.8, 0.0, -0.268]})
    assert not r.success and "pinch failed" in r.message
    assert ex.events[-1] == ("open",)
    assert len([e for e in ex.events if e[0] == "close"]) == int(cloth._f(ex, {}, "grasp_retries"))
    _assert_in_bounds(ex)


def test_fold_outside_the_workspace_is_refused_before_any_motion():
    ex = FakeExecutor("ur10e", SCENES["ur10e"])
    r = cloth.cloth_fold_single(ex, {"corner_label": "shirt left sleeve", "land_point": [-0.8, 0.95, -0.268]})
    assert not r.success and "workspace" in r.message
    assert ex.events == []


def test_ur10e_uses_its_own_gripper_numbers():
    ex = FakeExecutor("ur10e", SCENES["ur10e"])
    assert cloth.cloth_fold_single(ex, {"corner_label": "shirt left sleeve",
                                        "land_point": [-0.8, 0.0, -0.268]}).success
    block = family_block(None, "ur10e", "task_motion")["cloth"]
    assert ("close", float(block["force"])) in ex.events
    assert ("grip_intact",) in ex.events  # jaw object flag, not the jaw width
    assert all(m[3] is not None and m[3] <= 0.10 for m in ex.moves())  # every move has a slow speed


# Recorded from cloth.py before the family-agnostic change (commit c26a32d) with
# this fake: every commanded pose (x, y, z, rx, ry, rz), its velocity argument,
# and every jaw command.
BEFORE = json.loads((Path(__file__).parent / "cloth_fold_franka_before.json").read_text() or "{}")
FRANKA_SINGLE_PARAMS = {"corner_label": "left sleeve", "land_point": [0.45, -0.2, -0.02],
                        "yaw_deg": 90.0, "keep_x": True}
FRANKA_SHIRT_PARAMS = {"instruction": "fold the shirt"}


def _record(ex):
    return {"poses": [m[1] + m[2] for m in ex.moves()], "velocities": [m[3] for m in ex.moves()],
            "gripper": [list(e) for e in ex.events if e[0] in ("open", "close")]}


@pytest.mark.parametrize("name, skill, params", [("single", cloth.cloth_fold_single, FRANKA_SINGLE_PARAMS),
                                                 ("shirt", cloth.cloth_fold, FRANKA_SHIRT_PARAMS)])
def test_franka_commands_are_unchanged(name, skill, params):
    ex = FakeExecutor("franka", SCENES["franka"])
    assert skill(ex, dict(params)).success
    assert _record(ex) == BEFORE[name]


def test_shirt_fold_on_the_real_executor_with_a_mocked_ur10e_driver(monkeypatch):
    """The executor surface the skill relies on exists on ScoreExecutor for the UR10e family."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_selector_semantics import _FakePipeline, _MockRobot

    from spark_real.control.executor import ScoreExecutor

    class _UR(_MockRobot):
        robot_family, GRIPPER_TYPE = "ur10e", "robotiq_2f85"

        def __init__(self):
            super().__init__()
            self._tcp, self.path, self.forces = np.array([-0.7, 0.0, 0.1]), [], []

        def get_joint_positions(self):
            return np.zeros(6)

        def open_gripper(self, **kw):
            self._open = True

        def close_gripper(self, force=None, **kw):
            self._open = False
            self.forces.append(force)

        def move_linear(self, pose, velocity=0.1, asynchronous=False):
            super().move_linear(pose, velocity, asynchronous)
            self.path.append(np.array(pose[:3], dtype=float))

    monkeypatch.setattr(redetect, "redetect_single", lambda executor, label: None)
    pipeline = _FakePipeline()
    pipeline.config.robot_family = "ur10e"
    pipeline.profile = SimpleNamespace(raw={})
    robot = _UR()
    ex = ScoreExecutor(robot, detection_map={}, velocity=0.25, pipeline=pipeline)
    ex.update_detections({k: {"position_3d": v} for k, v in SCENES["ur10e"].items()})
    r = cloth.cloth_fold(ex, {"instruction": "fold the shirt"})
    assert r.success, r.message
    assert robot.forces == [30.0, 30.0, 30.0]
    path = np.array(robot.path)
    assert np.all(path >= ex.WORKSPACE_MIN - 1e-9) and np.all(path <= ex.WORKSPACE_MAX + 1e-9)
    assert path[:, 2].min() >= ex.TABLE_Z_FLOOR
