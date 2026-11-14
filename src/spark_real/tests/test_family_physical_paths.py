"""The physical-run behaviours on the Franka and bimanual executors, with fake drivers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_bimanual_executor_surface import _MockSafe  # noqa: E402
from test_selector_semantics import _FakePipeline, _MockRobot  # noqa: E402

from spark_real.control.bimanual_executor import BimanualScoreExecutor  # noqa: E402
from spark_real.control.recovery import rebind_before_approach, redetect_all, redetect_single  # noqa: E402
from spark_real.control.executor import ScoreExecutor  # noqa: E402

T_RIGHT_TO_LEFT = np.eye(4)
T_RIGHT_TO_LEFT[:3, 3] = [0.0, -0.8, 0.0]


class _Pipeline(_FakePipeline):
    cameras = ("birdview", "sideview", "wrist")

    def __init__(self, family="franka"):
        super().__init__()
        self.config.robot_family = family
        self.profile = SimpleNamespace(raw={})

    def capture(self):
        return {c: {"rgb": np.zeros((4, 4, 3), np.uint8)} for c in self.cameras}


class _BimanualPipeline(_Pipeline):
    cameras = ("external", "wrist_left", "wrist_right")

    def __init__(self):
        super().__init__("bimanual_franka")


def franka_executor():
    return ScoreExecutor(_MockRobot(), detection_map={}, pipeline=_Pipeline())


def bimanual_executor():
    return BimanualScoreExecutor(_MockSafe(), pipeline=_BimanualPipeline(), T_right_to_left=T_RIGHT_TO_LEFT)


def test_redetect_leaves_the_left_arm_map_in_its_own_frame():
    ex = bimanual_executor()
    ex.update_detections({"cup": {"position_3d": [0.4, 0.1, 0.05]}})
    ex._pipeline._detect_results = [SimpleNamespace(label="cup", position_3d=[0.45, 0.1, 0.05], confidence=0.9)]
    left, right = ex._arm_executors["left"], ex._arm_executors["right"]
    actions = [{"type": "move_to_keypoint", "params": {"keypoint_label": "cup"}}, {"type": "grasp"}]

    rebind_before_approach(left, "cup")
    redetect_single(left, "cup")
    redetect_all(left, [{}] + actions, 0)
    assert left.detection_map["cup"]["position_3d"] == pytest.approx([0.4, -0.7, 0.05])

    redetect_single(right, "cup")
    assert right.detection_map["cup"]["position_3d"] == pytest.approx([0.45, 0.1, 0.05])


CAMERA_SETS = {
    "ur10e": ("birdview", "sideview", "wrist"),
    "bimanual": ("external", "wrist_left", "wrist_right"),
}


def _captures(cameras, rgb=True):
    return {c: {"rgb": np.zeros((4, 4, 3), np.uint8) if rgb else None} for c in cameras}


def test_scene_views_names_the_configured_cameras():
    from spark_real.control.success_verifier import scene_views

    wrist_first = _captures(("wrist", "sideview", "birdview"))
    assert [c for c, _ in scene_views(wrist_first)] == ["birdview", "sideview", "wrist"]
    assert [c for c, _ in scene_views(_captures(CAMERA_SETS["bimanual"]))] == list(CAMERA_SETS["bimanual"])
    assert scene_views(_captures(CAMERA_SETS["ur10e"], rgb=False)) == []
    assert scene_views(None) == []


class _Planner:
    def __init__(self, complete=True):
        self.complete = complete
        self.asked = []

    def generate(self, contents, **kw):
        self.asked.append([c for c in contents if isinstance(c, str) and c.startswith("Camera")])
        return '{"success": true, "reason": "in the bowl"}'

    def assess_completion(self, instruction, images, evidence=""):
        self.asked.append([name for name, _ in images])
        return {"complete": self.complete, "confidence": 0.9, "reason": "in the bowl"}


@pytest.mark.parametrize("make, cameras", [
    (franka_executor, ["birdview", "sideview", "wrist"]),
    (lambda: bimanual_executor()._arm_executors["left"], ["external", "wrist_left", "wrist_right"]),
])
def test_planner_image_check_sees_every_family_camera(make, cameras, monkeypatch):
    monkeypatch.delenv("SPARK_VERIFY_LLM", raising=False)
    ex = make()
    ex._pipeline._planner = _Planner()
    assert ex._llm_sees_success({"task": "put the spoon in the bowl"})
    assert ex._pipeline._planner.asked == [[f"Camera '{c}':" for c in cameras]]


class _LastRung:
    """The pipeline surface _llm_last_resort uses."""

    from spark_real.pipeline_plan import ClosedLoopMixin as _Mixin

    _llm_last_resort = _Mixin._llm_last_resort

    def __init__(self, cameras, rgb=True, complete=True):
        import contextlib

        self.config = SimpleNamespace(llm_last_resort=True)
        self._planner = _Planner(complete)
        self._executor = SimpleNamespace(_destination_labels=set(), _release_proprio={},
                                         verify_outcome=SimpleNamespace(status="fail", reason=""))
        self.capture = lambda: _captures(cameras, rgb)
        self.detect = lambda captures, prompts, **kw: []
        self.merge_detections = lambda dets: dets
        self.activity = lambda name: contextlib.nullcontext()
        self.plans = []
        self.plan = lambda *a, **kw: self.plans.append(kw) or {"tree": {}}
        self.execute = lambda score, dets: []

    def run(self):
        outcome = SimpleNamespace(status="fail", reason="outside")
        return self._llm_last_resort("put the spoon in the bowl", ["spoon"], False, "fail", outcome, set(), [])


@pytest.mark.parametrize("family", CAMERA_SETS)
def test_last_rung_adjudicates_on_the_family_cameras(family):
    rung = _LastRung(CAMERA_SETS[family])
    success, status, _ = rung.run()
    assert (success, status) == (True, "pass")
    assert rung._planner.asked == [list(CAMERA_SETS[family])] and rung.plans == []


def test_last_rung_replan_gets_the_scene_cameras_only():
    rung = _LastRung(CAMERA_SETS["bimanual"], complete=False)
    rung.run()
    assert [name for name, _ in rung.plans[0]["scene_image"]] == ["external", "external (annotated keypoints)"]


def test_last_rung_never_passes_a_run_without_an_image():
    rung = _LastRung(CAMERA_SETS["bimanual"], rgb=False)
    success, status, _ = rung.run()
    assert (success, status) == (False, "fail")
    assert rung._planner.asked == [] and len(rung.plans) == 1


def test_second_look_marks_the_instances_on_the_family_scene_camera(monkeypatch):
    from spark_real import pipeline_plan
    from spark_real.pipeline_plan import PlanMixin

    monkeypatch.setattr(pipeline_plan, "annotate_for_planner", lambda rgb, dets: "marked")
    monkeypatch.setattr(pipeline_plan, "build_detection_details", lambda dets: [])
    spoons = [SimpleNamespace(label="spoon 1"), SimpleNamespace(label="spoon 2")]
    pipe = SimpleNamespace(capture=lambda: _captures(("wrist_left", "external")),
                           detect=lambda *a, **kw: spoons, merge_detections=lambda dets: dets)
    detections = [SimpleNamespace(label="spoon")]
    labels, _, image = PlanMixin._redetect_missing_instances(pipe, ["spoon 2"], detections, [("external", "raw")], None)
    assert labels == ["spoon 1", "spoon 2"]
    assert image == [("external", "raw"), ("external (annotated keypoints)", "marked")]


def ur10e_executor():
    robot = _MockRobot()
    robot.robot_family, robot.GRIPPER_TYPE = "ur10e", "robotiq_2f85"
    return ScoreExecutor(robot, detection_map={}, pipeline=_Pipeline("ur10e"))


@pytest.mark.parametrize("attr, ur10e, franka", [
    ("EMPTY_CLOSE_WIDTH_M", 0.011, 0.001),
    ("SOFT_HOLD_MIN_WIDTH_M", 0.015, 0.001),
    ("SOFT_SQUEEZE_MAX_FORCE", 80, 80),
    ("SOFT_SQUEEZE_SLACK_M", 0.005, 0.005),
    ("GRIPPER_EMPTY_THRESHOLD", 235, 235),
    ("GRIPPER_FULLY_CLOSED", 250, 250),
])
def test_gripper_numbers_come_from_the_family_config(attr, ur10e, franka):
    assert getattr(ScoreExecutor, attr) == ur10e
    assert getattr(ur10e_executor(), attr) == ur10e
    assert getattr(franka_executor(), attr) == franka


def test_machine_overlay_sets_a_gripper_number():
    pipe = _Pipeline("ur10e")
    pipe.profile.raw = {"grasp": {"empty_close_width_m": 0.004, "soft_squeeze_max_force": 50}}
    ex = ScoreExecutor(_MockRobot(), detection_map={}, pipeline=pipe)
    assert (ex.EMPTY_CLOSE_WIDTH_M, ex.SOFT_SQUEEZE_MAX_FORCE) == (0.004, 50)


def _width_check(ex, target, achieved):
    ex._last_grasp_target_width = target
    ex.robot.get_gripper_width = lambda: achieved
    ex._read_gripper_state = lambda: (None, None)
    ex.robot.get_tcp_force = lambda: (_ for _ in ()).throw(AssertionError("width alone decides"))
    return ex._verify_grasp()


def test_thin_hold_passes_the_width_check_on_the_franka_hand():
    assert _width_check(franka_executor(), target=0.010, achieved=0.006) is True
    assert _width_check(bimanual_executor()._arm_executors["right"], target=0.030, achieved=0.028) is True


def test_closed_on_nothing_is_not_a_width_hold_on_any_gripper():
    for ex in (ur10e_executor(), franka_executor(), bimanual_executor()._arm_executors["left"]):
        assert _width_check(ex, target=0.010, achieved=0.0005) is False


def _bowl_slots(monkeypatch, **fields):
    """Slots of a round bowl mask whose slot detector found no compartments."""
    import cv2

    from spark_real import perception_steps as pm

    monkeypatch.setattr(pm, "detect_slots", lambda **kw: [{"mode": "fallback"}, {"mode": "fallback"}])
    mask = np.zeros((120, 160), np.uint8)
    cv2.circle(mask, (80, 60), 40, 1, -1)
    bowl = SimpleNamespace(label="bowl", camera="external", mask=mask.astype(bool), orientation_angle=0.0, **fields)
    cal = SimpleNamespace(intrinsic_matrix=np.eye(3), extrinsic=np.eye(4))
    pipe = SimpleNamespace(SLOT_CONTAINER_LABELS={"bowl"}, _recentre_deep_container=lambda *a: None)
    captures = {"external": {"rgb": None, "depth": np.ones(mask.shape, np.float32), "calibration": cal}}
    pm.MergeMixin._enrich_with_slots(pipe, [bowl], captures)
    return getattr(bowl, "slots", None)


def test_round_bowl_gets_no_made_up_slots_without_the_fusion_gate(monkeypatch):
    assert _bowl_slots(monkeypatch) is None


def test_the_fusion_gate_reading_of_the_shape_is_kept(monkeypatch):
    assert _bowl_slots(monkeypatch, shape_round=True) is None
    assert len(_bowl_slots(monkeypatch, shape_round=False)) == 2


def test_task_words_and_tool_heads_reach_both_arms():
    ex = bimanual_executor()
    ex._task_instruction, ex._tool_heads = "put the spoon in the tray", {"spoon": "bowl end"}
    for arm in ex._arm_executors.values():
        assert arm._task_instruction == "put the spoon in the tray" and arm._tool_heads == {"spoon": "bowl end"}


def test_step_trace_names_the_arm(monkeypatch):
    from spark_real.control import common as primitive_trace

    written = []
    writer = SimpleNamespace(enabled=True, next_index=lambda: 0, write=written.append)
    monkeypatch.setattr(primitive_trace.TraceWriter, "from_pipeline", classmethod(lambda cls, pipeline: writer))
    result = SimpleNamespace(success=True, message="")
    bimanual_executor()._arm_executors["left"]._trace_step("wait", {}, result, 0.0)
    franka_executor()._trace_step("wait", {}, result, 0.0)
    assert written[0].extra["arm"] == "left" and "arm" not in written[1].extra


def test_missing_overlay_error_names_the_family_example():
    from spark_real import config

    for family, example in (("ur10e", "example"), ("franka", "franka_example"),
                            ("bimanual_franka", "bimanual_example")):
        with pytest.raises(FileNotFoundError, match=f"machines/{example}.yaml"):
            config.load_family_yaml(family, "no_such_host")
        assert (config.CONFIGS_DIR / "machines" / f"{example}.yaml").exists()


def test_bimanual_rebind_freshness_matches_the_single_arm_rigs():
    from spark_real.config import family_block

    ages = {f: family_block(None, f, "perception")["rebind"]["max_age_s"]
            for f in ("ur10e", "franka", "bimanual_franka")}
    assert set(ages.values()) == {0.5}


FAMILY_EXECUTORS = [franka_executor, lambda: bimanual_executor()._arm_executors["left"]]


@pytest.mark.parametrize("make", FAMILY_EXECUTORS)
def test_release_height_stays_above_the_rim_on_the_family_table(make):
    from spark_real.control.place import MIN_TCP_ABOVE_RIM_M, resolve_release_z

    ex = make()
    table = float(ex.TABLE_Z_FLOOR)
    bowl = {"rim_z_m": table + 0.05, "interior_z_m": table + 0.01, "height_samples": 500}
    z = ex._place_release_z(table + 0.30, bowl, {})
    want = resolve_release_z(table + 0.30, bowl, None, table_z=table, grasp_z_offset=ex.GRIPPER_OPEN_Z_OFFSET)
    assert want.source == "geometry" and z == pytest.approx(want.z)
    assert table + 0.05 + MIN_TCP_ABOVE_RIM_M <= z < table + 0.30
    assert ex._place_release_z(table + 0.30, {"position_3d": [0.4, 0.0, table]}, {}) == pytest.approx(table + 0.30)


@pytest.mark.parametrize("make, cameras", [
    (franka_executor, ("birdview", "sideview")),
    (lambda: bimanual_executor()._arm_executors["right"], ("external",)),
])
def test_inside_check_votes_on_the_family_cameras(make, cameras):
    from test_container_reference_frame import BOWL_MINOR, BOWL_XYZ, PLACE_ACTIONS, PLUSHIE_BESIDE_BOWL, det
    from test_verify_scope_verifier import CountingPipeline, plan, scene

    from spark_real.control import success_verifier as sv

    ex = make()
    ex.detection_map = plan()
    ex._pipeline = CountingPipeline(scene(cameras), cameras=cameras)
    outcome = sv.SuccessVerifier(executor=ex).verify({}, PLACE_ACTIONS)
    assert outcome.status == sv.PASS and ex._pipeline.detect_cameras == [sorted(cameras)]

    beside = [det("plushie", c, PLUSHIE_BESIDE_BOWL) for c in cameras]
    beside += [det("bowl", c, BOWL_XYZ, obb_minor_m=BOWL_MINOR, aspect_ratio=1.0) for c in cameras]
    ex._pipeline = CountingPipeline(beside, cameras=cameras)
    assert sv.SuccessVerifier(executor=ex).verify({}, PLACE_ACTIONS).status == sv.FAIL
