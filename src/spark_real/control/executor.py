"""ScoreExecutorCore: init, config/tuning, workspace bounds and task state.

Composes AbortMixin (executor_abort), BtRunnerMixin (bt_runner) and
DispatchMixin (bt_dispatch).
"""


from PIL import Image
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import cv2
import json
import logging
import math
import numpy as np
import os
import re
import threading
import time


from spark_core.util import as_bool, parse_json_reply
from spark_real.config import load_table_plane
from spark_real.config import family_block
from spark_real.control import common as primitive_timeouts
from spark_real.control import recovery as execution_recovery
from spark_real.control.bt import BtRunnerMixin
from spark_real.control.bt import DispatchMixin
from spark_real.control.common import AbortMixin
from spark_real.control.common import AbortRequested
from spark_real.control.common import AbortRequested, ExecutionResult
from spark_real.control.common import ExecutionResult
from spark_real.control.common import FRANKA_FAMILIES, apply_franka_bounds
from spark_real.control.common import TraceWriter
from spark_real.control.grasp import GraspMixin
from spark_real.control.motion import MotionMixin
from spark_real.control.place import ReleaseMixin
from spark_real.control.predicates import FAIL, PASS, UNVERIFIED, VerifyOutcome
from spark_real.control.success_verifier import SuccessVerifier, VerifyConfig, scene_views
from spark_real.control.trajectory import CartesianServo
from spark_real.control.verify_replay import ReplayMixin
from spark_real.control.verify_replay import _KEYPOINT_ACTIONS, RETRY_LIFT_M, RETRY_MIN_CLEAR_Z, _detection_fingerprint, _fail_has_evidence, _is_fresh, _keypoint_label, _replay_labels, llm_verify_enabled
from spark_real.grasp_depth_memory import GraspCalibration


logger = logging.getLogger(__name__)

# Statements that can only ADD speed. A script containing any of them is not a
# brake, whatever else it contains.
_ACCELERATING_STATEMENTS = ("movej(", "movel(", "movep(", "movec(", "servoj(", "speedj(")
_SPEEDL_VEC_RE = re.compile(r"speedl\(\[([^\]]*)\]")


def is_braking_script(script: str) -> bool:
    """True if every statement in ``script`` can only reduce arm speed.

    The abort-aware _send_script wrapper lets the brake through after the
    abort flag is set. A prefix test on "stopj"/"stopl" is too narrow: the
    Cartesian servo brakes with ``speedl([0,...])`` followed by ``stopl(1.0)``.
    """
    s = script.strip()
    if not s or not ("stopj(" in s or "stopl(" in s):
        return False
    if any(tok in s for tok in _ACCELERATING_STATEMENTS):
        return False
    for m in _SPEEDL_VEC_RE.finditer(s):
        try:
            vec = [float(x) for x in m.group(1).split(",")]
        except ValueError:
            return False  # unparseable speed: assume it accelerates
        if any(abs(v) > 1e-9 for v in vec):
            return False
    return True


class ScoreExecutorCore(AbortMixin, BtRunnerMixin, DispatchMixin):
    """Init, orchestration loop, dispatch, abort, and workspace logic.

    Base mixin for ScoreExecutor: __init__, the execute_score loop, action
    dispatch, abort machinery, and workspace bounds.
    """

    SAFE_HEIGHT_Z = 0.35
    APPROACH_OFFSET_Z = 0.0
    GRIPPER_OPEN_Z_OFFSET = -0.005
    # Same grasp as [2.103, -2.329, 0.059], expressed in the half-turn nearest
    # HOME_CONFIG (a parallel jaw is symmetric under 180 deg about the tool
    # axis); that value sat -177.8 deg in yaw from home and wrapped the
    # wrist-camera cable. Must match configs/ur10e_default.yaml.
    GRASP_ORIENTATION = [2.3038, 2.0802, -0.0048]
    WRIST_CAM_OFFSET_DEG = 81.0
    GRIPPER_EMPTY_THRESHOLD = 235
    GRIPPER_FULLY_CLOSED = 250
    # Release-side jaw thresholds (0=open .. 255=closed). GRIPPER_FULLY_CLOSED
    # answers "are the jaws still spread on something" AFTER a close; it says
    # nothing after an OPEN, where a released gripper reads ~0. These two are
    # the open-side pair: jaws at/near the open stop, or enough travel toward
    # open to prove the object let go. Bounds, not calibrated values; the
    # release witness logs the measured numbers so a rig run can tighten them.
    GRIPPER_RELEASED_MAX_POS = 60
    GRIPPER_RELEASE_MIN_TRAVEL = 40
    # Extra window the release witness may poll for the jaws to read open,
    # AFTER the driver's own open settle. Zero cost on a normal release (the
    # first sample already reads open).
    RELEASE_CONFIRM_TIMEOUT_S = 0.6
    # Brake decel for a URScript-capable driver that has no brake() of its own
    # (rad/s^2). UR10eDriver sizes its own from the commanded velocity; this is
    # only the last-resort literal.
    ABORT_DECEL_FALLBACK = 2.0
    MAX_GRASP_RETRIES = 3
    TABLE_Z_FLOOR = -0.276
    WORKSPACE_MIN = np.array([-1.1, -0.5, -0.27])
    WORKSPACE_MAX = np.array([-0.5, 0.7, 0.50])
    MAX_REACH = 1.15
    RECOVERY_Z_BIAS = [0.010, 0.020, 0.030]

    # Grasp / stack tuning. Overridable per-family via the `grasp:` and `stack:`
    # YAML blocks (see _apply_family_tuning); a family that omits them keeps
    # these defaults. Env vars remain as optional debug overrides.
    GRASP_DEPTH_M = 0.022            # descend this far below the perceived top
                                     # before clamping (TCP open->closed offset).
    GRASP_FORCE_EMPTY_N = 2.0        # force-verify: peak |dF| below this = empty.
                                     # gObj still leads.
    GRASP_LIFT_CHECK_M = 0.02        # confirming lift after clamp.
    GRASP_LIFT_DELTA_MIN_N = 0.5     # extra pull the confirming lift must add
                                     # for it to corroborate "holding".
    GRASP_YAW_AR_GATE = 1.8          # only do an oriented (yaw) grasp when the
                                     # object's aspect ratio exceeds this. Cubes
                                     # read ~1.1-1.5 from mask noise; real tools
                                     # ~2+ (1.4 fired 85 deg yaws on cubes).
    # Input-validity floors on the SAME oriented-grasp decision. AR cannot tell
    # a tool from a soft blob: the plushie read AR 3.14 at conf 0.24.
    GRASP_YAW_MIN_CONF = 0.45        # detection confidence floor for an OBB yaw.
    GRASP_YAW_MIN_OBB_CONF = 0.40    # OBB-axis confidence floor. Absent = unknown.
    STACK_CLEARANCE_M = 0.12         # begin place descent this far above target top.
    STACK_MAX_DESCENT_M = 0.15       # descent safety cap.
    STACK_CONTACT_FORCE_N = 5.0      # force-DELTA that counts as contact.
    STACK_HARD_STOP_DELTA_N = 25.0   # force-DELTA emergency stop.
    STACK_STEP_M = 0.004             # slow final-approach step size.
    STACK_APPROACH_GAP_M = 0.02      # fast approach stops this far above contact.
    STACK_DESCENT_VEL_FRAC = 0.10    # slow final-approach velocity fraction.

    def __init__(
        self,
        robot,
        detection_map=None,
        velocity=0.25,
        grasp_calibration: "GraspCalibration" = None,
        safe_height=None,
        pipeline=None,
        strict_placement_verify: bool = False,
    ):
        self.robot = robot
        self.detection_map = detection_map or {}
        try:
            from spark_real.control.verify_replay import remember_targets
            remember_targets(self)
        except Exception:  # noqa: BLE001
            pass
        self.velocity = velocity
        self.safe_height = safe_height or self.SAFE_HEIGHT_Z
        self._results = []
        self._abort = False
        # Abort epochs. _abort_epoch is bumped by every abort request;
        # _armed_epoch is stamped at the TASK boundary (reset_task_state).
        # execute_score clears the flag only when the two agree, so a stop
        # pressed during capture -> detect -> plan survives into the run.
        self._abort_epoch = 0
        self._armed_epoch = 0
        # Serialises bump-and-read against retract-if-unchanged, so the
        # watchdog cannot retract an operator stop that landed between its
        # own request and its retraction.
        self._abort_lock = threading.Lock()
        self._running = False
        self._wrist_servo_enabled = True
        # Wrist-camera refinement during approach/place/search; off by
        # default. Resolved from control.wrist_refine below.
        self._wrist_refine_enabled = False
        self._pipeline = pipeline
        self._recorder = None
        self._servo = CartesianServo(robot, rate_hz=125.0)
        # Everything scoped to ONE task lives in reset_task_state.
        self.reset_task_state()
        self._strict_placement_verify = strict_placement_verify
        self.grasp_calibration = grasp_calibration
        self._install_abort_aware_send_script()
        self._apply_family_workspace_bounds()
        self._resolve_wrist_refine_flag()
        self._apply_family_tuning()
        self._load_primitive_timeouts()

    def _apply_family_tuning(self):
        """Override grasp/stack tuning constants from the `grasp:`/`stack:` YAML
        blocks (profile.raw preferred, else <family>_default.yaml). Any
        missing key keeps the class default.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        if cfg is None:
            return
        family = (getattr(cfg, "robot_family", "ur10e") or "ur10e").lower()
        try:
            profile = getattr(self._pipeline, "profile", None)
            grasp = family_block(profile, family, "grasp")
            stack = family_block(profile, family, "stack")
            mapping = [
                (grasp, "depth_m", "GRASP_DEPTH_M"),
                (grasp, "force_empty_n", "GRASP_FORCE_EMPTY_N"),
                (grasp, "lift_check_m", "GRASP_LIFT_CHECK_M"),
                (grasp, "lift_delta_min_n", "GRASP_LIFT_DELTA_MIN_N"),
                (grasp, "yaw_aspect_ratio_gate", "GRASP_YAW_AR_GATE"),
                (grasp, "yaw_min_conf", "GRASP_YAW_MIN_CONF"),
                (grasp, "yaw_min_obb_conf", "GRASP_YAW_MIN_OBB_CONF"),
                (grasp, "released_max_pos", "GRIPPER_RELEASED_MAX_POS"),
                (grasp, "release_min_travel", "GRIPPER_RELEASE_MIN_TRAVEL"),
                (grasp, "release_confirm_timeout_s", "RELEASE_CONFIRM_TIMEOUT_S"),
                (grasp, "max_yaw_offset_deg", "_MAX_YAW_OFFSET_DEG"),
                (grasp, "empty_close_width_m", "EMPTY_CLOSE_WIDTH_M"),
                (grasp, "soft_hold_min_width_m", "SOFT_HOLD_MIN_WIDTH_M"),
                (grasp, "soft_squeeze_max_force", "SOFT_SQUEEZE_MAX_FORCE"),
                (grasp, "soft_squeeze_slack_m", "SOFT_SQUEEZE_SLACK_M"),
                (grasp, "empty_pos", "GRIPPER_EMPTY_THRESHOLD"),
                (grasp, "fully_closed_pos", "GRIPPER_FULLY_CLOSED"),
                (stack, "clearance_m", "STACK_CLEARANCE_M"),
                (stack, "max_descent_m", "STACK_MAX_DESCENT_M"),
                (stack, "contact_force_n", "STACK_CONTACT_FORCE_N"),
                (stack, "hard_stop_delta_n", "STACK_HARD_STOP_DELTA_N"),
                (stack, "step_m", "STACK_STEP_M"),
                (stack, "approach_gap_m", "STACK_APPROACH_GAP_M"),
                (stack, "descent_vel_frac", "STACK_DESCENT_VEL_FRAC"),
            ]
            applied = {}
            for block, key, attr in mapping:
                if isinstance(block, dict) and block.get(key) is not None:
                    setattr(self, attr, float(block[key]))
                    applied[attr] = getattr(self, attr)
            if applied:
                logger.info(
                    "ScoreExecutor grasp/stack tuning from config (%s): %s",
                    family, applied,
                )
            # Env override (quick experiment): SPARK_YAW_AR_GATE sets the oriented
            # (OBB-yaw) grasp gate; set it very high to force a plain top-down
            # grasp on objects whose aspect ratio would otherwise trip the yaw.
            _g = os.environ.get("SPARK_YAW_AR_GATE")
            if _g is not None:
                self.GRASP_YAW_AR_GATE = float(_g)
                logger.info(
                    "GRASP_YAW_AR_GATE=%.2f (SPARK_YAW_AR_GATE override)",
                    self.GRASP_YAW_AR_GATE,
                )
            # Cable limit for the oriented (OBB-yaw) grasp, in degrees.
            _y = os.environ.get("SPARK_MAX_YAW_OFFSET_DEG") or getattr(
                self, "_MAX_YAW_OFFSET_DEG", None
            )
            if _y is not None:
                self.MAX_YAW_OFFSET = math.radians(float(_y))
                logger.info(
                    "MAX_YAW_OFFSET=%.0f deg (oriented-grasp wrist-3 cable limit)",
                    float(_y),
                )
        except Exception as exc:
            logger.warning(
                "Could not load grasp/stack tuning: %s; keeping defaults", exc
            )

    def _resolve_wrist_refine_flag(self):
        """Read control.wrist_refine from the profile (or family YAML).

        Mirrors how _apply_family_workspace_bounds sources its values;
        defaults to False on any missing key or read error.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        if cfg is None:
            return
        family = (getattr(cfg, "robot_family", "ur10e") or "ur10e").lower()
        try:
            profile = getattr(self._pipeline, "profile", None)
            control = family_block(profile, family, "control")
            if "wrist_refine" in control:
                self._wrist_refine_enabled = bool(control["wrist_refine"])
                logger.info(
                    "ScoreExecutor wrist_refine=%s (control.wrist_refine)",
                    self._wrist_refine_enabled,
                )
        except Exception as exc:
            logger.warning(
                "Could not load control.wrist_refine: %s; "
                "keeping wrist refine disabled",
                exc,
            )
        # Env override: SPARK_WRIST_REFINE=1 forces wrist refinement on.
        _env = os.environ.get("SPARK_WRIST_REFINE")
        if _env is not None:
            self._wrist_refine_enabled = _env not in ("0", "false", "False", "")
            logger.info(
                "ScoreExecutor wrist_refine=%s (SPARK_WRIST_REFINE override)",
                self._wrist_refine_enabled,
            )

    def _load_primitive_timeouts(self):
        """Resolve the per-primitive wall-clock budgets.

        Starts from the defaults and overlays any timeouts: block in the
        resolved RobotProfile (preferred) or family YAML. Re-run from
        pipeline_init once _profile is wired so the profile override wins.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        family = (
            (getattr(cfg, "robot_family", None) or "").lower() if cfg is not None
            else ""
        )
        profile = getattr(self, "_profile", None)
        try:
            self._primitive_timeouts = primitive_timeouts.load_timeouts(
                profile=profile, family=family or None
            )
            logger.info(
                "ScoreExecutor primitive timeouts loaded (%d entries, default=%.1fs)",
                len(self._primitive_timeouts),
                self._primitive_timeouts.get(
                    "default", primitive_timeouts.DEFAULT_TIMEOUT_S
                ),
            )
        except Exception as exc:
            logger.warning(
                "Could not load primitive timeouts (%s); using paper defaults", exc
            )
            self._primitive_timeouts = primitive_timeouts.default_timeouts()
            self._primitive_timeouts["default"] = primitive_timeouts.DEFAULT_TIMEOUT_S

    def _apply_family_workspace_bounds(self):
        """Replace UR-default workspace bounds from the resolved profile.

        Sources configs/<family>_default.yaml plus any per-machine overlay.
        Falls back to class defaults if no pipeline/config is present, and
        re-reads the family YAML directly when the profile is missing/empty.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        if cfg is None:
            return
        family = (getattr(cfg, "robot_family", "ur10e") or "ur10e").lower()
        if family == "ur10e":
            # UR10e keeps its box, reach, table floor, and grasp rotvec in
            # the family YAML control: block.
            self._apply_ur10e_control_bounds()
        elif family in FRANKA_FAMILIES:
            apply_franka_bounds(self, family)

    def _apply_ur10e_control_bounds(self):
        """Load the UR10e executor box from the control: block.

        Sources the resolved profile (or ur10e_default.yaml directly);
        every key defaults to the class constant. Reads control.* (the
        executor/SafeRobot box), not the workspace: block.
        """
        try:
            control = family_block(getattr(self, "_profile", None), "ur10e", "control")
            if not control:
                return
            grasp = control.get("grasp_orientation")
            if grasp is not None:
                self.GRASP_ORIENTATION = [float(v) for v in grasp]
            ws_min = control.get("workspace_min")
            if ws_min is not None:
                self.WORKSPACE_MIN = np.array([float(v) for v in ws_min], dtype=float)
            ws_max = control.get("workspace_max")
            if ws_max is not None:
                self.WORKSPACE_MAX = np.array([float(v) for v in ws_max], dtype=float)
            if control.get("max_reach") is not None:
                self.MAX_REACH = float(control["max_reach"])
            # Table floor precedence: calibration file > YAML control.table_z_floor
            # > class default. The YAML value / class const stay as the FALLBACK.
            if control.get("table_z_floor") is not None:
                self.TABLE_Z_FLOOR = float(control["table_z_floor"])
                floor_src = "config: %.4f" % self.TABLE_Z_FLOOR
            else:
                floor_src = "class default: %.4f" % self.TABLE_Z_FLOOR
            cal = load_table_plane("ur10e")
            if cal is not None:
                surface_z = float(cal["surface_z"])
                # Margin is added so the floor sits a few mm ABOVE (less negative
                # than) the measured surface. Prefer the cal file's own margin.
                margin = float(
                    cal.get(
                        "safety_margin_m",
                        control.get("table_safety_margin_m", 0.003),
                    )
                )
                cal_floor = surface_z + margin
                # Sanity clamp: a cal recorded with a bad TCP reading must not
                # LOWER the safety floor below the config/class fallback by more
                # than 2 cm (that is how a bumped calibration drives the gripper
                # into the table). Raising the floor is always accepted.
                if cal_floor < self.TABLE_Z_FLOOR - 0.02:
                    logger.warning(
                        "ScoreExecutor REJECTING table calibration floor %.4f: "
                        "more than 2cm below fallback %.4f (suspect bad cal); "
                        "keeping fallback",
                        cal_floor,
                        self.TABLE_Z_FLOOR,
                    )
                else:
                    self.TABLE_Z_FLOOR = cal_floor
                    floor_src = "calibration: surface=%.4f +%dmm -> %.4f" % (
                        surface_z,
                        round(margin * 1000),
                        self.TABLE_Z_FLOOR,
                    )
            logger.info("ScoreExecutor table floor from %s", floor_src)
            logger.info(
                "ScoreExecutor UR10e control bounds from config: "
                "X=[%.2f,%.2f] Y=[%.2f,%.2f] Z=[%.2f,%.2f] reach=%.2f "
                "table_z_floor=%.3f grasp_rotvec=%s",
                self.WORKSPACE_MIN[0],
                self.WORKSPACE_MAX[0],
                self.WORKSPACE_MIN[1],
                self.WORKSPACE_MAX[1],
                self.WORKSPACE_MIN[2],
                self.WORKSPACE_MAX[2],
                self.MAX_REACH,
                self.TABLE_Z_FLOOR,
                self.GRASP_ORIENTATION,
            )
        except Exception as exc:
            logger.warning(
                "Could not load UR10e control bounds: %s; " "keeping class defaults",
                exc,
            )

    def _install_abort_aware_send_script(self):
        """
        Wrap robot._send_script (UR-only) to respect the abort flag.
        """
        executor_ref = self

        def make_wrapper(orig):
            def abort_aware_send(script):
                if (
                    executor_ref._running
                    and executor_ref._abort
                    and not is_braking_script(script)
                ):
                    raise AbortRequested()
                return orig(script)

            abort_aware_send._spark_abort_wrapped = True
            return abort_aware_send

        target = self.robot
        seen = set()
        while target is not None and id(target) not in seen:
            seen.add(id(target))
            if hasattr(target, "_send_script"):
                original_send = target._send_script
                if not getattr(original_send, "_spark_abort_wrapped", False):
                    try:
                        target._send_script = make_wrapper(original_send)
                    except AttributeError:
                        pass
            target = getattr(target, "_robot", None)

    def update_detections(self, detection_map):
        self.detection_map = detection_map
        # Snapshot the approved scene for the end-of-task spatial rung here,
        # not at verify time: mid-run re-detections mutate detection_map in
        # place.
        try:
            from spark_real.control.verify_replay import remember_targets
            remember_targets(self)
        except Exception:  # noqa: BLE001 - memory is an aid, never a blocker
            pass

    def _join_redetect(self, timeout: float = 5.0):
        """Bounded wait for a live post-release re-detect thread.

        Called before the executor starts any new detect so two detection
        passes never run concurrently against the stateful SAM3 predictor
        or mutate detection_map from two threads. Bounded so a wedged detect
        cannot hang a run.
        """
        thread = getattr(self, "_redetect_thread", None)
        if thread is None or not thread.is_alive():
            return
        try:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    "_join_redetect: re-detect thread still running after "
                    "%.1fs; proceeding (perception lock serializes SAM3)",
                    timeout,
                )
            else:
                self._redetect_thread = None
        except Exception as exc:
            logger.warning("_join_redetect failed: %s", exc)

    # Fields whose lifetime is ONE task. Attribute -> factory for a fresh value.
    # Anything a run writes and a later run reads belongs here.
    _TASK_SCOPED_STATE = {
        "_holding": bool,
        "_placed_labels": set,
        "_last_pick_label": str,
        "_last_place_label": str,
        "_last_keypoint_label": lambda: None,
        "_last_action_failed": bool,
        "_grasp_retry_count": dict,
        "_destination_labels": set,
        # Depth memory written by _approach_target and read by the grasp's
        # calibration record.
        "_grasp_perception_target_z": lambda: None,
        # Actual TCP z at the moment the jaws closed (grasp v2 / recovery
        # re-grasp). Read by the release give-back. Reset per pick in
        # _approach_target, per task here.
        "_actual_grasp_tcp_z": lambda: None,
        "_last_grasp_target_width": lambda: None,
        "_active_grasp_label": lambda: None,
        "_active_grasp_orient": lambda: None,
        "_active_grasp_strategy": lambda: None,
        # Release look-ahead written per-action by _run_actions/_run_branch
        # and read by the place transport's deep-dip side choice
        # (ReleaseMixin._dip_side_orient).
        "_pending_release_tilt": lambda: None,
    }

    def reset_task_state(self):
        """Drop every field scoped to ONE task.

        The executor is a server-lifetime singleton: one instance runs every
        task. Called from __init__ and from success_verifier.reset_run_state
        (the task boundary). NOT from execute_score: the closed loop calls
        that once per pass and _placed_labels must survive the passes of one
        task.
        """
        for attr, factory in self._TASK_SCOPED_STATE.items():
            setattr(self, attr, factory())
        # The task boundary is the ONE place an abort is legitimately
        # forgotten. Arming here lets execute_score distinguish a stale flag
        # from a stop pressed while this task was being planned.
        self._armed_epoch = getattr(self, "_abort_epoch", 0)
        self._abort = False
        # A post-release re-detection thread mutates detection_map in place,
        # so an unjoined one would rewrite the NEXT task's detections. Daemon
        # threads cannot be killed; wait briefly and say so if still running.
        thread = getattr(self, "_redetect_thread", None)
        if thread is not None:
            try:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(
                        "reset_task_state: re-detect thread from the previous "
                        "task is still running; it may overwrite detections"
                    )
            except Exception as exc:
                logger.warning("reset_task_state: re-detect join failed: %s", exc)
            self._redetect_thread = None


    def _check_workspace(self, pos):
        if np.linalg.norm(pos) > self.MAX_REACH:
            return False
        return bool(
            np.all(pos >= self.WORKSPACE_MIN) and np.all(pos <= self.WORKSPACE_MAX)
        )

    def _get_current_position(self):
        if hasattr(self.robot, "get_tcp_pose"):
            pose = self.robot.get_tcp_pose()
            if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                return pose[:3, 3].copy()
            return np.array(pose[:3])
        return np.zeros(3)

    _LEGATO_BLENDABLE = {"move_relative", "move_to_keypoint"}

    _LEAF_TYPES = {
        "move_to_keypoint",
        "grasp",
        "release",
        "move_relative",
        "open_drawer",
        "push_object",
        "turn_knob",
        "wait",
        "pour",
    }


    # Grammar slot aliases accepted everywhere the canonical name is expected.
    # The canonical name wins if both are present. Applied once at the dispatch
    # choke point so both builtin and skill primitives see normalized params.
    _PARAM_ALIASES = {
        "label": "keypoint_label",
        "target": "target_label",
    }


    _last_safety_reconcile_ts = 0.0


try:
    from spark_real.control.ik import solve_ik_fr3, solve_ik_ur10e, solve_ik_ur10e_rtde
except ImportError:
    solve_ik_ur10e = None
    solve_ik_ur10e_rtde = None
    solve_ik_fr3 = None

# IK seed when the joint read fails (not the ready pose; J5 off the
# wrist singularity).
_FR3_IK_SEED = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

from spark_real.control.common import (  # noqa: F401
    _LEGACY,
    _LEGACY_LIMITS_ENV,
    _SPEED_FRACTION_ENV,
    JOINT_ACC_CAP_RAD_S2,
    JOINT_VEL_FLOOR_RAD_S,
    JOINT_VEL_MARGIN,
    JOINT_VEL_REF_LINEAR_M_S,
    LINEAR_ACC_CAP_M_S2,
    LINEAR_VEL_CAP_M_S,
    UR10E_JOINT_VEL_RATED_RAD_S,
    joint_speed_fraction,
    leading_axis_vel_cap,
    legacy_joint_limits,
    resolve_joint_limits,
    resolve_linear_limits,
)
from spark_real.control.trajectory import PhraseMixin  # noqa: F401



class IkMixin(PhraseMixin):
    """
    IK-backed joint-space motion primitives mixed into ScoreExecutor.
    """

    def _robot_family(self):
        """Resolve the robot family off the driver."""
        fam = getattr(self.robot, "robot_family", None) or getattr(
            getattr(self.robot, "_robot", None), "robot_family", None
        )
        return str(fam or "").lower()

    def _movej_via_ik(self, position, orientation, velocity):
        """
        Solve IK in-process (control.ik), then movej over _send_script.

        Raises when the IK path is unavailable (SPARK_UR_PYROKI_IK=0, pyroki
        missing, non-UR family, no TCP offset) so the caller falls back to
        _movej_to_pose (firmware IK). The single-arm Franka solves with
        control.ik and moves through the driver's move_to_joint_config.
        """
        self._check_abort()
        family = self._robot_family()
        pos = np.array(position, dtype=float)
        orient = list(orientation)

        if family == "franka":
            return self._movej_via_fr3_ik(pos, orient, velocity)
        if family != "ur10e":
            raise RuntimeError(f"joint-IK path is UR10e-only (family={family!r})")
        if os.environ.get("SPARK_UR_PYROKI_IK", "1") != "1":
            raise RuntimeError("SPARK_UR_PYROKI_IK disabled; using legacy path")
        if solve_ik_ur10e_rtde is None:
            raise RuntimeError("UR10e pyroki IK not available")
        # The executor targets are RTDE-frame gripper-TIP poses; pyroki
        # solves URDF-frame tool0. solve_ik_rtde bridges the Rz(180) base
        # rotation + the pendant TCP offset. Without the offset that bridge
        # is unsafe, so fall back to the (frame-safe) legacy path.
        ur_tcp_off = None
        for src in (self.robot, getattr(self.robot, "_robot", None)):
            if src is not None and hasattr(src, "get_tcp_offset"):
                try:
                    ur_tcp_off = np.asarray(src.get_tcp_offset(), dtype=float)
                    break
                except Exception as exc:
                    logger.warning("[movej_via_ik:ur10e] get_tcp_offset failed: %s", exc)
        if ur_tcp_off is None:
            raise RuntimeError("UR10e TCP offset unavailable; using legacy path")
        default_seed = np.array(
            getattr(self.robot, "HOME_CONFIG",
                    [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]),
            dtype=float,
        )

        try:
            q_current = np.array(self.robot.get_joint_positions(), dtype=float)
        except Exception:
            q_current = default_seed

        q_target = solve_ik_ur10e_rtde(
            pos, orient, q_seed=q_current, tcp_offset=ur_tcp_off
        )
        if q_target is None:
            raise RuntimeError(
                f"pyroki IK failed for pos={pos.round(3)} orient={np.array(orient).round(3)}"
            )

        dq = np.abs(q_target - q_current[: q_target.size])
        logger.info(
            "[movej_via_ik:%s] pos=(%.3f,%.3f,%.3f) dq_max=%.2frad j%d",
            family or "?",
            pos[0],
            pos[1],
            pos[2],
            dq.max(),
            dq.argmax(),
        )

        # velocity is Cartesian m/s but joint moves want joint velocity (rad/s).
        # resolve_joint_limits owns that mapping AND the envelope; dq is passed
        # so a wrist-led move gets the wrist rating rather than the base's.
        vel_j, acc_j = resolve_joint_limits(velocity, dq=dq)
        logger.info(
            "[movej_via_ik:%s] v=%.2f rad/s a=%.2f rad/s^2 (lead j%d, cap %.2f)",
            family or "?",
            vel_j,
            acc_j,
            int(dq.argmax()),
            leading_axis_vel_cap(dq),
        )

        # movej over the URScript primary channel (_send_script): rtde_c.moveJ
        # silently no-ops after any speedl/gripper _send_script, so route the
        # joint target through the conflict-free URScript socket.
        # Fire-and-forget, then block on _wait_for_motion.
        self.robot.move_to_joint_config_urscript(
            q_target.tolist(), velocity=vel_j, acceleration=acc_j
        )
        # Hand the wait the joint target it just commanded: that is the
        # exact arrival test. Without it the wait falls back to Cartesian
        # proximity plus a velocity heuristic and burns the full timeout on
        # any move short enough to finish inside the URScript start lag.
        self._wait_for_motion(pos, q_target=q_target)

    # Blended multi-waypoint motion (UR10e only)

    def _ur_tcp_offset(self):
        """Pendant TCP offset (gripper tip relative to tool0), or None.

        Cached at connect by the driver; solve_ik_rtde cannot bridge the
        RTDE-frame tip target to a URDF tool0 target without it.
        """
        for src in (self.robot, getattr(self.robot, "_robot", None)):
            if src is not None and hasattr(src, "get_tcp_offset"):
                try:
                    return np.asarray(src.get_tcp_offset(), dtype=float)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[blend] get_tcp_offset failed: %s", exc)
        return None


    def _movej_to_pose(self, pose, velocity):
        """
        Joint-space move to a Cartesian pose (legacy fallback).

        UR fast path: emit movej(p[...]) URScript. Non-UR drivers fall back
        to move_linear or CartesianServo PD.
        """
        self._check_abort()
        # No solved joint target on this path (the controller runs its own IK),
        # so dq is unknown and the cap collapses to the slowest joint's rating.
        vel_j, acc_j = resolve_joint_limits(velocity, pose_fallback=True)
        if getattr(self.robot, "SUPPORTS_URSCRIPT", False) and hasattr(
            self.robot, "_send_script"
        ):
            ps = f"p[{pose[0]},{pose[1]},{pose[2]}," f"{pose[3]},{pose[4]},{pose[5]}]"
            if os.environ.get("SPARK_UR_MOVEL", "1") == "1":
                # Straight-line movel via the PRIMARY interface (_send_script),
                # NOT rtde_c.moveL: the gripper/speedl _send_script calls stop
                # the rtde_c control script, so rtde_c.moveL no-ops after a grasp
                # (arm never moves -> stack guard bails at full xy_err). movel
                # over _send_script works script-stopped AND keeps the TCP on a
                # straight line, killing the joint-space "wrenching" of movej.
                # movel v/a are Cartesian (m/s, m/s^2).
                vel_l, acc_l = resolve_linear_limits(velocity)
                sent = self.robot._send_script(f"movel({ps}, a={acc_l}, v={vel_l})")
            else:
                sent = self.robot._send_script(f"movej({ps}, a={acc_j}, v={vel_j})")
            if not sent:
                # Dropped command = failed primitive, never a silent pass
                # while _wait_for_motion times out on an arm that was never
                # told to move.
                raise RuntimeError(
                    "URScript move dropped (channel down); primitive failed"
                )
            self._wait_for_motion(np.array(pose[:3]))
            return

        if hasattr(self.robot, "move_linear"):
            try:
                self.robot.move_linear(list(pose), velocity=velocity)
                return
            except Exception as exc:
                logger.warning("move_linear failed (%s); trying CartesianServo PD", exc)
        if self._servo is not None and hasattr(self._servo, "move_to_pose"):
            try:
                self._servo.move_to_pose(list(pose), velocity=velocity)
                self._wait_stationary_after_servo()
                return
            except Exception as exc:
                logger.warning(
                    "CartesianServo PD fallback failed (%s); " "trying move_to_pose",
                    exc,
                )

        if hasattr(self.robot, "move_to_pose"):
            self.robot.move_to_pose(pose, velocity=velocity, wait=False)
            self._wait_for_motion(np.array(pose[:3]))
        elif hasattr(self.robot, "move_linear"):
            try:
                self.robot.move_linear(list(pose), velocity=velocity, asynchronous=True)
            except TypeError:
                self.robot.move_linear(list(pose), velocity=velocity)
            self._wait_for_motion(np.array(pose[:3]))

    def _wait_stationary_after_servo(
        self, max_wait: float = 2.0, vel_tol: float = 2e-3
    ):
        """
        Block until FR3 has fully decelerated after velocity-mode motion.
        """
        raw = getattr(self.robot, "_robot", self.robot)
        if hasattr(raw, "_robot"):
            raw = raw._robot
        try:
            t0 = time.time()
            prev_ok = False
            while time.time() - t0 < max_wait:
                try:
                    qdot = np.asarray(raw.state.dq, dtype=float)
                except Exception:
                    time.sleep(0.20)
                    return
                if np.max(np.abs(qdot)) < vel_tol:
                    if prev_ok:
                        return
                    prev_ok = True
                else:
                    prev_ok = False
                time.sleep(0.01)
        except Exception:
            time.sleep(0.20)

    def _movej_via_fr3_ik(self, pos, orient, velocity):
        """FR3 joint-space move: pyroki IK for the hand TCP, then a franky JointMotion."""
        if solve_ik_fr3 is None:
            raise RuntimeError("FR3 pyroki IK not available")
        try:
            q_current = np.asarray(self.robot.get_joint_positions(), dtype=float)
        except Exception:
            q_current = np.array(_FR3_IK_SEED)
        q_target = solve_ik_fr3(pos, orient, q_seed=q_current)
        if q_target is None:
            raise RuntimeError(f"FR3 IK failed for pos={pos.round(3)}")
        dq = np.abs(np.asarray(q_target) - q_current[:7])
        # Cartesian m/s -> joint rad/s, the same mapping the UR path uses.
        vel_j, _acc = resolve_joint_limits(velocity, dq=dq)
        logger.info(
            "[movej_via_ik:franka] pos=(%.3f,%.3f,%.3f) dq_max=%.2frad v=%.2frad/s",
            pos[0], pos[1], pos[2], float(dq.max()), vel_j,
        )
        self._check_abort()
        self.robot.move_to_joint_config(np.asarray(q_target).tolist(), velocity=vel_j)



# Replaying the last pick cycle is physical motion WITH a gripper release, so
# it stays off unless the operator asks for it. Config: verification.retry_on_fail.
RETRY_ON_FAIL_DEFAULT = False


def _ask_planner_task_done(executor, score) -> Optional[Tuple[bool, str]]:
    """The planner's look at the camera images: (done, reason). None when it is off or fails."""
    if not llm_verify_enabled(executor):
        return None
    try:
        captures = executor._pipeline.capture()
        # The operator's words: a planner's rewording can add requirements nobody asked for.
        task = getattr(executor, "_task_instruction", None) or (score or {}).get("task")
        contents = [
            f"A robot arm was asked to: {task!r}. The arm has "
            "moved clear. Looking only at these camera images, did the task succeed? Reply "
            'with JSON only: {"success": true or false, "reason": "one sentence"}.'
        ]
        # The wrist view looks straight down at the place after the lift; images only, no depth needed.
        for cam, rgb in scene_views(captures):
            contents += [f"Camera '{cam}':", Image.fromarray(rgb)]
        if len(contents) == 1:
            return None
        reply = executor._pipeline._planner.generate(contents, tag="verify", json_only=True)
        verdict = parse_json_reply(reply)
    except Exception as exc:  # noqa: BLE001 - never let the rung crash verify
        logger.warning("[llm-verify] rung failed: %s", exc)
        return None
    done = verdict.get("success") is True
    logger.info("[llm-verify] %s: %s", "pass" if done else "no", verdict.get("reason"))
    return done, verdict.get("reason")


class VerifyMixin(ReplayMixin):
    """
    Post-task verification, recovery, re-detection, failure snapshots.
    """

    # Last VerifyOutcome produced by this executor. Consumers (pipeline_run,
    # routes, the episode recorder, the BT library) must read THIS, not
    # ExecutionResult.success; execution is not achievement.
    verify_outcome: Optional[VerifyOutcome] = None

    def _verify_settings(self) -> Tuple[bool, bool, int]:
        """``(enabled, retry_on_fail, max_attempts)`` from the verification block."""
        enabled = VerifyConfig.from_pipeline(self._pipeline).enabled
        raw = getattr(getattr(self._pipeline, "profile", None), "raw", None) or {}
        block = raw.get("verification") if hasattr(raw, "get") else None
        block = block if isinstance(block, dict) else {}
        retry = as_bool(block.get("retry_on_fail"), RETRY_ON_FAIL_DEFAULT)
        default_max = 1 if self._strict_placement_verify else 2
        try:
            max_attempts = int(block.get("retry_max_attempts", default_max))
        except (TypeError, ValueError):
            max_attempts = default_max
        return enabled, retry, max(0, max_attempts)


    def _unocclude_for_verify(self):
        """Lift to safe height before the verify capture ("zoom out").

        An arm hovering centimeters over the placement occludes exactly the
        region every camera rung needs to see. One lift, current orientation
        preserved (a wrist reorientation here would be pure risk), skipped
        when already high enough or when anything about the read/move fails:
        failure to lift must never fail the verify.
        """
        try:
            pos = self._get_current_position()
            safe_z = float(getattr(self, "SAFE_HEIGHT_Z", 0.35))
            if pos is None or float(pos[2]) >= safe_z - 0.02:
                return
            orient = None
            cur = getattr(self, "current_orientation", None)
            if callable(cur):
                orient = cur()
            if orient is None:
                orient = list(self.GRASP_ORIENTATION)
            logger.info(
                "[verify] lifting %.0fmm to unocclude the scene before the "
                "verify capture", (safe_z - float(pos[2])) * 1000,
            )
            self._move_to([float(pos[0]), float(pos[1]), safe_z], orient)
        except Exception as exc:  # noqa: BLE001 - see docstring
            logger.warning("[verify] unocclude lift failed: %s", exc)

    def _fold_in_spatial_rung(self, outcome, score):
        """Rung 2: the remembered-region check, run only when the predicate
        verifier abstained. A filled slot defeats re-detection and identical
        sockets defeat label matching, but geometry still has an answer: does
        the placed object's mask now sit where the approved target region
        was? A pass here upgrades the abstain (with the evidence in the
        reason); a fail downgrades it to FAIL only when the object was
        confidently re-detected somewhere else. PASS/FAIL from the predicates
        are left alone; this rung breaks ties, it does not outvote
        measurements.
        """
        if outcome.status != UNVERIFIED:
            return outcome
        try:
            from spark_real.control.verify_replay import spatial_rung

            v = spatial_rung(self, score)
        except Exception as exc:  # noqa: BLE001 - never let the rung crash verify
            logger.warning("[spatial-verify] rung failed: %s", exc)
            return outcome
        if not isinstance(v, dict) or v.get("status") in (None, "abstain"):
            if isinstance(v, dict):
                outcome.gates["spatial"] = True  # abstain is not a failure
            return outcome
        outcome.gates["spatial"] = v["status"] == "pass"
        joiner = "; " if outcome.reason else ""
        if v["status"] == "pass":
            outcome.status = PASS
            outcome.reason = (
                outcome.reason + joiner + "spatial rung: " + v["detail"]
            )
        else:
            outcome.status = FAIL
            outcome.reason = (
                outcome.reason + joiner + "spatial rung: " + v["detail"]
            )
        logger.info("[spatial-verify] folded into verify: status=%s", outcome.status)
        return outcome

    def _fold_in_llm_rung(self, outcome, score):
        """Last rung: when geometry has no verdict, ask the planner LLM to look.

        It can only turn "unverified" into a pass. A "no" stays unverified with the reason noted,
        because a verify FAIL can replay motion and a model's opinion should not move the arm.
        SPARK_VERIFY_LLM=0 turns this off.
        """
        verdict = _ask_planner_task_done(self, score) if outcome.status == UNVERIFIED else None
        if verdict is None:
            return outcome
        done, reason = verdict
        outcome.reason = f"{outcome.reason}; llm rung ({'pass' if done else 'no'}): {reason}"
        if done:
            outcome.status = PASS
        return outcome

    def _llm_sees_success(self, score) -> bool:
        """The planner's image check, as a yes/no. False when it is off, fails or says no."""
        verdict = _ask_planner_task_done(self, score)
        return bool(verdict and verdict[0])

    def _fold_in_rungs(self, outcome, score):
        outcome = self._fold_in_release_proprio(outcome)
        outcome = self._fold_in_spatial_rung(outcome, score)
        return self._fold_in_llm_rung(outcome, score)

    def _fold_in_release_proprio(self, outcome):
        """Rung 0: the arm's own release-time evidence, folded into the
        camera verdict. Proprioception here only ever removes a pass, never
        grants one:

        - jaws never opened -> FAIL outright. The object was not released;
          whatever the cameras matched, the task did not happen.
        - commanded tilt not achieved -> a PASS downgrades to UNVERIFIED. The
          motion the plan asked for was dropped, so a visually-plausible
          result must not bank a BT that never executed as written.
        - proprio pass with cameras abstaining stays abstained: jaw + force
          agreement is necessary evidence, not sufficient.
        """
        v = getattr(self, "_release_proprio", None)
        if not isinstance(v, dict) or "verdict" not in v:
            return outcome
        outcome.gates["proprio"] = v["verdict"] != "fail"
        if v["verdict"] != "fail":
            return outcome
        if v.get("jaws_opened") is False:
            outcome.status = FAIL
            outcome.reason = (
                (outcome.reason + "; " if outcome.reason else "")
                + "proprio: jaws never opened -- object was not released"
            )
        elif v.get("commanded_pitch_achieved") is False and outcome.status == PASS:
            outcome.status = UNVERIFIED
            outcome.reason = (
                (outcome.reason + "; " if outcome.reason else "")
                + "proprio: commanded tilt was not achieved "
                f"(orient_err={v.get('orient_err_rad')}); not banking this run"
            )
        logger.info("[proprio-verdict] folded into verify: status=%s (%s)",
                    outcome.status, outcome.reason)
        return outcome

    def _run_post_task_verification(self, score: dict, actions: list):
        enabled, retry_on_fail, max_attempts = self._verify_settings()
        if not enabled:
            # Kill switch: no verifier, no capture, no retry, no motion.
            self.verify_outcome = VerifyOutcome(
                status=UNVERIFIED, reason="verification disabled by config"
            )
            logger.info("Verification disabled by config; no verify, no retry")
            return

        self._unocclude_for_verify()
        outcome = self._fold_in_rungs(self._verify_task_completion(score, actions), score)

        if outcome.status == PASS:
            return
        if not retry_on_fail or max_attempts <= 0:
            logger.info(
                "Verification %s; retry is off (verification.retry_on_fail), "
                "no replay",
                outcome.status,
            )
            return
        if outcome.status != FAIL:
            # Abstain, not a failure. Replaying here would drop a held object
            # on no evidence at all.
            logger.info("Verification %s (no verdict); not replaying", outcome.status)
            return
        if not _fail_has_evidence(outcome):
            logger.info("Verification fail with no failing gate or vote (%s); not replaying",
                        outcome.reason)
            return
        # A camera vote can fail a placement that is fine (an object the overhead camera cannot
        # see in a deep bowl), and the replay then takes it back out. Ask the planner to look
        # before moving: if it sees the task done, nothing is replayed. The verdict stays "fail"
        # here; the pipeline's last-resort rung adjudicates it with its own confidence gate.
        if self._llm_sees_success(score):
            outcome.reason = f"{outcome.reason}; planner image check sees the task done: no replay"
            logger.info("Verification fail, but the planner's image check sees the task done; not replaying "
                        "(the last-resort rung decides the verdict)")
            return

        # Attribution consult (verification.attribution, default off): the
        # lowest-responsible-layer table decides whether a replay is even the
        # right repair. PLAN (semantic fail / retries exhausted) means stop
        # and surface to the operator, never an autonomous LLM replan.
        # PERCEPTION/EXECUTION both proceed into the replay, whose
        # preconditions (re-perceive + ready-state) implement the
        # re-detect+re-bind and local-retry repairs.
        if execution_recovery.attribution_enabled(self):
            from spark_core.attribution import Layer, attribute_failure

            actions_list = actions or []
            start = self._find_last_pick_cycle(actions_list)
            pick_label = (
                _keypoint_label(actions_list[start]) if start is not None else ""
            )
            layer = attribute_failure(
                execution_recovery.build_verify_result(self, pick_label),
                None,
                0,
                max_local_retries=max(1, max_attempts),
            )
            logger.info(
                "Attribution: task-level fail on '%s' attributed to %s",
                pick_label or "?", layer.value,
            )
            if layer == Layer.PLAN:
                logger.warning(
                    "Attribution: PLAN layer -> not replaying; surfacing the "
                    "failure to the operator"
                )
                return

        self._replay_last_pick_cycle(score, actions, max_attempts)


    def _snapshot_failure_state(self, action_type: str, params: dict, result) -> None:
        """
        Save per-camera JPEGs + manifest when an action goes red.
        """
        if self._pipeline is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path(self._pipeline.config.output_dir) / "failures" / f"{ts}_{action_type}"
        out_root.mkdir(parents=True, exist_ok=True)

        try:
            caps = self._pipeline.capture()
        except Exception:
            caps = {}
        cam_files = []
        for cam_name, cd in (caps or {}).items():
            rgb = (cd or {}).get("rgb")
            if rgb is None:
                continue
            jpg = out_root / f"{cam_name}.jpg"
            try:
                cv2.imwrite(str(jpg), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cam_files.append(str(jpg))
            except Exception:
                pass

        robot_state = {}
        try:
            if hasattr(self.robot, "get_tcp_pose"):
                p = self.robot.get_tcp_pose()
                if hasattr(p, "tolist"):
                    robot_state["tcp_pose"] = p.tolist()
        except Exception:
            pass
        try:
            if hasattr(self.robot, "get_joint_positions"):
                j = self.robot.get_joint_positions()
                if hasattr(j, "tolist"):
                    robot_state["joints"] = j.tolist()
        except Exception:
            pass

        manifest = {
            "timestamp": ts,
            "action": action_type,
            "params": params,
            "message": getattr(result, "message", ""),
            "duration": getattr(result, "duration", 0.0),
            "robot": robot_state,
            "cameras": cam_files,
        }
        (out_root / "failure.json").write_text(json.dumps(manifest, indent=2, default=str))
        logger.info("Failure snapshot saved to %s (%d cameras)", out_root, len(cam_files))

    def _attempt_recovery(self, action_type, params, failed_result, actions, action_index):
        return execution_recovery.attempt_recovery(
            self, action_type, params, failed_result, actions, action_index
        )

    def _verify_task_completion(self, score, actions):
        """Run the predicate verifier and record its outcome.

        Appends an ExecutionResult so existing UI plumbing still sees a
        "verify" row, but the authoritative value is ``self.verify_outcome``:
        the row's ``success`` collapses the tri-state and must not be used to
        decide task success.
        """
        t0 = time.time()
        verifier = SuccessVerifier(executor=self)
        outcome = verifier.verify(score, actions)
        self.verify_outcome = outcome
        try:
            TraceWriter.from_pipeline(self._pipeline).write_verify(outcome, time.time() - t0)
        except Exception as exc:
            logger.warning("Verify trace not written: %s", exc)
        self._results.append(
            ExecutionResult(
                action_type="verify",
                success=(outcome.status == PASS),
                message=f"{outcome.status}: {outcome.reason}",
                duration=time.time() - t0,
            )
        )
        return outcome

    def _redetect_all(self, actions, current_index):
        execution_recovery.redetect_all(self, actions, current_index)

    def _redetect_single(self, label):
        execution_recovery.redetect_single(self, label)


class ScoreExecutor(
    ScoreExecutorCore,
    MotionMixin,
    IkMixin,
    GraspMixin,
    ReleaseMixin,
    VerifyMixin,
):
    """
    Executes SPARK behavior tree scores against any shim-compliant robot.

    Designed for composition, not inheritance: pass any driver (or
    SafeRobot wrapping one) that implements the shared shim interface
    (the UR10eDriver in this release). UR-specific shortcuts are guarded by
    hasattr checks.
    """

    pass


