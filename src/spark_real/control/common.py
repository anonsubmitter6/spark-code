"""Shared types for the ScoreExecutor family."""


from __future__ import annotations


from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from typing import Any, Dict, List, Optional
from typing import Callable, Dict, Optional
from typing import Iterator, Optional, Sequence
from typing import Optional
import json
import logging
import math
import numpy as np
import os
import threading
import time


from spark_real.config import family_block


# Leaf primitive names the executor dispatches directly (not via the skill
# registry). The planner validates emitted leaf types against this same set.
# The `action` meta-type and structural node types (sequence, selector,
# parallel, sync_barrier) are added locally at each site.
BUILTIN_PRIMITIVES = frozenset(
    {
        "move_to_keypoint",
        "grasp",
        "release",
        "move_relative",
        "wait",
    }
)


class AbortRequested(Exception):
    """Raised when the operator requests an emergency stop mid-motion.

    Propagated out of every motion-issuing helper so that no new motion
    command can overwrite the stop that was just sent to the controller.
    """

    pass


@dataclass
class ExecutionResult:
    action_type: str
    success: bool
    message: str = ""
    duration: float = 0.0


logger = logging.getLogger(__name__)


class AbortMixin:
    """Operator abort: epochs, brake, and abort-aware sleeps."""

    def note_abort_requested(self) -> int:
        """
        Raise the abort flag, advance the epoch, and return the new epoch.

        Every stop path must come through here rather than assigning
        ``_abort`` directly, or the request is invisible to execute_score's
        arming check. The epoch goes up FIRST, so a reader that sees
        ``_abort`` True has already seen the epoch move.

        The returned epoch identifies THIS request, so the primitive-timeout
        watchdog can retract its own abort without stepping on an operator
        stop that arrived in the same window.
        """
        with self._abort_lock:
            self._abort_epoch += 1
            self._abort = True
            return self._abort_epoch

    def retract_abort(self, epoch: int) -> bool:
        """
        Withdraw abort request ``epoch``. False unless it is the ONLY
        outstanding one.

        For the primitive-timeout watchdog, whose abort is a transient it must
        undo so recovery can run. TWO conditions, both needed:

        * nothing NEWER has landed (``_abort_epoch == epoch``): an operator
          stop raised while the primitive unwound;
        * nothing OLDER is still outstanding (``epoch - 1 == _armed_epoch``):
          an operator stop raised BEFORE the watchdog fired.

        Together: this request is the only thing between the task's arming
        point and now, so undoing it restores exactly the pre-timeout state.
        """
        with self._abort_lock:
            if self._abort_epoch != epoch:
                logger.warning(
                    "retract_abort(%d) refused: a newer stop (epoch %d) is "
                    "standing; leaving the abort in place",
                    epoch,
                    self._abort_epoch,
                )
                return False
            if epoch - 1 != self._armed_epoch:
                logger.warning(
                    "retract_abort(%d) refused: an earlier stop is still "
                    "outstanding (armed at epoch %d); leaving the abort in "
                    "place",
                    epoch,
                    self._armed_epoch,
                )
                return False
            self._armed_epoch = epoch
            self._abort = False
            return True

    def abort(self, epoch: Optional[int] = None) -> bool:
        """Request a stop AND brake the arm. True if the brake reached the arm.

        Setting the flag only stops this process from issuing new commands.
        URScript motion is fire-and-forget, so an unbraked abort leaves the arm
        running to its target while the executor unwinds. Brake here, at the
        single entry point every caller uses (/api/stop, /api/abort, the
        primitive-timeout watchdog).

        ``epoch`` is for a caller that has ALREADY called
        note_abort_requested() and is holding its epoch so it can retract
        exactly that request later (the watchdog). Passing it skips the bump
        so one stop is one epoch.
        """
        if epoch is None:
            self.note_abort_requested()
        self._servo.abort()
        braked = self._stop_robot()
        if braked:
            logger.warning("Abort requested; arm braked")
        else:
            logger.critical(
                "Abort requested but THE ARM WAS NOT BRAKED -- it may still be "
                "moving. Use the physical E-stop."
            )
        return braked

    def _stop_robot(self) -> bool:
        """Brake the arm. True only if a stop actually reached the driver.

        Order matters. The first branch is the UR escalation (URScript stopj on
        the primary socket, then a reopened socket, then Dashboard); it is the
        only thing that stops a fire-and-forget movej. rtde_c.stopJ is a no-op
        against a dead control script, so it must not be the first and only try.
        """
        emergency = getattr(self.robot, "emergency_stop", None)
        if callable(emergency):
            try:
                result = emergency()
                if isinstance(result, dict) and result.get("braked"):
                    return True
                logger.error("emergency_stop did not reach the arm: %s", result)
            except Exception as exc:  # noqa: BLE001 - fall through to the rest
                logger.warning("emergency_stop raised: %s", exc)
        # Any other URScript-capable driver: emit the brake directly. The
        # abort-aware _send_script wrapper lets braking scripts through.
        if getattr(self.robot, "SUPPORTS_URSCRIPT", False):
            send = getattr(self.robot, "_send_script", None)
            if callable(send):
                try:
                    if bool(send("stopj(%.2f)" % self.ABORT_DECEL_FALLBACK)):
                        return True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("stopj send failed: %s", exc)
        # Drivers without a URScript channel: try each stop method in order,
        # and do not stop at the first one that merely fails to raise.
        braked = False
        for name in ("stop_motion", "stop", "servo_stop"):
            fn = getattr(self.robot, name, None)
            if callable(fn):
                try:
                    # A driver that reports failure (False) has not braked.
                    braked = fn() is not False or braked
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s failed: %s", name, exc)
        return braked

    def _check_abort(self):
        """
        Raise AbortRequested if stop has been requested.
        """
        if self._abort:
            self._stop_robot()
            raise AbortRequested()

    def _abort_sleep(self, duration: float, tick: float = 0.05):
        """
        Abort-aware sleep. Polls the abort flag every tick seconds.
        """
        t_end = time.time() + duration
        while True:
            self._check_abort()
            remaining = t_end - time.time()
            if remaining <= 0:
                return
            time.sleep(min(tick, remaining))



# Budgets in seconds. ``wipe`` is the planner-grammar alias for
# ``constrained_scrub`` (10s) and is listed so a lookup by either name works.
# DEFAULT_TIMEOUT_S is applied to any primitive not named here.
DEFAULT_TIMEOUT_S = 8.0

PAPER_TIMEOUTS_S: Dict[str, float] = {
    "move_to_keypoint": 4.0,
    "move_relative": 2.0,
    "grasp": 1.0,
    "release": 1.0,
    "insert": 6.0,
    "push_object": 4.0,
    "open_drawer": 5.0,
    "wipe": 10.0,
    "constrained_scrub": 10.0,
}


# Primitives whose cost is perception, not motion. Kept separate so
# PAPER_TIMEOUTS_S stays exactly the paper's table.
#
# verify_placed budget, from measured cost:
#   scoped cost   1 prompt x 2 contributing cameras         = 2 SAM3 passes
#   per pass      ~ 7.5s (30s / 4 passes on sideview)
#   capture       reused inside the freshness window, else  ~ 1s
#   worst case    2 x 7.5 + 1                               ~ 16s
#   budget        16s + margin                              = 25.0s
# Shrink this as raw SAM3 latency comes down.
NON_PAPER_TIMEOUTS_S: Dict[str, float] = {
    "verify_placed": 25.0,
    # Pure proprioception: a gripper register read. Anything approaching this
    # is a wedged RTDE link, not a slow check.
    "verify_grasp": 5.0,
}


def default_timeouts() -> Dict[str, float]:
    """Paper-spec per-primitive budgets plus the perception-cost primitives."""
    out = dict(PAPER_TIMEOUTS_S)
    out.update(NON_PAPER_TIMEOUTS_S)
    return out


def load_timeouts(profile=None, family: Optional[str] = None) -> Dict[str, float]:
    """
    Resolve per-primitive timeouts, paper defaults overlaid by YAML.

    Merges any ``timeouts:`` block from the resolved RobotProfile
    (``profile.raw``) or from ``configs/<family>_default.yaml``.

    A ``default:`` key inside the YAML block overrides DEFAULT_TIMEOUT_S for
    unlisted primitives and is stored under the special key ``"default"``.
    Any malformed entry is skipped with a warning; the paper value stands.
    """
    timeouts = default_timeouts()
    timeouts["default"] = DEFAULT_TIMEOUT_S

    raw_block = family_block(profile, family or "", "timeouts")
    if isinstance(raw_block, dict):
        for name, val in raw_block.items():
            try:
                timeouts[str(name)] = float(val)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring non-numeric timeout for '%s' (%r)", name, val
                )
    return timeouts


def timeout_for(timeouts: Dict[str, float], action_type: str) -> float:
    """
    Look up the budget for ``action_type``, falling back to the default.
    """
    if action_type in timeouts:
        return float(timeouts[action_type])
    return float(timeouts.get("default", DEFAULT_TIMEOUT_S))


def run_with_timeout(
    executor,
    action_type: str,
    budget_s: float,
    fn: Callable,
) -> "object":
    """
    Run ``fn`` under a wall-clock budget, aborting via the executor on expiry.

    A daemon watchdog timer fires after ``budget_s`` seconds and invokes
    ``executor.abort()`` (the same path /api/stop uses), which raises
    AbortRequested inside the motion helpers and stops the CartesianServo.

    Returns whatever ``fn`` returns within budget. On a timeout it returns a
    failed ExecutionResult and clears the watchdog's transient abort flag. An
    AbortRequested raised by ``fn`` itself (operator stop) is re-raised
    unchanged.
    """
    if budget_s is None or budget_s <= 0:
        return fn()

    fired = {"timeout": False, "epoch": None}

    def _on_expire():
        fired["timeout"] = True
        logger.warning(
            "[timeout] primitive '%s' exceeded %.1fs budget; aborting motion "
            "and routing to recovery",
            action_type,
            budget_s,
        )
        try:
            # Claim an epoch for THIS abort so the retraction below undoes
            # exactly this request. abort(epoch=...) skips its own bump.
            note = getattr(executor, "note_abort_requested", None)
            if callable(note):
                fired["epoch"] = note()
                executor.abort(epoch=fired["epoch"])
            else:
                executor.abort()
        except Exception as exc:
            logger.warning("[timeout] executor.abort() raised: %s", exc)

    def _retract_watchdog_abort() -> bool:
        """
        Undo the watchdog's own abort. False unless it is the ONLY
        outstanding request; an operator stop raised before or after the
        watchdog fired must survive (see retract_abort for the rule).
        """
        retract = getattr(executor, "retract_abort", None)
        if callable(retract) and fired["epoch"] is not None:
            return bool(retract(fired["epoch"]))
        executor._abort = False
        return True

    watchdog = threading.Timer(float(budget_s), _on_expire)
    watchdog.daemon = True
    t0 = time.time()
    watchdog.start()
    try:
        result = fn()
    except AbortRequested:
        watchdog.cancel()
        if fired["timeout"] and _retract_watchdog_abort():
            # The watchdog's abort, not the operator's: convert to a failed
            # post-condition. A refused retraction means a genuine stop is
            # standing and falls through to `raise`.
            return ExecutionResult(
                action_type=action_type,
                success=False,
                message=(
                    f"{action_type} timed out after {budget_s:.1f}s "
                    f"(budget exceeded)"
                ),
                duration=time.time() - t0,
            )
        raise
    finally:
        watchdog.cancel()

    if fired["timeout"]:
        # fn returned (rather than raising) after the watchdog fired, e.g. a
        # primitive that swallows AbortRequested internally. Treat as a timed-
        # out failure and retract the transient abort flag.
        if not _retract_watchdog_abort():
            # A genuine operator stop landed while the primitive unwound; it
            # outranks the timeout and must end the run.
            raise AbortRequested()
        return ExecutionResult(
            action_type=action_type,
            success=False,
            message=(
                f"{action_type} timed out after {budget_s:.1f}s (budget exceeded)"
            ),
            duration=time.time() - t0,
        )
    return result




def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        try:
            return _jsonable(obj.to_dict())
        except Exception:
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


@dataclass
class PrimitiveTrace:
    index: int
    primitive: str
    params: Dict[str, Any] = field(default_factory=dict)
    detections_used: List[Dict[str, Any]] = field(default_factory=list)
    grasp_strategy_resolved: Optional[str] = None
    gates: Dict[str, Any] = field(default_factory=dict)
    votes: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _jsonable(
            {
                "index": self.index,
                "primitive": self.primitive,
                "params": self.params,
                "detections_used": self.detections_used,
                "grasp_strategy_resolved": self.grasp_strategy_resolved,
                "gates": self.gates,
                "votes": self.votes,
                "elapsed_s": round(float(self.elapsed_s), 3),
                **self.extra,
            }
        )


def detection_summary(det: Any) -> Dict[str, Any]:
    """Compact record of one detection: what the primitive actually acted on."""
    get = det.get if hasattr(det, "get") else (lambda k, d=None: getattr(det, k, d))
    keys = (
        "label",
        "confidence",
        "camera",
        "position_3d",
        "aspect_ratio",
        "obb_minor_m",
        "obb_confidence",
        "world_major_axis_rad",
        "low_quality",
        "reprompt_attempts",
    )
    return _jsonable({k: get(k, None) for k in keys})


class TraceWriter:
    """Writes primitive traces into ``<output_dir>/trace``."""

    def __init__(self, output_dir: Any, enabled: bool = True, save_images: bool = False):
        self.enabled = bool(enabled)
        self.save_images = bool(save_images)
        self.root: Optional[Path] = None
        if self.enabled and output_dir:
            try:
                self.root = Path(output_dir) / "trace"
                self.root.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("Trace disabled (cannot create dir): %s", exc)
                self.enabled = False
                self.root = None
        # Resume from what is already on disk: writers are constructed per
        # call (from_pipeline) and output_dir has no per-run component, so a
        # fresh counter would overwrite earlier attempts' traces.
        self._index = self._highest_on_disk()

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> "TraceWriter":
        raw = getattr(getattr(pipeline, "profile", None), "raw", None) or {}
        block = raw.get("trace") if hasattr(raw, "get") else None
        block = block if isinstance(block, dict) else {}
        out = getattr(getattr(pipeline, "config", None), "output_dir", None)
        return cls(
            out,
            enabled=bool(block.get("enabled", True)),
            save_images=bool(block.get("save_images", False)),
        )

    def _highest_on_disk(self) -> int:
        """Largest ``NNN_`` prefix already written under ``root``, else 0."""
        if self.root is None:
            return 0
        best = 0
        try:
            for path in self.root.glob("*.json"):
                head = path.name.split("_", 1)[0]
                if head.isdigit():
                    best = max(best, int(head))
        except Exception:
            return best
        return best

    def next_index(self) -> int:
        self._index += 1
        return self._index

    def write(self, trace: PrimitiveTrace) -> Optional[Path]:
        if not self.enabled or self.root is None:
            return None
        try:
            name = "".join(c if c.isalnum() or c in "-_" else "_" for c in trace.primitive)
            path = self.root / f"{trace.index:03d}_{name}.json"
            path.write_text(json.dumps(trace.to_dict(), indent=2, default=str))
            return path
        except Exception as exc:
            logger.warning("Trace write failed: %s", exc)
            return None

    def write_verify(self, outcome: Any, elapsed_s: float = 0.0) -> Optional[Path]:
        """Trace for the terminal verification step."""
        data = outcome.to_dict() if hasattr(outcome, "to_dict") else {"outcome": str(outcome)}
        return self.write(
            PrimitiveTrace(
                index=self.next_index(),
                primitive="verify",
                gates=data.get("gates", {}),
                votes=data.get("votes", []),
                elapsed_s=elapsed_s,
                extra={
                    "status": data.get("status"),
                    "predicates": data.get("predicates", []),
                    "reason": data.get("reason", ""),
                    "depth_source": data.get("depth_source"),
                },
            )
        )




# Fallback container half-extent (m) when a detection lacks slot geometry: the
# region radius is max(this, OBB-minor * 0.6), widened to cover detected slots.
TRAY_FALLBACK_HALF_EXTENT_M = 0.10


def container_region(det) -> Optional[tuple]:
    """Return (centroid_xy[2], radius_m) for a container detection, or None.

    None when the detection has no usable 3D centroid.
    """
    if det is None:
        return None
    get = det.get if hasattr(det, "get") else (lambda k, d=None: getattr(det, k, d))
    pos = get("position_3d", None)
    if pos is None:
        return None
    try:
        centroid_xy = np.asarray(pos[:2], dtype=float)
    except Exception:
        return None
    minor = float(get("obb_minor_m", 0.0) or 0.0)
    radius = max(TRAY_FALLBACK_HALF_EXTENT_M, minor * 0.6)
    # Widen to cover detected slots if present (real tray extent).
    slots = get("slots", None) or []
    for s in slots:
        wxyz = s.get("world_xyz") if hasattr(s, "get") else None
        if wxyz is None:
            continue
        try:
            d = float(np.linalg.norm(np.asarray(wxyz[:2], dtype=float) - centroid_xy))
            radius = max(radius, d + TRAY_FALLBACK_HALF_EXTENT_M)
        except Exception:
            continue
    return centroid_xy, radius


def xy_inside_container(xy, det) -> bool:
    """True if planar point ``xy`` lies within container ``det``'s extent."""
    region = container_region(det)
    if region is None:
        return False
    centroid_xy, radius = region
    try:
        p = np.asarray(xy[:2], dtype=float)
    except Exception:
        return False
    return bool(np.linalg.norm(p - centroid_xy) <= radius)



FRANKA_FAMILIES = ("franka", "bimanual_franka")
FRANKA_REACH_M = 0.855


def apply_franka_bounds(executor, family: str) -> None:
    raw = family_block(getattr(executor, "_profile", None), family)
    arm = getattr(executor, "arm", None)
    ws = raw.get(f"workspace_{arm}" if arm else "workspace") or {}
    if ws:
        executor.WORKSPACE_MIN = np.array(
            [float(ws.get("x_min", -0.85)), float(ws.get("y_min", -0.85)), float(ws.get("z_min", -0.05))]
        )
        executor.WORKSPACE_MAX = np.array(
            [float(ws.get("x_max", 0.85)), float(ws.get("y_max", 0.85)), float(ws.get("z_max", 1.20))]
        )
        executor.MAX_REACH = float(ws.get("reach_max", FRANKA_REACH_M))
    if raw.get("table_z_floor") is not None:
        executor.TABLE_Z_FLOOR = float(raw["table_z_floor"])
    control = raw.get("control") or {}
    grasp = control.get("grasp_orientation")
    executor.GRASP_ORIENTATION = [float(v) for v in grasp] if grasp else [math.pi, 0.0, 0.0]
    if control.get("max_velocity") is not None:
        executor.velocity = min(float(executor.velocity), float(control["max_velocity"]))
    logger.info(
        "ScoreExecutor %s%s bounds: min=%s max=%s reach=%.3f table_z_floor=%.3f grasp=%s v=%.2f",
        family,
        f"[{arm}]" if arm else "",
        np.round(executor.WORKSPACE_MIN, 3).tolist(),
        np.round(executor.WORKSPACE_MAX, 3).tolist(),
        executor.MAX_REACH,
        executor.TABLE_Z_FLOOR,
        executor.GRASP_ORIENTATION,
        executor.velocity,
    )


class _StillnessTracker:
    """Motion / stillness detector on a FIXED sample cadence.

    The poll rate and the motion test are independent: displacement is
    evaluated over its own ``sample_s`` window. ``speed_eps`` is 3 mm/s, i.e.
    0.3 mm of travel per 100 ms sample; over a 25 ms window it would be
    0.075 mm, inside TCP measurement noise, and a spuriously latched
    ``ever_moved`` lets the stillness exit fire early.
    """

    __slots__ = ("_sample_s", "_eps", "_need", "_prev", "_prev_t", "ever_moved", "still")

    def __init__(self, tcp, now, sample_s=0.1, speed_eps=0.003, still_samples=3):
        self._sample_s = float(sample_s)
        self._eps = float(speed_eps)
        self._need = int(still_samples)
        self._prev = np.asarray(tcp, dtype=float).copy()
        self._prev_t = float(now)
        self.ever_moved = False
        self.still = 0

    def update(self, tcp, now) -> None:
        """Fold one poll in; only whole sample windows change the verdict."""
        dt = float(now) - self._prev_t
        if dt < self._sample_s:
            return
        step_speed = float(np.linalg.norm(np.asarray(tcp, dtype=float) - self._prev)) / dt
        if step_speed > self._eps:
            self.ever_moved = True
            self.still = 0
        else:
            self.still += 1
        self._prev = np.asarray(tcp, dtype=float).copy()
        self._prev_t = float(now)

    def stopped(self) -> bool:
        """Enough consecutive still windows to call the arm stopped."""
        return self.still >= self._need



# Joint / Cartesian speed envelope: ONE source for every movej and movel this
# executor emits. The URScript defaults movej(q, a=1.4, v=1.05) and
# movel(pose, a=1.2, v=0.25) are 50% of the base/shoulder rating and 33% of
# the wrist rating, not UR10e capabilities.
#
# Rated maxima, UR10e tech sheet (base + shoulder 120 deg/s, elbow + wrist
# 1/2/3 180 deg/s):
#   https://www.universal-robots.com/manuals/EN/HTML/SW5_25/Content/
#   prod-usr-man/complianceUR10e/H_g5_sections/appendix_g5/tech_spec_sheet.htm
UR10E_JOINT_VEL_RATED_RAD_S = (2.094, 2.094, 3.142, 3.142, 3.142, 3.142)

# SAFETY MARGIN, stated explicitly: command at most 85% of the rated joint
# speed (15% headroom). The headroom is not decoration: the installed pendant
# safety configuration has its own Joint Speed Limit, and exceeding THAT is a
# protective stop (C150/C152), not a faster move. 85% also leaves room for the
# pendant Stopping Time / Stopping Distance limits, which throttle commanded
# speed so the arm can always stop inside them.
JOINT_VEL_MARGIN = 0.85

# UR publishes NO per-joint acceleration rating: movej "scales torques to be
# within hardware limits" itself, so this is a REQUEST ceiling, not a hardware
# number. 3.5 rad/s^2 is 2.5x the URScript default of 1.4. Acceleration is the
# dominant lever here (a trapezoid is triangular for any move shorter than
# v^2/a, and most primitives on this task are short), but very high acceleration combined with
# blends is UR's named trigger for C173 joint-torque-overload, so this is
# deliberately not pushed further without rig evidence.
JOINT_ACC_CAP_RAD_S2 = 3.5

# Never crawl: a tiny commanded velocity makes large reconfigurations take
# forever.
JOINT_VEL_FLOOR_RAD_S = 0.3

# The Cartesian `velocity` knob (PipelineConfig.velocity, default 0.25 m/s) is
# what the whole executor is parameterised by. This is the value of that knob
# which maps to the FULL leading-axis joint cap; below it the joint cap scales
# down proportionally.
JOINT_VEL_REF_LINEAR_M_S = 0.25

# movel is Cartesian. UR10e tool speed: 4 m/s max, "approx 1 m/s" typical in
# the tech-sheet table. Cap at half the typical figure.
LINEAR_VEL_CAP_M_S = 0.5

LINEAR_ACC_CAP_M_S2 = 1.5

# Ramp-in knob: SPARK_JOINT_SPEED_FRACTION scales the whole envelope (v and a
# together). 1.0 = the envelope above; ~0.6 reproduces a 1.05 rad/s velocity.
_SPEED_FRACTION_ENV = "SPARK_JOINT_SPEED_FRACTION"

# SPARK_LEGACY_JOINT_LIMITS restores the legacy clamps below exactly
# (one-env-var rollback).
_LEGACY_LIMITS_ENV = "SPARK_LEGACY_JOINT_LIMITS"

_LEGACY = {
    "vel_scale": 6.0,
    "vel_cap": 1.05,
    "acc_cap": 1.4,
    "pose_vel_scale": 3.0,
    "pose_vel_cap": 1.0,
    "lin_vel_cap": 0.25,
    "lin_acc_cap": 1.2,
}


def legacy_joint_limits() -> bool:
    """True when SPARK_LEGACY_JOINT_LIMITS restores the legacy clamps."""
    return os.environ.get(_LEGACY_LIMITS_ENV, "0") not in ("0", "false", "False", "")


def joint_speed_fraction() -> float:
    """Fraction of the whole speed envelope to command (0 < f <= 1)."""
    raw = os.environ.get(_SPEED_FRACTION_ENV)
    if raw is None:
        return 1.0
    try:
        return float(np.clip(float(raw), 0.05, 1.0))
    except ValueError:
        logger.warning("%s=%r ignored", _SPEED_FRACTION_ENV, raw)
        return 1.0


def leading_axis_vel_cap(dq) -> float:
    """Highest movej ``v`` that keeps EVERY joint inside its own rating.

    URScript movej applies ``v``/``a`` to the LEADING axis (the largest
    |dq|); every other joint is time-scaled to finish together, so joint i
    actually runs at ``v * dq_i / L``. A single global cap therefore has to be
    the slowest joint's rating (2.094 rad/s), but on a wrist-dominated move
    the base is barely turning and there is 50% more headroom available. The
    exact condition is ``v * dq_i / L <= rated_i`` for all i, i.e.
    ``v <= min_i(rated_i * L / dq_i)``.

    Falls back to the most conservative rating when dq is unknown or zero.
    """
    dq = np.abs(np.asarray(dq, dtype=float).reshape(-1))
    rated = np.asarray(UR10E_JOINT_VEL_RATED_RAD_S[: dq.size], dtype=float)
    if dq.size == 0 or rated.size != dq.size:
        return JOINT_VEL_MARGIN * min(UR10E_JOINT_VEL_RATED_RAD_S)
    lead = float(dq.max())
    if lead <= 1e-9:
        return JOINT_VEL_MARGIN * min(UR10E_JOINT_VEL_RATED_RAD_S)
    moving = dq > 1e-9
    cap = float(np.min(rated[moving] * lead / dq[moving]))
    return JOINT_VEL_MARGIN * cap


def resolve_joint_limits(velocity, dq=None, pose_fallback=False):
    """Map the Cartesian ``velocity`` knob to one movej ``(v, a)`` in rad, s.

    ``dq`` is the per-joint travel of the move about to be issued; passing it
    unlocks the wrist headroom described in leading_axis_vel_cap. Omit it and
    the cap collapses to the slowest joint's rating, which is what a move with
    no solved joint target has to assume.

    ``pose_fallback`` marks the legacy Cartesian path (_movej_to_pose), whose
    legacy clamps (velocity*3 capped at 1.0) differ from the IK path's
    (velocity*6 capped at 1.05). It only changes anything under
    SPARK_LEGACY_JOINT_LIMITS.
    """
    if legacy_joint_limits():
        if pose_fallback:
            vel_j = float(
                min(float(velocity) * _LEGACY["pose_vel_scale"], _LEGACY["pose_vel_cap"])
            )
        else:
            vel_j = float(
                np.clip(
                    float(velocity) * _LEGACY["vel_scale"],
                    JOINT_VEL_FLOOR_RAD_S,
                    _LEGACY["vel_cap"],
                )
            )
        return vel_j, float(min(vel_j * 2.0, _LEGACY["acc_cap"]))
    frac = joint_speed_fraction()
    cap = leading_axis_vel_cap(dq if dq is not None else []) * frac
    ref = max(JOINT_VEL_REF_LINEAR_M_S, 1e-6)
    vel_j = float(np.clip(float(velocity) / ref * cap, JOINT_VEL_FLOOR_RAD_S, cap))
    acc_j = float(min(vel_j * 2.0, JOINT_ACC_CAP_RAD_S2 * frac))
    return vel_j, acc_j


def resolve_linear_limits(velocity):
    """movel ``(v, a)`` in m/s, m/s^2 for the legacy Cartesian fallback."""
    if legacy_joint_limits():
        vel_l = float(min(float(velocity), _LEGACY["lin_vel_cap"]))
        return vel_l, float(min(max(vel_l, 0.05) * 4.0, _LEGACY["lin_acc_cap"]))
    frac = joint_speed_fraction()
    vel_l = float(min(float(velocity), LINEAR_VEL_CAP_M_S * frac))
    acc_l = float(min(max(vel_l, 0.05) * 4.0, LINEAR_ACC_CAP_M_S2 * frac))
    return vel_l, acc_l


# Mirrors of spark_real.recording.schema.ACTION_SOURCE_*.
# tests/test_episode_parity.py asserts these stay equal to the schema values.
SOURCE_VELOCITY = "velocity_cmd"
SOURCE_SERVO = "servo_cmd"

_DEFAULT_SOURCE = SOURCE_VELOCITY

_local = threading.local()


def current_source() -> str:
    """Provenance tag for a command issued from this thread right now."""
    return getattr(_local, "source", _DEFAULT_SOURCE)


@contextmanager
def command_source(name: str) -> Iterator[str]:
    """Tag every velocity command issued from this thread inside the block."""
    previous = getattr(_local, "source", _DEFAULT_SOURCE)
    _local.source = str(name)
    try:
        yield _local.source
    finally:
        _local.source = previous


@dataclass
class CommandSnapshot:
    """What the driver last commanded, with monotonic stamps."""

    velocity: Optional[np.ndarray] = None
    velocity_source: str = _DEFAULT_SOURCE
    velocity_time: float = -1.0
    gripper: Optional[float] = None
    gripper_time: float = -1.0


class CommandLatch:
    """Thread-safe holder for the most recent commanded velocity / gripper."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = CommandSnapshot()

    def latch_velocity(self, velocity: Sequence[float], source: Optional[str] = None) -> None:
        """Record a commanded base-frame twist. Never raises."""
        try:
            v = np.asarray(velocity, dtype=np.float64).ravel()[:6]
            if v.size < 6:
                v = np.pad(v, (0, 6 - v.size))
        except (TypeError, ValueError):
            return
        with self._lock:
            self._snapshot.velocity = v
            self._snapshot.velocity_source = source or current_source()
            self._snapshot.velocity_time = time.monotonic()

    def latch_gripper(self, position: float) -> None:
        """
        Record the COMMANDED normalized gripper target (0 open .. 1 closed).

        The measured Robotiq aperture is stale mid-motion and must never be
        substituted here.
        """
        try:
            g = float(np.clip(float(position), 0.0, 1.0))
        except (TypeError, ValueError):
            return
        with self._lock:
            self._snapshot.gripper = g
            self._snapshot.gripper_time = time.monotonic()

    def snapshot(self) -> CommandSnapshot:
        """Immutable copy of the current latch state."""
        with self._lock:
            s = self._snapshot
            return CommandSnapshot(
                velocity=None if s.velocity is None else s.velocity.copy(),
                velocity_source=s.velocity_source,
                velocity_time=s.velocity_time,
                gripper=s.gripper,
                gripper_time=s.gripper_time,
            )




class VerifyBlockError(ValueError):
    """A ``verify:`` block is malformed; the whole block must be rejected."""


STACKED_DZ_MIN_M = 0.025

_SCHEMA: Dict[str, Tuple[Tuple[str, ...], Dict[str, Any]]] = {
    "inside": (
        ("obj", "container"),
        {"xy_margin_m": -0.02, "z_tol_m": 0.06, "occlusion_ok": False},
    ),
    "on": (("obj", "surface"), {"xy_margin_m": 0.0, "z_max_m": 0.10}),
    "stacked": (
        ("obj", "base"),
        {"xy_tol_m": 0.035, "dz_min_m": STACKED_DZ_MIN_M, "dz_max_m": 0.120},
    ),
    "near": (("obj", "target", "max_dist_m"), {}),
    "removed_from": (("obj", "container"), {}),
    "held": (("obj", "value"), {}),
    "absent": (("obj",), {}),
}

# which params name a detection label
_LABEL_KEYS = ("obj", "container", "surface", "base", "target")

# ... and which of those name the REFERENCE FRAME rather than the object under
# test. The tree moves ``obj``; the container/surface/base/target it is judged
# against is scenery the tree never touched.
_REFERENCE_KEYS = ("container", "surface", "base", "target")


@dataclass(frozen=True)
class Predicate:
    pred: str
    params: Mapping[str, Any]

    @property
    def labels(self) -> List[str]:
        """Detection labels this predicate needs (``held`` needs none)."""
        return self._labels_for(_LABEL_KEYS)

    @property
    def reference_labels(self) -> List[str]:
        """Labels naming the reference frame this predicate measures against."""
        return self._labels_for(_REFERENCE_KEYS)

    @property
    def object_labels(self) -> List[str]:
        """Labels naming the object under test."""
        return self._labels_for(("obj",))

    def _labels_for(self, keys: Sequence[str]) -> List[str]:
        if self.pred == "held":
            return []
        out = []
        for key in keys:
            val = self.params.get(key)
            if isinstance(val, str) and val:
                out.append(val)
        return out

    def describe(self) -> str:
        if self.pred == "held":
            return f"held({self.params.get('obj', '?')}, {bool(self.params.get('value'))})"
        args = [str(self.params[k]) for k in _LABEL_KEYS if k in self.params]
        if self.pred == "near":
            args.append(f"{float(self.params['max_dist_m']):.3f}m")
        return f"{self.pred}({', '.join(args)})"


@dataclass
class VerifySpec:
    predicates: List[Predicate]
    min_conf: float = 0.35
    require_two_views: bool = False
    source: str = "planner"

    @property
    def labels(self) -> List[str]:
        return self._union("labels")

    @property
    def reference_labels(self) -> List[str]:
        return self._union("reference_labels")

    @property
    def object_labels(self) -> List[str]:
        return self._union("object_labels")

    def _union(self, attr: str) -> List[str]:
        seen, out = set(), []
        for p in self.predicates:
            for lab in getattr(p, attr):
                if lab not in seen:
                    seen.add(lab)
                    out.append(lab)
        return out

    def describe(self) -> List[str]:
        return [p.describe() for p in self.predicates]
