"""
Single-arm cloth folding skills the planner composes, for any single-arm
robot family.

    cloth_fold_single  pinch a corner/edge (stepped retries), lift, arc to the
                       mirror position across the fold axis (or land_point),
                       lay it down, release, retract.
                       post: pinched before the swing, jaw open after
    cloth_fold         T-shirt: +Y sleeve and -Y sleeve folded past the shirt
                       center, re-perceive, hem folded up to the collar; each
                       step is a cloth_fold_single.     post: every step's

Positions come from the detection map (sleeve, shirt, hem and collar
labels); heights, speeds, force and margins from the plan, else the family
`task_motion: cloth:` block, else the `task_motion:` block, else the defaults
here. Nothing in this file depends on the robot family: moves go through the
executor's _move_to, the pinch orientation through its top-down grasp
orientation and wrist yaw limit (grasp_strategy.compose_yaw), the jaws through
the driver protocol. The shirt lies with collar-hem along base X and the
sleeves along base Y; yaw 0 closes the jaws along base Y.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from spark_real.control.grasp_strategy import compose_yaw, max_yaw_offset, symmetric_yaw
from spark_real.skills import task_motion as tm
from spark_real.skills.primitives import _result
from spark_real.skills.registry import spark_skill

logger = logging.getLogger(__name__)

FOLD = {
    "approach_height": 0.08,
    "lift_height": 0.08,  # pinch-to-arc lift above the grasp (m)
    "arc_peak": 0.030,  # arc apex above the lift height (m)
    "place_height": 0.03,  # landing height above the grasp z (m)
    "arc_steps": 12,
    "arc_vel": 0.11,  # arc waypoint speed (executor velocity knob, m/s)
    "force": 30.0,  # pinch force in the gripper driver's unit (N on the Franka Hand)
    "past_center_m": 0.08,  # sleeves land this far past the shirt center line
    "grasp_dz": -0.02,  # cloth: pinch below the detected surface point
    "sleeve_yaw_deg": 90.0,  # jaws close along base X on a sleeve
    "hem_yaw_deg": 0.0,  # jaws close along base Y on the hem
}

# The pick numbers handed to task_motion.pick (see task_motion.DEFAULTS).
PICK_KEYS = ("retry_step_m", "grasp_retries", "held_width_m", "slow_vel", "close_settle_s",
             "open_settle_s", "held_check", "move_vel", "floor_margin_m")


def _f(ex, params, key):
    """Plan value, else `task_motion: cloth:`, else `task_motion:`/task_motion.DEFAULTS, else FOLD."""
    if params.get(key) is not None:
        return params[key]
    block = tm.cfg(ex, "cloth")
    if isinstance(block, dict) and block.get(key) is not None:
        return block[key]
    v = tm.cfg(ex, key)
    return FOLD.get(key) if v is None else v


def _pinch_orient(ex, yaw_deg: float):
    """Top-down orientation turned by yaw about base Z. A pinch is the same
    after a half turn, so a yaw past the wrist limit takes the equivalent one."""
    yaw = math.radians(float(yaw_deg))
    if abs(yaw) > max_yaw_offset(ex):
        yaw = symmetric_yaw(yaw)
    return compose_yaw(ex, yaw, context="cloth")


def _label(ex, name: str, *alts: str) -> str:
    """The first of name and its alternatives that is in the detection map."""
    return next((n for n in (name, *alts) if n in ex.detection_map), name)


def _outside_workspace(ex, points) -> list:
    lo, hi = ex.WORKSPACE_MIN[2], ex.WORKSPACE_MAX[2]
    return [np.round(p, 3).tolist() for p in points
            if not ex._check_workspace(np.array([p[0], p[1], min(max(p[2], lo), hi)], dtype=float))]


def _redetect(ex, label: str) -> None:
    """Refresh one label from the cameras when a pipeline is attached."""
    if getattr(ex, "_pipeline", None) is None or not label:
        return
    try:
        from spark_real.control.recovery import redetect_single

        redetect_single(ex, label)
    except Exception as exc:  # noqa: BLE001 - keep the previous detection
        logger.warning("re-detect of %r failed: %s", label, exc)


@spark_skill(
    "cloth_fold_single",
    "Pinch a cloth corner, lift, arc it across the fold axis, lay it down, release.",
    {"corner_label": str, "anchor_label": str, "anchor_midpoint": list, "land_point": list,
     "lift_height": float, "place_height": float, "force": float, "arc_steps": int,
     "approach_height": float, "release_after": bool, "yaw_deg": float, "grip_width": float},
)
def cloth_fold_single(ex, params: dict):
    t0 = time.time()
    corner = tm.det_xyz(ex, params.get("corner_label", "")) if params.get("corner_point") is None \
        else np.asarray(params["corner_point"], dtype=float)
    if params.get("land_point") is not None:
        land = np.asarray(params["land_point"], dtype=float)
    else:
        anchor = np.asarray(params["anchor_midpoint"], dtype=float) if params.get("anchor_midpoint") is not None \
            else tm.det_xyz(ex, params.get("anchor_label", ""))
        land = 2 * anchor - corner  # mirror across the fold axis
    orient = _pinch_orient(ex, params.get("yaw_deg", 0.0))
    if orient is None:
        return _result("cloth_fold_single", False, "pinch yaw is past the wrist limit", time.time() - t0)
    pick_params = {
        "force": _f(ex, params, "force"),
        "approach_dz": _f(ex, params, "approach_height"),
        "grasp_dz": _f(ex, params, "grasp_dz"),
        "lift_dz": _f(ex, params, "lift_height"),
        "target_width": params.get("grip_width"),
        **{k: _f(ex, params, k) for k in PICK_KEYS},
    }
    keep_x = bool(params.get("keep_x", False))
    pinch_z = tm.floor_z(ex, corner[2] + float(pick_params["grasp_dz"]), pick_params)
    top_z = pinch_z + float(pick_params["lift_dz"]) + float(_f(ex, params, "arc_peak"))
    end_xy = [corner[0] if keep_x else land[0], land[1]]
    outside = _outside_workspace(ex, [[*xy, z] for xy in (corner[:2], end_xy) for z in (pinch_z, top_z)])
    if outside:  # a clipped move would lay the cloth in the wrong place
        return _result("cloth_fold_single", False, f"fold leaves the workspace at {outside}", time.time() - t0)
    if not tm.pick(ex, corner, orient, pick_params):
        tm.open_jaws(ex, pick_params)
        return _result("cloth_fold_single", False, "pinch failed after retries", time.time() - t0)
    start = tm.tcp(ex)
    land_z = tm.floor_z(ex, corner[2] + float(pick_params["grasp_dz"]) + float(_f(ex, params, "place_height")),
                        pick_params)
    tm.arc(ex, start, land, start[2], float(_f(ex, params, "arc_peak")), orient,
           n=int(_f(ex, params, "arc_steps")), land_z=land_z, keep_x=keep_x, vel=float(_f(ex, params, "arc_vel")))
    if params.get("release_after", True):
        tm.open_jaws(ex, pick_params)
        ex._holding = False
        end = tm.tcp(ex)
        ex._move_to([end[0], end[1], end[2] + float(_f(ex, params, "lift_height"))], orient,
                    velocity=pick_params["move_vel"])
    opened = not params.get("release_after", True) or not ex.robot.is_object_detected()
    return _result("cloth_fold_single", opened, f"folded to {np.round(land[:2], 3).tolist()}", time.time() - t0)


@spark_skill(
    "cloth_fold",
    "T-shirt fold: both sleeves past the center line, then hem up to the collar.",
    {"instruction": str, "sleeve_labels": list, "shirt_label": str, "hem_label": str,
     "collar_label": str, "past_center_m": float, "sleeve_yaw_deg": float, "hem_yaw_deg": float},
)
def cloth_fold(ex, params: dict):
    t0 = time.time()
    # Also the names the perception prompt step gives the shirt parts.
    sleeves = [_label(ex, s, f"shirt {s}") for s in params.get("sleeve_labels") or ["left sleeve", "right sleeve"]]
    shirt = _label(ex, params.get("shirt_label", "shirt"), "shirt body")
    hem = _label(ex, params.get("hem_label", "hem"), "shirt hem")
    center = tm.det_xyz(ex, shirt)
    past = float(_f(ex, params, "past_center_m"))
    pts = sorted(((s, tm.det_xyz(ex, s)) for s in sleeves), key=lambda sp: -sp[1][1])
    steps = []
    for label, p in pts:  # +Y sleeve first; each lands past the center line
        side = 1.0 if p[1] > center[1] else -1.0
        land = [p[0], center[1] - side * past, p[2]]
        r = cloth_fold_single(ex, {**params, "corner_label": label, "land_point": land, "keep_x": True,
                                   "yaw_deg": _f(ex, params, "sleeve_yaw_deg")})
        steps.append((label, r.success))
        if not r.success:
            return _result("cloth_fold", False, f"{label} fold failed: {r.message}", time.time() - t0)
    for label in (hem, shirt, params.get("collar_label")):
        _redetect(ex, label)
    hem_p = tm.det_xyz(ex, hem)
    center = tm.det_xyz(ex, shirt)
    collar = tm.det_xyz(ex, params["collar_label"]) if params.get("collar_label") else 2 * center - hem_p
    r = cloth_fold_single(ex, {**params, "corner_label": hem, "land_point": [collar[0], center[1], hem_p[2]],
                               "yaw_deg": _f(ex, params, "hem_yaw_deg")})
    steps.append((hem, r.success))
    return _result("cloth_fold", all(ok for _, ok in steps), f"steps={steps}", time.time() - t0)
