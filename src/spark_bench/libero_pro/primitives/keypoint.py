"""
``move_to_keypoint``: the pick strategies (bowl rim, plate rim, top-down) and the matching
place. With SPARK_SKILLSET=shape or ask the rim pinch and the flat place are chosen and sized from the
mask's world points (``_rim_shape``, ``_centre_drop``), never from the label.

All entry points take ``(executor, params)``; helpers take the executor
as first arg and pull state (``holding``, ``last_pick_*``) from it.
"""
from __future__ import annotations

import copy
import os
from typing import Optional

import mujoco
import numpy as np

from spark_core import sim_detect, skillset
from spark_bench.libero_pro.primitives._common import (
    approach_descend,
    fail_pick,
    fuzzy_get_det,
    nudge_toward_base,
)
from spark_bench.libero_pro.motion import MOTION_FIX, reach_short, step_env
from spark_bench.libero_pro.telemetry import FRANKA_OPEN_APERTURE, find_finger_qpos_addrs

__all__ = ['handle']


_FLAT_PLACE_KW = ('plate', 'stove', 'cabinet', 'rack', 'dish')

# Gripper sizes for the shape rules (SPARK_SKILLSET=shape / ask).
FINGER_DEPTH = 0.03   # how far the fingertips go below an edge they pinch
RIM_CLEAR = 0.004     # fingertip clearance above the surface the object rests on


def _mask_pts(executor, det) -> Optional[np.ndarray]:
    """World points of a detection's mask; None when unavailable or fewer than 50."""
    mask = getattr(det, 'mask', None)
    if executor.sam3 is None or executor.depth is None or mask is None:
        return None
    try:
        pts = executor.sam3._backproject_mask(mask, executor.depth, executor.cam_pos, executor.cam_mat,
                                              executor.cam_fovy, executor.cam_w, executor.cam_h)
    except Exception:
        return None
    return pts if pts is not None and len(pts) >= 50 else None


def _centre_drop(pts: np.ndarray) -> Optional[float]:
    """How far the middle of the footprint lies below the top (95th-percentile z): the median z of the points
    within a quarter of the smaller footprint extent of the footprint centre. None when the view shows fewer
    than 20 points there (a deep interior hidden by its own wall)."""
    (x_lo, y_lo), (x_hi, y_hi) = np.percentile(pts[:, :2], [2, 98], axis=0)
    r = min(x_hi - x_lo, y_hi - y_lo) / 4
    sel = pts[np.hypot(pts[:, 0] - (x_lo + x_hi) / 2, pts[:, 1] - (y_lo + y_hi) / 2) < r]
    if len(sel) < 20:
        return None
    return float(np.percentile(pts[:, 2], 95) - np.median(sel[:, 2]))


def _rim_shape(pts: np.ndarray) -> dict:
    """Rim-pinch geometry of an object from its mask's world points (the jaws separate along world Y).

      wide  extent along the jaw axis (2nd-98th percentile) above the open aperture: a top-down close cannot
            span the object, so it is pinched at its +Y edge
      y     gripper centre, a quarter aperture inside that edge: the edge sits between the open jaws
      z     fingertips FINGER_DEPTH below the top, never lower than RIM_CLEAR above the base
      flat  the base clamp is active (the object is lower than the fingertips are long): the jaws close on a
            thin wall at the support surface, which needs the regulated close
      dy    gripper centre minus footprint centre; the place adds it back
      hang  how far the object reaches below the fingertips
    """
    y_lo, y_hi = sim_detect.jaw_span(pts)
    z_lo, z_top = np.percentile(pts[:, 2], [5, 95])
    y = y_hi - FRANKA_OPEN_APERTURE / 4
    z = max(z_top - FINGER_DEPTH, z_lo + RIM_CLEAR)
    return {'wide': bool(y_hi - y_lo > FRANKA_OPEN_APERTURE), 'y': float(y), 'z': float(z),
            'flat': bool(z_top - FINGER_DEPTH < z_lo + RIM_CLEAR),
            'dy': float(y - (y_lo + y_hi) / 2), 'hang': float(z - z_lo)}


def _flat_top(pts: Optional[np.ndarray]) -> bool:
    """A place target is a flat support (not a container) when the middle of its footprint is seen and lies
    less than FINGER_DEPTH below its top."""
    drop = _centre_drop(pts) if pts is not None else None
    return drop is not None and drop < FINGER_DEPTH


def handle(executor, params: dict) -> None:
    """
    ``move_to_keypoint`` - pick or place depending on ``executor.holding``.
    """
    label = params.get('keypoint_label', '') or ''
    det = fuzzy_get_det(executor.det_map, label)
    if det is None or det.position_3d is None:
        return
    target = det.position_3d.copy()
    target += np.array([params.get('offset_x', 0),
                         params.get('offset_y', 0),
                         params.get('offset_z', 0)])

    lbl = label.lower()
    # SPARK_SKILLSET=plain measures the pick without the rim strategies below.
    plain = skillset.plain_pick()
    geo = skillset.geometry_pick()   # 'shape' / 'ask': the points decide, not the label
    is_bowl = 'bowl' in lbl and not plain
    is_plate = ('plate' in lbl or 'dish' in lbl) and not is_bowl and not plain
    # Wipe targets (dirt/stain/marks) are not graspable: move to them with closed jaws, no pick.
    is_wipe_target = any(kw in lbl for kw in ('dirt', 'marks', 'stain', 'spill', 'smudge'))
    target[2] = max(target[2], 0.005)

    if executor.holding:
        top = (_top_surface(executor, det)
               if ' top' in f" {(getattr(executor, 'instruction', '') or '').lower()}" else None)
        if top is not None:
            # A tall place target (cabinet: mask spans its front face) has its centroid on the front, low;
            # releasing there dropped the wine bottle off the front edge (goal 2 position). Aim at the
            # centre of the surface it will rest on.
            target = top + np.array([params.get('offset_x', 0), params.get('offset_y', 0),
                                     params.get('offset_z', 0)])
        _do_place(executor, target, lbl, flat=_flat_top(_mask_pts(executor, det)) if geo else None)
    elif is_wipe_target:
        executor._move_to(target, False, steps=400)
    else:
        # ask: the plan's grasp_style; a re-approach of the same pick (pre-close capture) keeps it.
        style = str(params.get('grasp_style') or '').lower()
        if not style and label == executor.last_pick_label and getattr(executor, 'last_pick_shape', None):
            style = 'rim'
        shift = None
        if PICK_REGROUND:
            shift = _reground_pick(executor, label, target)
            if shift is not None:
                target[:2] += shift
                det = _shifted(det, shift)
        _do_pick(executor, label, det, target, lbl, is_bowl, is_plate,
                 rim={'shape': None, 'ask': style == 'rim'}.get(geo, False), shift=shift)


# Place


def _top_surface(executor, det) -> Optional[np.ndarray]:
    """Centre of the top surface of a tall (> 10 cm) place target from its mask's backprojected points:
    xy mean of the points within 2 cm of the 95th-percentile height, at that height. None when the target
    is short (plate, bowl, basket: the centroid already is the surface) or the mask is unavailable."""
    pts = _mask_pts(executor, det)
    if pts is None:
        return None
    z_hi = float(np.percentile(pts[:, 2], 95))
    if z_hi - float(np.percentile(pts[:, 2], 5)) < 0.10:
        return None
    sel = pts[pts[:, 2] > z_hi - 0.02]
    return np.array([float(sel[:, 0].mean()), float(sel[:, 1].mean()), z_hi])


def _do_place(executor, target: np.ndarray, lbl: str, flat: Optional[bool] = None) -> None:
    place_target = target.copy()
    # Remember where we intend to land the held object - the post-release
    # event capture verifies the placement against this.
    executor.last_place_target = target.copy()
    # Drawer-aware placement: after open_drawer ran, a place whose label
    # references the drawer/handle/cabinet should target the pulled-out
    # TRAY interior (tracked by the primitive), not the stale detection of
    # the closed-drawer handle.
    tray = getattr(executor, 'last_drawer_tray_pos', None)
    if tray is not None and any(k in lbl for k in
                                 ('drawer', 'handle', 'cabinet')):
        place_target = np.array([tray[0], tray[1], tray[2] + 0.10])
        # The post-release verification must check the tray landing
        # point, not the stale handle detection.
        executor.last_place_target = place_target.copy()
        ee = executor._ee()
        safe_z = max(float(ee[2]) + 0.05, float(place_target[2]) + 0.15)
        lift = ee.copy()
        lift[2] = safe_z
        executor._move_to(lift, False, steps=300)
        above = place_target.copy()
        above[2] = safe_z
        executor._move_to(above, False, steps=500)
        executor._move_to(place_target, False, steps=300)
        return

    if hasattr(executor.env, '_grip_offset'):
        # The bowl-rim grip leaves the bowl's centroid at EE - grip_offset_world
        # (gripper sits +Y of the rim).  To land the bowl centroid ON the
        # target XY, the EE must be at target + grip_offset.
        place_target[:2] += executor.env._grip_offset[:2]
    held_height = getattr(executor.env, '_held_obj_height', 0.0)
    is_flat_place = (any(kw in lbl for kw in _FLAT_PLACE_KW) and not skillset.plain_pick()
                     if flat is None else flat)
    if held_height > 0.05 and is_flat_place:
        safe_drop = max(held_height * 2.0 + 0.05, 0.20)
        place_target[2] = target[2] + safe_drop
    else:
        # Settle-crawl compensation.  The settle-to-contact crawl in
        # _do_top_down_pick grips the object N cm LOWER than the detected
        # keypoint, so the object sits N cm HIGHER at any commanded release
        # height.  The plan's place offset is authored against the un-crawled
        # grip, so give the crawl depth back at release.  Not applied on the
        # flat-place branch above, which already sets an absolute drop height
        # from the object's own extent.
        place_target[2] -= float(getattr(executor, 'last_crawl_dz', 0.0) or 0.0)
    # Three-phase transport (diagonal motion would drag the held bowl
    # through stove/plate edges):
    #   1. LIFT vertically at current XY to a safe clearance,
    #   2. XY-TRANSPORT horizontally at that height,
    #   3. DESCEND vertically to place_target.
    ee = executor._ee()
    safe_z = max(float(ee[2]) + 0.05,
                 float(place_target[2]) + 0.20)
    # Step 1: lift in place
    if ee[2] < safe_z - 0.02:
        lift = ee.copy()
        lift[2] = safe_z
        executor._move_to(lift, False, steps=400)
    # Step 2: XY transit at safe Z, monitored so a moved container is
    # caught mid-flight rather than after the descend.
    above = place_target.copy()
    above[2] = safe_z
    executor._move_to_monitored(lbl, above, False, steps=600)
    place_target = place_target + getattr(executor, 'monitor_shift', 0.0)   # a mid-flight retarget moved the hover point
    # Step 3: descend to place
    executor._move_to(place_target, False, steps=400)


# Pick

# Re-ground the pick target from above with the wrist camera before the grasp (SPARK_PICK_REGROUND=1): the
# agentview mask of a tall item is mostly its front face, so its centroid sits ~2 cm toward the camera.
PICK_REGROUND = os.environ.get('SPARK_PICK_REGROUND', '0') == '1'
# SPARK_PICK_REGROUND_GATE_M: largest accepted re-ground shift. In static scenes (LIBERO-PRO) nothing moves before
# the grasp and the real fixes are <= 2 cm, while an 18 cm gate lets a missed target bind a look-alike 8-15 cm away;
# the _rgs arm token sets 0.05. 0.18 covers targets relocated at approach.
PICK_REGROUND_GATE_M = float(os.environ.get('SPARK_PICK_REGROUND_GATE_M', '0.18'))


def _wrist_fix(executor, lbl: str, near_xy: np.ndarray):
    """xy of the wrist camera's instance of ``lbl`` nearest ``near_xy``, or None."""
    from spark_bench.libero_pro.perception import redetect_wrist
    fresh = redetect_wrist(executor.env, executor.sam3, executor.prompts_for_refresh, executor.cfg,
                           all_instances=True)
    cands = [d for d in (fresh.dets if fresh is not None else [])
             if d.position_3d is not None and fuzzy_get_det({d.label: d}, lbl) is not None]
    if not cands:
        return None
    xy = np.array([np.asarray(d.position_3d, float)[:2] for d in cands])
    return xy[int(np.argmin(np.linalg.norm(xy - near_xy, axis=1)))]


def _reground_pick(executor, lbl: str, target: np.ndarray):
    """xy shift for the pick target: from a hover above the bound position, the wrist camera's instance of
    ``lbl`` nearest to the binding (not the best-scored one: two bowls may be in view). The arm's approach can
    itself set the scene moving, so up to 4 fixes repeat 8 steps apart until two agree within 1 cm.
    None without a fix within PICK_REGROUND_GATE_M."""
    if executor.sam3 is None or not executor.prompts_for_refresh:
        return None
    hover = np.asarray(target, float).copy()
    hover[2] += 0.10
    executor._move_to(hover, True, steps=150, tol=0.02)   # a view point: no need to settle to 5 mm
    rec = {'label': lbl, 'fixes': []}
    executor.trial_meta.setdefault('pick_reground', []).append(rec)
    near, fix = np.asarray(target, float)[:2], None
    for k in range(4):
        if k:
            for _ in range(8):
                step_env(executor.env, np.array([0, 0, 0, 0, 0, 0, -1.0]))
        try:
            f = _wrist_fix(executor, lbl, near)
        except Exception as e:  # noqa: BLE001  (no fix: keep the binding)
            rec['err'] = repr(e)[:120]
            break
        if f is None:
            break
        rec['fixes'].append([round(float(x) * 100, 1) for x in f - target[:2]])
        still = fix is not None and float(np.linalg.norm(f - fix)) < 0.01
        fix, near = f, f
        if still:
            break
    if fix is None:
        return None
    shift = fix - np.asarray(target, float)[:2]
    rec['shift_cm'] = [round(float(x) * 100, 1) for x in shift]
    if float(np.linalg.norm(shift)) > PICK_REGROUND_GATE_M:
        rec['rejected'] = True
        return None
    return shift


def _shifted(det, shift):
    """A copy of ``det`` moved by ``shift`` in xy (its mask stays the agentview's)."""
    d = copy.copy(det)
    d.position_3d = np.asarray(det.position_3d, float).copy()
    d.position_3d[:2] += shift
    return d


def _do_pick(executor, label: str, det, target: np.ndarray,
              lbl: str, is_bowl: bool, is_plate: bool, rim=False, shift=None) -> None:
    """``rim``: False = label rules (is_bowl / is_plate); None = rim pinch when the points say the object is
    wider than the open jaws (shape); True = rim pinch because the plan asked for it (ask)."""
    # Remember pick context for grip-offset resolution at grasp time.
    executor.last_pick_label = label
    executor.last_pick_det_pos = (
        det.position_3d.copy() if det.position_3d is not None else None)
    executor.env._held_obj_height = 0.0
    # Reset per pick: only the top-down path crawls, and a stale depth from
    # an earlier pick must never bias this object's release height.
    executor.last_crawl_dz = 0.0
    executor.last_pick_shape = None
    executor.rim_off = None
    executor.pick_failed = None
    if getattr(executor.env, '_wrist_turned', False):   # an earlier wrist-turned pinch (SPARK_MOTION_FIX)
        _turn_wrist(executor, along_x=False)
        executor.env._wrist_turned = False

    pts = None
    mask = getattr(det, 'mask', None)
    if (executor.sam3 is not None and executor.depth is not None
            and mask is not None):
        pts = executor.sam3._backproject_mask(
            mask, executor.depth, executor.cam_pos, executor.cam_mat,
            executor.cam_fovy, executor.cam_w, executor.cam_h)
        if len(pts) > 50:
            # Held height, read by _do_place's flat-target drop.
            executor.env._held_obj_height = float(pts[:, 2].max() - pts[:, 2].min())

    if rim is not False:
        # A pick bound from the wrist camera (wrist_verify) has no agentview mask; it carries its own points.
        gpts = pts if pts is not None else getattr(det, 'pts_world', None)
        if gpts is not None and len(gpts) > 50:
            # The re-ground moved the target, not the points: move them with it.
            shape = _rim_shape(gpts if shift is None else gpts + np.array([shift[0], shift[1], 0.0]))
            take = bool(rim or shape['wide'])
            executor.trial_meta.setdefault('pick_shape', []).append(
                {'label': label, 'rim': take, 'view': 'agent' if pts is not None else 'wrist', **shape})
            if take:
                executor.last_pick_shape = shape
                _do_rim_pinch(executor, target, shape)
                return
        else:
            executor.trial_meta.setdefault('pick_shape', []).append({'label': label, 'rim': False, 'view': None})
    if is_bowl:
        _do_bowl_rim_pinch(executor, target)
    elif is_plate:
        _do_plate_rim_pinch(executor, target, pts=pts)
    else:
        _do_top_down_pick(executor, target)


def turned_rim(rim_target: np.ndarray, centre_y: float) -> np.ndarray:
    """The rim point for a pinch with the wrist turned a quarter turn (jaws along x): the same offset from the
    object's centre, on the robot-base (-X) side."""
    # ponytail: takes the x centre from the detection and the y offset as the x offset (round rims: bowl, plate);
    # an oblong object needs its own x extent from the mask points.
    out = np.array(rim_target, dtype=float)
    out[0] -= abs(out[1] - centre_y)
    out[1] = centre_y
    return out


def _turn_wrist(executor, along_x: bool) -> None:
    """Quarter turn of joint 7 so the jaws open along world x (or back along y); nothing when they already do.
    The turn lasts until it is turned back (the next pick)."""
    mujoco.mj_forward(executor.model, executor.data)
    if executor._finger_addrs is None:
        executor._finger_addrs = find_finger_qpos_addrs(executor.model)
    if not executor._finger_addrs:
        return
    jid = int(np.nonzero(executor.model.jnt_qposadr == executor._finger_addrs[0])[0][0])
    ax = executor.data.xaxis[jid]
    if (abs(ax[0]) > abs(ax[1])) == along_x:
        return
    q = executor._q_now().copy()
    q[6] += -np.pi / 2 if q[6] > 0 else np.pi / 2   # jaw symmetry: either way works, take the one away from the limit
    q[6] = float(np.clip(q[6], -2.8, 2.8))
    executor._joint_move(q, True, steps=1000)
    # The OSC keeps its last orientation goal while the rotation command is zero, so a position move after a
    # joint-space turn would turn the wrist back: set the goal to the pose the arm has now.
    ctrl = getattr(executor.env, 'env', executor.env).robots[0].controller
    ctrl.update(force=True)
    ctrl.reset_goal()
    executor.env._wrist_turned = along_x


def _rim_approach(executor, rim_target: np.ndarray, centre_y: float, **kw) -> Optional[np.ndarray]:
    """``approach_descend`` to a rim point; returns the rim point the hand is at, None when it reached neither.

    SPARK_MOTION_FIX: a hand that stops short (a tall neighbour on +Y under the hand body, which is 20 cm wide
    along the jaw axis) turns the wrist a quarter turn and pinches the -X rim with its thin side toward the
    neighbour; short there too, the pick is marked not reached and nothing closes."""
    approach_descend(executor, rim_target, **kw)
    if not MOTION_FIX or not reach_short(executor._ee(), rim_target):
        return rim_target
    ev = {'label': executor.last_pick_label, 'rim': [round(float(x), 4) for x in rim_target],
          'ee': [round(float(x), 4) for x in executor._ee()]}
    executor.trial_meta.setdefault('reach_events', []).append(ev)

    def lift_clear():   # clear of what stopped the hand before the wrist turns
        up = executor._ee()
        up[2] = max(up[2] + 0.03, rim_target[2] + kw.get('hi', 0.10))
        executor._move_to(up, True, steps=100, tol=0.02)
    lift_clear()
    _turn_wrist(executor, along_x=True)
    other = turned_rim(rim_target, centre_y)
    approach_descend(executor, other, **kw)
    ev['turned'], ev['ee_turned'] = [round(float(x), 4) for x in other], [round(float(x), 4) for x in executor._ee()]
    ev['reached'] = not reach_short(executor._ee(), other)
    if not ev['reached']:
        lift_clear()
        fail_pick(executor, 'not_reached')
        return None
    executor.rim_off = other[:2] - np.array([rim_target[0], centre_y])   # handle_grasp: the place adds it back
    return other


def _do_rim_pinch(executor, target: np.ndarray, shape: dict) -> None:
    """Rim pinch sized by ``_rim_shape``; the careful descent of the plate pinch when the object is flat."""
    rim_target = target.copy()
    rim_target[1] = shape['y']
    rim_target[2] = max(shape['z'], 0.005)
    centre_y = shape['y'] - shape['dy']
    if shape['flat']:
        _rim_approach(executor, rim_target, centre_y, hi=0.14, mid=0.03,
                      mid_steps=150, final_steps=200, high_steps=150)
    else:
        rim_target = _rim_approach(executor, rim_target, centre_y, hi=0.10, mid=0.02,
                                   mid_steps=100, final_steps=150)
        if rim_target is not None and np.linalg.norm(rim_target - executor._ee()) > 0.015:
            nudge_toward_base(executor, rim_target, mag=0.025)
    # What hangs below the fingertips, read by _do_place's flat-target drop.
    executor.env._held_obj_height = shape['hang']


def _do_bowl_rim_pinch(executor, target: np.ndarray) -> None:
    rim_target = target.copy()
    rim_target[1] += 0.025
    rim_target[2] -= 0.03
    rim_target[2] = max(rim_target[2], 0.005)
    rim_target = _rim_approach(executor, rim_target, float(target[1]), hi=0.10, mid=0.02,
                               mid_steps=100, final_steps=150)
    if rim_target is not None and np.linalg.norm(rim_target - executor._ee()) > 0.015:
        nudge_toward_base(executor, rim_target, mag=0.025)
    # Held bowl height, so _do_place's flat-target drop fires (akita black bowl, typical).
    executor.env._held_obj_height = 0.05


def _do_plate_rim_pinch(executor, target: np.ndarray,
                          pts: Optional[np.ndarray] = None) -> None:
    cfg = executor.cfg
    # Rim point from the PERCEIVED extent, not a fixed offset: the
    # jaws separate along world Y, so the pinch point must sit ON the
    # rim circle along +Y (a fixed +4 cm offset lands INSIDE a ~9 cm
    # plate and closes on the flat face).  Fingertips descend to just
    # above the plate's BASE so the pads span the rim lip, not brush
    # its top.
    # Pinch at the rim SHOULDER (~70% of the perceived radius), not the
    # outer edge: a plate's outer lip is a shallow wedge that squirts
    # out of a vertical parallel-jaw close, but the dish wall where the
    # slope is steepest (for LIBERO's plate also where the thin vertical
    # collision ring lives, r~=0.048 of r_outer~=0.069) presents
    # near-vertical contact faces that the jaws hold reliably.
    # Single close + immediate lift: a verify-and-resweep (probe lift +
    # aperture read + retry) leaks grip every extra second and a second
    # close cycle pops the wedged wall out.
    rim_dy = 0.045
    rim_z = max(target[2] - 0.018, 0.005)
    if pts is not None and len(pts) > 50:
        y_ext = float(np.percentile(pts[:, 1], 98)) - float(target[1])
        if y_ext > 0.02:
            rim_dy = float(np.clip(0.70 * y_ext, 0.02, 0.08))
        z_lo = float(np.percentile(pts[:, 2], 5))
        rim_z = max(z_lo + 0.004, 0.005)
    rim_target = target.copy()
    rim_target[1] += rim_dy
    rim_target[2] = rim_z
    _rim_approach(executor, rim_target, float(target[1]), hi=0.14, mid=0.03,
                  mid_steps=150, final_steps=200, high_steps=150)
    if cfg.verbose:
        print(f"[plate pinch] target=("
              f"{rim_target[0]:.3f},{rim_target[1]:.3f},{rim_target[2]:.3f}) "
              f"rim_dy={rim_dy:.3f}")


def _do_top_down_pick(executor, target: np.ndarray) -> None:
    above = target.copy()
    above[2] += 0.08
    # Monitored approach: re-check the scene between chunks so a mid-
    # flight displacement is caught in ~0.7 s instead of at pre-close.
    executor._move_to_monitored(
        getattr(executor, 'last_pick_label', '') or '',
        above, True, steps=150)
    target = target + getattr(executor, 'monitor_shift', 0.0)   # a mid-flight retarget moved the hover point
    residual = executor._move_to(target, True, steps=200)
    # Settle-to-contact: the detected keypoint is
    # the TOP surface the camera saw - especially for wrist-only
    # detections, whose backprojected z is the object's lid.  Closing at
    # that height pinches air above tall bodies (cans / cartons: fingers
    # shut to ~1 mm just over the shoulder and the object never moves).
    # Crawl down with the jaws open until the descent stalls (palm or
    # finger contact with the object / table resists the push) or a
    # bounded extra depth, then let the grasp primitive close at BODY
    # height.
    # The crawl MUST regulate XY while it pushes down.  A bare
    # ``action[2] = -1`` leaves the OSC x/y deltas at zero, which is not a
    # position hold: robosuite re-anchors the controller goal to the
    # current pose every step, so the arm sags along its reach axis under
    # the downward push.  Measured on goal/pos-t2: 5-6 crawl steps drifted
    # the EE +2.0 to +2.3 cm in +x (away from the base) off a wine-bottle
    # neck that had been reached to within 1 mm, and the jaws then shut on
    # air (aperture 1 mm) on 29 of 30 attempts.  Holding the pre-crawl XY
    # with the same proportional term ``move_to`` uses keeps the descent
    # vertical.
    hold_xy = executor._ee()[:2].copy()
    start_z = float(executor._ee()[2])
    prev_z, stall = start_z, 0
    for _ in range(50):
        ee_now = executor._ee()
        action = np.zeros(7)
        action[:2] = np.clip((hold_xy - ee_now[:2]) * 10.0 / 0.05, -1.0, 1.0)
        action[2] = -1.0
        action[6] = -1.0
        if step_env(executor.env, action):
            break
        z = float(executor._ee()[2])
        if start_z - z >= 0.035:
            break
        if prev_z - z < 5e-4:
            stall += 1
            if stall >= 3:
                break
        else:
            stall = 0
        prev_z = z
    # How far the crawl actually took us below the detected keypoint.  The
    # place path gives this back so the crawl does not silently raise the
    # release height (see _do_place).
    executor.last_crawl_dz = max(0.0, start_z - float(executor._ee()[2]))
    if residual > 0.015:
        target = nudge_toward_base(executor, target, mag=0.025)
    if MOTION_FIX and reach_short(executor._ee(), target):   # no opposite side to try on a top-down pick
        executor.trial_meta.setdefault('reach_events', []).append(
            {'label': executor.last_pick_label, 'rim': [round(float(x), 4) for x in target],
             'ee': [round(float(x), 4) for x in executor._ee()], 'reached': False})
        fail_pick(executor, 'not_reached')
