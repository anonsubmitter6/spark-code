"""Post-condition leaves and the behavior-tree walker for ``LiberoExecutor``, on with cfg.post_conditions.

What each check reads:
  verify_grasp     the finger joint positions now (the robot's own encoders). Below HELD_MIN_M the jaws met
                   nothing. Telemetry from the close is recorded with the verdict, not used in it.
  verify_placed    the finger joints before and after the release, then one fresh frame from each of the
                   agentview and wrist cameras, no motion. It fails only on positive evidence: nothing was
                   held at the release, the jaws did not open, or the object is seen only outside the target.
                   An unseen object abstains, and an abstention counts as success.
  search_keypoint  fresh frames from where the arm is, then a wrist-camera spiral around the last binding.

None of them reads a simulator pose or the success predicate. ``simulator_truth`` does, for offline scoring only: it
writes into the event's 'truth' field, which nothing reads back during a run.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from spark_bench.dyn.bodies import find_free_joint_for_hint
from spark_bench.libero_pro.perception import redetect_agentview, redetect_wrist
from spark_bench.libero_pro.primitives._common import fuzzy_get_det, mask_points_3d
from spark_bench.libero_pro.telemetry import find_finger_qpos_addrs, read_aperture
from spark_core import bt_grammar

HELD_MIN_M = 0.0018       # an empty close ends near 1 mm; a plate rim wall is 2.5 mm
OPEN_MIN_M = 0.06         # jaws released: the Franka opens to 0.08 m
OPEN_RISE_M = 0.002
MIN_CONF = 0.25           # weaker detections are not evidence either way
FOOTPRINT_DEFAULT_M = 0.08
FOOTPRINT_MARGIN_M = 0.01
SEARCH_RADIUS_M = 0.15
SEARCH_POINTS = 8
SEARCH_HEIGHT_M = 0.20
MAX_TILT_RAD = np.pi / 2
LIFT_M = 0.10


def simulator_truth(ex) -> dict:
    """Simulator state at a check, stored for later scoring. Never returned to the executor's decisions."""
    out: dict = {}
    try:
        model, data = ex.model, ex.data
        def name(b):
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ''

        fingers: dict = {}
        for i in range(int(data.ncon)):
            c = data.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            for f, o in ((b1, b2), (b2, b1)):
                if 'finger' in name(f) and 'gripper' not in name(o) and 'finger' not in name(o):
                    fingers.setdefault(name(o), set()).add(name(f))
        out['held_bodies'] = sorted(k for k, v in fingers.items() if len(v) >= 2)
        hint = getattr(ex.cfg, '_pc_pick_hint', '')
        adr, out['pick_body'] = find_free_joint_for_hint(model, hint, mj=mujoco)
        if adr is not None:
            out['pick_xyz'] = [round(float(x), 4) for x in data.qpos[adr:adr + 3]]
        out['predicate'] = bool(ex.env.check_success())
    except Exception as e:  # noqa: BLE001 - analysis must not touch the trial
        out['truth_error'] = repr(e)[:120]
    return out


def _record_check(ex, check: str, status: str, reason: str, **kw) -> bool:
    rec = {'check': check, 'status': status, 'reason': reason, 'truth': simulator_truth(ex), **kw}
    ex.trial_meta.setdefault('pc_events', []).append(rec)
    return status != 'fail'


def aperture(ex) -> Optional[float]:
    if ex._finger_addrs is None:
        ex._finger_addrs = find_finger_qpos_addrs(ex.model)
    return read_aperture(ex.data, ex._finger_addrs)


def verify_grasp(ex, params: dict) -> bool:
    ap = aperture(ex)
    g = (ex.trial_meta.get('grasp_outcomes') or [{}])[-1].get('outcome')
    if ap is None:
        return _record_check(ex, 'verify_grasp', 'abstain', 'no finger joints', telemetry=g)
    ok = ap >= HELD_MIN_M
    if not ok:
        ex.holding = False
    return _record_check(ex, 'verify_grasp', 'pass' if ok else 'fail', 'aperture', aperture_m=round(ap, 5), telemetry=g)


def phrases_for(ex, label: str) -> set:
    """``label``, what it resolves to in the binding, and the other detection phrases of the same object."""
    from spark_bench.fair.planning import same_object_phrases
    out = {label, *same_object_phrases(label)}
    det = fuzzy_get_det(ex.det_map, label)
    key = next((k for k, v in ex.det_map.items() if v is det), None)
    if key is not None:
        out |= {key, *same_object_phrases(key)}
    return out


def fresh_instances(ex, labels: set, *, wrist: bool = True, agentview: bool = True) -> list:
    """Confident fresh detections of ``labels``, with the camera that saw each, from one frame per camera."""
    rename = getattr(ex.cfg, '_mp_phrase_to_concept', None) or {}
    out = []
    for cam, fn, on in (('agentview', redetect_agentview, agentview), ('wrist', redetect_wrist, wrist)):
        if not on:
            continue
        try:
            fresh = fn(ex.env, ex.sam3, ex.prompts_for_refresh, ex.cfg, all_instances=True)
        except Exception as e:  # noqa: BLE001 - a condition never raises; the clock may have run out
            ex.trial_meta.setdefault('pc_errors', []).append(repr(e)[:120])
            continue
        for d in (fresh.dets if fresh is not None else []):
            if (d.position_3d is not None and float(d.confidence) >= MIN_CONF
                    and rename.get(d.label, d.label) in labels):
                out.append((cam, d))
    return out


def _footprint(ex, container: str, fresh_container: list):
    """(center xy, half extent xy) of the target: fresh centroid if seen, extents from the bound mask."""
    det = fuzzy_get_det(ex.det_map, container)
    if det is None and not fresh_container:
        return None
    half = np.array([FOOTPRINT_DEFAULT_M, FOOTPRINT_DEFAULT_M])
    pts = mask_points_3d(ex, det) if det is not None else None
    if pts is not None:
        half = np.clip((np.percentile(pts[:, :2], 95, axis=0) - np.percentile(pts[:, :2], 5, axis=0)) / 2, 0.03, 0.2)
    ag = [d for cam, d in fresh_container if cam == 'agentview']
    src = ag[0] if ag else (det if det is not None else fresh_container[0][1])
    return np.asarray(src.position_3d, float)[:2], half + FOOTPRINT_MARGIN_M


def verify_placed(ex, params: dict) -> bool:
    obj = params.get('obj') or params.get('keypoint_label') or ex.last_pick_label or ''
    container = params.get('container') or params.get('target_label') or ''
    before, after = ex._aperture_at_release, aperture(ex)
    kw = {'obj': obj, 'container': container, 'aperture_before_m': before, 'aperture_after_m': after}
    if before is not None and before < HELD_MIN_M:
        return _record_check(ex, 'verify_placed', 'fail', 'nothing held at release', **kw)
    # The jaws open slowly: 1 cm a moment after a rim pinch is normal, so only no rise at all counts.
    if after is not None and before is not None and after < OPEN_MIN_M and after - before < OPEN_RISE_M:
        return _record_check(ex, 'verify_placed', 'fail', 'jaws did not open', **kw)
    if not obj or not container or ex.sam3 is None or not ex.prompts_for_refresh:
        return _record_check(ex, 'verify_placed', 'abstain', 'no labels or no perception', **kw)
    both = fresh_instances(ex, phrases_for(ex, obj) | phrases_for(ex, container))
    objs = phrases_for(ex, obj)
    fp = _footprint(ex, container, [(c, d) for c, d in both if d.label not in objs])
    seen = [(c, d) for c, d in both if d.label in objs]
    kw['seen'] = [[c, d.label, round(float(d.confidence), 2), [round(float(x), 3) for x in d.position_3d[:3]]]
                  for c, d in seen]
    if fp is None or not seen:
        return _record_check(ex, 'verify_placed', 'abstain', 'object not seen' if fp is not None else 'target unknown', **kw)
    center, half = fp
    kw['footprint'] = [[round(float(x), 3) for x in center], [round(float(x), 3) for x in half]]
    inside = [np.all(np.abs(np.asarray(d.position_3d, float)[:2] - center) <= half) for _, d in seen]
    return _record_check(ex, 'verify_placed', 'pass' if any(inside) else 'fail',
                  'seen in target' if any(inside) else 'seen only outside target', **kw)


def _rebind(ex, label: str, det, cam: str) -> None:
    pos = np.asarray(det.position_3d, float).copy()
    # A wrist mask does not fit the agentview depth the pick reads heights from.
    ex.det_map[label] = dataclasses.replace(det, position_3d=pos, mask=det.mask if cam == 'agentview' else None)
    ex.plan_binding[label] = pos.copy()
    ex.agent_ref[label] = pos.copy()


def search_keypoint(ex, params: dict) -> bool:
    """Re-perceive from here, then along a spiral at hover height around the last binding (aim: the binding,
    or the arm if there is none). The nearest fresh instance replaces the binding."""
    label = params.get('keypoint_label') or params.get('target_label') or ex.last_pick_label or ''
    if not label or ex.sam3 is None or not ex.prompts_for_refresh:
        return _record_check(ex, 'search_keypoint', 'abstain', 'no label or no perception', found=False) and False
    labels = phrases_for(ex, label)
    det = fuzzy_get_det(ex.det_map, label)
    origin = np.asarray(det.position_3d, float) if det is not None else ex._ee()

    def take(found: list, where: str) -> bool:
        cam, best = min(found, key=lambda f: float(np.linalg.norm(np.asarray(f[1].position_3d, float)[:2] - origin[:2])))
        _rebind(ex, label, best, cam)
        _record_check(ex, 'search_keypoint', 'pass', where, found=True, camera=cam, label_seen=best.label,
               new_xyz=[round(float(x), 4) for x in best.position_3d[:3]],
               moved_cm=round(100 * float(np.linalg.norm(np.asarray(best.position_3d, float)[:2] - origin[:2])), 1))
        return True

    found = fresh_instances(ex, labels)
    if found:
        return take(found, 'in view')
    z = max(float(ex._ee()[2]), float(origin[2]) + SEARCH_HEIGHT_M)
    for i in range(SEARCH_POINTS):
        r, th = SEARCH_RADIUS_M * (i + 1) / SEARCH_POINTS, 2 * np.pi * i / 4
        ex._move_to(np.array([origin[0] + r * np.cos(th), origin[1] + r * np.sin(th), z]), not ex.holding,
                    steps=60, tol=0.02)
        found = fresh_instances(ex, labels, agentview=False)
        if found:
            return take(found, f'spiral point {i + 1}')
    return _record_check(ex, 'search_keypoint', 'fail', 'not found', found=False)


def tilt_wrist(ex, params: dict) -> bool:
    """Rotate in place about a local gripper axis (x, y or z), at most 90 degrees. Not a condition."""
    axis = str(params.get('axis', 'y')).lower()
    if axis not in ('x', 'y', 'z'):
        return _record_check(ex, 'tilt_wrist', 'abstain', f'bad axis {axis!r}')
    angle = params.get('angle_rad')
    angle = float(np.deg2rad(float(params.get('angle', 0.0)))) if angle is None else float(angle)
    angle = float(np.clip(angle, -MAX_TILT_RAD, MAX_TILT_RAD))
    mujoco.mj_forward(ex.model, ex.data)
    R = ex.data.site_xmat[ex.ee_site].reshape(3, 3).copy()
    target = R @ Rotation.from_euler(axis, angle).as_matrix()
    pos_err, rot_err = ex._move_to_pose(ex._ee(), target, None, steps=100)
    return _record_check(ex, 'tilt_wrist', 'pass', 'moved', axis=axis, angle_rad=round(angle, 3),
                  rot_err_rad=round(float(rot_err), 3), pos_err_m=round(float(pos_err), 4))


CONDITIONS = {'verify_grasp': verify_grasp, 'verify_placed': verify_placed, 'search_keypoint': search_keypoint,
              'tilt_wrist': tilt_wrist}
VERDICT_LEAVES = ('verify_grasp', 'verify_placed')


class _TreeDone(Exception):
    def __init__(self, ok: bool):
        self.ok = ok


class TreeMixin:
    """Walks a score as the planner wrote it. A failed verify fails its sequence; ``retry`` and ``fallback``
    then act on that, and a failure that reaches the root goes to the runner's recovery loop."""

    _aperture_at_release: Optional[float] = None

    def _run_tree(self, score: dict) -> None:
        self._budget = bt_grammar.MAX_RECOVERY_ATTEMPTS_PER_SCORE
        self._leaf_idx = 0
        self._verdicts = 0
        stats = {'reattempts': 0, 'no_handler': []}
        self._tree_stats = stats
        try:
            ok = self._walk(score.get('tree', {}))
        except _TreeDone as d:
            ok = d.ok
        stats['ok'] = bool(ok)
        self.trial_meta.setdefault('tree_runs', []).append(stats)
        if self._verdicts:  # with no verify in the plan, the runner keeps its own evidence
            self.trial_meta['tree_ok'] = bool(ok)

    def _walk(self, node) -> bool:
        if not isinstance(node, dict):
            return True
        t = node.get('type')
        children = node.get('children', []) or []
        if t == 'sequence':
            return self._walk_sequence(children)
        if t in bt_grammar.RETRY_TYPES:
            for attempt in range(bt_grammar.clamp_retry_attempts((node.get('params') or {}).get('max_attempts'))):
                if attempt and not self._next_attempt():
                    return False
                if self._walk_sequence(children):
                    return True
            return False
        if t in bt_grammar.SELECTOR_TYPES:
            for i, child in enumerate(children[:bt_grammar.MAX_SELECTOR_BRANCHES]):
                if i and not self._next_attempt():
                    return False
                if self._walk(child):
                    return True
            return not children
        return self._leaf(node)

    def _walk_sequence(self, children) -> bool:
        return all(self._walk(c) for c in children)

    def _next_attempt(self) -> bool:
        """Spend one re-attempt: the arm opens if it holds nothing, lifts clear, and the next keypoint re-detects."""
        if self._budget <= 0:
            return False
        self._budget -= 1
        self._tree_stats['reattempts'] += 1
        self.holding = False
        ee = self._ee()
        ee[2] += LIFT_M
        self._move_to(ee, True, steps=100, tol=0.02)
        self._last_perception_t = 0.0
        return True

    def _leaf(self, node: dict) -> bool:
        if self._goal_satisfied():
            raise _TreeDone(True)
        t = node.get('type')
        idx = self._leaf_idx
        self._leaf_idx += 1
        fn = CONDITIONS.get(t)
        if fn is None:
            from spark_bench.libero_pro.executor import _HANDLERS
            if t not in _HANDLERS:
                self._tree_stats['no_handler'].append(t)
            if t == 'release':
                self._aperture_at_release = aperture(self)
            if not self.dispatch_action(idx, node) or self.abort is not None:
                raise _TreeDone(False)
            # SPARK_MOTION_FIX: not reached / jaws met; retry and fallback nodes act on it
            if getattr(self, 'pick_failed', None):
                self.pick_failed = None
                return False
            return True
        trace = getattr(self.cfg, '_trace', None)
        if trace is not None:
            trace.enter_note(idx, t, (node.get('params') or {}).get('keypoint_label'))
        ok = fn(self, node.get('params') or {})
        if t in VERDICT_LEAVES:
            self._verdicts += 1
        return ok
