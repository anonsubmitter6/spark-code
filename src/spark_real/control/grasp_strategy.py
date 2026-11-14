"""Planner-directed grasp orientation.

A plan may name ``grasp_strategy`` (and an explicit ``grasp_yaw_deg``) on
``move_to_keypoint`` or ``grasp``; absent both, ``auto`` applies the
``aspect_ratio >= GRASP_YAW_AR_GATE`` test plus input-validity preconditions.

Every yaw is checked against the wrist-3 cable limit AFTER composition, on
the rotation that will actually be commanded, not on the requested angle.
"""


from scipy.spatial.transform import Rotation
import logging
import math
import numpy as np
import os


from spark_real.perception.detection_fusion import measure_mask


logger = logging.getLogger(__name__)

# "cgn" and "se3" name 6-DOF backends the frozen planner prompt still lists;
# they are not in this release and resolve to "auto" with a warning.
STRATEGIES = ("auto", "topdown", "obb", "cgn", "se3")
UNAVAILABLE_STRATEGIES = ("cgn", "se3")
DEFAULT_STRATEGY = "auto"

# Input-validity floors for the `auto` route. A yaw read off a 24%-confidence
# blob is noise, not policy. Config: grasp.yaw_min_conf / grasp.yaw_min_obb_conf.
DEFAULT_YAW_MIN_CONF = 0.45
DEFAULT_YAW_MIN_OBB_CONF = 0.40

# Fallback wrist-3 cable limit when the executor carries none.
DEFAULT_MAX_YAW_OFFSET_RAD = math.radians(100.0)  # see executor_motion.MAX_YAW_OFFSET

# A "top-down" orientation must keep tool-Z this close to straight down.
TOPDOWN_TILT_TOL_DEG = 5.0

_WORLD_DOWN = np.array([0.0, 0.0, -1.0])


from spark_core.util import det_field as _field  # noqa: E402


def _tuned(executor, attr: str, env: str, default: float) -> float:
    """env override > executor attribute > module default."""
    raw = os.environ.get(env)
    if raw is None:
        raw = getattr(executor, attr, None)
    try:
        return float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not a number; using %.2f", env, raw, default)
        return float(default)


def yaw_min_conf(executor) -> float:
    return _tuned(executor, "GRASP_YAW_MIN_CONF", "SPARK_YAW_MIN_CONF",
                  DEFAULT_YAW_MIN_CONF)


def yaw_min_obb_conf(executor) -> float:
    return _tuned(executor, "GRASP_YAW_MIN_OBB_CONF", "SPARK_YAW_MIN_OBB_CONF",
                  DEFAULT_YAW_MIN_OBB_CONF)


def _live_fusion_gate(executor):
    """The pipeline's fusion gate, or None when fusion is switched off."""
    gate = getattr(getattr(executor, "_pipeline", None), "_fusion_gate", None)
    return gate if getattr(gate, "cfg", None) is not None else None


def axis_trust(detection, executor):
    """(value, source) for the mask-measured OBB axis trust, else (None, "").

    Prefers the value the perception gate already stamped. Falls back to
    measuring the detection's own `_mask` (click/box detections never pass
    the perception seam). Only runs when fusion is on.
    """
    v = _field(detection, "axis_trust", None)
    if v is not None:
        try:
            return float(v), "axis_trust"
        except (TypeError, ValueError):
            logger.warning("[grasp-strategy] axis_trust=%r is not a number", v)
            return None, ""

    gate = _live_fusion_gate(executor)
    if gate is None:
        return None, ""
    mask = _field(detection, "_mask", None)
    if mask is None:
        mask = _field(detection, "mask", None)
    if mask is None:
        return None, ""
    try:
        q = measure_mask(
            mask,
            reported_aspect_ratio=_field(detection, "aspect_ratio", None),
            th=gate.cfg.quality,
        )
    except Exception as exc:  # noqa: BLE001 - never let scoring stop a grasp
        logger.warning("[grasp-strategy] mask axis measurement failed (%s)", exc)
        return None, ""
    return float(q.axis_trust), "axis_trust(mask)"


def max_yaw_offset(executor) -> float:
    """Wrist-3 cable limit in radians (grasp.max_yaw_offset_deg)."""
    lim = getattr(executor, "MAX_YAW_OFFSET", None)
    try:
        lim = float(lim)
    except (TypeError, ValueError):
        lim = None
    return lim if lim and lim > 0 else DEFAULT_MAX_YAW_OFFSET_RAD


def base_orientation(executor) -> np.ndarray:
    return np.asarray(
        getattr(executor, "GRASP_ORIENTATION", [np.pi, 0.0, 0.0]), dtype=float
    )


def symmetric_yaw(yaw: float) -> float:
    """Reduce a two-jaw-symmetric yaw into (-pi/2, pi/2]."""
    y = float(yaw) % np.pi
    if y > np.pi / 2:
        y -= np.pi
    return y


def measured_yaw_offset(orientation, base) -> float:
    """World-Z yaw of `orientation` relative to `base`, from the matrices."""
    rel = (
        Rotation.from_rotvec(np.asarray(orientation, dtype=float))
        * Rotation.from_rotvec(np.asarray(base, dtype=float)).inv()
    ).as_matrix()
    return float(np.arctan2(rel[1, 0], rel[0, 0]))


def tool_z_tilt_deg(orientation) -> float:
    """Angle (deg) between the tool Z axis and straight down."""
    tool_z = Rotation.from_rotvec(np.asarray(orientation, dtype=float)).as_matrix()[:, 2]
    return float(np.rad2deg(np.arccos(np.clip(np.dot(tool_z, _WORLD_DOWN), -1.0, 1.0))))


def compose_yaw(executor, yaw: float, context: str = "grasp"):
    """Compose a world-Z yaw onto GRASP_ORIENTATION, cable limit enforced.

    Returns the rotvec list, or None when the COMMANDED rotation exceeds the
    limit (caller falls back to top-down). The limit is measured on the
    composed rotation, so a strategy that supplies a full pose is checked the
    same way an OBB yaw is.
    """
    orient = executor._oriented_grasp(float(yaw))
    if not check_yaw_limit(executor, orient, context):
        return None
    # _oriented_grasp CLIPS to the limit. A clipped yaw is a crooked grasp, so
    # refuse it and let the caller command a defined top-down instead.
    measured = measured_yaw_offset(orient, base_orientation(executor))
    if abs(measured - float(yaw)) > np.deg2rad(1.0):
        logger.warning(
            "[grasp-strategy] %s: requested yaw %.1f deg was clipped to %.1f "
            "by the wrist-3 cable limit -> refusing (top-down instead)",
            context, np.rad2deg(float(yaw)), np.rad2deg(measured),
        )
        return None
    return orient


def check_yaw_limit(executor, orientation, context: str = "grasp") -> bool:
    """True when `orientation` is within the wrist-3 cable envelope."""
    limit = max_yaw_offset(executor)
    measured = measured_yaw_offset(orientation, base_orientation(executor))
    if abs(measured) > limit + 1e-6:
        logger.warning(
            "[grasp-strategy] %s: composed yaw %.1f deg exceeds the wrist-3 "
            "cable limit %.1f deg -> refusing this orientation",
            context, np.rad2deg(measured), np.rad2deg(limit),
        )
        return False
    return True


def resolve_strategy(params, detection, executor):
    """Return (strategy, reason). Never raises, never returns 'auto'."""
    params = params or {}
    requested = params.get("grasp_strategy")
    forced = os.environ.get("SPARK_GRASP_STRATEGY")
    if forced:
        requested = forced
    requested = str(requested or DEFAULT_STRATEGY).strip().lower()
    if requested not in STRATEGIES:
        logger.warning(
            "[grasp-strategy] unknown grasp_strategy %r; using auto", requested
        )
        requested = DEFAULT_STRATEGY

    # Loud, not a silent no-op: the planner asked for a 6-DOF backend and is
    # not getting it.
    if requested in UNAVAILABLE_STRATEGIES:
        logger.warning(
            "[grasp-strategy] planner asked for %r, which this release does not "
            "ship; falling back to auto", requested,
        )
        requested = DEFAULT_STRATEGY

    if requested != DEFAULT_STRATEGY:
        return requested, "planner"

    # auto: AR gate plus input-validity preconditions.
    if detection is None:
        return "topdown", "no detection"
    if bool(_field(detection, "low_quality", False)):
        # low_quality covers two claims: "mask is round" (no axis, must veto)
        # and "world OBB is inflated" (a depth claim that flips run to run on
        # shiny tools). Refuse only when the MASK says there is no axis; when
        # only the depth-derived OBB is suspect, use the ray-plane axis.
        if not bool(_field(detection, "shape_round", False)):
            plane = _plane_axis(detection, gate_for(executor))
            if plane is not None:
                return (
                    "plane",
                    f"low_quality is depth-driven (ar_inflated="
                    f"{bool(_field(detection, 'ar_inflated', False))}), mask ar "
                    f"{plane[1]:.2f} still has an axis -> ray-plane",
                )
        return "topdown", "mask flagged low_quality"

    ar = float(_field(detection, "aspect_ratio", 1.0) or 1.0)
    gate = float(getattr(executor, "GRASP_YAW_AR_GATE", 1.8))
    if ar < gate:
        # `aspect_ratio` is the WORLD OBB's, from the depth cloud, which a
        # thin specular tool can deflate (e.g. 1.40 against a 1.60 gate). Ask
        # the mask before giving up; a round object fails both.
        plane = _plane_axis(detection, gate)
        if plane is not None:
            return (
                "plane",
                f"world ar {ar:.2f} < gate {gate:.2f}, but the mask ar is "
                f"{plane[1]:.2f} -> ray-plane axis",
            )
        return "topdown", f"ar {ar:.2f} < gate {gate:.2f}"

    # Prefer the fusion gate's `fused_confidence` over SAM3's own score.
    conf, conf_src = _field(detection, "fused_confidence", None), "fused_conf"
    if conf is None:
        conf, conf_src = _field(detection, "confidence", None), "conf"
    min_conf = yaw_min_conf(executor)
    if conf is not None and float(conf) < min_conf:
        return "topdown", f"{conf_src} {float(conf):.2f} < {min_conf:.2f} (ar {ar:.2f})"

    min_obb = yaw_min_obb_conf(executor)
    # Two independent OBB-axis measurements: obb_confidence (depth) and
    # axis_trust (mask alone). Either may veto. Absent is UNKNOWN, not zero.
    obb_sources = {"obb_conf": _field(detection, "obb_confidence", None)}
    trust, trust_src = axis_trust(detection, executor)
    if trust is not None:
        obb_sources[trust_src] = trust
    for name, value in obb_sources.items():
        if value is not None and float(value) < min_obb:
            # The point-cloud OBB is untrustworthy, but that is a DEPTH claim.
            # The ray-plane axis (silhouette plus one centroid depth) survives
            # thin, shiny, low-return objects such as silverware and tools.
            plane = _plane_axis(detection, gate)
            if plane is not None:
                return (
                    "plane",
                    f"{name} {float(value):.2f} < {min_obb:.2f}, "
                    f"falling back to the ray-plane axis (ar {plane[1]:.2f})",
                )
            return "topdown", f"{name} {float(value):.2f} < {min_obb:.2f}"

    return "obb", f"ar {ar:.2f} >= gate {gate:.2f}, {conf_src} ok"


def gate_for(executor) -> float:
    """The elongation gate an oriented grasp must clear."""
    return float(getattr(executor, "GRASP_YAW_AR_GATE", 1.8))


def _plane_axis(detection, gate: float):
    """(angle_rad, mask_ar) from the ray-plane axis, or None if unusable.

    Requires the MASK's own aspect ratio to clear the same elongation gate the
    OBB path uses. A round mask has no axis to grasp by whatever the depth
    said, so this must not resurrect one.
    """
    angle = _field(detection, "plane_orientation_angle", None)
    if angle is None:
        return None
    ar = _field(detection, "plane_aspect_ratio", None)
    if ar is None:
        return None
    try:
        angle_f, ar_f = float(angle), float(ar)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(angle_f) or not np.isfinite(ar_f) or ar_f < gate:
        return None
    return angle_f, ar_f


def _record_held_axis(detection, executor, params=None):
    """Remember which WAY the object about to be grasped points.

    A PCA axis is a line, so it cannot say which end of a screwdriver is the
    tip. The place compares the held object's heavy end against the
    receptacle's to avoid seating the tool end-for-end. Recorded here because
    resolve_grasp_orientation is the one funnel every grasp path passes through.
    """
    executor._held_mask = None
    executor._held_axis_img = None
    executor._held_axis_sign = None
    try:
        import numpy as _np

        from spark_core.mask_obb import _pca_obb
        from spark_real.perception.masks import heavy_end_sign

        mask = _field(detection, "_mask", None)
        src = _field(detection, "label", None) or (
            (params or {}).get("keypoint_label") if isinstance(params, dict) else None
        )
        # Direction lives in the WHOLE tool, not the grasped sub-part: a
        # screwdriver handle is symmetric end-for-end, only the shaft breaks
        # the tie.
        try:
            from spark_real.control.grasp import _parent_object_label

            parent = _parent_object_label(src, getattr(executor, "detection_map", None))
            if parent is not None:
                pm = (executor.detection_map.get(parent) or {}).get("_mask")
                if pm is not None:
                    mask, src = pm, parent
        except Exception:  # noqa: BLE001 - sub-part upgrade is best-effort
            pass

        if mask is None:
            logger.info(
                "[held-axis] '%s' has no mask; the place cannot tell its ends "
                "apart and will fall back to the undirected axis",
                src,
            )
            return

        arr = _np.asarray(mask)
        ang, ar, _, _ = _pca_obb(arr)
        sgn = float(heavy_end_sign(arr, ang))
        executor._held_mask = arr
        executor._held_axis_img = float(ang)
        executor._held_axis_sign = sgn
        logger.info(
            "[held-axis] '%s' image axis %.1f deg, ar %.2f, heavy end %s",
            src, _np.rad2deg(ang), ar,
            {1.0: "+", -1.0: "-", 0.0: "UNKNOWN (too symmetric to call)"}.get(sgn, "?"),
        )
    except Exception as exc:  # noqa: BLE001 - never let this stop a grasp
        logger.warning("[held-axis] could not record the held mask: %s", exc)


def place_aware_yaw(executor, yaw, context):
    """Choose between yaw and yaw+-180 so the PLACE that follows is reachable.

    A parallel gripper closes identically at both, but the choice fixes which
    way the held object points and hence the wrist angle the place will need
    (a bad branch can push the place past the wrist limit and seat
    the object end-for-end).

    Returns the chosen yaw, or the input unchanged whenever the preview
    cannot be computed (no pending place, no masks, fit undecided). It never
    picks an unreachable branch over a reachable one, and when both are
    reachable it defers to the caller's least-travel choice.
    """
    if context != "grasp":
        return yaw
    label = getattr(executor, "_pending_place_label", None)
    held = getattr(executor, "_held_mask", None)
    if not label or held is None:
        return yaw
    try:
        target = ((getattr(executor, "detection_map", None) or {}).get(label) or {}).get(
            "_mask"
        )
        if target is None:
            return yaw
        from spark_real.perception.masks import best_fit_rotation

        ang, iou, margin = best_fit_rotation(np.asarray(held), np.asarray(target))
        if ang is None:
            return yaw
        min_margin = float(getattr(executor, "JIGSAW_MIN_MARGIN", 0.05))
        min_iou = float(getattr(executor, "JIGSAW_MIN_IOU", 0.20))
        if margin < min_margin or iou < min_iou:
            return yaw  # the fit will not steer the place either; nothing to preview
        limit = float(gate_yaw_limit(executor))

        def _place_yaw(branch):
            w = branch + float(ang)
            return (w + np.pi) % (2 * np.pi) - np.pi

        alt = yaw + (np.pi if yaw < 0 else -np.pi)
        alt = (alt + np.pi) % (2 * np.pi) - np.pi
        here_ok = abs(_place_yaw(yaw)) <= limit + 1e-9
        alt_ok = abs(_place_yaw(alt)) <= limit + 1e-9
        if here_ok or not alt_ok:
            return yaw  # already fine, or neither branch helps
        logger.warning(
            "[place-aware grasp] grasping at %.1f deg would need %.1f deg at "
            "the place (limit %.0f) and seat the object end-for-end; taking "
            "the equivalent %.1f deg branch instead, whose place is %.1f deg",
            np.rad2deg(yaw), np.rad2deg(_place_yaw(yaw)), np.rad2deg(limit),
            np.rad2deg(alt), np.rad2deg(_place_yaw(alt)),
        )
        return alt
    except Exception as exc:  # noqa: BLE001 - a failed preview keeps the old choice
        logger.warning("[place-aware grasp] preview failed (%s); keeping %.1f deg",
                       exc, np.rad2deg(yaw))
        return yaw


def gate_yaw_limit(executor) -> float:
    """The wrist-3 yaw budget actually in force (config overrides the class)."""
    return float(getattr(executor, "MAX_YAW_OFFSET", DEFAULT_MAX_YAW_OFFSET_RAD))

def resolve_grasp_orientation(
    params, detection, executor, yaw_override_rad=None, context="grasp"
):
    """Return (orientation_rotvec, strategy_used).

    Pure except for the executor's current-TCP read inside
    _nearest_symmetric_yaw. `yaw_override_rad` is a caller-supplied world yaw
    (wrist refinement, slot direction) that replaces the detection OBB angle.
    """
    if context == "grasp":
        _record_held_axis(detection, executor, params)
    base = list(base_orientation(executor))
    try:
        strategy, reason = resolve_strategy(params, detection, executor)
    except Exception as exc:  # noqa: BLE001 - never let strategy math stop motion
        logger.warning("[grasp-strategy] resolve failed (%s); top-down", exc)
        return base, "topdown"

    # A caller-supplied yaw (slot geometry, wrist refinement) or the planner's
    # grasp_yaw_deg OUTRANKS every veto below: resolve_strategy's vetoes are
    # statements about the HELD OBJECT's mask and depth, and a slot's
    # direction is a property of the tool bed, not of the object.
    explicit_yaw = yaw_override_rad
    if explicit_yaw is None:
        _pyaw = (params or {}).get("grasp_yaw_deg")
        if _pyaw is not None:
            try:
                explicit_yaw = np.deg2rad(float(_pyaw))
            except (TypeError, ValueError):
                explicit_yaw = None

    # Only when the strategy would otherwise DROP the yaw; the "obb" branch
    # below consumes grasp_yaw_deg itself.
    if explicit_yaw is not None and strategy == "topdown":
        try:
            yaw_o = float(explicit_yaw)
        except (TypeError, ValueError):
            yaw_o = None
        if yaw_o is not None and np.isfinite(yaw_o):
            yaw_o = symmetric_yaw(yaw_o)
            if hasattr(executor, "_nearest_symmetric_yaw"):
                yaw_o = executor._nearest_symmetric_yaw(yaw_o)
            orient_o = compose_yaw(executor, yaw_o, context=context)
            if orient_o is not None:
                logger.info(
                    "[grasp-strategy] %s: caller yaw=%.1f deg (slot/refinement) "
                    "overrides strategy '%s' (%s)",
                    context, np.rad2deg(yaw_o), strategy, reason,
                )
                return orient_o, "caller_yaw"
            logger.info(
                "[grasp-strategy] %s: caller yaw refused by the wrist limit; "
                "falling through to '%s'", context, strategy,
            )

    if strategy == "topdown" and context == "place":
        # PLACING is not GRASPING. resolve_strategy's vetoes are about closing
        # jaws ON the object (fill, solidity, fragmentation, depth agreement),
        # not about which WAY the target lies. If the target is ELONGATED,
        # align to it, preferring the mask's aspect ratio over the
        # depth-derived one, and track which field supplied each number for
        # the log.
        _ar = _ar_src = None
        for _k in ("mask_aspect_ratio", "plane_aspect_ratio", "aspect_ratio"):
            _v = _field(detection, _k, None)
            if _v is not None:
                _ar, _ar_src = _v, _k
                break
        _axis = _axis_src = None
        for _k in ("plane_orientation_angle", "world_major_axis_rad", "orientation_angle"):
            _v = _field(detection, _k, None)
            if _v is not None:
                _axis, _axis_src = _v, _k
                break
        try:
            _ar_f = float(_ar) if _ar is not None else None
            _axis_f = float(_axis) if _axis is not None else None
        except (TypeError, ValueError):
            _ar_f = _axis_f = None
        if (
            _ar_f is not None
            and _axis_f is not None
            and np.isfinite(_ar_f)
            and np.isfinite(_axis_f)
            and _ar_f >= gate_for(executor)
        ):
            yaw_t = symmetric_yaw(_axis_f)
            if hasattr(executor, "_nearest_symmetric_yaw"):
                yaw_t = executor._nearest_symmetric_yaw(yaw_t)
            orient_t = compose_yaw(executor, yaw_t, context=context)
            if orient_t is not None:
                logger.info(
                    "[grasp-strategy] place: aligning to the TARGET's axis "
                    "yaw=%.1f deg [axis from %s] (target ar %.2f [from %s]); "
                    "'%s' was a grasp-quality veto and does not describe the "
                    "target's direction",
                    np.rad2deg(yaw_t), _axis_src, _ar_f, _ar_src, reason,
                )
                return orient_t, "target_axis"

    if strategy == "topdown":
        if context == "place":
            logger.info(
                "[grasp-strategy] place: topdown (%s); target-axis fallback "
                "declined: ar=%s [from %s] gate=%.2f, axis=%s [from %s]",
                reason,
                (f"{float(_ar):.2f}" if _ar is not None else "none"),
                _ar_src,
                gate_for(executor),
                (f"{float(_axis):.3f}rad" if _axis is not None else "none"),
                _axis_src,
            )
            return base, "topdown"
        logger.info("[grasp-strategy] %s: topdown (%s)", context, reason)
        return base, "topdown"

    if strategy == "plane":
        plane = _plane_axis(detection, float(getattr(executor, "GRASP_YAW_AR_GATE", 1.8)))
        if plane is None:
            logger.info("[grasp-strategy] %s: plane axis vanished -> topdown", context)
            return base, "topdown"
        yaw = symmetric_yaw(float(plane[0]))
        if hasattr(executor, "_nearest_symmetric_yaw"):
            yaw = executor._nearest_symmetric_yaw(yaw)
        orient = compose_yaw(executor, yaw, context=context)
        if orient is None:
            return base, "topdown"
        logger.info(
            "[grasp-strategy] %s: plane yaw=%.1f deg from the ray-plane axis (%s)",
            context, np.rad2deg(yaw), reason,
        )
        return orient, "plane"

    # obb
    yaw_deg = (params or {}).get("grasp_yaw_deg")
    if yaw_deg is not None:
        yaw = np.deg2rad(float(yaw_deg))
        src = f"grasp_yaw_deg={float(yaw_deg):.1f}"
    elif yaw_override_rad is not None:
        yaw = float(yaw_override_rad)
        src = "caller yaw"
    else:
        raw = _field(detection, "orientation_angle", None)
        if raw is None:
            logger.info(
                "[grasp-strategy] %s: obb requested but the detection has no "
                "orientation_angle -> topdown", context,
            )
            return base, "topdown"
        yaw = float(raw)
        src = "detection OBB"

    yaw = symmetric_yaw(yaw)
    if hasattr(executor, "_nearest_symmetric_yaw"):
        yaw = executor._nearest_symmetric_yaw(yaw)
    # Least-travel chose a branch knowing only the grasp; let the destination
    # override it when that would strand the place past the wrist limit.
    yaw = place_aware_yaw(executor, yaw, context)
    orient = compose_yaw(executor, yaw, context=context)
    if orient is None:
        return base, "topdown"
    logger.info(
        "[grasp-strategy] %s: obb yaw=%.1f deg from %s (%s)",
        context, np.rad2deg(yaw), src, reason,
    )
    return orient, "obb"




class ApproachMixin:
    """Approach to a grasp target and the Cartesian servo move."""

    def _approach_target(self, target: np.ndarray, detection: dict = None, params: dict = None):
        """
        Approach target: lift, position above, wrist refine, servo descend.

        `params` are the plan node's params; only grasp_strategy /
        grasp_yaw_deg are read. Omitting them is the auto route.
        """
        self._check_abort()
        if not self._holding:
            self._ensure_jaws_open("pre-approach")
        # Orientation source is the planner's `grasp_strategy` when it named
        # one, otherwise the auto route (AR gate + input-validity floors).
        orient, strategy = resolve_grasp_orientation(
            params or {}, detection, self, context="approach"
        )
        self._active_grasp_strategy = strategy
        self._active_grasp_label = self._last_keypoint_label or ""
        adjusted = target.copy()
        _gc_label = self._last_keypoint_label or ""
        _gc_learned = (
            self.grasp_calibration.learned_offset(_gc_label)
            if self.grasp_calibration is not None
            else None
        )
        _gc_initial = _gc_learned if _gc_learned is not None else self.GRIPPER_OPEN_Z_OFFSET
        adjusted[2] += _gc_initial
        if _gc_learned is not None:
            logger.info(
                "Grasp learned-offset for %r: %.3f m (%d samples)",
                _gc_label,
                _gc_learned,
                self.grasp_calibration._stats.get(
                    _gc_label.lower().rstrip("0123456789 "), type("_", (), {"n": 0})()
                ).n,
            )
        self._grasp_perception_target_z = float(target[2])
        # Per-pick reset: a stale close-height from the PREVIOUS pick must
        # never feed this pick's release give-back.
        self._actual_grasp_tcp_z = None

        current = self._get_current_position()
        safe_z = self._safe_clearance_z(current, adjusted)
        label = self._last_keypoint_label or ""
        # The descent target only exists AFTER a wrist capture, so a phrase
        # can only absorb it when refinement is off (the default here).
        refine_pending = bool(
            self._wrist_refine_enabled and label and self._wrist_refine_mode() != "off"
        )
        # A UR10e transit holds yaw: lift at the current yaw, arrive at the grasp yaw, or the
        # wrist unwinds to yaw 0 and back. A wrist capture must stay at GRASP_ORIENTATION
        # (WRIST_CAM_OFFSET_DEG assumes it). ponytail: FR3 still unwinds, it is unvalidated there.
        hold_yaw = self._robot_family() == "ur10e" and not refine_pending
        lift_orient = self.current_orientation() if hold_yaw else list(self.GRASP_ORIENTATION)
        above_orient = list(orient) if hold_yaw else list(self.GRASP_ORIENTATION)
        transit = []
        if np.linalg.norm(adjusted - current) > 0.5:
            lift = current.copy()
            lift[2] = safe_z
            transit.append((lift, lift_orient, None, "lift"))
            above = adjusted.copy()
            above[2] = safe_z
            transit.append((above, above_orient, None, "above"))
        else:
            above = adjusted.copy()
            above[2] = max(adjusted[2] + 0.10, safe_z)
            transit.append((above, above_orient, None, "above"))

        if self._blend_enabled():
            phrase = list(transit)
            with_descent = not refine_pending
            if with_descent:
                phrase.append(
                    (
                        adjusted.copy(),
                        list(orient),
                        self.velocity * 0.9,
                        "grasp_descent",
                    )
                )
            if self._blend_transit(phrase, phrase="approach"):
                if with_descent:
                    # Terminal row was r=0: the arm IS stopped at the grasp pose.
                    return
                transit = []
        for pos_i, ori_i, _v, _lbl in transit:
            self._move_to(pos_i, ori_i)

        # Wrist-camera refinement, gated by control.wrist_refine (off by
        # default); when off the servo goes straight to the birdview/sideview
        # target.
        if label:
            refined = self._refine_with_wrist(
                label, adjusted, z_plane=self._grasp_perception_target_z
            )
            if refined is not None:
                new_pos = np.array(refined["position_3d"])
                xy_diff = np.linalg.norm(new_pos[:2] - adjusted[:2])
                if xy_diff < 0.05:
                    adjusted[:2] = new_pos[:2]
                    # ray_plane's Z IS the prior it was handed; re-applying it
                    # would add the gripper offset twice. Only the depth path
                    # can correct Z.
                    if refined.get("source") != "ray_plane" and (
                        abs(new_pos[2] - adjusted[2]) < 0.03
                    ):
                        adjusted[2] = new_pos[2] + self.GRIPPER_OPEN_Z_OFFSET
                    logger.info("Wrist XY refine: delta=%.3fm", xy_diff)
                    # The fix is the same object, so its geometry applies too. Past the
                    # gate it is some other object, and so is its yaw.
                    # Re-resolve with the wrist's own AR/angle, keeping the source
                    # detection's confidence: same strategy rules, fresher geometry.
                    img_deg = np.rad2deg(refined["orientation_angle"])
                    yaw_deg = (img_deg % 180) - self.WRIST_CAM_OFFSET_DEG
                    if yaw_deg > 90:
                        yaw_deg -= 180
                    if yaw_deg < -90:
                        yaw_deg += 180
                    wrist_det = dict(detection or {})
                    wrist_det["aspect_ratio"] = refined["aspect_ratio"]
                    wrist_det["orientation_angle"] = refined["orientation_angle"]
                    wrist_orient, wrist_strategy = resolve_grasp_orientation(
                        params or {},
                        wrist_det,
                        self,
                        yaw_override_rad=np.deg2rad(yaw_deg),
                        context="wrist-refine",
                    )
                    if wrist_strategy == "obb":
                        orient = wrist_orient
                        self._active_grasp_strategy = wrist_strategy
                        logger.info(
                            "Wrist grasp: img=%.1f deg -> yaw=%.1f deg (ar=%.1f)",
                            img_deg,
                            yaw_deg,
                            refined["aspect_ratio"],
                        )
                else:
                    logger.info("Wrist XY too far (%.3fm), keeping birdview", xy_diff)

        # Persist the (possibly OBB-yawed) orientation so the grasp primitive's
        # descent/clamp/lift reuse it instead of reverting to GRASP_ORIENTATION
        # right before the jaws close. See _grasp_v2.
        self._active_grasp_orient = list(orient)
        # Remember which WAY the object points, so the place can put its heavy
        # end where the receptacle's heavy end is (a PCA axis is only a line).
        self._held_axis_sign = None
        self._held_axis_img = None
        # Clear the mask WITH the sign, or an approach without a detection
        # (recovery's retract_retry) registers the PREVIOUS pick's silhouette.
        self._held_mask = None
        try:
            _m = detection.get("_mask") if isinstance(detection, dict) else None
            if _m is not None:
                from spark_core.mask_obb import _pca_obb
                from spark_real.perception.masks import heavy_end_sign

                _a, _ar, _, _ = _pca_obb(np.asarray(_m))
                _sgn = heavy_end_sign(np.asarray(_m), _a)
                self._held_axis_img = float(_a)
                self._held_axis_sign = float(_sgn)
                self._held_mask = np.asarray(_m)
                logger.info(
                    "[held-axis] image axis %.1f deg, heavy end %s (ar %.2f)",
                    np.rad2deg(_a),
                    {1.0: "+", -1.0: "-", 0.0: "UNKNOWN"}.get(_sgn, "?"),
                    _ar,
                )
        except Exception as exc:  # noqa: BLE001 - never let this stop a grasp
            logger.warning("[held-axis] could not measure: %s", exc)
        # Final oriented descent as a JOINT move (pyroki IK -> movej): smooth,
        # fast, hits the yaw exactly. The Cartesian PD servo is jittery near
        # convergence and slow; fall back to it only if the joint-IK path is
        # unavailable or fails.

        if self._robot_family() == "ur10e":
            try:
                self._movej_via_ik(adjusted, orient, self.velocity * 0.9)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[approach] joint-IK descend failed (%s); Cartesian-servo fallback",
                    exc,
                )
        self._servo_to(adjusted, orient, velocity=self.velocity * 0.9)

    def _servo_to(self, position: np.ndarray, orientation: list, velocity: float = 0.15) -> bool:
        """
        Servo to a target pose with precise orientation control.

        Returns False when the servo STALLED (gave up against an unreachable
        target, no movel retry issued, TCP parked where it got stuck). True
        otherwise (converged, or a timeout the movel fallback finished). The
        place descent must not treat a stall as an arrival.
        """
        self._check_abort()
        target_pose = list(position) + list(orientation)
        self._servo.max_vel_linear = min(velocity, 0.25)
        converged = self._servo.move_to_pose(target_pose, velocity=velocity)
        self._check_abort()
        stalled = False
        if not converged:
            # A STALL means the servo held a command into an unreachable
            # target. A movel to that same pose would drive into the identical
            # obstruction, risking a protective stop. Accept the current pose
            # and let the caller's verification decide. A TIMEOUT is different:
            # the servo was still closing, so movel can finish the job.
            if getattr(self._servo, "last_exit", "") == "stalled":
                stalled = True
                logger.warning(
                    "Servo stalled at pos_err=%.4f ori_err=%.4f; NOT retrying "
                    "with movel (same unreachable target)",
                    self._servo.last_pos_err,
                    self._servo.last_ori_err,
                )
            else:
                logger.warning("Servo did not converge, falling back to movel")
                self._move_to_linear(position, orientation, velocity=velocity)
        # Hand the controller back before anything else uploads a program.
        # The servo streams servoL commands and OWNS the controller until it is
        # stopped; a gripper command is a URScript upload, and uploading one
        # while the servo still holds the channel silently loses it (jaw
        # register 175.95 -> 175.95, zero travel).
        try:
            if hasattr(self.robot, "servo_stop"):
                self.robot.servo_stop()
        except Exception as exc:  # noqa: BLE001 - never let cleanup stop motion
            logger.warning("servo_stop after servo failed: %s", exc)

        actual = self._servo._get_tcp_pose()
        actual_rotvec = actual[3:6]
        R_actual = Rotation.from_rotvec(actual_rotvec).as_matrix()
        tool_x_world = R_actual @ np.array([1, 0, 0])
        tool_y_world = R_actual @ np.array([0, 1, 0])
        logger.info(
            "Servo result: tool_X=%.1f deg, tool_Y=%.1f deg (closing axis)",
            np.rad2deg(np.arctan2(tool_x_world[1], tool_x_world[0])),
            np.rad2deg(np.arctan2(tool_y_world[1], tool_y_world[0])),
        )
        return not stalled
