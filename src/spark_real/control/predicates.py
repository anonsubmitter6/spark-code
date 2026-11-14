"""Parse a plan's verify block, or derive a default one from the actions it ran."""


from __future__ import annotations


from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple
from typing import Any, Dict, List, Optional, Tuple
import logging
import math
import numpy as np
import time


from spark_core.util import as_bool, det_field
from spark_core.util import det_field
from spark_real.bt_library import LabelResolvingDetectionMap, canon_label
from spark_real.control.common import _LABEL_KEYS, _REFERENCE_KEYS, _SCHEMA, STACKED_DZ_MIN_M, Predicate, VerifyBlockError, VerifySpec
from spark_real.control.common import _SCHEMA, STACKED_DZ_MIN_M, Predicate, VerifyBlockError, VerifySpec
from spark_real.control.common import xy_inside_container


def _norm_bool(val: Any) -> bool:
    if isinstance(val, str):
        return as_bool(val, False)
    return bool(val)


def parse_verify_block(
    block: Any,
    *,
    label_resolver: Optional[Callable[[str], Optional[str]]] = None,
    default_min_conf: float = 0.35,
    source: str = "planner",
) -> VerifySpec:
    """Parse a ``verify:`` block. Raises :class:`VerifyBlockError` on anything
    malformed; the whole block is rejected, never partially honoured."""
    if not isinstance(block, Mapping):
        raise VerifyBlockError(f"verify block must be a mapping, got {type(block)}")
    items = block.get("all")
    if not isinstance(items, (list, tuple)) or not items:
        raise VerifyBlockError("verify block needs a non-empty 'all' list")

    preds: List[Predicate] = []
    for raw in items:
        if not isinstance(raw, Mapping):
            raise VerifyBlockError(f"predicate entry must be a mapping: {raw!r}")
        name = str(raw.get("pred", "")).strip()
        if name not in _SCHEMA:
            raise VerifyBlockError(f"unknown predicate '{name}'")
        required, optional = _SCHEMA[name]
        params: Dict[str, Any] = dict(optional)
        for key in required:
            if key not in raw or raw[key] is None:
                raise VerifyBlockError(f"'{name}' missing required param '{key}'")
            params[key] = raw[key]
        for key in optional:
            if key in raw and raw[key] is not None:
                params[key] = raw[key]
        # types
        if name == "held":
            params["value"] = _norm_bool(params["value"])
        if name == "inside":
            params["occlusion_ok"] = _norm_bool(params["occlusion_ok"])
        for key in (
            "xy_margin_m",
            "z_tol_m",
            "z_max_m",
            "xy_tol_m",
            "dz_min_m",
            "dz_max_m",
            "max_dist_m",
        ):
            if key in params:
                try:
                    params[key] = float(params[key])
                except (TypeError, ValueError):
                    raise VerifyBlockError(f"'{name}.{key}' is not a number")
        pred = Predicate(name, params)
        if label_resolver is not None:
            for lab in pred.labels:
                if not label_resolver(lab):
                    raise VerifyBlockError(f"'{name}' references unknown label '{lab}'")
        preds.append(pred)

    min_conf = block.get("min_conf", default_min_conf)
    try:
        min_conf = float(min_conf)
    except (TypeError, ValueError):
        raise VerifyBlockError("min_conf is not a number")
    return VerifySpec(
        predicates=preds,
        min_conf=min_conf,
        require_two_views=_norm_bool(block.get("require_two_views", False)),
        source=source,
    )


_GRASP_TYPES = frozenset(
    ("grasp", "grasp_se3", "grasp_top_down", "grasp_cgn", "grasp_se3_flow")
)

# Primitives that push, pull or drag the object they NAME (as opposed to the
# object already in the jaws). A label one of these touched has a stale
# plan-time pose, so it can never serve as a plan-time reference frame.
_DISPLACING_TYPES = frozenset(
    (
        "open_drawer",
        "close_drawer",
        "pull",
        "push",
        "push_object",
        "drag",
        "wiggle",
        "screw",
        "compliant_push",
        "compliant_insert",
        "throw_to",
        "grasp_perturb",
        "hold_in_place",
        "tilt_wrist",
    )
)

_CONTAINER_MIN_MINOR_M = 0.08


def _labels_touched_by(actions: Sequence[Mapping], types_) -> List[str]:
    """Labels the given action types name, in order, deduplicated.

    A primitive that omits its own label inherits the one from the preceding
    ``move_to_keypoint``.
    """
    out: List[str] = []
    pending: Optional[str] = None
    for act in actions or []:
        atype = str(act.get("type", act.get("name", "")))
        params = act.get("params", {}) or {}
        label = params.get("keypoint_label") or params.get("label")
        if atype == "move_to_keypoint":
            pending = label
            continue
        if atype in types_:
            lab = label or params.get("joint_name") or pending
            if lab and lab not in out:
                out.append(str(lab))
    return out


def _is_container(det: Any) -> bool:
    if det is None:
        return False
    if det_field(det, "slots"):
        return True
    return float(det_field(det, "obb_minor_m", 0.0) or 0.0) >= _CONTAINER_MIN_MINOR_M


def grasped_labels(actions: Sequence[Mapping]) -> List[str]:
    """Labels the tree actually closed the jaws on, in order.

    These are the only objects allowed to have moved during the run, which is
    what lets the verifier re-identify them geometrically after a re-detection
    (see ``success_verifier.IdentityBinder``).
    """
    return _labels_touched_by(actions, _GRASP_TYPES)


def displaced_labels(actions: Sequence[Mapping]) -> List[str]:
    """Every label the tree physically moved, carried OR shoved.

    Superset of :func:`grasped_labels`. Decides whether a label's plan-time
    detection may be reused as a reference frame: everything NOT in this set is
    scenery whose plan-time pose is still valid at verify time.
    """
    out = _labels_touched_by(actions, _GRASP_TYPES | _DISPLACING_TYPES)
    for act in actions or []:
        if str(act.get("type", act.get("name", ""))) != "sweep":
            continue
        # sweep's target_label is the container swept INTO (scenery). Only the
        # object_labels move.
        raw = (act.get("params", {}) or {}).get("object_labels") or ""
        objs = (
            list(raw)
            if isinstance(raw, (list, tuple))
            else [o.strip() for o in str(raw).split(",") if o.strip()]
        )
        for obj in objs:
            if obj and obj not in out:
                out.append(str(obj))
    return out


def derive_default_verify(
    actions: Sequence[Mapping],
    detections: Optional[Mapping[str, Any]] = None,
    *,
    min_conf: float = 0.35,
) -> Optional[VerifySpec]:
    """Derive a predicate block from a flattened action list.

    Returns None when nothing is derivable (pour/scrub/cloth/bimanual); the
    caller must then report ``unverified``, never ``pass``.
    """
    detections = detections or {}
    preds: List[Predicate] = []
    seen = set()

    def add(pred: Predicate):
        key = pred.describe()
        if key not in seen:
            seen.add(key)
            preds.append(pred)

    last_grasped: Optional[str] = None
    pending_keypoint: Optional[str] = None
    place_label: Optional[str] = None
    holding = False

    for act in actions:
        atype = str(act.get("type", act.get("name", "")))
        params = act.get("params", {}) or {}
        label = params.get("keypoint_label") or params.get("label")

        if atype == "move_to_keypoint":
            pending_keypoint = label
            if holding and label:
                place_label = label
            continue

        if atype in _GRASP_TYPES:
            last_grasped = label or pending_keypoint or last_grasped
            holding = True
            place_label = None
            continue

        if atype == "place_in_slot":
            cont = params.get("container_label")
            if last_grasped and cont:
                add(
                    Predicate(
                        "inside",
                        {
                            "obj": last_grasped,
                            "container": cont,
                            "xy_margin_m": -0.02,
                            "z_tol_m": 0.06,
                            "occlusion_ok": False,
                        },
                    )
                )
            holding = False
            place_label = None
            continue

        if atype == "stack":
            base = params.get("target_label")
            if last_grasped and base:
                add(
                    Predicate(
                        "stacked",
                        {
                            "obj": last_grasped,
                            "base": base,
                            "xy_tol_m": 0.035,
                            "dz_min_m": STACKED_DZ_MIN_M,
                            "dz_max_m": 0.120,
                        },
                    )
                )
            holding = False
            continue

        if atype == "sweep":
            target = params.get("target_label") or params.get("area_label")
            raw_objs = params.get("object_labels") or ""
            objs = (
                list(raw_objs)
                if isinstance(raw_objs, (list, tuple))
                else [o.strip() for o in str(raw_objs).split(",") if o.strip()]
            )
            for obj in objs:
                if target:
                    add(
                        Predicate(
                            "inside",
                            {
                                "obj": obj,
                                "container": target,
                                "xy_margin_m": -0.02,
                                "z_tol_m": 0.06,
                                "occlusion_ok": False,
                            },
                        )
                    )
            continue

        if atype == "release":
            if last_grasped and place_label:
                cont_det = detections.get(place_label)
                if _is_container(cont_det):
                    add(
                        Predicate(
                            "inside",
                            {
                                "obj": last_grasped,
                                "container": place_label,
                                "xy_margin_m": -0.02,
                                "z_tol_m": 0.06,
                                "occlusion_ok": False,
                            },
                        )
                    )
                else:
                    add(
                        Predicate(
                            "on",
                            {
                                "obj": last_grasped,
                                "surface": place_label,
                                "xy_margin_m": 0.0,
                                "z_max_m": 0.10,
                            },
                        )
                    )
            holding = False
            place_label = None
            continue

    if not preds:
        return None
    if last_grasped:
        add(Predicate("held", {"obj": last_grasped, "value": False}))
    return VerifySpec(predicates=preds, min_conf=min_conf, source="derived")


Vote = str  # "pass" | "fail" | "abstain"
Status = str  # "pass" | "fail" | "unverified"
Mode = str  # "3d" | "2d" | "none"

PASS = "pass"
FAIL = "fail"
ABSTAIN = "abstain"
UNVERIFIED = "unverified"

# Slot bounds are padded by this much before they define the container extent.
SLOT_PAD_M = 0.02
# An object may sit this far below the container's own centroid z and still
# count as inside (rim-referenced centroids read high on deep bins).
INSIDE_Z_BELOW_M = 0.02
# Smallest margin-adjusted half-extent that still carries information. A
# re-detected centroid moves ~1-2cm between passes (the binder's "stationary"
# tolerance is 6cm), so a pass region narrower than 1cm is finer than the
# sensor and the test must abstain, not fail. Example: an occluded bowl mask
# with half_b=1.9cm under inside's default -2cm margin collapses to 0.0.
DEGENERATE_EXTENT_M = 0.01
# `inside`: share of the object's footprint that must lie over the container when its centre does not.
INSIDE_MIN_OVERLAP = 0.5
# Half-extent above which a container footprint is a bad detection, not a container.
OVERSIZE_EXTENT_M = 0.30
# Count-based ``inside`` (numbered look-alike instances): detections weaker than this are
# not counted, and a piece counts as inside only within this height of the container.
COUNT_MIN_CONF = 0.20
COUNT_INSIDE_Z_TOL_M = 0.10


@dataclass
class CameraVote:
    camera: str
    predicate: str
    vote: Vote
    mode: Mode
    confidence: float
    detail: str


@dataclass
class VerifyOutcome:
    status: Status
    predicates: List[str] = field(default_factory=list)
    votes: List[CameraVote] = field(default_factory=list)
    gates: Dict[str, bool] = field(default_factory=dict)
    reason: str = ""
    depth_source: str = "none"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "predicates": list(self.predicates),
            "votes": [
                {
                    "camera": v.camera,
                    "predicate": v.predicate,
                    "vote": v.vote,
                    "mode": v.mode,
                    "confidence": round(float(v.confidence), 3),
                    "detail": v.detail,
                }
                for v in self.votes
            ],
            "gates": dict(self.gates),
            "reason": self.reason,
            "depth_source": self.depth_source,
        }


@dataclass
class EvalConfig:
    """Thresholds for a single camera's evaluation."""

    min_conf: float = 0.35
    fallback_2d_containment: float = 0.90
    inside_xy_margin_m: float = -0.02
    inside_z_tol_m: float = 0.06
    z_tol_scale: float = 1.0  # 2.0 on monocular (DA3) depth


@dataclass
class SceneView:
    """One camera's view of the scene, plus non-visual state.

    ``lookup`` maps a predicate label to the ONE physical detection that label
    was written about, or None. ``explain`` (optional) says why a lookup came
    back empty; "not detected" and "three candidates, identity ambiguous" are
    both abstentions but very different diagnoses.
    """

    camera: str
    lookup: Callable[[str], Optional[Any]]
    held: Optional[bool] = None
    depth_source: str = "hardware"
    explain: Optional[Callable[[str], str]] = None


# predicate schema: required params -> optional params with defaults


def _xyz(det: Any) -> Optional[np.ndarray]:
    pos = det_field(det, "position_3d")
    if pos is None:
        return None
    try:
        arr = np.asarray(pos, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3]


def _conf(det: Any) -> float:
    try:
        return float(det_field(det, "confidence", 0.0) or 0.0)
    except Exception:
        return 0.0


def _mask(det: Any) -> Optional[np.ndarray]:
    m = det_field(det, "mask")
    if m is None:
        return None
    try:
        arr = np.asarray(m)
    except Exception:
        return None
    return arr if arr.ndim == 2 and arr.size else None


# container geometry


def _container_obb(det: Any) -> Tuple[Optional[Tuple], str]:
    """Return ((centre_xy, half_a, half_b, theta), source-tag).

    The tag is echoed into the vote detail; a reference frame taken from the
    plan-time detection rather than the verify re-detection is marked ``@plan``.
    """
    tag = "@plan" if det_field(det, "plan_anchored") else ""
    xyz = _xyz(det)
    if xyz is None:
        return None, "no-position" + tag
    centre = xyz[:2].astype(float)

    theta = det_field(det, "world_major_axis_rad")
    theta = float(theta) if theta is not None else 0.0
    ar = float(det_field(det, "aspect_ratio", 1.0) or 1.0)
    # PCA inversion: a sub-1 aspect ratio means the stored "major" axis is
    # actually the short one (same guard as mask_geometry.resolve_slot_direction).
    if 0.0 < ar < 1.0:
        theta += math.pi / 2.0
        ar = 1.0 / ar

    slots = det_field(det, "slots") or []
    a_max = b_max = 0.0
    n_slots = 0
    ca, sa = math.cos(theta), math.sin(theta)
    for slot in slots:
        wxyz = det_field(slot, "world_xyz")
        if wxyz is None:
            continue
        try:
            p = np.asarray(wxyz, dtype=float).reshape(-1)[:2] - centre
        except Exception:
            continue
        a_max = max(a_max, abs(p[0] * ca + p[1] * sa))
        b_max = max(b_max, abs(-p[0] * sa + p[1] * ca))
        n_slots += 1
    if n_slots:
        return (centre, a_max + SLOT_PAD_M, b_max + SLOT_PAD_M, theta), "slots" + tag

    minor = float(det_field(det, "obb_minor_m", 0.0) or 0.0)
    if minor > 0.0:
        # obb_minor_m is the FULL short-axis length, so halve it.
        half_b = minor / 2.0
        half_a = minor * max(ar, 1.0) / 2.0
        return (centre, half_a, half_b, theta), "obb" + tag

    # No slots and no measured width: the container's extent is UNKNOWN.
    # Abstain rather than invent a radius; a camera that DID measure the
    # container still decides.
    return None, "no-extent" + tag


def container_obb_xy(det: Any) -> Optional[Tuple]:
    """``(centre_xy, half_a, half_b, theta)`` for a container detection."""
    obb, _ = _container_obb(det)
    return obb


def _obb_contains(point_xy: np.ndarray, obb: Tuple, margin: float) -> bool:
    centre, half_a, half_b, theta = obb
    d = np.asarray(point_xy, dtype=float)[:2] - np.asarray(centre, dtype=float)
    ca, sa = math.cos(theta), math.sin(theta)
    a = abs(d[0] * ca + d[1] * sa)
    b = abs(-d[0] * sa + d[1] * ca)
    # A margin may shrink the box but never invert it.
    return a <= max(half_a + margin, 0.0) and b <= max(half_b + margin, 0.0)


def _share_inside(box: Tuple, other: Tuple, n: int) -> float:
    """Fraction of ``box`` (sampled on an n x n grid) that lies inside ``other``."""
    centre, half_a, half_b, theta = box
    ca, sa = math.cos(theta), math.sin(theta)
    g = np.linspace(-1.0, 1.0, n)
    inside = 0
    for u in g:
        for v in g:
            p = np.asarray(centre, dtype=float) + np.array(
                [u * half_a * ca - v * half_b * sa, u * half_a * sa + v * half_b * ca])
            inside += _obb_contains(p, other, 0.0)
    return inside / float(n * n)


def footprint_overlap(obj: Any, cont_obb: Tuple, n: int = 9) -> Optional[float]:
    """Larger of two shares: the object's footprint inside the container's box, and the
    container's box inside the object's footprint. None when the object's extent is unknown."""
    o_obb, _ = _container_obb(obj)
    if o_obb is None:
        return None
    # The reverse share too: an object wider than its container covers it without half of
    # itself being over it.
    return max(_share_inside(o_obb, cont_obb, n), _share_inside(cont_obb, o_obb, n))


def _own_height(det: Any) -> float:
    """Measured top-to-bottom height of a detection (0 when unknown), capped at 20 cm."""
    try:
        top, bottom = det_field(det, "rim_z_m"), det_field(det, "interior_z_m")
        h = float(top) - float(bottom)
    except (TypeError, ValueError):
        return 0.0
    return min(max(h, 0.0), 0.20) if math.isfinite(h) else 0.0


# public aliases (used by success_verifier's release witness and its
# plan-anchored reference builder)
obb_contains = _obb_contains
detection_xyz = _xyz
detection_field = det_field


def mask_containment(obj_mask: Any, container_mask: Any) -> Optional[float]:
    """Fraction of the object mask inside the container mask, same camera."""
    a, b = _mask(obj_mask), _mask(container_mask)
    if a is None or b is None or a.shape != b.shape:
        return None
    am = a.astype(bool)
    area = int(am.sum())
    if area <= 0:
        return None
    return float(np.logical_and(am, b.astype(bool)).sum()) / float(area)


# evaluation


def _vote(view, pred, vote, mode, conf, detail) -> CameraVote:
    return CameraVote(
        camera=view.camera,
        predicate=pred.describe(),
        vote=vote,
        mode=mode,
        confidence=float(conf),
        detail=detail,
    )


def _binding_note(view: SceneView, label: Any) -> str:
    """Why ``view.lookup`` came back empty, if the view can say."""
    if view.explain is None or not label:
        return ""
    try:
        note = view.explain(label)
    except Exception:
        return ""
    return f": {note}" if note else ""


def _resolve_pair(view: SceneView, pred: Predicate, cfg: EvalConfig, keys):
    """Bind labels -> detections; returns (dets, abstain_vote_or_None)."""
    dets = []
    for key in keys:
        label = pred.params.get(key)
        det = view.lookup(label) if label else None
        if det is None:
            return None, _vote(
                view, pred, ABSTAIN, "none", 0.0, f"'{label}' unbound{_binding_note(view, label)}"
            )
        dets.append(det)
    conf = min(_conf(d) for d in dets)
    if conf < cfg.min_conf:
        return None, _vote(
            view,
            pred,
            ABSTAIN,
            "none",
            0.0,
            f"confidence {conf:.2f} < min_conf {cfg.min_conf:.2f}",
        )
    return dets, None


def _eval_containment(
    view: SceneView, pred: Predicate, cfg: EvalConfig, obj_key: str, cont_key: str
) -> CameraVote:
    """Shared body of ``inside`` / ``on`` / ``removed_from``."""
    dets, abstain = _resolve_pair(view, pred, cfg, (obj_key, cont_key))
    if abstain is not None:
        return abstain
    obj, cont = dets
    conf = min(_conf(obj), _conf(cont))
    o_xyz, c_xyz = _xyz(obj), _xyz(cont)

    if o_xyz is not None and c_xyz is not None:
        obb, src = _container_obb(cont)
        if obb is None:
            return _vote(
                view, pred, ABSTAIN, "none", 0.0, f"container extent unknown ({src})"
            )
        if pred.pred == "on":
            margin = float(pred.params.get("xy_margin_m", 0.0))
            z_lo, z_hi = 0.0, float(pred.params.get("z_max_m", 0.10))
            dz = float(o_xyz[2] - c_xyz[2])
            z_ok = z_lo <= dz <= z_hi * cfg.z_tol_scale
        else:
            margin = float(pred.params.get("xy_margin_m", cfg.inside_xy_margin_m))
            z_tol = float(pred.params.get("z_tol_m", cfg.inside_z_tol_m))
            z_tol *= cfg.z_tol_scale
            dz = float(o_xyz[2] - c_xyz[2])
            # The object's position is its visible top, so a tall object sitting in a shallow
            # container reads its own height above the container. Allow for that height.
            z_tol += _own_height(obj)
            # The container's position is its rim, so a thin object lying on the floor of a deep
            # bowl reads the bowl's depth BELOW it. Allow for that depth.
            z_ok = -(INSIDE_Z_BELOW_M * cfg.z_tol_scale + _own_height(cont)) <= dz <= z_tol
        # A collapsed margin-adjusted extent carries NO information: every
        # object position fails the xy test, and for removed_from (which
        # inverts the test) it would be a free PASS. Abstain.
        eff_a = max(float(obb[1]) + margin, 0.0)
        eff_b = max(float(obb[2]) + margin, 0.0)
        if min(eff_a, eff_b) < DEGENERATE_EXTENT_M:
            return _vote(
                view,
                pred,
                ABSTAIN,
                "none",
                0.0,
                f"container extent collapses under margin {margin * 100:+.1f}cm: "
                f"half=({obb[1] * 100:.1f},{obb[2] * 100:.1f})cm -> "
                f"({eff_a * 100:.1f},{eff_b * 100:.1f})cm, "
                f"below the {DEGENERATE_EXTENT_M * 100:.0f}cm information floor "
                f"(obb={src})",
            )
        # The opposite failure: a footprint larger than any container on this table (a tipped
        # bowl re-detected as a 95 cm box). Every position passes the xy test, a free PASS. Abstain.
        if max(float(obb[1]), float(obb[2])) > OVERSIZE_EXTENT_M:
            return _vote(view, pred, ABSTAIN, "none", 0.0,
                         f"container extent is implausible: half=({obb[1] * 100:.1f},{obb[2] * 100:.1f})cm "
                         f"> {OVERSIZE_EXTENT_M * 100:.0f}cm (obb={src})")
        xy_ok = _obb_contains(o_xyz[:2], obb, margin)
        overlap = None
        if pred.pred == "inside" and not xy_ok:
            # One point is a weak test for a long or leaning object (a spoon on the rim has its
            # centre near the edge). Ask how much of the object's footprint lies over the
            # container; at least INSIDE_MIN_OVERLAP of it counts as inside.
            overlap = footprint_overlap(obj, obb)
            if overlap is not None and overlap >= INSIDE_MIN_OVERLAP:
                xy_ok = True
        ok = bool(xy_ok and z_ok)
        detail = (
            f"3d xy_ok={xy_ok} dz={dz*100:+.1f}cm obb={src} "
            f"half=({obb[1]*100:.1f},{obb[2]*100:.1f})cm"
            + (f" overlap={overlap:.2f}" if overlap is not None else "")
        )
        if pred.pred == "removed_from":
            return _vote(view, pred, FAIL if ok else PASS, "3d", conf, "removed_from: " + detail)
        return _vote(view, pred, PASS if ok else FAIL, "3d", conf, detail)

    # 2D fallback: only when this camera lacks 3D for object or container.
    # A threshold above 1.0 is the kill switch: abstain rather than fail
    # everything (a fail would veto every other camera).
    if cfg.fallback_2d_containment > 1.0:
        return _vote(view, pred, ABSTAIN, "none", 0.0, "no 3d; 2d fallback disabled")
    frac = mask_containment(obj, cont)
    if frac is None:
        return _vote(view, pred, ABSTAIN, "none", 0.0, "no 3d and no comparable masks")
    ok = frac >= cfg.fallback_2d_containment
    detail = f"2d containment={frac:.2f} (>= {cfg.fallback_2d_containment:.2f})"
    if pred.pred == "removed_from":
        return _vote(view, pred, FAIL if ok else PASS, "2d", conf, detail)
    return _vote(view, pred, PASS if ok else FAIL, "2d", conf, detail)


def is_numbered(label: Any) -> bool:
    """'silverware 2' is one of several look-alike instances; 'tray' is not."""
    return bool(label) and str(label)[-1:].isdigit()


def plan_instances_inside(plan_map, base, cont, moved) -> int:
    """How many numbered ``base`` instances the tree did NOT move already lay inside ``cont``
    at plan time. They are still there, so the count that must be inside starts from them."""
    if cont is None or _xyz(cont) is None or _container_obb(cont)[0] is None:
        return 0
    n = 0
    for key in list(plan_map):
        if key in moved or not is_numbered(key) or str(key).rsplit(" ", 1)[0] != base:
            continue
        # The entry stored under this key: LabelResolvingDetectionMap.get resolves labels.
        xyz = _xyz(dict.get(plan_map, key) if isinstance(plan_map, dict) else plan_map.get(key))
        if xyz is not None and xy_inside_container(xyz[:2], cont):
            n += 1
    return n


def count_inside_vote(view, pred, cam_dets, base, n_needed, n_total, cfg):
    """``inside`` for numbered look-alike instances, by count instead of identity.

    Which spoon is "silverware 2" cannot be told once the spoons lie in the tray, and
    it does not matter: the claim is that enough of them are inside. Every detection
    of ``base`` in this camera is tested against the container's footprint.
    ``n_total`` instances exist and ``n_needed`` of them must be inside, so up to
    ``n_total - n_needed`` may still lie outside (not yet placed, or not part of the task).
    FAIL: more than that are outside, or all are seen and too few are inside.
    PASS: at least ``n_needed`` are inside. ABSTAIN otherwise (pieces hidden, container unknown).
    """
    cont = view.lookup(pred.params.get("container"))
    c_xyz = _xyz(cont) if cont is not None else None
    obb, src = _container_obb(cont) if cont is not None else (None, "container unbound")
    if c_xyz is None or obb is None:
        return _vote(view, pred, ABSTAIN, "none", 0.0, f"count: container extent unknown ({src})")
    z_tol = COUNT_INSIDE_Z_TOL_M * cfg.z_tol_scale
    inside, outside = [], 0
    for det in cam_dets:
        label = str(det_field(det, "label", "") or "")
        if (label.rsplit(" ", 1)[0] if is_numbered(label) else label) != base:
            continue
        xyz = _xyz(det)
        if xyz is None or _conf(det) < COUNT_MIN_CONF:
            continue
        # The executor's own container region (whole tray), not the slot-centre box: a piece
        # lying toward the rim is still in the tray.
        if xy_inside_container(xyz[:2], cont) and abs(float(xyz[2] - c_xyz[2])) <= z_tol:
            inside.append(_conf(det))
        else:
            outside += 1
    detail = (f"count (which instance is which is ambiguous): {len(inside)} of '{base}' inside, "
              f"{outside} outside (need {n_needed}, {n_total} in all; obb={src})")
    conf = min(inside) if inside else 0.0
    if outside > max(n_total - n_needed, 0) or (len(inside) + outside >= n_total and len(inside) < n_needed):
        return _vote(view, pred, FAIL, "3d", conf, detail)
    # When every instance belongs inside, "none left outside" is the claim itself: pieces that
    # overlap in the container and merge into one detection must not block the pass.
    none_left = n_total <= n_needed and outside == 0 and inside
    if len(inside) >= n_needed or none_left:
        return _vote(view, pred, PASS, "3d", max(conf, cfg.min_conf), detail)
    return _vote(view, pred, ABSTAIN, "none", 0.0, detail)


def _eval_stacked(view: SceneView, pred: Predicate, cfg: EvalConfig) -> CameraVote:
    dets, abstain = _resolve_pair(view, pred, cfg, ("obj", "base"))
    if abstain is not None:
        return abstain
    obj, base = dets
    conf = min(_conf(obj), _conf(base))
    o, b = _xyz(obj), _xyz(base)
    if o is None or b is None:
        return _vote(view, pred, ABSTAIN, "none", 0.0, "stacked needs 3d, none here")
    d_xy = float(np.linalg.norm(o[:2] - b[:2]))
    dz = float(o[2] - b[2])
    xy_tol = float(pred.params.get("xy_tol_m", 0.035))
    # ponytail: fixed gap; a top object 1 cm above the base pose is on the table, not on the base.
    # Use the base height if thin objects get stacked
    dz_min = float(pred.params.get("dz_min_m", STACKED_DZ_MIN_M))
    dz_max = float(pred.params.get("dz_max_m", 0.120)) * cfg.z_tol_scale
    ok = d_xy <= xy_tol and dz_min <= dz <= dz_max
    return _vote(
        view,
        pred,
        PASS if ok else FAIL,
        "3d",
        conf,
        f"3d d_xy={d_xy*100:.1f}cm dz={dz*100:+.1f}cm",
    )


def _eval_near(view: SceneView, pred: Predicate, cfg: EvalConfig) -> CameraVote:
    dets, abstain = _resolve_pair(view, pred, cfg, ("obj", "target"))
    if abstain is not None:
        return abstain
    obj, tgt = dets
    conf = min(_conf(obj), _conf(tgt))
    o, t = _xyz(obj), _xyz(tgt)
    if o is None or t is None:
        return _vote(view, pred, ABSTAIN, "none", 0.0, "near needs 3d, none here")
    dist = float(np.linalg.norm(o - t))
    ok = dist <= float(pred.params["max_dist_m"])
    return _vote(view, pred, PASS if ok else FAIL, "3d", conf, f"3d dist={dist*100:.1f}cm")


def _eval_held(view: SceneView, pred: Predicate, cfg: EvalConfig) -> CameraVote:
    want = bool(pred.params.get("value"))
    have = view.held
    if have is None:
        # Never abstains: no witness at all reads as "not holding" only when
        # the caller supplied nothing; treat as a failure to prove release.
        return _vote(view, pred, FAIL, "none", 1.0, "no grip state available")
    return _vote(
        view,
        pred,
        PASS if bool(have) == want else FAIL,
        "none",
        1.0,
        f"held={bool(have)} want={want}",
    )


def _eval_absent(view: SceneView, pred: Predicate, cfg: EvalConfig) -> CameraVote:
    """Asymmetric on purpose: ``absent`` can FAIL, and can never PASS.

    A SAM3 miss, an occlusion, and an object genuinely removed all produce the
    same non-detection, so a non-detection abstains (mode ``none``); only a
    positive re-detection is decisive, and it decides FAIL.
    """
    label = pred.params.get("obj")
    det = view.lookup(label)
    if det is None:
        return _vote(
            view,
            pred,
            ABSTAIN,
            "none",
            0.0,
            f"'{label}' not detected -- absence of evidence{_binding_note(view, label)}",
        )
    conf = _conf(det)
    if conf < cfg.min_conf:
        return _vote(
            view, pred, ABSTAIN, "none", 0.0, f"detected at {conf:.2f} < min_conf {cfg.min_conf:.2f}"
        )
    return _vote(view, pred, FAIL, "3d", conf, f"still detected at {conf:.2f}")


def evaluate_predicate(pred: Predicate, view: SceneView, cfg: EvalConfig) -> CameraVote:
    """Evaluate one predicate in one camera. Never raises for scene reasons."""
    if pred.pred in ("inside", "removed_from"):
        return _eval_containment(view, pred, cfg, "obj", "container")
    if pred.pred == "on":
        return _eval_containment(view, pred, cfg, "obj", "surface")
    if pred.pred == "stacked":
        return _eval_stacked(view, pred, cfg)
    if pred.pred == "near":
        return _eval_near(view, pred, cfg)
    if pred.pred == "held":
        return _eval_held(view, pred, cfg)
    if pred.pred == "absent":
        return _eval_absent(view, pred, cfg)
    return _vote(view, pred, ABSTAIN, "none", 0.0, f"unknown predicate {pred.pred}")


def camera_verdict(votes: Sequence[CameraVote]) -> Vote:
    """AND over predicates, abstain-absorbing."""
    if not votes:
        return ABSTAIN
    if any(v.vote == FAIL for v in votes):
        return FAIL
    if any(v.vote == ABSTAIN for v in votes):
        return ABSTAIN
    return PASS


# parsing


# derivation from tree structure (so the cached BT library verifies)




logger = logging.getLogger(__name__)

GATE_NAMES = ("grasp", "transport", "release")


@dataclass
class ReleaseWitness:
    held_before: bool
    held_after: bool
    tcp_xyz: Tuple[float, float, float]
    container_label: Optional[str] = None
    inside_region: Optional[bool] = None
    tcp_dz_to_container: Optional[float] = None
    # Jaw evidence behind held_after. ``released`` is the gate's verdict:
    # True = the jaws demonstrably opened, False = they did not, None = no
    # usable read (abstain). The raw numbers are logged so the next rig run
    # calibrates GRIPPER_RELEASED_MAX_POS / GRIPPER_RELEASE_MIN_TRAVEL.
    released: Optional[bool] = None
    jaw_pos_before: Optional[float] = None
    jaw_pos_after: Optional[float] = None
    obj_after: Optional[bool] = None
    confirm_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "held_before": bool(self.held_before),
            "held_after": bool(self.held_after),
            "tcp_xyz": [float(v) for v in self.tcp_xyz],
            "container_label": self.container_label,
            "inside_region": self.inside_region,
            "tcp_dz_to_container": self.tcp_dz_to_container,
            "released": self.released,
            "jaw_pos_before": self.jaw_pos_before,
            "jaw_pos_after": self.jaw_pos_after,
            "obj_after": self.obj_after,
            "confirm_s": round(float(self.confirm_s), 3),
        }


def _gates(executor) -> Dict[str, Optional[bool]]:
    flags = getattr(executor, "_verify_gates", None)
    if not isinstance(flags, dict):
        flags = {}
        try:
            executor._verify_gates = flags
        except Exception:
            pass
    return flags


def set_gate(executor, name: str, value: Optional[bool], detail: str = "") -> None:
    if executor is None or name not in GATE_NAMES:
        return
    _gates(executor)[name] = value
    if value is False:
        logger.warning("Verify gate '%s' FAILED%s", name, f": {detail}" if detail else "")


def record_grasp_verdict(executor, holding: bool, detail: str = "", verdict: Any = None) -> None:
    """G1: called from the grasp verify path with its structured verdict.

    ``verdict`` is the executor's own richer record (control.executor_grasp's
    GraspVerdict) when it has one; it is stored verbatim so the ASPIRE trace
    keeps the evidence behind the boolean.
    """
    set_gate(executor, "grasp", bool(holding), detail)
    if holding:
        # A successful re-grasp clears an earlier transport drop.
        if _gates(executor).get("transport") is False:
            set_gate(executor, "transport", True, "re-grasp after drop")
    try:
        executor._last_grasp_verdict = (
            verdict if verdict is not None else {"holding": bool(holding), "detail": detail}
        )
    except Exception:
        pass


def _verdict_holding(verdict: Any) -> Optional[bool]:
    """Read the boolean out of either verdict shape (dict or GraspVerdict)."""
    if verdict is None:
        return None
    if isinstance(verdict, dict):
        val = verdict.get("holding")
    else:
        val = getattr(verdict, "held", None)
    return None if val is None else bool(val)


def note_transport_drop(executor, waypoint: str = "") -> None:
    """G2: called when a stationary-waypoint grip check says DROPPED.

    Also upgrades the stored grasp verdict's typed outcome to SLIP when the
    grasp had classified SECURED: a secured close followed by a lost grip
    at a stationary waypoint is the Robotiq analog of the sim's
    aperture-collapse SLIP. Additive metadata only.
    """
    set_gate(executor, "transport", False, f"drop at {waypoint}")
    try:
        from spark_core.grasp_outcome import (
            GraspOutcome,
            slip_after_secured,
        )

        v = getattr(executor, "_last_grasp_verdict", None)
        prev_raw = (
            v.get("outcome") if isinstance(v, dict)
            else getattr(v, "outcome", None)
        )
        prev = GraspOutcome(prev_raw) if prev_raw else None
        slip = slip_after_secured(prev, False)
        if slip is not None:
            if isinstance(v, dict):
                v["outcome"] = slip.value
            else:
                v.outcome = slip.value
            logger.warning(
                "Grasp outcome: SECURED -> SLIP (drop at %s)", waypoint
            )
    except Exception:  # noqa: BLE001 - annotation must never break the gate
        pass


def reset_run_state(executor, task_scope: bool = True) -> None:
    """Drop every verdict/gate carried by the previous run.

    An executor instance outlives a task, so without this the next run would
    inherit the last run's outcome, gates and release witness, and a stale
    ``pass`` would be read as this run's success.

    ``task_scope`` also clears the executor's own per-task bookkeeping
    (_placed_labels, _holding, the grasp retry budget, ...) via
    ScoreExecutor.reset_task_state. Pass False from a CLOSED-LOOP PASS
    boundary: the passes of one task share that state (_placed_labels is what
    tells pass 3 that pass 1 already handled an object), so clearing it there
    would make the loop re-pick an object already sitting in the receptacle.
    """
    if executor is None:
        return
    for attr, value in (
        ("verify_outcome", None),
        ("_verify_gates", {}),
        ("_release_witness", None),
        ("_last_grasp_verdict", None),
    ):
        try:
            setattr(executor, attr, value)
        except Exception:
            pass
    if not task_scope:
        return
    reset = getattr(executor, "reset_task_state", None)
    if callable(reset):
        try:
            reset()
        except Exception as exc:
            logger.warning("reset_run_state: per-task reset failed: %s", exc)


def _read_holding(executor) -> Optional[bool]:
    """Best available live grip state (gObj-primary when the rig exposes it)."""
    try:
        if hasattr(executor, "_grip_intact") and executor._gripper_type() == "robotiq_2f85":
            return bool(executor._grip_intact())
    except Exception as exc:
        logger.warning("Release witness: grip read failed: %s", exc)
    held = getattr(executor, "_holding", None)
    return None if held is None else bool(held)


def _is_robotiq(executor) -> bool:
    try:
        return executor._gripper_type() == "robotiq_2f85"
    except Exception:
        return False


def _has_jaw_source(executor) -> bool:
    """Does the driver expose any jaw state (register, position or width)?"""
    if _is_robotiq(executor):
        return True
    robot = getattr(executor, "robot", None)
    return any(
        hasattr(robot, name) for name in ("get_gripper_position", "get_gripper_width")
    )


def _jaw_sample(executor) -> Tuple[Optional[bool], Optional[float]]:
    """One fresh jaw sample: ``(object_flag, jaw_pos)``, 0=open .. 255=closed.

    Any driver: the executor's _get_gripper_position rescales a non-Robotiq
    position (or width) onto the Robotiq count scale, so the open-side
    thresholds below apply to every gripper. ``(None, None)`` only when the
    driver has no position source or the read failed. On the Robotiq,
    _read_gripper_state forces a publish and waits for the controller to
    acknowledge it, so this is a current sample, not a stale register.
    """
    try:
        obj, pos = executor._read_gripper_state()
    except Exception as exc:
        logger.warning("Release witness: jaw read failed: %s", exc)
        return None, None
    return (
        None if obj is None else bool(obj),
        None if pos is None else float(pos),
    )


def _jaws_opened(
    executor,
    pos_before: Optional[float],
    pos_after: Optional[float],
    obj_after: Optional[bool],
) -> Optional[bool]:
    """Did the jaws open? None when there is no usable position read.

    Reaching the open stop is conclusive on its own: a gObj that still reads
    contact there is a stale flag, not an object. Short of the stop, gObj=1
    means the jaws were halted BY something, so only travel with gObj clear
    counts. The travel clause carries the thin object, whose jaws legitimately
    stop before the absolute threshold.
    """
    if pos_after is None:
        return None
    max_pos = float(getattr(executor, "GRIPPER_RELEASED_MAX_POS", 60.0))
    min_travel = float(getattr(executor, "GRIPPER_RELEASE_MIN_TRAVEL", 40.0))
    if pos_after <= max_pos:
        return True
    if obj_after is True:
        return False
    if pos_before is not None and (pos_before - pos_after) >= min_travel:
        return True
    return False


def _confirm_jaws_open(
    executor, pos_before: Optional[float]
) -> Tuple[Optional[bool], Optional[bool], Optional[float], float]:
    """Poll for the jaws to read open. ``(opened, gObj, pos, waited_s)``.

    Only safe once the open URScript has finished: every sample uploads a
    program of its own, which would cancel an rq_move_and_wait still running.
    The caller's settle guarantees that. Costs nothing on a normal release;
    the first sample already reads open.
    """
    timeout = float(getattr(executor, "RELEASE_CONFIRM_TIMEOUT_S", 0.6))
    t0 = time.monotonic()
    obj, pos = _jaw_sample(executor)
    opened = _jaws_opened(executor, pos_before, pos, obj)
    while opened is False and (time.monotonic() - t0) < timeout:
        time.sleep(0.05)
        obj, pos = _jaw_sample(executor)
        opened = _jaws_opened(executor, pos_before, pos, obj)
    return opened, obj, pos, time.monotonic() - t0


def begin_release_witness(executor) -> Dict[str, Any]:
    """G3 part 1: call immediately BEFORE opening the gripper."""
    state = {"held_before": _read_holding(executor)}
    _, state["jaw_pos_before"] = _jaw_sample(executor)
    try:
        state["tcp_xyz"] = [float(v) for v in executor._get_current_position()[:3]]
    except Exception:
        state["tcp_xyz"] = [0.0, 0.0, 0.0]
    return state


def finish_release_witness(
    executor, state: Dict[str, Any], container_label: Optional[str] = None
) -> Optional[ReleaseWitness]:
    """G3 part 2: call after the open + settle. Stores the witness on the executor.

    ``held_after`` must NOT be read with _grip_intact. That predicate answers
    "are the jaws still spread on something" after a CLOSE, and its position
    clause (pos < GRIPPER_FULLY_CLOSED = holding) is True for every OPEN
    gripper (pos ~0 on a released 2F-85).
    Post-open the meaningful question is the opposite one: did the jaws reach
    open, or did something stop them.
    """
    try:
        pos_before = state.get("jaw_pos_before")
        opened, obj_after, pos_after, waited = _confirm_jaws_open(executor, pos_before)
        if opened is not None:
            released = opened
        elif _has_jaw_source(executor):
            # The rig HAS a jaw source and it did not read: abstain. Falling
            # back to _grip_intact here would answer "DROPPED" from the same
            # unreadable state and pass the gate on no evidence at all.
            released = None
        else:
            # No jaw state at all: the executor's own flag is the only signal.
            held = _read_holding(executor)
            released = None if held is None else (not held)
        witness = ReleaseWitness(
            held_before=bool(state.get("held_before")),
            held_after=(released is False),
            tcp_xyz=tuple(state.get("tcp_xyz") or (0.0, 0.0, 0.0)),
            container_label=container_label,
            released=released,
            jaw_pos_before=pos_before,
            jaw_pos_after=pos_after,
            obj_after=obj_after,
            confirm_s=waited,
        )
        label = container_label or getattr(executor, "_last_place_label", "") or ""
        det = (getattr(executor, "detection_map", None) or {}).get(label)
        obb = container_obb_xy(det) if det is not None else None
        if obb is not None:
            witness.container_label = label
            witness.inside_region = bool(
                obb_contains(np.asarray(witness.tcp_xyz[:2], dtype=float), obb, 0.0)
            )
            cxyz = detection_xyz(det)
            if cxyz is not None:
                witness.tcp_dz_to_container = float(witness.tcp_xyz[2] - cxyz[2])
        executor._release_witness = witness
        open_max = getattr(executor, "GRIPPER_RELEASED_MAX_POS", 60)
        min_travel = getattr(executor, "GRIPPER_RELEASE_MIN_TRAVEL", 40)
        detail = (
            f"held_before={witness.held_before} released={released} "
            f"jaw {pos_before} -> {pos_after} "
            f"(open<={open_max}, travel>={min_travel}) "
            f"gObj_after={obj_after} confirm={waited:.2f}s"
        )
        if released is None:
            # Abstain rather than invent a verdict: collect_gates reads a
            # missing gate as "the primitive never reported", and a fabricated
            # failure here would fail every task on a rig with no jaw register.
            logger.warning("Release witness: no grip evidence, abstaining (%s)", detail)
        else:
            set_gate(executor, "release", bool(witness.held_before and released), detail)
        logger.info("Release witness: %s", witness.to_dict())
        return witness
    except Exception as exc:
        logger.warning("Release witness failed: %s", exc)
        return None


def collect_gates(executor) -> Tuple[Dict[str, bool], List[str]]:
    """Resolve the three gates to booleans plus the reasons any of them failed.

    An unknown gate reads True (the primitive never ran); only explicit
    negative evidence fails, and the vision path still has to pass.
    """
    flags = dict(_gates(executor))
    if "grasp" not in flags:
        held = _verdict_holding(getattr(executor, "_last_grasp_verdict", None))
        if held is not None:
            flags["grasp"] = held
        else:
            results = [
                r
                for r in (getattr(executor, "_results", None) or [])
                if getattr(r, "action_type", "") == "grasp"
            ]
            if results:
                flags["grasp"] = bool(results[-1].success)
    witness = getattr(executor, "_release_witness", None)
    if "release" not in flags and isinstance(witness, ReleaseWitness):
        # Same verdict finish_release_witness computed, never a re-derivation:
        # released=None means it abstained, and re-deriving from held_after
        # would resurrect the false negative it exists to avoid.
        if witness.released is not None:
            flags["release"] = bool(witness.held_before and witness.released)

    gates, failures = {}, []
    for name in GATE_NAMES:
        val = flags.get(name)
        gates[name] = True if val is None else bool(val)
        if val is False:
            failures.append(name)
    return gates, failures



# How far a re-detected object may sit from its plan-time position and still
# count as the same physical object that never moved. Covers centroid jitter
# and depth noise; well under the spacing of distinct utensils on a table.
STATIC_MATCH_TOL_M = 0.06

# Detection fields a predicate reads off a reference object (container /
# surface / base / target). These are the ones taken from the PLAN-TIME
# detection when the reference is plan-anchored; ``mask`` and ``camera`` are
# deliberately NOT here; see _plan_anchored_reference.
_REFERENCE_FIELDS = (
    "label",
    "position_3d",
    "aspect_ratio",
    "world_major_axis_rad",
    "obb_minor_m",
    "slots",
)


def _plan_anchored_reference(plan_det: Any, observed: Any) -> Dict[str, Any]:
    """A reference detection whose POSE is plan-time and whose MASK is now.

    World-frame geometry (position, OBB, slots) comes from the plan-time
    detection: that is the pose the arm actually serviced, and it predates the
    occlusion the placement itself created.

    ``mask`` and ``camera`` must NOT come from plan time. The 2D containment
    fallback compares two masks pixel-for-pixel in ONE camera's image, and the
    plan-time map is a cross-camera merge; its mask may belong to a different
    view entirely. So the image-space fields stay with the fresh detection in
    this camera, and the world-frame fields come from plan time.

    ``confidence`` likewise comes from the fresh detection: it answers "did
    this camera see the container just now", which is what licenses reusing the
    plan-time pose at all. Falls back to the plan-time value when the
    re-detection has none.
    """
    ref: Dict[str, Any] = {name: detection_field(plan_det, name) for name in _REFERENCE_FIELDS}
    conf = detection_field(observed, "confidence") if observed is not None else None
    ref["confidence"] = conf if conf is not None else detection_field(plan_det, "confidence")
    # ``observed`` is None when verify_scope dropped this reference's prompt
    # altogether (verification.scope.anchor_references). There is then no
    # image-space evidence in this camera at all, so the 2D containment
    # fallback abstains, which is why the prompt is only dropped when that
    # fallback is switched off. The world-frame fields, which is all the 3D
    # test reads, are unaffected.
    ref["mask"] = detection_field(observed, "mask") if observed is not None else None
    ref["camera"] = detection_field(observed, "camera") if observed is not None else None
    ref["plan_anchored"] = True
    return ref


def _plan_key(plan_map: Any, label: Any) -> Optional[str]:
    """The plan-time detection key a predicate label refers to, or None.

    NOT ``LabelResolvingDetectionMap.resolve_label``. That resolver exists to
    keep a cached tree DRIVING when labels drift, so it deliberately collapses
    ``"fork"`` (and even ``"fork 9"``) onto the lowest-numbered instance. That
    is the right call for dispatching a primitive and the wrong one here: this
    key decides WHICH physical object a success predicate is about, and a
    silent collapse binds the predicate to a fork the tree never touched.

    So: an exact key, a declared role alias, or a canonical base with exactly
    one plan-time instance. More than one instance is ambiguous and returns
    None, which the binder and the predicates read as abstain.
    """
    key = str(label or "").strip()
    if not key or not plan_map:
        return None
    # Iteration yields the real detected labels for both map types.
    keys = list(plan_map)
    if key in keys:
        return key
    # A role alias is an authoritative binding published by the task's
    # perception contract, so it outranks the ambiguity rule below.
    alias = (getattr(plan_map, "aliases", None) or {}).get(key)
    if alias in keys:
        return alias
    want = canon_label(key)
    hits = [k for k in keys if canon_label(k) == want]
    if len(hits) > 1:
        return None
    if hits:
        return hits[0]
    # No instance of this base at all: sub-part / super-part resolution
    # ("knife" -> "knife handle") is unambiguous by construction.
    if isinstance(plan_map, LabelResolvingDetectionMap):
        return plan_map.resolve_label(key)
    return None


def _xy_dist(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


class IdentityBinder:
    """Bind a predicate label to the ONE physical object it was written about.

    Instance numbers are assigned per detection pass and permute freely between
    passes: SAM3 orders them geometrically, so the moment an object moves the
    numbering re-sorts. Re-resolving ``"fork 1"`` against the verify-time
    detections therefore binds the predicate to whatever object happens to be
    first NOW. Identity must come from geometry, not the number.

    Two rules, both anchored on the plan-time detection map the tree was
    written against:

      * a label the tree never grasped did not move, so it binds to the
        re-detection nearest its plan-time position, and only when that
        nearest match is unique inside the tolerance;
      * a label the tree DID grasp is the only object allowed to have moved,
        so it binds to the re-detection left over once every *other* plan-time
        instance of the same base label has claimed its own, still-stationary
        re-detection.

    ``plan_anchored`` is a third, narrower rule for REFERENCE objects (the
    container/surface/base a predicate measures against, when the tree never
    touched it). Those bind to their PLAN-TIME pose and are not re-located at
    all; see :meth:`_bind_reference`.

    Every other case (a sibling that vanished, two candidates on top of each
    other, more than one unaccounted detection, no plan-time anchor) returns
    None, which the predicates read as abstain. Abstaining is a fine outcome;
    binding to the wrong object is the failure this class prevents.
    """

    def __init__(
        self,
        plan_map: Any,
        detections: Iterable[Any],
        manipulated: Optional[Set[str]] = None,
        tol: float = STATIC_MATCH_TOL_M,
        plan_anchored: Optional[Set[str]] = None,
        pose_only: Optional[Set[str]] = None,
    ):
        self._plan = plan_map if isinstance(plan_map, Mapping) else {}
        self._manipulated = set(manipulated or ())
        self._plan_anchored = set(plan_anchored or ())
        # Plan keys whose prompt was never issued (verify_scope dropped it), so
        # "no candidate in this camera" is the scope's doing and must not abstain.
        self._pose_only = set(pose_only or ())
        self._tol = float(tol)
        self._by_base: Dict[str, List[Tuple[str, Any, Optional[np.ndarray]]]] = {}
        for det in detections or ():
            label = str(getattr(det, "label", "") or "")
            self._by_base.setdefault(canon_label(label), []).append(
                (label, det, detection_xyz(det))
            )
        self._cache: Dict[str, Tuple[Optional[Any], str]] = {}

    # public surface consumed by SceneView

    def lookup(self, label: Any) -> Optional[Any]:
        return self._resolve(label)[0]

    def explain(self, label: Any) -> str:
        return self._resolve(label)[1]

    def _resolve(self, label: Any) -> Tuple[Optional[Any], str]:
        key = str(label or "")
        if key not in self._cache:
            self._cache[key] = self._bind(key)
            det, why = self._cache[key]
            if det is None:
                logger.info("Identity binder: %r unbound (%s)", key, why)
            else:
                logger.info("Identity binder: %r -> %s", key, why)
        return self._cache[key]

    # binding

    def _plan_siblings(self, base: str, exclude: Optional[str]):
        """Plan-time instances of the same base label, minus the bound one."""
        return [
            (k, detection_xyz(v))
            for k, v in self._plan.items()
            if canon_label(k) == base and k != exclude
        ]

    def _bind(self, label: str) -> Tuple[Optional[Any], str]:
        base = canon_label(label)
        cands = list(self._by_base.get(base, ()))
        plan_key = _plan_key(self._plan, label)
        if not cands:
            if plan_key is not None and plan_key in self._pose_only:
                plan_det = dict.get(self._plan, plan_key)
                if detection_xyz(plan_det) is None:
                    return None, f"reference {plan_key!r} has no plan-time 3d anchor"
                return _plan_anchored_reference(plan_det, None), (
                    f"plan-time pose of {plan_key!r} (prompt not issued: "
                    f"verification.scope.anchor_references)"
                )
            return None, "not detected in this camera"

        # A sibling the tree also moved has no anchor left to claim a detection with.
        siblings = [
            s for s in self._plan_siblings(base, plan_key) if s[0] not in self._manipulated
        ]
        if plan_key is None:
            # No anchor to check against. A label the scene has exactly one of
            # is still the only candidate there is; anything else is ambiguous.
            if len(cands) == 1 and not siblings:
                return cands[0][1], f"sole {base!r} in view (no plan-time anchor)"
            return None, f"{len(cands)} {base!r} detections and no plan-time anchor"
        # No "sole candidate" shortcut past here. Being the only detection in
        # view is not evidence of identity: an object the tree never grasped
        # that is now 28cm from where it was planned is either a different
        # object or one that moved unobserved, and binding to it is exactly the
        # wrong-object pass this class exists to stop. Geometry decides.
        if plan_key in self._manipulated:
            return self._bind_moved(base, cands, siblings)
        if plan_key in self._plan_anchored:
            return self._bind_reference(cands, plan_key)
        return self._bind_static(base, cands, plan_key)

    def _bind_reference(self, cands, plan_key) -> Tuple[Optional[Any], str]:
        """A reference frame the tree never touched: use its plan-time pose.

        Re-detecting the container of a containment predicate is a pure loss.
        The tree does not touch it, a bowl on a table does not move on its own,
        and the object JUST PLACED occludes its interior, so the surviving
        mask is a partial rim and the depth behind those pixels is sampled off
        the placed object, which slides the backprojected centroid along the
        camera ray. Measured: 8.5cm (birdview), 13.7cm and 17.3cm (sideview) of
        apparent motion on a bowl of radius 8cm, on places that physically
        succeeded. A successful place is exactly what corrupts this measurement.

        The re-detection is still REQUIRED to exist: this method is only
        reached with a non-empty candidate list, so a container that vanished
        from a camera's view still abstains there. What the re-detection may no
        longer do is supply the pose.

        Disagreement policy: a re-detection far from the plan-time anchor is
        LOGGED (the distance goes into the bind explanation and the trace) but
        does not change the binding and does not force an abstain. There is no
        honest threshold available: the drift induced by occlusion is
        unbounded along the camera ray, so any cutoff loose enough to admit the
        17.3cm case is loose enough to admit a container that genuinely moved.
        What guards that case instead is presence (above), the exclusion of any
        container the tree itself displaced (``displaced_labels``), and the
        degenerate-extent abstain in success_predicates.
        """
        plan_det = dict.get(self._plan, plan_key)
        anchor = detection_xyz(plan_det)
        if anchor is None:
            return None, f"reference {plan_key!r} has no plan-time 3d anchor"
        scored = sorted(
            ((_xy_dist(c[2], anchor), c) for c in cands if c[2] is not None),
            key=lambda t: t[0],
        )
        if scored:
            nearest = scored[0][1]
            seen = f"re-detected {scored[0][0] * 100:.1f}cm away"
        else:
            nearest = cands[0]
            seen = "re-detected without 3d"
        ref = _plan_anchored_reference(plan_det, nearest[1])
        return ref, (
            f"plan-time pose of {plan_key!r} ({seen}; the re-detection is not "
            f"used as the reference frame)"
        )

    def _bind_moved(self, base, cands, siblings) -> Tuple[Optional[Any], str]:
        """The grasped object is whatever the stationary siblings do not claim."""
        unclaimed = list(cands)
        for name, anchor in siblings:
            if anchor is None:
                return None, f"un-manipulated {name!r} has no plan-time 3d anchor"
            near = []
            for c in unclaimed:
                d = _xy_dist(c[2], anchor)  # None (no 3d) is not a match; 0.0 is
                if d is not None and d <= self._tol:
                    near.append(c)
            if len(near) != 1:
                return None, (
                    f"{len(near)} detections within {self._tol * 100:.0f}cm of un-manipulated "
                    f"{name!r}; identity ambiguous"
                )
            unclaimed.remove(near[0])
        if len(unclaimed) != 1:
            return None, f"{len(unclaimed)} unaccounted {base!r} detections after elimination"
        if unclaimed[0][2] is None and len(cands) > 1:
            return None, f"leftover {base!r} carries no 3d; identity unconfirmable"
        return unclaimed[0][1], f"{unclaimed[0][0]!r} by elimination ({len(siblings)} sibling(s))"

    def _bind_static(self, base, cands, plan_key) -> Tuple[Optional[Any], str]:
        """An object the tree never touched must still be where it was."""
        anchor = detection_xyz(dict.get(self._plan, plan_key))
        if anchor is None:
            return None, f"{plan_key!r} has no plan-time 3d anchor and {len(cands)} candidates"
        scored = sorted(
            ((_xy_dist(c[2], anchor), c) for c in cands if c[2] is not None),
            key=lambda t: t[0],
        )
        if not scored:
            return None, f"no {base!r} candidate carries 3d"
        if len(scored) < len(cands):
            return None, f"{len(cands) - len(scored)} {base!r} candidate(s) lack 3d; ambiguous"
        tol = self._static_tol(dict.get(self._plan, plan_key))
        best_d, best = scored[0]
        if best_d > tol:
            return None, f"nearest {base!r} is {best_d * 100:.1f}cm from its plan-time position"
        if len(scored) > 1 and scored[1][0] <= tol:
            return None, f"two {base!r} candidates within {tol * 100:.0f}cm; ambiguous"
        return best[1], f"{best[0]!r} {best_d * 100:.1f}cm from its plan-time position"

    def _static_tol(self, plan_det: Any) -> float:
        """Drift a stationary object's *measured centroid* may show.

        The flat tolerance assumes the mask is stable, which holds for an
        object nothing happened to. It does NOT hold for the container just
        placed into: the object now occupying it occludes its interior, so the
        surviving mask is a partial rim whose centroid sits well off the true
        centre (measured: mask fill 0.75 -> 0.37 and a 7.3cm centroid shift on
        a bowl that did not move).

        A stationary container's centroid cannot drift beyond its own radius,
        so that radius is the physically meaningful bound. Objects with no
        measured extent (no slots, no obb_minor_m) keep the flat tolerance;
        widening those is what would let a utensil dropped BESIDE the tray
        bind to the tray's own anchor.

        Never shrinks below the flat value.
        """
        obb = container_obb_xy(plan_det) if plan_det is not None else None
        if obb is None:
            return self._tol
        _, half_a, half_b, _ = obb
        return max(self._tol, float(max(half_a, half_b)))
