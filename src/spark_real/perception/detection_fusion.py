"""Fuse a second detector's boxes with SAM3's masks, and repair bad masks.

Two capabilities, deliberately independent so the useful one does not wait on
the unavailable one:

  DetectionGate(...)  quality gate + ASPIRE repair. Works with NO new model.
  BoxProposer         optional second opinion, folded in when available.

FUSION RULE
-----------
Association. For every SAM3 detection with a mask, the mask's tight bbox is
scored by IoU against every proposal box. The best-IoU proposal above
`assoc_iou` is the partner. Association is CLASS-AGNOSTIC by design: a box
that overlaps says "an independently-trained detector also sees an object
here", which is worth having even when the class names differ.

Confidence combination. Log-odds pooling, i.e. treat the two detectors as
conditionally-independent evidence and add their logits:

    fused_logit = logit(sam3_conf) + w * logit(prop_conf)

`w` is the weight on the second opinion (< 1: corroborating, not
authoritative). w_class applies when the labels agree through the synonym
map, w_box when only the geometry agrees.

Disagreement, all four cases spelled out:

  AGREE_CLASS  box overlaps and names the same thing -> pool with w_class.
  AGREE_BOX    box overlaps, names something else -> pool with w_box, and cap
               the result at `box_only_ceiling`. Geometric corroboration
               cannot certify identity, only presence.
  MISS         the label IS in the proposer's vocabulary, the label is one the
               proposer is TRUSTED for (see below), and NO box overlaps ->
               genuine disagreement, subtract `miss_penalty`.
  ABSTAIN      the label is OUT of vocabulary, the label is out of the
               proposer's per-label allowlist, or the proposer was unavailable
               -> confidence UNCHANGED. This is the case that matters most for
               correctness here: RF-DETR is closed-set COCO, and
               `tray`/`bin`/`block`/`pen` are not COCO classes. Reading silence
               about an out-of-vocab object as a veto would delete four of the
               seven corpus tasks.

  ORPHAN       a box with no SAM3 mask. A detection is never synthesised from
               a box: no mask means no geometry. The box is retained as a
               geometric prompt for ASPIRE step 2.

PER-LABEL COMPETENCE (`proposer.labels`)
----------------------------------------
Vocabulary membership is necessary but NOT sufficient. A label can be in the
closed set and still be one the detector is bad at, and then its silence gets
scored as disagreement against a correct mask ("plushie" resolves into COCO
via the DEFAULT_SYNONYMS "teddy bear" entry, so covers() calls it
in-vocabulary, yet RF-DETR boxes no teddy bear and docks a correct SAM3 mask
under the verify gate's 0.35).

`proposer.labels` is therefore an ALLOWLIST of the labels this detector has
earned an opinion on. A detection outside it takes the ABSTAIN path whole: no
association, no partner, no bonus, no penalty, fused == sam3 bit for bit. An
EMPTY or ABSENT list means "no per-label opinion", i.e. the vocabulary alone
gates, so this is opt-in and cannot silently change an existing deployment.

Matching reuses `_canonical_class`, the same lookup `covers()` runs, on BOTH
sides of the comparison -- so synonyms (mug -> cup), regular plurals (bowls ->
bowl), instance suffixes (bowl 2 -> bowl) and the sub-part prompts SAM3 is
really asked for (knife handle -> knife) all resolve onto one entry. There is
no second matching scheme. An entry that resolves to nothing in the vocabulary
(e.g. `tray`) is compared as plain text; it can never be a MISS anyway, since
covers() already abstains on it.

WHERE THE FUSED NUMBER LANDS
----------------------------
The gate stamps three fields and rewrites nothing else by default:

  low_quality        already consumed: control/grasp_strategy.py forces
                     top-down on it.
  fused_confidence   the pooled number, gating grasp.yaw_min_conf. Written to
                     its OWN field, not over `confidence`, so the planner
                     prompt and the primitive trace keep reporting what SAM3
                     actually said. grasp_strategy prefers it when present.
  axis_trust         mask-measured counterpart of perception's obb_confidence,
                     gating grasp.yaw_min_obb_conf. min() of the two wins, so
                     either measurement can veto a wrist rotation alone.

`write_confidence: true` additionally overwrites `confidence` for callers who
want one number everywhere; it is OFF because it changes planner text.

GETTING THOSE STAMPS TO THE EXECUTOR
------------------------------------
The executor does not read ObjectDetection; it reads a plain dict built in
pipeline_execution.execute(). `fusion_map_fields()` below is the one function
that names the keys, so the dict builder needs a single line and the emission
convention lives here next to the semantics it has to honour: a field is
emitted only when it was MEASURED, because grasp_strategy reads an absent key
as unknown and a present low value as a veto.
"""


from __future__ import annotations


from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence
from typing import Dict, List, Optional, Sequence
from typing import Optional
import cv2
import logging
import math
import numpy as np
import os


from spark_core.util import det_field


logger = logging.getLogger(__name__)

# COCO class -> the corpus/prompt words that should count as the same thing.
# Only used to decide "did the two detectors name the same object", never to
# rename anything downstream.
DEFAULT_SYNONYMS: Dict[str, tuple] = {
    "teddy bear": ("plushie", "plush", "stuffed animal", "toy", "teddy"),
    "bowl": ("bowl", "dish"),
    "knife": ("knife", "blade"),
    "spoon": ("spoon",),
    "fork": ("fork",),
    "cup": ("cup", "mug"),
    "bottle": ("bottle",),
    "scissors": ("scissors",),
    "book": ("book",),
    "cell phone": ("phone", "cellphone"),
}


# Sub-part phrasings. SAM3 answers "knife handle" much better than it answers
# "the graspable end of a knife", and a handle mask has a real major axis where
# a whole-tool mask often does not.
SUBPART_SUFFIXES: tuple = ("handle", "grip")


def default_alt_prompts(label: str, elongated: bool = False, limit: int = 4) -> List[str]:
    """Reprompt vocabulary for ASPIRE when the caller supplies none.

    Needs no second model: synonyms come from DEFAULT_SYNONYMS, sub-parts from
    SUBPART_SUFFIXES. `elongated` (the detection claimed a major axis) puts the
    sub-part phrasings first, because that is the failure where a handle mask
    is the actual repair.
    """
    base = _strip_instance(label)
    if not base:
        return []
    words: List[str] = []
    for cls, syn in DEFAULT_SYNONYMS.items():
        if base == cls or base in syn or any(w in base for w in syn):
            words.extend([cls, *syn])
    subparts = [f"{base} {s}" for s in SUBPART_SUFFIXES]
    ordered = subparts + words if elongated else words + subparts

    out: List[str] = []
    for w in ordered:
        w = w.strip().lower()
        if w and w != base and w not in out:
            out.append(w)
    return out[:limit]


@dataclass
class BoxProposal:
    """One box from a secondary detector, in pixels of the source image."""

    label: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) pixels
    source: str = "unknown"

    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class ProposalSet:
    """Everything one proposer pass returns.

    `available` False means the backend could not run. Callers MUST treat that
    as abstain -- an unavailable detector is not a detector that saw nothing.
    """

    proposals: List[BoxProposal] = field(default_factory=list)
    vocabulary: frozenset = frozenset()
    available: bool = True
    source: str = "unknown"
    note: str = ""

    def covers(self, label: str, synonyms: Optional[Dict[str, tuple]] = None) -> bool:
        """True when `label` is inside this detector's closed vocabulary."""
        if not self.available or not self.vocabulary:
            return False
        return _canonical_class(label, self.vocabulary, synonyms) is not None


class BoxProposer:
    """Interface. Implementations must not raise from `propose`."""

    name = "base"

    def vocabulary(self) -> frozenset:
        raise NotImplementedError

    def propose(self, rgb: np.ndarray, labels: Sequence[str] = ()) -> ProposalSet:
        raise NotImplementedError


class NullProposer(BoxProposer):
    """Always abstains. The default, so nothing changes until opt-in."""

    name = "null"

    def vocabulary(self) -> frozenset:
        return frozenset()

    def propose(self, rgb, labels=()) -> ProposalSet:
        return ProposalSet(available=False, source=self.name, note="disabled")


class StaticProposer(BoxProposer):
    """Replays a fixed list. Real code path, used by tests and by replaying a
    detector that was run offline."""

    name = "static"

    def __init__(self, proposals: Sequence[BoxProposal], vocabulary: Sequence[str] = ()):
        self._proposals = list(proposals)
        self._vocab = frozenset(v.lower() for v in vocabulary) or frozenset(
            p.label.lower() for p in self._proposals
        )

    def vocabulary(self) -> frozenset:
        return self._vocab

    def propose(self, rgb, labels=()) -> ProposalSet:
        return ProposalSet(
            proposals=list(self._proposals),
            vocabulary=self._vocab,
            available=True,
            source=self.name,
        )


class RFDETRProposer(BoxProposer):
    """RF-DETR backend. Fails loudly when the package is missing.

    Kept as a real adapter rather than a stub so that installing the package
    is the only thing standing between here and a working second opinion.
    """

    name = "rfdetr"
    INSTALL_HINT = (
        "rfdetr is not installed in this env. "
        "conda run -n spark_conda pip install rfdetr  "
        "(~16 MB of wheels, does not touch torch/torchvision/numpy; the "
        "checkpoint is a separate 135 MB-1.5 GB runtime download). "
        "NOTE: RF-DETR is closed-set over 80 COCO classes and cannot replace "
        "SAM3's open-vocabulary role."
    )

    def __init__(self, variant: str = "medium", threshold: float = 0.35, device: str = "cuda"):
        try:
            import rfdetr  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(f"RFDETRProposer unavailable: {self.INSTALL_HINT}") from exc

        from rfdetr import RFDETRMedium, RFDETRNano  # noqa: F401  (import-time check)

        self.threshold = float(threshold)
        self.variant = variant
        builder = {"nano": RFDETRNano, "medium": RFDETRMedium}.get(variant, RFDETRMedium)
        self._model = builder(device=device)
        self._vocab = frozenset(c.lower() for c in self._coco_classes())

    @staticmethod
    def _coco_classes() -> Sequence[str]:
        from rfdetr.assets.coco_classes import COCO_CLASSES

        if isinstance(COCO_CLASSES, dict):
            return [str(v) for v in COCO_CLASSES.values()]
        return [str(v) for v in COCO_CLASSES]

    def vocabulary(self) -> frozenset:
        return self._vocab

    def propose(self, rgb, labels=()) -> ProposalSet:
        try:
            from PIL import Image

            det = self._model.predict(Image.fromarray(rgb), threshold=self.threshold)
            out = []
            for xyxy, conf, cid in zip(det.xyxy, det.confidence, det.class_id):
                out.append(
                    BoxProposal(
                        label=self._class_name(int(cid)),
                        confidence=float(conf),
                        box=tuple(float(v) for v in xyxy),
                        source=self.name,
                    )
                )
            return ProposalSet(out, self._vocab, True, self.name)
        except Exception as exc:  # noqa: BLE001 - a detector must never stop perception
            logger.warning("[proposer] rfdetr pass failed (%s); abstaining", exc)
            return ProposalSet(available=False, source=self.name, note=str(exc))

    def _class_name(self, cid: int) -> str:
        """Map a model class id to its COCO name.

        COCO_CLASSES is keyed by CATEGORY ID (1..90, 80 entries, with gaps),
        so it must be looked up by key, not by position: indexing .values()
        positionally shifts every label and drops the ids past the 80th (id
        88, "teddy bear", would come back as the string "88").
        """
        from rfdetr.assets.coco_classes import COCO_CLASSES

        if isinstance(COCO_CLASSES, dict):
            name = COCO_CLASSES.get(cid)
            if name is None and 0 <= cid < len(COCO_CLASSES):
                # Some builds emit contiguous indices instead of category ids.
                name = list(COCO_CLASSES.values())[cid]
            return str(name).lower() if name is not None else str(cid)
        classes = list(COCO_CLASSES)
        return str(classes[cid]).lower() if 0 <= cid < len(classes) else str(cid)


class TorchvisionCOCOProposer(BoxProposer):
    """COCO detector from torchvision. The one second opinion that RUNS HERE.

    Same closed-set 80-class COCO signal RF-DETR would give, with no install:
    torchvision is already a hard dependency of this env. CPU by default --
    the live server owns the GPU, and 1.9 s on CPU is cheap next to one 29 s
    pick-and-place.

    Fails loudly: a missing weight cache with no network raises rather than
    quietly proposing nothing.
    """

    name = "torchvision"
    VARIANTS = ("fasterrcnn", "fasterrcnn_mobilenet", "retinanet")
    INSTALL_HINT = (
        "torchvision.models.detection is unavailable. The COCO weights are a "
        "one-time ~160 MB download into the torch hub cache (torch.hub.get_dir()); with no "
        "network, pre-seed that cache on a connected machine."
    )

    def __init__(self, variant: str = "fasterrcnn", threshold: float = 0.35, device: str = "cpu"):
        try:
            import torch
            from torchvision.models import detection as tvdet
        except ImportError as exc:
            raise RuntimeError(f"TorchvisionCOCOProposer unavailable: {self.INSTALL_HINT}") from exc

        builders = {
            "fasterrcnn": (tvdet.fasterrcnn_resnet50_fpn, tvdet.FasterRCNN_ResNet50_FPN_Weights),
            "fasterrcnn_mobilenet": (
                tvdet.fasterrcnn_mobilenet_v3_large_fpn,
                tvdet.FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            ),
            "retinanet": (tvdet.retinanet_resnet50_fpn, tvdet.RetinaNet_ResNet50_FPN_Weights),
        }
        if variant not in builders:
            raise RuntimeError(
                f"unknown torchvision variant {variant!r}; pick one of {self.VARIANTS}"
            )
        builder, weights_enum = builders[variant]
        weights = weights_enum.COCO_V1
        try:
            model = builder(weights=weights)
        except Exception as exc:  # noqa: BLE001 - no network + no cache lands here
            raise RuntimeError(
                f"TorchvisionCOCOProposer could not load {variant} COCO weights "
                f"({exc}). {self.INSTALL_HINT}"
            ) from exc

        self._torch = torch
        self.variant = variant
        self.threshold = float(threshold)
        self.device = str(device)
        self._model = model.eval().to(self.device)
        # "N/A" placeholders exist in the COCO category list; drop them so they
        # cannot be matched as a class name.
        self._categories = [str(c) for c in weights.meta["categories"]]
        self._vocab = frozenset(c.lower() for c in self._categories if c and c != "N/A")
        logger.info(
            "[proposer] torchvision %s on %s, %d COCO classes, threshold %.2f",
            variant,
            self.device,
            len(self._vocab),
            self.threshold,
        )

    def vocabulary(self) -> frozenset:
        return self._vocab

    def propose(self, rgb, labels=()) -> ProposalSet:
        try:
            arr = np.ascontiguousarray(np.asarray(rgb))
            if arr.ndim != 3 or arr.shape[2] < 3:
                return ProposalSet(available=False, source=self.name, note="rgb is not HxWx3")
            x = self._torch.from_numpy(arr[:, :, :3].copy()).permute(2, 0, 1).float()
            if arr.dtype == np.uint8:
                x = x / 255.0
            x = x.to(self.device)
            with self._torch.no_grad():
                out = self._model([x])[0]
            props = []
            for box, score, cid in zip(out["boxes"], out["scores"], out["labels"]):
                s = float(score)
                if s < self.threshold:
                    continue
                props.append(
                    BoxProposal(
                        label=self._class_name(int(cid)),
                        confidence=s,
                        box=tuple(float(v) for v in box.tolist()),
                        source=f"{self.name}:{self.variant}",
                    )
                )
            return ProposalSet(props, self._vocab, True, self.name)
        except Exception as exc:  # noqa: BLE001 - a detector must never stop perception
            logger.warning("[proposer] torchvision pass failed (%s); abstaining", exc)
            return ProposalSet(available=False, source=self.name, note=str(exc))

    def _class_name(self, cid: int) -> str:
        if 0 <= cid < len(self._categories):
            return self._categories[cid].lower()
        return str(cid)


_BACKENDS = {
    "null": NullProposer,
    "rfdetr": RFDETRProposer,
    "torchvision": TorchvisionCOCOProposer,
}

_PROPOSER_KWARGS = ("variant", "threshold", "device")


def build_proposer(cfg: Optional[dict]) -> BoxProposer:
    """Config-gated construction. Absent/off config -> NullProposer.

    A backend that was explicitly ASKED FOR and cannot be built RAISES. The
    operator turning fusion on and getting silence instead of protection is the
    failure mode that matters here; `strict: false` degrades to null instead.
    """
    cfg = dict(cfg or {})
    backend = str(cfg.get("backend", "null")).lower()
    env = os.environ.get("SPARK_BOX_PROPOSER")
    if env:
        backend = env.lower()

    if backend in ("null", "off", "none", ""):
        return NullProposer()
    if not cfg.get("enabled", False) and not env:
        # Backend named but never switched on. Not an error, not a warning.
        return NullProposer()

    strict = bool(cfg.get("strict", True))
    cls = _BACKENDS.get(backend)
    if cls is None:
        msg = f"unknown box proposer backend {backend!r}; known: {sorted(_BACKENDS)}"
        if strict:
            raise RuntimeError(msg)
        logger.warning("[proposer] %s; using null", msg)
        return NullProposer()
    try:
        return cls(**{k: v for k, v in cfg.items() if k in _PROPOSER_KWARGS})
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise RuntimeError(f"box proposer {backend!r} was requested but is unusable: {exc}")
        logger.warning("[proposer] %s unavailable (%s); using null", backend, exc)
        return NullProposer()


def _canonical_class(label: str, vocabulary, synonyms=None) -> Optional[str]:
    """Map a SAM3 prompt onto a vocabulary class, or None if out of vocab."""
    syn = DEFAULT_SYNONYMS if synonyms is None else synonyms
    lab = _strip_instance(label)
    if lab in vocabulary:
        return lab
    for cls, words in syn.items():
        if cls not in vocabulary:
            continue
        if lab == cls or any(w in lab for w in words):
            return cls
    # last resort: a vocabulary word appearing inside a compound prompt
    for cls in vocabulary:
        if cls in lab:
            return cls
    return None


def _strip_instance(label: str) -> str:
    """'blue block 2' -> 'blue block'."""
    parts = str(label).strip().lower().split()
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts)



_MIN_PIXELS = 24  # below this nothing is measurable


@dataclass
class QualityThresholds:
    """All tunable. Loaded from config; these defaults are the shipped policy."""

    min_pixels: int = _MIN_PIXELS
    min_fill: float = 0.45  # mask_area / OBB area
    min_solidity: float = 0.70  # mask_area / convex hull area; a spoon (bowl + thin handle) sits at 0.73-0.79
    min_component_frac: float = 0.85  # largest connected component share
    max_angle_swing_deg: float = 12.0  # axis movement under erode/dilate
    max_ar_inflation: float = 1.6  # reported_ar / mask_ar ceiling, see below
    min_mask_ar: float = 1.5  # a yaw needs elongation in the mask itself
    perturb_frac: float = 0.06  # erosion radius as a fraction of sqrt(area)


@dataclass
class MaskQuality:
    """Measured geometry trust for one mask."""

    pixels: int = 0
    mask_aspect_ratio: float = 1.0
    major_axis_deg: float = 0.0
    fill: float = 0.0
    solidity: float = 0.0
    component_frac: float = 0.0
    angle_swing_deg: float = 180.0
    ar_inflation: float = 1.0  # reported_ar / mask_ar; >1 means depth stretched it
    measured: bool = False
    angle_trustworthy: bool = False
    # WHY the axis is untrustworthy, split by whether a different mask could
    # repair it. Both of these are statements about the object's SHAPE that
    # the pixels AGREE with, so no re-prompt can change them; the mask-
    # integrity failures (fill / components / swing) are the ones a better
    # mask fixes. Consumers wanting "can I aim a wrist" still read
    # angle_trustworthy -- these only explain it.
    shape_round: bool = False   # mask has no major axis to align a wrist to
    ar_inflated: bool = False   # world OBB stretched by depth noise, not text
    geometry_trust: float = 0.0  # [0,1], 0 = do not use this shape
    # Mask-measured answer to the SAME question perception's obb_confidence
    # asks: how well-defined is this OBB's major axis. 0.0 when there is no
    # usable axis at all, so it can gate grasp.yaw_min_obb_conf directly.
    axis_trust: float = 0.0
    reasons: list = field(default_factory=list)

    def describe(self) -> str:
        if not self.measured:
            return "unmeasurable mask"
        return (
            f"trust={self.geometry_trust:.2f} axis_trust={self.axis_trust:.2f} "
            f"fill={self.fill:.2f} "
            f"solid={self.solidity:.2f} cc={self.component_frac:.2f} "
            f"swing={self.angle_swing_deg:.1f}deg mask_ar={self.mask_aspect_ratio:.2f}"
            + (f" [{', '.join(self.reasons)}]" if self.reasons else "")
        )


def _as_u8(mask) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim != 2 or m.size == 0:
        return None
    return (m > 0).astype(np.uint8)


def _major_axis(mask_u8):
    """(angle_deg in [0,180), major_px, minor_px, obb_area) from minAreaRect."""
    pts = cv2.findNonZero(mask_u8)
    if pts is None or len(pts) < 5:
        return None
    (_, _), (w, h), ang = cv2.minAreaRect(pts)
    if w <= 0 or h <= 0:
        return None
    # minAreaRect's angle names the `w` edge; rotate to name the LONG edge.
    major, minor = (w, h) if w >= h else (h, w)
    deg = ang if w >= h else ang + 90.0
    return deg % 180.0, float(major), float(minor), float(w * h)


def _angdiff(a: float, b: float) -> float:
    """Difference of two undirected axes, in [0,90]."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def measure_mask(
    mask, reported_aspect_ratio: Optional[float] = None, th: Optional[QualityThresholds] = None
) -> MaskQuality:
    """Score one mask. Never raises."""
    th = th or QualityThresholds()
    q = MaskQuality()
    try:
        m = _as_u8(mask)
        if m is None:
            q.reasons.append("no mask")
            return q
        q.pixels = int(m.sum())
        if q.pixels < th.min_pixels:
            q.reasons.append(f"only {q.pixels}px")
            return q

        axis = _major_axis(m)
        if axis is None:
            q.reasons.append("degenerate OBB")
            return q
        deg, major, minor, obb_area = axis
        q.major_axis_deg = deg
        q.mask_aspect_ratio = major / max(minor, 1e-6)
        q.fill = q.pixels / max(obb_area, 1e-6)

        hull = cv2.convexHull(cv2.findNonZero(m))
        q.solidity = q.pixels / max(cv2.contourArea(hull), 1e-6)

        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            q.component_frac = float(areas.max()) / max(q.pixels, 1)

        q.angle_swing_deg = _axis_swing(m, deg, th.perturb_frac)

        if reported_aspect_ratio is not None:
            q.ar_inflation = float(reported_aspect_ratio) / max(q.mask_aspect_ratio, 1e-6)

        q.measured = True
        _verdict(q, th, reported_aspect_ratio)
        return q
    except Exception as exc:  # noqa: BLE001 - scoring must never stop perception
        logger.warning("[mask-quality] measure failed (%s)", exc)
        q.reasons.append("measure error")
        return q


def _axis_swing(m: np.ndarray, base_deg: float, perturb_frac: float) -> float:
    """Max axis movement under a small erode and dilate.

    This is the direct test of whether the axis is real or is being set by a
    handful of boundary pixels.
    """
    k = max(1, int(round(perturb_frac * math.sqrt(float(m.sum())))))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    swing = 0.0
    for op in (cv2.MORPH_ERODE, cv2.MORPH_DILATE):
        pert = cv2.morphologyEx(m, op, kern)
        if int(pert.sum()) < _MIN_PIXELS:
            continue  # eroded away: contributes no evidence either way
        axis = _major_axis(pert)
        if axis is not None:
            swing = max(swing, _angdiff(base_deg, axis[0]))
    return swing


def _verdict(q: MaskQuality, th: QualityThresholds, reported_ar) -> None:
    """Fold the metrics into geometry_trust and angle_trustworthy."""
    checks = []
    if q.fill < th.min_fill:
        q.reasons.append(f"fill {q.fill:.2f}<{th.min_fill:.2f}")
        checks.append(0.0)
    else:
        checks.append(min(1.0, (q.fill - th.min_fill) / max(1e-6, 1.0 - th.min_fill)))

    if q.solidity < th.min_solidity:
        q.reasons.append(f"solidity {q.solidity:.2f}<{th.min_solidity:.2f}")
        checks.append(0.0)
    else:
        checks.append(1.0)

    if q.component_frac < th.min_component_frac:
        q.reasons.append(f"fragmented cc={q.component_frac:.2f}")
        checks.append(0.0)
    else:
        checks.append(1.0)

    if q.angle_swing_deg > th.max_angle_swing_deg:
        q.reasons.append(f"axis swings {q.angle_swing_deg:.1f}deg")
        checks.append(0.0)
    else:
        checks.append(1.0 - q.angle_swing_deg / max(th.max_angle_swing_deg, 1e-6))

    # Directional on purpose. Backprojection noise STRETCHES the world OBB, it
    # does not shrink it, so reported >> mask is a fabricated axis. The reverse
    # (mask longer than world) is ordinary perspective foreshortening, benign.
    inflated = reported_ar is not None and q.ar_inflation > th.max_ar_inflation
    q.ar_inflated = bool(inflated)
    if inflated:
        q.reasons.append(
            f"reported ar {float(reported_ar):.2f} is {q.ar_inflation:.1f}x the "
            f"mask ar {q.mask_aspect_ratio:.2f}"
        )
        checks.append(0.0)

    # The record claims elongation the mask does not have, so there is no
    # major axis to align a wrist to.
    round_mask = q.mask_aspect_ratio < th.min_mask_ar
    q.shape_round = bool(round_mask)
    if round_mask:
        q.reasons.append(f"mask ar {q.mask_aspect_ratio:.2f} is round")

    q.geometry_trust = float(min(checks)) if checks else 0.0
    # A yaw needs BOTH a stable axis and a shape that actually has one.
    q.angle_trustworthy = (
        q.measured
        and q.angle_swing_deg <= th.max_angle_swing_deg
        and q.fill >= th.min_fill
        and q.component_frac >= th.min_component_frac
        and not inflated
        and not round_mask
    )
    # No usable axis -> 0, so a min() against perception's obb_confidence
    # vetoes. Deliberately NOT scaled by elongation: mask_ar shorter than the
    # world OBB is foreshortening, and min_mask_ar above is already the one
    # elongation policy. This is a stability/quality number only.
    q.axis_trust = float(q.geometry_trust) if q.angle_trustworthy else 0.0



AGREE_CLASS = "agree_class"

AGREE_BOX = "agree_box"

MISS = "miss"

ABSTAIN = "abstain"

_EPS = 1e-4


@dataclass
class FusionConfig:
    """Config-gated. Every default here is the OFF / no-change behaviour."""

    enabled: bool = False  # master switch for the whole gate
    # False => the gate annotates low_quality but never rewrites confidence.
    write_confidence: bool = False
    assoc_iou: float = 0.40
    w_class: float = 0.60
    w_box: float = 0.25
    box_only_ceiling: float = 0.85
    # MEASURED, not guessed. Log-odds cost of "the second detector saw nothing
    # here". The correct value is log(1 - recall) for that detector on THIS
    # rig's objects, and the corroboration scan in tests/fusion_plushie_demo.py
    # measures it: a COCO fasterrcnn boxed a real, visible plushie in 3 of 8
    # real frames, so 1-recall = 0.625 and the penalty is -log(0.625) = 0.47.
    # Re-run the scan to recalibrate for a different backend.
    miss_penalty: float = 0.47  # logits
    low_quality_conf: float = 0.40  # mirrors perception.reprompt_min_conf
    reprompt_max_attempts: int = 2
    # A reprompt result must be the SAME OBJECT. Without this, "teddy bear"
    # returning a mask on the far side of the table gets adopted because it
    # scored better, and the arm goes to the wrong place.
    reprompt_min_iou: float = 0.30
    proposer: dict = field(default_factory=dict)
    quality: QualityThresholds = field(default_factory=QualityThresholds)

    def proposer_labels(self) -> tuple:
        """`proposer.labels`, normalised. Empty tuple = gate on vocabulary only.

        Lives here rather than in build_proposer because it is a FUSION
        decision, not a detector one: the model still runs and still reports
        every box it sees -- orphan_boxes() needs the whole set to answer
        "what else is on the table"; the gate just declines to SCORE the
        labels it is bad at.
        """
        raw = self.proposer.get("labels") if isinstance(self.proposer, dict) else None
        if raw is None:
            return ()
        if isinstance(raw, str):
            raw = [raw]
        out = []
        for entry in raw:
            e = _strip_instance(entry)
            if e and e not in out:
                out.append(e)
        return tuple(out)

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "FusionConfig":
        cfg = dict(cfg or {})
        qcfg = cfg.pop("quality", None) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "quality"}
        kwargs = {k: v for k, v in cfg.items() if k in known}
        out = cls(**kwargs)
        for k, v in qcfg.items():
            if hasattr(out.quality, k):
                setattr(out.quality, k, v)
        return out


@dataclass
class FusionResult:
    """Per-detection verdict. Carries the whole audit trail."""

    label: str
    sam3_confidence: float
    fused_confidence: float
    agreement: str = ABSTAIN
    partner: Optional[BoxProposal] = None
    iou: float = 0.0
    quality: MaskQuality = field(default_factory=MaskQuality)
    low_quality: bool = False
    angle_trustworthy: bool = False
    reprompt_attempts: int = 0
    notes: list = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"{self.label}: sam3={self.sam3_confidence:.2f} -> "
            f"fused={self.fused_confidence:.2f} [{self.agreement}] "
            f"low_quality={self.low_quality} yaw_ok={self.angle_trustworthy} "
            f"| {self.quality.describe()}" + (f" | {'; '.join(self.notes)}" if self.notes else "")
        )


def _logit(p: float) -> float:
    p = min(max(float(p), _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def mask_bbox(mask) -> Optional[tuple]:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim != 2 or not m.any():
        return None
    ys, xs = np.nonzero(m > 0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(ua, 1e-9)


def _set_field(det, key, value) -> bool:
    """Write a field on a detection given as a dict OR a dataclass.

    The detection_map the executor actually reads is a plain dict.
    """
    if det is None:
        return False
    try:
        if hasattr(det, "__setitem__") and hasattr(det, "get"):
            det[key] = value
        else:
            setattr(det, key, value)
        return True
    except Exception as exc:  # noqa: BLE001 - a read-only detection is not fatal
        logger.warning("[gate] could not stamp %s on %r (%s)", key, type(det).__name__, exc)
        return False


# Perception-measured scalars copied verbatim onto the executor's
# `detection_map` entry, in ADDITION to the gate's own two. `rim_z_m` /
# `interior_z_m` / `height_samples` are the world-Z profile of the mask's depth
# cloud (perception.mask_geometry.mask_height_profile); control.place
# reads them to place the jaws relative to a container's rim and interior floor
# instead of at a planner-guessed `offset_z`. They ride this channel because it
# already implements the exact convention they need -- emit only when measured,
# so an absent key means "not measured" and the caller falls back.
MAP_GEOMETRY_FIELDS = (
    "rim_z_m",
    "interior_z_m",
    "height_samples",
    # The MASK-derived twins of detection_map's depth-derived orientation_angle
    # and aspect_ratio, so the executor's "depth disagrees with the mask"
    # fallback has the mask numbers to read.
    "mask_aspect_ratio",
    "plane_orientation_angle",
    "plane_aspect_ratio",
)


def fusion_map_fields(det) -> Dict[str, float]:
    """Perception's channel into the executor's `detection_map` dict.

    Emits a key ONLY when perception actually measured it. grasp_strategy reads
    an absent key as unknown and a present low value as a veto, so publishing
    a default 0.0 for a detection the gate never saw would kill every oriented
    grasp -- the same convention `obb_confidence` already follows, and the same
    one release_height depends on for its fallback.
    """
    out: Dict[str, float] = {}
    for key in ("fused_confidence", "axis_trust") + MAP_GEOMETRY_FIELDS:
        v = det_field(det, key, None)
        if v is None:
            continue
        try:
            out[key] = float(v)
        except (TypeError, ValueError):
            logger.warning("[gate] %s=%r on %r is not a number; dropped", key, v, det)
    return out


def _canon_or_raw(label: str, vocabulary) -> str:
    """Canonical vocabulary class for `label`, else the bare label.

    The SAME `_canonical_class` the vocabulary check runs, so synonyms, regular
    plurals, instance suffixes and sub-part prompts resolve identically on both
    sides of an allowlist comparison. The raw fallback keeps a non-vocabulary
    allowlist entry ('tray') meaningful as plain text instead of collapsing to
    None and matching every other out-of-vocabulary label.
    """
    return _canonical_class(label, vocabulary) or _strip_instance(label)


def proposer_trusts(label: str, proposals: ProposalSet, cfg: FusionConfig) -> bool:
    """Is this a label the proposer is allowed to have an opinion about?

    True for everything when no allowlist is configured, so a deployment that
    does not opt in is unchanged.
    """
    allow = cfg.proposer_labels()
    if not allow:
        return True
    canon = _canon_or_raw(label, proposals.vocabulary)
    return any(_canon_or_raw(entry, proposals.vocabulary) == canon for entry in allow)


def fuse_one(det, proposals: ProposalSet, cfg: FusionConfig) -> FusionResult:
    """Apply the fusion rule to a single detection. Pure, no I/O."""
    label = str(det_field(det, "label", "?"))
    sam3_conf = float(det_field(det, "confidence", 0.0))
    res = FusionResult(label=label, sam3_confidence=sam3_conf, fused_confidence=sam3_conf)

    res.quality = measure_mask(
        det_field(det, "mask", None),
        reported_aspect_ratio=det_field(det, "aspect_ratio", None),
        th=cfg.quality,
    )
    res.angle_trustworthy = res.quality.angle_trustworthy

    bbox = mask_bbox(det_field(det, "mask", None)) or det_field(det, "bbox", None)
    # Competence check FIRST. A label the proposer is not trusted for gets no
    # association at all: its boxes are not evidence here, so they must not
    # reach AGREE_BOX (a bonus) any more than its silence reaches MISS (a
    # penalty). Skipping the scan is also what leaves `partner` None, which is
    # what keeps ASPIRE from reprompting off an untrusted box.
    trusted = proposer_trusts(label, proposals, cfg)
    best, best_iou = None, 0.0
    if trusted and proposals.available and bbox is not None:
        for p in proposals.proposals:
            i = box_iou(bbox, p.box)
            if i > best_iou:
                best, best_iou = p, i

    in_vocab = trusted and proposals.covers(label)
    if best is not None and best_iou >= cfg.assoc_iou:
        res.partner, res.iou = best, best_iou
        same = _canonical_class(label, proposals.vocabulary) == _strip_instance(best.label)
        if same:
            res.agreement = AGREE_CLASS
            res.fused_confidence = _sigmoid(
                _logit(sam3_conf) + cfg.w_class * _logit(best.confidence)
            )
        else:
            res.agreement = AGREE_BOX
            res.fused_confidence = min(
                cfg.box_only_ceiling,
                _sigmoid(_logit(sam3_conf) + cfg.w_box * _logit(best.confidence)),
            )
            res.notes.append(f"box says '{best.label}', sam3 says '{label}'")
    elif proposals.available and in_vocab:
        res.agreement = MISS
        res.fused_confidence = _sigmoid(_logit(sam3_conf) - cfg.miss_penalty)
        res.notes.append(f"'{label}' is in the proposer vocabulary but no box overlapped")
    else:
        res.agreement = ABSTAIN
        if not proposals.available:
            res.notes.append(f"no proposer ({proposals.note or 'unavailable'})")
        elif not trusted:
            res.notes.append(
                f"'{label}' is not a label this proposer is trusted for "
                f"(proposer.labels); confidence untouched"
            )
        else:
            res.notes.append(f"'{label}' out of proposer vocabulary; confidence untouched")

    res.low_quality = bool(
        res.fused_confidence < cfg.low_quality_conf or not res.quality.angle_trustworthy
    )
    return res


def orphan_boxes(dets, proposals: ProposalSet, cfg: FusionConfig) -> List[BoxProposal]:
    """Boxes that matched no mask. Candidate geometric prompts for ASPIRE."""
    if not proposals.available:
        return []
    boxes = [mask_bbox(det_field(d, "mask", None)) or det_field(d, "bbox", None) for d in dets]
    boxes = [b for b in boxes if b is not None]
    out = []
    for p in proposals.proposals:
        if all(box_iou(b, p.box) < cfg.assoc_iou for b in boxes):
            out.append(p)
    return out




class DetectionGate:
    """Quality gate + ASPIRE repair over a list of detections.

    ASPIRE -- search, observe, REPROMPT, recover -- as a concrete policy.
    NO SECOND MODEL IS REQUIRED for any of it: the trigger is measured off the
    mask, and the reprompt vocabulary is generated from the label.

      TRIGGER   a detection is low_quality, i.e. fused confidence below
                `low_quality_conf` OR its mask geometry is not trustworthy
                enough to aim a wrist (`angle_trustworthy` False). Note this is
                a QUALITY trigger and is orthogonal to the count-gated
                escalation already in detect_for_task(), which asks a different
                question (did we find the right NUMBER of objects).

      STEP 1    REPROMPT conditioned on geometry, when and only when a second
                detector ASSOCIATED a box with this mask: re-run the base
                prompt boxed to the partner. That is an independent
                observation of the same object, so it goes first. Orphan
                boxes are NOT used here -- an orphan matched no mask by
                definition, so prompting with one returns a different object
                and rule 3 would refuse it. orphan_boxes() reports them for
                the caller's own missed-object search, a different problem.
      STEP 2    REPROMPT with alternate text. Caller-supplied `alt_prompts`
                win; absent an entry, box_proposals.default_alt_prompts()
                generates synonyms plus sub-part phrasings ("knife handle"),
                sub-parts FIRST when the detection claimed elongation, because
                that is the failure where a handle mask is the real repair.
                This step needs no second model and is a complete path alone.
      RECOVER   after `reprompt_max_attempts` (default 2), keep the best
                candidate seen and mark it low_quality=True. The live grasp
                resolver already turns low_quality into a top-down grasp, so
                the object is still picked, just not with an invented yaw. The
                decision is degraded; the object is never dropped.

    RETRY POLICY, all four rules:
      1. Budget. At most `reprompt_max_attempts` redetect() calls per
         detection, counted whether or not a candidate comes back. A capture
         is ~0.5 s and this runs inside the motion path.
      2. Monotone acceptance. A candidate replaces the incumbent only if
         confidence*geometry scores strictly higher, so a retry can never make
         things worse than not retrying.
      3. Same object. A candidate whose mask bbox overlaps the incumbent's by
         less than `reprompt_min_iou` is REFUSED however well it scores -- a
         better mask of a different object is the worst possible outcome.
      4. Early exit. The loop stops the moment a candidate clears the trigger;
         it does not keep shopping for a higher number.
    """

    def __init__(self, config: Optional[dict] = None, proposer=None):
        self.cfg = config if isinstance(config, FusionConfig) else FusionConfig.from_dict(config)
        self.proposer = proposer if proposer is not None else build_proposer(self.cfg.proposer)

    def score(self, detections: Sequence, rgb=None) -> List[FusionResult]:
        """Fuse and score without mutating anything."""
        props = ProposalSet(available=False, note="no rgb")
        if rgb is not None:
            try:
                props = self.proposer.propose(
                    rgb, [str(det_field(d, "label", "")) for d in detections]
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[gate] proposer raised (%s); abstaining", exc)
                props = ProposalSet(available=False, note=str(exc))
        self._last_proposals = props
        return [fuse_one(d, props, self.cfg) for d in detections]

    def apply(
        self,
        detections: Sequence,
        rgb=None,
        redetect: Optional[Callable[[str, Optional[tuple]], object]] = None,
        alt_prompts: Optional[Dict[str, Sequence[str]]] = None,
    ) -> List[FusionResult]:
        """Score, run ASPIRE on whatever fails, then stamp the detections.

        `redetect(prompt, box)` is supplied by the caller (the pipeline) and
        returns a replacement detection or None. Keeping it a callback is what
        lets this module stay free of any pipeline import and be tested with a
        fake. `box` is (x1,y1,x2,y2) pixels or None.
        """
        if not self.cfg.enabled:
            return []

        results = self.score(detections, rgb)
        for det, res in zip(detections, results):
            if res.low_quality and redetect is not None and (
                not self._shape_only(res) or res.partner is not None
            ):
                self._aspire(det, res, redetect, alt_prompts or {})
            self._stamp(det, res)
        return results

    def _shape_only(self, res: FusionResult) -> bool:
        """Can a DIFFERENT MASK of this object plausibly clear the trigger?

        A reprompt only changes the text, so it can only repair a bad mask,
        not a true statement about the object's shape. Reprompt on LOW
        CONFIDENCE (a better prompt may find the object) or on MASK-INTEGRITY
        failure (holes, fragments, an unstable axis). Never on shape alone: a
        round object has no yaw to recover, and an inflated world OBB is depth
        noise, not a text problem. Both still set angle_trustworthy False, so
        the resolver still grasps top-down; only the retry is skipped, never
        the safe degrade.
        """
        q = res.quality
        if res.fused_confidence < self.cfg.low_quality_conf:
            return False
        if not q.measured:
            return False
        th = self.cfg.quality
        if (
            q.fill < th.min_fill
            or q.component_frac < th.min_component_frac
            or q.angle_swing_deg > th.max_angle_swing_deg
        ):
            return False
        return bool(q.shape_round or q.ar_inflated)


    def _candidate_score(self, res: FusionResult) -> float:
        """Rank candidates by confidence AND usable geometry."""
        return res.fused_confidence * (0.5 + 0.5 * res.quality.geometry_trust)

    def _attempts(self, det, res: FusionResult, alt_prompts) -> List[tuple]:
        """(prompt, box) list in the order they will be tried."""
        base_label = _strip_instance(res.label)
        attempts: List[tuple] = []
        if res.partner is not None:
            attempts.append((base_label, res.partner.box))

        if self._shape_only(res):
            # Partner box only (it can return a differently-shaped mask). Text
            # cannot: see _shape_only.
            res.notes.append(
                "text reprompts skipped: low_quality is shape-only ("
                + ", ".join(
                    n
                    for n, on in (
                        ("round", res.quality.shape_round),
                        ("ar inflated", res.quality.ar_inflated),
                    )
                    if on
                )
                + "), which no prompt can change"
            )
            return attempts

        supplied = list(alt_prompts.get(base_label, ()) or [])
        if not supplied:
            # Order off the MEASURED mask elongation, never the reported one:
            # the reported number is the thing under suspicion. A long mask is
            # a tool, so ask for its handle; a round mask is a blob, so ask for
            # it by another name.
            elongated = res.quality.mask_aspect_ratio >= self.cfg.quality.min_mask_ar
            supplied = default_alt_prompts(base_label, elongated=elongated)
        attempts.extend((p, None) for p in supplied)
        return attempts

    def _aspire(self, det, res: FusionResult, redetect, alt_prompts) -> None:
        attempts = self._attempts(det, res, alt_prompts)
        incumbent_bbox = mask_bbox(det_field(det, "mask", None)) or det_field(det, "bbox", None)
        best_score = self._candidate_score(res)
        props = getattr(self, "_last_proposals", ProposalSet(available=False))

        for prompt, box in attempts[: self.cfg.reprompt_max_attempts]:
            res.reprompt_attempts += 1
            try:
                cand = redetect(prompt, box)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[aspire] reprompt %r failed (%s)", prompt, exc)
                continue
            if cand is None:
                continue
            if not self._same_object(incumbent_bbox, cand):
                res.notes.append(f"reprompt '{prompt}' rejected (different object)")
                continue
            cres = fuse_one(cand, props, self.cfg)
            cscore = self._candidate_score(cres)
            if cscore <= best_score:
                res.notes.append(f"reprompt '{prompt}' rejected ({cscore:.2f}<={best_score:.2f})")
                continue

            res.notes.append(
                f"reprompt '{prompt}'{' +box' if box else ''} accepted "
                f"({best_score:.2f}->{cscore:.2f})"
            )
            best_score = cscore
            self._adopt(det, cand)
            keep_attempts, keep_notes = res.reprompt_attempts, res.notes
            cres.reprompt_attempts, cres.notes = keep_attempts, keep_notes
            res.__dict__.update(cres.__dict__)
            if not res.low_quality:
                break

        if res.low_quality:
            res.notes.append("recovered as low_quality: top-down, no yaw")

    def _same_object(self, incumbent_bbox, cand) -> bool:
        """Retry rule 3. Unknown geometry on either side is not a rejection."""
        cand_bbox = mask_bbox(det_field(cand, "mask", None)) or det_field(cand, "bbox", None)
        if incumbent_bbox is None or cand_bbox is None:
            return True
        return box_iou(incumbent_bbox, cand_bbox) >= self.cfg.reprompt_min_iou

    @staticmethod
    def _adopt(det, cand) -> None:
        """Copy an accepted candidate's geometry onto the live detection."""
        for f in (
            "mask",
            "bbox",
            "confidence",
            "centroid_2d",
            "mask_area",
            "aspect_ratio",
            "orientation_angle",
            "obb_minor_m",
            "obb_confidence",
            "position_3d",
            "depth_meters",
        ):
            v = det_field(cand, f, None)
            if v is not None:
                _set_field(det, f, v)

    def _stamp(self, det, res: FusionResult) -> None:
        """Write the verdict onto the detection's reserved fields.

        `fused_confidence` and `axis_trust` are separate fields on purpose:
        they are the gate's channel into grasp_strategy, and writing them costs
        the planner prompt and the trace nothing.
        """
        _set_field(det, "low_quality", bool(res.low_quality))
        _set_field(det, "reprompt_attempts", int(res.reprompt_attempts))
        _set_field(det, "fused_confidence", float(res.fused_confidence))
        _set_field(det, "axis_trust", float(res.quality.axis_trust))
        # WHY the axis was refused, so a consumer can tell a DEPTH statement
        # from a MASK statement. low_quality alone collapses "the point cloud
        # stretched this OBB" (noisy, flips run to run) and "this object is
        # round" (stable, permanent) into one bit.
        _set_field(det, "ar_inflated", bool(res.quality.ar_inflated))
        _set_field(det, "shape_round", bool(res.quality.shape_round))
        _set_field(det, "mask_aspect_ratio", float(res.quality.mask_aspect_ratio))
        if self.cfg.write_confidence:
            _set_field(det, "confidence", float(res.fused_confidence))
