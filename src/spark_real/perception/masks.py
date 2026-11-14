"""Slot detector for compartmentalized trays / silverware drawers.

Given a birdview RGB-D capture and a SAM3 tray mask, find the individual
compartments ("slots") inside the tray so a heterogeneous batch of
objects (knife, fork, spoon) can each be placed in its own slot.

The key signal is surface normals computed from the depth image:
flat slot FLOORS have normals pointing world +Z, while slot DIVIDERS
have normals pointing horizontally. Pixels that are (a) inside the
tray mask, (b) horizontal-up by normal, and (c) at the LOW end of the
world-Z distribution form per-slot connected components.

Pipeline (mirrors the style in ``spark_perception.py``):

  1. Sobel on depth(meters) plus K (intrinsics) gives per-pixel cam-frame
     normals N_cam.
  2. Rotate to base frame: N_base = R_cam_to_base @ N_cam.
  3. Keep up-facing pixels (N_base.z > 0.9), then keep those within
     slot_floor_band_m of the 25th-percentile world-Z.
  4. Connected components on the surviving pixels become slots.
  5. Order slots by major-axis projection (world XY PCA of the tray mask).

Returns a list of dicts; each describes one slot in world frame.
Each slot also carries its floor's own long axis (axis_rad, axis_ar), so a
tray's layout is measured rather than assumed. Falls back to N evenly-spaced
stripes across the tray's short axis when the normal-based path produces too
few clusters (flat tray, no visible dividers, bad depth).

No SPARK server / no robot motion is touched by this module; it's a
pure perception utility intended to be called by the executor BEFORE
deciding placement targets.
"""


from __future__ import annotations


from pathlib import Path
from typing import Dict, List
from typing import List, Optional
import cv2
import json
import logging
import numpy as np
import os
import tempfile
import threading


from spark_core.mask_obb import _pca_obb
from spark_core.util import det_field


logger = logging.getLogger(__name__)

# A slot floor this elongated has a direction; a squarer one does not.
SLOT_AXIS_MIN_AR = 1.5


def _floor_axis(xy: np.ndarray) -> tuple:
    """(long-axis angle in rad, aspect ratio) of a slot floor's world-XY points."""
    vals, vecs = np.linalg.eigh(np.cov(xy.T))
    return float(np.arctan2(vecs[1, 1], vecs[0, 1])), float(np.sqrt(vals[1] / max(vals[0], 1e-12)))


def slot_axis_consensus(slots) -> Optional[float]:
    """Pixel-weighted direction (mod pi) of the elongated slot floors, or None.

    None when no floor has a direction, or when they disagree (a tray with
    perpendicular compartments has no single direction; ask per slot instead).
    """
    floors = [s for s in slots if s.get("axis_ar", 0.0) >= SLOT_AXIS_MIN_AR]
    total = sum(s["n_pixels"] for s in floors)
    if not total:
        return None
    v = sum(s["n_pixels"] * np.exp(2j * s["axis_rad"]) for s in floors)
    return float(np.angle(v)) / 2.0 if abs(v) >= 0.5 * total else None


def _compute_normals_from_depth(
    depth_m: np.ndarray,
    K: np.ndarray,
    smooth_ksize: int = 5,
) -> np.ndarray:
    """
    Per-pixel surface normals in camera frame from a metric depth map.

    For each pixel (u, v) with depth d:
        P_cam = ((u-cx)*d/fx, (v-cy)*d/fy, d)            # OpenCV
    Tangent vectors come from finite differences via Sobel on the depth
    image; the normal is the normalised cross product of the row/col
    tangents. Output normals always point toward the camera (z < 0 in
    OpenCV camera frame); the sign is handled on conversion to world.

    A small bilateral pre-smoothing tames depth quantisation noise so
    horizontal slot floors don't look bumpy.

    Returns (H, W, 3) float32. Invalid pixels (d<=0) get zeros.
    """
    h, w = depth_m.shape
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # Pre-smooth depth to suppress single-pixel noise without blurring
    # across slot dividers (bilateral preserves depth edges).
    d = depth_m.astype(np.float32)
    valid = d > 0.01
    d_smooth = cv2.bilateralFilter(d, d=5, sigmaColor=0.01, sigmaSpace=5)
    # Bilateral can zero out invalid neighbourhoods; keep originals there.
    d_smooth = np.where(valid, d_smooth, d)

    # Build pixel-grid coordinates once.
    us = np.arange(w, dtype=np.float32)
    vs = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    # Camera-frame XYZ for every pixel.
    X = (uu - cx) * d_smooth / fx
    Y = (vv - cy) * d_smooth / fy
    Z = d_smooth

    # Sobel along u (columns) and v (rows). Larger ksize averages out
    # more depth noise on the slot floor.
    ksize = smooth_ksize if smooth_ksize in (3, 5, 7) else 5
    dX_du = cv2.Sobel(X, cv2.CV_32F, 1, 0, ksize=ksize)
    dY_du = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=ksize)
    dZ_du = cv2.Sobel(Z, cv2.CV_32F, 1, 0, ksize=ksize)
    dX_dv = cv2.Sobel(X, cv2.CV_32F, 0, 1, ksize=ksize)
    dY_dv = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=ksize)
    dZ_dv = cv2.Sobel(Z, cv2.CV_32F, 0, 1, ksize=ksize)

    # Normal = (dP/dv) x (dP/du). Sign chosen so a flat surface in
    # front of the camera yields N_cam ~ (0, 0, -1) (pointing back at
    # the camera in OpenCV convention).
    nx = dY_dv * dZ_du - dZ_dv * dY_du
    ny = dZ_dv * dX_du - dX_dv * dZ_du
    nz = dX_dv * dY_du - dY_dv * dX_du

    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx /= norm
    ny /= norm
    nz /= norm

    normals = np.stack([nx, ny, nz], axis=-1)
    normals[~valid] = 0.0
    return normals


def _tray_axes_world(
    tray_mask: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    T_cam_to_base: np.ndarray,
) -> Optional[dict]:
    """
    World-XY PCA on the tray mask. Returns mean, major, minor (XY).

    Mirrors ``_world_xy_pca_obb`` in spark_perception.py, same idea,
    same OpenCV-convention backprojection. Returns None if the tray
    has too few valid depth pixels for a stable axis.
    """
    ys, xs = np.where(tray_mask > 0)
    if len(xs) < 50:
        return None
    depths = depth_m[ys, xs].astype(np.float64)
    valid = (depths > 0.01) & (depths < 10.0)
    if int(valid.sum()) < 50:
        return None
    xs_v = xs[valid].astype(np.float64)
    ys_v = ys[valid].astype(np.float64)
    d_v = depths[valid]
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx_k = float(K[0, 2])
    cy_k = float(K[1, 2])
    x_cam = (xs_v - cx_k) * d_v / fx
    y_cam = (ys_v - cy_k) * d_v / fy
    z_cam = d_v
    pts_cam = np.column_stack([x_cam, y_cam, z_cam])
    R = T_cam_to_base[:3, :3]
    t = T_cam_to_base[:3, 3]
    pts_world = (R @ pts_cam.T).T + t
    xy = pts_world[:, :2]
    mean_xy = xy.mean(axis=0)
    cov = np.cov((xy - mean_xy).T)
    if cov.ndim != 2 or cov.shape != (2, 2):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, 1]  # largest variance
    minor = eigvecs[:, 0]
    proj_mj = (xy - mean_xy) @ major
    proj_mn = (xy - mean_xy) @ minor
    return {
        "mean_xy": mean_xy,
        "major": major,
        "minor": minor,
        "major_len_m": float(proj_mj.max() - proj_mj.min()),
        "minor_len_m": float(proj_mn.max() - proj_mn.min()),
        "z_med": float(np.median(pts_world[:, 2])),
    }


def _backproject_pixels_to_world(
    ys: np.ndarray,
    xs: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    T_cam_to_base: np.ndarray,
) -> np.ndarray:
    """
    Backproject a batch of (v, u) image pixels to world XYZ.

    Returns (N, 3) float64 with NaN rows for pixels with invalid depth.
    """
    d = depth_m[ys, xs].astype(np.float64)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx_k = float(K[0, 2])
    cy_k = float(K[1, 2])
    x_cam = (xs.astype(np.float64) - cx_k) * d / fx
    y_cam = (ys.astype(np.float64) - cy_k) * d / fy
    z_cam = d
    pts_cam = np.column_stack([x_cam, y_cam, z_cam])
    R = T_cam_to_base[:3, :3]
    t = T_cam_to_base[:3, 3]
    pts_world = (R @ pts_cam.T).T + t
    pts_world[d <= 0.01] = np.nan
    return pts_world


def _fallback_axis_slots(
    tray_axes: dict,
    n_slots: int,
    lift_offset_m: float,
) -> List[dict]:
    """
    N evenly-spaced poses across the tray's short axis.

    Used when the normal-based detector finds <2 reliable components, so the
    layout is unknown: this assumes slots along the long side, the same default
    as resolve_slot_direction. Outermost centers are inset by half a stripe so
    the release pose stays inside the tray footprint, not on the rim.
    """
    if n_slots < 1:
        n_slots = 1
    mean_xy = tray_axes["mean_xy"]
    minor = tray_axes["minor"]
    L = tray_axes["minor_len_m"]
    # Inset by one half-slot from each end so centers don't sit on rims.
    slot_len = L / max(n_slots, 1)
    starts = -L / 2 + slot_len / 2 + slot_len * np.arange(n_slots)
    z = tray_axes["z_med"] + lift_offset_m
    slots = []
    for i, s in enumerate(starts):
        xy = mean_xy + s * minor
        slots.append(
            {
                "slot_idx": int(i),
                "world_xyz": np.array([xy[0], xy[1], z], dtype=np.float64),
                "n_pixels": 0,
                "confidence": 0.3,
                "mode": "fallback",
            }
        )
    return slots


# Public API.


def detect_slots(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    tray_mask: np.ndarray,
    K: np.ndarray,
    T_cam_to_base: np.ndarray,
    depth_scale: float = 1.0,
    n_fallback_slots: int = 2,
    min_slot_area_px: int = 200,
    slot_floor_band_m: float = 0.015,
    lift_offset_m: float = 0.04,
    normal_up_threshold: float = 0.9,
) -> List[dict]:
    """
    Detect compartments inside a tray using depth surface normals.

    Args:
        rgb: (H, W, 3) uint8. Only used for shape checks / future RGB
            cues; not consumed by the current detector.
        depth_m: (H, W) float metric depth in METERS, aligned to the
            color frame. Multiplied internally by ``depth_scale``.
        tray_mask: (H, W) bool or uint8. SAM3 mask of the tray.
        K: (3, 3) color-camera intrinsics (OpenCV convention).
        T_cam_to_base: (4, 4) homogeneous transform from camera to
            robot base frame.
        depth_scale: per-camera depth bias correction (see SPARK
            ``depth_scale_correction`` in the handeye JSON).
        n_fallback_slots: number of evenly-spaced slots to emit across
            the tray's short axis when the normal-based detector
            can't find >=2 components. Caller should pass the actual
            object count when known so the fallback matches.
        min_slot_area_px: minimum connected-component area for a
            slot. Smaller components are dropped as noise.
        slot_floor_band_m: world-Z thickness around the 25th-percentile
            up-pixel Z to count as "slot floor".
        lift_offset_m: world-Z offset added to each slot's floor for
            the released-above-floor pose.
        normal_up_threshold: minimum N_base.z to count as "up-facing"
            (default 0.9 ~ within 26 deg of world +Z).

    Returns:
        List of dicts ordered side by side, across the slots. Each dict:
            slot_idx: int
            world_xyz: np.ndarray  shape (3,) float64
            n_pixels: int          number of slot-floor pixels in component
            confidence: float      0..1
            mode: str              "normals" or "fallback"
            axis_rad, axis_ar: float  the floor's long axis ("normals" only)

    On total perception failure (no tray axes computable) returns an
    empty list; the caller should treat this the same as "no slots".
    """
    # step 0: input validation + scale
    if depth_m is None or rgb is None or tray_mask is None:
        return []
    if depth_m.ndim != 2:
        return []
    if rgb.shape[:2] != depth_m.shape:
        logger.warning(
            "rgb shape %s and depth shape %s disagree", rgb.shape[:2], depth_m.shape
        )
    if tray_mask.dtype != bool:
        tray_mask = tray_mask > 0
    if K is None or T_cam_to_base is None:
        return []
    K = np.asarray(K, dtype=np.float64)
    T_cam_to_base = np.asarray(T_cam_to_base, dtype=np.float64)

    depth_m = depth_m.astype(np.float32)
    if depth_scale and float(depth_scale) != 1.0:
        depth_m = depth_m * float(depth_scale)

    # step 1: tray axes for ordering / fallback
    tray_axes = _tray_axes_world(tray_mask, depth_m, K, T_cam_to_base)
    if tray_axes is None:
        logger.warning("slot_detector: tray has too few valid-depth pixels")
        return []

    # step 2: per-pixel normals in camera frame, transform to world
    normals_cam = _compute_normals_from_depth(depth_m, K)
    R = T_cam_to_base[:3, :3]
    H, W = depth_m.shape
    # Flatten matrix multiply: (3,3) @ (3, N) -> (3, N).
    n_flat = normals_cam.reshape(-1, 3).T  # (3, H*W)
    n_world_flat = (R @ n_flat).T  # (H*W, 3)
    n_world = n_world_flat.reshape(H, W, 3)

    # Use |Nz| so the test is sign-agnostic: a horizontal surface gives a
    # large |Nz| regardless of which side the camera-ward normal points.
    up_mask = (np.abs(n_world[:, :, 2]) > float(normal_up_threshold)) & tray_mask
    up_mask &= depth_m > 0.01

    n_up = int(up_mask.sum())
    if n_up < min_slot_area_px:
        logger.info(
            "slot_detector: only %d up-pixels in tray (min %d), falling back",
            n_up,
            min_slot_area_px,
        )
        return _fallback_axis_slots(tray_axes, n_fallback_slots, lift_offset_m)

    # step 3: filter up-pixels down to slot floors (lowest world Z)
    ys, xs = np.where(up_mask)
    pts_world = _backproject_pixels_to_world(ys, xs, depth_m, K, T_cam_to_base)
    finite = np.isfinite(pts_world[:, 2])
    if int(finite.sum()) < min_slot_area_px:
        return _fallback_axis_slots(tray_axes, n_fallback_slots, lift_offset_m)
    pts_world = pts_world[finite]
    ys = ys[finite]
    xs = xs[finite]
    z_world = pts_world[:, 2]

    # Slot floor altitude: the deepest cluster of horizontal pixels (the
    # slot interiors). Dividers often have more birdview area than the
    # interiors, so a single percentile is fragile; instead histogram the
    # up-pixel Z values and pick the lowest bin with significant count.
    z_lo = float(np.percentile(z_world, 2))  # robust min (drops noise)
    z_hi = float(np.percentile(z_world, 98))  # robust max
    n_bins = 24
    bin_edges = np.linspace(z_lo, z_hi, n_bins + 1)
    counts, _ = np.histogram(z_world, bins=bin_edges)
    # Significance threshold: a bin counts as a "real surface" if it
    # has at least min_slot_area_px / n_bins pixels (uniform-distribution
    # baseline scaled by half).
    sig_thresh = max(int(min_slot_area_px) // (n_bins // 2), 50)
    sig_bins = np.where(counts >= sig_thresh)[0]
    if len(sig_bins) == 0:
        logger.info(
            "slot_detector: no significant horizontal band in Z; "
            "falling back to axis stripes"
        )
        return _fallback_axis_slots(tray_axes, n_fallback_slots, lift_offset_m)
    # LOWEST significant bin = slot-floor altitude (deepest horizontal
    # surface inside the tray mask).
    lo_bin = int(sig_bins[0])
    z_floor = float(bin_edges[lo_bin])
    logger.info(
        "slot_detector: Z range [%.3f, %.3f], slot_floor band starts at "
        "z=%.3f (bin %d/%d), n_sig_bins=%d",
        z_lo,
        z_hi,
        z_floor,
        lo_bin,
        n_bins,
        len(sig_bins),
    )
    floor_keep = z_world < (z_floor + slot_floor_band_m)
    if int(floor_keep.sum()) < min_slot_area_px:
        return _fallback_axis_slots(tray_axes, n_fallback_slots, lift_offset_m)

    floor_ys = ys[floor_keep]
    floor_xs = xs[floor_keep]
    floor_pts = pts_world[floor_keep]

    # Rebuild a 2D mask of just the slot-floor pixels for connected-
    # components labelling.
    slot_floor_mask = np.zeros((H, W), dtype=np.uint8)
    slot_floor_mask[floor_ys, floor_xs] = 1

    # Light morphological cleanup: close 1-pixel gaps that confuse
    # connectedComponents (slot floors aren't perfectly contiguous
    # at the rim), then open to drop salt-and-pepper noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    slot_floor_mask = cv2.morphologyEx(slot_floor_mask, cv2.MORPH_CLOSE, kernel)
    slot_floor_mask = cv2.morphologyEx(slot_floor_mask, cv2.MORPH_OPEN, kernel)

    # step 4: connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        slot_floor_mask,
        connectivity=8,
    )

    # Labels: 0 is background; 1..n_labels-1 are components.
    slot_components = []
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_slot_area_px:
            continue
        comp_mask = labels == lbl
        # Restrict to the floor-pixel set (comp_mask is in image coords,
        # but only floor pixels have a depth backprojection; the world
        # points are re-picked from floor_pts at the matching rows).
        comp_pix_in_floor = comp_mask[floor_ys, floor_xs]
        if int(comp_pix_in_floor.sum()) < min_slot_area_px:
            continue
        comp_world = floor_pts[comp_pix_in_floor]
        # Per-axis median for robustness.
        cx_w = float(np.median(comp_world[:, 0]))
        cy_w = float(np.median(comp_world[:, 1]))
        cz_w = float(np.median(comp_world[:, 2]))
        axis_rad, axis_ar = _floor_axis(comp_world[:, :2])
        slot_components.append(
            {
                "world_xy": np.array([cx_w, cy_w]),
                "world_z_floor": cz_w,
                "n_pixels": int(comp_pix_in_floor.sum()),
                "axis_rad": axis_rad,
                "axis_ar": axis_ar,
            }
        )

    if len(slot_components) < 2:
        logger.info(
            "slot_detector: found %d slot component(s), falling back to "
            "%d-stripe along major axis",
            len(slot_components),
            n_fallback_slots,
        )
        return _fallback_axis_slots(tray_axes, n_fallback_slots, lift_offset_m)

    # step 5: order slots side by side, across their own direction. A tray
    # whose slots run along its long side has no spread along that axis.
    mean_xy = tray_axes["mean_xy"]
    axis = slot_axis_consensus(slot_components)
    across = (
        tray_axes["major"]
        if axis is None
        else np.array([-np.sin(axis), np.cos(axis)])
    )
    for sc in slot_components:
        sc["proj"] = float((sc["world_xy"] - mean_xy) @ across)
    slot_components.sort(key=lambda s: s["proj"])

    # step 6: build the output list
    # Confidence proxy: pixel count vs the largest component. Trays
    # with very-unequal components likely have a partial occlusion or
    # one slot already full of an object; flag the small ones.
    max_n = max(sc["n_pixels"] for sc in slot_components)
    out: List[dict] = []
    for i, sc in enumerate(slot_components):
        xy = sc["world_xy"]
        z = sc["world_z_floor"] + float(lift_offset_m)
        conf = 0.5 + 0.5 * (sc["n_pixels"] / float(max_n))
        conf = float(min(max(conf, 0.0), 1.0))
        out.append(
            {
                "slot_idx": i,
                "world_xyz": np.array([xy[0], xy[1], z], dtype=np.float64),
                "n_pixels": int(sc["n_pixels"]),
                "confidence": conf,
                "mode": "normals",
                "axis_rad": sc["axis_rad"],
                "axis_ar": sc["axis_ar"],
            }
        )
    return out



# Below this a tray's world OBB axis is too close to square to name its long side.
TRAY_OBB_AR_TRUST = 1.5

_HEAVY_END_IMBALANCE_GATE = 0.08
_HEAVY_END_MIN_PIXELS = 50


def _largest_component(mask8: np.ndarray) -> np.ndarray:
    """The largest 8-connected component of a binary mask.

    SAM3 masks fragment, and satellite blobs drag the extent midpoint of a
    DIRECTED measurement toward a confidently wrong sign. The body of the
    object is the largest component; measure that.
    """
    n, lab = cv2.connectedComponents(mask8, connectivity=8)
    if n <= 2:  # background + at most one component
        return mask8
    counts = np.bincount(lab.ravel())
    counts[0] = 0  # background
    return (lab == int(np.argmax(counts))).astype(np.uint8)


def heavy_end_sign(mask: np.ndarray, axis_rad: float) -> float:
    """Which END of `axis_rad` the mask's fat half lies toward: +1.0 / -1.0.

    A PCA axis is a LINE -- identical under a 180 deg flip -- so it cannot say
    which end of a screwdriver is the tip. This breaks that tie with the only
    directed signal a silhouette has: one end being fatter than the other.
    Pixels are projected onto the axis direction and split at the midpoint of
    the (percentile-trimmed) extent; whichever half holds more mask pixels is
    the heavy end. +1.0 means it lies toward +(cos axis, sin axis) in image
    coords (x right, y down), -1.0 the opposite end.

    Returns 0.0 -- "too symmetric to call" -- below _HEAVY_END_IMBALANCE_GATE,
    and every consumer must treat 0.0 as an abstention, never a direction. A
    bare handle mask is symmetric end-for-end, so the direction must be
    measured on the WHOLE tool (see _record_held_axis's parent-label upgrade).
    """
    m8 = _largest_component((np.asarray(mask) > 0).astype(np.uint8))
    ys, xs = np.where(m8 > 0)
    if len(xs) < _HEAVY_END_MIN_PIXELS:
        return 0.0
    # The caller's axis usually comes from _pca_obb over the FULL mask. When
    # fragments dragged that axis away from the body's own, a sign measured
    # along it describes the fragments, not the object. If the two axes
    # disagree by more than 45 deg the premise is broken: abstain.
    ang_cc, _, _, _ = _pca_obb(m8)
    d_ax = abs(ang_cc - float(axis_rad)) % np.pi
    if min(d_ax, np.pi - d_ax) > np.pi / 4:
        return 0.0
    proj = xs * float(np.cos(axis_rad)) + ys * float(np.sin(axis_rad))
    # Trim the extent, not the mass: a few speck pixels must not drag the
    # midpoint, but every real pixel still votes on which half is heavier.
    lo, hi = np.percentile(proj, [2.0, 98.0])
    if hi - lo < 1e-6:
        return 0.0
    mid = 0.5 * (lo + hi)
    n_pos = int(np.count_nonzero(proj > mid))
    n_neg = int(np.count_nonzero(proj < mid))
    total = n_pos + n_neg
    if total == 0:
        return 0.0
    imbalance = (n_pos - n_neg) / float(total)
    if abs(imbalance) < _HEAVY_END_IMBALANCE_GATE:
        return 0.0
    return 1.0 if imbalance > 0 else -1.0


# best_fit_rotation resamples both masks onto this square canvas. 192 px keeps
# a full 360-sweep of warps under ~10 ms while leaving a ~2 px quantization,
# far under the 90-deg-class margins the consumers gate on.
_FIT_CANVAS_PX = 192
_FIT_STEP_DEG = 3.0
_FIT_MIN_PIXELS = 50


def best_fit_rotation(held_mask: np.ndarray, target_mask: np.ndarray):
    """Image rotation that best registers `held_mask` onto `target_mask`.

    Returns (angle_rad, best_iou, margin):
      angle_rad  signed image-frame rotation in (-pi, pi]. Rotating the held
                 silhouette by +angle (the same convention as _pca_obb angles:
                 atan2 in image coords, x right, y down) best overlays it on
                 the target, i.e. angle ~= target_axis - held_axis when the
                 shapes are elongated.
      best_iou   IoU at that rotation, both masks centroid-centered and drawn
                 at a SHARED scale -- relative size is real signal (a cutout
                 slightly larger than its tool still overlaps well; a bowl
                 twice its size does not).
      margin     best_iou minus the best IoU among rotations >= 90 deg away.
                 This is the DIRECTED part of the answer: for a directional
                 shape in a directional recess the 180-off candidate scores
                 visibly worse and the margin is large; for a round or
                 twofold-symmetric pair every rotation ties and the margin
                 collapses toward 0 -- which is the honest "shape does not
                 decide" answer the JIGSAW_MIN_MARGIN gate turns into an
                 abstention (executor_motion, gate 0.05).

    (0.0, 0.0, 0.0) when either mask is degenerate (< _FIT_MIN_PIXELS).
    """
    # Largest component only, for the same reason as heavy_end_sign: a
    # satellite blob shifts the centroid and radius, and the registration then
    # scores rotations of the wrong center.
    h8 = _largest_component((np.asarray(held_mask) > 0).astype(np.uint8))
    t8 = _largest_component((np.asarray(target_mask) > 0).astype(np.uint8))
    if int(h8.sum()) < _FIT_MIN_PIXELS or int(t8.sum()) < _FIT_MIN_PIXELS:
        return 0.0, 0.0, 0.0

    c = _FIT_CANVAS_PX
    ctr = (c - 1) / 2.0

    def _centroid_and_radius(m):
        ys, xs = np.where(m > 0)
        cx, cy = float(xs.mean()), float(ys.mean())
        r = float(np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).max()))
        return cx, cy, max(r, 1.0)

    hx, hy, hr = _centroid_and_radius(h8)
    tx, ty, tr = _centroid_and_radius(t8)
    # ONE scale for both, sized so the larger just fits under every rotation.
    s = (c / 2.0 - 2.0) / max(hr, tr)

    def _to_canvas(m, cx, cy):
        M = np.array([[s, 0.0, ctr - s * cx], [0.0, s, ctr - s * cy]])
        return cv2.warpAffine(m, M, (c, c), flags=cv2.INTER_NEAREST)

    held_c = _to_canvas(h8, hx, hy)
    tgt_c = _to_canvas(t8, tx, ty).astype(bool)
    tgt_area = int(tgt_c.sum())

    angles = np.deg2rad(np.arange(0.0, 360.0, _FIT_STEP_DEG))
    ious = np.zeros(len(angles))
    for i, a in enumerate(angles):
        ca, sa = float(np.cos(a)), float(np.sin(a))
        # Forward map [[c,-s],[s,c]] about the canvas center: a pixel at image
        # angle b moves to b+a, matching _pca_obb's atan2 convention, so the
        # returned rotation composes directly with axis angles.
        M = np.array(
            [[ca, -sa, ctr - ca * ctr + sa * ctr],
             [sa, ca, ctr - sa * ctr - ca * ctr]]
        )
        rot = cv2.warpAffine(held_c, M, (c, c), flags=cv2.INTER_NEAREST).astype(bool)
        inter = int(np.count_nonzero(rot & tgt_c))
        union = int(rot.sum()) + tgt_area - inter
        ious[i] = inter / union if union > 0 else 0.0

    best = int(np.argmax(ious))
    best_iou = float(ious[best])
    # Circular distance to the winner; rivals live >= 90 deg away.
    d = np.abs(np.rad2deg(angles) - np.rad2deg(angles[best]))
    d = np.minimum(d, 360.0 - d)
    far = d >= 90.0
    rival = float(ious[far].max()) if far.any() else 0.0
    ang = float((angles[best] + np.pi) % (2 * np.pi) - np.pi)
    return ang, best_iou, best_iou - rival


def resolve_slot_direction(det, container_label="", slot_idx=None):
    """World angle (rad) a placed utensil must lie ALONG, and its source.

    Shared by place_in_slot and the generic place path. `det` may be a dict
    (detection_map) or an ObjectDetection.

      1. This slot's own floor, when it is elongated (mixed-compartment trays).
      2. The consensus of all measured slot floors.
      3. No floor measured (dividers below the depth resolution): the
         container's long axis. A tray whose slots run across its short side
         needs the planner's grasp_yaw_deg here.
    """
    # ponytail: step 3 assumes slots along the long side; read dividers from RGB if a rig needs otherwise.
    measured = [s for s in (det_field(det, "slots", None) or []) if s.get("mode") == "normals"]
    for s in measured:
        if s.get("slot_idx") == slot_idx and s.get("axis_ar", 0.0) >= SLOT_AXIS_MIN_AR:
            return float(s["axis_rad"]), f"slot_{slot_idx}_floor(AR={s['axis_ar']:.2f})"
    axis = slot_axis_consensus(measured)
    if axis is not None:
        return axis, f"slot_floors(n={len(measured)})"

    world_major = det_field(det, "world_major_axis_rad", None)
    if world_major is None:
        world_major = det_field(det, "orientation_angle", None)
    tray_ar = float(det_field(det, "aspect_ratio", 1.0) or 1.0)
    if world_major is not None and 1e-6 < tray_ar < 1.0:  # the stored "major" is the short axis
        world_major, tray_ar = float(world_major) + np.pi / 2.0, 1.0 / tray_ar
    if world_major is not None and tray_ar >= TRAY_OBB_AR_TRUST:
        return float(world_major), f"container_long_axis(AR={tray_ar:.2f})"
    plane_axis = det_field(det, "plane_orientation_angle", None)
    if plane_axis is not None:
        return float(plane_axis), "container_long_axis(ray_plane)"
    logger.warning(
        "[slot_direction] no reliable direction for '%s' (AR %.2f); placement yaw may be wrong",
        container_label,
        tray_ar,
    )
    return float(world_major or 0.0), "container_obb(low_confidence)"



SCHEMA = 1


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in str(text).strip()]
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "task"


class AnnotationAnchors:
    """JSON-backed {task: {label: [anchor, ...]}} with atomic writes."""

    def __init__(self, path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, List[dict]]] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text())
            if isinstance(raw, dict) and raw.get("schema") == SCHEMA:
                self._data = raw.get("tasks", {}) or {}
            elif isinstance(raw, dict):
                logger.info(
                    "annotation anchors: schema %r != %d; starting empty",
                    raw.get("schema"), SCHEMA,
                )
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001 - a bad file is an empty store
            logger.warning("annotation anchors: unreadable (%s); starting empty", exc)

    def _flush(self) -> None:
        payload = json.dumps(
            {"schema": SCHEMA, "tasks": self._data}, indent=2, sort_keys=True
        )
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=str(self._path.parent), prefix=".annotation_anchors."
            )
            with os.fdopen(fd, "w") as fh:
                fh.write(payload)
            os.replace(tmp, self._path)
        except Exception as exc:  # noqa: BLE001 - never fail a run over this
            logger.warning("annotation anchors: could not persist (%s)", exc)

    # write
    def save(self, task, label, camera, box_norm, note="") -> dict:
        """Record one normalised box for (task, label, camera). Replaces any
        previous anchor for that exact triple -- the newest annotation is the
        operator's current intent, and keeping both would make the fold-in
        order decide which wins."""
        if not task or not label or box_norm is None or len(box_norm) != 4:
            raise ValueError("task, label and a 4-element box are required")
        x1, y1, x2, y2 = (float(v) for v in box_norm)
        if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
            raise ValueError(f"box must be normalised 0..1, got {box_norm}")
        anchor = {
            "label": str(label),
            "camera": str(camera or ""),
            "box_norm": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
            "note": str(note or ""),
        }
        key = _slug(task)
        with self._lock:
            per_task = self._data.setdefault(key, {})
            kept = [
                a for a in per_task.get(str(label), [])
                if a.get("camera") != anchor["camera"]
            ]
            kept.append(anchor)
            per_task[str(label)] = kept
            self._flush()
        logger.info(
            "annotation anchors: saved %r/%r on %s -> %s",
            task, label, anchor["camera"], anchor["box_norm"],
        )
        return anchor

    # read
    def for_task(self, task) -> Dict[str, List[dict]]:
        return dict(self._data.get(_slug(task), {})) if task else {}

    def labels(self, task) -> List[str]:
        return sorted(self.for_task(task).keys())

    def drop(self, task, label=None) -> int:
        key = _slug(task)
        with self._lock:
            if key not in self._data:
                return 0
            if label is None:
                n = sum(len(v) for v in self._data[key].values())
                del self._data[key]
            else:
                n = len(self._data[key].pop(str(label), []))
                if not self._data[key]:
                    del self._data[key]
            self._flush()
        return n

    @staticmethod
    def pixels(anchor, width, height):
        """Normalised anchor -> (x1, y1, x2, y2) pixels for this image size."""
        x1, y1, x2, y2 = anchor["box_norm"]
        return (
            int(round(x1 * float(width))), int(round(y1 * float(height))),
            int(round(x2 * float(width))), int(round(y2 * float(height))),
        )


# A direction whose |dz| is under this is treated as parallel to the plane:
# the intersection is either nonexistent or so far out that the pixel carries
# no usable lateral information.
PARALLEL_EPS = 1e-6


def unpack_intrinsics(intr):
    """Pull (fx, fy, cx, cy) out of a calibration object, mapping, or sequence.

    Accepts CameraCalibration (attributes), a dict, or a 4-sequence in
    fx, fy, cx, cy order.
    """
    if hasattr(intr, "fx"):
        return float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy)
    if isinstance(intr, dict):
        return (
            float(intr["fx"]),
            float(intr["fy"]),
            float(intr["cx"]),
            float(intr["cy"]),
        )
    fx, fy, cx, cy = intr
    return float(fx), float(fy), float(cx), float(cy)


def pixel_ray_base(u: float, v: float, intr, T_cam_to_base: np.ndarray):
    """Ray through pixel (u, v), expressed in the robot base frame.

    Returns (origin, direction) with direction unit-length. origin is the
    camera centre, i.e. the translation column of T_cam_to_base.
    """
    fx, fy, cx, cy = unpack_intrinsics(intr)
    T = np.asarray(T_cam_to_base, dtype=float)
    # Pinhole: the ray through (u, v) hits z_cam=1 at these camera-frame coords.
    d_cam = np.array([(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0])
    direction = T[:3, :3] @ d_cam
    norm = float(np.linalg.norm(direction))
    if norm < PARALLEL_EPS:
        raise ValueError("degenerate camera->base rotation")
    return T[:3, 3].astype(float).copy(), direction / norm


def intersect_ray_plane(origin, direction, z_plane: float):
    """Intersect a ray with the horizontal plane z = z_plane.

    Returns the xyz hit point, or None when the ray is parallel to the plane
    or points AWAY from it (t <= 0 would put the "hit" behind the camera).
    """
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    if abs(d[2]) < PARALLEL_EPS:
        return None
    t = (float(z_plane) - o[2]) / d[2]
    if t <= 0.0:
        return None
    return o + t * d


def pixel_to_plane_point(
    u: float, v: float, intr, T_cam_to_base: np.ndarray, z_plane: float
):
    """pixel_ray_base + intersect_ray_plane. None on either degenerate case."""
    origin, direction = pixel_ray_base(u, v, intr, T_cam_to_base)
    return intersect_ray_plane(origin, direction, z_plane)


def project_point_to_pixel(point, intr, T_cam_to_base: np.ndarray):
    """Inverse of pixel_to_plane_point: base-frame xyz -> (u, v).

    Returns None when the point is at or behind the camera plane (z_cam <= 0).
    Used to round-trip the geometry in tests and to sanity-check a refinement.
    """
    fx, fy, cx, cy = unpack_intrinsics(intr)
    T = np.asarray(T_cam_to_base, dtype=float)
    p_cam = T[:3, :3].T @ (np.asarray(point, dtype=float) - T[:3, 3])
    if p_cam[2] <= PARALLEL_EPS:
        return None
    return (
        fx * p_cam[0] / p_cam[2] + cx,
        fy * p_cam[1] / p_cam[2] + cy,
    )
