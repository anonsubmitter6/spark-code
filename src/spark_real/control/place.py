"""Release dip side: rotate the approach yaw so a tilted release lands on the allowed side."""


from __future__ import annotations


from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from typing import Any, Optional, Tuple
import cv2
import json
import logging
import math
import numpy as np
import os
import re
import time


from spark_core.mask_obb import _cloud_median_world, _world_xy_pca_obb
from spark_core.util import as_bool, det_field
from spark_core.util import det_field
from spark_real import recording as run_timing
from spark_real.control import success_verifier
from spark_real.control.common import ExecutionResult
from spark_real.control.common import container_region
from spark_real.control.grasp_strategy import compose_yaw, resolve_grasp_orientation
from spark_real.control.predicates import container_obb_xy, obb_contains
from spark_real.perception.masks import resolve_slot_direction
from spark_real.routes import state as _routes_state


logger = logging.getLogger(__name__)


class ReleaseDipMixin:
    """Release dip side: rotate the approach yaw so a tilted release lands on the allowed side."""

    def _dip_side_orient(self, orient, yaw_aligned: bool = False):
        """Fold the deep-dip side choice into the place-approach orientation.

        Called by MotionMixin._transport_to with the approach orientation it
        is about to command. When the upcoming release (executor_core's
        _pending_release_tilt look-ahead) asks for a dip past the positive
        budget (MAX_POS_PITCH_RAD, the wrist-camera clearance) and `orient`
        would put the dip on that side (tool_y_world[0] > 0 at the release's
        own pitch_sign test), rotate the approach yaw so the dip lands on the
        negative side and gets the 90 deg budget instead. The yaw rides the
        transport's horizontal transit, so it costs no extra wrist travel at
        the release.

        Two cases:
          * `yaw_aligned` False (bin/bowl: no slot, no shape fit, no OBB
            adopted): the yaw is free, so pick the smallest yaw that puts
            tool Y DIP_SIDE_MARGIN_RAD past the boundary, cable limit
            enforced by compose_yaw.
          * `yaw_aligned` True: the yaw carries a slot/fit/OBB alignment and
            only its 180-twin preserves it (a two-jaw grasp is symmetric mod
            180). Try _flip_yaw_180; when the twin exceeds the wrist-3 cable
            limit, keep the alignment and let the release clamp to 40 deg. A
            shallow dip beats a crooked seat.

        Returns `orient` (possibly replaced). Never raises: any read failure
        keeps the alignment untouched and the release clamps.
        """
        pend = getattr(self, "_pending_release_tilt", None)
        if not pend:
            return orient
        try:
            tilt = abs(float(pend.get("tilt_angle", 0.0) or 0.0))
        except (TypeError, ValueError, AttributeError):
            return orient
        if tilt <= self.MAX_POS_PITCH_RAD + 1e-9:
            return orient
        try:
            tool_y = Rotation.from_rotvec(
                np.asarray(orient, dtype=float)
            ).apply([0.0, 1.0, 0.0])
        except Exception as exc:  # noqa: BLE001 - never stop a transport
            logger.warning("[dip-side] could not read orient (%s); unchanged", exc)
            return orient
        if tool_y[0] <= 0.0:
            logger.info(
                "[dip-side] release wants %.0f deg; approach already on the "
                "negative-pitch side (tool_y_x=%.2f), budget %.0f deg",
                np.rad2deg(tilt), tool_y[0], np.rad2deg(self.MAX_NEG_PITCH_RAD),
            )
            return orient

        if yaw_aligned:
            flipped = self._flip_yaw_180(orient)
            if flipped is not None:
                logger.info(
                    "[dip-side] release wants %.0f deg > %.0f positive budget; "
                    "flipping the ALIGNED place yaw 180 (grasp-symmetric) so "
                    "the dip gets the %.0f deg negative budget",
                    np.rad2deg(tilt), np.rad2deg(self.MAX_POS_PITCH_RAD),
                    np.rad2deg(self.MAX_NEG_PITCH_RAD),
                )
                return flipped
            logger.warning(
                "[dip-side] release wants %.0f deg but the aligned yaw's 180 "
                "twin exceeds the wrist-3 cable limit; keeping the alignment, "
                "the release will clamp to %.0f deg",
                np.rad2deg(tilt), np.rad2deg(self.MAX_POS_PITCH_RAD),
            )
            return orient

        # Free yaw: aim tool Y's world heading past the +-90 deg boundary by
        # the margin, with the least wrist travel from base. The heading of
        # tool Y after a world-Z yaw psi is (delta + psi), where delta is its
        # heading at the base orientation, and pitch_sign goes negative when
        # cos(delta + psi) < 0.
        from spark_real.control.grasp_strategy import base_orientation, compose_yaw

        try:
            ty_base = Rotation.from_rotvec(base_orientation(self)).apply(
                [0.0, 1.0, 0.0]
            )
            delta = float(np.arctan2(ty_base[1], ty_base[0]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[dip-side] base orientation unreadable (%s)", exc)
            return orient
        edge = np.pi / 2.0 + self.DIP_SIDE_MARGIN_RAD
        candidates = sorted(
            (
                float((s * edge - delta + np.pi) % (2 * np.pi) - np.pi)
                for s in (1.0, -1.0)
            ),
            key=abs,
        )
        for psi in candidates:
            picked = compose_yaw(self, psi, context="place-dip")
            if picked is None:
                continue  # over the cable limit; try the other side
            logger.info(
                "[dip-side] release wants %.0f deg > %.0f positive budget; "
                "free yaw -> %.1f deg so the dip gets the %.0f deg negative "
                "budget (folded into the approach, no extra motion)",
                np.rad2deg(tilt), np.rad2deg(self.MAX_POS_PITCH_RAD),
                np.rad2deg(psi), np.rad2deg(self.MAX_NEG_PITCH_RAD),
            )
            return picked
        logger.warning(
            "[dip-side] release wants %.0f deg but no dip-side yaw fits the "
            "wrist-3 cable limit; the release will clamp to %.0f deg",
            np.rad2deg(tilt), np.rad2deg(self.MAX_POS_PITCH_RAD),
        )
        return orient


# Where the held object's bottom is aimed, above the container's interior
# floor. This IS the remaining fall.
FLOOR_TARGET_GAP_M = 0.02
# Hard lower bound on the same gap. Everything above is aiming; this is the
# line the clamp enforces.
MIN_FLOOR_CLEARANCE_M = 0.01
# Deep containers: gap between the held object's bottom and the rim plane at release.
# The most the upper bound allows: the measured hang of a soft object reads up to 4 cm short,
# and a hang read short puts the object below the rim, where it presses the rim and tips a light bowl.
RIM_RELEASE_GAP_M = 0.05
# The TCP never comes closer than this to the rim plane, and never crosses it.
MIN_TCP_ABOVE_RIM_M = 0.02
# ... and never sits further above the rim than this. A larger gap is a drop,
# not a place. Caps a rim reading that came back too high.
MAX_TCP_ABOVE_RIM_M = 0.05

# How far the held object hangs below the TCP when it cannot be measured.
# Errs LONG on purpose: a longer assumed overhang raises the release.
# Below this, a measured top-to-bottom extent is not a measurement: a depth
# camera cannot see an object's underside, so anything thinner than this is
# noise or an inversion. Triggers the OBB-minor fallback in held_drop.
MIN_MEASURABLE_OVERHANG_M = 0.005

HELD_DROP_FALLBACK_M = 0.06
# Nothing this gripper carries hangs further below the TCP than this; a larger
# reading is a mask that swallowed the table, not a very long object.
MAX_HELD_DROP_M = 0.15

# Validity band for a measured container. A rim more than this above the table
# is not a container placed into from above, and an interior below the table is
# depth punching through the container to the surface it stands on.
MAX_CONTAINER_RIM_ABOVE_TABLE_M = 0.40
# Minimum depth pixels inside the mask before its height profile counts as
# measured rather than as three stray points.
MIN_PROFILE_SAMPLES = 50


@dataclass
class ReleaseHeight:
    """The resolved place-target Z, plus everything that decided it."""

    z: float
    source: str  # "geometry" | "plan-strict" | "plan-no-geometry"
    detail: str
    held_drop_m: float
    rim_z: Optional[float] = None
    interior_z: Optional[float] = None
    geometric_z: Optional[float] = None
    lo: Optional[float] = None
    hi: Optional[float] = None
    clamped: bool = False

    def describe(self) -> str:
        if self.geometric_z is None:
            return f"z={self.z:.3f} ({self.source}: {self.detail})"
        return (
            f"z={self.z:.3f} ({self.source}: rim={self.rim_z:.3f} "
            f"interior={self.interior_z:.3f} drop={self.held_drop_m*100:.1f}cm "
            f"band=[{self.lo:.3f},{self.hi:.3f}]"
            f"{' CLAMPED' if self.clamped else ''}; {self.detail})"
        )


def _finite(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def height_profile(det: Any) -> Tuple[Optional[Tuple[float, float]], str]:
    """``((rim_z, interior_z), reason)`` for a detection, or ``(None, reason)``.

    Every rejection is a distinct reason string so a run's log says WHICH way
    the measurement was unusable, not just that it was.
    """
    if det is None:
        return None, "no detection"
    rim = _finite(det_field(det, "rim_z_m"))
    interior = _finite(det_field(det, "interior_z_m"))
    if rim is None or interior is None:
        return None, "no measured height profile"
    n = _finite(det_field(det, "height_samples", 0.0)) or 0.0
    if n < MIN_PROFILE_SAMPLES:
        return None, f"only {n:.0f} depth samples in the mask"
    if interior > rim + 1e-6:
        return None, f"inverted profile (rim {rim:.3f} below interior {interior:.3f})"
    return (rim, interior), "measured"


def _container_profile(
    det: Any, table_z: Optional[float]
) -> Tuple[Optional[Tuple[float, float]], str]:
    """``height_profile`` plus the checks that only apply to a container."""
    prof, reason = height_profile(det)
    if prof is None:
        return None, reason
    rim, interior = prof
    if table_z is not None and math.isfinite(table_z):
        if rim > table_z + MAX_CONTAINER_RIM_ABOVE_TABLE_M:
            return None, (
                f"rim {rim:.3f} is more than "
                f"{MAX_CONTAINER_RIM_ABOVE_TABLE_M*100:.0f}cm above the table"
            )
        # Depth that punched through a translucent container reads the surface
        # it stands ON. Nothing inside it can be below that surface.
        if interior < table_z:
            interior = float(table_z)
            reason = "measured (interior raised to the table plane)"
    return (rim, interior), reason


def held_drop(held: Any, grasp_z_offset: float = 0.0) -> Tuple[float, str]:
    """How far the held object hangs below the TCP, in metres.

    ``grasp_z_offset`` is the executor's ``GRIPPER_OPEN_Z_OFFSET``: the grasp
    descended to the object's perceived Z plus that offset, so the TCP-to-
    bottom distance is ``(perceived_z + offset) - measured_bottom``.
    """
    prof, reason = height_profile(held)
    top = _finite(det_field(held, "position_3d", [None, None, None])[2]) if held else None
    if prof is None or top is None:
        return HELD_DROP_FALLBACK_M, f"held extent unknown ({reason}); fallback"
    _rim, bottom = prof
    raw = (top + float(grasp_z_offset)) - bottom
    # A depth camera sees an object's TOP surface, never its underside, so for
    # a tool lying flat this difference collapses to noise and can invert
    # (raw <= 0 would model the tool as hanging nothing below the gripper).
    # The OBB minor axis (measured SHORT dimension, e.g. 13.4 mm for a
    # screwdriver) is a better estimate of the protrusion for anything a
    # parallel gripper can close on. Use it when the extent is degenerate.
    if raw < MIN_MEASURABLE_OVERHANG_M:
        minor = _finite(det_field(held, "obb_minor_m", None)) if held else None
        if minor is not None and minor > MIN_MEASURABLE_OVERHANG_M:
            drop = min(float(minor), MAX_HELD_DROP_M)
            return drop, (
                f"held overhang from OBB minor {minor*1000:.0f}mm "
                f"(depth extent {raw*1000:.0f}mm is degenerate)"
            )
    drop = min(max(raw, 0.0), MAX_HELD_DROP_M)
    note = "measured"
    if raw > MAX_HELD_DROP_M:
        note = f"measured {raw*100:.0f}cm, capped"
    # A squashed soft object reads shorter than it hangs. SPARK_HELD_DROP_MIN_M floors the
    # overhang. Off by default.
    floor = min(float(os.environ.get("SPARK_HELD_DROP_MIN_M", "0") or 0), MAX_HELD_DROP_M)
    if drop < floor:
        return floor, f"held overhang {note} {drop*100:.1f}cm, raised to the {floor*100:.0f}cm floor"
    return drop, f"held overhang {note}"


def resolve_release_z(
    plan_z: float,
    container: Any,
    held: Any,
    table_z: Optional[float] = None,
    strict: bool = False,
    grasp_z_offset: float = 0.0,
) -> ReleaseHeight:
    """Resolve the Z a place transport should fly to. See the module docstring."""
    drop, drop_note = held_drop(held, grasp_z_offset=grasp_z_offset)
    plan_z = float(plan_z)

    if strict:
        return ReleaseHeight(
            z=plan_z,
            source="plan-strict",
            detail="strict_offset_z set; the plan's offset_z is authoritative",
            held_drop_m=drop,
        )

    prof, reason = _container_profile(container, table_z)
    if prof is None:
        return ReleaseHeight(
            z=plan_z, source="plan-no-geometry", detail=reason, held_drop_m=drop
        )

    rim, interior = prof
    depth = max(float(rim) - float(interior), 0.0)

    # Aiming FLOOR_TARGET_GAP above the floor is wrong for a recess shallower
    # than that gap: the aim point lands above the rim. For such a recess aim
    # at HALF its depth (a 1.7 cm cutout aims at 0.85 cm); a container at
    # least as deep as the gap (a 3.9 cm bowl) keeps the full 2 cm.
    target_gap = (
        FLOOR_TARGET_GAP_M
        if depth >= FLOOR_TARGET_GAP_M
        else max(depth * 0.5, 0.0)
    )
    nominal = interior + target_gap + drop
    # A deep container (2 cm or more): do not lower the object below the rim. Its measured centre
    # is uncertain (a deep, reflective bowl), so an object lowered inside lands on the rim and is
    # pressed into it. Let go with the object's bottom just above the rim and let it fall in.
    # SPARK_RELEASE_AT_RIM=0 restores the lowering to 2 cm above the floor.
    if depth >= FLOOR_TARGET_GAP_M and os.environ.get("SPARK_RELEASE_AT_RIM", "1") != "0":
        nominal = max(nominal, rim + drop + RIM_RELEASE_GAP_M)

    # Rim guard: the jaws never cross the rim plane. For a recess shallower
    # than the fixed 2 cm margin (e.g. rim -0.253, interior -0.270: the floor
    # target needs TCP -0.2466, 6.4 mm above the rim but 13.6 mm below
    # rim+2cm), shrink the margin to what still lets the object reach its
    # floor target, never below zero. A deeper container keeps the full 2 cm.
    rim_guard = MIN_TCP_ABOVE_RIM_M
    if depth < MIN_TCP_ABOVE_RIM_M:
        needed = (interior + MIN_FLOOR_CLEARANCE_M + drop) - rim
        rim_guard = min(MIN_TCP_ABOVE_RIM_M, max(needed, 0.0))

    lo = max(rim + rim_guard, interior + MIN_FLOOR_CLEARANCE_M + drop)
    hi = rim + MAX_TCP_ABOVE_RIM_M + drop
    # rim >= interior is guaranteed by height_profile, so lo <= hi always; the
    # max() keeps clip() well defined even if a future edit breaks that.
    hi = max(hi, lo)
    geometric = min(max(nominal, lo), hi)

    # Geometry may only lower the plan; the bounds may only raise it.
    chosen = min(plan_z, geometric)
    clamped = chosen < lo - 1e-12
    chosen = min(max(chosen, lo), hi)

    return ReleaseHeight(
        z=float(chosen),
        source="geometry",
        detail=f"{reason}; {drop_note}",
        held_drop_m=drop,
        rim_z=rim,
        interior_z=interior,
        geometric_z=float(geometric),
        lo=float(lo),
        hi=float(hi),
        clamped=bool(clamped),
    )




class ReleaseMixin(ReleaseDipMixin):
    """
    Release action with optional tilt-and-insert.
    """

    # Wrist pitch budget, by side. Positive pitch swings the wrist-mounted
    # RealSense toward the dip and 40 deg is where its clearance runs out:
    # a physical limit on this rig, not a tuning knob; do NOT raise it.
    # Negative pitch swings the camera away and has 90 deg of room.
    #
    # Human demonstrations ("put the pen in the bin", UR10e @15Hz) release
    # at 52-64 deg from vertical, only reachable on the negative side.
    # _dip_side_orient chooses the place-approach yaw at transport time so
    # the dip lands on the negative-pitch side. No at-release 180 wrist
    # flip: it produces a near-360 wrist spin and blows the release budget.
    MAX_POS_PITCH_RAD = float(np.deg2rad(40))
    MAX_NEG_PITCH_RAD = float(np.deg2rad(90))

    # How far past the pitch-sign boundary the dip-side yaw aims tool Y
    # (rad). The release re-derives pitch_sign from the live TCP, so a yaw
    # that lands tool-Y exactly on the boundary would let servo residuals
    # flip the sign back to the 40 deg side. 10 deg puts |tool_y_world[0]|
    # ~0.17 past zero, far above the ~0.001 rad servo residual.
    DIP_SIDE_MARGIN_RAD = float(np.deg2rad(10))

    # Raise commanded by the tilt movel, above the resolved release z (m).
    # The resolved z parks the TCP rim+2..5cm (+held drop) over the
    # container (release_height.resolve_release_z); pitching 50+ deg there
    # sweeps the held tool's tip through the rim annulus. Teleop ramps the
    # tilt at z=0.11-0.21m and flies the last 3-7cm of descent with the tilt
    # already at max. 0.10 puts the ramp at ~rim+12-21cm on the measured pen
    # geometry; the tilted insert then descends back to the resolved z.
    TILT_RAMP_RAISE_M = 0.10

    # Hold still at max tilt before the jaws open (s). Teleop reaches max
    # tilt 0.5-1.5s before the release and the gripper opens over ~0.7s with
    # the TCP motionless. Opening the instant the insert movel returns hands
    # the pen the arm's residual motion instead of a clean drop. 1.0 = middle
    # of the human window.
    TILT_SETTLE_S = 1.0


    # Settle (s) between commanding the open and letting the release witness
    # sample the jaw registers. This one stays a sleep:
    #
    #   * it cannot become a condition wait. The witness reads the jaws by
    #     uploading a publish program, and a new program REPLACES the running
    #     rq_move_and_wait open, so polling the open would cancel the open and
    #     leave the object still gripped.
    #   * open_gripper does not pass a speed_norm, so the jaw speed on this
    #     rig is unmeasured, and 0.5 s (on top of the driver's own 0.5 s) is
    #     already optimistic against the datasheet's slowest 20 mm/s.
    #     Shortening it on a guess risks reporting "never released" on a
    #     release that worked.
    #
    # Env SPARK_RELEASE_OPEN_SETTLE_S lets the operator tune it after
    # measuring the real open time (the witness logs jaw_pos_before/after).
    RELEASE_OPEN_SETTLE_S = 0.5

    def _release_open_settle_s(self) -> float:
        raw = os.environ.get("SPARK_RELEASE_OPEN_SETTLE_S")
        if raw is None:
            return float(self.RELEASE_OPEN_SETTLE_S)
        try:
            return max(0.0, float(raw))
        except ValueError:
            logger.warning("SPARK_RELEASE_OPEN_SETTLE_S=%r ignored", raw)
            return float(self.RELEASE_OPEN_SETTLE_S)

    def _transport_delivered(self) -> tuple:
        """Did the transport this release depends on put the TCP over the container?

        Returns ``(ok, detail)``. ``ok`` is True whenever there is nothing to
        judge: not holding anything, no place transport ran, the record
        belongs to an earlier task (``_last_place_label`` is reset per task and
        must still match), or the container's extent is unknown. The gate only
        ever fires on positive evidence that the TCP is outside the container
        it was sent to; see MotionMixin._place_arrival for the tolerance.

        The verdict is recomputed here, against the TCP as it stands with the
        jaws about to open; the stored record only says which container a
        place transport was aimed at. A recovery may have re-approached and
        fixed it, or a move may have drifted off the container after a clean
        arrival.
        """
        if not self._holding:
            return True, ""
        record = getattr(self, "_place_arrival_record", None)
        if not record:
            return True, ""
        label = record.get("label") or ""
        if not label or label != getattr(self, "_last_place_label", ""):
            return True, ""
        return self._place_arrival(label, self.detection_map.get(label))


    # Proprioceptive release verdict
    #
    # The arm's own evidence about whether the place worked, collected in the
    # seconds around the jaw opening. Camera verification is weakest where
    # these signals are strongest: a filled slot no longer looks like a slot,
    # and a tray of identical sockets defeats label re-matching. Every signal
    # reports pass/fail/None (None = could not measure, never a guess); the
    # aggregate is "fail" on any hard contradiction, "pass" when everything
    # measurable agrees, else "uncertain".

    def _collect_release_proprio(self, commanded_orient=None):
        v = {
            "commanded_pitch_achieved": None,
            "carry_force_n": None,
            "jaws_opened": None,       # part 2 fills these two
            "weight_dropped": None,
        }
        # Commanded vs achieved orientation. This catches a dropped motion
        # command (a tilt movel that was silently a no-op while the log said
        # 'completed'). 0.15 rad (~9 deg) is far above servo residuals
        # (~0.001) and far below the commanded 32 deg tilt.
        if commanded_orient is not None:
            try:
                actual = self.current_orientation()
                if actual is not None:
                    r_cmd = Rotation.from_rotvec(list(commanded_orient))
                    r_act = Rotation.from_rotvec(list(actual))
                    err = float((r_cmd.inv() * r_act).magnitude())
                    v["commanded_pitch_achieved"] = bool(err < 0.15)
                    v["orient_err_rad"] = round(err, 4)
            except Exception as exc:  # noqa: BLE001 - verdict must not block release
                logger.warning("release proprio: orientation read failed: %s", exc)
        # Carried load just before the jaws open, for the weight-drop
        # comparison in part 2. Averaged briefly because the UR force
        # estimate is ~+-2 N noisy; a light pen stays below the floor and the
        # signal reports None rather than guessing.
        try:
            fz = []
            for _ in range(5):
                fz.append(float(self.robot.get_tcp_force()[2]))
                time.sleep(0.02)
            v["carry_force_n"] = round(float(np.mean(fz)), 2)
        except Exception:
            pass
        return v

    _PROPRIO_WEIGHT_FLOOR_N = 1.5  # below this the drop is inside sensor noise

    def _finish_release_proprio(self):
        v = getattr(self, "_release_proprio", None)
        if v is None:
            return
        # Jaw evidence comes from the witness the verifier just finished; keyed
        # on the same record verify_scope.still_holding reads.
        w = getattr(self, "_release_witness", None)
        released = None
        if w is not None:
            released = getattr(w, "released", None)
            if released is None and isinstance(w, dict):
                released = w.get("released")
        v["jaws_opened"] = released
        # Weight after the open: carried minus current should be about the
        # object's weight. Only meaningful when the carried reading existed
        # and the difference clears the noise floor in SOME direction.
        try:
            if v.get("carry_force_n") is not None:
                fz = []
                for _ in range(5):
                    fz.append(float(self.robot.get_tcp_force()[2]))
                    time.sleep(0.02)
                drop = float(v["carry_force_n"] - float(np.mean(fz)))
                v["weight_drop_n"] = round(drop, 2)
                if abs(drop) >= self._PROPRIO_WEIGHT_FLOOR_N:
                    v["weight_dropped"] = bool(drop < 0)
        except Exception:
            pass
        # Aggregate. Hard contradictions fail; all-green passes; anything
        # unmeasurable stays uncertain rather than optimistic.
        hard_fail = (
            v.get("jaws_opened") is False
            or v.get("commanded_pitch_achieved") is False
        )
        measured = [
            x for x in (v.get("jaws_opened"), v.get("commanded_pitch_achieved"),
                        v.get("weight_dropped"))
            if x is not None
        ]
        v["verdict"] = (
            "fail" if hard_fail
            else ("pass" if measured and all(measured) else "uncertain")
        )
        logger.info("[proprio-verdict] %s | %s", v["verdict"],
                    {k: x for k, x in v.items() if k != "verdict"})

    def _release(self, params: dict, t0: float) -> ExecutionResult:
        self._check_abort()

        # Arrival gate. Opening the jaws is irreversible, so it is gated on the
        # object actually being over the container, and the failure is
        # reported (failed action + the transport verify gate) rather than
        # swallowed: an arm that silently refuses to let go and holds the
        # object forever is worse than the drop. executor_core's failure path
        # runs recovery, then force-verifies the grip and keeps the object
        # rather than dumping it wherever the arm happens to be.
        delivered, why = self._transport_delivered()
        if not delivered:
            success_verifier.set_gate(self, "transport", False, why)
            logger.warning("Release REFUSED, transport did not arrive: %s", why)
            return ExecutionResult(
                action_type="release",
                success=False,
                message=f"Refused to open the jaws: {why}",
                duration=time.time() - t0,
            )

        release_tcp = self._get_current_position()
        logger.info(
            "Release TCP: [%.3f, %.3f, %.3f] (z=%.3f)",
            release_tcp[0],
            release_tcp[1],
            release_tcp[2],
            release_tcp[2],
        )

        tilt_angle = abs(params.get("tilt_angle", 0.0))
        tilted = False
        signed_angle = 0.0

        if tilt_angle > 0.01:
            planner_pitch = float(params.get("pitch_sign", 1.0))
            pitch_sign = planner_pitch

            try:
                obs = self.robot.get_observation()
                tcp_pose = obs["tcp_pose"]
                current_rot = Rotation.from_rotvec(tcp_pose[3:6])
                tool_y_world = current_rot.apply([0, 1, 0])
                pitch_sign = 1.0 if tool_y_world[0] > 0 else -1.0
                logger.info(
                    "Pitch sign from TCP: tool_Y_world=(%.2f,%.2f,%.2f) "
                    "obj extends in X=%.2f -> pitch_sign=%.0f (planner hint=%.0f)",
                    *tool_y_world,
                    tool_y_world[0],
                    pitch_sign,
                    planner_pitch,
                )
            except Exception as e:
                logger.warning(
                    "Could not read TCP for pitch sign: %s, using the planner hint", e
                )
                pitch_sign = planner_pitch

            MAX_POS_PITCH = self.MAX_POS_PITCH_RAD
            MAX_NEG_PITCH = self.MAX_NEG_PITCH_RAD

            # Tilt onto the orientation the arm is actually holding, not onto
            # the bare top-down constant: the current TCP rotation carries the
            # yaw the place just aligned, and pitching it about the tool X
            # keeps the alignment and adds the dip.
            try:
                _obs = self.robot.get_observation()
                base_rot = Rotation.from_rotvec(_obs["tcp_pose"][3:6])
                logger.info(
                    "Tilt base: current TCP orientation (keeps the placement yaw)"
                )
            except Exception as _e:  # noqa: BLE001
                logger.warning(
                    "Could not read TCP for the tilt base (%s); falling back to "
                    "GRASP_ORIENTATION, which DROPS any placement yaw", _e
                )
                base_rot = Rotation.from_rotvec(self.GRASP_ORIENTATION)

            # Positive pitch past its limit: clamp, never flip. A 180 wrist
            # flip about z to pitch negative instead costs a full extra wrist
            # revolution and can blow the release budget mid-insert. Excess is
            # clamped by the shared min() below and logged here so a plan
            # asking for the moon is visible without being obeyed.
            #
            # Deep dips are not lost by this clamp: _dip_side_orient chose the
            # place-approach yaw at transport time so a >40 deg request lands
            # on the negative side (90 deg budget) with pitch_sign=-1 and this
            # branch never fires. Reaching it means that choice was impossible
            # (an aligned yaw whose 180 twin exceeds the cable limit, or no
            # transport look-ahead ran); then a shallow 40 deg dip is the
            # right answer, not an at-release reorientation.
            if pitch_sign > 0 and tilt_angle > MAX_POS_PITCH:
                logger.info(
                    "Positive pitch %.0f deg exceeds limit (%.0f); clamping "
                    "(no 180 flip; the dip-side yaw choice was not available)",
                    np.rad2deg(tilt_angle),
                    np.rad2deg(MAX_POS_PITCH),
                )

            if pitch_sign > 0:
                tilt_angle = min(tilt_angle, MAX_POS_PITCH)
            else:
                tilt_angle = min(tilt_angle, MAX_NEG_PITCH)

            signed_angle = -pitch_sign * tilt_angle
            tilt = Rotation.from_euler("x", signed_angle)
            tilted_rot = base_rot * tilt
            tilted_orient = tilted_rot.as_rotvec().tolist()
            logger.info("Tilted release: pitch %+.0f deg", np.rad2deg(signed_angle))

            # Keep the dipping tip over the spot the transport delivered.
            # The tip retracts toward the TCP by half_length*(1-cos) as the
            # tool tilts, so the TCP shifts the same amount along the tool-Y
            # horizontal heading (the direction the object's far end points);
            # a world-X shift would walk the tip sideways off the container
            # under the dip-side yaw. Reduces to pitch_sign*X at zero yaw.
            # Teleop dxy during the dip is small (<=8cm across episodes);
            # this gives 1.9cm at 40 deg.
            half_length = 0.08
            ty_xy = base_rot.apply([0.0, 1.0, 0.0])[:2]
            ty_norm = float(np.linalg.norm(ty_xy))
            tip_dir = (
                ty_xy / ty_norm if ty_norm > 1e-6 else np.array([pitch_sign, 0.0])
            )
            offset_xy = tip_dir * half_length * (1 - np.cos(tilt_angle))
            # Tilt high, not at the release z. The tilt movel raises the TCP
            # by TILT_RAMP_RAISE_M while the tilt ramps (teleop shows the tilt
            # at z=0.11-0.21m, never at the rim); the ~10cm of travel at 0.3x
            # velocity stretches the ramp to the human's 2.5-3s. The tilted
            # insert below then descends back to the resolved release z.
            tilt_pos = release_tcp.copy()
            tilt_pos[0] += offset_xy[0]
            tilt_pos[1] += offset_xy[1]
            tilt_pos[2] += self.TILT_RAMP_RAISE_M
            logger.info(
                "Tilt TCP offset: half_len=%.0fmm, offset_xy=(%+.0f,%+.0f)mm, "
                "raise=%.0fmm -> tilt_pos=(%.3f,%.3f,%.3f)",
                half_length * 1000,
                offset_xy[0] * 1000,
                offset_xy[1] * 1000,
                self.TILT_RAMP_RAISE_M * 1000,
                *tilt_pos,
            )

            try:
                logger.info("Executing tilt movel (ramp at height)...")
                self._move_to_linear(
                    tilt_pos, tilted_orient, velocity=self.velocity * 0.3
                )
                logger.info("Tilt movel completed")
                tilted = True
            except Exception as e:
                logger.warning("Tilt failed: %s - releasing straight", e)
                tilted = False

            if tilted:
                # Descend, tilted, back to the resolved release z: the z the
                # place transport arrived at, which release_height already
                # bounded to keep the TCP off the rim plane (teleop: 12-20cm
                # of z during the dip, jaws opening at the rim plane). Known
                # conservatism: a tilted tool hangs cos(tilt) of its overhang,
                # so the object's bottom ends a couple of cm higher than the
                # vertical-drop geometry aimed; the rim guard is the harder
                # constraint.
                insert_pos = self._get_current_position()
                insert_pos[2] = max(release_tcp[2], self.TABLE_Z_FLOOR + 0.02)
                logger.info(
                    "Inserting tilted: descending to resolved z=%.3f "
                    "(%.0fmm below the ramp)",
                    insert_pos[2],
                    (tilt_pos[2] - insert_pos[2]) * 1000,
                )
                try:
                    self._move_to_linear(
                        insert_pos, tilted_orient, velocity=self.velocity * 0.2
                    )
                    logger.info("Insert descent completed")
                except Exception as e:
                    logger.warning("Insert descent failed: %s", e)
                # Hold still at max tilt before the jaws open (see
                # TILT_SETTLE_S). Abort-aware like every other release sleep.
                self._abort_sleep(self.TILT_SETTLE_S)

        # G3 release witness: read the grip immediately before opening, and
        # again after the settle. held_before == False means the object was
        # already gone; the run fails with no capture and no SAM3, which is
        # the cheap catch for a drop that happened mid-transport.
        _witness_state = success_verifier.begin_release_witness(self)
        # Proprioceptive verdict, part 1: sample what the arm itself can attest
        # before the jaws open (commanded pose reached, commanded tilt
        # achieved, force carried).
        self._release_proprio = self._collect_release_proprio(
            commanded_orient=(tilted_orient if tilted else None),
        )
        self.robot.open_gripper()
        # This settle is load-bearing, not padding: the witness reads the jaw
        # registers by uploading a publish program, which would REPLACE the
        # rq_move_and_wait open still running on the controller. open_gripper
        # sleeps 0.5 s of its own, so the jaws are parked by the time the
        # witness samples them. It then confirms by polling, not by waiting
        # longer (success_verifier._confirm_jaws_open). See
        # RELEASE_OPEN_SETTLE_S for why this one keeps its full duration.
        self._abort_sleep(self._release_open_settle_s())
        # Clear _holding FIRST: on a rig with no jaw register, the witness
        # falls back to this flag, and reading it while still True would
        # report the object as never released on every successful place.
        self._holding = False
        # Consumed: the arrival question this record answered is now settled.
        self._place_arrival_record = None
        success_verifier.finish_release_witness(
            self, _witness_state, self._last_place_label
        )
        # Proprioceptive verdict, part 2: the witness has now read the jaws.
        self._finish_release_proprio()

        # Retract straight up in the current orientation. A release is a grip
        # action and must not command a yaw: reorienting the wrist after the
        # jaws are open is wasted motion, and snapping to GRASP_ORIENTATION
        # from a tilted place pose can swing the wrist ~70-85 deg and overrun
        # the release timeout budget. The next task's pre-home restores the
        # base orientation anyway. Read the current orientation via
        # get_observation() (the normalized cross-family pose interface); if
        # it cannot be read, do not guess a different one.
        orient = None
        try:
            obs = self.robot.get_observation()
            # `a or b` on a numpy array raises "truth value of an array with
            # more than one element is ambiguous"; compare to None.
            tcp = obs.get("tcp_pose")
            if tcp is None:
                tcp = obs.get("tcp_pos")
            if tcp is not None and len(tcp) >= 6:
                orient = list(np.array(tcp[3:6], dtype=float))
                logger.info(
                    "Release retract: holding the CURRENT orientation %s",
                    [round(float(v), 3) for v in orient],
                )
            else:
                logger.warning(
                    "Release retract: get_observation gave no tcp_pose (%r); "
                    "falling back to GRASP_ORIENTATION, which will SNAP the "
                    "wrist back to base after the release",
                    list(obs.keys()) if isinstance(obs, dict) else type(obs),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Release retract: could not read the current orientation (%s); "
                "falling back to GRASP_ORIENTATION -- expect a wrist snap", exc
            )
        if orient is None:
            logger.warning(
                "Release retract: current orientation unreadable; skipping the "
                "reorientation entirely rather than snapping to base"
            )
        current = self._get_current_position()
        retract = current.copy()
        # Retract height is a parameter so the opening jaws clear neighbouring
        # objects (a couple of centimetres) without stacking on a lift the
        # plan adds itself: 8 cm plus a move_relative dz 0.05 is 13 cm up
        # against compliant_push's 6 cm of travel.
        retract_z = float(params.get("retract_z", 0.08))
        retract[2] += retract_z
        if abs(retract_z - 0.08) > 1e-6:
            logger.info("Release retract: %.3f m (planner-set)", retract_z)
        if orient is not None and abs(retract_z) > 1e-6:
            self._move_to(retract, orient)
        elif orient is None and abs(retract_z) > 1e-6:
            # The orientation read failed, but the lift must still happen:
            # skipping it leaves the open jaws parked centimeters beside the
            # just-placed object, a worse failure than the wrist snap. Try the
            # executor's own TCP read (a different path than get_observation)
            # before paying the snap to base.
            fb = None
            cur = getattr(self, "current_orientation", None)
            if callable(cur):
                fb = cur()
            self._move_to(
                retract, list(fb) if fb is not None else list(self.GRASP_ORIENTATION)
            )
        msg = f"Released at z={self._get_current_position()[2]:.3f}"
        if tilted:
            msg += f" (tilted {np.rad2deg(signed_angle):+.0f} deg, inserted)"

        # A release that did not open the jaws is a failed release. Returning
        # success here lets the tree go on to close_gripper + compliant_push
        # and press the still-held tool into the bench (~89 N) until the
        # controller protective-stops. Reporting the failure routes the same
        # situation into recovery instead, which lifts.
        #
        # Positive evidence only: `released is False` means the jaws were
        # measured shut. An unknown (None) still passes, because the registers
        # are known to read stale and a false failure here would strand a
        # perfectly good place.
        witness = getattr(self, "_release_witness", None)
        released = None
        if witness is not None:
            released = getattr(witness, "released", None)
            if released is None and isinstance(witness, dict):
                released = witness.get("released")
        if released is False:
            logger.warning(
                "Release FAILED: the jaws never opened (%s). Reporting failure "
                "so recovery runs -- continuing here is what pressed a held "
                "tool into the bench on 2026-08-24.", msg,
            )
            return ExecutionResult(
                action_type="release",
                success=False,
                message="Release did not open the jaws; still holding",
                duration=time.time() - t0,
            )
        return ExecutionResult(
            action_type="release", success=True, message=msg, duration=time.time() - t0
        )




class PlaceMixin:
    """Transport to a place target: release height, standoff, descent and arrival."""

    def _place_release_z(self, plan_z: float, container, params: dict) -> float:
        """Resolve the Z a place transport flies to; see control/release_height.

        Binds executor state (held detection, table plane, grasp Z offset) to
        the pure resolver and logs the verdict. Every branch returns a number;
        an unmeasurable container returns ``plan_z`` unchanged.
        """
        held_label = getattr(self, "_active_grasp_label", "") or ""
        held = self.detection_map.get(held_label) if held_label else None
        strict = bool(params.get("strict_offset_z", False))
        # Effective grasp-z offset for held_drop. The static constant assumes
        # the TCP closed at perceived_top + GRIPPER_OPEN_Z_OFFSET; the real
        # grasp descends further (SPARK_GRASP_DEPTH_M, contact guard, learned
        # per-label offsets, recovery Z bias), so gripping N cm lower means the
        # object hangs N cm LESS below the TCP. Behind place.release_giveback
        # (default off); plan-strict and plan-no-geometry bypass the drop
        # model either way.
        gz = float(self.GRIPPER_OPEN_Z_OFFSET)
        if self._release_giveback_enabled():
            actual = getattr(self, "_actual_grasp_tcp_z", None)
            ptz = getattr(self, "_grasp_perception_target_z", None)
            if actual is not None and ptz is not None:
                gz = float(actual) - float(ptz)
                logger.info(
                    "Release give-back: effective grasp_z_offset=%.3f "
                    "(actual close TCP %.3f - perceived top %.3f; constant "
                    "was %.3f)",
                    gz,
                    float(actual),
                    float(ptz),
                    float(self.GRIPPER_OPEN_Z_OFFSET),
                )
        try:
            out = resolve_release_z(
                plan_z=plan_z,
                container=container,
                held=held,
                table_z=float(self.TABLE_Z_FLOOR),
                strict=strict,
                grasp_z_offset=gz,
            )
        except Exception as exc:  # noqa: BLE001 - a height quirk must not stop a place
            logger.warning(
                "Release height: could not resolve (%s); keeping the plan's " "z=%.3f",
                exc,
                plan_z,
            )
            return float(plan_z)
        if abs(out.z - plan_z) < 1e-6:
            logger.info("Release height: %s (unchanged)", out.describe())
        else:
            logger.info(
                "Release height: plan z=%.3f -> %.3f (%+.1f cm) %s",
                plan_z,
                out.z,
                (out.z - plan_z) * 100.0,
                out.describe(),
            )
        self._last_release_height = out
        return float(out.z)

    def _release_giveback_enabled(self) -> bool:
        """place.release_giveback in the family YAML (default OFF = the static
        GRIPPER_OPEN_Z_OFFSET). SPARK_RELEASE_GIVEBACK env overrides."""
        env = os.environ.get("SPARK_RELEASE_GIVEBACK")
        if env is not None:
            return env.strip() not in ("", "0", "false", "no")
        try:
            profile = getattr(self._pipeline, "profile", None)
            raw = getattr(profile, "raw", None) or {}
            return bool((raw.get("place") or {}).get("release_giveback", False))
        except Exception:  # noqa: BLE001
            return False

    def _place_arrival(self, label: str, detection) -> tuple:
        """Did the arm end up OVER container ``label``? -> (ok, detail).

        ``ok`` is True whenever the question cannot be answered: a container
        with no slots and no measured width has an UNKNOWN extent
        (success_predicates._container_obb returns None). Missing perception
        never blocks a release.

        The test is the container's own OBB in XY, widened by
        PLACE_ARRIVAL_XY_MARGIN_M, evaluated against the CURRENT TCP, not
        against the pose the servo was commanding.
        """
        obb = container_obb_xy(detection) if detection is not None else None
        if obb is None:
            return True, ""
        tcp = self._get_current_position()
        if obb_contains(tcp[:2], obb, self.PLACE_ARRIVAL_XY_MARGIN_M):
            return True, ""
        centre, half_a, half_b, theta = obb
        d = np.asarray(tcp[:2], dtype=float) - np.asarray(centre, dtype=float)
        ca, sa = np.cos(theta), np.sin(theta)
        along = abs(d[0] * ca + d[1] * sa)
        across = abs(-d[0] * sa + d[1] * ca)
        detail = (
            f"TCP ({tcp[0]:.3f},{tcp[1]:.3f}) is outside '{label or '?'}': "
            f"{along * 100:.1f}/{across * 100:.1f} cm from centre vs "
            f"half=({half_a * 100:.1f},{half_b * 100:.1f})cm "
            f"+{self.PLACE_ARRIVAL_XY_MARGIN_M * 100:.0f}cm margin"
        )
        return False, detail

    def _note_place_arrival(self, label: str, arrived: bool, detail: str) -> None:
        """Record that a place transport ran, and how it went.

        Read by ReleaseMixin._transport_delivered, which uses the LABEL to know
        which container to judge against and recomputes the verdict from the
        live TCP; ``arrived``/``detail`` feed the failed-action message and
        the log. The release ignores a record whose label no longer matches
        ``_last_place_label``.
        """
        self._place_arrival_record = {
            "label": label,
            "arrived": bool(arrived),
            "detail": detail,
        }

    def _transport_to(
        self,
        target: np.ndarray,
        target_detection: dict = None,
        target_label: str = "",
        params: dict = None,
        lead=None,
    ):
        """
        Lift, horizontal move, servo descend with orientation matching.

        `params` are the plan node's params; only grasp_strategy /
        grasp_yaw_deg are read.

        `lead` is an optional list of ``(position, orientation, velocity,
        label)`` transit waypoints to fly BEFORE the lift, as part of the same
        blended phrase (the legato look-ahead folds a preceding
        ``move_relative`` into the transport). Ignored when blending is off.

        Returns True if the transport reached its place pose. False when:

          * the grip check found a DROP (``_holding`` is already cleared), or
          * the arm did not ARRIVE inside the target container's planar
            footprint (``_place_arrival``). The object is still held; the
            caller reports the action failed so the tree can recover.
        """
        self._check_abort()
        self._last_place_label = target_label
        self._place_arrival_record = None

        orient = self.GRASP_ORIENTATION
        # Does `orient` end up carrying a slot/fit/OBB alignment? Read by the
        # deep-dip side choice below: an aligned yaw may only be traded for
        # its grasp-symmetric 180 twin, a free yaw may be picked outright.
        yaw_aligned = False
        # Slotted-container safety net for a generic place (move_to_keypoint
        # + release): derive the slot direction the SAME way place_in_slot does.
        slots = target_detection.get("slots") if target_detection else None
        if slots:
            slot_dir, slot_src = resolve_slot_direction(target_detection, target_label)
            yaw = (slot_dir + np.pi / 2.0) % np.pi - np.pi / 2.0
            yaw = self._nearest_symmetric_yaw(yaw)
            # Slot direction is a validated source and bypasses the strategy
            # gates, but the cable limit still applies to the composed rotation.
            slot_orient = compose_yaw(self, yaw, context="place-slots")
            if slot_orient is not None:
                # Same directed head/tail decision as the obb place path and
                # place_in_slot; axis alignment alone (mod pi) is a coin flip
                # on which end of the slot the tool's tip lands in.
                orient = self._orient_head_to_head(slot_orient, target_detection)
                yaw_aligned = True
            logger.info(
                "[place_orient] slotted container '%s': aligning to slots "
                "(dir=%.1f deg src=%s) -> yaw=%.1f deg, NOT raw OBB",
                target_label or "?",
                np.rad2deg(slot_dir),
                slot_src,
                np.rad2deg(yaw),
            )
        elif target_detection:
            target_orient = target_detection.get("orientation_angle")

            # SHAPE FIT FIRST; it decides on its own whether orientation
            # matters. A slot or tool cut-out is shaped like the thing that
            # goes in it, so one rotation fits with a large margin; for a bowl
            # the margin collapses and the fit says "do not rotate".
            self._place_fit_indifferent = False
            fitted = self._fit_place_yaw(target_detection)
            if fitted is None and self._place_fit_indifferent:
                # The fit overlaps equally at every rotation: no yaw for this place. Without this
                # the axis path turns the wrist to a "long axis" that an occluded bowl mask invents.
                logger.info("[place_orient] the shape fit is the same at every rotation; no yaw for '%s'",
                            target_label or "?")
                target_orient = None
            if fitted is not None:
                orient = fitted
                yaw_aligned = True
                logger.info("[place_orient] using the shape fit for the place")
                target_orient = None  # the fit is the only source for this target

            # The axis path gives the right LINE; the fit is also used for the
            # half it answers without any image-to-world frame conversion:
            # whether that line should be flipped 180 deg. See
            # _orient_head_to_head, which compares both in one image frame.

            if target_orient is not None and abs(float(target_orient)) > 1e-3:
                place_orient, place_strategy = resolve_grasp_orientation(
                    params or {}, target_detection, self, context="place"
                )
                # Adopt ANY resolved yaw ("obb", "plane", "target_axis",
                # "caller_yaw").
                if place_strategy != "topdown" and (
                    place_orient is not None
                ):
                    orient = place_orient
                    place_orient = self._orient_head_to_head(
                        place_orient, target_detection
                    )
                    orient = place_orient
                    yaw_aligned = True
                    logger.info(
                        "[place_orient] adopting '%s' yaw for the place",
                        place_strategy,
                    )
            else:
                logger.info(
                    "[place_orient] skipping yaw-align for '%s' " "(orient<=1mrad or missing)",
                    target_label or "?",
                )

        # DEEP-DIP SIDE CHOICE, decided BEFORE the transport flies and folded
        # into the approach orientation. When the upcoming release asks for a
        # dip past the positive pitch budget (wrist-camera clearance, 40 deg),
        # rotate the approach yaw so the dip lands on the NEGATIVE side (90 deg
        # budget). See ReleaseMixin._dip_side_orient for the rules.
        orient = self._dip_side_orient(orient, yaw_aligned=yaw_aligned)

        place_target = target.copy()
        logger.info("Place '%s': target_z=%.3f", target_label, place_target[2])

        current_pose = self._get_current_position()
        clearance_from = np.asarray(lead[-1][0], dtype=float) if lead else current_pose
        safe_z = self._safe_clearance_z(clearance_from, place_target)
        lift = clearance_from.copy()
        lift[2] = safe_z
        above_target = place_target.copy()
        above_target[2] = safe_z
        # Hand the servo only the last centimetres of the descent. The servo
        # is a PD loop whose gain falls with the error (kp_pos=2.0,
        # pos_threshold=3 mm), so it crawls near convergence (3.2 s for the
        # full descent). A movej covers the free-space part; the servo keeps
        # the part that can touch something, with its compliance and its
        # unreachable-target stall exit intact.
        standoff = place_target.copy()
        standoff[2] = min(
            float(place_target[2]) + self._place_standoff_m(place_target), safe_z
        )

        if self._blend_enabled():
            phrase = list(lead or [])
            # Lift at the orientation the arm is ALREADY holding, not the base
            # constant: snapping to GRASP_ORIENTATION would rotate the wrist
            # to zero on the way up and out to the place yaw on the way
            # across. Holding the carry angle leaves a SINGLE rotation during
            # the horizontal transit.
            carry_orient = getattr(self, "_active_grasp_orient", None)
            if not carry_orient:
                try:
                    carry_orient = list(
                        np.asarray(self.robot.get_tcp_pose(), dtype=float)[3:6]
                    )
                except Exception:  # noqa: BLE001
                    carry_orient = list(self.GRASP_ORIENTATION)
            phrase.append((lift, list(carry_orient), None, "lift"))
            phrase.append((above_target, list(orient), None, "above_target"))
            if standoff[2] < above_target[2] - 1e-4:
                # Terminal row of the SAME program: no extra upload; the r=0
                # stop is the one the exact joint-arrival test applies to.
                phrase.append((standoff, list(orient), None, "place_standoff"))
            if self._blend_transit(phrase, final_radius_m=0.0, phrase="transport"):
                # ONE grip check, at the parked end of the phrase. A
                # mid-transport check sends a gripper URScript, which REPLACES
                # the running program and would cancel the blend; a passive
                # register read cannot see a drop since the last publish.
                if not self._transport_grip_ok("above_target"):
                    self._holding = False
                    return False
                return self._place_descent(place_target, orient, target_label, target_detection)
            for pos_i, ori_i, _v, _lbl in lead or []:
                self._move_to(pos_i, ori_i)

        self._move_to(lift, self.current_orientation())

        # Per-waypoint drop detection at EACH stationary transport arrival
        # (lift, then above-target). On a confirmed drop, clear _holding and
        # return; the caller fails the action and recovery re-grasps.
        if not self._transport_grip_ok("lift"):
            self._holding = False
            return False

        self._move_to(above_target, orient)
        if not self._transport_grip_ok("above_target"):
            self._holding = False
            return False

        # Same standoff as the blended phrase, at the cost of one extra
        # program upload.
        if standoff[2] < above_target[2] - 1e-4:
            self._move_to(standoff, orient)
        return self._place_descent(place_target, orient, target_label, target_detection)

    def _place_standoff_m(self, place_target) -> float:
        """How much of the descent the slow servo must own, in metres.

        A held object first meets the container rim when its BOTTOM reaches
        rim height, i.e. at TCP z = rim_z + held_drop. For anything taller
        than the container is deep that point is ABOVE the release height,
        and contact would land in the fast blended move (no compliance, no
        stall exit). Example: rim -0.220, held_drop 10.3 cm, release z -0.139:
        contact begins at -0.117, but a fixed 2 cm standoff takes over at
        -0.119.

        So: start the slow phase strictly ABOVE the first possible contact,
        with a margin for the held-extent estimate. Short objects keep the
        fixed 2 cm because their computed value is smaller.
        """
        # SPARK_PLACE_SLOW_FROM_M: start the slow descent at least this far above the release
        # height (a soft object's hang reads short, so the fast move can end inside the contact zone).
        slow_from_m = float(os.environ.get("SPARK_PLACE_SLOW_FROM_M", "0") or 0)
        base = max(float(self.PLACE_SERVO_STANDOFF_M), min(slow_from_m, self.PLACE_SERVO_STANDOFF_MAX_M))
        geom = getattr(self, "_last_release_height", None)
        rim = getattr(geom, "rim_z", None)
        drop = getattr(geom, "held_drop_m", None)
        if rim is None or drop is None:
            return base
        try:
            contact_z = float(rim) + float(drop)
            need = (contact_z - float(place_target[2])) + self.PLACE_CONTACT_MARGIN_M
        except (TypeError, ValueError):
            return base
        if not np.isfinite(need) or need <= base:
            return base
        need = min(need, self.PLACE_SERVO_STANDOFF_MAX_M)
        logger.info(
            "Place standoff: %.1f cm (rim=%.3f + held_drop=%.3f -> first "
            "contact at z=%.3f, %.1f cm above the release height); the slow "
            "servo owns the whole contact zone instead of the blend",
            need * 100.0,
            float(rim),
            float(drop),
            contact_z,
            (contact_z - float(place_target[2])) * 100.0,
        )
        return need

    def _place_descent(self, place_target, orient, target_label: str, target_detection) -> bool:
        """Final servo of a transport, plus the arrival question it answers.

        A stall alone does NOT fail the transport: a 4.9 mm stall in the
        middle of a bowl has delivered the object. What fails it is ending up
        outside the container's own footprint.
        """
        # ORIENTATION IS SETTLED AT HEIGHT, NEVER IN THE CONTACT ZONE. A held
        # tool extends far below the TCP, so a wrist rotation near the rim
        # sweeps a volume the planner never cleared (a held screwdriver swept
        # into the tool bed and the controller went into PROTECTIVE_STOP). If
        # orientation error remains here, rise to a clearance height, fix the
        # wrist THERE, and come back down; the contact-zone servo then owns
        # position only.
        try:
            cur_o = self.current_orientation()
            if cur_o is not None and orient is not None:
                from scipy.spatial.transform import Rotation as _R

                ori_err = float(
                    (_R.from_rotvec(list(orient)).inv()
                     * _R.from_rotvec(list(cur_o))).magnitude()
                )
                if ori_err > self.PLACE_ORI_FIX_GATE_RAD:
                    fix = np.array(place_target, dtype=float).copy()
                    fix[2] = float(place_target[2]) + self.ROTATE_CLEARANCE_M
                    logger.warning(
                        "[place] %.1f deg of orientation left at the "
                        "standoff; correcting at z=%.3f (+%.0f cm clearance) "
                        "before the contact-zone descent",
                        np.rad2deg(ori_err), fix[2],
                        self.ROTATE_CLEARANCE_M * 100,
                    )
                    self._move_to(fix.tolist(), list(orient))
        except Exception as exc:  # noqa: BLE001 - the descent must still run
            logger.warning("[place] orientation pre-check failed: %s", exc)
        # 0.25, not 0.5: this servo owns the rim-contact zone (see
        # _place_standoff_m), so its speed sets the impulse delivered to the
        # container on contact. Keeps a light bowl from being knocked over.
        descent_scale = float(os.environ.get("SPARK_PLACE_DESCENT_SCALE", "0.25") or 0.25)
        landed = self._servo_to(place_target, orient, velocity=self.velocity * descent_scale)
        arrived, detail = self._place_arrival(target_label, target_detection)
        self._note_place_arrival(target_label, arrived, detail)
        if arrived:
            if not landed:
                logger.info(
                    "Place '%s': servo stalled but the TCP is over the "
                    "container; releasing here is still correct",
                    target_label or "?",
                )
            return True
        logger.warning(
            "Place '%s' DID NOT ARRIVE (%s): %s",
            target_label or "?",
            "servo stalled" if not landed else "servo converged",
            detail,
        )
        return False

    def _transport_grip_ok(self, waypoint: str) -> bool:
        """Maintain + verify the grip at a STATIONARY transport waypoint.

        Robotiq path: VERIFY FIRST via the gObj-primary _grip_intact() (one
        forced publish + two register reads, ~30-200 ms, no jaw motion). Only
        if that does not confirm, re-squeeze to the fully-closed POSITION at
        GRASP_CLOSE_FORCE to take up slow slip, then re-verify. A False after
        that is a real drop.

        Non-robotiq path: delegate to the force-based
        _verify_holding_during_transport().
        """
        if self._gripper_type() != "robotiq_2f85":
            return self._verify_holding_during_transport()
        if self._grip_intact():
            return True
        close_force = int(os.environ.get("SPARK_GRASP_CLOSE_FORCE", self.GRASP_CLOSE_FORCE))
        logger.info(
            "Transport '%s': grip unconfirmed, re-squeezing to take up slack",
            waypoint,
        )
        self._robotiq_resqueeze(speed=50, force=close_force, settle=self.GRASP_VERIFY_SETTLE_S)
        intact = self._grip_intact()
        if not intact:
            logger.warning("Transport: object DROPPED at waypoint '%s'", waypoint)
            # G2: the run can no longer pass without a later successful re-grasp.
            success_verifier.note_transport_drop(self, waypoint)
        return intact




class PlaceYawMixin:
    """Place-time wrist yaw: fit to the target and head-to-head orientation."""

    def _fit_place_yaw(self, target_detection):
        """Place yaw from SHAPE AGREEMENT, or None when shape cannot decide.

        Registers the held object's mask against the receptacle's over every
        rotation and keeps the best. Returns a composed orientation when the
        winner beats every orientation >=90 deg away by `JIGSAW_MIN_MARGIN`,
        and None otherwise (a round container).

        The angle is applied as a DELTA on the orientation the arm is already
        carrying, so it never depends on an image-to-world sign convention.
        """
        try:
            held = getattr(self, "_held_mask", None)
            tgt = (
                target_detection.get("_mask")
                if isinstance(target_detection, dict)
                else None
            )
            if held is None or tgt is None:
                return None
            from spark_real.perception.masks import best_fit_rotation

            ang, iou, margin = best_fit_rotation(held, tgt)
            logger.info(
                "[fit] rotation %.1f deg  IoU %.3f  margin %.3f "
                "(need margin>=%.3f, IoU>=%.3f)",
                np.rad2deg(ang), iou, margin,
                self.JIGSAW_MIN_MARGIN, self.JIGSAW_MIN_IOU,
            )
            # A good overlap at every rotation (IoU high, no margin between rotations) is an answer:
            # the target is round for this object, so the place needs no yaw at all.
            self._place_fit_indifferent = bool(iou >= self.JIGSAW_MIN_IOU and margin < self.JIGSAW_MIN_MARGIN)
            if margin < self.JIGSAW_MIN_MARGIN or iou < self.JIGSAW_MIN_IOU:
                logger.info(
                    "[fit] shape does not decide the orientation here "
                    "(margin %.3f); leaving the yaw to the axis path",
                    margin,
                )
                return None

            from spark_real.control.grasp_strategy import (
                base_orientation,
                compose_yaw,
                measured_yaw_offset,
            )

            cur = measured_yaw_offset(
                self._active_grasp_orient or base_orientation(self),
                base_orientation(self),
            )
            # want = cur + ang: the fit reports the signed rotation that
            # carries the held shape onto the target, ADDED to the carry yaw,
            # no extra 90 and no negation (verified: cur -5.7, ang 273 = -87
            # -> want -92.7 -> tool_X -8.6 against a measured -9). `cur - ang`
            # and `cur - ang - 90` were each wrong by a clean 90 or 180.
            want = cur + float(ang)
            want = (want + np.pi) % (2 * np.pi) - np.pi
            if abs(want) > self.MAX_YAW_OFFSET + 1e-9:
                alt = want + (np.pi if want < 0 else -np.pi)
                if abs(alt) <= self.MAX_YAW_OFFSET + 1e-9:
                    # Substitute only when the shapes cannot tell the two ends
                    # apart. A DECISIVE fit means the 180-equivalent is the
                    # orientation the fit rejected: refusing to orient beats
                    # orienting backwards (grasp_strategy.place_aware_yaw
                    # prevents most of these upstream).
                    if margin >= 2.0 * self.JIGSAW_MIN_MARGIN:
                        logger.warning(
                            "[fit] wanted %.1f deg (past the wrist limit) and "
                            "the fit is DECISIVE (margin %.3f), so the "
                            "180-equivalent %.1f deg would seat the object "
                            "end-for-end; refusing to orient rather than "
                            "placing it backwards",
                            np.rad2deg(want), margin, np.rad2deg(alt),
                        )
                        return None
                    logger.warning(
                        "[fit] wanted %.1f deg (past the wrist limit); using "
                        "the 180-equivalent %.1f deg -- margin %.3f says the "
                        "ends are interchangeable here",
                        np.rad2deg(want), np.rad2deg(alt), margin,
                    )
                    want = alt
                else:
                    logger.warning(
                        "[fit] wanted %.1f deg, beyond the wrist limit and no "
                        "reachable equivalent; leaving the yaw to the axis path",
                        np.rad2deg(want),
                    )
                    return None
            return compose_yaw(self, want, context="place")
        except Exception as exc:  # noqa: BLE001 - never let this stop a place
            logger.warning("[fit] failed (%s); falling back to the axis path", exc)
            return None

    def _orient_head_to_head(self, place_orient, target_detection):
        """Flip the place yaw by 180 deg when the object would go in backwards.

        The yaw resolved so far aligns the two AXES; a PCA axis is a line,
        identical under a 180 deg flip, and the wrong half is a tool seated
        end-for-end.

        Both masks break the tie by being fatter at one end
        (mask_geometry.heavy_end_sign). The comparison stays in ONE image
        frame, so no image-to-world sign convention can invert it.

        Requires BOTH heavy ends to be known. Either coming back 0.0 means the
        shape is too symmetric to call, and the yaw passes through untouched.
        """
        # FIRST: jigsaw. Which rotation makes the two SHAPES coincide answers
        # direction and angle together. Only fall through to the heavy-end
        # heuristic when the shapes cannot decide it.
        try:
            mask_t0 = (
                target_detection.get("_mask")
                if isinstance(target_detection, dict)
                else None
            )
            mask_h0 = getattr(self, "_held_mask", None)
            if mask_t0 is not None and mask_h0 is not None:
                from spark_real.perception.masks import best_fit_rotation

                _ang, _iou, _margin = best_fit_rotation(mask_h0, mask_t0)
                logger.info(
                    "[jigsaw] best-fit rotation %.1f deg, IoU %.3f, margin %.3f "
                    "(need margin >= %.3f)",
                    np.rad2deg(_ang), _iou, _margin, self.JIGSAW_MIN_MARGIN,
                )
                if _margin >= self.JIGSAW_MIN_MARGIN and _iou >= self.JIGSAW_MIN_IOU:
                    ang_h = getattr(self, "_held_axis_img", None)
                    mt, _, _, _ = __import__(
                        "spark_core.mask_obb",
                        fromlist=["_pca_obb"],
                    )._pca_obb(np.asarray(mask_t0))
                    if ang_h is not None:
                        # Does the shape fit agree with the axis alignment, or
                        # with its 180 flip? Same image frame throughout.
                        def _w(a):
                            return (a + np.pi) % (2 * np.pi) - np.pi

                        axis_rot = _w(mt - ang_h)
                        # Reduce into [-pi/2, pi/2] BEFORE comparing. A PCA
                        # eigenvector's sign is arbitrary, so mt - ang_h is
                        # only defined mod pi (36/50 vs 50/50 correct flips on
                        # synthetic trials). Same reduction as the heavy-end
                        # rung below.
                        if axis_rot > np.pi / 2:
                            axis_rot -= np.pi
                        elif axis_rot < -np.pi / 2:
                            axis_rot += np.pi
                        if abs(_w(_ang - axis_rot)) > np.pi / 2:
                            flipped = self._flip_yaw_180(place_orient)
                            if flipped is not None:
                                logger.info(
                                    "[jigsaw] shape fit disagrees with the axis "
                                    "alignment by ~180; flipping the place yaw"
                                )
                                return flipped
                            logger.warning(
                                "[jigsaw] shape fit wants a 180 flip but it "
                                "exceeds the wrist limit; seating end-for-end"
                            )
                        else:
                            logger.info("[jigsaw] shape fit agrees; no flip")
                        return place_orient
        except Exception as exc:  # noqa: BLE001
            logger.warning("[jigsaw] failed (%s); falling back to heavy-end", exc)

        try:
            sgn_h = getattr(self, "_held_axis_sign", None)
            ang_h = getattr(self, "_held_axis_img", None)
            mask_t = (
                target_detection.get("_mask")
                if isinstance(target_detection, dict)
                else None
            )
            if sgn_h in (None, 0.0) or ang_h is None or mask_t is None:
                logger.info(
                    "[head-to-head] not decidable (held sign=%s, target mask=%s); "
                    "leaving the yaw as resolved",
                    sgn_h,
                    "yes" if mask_t is not None else "no",
                )
                return place_orient

            from spark_core.mask_obb import _pca_obb
            from spark_real.perception.masks import heavy_end_sign

            ang_t, ar_t, _, _ = _pca_obb(np.asarray(mask_t))
            sgn_t = heavy_end_sign(np.asarray(mask_t), ang_t)
            if sgn_t == 0.0:
                logger.info(
                    "[head-to-head] target heavy end UNKNOWN (ar %.2f); leaving "
                    "the yaw as resolved",
                    ar_t,
                )
                return place_orient

            def _wrap(a):
                return (a + np.pi) % (2 * np.pi) - np.pi

            dir_h = ang_h + (0.0 if sgn_h > 0 else np.pi)
            dir_t = ang_t + (0.0 if sgn_t > 0 else np.pi)
            # The rotation the pipeline effectively applied, reduced mod 180,
            # which is all an axis alignment can express.
            rot = _wrap(ang_t - ang_h)
            if rot > np.pi / 2:
                rot -= np.pi
            elif rot < -np.pi / 2:
                rot += np.pi
            residual = _wrap(dir_t - (dir_h + rot))
            if abs(residual) <= np.pi / 2:
                logger.info(
                    "[head-to-head] heads agree (residual %.0f deg); no flip",
                    np.rad2deg(residual),
                )
                return place_orient

            flipped = self._flip_yaw_180(place_orient)
            if flipped is None:
                logger.warning(
                    "[head-to-head] object would seat BACKWARDS (residual "
                    "%.0f deg) but the 180 flip exceeds the wrist limit; "
                    "placing anyway, end-for-end",
                    np.rad2deg(residual),
                )
                return place_orient
            logger.info(
                "[head-to-head] object would seat BACKWARDS (residual %.0f deg); "
                "flipping the place yaw 180",
                np.rad2deg(residual),
            )
            return flipped
        except Exception as exc:  # noqa: BLE001 - never let this stop a place
            logger.warning("[head-to-head] failed (%s); yaw unchanged", exc)
            return place_orient

    def _flip_yaw_180(self, orient):
        """`orient` rotated 180 deg about world Z, or None if out of limit."""
        from spark_real.control.grasp_strategy import (
            base_orientation,
            compose_yaw,
            measured_yaw_offset,
        )

        cur = measured_yaw_offset(orient, base_orientation(self))
        for cand in (cur + np.pi, cur - np.pi):
            if abs(cand) <= self.MAX_YAW_OFFSET + 1e-9:
                return compose_yaw(self, cand, context="place")
        return None



# The `placement:` block of the robot profile. Every threshold is a knob: tune on the robot.
DEFAULTS = {
    "pointed_slots": False,
    "pointing_camera": "birdview",
    "pointing_crop_margin_px": 40,     # image border kept around the container in the crop sent to the planner
    "pointing_max_reasks": 1,          # extra requests after a missing / outside point, then the index path
    "pointing_coord_scale": 1000,      # reply coordinates run 0..scale across the crop (Qwen-VL style); 0 = crop pixels
    "pointing_min_mask_px": 300,       # a smaller SAM3 mask is not a compartment
    "pointing_max_mask_frac": 0.6,     # of the container mask; a larger SAM3 mask is the whole container
    "pointing_min_inside_frac": 0.8,   # share of the SAM3 mask that must lie on the container
    "pointing_min_axis_ar": 1.3,       # below this the compartment has no long axis; the earlier slot direction stays
    "pointing_pixel_radius_px": 6,     # disc around the point, used when the SAM3 mask is refused
}

PROMPT = (
    "This camera image shows a container, outlined in green. A robot arm is holding {held} and will "
    "put it down in the container. Point at the centre of a free compartment of the container that "
    "the held object fits; avoid compartments that are covered or already hold something. {coords} "
    'Reply with JSON only: {{"point": [x, y]}}. If no compartment is free, reply {{"point": null}}.'
)


def settings(pipeline) -> dict:
    out = dict(DEFAULTS)
    raw = getattr(getattr(pipeline, "profile", None), "raw", None) or {}
    block = raw.get("placement") if hasattr(raw, "get") else None
    if isinstance(block, dict):
        out.update({k: block[k] for k in out if k in block})
    return out


def enabled(pipeline) -> bool:
    """UI toggle when set, else ``placement.pointed_slots`` (default off)."""
    if _routes_state.pointed_slots is not None:
        return bool(_routes_state.pointed_slots)
    return as_bool(settings(pipeline)["pointed_slots"], False)


def parse_point(reply):
    """The [x, y] in a planner reply, or None.

    Accepts {"point": [x, y]}, {"x": .., "y": ..}, [{"point_2d": [x, y]}], fenced JSON, or text
    that holds exactly two numbers.
    """
    text = str(reply or "")
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    try:
        obj = json.loads(m.group()) if m else None
    except ValueError:
        obj = None
    while isinstance(obj, list) and obj and isinstance(obj[0], (dict, list)):
        obj = obj[0]
    if isinstance(obj, dict):
        if "x" in obj and "y" in obj:
            obj = [obj["x"], obj["y"]]
        else:
            obj = next((obj[k] for k in ("point", "point_2d", "coordinate", "coordinates") if k in obj), None)
    if obj is None and m is None:
        obj = re.findall(r"-?\d+(?:\.\d+)?", text)
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        try:
            x, y = (float(v) for v in obj if not isinstance(v, bool))
            return x, y
        except (TypeError, ValueError):
            return None
    return None


def ask_slot_point(planner, crop, held, cfg, hint=""):
    """One planner request; returns the point in crop pixels, or None.

    The function name is what run_timing records as the caller ("by") of this request.
    """
    h, w = crop.shape[:2]
    scale = float(cfg["pointing_coord_scale"] or 0)
    coords = (
        f"Give x and y on a 0 to {scale:g} scale: [0, 0] is the top-left corner of the image and "
        f"[{scale:g}, {scale:g}] is the bottom-right corner."
        if scale > 0
        else f"Give x and y in pixels; the image is {w} wide and {h} high, [0, 0] is the top-left corner."
    )
    text = PROMPT.format(held=held, coords=coords) + (" " + hint if hint else "")
    reply = planner.generate([text, crop], tag="point_slot", json_only=True)
    pt = parse_point(reply)
    logger.info("[point-slot] planner reply %r -> %s", str(reply)[:120], pt)
    if pt is None:
        return None
    if scale > 0:
        return pt[0] * w / scale, pt[1] * h / scale
    return pt


def _sam3_point(pipeline, rgb, depth, px, py, cam):
    """SAM3 point prompt at an image pixel: the same call /api/detect_click makes. Returns a mask or None."""
    from spark_real.routes.detection import _run_point_prompt  # imports torch

    dets = _run_point_prompt(pipeline, rgb, depth, px, py, "compartment", cam_name=cam)
    return np.asarray(dets[0].mask) > 0 if dets else None


def _container_mask(pipeline, cam, frame, label, det):
    """A fresh mask of the container on the pointing camera (the stored one is from plan time)."""
    base = label.rsplit(" ", 1)[0] if label[-1:].isdigit() else label
    dets = [d for d in pipeline.detect({cam: frame}, prompts=[base], use_fusion=False)
            if getattr(d, "mask", None) is not None]
    if not dets:
        return None
    here = det_field(det, "position_3d")

    def dist(d):
        p = getattr(d, "position_3d", None)
        return 0.0 if p is None or here is None else float(np.linalg.norm(np.asarray(p[:2]) - np.asarray(here[:2])))

    mask = np.asarray(min(dets, key=dist).mask) > 0
    return mask if mask.shape == frame["rgb"].shape[:2] and mask.any() else None


def _world(mask, depth, cal):
    """(xy, long-axis angle, aspect ratio) of an image mask in the robot base frame; xy None without depth."""
    if depth is None or cal is None:
        return None, None, 0.0
    k = dict(fx=cal.fx, fy=cal.fy, cx_k=cal.cx, cy_k=cal.cy, use_opencv=True)
    pos, _ = _cloud_median_world(mask, depth, cal.rotation_matrix, cal.position, **k)
    if pos is None:
        return None, None, 0.0
    axis, ar, _, _ = _world_xy_pca_obb(mask, depth, cal.position, cal.rotation_matrix, **k)
    return np.asarray(pos[:2], dtype=float), axis, float(ar or 0.0)


def pointed_slot(executor, container_label, det):
    """Ask the planner for a free compartment and turn its point into a placement target.

    Returns ``{"xy", "axis_rad" (None = keep the earlier slot direction), "source", "pixel"}`` or
    ``None`` (the caller then uses the slot index).
    """
    try:
        return _pointed_slot(executor, container_label, det)
    except Exception as exc:  # noqa: BLE001 - pointing is optional; the index path is the fallback
        logger.warning("[point-slot] failed (%s); using the slot index", exc)
        return None


def _pointed_slot(executor, container_label, det):
    pipeline = executor._pipeline
    planner = getattr(pipeline, "_planner", None)
    cfg = settings(pipeline)
    cam = cfg["pointing_camera"]
    frame = (pipeline.capture() or {}).get(cam) or {}
    rgb, depth = frame.get("rgb"), frame.get("depth")
    if planner is None or rgb is None:
        logger.warning("[point-slot] no planner or no '%s' image; using the slot index", cam)
        return None
    cont = _container_mask(pipeline, cam, frame, container_label, det)
    if cont is None:
        logger.warning("[point-slot] '%s' not found on '%s'; using the slot index", container_label, cam)
        return None

    ys, xs = np.where(cont)
    m = int(cfg["pointing_crop_margin_px"])
    x0, y0 = max(int(xs.min()) - m, 0), max(int(ys.min()) - m, 0)
    x1, y1 = min(int(xs.max()) + m + 1, cont.shape[1]), min(int(ys.max()) + m + 1, cont.shape[0])
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    contours, _ = cv2.findContours(cont[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, contours, -1, (0, 255, 0), 2)

    held = getattr(executor, "_last_pick_label", "") or "an object"
    px = py = None
    hint = ""
    for _ in range(1 + int(cfg["pointing_max_reasks"])):
        pt = ask_slot_point(planner, crop, held, cfg, hint)
        if pt is None:
            hint = "Your last reply had no point. Reply with the JSON only."
            continue
        u, v = int(round(x0 + pt[0])), int(round(y0 + pt[1]))
        if 0 <= v < cont.shape[0] and 0 <= u < cont.shape[1] and cont[v, u]:
            px, py = u, v
            break
        hint = "Your last point was outside the green outline. Point inside the container."
    if px is None:
        logger.warning("[point-slot] no usable point for '%s'; using the slot index", container_label)
        run_timing.note("pointed_slot", container=container_label, ok=False)
        return None

    cal = frame.get("calibration")
    xy = axis = None
    source = "pixel"
    try:
        comp = _sam3_point(pipeline, rgb, depth, px, py, cam)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[point-slot] SAM3 point prompt failed (%s); using the pixel itself", exc)
        comp = None
    if comp is not None:
        n = int(comp.sum())
        if (cfg["pointing_min_mask_px"] <= n <= cfg["pointing_max_mask_frac"] * cont.sum()
                and (comp & cont).sum() >= cfg["pointing_min_inside_frac"] * n):
            xy, axis, ar = _world(comp, depth, cal)
            if xy is not None:
                source = "sam3_mask"
                if axis is None or ar < cfg["pointing_min_axis_ar"]:
                    axis = None
        else:
            logger.info("[point-slot] SAM3 mask of %d px refused (container %d px); using the pixel itself",
                        n, int(cont.sum()))
    if xy is None:
        disc = np.zeros(cont.shape, np.uint8)
        cv2.circle(disc, (px, py), int(cfg["pointing_pixel_radius_px"]), 1, -1)
        xy, _, _ = _world(disc, depth, cal)
        axis = None
    region = container_region(det)
    if xy is None or region is None or np.linalg.norm(xy - region[0]) > region[1]:
        logger.warning("[point-slot] point (%d, %d) gives no position inside '%s'; using the slot index",
                       px, py, container_label)
        run_timing.note("pointed_slot", container=container_label, ok=False, pixel=[px, py])
        return None
    logger.info("[point-slot] '%s': pixel (%d, %d) on '%s' -> xy=(%.3f, %.3f) axis=%s source=%s",
                container_label, px, py, cam, xy[0], xy[1],
                "kept" if axis is None else f"{np.rad2deg(axis):.1f} deg", source)
    run_timing.note("pointed_slot", container=container_label, ok=True, pixel=[px, py], source=source,
                    xy=[round(float(xy[0]), 3), round(float(xy[1]), 3)],
                    axis_deg=None if axis is None else round(float(np.rad2deg(axis)), 1))
    return {"xy": xy, "axis_rad": axis, "source": source, "pixel": (px, py)}
