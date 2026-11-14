"""Single Cartesian moves and the wait-for-arrival loop."""


from PIL import Image as PILImage
from scipy.spatial.transform import Rotation
from typing import Optional
import cv2
import logging
import numpy as np
import os
import time


from spark_real.calibration import compute_wrist_camera_extrinsic
from spark_real.config import load_table_plane
from spark_real.config import family_block
from spark_real.control.common import AbortRequested
from spark_real.control.common import ExecutionResult
from spark_real.control.common import _StillnessTracker
from spark_real.control.grasp_strategy import ApproachMixin
from spark_real.control.place import PlaceMixin
from spark_real.control.place import PlaceYawMixin
from spark_real.control.recovery import grasp_target_in_container
from spark_real.control.trajectory import UR_PROGRESS_REGISTER, decode_blend_progress, joint_move_seconds
from spark_real.perception.masks import intersect_ray_plane, pixel_ray_base


logger = logging.getLogger(__name__)


class MotionIOMixin:
    """Single Cartesian moves and the wait-for-arrival loop."""

    def _move_to(self, position, orientation, velocity=None):
        """
        Move with workspace protection; uses movej for large moves.
        """
        self._check_abort()
        vel = velocity or self.velocity
        target_raw = np.array(position, dtype=float)
        target = target_raw.copy()
        if not self._check_workspace(target):
            target = np.clip(target, self.WORKSPACE_MIN, self.WORKSPACE_MAX)
            logger.info(
                "Move target CLIPPED by workspace: requested (%.3f,%.3f,%.3f) "
                "-> commanded (%.3f,%.3f,%.3f)",
                target_raw[0],
                target_raw[1],
                target_raw[2],
                target[0],
                target[1],
                target[2],
            )

        pose = list(target) + list(orientation)
        current = self._get_current_position()
        dist = np.linalg.norm(target - current)
        logger.info(
            "MOVE_TO target=(%.3f,%.3f,%.3f) orient=(%.2f,%.2f,%.2f) "
            "from=(%.3f,%.3f,%.3f) dist=%.3fm v=%.2f",
            target[0],
            target[1],
            target[2],
            orientation[0],
            orientation[1],
            orientation[2],
            current[0],
            current[1],
            current[2],
            dist,
            vel,
        )
        if dist > 1.0:
            logger.warning("Move %.3fm too large, skipping", dist)
            return
        try:
            self._movej_via_ik(target, orientation, vel)
        except AbortRequested:
            raise
        except Exception as e:
            logger.warning("_movej_via_ik failed (%s); trying _movej_to_pose", e)
            try:
                # Full velocity: on UR10e _movej_to_pose IS the primary path
                # (controller firmware IK); halving it would make every UR
                # move crawl at ~0.375 rad/s.
                self._movej_to_pose(pose, vel)
            except Exception:
                raise

    def _issue_linear(self, pose, vel):
        """
        Dispatch a Cartesian move to whichever driver method exists.
        """
        if hasattr(self.robot, "move_to_pose"):
            self.robot.move_to_pose(pose, velocity=vel, wait=False)
        elif hasattr(self.robot, "move_linear"):
            try:
                self.robot.move_linear(pose, velocity=vel, asynchronous=True)
            except TypeError:
                self.robot.move_linear(pose, velocity=vel)

    def _move_to_linear(self, position, orientation, velocity=None):
        """
        Force linear (Cartesian) move to preserve TCP orientation.
        """
        self._check_abort()
        vel = velocity or self.velocity
        target = np.array(position)
        if not self._check_workspace(target):
            target = np.clip(target, self.WORKSPACE_MIN, self.WORKSPACE_MAX)
        pose = list(target) + list(orientation)
        dist = np.linalg.norm(target - self._get_current_position())
        if dist > 1.0:
            logger.warning("Move %.3fm too large, skipping", dist)
            return
        try:
            self._issue_linear(pose, vel)
            # Budget the wait to the move actually commanded. There is no
            # q_target on this path (the driver runs its own Cartesian IK), so
            # the only exits are proximity and stillness-after-motion, and a
            # movel short enough to finish inside the URScript start lag hits
            # neither. 3x the ideal duration plus the start lag and a second
            # of slack, capped at 15 s.
            ideal = dist / max(vel, 1e-3)
            budget = float(
                np.clip(
                    3.0 * ideal + self.URSCRIPT_START_LAG_S + 1.0,
                    self._min_motion_budget_s(),
                    15.0,
                )
            )
            arrived = self._wait_for_motion(target, timeout=budget)
            if arrived is False:
                # The arm never moved and is not at the target: the command
                # was dropped (dead RTDE control script, refused URScript).
                # Raising lets callers with a fallback (e.g. the tilt release)
                # take it instead of proceeding as if the pose was reached.
                raise RuntimeError(
                    f"linear move to {list(np.round(target, 3))} produced no "
                    "motion (command dropped)"
                )
        except AbortRequested:
            raise
        except Exception as e:
            if any(k in str(e).lower() for k in ("singular", "protective", "accel")):
                try:
                    self._movej_to_pose(pose, vel * 0.5)
                except Exception:
                    raise
            else:
                raise

    def _min_motion_budget_s(self) -> float:
        """Floor for any arrival budget, tied to the program start lag.

        A fast move has a small `ideal` and collapses onto the floor; a floor
        that does not clear the upload lag by a real margin times out before
        the arm has moved and reads as a motion failure.
        """
        return float(self.URSCRIPT_START_LAG_S + self.MOTION_BUDGET_SLACK_S)

    def _wait_for_motion(self, target, timeout=15.0, q_target=None):
        """
        Block until a fire-and-forget URScript movej lands.

        ``q_target`` is the authoritative signal: the controller is done when
        the measured joints equal the commanded ones. Cartesian proximity
        alone is not, because the commanded pose and the achieved TCP
        routinely differ by more than the 5 mm gate (IK rounding, TCP offset),
        and the stillness exit cannot fire on a move that finished inside the
        start lag.
        """
        t0 = time.time()
        motion = _StillnessTracker(
            self._get_current_position(),
            t0,
            sample_s=self.MOTION_SAMPLE_S,
            still_samples=self.MOTION_STILL_SAMPLES,
        )
        next_record = t0

        def _arrived():
            """Retire the driver's motion lease: this move is provably done."""
            clear = getattr(self.robot, "clear_motion_lease", None)
            if callable(clear):
                try:
                    clear()
                except Exception:  # noqa: BLE001 - never block on bookkeeping
                    pass

        while time.time() - t0 < timeout:
            self._check_abort()
            time.sleep(self.MOTION_POLL_S)
            now = time.time()
            tcp = self._get_current_position()
            # The recorder keeps its own ~10 Hz cadence: the faster poll exists
            # to cut arrival latency, not to change what gets logged.
            if self._recorder and now >= next_record:
                self._recorder.record()
                next_record = now + self.MOTION_SAMPLE_S
            # 1. Joints match the command: unambiguously arrived.
            if q_target is not None:
                try:
                    q = np.asarray(self.robot.get_joint_positions(), dtype=float)
                    if np.max(np.abs(q - np.asarray(q_target, dtype=float))) < (
                        self.JOINT_ARRIVE_TOL
                    ):
                        _arrived()
                        return True
                except Exception:  # noqa: BLE001 - fall through to the other tests
                    pass
            # 2. Cartesian proximity (kept: works when q_target is unavailable).
            if np.linalg.norm(tcp - target) < 0.005:
                _arrived()
                return True
            # 3. Stillness, past the start lag AND only after the arm was seen
            #    to move: if the URScript upload lag runs past
            #    URSCRIPT_START_LAG_S, "still" means "not started yet", not
            #    "arrived" (returned ~20 cm short, grasp closed above the
            #    object). The joint-target test above is exact and needs no
            #    motion history.
            motion.update(tcp, now)
            settled = motion.stopped() and now - t0 > self.URSCRIPT_START_LAG_S
            if settled and motion.ever_moved:
                _arrived()
                return True
        dist = float(np.linalg.norm(self._get_current_position() - target))
        logger.warning(
            "[_wait_for_motion] %.1fs timeout with no arrival signal " "(moved=%s, dist=%.3fm)",
            timeout,
            motion.ever_moved,
            dist,
        )
        # A timeout where the arm NEVER MOVED and is still away from the
        # target is a dropped command, not a slow one. False lets callers
        # distinguish "arm is basically there" from "nothing happened".
        return bool(motion.ever_moved or dist < 0.02)




class KeypointMixin:
    """move_to_keypoint and the orientation helpers it uses."""

    def current_orientation(self, fallback=None):
        """The orientation the arm is actually holding, or `fallback`.

        Translations must not re-pick a wrist angle: passing the fixed
        GRASP_ORIENTATION into a pure move UNWINDS whatever yaw the task had
        established.

        `obs.get("tcp_pose") or obs.get(...)` raises on a numpy array, so the
        None comparisons are explicit.
        """
        try:
            obs = self.robot.get_observation()
            tcp = obs.get("tcp_pose")
            if tcp is None:
                tcp = obs.get("tcp_pos")
            if tcp is not None and len(tcp) >= 6:
                return list(np.asarray(tcp[3:6], dtype=float))
        except Exception as exc:  # noqa: BLE001
            logger.warning("current_orientation: read failed (%s)", exc)
        return fallback if fallback is not None else list(self.GRASP_ORIENTATION)

    def _nearest_symmetric_yaw(self, yaw: float) -> float:
        """
        Pick yaw or yaw+-pi, whichever needs the least wrist travel.

        A two-jaw gripper grasps identically at yaw and yaw+pi, so <=90 deg
        of travel always suffices. Candidates stay inside MAX_YAW_OFFSET; on
        any read failure the canonical yaw passes through unchanged.
        """
        try:
            tcp = np.asarray(self.robot.get_tcp_pose(), dtype=float)
            R_cur = Rotation.from_rotvec(tcp[3:6])
            R_base = Rotation.from_rotvec(np.asarray(self.GRASP_ORIENTATION, dtype=float))
            rel = (R_cur * R_base.inv()).as_matrix()
            cur_yaw = float(np.arctan2(rel[1, 0], rel[0, 0]))
        except Exception:
            return yaw
        best = yaw
        for cand in (yaw - np.pi, yaw + np.pi):
            if abs(cand) <= self.MAX_YAW_OFFSET and abs(cand - cur_yaw) < abs(best - cur_yaw):
                best = cand
        if best != yaw:
            logger.info(
                "[yaw-symmetry] %.1f deg -> %.1f deg (current %.1f, travel "
                "%.1f instead of %.1f deg)",
                np.rad2deg(yaw),
                np.rad2deg(best),
                np.rad2deg(cur_yaw),
                np.rad2deg(abs(best - cur_yaw)),
                np.rad2deg(abs(yaw - cur_yaw)),
            )
        return best

    def _move_to_keypoint(self, params: dict, t0: float) -> ExecutionResult:
        self._check_abort()
        label = params.get("keypoint_label", "")
        self._last_keypoint_label = label
        self._pick_approach_skipped = False
        # offset_z default 0.0: move_to_keypoint takes the gripper TO the
        # keypoint so the following grasp closes on it (the grasp descent-retry
        # only spans ~7.5cm). The planner emits explicit offsets (0 picks,
        # 0.04+ places); this only governs plans/BTs that omit the field.
        offset = np.array(
            [
                params.get("offset_x", 0.0),
                params.get("offset_y", 0.0),
                params.get("offset_z", 0.0),
            ]
        )

        det = self.detection_map.get(label)
        if det is None:
            return ExecutionResult(
                action_type="move_to_keypoint",
                success=False,
                message=f"Object '{label}' not found in detections",
                duration=time.time() - t0,
            )

        # Hard guard for pick approaches: if this object already sits inside
        # the tray / container, skip the pick entirely (it is done). Only
        # applies when not holding so placement moves into the container are
        # never blocked.
        if not self._holding:
            try:
                _in = grasp_target_in_container(self, label)
            except Exception:
                _in = None
            # Only skip the pick if this run placed the object into the
            # container (a re-pick guard for stacking recovery). An object that
            # STARTED inside a container is not in _placed_labels and must be
            # picked normally.
            if _in is not None and label in self._placed_labels:
                self._placed_labels.add(label)
                self._pick_approach_skipped = True  # the grasp that follows has nothing to close on
                logger.info(
                    "move_to_keypoint '%s': already in container " "'%s', skipping pick approach",
                    label,
                    _in,
                )
                return ExecutionResult(
                    action_type="move_to_keypoint",
                    success=True,
                    message=f"'{label}' already in container '{_in}', " f"skipping",
                    duration=time.time() - t0,
                )

        pos = np.array(det["position_3d"])
        if pos[2] < self.TABLE_Z_FLOOR:
            logger.warning("Detection '%s' Z=%.3f below floor, clamping", label, pos[2])
            pos[2] = self.TABLE_Z_FLOOR
        target = pos + offset

        if self._holding:
            # PLACE. Recompute the Z from the container's MEASURED rim and
            # interior floor plus the held object's overhang; the plan's
            # offset_z survives as an upper bound, and an unmeasurable
            # container leaves it untouched (see control/place.py).
            # XY is never touched here.
            target[2] = self._place_release_z(float(target[2]), det, params)
            delivered = self._transport_to(
                target, target_detection=det, target_label=label, params=params
            )
            if not self._holding:
                return ExecutionResult(
                    action_type="move_to_keypoint",
                    success=False,
                    message=f"Object lost during transport to '{label}'",
                    duration=time.time() - t0,
                )
            if not delivered:
                # Still holding, but not over the target. A failed action:
                # executor_core runs recovery, and the reset path does NOT open
                # the gripper while it is still holding.
                record = getattr(self, "_place_arrival_record", None) or {}
                return ExecutionResult(
                    action_type="move_to_keypoint",
                    success=False,
                    message=(
                        f"Did not arrive over '{label}': "
                        f"{record.get('detail', 'transport stalled short')}"
                    ),
                    duration=time.time() - t0,
                )
        else:
            self._approach_target(target, detection=det, params=params)

        return ExecutionResult(
            action_type="move_to_keypoint",
            success=True,
            message=f"Moved to '{label}' at {target.tolist()}",
            duration=time.time() - t0,
        )

    def _oriented_grasp(self, yaw_offset: float = 0.0) -> list:
        """Post-rotate GRASP_ORIENTATION by Rz(yaw) about world Z.

        The closing direction ends up perpendicular to the OBB major axis.
        """
        clamped = np.clip(yaw_offset, -self.MAX_YAW_OFFSET, self.MAX_YAW_OFFSET)
        if abs(clamped - yaw_offset) > 0.01:
            logger.warning(
                "_oriented_grasp: yaw clamped from %.1f to %.1f deg",
                np.rad2deg(yaw_offset),
                np.rad2deg(clamped),
            )
        logger.info("_oriented_grasp: yaw_offset=%.1f deg", np.rad2deg(clamped))
        base = Rotation.from_rotvec(self.GRASP_ORIENTATION)
        yaw = Rotation.from_euler("z", clamped)
        return (yaw * base).as_rotvec().tolist()

    def _safe_clearance_z(self, start: np.ndarray, end: np.ndarray) -> float:
        """
        Safety lift 15 cm above the object/end Z.
        """
        base_z = end[2] + 0.15
        if not self.detection_map:
            return base_z
        max_obstacle_z = base_z
        for label, det in self.detection_map.items():
            pos = det.get("position_3d")
            if pos is None:
                continue
            obs = np.array(pos)
            path_dir = end[:2] - start[:2]
            path_len = np.linalg.norm(path_dir)
            if path_len < 0.01:
                continue
            path_unit = path_dir / path_len
            to_obs = obs[:2] - start[:2]
            proj = np.clip(np.dot(to_obs, path_unit), 0, path_len)
            closest = start[:2] + proj * path_unit
            dist_xy = np.linalg.norm(obs[:2] - closest)
            if dist_xy < self.OBSTACLE_XY_RADIUS:
                obstacle_top = obs[2] + self.CLEARANCE_MARGIN
                if obstacle_top > max_obstacle_z:
                    max_obstacle_z = obstacle_top
                    logger.info(
                        "Clearance: '%s' at z=%.3f, clearing at z=%.3f",
                        label,
                        obs[2],
                        obstacle_top,
                    )
        return max(max_obstacle_z, base_z)

    def _control_config(self) -> dict:
        """The resolved `control:` block (profile.raw preferred, else family YAML).

        Returns {} on any missing pipeline/config/file.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        if cfg is None:
            return {}
        family = (getattr(cfg, "robot_family", "ur10e") or "ur10e").lower()
        return family_block(getattr(self._pipeline, "profile", None), family, "control")




class BlendMixin:
    """Blended UR10e paths: config, progress register, motion lease and halt."""

    def _blend_cfg(self) -> dict:
        """Resolve the blended-path knobs once per executor.

        Precedence: env SPARK_UR_BLEND (on/off only) > control.blend_* in the
        resolved profile / family YAML > the class defaults above.

        Default ON for the transit chains (approach + transport). Safety:
          * abort: a blended path emits URScript stopj on the same socket and
            provably brakes (_halt_arm).
          * every row goes through the workspace clip (_clip_blend_waypoints)
            and the joint/singularity filter (_blend_rows_pass_safety_filter).
          * the terminal row is forced to r=0 and a descent row keeps
            BLEND_DESCENT_RADIUS_M = 2 cm, so an over-large radius cannot
            corner-cut into the object.
        PolyScope behaviour on an over-large blend radius (silent reduction vs
        runtime error) must be read off the pendant log on the first rig run.
        control.blend_motion: false (or SPARK_UR_BLEND=0) restores per-move
        dispatch.
        """
        cached = getattr(self, "_blend_cfg_cached", None)
        if cached is not None:
            return cached
        control = self._control_config()
        enabled = bool(control.get("blend_motion", self.BLEND_MOTION_DEFAULT))
        env = os.environ.get("SPARK_UR_BLEND")
        if env is not None:
            enabled = env not in ("0", "false", "False", "")
        cfg = {
            "enabled": enabled,
            "radius_m": float(control.get("blend_radius_m", self.BLEND_RADIUS_M)),
            "descent_radius_m": float(
                control.get("blend_descent_radius_m", self.BLEND_DESCENT_RADIUS_M)
            ),
            "min_radius_m": float(control.get("blend_min_radius_m", self.BLEND_MIN_RADIUS_M)),
            "max_rows": int(control.get("blend_max_rows", self.BLEND_MAX_ROWS)),
            "register": int(control.get("blend_progress_register", UR_PROGRESS_REGISTER)),
            # Fold a preceding move_relative into the transport phrase, i.e.
            # blend ACROSS a BT node boundary. Separate switch, default off;
            # see _execute_ur_blend_phrase.
            "cross_node": bool(control.get("blend_cross_node", False)),
        }
        env_x = os.environ.get("SPARK_UR_BLEND_CROSS_NODE")
        if env_x is not None:
            cfg["cross_node"] = env_x not in ("0", "false", "False", "")
        self._blend_cfg_cached = cfg
        logger.info(
            "Blended motion: %s (r=%.3fm, descent r=%.3fm, max_rows=%d, reg=%d, " "cross_node=%s)",
            "ON" if enabled else "off",
            cfg["radius_m"],
            cfg["descent_radius_m"],
            cfg["max_rows"],
            cfg["register"],
            cfg["cross_node"],
        )
        return cfg

    def _blend_enabled(self) -> bool:
        """True when transit chains may be issued as one blended path."""
        return bool(self._blend_cfg()["enabled"])

    def _blend_register(self) -> int:
        return self._blend_cfg()["register"]

    def _rtde_receive(self):
        """The rtde_receive handle, wherever it sits in the wrapper chain."""
        obj = self.robot
        for _ in range(4):
            if obj is None:
                return None
            r = getattr(obj, "_rtde_r", None)
            if r is not None:
                return r
            obj = getattr(obj, "_robot", None)
        return None

    def _read_blend_progress(self):
        """Current value of the blend progress register, or None if unreadable.

        A pure rtde_receive read: it uploads nothing, so unlike a gripper
        publish it can never replace the running program.
        """
        reg = self._blend_register()
        fn = getattr(self.robot, "get_output_int_register", None)
        if callable(fn):
            try:
                return int(fn(reg))
            except Exception:  # noqa: BLE001
                return None
        rtde_r = self._rtde_receive()
        if rtde_r is None:
            return None
        try:
            return int(rtde_r.getOutputIntRegister(reg))
        except Exception:  # noqa: BLE001 - build may not surface this register
            return None

    def _refresh_motion_lease(self):
        """Keep the driver's motion lease alive for the length of a path.

        _MOVEJ_LEASE_S is 1.5 s, sized for ONE move's upload lag. If the lease
        lapses mid-path a single _publish_gripper_state replaces the running
        program and cancels the whole path. Re-noting a movej pushes the lease
        out by another 1.5 s; _wait_for_blended_path retires it on arrival.
        """
        note = getattr(self.robot, "_note_motion_script", None)
        if callable(note):
            try:
                note("movej(")
            except Exception:  # noqa: BLE001 - never block on bookkeeping
                pass

    def _retire_motion_lease(self):
        clear = getattr(self.robot, "clear_motion_lease", None)
        if callable(clear):
            try:
                clear()
            except Exception:  # noqa: BLE001
                pass

    def _halt_arm(self) -> bool:
        """Decelerate the arm NOW. True only if the brake left this process.

        A new send REPLACES the running program, so a URScript stop both brakes
        the arm and kills a blended path. No hardcoded ``stopj(2.0)`` here:
        overshoot is ``v**2 / 2a``, so a FIXED decel gets quadratically worse
        as commanded velocity rises (at 1.78 rad/s, ``stopj(2.0)`` is 0.79 rad
        of travel AFTER the stop, roughly 0.8 m of TCP at full reach). The
        driver sizes its decel from the speed it last commanded
        (UR10eDriver.brake_decel).

        Order:
          1. the driver's own ``brake()``, the only branch that knows what
             velocity was commanded;
          2. the executor's ``_stop_robot()``, the shared escalation ladder
             (reopened socket, then Dashboard); its URScript branch emits a
             FIXED ABORT_DECEL_FALLBACK, hence (1) first;
          3. a literal stopj, for a bare MotionMixin with neither.
        """
        ok = False
        brake = getattr(self.robot, "brake", None)
        if callable(brake):
            try:
                ok = bool(brake())
            except Exception as exc:  # noqa: BLE001 - a stop never raises
                logger.warning("[blend] driver brake failed: %s", exc)
        if not ok:
            stop = getattr(self, "_stop_robot", None)
            if callable(stop):
                try:
                    ok = bool(stop())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[blend] brake escalation failed: %s", exc)
        if not ok:
            send = getattr(self.robot, "_send_script", None)
            if callable(send):
                try:
                    ok = bool(send("stopj(%.2f)" % self.BLEND_ABORT_DECEL))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[blend] stopj send failed: %s", exc)
        self._retire_motion_lease()
        return ok

    def _wait_for_blended_path(self, rows, epoch, start_pos=None, q_start=None) -> bool:
        """Block until a blended URScript path lands. True only on arrival.

        Arrival is the FINAL waypoint's joint target, nothing else.
        Intermediate waypoints are never reached (the arm cuts each corner at
        distance r), so a per-waypoint joint match would never fire and a
        per-waypoint Cartesian test would never come within 5 mm. The last row
        is forced to r=0 (waypoints.resolve_blend_radii) so the terminal
        waypoint is a true stop.

        Stillness is evaluated over the WHOLE path, never per waypoint, and
        may only mean "arrived" if motion has been observed; a path's upload
        lag is longer than one move's.
        """
        n = len(rows)
        final = rows[-1]
        target = final.position
        q_target = np.asarray(final.q, dtype=float)
        register_ok = self._read_blend_progress() is not None
        if not register_ok:
            logger.warning(
                "[blend] progress register %d unreadable: no mid-path stall "
                "detection, only the whole-path budget",
                self._blend_register(),
            )

        # Budget the WHOLE path, not one move: 15 s is both too long for a
        # 2 cm reposition and too short for a 5-row transport. Same shape as
        # _move_to_linear's per-move budget (3x ideal + start lag + slack).
        # Without q_start the first segment is budgeted off a half-turn rather
        # than under-budgeted.
        ideal = 0.0
        q_prev = None if q_start is None else np.asarray(q_start, dtype=float)
        for row in rows:
            dq = (
                float(np.max(np.abs(row.q - q_prev[: row.q.size]))) if q_prev is not None else np.pi
            )
            ideal += joint_move_seconds(dq, row.velocity, row.acceleration)
            q_prev = row.q
        timeout = float(
            np.clip(
                3.0 * ideal + self.URSCRIPT_START_LAG_S + 1.0,
                self._min_motion_budget_s(),
                self.BLEND_TIMEOUT_CAP_S,
            )
        )

        t0 = time.time()
        motion = _StillnessTracker(
            self._get_current_position(),
            t0,
            sample_s=self.MOTION_SAMPLE_S,
            still_samples=self.MOTION_STILL_SAMPLES,
        )
        next_record = t0
        progress = None
        progress_at = t0

        def _arrived(why: str) -> bool:
            self._retire_motion_lease()
            logger.info(
                "[blend] path complete after %.2fs (%s, %d rows)",
                time.time() - t0,
                why,
                n,
            )
            return True

        while time.time() - t0 < timeout:
            if self._abort:
                # Reach the arm BEFORE raising (see _halt_arm); otherwise an
                # aborted blended path keeps running to its last waypoint
                # after the executor has unwound.
                self._halt_arm()
            self._check_abort()
            time.sleep(self.MOTION_POLL_S)
            now = time.time()
            self._refresh_motion_lease()
            tcp = self._get_current_position()
            if self._recorder and now >= next_record:
                self._recorder.record()
                next_record = now + self.MOTION_SAMPLE_S

            # 1. Joints match the FINAL commanded row: unambiguously arrived.
            try:
                q = np.asarray(self.robot.get_joint_positions(), dtype=float)
                if np.max(np.abs(q[: q_target.size] - q_target)) < self.JOINT_ARRIVE_TOL:
                    return _arrived("joint target")
            except Exception:  # noqa: BLE001 - fall through to the other tests
                pass
            # 2. Cartesian proximity to the final waypoint.
            if np.linalg.norm(tcp - target) < 0.005:
                return _arrived("cartesian proximity")

            motion.update(tcp, now)

            # 3. Progress register. `n` means the program ran past its last
            #    movej, so the controller itself reports the path finished.
            #    Trustworthy only because _next_blend_epoch stepped off
            #    whatever value the register was already holding.
            k = decode_blend_progress(self._read_blend_progress(), epoch, n)
            if k is not None and (progress is None or k > progress):
                progress = k
                progress_at = now
                if k >= n:
                    return _arrived("progress register")
            # 4. Stall: started, still before the LAST row, register frozen,
            #    TCP still. Intermediate rows only: a stopped arm on the final
            #    row is the ordinary end of a move, handled by (5).
            if (
                progress is not None
                and progress < n - 1
                and now - progress_at > self.BLEND_STALL_S
                and motion.stopped()
            ):
                logger.warning(
                    "[blend] STALLED at row %d/%d after %.1fs (register frozen, "
                    "TCP still); stopping the path",
                    progress,
                    n,
                    now - t0,
                )
                self._halt_arm()
                return False
            # 5. Whole-path stillness, gated on observed motion (never an exit
            #    without ever_moved). Also refused while the register says the
            #    program is still inside its rows: a mid-path stop looks like a
            #    finished path from the outside. On the LAST row, or with a
            #    mute register, this is the per-move wait's test.
            settled = motion.stopped() and now - t0 > self.URSCRIPT_START_LAG_S
            if settled and motion.ever_moved and (progress is None or progress >= n - 1):
                return _arrived("settled after observed motion")

        logger.warning(
            "[blend] %.1fs timeout, no arrival signal (moved=%s, row=%s/%d, "
            "dist=%.3fm); stopping the path",
            timeout,
            motion.ever_moved,
            progress,
            n,
            float(np.linalg.norm(self._get_current_position() - target)),
        )
        self._halt_arm()
        return False


try:
    import torch
except ImportError:
    torch = None



class WristRefineMixin:
    """Wrist-camera refinement of a target position (depth or ray-plane)."""

    def _read_wrist_refine_mode_config(self) -> Optional[str]:
        """control.wrist_refine_mode from the profile, else the family YAML."""
        return self._control_config().get("wrist_refine_mode")

    def _wrist_refine_mode(self) -> str:
        """Resolve the wrist refinement mode: off | depth | ray_plane.

        ray_plane (default) needs RGB only, so it works with the wrist D435i
        opened color-only (streaming color+depth+IMU wedges the USB event
        ring on this rig). depth uses mask-median-depth backprojection. Cached
        per executor; SPARK_WRIST_REFINE_MODE overrides.
        """
        cached = getattr(self, "_wrist_refine_mode_cached", None)
        if cached is not None:
            return cached
        # _wrist_refine_mode_cfg is set by executor_core; fall back to reading
        # it here so this works standalone.
        mode = getattr(self, "_wrist_refine_mode_cfg", None)
        if mode is None:
            mode = self._read_wrist_refine_mode_config()
        mode = os.environ.get("SPARK_WRIST_REFINE_MODE") or mode
        mode = str(mode or self.WRIST_REFINE_MODE_DEFAULT).strip().lower()
        if mode not in self.WRIST_REFINE_MODES:
            logger.warning(
                "Unknown wrist_refine_mode %r; using %s",
                mode,
                self.WRIST_REFINE_MODE_DEFAULT,
            )
            mode = self.WRIST_REFINE_MODE_DEFAULT
        self._wrist_refine_mode_cached = mode
        logger.info("Wrist refine mode: %s", mode)
        return mode

    def _wrist_plane_z(
        self,
        coarse_target: np.ndarray,
        cam_z: float,
        z_plane: Optional[float] = None,
        tcp_z: Optional[float] = None,
    ) -> tuple:
        """Pick the horizontal plane the wrist ray is intersected with.

        Priority: explicit object-Z prior > the coarse target's own Z (the
        Kinect Z is reliable here; the wrist is wanted for lateral fixes) >
        the calibrated table surface > TABLE_Z_FLOOR (itself config-fed via
        control.table_z_floor).

        A prior is only taken if it clears WRIST_RAY_MIN_STANDOFF_M below BOTH
        the camera (a grazing ray runs away laterally) and the TCP (a "target"
        at the gripper's own height is not an object: search_keypoint passes
        its TCP waypoint as the coarse target, and the wrist cam sits above
        the TCP, so the camera test alone would wave it through).

        Returns (z, source).
        """
        coarse = np.asarray(coarse_target, dtype=float)
        for z, src in ((z_plane, "object_prior"), (coarse[2], "coarse_target")):
            if z is None:
                continue
            z = float(z)
            if cam_z - z <= self.WRIST_RAY_MIN_STANDOFF_M:
                continue
            if tcp_z is not None and tcp_z - z <= self.WRIST_RAY_MIN_STANDOFF_M:
                continue
            return z, src
        cal = load_table_plane(self._robot_family())
        if cal is not None:
            return float(cal["surface_z"]), "table_cal"
        return float(self.TABLE_Z_FLOOR), "table_z_floor"

    def _wrist_pos_from_depth(self, rs_cal, depth, mask, cx_det, cy_det, label):
        """Depth path: median depth inside the mask, backprojected."""
        if depth is None:
            logger.info(
                "Wrist refine: no depth stream (camera is color-only); "
                "use control.wrist_refine_mode: ray_plane"
            )
            return None, None
        mask_depths = depth[mask > 0]
        valid = mask_depths[(mask_depths > 0.01) & (mask_depths < 1.0)]
        if len(valid) < 5:
            logger.info("Wrist refine: no valid depth for '%s'", label)
            return None, None
        d_val = float(np.median(valid))
        x_cam = (cx_det - rs_cal.cx) * d_val / rs_cal.fx
        y_cam = (cy_det - rs_cal.cy) * d_val / rs_cal.fy
        pos = rs_cal.rotation_matrix @ np.array([x_cam, y_cam, d_val]) + rs_cal.position
        return pos, "depth=%.3f" % d_val

    def _wrist_pos_from_ray(
        self, rs_cal, cx_det, cy_det, coarse_target, z_plane, label, tcp_z=None
    ):
        """RGB-only path: mask-centroid ray x horizontal plane."""
        origin, direction = pixel_ray_base(cx_det, cy_det, rs_cal, rs_cal.extrinsic)
        plane_z, src = self._wrist_plane_z(coarse_target, float(origin[2]), z_plane, tcp_z)
        pos = intersect_ray_plane(origin, direction, plane_z)
        if pos is None:
            logger.info(
                "Wrist refine: ray for '%s' misses plane z=%.3f (%s) -- "
                "parallel or pointing away",
                label,
                plane_z,
                src,
            )
            return None, None
        return pos, "plane_z=%.3f (%s)" % (plane_z, src)

    def _refine_with_wrist(
        self,
        label: str,
        coarse_target: np.ndarray,
        z_plane: Optional[float] = None,
    ) -> Optional[dict]:
        """Capture wrist image, run SAM3, compute OBB orientation in tool frame.

        Returns a dict with refined position_3d, orientation_angle,
        aspect_ratio and the source mode, or None on failure. Gated off by
        default via control.wrist_refine; when off this returns None and never
        touches the camera. control.wrist_refine_mode picks how the 3D point
        is recovered (see _wrist_refine_mode). z_plane is the caller's
        object-height prior for ray_plane.
        """
        if not self._wrist_refine_enabled:
            return None
        if self._pipeline is None:
            return None
        mode = self._wrist_refine_mode()
        if mode == "off":
            return None
        rs = self._pipeline._realsense
        rs_cal = self._pipeline._realsense_cal
        perception = self._pipeline._perception
        if rs is None or rs_cal is None or perception is None:
            return None

        try:
            if torch is None:
                return None

            rgb, depth = rs.read()
            if rgb is None:
                return None

            tcp_pose = self._servo._get_tcp_pose()
            rs_cal.extrinsic = compute_wrist_camera_extrinsic(
                tcp_pose=tcp_pose,
                tool_offset=self._pipeline._wrist_tool_offset,
            )

            sam3 = perception._sam3
            if sam3 is None:
                return None
            pil_img = PILImage.fromarray(rgb)
            state = sam3.set_image(pil_img)
            state = sam3.set_text_prompt(prompt=label, state=state)
            masks = state.get("masks", torch.tensor([]))
            scores = state.get("scores", torch.tensor([]))
            if masks.numel() == 0:
                logger.info("Wrist refine: '%s' not found", label)
                return None

            # The instance nearest the binding, not the best-scored one: two spoons
            # may be in view. Masks far below the top score are noise, not candidates.
            floor = float(scores.max()) * self.WRIST_MIN_RELATIVE_SCORE
            best = None
            for m, score in zip(masks, scores):
                mask = m.cpu().numpy().squeeze()
                ys, xs = np.where(mask > 0)
                if float(score) < floor or len(xs) < 20:
                    continue
                cx_det, cy_det = float(xs.mean()), float(ys.mean())
                if mode == "depth":
                    pos, detail = self._wrist_pos_from_depth(
                        rs_cal, depth, mask, cx_det, cy_det, label
                    )
                else:
                    pos, detail = self._wrist_pos_from_ray(
                        rs_cal,
                        cx_det,
                        cy_det,
                        coarse_target,
                        z_plane,
                        label,
                        tcp_z=float(tcp_pose[2]),
                    )
                if pos is None:
                    continue
                dist = float(np.linalg.norm(np.asarray(pos)[:2] - np.asarray(coarse_target)[:2]))
                if best is None or dist < best[0]:
                    best = (dist, mask, pos, detail)
            if best is None:
                return None
            _, mask, pos_world, detail = best

            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            wrist_angle = 0.0
            aspect_ratio = 1.0
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) >= 5:
                    rect = cv2.minAreaRect(largest)
                    box = cv2.boxPoints(rect)
                    s1 = box[1] - box[0]
                    s2 = box[2] - box[1]
                    l1, l2 = np.linalg.norm(s1), np.linalg.norm(s2)
                    major = s1 if l1 >= l2 else s2
                    long, short = max(l1, l2), min(l1, l2)
                    if short > 0:
                        aspect_ratio = float(long / short)
                    wrist_angle = float(np.arctan2(major[1], major[0]))

            logger.info(
                "Wrist refine '%s' [%s]: img_angle=%.1f deg, ar=%.1f, %s, "
                "pos=(%.3f, %.3f, %.3f)",
                label,
                mode,
                np.rad2deg(wrist_angle),
                aspect_ratio,
                detail,
                pos_world[0],
                pos_world[1],
                pos_world[2],
            )
            return {
                "position_3d": pos_world,
                "orientation_angle": wrist_angle,
                "aspect_ratio": aspect_ratio,
                "source": mode,
            }
        except Exception as e:
            logger.warning("Wrist refine failed: %s", e)
            return None




class MotionMixin(
    KeypointMixin,
    WristRefineMixin,
    ApproachMixin,
    PlaceMixin,
    PlaceYawMixin,
    MotionIOMixin,
    BlendMixin,
):
    """
    Movement primitives mixed into ScoreExecutor.
    """

    # Hard wrist-3 cable limit, not a tuning knob: the wrist-camera USB cable
    # will not take more travel than this. Config: grasp.max_yaw_offset_deg
    # (env SPARK_MAX_YAW_OFFSET_DEG). Enforced on the COMPOSED rotation by
    # grasp_strategy.check_yaw_limit, not just on the requested angle.
    # 100, not 90: a two-jaw gripper is symmetric mod 180, and at a 90 cap the
    # nearer equivalent of a yaw near +-90 (e.g. +92.9 for -87.1) is refused
    # by a few degrees, so the wrist takes the LONGER path. 10 deg of headroom.
    MAX_YAW_OFFSET = np.deg2rad(100)
    CLEARANCE_MARGIN = 0.12
    OBSTACLE_XY_RADIUS = 0.12

    # How much of the place descent the Cartesian servo keeps (m). Everything
    # above this is flown as a movej/blend row; the servo owns every millimetre
    # where the held object can contact the container.
    PLACE_SERVO_STANDOFF_M = 0.02
    # Extra clearance above the first possible rim contact, covering the
    # held-extent estimate's own error. See _place_standoff_m.
    PLACE_CONTACT_MARGIN_M = 0.015
    # Ceiling on the slow phase. The servo crawls near convergence, so an
    # unbounded standoff would turn a bad held-extent number into a 10 s place.
    PLACE_SERVO_STANDOFF_MAX_M = 0.12

    WRIST_REFINE_MODES = ("off", "depth", "ray_plane")
    WRIST_REFINE_MODE_DEFAULT = "ray_plane"
    WRIST_MIN_RELATIVE_SCORE = 0.5
    # A plane prior nearer the camera than this is rejected: the ray grazes
    # the plane and the intersection runs away laterally.
    WRIST_RAY_MIN_STANDOFF_M = 0.03

    # How far OUTSIDE a container's own planar footprint the TCP may sit and
    # still count as having arrived over it. The ONLY tolerance the place path
    # judges arrival with (see _place_arrival); not the servo's pos_threshold
    # (3 mm), which says nothing about whether the object is over the container.
    # +2 cm mirrors the verifier's containment margin
    # (success_predicates.EvalConfig.inside_xy_margin_m = -0.02): anything
    # refused here is at least 4 cm outside a region that could have passed
    # verify, so the gate is never the reason a good task refuses to let go.
    PLACE_ARRIVAL_XY_MARGIN_M = 0.02


    # A shape fit must beat every orientation >=90 deg away by this much
    # before it is allowed to decide direction; below it the object is too
    # symmetric to call and the heavy-end test gets a turn.
    JIGSAW_MIN_MARGIN = 0.05
    JIGSAW_MIN_IOU = 0.20


    # Above this residual orientation error, the wrist is corrected at
    # clearance height before any contact-zone motion (5 deg: well above
    # servo convergence residuals ~0.06 deg, well below anything that sweeps
    # a held tool into a rim).
    PLACE_ORI_FIX_GATE_RAD = np.deg2rad(5.0)
    # Clearance above the place target for wrist rotations with a held
    # object: longer than the tool extent below the jaws (~10 cm screwdriver)
    # plus margin.
    ROTATE_CLEARANCE_M = 0.15


    # Joint tolerance counting a URScript movej as arrived (rad, per joint).
    JOINT_ARRIVE_TOL = 0.01
    # URScript upload/start lag on the primary socket: the arm is stationary for
    # up to ~1 s after the send, so stillness before this means "not started".
    URSCRIPT_START_LAG_S = 1.0
    # Arrival-poll cadence. The exact tests (joint target, Cartesian proximity)
    # are cheap register reads, so polling at 25 ms rather than 100 ms removes
    # up to 75 ms of dead time per program (~0.5 s over a pick-and-place).
    # MOTION_SAMPLE_S stays the window the displacement test is evaluated
    # over; see _StillnessTracker.
    MOTION_POLL_S = 0.025
    MOTION_SAMPLE_S = 0.1
    MOTION_STILL_SAMPLES = 3
    # Slack added on top of 3x the ideal move duration when budgeting a wait.
    MOTION_BUDGET_SLACK_S = 1.5

    # Blended-path defaults. Config: control.blend_motion / blend_radius_m /
    # blend_descent_radius_m / blend_min_radius_m / blend_max_rows; env
    # SPARK_UR_BLEND overrides the on/off switch. See _blend_cfg.
    BLEND_MOTION_DEFAULT = True
    BLEND_RADIUS_M = 0.05
    # A phrase that ends in a grasp/place descent keeps that descent
    # essentially vertical: 2 cm of corner-cut into a 10-15 cm descent, not 5.
    BLEND_DESCENT_RADIUS_M = 0.02
    BLEND_MIN_RADIUS_M = 0.01
    BLEND_MAX_ROWS = 8
    BLEND_TIMEOUT_CAP_S = 45.0
    # No progress-register advance AND a still TCP for this long, mid-path, is
    # the stall signature (vs a 15.0 s grind into a mechanical stop ending as
    # a silent full-path timeout).
    BLEND_STALL_S = 3.0
    # Last-resort brake decel (rad/s^2) for a driver with no velocity-sized
    # brake() of its own. Only reached by bare-mixin test doubles; the real
    # driver computes this from the speed it last commanded.
    BLEND_ABORT_DECEL = 4.0
