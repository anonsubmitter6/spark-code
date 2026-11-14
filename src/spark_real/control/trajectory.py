"""Trajectory recording and visualization for score execution."""


from __future__ import annotations


from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import List, Optional, Sequence
import logging
import math
import matplotlib
import numpy as np
import os
import time


from spark_real.control.common import ExecutionResult
from spark_real.control.common import SOURCE_SERVO, command_source


logger = logging.getLogger(__name__)

# Why one URScript program instead of rtde_c.moveJ(path):
#   * every move on this rig already goes out over the raw URScript socket
#     (ur10e_driver._send_script, port 30002) because rtde_c.moveJ silently
#     no-ops once a speedl / gripper script has stopped its control script,
#     and _send_script itself calls rtde_c.stopScript() before every move.
#   * consecutive _send_script("movej(...)") calls do NOT blend: each send
#     REPLACES the running program, so send #2 cancels move #1 mid-flight.
#     Blending therefore requires all rows inside ONE program in ONE send.
#     That shape is already proven here by the gripper publisher, which
#     uploads a multi-line def/end block plus its call in a single send.
#
# Emitted shape (last row always r=0, so the terminal waypoint is a true
# stop and the executor's exact joint-arrival test still applies):
#
#   def spark_blend_path():
#     write_output_integer_register(15, 1000)
#     movej([...],a=1.4,v=1.05,r=0.05)
#     write_output_integer_register(15, 1001)
#     movej([...],a=1.4,v=1.05,r=0)
#     write_output_integer_register(15, 1002)
#   end
#   spark_blend_path()
#
# The register writes are a mid-path progress signal. They are inside the
# same program, so they cannot cancel anything (unlike a fresh send), and
# they need no control script. Without them a stall inside a multi-second
# path would only surface as a silent full-path timeout (the measured 15.0 s /
# 1817-speedl grind into a mechanical stop). "Register not advancing AND TCP
# still" is the stall signature.

# 12/13/14 belong to the gripper publisher (_POS/_OBJ/_SEQ_REGISTER).
UR_PROGRESS_REGISTER = 15
# value = epoch * STRIDE + row_index, so a register left over from the
# PREVIOUS path can never be mistaken for progress on this one.
UR_PROGRESS_STRIDE = 1000
UR_MAX_EPOCH = 200
UR_BLEND_FUNC = "spark_blend_path"

# A junction blend radius may not exceed this fraction of either adjoining
# segment. r_i <= 0.45*min(L_i, L_i+1) also guarantees non-overlap of
# consecutive blend regions (r_i + r_i+1 <= 0.9 * L_i+1). PolyScope does no
# validation (rtde_control.Path.toScriptCode() accepted 0.05 blends on
# ~0.1 rad segments), so the policing is here.
UR_BLEND_SEG_FRAC = 0.45
# Below this a blend is not worth the corner-cut; emit a full stop instead.
UR_BLEND_MIN_M = 0.01
# Two rows whose joint targets agree this closely are the same pose: the row
# is a no-op and is dropped rather than blended (this is _transport_to's
# first lift, which is ~0 m long right after a move_relative(dz=+0.20)).
UR_ROW_MERGE_RAD = 0.01
# Reject a phrase whose IK jumps this far on one row: an elbow/wrist branch
# flip mid-path becomes one continuous blended wrench instead of a discrete,
# visible reconfiguration between two stopped moves.
UR_MAX_ROW_DQ_RAD = 2.6


@dataclass
class JointRow:
    """One ``movej`` row of a blended URScript path.

    Attributes:
        q: (6,) joint target [rad].
        position: (3,) TCP position this row commands, base frame [m]. Only
            used to size blend radii (``r`` is metres of TCP path even for
            movej) and to budget the wait; never sent.
        velocity: joint velocity [rad/s] -> ``v=``.
        acceleration: joint acceleration [rad/s^2] -> ``a=``.
        blend: blend radius [m] -> ``r=``. 0 means decelerate to a stop.
        label: free text for logs.
    """

    q: np.ndarray
    position: np.ndarray
    velocity: float
    acceleration: float
    blend: float = 0.0
    label: str = ""

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=float).reshape(-1)
        self.position = np.asarray(self.position, dtype=float).reshape(3)


def _fmt_num(x: float) -> str:
    """Format like ur_rtde's own PathEntry emitter (C++ ostream default: %g)."""
    return f"{float(x):g}"


def format_movej_row(q: Sequence[float], velocity: float, acceleration: float, blend: float) -> str:
    """One ``movej([...],a=,v=,r=)`` line.

    Byte-identical to ``rtde_control.PathEntry(MoveJ, PositionJoints,
    [*q, v, a, r]).toScriptCode()`` minus its leading tab (asserted by
    tests/test_blend_path.py), so the wire contract is ur_rtde's.
    """
    qs = ",".join(_fmt_num(v) for v in list(q)[:6])
    return (
        f"movej([{qs}],a={_fmt_num(acceleration)}," f"v={_fmt_num(velocity)},r={_fmt_num(blend)})"
    )


def resolve_blend_radii(
    start_position: Sequence[float],
    rows: List[JointRow],
    radius_m: float,
    final_radius_m: float = 0.0,
    min_radius_m: float = UR_BLEND_MIN_M,
    merge_rad: float = UR_ROW_MERGE_RAD,
    start_q: Optional[Sequence[float]] = None,
) -> tuple:
    """Drop no-op rows and size every junction blend. Mutates ``rows``' blends.

    ``radius_m`` is the requested junction radius; each junction is clamped to
    ``UR_BLEND_SEG_FRAC`` of BOTH adjoining segments, and anything below
    ``min_radius_m`` becomes a full stop. The LAST row is always r=0;
    ur_rtde's own emitter does the same, and the executor's arrival test
    depends on that terminal waypoint being a true stop.

    ``final_radius_m`` caps the junction leading INTO the last row separately,
    so a phrase that ends in a grasp descent can keep that descent essentially
    vertical (a 5 cm corner-cut into a 15 cm descent is not wanted) while the
    transit junctions before it stay generous.

    Returns (kept_rows, dropped_labels).
    """
    kept: List[JointRow] = []
    dropped: List[str] = []
    prev_q = None if start_q is None else np.asarray(start_q, dtype=float).reshape(-1)
    for i, row in enumerate(rows):
        last = i == len(rows) - 1
        if (
            prev_q is not None
            and not last
            and prev_q.size >= row.q.size
            and float(np.max(np.abs(row.q - prev_q[: row.q.size]))) < merge_rad
        ):
            # Same pose as the current or previous target: a no-op row.
            dropped.append(row.label or f"row{i}")
            continue
        kept.append(row)
        prev_q = row.q
    if not kept:
        return kept, dropped

    pts = [np.asarray(start_position, dtype=float).reshape(3)] + [r.position for r in kept]
    seg = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(kept))]
    for i, row in enumerate(kept):
        if i == len(kept) - 1:
            row.blend = 0.0  # terminal waypoint is a true stop
            continue
        want = radius_m
        if i == len(kept) - 2:
            want = min(want, final_radius_m) if final_radius_m > 0 else want
        r = min(want, UR_BLEND_SEG_FRAC * seg[i], UR_BLEND_SEG_FRAC * seg[i + 1])
        row.blend = float(r) if r >= min_radius_m else 0.0
    return kept, dropped


def build_ur_blend_program(
    rows: List[JointRow],
    epoch: int,
    progress_register: int = UR_PROGRESS_REGISTER,
    func_name: str = UR_BLEND_FUNC,
) -> str:
    """Wrap ``rows`` into one uploadable URScript program.

    ``epoch`` namespaces the progress register so a value left by the previous
    path is not read as progress on this one. Row k is *entered* when the
    register reads ``epoch*STRIDE + k``; ``+ len(rows)`` means the program ran
    to the end.
    """
    if not rows:
        raise ValueError("build_ur_blend_program: no rows")
    base = int(epoch) * UR_PROGRESS_STRIDE
    lines = [f"def {func_name}():"]
    for k, row in enumerate(rows):
        if progress_register is not None:
            lines.append(f"  write_output_integer_register({progress_register}, {base + k})")
        lines.append("  " + format_movej_row(row.q, row.velocity, row.acceleration, row.blend))
    if progress_register is not None:
        lines.append(f"  write_output_integer_register({progress_register}, {base + len(rows)})")
    lines.append("end")
    lines.append(f"{func_name}()")
    return "\n".join(lines) + "\n"


def blend_progress_value(epoch: int, row_index: int) -> int:
    """The register value written on entering ``row_index`` of ``epoch``."""
    return int(epoch) * UR_PROGRESS_STRIDE + int(row_index)


def decode_blend_progress(value, epoch: int, n_rows: int):
    """Row index from a register read, or None if it is not this path's epoch."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    base = int(epoch) * UR_PROGRESS_STRIDE
    if base <= v <= base + n_rows:
        return v - base
    return None


def joint_move_seconds(dq_max: float, velocity: float, acceleration: float) -> float:
    """Trapezoidal duration of one movej, from the leading joint's travel.

    Used only to BUDGET the wait (3x this plus the start lag), never to
    decide arrival.
    """
    d = abs(float(dq_max))
    v = max(float(velocity), 1e-6)
    a = max(float(acceleration), 1e-6)
    if d <= v * v / a:
        return 2.0 * math.sqrt(d / a)
    return d / v + v / a


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from spark_real.config import output_root  # noqa: E402


OUTPUT_DIR = output_root() / "trajectories"


class TrajectoryRecorder:
    """
    Records robot state during execution and generates plots.
    """

    def __init__(self, robot):
        self.robot = robot
        self.start_time = time.time()
        self.data = {
            "timestamps": [],
            "tcp_poses": [],
            "joint_positions": [],
            "gripper_positions": [],
            "action_labels": [],
            "action_transitions": [],
        }
        self._current_label = "idle"

    def set_action_label(self, label: str):
        self._current_label = label

    def mark_transition(self, label: str):
        t = time.time() - self.start_time
        self.data["action_transitions"].append((t, label))

    def record(self):
        """
        Sample current robot state (~10Hz from motion wait loops).
        """
        try:
            obs = self.robot.get_observation()
            tcp = np.array(obs["tcp_pose"][:3])
            joints = np.array(obs["joint_positions"][:6])
            gripper = float(obs.get("gripper_position", 0.0))
            t = time.time() - self.start_time

            self.data["timestamps"].append(t)
            self.data["tcp_poses"].append(tcp.tolist())
            self.data["joint_positions"].append(joints.tolist())
            self.data["gripper_positions"].append(gripper)
            self.data["action_labels"].append(self._current_label)
        except Exception as e:
            logger.debug("Trajectory record skipped: %s", e)

    def save(self):
        """
        Save trajectory data (.npz) and plot (.png).
        """
        traj = self.data
        if len(traj["timestamps"]) < 2:
            logger.info(
                "Trajectory too short to save (%d samples)", len(traj["timestamps"])
            )
            return

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_DIR / ts_str
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamps = np.array(traj["timestamps"])
        tcp_poses = np.array(traj["tcp_poses"])
        joint_positions = np.array(traj["joint_positions"])
        gripper_positions = np.array(traj["gripper_positions"])
        action_labels = np.array(traj["action_labels"], dtype=object)

        npz_path = out_dir / "trajectory.npz"
        np.savez(
            npz_path,
            timestamps=timestamps,
            tcp_poses=tcp_poses,
            joint_positions=joint_positions,
            gripper_positions=gripper_positions,
            action_labels=action_labels,
        )
        logger.info(
            "Trajectory saved: %s (%d samples, %.1fs)",
            npz_path,
            len(timestamps),
            timestamps[-1],
        )

        try:
            _save_plot(out_dir, traj)
        except Exception as e:
            logger.warning("Failed to save trajectory plot: %s", e)


def _ema_smooth(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Apply exponential moving average smoothing.
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def _padded_limits(data: np.ndarray, min_range: float = 0.05) -> tuple:
    """
    Axis limits with minimum range to avoid over-zooming on noise.
    """
    dmin, dmax = float(data.min()), float(data.max())
    drange = dmax - dmin
    if drange < min_range:
        center = (dmin + dmax) / 2
        return center - min_range / 2, center + min_range / 2
    padding = drange * 0.1
    return dmin - padding, dmax + padding


def _save_plot(out_dir: Path, trajectory: dict):
    """
    Save 4-subplot trajectory visualization.
    """
    tcp_poses = np.array(trajectory["tcp_poses"])
    timestamps = np.array(trajectory["timestamps"])
    gripper_positions = np.array(trajectory["gripper_positions"])
    transitions = trajectory.get("action_transitions", [])

    if len(tcp_poses) < 2:
        return

    x_raw, y_raw, z_raw = tcp_poses[:, 0], tcp_poses[:, 1], tcp_poses[:, 2]
    x = _ema_smooth(x_raw)
    y = _ema_smooth(y_raw)
    z = _ema_smooth(z_raw)

    x_lim = _padded_limits(x)
    y_lim = _padded_limits(y)
    z_lim = _padded_limits(z)

    trans_idx = [
        (int(np.argmin(np.abs(timestamps - t))), label) for t, label in transitions
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(trans_idx), 1)))

    fig = plt.figure(figsize=(16, 10))

    # 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(x_raw, y_raw, z_raw, "gray", alpha=0.2, linewidth=0.5)
    scatter = ax1.scatter(x, y, z, c=timestamps, cmap="viridis", s=20)
    ax1.plot(x, y, z, "b-", alpha=0.5, linewidth=1.5)
    ax1.scatter([x[0]], [y[0]], [z[0]], c="green", s=100, marker="o", label="Start")
    ax1.scatter([x[-1]], [y[-1]], [z[-1]], c="red", s=100, marker="x", label="End")
    for ci, (idx, label) in enumerate(trans_idx):
        ax1.scatter(
            [x[idx]],
            [y[idx]],
            [z[idx]],
            color=colors[ci],
            s=120,
            marker="D",
            edgecolors="black",
            linewidths=0.5,
        )
        ax1.text(x[idx], y[idx], z[idx], f" {label}", fontsize=6, color=colors[ci])
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.set_zlim(z_lim)
    ax1.set_title("3D TCP Trajectory (EMA smoothed)")
    ax1.legend(loc="upper left", fontsize=7)
    plt.colorbar(scatter, ax=ax1, label="Time (s)", shrink=0.6)

    # XY top-down
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(x_raw, y_raw, "gray", alpha=0.3, linewidth=0.5)
    ax2.scatter(x, y, c=timestamps, cmap="viridis", s=20)
    ax2.plot(x, y, "b-", alpha=0.5, linewidth=1.5)
    ax2.scatter([x[0]], [y[0]], c="green", s=100, marker="o", label="Start")
    ax2.scatter([x[-1]], [y[-1]], c="red", s=100, marker="x", label="End")
    for ci, (idx, label) in enumerate(trans_idx):
        ax2.scatter(
            [x[idx]],
            [y[idx]],
            color=colors[ci],
            s=80,
            marker="D",
            edgecolors="black",
            linewidths=0.5,
        )
        ax2.annotate(
            label,
            (x[idx], y[idx]),
            fontsize=6,
            color=colors[ci],
            textcoords="offset points",
            xytext=(5, 5),
        )
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.set_title("Top View (XY)")
    ax2.legend(fontsize=7)
    ax2.ticklabel_format(useOffset=False, style="plain")

    # XZ side view
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x_raw, z_raw, "gray", alpha=0.3, linewidth=0.5)
    ax3.scatter(x, z, c=timestamps, cmap="viridis", s=20)
    ax3.plot(x, z, "b-", alpha=0.5, linewidth=1.5)
    ax3.scatter([x[0]], [z[0]], c="green", s=100, marker="o", label="Start")
    ax3.scatter([x[-1]], [z[-1]], c="red", s=100, marker="x", label="End")
    for ci, (idx, label) in enumerate(trans_idx):
        ax3.scatter(
            [x[idx]],
            [z[idx]],
            color=colors[ci],
            s=80,
            marker="D",
            edgecolors="black",
            linewidths=0.5,
        )
        ax3.annotate(
            label,
            (x[idx], z[idx]),
            fontsize=6,
            color=colors[ci],
            textcoords="offset points",
            xytext=(5, 5),
        )
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_xlim(x_lim)
    ax3.set_ylim(z_lim)
    ax3.set_title("Side View (XZ)")
    ax3.legend(fontsize=7)
    ax3.ticklabel_format(useOffset=False, style="plain")

    # Gripper timeline
    ax4 = fig.add_subplot(2, 2, 4)
    grip_pct = np.array(gripper_positions) * 100
    ax4.plot(timestamps, grip_pct, "g-", linewidth=2)
    ax4.fill_between(timestamps, 0, grip_pct, alpha=0.3)
    for ci, (t_trans, label) in enumerate(transitions):
        ax4.axvline(x=t_trans, color=colors[ci], linestyle="--", alpha=0.7, linewidth=1)
        ax4.text(
            t_trans,
            102,
            label,
            fontsize=5,
            rotation=45,
            ha="left",
            va="bottom",
            color=colors[ci],
        )
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Gripper (%)")
    ax4.set_title("Gripper Position Over Time")
    ax4.set_ylim(-5, 115)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "trajectory_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Trajectory plot saved: %s", plot_path)




class CartesianServo:
    """
    Velocity-based Cartesian servo for UR10e via speedl.
    """

    # Slowest commanded linear approach while outside pos_threshold (m/s).
    MIN_APPROACH_VEL_LINEAR = 0.03

    def __init__(self, robot, rate_hz: float = 125.0):
        self.robot = robot
        # Drivers without URScript are capped at 30 Hz; UR's URScript speedl
        # tolerates the higher rate.
        if not getattr(robot, "SUPPORTS_URSCRIPT", False):
            rate_hz = min(rate_hz, 30.0)
        self.dt = 1.0 / rate_hz

        # PD gains kept conservative: near a wrist singularity small Cartesian
        # velocities map to large joint velocities.
        self.kp_pos = 2.0
        self.kp_ori = 1.5
        self.kd_pos = 0.5
        self.kd_ori = 0.4

        # Conservative linear/angular caps for stable PD servoing.
        self.max_vel_linear = 0.15  # m/s
        self.max_vel_angular = 0.6  # rad/s
        # Slowest the arm may approach while still outside pos_threshold. See
        # _compute_velocity: without this the proportional law crawls the last
        # centimetres. Small enough to stop inside one tick of the arrival
        # tolerance (0.03 m/s * 8 ms = 0.24 mm, vs a 3 mm threshold).
        self.min_vel_linear = self.MIN_APPROACH_VEL_LINEAR
        self.accel = 1.0  # m/s^2 for speedl

        # Convergence thresholds
        self.pos_threshold = 0.003  # 3mm
        # ~3 deg. The speedl angular PD has a steady-state yaw error above
        # 1 deg on oriented grasps, so a tighter tolerance never converges
        # and the servo runs the full timeout.
        self.ori_threshold = 0.05
        self.settle_time = 0.15  # dwell at target before exit

        # Stall detection. A target a few mm past a mechanical stop (object
        # bottomed out on a rim, target Z estimate slightly low) never
        # converges: the PD holds a small steady command into the stop for the
        # whole timeout. Measured offline: 4 mm of unreachability turned a
        # 3.8 s descent into 15.01 s / 1877 consecutive speedl commands. If
        # neither error has improved by a meaningful margin for this long,
        # more time will not help.
        self.stall_window = 1.2  # s without progress before giving up
        self.stall_pos_eps = 0.0005  # m of improvement that counts as progress
        self.stall_ori_eps = 0.005  # rad, ditto

        self._abort = False
        self._prev_vel = np.zeros(6)
        # Diagnostics for the caller / logs, set by every move_to_pose call.
        self.last_pos_err = float("nan")
        self.last_ori_err = float("nan")
        self.last_exit = "none"

    def abort(self):
        self._abort = True

    def _get_tcp_pose(self) -> np.ndarray:
        """
        Read current TCP pose [x, y, z, rx, ry, rz].
        """
        if hasattr(self.robot, "get_tcp_pose"):
            p = self.robot.get_tcp_pose()
            if isinstance(p, np.ndarray) and p.shape == (4, 4):
                pos = p[:3, 3]
                rotvec = Rotation.from_matrix(p[:3, :3]).as_rotvec()
                return np.concatenate([pos, rotvec])
            return np.array(p[:6])
        # No zeros fallback: that would make the servo chase from the origin.
        raise AttributeError(
            "CartesianServo needs a robot with get_tcp_pose; got "
            f"{type(self.robot).__name__}."
        )

    def _send_speedl(self, velocity: np.ndarray):
        """
        Send a base-frame Cartesian velocity command.

        Family-aware: UR robots get URScript ``speedl`` via _send_script;
        other drivers take the same 6-vector through their send_velocity
        shim. Dispatch is gated on the SUPPORTS_URSCRIPT flag,
        not ``hasattr(_send_script)``: the SafeRobot wrapper defines
        ``_send_script`` as a method that always raises for non-UR drivers.
        """
        v = velocity.tolist()
        with command_source(SOURCE_SERVO):
            if getattr(self.robot, "SUPPORTS_URSCRIPT", False) and hasattr(
                self.robot, "_send_script"
            ):
                # The URScript branch is kept separate from send_velocity on
                # purpose: on UR the driver is wrapped in SafeRobot, whose
                # send_velocity runs the CBF-QP while _send_script passes
                # speedl through untouched. The commanded twist is latched
                # explicitly so there is still exactly one latch object.
                latch = getattr(self.robot, "latch_command_velocity", None)
                if callable(latch):
                    latch(v)
                script = (
                    f"speedl([{v[0]:.5f}, {v[1]:.5f}, {v[2]:.5f}, "
                    f"{v[3]:.5f}, {v[4]:.5f}, {v[5]:.5f}], "
                    f"{self.accel}, {self.dt + 0.02})"
                )
                self.robot._send_script(script)
            elif hasattr(self.robot, "send_velocity"):
                try:
                    self.robot.send_velocity(
                        v, acceleration=self.accel, duration=self.dt + 0.02
                    )
                except TypeError:
                    # Legacy positional / time_duration variant
                    self.robot.send_velocity(v, self.accel, self.dt + 0.02)

    def _stop(self):
        """
        Decelerate to zero. Family-aware; see _send_speedl.
        """
        # Zero is a command too: without this the recorder would keep
        # reporting the last non-zero twist through the whole settle dwell.
        latch = getattr(self.robot, "latch_command_velocity", None)
        if callable(latch):
            latch([0.0] * 6, SOURCE_SERVO)
        if getattr(self.robot, "SUPPORTS_URSCRIPT", False) and hasattr(
            self.robot, "_send_script"
        ):
            self.robot._send_script("speedl([0,0,0,0,0,0], 1.0, 0.1)\nstopl(1.0)")
        elif hasattr(self.robot, "stop_velocity"):
            try:
                self.robot.stop_velocity()
            except Exception:
                pass
        elif hasattr(self.robot, "send_velocity"):
            try:
                self.robot.send_velocity(
                    [0, 0, 0, 0, 0, 0], acceleration=1.0, duration=0.1
                )
            except TypeError:
                self.robot.send_velocity([0, 0, 0, 0, 0, 0], 1.0, 0.1)

    def _compute_velocity(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute 6D Cartesian velocity command from pose error.
        """
        pos_err = target[:3] - current[:3]

        # Orientation error as compact rotvec
        R_cur = Rotation.from_rotvec(current[3:6])
        R_tgt = Rotation.from_rotvec(target[3:6])
        R_err = R_tgt * R_cur.inv()
        ori_err = R_err.as_rotvec()

        # PD control (damping on velocity change)
        v_pos = self.kp_pos * pos_err - self.kd_pos * self._prev_vel[:3]
        v_ori = self.kp_ori * ori_err - self.kd_ori * self._prev_vel[3:]

        vel = np.concatenate([v_pos, v_ori])

        # Clamp linear and angular separately
        lin_norm = np.linalg.norm(vel[:3])
        if lin_norm > self.max_vel_linear:
            vel[:3] *= self.max_vel_linear / lin_norm
        ang_norm = np.linalg.norm(vel[3:])
        if ang_norm > self.max_vel_angular:
            vel[3:] *= self.max_vel_angular / ang_norm

        # Floor the approach speed while still OUTSIDE the arrival tolerance.
        # v = kp * err decays exponentially, so the tail dominates the move.
        # Measured descending into the bowl: 12.7 cm in 0.4 s, then 2.2 cm in
        # 2.2 s, ending at 0.001 m/s. The floor applies only outside
        # pos_threshold; flooring inside it would command motion at the target
        # and buzz around it forever.
        pos_dist = float(np.linalg.norm(pos_err))
        if pos_dist > self.pos_threshold:
            lin_norm = float(np.linalg.norm(vel[:3]))
            floor = min(self.min_vel_linear, self.max_vel_linear)
            if 0.0 < lin_norm < floor:
                vel[:3] *= floor / lin_norm
            elif lin_norm <= 0.0:
                # PD damping cancelled the command outright while still short
                # of the target: drive straight at it rather than stall.
                vel[:3] = pos_err / max(pos_dist, 1e-9) * floor

        return vel

    def move_to_pose(
        self, target_pose: list, velocity: float = 0.15, timeout: float = 15.0
    ) -> bool:
        """
        Drive TCP to target pose using Cartesian velocity servo.

        Args:
            target_pose: [x, y, z, rx, ry, rz] target in axis-angle
            velocity: Max linear velocity scale (adjusts kp)
            timeout: Abort after this many seconds

        Returns:
            True if converged, False if timed out or aborted
        """
        self._abort = False
        self._prev_vel = np.zeros(6)
        target = np.array(target_pose, dtype=np.float64)

        # `velocity` reaches the loop only through the caller's max_vel_linear
        # clamp (see executor_motion._servo_to), not through the gains.

        t0 = time.time()
        settle_start = None
        best_pos = np.inf
        best_ori = np.inf
        last_progress = t0
        pos_err = np.inf
        ori_err = np.inf

        while time.time() - t0 < timeout:
            if self._abort:
                self._stop()
                self.last_exit = "abort"
                return False

            current = self._get_tcp_pose()

            pos_err = np.linalg.norm(target[:3] - current[:3])
            R_cur = Rotation.from_rotvec(current[3:6])
            R_tgt = Rotation.from_rotvec(target[3:6])
            ori_err = np.linalg.norm((R_tgt * R_cur.inv()).as_rotvec())

            if pos_err < self.pos_threshold and ori_err < self.ori_threshold:
                if settle_start is None:
                    settle_start = time.time()
                elif time.time() - settle_start > self.settle_time:
                    self._stop()
                    self._finish("converged", pos_err, ori_err)
                    logger.info(
                        "Servo converged: pos_err=%.4f ori_err=%.4f (%.2fs)",
                        pos_err,
                        ori_err,
                        time.time() - t0,
                    )
                    return True
            else:
                settle_start = None
                # Progress on EITHER channel resets the clock: a descent that
                # has reached its Z but is still rotating is making progress.
                now = time.time()
                if pos_err < best_pos - self.stall_pos_eps:
                    best_pos = pos_err
                    last_progress = now
                if ori_err < best_ori - self.stall_ori_eps:
                    best_ori = ori_err
                    last_progress = now
                if now - last_progress > self.stall_window:
                    self._stop()
                    self._finish("stalled", pos_err, ori_err)
                    logger.warning(
                        "Servo STALLED: no progress for %.1fs at pos_err=%.4f "
                        "ori_err=%.4f (%.2fs elapsed). Target is unreachable "
                        "-- blocked, or past a mechanical stop.",
                        self.stall_window,
                        pos_err,
                        ori_err,
                        now - t0,
                    )
                    return False

            vel = self._compute_velocity(current, target)
            self._send_speedl(vel)
            self._prev_vel = vel

            time.sleep(self.dt)

        self._stop()
        current = self._get_tcp_pose()
        pos_err = np.linalg.norm(target[:3] - current[:3])
        self._finish("timeout", pos_err, ori_err)
        logger.warning(
            "Servo timed out: pos_err=%.4f (timeout=%.1fs)", pos_err, timeout
        )
        return False

    def _finish(self, exit_reason: str, pos_err: float, ori_err: float) -> None:
        """Record why the last move_to_pose ended, for the caller and the logs."""
        self.last_exit = exit_reason
        self.last_pos_err = float(pos_err)
        self.last_ori_err = float(ori_err)


try:
    from spark_real.control.ik import solve_ik_ur10e, solve_ik_ur10e_rtde
except ImportError:
    solve_ik_ur10e = None
    solve_ik_ur10e_rtde = None
from spark_real.control.common import resolve_joint_limits



class PhraseMixin:
    """Blended UR10e phrases: IK every waypoint, safety-filter the rows, play one URScript path."""

    def _next_blend_epoch(self, n_rows: int) -> int:
        """Pick a progress-register epoch the CURRENT register value is not in.

        The register survives across paths and across process restarts, so a
        stale value inside this path's window would read as progress, and a
        stale terminal value would read as "path complete" before the program
        had written a byte. Reading first and stepping off the held value
        closes that: never infer from a signal this path has not been observed
        to produce.
        """
        held = self._read_blend_progress()
        ep = getattr(self, "_blend_epoch", 0) % UR_MAX_EPOCH + 1
        for _ in range(UR_MAX_EPOCH):
            if decode_blend_progress(held, ep, n_rows) is None:
                break
            ep = ep % UR_MAX_EPOCH + 1
        self._blend_epoch = ep
        return ep

    def _solve_blend_rows(self, waypoints, q_start):
        """IK every waypoint with seed chaining; None rejects the whole phrase.

        Pre-validating the ENTIRE path before the first byte is sent is the
        point: a mid-path IK failure cannot be recovered once the program is
        running, and an IK branch flip on one row becomes a single continuous
        blended wrench instead of a discrete reconfiguration between two
        stopped moves.
        """
        if solve_ik_ur10e_rtde is None:
            logger.info("[blend] UR10e pyroki IK unavailable; no blended path")
            return None
        if os.environ.get("SPARK_UR_PYROKI_IK", "1") != "1":
            return None
        tcp_off = self._ur_tcp_offset()
        if tcp_off is None:
            logger.info("[blend] TCP offset unavailable; no blended path")
            return None

        q_seed = np.asarray(q_start, dtype=float).reshape(-1)
        rows = []
        for i, wp in enumerate(waypoints):
            pos = np.asarray(wp[0], dtype=float).reshape(3)
            orient = list(wp[1])
            vel = float(wp[2]) if len(wp) > 2 and wp[2] else self.velocity
            label = str(wp[3]) if len(wp) > 3 else f"row{i}"
            q = solve_ik_ur10e_rtde(pos, orient, q_seed=q_seed, tcp_offset=tcp_off)
            if q is None:
                logger.info(
                    "[blend] IK failed at %s pos=%s; falling back to per-move",
                    label,
                    pos.round(3).tolist(),
                )
                return None
            dq = float(np.max(np.abs(np.asarray(q, dtype=float) - q_seed[: q.size])))
            if dq > UR_MAX_ROW_DQ_RAD:
                logger.info(
                    "[blend] %s needs dq_max=%.2f rad (> %.2f): IK branch flip, "
                    "refusing to blend through it",
                    label,
                    dq,
                    UR_MAX_ROW_DQ_RAD,
                )
                return None
            vel_j, acc_j = resolve_joint_limits(
                vel, dq=np.asarray(q, dtype=float) - q_seed[: q.size]
            )
            rows.append(
                JointRow(
                    q=q,
                    position=pos,
                    velocity=vel_j,
                    acceleration=acc_j,
                    label=label,
                )
            )
            q_seed = np.asarray(q, dtype=float)
        return rows

    def _clip_blend_waypoints(self, waypoints, start_pos, phrase):
        """Apply _move_to's workspace clip + oversize refusal to every row.

        Per-move dispatch gets both of these inside _move_to; a phrase must not
        bypass them just because it takes one send instead of N. Returns the
        clipped waypoint list, or None to refuse the phrase.
        """
        out = []
        prev = np.asarray(start_pos, dtype=float)
        for wp in waypoints:
            raw = np.asarray(wp[0], dtype=float).reshape(3)
            pos = raw
            if not self._check_workspace(pos):
                pos = np.clip(raw, self.WORKSPACE_MIN, self.WORKSPACE_MAX)
                logger.info(
                    "[blend] %s row CLIPPED by workspace: (%.3f,%.3f,%.3f) -> "
                    "(%.3f,%.3f,%.3f)",
                    phrase,
                    raw[0],
                    raw[1],
                    raw[2],
                    pos[0],
                    pos[1],
                    pos[2],
                )
            seg = float(np.linalg.norm(pos - prev))
            if seg > 1.0:
                logger.warning(
                    "[blend] %s segment %.3fm too large; per-move dispatch",
                    phrase,
                    seg,
                )
                return None
            out.append((pos,) + tuple(wp[1:]))
            prev = pos
        return out

    def _blend_rows_pass_safety_filter(self, rows, phrase) -> bool:
        """Run the driver wrapper's movej joint filter over EVERY row.

        Deliberately redundant with SafeRobot's own per-row movej scan, for
        two reasons:

          - It degrades to per-move dispatch (return False) instead of
            refusing the motion outright, so a single near-singular waypoint
            costs the blend, not the task.
          - It is the only joint check in the chain when self.robot is a bare
            driver with no SafeRobot wrapper.

        Blending must not buy speed with safety, and a second independent
        evaluation of the same barrier is cheap.

        Preference order: a public per-target predicate on the wrapper if one
        exists, else the wrapper's OWN epsilon applied with the same predicate
        shape SafeRobot uses (so there is still one source for the number). A
        chain with no filter in it (a bare driver) has nothing to enforce.
        """
        node = self.robot
        for _ in range(4):
            if node is None:
                return True
            check = getattr(node, "check_joint_target_safe", None)
            if callable(check):
                for row in rows:
                    if not check(row.q.tolist()):
                        logger.warning(
                            "[blend] %s row %s refused by the safety filter; "
                            "per-move dispatch",
                            phrase,
                            row.label,
                        )
                        return False
                return True
            eps = getattr(getattr(node, "_cfg", None), "eps_singularity", None)
            if eps is not None:
                for row in rows:
                    q2 = float(row.q[2])
                    if (q2**2) * ((q2 - np.pi) ** 2) - float(eps) ** 2 < 0:
                        logger.warning(
                            "[blend] %s row %s near the elbow singularity "
                            "(q2=%.3f); per-move dispatch",
                            phrase,
                            row.label,
                            q2,
                        )
                        return False
                return True
            node = getattr(node, "_robot", None)
        return True

    def _blend_transit(self, waypoints, final_radius_m=None, phrase="phrase") -> bool:
        """Fly ``waypoints`` as ONE blended URScript program. True = arrived.

        ``waypoints`` is a list of ``(position, orientation[, velocity[,
        label]])``. Every row but the last carries a blend radius, so the arm
        cuts the corner at each intermediate waypoint instead of decelerating
        to a stop; the last row is always ``r=0`` and is therefore a true stop
        the exact joint-arrival test still applies to.

        Returns False (having commanded nothing, or having stopped the arm)
        whenever the phrase cannot be trusted (blending disabled, wrong family,
        IK rejection, nothing left to blend after dropping no-op rows, a failed
        send, a stall, or a timeout). Callers must treat False as "do it the
        old way from the current pose".
        """
        if not self._blend_enabled():
            return False
        if self._robot_family() != "ur10e":
            return False
        if not hasattr(self.robot, "_send_script"):
            return False
        self._check_abort()

        cfg = self._blend_cfg()
        max_rows = int(cfg["max_rows"])
        if len(waypoints) < 2:
            return False
        if len(waypoints) > max_rows:
            logger.info(
                "[blend] %s has %d rows (> blend_max_rows=%d); per-move dispatch",
                phrase,
                len(waypoints),
                max_rows,
            )
            return False

        try:
            q_start = np.asarray(self.robot.get_joint_positions(), dtype=float)
        except Exception as exc:  # noqa: BLE001
            logger.info("[blend] cannot read joints (%s); per-move dispatch", exc)
            return False
        start_pos = self._get_current_position()

        waypoints = self._clip_blend_waypoints(waypoints, start_pos, phrase)
        if waypoints is None:
            return False

        rows = self._solve_blend_rows(waypoints, q_start)
        if not rows:
            return False
        if not self._blend_rows_pass_safety_filter(rows, phrase):
            return False

        kept, dropped = resolve_blend_radii(
            start_pos,
            rows,
            radius_m=float(cfg["radius_m"]),
            final_radius_m=(
                float(cfg["descent_radius_m"])
                if final_radius_m is None
                else float(final_radius_m)
            ),
            min_radius_m=float(cfg["min_radius_m"]),
            start_q=q_start,
        )
        if dropped:
            logger.info("[blend] %s: dropped no-op rows %s", phrase, dropped)
        if len(kept) < 2:
            # Nothing left to blend: one real move is exactly what the
            # unblended path would issue, so let the caller issue it.
            return False

        epoch = self._next_blend_epoch(len(kept))
        program = build_ur_blend_program(kept, epoch, self._blend_register())
        logger.info(
            "[blend] %s: %d rows, r=[%s], epoch=%d",
            phrase,
            len(kept),
            ", ".join(f"{r.blend:.3f}" for r in kept),
            epoch,
        )
        if not self.robot._send_script(program):
            logger.warning("[blend] program send failed; per-move dispatch")
            self._retire_motion_lease()
            return False
        return self._wait_for_blended_path(
            kept, epoch, start_pos=start_pos, q_start=q_start
        )

    def _execute_ur_blend_phrase(self, run) -> bool:
        """UR10e backend for the legato look-ahead: one blended path per phrase.

        Accepts only one shape: a run of
        ``move_relative`` notes, optionally closed by ONE terminal
        ``move_to_keypoint`` while holding. That terminal note is the transport,
        so it is handed to ``_transport_to`` as a lead-in rather than reduced to
        a bare "go to the detection", which is what the franky path above does
        and which would skip the clearance lift and the place descent entirely.

        Returns False without commanding anything on any shape it does not
        recognise, so the caller falls back to per-action dispatch.
        """
        if not self._blend_enabled():
            return False

        lead = []
        running = np.asarray(self._get_current_position(), dtype=float).copy()
        terminal = None
        for k, note in enumerate(run):
            atype = note.get("type", note.get("name", ""))
            params = note.get("params", {}) or {}
            if atype == "move_relative":
                running = running + np.array(
                    [
                        params.get("dx", 0.0),
                        params.get("dy", 0.0),
                        params.get("dz", 0.0),
                    ],
                    dtype=float,
                )
                lead.append(
                    (
                        running.copy(),
                        self.current_orientation(),
                        self.velocity,
                        "rel%d(dz=%.2f)" % (k, params.get("dz", 0.0)),
                    )
                )
            elif atype == "move_to_keypoint" and k == len(run) - 1 and self._holding:
                if not self._blend_cfg()["cross_node"]:
                    # OFF BY DEFAULT, and not for want of speed.
                    # executor_core._run_actions skips its whole failure/recovery
                    # block when a legato phrase returns True (`i = j; continue`),
                    # so a drop detected inside the phrase would be reported and
                    # then ignored: no _attempt_recovery, no re-grasp, and the
                    # following release would open on nothing. Folding the
                    # preceding move_relative into the transport is worth ~1 s,
                    # which is not worth silently losing drop recovery. Turning
                    # this on needs _run_actions to consume a phrase's per-note
                    # results; until then the
                    # transport still blends internally, inside _transport_to.
                    return False
                terminal = note
            else:
                return False

        t0 = time.time()
        if terminal is None:
            if not self._blend_transit(lead, final_radius_m=0.0, phrase="move_rel"):
                return False
        else:
            params = terminal.get("params", {}) or {}
            label = params.get("keypoint_label", "")
            det = self.detection_map.get(label)
            if det is None:
                return False
            pos = np.asarray(det["position_3d"], dtype=float).copy()
            if pos[2] < self.TABLE_Z_FLOOR:
                pos[2] = self.TABLE_Z_FLOOR
            target = pos + np.array(
                [
                    params.get("offset_x", 0.0),
                    params.get("offset_y", 0.0),
                    params.get("offset_z", 0.0),
                ],
                dtype=float,
            )
            self._last_keypoint_label = label
            ok = self._transport_to(
                target,
                target_detection=det,
                target_label=label,
                params=params,
                lead=lead,
            )
            if not ok or not self._holding:
                # Drop detected mid-transport. The notes ARE consumed (the arm
                # moved), so report them here exactly as _move_to_keypoint would
                # rather than replaying the phrase.
                self._append_phrase_results(run, time.time() - t0, failed_last=True)
                return True

        self._append_phrase_results(run, time.time() - t0)
        return True

    def _append_phrase_results(self, run, elapsed: float, failed_last: bool = False):
        """One ExecutionResult per note of a phrase that played as one motion."""
        per = elapsed / max(len(run), 1)
        for k, note in enumerate(run):
            atype = note.get("type", note.get("name", ""))
            last = k == len(run) - 1
            bad = failed_last and last
            self._results.append(
                ExecutionResult(
                    action_type=atype,
                    success=not bad,
                    message=(
                        "Object lost during transport"
                        if bad
                        else "Played as one blended phrase (%d notes)" % len(run)
                    ),
                    duration=per,
                )
            )

    def _execute_legato_phrase(self, run, run_start_idx: int, total: int):
        """
        Play a sequence of blendable actions as one joint-space phrase.

        UR10e only: one URScript blended path (_execute_ur_blend_phrase).
        Returns False when the phrase cannot play, so the caller falls back
        to per-action dispatch.
        """
        if self._robot_family() != "ur10e":
            return False
        return self._execute_ur_blend_phrase(run)
