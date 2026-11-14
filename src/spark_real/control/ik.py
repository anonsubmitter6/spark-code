"""
Inverse kinematics for the UR10e and the FR3 with pyroki (JAX).

Both arms solve a jaxls Levenberg-Marquardt factor graph (pose, limit and
rest costs) on a URDF from robot_descriptions, JIT-compiled on first use.

UR10e (6 joints, `tool0` flange): solve_ik_ur10e, solve_ik_ur10e_rtde (takes
the RTDE TCP pose and removes the pendant TCP offset), fk_ur10e.
FR3 (7 joints, `fr3_hand_tcp`, adds manipulability and optional
self-collision costs): solve_ik_fr3, fk_fr3.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Union

# CPU pinned: GPU JAX linalg solves fail on a mixed nvidia wheel set.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import jax
    import jax.numpy as jnp
    import jaxlie
    import jaxls
    import pyroki
    from jaxls import Cost, Var, VarValues
    from jax import Array
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except ImportError:
    jax = jnp = jaxlie = jaxls = pyroki = None
    Cost = Var = VarValues = Array = None
    load_robot_description = None

logger = logging.getLogger(__name__)

# UR10e IK seed / rest pose (elbow-up wrist-down grasp posture). Kept distinct
# from ur10e_driver.HOME_CONFIG so the solver rests toward a downward-facing
# grasp configuration rather than the mechanical home.
_UR10E_REST = np.array([1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float32)

_UR10E_DESC = "ur10e_description"
_UR10E_MAX_POS_ERR_M = 0.05
_UR10E_MAX_ORI_ERR_RAD = np.deg2rad(2.0)
_WRIST3_UNWIND_RAD = 4.0
_UR10E_TCP_LINK = "tool0"

_ur_robot = None
_ur_tli: Optional[int] = None
_ur_nact: Optional[int] = None
_ur_solver = None
_ur_last_q: Optional[np.ndarray] = None


def _ur_load():
    global _ur_robot, _ur_tli, _ur_nact
    if _ur_robot is not None:
        return
    if pyroki is None or load_robot_description is None:
        raise RuntimeError(
            "pyroki/JAX/robot_descriptions not installed; UR10e pyroki IK unavailable"
        )
    urdf = load_robot_description(_UR10E_DESC)
    _ur_robot = pyroki.Robot.from_urdf(urdf)
    names = list(_ur_robot.links.names)
    if _UR10E_TCP_LINK not in names:
        raise RuntimeError(f"No '{_UR10E_TCP_LINK}' link in UR10e URDF. Available: {names}")
    _ur_tli = names.index(_UR10E_TCP_LINK)
    _ur_nact = int(_ur_robot.joints.num_actuated_joints)
    logger.info("UR10e IK (pyroki): %d links, %d joints, tcp=%s", len(names), _ur_nact, _UR10E_TCP_LINK)


def _ur_build():
    global _ur_solver
    if _ur_solver is not None:
        return
    pk = pyroki.costs
    tli = jnp.array(_ur_tli, dtype=jnp.int32)

    def _solve(tp, tq, sd, pw, ow, rw, lw):
        pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(tq), tp)
        jv = _ur_robot.joint_var_cls(0)
        fs = [
            pk.pose_cost(_ur_robot, jv, pose, tli, pos_weight=pw, ori_weight=ow),
            pk.limit_cost(_ur_robot, jv, weight=lw),
            pk.rest_cost(jv, rest_pose=sd, weight=rw),
        ]
        prob = jaxls.LeastSquaresProblem(fs, [jv]).analyze()
        sol = prob.solve(
            initial_vals=jaxls.VarValues.make([jv.with_value(sd)]),
            linear_solver="dense_cholesky",
            termination=jaxls.TerminationConfig(max_iterations=150),
            verbose=False,
        )
        qf = sol[jv]
        T = jaxlie.SE3(_ur_robot.forward_kinematics(qf)[_ur_tli])
        return qf, T.translation(), T.rotation().as_matrix()

    _ur_solver = jax.jit(_solve)


def _ur_seed(q_seed: np.ndarray) -> np.ndarray:
    """Coerce an arbitrary seed to a 6-vec float32 (pad/truncate with rest)."""
    s = _UR10E_REST.copy()
    q = np.asarray(q_seed, dtype=np.float32).ravel()
    n = min(6, q.size)
    s[:n] = q[:n]
    return s


def solve_ik_ur10e(
    target_pos: np.ndarray,
    target_orient: Union[np.ndarray, list],
    q_seed: np.ndarray,
    pos_weight: float = 300.0,
    ori_weight: float = 100.0,
    rest_weight: float = 1.5,
    limit_weight: float = 200.0,
    **_kw,
) -> Optional[np.ndarray]:
    """
    Solve UR10e IK for the tool0 frame.

    Args:
        target_pos: (3,) world position of tool0 (meters, robot base frame).
        target_orient: (3,) axis-angle rotvec or (3,3) rotation matrix.
        q_seed: current joint config (>=6 used); seeds the solve + tie-break.

    Returns:
        (6,) joint positions, or None on failure / >5cm or >2deg residual.
    """
    global _ur_last_q
    _ur_load()
    _ur_build()

    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    to = np.asarray(target_orient, dtype=float)
    if to.shape == (3,):
        Rt = Rotation.from_rotvec(to).as_matrix()
    elif to.shape == (3, 3):
        Rt = to
    else:
        raise ValueError(f"target_orient must be (3,) or (3,3), got {to.shape}")

    sd = _ur_seed(q_seed)

    def _one(R, seed):
        qx = Rotation.from_matrix(R).as_quat()  # xyzw
        qw = np.array([qx[3], qx[0], qx[1], qx[2]], dtype=np.float32)  # wxyz
        try:
            qj, ap, aR = _ur_solver(
                jnp.asarray(target_pos, jnp.float32),
                jnp.asarray(qw, jnp.float32),
                jnp.asarray(seed, jnp.float32),
                jnp.float32(pos_weight),
                jnp.float32(ori_weight),
                jnp.float32(rest_weight),
                jnp.float32(limit_weight),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ur pyroki IK] solve raised: %s", exc)
            return None, None, None
        # numpy conversion OUTSIDE the jit (asarray inside traces->error).
        return (
            np.asarray(qj, dtype=float).ravel(),
            np.asarray(ap, dtype=float),
            np.asarray(aR, dtype=float),
        )

    # The commanded yaw is the yaw; callers that may use the two-jaw symmetry choose the
    # branch themselves (grasp_strategy.symmetric_yaw).
    # Seeds: the current wrist, then wrist-3 turned by +-90: LM stalls on a long yaw otherwise.
    cands = []
    for turn in (0.0, np.pi / 2, -np.pi / 2):
        q, ap, aR = _one(Rt, sd + np.array([0, 0, 0, 0, 0, turn], dtype=np.float32))
        if q is None or q.size < 6:
            continue
        err = float(np.linalg.norm(ap - target_pos))
        ori = float(Rotation.from_matrix(aR.T @ Rt).magnitude())
        if err <= _UR10E_MAX_POS_ERR_M and ori <= _UR10E_MAX_ORI_ERR_RAD:
            cands.append((q, ap, float(np.max(np.abs(q[:6] - sd[:6]))), err))
    if not cands:
        logger.warning("[ur pyroki IK] no candidate reached the pose")
        return None
    cands.sort(key=lambda c: c[2])  # smallest joint swing
    qb, ab, dqb, eb = cands[0]

    qb = np.asarray(qb, dtype=float).copy()
    # Wrist-3 is 2pi-periodic and limit_cost is soft, so ties near a half turn walk it toward
    # its +-2pi stop. Unwind only past _WRIST3_UNWIND_RAD: a wrap at +-pi spins a full turn on a 1 cm move.
    if abs(qb[5]) > _WRIST3_UNWIND_RAD:
        qb[5] = (qb[5] + np.pi) % (2 * np.pi) - np.pi
    logger.info("[ur pyroki IK] solved dq=%.2f pos_err=%.4fm", dqb, eb)
    _ur_last_q = np.asarray(qb, dtype=np.float32).copy()
    return qb[:6]


# Hardware frame handling (RTDE base <-> URDF base_link).
# The UR RTDE base frame is a pure Rz(180) from the URDF base_link frame pyroki
# uses. getTCPOffset() gives the pendant TCP (gripper tip relative to tool0).
# solve_ik_ur10e_rtde converts an RTDE-frame gripper-TIP target into a URDF-frame
# tool0 target and solves.


def _se3(p, rv_or_R):
    T = np.eye(4)
    a = np.asarray(rv_or_R, float)
    T[:3, :3] = a if a.shape == (3, 3) else Rotation.from_rotvec(a).as_matrix()
    T[:3, 3] = np.asarray(p, float).reshape(3)
    return T


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _rz180():
    return Rotation.from_euler("z", np.pi).as_matrix()


def solve_ik_ur10e_rtde(
    tip_pos: np.ndarray,
    tip_orient: Union[np.ndarray, list],
    q_seed: np.ndarray,
    tcp_offset: Union[np.ndarray, list],
    **kw,
) -> Optional[np.ndarray]:
    """
    Solve UR10e IK for a gripper-TIP target expressed in the RTDE base frame.

    Handles the two hardware frame facts bare solve_ik_ur10e does not:
      1. RTDE base <-> URDF base_link differ by Rz(180 deg).
      2. The pendant TCP (getTCPOffset) offsets the gripper tip from tool0.

    Args:
        tip_pos: (3,) gripper-tip target, RTDE base frame.
        tip_orient: (3,) rotvec or (3,3) matrix, tip orientation, RTDE frame.
        q_seed: current joints (seed + tie-break).
        tcp_offset: (6,) [x,y,z,rx,ry,rz] from RTDEControl.getTCPOffset()
            (tip relative to tool0, tool0 frame).
    Returns:
        (6,) joints, or None on failure.
    """
    to = np.asarray(tip_orient, float)
    R_tip = to if to.shape == (3, 3) else Rotation.from_rotvec(to.reshape(3)).as_matrix()
    tcp = np.asarray(tcp_offset, float).reshape(6)
    T_base_tip = _se3(np.asarray(tip_pos, float).reshape(3), R_tip)
    T_tool0_tip = _se3(tcp[:3], tcp[3:])
    T_base_tool0 = T_base_tip @ _inv(T_tool0_tip)   # RTDE-frame tool0 target
    Rz = _rz180()                                    # URDF <- RTDE (== its own inverse)
    pos_urdf = Rz @ T_base_tool0[:3, 3]
    R_urdf = Rz @ T_base_tool0[:3, :3]
    return solve_ik_ur10e(pos_urdf, R_urdf, q_seed, **kw)


def fk_ur10e(q: np.ndarray) -> tuple:
    """Forward kinematics for tool0. Returns (pos[3], R[3x3])."""
    _ur_load()
    qf = _ur_seed(q)
    T = jaxlie.SE3(_ur_robot.forward_kinematics(jnp.asarray(qf))[_ur_tli])
    return (
        np.asarray(T.translation(), dtype=float),
        np.asarray(T.rotation().as_matrix(), dtype=float),
    )


_FR3_DESC = "fr3_description"
_urdf = None
_fr_robot = _rcoll = _ths = None
_fr_tli = _fr_nact = None
_solvers: dict = {}
_fr_last_q: Optional[np.ndarray] = None
_VEL_DT, _VEL_W = 0.01, 0.1
_SC_M, _SC_W, _WC_M, _WC_W, _TZ = 0.02, 5.0, 0.02, 5.0, -0.025


def _fr_load():
    global _fr_robot, _fr_tli, _fr_nact, _urdf
    if _fr_robot is not None:
        return
    if pyroki is None or load_robot_description is None:
        raise RuntimeError("pyroki/JAX/robot_descriptions not installed; FR3 pyroki IK unavailable")
    _urdf = load_robot_description(_FR3_DESC)
    _fr_robot = pyroki.Robot.from_urdf(_urdf)
    names = list(_fr_robot.links.names)
    for c in ("fr3_hand_tcp", "fr3_hand", "panda_hand_tcp"):
        if c in names:
            _fr_tli = names.index(c)
            logger.info("FR3 IK (pyroki): %d links, tcp=%s", len(names), c)
            break
    if _fr_tli is None:
        raise RuntimeError(f"No TCP link in URDF. Available: {names[:20]}...")
    _fr_nact = int(_fr_robot.joints.num_actuated_joints)


def _load_coll():
    global _rcoll, _ths
    if _rcoll is not None:
        return
    _rcoll = pyroki.collision.RobotCollision.from_urdf(_urdf)
    _ths = pyroki.collision.HalfSpace.from_point_and_normal(
        point=jnp.array([0.0, 0.0, _TZ], dtype=jnp.float32),
        normal=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
    )


def _fr_solver(sm: bool, co: bool):
    k = (sm, co)
    if k not in _solvers:
        _fr_load()
        if co:
            _load_coll()
        _solvers[k] = _fr_build(sm, co)
    return _solvers[k]


def _fr_build(sm: bool, co: bool):
    pk = pyroki.costs
    tli = jnp.array(_fr_tli, dtype=jnp.int32)
    if sm:

        @Cost.create_factory
        def _vc(v: VarValues, r, jv: Var[Array], p: Array, dt: float, w) -> Array:
            return (
                jnp.maximum(0.0, jnp.abs((v[jv] - p) / dt) - r.joints.velocity_limits)
                * w
            ).flatten()

    def _solve(tp, tq, sd, pw, ow, rw, lw, mw, *ex):
        ex = iter(ex)
        if sm:
            prev, vw = next(ex), next(ex)
        if co:
            scw, wcw, tg = next(ex), next(ex), next(ex)
        pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(tq), tp)
        jv = _fr_robot.joint_var_cls(0)
        fs = [
            pk.pose_cost(_fr_robot, jv, pose, tli, pos_weight=pw, ori_weight=ow),
            pk.limit_cost(_fr_robot, jv, weight=lw),
            pk.rest_cost(jv, rest_pose=sd, weight=rw),
            pk.manipulability_cost(_fr_robot, jv, tli, weight=mw),
        ]
        if sm:
            fs.append(_vc(_fr_robot, jv, prev, _VEL_DT, vw))
        if co:
            fs.append(
                pk.self_collision_cost(
                    _fr_robot, robot_coll=_rcoll, joint_var=jv, margin=_SC_M, weight=scw
                )
            )
            fs.append(
                pk.world_collision_cost(
                    _fr_robot,
                    robot_coll=_rcoll,
                    joint_var=jv,
                    world_geom=tg,
                    margin=_WC_M,
                    weight=wcw,
                )
            )
        prob = jaxls.LeastSquaresProblem(fs, [jv]).analyze()
        sol = prob.solve(
            initial_vals=jaxls.VarValues.make([jv.with_value(sd)]),
            linear_solver="dense_cholesky",
            termination=jaxls.TerminationConfig(max_iterations=150),
            verbose=False,
        )
        qf = sol[jv]
        return qf, jaxlie.SE3(_fr_robot.forward_kinematics(qf)[_fr_tli]).translation()

    return jax.jit(_solve)


def _fr_seed(q7):
    s = np.zeros(_fr_nact, dtype=np.float32)
    n = min(7, _fr_nact)
    s[:n] = q7[:n]
    if abs(s[4]) < 0.30:
        s[4] = -0.50 if s[4] <= 0 else 0.50
    if _fr_nact > 7:
        s[7:] = 0.02
    return s


def _wj7(q, ref):
    q7 = q[6]
    best, bok, bd = q7, abs(q7) <= 2.85, abs(q7 - ref)
    for d in (-np.pi, np.pi):
        c, ok, di = q7 + d, abs(q7 + d) <= 2.85, abs(q7 + d - ref)
        if (ok and not bok) or (ok == bok and di < bd):
            best, bok, bd = c, ok, di
    q[6] = best


def solve_ik_fr3(
    target_pos: np.ndarray,
    target_orient: Union[np.ndarray, list],
    q_seed: np.ndarray,
    pos_weight: float = 300.0,
    ori_weight: float = 100.0,
    rest_weight: float = 1.5,
    limit_weight: float = 200.0,
    manip_weight: float = 1.0,
    **_kw,
) -> Optional[np.ndarray]:
    """
    Solve FR3 IK. Returns 7-vec joint positions or None on failure.
    """
    global _fr_last_q
    _fr_load()
    uv = os.environ.get("SPARK_IK_VEL_SMOOTHING", "0") == "1"
    uc = os.environ.get("SPARK_IK_COLLISION", "0") == "1"
    slv = _fr_solver(uv, uc)

    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    to = np.asarray(target_orient, dtype=float)
    if to.shape == (3,):
        Rt = Rotation.from_rotvec(to).as_matrix()
    elif to.shape == (3, 3):
        Rt = to
    else:
        raise ValueError(f"target_orient must be (3,) or (3,3), got {to.shape}")

    q_seed = np.asarray(q_seed, dtype=float).ravel()
    if q_seed.size < 7:
        raise ValueError(f"q_seed needs >= 7 entries, got {q_seed.size}")
    sd = _fr_seed(q_seed)
    pv = _fr_last_q if _fr_last_q is not None else sd

    def _one(R, rw):
        qx = Rotation.from_matrix(R).as_quat()
        qw = np.array([qx[3], qx[0], qx[1], qx[2]], dtype=np.float32)
        a = (
            jnp.asarray(target_pos, jnp.float32),
            jnp.asarray(qw, jnp.float32),
            jnp.asarray(sd, jnp.float32),
            jnp.float32(pos_weight),
            jnp.float32(ori_weight),
            jnp.float32(rw),
            jnp.float32(limit_weight),
            jnp.float32(manip_weight),
        )
        ex = ()
        if uv:
            ex += (jnp.asarray(pv, jnp.float32), jnp.float32(_VEL_W))
        if uc:
            ex += (jnp.float32(_SC_W), jnp.float32(_WC_W), _ths)
        try:
            qj, ap = slv(*a, *ex)
        except Exception:
            return None, None
        return np.asarray(qj, dtype=float).ravel(), np.asarray(ap, dtype=float)

    # Solve both yaw orientations (gripper symmetry: yaw and yaw+pi).
    Ra = Rt @ Rotation.from_euler("z", np.pi).as_matrix()
    cands = []
    for R, tag in ((Rt, "primary"), (Ra, "yaw+pi")):
        q, ap = _one(R, rest_weight)
        if q is None or q.size < 7:
            continue
        _wj7(q, q_seed[6])
        cands.append(
            (
                q,
                ap,
                np.max(np.abs(q[:7] - q_seed[:7])),
                float(np.linalg.norm(ap - target_pos)),
                tag,
            )
        )

    if not cands:
        logger.warning("[pyroki IK] no candidate solved")
        return None
    cands = [c for c in cands if c[3] <= 0.05]
    if not cands:
        logger.warning("[pyroki IK] all candidates >5cm off")
        return None
    # Prefer J5 away from the wrist singularity; do not add an upper band
    # (it breaks candidate continuity across chained waypoints). The hard
    # clamp below handles the limit case.
    ns = [c for c in cands if abs(c[0][4]) >= 0.25]
    if ns:
        cands = ns
    cands.sort(key=lambda c: c[2])
    qb, _, dqb, eb, tb = cands[0]

    # Heavy-rest retry for large swing or J5 near singularity.
    if dqb > 1.0 or abs(qb[4]) < 0.25:
        logger.info("[pyroki IK] dq=%.2f/J5=%.2f; heavy-rest retry", dqb, qb[4])
        q2, a2 = _one(Rt if tb == "primary" else Ra, rest_weight * 4)
        if q2 is not None and q2.size >= 7:
            _wj7(q2, q_seed[6])
            d2, e2 = np.max(np.abs(q2[:7] - q_seed[:7])), float(
                np.linalg.norm(a2 - target_pos)
            )
            if e2 <= 0.05 and d2 < dqb:
                qb, dqb, eb = q2, d2, e2

    if eb > 0.05:
        return None
    # Hard-clamp to joint limits minus a margin: the soft limit_cost can
    # let a solution land just past a bound, which libfranka rejects with
    # joint_position_limits_violation and a slow fallback cascade.
    lo = np.asarray(_fr_robot.joints.lower_limits, dtype=float)[:7] + 0.02
    hi = np.asarray(_fr_robot.joints.upper_limits, dtype=float)[:7] - 0.02
    qb = np.asarray(qb, dtype=float).copy()
    qb[:7] = np.clip(qb[:7], lo, hi)
    logger.info("[pyroki IK] solved dq=%.2f pos_err=%.4fm", dqb, eb)
    _fr_last_q = np.asarray(qb, dtype=np.float32).copy()
    return qb[:7]


def fk_fr3(q: np.ndarray) -> tuple:
    """
    Forward kinematics for TCP. Returns (pos[3], R[3x3]).
    """
    _fr_load()
    qf = np.zeros(_fr_nact, dtype=np.float32)
    q = np.asarray(q, dtype=float).ravel()
    qf[: min(7, _fr_nact)] = q[: min(7, _fr_nact)]
    if _fr_nact > 7:
        qf[7:] = 0.02
    T = jaxlie.SE3(_fr_robot.forward_kinematics(jnp.asarray(qf))[_fr_tli])
    return np.asarray(T.translation(), dtype=float), np.asarray(
        T.rotation().as_matrix(), dtype=float
    )
