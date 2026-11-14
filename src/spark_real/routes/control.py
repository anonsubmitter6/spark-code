from __future__ import annotations


from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from scipy.spatial.transform import Rotation
from typing import Optional
import logging
import numpy as np
import time


from spark_real.routes import state
from spark_real.routes.state import ConnectRobotRequest, GripperRequest, GripperPositionRequest, GraspTestRequest
from spark_real.routes.state import MoveRequest, VelocityRequest


router = APIRouter()


@router.post("/api/move_relative")
async def move_relative(req: MoveRequest):
    """
    Move robot by sending a short velocity command (teleop-style).
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    if state.executor_running:
        return {"success": True, "skipped": "executor running"}
    if state.cal_active:
        return {"success": True, "skipped": "calibration active"}
    try:
        speed = 0.08
        vx = speed if req.dx > 0 else (-speed if req.dx < 0 else 0)
        vy = speed if req.dy > 0 else (-speed if req.dy < 0 else 0)
        vz = speed if req.dz > 0 else (-speed if req.dz < 0 else 0)
        try:
            pipeline._robot.send_velocity(
                [vx, vy, vz, 0, 0, 0],
                acceleration=0.3,
                duration=0.3,
            )
        except TypeError:
            pipeline._robot.send_velocity(
                [vx, vy, vz, 0, 0, 0],
                acceleration=0.3,
                time_duration=0.3,
            )
        return {"success": True, "velocity": [vx, vy, vz]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/velocity")
async def send_velocity(req: VelocityRequest):
    """
    Send EMA-smoothed velocity command for continuous teleop.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    if state.executor_running:
        return {"success": True, "skipped": "executor running"}
    if state.cal_active:
        return {"success": True, "skipped": "calibration active"}
    try:
        if state.smoothed_vel is None:
            state.smoothed_vel = np.zeros(6)
        raw = np.array([req.vx, req.vy, req.vz, req.wrx, req.wry, req.wrz])
        if state.MAX_DOWN_VEL is not None and raw[2] < -state.MAX_DOWN_VEL:
            raw[2] = -state.MAX_DOWN_VEL
        state.smoothed_vel = (
            state.VEL_ALPHA * raw + (1 - state.VEL_ALPHA) * state.smoothed_vel
        )
        state.smoothed_vel[np.abs(state.smoothed_vel) < 1e-4] = 0.0
        try:
            pipeline._robot.send_velocity(
                state.smoothed_vel.tolist(),
                acceleration=0.4,
                duration=req.duration,
            )
        except TypeError:
            pipeline._robot.send_velocity(
                state.smoothed_vel.tolist(),
                acceleration=0.4,
                time_duration=req.duration,
            )
        return {"success": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/test_yaw")
def test_yaw():
    """
    Diagnostic: cycle through 3 wrist yaw angles.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    try:
        GRASP_ORI = [2.22, -2.22, 0.0]
        base_rot = Rotation.from_rotvec(GRASP_ORI)
        safe_pose = [-0.80, 0.10, 0.10] + GRASP_ORI
        ps = ",".join(f"{v:.4f}" for v in safe_pose)
        pipeline._robot._send_script(f"movel(p[{ps}], a=0.3, v=0.15)")
        time.sleep(5)

        results = []
        for yaw_deg in [0, 30, 60]:
            yaw_rad = np.deg2rad(yaw_deg)
            final_rot = (
                (Rotation.from_euler("z", yaw_rad) * base_rot).as_rotvec().tolist()
            )
            pose = [-0.80, 0.10, 0.10] + final_rot
            ps = ",".join(f"{v:.4f}" for v in pose)
            pipeline._robot._send_script(f"movel(p[{ps}], a=0.3, v=0.10)")
            time.sleep(5)
            raw = np.array(pipeline._robot.get_tcp_pose())
            if raw.shape == (4, 4):
                R = raw[:3, :3]
            else:
                R = Rotation.from_rotvec(raw[3:6]).as_matrix()
            tx = R @ [1, 0, 0]
            results.append(
                {
                    "yaw_deg": yaw_deg,
                    "tool_X_deg": round(float(np.rad2deg(np.arctan2(tx[1], tx[0]))), 1),
                }
            )
        return {"results": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


logger = logging.getLogger("spark_server")


@router.post("/api/connect_robot")
def connect_robot(req: ConnectRobotRequest):
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    target_ip = (req.robot_ip or "").strip() or pipeline.config.robot_ip
    if not target_ip:
        return {"connected": False, "error": "No robot IP configured."}
    try:
        pipeline.config.robot_ip = target_ip
        pipeline._init_robot()
        connected = pipeline._robot is not None
        return {
            "connected": connected,
            "robot_ip": target_ip,
            "robot_family": getattr(pipeline.config, "robot_family", "ur10e"),
        }
    except Exception as e:
        return {"connected": False, "robot_ip": target_ip, "error": str(e)}


@router.post("/api/robot/release")
def release_robot():
    """Disconnect the robot so an external process (teleop) can own the
    exclusive RTDE channel. Reacquire with POST /api/connect_robot."""
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    robot = getattr(pipeline, "_robot", None)
    if robot is None:
        return {"released": True, "note": "no robot connected"}
    inner = getattr(robot, "_robot", None) or robot  # unwrap SafeRobot
    try:
        inner.disconnect()
    except Exception as e:  # noqa: BLE001
        logger.warning("robot release: disconnect raised: %s", e)
    pipeline._robot = None
    pipeline._executor = None
    logger.info("Robot released for external control (reacquire: /api/connect_robot)")
    return {"released": True}


@router.post("/api/stream_mode")
async def set_stream_mode(req: dict):
    state.stream_mode = req.get("mode", "rgb")
    return {"mode": state.stream_mode}


@router.post("/api/stream_camera")
async def set_stream_camera(req: dict):
    state.stream_camera = req.get("camera", "all")
    return {"camera": state.stream_camera}


@router.post("/api/gripper")
def control_gripper(req: GripperRequest):
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    if req.action == "open":
        pipeline._robot.open_gripper()
    elif req.action == "close":
        pipeline._robot.close_gripper()
    return {"action": req.action, "success": True}


@router.post("/api/gripper_position")
def control_gripper_position(req: GripperPositionRequest):
    """
    Variable-width gripper command. position 0.0 (open) to 1.0 (closed).
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    pos = float(max(0.0, min(1.0, req.position)))
    try:
        pipeline._robot.set_gripper_position(pos, speed=req.speed, force=req.force)
    except TypeError:
        pipeline._robot.set_gripper_position(pos)
    except AttributeError:
        return JSONResponse(
            status_code=400, content={"error": "Driver has no set_gripper_position."}
        )
    return {"position": pos, "success": True}


@router.post("/api/grasp_test")
def grasp_test(req: GraspTestRequest):
    """
    Run one calibrated grasp_to_width and return metrics.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    if state.executor_running:
        return JSONResponse(status_code=409, content={"error": "Executor running"})
    robot = pipeline._robot
    if not hasattr(robot, "grasp_to_width"):
        return JSONResponse(
            status_code=400, content={"error": "No grasp_to_width support"}
        )

    def _read_wrench():
        try:
            if hasattr(robot, "get_tcp_force"):
                ft = np.asarray(robot.get_tcp_force(), dtype=float)
                if ft.size >= 3:
                    return [float(ft[0]), float(ft[1]), float(ft[2])]
        except Exception:
            pass
        return [0.0, 0.0, 0.0]

    try:
        robot.open_gripper()
    except Exception as exc:
        return JSONResponse(
            status_code=500, content={"error": f"open_gripper failed: {exc}"}
        )
    time.sleep(1.0)
    pre = _read_wrench()

    try:
        success_flag = bool(
            robot.grasp_to_width(
                width=float(req.target_width),
                force=float(req.force),
                speed=float(req.speed),
            )
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500, content={"error": f"grasp_to_width raised: {exc}"}
        )

    time.sleep(float(req.settle_s))
    try:
        achieved_w = float(robot.get_gripper_width())
    except Exception:
        achieved_w = 0.0

    post = _read_wrench()
    delta_mag = float(np.linalg.norm([post[i] - pre[i] for i in range(3)]))
    inferred = bool(
        0.005 < achieved_w < 0.085 and abs(achieved_w - req.target_width) < 0.020
    )
    if req.release_after:
        try:
            robot.open_gripper()
        except Exception:
            pass

    return {
        "label": req.label,
        "target_width": req.target_width,
        "achieved_width": round(achieved_w, 4),
        "driver_grasp_success": success_flag,
        "delta_force_mag": round(delta_mag, 3),
        "inferred_holding": inferred,
    }



def _home_target_for(pipeline):
    """Home joints from the profile, else the connected driver's HOME_CONFIG."""
    profile = getattr(pipeline, "profile", None)
    home = profile.home_config() if profile is not None else None
    if home:
        return list(home)
    robot = getattr(pipeline, "_robot", None)
    robot = getattr(robot, "_robot", robot)
    return list(robot.HOME_CONFIG)


@router.post("/api/home")
def go_home():
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    target = _home_target_for(pipeline)

    for stop_fn in ("stop", "stop_motion", "servo_stop"):
        if hasattr(pipeline._robot, stop_fn):
            try:
                getattr(pipeline._robot, stop_fn)()
                break
            except Exception:
                pass
    time.sleep(0.3)

    try:
        if hasattr(pipeline._robot, "_send_script"):
            joints_str = ",".join(f"{j:.4f}" for j in target)
            if not pipeline._robot._send_script(
                f"movej([{joints_str}], a=1.0, v=0.8)"
            ):
                # A dropped home command must not return success:true after
                # the poll loop breaks on a read exception.
                return JSONResponse(
                    status_code=500,
                    content={"error": "home movej dropped: URScript channel down"},
                )
        elif hasattr(pipeline._robot, "go_home"):
            pipeline._robot.go_home()
        else:
            pipeline._robot.move_to_joint_config(
                target,
                velocity=0.8,
                acceleration=1.0,
            )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"go_home failed: {e}"})

    for _ in range(100):
        time.sleep(0.1)
        try:
            joints = np.array(pipeline._robot.get_joint_positions())
            if len(joints) == len(target) and np.allclose(joints, target, atol=0.05):
                break
        except Exception:
            break
    return {"success": True}


@router.post("/api/stop")
async def stop_robot():
    """Brake the arm. The operator's stop button ends up here.

    This must reach the arm, not just set a flag: URScript motion is
    fire-and-forget, so an unbraked stop lets the arm run on to its target.
    Both entry points below brake -- the executor's abort() for the running
    score, and the driver's own escalation (URScript stopj -> reopened socket
    -> bounded Dashboard stop) for the arm itself, which is also the path when
    no score is running at all. The reply reports whether a stop actually left
    this process; `braked: false` means THE ARM MAY STILL BE MOVING.
    """
    state.go_cancelled = True  # a run waiting at the start gate is cancelled, not started
    state.go_event.set()
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    errors = []
    steps = []
    braked = False

    if pipeline._executor is not None:
        try:
            braked = bool(pipeline._executor.abort())
            steps.append("executor.abort" if braked else "executor.abort:no-brake")
        except Exception as e:
            errors.append(f"executor.abort: {e}")

    # Brake the driver directly whenever the executor did not: no score
    # running, abort() raised, or its escalation failed. Skipped once the arm is
    # already decelerating, because a second stop program REPLACES the first.
    # The escalation is the driver's (it owns the socket and the Dashboard
    # client); the old stop/stop_motion/servo_stop first-hit-wins loop is gone
    # -- it resolved to rtde_c.stopJ, which cannot stop this rig, and its
    # success masked the failure of the URScript stop above it.
    if not braked:
        emergency = getattr(pipeline._robot, "emergency_stop", None)
        if callable(emergency):
            try:
                result = emergency()
                steps.extend(result.get("steps", []))
                braked = bool(result.get("braked"))
            except Exception as e:
                errors.append(f"emergency_stop: {e}")
        else:
            for stop_fn in ("stop_motion", "stop", "servo_stop"):
                fn = getattr(pipeline._robot, stop_fn, None)
                if callable(fn):
                    try:
                        braked = fn() is not False or braked
                        steps.append(stop_fn)
                    except Exception as e:
                        errors.append(f"{stop_fn}: {e}")

    try:
        if pipeline._executor is not None:
            pipeline._executor._running = False
            # Belt-and-braces after executor.abort() above, but it must go
            # through note_abort_requested: a bare `_abort = True` does not
            # advance the abort epoch, so execute_score would wipe it when the
            # run finally reaches it. That matters most when abort() above
            # raised or when no score is running YET -- the executor thread is
            # in capture/detect/plan and execute_score is still seconds away.
            note = getattr(pipeline._executor, "note_abort_requested", None)
            if callable(note):
                note()
            else:
                pipeline._executor._abort = True
        state.executor_running = False
        with pipeline._activity_lock:
            pipeline._activity_stack.clear()
    except Exception:
        pass

    if not braked:
        logger.critical("/api/stop: THE ARM WAS NOT BRAKED (%s)", errors or steps)
    return {"success": True, "braked": braked, "steps": steps, "errors": errors}


@router.post("/api/recover")
def recover_robot():
    """
    Clear protective stop / reflex state so teleop can continue.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})

    robot = pipeline._robot
    actions, errors = [], []

    for fn in ("stop_velocity", "stop_motion", "servo_stop"):
        if hasattr(robot, fn):
            try:
                getattr(robot, fn)()
                actions.append(fn)
                break
            except Exception as e:
                errors.append(f"{fn}: {e}")

    for fn in (
        "recover_from_errors",
        "clear_protective_stop",
        "_send_dashboard_command",
    ):
        if hasattr(robot, fn):
            try:
                if fn == "_send_dashboard_command":
                    getattr(robot, fn)("close safety popup")
                else:
                    getattr(robot, fn)()
                actions.append(fn)
                break
            except Exception as e:
                errors.append(f"{fn}: {e}")

    if hasattr(robot, "cbf_deviated"):
        robot.cbf_deviated = False
        actions.append("cleared_cbf_deviated")

    nudge_info = _nudge_off_limit_joints(robot)
    if nudge_info:
        actions.append(nudge_info)

    return {"success": True, "actions": actions, "errors": errors}


def _nudge_off_limit_joints(robot):
    """
    Nudge joints near limits partway toward home.
    """
    try:
        target_robot = robot
        for attr in ("_robot", "_driver"):
            inner = getattr(target_robot, attr, None)
            if inner is not None:
                target_robot = inner
        limits = getattr(target_robot, "JOINT_LIMITS", None)
        home = getattr(target_robot, "HOME_CONFIG", None)
        if not limits or not home or not hasattr(robot, "get_joint_positions"):
            return None
        q_now = list(robot.get_joint_positions())
        if len(q_now) != len(limits):
            return None
        q_target = list(q_now)
        nudged = []
        for i, (q, (lo, hi), q_home) in enumerate(zip(q_now, limits, home)):
            rng = hi - lo
            if rng <= 0:
                continue
            margin = rng * 0.12
            if (q - lo) < margin or (hi - q) < margin:
                q_target[i] = q + 0.35 * (q_home - q)
                nudged.append(i)
        if not nudged:
            return None
        if hasattr(robot, "move_to_joint_config"):
            try:
                robot.move_to_joint_config(q_target, velocity=0.25)
            except TypeError:
                robot.move_to_joint_config(q_target)
            return f"nudged_joints={nudged}"
    except Exception:
        pass
    return None


@router.get("/api/robot_state")
async def get_robot_state():
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return {"connected": False}
    try:
        if hasattr(pipeline._robot, "get_observation"):
            obs = pipeline._robot.get_observation()
        else:
            obs = {}
            for attr, key in (
                ("get_joint_positions", "joint_positions"),
                ("get_tcp_pose", "tcp_pose"),
                ("get_gripper_position", "gripper_position"),
            ):
                fn = getattr(pipeline._robot, attr, None)
                if callable(fn):
                    try:
                        obs[key] = fn()
                    except Exception:
                        pass
            obs.setdefault("gripper_position", 0.0)
    except Exception as e:
        return {"connected": False, "error": str(e)}

    def _list_or_none(val):
        if val is None:
            return None
        return val.tolist() if hasattr(val, "tolist") else list(val)

    raw_grip = obs.get("gripper_position")
    grip_value = None
    if raw_grip is not None:
        try:
            grip_value = float(raw_grip)
        except (TypeError, ValueError):
            pass

    # gObj object-detect flag + TCP speed (diagnostic: lets a caller confirm the
    # arm is stationary before trusting the gripper register read).
    obj_detected = None
    try:
        if hasattr(pipeline._robot, "is_object_detected"):
            obj_detected = bool(pipeline._robot.is_object_detected())
    except Exception:
        pass
    tcp_speed = None
    try:
        rr = getattr(pipeline._robot, "_rtde_r", None) or getattr(
            getattr(pipeline._robot, "_robot", None), "_rtde_r", None
        )
        if rr is not None:
            spd = np.asarray(rr.getActualTCPSpeed(), dtype=float)
            if spd.size >= 3:
                tcp_speed = float(np.linalg.norm(spd[:3]))
    except Exception:
        pass

    return {
        "connected": True,
        "joint_positions": _list_or_none(obs.get("joint_positions")),
        "tcp_pose": _list_or_none(obs.get("tcp_pose")),
        "gripper_position": grip_value,
        "object_detected": obj_detected,
        "tcp_speed": tcp_speed,
    }


# Teleop endpoints (move_relative / velocity / test_yaw) live in control_teleop.py.


logger = logging.getLogger("spark_server")


class SaveEpisodeRequest(BaseModel):
    # PASS/FAIL is recorded, never inferred. Task-success predicates are not
    # reliable enough yet to label a corpus with, and a FAILED episode is not
    # rubbish -- it is the negative half a verifier study needs and the
    # expensive half to collect deliberately. So a failed run is SAVED with
    # outcome="fail", not discarded.
    outcome: Optional[str] = None  # "pass" | "fail" | None (unlabelled)
    task_name: Optional[str] = None


class PromoteRequest(BaseModel):
    hash: str
    promoted: bool = True


@router.post("/api/episode/save")
async def episode_save(req: SaveEpisodeRequest):
    rec = state.current_episode
    if rec is None:
        return JSONResponse(
            status_code=404, content={"error": "No cached episode to save"}
        )
    task_name = req.task_name or rec.instruction or "episode"
    outcome = (req.outcome or "").strip().lower() or None
    if outcome not in (None, "pass", "fail"):
        return JSONResponse(
            status_code=400,
            content={"error": f"outcome must be pass/fail, got {outcome!r}"},
        )
    try:
        kept = rec.save_as(task_name, outcome=outcome)
    except Exception as exc:
        logger.error("episode_save failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})
    summary = rec.summary()
    summary["kept_path"] = str(kept)
    summary["outcome"] = outcome
    logger.info("episode_save: %r outcome=%s -> %s", task_name, outcome, kept)
    return {"saved": True, "outcome": outcome, "episode": summary}


@router.post("/api/episode/discard")
async def episode_discard():
    rec = state.current_episode
    if rec is None:
        return {"discarded": False, "reason": "no cached episode"}
    try:
        rec.discard()
    except Exception as exc:
        logger.warning("episode_discard failed: %s", exc)
    state.current_episode = None
    return {"discarded": True}


# The /api/episode/* bundle below stores bt.yaml + result.json beside a run.


@router.get("/api/library/list")
async def library_list():
    pipeline = state.pipeline
    if pipeline is None or pipeline._bt_library is None:
        return {"entries": []}
    return {"entries": pipeline._bt_library.list_all()}


@router.post("/api/library/promote")
async def library_promote(req: PromoteRequest):
    pipeline = state.pipeline
    if pipeline is None or pipeline._bt_library is None:
        return JSONResponse(
            status_code=400, content={"error": "BT library not initialised"}
        )
    entry = pipeline._bt_library.set_promoted(req.hash, req.promoted)
    if entry is None:
        return JSONResponse(
            status_code=404, content={"error": f"No BT entry with hash " f"{req.hash}"}
        )
    return {
        "hash": entry.hash,
        "promoted": entry.promoted,
        "instruction": entry.instruction,
    }
