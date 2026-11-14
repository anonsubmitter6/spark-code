from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pathlib import Path
import asyncio
import json
import logging
import os
import threading
import time
import yaml


from spark_core.util import base_label
from spark_real import recording as run_timing
from spark_real.control.predicates import UNVERIFIED
from spark_real.control.success_verifier import reset_run_state, task_success
from spark_real.recording import EpisodeRecorder
from spark_real.perception.prompts import PromptCountMismatch
from spark_real.pipeline import SPARKRealPipeline, PipelineConfig
from spark_real.pipeline_plan import build_detection_details
from spark_real.pipeline import user_aborted
from spark_real.planning.plan_annotate import annotate_for_planner
from spark_real.routes import state
from spark_real.routes.detection import router as _detection_router
from spark_real.routes.state import ExecuteRequest, PlanRequest
from spark_real.routes.streaming import get_available_cameras
from spark_real.routes.visualization import encode_image, depth_to_colormap
from spark_real.routes.visualization import serialize_detections


logger = logging.getLogger("spark_server")
router = APIRouter()


def _pretask_home(pipeline):
    """
    Open gripper, clear reflex, go home. Best-effort.
    """
    try:
        pipeline._robot.open_gripper()
    except Exception as exc:
        logger.warning("pre-task gripper open failed: %s", exc)
    try:
        pipeline._robot.recover_from_errors()
    except Exception:
        pass
    for method in ("go_home", "move_home", "home"):
        if hasattr(pipeline._robot, method):
            try:
                getattr(pipeline._robot, method)()
                logger.info("Pre-task home complete (via %s)", method)
                return
            except Exception as exc:
                logger.warning("pre-task home (%s) failed: %s", method, exc)


def _wait_for_go(pipeline):
    """Block until GO is pressed: (go, seconds waited). Without a GO the video recording is stopped."""
    state.go_cancelled = False
    state.go_event.clear()
    logger.info("[start-gate] plan ready; press GO to start the motion")
    gate_t0 = time.time()
    go = state.go_event.wait(timeout=300) and not state.go_cancelled
    if not go:
        logger.warning("[start-gate] no GO (cancelled or 300 s passed); run not executed")
        try:
            pipeline.stop_video_recording()
        except Exception:  # noqa: BLE001
            pass
    return go, time.time() - gate_t0


def _robot_ready(pipeline):
    """
    Return (ok, error_msg). Verifies FCI is up and joint state readable.
    """
    if state.executor_running:
        return False, "an execution is already in progress"
    if pipeline is None:
        return False, "Pipeline not initialized"
    robot = getattr(pipeline, "_robot", None)
    if robot is None:
        return False, "Robot driver not constructed yet"
    drv = robot
    for _ in range(4):
        if hasattr(drv, "_connected"):
            if not getattr(drv, "_connected", False):
                return False, "Robot driver reports DISCONNECTED"
            # Probe the ACTUAL link, not the flag: after a controller reboot
            # or network drop the flag stays True while every RTDE read
            # raises deep inside the executor. health_check clears the flag
            # on failure so the state is consistent afterwards.
            health = getattr(drv, "health_check", None)
            if callable(health) and not health():
                return False, (
                    "Robot RTDE link is down (health check failed); "
                    "reconnect via /api/connect_robot"
                )
            break
        drv = getattr(drv, "_robot", None)
        if drv is None:
            break
    try:
        if hasattr(robot, "get_joint_positions"):
            q = robot.get_joint_positions()
            if q is None or len(q) < 6:
                return False, "Robot joint state unavailable"
    except Exception as exc:
        return False, f"Robot read failed: {exc}"
    return True, None


@router.post("/api/execute_approved")
async def execute_approved(req: ExecuteRequest):
    """
    Execute with previously approved detections.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    ok, why = _robot_ready(pipeline)
    if not ok:
        return JSONResponse(
            status_code=400, content={"error": f"Cannot execute: {why}."}
        )
    if state.pending_detections is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No pending detections; run detect_approve first"},
        )
    if not state.execute_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409, content={"error": "execution already in progress"}
        )

    with state.progress_lock:
        state.progress_log.clear()
    loop = asyncio.get_event_loop()

    def _exec_sync():
        t0 = time.time()
        run_timing.reset()
        instruction = req.instruction or state.pending_instruction
        detections = state.pending_detections
        captures = state.pending_captures
        # Only detect_approve fills pending_captures. A run whose detections
        # came from click/box prompts (the "Execute (keep detections)" path)
        # arrives with detections but NO frames.
        # The detections are the operator's work and must be kept; the frames
        # are just planner context, so grab fresh ones. capture() only reads
        # from already-streaming cameras -- no device open, no USB cycle.
        if captures is None:
            try:
                captures = pipeline.capture()
            except Exception as exc:  # noqa: BLE001 - planner can run blind
                logger.warning(
                    "execute_approved: no pending captures and a fresh "
                    "capture failed (%s); planning without a scene image",
                    exc,
                )
                captures = {}

        # run_task enriches its detections; this path must too, or place_in_slot has no slots.
        try:
            pipeline._enrich_with_slots(detections, captures)
        except Exception as exc:  # noqa: BLE001 - a place still works from the tray outline
            logger.warning("execute_approved: slot detection failed: %s", exc)

        # This endpoint bypasses run_task, and a dry run never reaches
        # execute() either -- without this the verdict read below is the
        # PREVIOUS run's, so a dry run that moved nothing reports its pass.
        reset_run_state(getattr(pipeline, "_executor", None))

        scene_img = None
        scene_cam = None
        for cam_name in ["birdview", "sideview"]:
            cam_data = captures.get(cam_name)
            if cam_data and cam_data.get("rgb") is not None:
                scene_img = cam_data["rgb"]
                scene_cam = cam_name
                break
        if scene_img is not None:
            scene_img = annotate_for_planner(scene_img, detections)

        detection_details = build_detection_details(detections)

        # Tier-2 success override. This endpoint bypasses run_task, so without
        # this the task YAML's `verify:` block would apply on /api/execute but
        # silently not here. Cleared when the task has none, so the previous
        # task's predicate cannot leak in.
        _spec = pipeline.task_spec(instruction)
        if pipeline._executor is not None:
            pipeline._executor._task_verify_block = (
                dict(_spec.verify) if _spec is not None and _spec.verify else None
            )

        with pipeline.activity("planning"):
            score = pipeline.plan(
                instruction,
                detections,
                scene_image=scene_img,
                detection_details=detection_details,
                temperature=req.temperature,
                scene_camera=scene_cam,
            )

        if state.current_episode is not None:
            try:
                state.current_episode.discard()
            except Exception:
                pass
            state.current_episode = None

        rec = None
        video_paths = {}
        do_execute = not req.dry_run and pipeline._robot is not None
        # Silent-success guard: a real run that expected a robot but whose
        # driver/executor never constructed would otherwise return
        # exec_results=[] and report success having moved nothing. Gate on
        # the RobotProfile.no_robot flag (server.py sets it from --no-robot)
        # so genuine plan-only --no-robot runs stay valid.
        _profile = getattr(pipeline, "profile", None)
        _expected_robot = not bool(getattr(_profile, "no_robot", False))
        _driver_missing = pipeline._robot is None or pipeline._executor is None
        if not req.dry_run and _driver_missing and _expected_robot:
            return {
                "success": False,
                "error": "robot driver unavailable (perception-only boot)",
                "instruction": instruction,
                "detections": detection_details,
                "plan": score,
                "plan_yaml": yaml.dump(score, default_flow_style=False),
                "execution_results": [],
            }
        _episode_enabled = os.environ.get("SPARK_ENABLE_EPISODE_RECORDER", "0") == "1"
        if do_execute and _episode_enabled:
            try:
                rec = EpisodeRecorder(
                    root=Path(pipeline.config.output_dir).parent,
                    instruction=instruction,
                )
                rec.begin(pipeline._robot)
                state.current_episode = rec
            except Exception:
                rec = None
            try:
                pipeline.start_video_recording(label=instruction)
            except Exception:
                pass

        phase_s = {}
        if do_execute:
            _pretask_home(pipeline)
            if os.environ.get("SPARK_START_GATE", "0") == "1":
                phase_s["pre_go_s"] = round(time.time() - t0, 2)  # Execute pressed -> plan ready and arm homed
                do_execute, waited_s = _wait_for_go(pipeline)
                phase_s["go_wait_s"] = round(waited_s, 2)
                t0 += waited_s  # the wait is not trial time
        if do_execute:
            state.executor_running = True
            try:
                exec_results = pipeline.execute(score, detections)
            finally:
                state.executor_running = False
        else:
            exec_results = []

        if do_execute:
            try:
                vp = pipeline.stop_video_recording()
                if isinstance(vp, dict):
                    video_paths = {k: Path(v) for k, v in vp.items() if v is not None}
                elif vp:
                    video_paths = {"main": Path(vp)}
            except Exception:
                pass

        duration = time.time() - t0
        # The verifier's verdict, not "the last primitive returned True".
        _executor = getattr(pipeline, "_executor", None)
        outcome = getattr(_executor, "verify_outcome", None) if _executor else None
        success = task_success(outcome)
        verify_status = outcome.status if outcome is not None else UNVERIFIED
        # run_task ends with the planner's last-resort rung (look at the scene, accept a finished
        # task, else replan once with the failure). This route bypasses run_task, so the rung is
        # called here; without it a failed check on a UI run simply ends the run.
        if do_execute and exec_results and not user_aborted(exec_results) \
                and os.environ.get("SPARK_LLM_LAST_RESORT", "1") != "0":
            prompts = state.pending_prompts or sorted(
                {base_label(getattr(d, "label", "") or "") for d in (detections or [])} - {""})
            try:
                success, verify_status, outcome = pipeline._llm_last_resort(
                    instruction, prompts, success, verify_status, outcome, None, exec_results)
            except Exception as exc:  # noqa: BLE001 - the rung never masks the run's own verdict
                logger.warning("execute_approved: last-resort rung failed: %s", exc)
            duration = time.time() - t0
        verify_dict = outcome.to_dict() if outcome is not None else None

        pipeline._maybe_save_bt(
            instruction,
            score,
            detections,
            success,
            # Honor the request instead of hardcoding True. None falls through
            # to config.save_to_library (True by default).
            save_to_library=req.save_to_library,
            executed=bool(exec_results),
            plan_source=getattr(pipeline, "_last_plan_source", None),
            bt_hash=getattr(pipeline, "_last_bt_hash", None),
            user_aborted=user_aborted(exec_results),
            verify_status=verify_status,
        )

        if rec is not None:
            result_payload = {
                "success": success,
                "verify": verify_dict,
                "verify_status": verify_status,
                "duration": duration,
                "execution_results": [
                    {
                        "action": r.action_type,
                        "success": r.success,
                        "message": r.message,
                        "duration": r.duration,
                    }
                    for r in exec_results
                ],
            }
            try:
                rec.end(
                    score=score,
                    detections=detections,
                    result=result_payload,
                    video_paths=video_paths,
                )
            except Exception:
                pass

        # This path writes no result.json, so the run's timings go to one line of a JSONL file.
        try:
            record = {"tag": os.environ.get("SPARK_RUN_TAG"), **phase_s}
            if "pre_go_s" in phase_s:
                # With the start gate: duration = pre_go_s + exec_s (the GO wait is in neither).
                record["exec_s"] = round(duration - phase_s["pre_go_s"], 2)
            record.update({
                "time": time.strftime("%Y%m%d_%H%M%S"), "instruction": instruction, "success": success,
                "verify_status": verify_status, "duration": duration, "executed": bool(exec_results),
                "plan_source": getattr(pipeline, "_last_plan_source", None),
                "skills": [[r.action_type, r.success, r.duration] for r in exec_results],
                "timings": run_timing.snapshot(),
            })
            run_log = Path(pipeline.config.output_dir) / os.environ.get("SPARK_RUN_LOG", "approved_runs.jsonl")
            with open(run_log, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:  # noqa: BLE001 - recording never fails a run
            logger.debug("run log not written: %s", exc)

        return {
            "instruction": instruction,
            "success": success,
            # Tri-state verdict. `success` is exactly verify_status == "pass";
            # "unverified" is NOT a failure and must be shown as its own state.
            "verify_status": verify_status,
            "verify": verify_dict,
            "duration": duration,
            "detections": detection_details,
            "plan": score,
            "plan_yaml": yaml.dump(score, default_flow_style=False),
            "execution_results": [
                {
                    "action": r.action_type,
                    "success": r.success,
                    "message": r.message,
                    "duration": r.duration,
                }
                for r in exec_results
            ],
            "episode": rec.summary() if rec is not None else None,
            # Where the executed tree came from.
            "plan_source": getattr(pipeline, "_last_plan_source", None),
            "bt_hash": getattr(pipeline, "_last_bt_hash", None),
        }

    try:
        return await loop.run_in_executor(None, _exec_sync)
    finally:
        state.execute_lock.release()


@router.post("/api/go")
async def start_gate_go(body: dict = None):
    """Release (or cancel) a run that waits at the start gate."""
    state.go_cancelled = bool((body or {}).get("cancel", False))
    state.go_event.set()
    return {"go": not state.go_cancelled}


@router.post("/api/run_bt")
async def run_bt_yaml(body: dict):
    """
    Run a pre-cooked BT (YAML or dict) directly through the executor.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    plan = body.get("plan")
    if plan is None and body.get("yaml"):
        plan = yaml.safe_load(body["yaml"])
    if not plan:
        return JSONResponse(
            status_code=400, content={"error": "Provide 'yaml' or 'plan'"}
        )

    if not state.execute_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409, content={"error": "execution already in progress"}
        )

    detections = state.pending_detections or []
    loop = asyncio.get_event_loop()
    skip_prelude = bool(body.get("skip_prelude", False))

    def _run():
        with state.progress_lock:
            state.progress_log.clear()
        if not skip_prelude:
            _pretask_home(pipeline)
        state.executor_running = True
        try:
            exec_results = pipeline.execute(plan, detections)
        finally:
            state.executor_running = False
        return [
            {
                "action": r.action_type,
                "success": r.success,
                "message": r.message,
                "duration": r.duration,
            }
            for r in exec_results
        ]

    try:
        results = await loop.run_in_executor(None, _run)
    finally:
        state.execute_lock.release()
    return {"execution_results": results}


@router.post("/api/plan")
def generate_plan(req: PlanRequest):
    # Sync (def) handler: FastAPI runs it in the anyio threadpool. The body
    # does capture + SAM3 + LLM inline -- seconds to minutes of blocking
    # work that would freeze the event loop (and with it /api/stop and the
    # camera stream) for the whole call.
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    with pipeline.activity("capturing"):
        captures = pipeline.capture()
    # The camera name travels with the image.
    scene_cam, scene_img = next(
        ((name, c["rgb"]) for name, c in captures.items() if c.get("rgb") is not None),
        (None, None),
    )
    # Prompt source, in order of preference:
    #   1. whatever the caller asked for
    #   2. the task's frozen contract in configs/tasks/*.yaml -- pinned
    #      vocabulary, declared object counts, deterministic instance
    #      ordering, no LLM
    #   3. LLM vision prompt-gen (temperature 0.3, so the vocabulary itself
    #      drifts run to run -- fine for a preview, not for collection)
    #   4. the regex extractor
    # Same source as detect_approve, so a plan preview detects with the same
    # vocabulary an actual run would; otherwise the preview's labels diverge
    # from the BT's and the preview is unexecutable.
    prompts = req.prompts
    spec = pipeline.task_spec(req.instruction) if prompts is None else None
    if spec is None:
        if prompts is None and pipeline._planner is not None and scene_img is not None:
            try:
                with pipeline.activity("planning"):
                    prompts = pipeline._planner.generate_prompts(
                        req.instruction,
                        scene_img,
                    )
            except Exception:
                prompts = pipeline._extract_prompts(req.instruction)
        elif prompts is None:
            prompts = pipeline._extract_prompts(req.instruction)

    resolution = None
    with pipeline.activity("detecting"):
        if spec is not None:
            try:
                detections, _all, resolution, spec, prompts = pipeline.detect_for_task(
                    captures, req.instruction, spec=spec
                )
            except PromptCountMismatch as exc:
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": str(exc),
                        "instruction": req.instruction,
                        "reason": "prompt_count_mismatch",
                    },
                )
        else:
            detections = pipeline.detect(captures, prompts, multi_instance=True)
    # Parity with /api/execute: the model gets the SAME annotated image and the
    # SAME geometry, so a plan previewed here can make the same grasp-strategy
    # call.
    scene_img_annotated = (
        annotate_for_planner(scene_img, detections) if scene_img is not None else None
    )
    with pipeline.activity("planning"):
        score = pipeline.plan(
            req.instruction,
            detections,
            scene_image=scene_img_annotated,
            detection_details=build_detection_details(detections),
            temperature=req.temperature,
            scene_camera=scene_cam,
        )
    return {
        "instruction": req.instruction,
        "detections": serialize_detections(detections),
        "plan": score,
        "plan_yaml": yaml.dump(score, default_flow_style=False),
        "prompts": list(prompts or []),
        "task_spec": None if spec is None else spec.task,
        "detection_counts": None if resolution is None else resolution.counts,
        "plan_source": getattr(pipeline, "_last_plan_source", None),
        "bt_hash": getattr(pipeline, "_last_bt_hash", None),
    }


@router.post("/api/execute")
async def execute_task(req: ExecuteRequest):
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    ok, why = _robot_ready(pipeline)
    if not ok:
        return JSONResponse(
            status_code=400, content={"error": f"Cannot execute: {why}."}
        )
    if not state.execute_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409, content={"error": "execution already in progress"}
        )
    try:
        return await _execute_task_locked(pipeline, req)
    finally:
        state.execute_lock.release()


async def _execute_task_locked(pipeline, req: ExecuteRequest):
    """Body of /api/execute; caller holds state.execute_lock."""
    with state.progress_lock:
        state.progress_log.clear()
    loop = asyncio.get_event_loop()

    if req.record_video:
        try:
            await loop.run_in_executor(
                None,
                lambda: pipeline.start_video_recording(label=req.instruction),
            )
        except Exception:
            pass

    if not req.dry_run:
        await loop.run_in_executor(None, lambda: _pretask_home(pipeline))

    # Per-run closed-loop override: temporarily raise max_task_passes so a
    # single execute can re-perceive and re-plan over what remains. Restored
    # after the run so the server default is untouched.
    _prev_passes = getattr(pipeline.config, "max_task_passes", 1)
    if req.closed_loop_passes is not None:
        pipeline.config.max_task_passes = max(1, int(req.closed_loop_passes))
    _mismatch = None
    result = None
    if not req.dry_run:
        state.executor_running = True
    try:
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.run_task(
                instruction=req.instruction,
                prompts=req.prompts,
                execute=not req.dry_run,
                save_to_library=req.save_to_library,
            ),
        )
    except PromptCountMismatch as exc:
        # A registered task saw the wrong number of objects and its policy is
        # `abort`. Deliberately fatal: a mislabeled episode poisons the
        # training set far worse than losing one episode to a re-run.
        _mismatch = exc
    finally:
        pipeline.config.max_task_passes = _prev_passes
        state.executor_running = False

    if _mismatch is not None:
        return JSONResponse(
            status_code=422,
            content={
                "error": str(_mismatch),
                "instruction": req.instruction,
                "reason": "prompt_count_mismatch",
            },
        )

    _video_path = None
    if req.record_video:
        try:
            _video_path = await loop.run_in_executor(
                None,
                lambda: pipeline.stop_video_recording(),
            )
        except Exception:
            pass

    if not req.dry_run:
        try:
            await loop.run_in_executor(None, lambda: _pretask_home(pipeline))
        except Exception:
            pass

    return {
        "instruction": result.instruction,
        "success": result.success,
        # Tri-state verdict, same as /api/execute_approved. `success` is
        # exactly verify_status == "pass"; "unverified" is not a failure.
        "verify_status": getattr(result, "verify_status", UNVERIFIED),
        "verify": getattr(result, "verify", None),
        "duration": result.duration,
        "detections": result.detections,
        "plan": result.plan,
        "execution_results": result.execution_results,
        "video_path": str(_video_path) if _video_path else None,
        # Plan provenance: which cached tree ran and how it was matched.
        # "llm" here during a collection session means the cache missed.
        "plan_source": result.plan_source,
        "bt_hash": result.bt_hash,
        "label_resolutions": result.label_resolutions,
    }


logger = logging.getLogger("spark_server")

# Merge sub-routers so server.py sees a single router
router.include_router(_detection_router)


@router.post("/api/abort")
async def abort_task():
    """
    Cancel any in-flight task without killing the server.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return {"aborted": False, "reason": "pipeline not running"}
    executor = getattr(pipeline, "_executor", None) or getattr(
        pipeline, "executor", None
    )
    actions = []
    if executor is not None:
        # Through note_abort_requested, never `executor._abort = True`: a bare
        # assignment does not advance the abort epoch, so execute_score's
        # arming check cannot see it and wipes it when the run starts. That is
        # the whole window /api/abort exists to cover -- the executor thread is
        # still in capture/detect/plan and has not reached execute_score yet.
        note = getattr(executor, "note_abort_requested", None)
        if callable(note):
            note()
            actions.append("executor.note_abort_requested")
        elif hasattr(executor, "_abort"):
            executor._abort = True
            actions.append("set executor._abort")
    robot = getattr(pipeline, "_robot", None)
    if robot is not None:
        for method in ("stop_velocity", "stop_motion", "recover_from_errors"):
            fn = getattr(robot, method, None)
            if callable(fn):
                try:
                    fn()
                    actions.append(method)
                except Exception as exc:
                    actions.append(f"{method}:err={exc}")
    return {"aborted": bool(actions), "actions": actions}


@router.post("/api/shutdown")
def shutdown_server():
    """
    Release Kinects + robot and exit the process cleanly.
    """
    pipeline = state.pipeline
    if pipeline is not None:
        try:
            pipeline.shutdown()
        except Exception as exc:
            logger.warning("pipeline shutdown raised: %s", exc)

    def _exit_soon():
        time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=_exit_soon, daemon=True).start()
    return {"shutdown": "ok"}


@router.get("/api/status")
async def get_status():
    pipeline = state.pipeline
    if pipeline is None:
        return {"initialized": False, "error": "Pipeline not started"}
    status = pipeline.get_status()
    status["active_camera"] = state.active_camera
    status["available_cameras"] = get_available_cameras()
    return status


_ROBOT_DISPLAY = {
    "ur10e": {"name": "UR10e", "icon": "\U0001f916"},
    "franka": {"name": "Franka FR3", "icon": "\U0001f916"},
    "bimanual_franka": {"name": "Bimanual Franka", "icon": "\U0001f916"},
}


@router.get("/api/robot")
async def get_robot():
    """
    Return the active robot family + model for UI banner.
    """
    pipeline = state.pipeline
    family = "ur10e"
    model = "UR10e"
    if pipeline is not None and getattr(pipeline, "config", None) is not None:
        family = getattr(pipeline.config, "robot_family", family) or family
        model = getattr(pipeline.config, "robot_model", model) or model
    disp = _ROBOT_DISPLAY.get(family, _ROBOT_DISPLAY["ur10e"])
    return {
        "family": family,
        "model": model,
        "name": disp["name"],
        "icon": disp["icon"],
    }


@router.post("/api/initialize")
async def initialize_pipeline():
    async with state.pipeline_lock:
        if state.pipeline is not None and state.pipeline._initialized:
            return {"status": "already_initialized", **state.pipeline.get_status()}
        # Build through the SAME profile machinery main() uses, for the
        # family the server was launched with. A bare PipelineConfig()
        # ignores --robot and takes the dataclass default family
        # (pipeline_types.py robot_family, "ur10e"), so on any other rig
        # this endpoint would build the wrong grasp orientation, camera roles
        # and workspace.
        profile = None
        if state.launch_config is not None:
            from spark_real.config import load_profile

            profile = load_profile(state.launch_config)
            config = PipelineConfig(**profile.to_pipeline_kwargs())
        else:
            config = PipelineConfig(robot_ip="", use_kinect=True, use_realsense=None)

        # RELEASE THE OLD PIPELINE BEFORE BUILDING A NEW ONE. Reaching here
        # with state.pipeline set means a previous pipeline exists whose
        # initialize() did not finish -- and a half-built pipeline still owns
        # whatever it DID open: PyK4A handles with live capture threads, the
        # RealSense stream, the RTDE connection. Overwriting the reference
        # would leave two handles on one depth MCU, with the old capture
        # threads calling get_capture() forever and nothing releasing them
        # (atexit's _release_devices only walks state.pipeline). shutdown()
        # is idempotent and each close is isolated, so this is safe even on a
        # pipeline that barely got started.
        old = state.pipeline
        if old is not None:
            logger.info("initialize: releasing the previous, uninitialized pipeline")
            try:
                old.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("initialize: previous pipeline shutdown raised: %s", exc)

        # Build and initialize into a LOCAL, and publish only on success.
        # Assigning state.pipeline first would expose a half-constructed
        # object to every concurrent reader (stream pool, /api/status,
        # recorder) for the whole multi-second initialize, and a raising
        # initialize() would leave that wreck installed as the live pipeline.
        # state.pipeline stays None/old until the new one is ready.
        pipeline = SPARKRealPipeline(config, profile=profile)
        state.pipeline = None
        try:
            pipeline.initialize()
        except BaseException:
            logger.critical(
                "initialize: pipeline.initialize() failed; releasing whatever "
                "it opened rather than leaving devices held by an unreachable "
                "object"
            )
            try:
                pipeline.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("initialize: cleanup shutdown raised: %s", exc)
            raise
        state.pipeline = pipeline
        return {"status": "initialized", **pipeline.get_status()}


@router.get("/api/capture")
def capture_images():
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    captures = pipeline.capture()
    result = {}
    for cam_name, data in captures.items():
        rgb = data["rgb"]
        depth = data["depth"]
        rgb_b64 = encode_image(rgb)
        depth_b64 = (
            encode_image(depth_to_colormap(depth)) if depth is not None else None
        )
        result[cam_name] = {
            "rgb": rgb_b64,
            "depth": depth_b64,
            "width": rgb.shape[1],
            "height": rgb.shape[0],
        }
        if cam_name == state.active_camera or state.get_last_frame()[0] is None:
            state.set_last_frame(rgb, depth, cam_name)
    return result


@router.get("/api/progress")
async def get_progress(since: int = 0):
    """
    Get recent pipeline progress log entries.

    `since` is a monotonic sequence number, not an index: pass the `seq`
    of the last entry you rendered and you get strictly what followed it.
    Callers that omit it get the whole retained buffer, which is the
    pre-existing contract, so an old frontend against a new server keeps
    working.

    `next_since` is what the caller should send next time. It is returned
    even when `log` is empty -- otherwise a poll that happens to land on
    an empty window would leave the client's cursor stale.
    """
    with state.progress_lock:
        if since > 0:
            entries = [e for e in state.progress_log if e.get("seq", 0) > since]
        else:
            entries = list(state.progress_log)
        latest = state.progress_seq
    return {"log": entries, "next_since": latest}


@router.delete("/api/progress")
async def clear_progress():
    """
    Clear progress log.

    Deliberately does NOT reset progress_seq. The counter is monotonic for
    the life of the process so that a client holding a pre-clear cursor
    cannot be handed entries it already rendered. `next_since` in the
    response lets the caller resynchronise in the same round trip.
    """
    with state.progress_lock:
        state.progress_log.clear()
        latest = state.progress_seq
    return {"cleared": True, "next_since": latest}
