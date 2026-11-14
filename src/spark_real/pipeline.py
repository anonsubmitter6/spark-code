"""SPARK Real Pipeline - End-to-end orchestrator for physical robot deployment.

Coordinates: Camera capture -> SAM3 perception -> LLM planning -> robot execution.
"""


from PIL import Image
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import Dict, List, Optional
from typing import List, Optional
import json
import logging
import numpy as np
import os
import re
import threading
import time
import yaml


from spark_core.planner import SPARKPlanner
from spark_core.sam3 import SPARKPerception
from spark_real import recording as run_timing
from spark_real.bt_library import BTLibrary
from spark_real.calibration import CameraCalibration
from spark_real.config import SparkConfig, load_profile, output_root
from spark_real.control.executor import ScoreExecutor
from spark_real.control.predicates import UNVERIFIED
from spark_real.control.success_verifier import reset_run_state, task_success
from spark_real.grasp_depth_memory import GraspCalibration
from spark_real.perception.cameras import RealSenseCamera
from spark_real.perception.masks import AnnotationAnchors
from spark_real.perception.prompts import LearnedPrompts
from spark_real.perception_steps import CaptureMixin
from spark_real.perception_steps import FusionMixin
from spark_real.perception_steps import MergeMixin
from spark_real.perception_steps import TaskPromptMixin
from spark_real.perception_steps import _NULL_BACKENDS, _PROPOSER_OFF_LABEL, MAX_ON_TABLE_HEIGHT_M, PROPOSER_MODE_ALWAYS, PROPOSER_MODE_AUTO, PROPOSER_MODE_OFF, PROPOSER_MODES, normalize_proposer_mode
from spark_real.pipeline_init import BimanualInitMixin
from spark_real.pipeline_init import InitMixin
from spark_real.pipeline_init import PipelineConfig, TaskResult
from spark_real.pipeline_init import TaskResult
from spark_real.pipeline_plan import CACHE_PLAN_SOURCES
from spark_real.pipeline_plan import ClosedLoopMixin
from spark_real.pipeline_plan import ExecutionMixin
from spark_real.pipeline_plan import build_detection_details, verify_block_labels
from spark_real.planning.bt_seeds import ensure_seeded
from spark_real.planning.plan_annotate import annotate_for_planner
from spark_real.recording import VideoRecorder
from spark_real.routes import streaming


logger = logging.getLogger(__name__)

# The message ScoreExecutor attaches when the operator hits stop. A robot
# reflex abort carries a different message and IS a real failure.
ABORT_MESSAGE = "Aborted by user"


def user_aborted(exec_results) -> bool:
    """
    True when the operator killed the run mid-execution.

    Accepts both the live ``ActionResult`` objects and the plain dicts the
    /scores page loads back from JSON, so one definition serves both.
    """
    for r in exec_results or []:
        message = r.get("message") if isinstance(r, dict) else getattr(r, "message", None)
        if ABORT_MESSAGE in (message or ""):
            return True
    return False


# Capture saving, video recording, and result persistence.
# Mixed into the pipeline class.
class IOMixin:

    def _save_captures(self, captures: Dict, timestamp: str) -> Dict[str, str]:
        """
        Save captured images to disk.
        """
        paths = {}
        out = Path(self.config.output_dir) / timestamp
        out.mkdir(parents=True, exist_ok=True)
        for cam_name, data in captures.items():
            rgb_path = str(out / f"{cam_name}_rgb.jpg")
            Image.fromarray(data["rgb"]).save(rgb_path)
            paths[f"{cam_name}_rgb"] = rgb_path
            if data["depth"] is not None:
                depth_path = str(out / f"{cam_name}_depth.npy")
                np.save(depth_path, data["depth"])
                paths[f"{cam_name}_depth"] = depth_path
        return paths

    def start_video_recording(self, label: str = "") -> None:
        """
        Begin server-side mp4 recording of every active camera, in
        parallel. One mp4 per camera (sideview, birdview, wrist where
        present), output:
            {output_dir}/videos/{ts}_{label}_{cam}.mp4
        Frames are pulled at ~10 fps directly from each Kinect /
        RealSense via the AzureKinectCamera.read() helper, so the
        recordings are independent of which camera is selected in the
        viewer and survive a closed browser.
        """
        if self._video_recorders:
            # Already recording, don't double-start.
            return

        out_dir = Path(self.config.output_dir).parent / "videos"

        def _make_provider(cam_name):
            def _grab():
                try:
                    rgb, _ = streaming.capture_single_camera(cam_name, need_depth=False)
                except Exception:
                    rgb = None
                return rgb

            return _grab

        # Figure out which cameras are actually live.
        live = []
        if self._kinect is not None:
            live.append("sideview")
        if self._kinect2 is not None:
            live.append("birdview")
        if self._realsense is not None:
            live.append("wrist")

        self._video_recorders = {}
        for cam in live:
            rec = VideoRecorder(_make_provider(cam), out_dir, fps=10)
            rec.start(f"{label}_{cam}" if label else cam)
            self._video_recorders[cam] = rec

    def stop_video_recording(self):
        """
        Stop every active recording and return a dict of {cam: path}.

        For backwards compat with callers that expected a single Path,
        returns just the first path if exactly one stream was recorded.
        """
        if not self._video_recorders:
            return None
        paths = {}
        for cam, rec in self._video_recorders.items():
            try:
                p = rec.stop()
                if p is not None:
                    paths[cam] = p
            except Exception as exc:
                logger.warning("stop_video_recording (%s) failed: %s", cam, exc)
        self._video_recorders = {}
        if len(paths) == 1:
            return next(iter(paths.values()))
        return paths or None

    def _maybe_save_bt(
        self,
        instruction,
        score,
        detections,
        success,
        save_to_library=None,
        executed=True,
        plan_source=None,
        bt_hash=None,
        user_aborted=False,
        verify_status=None,
    ):
        """
        Feed a finished run's outcome back into the BT library.

        This is the capture-once-reuse-forever hinge: a planner-on run that
        verifies successful lands here, gets stored with success=1, and every
        later run of the same task resolves to it with no LLM call.

        Two write paths, deliberately distinct:

          * plan came from the LLM  -> ``add()``: mint (or merge into) the
            entry for this (instruction, tree) pair.
          * plan was SERVED from the library -> ``bump()`` the entry that
            actually ran, and record this run's phrasing as an alias on it.
            ``add()`` here would mint a near-duplicate keyed on the new
            phrasing.

        ``executed=False`` (dry run, no robot, or a plan-only call) never
        writes: the caller's ``success`` defaults to True when there are no
        execution results, so a dry run would otherwise mint a
        cache-eligible entry that nothing ever verified.

        ``verify_status`` is the tri-state verdict. ``unverified`` bumps
        neither success nor fail -- it reinforces nothing and suppresses
        nothing. Treating it as a failure would cool the cache on the
        verifier's own blind spots; treating it as a success would reinforce
        an unverified tree. It does mint the row (success=0, below the
        ``min_success`` floor that makes an entry servable) so the abstain is
        countable on /scores instead of leaving no trace at all.
        """
        if self._bt_library is None:
            return
        # No-adaptation control: a frozen library must not learn from the run
        # it is scoring. See PipelineConfig.bt_frozen.
        from spark_real.pipeline_plan import _bt_frozen

        if _bt_frozen(self):
            logger.info(
                "BT library FROZEN: not recording '%s'; this run plans with "
                "the context trial 1 had", instruction,
            )
            return
        gate = save_to_library if save_to_library is not None else self.config.save_to_library
        if not gate:
            return
        if not executed:
            logger.info(
                "BT library: not recording %r (nothing executed; a dry run "
                "must not mint a cache-eligible entry)",
                instruction,
            )
            return
        if user_aborted:
            # The operator hit stop. That is not a task outcome: penalising the
            # tree would blame it for a decision the operator made, and
            # crediting it would bank a success nobody verified. The /scores
            # page already excludes these runs; the library must agree.
            logger.info(
                "BT library: not recording %r (aborted by user, not a task outcome)",
                instruction,
            )
            return
        if not score:
            return

        objects = []
        for d in detections or []:
            # detections may be ObjectDetection or already plain dicts
            lbl = getattr(d, "label", None) or (d.get("label") if isinstance(d, dict) else None)
            if lbl:
                objects.append(lbl)

        if verify_status == UNVERIFIED:
            # No verdict was reached. Count it -- MINTING the row if this was
            # an LLM-planned run with no hash yet -- so a systematically
            # abstaining verifier is visible on /scores instead of the cache
            # silently going cold. success stays 0, which is below lookup()'s
            # min_success floor, so the row is countable but not yet servable.
            entry = self._bt_library.note_unverified(
                bt_hash, instruction=instruction, score=score, objects=objects
            )
            logger.info(
                "BT library: not recording %r (verification %s -- neither "
                "reinforces nor suppresses)%s",
                instruction,
                UNVERIFIED,
                f"; entry {entry.hash} unverified={entry.unverified}" if entry else "",
            )
            return
        try:
            served = plan_source in CACHE_PLAN_SOURCES and bool(bt_hash)
            if served:
                entry = self._bt_library.bump(bt_hash, bool(success), alias=instruction)
                if entry is not None:
                    logger.info(
                        "BT library: %s %s (%r) -> success=%d fail=%d%s",
                        "reinforced" if success else "penalised",
                        entry.hash,
                        entry.instruction,
                        entry.success,
                        entry.fail,
                        " [promoted]" if entry.promoted else "",
                    )
                    return
                logger.warning(
                    "BT library: served hash %s vanished before feedback; " "falling back to add()",
                    bt_hash,
                )
            entry = self._bt_library.add(instruction, score, objects=objects, success=bool(success))
            logger.info(
                "BT library: stored %s (%r) from %s -> success=%d fail=%d%s",
                entry.hash,
                entry.instruction,
                plan_source or "unknown",
                entry.success,
                entry.fail,
                " [promoted]" if entry.promoted else "",
            )
        except Exception as e:
            logger.warning("BT library save failed: %s", e)

    def _save_result(self, result: TaskResult, timestamp: str):
        """
        Save task result to JSON.
        """
        out = Path(self.config.output_dir) / timestamp
        out.mkdir(parents=True, exist_ok=True)
        result_dict = {
            "instruction": result.instruction,
            "timestamp": result.timestamp,
            "detections": result.detections,
            "plan": result.plan,
            "execution_results": result.execution_results,
            "success": result.success,
            "duration": result.duration,
            "captures": result.captures,
            # Plan provenance. Without it a saved run cannot be attributed to
            # a cached tree vs. a fresh LLM call after the fact, which is the
            # whole point of running collection offline.
            "plan_source": getattr(result, "plan_source", None),
            "bt_hash": getattr(result, "bt_hash", None),
            "label_resolutions": getattr(result, "label_resolutions", None) or {},
            # Tri-state verdict + the evidence behind it. `success` above is
            # exactly (verify_status == "pass"); /scores must be able to tell
            # an unverified run from a failed one.
            "verify_status": getattr(result, "verify_status", "unverified"),
            "verify": getattr(result, "verify", None),
            "timings": run_timing.snapshot(),
        }
        with open(out / "result.json", "w") as f:
            json.dump(result_dict, f, indent=2, default=str)




# Full capture -> detect -> plan -> execute orchestration.
# Mixed into the pipeline class.
class RunMixin(ClosedLoopMixin):

    def run_task(
        self,
        instruction: str,
        prompts: List[str] = None,
        execute: bool = True,
        save_to_library: Optional[bool] = None,
    ) -> TaskResult:
        """
        Full pipeline: capture -> detect -> plan -> execute.

        When ``config.max_task_passes`` > 1 the pipeline runs closed-loop:
        after each behavior tree it re-perceives the scene, decides which
        targets still need handling (generic, any task), and re-plans over
        the fresh detections until the goal holds or the pass budget is
        spent. Single-pass behavior (max_task_passes == 1) is unchanged.
        """
        t0 = time.time()
        run_timing.reset()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("Task: %s", instruction)

        # Clear the previous run's verdict HERE, not in execute(). A dry run,
        # a no-robot run or a capture failure never reaches execute(), and the
        # verdict read at the end of this method would then be the LAST run's
        # -- reporting a stale `pass` as this run's success.
        reset_run_state(getattr(self, "_executor", None))

        # Capture
        logger.info("[1/5] Capturing images...")
        with self.activity("capturing"):
            captures = self.capture()
        if not captures:
            return TaskResult(
                instruction=instruction,
                timestamp=timestamp,
                detections=[],
                plan={},
                execution_results=[],
                success=False,
                duration=time.time() - t0,
            )

        saved_paths = {}
        if self.config.save_captures:
            saved_paths = self._save_captures(captures, timestamp)

        # Detect: two-phase approach:
        #   Phase 1: If no prompts given, ask the planner to look at the scene
        #            and decide what to detect.
        #   Phase 2: Run SAM3 with those prompts, then send annotated
        #            results back to the planner for planning.
        # Collect EVERY fixed view, not the first one that exists. A single
        # camera foreshortens: an elongated tool lying away from that image
        # plane reads far rounder than it is. The API takes several images in
        # one request, so send all and let the model cross-check.
        # scene_cam stays the primary (first) view; the annotation overlay
        # is defined against exactly one camera.
        scene_img_raw = None
        scene_cam = None
        scene_raw_views = []
        for cam_name in ["birdview", "sideview"]:
            cam_data = captures.get(cam_name)
            if cam_data and cam_data.get("rgb") is not None:
                scene_raw_views.append((cam_name, cam_data["rgb"]))
                if scene_img_raw is None:
                    scene_img_raw = cam_data["rgb"]
                    scene_cam = cam_name

        use_multi_instance = False

        # Resolve the cached BT ONCE, up front. Two things ride on it:
        #   - the SAM3 detection prompts are harvested from the tree's own
        #     labels, which skips the LLM prompt-generation call so a
        #     cache-only run is genuinely offline, and
        #   - plan() below is handed the SAME match, so the harvest and the
        #     executed tree can never disagree about which BT this run is.
        # In cache_only mode a miss raises BTCacheMiss out of run_task
        # rather than silently falling through to the planner.
        _plan_mode = self.plan_mode()
        cached_match = None
        if self._bt_library is not None:
            cached_match = self.resolve_cached_bt(instruction)
        if prompts is None and cached_match is not None:
            harvested = self._harvest_cached_prompts(cached_match)
            if harvested:
                prompts = harvested
                use_multi_instance = True  # treat as batch (matches the LLM path)
        # Task-prompt registry (configs/tasks/*.yaml): the task's frozen
        # perception contract -- pinned vocabulary, declared object counts,
        # and a deterministic geometric instance ordering. Sits ahead of BOTH
        # the regex extractor and the LLM because it is the only prompt source
        # that is identical run to run; generate_prompts runs at temperature
        # 0.3, so the prompt VOCABULARY itself drifts, which is exactly the
        # non-repeatability a demo corpus must not have.
        #
        # A miss is not an error: unregistered instructions fall through to
        # the paths below unchanged.
        task_spec = self.task_spec(instruction)
        # Tier-2 success override (spec section 1.5). Set even when the task supplies no
        # prompts, and CLEARED otherwise so the previous task's predicate does
        # not leak into this one. The planner's own `verify:` still wins.
        if self._executor is not None:
            self._executor._task_verify_block = (
                dict(task_spec.verify) if task_spec is not None and task_spec.verify else None
            )
            if task_spec is not None and task_spec.verify:
                logger.info(
                    "[2/5] task registry %r: verify block from %s",
                    task_spec.task,
                    task_spec.source_path,
                )
        if prompts is None and task_spec is not None:
            prompts = list(task_spec.prompts)
            use_multi_instance = True
            logger.info(
                "[2/5] task registry %r: prompts %s (frozen in version "
                "control -- no LLM call)",
                task_spec.task,
                prompts,
            )
        if prompts is None and _plan_mode == "cache_only":
            # Strict offline: never fall through to the planner for prompts.
            # The regex extractor is a legitimate offline fallback; the LLM is
            # not. The task registry above covers the demo tasks without
            # either; this is the backstop for an unregistered instruction.
            prompts = self._extract_prompts(instruction)
            logger.info(
                "[2/5] plan_mode=cache_only: no cached BT labels to harvest; "
                "using regex prompts %s (no LLM call)",
                prompts,
            )
        if prompts is None:
            if self._planner is not None and scene_img_raw is not None:
                logger.info("[2/5] Asking the planner for detection prompts...")
                try:
                    with self.activity("planning"):
                        prompts = self._planner.generate_prompts(instruction, scene_img_raw)
                    use_multi_instance = True  # LLM prompts -> batch task -> multi-instance
                    logger.info("  LLM prompts: %s", prompts)
                except Exception as e:
                    logger.warning(
                        "  LLM prompt generation failed: %s, falling back to regex",
                        e,
                    )
                    prompts = self._extract_prompts(instruction)
            else:
                prompts = self._extract_prompts(instruction)

        # SAM3 detection: multi-instance only for LLM-generated prompts (batch tasks).
        # User-provided prompts get top-1 per prompt (specific objects).
        logger.info(
            "[3/5] Detecting objects: %s (multi_instance=%s)",
            prompts,
            use_multi_instance,
        )
        with self.activity("detecting"):
            if task_spec is not None:
                # Registered task: detect under its contract. Adds the count
                # gate (with the relax-then-alt-prompts escalation) and the
                # declared instance ordering, so "blue block 1" is the same
                # physical block every run and a cached BT keeps binding.
                # Raises PromptCountMismatch when the gate fails and the
                # task's policy is `abort` -- deliberately fatal, because a
                # mislabeled episode poisons a training set silently.
                (
                    detections,
                    all_detections,
                    _resolution,
                    task_spec,
                    prompts,
                ) = self.detect_for_task(
                    captures, instruction, spec=task_spec, prompts=prompts
                )
            else:
                all_detections = self.detect(
                    captures, prompts, multi_instance=use_multi_instance
                )
                detections = self.merge_detections(all_detections)
            # Enrich container-like detections with slot poses + world-
            # frame major-axis orientation so the place_in_slot primitive
            # can drop heterogeneous items into distinct slots with
            # the gripper yaw aligned to the container's long axis.
            try:
                self._enrich_with_slots(detections, captures)
            except Exception as _e:
                logger.warning("slot enrichment failed (continuing without slots): %s", _e)
        logger.info(
            "  Found %d unique objects (%d total across cameras)",
            len(detections),
            len(all_detections),
        )
        for d in detections:
            pos_str = (
                f"({d.position_3d[0]:.3f}, {d.position_3d[1]:.3f}, {d.position_3d[2]:.3f})"
                if d.position_3d is not None
                else "N/A"
            )
            logger.info(
                "  - %s: conf=%.3f, pos=%s, cam=%s",
                d.label,
                d.confidence,
                pos_str,
                getattr(d, "camera", "?"),
            )

        # Plan: send the planner the annotated image + detection details
        # so it can filter false positives and generate the plan.
        logger.info("[4/5] Generating plan...")
        scene_img_annotated = None
        if scene_img_raw is not None:
            cam_dets = [d for d in all_detections if getattr(d, "camera", None) == scene_cam]
            # Planner-specific overlay: adds the OBB rectangle and major-axis
            # arrow on top of the tint+contour, so the model can SEE whether an
            # object has a meaningful long axis. _annotate_image stays as-is
            # for the UI and the detection routes.
            scene_img_annotated = annotate_for_planner(
                scene_img_raw, cam_dets if cam_dets else detections
            )

        # Annotate each view against ITS OWN camera's detections -- an overlay
        # drawn from another camera's pixels would be worse than no overlay.
        scene_views = []
        for _cam, _raw in scene_raw_views:
            _dets = [d for d in all_detections if getattr(d, "camera", None) == _cam]
            if not _dets:
                continue
            try:
                scene_views.append((_cam, annotate_for_planner(_raw, _dets)))
            except Exception as _exc:  # noqa: BLE001
                logger.warning("[plan] could not annotate %s: %s", _cam, _exc)
        if len(scene_views) > 1:
            logger.info(
                "[plan] sending %d camera views to the planner: %s",
                len(scene_views),
                ", ".join(c for c, _ in scene_views),
            )
        elif scene_img_annotated is not None:
            scene_views = [(scene_cam, scene_img_annotated)]

        detection_details = build_detection_details(detections)

        with self.activity("planning"):
            score = self.plan(
                instruction,
                detections,
                scene_image=(scene_views if scene_views else scene_img_annotated),
                detection_details=detection_details,
                scene_camera=scene_cam,
            )
        plan_source = getattr(self, "_last_plan_source", None) or "llm"
        bt_hash = getattr(self, "_last_bt_hash", None)
        logger.info(
            "  Plan (source=%s, bt=%s):\n%s",
            plan_source,
            bt_hash or "-",
            yaml.dump(score, default_flow_style=False),
        )

        exec_results = []
        if execute and self._robot is not None:
            logger.info("[5/5] Executing on robot...")
            exec_results = self.execute(score, detections)
        else:
            logger.info("[5/5] Skipping execution (dry run or no robot)")

        exec_results, goal_unhandled = self._closed_loop_passes(
            instruction, prompts, score, exec_results, execute, plan_source, bt_hash
        )

        # Task success is the verifier's verdict and nothing else. An
        # ExecutionResult says a primitive RAN, not that the task was
        # achieved: a "last action succeeded" fallback banks a fork dropped
        # beside the tray as a success.
        outcome = getattr(self._executor, "verify_outcome", None) if self._executor else None
        success = task_success(outcome)
        verify_status = outcome.status if outcome is not None else UNVERIFIED
        # Closed-loop ground truth can only ever REMOVE a pass: a target still
        # outside the destination means the task did not succeed, whatever the
        # predicate said.
        if goal_unhandled:
            if success:
                logger.warning(
                    "Verifier said pass but %d target(s) are still outside the "
                    "destination %s; downgrading to fail",
                    len(goal_unhandled),
                    goal_unhandled,
                )
            success = False
            verify_status = "fail"

        success, verify_status, outcome = self._llm_last_resort(
            instruction, prompts, success, verify_status, outcome, goal_unhandled, exec_results
        )
        duration = time.time() - t0

        result = TaskResult(
            instruction=instruction,
            timestamp=timestamp,
            detections=[
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "position_3d": (
                        d.position_3d.tolist()
                        if isinstance(d.position_3d, np.ndarray)
                        else d.position_3d
                    ),
                    "centroid_2d": d.centroid_2d,
                    "depth_meters": d.depth_meters,
                    "orientation_angle": d.orientation_angle,
                    "aspect_ratio": d.aspect_ratio,
                    "obb_minor_m": float(getattr(d, "obb_minor_m", 0.0) or 0.0),
                }
                for d in detections
            ],
            plan=score,
            execution_results=[
                {
                    "action": r.action_type,
                    "success": r.success,
                    "message": r.message,
                    "duration": r.duration,
                }
                for r in exec_results
            ],
            success=success,
            duration=duration,
            captures=saved_paths,
        )

        # Plan provenance for the run record.
        result.plan_source = plan_source
        result.bt_hash = bt_hash
        result.label_resolutions = dict(getattr(self, "_last_label_resolutions", None) or {})
        # The verdict itself, so /scores and the recorded episode can show WHY
        # a run passed or failed rather than just a bare boolean.
        result.verify = outcome.to_dict() if outcome is not None else None
        result.verify_status = verify_status

        self._task_history.append(result)
        self._save_result(result, timestamp)
        # Automatic success feedback. ``executed`` is the guard that keeps a
        # dry run out of the cache; ``plan_source``/``bt_hash`` route a
        # cache-served run to bump()+alias instead of minting a duplicate.
        self._maybe_save_bt(
            instruction,
            score,
            detections,
            success,
            save_to_library=save_to_library,
            executed=bool(exec_results),
            plan_source=plan_source,
            bt_hash=bt_hash,
            user_aborted=user_aborted(exec_results),
            verify_status=verify_status,
        )
        logger.info(
            "Task complete (%.1fs, verify=%s, success=%s, plan_source=%s)",
            duration,
            verify_status,
            success,
            plan_source,
        )
        return result

    def _harvest_cached_prompts(self, cached_match):
        """SAM3 prompts from a cached BT's labels and verify block, or None.

        BTs carry instance-suffixed labels ("knife handle 1"); SAM3 wants the
        de-instanced prompt, so trailing " <N>" is stripped and the list
        deduplicated in insertion order.
        """
        cached_instr = cached_match.entry.instruction
        cached_score = cached_match.score
        if cached_score is not None:
            # BTs carry instance-suffixed labels like "knife handle 1",
            # "knife handle 2". SAM3 wants the de-instanced prompt
            # ("knife handle") and produces those numbered labels
            # itself in multi-instance mode. Strip trailing " <N>"
            # and dedupe while preserving insertion order so the
            # prompt list is deterministic.
            seen = set()
            cached_labels = []

            def _harvest(node):
                if not isinstance(node, dict):
                    return
                p = node.get("params") or {}
                # Pull any label-shaped param so the harvested
                # prompt set covers every label the BT will look up
                # at runtime (place_in_slot's container_label, e.g.
                # "tray"), not just the grasp's keypoint_label.
                for k in (
                    "keypoint_label",
                    "container_label",
                    "target_label",
                    "area_label",
                    "source_label",
                ):
                    lbl = p.get(k)
                    if not lbl:
                        continue
                    base = re.sub(r"\s+\d+$", "", lbl).strip()
                    if base and base not in seen:
                        seen.add(base)
                        cached_labels.append(base)
                for child in node.get("children") or []:
                    _harvest(child)

            _harvest((cached_score or {}).get("tree", {}))
            # Predicate labels sit outside the tree. Without them a
            # cache-served run detects only what it manipulates, and the
            # verifier abstains in every camera -> `unverified` forever.
            for lbl in sorted(verify_block_labels(cached_score)):
                base = re.sub(r"\s+\d+$", "", lbl).strip()
                if base and base not in seen:
                    seen.add(base)
                    cached_labels.append(base)
            if cached_labels:
                logger.info(
                    "[2/5] cache %s: harvested prompts %s from cached BT "
                    "(matched %r) -- no prompt-generation LLM call",
                    cached_match.source,
                    cached_labels,
                    cached_instr,
                )
            return cached_labels or None
        return None




class PerceptionMixin(FusionMixin, CaptureMixin, MergeMixin, TaskPromptMixin):

    # Count-gate outcome of the most recent spec-driven merge_detections.
    # Class-level default because this is a mixin with no __init__ of its own.
    _last_resolution = None
    _last_captures = None
    _last_captures_t = None
    _prompt_registry = None

    # Detector fusion / ASPIRE quality gate (config-gated, default OFF)
    # `None` = not resolved yet, `False` = resolved and OFF, otherwise the gate.
    _fusion_gate = None
    _fusion_gate_error = None

    FUSION_ENV = "SPARK_DETECTION_FUSION"
    PROPOSER_MODE_ENV = "SPARK_PROPOSER_MODE"


    MERGE_PROXIMITY_M = 0.15  # 15cm, loose enough for cross-camera calibration drift


    # Container labels whose detections should receive slot poses +
    # world-frame major-axis orientation. Slot detection runs in birdview
    # using surface-normal clustering and falls back to stripe-along-axis
    # when no compartments are separable. Matched case-insensitively
    # against the detection's base label (digit suffix from multi-
    # instance numbering stripped).
    SLOT_CONTAINER_LABELS = frozenset(
        {
            "tray",
            "grey tray",
            "silverware tray",
            "plate",
            "dish",
            "bowl",
            "dustpan",
            "pan",
            "drawer",
            "container",
            "compartment",
            "box",
        }
    )


    # Camera preference for the frame the LLM is shown. Birdview first for the
    # same reason merge_detections trusts it first: it is the view the merged
    # detection's XY comes from, so it is the view whose vocabulary has to work.
    _PROPOSAL_CAM_ORDER = ("birdview", "sideview", "wrist")


try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def resolve_config_path(path_str) -> Path:
    """
    Resolve a config path to an absolute one.

    Absolute (and ``~``) paths are returned as given; relative ones are
    anchored on the parent of output_root() (SPARK_OUTPUT_DIR), so the
    BT library and learned state sit next to the run outputs.
    """
    p = Path(str(path_str)).expanduser()
    if not p.is_absolute():
        p = output_root().parent / p
    return p.resolve()


# Cached result of the RealSense availability probe. Probed ONCE per
# process: on the RSUSB librealsense backend (mandatory on this host, see
# /etc/modprobe.d/blacklist-uvcvideo.conf) every rs.context() re-inits
# libusb and re-walks the bus, and repeated enumeration is exactly the
# lifecycle churn that corrupts the fragile xHCI controller here.
_REALSENSE_PROBE_RESULT: Optional[bool] = None


def _realsense_device_available() -> bool:
    """
    Return True iff pyrealsense2 enumerates at least one camera.

    Used to default-init the wrist RealSense when one is plugged in,
    without requiring an explicit flag or hardcoded /dev path. The bus is
    probed once per process and the answer cached; a camera plugged in
    later needs a server restart to be picked up (it always did in
    practice: this is only consulted during initialize()).
    """
    global _REALSENSE_PROBE_RESULT
    if _REALSENSE_PROBE_RESULT is not None:
        return _REALSENSE_PROBE_RESULT
    if rs is None:
        return False
    try:
        _REALSENSE_PROBE_RESULT = len(rs.context().query_devices()) > 0
    except Exception:
        _REALSENSE_PROBE_RESULT = False
    return _REALSENSE_PROBE_RESULT


run_timing.install()


class SPARKRealPipeline(
    InitMixin, BimanualInitMixin, PerceptionMixin, ExecutionMixin, RunMixin, IOMixin
):
    """
    End-to-end pipeline for SPARK physical deployment.
    """

    def __init__(self, config: PipelineConfig = None, profile=None):
        self.config = config or PipelineConfig()
        # Resolved per-system config (gripper, cameras, home, workspace): the
        # single source each subsystem reads. Built from the family when not
        # supplied so non-server callers (routes, tests) still get one.
        if profile is None:
            profile = load_profile(SparkConfig(family=self.config.robot_family))
        self.profile = profile
        # SPARK_VELOCITY_OVERRIDE env var lets a single run drop the
        # Cartesian motion speed (m/s) without editing the default. Lower
        # nominal speed gives Ruckig more headroom to smooth the path near
        # a singularity and avoid joint-velocity reflexes during transit.
        try:
            _vel_env = os.environ.get("SPARK_VELOCITY_OVERRIDE")
            if _vel_env is not None and float(_vel_env) > 0.0:
                self.config.velocity = float(_vel_env)
                logger.info(
                    "SPARK_VELOCITY_OVERRIDE: capping Cartesian "
                    "velocity at %.3f m/s (default 0.25)",
                    self.config.velocity,
                )
        except (ValueError, TypeError):
            pass
        self._kinect = None  # Side view (master Kinect)
        self._kinect2 = None  # Bird view (subordinate Kinect)
        self._multi_k4a = None  # MultiAzureKinectManager for synced capture
        self._kinect_serial = None
        self._kinect2_serial = None
        self._realsense: Optional[RealSenseCamera] = None
        self._perception: Optional[SPARKPerception] = None
        self._planner: Optional[SPARKPlanner] = None
        self._robot = None
        self._executor: Optional[ScoreExecutor] = None
        self._kinect_cal: Optional[CameraCalibration] = None
        self._kinect2_cal: Optional[CameraCalibration] = None
        self._realsense_cal: Optional[CameraCalibration] = None
        # Unified role-keyed view over the single-arm cameras, built in
        # initialize() from the already-open slots. Private on purpose: see
        # _build_unified_camera_registry for why single-arm must not expose a
        # public `camera_registry` (routes/streaming.py keys off that name).
        self._camera_registry = None
        self._initialized = False
        self._task_history: List[TaskResult] = []
        self._video_recorder: Optional[VideoRecorder] = None
        # Multi-camera recording: one VideoRecorder per live camera.
        self._video_recorders: dict = {}
        # Voyager-style memory: append every successful BT score so future
        # plans can few-shot from prior real-robot wins -- and, once a task
        # has one verified run, serve that exact tree back with no LLM call.
        #
        # The root is derived from output_dir, never from the cwd: a relative
        # output_dir is anchored by resolve_config_path. bt.library_dir
        # overrides, and an absolute one is honoured verbatim.
        self._output_root = resolve_config_path(self.config.output_dir)
        self._bt_library_root = (
            resolve_config_path(self.config.bt_library_dir)
            if self.config.bt_library_dir
            else self._output_root.parent / "bt_library"
        )
        # Logged at INFO on construction so a mid-collection cwd change shows
        # up in the log as a changed root, instead of looking like data loss.
        logger.info(
            "BT library root: %s (output_dir=%r, cwd=%s)",
            self._bt_library_root,
            self.config.output_dir,
            os.getcwd(),
        )
        self._bt_library = BTLibrary(
            self._bt_library_root,
            min_similarity=self.config.bt_min_similarity,
            auto_promote_after=self.config.bt_auto_promote_after,
        )
        # Prime it from the packaged seed YAMLs (configs/bt_seeds/) so a fresh
        # checkout serves the demo tasks from cache on its very first request.
        # Idempotent; never overwrites a live entry with a higher success count.
        # Skipped while collecting a fresh corpus -- see bt_seed_on_start.
        if self.config.bt_seed_on_start:
            ensure_seeded(self._bt_library, seed_dir=self.config.bt_seed_dir)
        else:
            logger.info(
                "BT seeds: not installed (bt.seed_on_start=false); tasks without a "
                "live entry will be planned, not served from cache"
            )
        # Per-object grasp depth-correction memory. Same anchoring as the BT
        # library: this is persistent learned state, so it must not follow the
        # cwd either.
        # Learned SAM3 prompts (spark_bench's tuned-prompts tier): phrases the
        # LLM rung proposed and the count gate accepted, persisted per
        # task+group so the next run tries them before paying another API
        # call. Same anchoring rule as the BT library and grasp calibration:
        # learned state must not follow the cwd.
        self._learned_prompts = LearnedPrompts(
            self._output_root.parent / "learned_prompts.json"
        )
        # Operator click/box annotations, saved per task. Same anchoring rule.
        self._annotation_anchors = AnnotationAnchors(
            self._output_root.parent / "annotation_anchors.json"
        )
        self._grasp_calibration = (
            GraspCalibration(self._output_root.parent / "grasp_calibration.json")
            if self.config.use_grasp_calibration
            else None
        )
        # Activity tracking surfaced via /api/status for the UI status pill.
        # Stack so nested/overlapping activities don't lose state.
        self._activity_stack: List[str] = []
        self._activity_lock = threading.Lock()
        # Per-device read locks. AzureKinectCamera.read() only reads cached
        # numpy arrays under the camera's own lock; the ONLY thread that
        # touches the k4a handle is the capture thread, which these locks do
        # not gate at all. What they buy: one consumer at a time sees a
        # consistent rgb/depth/ir triple, and the deferred rectification in
        # read() is not run twice on the same frame.
        # Anything that needs real exclusion from the device must gate on
        # the capture thread (close() joins it), not on these.
        self._kinect_read_lock = threading.Lock()
        self._kinect2_read_lock = threading.Lock()
        self._realsense_read_lock = threading.Lock()
        # Wrist-camera offset from TCP, family-aware.  Use the per-family
        # default unless the config supplies an explicit override.
        self._wrist_tool_offset = self._resolve_wrist_tool_offset()

    def initialize(self):
        """
        Initialize all pipeline components.
        """
        logger.info("Initializing SPARK Real Pipeline...")

        if self.config.use_kinect:
            self._init_kinects()

        if self.config.use_realsense is None:
            self.config.use_realsense = _realsense_device_available()
            logger.info(
                "RealSense auto-detect: %s",
                "device found" if self.config.use_realsense else "no device, skipping",
            )

        if self.config.use_realsense:
            logger.info("Opening RealSense...")
            try:
                self._realsense = RealSenseCamera(
                    serial=self.config.realsense_serial or None,
                    color_only=getattr(self.config, "realsense_color_only", True),
                )
                self._realsense.open()
                self._realsense_cal = CameraCalibration(
                    name="wrist",
                    width=self._realsense.config.width,
                    height=self._realsense.config.height,
                    fx=self._realsense.config.fx,
                    fy=self._realsense.config.fy,
                    cx=self._realsense.config.cx,
                    cy=self._realsense.config.cy,
                    depth_scale=float(getattr(self, "_wrist_depth_scale", 1.0)),
                    depth_offset=float(getattr(self, "_wrist_depth_offset", 0.0)),
                )
                logger.info(
                    "RealSense ready (%dx%d, depth_scale=%.4f, " "depth_offset=%.1fmm)",
                    self._realsense_cal.width,
                    self._realsense_cal.height,
                    self._realsense_cal.depth_scale,
                    self._realsense_cal.depth_offset * 1000.0,
                )
            except Exception as e:
                # Wrist camera is optional, used only for grasp refinement.
                # Bird/sideview Kinects can drive the pipeline on their own.
                logger.warning(
                    "RealSense unavailable (%s), continuing without "
                    "wrist camera. Grasp refinement will be skipped.",
                    e,
                )
                self._realsense = None
                self._realsense_cal = None

        self._load_handeye_calibrations()

        # Unified, role-keyed view over the cameras opened above.
        self._build_unified_camera_registry()

        logger.info("Loading SAM3 perception...")
        self._perception = SPARKPerception(sam3_threshold=self.config.sam3_threshold)
        self._perception.load_models(load_da3=not self.config.use_hardware_depth)
        logger.info("SAM3 perception ready")

        # Backend and model come from SPARK_LLM_BACKEND and SPARK_LLM_MODEL (spark_core.llm).
        self._planner = SPARKPlanner(
            robot_family=self.config.robot_family,
            temperature=self.config.planner_temperature,
        )
        logger.info(
            "Planner ready (model=%s, family=%s)",
            self._planner.model,
            self.config.robot_family,
        )

        # Robot is optional; can run perception-only
        if self.config.robot_ip:
            self._init_robot()

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        self._initialized = True
        logger.info("Pipeline initialization complete")

    @property
    def task_history(self) -> List[TaskResult]:
        """
        Get history of all tasks run in this session.
        """
        return self._task_history

    @contextmanager
    def activity(self, name: str):
        """
        Context manager: mark the pipeline as busy with ``name``.

        Stacks so that nested or overlapping activities still report
        correctly. The latest pushed name wins for the status field.
        Robot execution (``executor._running``) is reported separately
        and overrides any pushed name.
        """
        with self._activity_lock:
            self._activity_stack.append(name)
        try:
            yield
        finally:
            with self._activity_lock:
                # Pop the matching entry, usually the last one, but be
                # defensive in case something else mutated the stack.
                try:
                    idx = (
                        len(self._activity_stack)
                        - 1
                        - self._activity_stack[::-1].index(name)
                    )
                    self._activity_stack.pop(idx)
                except ValueError:
                    pass

    @property
    def activity_state(self) -> str:
        """
        One of: idle | capturing | detecting | planning | executing.

        Robot execution beats any pushed activity (because it's the
        operator-visible motion). Otherwise the most recently pushed
        activity wins, falling back to ``idle``.
        """
        if self._executor is not None and self._executor._running:
            return "executing"
        with self._activity_lock:
            if self._activity_stack:
                return self._activity_stack[-1]
        return "idle"

    def get_status(self) -> Dict:
        """
        Get current pipeline status for the frontend.
        """
        # Per-camera frame age: 'connected' only says the handle exists; a
        # wedged Kinect is connected AND frozen. The UI shows a dead camera
        # instead of a frozen picture using this.
        frame_age = {}
        for name, dev in (
            ("sideview", self._kinect),
            ("birdview", self._kinect2),
            ("wrist", self._realsense),
        ):
            if dev is not None and hasattr(dev, "last_frame_age_s"):
                try:
                    frame_age[name] = dev.last_frame_age_s()
                except Exception:  # noqa: BLE001 - status must never raise
                    frame_age[name] = None
        return {
            "initialized": self._initialized,
            "kinect_connected": self._kinect is not None,
            "kinect2_connected": self._kinect2 is not None,
            "realsense_connected": self._realsense is not None,
            "camera_frame_age_s": frame_age,
            "robot_connected": self._robot is not None,
            "perception_ready": self._perception is not None
            and self._perception._sam3 is not None,
            "planner_ready": self._planner is not None,
            "tasks_completed": len(self._task_history),
            "robot_ip": self.config.robot_ip,
            "robot_family": self.config.robot_family,
            "robot_model": self.config.robot_model,
            "activity": self.activity_state,
        }

    def _stop_reader_threads(self):
        """
        Quiesce every background thread that reads a device or the robot.

        Called first from shutdown(), so no close() or disconnect() below can
        land while one of these threads still holds a handle. Each is isolated:
        a recorder that will not stop must not prevent the devices being
        released.
        """
        if getattr(self, "_video_recorders", None):
            try:
                self.stop_video_recording()
            except Exception as exc:  # noqa: BLE001
                logger.warning("stop_video_recording raised during shutdown: %s", exc)

    def shutdown(self):
        """
        Clean shutdown of all components.

        Each device close is isolated so a hang or exception in one
        (a robot disconnect can hang) does not prevent the others
        from being released. Cameras go first because their depth MCUs
        brick if left mid-stream; the robot just sits idle if disconnect
        is skipped.
        """
        logger.info("Shutting down pipeline...")
        # STOP AND JOIN THE READERS BEFORE CLOSING WHAT THEY READ.
        # The per-camera VideoRecorder threads poll
        # streaming.capture_single_camera at 10 Hz for the whole task;
        # unjoined they would read cameras that shutdown is concurrently
        # releasing, and their mp4s would never be encoded (stop() is the only
        # thing that writes the file). Joining here is also what makes the
        # ordering assertable.
        self._stop_reader_threads()
        # Quiesce the wrist camera's recovery machinery FIRST (non-blocking):
        # closing a Kinect can take seconds, and a RealSense hardware_reset
        # firing mid-Kinect-close is exactly the overlapping-lifecycle
        # pattern that corrupts the shared xHCI controller. request_stop()
        # only sets flags; the actual close happens in the loop below, after
        # the Kinects.
        # getattr throughout: /api/initialize calls shutdown() on a pipeline
        # whose __init__ or initialize() did NOT finish, to release the
        # devices it did manage to open. A bare self._realsense on such an
        # object would raise AttributeError and abort the teardown before the
        # Kinects are closed.
        realsense = getattr(self, "_realsense", None)
        if realsense is not None and hasattr(realsense, "request_stop"):
            try:
                realsense.request_stop()
            except Exception as exc:
                logger.warning("realsense request_stop raised: %s", exc)
        devices = [
            ("kinect (sideview)", getattr(self, "_kinect", None)),
            ("kinect2 (birdview)", getattr(self, "_kinect2", None)),
            ("realsense", realsense),
        ]
        # Bimanual: every camera opened from the roster, each closed once.
        registry = getattr(self, "camera_registry", None)
        for role, entry in (registry or []):
            if all(entry.device is not d for _, d in devices):
                devices.append((role, entry.device))
        for label, dev in devices:
            if dev is None:
                continue
            try:
                dev.close()
            except Exception as exc:
                logger.warning("%s close raised: %s", label, exc)
        robot = getattr(self, "_robot", None)
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as exc:
                logger.warning("robot disconnect raised: %s", exc)
        self._initialized = False
        logger.info("Pipeline shutdown complete")
