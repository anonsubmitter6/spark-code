"""Replay of the last pick cycle after an evidenced verification failure."""


from typing import Any, List, Optional, Sequence, Tuple
from typing import Optional, Sequence
import logging
import numpy as np
import os


from spark_real.control.predicates import FAIL, PASS
from spark_real.control.predicates import FAIL, VerifyOutcome


# Pre-retry retreat. The lift is vertical only and happens before any lateral
# transit, so the tool clears the container it just failed over.
RETRY_LIFT_M = 0.10

RETRY_MIN_CLEAR_Z = 0.05

# Action types that name a keypoint the replay will drive to.
_KEYPOINT_ACTIONS = ("move_to_keypoint", "grasp_se3")


def llm_verify_enabled(executor) -> bool:
    """The planner's image check runs unless SPARK_VERIFY_LLM=0 or there is no planner."""
    planner = getattr(getattr(executor, "_pipeline", None), "_planner", None)
    return planner is not None and os.environ.get("SPARK_VERIFY_LLM") != "0"


def _keypoint_label(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    params = action.get("params") or {}
    return params.get("keypoint_label", "") or ""


def _replay_labels(actions: Sequence[dict]) -> List[str]:
    """Keypoint labels a replay segment drives to, in order, de-duplicated."""
    out: List[str] = []
    for action in actions or ():
        if not isinstance(action, dict):
            continue
        if action.get("type", action.get("name", "")) not in _KEYPOINT_ACTIONS:
            continue
        label = _keypoint_label(action)
        if label and label not in out:
            out.append(label)
    return out


def _detection_fingerprint(detection_map: Any, label: str) -> Optional[Tuple]:
    """``(entry, xyz)`` for ``label`` in the detection map, or ``None``.

    The entry OBJECT is kept, not its ``id``: holding the reference is what
    stops a freed dict's address being recycled by its replacement and read
    back as "unchanged".
    """
    get = getattr(detection_map, "get", None)
    if not callable(get):
        return None
    entry = get(label)
    if entry is None:
        return None
    if hasattr(entry, "get"):
        pos = entry.get("position_3d")
    else:
        pos = getattr(entry, "position_3d", None)
    try:
        xyz = None if pos is None else tuple(float(v) for v in pos)
    except (TypeError, ValueError):
        xyz = None
    return (entry, xyz)


def _is_fresh(before: Optional[Tuple], after: Optional[Tuple]) -> bool:
    """Did a re-detection actually write something for this label?

    ``execution_recovery.redetect_single`` leaves the stale entry untouched
    when the detector finds nothing ("not found, keeping old position"), which
    is otherwise indistinguishable from a successful re-read.
    """
    if after is None:
        return False
    if before is None:
        return True
    return after[0] is not before[0] or after[1] != before[1]


def _fail_has_evidence(outcome: VerifyOutcome) -> bool:
    """Did this ``fail`` come from evidence, or was it synthesised?

    ``verification.on_missing: fail`` turns "no predicate derivable" (a
    structural abstain, no capture, no vote) into a fail. Replaying a pick
    cycle off that opens the gripper on the strength of nothing at all, which
    is the same "abstain moves the robot" bug the retry gate exists to stop.
    A failed gate or a camera voting fail is evidence; nothing else is.
    """
    if any(v is False for v in (outcome.gates or {}).values()):
        return True
    return any(getattr(v, "vote", "") == FAIL for v in (outcome.votes or ()))


logger = logging.getLogger(__name__)


class ReplayMixin:
    """Replay of the last pick cycle after an evidenced verification failure."""

    def _replay_last_pick_cycle(self, score: dict, actions: list, max_attempts: int):
        """Opt-in recovery replay. Only ever reached on an explicit ``fail``.

        Bounded by ``max_attempts``. Every attempt establishes the arm state
        and re-perceives BEFORE it moves laterally; either precondition failing
        ends the loop rather than degrading into a blind replay.
        """
        last_pick_start = self._find_last_pick_cycle(actions)
        if last_pick_start is None:
            logger.info("Verification fail but no pick cycle to replay")
            return

        retry_actions = actions[last_pick_start:]
        pick_label = _keypoint_label(retry_actions[0])
        labels = _replay_labels(retry_actions)
        if not pick_label:
            logger.info("Verification fail but the pick cycle names no keypoint")
            return

        # A SUCCESSFUL release already added the object to _placed_labels and
        # cleared _last_pick_label. Both bits of bookkeeping block this retry:
        #   * executor_motion short-circuits move_to_keypoint with "already in
        #     container, skipping pick approach" for a placed label, so the
        #     replayed grasp would close on air;
        #   * the pre-approach re-detect is skipped for a placed label, so the
        #     replay would aim at the plan-time pose.
        # Un-place it for the duration of the retry; a completed retry re-marks
        # it in _run_replay_actions.
        if pick_label in self._placed_labels:
            self._placed_labels.discard(pick_label)
            logger.info(
                "Retry: un-placed '%s' so the replay re-detects it and does not "
                "skip the pick approach",
                pick_label,
            )

        for attempt in range(max_attempts):
            if self._abort:
                break

            logger.warning(
                "Verification FAILED (attempt %d/%d), replaying the last pick "
                "cycle -- this opens the gripper",
                attempt + 1,
                max_attempts,
            )

            if not self._establish_retry_ready_state():
                break
            if not self._reperceive_for_retry(pick_label, labels):
                break

            self._run_replay_actions(retry_actions, attempt, pick_label)
            if self._abort:
                break

            # Same rungs as the first check: without them a replay that ends "unverified" never
            # reaches the planner's image check.
            outcome = self._fold_in_rungs(self._verify_task_completion(score, actions), score)
            self.verify_outcome = outcome
            if outcome.status == PASS and llm_verify_enabled(self) and not self._llm_sees_success(score):
                # The scene was disturbed by a failed placement and a replay; a camera pass here is
                # confirmed by the planner's look. A "no" makes it a fail, and the last rung decides.
                outcome.status = FAIL
                outcome.reason = f"{outcome.reason}; planner image check after the replay: not done"
                self.verify_outcome = outcome
                logger.info("Verification passed on replay %d by the cameras, but the planner's image "
                            "check says not done; fail (the last-resort rung decides)", attempt + 1)
                return
            if outcome.status == PASS:
                logger.info("Verification passed on replay %d", attempt + 1)
                return
            if outcome.status != FAIL or not _fail_has_evidence(outcome):
                # Same rule as the entry gate: only an evidenced fail continues.
                logger.info(
                    "Verification %s after replay %d; stopping", outcome.status, attempt + 1
                )
                return

        logger.warning("Giving up after %d replay attempt(s)", max_attempts)

    def _establish_retry_ready_state(self) -> bool:
        """Bring the arm to the ONE state a pick replay may start from.

        The replayed tree opens with ``move_to_keypoint -> grasp``: a lateral
        transit, a descent, and a close. That is safe only from jaws open, jaws
        EMPTY, tool lifted clear. After a failed place the arm may be open over
        the container, closed on nothing, still holding the object, or parked
        mid-transit, so none of it is assumed here; it is commanded and then
        checked. Returns False when the state could not be established, in
        which case the caller must not move.
        """
        self._check_abort()
        if getattr(self, "_holding", False):
            logger.warning(
                "Retry: the executor still believes it holds an object; opening "
                "at the current pose (the lowest drop available) before transit"
            )

        # Open in place. If something is still in the jaws this drops it, and
        # the current pose is the shortest fall on offer; lifting first would
        # only increase it.
        self.robot.open_gripper()
        self._abort_sleep(0.3)
        self._holding = False

        # Confirm empty. A soft object (the plushie) can stay seated in a
        # partly-open jaw; re-approaching with it aboard drags it through the
        # descent. Same signal _run_actions' own failure recovery trusts.
        verify_grasp = getattr(self, "_verify_grasp", None)
        if callable(verify_grasp):
            try:
                if verify_grasp():
                    logger.error(
                        "Retry ABANDONED: the gripper still reads HOLDING after "
                        "an open; not replaying a pick cycle with a loaded jaw"
                    )
                    self._holding = True
                    return False
            except Exception as exc:
                logger.warning(
                    "Retry: grip state unreadable (%s); treating the jaws as open",
                    exc,
                )

        # Lift clear. Vertical only, and before any lateral motion.
        self._check_abort()
        current = self._get_current_position()
        current[2] = max(current[2] + RETRY_LIFT_M, RETRY_MIN_CLEAR_Z)
        self._move_to(current, self.GRASP_ORIENTATION)
        return True

    def _reperceive_for_retry(self, pick_label: str, labels: Sequence[str]) -> bool:
        """Re-detect everything the replay drives to. False = do not move.

        The object is not where the plan said it was (that is why the place
        failed), so replaying against the plan-time detection map re-commits
        the same error. The pick target must therefore be re-detected AND the
        re-detection must have landed: ``redetect_single`` silently keeps the
        old position when the detector finds nothing, so each entry is
        fingerprinted either side of the call.
        """
        det_map = getattr(self, "detection_map", None)
        if det_map is None:
            logger.error("Retry ABANDONED: no detection map to refresh")
            return False

        fresh = False
        for label in labels:
            before = _detection_fingerprint(det_map, label)
            self._redetect_single(label)
            landed = _is_fresh(before, _detection_fingerprint(det_map, label))
            logger.info(
                "Retry re-perceive '%s': %s", label, "fresh" if landed else "STALE"
            )
            if label == pick_label:
                fresh = landed

        if not fresh:
            logger.error(
                "Retry ABANDONED: '%s' could not be re-perceived; replaying "
                "against the plan-time pose would repeat the miss",
                pick_label,
            )
            return False
        return True

    def _run_replay_actions(self, retry_actions: list, attempt: int, pick_label: str):
        """Dispatch the replayed segment. Perception is already refreshed."""
        for ri, action in enumerate(retry_actions):
            if self._abort:
                return
            atype = action.get("type", action.get("name", "unknown"))
            params = action.get("params", {})
            logger.info(
                "[RETRY-%d %d/%d] %s %s",
                attempt + 1,
                ri + 1,
                len(retry_actions),
                atype,
                params,
            )
            self._recorder.set_action_label(f"retry-{attempt+1} {atype}")

            result = self._dispatch_action(atype, params)
            self._results.append(result)

            if result.success:
                if atype == "release" and pick_label:
                    # Mirror _run_actions' post-release bookkeeping: the replay
                    # bypasses it, and pipeline_run's closed loop reads
                    # _placed_labels to decide what is still outstanding.
                    self._placed_labels.add(pick_label)
                    logger.info("Retry: re-marked '%s' as placed", pick_label)
                continue

            recovery = self._attempt_recovery(atype, params, result, retry_actions, ri)
            if recovery and recovery.success:
                self._results[-1] = recovery
                continue
            return

    @staticmethod
    def _find_last_pick_cycle(actions: list) -> Optional[int]:
        for j in range(len(actions) - 1, -1, -1):
            if actions[j].get("type") == "move_to_keypoint":
                for k in range(j, -1, -1):
                    if actions[k].get("type") == "move_to_keypoint":
                        has_grasp = any(
                            actions[m].get("type") == "grasp"
                            for m in range(k, min(k + 3, len(actions)))
                        )
                        if has_grasp:
                            return k
                break
        return None


logger = logging.getLogger("spark_server")

# The placed object's mask centroid must land within this fraction of the
# remembered target bbox's diagonal from the bbox center. Loose on purpose:
# a screwdriver seated in its slot can hang a few cm past either end.
CENTER_FRAC = 0.75
# Minimum confidence for the end-capture object detection to count as
# evidence in either direction. Below this the rung abstains.
MIN_CONF = 0.35


def remember_targets(executor):
    """Snapshot the pre-execution detection map's masks and boxes.

    Taken ONCE when the executor receives its detection map; mid-run
    re-detections overwrite detection_map in place. The memory must be the
    scene as APPROVED, not as mutated.
    """
    mem = {}
    for label, det in (executor.detection_map or {}).items():
        if not isinstance(det, dict):
            continue
        bbox = det.get("bbox")
        cam = det.get("_camera")
        if bbox is None or cam is None:
            continue
        mem[str(label)] = {
            "bbox": [float(v) for v in bbox],
            "camera": str(cam),
            "confidence": float(det.get("confidence") or 0.0),
        }
    executor._spatial_memory = mem
    if mem:
        logger.info("[spatial-verify] remembered %d region(s): %s",
                    len(mem), ", ".join(sorted(mem)))


def spatial_rung(executor, score=None):
    """The end-of-task check. Returns a dict verdict or None (abstain).

    {"status": "pass"|"fail"|"abstain", "detail": str, ...evidence}
    """
    mem = getattr(executor, "_spatial_memory", None) or {}
    target_label = getattr(executor, "_last_place_label", "") or ""
    obj_label = getattr(executor, "_active_grasp_label", "") or ""
    if not mem or not target_label or not obj_label:
        return None
    remembered = mem.get(target_label)
    if remembered is None:
        return None

    pipeline = getattr(executor, "_pipeline", None)
    if pipeline is None:
        return None

    # One wide capture with the arm retracted. Detect ONLY the object's
    # label: the target is not re-detected, because "the slot is gone" is a
    # success signal, not a matching failure.
    try:
        captures = pipeline.capture()
        dets = pipeline.detect(captures, [obj_label])
        merged = pipeline.merge_detections(dets)
    except Exception as exc:  # noqa: BLE001 - verification must not crash a run
        logger.warning("[spatial-verify] capture/detect failed: %s", exc)
        return None

    cam = remembered["camera"]
    x1, y1, x2, y2 = remembered["bbox"]
    cx_t, cy_t = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    diag = float(np.hypot(x2 - x1, y2 - y1)) or 1.0

    best = None
    for d in merged:
        if str(getattr(d, "label", "")) != obj_label:
            continue
        if str(getattr(d, "camera", "")) != cam:
            continue
        if float(getattr(d, "confidence", 0) or 0) < MIN_CONF:
            continue
        c2d = getattr(d, "centroid_2d", None)
        if c2d is None:
            continue
        dist = float(np.hypot(c2d[0] - cx_t, c2d[1] - cy_t)) / diag
        if best is None or dist < best["dist_frac"]:
            best = {
                "dist_frac": round(dist, 3),
                "confidence": round(float(d.confidence), 2),
            }

    if best is None:
        # Not visible on the remembered camera: seated deep or dropped out
        # of frame. Abstain rather than guess.
        return {"status": "abstain",
                "detail": f"'{obj_label}' not re-detected on {cam} "
                          f"(conf floor {MIN_CONF})"}

    ok = best["dist_frac"] <= CENTER_FRAC
    verdict = {
        "status": "pass" if ok else "fail",
        "detail": (
            f"'{obj_label}' centroid {best['dist_frac']:.2f} of the "
            f"remembered '{target_label}' bbox diagonal from its center on "
            f"{cam} (gate {CENTER_FRAC}, conf {best['confidence']})"
        ),
        **best,
        "camera": cam,
    }
    logger.info("[spatial-verify] %s: %s", verdict["status"], verdict["detail"])
    return verdict
