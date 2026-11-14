"""Behavior-tree node handlers (sequence, selector, retry), recovery budget and leaf dispatch."""


from typing import List
import logging
import numpy as np
import threading
import time


from spark_core import bt_grammar
from spark_real.control import common as primitive_timeouts
from spark_real.control import recovery as execution_recovery
from spark_real.control.common import AbortRequested, ExecutionResult
from spark_real.control.common import BUILTIN_PRIMITIVES, AbortRequested, ExecutionResult
from spark_real.control.trajectory import TrajectoryRecorder
from spark_real.skills import registry as skill_registry


logger = logging.getLogger(__name__)

_LABEL_PARAMS = ("keypoint_label", "container_label", "target_label")


class DispatchMixin:
    """Behavior-tree node handlers (sequence, selector, retry), recovery budget and leaf dispatch."""

    def _flatten_tree(self, node: dict) -> List[dict]:
        """Flatten a score tree into the action list ``_run_actions`` walks.

        Sequences collapse. A ``selector``/``fallback`` or ``retry`` node does
        NOT: it becomes ONE opaque marker action carrying its already-flattened
        branches, otherwise "try A, else B" would flatten to "do A, then do B".
        """
        node_type = node.get("type", "")

        if node_type in bt_grammar.SELECTOR_TYPES:
            branches = []
            for child in node.get("children") or []:
                branch = self._flatten_tree(child)
                if branch:
                    branches.append(branch)
            if len(branches) > bt_grammar.MAX_SELECTOR_BRANCHES:
                logger.warning(
                    "selector has %d branches; running the first %d",
                    len(branches),
                    bt_grammar.MAX_SELECTOR_BRANCHES,
                )
                branches = branches[: bt_grammar.MAX_SELECTOR_BRANCHES]
            return [
                {
                    "type": "selector",
                    "_bt_selector": True,
                    "branches": branches,
                    "params": dict(node.get("params") or {}),
                }
            ]

        if node_type in bt_grammar.RETRY_TYPES:
            body = []
            for child in node.get("children") or []:
                body.extend(self._flatten_tree(child))
            attempts = bt_grammar.clamp_retry_attempts(
                (node.get("params") or {}).get("max_attempts")
            )
            return [
                {
                    "type": "retry",
                    "_bt_retry": True,
                    "branch": body,
                    "max_attempts": attempts,
                    "params": dict(node.get("params") or {}),
                }
            ]

        if node_type == "sequence" or (
            "children" in node and node_type not in self._LEAF_TYPES
        ):
            actions = []
            for child in node.get("children", []):
                actions.extend(self._flatten_tree(child))
            return actions
        return [node]

    def _leaf_actions(self, actions: List[dict]) -> List[dict]:
        """Expand control-flow markers into the leaves they can run.

        Downstream consumers (post-task verification, the pick-cycle replay)
        reason about "which labels did this plan touch", not about control
        flow.
        """
        out: List[dict] = []
        for action in actions:
            if action.get("_bt_selector"):
                for branch in action.get("branches") or []:
                    out.extend(self._leaf_actions(branch))
            elif action.get("_bt_retry"):
                out.extend(self._leaf_actions(action.get("branch") or []))
            else:
                out.append(action)
        return out

    def _spend_recovery_budget(self, what: str) -> bool:
        """Charge one recovery re-attempt against the per-score budget.

        Returns False once the budget is gone, which makes the enclosing node
        fail cleanly. The per-node caps alone are not enough: retry(3) nested
        inside fallback(3) multiplies to nine attempts, and nine blind
        re-approaches on a real arm is not recovery, it is thrashing.
        """
        used = getattr(self, "_recovery_attempts_used", 0)
        if used >= bt_grammar.MAX_RECOVERY_ATTEMPTS_PER_SCORE:
            logger.warning(
                "recovery budget exhausted (%d/%d); refusing %s",
                used,
                bt_grammar.MAX_RECOVERY_ATTEMPTS_PER_SCORE,
                what,
            )
            return False
        self._recovery_attempts_used = used + 1
        logger.info(
            "recovery attempt %d/%d: %s",
            self._recovery_attempts_used,
            bt_grammar.MAX_RECOVERY_ATTEMPTS_PER_SCORE,
            what,
        )
        return True

    @staticmethod
    def _first_label_of(actions: List[dict]) -> str:
        for action in actions:
            label = (action.get("params") or {}).get("keypoint_label")
            if label:
                return str(label)
        return ""

    def _reset_before_recovery(self, next_actions: List[dict]) -> None:
        """Put the arm in a known state before a recovery branch runs.

        Mirrors the reset the flat path already does on a failed action: drop
        nothing that is still gripped, clear the workspace vertically, return
        to home, and re-ground the label the next branch is about to reach for.
        Every step is best-effort: a reset that raises must not mask the
        failure that triggered it.
        """
        self._check_abort()

        still_holding = False
        if self._holding:
            try:
                still_holding = self._verify_grasp()
            except Exception as exc:
                logger.warning("Recovery reset: grasp verify failed: %s", exc)
        if not still_holding:
            try:
                self.robot.open_gripper()
                self._abort_sleep(0.3)
            except AbortRequested:
                raise
            except Exception as exc:
                logger.warning("Recovery reset: open_gripper failed: %s", exc)
            self._holding = False

        try:
            cur = self._get_current_position()
            if cur[2] < 0.45:
                cur[2] = min(cur[2] + 0.10, 0.45)
                # A reset LIFT is a translation; re-commanding GRASP_ORIENTATION
                # here would unwind the task's yaw at v=2.67 rad/s.
                self._move_to(cur, self.current_orientation())
        except AbortRequested:
            raise
        except Exception as exc:
            logger.warning("Recovery reset: lift failed: %s", exc)

        go_home = getattr(self.robot, "go_home", None)
        if callable(go_home):
            try:
                go_home()
            except AbortRequested:
                raise
            except Exception as exc:
                logger.warning("Recovery reset: go_home failed: %s", exc)

        if self._pipeline is not None:
            label = self._first_label_of(self._leaf_actions(next_actions))
            if label and label not in self._placed_labels:
                try:
                    self._redetect_single(label)
                except AbortRequested:
                    raise
                except Exception as exc:
                    logger.warning("Recovery reset: re-detect failed: %s", exc)

    def _skip_failed_cycle(self, actions: List[dict], index: int) -> int:
        """Index of the next action worth running after a control-flow node
        exhausted its recovery.

        Once the acquire has genuinely failed, the transport and the place
        that follow it would move an empty gripper to a container and open it.
        Skip to the next action that can start a NEW pick cycle (one whose
        leaves contain a grasp), which for a single-object plan is the end,
        and for a multi-object plan is the next object.
        """
        j = index + 1
        while j < len(actions):
            leaves = self._leaf_actions([actions[j]])
            if any(a.get("type") in ("grasp", "grasp_se3") for a in leaves):
                return j
            self._results.append(
                ExecutionResult(
                    action_type=actions[j].get("type", "unknown"),
                    success=False,
                    message="Skipped (recovery for the preceding step failed)",
                )
            )
            j += 1
        return j

    def _run_branch(self, actions: List[dict]) -> bool:
        """Run one branch under STRICT sequence semantics.

        Strict means: stop at the first failure and report it. The enclosing
        selector's next branch, or the enclosing retry's next attempt, IS the
        reaction to the failure.
        """
        for k, action in enumerate(actions):
            self._check_abort()
            # Same release look-ahead as _run_actions: a recovery re-run must
            # steer its approach yaw by the same upcoming dip the first try did.
            self._pending_release_tilt = self._peek_release_tilt(actions[k:])
            self._pending_place_label = self._peek_place_label(actions[k + 1:])
            if action.get("_bt_selector"):
                result = self._run_selector(action)
            elif action.get("_bt_retry"):
                result = self._run_retry(action)
            else:
                result = self._run_leaf(action)
            self._results.append(result)
            self._last_action_failed = not bool(result.success)
            if not result.success:
                return False
        return True

    def _run_leaf(self, action: dict) -> ExecutionResult:
        action_type = action.get("type", action.get("name", "unknown"))
        params = action.get("params", {}) or {}
        logger.info("[branch] %s %s", action_type, params)
        try:
            self._recorder.set_action_label(action_type)
            self._recorder.mark_transition(action_type)
            self._recorder.record()
        except Exception as exc:
            logger.debug("Recorder skipped for %s: %s", action_type, exc)

        if action_type in ("move_to_keypoint", "grasp_se3") and not self._holding:
            self._last_pick_label = params.get("keypoint_label", "")

        result = self._dispatch_action(action_type, params)
        if not result.success:
            logger.warning("[branch] %s failed: %s", action_type, result.message)
        if result.success and action_type == "release" and self._last_pick_label:
            self._placed_labels.add(self._last_pick_label)
            self._last_pick_label = ""
        return result

    def _run_selector(self, marker: dict) -> ExecutionResult:
        branches = marker.get("branches") or []
        total = len(branches)
        if total == 0:
            return ExecutionResult(
                action_type="selector", success=True, message="empty selector"
            )
        for idx, branch in enumerate(branches):
            if idx > 0:
                if not self._spend_recovery_budget(
                    f"selector branch {idx + 1}/{total}"
                ):
                    return ExecutionResult(
                        action_type="selector",
                        success=False,
                        message=(
                            f"All {total} selector branches failed "
                            f"(recovery budget exhausted after {idx})"
                        ),
                    )
                self._reset_before_recovery(branch)
            if self._run_branch(branch):
                return ExecutionResult(
                    action_type="selector",
                    success=True,
                    message=f"branch {idx + 1}/{total} succeeded",
                )
            logger.warning("selector branch %d/%d failed", idx + 1, total)
        return ExecutionResult(
            action_type="selector",
            success=False,
            message=f"All {total} selector branches failed",
        )

    def _run_retry(self, marker: dict) -> ExecutionResult:
        branch = marker.get("branch") or []
        if not branch:
            return ExecutionResult(
                action_type="retry", success=True, message="empty retry"
            )
        attempts = bt_grammar.clamp_retry_attempts(marker.get("max_attempts"))
        for k in range(attempts):
            if k > 0:
                if not self._spend_recovery_budget(f"retry attempt {k + 1}/{attempts}"):
                    return ExecutionResult(
                        action_type="retry",
                        success=False,
                        message=(
                            f"retry gave up after {k}/{attempts} attempts "
                            "(recovery budget exhausted)"
                        ),
                    )
                self._reset_before_recovery(branch)
            if self._run_branch(branch):
                return ExecutionResult(
                    action_type="retry",
                    success=True,
                    message=f"succeeded on attempt {k + 1}/{attempts}",
                )
            logger.warning("retry attempt %d/%d failed", k + 1, attempts)
        return ExecutionResult(
            action_type="retry",
            success=False,
            message=f"all {attempts} retry attempts failed",
        )

    def _normalize_params(self, params: dict) -> dict:
        if not params:
            return params
        if not any(a in params for a in self._PARAM_ALIASES):
            return params
        norm = dict(params)
        for alias, canonical in self._PARAM_ALIASES.items():
            if alias in norm and canonical not in norm:
                norm[canonical] = norm[alias]
        return norm

    def _dispatch_action(self, action_type: str, params: dict) -> ExecutionResult:
        """Normalize grammar aliases, then run under the primitive's budget.

        On a budget overrun the watchdog aborts the motion and returns a
        failed result so the caller's recovery path handles it like any
        failed post-condition.
        """
        params = self._normalize_params(params)
        budget = primitive_timeouts.timeout_for(
            getattr(self, "_primitive_timeouts", None) or {}, action_type
        )
        t_start = time.time()
        result = primitive_timeouts.run_with_timeout(
            self,
            action_type,
            budget,
            lambda: self._dispatch_action_body(action_type, params),
        )
        self._trace_step(action_type, params, result, t_start)
        return result

    def _trace_step(self, action_type, params, result, t_start) -> None:
        """One record per executed step: what ran, on which detection, when, and how it ended."""
        try:
            from spark_real.control.common import PrimitiveTrace, TraceWriter, detection_summary

            writer = TraceWriter.from_pipeline(self._pipeline)
            if not writer.enabled:
                return
            label = next((params.get(k) for k in _LABEL_PARAMS
                          if isinstance((params or {}).get(k), str)), None)
            det = (self.detection_map or {}).get(label) if label else None
            writer.write(PrimitiveTrace(
                index=writer.next_index(), primitive=action_type, params=dict(params or {}),
                detections_used=[detection_summary({**det, "label": label})] if det else [],
                elapsed_s=time.time() - t_start,
                extra={"success": bool(result.success), "message": result.message,
                       "t_start": round(t_start, 3), "t_end": round(time.time(), 3),
                       "holding_after": bool(getattr(self, "_holding", False)),
                       **({"arm": self.arm} if getattr(self, "arm", None) else {})},
            ))
        except Exception as exc:  # noqa: BLE001 - tracing never fails a run
            logger.debug("step trace skipped: %s", exc)

    def _reconcile_after_protective_stop(self, action_type: str, params: dict):
        """One world-model check at the first action boundary after an
        auto-cleared protective stop.

        The unlock path (ur10e_driver._check_safety_and_recover) restores
        CONTROL, but a protective stop means a COLLISION happened: the held
        object may have been knocked from the jaws (ask the jaws) and the
        involved detection may have moved (re-bind).
        """
        ts = float(getattr(self.robot, "last_protective_recovery_ts", 0.0) or 0.0)
        if ts <= self._last_safety_reconcile_ts:
            return
        self._last_safety_reconcile_ts = ts
        logger.warning(
            "[post-stop] reconciling before '%s': the last protective stop "
            "was auto-cleared, so the collision's effects are checked now",
            action_type,
        )
        # 1. Still holding? A drop here fails the CURRENT action so recovery
        #    re-grasps instead of transporting empty jaws to the target.
        if getattr(self, "_holding", False):
            try:
                if not self._grip_intact():
                    logger.warning(
                        "[post-stop] the jaws are EMPTY; the collision cost "
                        "us the object -- marking not-holding so recovery "
                        "re-grasps"
                    )
                    self._holding = False
            except Exception:  # noqa: BLE001 - unknown grip stays assumed-held
                pass
        # 2. Re-bind the label this action is about, if it names one. The
        #    collision may have moved it (or been CAUSED by it moving).
        label = (params or {}).get("keypoint_label") or (params or {}).get(
            "container_label"
        )
        if label:
            try:
                from spark_real.control.recovery import (
                    rebind_before_approach,
                )

                rebind_before_approach(self, label)
            except Exception:  # noqa: BLE001
                pass

    def _rebind_action_labels(self, params) -> None:
        """Every skill that names a scene object gets a fresh detection of it first.
        The object in the jaws is skipped: a camera cannot relocate it."""
        from spark_core.binding import _labels_match
        from spark_real.control.recovery import rebind_before_approach
        from spark_real.control.recovery import _held_label_of

        held = _held_label_of(self)
        for key in _LABEL_PARAMS:
            label = (params or {}).get(key)
            if not label or not isinstance(label, str):
                continue
            if held and _labels_match(label, held):
                continue
            try:
                rebind_before_approach(self, label)
            except AbortRequested:
                raise
            except Exception:  # noqa: BLE001 - a failed rebind keeps the old binding
                pass

    def _dispatch_action_body(
        self, action_type: str, params: dict
    ) -> ExecutionResult:
        t0 = time.time()
        try:
            self._reconcile_after_protective_stop(action_type, params)
            self._current_action_type = action_type
            self._rebind_action_labels(params)
            builtin = set(BUILTIN_PRIMITIVES) | {"action"}
            if action_type not in builtin and skill_registry.get(action_type) is not None:
                return skill_registry.dispatch(action_type, self, params)

            if action_type in ("move_to_keypoint", "action"):
                name = params.get("name", action_type)
                dispatch_map = {
                    "move_to_keypoint": self._move_to_keypoint,
                    "grasp": self._grasp,
                    "release": self._release,
                    "move_relative": self._move_relative,
                }
                handler = dispatch_map.get(name)
                if handler:
                    return handler(params, t0)
                if action_type == "action" and name != action_type:
                    return self._dispatch_action_body(name, params)
                return self._move_to_keypoint(params, t0)
            elif action_type == "grasp":
                return self._grasp(params, t0)
            elif action_type == "release":
                return self._release(params, t0)
            elif action_type == "move_relative":
                return self._move_relative(params, t0)
            elif action_type == "wait":
                return self._wait(params, t0)
            else:
                return ExecutionResult(
                    action_type=action_type,
                    success=False,
                    message=f"Unknown action type: {action_type}",
                    duration=time.time() - t0,
                )
        except AbortRequested:
            raise
        except Exception as e:
            return ExecutionResult(
                action_type=action_type,
                success=False,
                message=str(e),
                duration=time.time() - t0,
            )

    def _move_relative(self, params: dict, t0: float) -> ExecutionResult:
        self._check_abort()
        dx = params.get("dx", 0.0)
        dy = params.get("dy", 0.0)
        dz = params.get("dz", 0.0)
        current = self._get_current_position()
        self._move_to(current + np.array([dx, dy, dz]), self.current_orientation())
        return ExecutionResult(
            action_type="move_relative",
            success=True,
            message=f"Moved by ({dx}, {dy}, {dz})",
            duration=time.time() - t0,
        )

    def _wait(self, params: dict, t0: float) -> ExecutionResult:
        # duration defaults to 0.5s when the plan omits it.
        duration = params.get("duration", 0.5)
        self._abort_sleep(duration)
        return ExecutionResult(
            action_type="wait",
            success=True,
            message=f"Waited {duration}s",
            duration=time.time() - t0,
        )




class BtRunnerMixin:
    """Score execution entry point and the flat action loop with legato look-ahead."""

    def execute_score(self, score: dict) -> List[ExecutionResult]:
        self._results = []
        # NOT an unconditional `self._abort = False`: a stop pressed after this
        # task was armed (during capture -> detect -> plan) must survive into
        # the run rather than be wiped by it.
        self._abort = self._abort_epoch != self._armed_epoch
        if self._abort:
            logger.warning(
                "execute_score: a stop was requested after this task was "
                "armed (epoch %d != %d); honouring it instead of starting",
                self._abort_epoch,
                self._armed_epoch,
            )
        self._running = True
        self._recorder = TrajectoryRecorder(self.robot)

        tree = score.get("tree", score)
        actions = self._flatten_tree(tree)
        # Recovery budget is scoped to ONE execute_score call: a closed-loop
        # pass gets its own six re-attempts, but a single pass can never spend
        # more however deeply the plan nests retry inside fallback.
        self._recovery_attempts_used = 0
        # Post-task verification and the replay path both want a plain list of
        # leaves: every leaf of every branch.
        leaf_actions = self._leaf_actions(actions)
        logger.info(
            "Executing %d actions (%d leaves)", len(actions), len(leaf_actions)
        )

        aborted = False
        try:
            try:
                self._run_actions(actions)
            except AbortRequested:
                logger.warning("Execution aborted by user")
                self._results.append(
                    ExecutionResult(
                        action_type="abort", success=False, message="Aborted by user"
                    )
                )
                aborted = True
                self._stop_robot()

            if self._pipeline is not None and not aborted:
                try:
                    self._run_post_task_verification(score, leaf_actions)
                except AbortRequested:
                    logger.warning("Post-task verification aborted by user")
                    aborted = True

            try:
                self._recorder.save()
            except Exception as e:
                logger.warning("Failed to save trajectory: %s", e)

            return self._results
        finally:
            self._running = False

    def _is_legato_blendable(self, action: dict) -> bool:
        """
        Return True if this action can be a legato note (vs barrier).
        """
        if action is None:
            return False
        atype = action.get("type", action.get("name", ""))
        if atype not in self._LEGATO_BLENDABLE:
            return False
        if atype == "move_to_keypoint" and not self._holding:
            return False
        return True

    @staticmethod
    def _peek_release_tilt(actions):
        """Params of the first ``release`` leaf in ``actions``, else None.

        Depth-first through selector/retry markers, in execution order. Lets
        the PLACE TRANSPORT know what dip the release will ask for: a tilt
        past the positive pitch budget (the wrist-camera clearance, 40 deg) is
        only achievable when the approach yaw already put the dip on the
        negative-pitch side (90 deg budget), see ReleaseMixin._dip_side_orient.
        """
        for node in actions or []:
            if not isinstance(node, dict):
                continue
            if node.get("_bt_selector"):
                for branch in node.get("branches") or []:
                    hit = BtRunnerMixin._peek_release_tilt(branch)
                    if hit is not None:
                        return hit
                continue
            if node.get("_bt_retry"):
                hit = BtRunnerMixin._peek_release_tilt(node.get("branch"))
                if hit is not None:
                    return hit
                continue
            if node.get("type", node.get("name", "")) == "release":
                return node.get("params", {}) or {}
        return None

    @staticmethod
    def _peek_place_label(actions):
        """Label of the next PLACE destination in ``actions``, else None.

        Sibling of _peek_release_tilt: it lets the grasp choose between the
        two equivalent grasp yaws knowing the destination. A parallel gripper
        closes identically at yaw and yaw+180, but the choice fixes which way
        the held object points, and therefore the yaw the place will need
        (which can land past the wrist limit and seat the tool
        END-FOR-END).
        """
        for node in actions or []:
            if not isinstance(node, dict):
                continue
            if node.get("_bt_selector"):
                for branch in node.get("branches") or []:
                    hit = BtRunnerMixin._peek_place_label(branch)
                    if hit is not None:
                        return hit
                continue
            if node.get("_bt_retry"):
                hit = BtRunnerMixin._peek_place_label(node.get("branch"))
                if hit is not None:
                    return hit
                continue
            params = node.get("params", {}) or {}
            kind = node.get("type", node.get("name", ""))
            if kind == "place_in_slot":
                return params.get("container_label")
            if kind == "move_to_keypoint":
                # The transport that delivers to the place is the FIRST
                # keypoint move after the grasp; the grasp's own approach has
                # already been consumed by the time this look-ahead runs.
                return params.get("keypoint_label")
        return None

    def _run_actions(self, actions):
        i = 0
        while i < len(actions):
            action = actions[i]
            # Refresh the release look-ahead every step so it can neither go
            # stale (a consumed release must stop steering later transports)
            # nor be missed (legato phrases and recovery both re-enter here).
            self._pending_release_tilt = self._peek_release_tilt(actions[i:])
            # Skip the current node: the NEXT destination is what matters.
            self._pending_place_label = self._peek_place_label(actions[i + 1:])
            if self._abort:
                logger.warning("Execution aborted by user")
                self._results.append(
                    ExecutionResult(
                        action_type="abort", success=False, message="Aborted by user"
                    )
                )
                break

            # Control-flow markers are handled first: they are not
            # dispatchable leaves, they own their children's failure handling,
            # and the flat path's recovery must not double up on theirs.
            if action.get("_bt_selector") or action.get("_bt_retry"):
                if action.get("_bt_selector"):
                    result = self._run_selector(action)
                else:
                    result = self._run_retry(action)
                self._results.append(result)
                self._last_action_failed = not bool(result.success)
                if not result.success:
                    i = self._skip_failed_cycle(actions, i)
                    continue
                i += 1
                continue

            # Legato look-ahead
            if self._is_legato_blendable(action):
                run = [action]
                j = i + 1
                while j < len(actions) and self._is_legato_blendable(actions[j]):
                    run.append(actions[j])
                    j += 1
                if len(run) >= 2:
                    logger.info(
                        "[%d-%d/%d] legato phrase: %s",
                        i + 1,
                        j,
                        len(actions),
                        [a.get("type") for a in run],
                    )
                    ok = self._execute_legato_phrase(run, i, len(actions))
                    if ok:
                        i = j
                        continue

            action_type = action.get("type", action.get("name", "unknown"))
            params = action.get("params", {})
            logger.info("[%d/%d] %s %s", i + 1, len(actions), action_type, params)

            if action_type == "move_to_keypoint":
                kp = params.get("keypoint_label", "")
                label = f"move_to {kp}" if kp else action_type
            elif action_type == "move_relative":
                label = "move_rel"
            else:
                label = action_type
            self._recorder.set_action_label(label)
            self._recorder.mark_transition(label)
            self._recorder.record()

            # Re-detect before approaching when previous action failed
            if (
                action_type in ("move_to_keypoint", "grasp_se3")
                and i > 0
                and not self._holding
                and self._pipeline is not None
                and getattr(self, "_last_action_failed", False)
            ):
                kp_label = params.get("keypoint_label", "")
                if kp_label and kp_label not in self._placed_labels:
                    self._join_redetect()
                    self._redetect_single(kp_label)

            if action_type == "move_to_keypoint" and not self._holding:
                self._last_pick_label = params.get("keypoint_label", "")
            elif action_type == "grasp_se3" and not self._holding:
                self._last_pick_label = params.get("keypoint_label", "")

            result = self._dispatch_action(action_type, params)
            self._results.append(result)
            self._last_action_failed = not bool(result.success)

            if result.success and action_type == "release":
                if (
                    self._last_pick_label
                    and self._pipeline is not None
                    and (
                        getattr(self, "strict_placement_verify", False)
                        # The per-release two-anchor verdict + rebind also
                        # rides this capture (verification.release_verdict).
                        or execution_recovery.release_verdict_enabled(self)
                    )
                ):
                    try:
                        execution_recovery.verify_placement(
                            self, self._last_pick_label, self._last_place_label
                        )
                    except Exception as e:
                        logger.warning("Post-release verification error: %s", e)
                if self._last_pick_label:
                    self._placed_labels.add(self._last_pick_label)
                    logger.info("Marked '%s' as placed", self._last_pick_label)
                    self._last_pick_label = ""
                if self._pipeline is not None:
                    remaining = actions[i + 1 :]
                    has_next_pick = any(
                        a.get("type") in ("move_to_keypoint", "grasp_se3")
                        for a in remaining
                    )
                    if has_next_pick:
                        # Never two re-detect threads at once.
                        self._join_redetect()
                        self._redetect_thread = threading.Thread(
                            target=self._redetect_all,
                            args=(actions, i),
                            daemon=True,
                        )
                        self._redetect_thread.start()
                        logger.info(
                            "Inter-grasp: skipping home, going direct to next pick"
                        )

            if not result.success:
                logger.warning("Action %d failed: %s", i + 1, result.message)
                msg_lower = (result.message or "").lower()
                if any(
                    k in msg_lower
                    for k in ("reflex", "singular", "discontinuity", "control")
                ):
                    try:
                        self._snapshot_failure_state(action_type, params, result)
                    except Exception as _snap_e:
                        logger.warning("Failure snapshot skipped: %s", _snap_e)
                recovery = self._attempt_recovery(
                    action_type, params, result, actions, i
                )
                if recovery is not None and recovery.success:
                    self._results[-1] = recovery
                    i += 1
                    continue
                logger.warning("Action failed, resetting")
                self._check_abort()
                still_holding = False
                if self._holding:
                    try:
                        still_holding = self._verify_grasp()
                        logger.info(
                            "Recovery: force-verify says %s; " "%s open the gripper",
                            "HOLDING" if still_holding else "EMPTY",
                            "skipping" if still_holding else "going to",
                        )
                    except Exception as _vexc:
                        logger.warning("Recovery verify failed: %s", _vexc)
                if not still_holding:
                    self.robot.open_gripper()
                    self._abort_sleep(0.3)
                    self._holding = False
                cur = self._get_current_position()
                if cur[2] < 0.45:
                    cur[2] = min(cur[2] + 0.10, 0.45)
                    self._move_to(cur, self.current_orientation())
                # Post-failure J5 escape from singularity
                try:
                    q_now = np.asarray(self.robot.get_joint_positions(), dtype=float)
                    if abs(q_now[4]) < 0.5:
                        q_escape = q_now[:7].copy()
                        q_escape[4] = -0.8
                        logger.info(
                            "Recovery: J5=%.3f near singular; "
                            "escape-nudge to -0.8 before next cycle",
                            float(q_now[4]),
                        )
                        self.robot.move_to_joint_config(
                            q_escape.tolist(), velocity=self.velocity * 0.5
                        )
                        try:
                            self._wait_stationary_after_servo()
                        except Exception:
                            pass
                except Exception as _e:
                    logger.debug("Recovery J5 escape skipped: %s", _e)
                # Retry-once on grasp failure
                if action_type in ("grasp", "grasp_se3"):
                    kp = params.get("keypoint_label", "")
                    if not hasattr(self, "_grasp_retry_count"):
                        self._grasp_retry_count = {}
                    retries = self._grasp_retry_count.get(kp, 0) if kp else 0
                    if kp and retries < 1:
                        self._grasp_retry_count[kp] = retries + 1
                        logger.info(
                            "[retry] grasp '%s' failed; retrying once with "
                            "fresh detection (attempt %d)",
                            kp,
                            retries + 2,
                        )
                        continue
                    if kp:
                        logger.info(
                            "[retry] grasp '%s' failed twice; giving up, "
                            "skipping cycle",
                            kp,
                        )
                    while i + 1 < len(actions):
                        next_type = actions[i + 1].get("type", "")
                        if next_type in ("grasp", "grasp_se3"):
                            break
                        i += 1
                        self._results.append(
                            ExecutionResult(
                                action_type=next_type,
                                success=False,
                                message="Skipped (prior grasp failed)",
                            )
                        )
                    logger.info("Skipped to next grasp cycle")

            i += 1
