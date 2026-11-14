"""Per-step record of one displacement trial, for analysis after the run.

Nothing here feeds the system: the executor never reads a trace, and the true object positions are saved
for analysis only. Recording reads simulator arrays that the step already computed. It runs no physics,
no renders and no random draws, and saved frames are copies of the images the step already rendered.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np

from spark_bench.dyn.bodies import find_free_joint_for_hint, free_object_joints
from spark_bench.libero_pro.motion import find_ee_site
from spark_bench.libero_pro.primitives._common import fuzzy_get_det
from spark_bench.libero_pro.telemetry import find_finger_qpos_addrs, read_aperture

log = logging.getLogger(__name__)

# Columns of ``rows``; the body positions follow (three per name in info['body_names']).
COL_T, COL_EE, COL_AP, COL_CMD, COL_NOTE, COL_ATTEMPT, COL_HOLD, COL_AIM, COL_BODIES = 0, 1, 4, 5, 6, 7, 8, 9, 12


def trace_stem(phase: str, magnitude_cm: float, trial: int) -> str:
    return f'{phase}_{magnitude_cm:g}cm_T{trial}'


class TrialTrace:
    """Wraps ``env.step`` like the injector does and samples after each step.

    A row is kept every ``stride`` steps and whenever the behavior-tree note or the displacement state
    changes, so the first row of each note is exact. ``frame_stride`` > 0 also saves the rendered images.
    Physics stepped by ``joint_move`` bypasses env.step: those steps have no rows, but the row clock is the
    env's own step counter, so they still count as time.
    """

    def __init__(self, env, path: Path, pick_hint: str, place_hint: str, meta: dict,
                 stride: int = 5, frame_stride: int = 0):
        self.env = env
        self.path = Path(path)
        self.meta = meta
        self.stride = int(stride)
        self.frame_stride = int(frame_stride)
        self.model = env.sim.model._model
        self.data = env.sim.data._data
        self.site = find_ee_site(self.model)
        self.fingers = find_finger_qpos_addrs(self.model)
        joints = free_object_joints(self.model, mujoco)
        self.adrs = [j[0] for j in joints]
        self.body_names = [j[2] for j in joints]
        self.pick_body = find_free_joint_for_hint(self.model, pick_hint, mj=mujoco)[1]
        self.place_body = find_free_joint_for_hint(self.model, place_hint, mj=mujoco)[1]
        self.initial = self._bodies().tolist()
        self.rows: list[list[float]] = []
        self.notes: list[dict] = []
        self.note = -1
        self.attempt_i = -1
        self.executor: Any = None
        self.perturb_row: Optional[int] = None
        self.perturb_t: Optional[int] = None
        self._force_row = True
        self._n = 0
        self._orig_step = None
        self.sample_error: Optional[str] = None

    def install(self) -> None:
        self._orig_step = self.env.step

        def _patched(action):
            out = self._orig_step(action)
            try:
                self._sample(action, out)
            except Exception as e:  # noqa: BLE001 - motion loops read a step exception as the end of the episode
                self.sample_error = repr(e)
            return out

        self.env.step = _patched

    def uninstall(self) -> None:
        if self._orig_step is not None:
            self.env.step = self._orig_step
            self._orig_step = None

    def close(self) -> None:
        self.uninstall()
        try:
            self.save()
        except Exception as e:  # noqa: BLE001 - an analysis record must not end the run
            log.warning('trace save failed: %s', e)

    def _clock(self) -> int:
        return int(getattr(getattr(self.env, 'env', self.env), 'timestep', self._n))

    def start_attempt(self, executor) -> None:
        """One executor.run: the first plan execution or one recovery re-execution."""
        self.executor = executor
        self.attempt_i += 1
        self._set_note(-1, 'attempt_start', None)

    def end_attempt(self) -> None:
        self._set_note(-1, 'between_attempts', None)

    def enter_note(self, idx: int, note_type: str, label: Optional[str]) -> None:
        self._set_note(idx, note_type, label)

    def _set_note(self, idx: int, note_type: Optional[str], label: Optional[str]) -> None:
        self.notes.append({'attempt': self.attempt_i, 'idx': int(idx), 'type': str(note_type), 'label': label})
        self.note = len(self.notes) - 1
        self._force_row = True

    def _bodies(self) -> np.ndarray:
        return np.array([self.data.qpos[a:a + 3] for a in self.adrs], dtype=np.float32).reshape(-1, 3)

    def _aim(self) -> list[float]:
        """Where the executor believes the current note's target is (pick label when the note has none)."""
        ex, label = self.executor, self.notes[self.note]['label'] if self.notes else None
        if ex is None:
            return [np.nan] * 3
        det = fuzzy_get_det(ex.det_map, label or ex.last_pick_label or '')
        return [np.nan] * 3 if det is None else [float(x) for x in det.position_3d]

    def _sample(self, action, out) -> None:
        self._n += 1
        fired = self.perturb_row is None and 't_perturb' in self.meta
        if fired or self._force_row or self._n % self.stride == 0:
            clock = self._clock()
            if fired:
                self.perturb_row, self.perturb_t = len(self.rows), clock
            ap = read_aperture(self.data, self.fingers)
            self.rows.append([
                clock, *(float(x) for x in self.data.site_xpos[self.site]),
                np.nan if ap is None else ap, float(np.asarray(action).reshape(-1)[6]),
                self.note, self.attempt_i, float(bool(getattr(self.executor, 'holding', False))),
                *self._aim(), *self._bodies().ravel()])
            self._force_row = False
        if self.frame_stride and self._n % self.frame_stride == 0:
            self._save_frame(out[0])

    def _save_frame(self, obs: dict) -> None:
        import cv2
        d = self.path.with_suffix('')
        d.mkdir(parents=True, exist_ok=True)
        views = [obs[k][::-1] for k in ('agentview_image', 'robot0_eye_in_hand_image') if k in obs]
        if views:
            cv2.imwrite(str(d / f'{self._clock():05d}.jpg'), cv2.cvtColor(np.hstack(views), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 70])

    def save(self) -> None:
        info = {'body_names': self.body_names, 'pick_body': self.pick_body, 'place_body': self.place_body,
                'initial_bodies': self.initial, 'notes': self.notes, 'perturb_row': self.perturb_row,
                'perturb_t': self.perturb_t, 'stride': self.stride, 'n_env_steps': self._n,
                'final_timestep': self._clock(),
                'horizon': int(getattr(getattr(self.env, 'env', self.env), 'horizon', 0)),
                'sample_error': self.sample_error}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, rows=np.asarray(self.rows, dtype=np.float32), info=json.dumps(info))
