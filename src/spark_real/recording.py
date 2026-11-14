"""Per-episode artifact recorder for spark_real.

One ``EpisodeRecorder`` owns a cache folder for the currently-running
task. It bundles:

  * ``bt.yaml``         : the planner's score (the BT we executed)
  * ``trajectory.npz``  : robot state @ ~30 Hz (TCP pose, joints,
                            wrench, gripper width, timestamps)
  * ``log.txt``         : slice of the executor log captured during
                            the episode (mirrors what the user saw
                            stream in the progress panel)
  * ``result.json``     : final outcome (success/fail, duration,
                            execution_results, detection summary)
  * ``video_<cam>.<ext>`` : per-camera videos owned by the pipeline's
                             ``VideoRecorder`` (not written here, just
                             copied into the bundle on save())

The cache lives at ``<output_dir>/episodes_cache/<episode_id>/``.
``save_as(task_name)`` renames it to ``<output_dir>/episodes_kept/
<task_name>_<episode_id>/``. ``discard()`` deletes it. Starting a new
episode automatically clears any prior cache (kept bundles are not
touched).

Thread model:
  * The trajectory logger runs as a daemon thread polling the robot
    driver at ~30 Hz. Started by ``begin()``, stopped by ``end()``.
  * Reads are best-effort: any driver call that raises is logged once
    and skipped on the next tick. The recorder never raises out.
"""


from __future__ import annotations


from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import Any, Dict, List, Optional
from typing import Callable, Optional
import functools
import imageio.v3 as iio
import json
import logging
import numpy as np
import os
import shutil
import subprocess
import sys
import threading
import time
import yaml


logger = logging.getLogger(__name__)


def slug(text: str) -> str:
    """Filesystem-safe token."""
    s = "".join(c if c.isalnum() or c in "-_." else "_" for c in (text or "")).strip("_")
    return s or "task"


def pad_rows(rows, width: int) -> np.ndarray:
    """(N, width) float64; short or missing rows are NaN-padded."""
    out = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        arr = np.asarray(row, dtype=np.float64).ravel()[:width]
        out[i, : arr.size] = arr
    return out


@dataclass
class _TrajBuffer:
    """
    Single-arm trajectory buffer.
    """

    ts: List[float] = field(default_factory=list)
    tcp_pose: List[List[float]] = field(default_factory=list)  # x,y,z,rx,ry,rz
    joints: List[List[float]] = field(default_factory=list)
    wrench: List[List[float]] = field(default_factory=list)  # fx..fz tx..tz
    gripper_width: List[float] = field(default_factory=list)
    gripper_pos: List[float] = field(default_factory=list)  # 0..255 scale

    def add(
        self, ts: float, *, tcp_pose=None, joints=None, wrench=None, gw=None, gp=None
    ) -> None:
        self.ts.append(ts)
        self.tcp_pose.append(list(tcp_pose) if tcp_pose is not None else [])
        self.joints.append(list(joints) if joints is not None else [])
        self.wrench.append(list(wrench) if wrench is not None else [])
        self.gripper_width.append(float(gw) if gw is not None else float("nan"))
        self.gripper_pos.append(float(gp) if gp is not None else float("nan"))

    def to_npz_dict(self) -> Dict[str, np.ndarray]:
        pad = pad_rows
        return {
            "ts": np.asarray(self.ts, dtype=np.float64),
            "tcp_pose": pad(self.tcp_pose, 6),
            "joints": pad(self.joints, 7),
            "wrench": pad(self.wrench, 6),
            "gripper_width": np.asarray(self.gripper_width, dtype=np.float64),
            "gripper_pos": np.asarray(self.gripper_pos, dtype=np.float64),
        }


class EpisodeRecorder:
    """
    Cache + trajectory thread for a single episode.

    Lifecycle:
        rec = EpisodeRecorder(root, instruction)
        rec.begin(robot)
        ... task runs ...
        rec.end(score=..., detections=..., result=...)
        rec.save_as("pick_the_plushie")   OR   rec.discard()
    """

    POLL_HZ = 30.0

    def __init__(self, root: Path, instruction: str):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.episode_id = time.strftime("%Y%m%d_%H%M%S")
        self.cache_dir = self.root / "episodes_cache" / self.episode_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.instruction = instruction
        self._buf = _TrajBuffer()
        self._stop = threading.Event()
        self._paused = False  # set True via pause() during model.sample
        self._thread: Optional[threading.Thread] = None
        self._t_start: Optional[float] = None
        self._robot: Any = None
        self._video_paths: Dict[str, Path] = {}
        self._final: Dict[str, Any] = {}
        self._exhausted = False

    # lifecycle

    def begin(self, robot: Any) -> None:
        """Start trajectory logging. Idempotent."""
        if self._thread is not None:
            return
        self._robot = robot
        self._buf = _TrajBuffer()
        self._t_start = time.time()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"episode-rec-{self.episode_id}"
        )
        self._thread.start()
        logger.info(
            "EpisodeRecorder: started episode %s in %s", self.episode_id, self.cache_dir
        )

    def end(
        self,
        *,
        score: Optional[Dict[str, Any]] = None,
        detections: Optional[List[Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        video_paths: Optional[Dict[str, Path]] = None,
        log_text: Optional[str] = None,
    ) -> None:
        """
        Stop the trajectory thread and flush all artifacts to cache.
        """
        if self._exhausted:
            return
        self._exhausted = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # BT YAML
        if score is not None:
            try:
                (self.cache_dir / "bt.yaml").write_text(
                    yaml.safe_dump(score, sort_keys=False)
                )
            except Exception as exc:
                logger.warning("EpisodeRecorder: bt.yaml write failed: %s", exc)

        # Trajectory
        try:
            np.savez_compressed(
                self.cache_dir / "trajectory.npz", **self._buf.to_npz_dict()
            )
        except Exception as exc:
            logger.warning("EpisodeRecorder: trajectory.npz failed: %s", exc)

        # Video paths; the actual files are owned by VideoRecorder
        # in pipeline.py; we just record where they live so the save()
        # step can copy them in.
        self._video_paths = {
            k: Path(v) for k, v in (video_paths or {}).items() if v is not None
        }

        # Result
        try:
            payload = {
                "episode_id": self.episode_id,
                "instruction": self.instruction,
                "started_at": self._t_start,
                "ended_at": time.time(),
                "duration_s": (time.time() - (self._t_start or time.time())),
                "n_samples": len(self._buf.ts),
                "video_files": {k: str(v) for k, v in self._video_paths.items()},
            }
            if detections:
                payload["detections"] = [
                    {
                        "label": getattr(d, "label", None) or d.get("label"),
                        "confidence": float(
                            getattr(d, "confidence", 0.0)
                            if not isinstance(d, dict)
                            else d.get("confidence", 0.0)
                        ),
                    }
                    for d in detections
                ]
            if result is not None:
                payload.update(result)
            (self.cache_dir / "result.json").write_text(
                json.dumps(payload, indent=2, default=str)
            )
            self._final = payload
        except Exception as exc:
            logger.warning("EpisodeRecorder: result.json failed: %s", exc)

        # Log slice
        if log_text:
            try:
                (self.cache_dir / "log.txt").write_text(log_text)
            except Exception as exc:
                logger.warning("EpisodeRecorder: log.txt failed: %s", exc)

        logger.info(
            "EpisodeRecorder: ended episode %s (%d samples, " "cache %s)",
            self.episode_id,
            len(self._buf.ts),
            self.cache_dir,
        )

    # traj poll

    def pause(self) -> None:
        """
        Pause polling without stopping the recorder thread.

        Frees the GIL for a heavy inference call.
        """
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def _loop(self) -> None:
        period = 1.0 / max(1.0, self.POLL_HZ)
        warned: Dict[str, bool] = {}
        while not self._stop.is_set():
            if getattr(self, "_paused", False):
                # When paused, wait on the stop event (rechecking every 200ms)
                # instead of busy-polling the paused flag.
                self._stop.wait(timeout=0.2)
                continue
            t0 = time.time()
            self._buf.add(t0, **self._read_arm_kw(warned))
            sleep_for = max(0.0, period - (time.time() - t0))
            self._stop.wait(timeout=sleep_for)

    def _read_arm_kw(self, warned: Dict[str, bool]) -> Dict[str, Any]:
        """Poll the arm's state; failed reads are skipped."""
        kw: Dict[str, Any] = {}
        try:
            if hasattr(self._robot, "get_tcp_pose"):
                p = self._robot.get_tcp_pose()
                if isinstance(p, np.ndarray) and p.shape == (4, 4):
                    pos = p[:3, 3]
                    rv = Rotation.from_matrix(p[:3, :3]).as_rotvec()
                    kw["tcp_pose"] = [pos[0], pos[1], pos[2], rv[0], rv[1], rv[2]]
                elif p is not None:
                    kw["tcp_pose"] = list(p)[:6]
        except Exception as exc:
            if not warned.get("tcp"):
                logger.warning("EpisodeRecorder: tcp_pose read failed: %s", exc)
                warned["tcp"] = True
        try:
            if hasattr(self._robot, "get_joint_positions"):
                j = self._robot.get_joint_positions()
                if j is not None:
                    kw["joints"] = list(j)
        except Exception:
            pass
        try:
            if hasattr(self._robot, "get_tcp_force"):
                f = self._robot.get_tcp_force()
                if f is not None:
                    kw["wrench"] = list(f)[:6]
        except Exception:
            pass
        try:
            if hasattr(self._robot, "get_gripper_width"):
                kw["gw"] = float(self._robot.get_gripper_width())
        except Exception:
            pass
        try:
            if hasattr(self._robot, "get_gripper_position"):
                kw["gp"] = float(self._robot.get_gripper_position())
        except Exception:
            pass
        return kw

    # save / discard

    def save_as(
        self,
        task_name: str,
        kept_root: Optional[Path] = None,
        outcome: Optional[str] = None,
    ) -> Path:
        """
        Move the cache folder to ``episodes_kept/<task_name>_<id>/``
        and copy in any referenced videos. Returns the kept-folder path.
        """
        safe = slug(task_name) if str(task_name).strip() else "episode"
        # The label goes in the FOLDER NAME as well as the metadata, so a
        # corpus can be split with a glob and never depends on a reader
        # parsing json correctly.
        if outcome in ("pass", "fail"):
            safe = f"{safe}__{outcome}"
        kept = (kept_root or self.root) / "episodes_kept" / f"{safe}_{self.episode_id}"
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.cache_dir), str(kept))

        for cam, src in self._video_paths.items():
            try:
                if src and src.exists():
                    shutil.copy2(src, kept / src.name)
            except Exception as exc:
                logger.warning("EpisodeRecorder: copy video %s failed: %s", src, exc)
        self.cache_dir = kept
        if outcome in ("pass", "fail"):
            try:
                meta_path = kept / "metadata.json"
                meta = {}
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                meta["outcome"] = outcome
                meta["outcome_source"] = "operator"
                meta_path.write_text(json.dumps(meta, indent=2))
            except Exception as exc:  # noqa: BLE001 - never lose the episode
                logger.warning("EpisodeRecorder: could not stamp outcome: %s", exc)
        logger.info(
            "EpisodeRecorder: saved episode to %s (outcome=%s)", kept, outcome
        )
        return kept

    def discard(self) -> None:
        """
        Wipe the cache folder. Idempotent.
        """
        if self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
            except Exception as exc:
                logger.warning("EpisodeRecorder: discard failed: %s", exc)
        logger.info("EpisodeRecorder: discarded episode %s", self.episode_id)

    def summary(self) -> Dict[str, Any]:
        out = {
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "cache_dir": str(self.cache_dir),
            "exhausted": self._exhausted,
            "n_samples": len(self._buf.ts),
        }
        out.update(self._final)
        return out




try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None



def _resolve_ffmpeg_exe() -> str:
    """
    Return an absolute path to the ffmpeg binary.

    The posix_spawn fast path requires an absolute executable
    (``os.path.dirname(executable)`` truthy in subprocess.Popen). Prefer
    imageio_ffmpeg's bundled binary, else the ffmpeg found on PATH, last
    resort the bare name 'ffmpeg'.
    """
    try:
        if imageio_ffmpeg is None:
            raise ImportError("imageio_ffmpeg not available")
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


def _encode_h264_posix_spawn(frames, out_path: Path, fps: int) -> None:
    """
    Pipe raw RGB frames into ffmpeg for H.264 encoding.

    Why this exists: imageio's ``iio.imwrite`` spawns ffmpeg with a
    ``preexec_fn``, which forces Python onto the fork()+exec() path.
    fork() in a multithreaded process is unsafe per POSIX, and the
    server has JAX worker threads (pyroki) live by the first task, so
    the fork path can deadlock the encoder and the event loop.

    Instead we call ffmpeg via subprocess.Popen with arguments chosen to
    hit Python's posix_spawn fast path (subprocess.py _USE_POSIX_SPAWN):
      * absolute executable path (imageio's bundled ffmpeg)
      * preexec_fn=None
      * close_fds=False (posix_spawn cannot close arbitrary inherited fds)
      * no start_new_session, gid/uid manipulation, etc.

    Trade-off: posix_spawn cannot scrub inherited fds, so ffmpeg starts
    with the parent's open fds (Kinect handles, sockets). ffmpeg only
    reads stdin and writes the output path and never references them;
    they are freed when ffmpeg exits.
    """
    arr = np.ascontiguousarray(np.asarray(frames, dtype=np.uint8))
    n, h, w, c = arr.shape
    assert c == 3, f"expected RGB frames, got {arr.shape}"

    ffmpeg = _resolve_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "15",
        # H.264 requires even dimensions; mirror imageio's scale rescue.
        "-vf",
        f"scale={w if w % 2 == 0 else w + 1}:{h if h % 2 == 0 else h + 1}",
        "-v",
        "warning",
        str(out_path),
    ]

    # Open output as a write fd so it doesn't depend on cwd. Letting
    # ffmpeg open() it directly is fine; we just pass the path in argv.
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        close_fds=False,  # << posix_spawn requires this
        preexec_fn=None,  # << posix_spawn requires this
        start_new_session=False,
        env=os.environ.copy(),
    )

    # Stream frames in. ffmpeg buffers ~30MB internally; for a 2-minute
    # 1080p clip the kernel pipe (1MB) will fill up and we'll block on
    # write(), which is fine, that's flow control.
    try:
        proc.stdin.write(arr.tobytes())
        proc.stdin.close()
    except BrokenPipeError as exc:
        # ffmpeg died early, drain stderr for the actual reason.
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg pipe broken before encode: {err}") from exc

    err_bytes = proc.stderr.read() if proc.stderr else b""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"ffmpeg exited rc={rc}: {err_bytes.decode(errors='replace')}"
        )


class VideoRecorder:
    """
    Per-task recorder.  Frames captured by a ``frame_provider`` callback
    that returns an HxWx3 uint8 array, called from a background thread.
    """

    def __init__(
        self,
        frame_provider: Callable[[], Optional[np.ndarray]],
        out_dir: Path,
        fps: int = 10,
        max_seconds: float = 600.0,
    ):
        self.frame_provider = frame_provider
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.max_frames = int(self.fps * max_seconds)

        self._frames: deque = deque(maxlen=self.max_frames)
        # Guards the buffer against a capture thread that outlived stop()'s
        # join and is still appending while the encode reads.
        self._frames_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._label: str = ""

    # lifecycle

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, label: str = "") -> None:
        if self.is_recording:
            logger.warning("VideoRecorder.start: already running, ignoring")
            return
        self._label = label or "task"
        with self._frames_lock:
            self._frames.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="VideoRecorder"
        )
        self._thread.start()
        logger.info("VideoRecorder started: label=%r fps=%d", label, self.fps)

    # How long stop() waits for the capture thread. The frame provider can
    # block for a full staleness deadline inside a camera read (2 s), so a
    # 2 s join lands exactly on that boundary.
    JOIN_TIMEOUT_S = 5.0

    def stop(self) -> Optional[Path]:
        """
        Stop the capture thread and write the mp4.  Returns the file path,
        or None if no frames were captured.

        The join result is checked: the frame provider blocks up to a
        camera's 2 s staleness deadline, so a join timeout is the normal
        outcome on a wedged camera. The buffer is detached under a lock, so a
        thread that will not die appends into a deque nobody reads and cannot
        contaminate the encode ("deque mutated during iteration") or the next
        recording.
        """
        if not self.is_recording:
            return None
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.JOIN_TIMEOUT_S)
            if thread.is_alive():
                logger.error(
                    "VideoRecorder capture thread did not exit within %.0fs "
                    "(frame provider blocked?); detaching its buffer and "
                    "encoding what was captured. The thread is a daemon and "
                    "will exit with the process.",
                    self.JOIN_TIMEOUT_S,
                )
        # Detach: the zombie loop keeps a live deque to append into, and the
        # encode gets a private snapshot nobody mutates.
        with self._frames_lock:
            frames = list(self._frames)
            self._frames = deque(maxlen=self.max_frames)
        self._thread = None
        return self._encode(frames)

    # internals

    def _loop(self) -> None:
        period = 1.0 / max(1, self.fps)
        next_t = time.time()
        consecutive_failures = 0
        while not self._stop_event.is_set():
            try:
                frame = self.frame_provider()
                if frame is not None and frame.size > 0:
                    with self._frames_lock:
                        self._frames.append(frame)
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures % 20 == 1:  # don't spam the log
                    logger.warning("VideoRecorder frame grab failed: %s", e)
            next_t += period
            sleep = max(0.0, next_t - time.time())
            self._stop_event.wait(timeout=sleep)
            # fall-behind guard: if we're chronically late, skip ahead
            if time.time() - next_t > period * 5:
                next_t = time.time()

    def _encode(self, frames=None) -> Optional[Path]:
        # `frames` is the detached snapshot stop() hands over; the default
        # keeps the historical signature usable from a test or a REPL.
        if frames is None:
            with self._frames_lock:
                frames = list(self._frames)
        if len(frames) < 2:
            logger.info("VideoRecorder: only %d frame(s), not encoding", len(frames))
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c for c in self._label if c.isalnum() or c in "_-")[:48]
        path = self.out_dir / f"{ts}_{safe_label}.mp4"
        # Coerce to a contiguous (N, H, W, 3) uint8 array.
        # All frames must share H, W, channels.  Resize-on-mismatch is not
        # implemented; drop mismatched ones (rare) so encoding doesn't error.
        h, w = frames[0].shape[:2]
        clean = [f for f in frames if f.shape[:2] == (h, w) and f.dtype == np.uint8]
        if len(clean) < len(frames):
            logger.info(
                "VideoRecorder: dropped %d size-mismatched frames",
                len(frames) - len(clean),
            )
        if len(clean) < 2:
            return None
        try:
            _encode_h264_posix_spawn(clean, path, fps=self.fps)
        except Exception as e:
            logger.error("VideoRecorder encode failed: %s", e)
            return None
        logger.info(
            "VideoRecorder: wrote %s (%d frames @ %d fps)", path, len(clean), self.fps
        )
        return path

    def __len__(self) -> int:
        with self._frames_lock:
            return len(self._frames)




_calls = []
_events = []


def reset() -> None:
    _calls.clear()
    _events.clear()
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 - no GPU, no torch: timing still works
        pass


def note(kind: str, **kw) -> None:
    """One run event for the log (for example a re-detect that found the object moved)."""
    _events.append({"kind": kind, "t": round(time.time(), 2), **kw})


def snapshot() -> dict:
    calls = list(_calls)
    out = {"calls": calls, "events": list(_events)}
    for fn in sorted({c["fn"] for c in calls}):
        mine = [c["s"] for c in calls if c["fn"] == fn]
        out[fn] = {"n": len(mine), "total_s": round(sum(mine), 3)}
    try:
        import torch
        out["gpu_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        out["gpu_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)
    except Exception:  # noqa: BLE001
        pass
    return out


def _wrap(cls, name: str, fn: str) -> None:
    f = getattr(cls, name, None)
    if f is None or getattr(f, "_timed", False):
        return

    @functools.wraps(f)
    def timed(*a, **k):
        by = sys._getframe(1).f_code.co_name
        # callers above `by`: tells a recovery re-detect from a pre-step one
        via, frame = [], sys._getframe(1).f_back
        while frame is not None and len(via) < 5:
            via.append(frame.f_code.co_name)
            frame = frame.f_back
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            _calls.append({"fn": fn, "by": by, "via": via, "tag": k.get("tag"), "s": round(time.perf_counter() - t, 3)})
    timed._timed = True
    setattr(cls, name, timed)


def install() -> None:
    from spark_core.llm import LLMClient
    from spark_real.control.motion import WristRefineMixin
    from spark_real.perception_steps import CaptureMixin
    _wrap(LLMClient, "generate", "llm")
    _wrap(CaptureMixin, "detect", "sam3_detect")
    _wrap(WristRefineMixin, "_refine_with_wrist", "wrist_reground")


if __name__ == "__main__":
    class _P:
        def detect(self, tag=None):
            time.sleep(0.01)
            return 1

    def recovery(p):
        return p.detect(tag="x")

    _wrap(_P, "detect", "sam3_detect")
    _wrap(_P, "detect", "sam3_detect")   # second wrap is a no-op
    reset()
    assert recovery(_P()) == 1
    s = snapshot()
    assert s["sam3_detect"]["n"] == 1 and s["calls"][0]["by"] == "recovery" and s["calls"][0]["tag"] == "x", s
    assert 0.009 < s["sam3_detect"]["total_s"] < 0.1, s
    reset()
    assert snapshot()["calls"] == []
    print("ok", s)
