"""Camera basics: the process-wide USB lifecycle lock, teardown tracking, CameraConfig and a plain USB camera."""


from __future__ import annotations


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from typing import Optional, Tuple
from typing import Tuple
import configparser
import glob
import logging
import numpy as np
import os
import threading
import time as _time


from spark_real.calibration import CameraCalibration


try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
try:
    import yaml
except ImportError:
    yaml = None


# One process-wide lock serializing EVERY USB lifecycle transition (device
# open, close, hardware_reset, bus re-enumeration) across all camera classes.
# Steady-state streaming is fine; concurrent lifecycle churn corrupts the
# xHCI controller's TRB rings (kernel logs "Event dma ... not part of TD")
# and can hard-kill the host. RLock so a composite transition (e.g.
# /api/kinect_reset closing then reopening) can nest the per-device close
# under its own hold.
USB_LIFECYCLE_LOCK = threading.RLock()

# Count of device teardowns whose SDK call has been ABANDONED but may still
# be executing. close() runs device.stop() on a worker and gives up waiting
# after 4 s, releasing USB_LIFECYCLE_LOCK while libk4a may still be issuing
# SET_INTERFACE, endpoint halts and hub traffic on the shared controller.
# Anything that would touch the bus (uhubctl power-cycle, sysfs deauthorize,
# hardware_reset, a fresh open) must refuse while this is non-zero.
# Incremented before the release worker starts, decremented by the worker
# itself, so it stays set for as long as the SDK call really runs.
_TEARDOWN_LOCK = threading.Lock()

_TEARDOWN_IN_FLIGHT = 0


def teardown_in_flight() -> bool:
    """True while an abandoned-but-still-running device teardown may be
    driving USB traffic. See _TEARDOWN_IN_FLIGHT."""
    with _TEARDOWN_LOCK:
        return _TEARDOWN_IN_FLIGHT > 0


def _teardown_begin() -> None:
    global _TEARDOWN_IN_FLIGHT
    with _TEARDOWN_LOCK:
        _TEARDOWN_IN_FLIGHT += 1


def _teardown_end() -> None:
    global _TEARDOWN_IN_FLIGHT
    with _TEARDOWN_LOCK:
        _TEARDOWN_IN_FLIGHT = max(0, _TEARDOWN_IN_FLIGHT - 1)


def await_teardown_clear(timeout: float = 15.0, poll: float = 0.1) -> bool:
    """
    Block until no abandoned teardown is in flight. Returns False on timeout.

    Callers about to touch the bus (open, reset, power-cycle) use this
    instead of assuming USB_LIFECYCLE_LOCK covered the teardown -- it does
    not once close() has stopped waiting for libk4a.
    """
    deadline = _time.monotonic() + timeout
    while teardown_in_flight():
        if _time.monotonic() >= deadline:
            return False
        _time.sleep(poll)
    return True


def usb_reset_disabled() -> bool:
    """
    True (reset paths OFF) unless SPARK_ALLOW_USB_RESET is set.

    Every software-initiated USB reset path (the Kinect capture loop's
    bounded device stop/start, the RealSense hardware_reset recovery, and
    /api/kinect_reset) fires unattended, precisely when the controller is
    ALREADY misbehaving (40 consecutive capture failures, 25 consecutive
    frame timeouts), and reset traffic on a sick controller precedes the
    ring desync in the kernel log. A degraded wrist camera is strictly
    better than a dead host.

    SPARK_NO_USB_RESET still wins if set. Read at call time, not import time.
    """
    def _set(name: str) -> bool:
        # kill switch: any value except empty/0/false counts, so a typo cannot re-enable resets
        return os.environ.get(name, "") not in ("", "0", "false", "False")
    if _set("SPARK_NO_USB_RESET"):
        return True
    return not _set("SPARK_ALLOW_USB_RESET")


class CameraConfig:
    """
    Camera intrinsics and extrinsics.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fx: float = 615.0,
        fy: float = 615.0,
        cx: float = 320.0,
        cy: float = 240.0,
        extrinsic: np.ndarray = None,
    ):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.extrinsic = extrinsic if extrinsic is not None else np.eye(4)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """
        3x3 camera intrinsic matrix.
        """
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def backproject(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        Backproject pixel + depth to 3D camera frame.
        """
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        return np.array([x, y, depth])

    def to_world(self, point_camera: np.ndarray) -> np.ndarray:
        """
        Transform point from camera frame to world frame.
        """
        p_hom = np.append(point_camera, 1.0)
        return (self.extrinsic @ p_hom)[:3]

    @classmethod
    def from_yaml(cls, path: str) -> "CameraConfig":
        """
        Load from YAML calibration file.
        """
        if yaml is None:
            raise RuntimeError("PyYAML not installed; CameraConfig.from_yaml unavailable")

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            width=data.get("width", 640),
            height=data.get("height", 480),
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            extrinsic=(
                np.array(data["extrinsic"]).reshape(4, 4)
                if "extrinsic" in data
                else None
            ),
        )


class USBCamera:
    """
    Simple USB camera via OpenCV.
    """

    def __init__(self, device_id: int = 0, config: CameraConfig = None):
        self.device_id = device_id
        self.config = config or CameraConfig()
        self._cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self.device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.device_id}")

    def read(self, **kwargs) -> Tuple[np.ndarray, None]:
        """
        Returns (rgb_image, None). No depth from a USB camera; accepts and
        ignores depth/ir kwargs so registry-wide read loops (streaming) can
        call every camera uniformly.
        """
        if self._cap is None:
            return None, None
        ret, frame = self._cap.read()
        if not ret:
            return None, None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None

    def close(self):
        if self._cap:
            self._cap.release()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


try:
    import pyrealsense2 as rs

    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

logger = logging.getLogger(__name__)

# Shared librealsense context. On the RSUSB backend (mandatory on this host;
# see /etc/modprobe.d/blacklist-uvcvideo.conf) every rs.context() re-inits
# libusb and re-walks the bus, so fresh contexts per call multiply
# enumeration traffic. One context per process, created on first use.
_RS_CONTEXT = None


def _get_rs_context():
    global _RS_CONTEXT
    if _RS_CONTEXT is None:
        _RS_CONTEXT = rs.context()
    return _RS_CONTEXT


class RealSenseCamera:
    """
    Intel RealSense D435/D455 RGB-D camera with background capture thread.
    """

    # hardware_reset recovery is bounded PER PROCESS (class-level counter):
    # a reset storm on a wedged link is a prime xHCI-corruption aggravator on
    # this host, where the D435i must run direct to the motherboard and a
    # reset is anything but cheap. Gaps between attempts back off 30/60 s.
    MAX_RECOVER_ATTEMPTS = 3
    RECOVER_BACKOFF_BASE_S = 30.0
    CLOSE_JOIN_TIMEOUT_S = 10.0
    _recover_attempts_total = 0
    _last_recover_ts = None

    def __init__(
        self, serial: str = None, config: CameraConfig = None, color_only: bool = False
    ):
        if not HAS_REALSENSE:
            raise ImportError("pyrealsense2 not installed. pip install pyrealsense2")
        self.serial = serial
        self.config = config or CameraConfig()
        # Stream color only (skip depth) to roughly halve the SuperSpeed UVC
        # bandwidth: the D435i color+depth+IMU triplet on a marginal link
        # desyncs the xHCI event ring ("Event dma ... not part of TD" on the
        # video endpoint) and the stream wedges. The wrist is a 2D camera, so
        # depth is dead weight.
        self._color_only = bool(color_only)
        # Env override (reversible test hook): SPARK_WRIST_COLOR_ONLY=0 re-enables
        # depth without touching config, e.g. to trial wrist-depth refinement.
        _env = os.environ.get("SPARK_WRIST_COLOR_ONLY")
        if _env is not None:
            self._color_only = _env not in ("0", "false", "False", "")
        self._pipeline = None
        self._align = None
        self._latest_rgb = None
        self._latest_depth = None
        self._lock = __import__("threading").Lock()
        self._thread = None
        self._running = False
        # Set once at the start of shutdown (request_stop). The capture loop
        # and recovery path poll it so close() can interrupt a recovery
        # mid-flight instead of racing it.
        self._closing = threading.Event()
        self._reset_disabled_logged = False
        self._recover_exhausted_logged = False
        # Teardown reentrancy state, same rationale as AzureKinectCamera:
        # pipeline.shutdown(), /api/kinect_reset, the SIGTERM handler and
        # atexit can all reach close() and none of them coordinate. Without
        # this, two closers each call rs.pipeline.stop() on the SAME
        # librealsense handle.
        self._close_lock = threading.Lock()
        self._closed = False
        self._close_in_progress = False

    def open(self):
        # Probe the USB link before configuring streams: D435i firmware needs
        # SuperSpeed for the color+depth+IMU triplet, so a USB 2.x link falls
        # back to color-only at low fps for calibration and 2D detection.
        # The whole open (probe + stream start) is a lifecycle transition and
        # is serialized against every other open/close/reset in the process.
        with USB_LIFECYCLE_LOCK:
            usb_type = "3.2"
            try:
                ctx = _get_rs_context()
                for d in ctx.query_devices():
                    if (
                        not self.serial
                        or d.get_info(rs.camera_info.serial_number) == self.serial
                    ):
                        usb_type = d.get_info(rs.camera_info.usb_type_descriptor)
                        break
            except Exception:
                pass
            usb2_mode = usb_type.startswith("2.")
            if usb2_mode:
                logger.warning(
                    "RealSense link is USB %s (not SuperSpeed): depth disabled, "
                    "color forced to 6 fps. Replace the cable with a SuperSpeed-"
                    "rated USB-A/C to USB-C to restore depth + 30 fps.",
                    usb_type,
                )

            self._usb2_mode = usb2_mode
            self._start_pipeline()

        # Start background capture thread
        self._running = True
        self._thread = __import__("threading").Thread(
            target=self._capture_loop, daemon=True
        )
        self._thread.start()

    def _start_pipeline(self):
        """
        Build and start the stream pipeline.

        Shared by open() and the capture-loop recovery path, so it must be
        safe to call repeatedly.
        """
        self._pipeline = rs.pipeline()
        rs_config = rs.config()
        if self.serial:
            rs_config.enable_device(self.serial)
        color_fps = 6 if self._usb2_mode else 15
        rs_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.rgb8,
            color_fps,
        )
        self._depth_enabled = False
        if not self._usb2_mode and not self._color_only:
            try:
                rs_config.enable_stream(
                    rs.stream.depth,
                    self.config.width,
                    self.config.height,
                    rs.format.z16,
                    15,
                )
                self._depth_enabled = True
            except Exception:
                pass
        profile = self._pipeline.start(rs_config)
        if self._depth_enabled:
            self._align = rs.align(rs.stream.color)
        else:
            self._align = None

        intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.config.fx = intrinsics.fx
        self.config.fy = intrinsics.fy
        self.config.cx = intrinsics.ppx
        self.config.cy = intrinsics.ppy

        # Drop warmup frames so the first read() is usable: early frames are
        # underexposed while auto-exposure settles and depth comes back nearly
        # empty. Tolerate early timeouts so a flaky link cannot wedge open.
        for _ in range(30):
            if self._closing.is_set():
                break
            try:
                self._pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                break

    def _recovery_allowed(self) -> bool:
        """
        Gate on hardware_reset recovery: kill-switch, per-process attempt
        bound, and exponential backoff between attempts.

        Returns False (with one-time logging) instead of raising so the
        capture loop can keep serving whatever frames still arrive; a
        disallowed recovery means the wrist camera degrades to no-frames
        rather than generating reset traffic on the shared controller.
        """
        if self._closing.is_set() or not self._running:
            return False
        if teardown_in_flight():
            # A Kinect stop() is still running on the shared controller even
            # though USB_LIFECYCLE_LOCK is free. A hardware_reset now is
            # exactly the overlapping-lifecycle pattern the lock exists to prevent.
            logger.warning(
                "RealSense: teardown in flight on another camera; skipping "
                "hardware_reset recovery"
            )
            return False
        if usb_reset_disabled():
            if not self._reset_disabled_logged:
                logger.warning(
                    "USB reset paths disabled (SPARK_ALLOW_USB_RESET unset): "
                    "RealSense hardware_reset recovery off; wrist camera "
                    "degrades to no-frames until replug/restart"
                )
                self._reset_disabled_logged = True
            return False
        cls = RealSenseCamera
        if cls._recover_attempts_total >= self.MAX_RECOVER_ATTEMPTS:
            if not self._recover_exhausted_logged:
                logger.warning(
                    "RealSense: %d/%d hardware_reset attempts used this "
                    "process; giving up on reset recovery (wrist camera "
                    "degraded). Replug or restart to retry.",
                    cls._recover_attempts_total,
                    self.MAX_RECOVER_ATTEMPTS,
                )
                self._recover_exhausted_logged = True
            return False
        if cls._last_recover_ts is not None:
            # Gap after attempt N is base * 2**(N-1): 30 s, 60 s, ...
            gap = self.RECOVER_BACKOFF_BASE_S * (
                2.0 ** (cls._recover_attempts_total - 1)
            )
            if _time.monotonic() - cls._last_recover_ts < gap:
                return False
        return True

    def _try_recover(self):
        """
        Recover a wedged D435i via hardware_reset.

        The camera can wedge into frame-timeout storms (USB link up, zero
        frames); hardware_reset clears it short of replugging. Stop the dead
        pipeline, reset, wait for re-enumeration, restart streams.

        Bounded (MAX_RECOVER_ATTEMPTS per process, exponential backoff) and
        serialized under USB_LIFECYCLE_LOCK: a reset must never overlap a
        Kinect open/close on the same xHCI controller. Honors the
        reset kill-switch via _recovery_allowed(), and aborts
        early if close() is racing it (self._closing).
        """
        if not self._recovery_allowed():
            return False
        with USB_LIFECYCLE_LOCK:
            if self._closing.is_set():
                return False
            cls = RealSenseCamera
            cls._recover_attempts_total += 1
            cls._last_recover_ts = _time.monotonic()
            logger.warning(
                "RealSense: no frames, attempting hardware_reset recovery "
                "(attempt %d/%d this process)",
                cls._recover_attempts_total,
                self.MAX_RECOVER_ATTEMPTS,
            )
            try:
                self._pipeline.stop()
            except Exception:
                pass
            try:
                ctx = _get_rs_context()
                for d in ctx.query_devices():
                    if (
                        not self.serial
                        or d.get_info(rs.camera_info.serial_number) == self.serial
                    ):
                        d.hardware_reset()
                        break
                # Re-enumeration takes ~4-5 s. Event-wait so close() can
                # interrupt instead of racing a sleeping recovery.
                if self._closing.wait(6.0):
                    return False
            except Exception as exc:
                logger.warning("RealSense hardware_reset failed: %s", exc)
            # Re-probe the link speed before re-opening: the camera may have
            # been replugged onto a different port since open() (USB2 at boot,
            # SuperSpeed now, or vice versa), and _usb2_mode decides whether
            # depth streams are enabled.
            try:
                ctx = _get_rs_context()
                for d in ctx.query_devices():
                    if (
                        not self.serial
                        or d.get_info(rs.camera_info.serial_number) == self.serial
                    ):
                        usb_type = d.get_info(rs.camera_info.usb_type_descriptor)
                        new_usb2 = usb_type.startswith("2.")
                        if new_usb2 != self._usb2_mode:
                            logger.info(
                                "RealSense link changed: usb2_mode %s -> %s",
                                self._usb2_mode,
                                new_usb2,
                            )
                            self._usb2_mode = new_usb2
                        break
            except Exception:
                pass
            if self._closing.is_set():
                return False
            try:
                self._start_pipeline()
                logger.info("RealSense: recovered after hardware_reset")
                return True
            except Exception as exc:
                logger.warning("RealSense re-open after reset failed: %s", exc)
                return False

    def _capture_loop(self):
        """
        Background thread: continuously grab frames, keep latest.
        """
        frame_count = 0
        fail_count = 0
        consec_fails = 0
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=500)
                if self._align:
                    aligned = self._align.process(frames)
                    color_frame = aligned.get_color_frame()
                    depth_frame = aligned.get_depth_frame()
                else:
                    color_frame = frames.get_color_frame()
                    depth_frame = None
                if color_frame:
                    rgb = np.asanyarray(color_frame.get_data()).copy()
                    depth = None
                    if depth_frame:
                        depth = (
                            np.asanyarray(depth_frame.get_data()).astype(np.float32)
                            / 1000.0
                        )
                    with self._lock:
                        self._latest_rgb = rgb
                        self._latest_depth = depth
                    frame_count += 1
                    consec_fails = 0
                    if frame_count % 100 == 0:
                        logger.debug("RealSense: %d frames captured", frame_count)
                _time.sleep(0.01)
            except RuntimeError as e:
                fail_count += 1
                consec_fails += 1
                if fail_count <= 3 or fail_count % 50 == 0:
                    logger.warning("RealSense frame timeout #%d: %s", fail_count, e)
                # Self-heal: a long run of consecutive misses means the link
                # is wedged, not flaky, so try to recover. _recovery_allowed
                # enforces the kill-switch, the per-process attempt bound,
                # and exponential backoff between attempts.
                if consec_fails >= 25 and self._recovery_allowed():
                    if self._try_recover():
                        consec_fails = 0
                # Event-wait instead of sleep so close() interrupts promptly.
                self._closing.wait(0.1)

    def read(self, **kwargs) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns latest cached (rgb, depth). Non-blocking.
        """
        with self._lock:
            return self._latest_rgb, self._latest_depth

    def request_stop(self):
        """
        Non-blocking quiesce: stop the capture loop and abort any in-flight
        recovery at its next checkpoint. Called by pipeline.shutdown() BEFORE
        the Kinects close, so a hardware_reset can never fire while another
        camera on the same controller is mid-close. close() completes the
        teardown afterwards; request_stop() alone touches no USB state.
        """
        self._running = False
        self._closing.set()

    def close(self):
        self.request_stop()
        # LOCK ORDER. _close_lock is held ONLY around the flag transitions,
        # never across _close_locked, which joins the capture thread and
        # then takes USB_LIFECYCLE_LOCK. Nesting _close_lock ->
        # USB_LIFECYCLE_LOCK would invert against
        # /api/kinect_reset._kinect_reset_locked, which already holds
        # USB_LIFECYCLE_LOCK when it reaches a camera close.
        with self._close_lock:
            if self._closed:
                return
            if self._close_in_progress:
                logger.warning(
                    "RealSense: close() already in progress on another "
                    "thread; not racing it"
                )
                return
            self._close_in_progress = True
        try:
            self._close_locked()
        finally:
            with self._close_lock:
                self._close_in_progress = False

    def _close_locked(self):
        """Body of close(); exactly one thread is in here at a time."""
        t = self._thread
        if t is not None:
            # JOIN BEFORE CLOSE: the capture thread owns the pipeline handle
            # (wait_for_frames, recovery). _closing interrupts its sleeps and
            # any recovery at the next checkpoint, so this join is normally
            # near-instant.
            t.join(timeout=self.CLOSE_JOIN_TIMEOUT_S)
            if t.is_alive():
                logger.warning(
                    "RealSense capture thread did not exit within %.0fs; "
                    "SKIPPING pipeline.stop() rather than stopping the "
                    "stream out from under a live reader (that race "
                    "corrupts xHCI state). Device releases when the "
                    "process exits.",
                    self.CLOSE_JOIN_TIMEOUT_S,
                )
                return
            self._thread = None
        pipe = self._pipeline
        if pipe is not None:
            # Drop the reference BEFORE stopping so nothing can hand the same
            # handle to a second stop() even if this one raises.
            self._pipeline = None
            with USB_LIFECYCLE_LOCK:
                try:
                    pipe.stop()
                except Exception:
                    pass
        self._closed = True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


try:
    import pyzed.sl as sl  # type: ignore  # ZED SDK optional
except ImportError:
    sl = None



_DEPTH_MODE_MAP = {
    # Strings come from the YAML config; map to pyzed.sl.DEPTH_MODE enum
    # at runtime (only if the SDK is importable).
    "NEURAL": "NEURAL",
    "ULTRA": "ULTRA",
    "QUALITY": "QUALITY",
    "PERFORMANCE": "PERFORMANCE",
    "STANDARD": "STANDARD",
}


class ZEDMiniCamera:
    """
    Stereo + depth wrapper for the ZED Mini.

    Serves as the primary localization camera for the bimanual rig,
    complementary to the two wrist RealSenses (which are occluded during
    handoff).
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        depth_mode: str = "NEURAL",
        flip: str = "auto",
        simulate: bool = False,
    ):
        self.serial = serial or ""
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.depth_mode = depth_mode.upper()
        # flip: "auto" (SDK picks via IMU), "on" (force 180-degree image flip),
        # "off" (no flip). Use "on" if the ZED is physically mounted inverted
        # AND the SDK's auto detect is not kicking in.
        self.flip = str(flip or "auto").lower()
        self._zed = None
        self._sim_mode = bool(simulate)
        self._opened = False
        self._lock = threading.Lock()
        # CameraConfig doesn't carry a name field; the camera registry
        # holds the role string. Use width*0.7 as a placeholder focal
        # length until open() reads the SDK intrinsics.
        self.config = CameraConfig(
            width=self.width,
            height=self.height,
            fx=self.width * 0.7,
            fy=self.width * 0.7,
            cx=self.width / 2,
            cy=self.height / 2,
        )

    # lifecycle
    def open(self, serial: Optional[str] = None) -> bool:
        if serial:
            self.serial = serial
        if self._sim_mode:
            self._opened = True
            return True
        if sl is None:
            raise RuntimeError(
                "camera type 'zed_mini' needs the ZED SDK and its Python wrapper "
                "(pyzed); not installed on this host"
            )

        zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = self._resolution_for(sl)
        init.camera_fps = self.fps
        # SDK 5.x dropped DEPTH_MODE.STANDARD; resolve the default via
        # getattr too because Python evaluates the default arg eagerly and
        # referencing a missing enum member there crashes even when the
        # primary lookup would succeed.
        _default_depth_mode = getattr(
            sl.DEPTH_MODE, "PERFORMANCE", getattr(sl.DEPTH_MODE, "NEURAL", None)
        )
        init.depth_mode = getattr(
            sl.DEPTH_MODE,
            _DEPTH_MODE_MAP.get(self.depth_mode, "NEURAL"),
            _default_depth_mode,
        )
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        # Image flip control. SDK 5.x exposes sl.FLIP_MODE.{AUTO,ON,OFF}
        # on InitParameters.camera_image_flip. Wrap in getattr so older SDK
        # builds without the enum still work (fall back to no flip).
        _flip_mode_enum = getattr(sl, "FLIP_MODE", None)
        if _flip_mode_enum is not None:
            _flip_choice = {
                "auto": getattr(_flip_mode_enum, "AUTO", None),
                "on": getattr(_flip_mode_enum, "ON", None),
                "off": getattr(_flip_mode_enum, "OFF", None),
            }.get(self.flip, getattr(_flip_mode_enum, "AUTO", None))
            if _flip_choice is not None:
                try:
                    init.camera_image_flip = _flip_choice
                except Exception:
                    logger.warning(
                        "ZED: SDK did not accept camera_image_flip=%s", self.flip
                    )
        if self.serial:
            try:
                init.set_from_serial_number(int(self.serial))
            except Exception:
                logger.warning(
                    "invalid ZED serial %r; opening any available unit", self.serial
                )

        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")

        # Pull intrinsics from the SDK.
        ci = zed.get_camera_information().camera_configuration
        cp_left = ci.calibration_parameters.left_cam
        self.config = CameraConfig(
            width=self.width,
            height=self.height,
            fx=float(cp_left.fx),
            fy=float(cp_left.fy),
            cx=float(cp_left.cx),
            cy=float(cp_left.cy),
        )
        self._zed = zed
        self._opened = True
        return True

    def close(self) -> None:
        if self._zed is not None and not self._sim_mode:
            try:
                self._zed.close()
            except Exception:
                logger.exception("ZED close failed")
        self._opened = False

    # capture
    def read(self, depth: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        with self._lock:
            if self._sim_mode:
                return self._sim_frame(depth=depth)
            if self._zed is None:
                return None, None

            rt = sl.RuntimeParameters()
            err = self._zed.grab(rt)
            if err != sl.ERROR_CODE.SUCCESS:
                logger.warning("ZED grab error %s", err)
                return None, None
            mat_rgb = sl.Mat()
            self._zed.retrieve_image(mat_rgb, sl.VIEW.LEFT)
            rgb = mat_rgb.get_data()[:, :, :3].copy()
            # ZED returns BGR; convert to RGB for SAM3.
            rgb = rgb[..., ::-1].copy()
            depth_np: Optional[np.ndarray] = None
            if depth:
                mat_d = sl.Mat()
                self._zed.retrieve_measure(mat_d, sl.MEASURE.DEPTH)
                depth_np = mat_d.get_data().astype(np.float32)
                # ZED uses NaN for invalid depth; promote to 0 so consumers
                # that expect numeric arrays don't blow up. SAM3 / DA3 paths
                # treat 0 as "no depth".
                depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
            return rgb, depth_np

    def read_stereo(
        self, hires: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Return (left_bgr, right_bgr, depth_f32) from the ZED stereo pair.

        When hires=True, retrieves at the camera's native resolution
        regardless of the configured width/height (for calibration).
        """
        with self._lock:
            if self._sim_mode:
                f, d = self._sim_frame(depth=True)
                return f, f.copy(), d
            if self._zed is None:
                raise RuntimeError("ZED is not open")
            err = self._zed.grab(sl.RuntimeParameters())
            if err != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"ZED grab error {err}")
            mat_l = sl.Mat()
            mat_r = sl.Mat()
            mat_d = sl.Mat()
            self._zed.retrieve_image(mat_l, sl.VIEW.LEFT)
            self._zed.retrieve_image(mat_r, sl.VIEW.RIGHT)
            self._zed.retrieve_measure(mat_d, sl.MEASURE.DEPTH)
            left = mat_l.get_data()[:, :, :3].copy()
            right = mat_r.get_data()[:, :, :3].copy()
            depth = np.nan_to_num(
                mat_d.get_data().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            return left, right, depth

    def get_camera_info(self) -> dict:
        """
        Return SDK intrinsics + baseline for stereo triangulation.
        """
        with self._lock:
            if self._sim_mode or self._zed is None:
                return {"sim": True}
            info = self._zed.get_camera_information()
            cp = info.camera_configuration.calibration_parameters
            return {
                "fx_l": cp.left_cam.fx,
                "fy_l": cp.left_cam.fy,
                "cx_l": cp.left_cam.cx,
                "cy_l": cp.left_cam.cy,
                "fx_r": cp.right_cam.fx,
                "fy_r": cp.right_cam.fy,
                "cx_r": cp.right_cam.cx,
                "cy_r": cp.right_cam.cy,
                "baseline_m": abs(cp.stereo_transform.get_translation().get()[0])
                or self._read_baseline_from_conf(),
            }

    def _read_baseline_from_conf(self) -> float:
        """
        Read baseline from /usr/local/zed/settings/SN*.conf.
        """
        for p in glob.glob("/usr/local/zed/settings/SN*.conf"):
            cfg = configparser.ConfigParser()
            cfg.read(p)
            for sec in cfg.sections():
                if cfg.has_option(sec, "Baseline"):
                    return float(cfg.get(sec, "Baseline")) / 1000.0
        return 0.0

    # simulator
    def _sim_frame(self, depth: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        h, w = self.height, self.width
        # Checker pattern for sanity-checking the frontend tile.
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        sq = 32
        for y in range(0, h, sq):
            for x in range(0, w, sq):
                v = 220 if ((x // sq) + (y // sq)) % 2 else 40
                rgb[y : y + sq, x : x + sq] = (v, v // 2, 255 - v)
        depth_np = None
        if depth:
            depth_np = np.full((h, w), 0.80, dtype=np.float32)
        return rgb, depth_np

    # helpers
    def _resolution_for(self, sl):
        """
        Pick the lowest sl.RESOLUTION >= the requested width.

        ZED Mini supports HD2K (2208x1242), HD1080, HD720, VGA;
        defaults to HD720 if nothing matches.
        """
        if self.width >= 2208:
            return sl.RESOLUTION.HD2K
        if self.width >= 1920:
            return sl.RESOLUTION.HD1080
        if self.width >= 1280:
            return sl.RESOLUTION.HD720
        return sl.RESOLUTION.VGA



# Camera driver registry: config `cameras[].type` -> the module its driver
# needs. Any family may name any type. "azure_kinect" and "usb_camera"/"uvc"
# are accepted spellings of "kinect" and "webcam".
_SDK = {"kinect": "pyk4a", "realsense": "pyrealsense2", "zed_mini": "pyzed.sl", "webcam": "cv2"}
CAMERA_TYPES = tuple(_SDK)
_TYPE_ALIASES = {"azure_kinect": "kinect", "usb_camera": "webcam", "uvc": "webcam"}
# (role, type) pairs the single-arm pipeline opens itself (Kinect sync order,
# retries, RealSense recovery); every other roster entry opens generically.
PIPELINE_OPENED = {("sideview", "kinect"), ("birdview", "kinect"), ("wrist", "realsense")}


def _is_plugin(kind: str) -> bool:
    # A driver outside this package: "package.module:ClassName" (see _open_plugin).
    return ":" in kind


def camera_type(entry: Dict[str, Any]) -> str:
    """The canonical type of a `cameras:` entry; ValueError if no driver is registered."""
    raw = str(entry.get("type") or "")
    if _is_plugin(raw):
        return raw
    raw = raw.lower()
    kind = _TYPE_ALIASES.get(raw, raw)
    if kind not in CAMERA_TYPES:
        raise ValueError(
            f"camera role {entry.get('role')!r}: type {raw or '<missing>'!r} has no driver; "
            f"registered: {CAMERA_TYPES}, or a class as 'package.module:ClassName'"
        )
    return kind


def require_driver(kind: str, role: str = "") -> None:
    """RuntimeError when the type's SDK (its transport) is not installed here."""
    import importlib

    module = kind.split(":")[0] if _is_plugin(kind) else _SDK[kind]
    try:
        importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(
            f"camera role {role!r}: type {kind!r} needs {module} ({exc}); "
            "install its SDK or remove the entry from `cameras:`"
        ) from None


KINECT_ROLES = ("sideview", "birdview")  # the single-arm pipeline's two Kinect slots


def assign_kinect_roles(
    serials: Dict[int, Optional[str]],
    roster: List[Dict[str, Any]],
    master_serial: str = "",
    use_hw_sync: bool = False,
) -> List[tuple]:
    """
    (device_id, role, sync_mode) for each connected Kinect, in open order.

    A Kinect whose serial is in the `cameras:` roster takes that entry's role;
    the others fill the remaining Kinect roles in roster order, so a rig with
    one Kinect, or Kinects whose serials are not set, needs no serials.
    With hardware sync the master is ``master_serial`` when set, else the
    roster entry with `sync: master`, else sideview; subordinates open first.
    """
    entries = [e for e in roster if camera_type(e) == "kinect" and e.get("role") in KINECT_ROLES
               and e.get("enabled", True) is not False]
    roles = [e["role"] for e in entries] or list(KINECT_ROLES)
    roles += [r for r in KINECT_ROLES if r not in roles]
    by_serial = {str(e["serial"]): e["role"] for e in entries if e.get("serial")}
    if master_serial:
        by_serial = {sn: r for sn, r in by_serial.items() if r != "sideview"}
        by_serial[str(master_serial)] = "sideview"
    assigned = {d: by_serial[str(sn)] for d, sn in serials.items() if sn is not None and str(sn) in by_serial}
    free = [r for r in roles if r not in assigned.values()]
    unassigned = [d for d in sorted(serials) if d not in assigned]
    # Two Kinects and no serial says which is which: device order is a coin flip, and a wrong
    # guess swaps the overhead and side extrinsics (every 3D position wrong, no error anywhere).
    if len(unassigned) >= 2 and os.environ.get("SPARK_ALLOW_GUESSED_CAMERA_ROLES", "0") != "1":
        raise RuntimeError(
            "Kinects " + ", ".join(f"{d} (SN={serials[d]})" for d in unassigned) + " match no `serial:` in the "
            "`cameras:` roster, so their roles (sideview / birdview) would be guessed from the device order. "
            "Put each serial under its role in configs/machines/<host>.yaml (see machines/example.yaml) and "
            "start with --machine <host>. SPARK_ALLOW_GUESSED_CAMERA_ROLES=1 keeps the device-order guess."
        )
    for d in sorted(serials):
        if d not in assigned and free:
            assigned[d] = free.pop(0)
            logger.warning("Kinect %d (SN=%s) has no roster serial; using role %r", d, serials[d], assigned[d])
    master_role = "sideview"
    if not master_serial:
        master_role = next((e["role"] for e in entries if str(e.get("sync", "")).lower() == "master"), "sideview")
    out = []
    for d in sorted(assigned):
        role = assigned[d]
        sync = ("MASTER" if role == master_role else "SUBORDINATE") if use_hw_sync else "STANDALONE"
        out.append((d, role, sync))
    out.sort(key=lambda t: t[2] == "MASTER")  # subordinates listen before the master triggers
    return out


@dataclass
class CameraEntry:
    """One open camera with its calibration."""

    role: str  # e.g. 'sideview', 'birdview', 'wrist', 'external', 'wrist_left'
    kind: str  # one of CAMERA_TYPES
    device: Any
    calibration: CameraCalibration
    arm: Optional[str] = None
    tool_offset_xyz: Optional[List[float]] = None  # TCP->camera (wrist cams)
    serial: str = ""


@dataclass
class CameraRegistry:
    """Role-keyed cameras; ``roles`` order is fixed for a stable UI layout."""

    cameras: Dict[str, CameraEntry] = field(default_factory=dict)
    roles: List[str] = field(default_factory=list)
    primary_role: str = ""
    wrist_for_arm: Dict[str, str] = field(default_factory=dict)  # arm -> role

    def __getitem__(self, role: str) -> CameraEntry:
        return self.cameras[role]

    def get(self, role: str) -> Optional[CameraEntry]:
        return self.cameras.get(role)

    def __contains__(self, role: str) -> bool:
        return role in self.cameras

    def __iter__(self):
        for r in self.roles:
            yield r, self.cameras[r]


def spec(profile) -> Dict[str, Dict[str, Any]]:
    """Role -> copy of the YAML camera entry. Entries without a role are skipped."""
    out: Dict[str, Dict[str, Any]] = {}
    for entry in profile.cameras():
        role = entry.get("role")
        if not role:
            logger.warning("camera spec missing 'role'; skipping: %s", entry)
            continue
        camera_type(entry)
        out[role] = dict(entry)
    return out


def registry_from_open_devices(
    entries: List[CameraEntry], primary_role: str = ""
) -> CameraRegistry:
    """Wrap already-open devices; the primary role is listed first."""
    registry = CameraRegistry(primary_role=primary_role)
    for entry in entries:
        registry.cameras[entry.role] = entry
        registry.roles.append(entry.role)
    if primary_role in registry.cameras:
        registry.roles.remove(primary_role)
        registry.roles.insert(0, primary_role)
    return registry


# Roster keys the registry consumes itself; a plugin class gets every other key.
_REGISTRY_KEYS = ("role", "type", "arm", "tool_offset_xyz", "enabled", "sync")


def _open_plugin(kind: str, spec: Dict[str, Any]):
    """
    A camera class named in config as "package.module:ClassName", built with the
    entry's other keys as keyword arguments. It must provide open(), close(),
    read() -> (rgb HxWx3 uint8, depth HxW in meters or None) and a `config` with
    width, height, fx, fy, cx, cy (camera_base.CameraConfig).
    """
    import importlib

    module, _, name = kind.partition(":")
    cls = getattr(importlib.import_module(module), name)
    return cls(**{k: v for k, v in spec.items() if k not in _REGISTRY_KEYS})


def _open_device(kind: str, spec: Dict[str, Any]):
    """Construct and open one camera from its config entry."""

    width, height = int(spec.get("width", 640)), int(spec.get("height", 480))
    if _is_plugin(kind):
        device = _open_plugin(kind, spec)
    elif kind == "zed_mini":

        device = ZEDMiniCamera(
            serial=spec.get("serial") or None,
            width=width,
            height=height,
            fps=int(spec.get("fps", 30)),
            depth_mode=spec.get("depth_mode", "NEURAL"),
            flip=spec.get("flip", "auto"),
        )
    elif kind == "realsense":

        device = RealSenseCamera(serial=spec.get("serial") or None, config=CameraConfig(width=width, height=height))
    elif kind == "kinect":
        from spark_real.perception.kinect import AzureKinectCamera

        device = AzureKinectCamera(
            color_resolution=spec.get("resolution", "720P"),
            depth_mode=spec.get("depth_mode", "WFOV_2X2BINNED"),
            camera_fps=int(spec.get("fps", 15)),
        )
    else:

        # `device`: an index or a /dev/v4l/by-path/... string (identical
        # webcams often share one USB serial, so by-id collides).
        dev_id = spec.get("device", spec.get("device_id"))
        if dev_id is None:
            raise ValueError("webcam `device` is not set")
        device = USBCamera(device_id=dev_id, config=CameraConfig(width=width, height=height))
    device.open()
    return device


def build_camera_registry(
    cam_specs: List[Dict[str, Any]], primary_role: str = "", registry: Optional[CameraRegistry] = None
) -> CameraRegistry:
    """
    Open every camera in a `cameras:` roster, keyed by role (added to
    ``registry`` when given).

    Before anything opens, a type with no driver raises ValueError and a
    type whose SDK is missing raises RuntimeError. A camera that then fails
    to open (unplugged, busy) is logged and left out, so nothing downstream
    reads a dead or simulated device. The primary role is listed first, then
    the left and right wrist cameras.
    """
    kinds = [(spec, camera_type(spec)) for spec in cam_specs if spec.get("role")]
    for spec, kind in kinds:
        require_driver(kind, spec["role"])
    if registry is None:
        registry = CameraRegistry(primary_role=primary_role or (kinds[0][0]["role"] if kinds else ""))
    for spec, kind in kinds:
        role = spec["role"]
        try:
            device = _open_device(kind, spec)
        except Exception as exc:
            logger.error("camera %r (%s) did not open: %s", role, kind, exc)
            continue
        c = device.config
        entry = CameraEntry(
            role=role,
            kind=kind,
            device=device,
            # Extrinsic stays identity until a calibration file sets it.
            calibration=CameraCalibration(
                name=role, width=int(c.width), height=int(c.height),
                fx=float(c.fx), fy=float(c.fy), cx=float(c.cx), cy=float(c.cy),
            ),
            arm=spec.get("arm"),
            tool_offset_xyz=spec.get("tool_offset_xyz"),
            serial=str(spec.get("serial") or ""),
        )
        registry.cameras[role] = entry
        registry.roles.append(role)
        if entry.arm in ("left", "right"):
            registry.wrist_for_arm[entry.arm] = role
    preferred = [registry.primary_role, registry.wrist_for_arm.get("left"), registry.wrist_for_arm.get("right")]
    head = [r for r in preferred if r and r in registry.cameras]
    registry.roles = head + [r for r in registry.roles if r not in head]
    logger.info("camera registry: roles=%s primary=%s", registry.roles, registry.primary_role)
    return registry
