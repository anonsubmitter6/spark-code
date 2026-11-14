"""Shared server state accessible by all route modules."""


from pydantic import BaseModel
from typing import List, Optional
from typing import Optional
import asyncio
import threading


class DetectRequest(BaseModel):
    prompts: List[str]
    multi_instance: bool = False
    # Optional task string. When it resolves in the prompt registry the task's
    # declared instance ordering and per-instance-Z policy are applied to the
    # merged set, so labels match what a cached behaviour tree expects. An
    # empty prompts list then falls back to the task's registered prompts.
    instruction: Optional[str] = None


class ClickDetectRequest(BaseModel):
    # Non-empty -> persist this annotation for that task so /api/execute
    # can reuse it. Empty/None -> one-off rescue, nothing is stored.
    save_for_task: Optional[str] = None
    x: float
    y: float
    img_width: float
    img_height: float
    label: str = ""


class BoxDetectRequest(BaseModel):
    # Non-empty -> persist this annotation for that task so /api/execute
    # can reuse it. Empty/None -> one-off rescue, nothing is stored.
    save_for_task: Optional[str] = None
    x1: float
    y1: float
    x2: float
    y2: float
    img_width: float
    img_height: float
    label: str = ""


class PlanRequest(BaseModel):
    instruction: str
    prompts: Optional[List[str]] = None
    temperature: Optional[float] = None


class ExecuteRequest(BaseModel):
    instruction: str
    prompts: Optional[List[str]] = None
    dry_run: bool = False
    save_to_library: Optional[bool] = None
    record_video: bool = False
    temperature: Optional[float] = None
    # Per-run closed-loop override. None = use config.max_task_passes.
    # >1 re-perceives and re-plans over what remains after each pass until
    # the goal holds (handles dropped/missed objects and scene perturbation).
    closed_loop_passes: Optional[int] = None


class DetectApproveRequest(BaseModel):
    instruction: str
    prompts: Optional[List[str]] = None
    # Low-light test path: when True, swap each Kinect's RGB frame for
    # its active-IR image before feeding to SAM3. The wrist camera (if
    # present) stays on its normal RGB since RealSense doesn't have an
    # IR substitute via this code path.
    use_ir: bool = False


class GripperRequest(BaseModel):
    action: str


class GraspTestRequest(BaseModel):
    """
    Calibrate a width-targeted grasp on a known object.

    Drives a single grasp_to_width(target_width, force) cycle and
    returns the achieved width plus the TCP-force delta. Used by
    scripts/grasp_profile.py to build a per-category calibration
    table that informs the planner prompt defaults.
    """

    target_width: float
    force: float = 20.0
    speed: float = 0.05
    settle_s: float = 1.5
    label: Optional[str] = None
    release_after: bool = True


class GripperPositionRequest(BaseModel):
    # 0.0 = fully open, 1.0 = fully closed. Continuous; analog gamepad
    # triggers stream this at a modest rate.
    position: float
    speed: Optional[float] = None
    force: Optional[float] = None


class MoveRequest(BaseModel):
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0


class VelocityRequest(BaseModel):
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wrx: float = 0.0
    wry: float = 0.0
    wrz: float = 0.0
    duration: float = 0.25


class ConnectRobotRequest(BaseModel):
    # Empty string = use the currently configured robot_ip on the pipeline
    # (set by server.py from --ip or the machine overlay).
    robot_ip: str = ""


class AnchorRequest(BaseModel):
    x: float
    y: float
    img_width: float
    img_height: float


class SwitchCameraRequest(BaseModel):
    camera: str


class CalibPointRequest(BaseModel):
    robot_x: float
    robot_y: float
    robot_z: float
    camera_x: float
    camera_y: float
    camera_z: float


pipeline = None
pipeline_lock = asyncio.Lock()

# The SparkConfig the server was LAUNCHED with (set by server.main before
# uvicorn starts). /api/initialize builds from this so a --no-init boot on
# rebuilds the launched configuration, not a default one.
launch_config = None

# BT library runtime overrides for the LLM-vs-cache resolver in
# pipeline.plan() / pipeline.run_task(). Both default to None so the
# legacy SPARK_DISABLE_LLM env var still governs at boot.
#
# use_cached_bt: None = fall back to env var, True/False = force.
#   Toggled by POST /api/bt/toggle_cache from the frontend panel.
# pinned_bt_hash: single-shot pointer to a specific cached BT. plan()
#   consumes and clears it so the next call returns to normal resolution.
use_cached_bt: Optional[bool] = None
pinned_bt_hash: Optional[str] = None

# RF-DETR second-opinion runtime override, same shape as use_cached_bt:
# None = fall back to config (YAML, then SPARK_PROPOSER_MODE), otherwise one of
# pipeline_perception.PROPOSER_MODES ("auto" / "always" / "off"). Written only
# by POST /api/perception/proposer_mode, which also drops the pipeline's cached
# fusion gate -- setting this alone is inert until the gate is rebuilt.
proposer_mode: Optional[str] = None

# Pointed-slot placement (control/place.py): None = placement.pointed_slots from the
# profile, True/False = forced by POST /api/placement/pointed_slots. Not persisted: a restart
# returns to the profile.
pointed_slots: Optional[bool] = None

active_camera = "sideview"

# The most recently published frame, as ONE consistent (rgb, depth, camera)
# triple. Always publish with set_last_frame() and consume with
# get_last_frame(); never assign or read the three names directly.
#
# Three producers write it concurrently (capture_frame_sync on the
# _STREAM_POOL, the /ws/camera loop, /api/capture). Read as three separate
# globals, a tiled publish interleaved with a single-camera publish hands
# /api/detect_click one camera's RGB, another camera's depth and a third
# camera's extrinsics; the Kinects share a resolution, so nothing raises and
# the arm moves to a coordinate that corresponds to nothing in the scene.
last_rgb = None
last_depth = None
last_cam_name = None
_last_frame_lock = threading.Lock()


def set_last_frame(rgb, depth, cam_name):
    """Publish an rgb/depth/camera triple as one indivisible update."""
    global last_rgb, last_depth, last_cam_name
    with _last_frame_lock:
        last_rgb = rgb
        last_depth = depth
        last_cam_name = cam_name


def get_last_frame():
    """
    Return the last published ``(rgb, depth, cam_name)``, guaranteed to be
    from a single camera and a single capture.

    Consumers must take all three from ONE call. Re-reading
    ``state.last_depth`` afterwards reintroduces exactly the tear this
    exists to remove.
    """
    with _last_frame_lock:
        return last_rgb, last_depth, last_cam_name

# Progress log: pipeline steps pushed here, frontend polls /api/progress.
#
# Entries are {"seq": int, "ts": float, "msg": str}. `seq` is a monotonic
# counter that NEVER resets while the buffer trims, and it is what the
# frontend uses to ask for "everything after N" (GET /api/progress?since=N).
#
# A client-side list index would silently point at the wrong element once
# the buffer trims from the front.
#
# progress_seq is guarded by progress_lock, like the buffer itself.
progress_log: list = []
progress_seq: int = 0
progress_lock = threading.Lock()

# Buffer cap. Generous because entries are small and `since` means a poll
# only ever transfers what is new; the cap only bounds how far a client can
# fall behind before it misses lines.
PROGRESS_MAX = 500

# Pending detections waiting for user approval
pending_detections = None
pending_captures = None
pending_instruction = None
pending_prompts = None

# Camera stream state
stream_mode = "rgb"
stream_camera = "all"

# Tile layout for click-to-camera mapping
tile_layout = []

# EMA-smoothed velocity for teleop
smoothed_vel = None
# EMA smoothing factor for teleop velocity. 0.4 is the default; higher
# values reduce smoothing and increase jitter.
VEL_ALPHA = 0.4

# Maximum downward (-Z, world-frame) velocity for /api/velocity commands.
# Caps both teleop and any other client that streams Cartesian velocity to
# the robot, so server-side safety doesn't depend on the frontend obeying
# its own gain map. Only descent gets clamped; None = no cap.
MAX_DOWN_VEL = 0.03

# Execution serialization. Acquired NON-BLOCKING at the top of
# /api/execute, /api/execute_approved and /api/run_bt; a second request
# while one run is in flight gets 409 instead of a second ScoreExecutor
# traversal over the same driver (interleaved URScript programs replace
# each other on the controller; the second run's completion would clear
# executor_running while the first is still driving the arm).
execute_lock = threading.Lock()
# Start gate (SPARK_START_GATE=1): a planned run waits for POST /api/go before the first motion.
go_event = threading.Event()
go_cancelled = False

# Executor mutex. True while /api/execute_approved / /api/execute is
# actively driving the arm. While True, teleop endpoints
# (/api/velocity, /api/move_relative) become no-ops so they don't
# interleave Cartesian-velocity commands with the executor's
# Cartesian-pose / joint motions.
executor_running = False

# Calibration mutex. True while /api/calibrate/move_to_pose is
# driving the CartesianServo. Same rationale as executor_running:
# any concurrent /api/velocity call (from teleop with stick drift
# past the deadzone) would fight the servo, leaving the robot
# averaged near zero motion, hitting a servo timeout.
cal_active = False

# Current episode recorder (EpisodeRecorder instance). Holds the
# cache for the most recent run (success or fail). Cleared on the
# next execute_approved call. The user reviews via the result panel
# and saves with /api/episode/save (renames cache -> episodes_kept/)
# or discards with /api/episode/discard.
current_episode = None
