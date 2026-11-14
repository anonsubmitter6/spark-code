from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import logging
import numpy as np
import time


from spark_real.calibration import CameraCalibration
from spark_real.config import calibrations_dir
from spark_real.config import calibrations_dir, family_block
from spark_real.config import output_root
from spark_real.control.bimanual_executor import BimanualScoreExecutor
from spark_real.control.bimanual_safe_robot import BaseFrames, BimanualSafeRobot, InterArmConfig
from spark_real.control.executor import ScoreExecutor
from spark_real.control.safe_robot import SafeRobot, SafetyConfig
from spark_real.control.safe_robot import SafetyConfig
from spark_real.perception.cameras import CameraEntry
from spark_real.perception.cameras import PIPELINE_OPENED, build_camera_registry, camera_type, spec as camera_spec
from spark_real.perception.cameras import assign_kinect_roles, registry_from_open_devices
from spark_real.perception.cameras import build_camera_registry
from spark_real.perception.kinect import AzureKinectCamera
from spark_real.robots.factory import make_robot_driver


@dataclass
class PipelineConfig:
    """
    Configuration for the real robot pipeline.
    """

    # robot_ip is set by server.py from --ip / configs/<family>_default.yaml.
    # The empty default is a sentinel meaning "no robot configured"; code
    # paths that try to connect should treat "" as a "skip connect" signal.
    robot_ip: str = ""

    # Robot identity surfaced to the frontend so the UI can swap banner +
    # control panel per family. server.py overrides via --robot.
    robot_family: str = "ur10e"
    robot_model: str = "UR10e"  # human-readable model string

    use_kinect: bool = True
    use_realsense: Optional[bool] = None
    """
    None = auto (init iff a RealSense is enumerated); True/False forces.
    """
    # 720P halves the color stream's USB isochronous reservation vs 1080P;
    # intrinsics follow the resolution, so hand-eye stays valid.
    kinect_resolution: str = "720P"
    kinect_master_serial: str = ""  # names the sideview (master) Kinect
    kinect_use_hw_sync: bool = False  # Set True only if sync cable connected
    # Depth modes (Azure Kinect DK):
    #   NFOV_UNBINNED:  75x65 FOV,  640x576, 30 fps, range 0.5-3.86 m
    #   NFOV_2X2BINNED: 75x65 FOV,  320x288, 30 fps, range 0.5-5.46 m
    #   WFOV_UNBINNED:  120x120,    1024x1024, 15 fps, range 0.25-2.21 m
    #   WFOV_2X2BINNED: 120x120,    512x512,  30 fps, range 0.25-2.88 m
    # WFOV_2X2BINNED is the right default for tabletop manipulation:
    # full 30 fps for the VLA loop, wide enough that one Kinect sees the
    # whole cell, and close-range support down to 25 cm. Override in the
    # per-family YAML's `kinect_depth_mode` field.
    kinect_depth_mode: str = "WFOV_2X2BINNED"
    # 15 fps halves USB and depth-engine load vs 30 with two Kinects on one
    # host controller.
    kinect_fps: int = 15
    realsense_serial: str = ""
    # Stream the wrist RealSense color-only (no depth) to halve SuperSpeed USB
    # bandwidth and avoid the xHCI event-ring desync that drops the stream.
    # Depth is unused while wrist_refine is off; set False to restore it.
    realsense_color_only: bool = True

    sam3_threshold: float = 0.03
    use_hardware_depth: bool = True

    planner_temperature: float = 0.3

    velocity: float = 0.25
    safe_height: float = 0.35
    # Recovery search floor (robot frame, open-gripper TCP at the work
    # surface). None means: use the executor's config-fed TABLE_Z_FLOOR
    # for the active family. Set per machine in the family YAML
    # (table_height) only when the rig has a raised surface like the
    # original UR10e cell's foam mat (-0.28).
    table_height: float | None = None

    output_dir: str = field(default_factory=lambda: str(output_root() / "real_runs"))
    save_captures: bool = True

    # Behavior-tree cache. Once a task has one verified run in the library,
    # every later run of it is served from disk with no LLM call. Set via the
    # `bt:` block in configs/<family>_default.yaml.
    bt_library_dir: Optional[str] = None
    """Library root. None => <output_dir>/../bt_library (NOT cwd-relative)."""
    bt_seed_dir: Optional[str] = None
    """Seed YAML root. None => the packaged configs/bt_seeds."""
    bt_frozen: bool = False
    """
    Freeze the BT library: serve NOTHING as few-shot context and record
    nothing back.

    The no-adaptation control: no trial of a task is seen before that task is
    scored, so with bt_frozen set, trial N plans with exactly the context
    trial 1 had.

    Env override: SPARK_BT_FROZEN=1.
    """

    bt_seed_on_start: bool = True
    """
    Install the packaged seed trees into the library at startup.

    Turn this OFF when collecting a fresh demonstration corpus: a seeded tree
    is cache-eligible immediately, so the task would be served from disk
    instead of being planned, and the run you meant to record fresh never
    calls the planner at all.
    """
    bt_plan_mode: str = "auto"
    # The recovery ladder's terminal rung: after every LLM-free rung fails
    # WITH evidence, consult the planner once with the failure context.
    # Off: surface the failure and stop.
    llm_last_resort: bool = True
    """auto (cache first, LLM on miss) | cache_only (miss raises) | llm."""
    bt_min_similarity: float = 0.75
    """Jaccard floor for fuzzy retrieval. Below it is a miss, not a hit."""
    bt_auto_promote_after: int = 3
    """Successes (with zero failures) before an entry auto-promotes."""

    # Frozen per-task SAM3 prompt contracts (prompts, expected object counts,
    # deterministic instance ordering). None => the packaged configs/tasks.
    # Also overridable by $SPARK_TASK_PROMPTS.
    task_prompts_dir: Optional[str] = None
    """Directory of task prompt YAMLs; see perception/prompts.py."""

    # Experimental: strict placement verification.
    # When True, replaces the binary mask-intersection check with a
    # containment ratio (>= 50% of the picked object's mask must lie
    # inside the place target's mask), allows the post-task retry to
    # re-detect the placed object's current location, and caps retries
    # at 1. Catches "tumbled out next to container" failures that the
    # legacy verify reports as success.
    # Off by default, flip via --strict-verify on the server CLI.
    strict_placement_verify: bool = False

    save_to_library: bool = True
    """
    If True, every successful task appends its (instruction, score) to bt_library/.
    """

    # Closed-loop task execution (generic, all families).
    # After a behavior tree runs, the pipeline re-perceives the scene and
    # checks whether the target objects actually reached their destination
    # (in a receptacle, swept into a dustpan, lifted off the source). Any
    # target that did NOT make it, plus any object the scene was perturbed
    # with mid-task (swapped utensil, added item, moved receptacle), drives
    # another plan-and-execute pass. Already-handled objects stay closed via
    # the destination filter and the executor's placed-label set, so the
    # re-plan never re-visits a node already taken care of.
    # max_task_passes == 1 is single shot.
    max_task_passes: int = 1
    """Outer closed-loop cap. 1 = single shot (legacy). >1 re-perceives and
    re-plans over what remains until the goal holds or the budget is spent."""
    closed_loop_settle_s: float = 0.8
    """Seconds to let the scene settle (arm home, objects still) before the
    between-pass re-perception so SAM3 sees the workspace, not motion."""
    closed_loop_clear_radius_m: float = 0.12
    """A target whose centroid is within this distance of a destination
    centroid is treated as handled (inside the receptacle / swept in). Used
    as a floor; the destination's own OBB minor radius widens it per object."""

    use_grasp_calibration: bool = True
    """
    If True, ScoreExecutor uses per-object learned grasp depth offsets.
    """

    # Wrist-camera translation from the TCP (identity rotation). None: use
    # <calibrations>/handeye_wrist.json, else a placeholder offset.
    wrist_tool_offset_xyz: Optional[List[float]] = None


@dataclass
class TaskResult:
    """
    Result from a single task execution.
    """

    instruction: str
    timestamp: str
    detections: List[Dict]
    plan: Dict
    execution_results: List[Dict]
    success: bool
    duration: float
    captures: Dict[str, str] = field(default_factory=dict)

    # Plan provenance. Declared (not just set as ad-hoc attributes) because
    # the recorder copies them verbatim into metadata["spark"], so an episode
    # on disk says exactly which tree drove it and where that tree came from.
    plan_source: Optional[str] = None
    """pin | exact | alias | jaccard | seed | llm -- see BTLibrary.resolve."""
    bt_hash: Optional[str] = None
    """Content hash of the executed behavior tree, or None if unlibraried."""
    label_resolutions: Dict[str, str] = field(default_factory=dict)
    """Cached-BT label -> the differently-named detection it re-bound to."""
    # Task-success verdict (control/success_verifier.VerifyOutcome.to_dict()).
    # ``success`` above is exactly ``verify_status == "pass"``; this carries the
    # evidence -- predicates, per-camera votes, gates, depth source.
    verify: Optional[Dict] = None
    """Full VerifyOutcome dict, or None when no verdict was produced."""
    verify_status: str = "unverified"
    """pass | fail | unverified. `unverified` must not train and must not
    reinforce a cached tree; it is not a quieter kind of failure."""


logger = logging.getLogger(__name__)

try:
    import pyk4a

    HAS_PYK4A = True
except ImportError:
    HAS_PYK4A = False


# Camera, calibration, and robot bring-up for SPARKRealPipeline.
# Mixed into the pipeline class; uses attributes set in __init__.
class InitMixin:

    def _resolve_wrist_tool_offset(self) -> np.ndarray:
        """
        Build the 4x4 TCP-to-camera transform.

        Priority order:
          1. Explicit ``wrist_tool_offset_xyz`` in PipelineConfig
             (translation-only override).
          2. Measured hand-eye calibration at
             ``<calibrations_dir>/handeye_wrist.json``: the FULL 4x4.
          3. A placeholder translation (identity rotation).
        """
        offset_xyz = self.config.wrist_tool_offset_xyz
        if offset_xyz is not None:
            T = np.eye(4)
            T[0, 3] = float(offset_xyz[0])
            T[1, 3] = float(offset_xyz[1])
            T[2, 3] = float(offset_xyz[2])
            return T

        handeye_path = calibrations_dir() / "handeye_wrist.json"
        if handeye_path.exists():
            try:
                data = json.loads(handeye_path.read_text())
                # Accept both old AX=XB ("transform_4x4") and new RGB-D
                # Procrustes ("T_cam_to_base_4x4") output formats.
                T_key = (
                    "T_cam_to_base_4x4"
                    if "T_cam_to_base_4x4" in data
                    else "transform_4x4"
                )
                T = np.array(data[T_key], dtype=np.float64)
                if T.shape != (4, 4):
                    raise ValueError(f"{T_key} must be 4x4, got {T.shape}")
                # Stash the per-camera depth correction so it can be applied
                # in capture() once _realsense_cal is created. We can't set
                # _realsense_cal.depth_scale/offset here because that cal
                # object doesn't exist yet at pipeline init time.
                self._wrist_depth_scale = float(data.get("depth_scale_correction", 1.0))
                self._wrist_depth_offset = float(
                    data.get("depth_offset_correction", 0.0)
                )
                logger.info(
                    "wrist tool_offset: loaded from handeye_wrist.json "
                    "(rmse=%.1fmm, depth_scale=%.4f, depth_offset=%.1fmm, "
                    "pairs/poses=%d)",
                    float(data.get("rmse_mm", data.get("residual_mm", 0.0))),
                    self._wrist_depth_scale,
                    self._wrist_depth_offset * 1000.0,
                    int(data.get("num_pairs", data.get("num_poses", 0))),
                )
                return T
            except Exception as exc:
                logger.warning(
                    "wrist tool_offset: failed to load handeye_wrist.json "
                    "(%s); using default tool offset.",
                    exc,
                )

        offset_xyz = [0.05, 0.00, -0.10]
        T = np.eye(4)
        T[0, 3] = float(offset_xyz[0])
        T[1, 3] = float(offset_xyz[1])
        T[2, 3] = float(offset_xyz[2])
        return T

    def _init_kinects(self):
        """
        Initialize Azure Kinect cameras.

        When ``config.kinect_use_hw_sync`` is True (3.5mm sync cable
        connected between two Kinects), the **subordinate** is opened
        FIRST in SUBORDINATE mode (it then waits silently for the master
        trigger), THEN the master is opened in MASTER mode and starts
        firing pulses. This is mandatory per the K4A SDK and also keeps
        USB bandwidth manageable when both Kinects share one host
        controller: the subordinate doesn't free-run.

        Without the sync cable (standalone), both Kinects run free and
        bandwidth must come from separate USB host controllers.
        """
        if not HAS_PYK4A:
            logger.warning("pyk4a not available, skipping Kinect init")
            return

        num_kinects = pyk4a.connected_device_count()
        logger.info("Found %d Azure Kinect device(s)", num_kinects)

        master_serial = self.config.kinect_master_serial
        use_hw_sync = bool(getattr(self.config, "kinect_use_hw_sync", False))
        serial_map = {}  # serial -> device_id
        for dev_id in range(num_kinects):
            try:
                dev = pyk4a.PyK4A(device_id=dev_id)
                dev.open()
                serial_map[dev.serial] = dev_id
                dev.close()
            except Exception:
                pass
        logger.info("Camera serials: %s", serial_map)

        roster = self.profile.cameras() if self.profile is not None else []
        serials = {d: None for d in range(num_kinects)}
        serials.update({d: sn for sn, d in serial_map.items()})
        plan = assign_kinect_roles(serials, roster, master_serial, use_hw_sync)
        logger.info("Kinect open plan (device, role, sync): %s", plan)

        for dev_id, name, sync_mode in plan:
            serial = serials.get(dev_id)
            logger.info(
                "Opening Kinect %d as %s (SN=%s, sync=%s)...",
                dev_id,
                name,
                serial or "?",
                sync_mode,
            )
            try:
                # Subordinate delay depends on the master's depth laser
                # pulse width, which differs by depth mode. NFOV pulses
                # are ~125us so 160us margin is fine; WFOV pulses are
                # nearly an order of magnitude longer, so use 1600us to
                # avoid cross-illumination causing dropped frames and
                # blocking open() forever in warmup.
                sub_delay = 0
                if sync_mode == "SUBORDINATE":
                    if self.config.kinect_depth_mode.startswith("WFOV"):
                        sub_delay = 1600
                    else:
                        sub_delay = 160
                cam = AzureKinectCamera(
                    device_id=dev_id,
                    color_resolution=self.config.kinect_resolution,
                    depth_mode=self.config.kinect_depth_mode,
                    sync_mode=sync_mode,
                    subordinate_delay_usec=sub_delay,
                    camera_fps=self.config.kinect_fps,
                )
                for attempt in range(2):
                    try:
                        cam.open()
                        break
                    except Exception:
                        if attempt == 0:
                            logger.warning(
                                "Kinect %d failed, retrying in 2s...", dev_id
                            )
                            time.sleep(2)
                        else:
                            raise
                # From here on `cam` is open. Any exception before the
                # outer except catches it would leak the handle, so
                # close it explicitly on failure.
                try:
                    cal = CameraCalibration(
                        name=name,
                        width=cam.config.width,
                        height=cam.config.height,
                        fx=cam.config.fx,
                        fy=cam.config.fy,
                        cx=cam.config.cx,
                        cy=cam.config.cy,
                        # Carry any config-level extrinsic across instead of
                        # silently dropping it: the fallback for a camera whose
                        # config carries an extrinsic but has no hand-eye file
                        # (_load_handeye_calibrations() overwrites cal.extrinsic
                        # when one exists).
                        extrinsic=getattr(cam.config, "extrinsic", None),
                    )
                    if name == "sideview":
                        self._kinect = cam
                        self._kinect_cal = cal
                    else:
                        self._kinect2 = cam
                        self._kinect2_cal = cal
                except Exception:
                    try:
                        cam.close()
                    except Exception:
                        pass
                    raise
                logger.info(
                    "Kinect %d (%s) ready (%dx%d, fx=%.1f)",
                    dev_id,
                    name,
                    cal.width,
                    cal.height,
                    cal.fx,
                )
                # Stagger camera starts to avoid USB contention
                time.sleep(3)
            except Exception as e:
                logger.warning("Kinect %d failed: %s", dev_id, e)

    def _load_handeye_calibrations(self):
        """
        Apply saved hand-eye calibrations to the live Kinect cals.

        The cam-to-base 4x4 transforms are read from
        ``<calibrations_dir>/handeye_{birdview,sideview,wrist}.json``.
        Without this loader the Kinect cals stay at identity and detect()
        backprojects every pixel ~1 m below the table (the OpenGL fallback
        path triggers because ``intrinsic_matrix`` is gated on
        ``not np.allclose(extrinsic, np.eye(4))``).

        Wrist is intentionally skipped here: its extrinsic is recomputed
        every frame from the live TCP pose in ``capture()`` so a fixed
        cam-to-base load would be wrong the instant the arm moves.
        """
        cal_dir = calibrations_dir()
        targets = [
            ("birdview", "handeye_birdview.json", self._kinect2_cal),
            ("sideview", "handeye_sideview.json", self._kinect_cal),
        ]
        for name, fname, cal in targets:
            if cal is None:
                continue
            path = cal_dir / fname
            if not path.exists():
                logger.info("hand-eye: no %s file at %s", name, path)
                continue
            try:
                data = json.loads(path.read_text())
                # Accept either format:
                #   {transform_4x4: [[...]], ...}              (legacy
                #     AX=XB hand-eye output)
                #   {T_cam_to_base_4x4: [[...]],
                #    depth_scale_correction: 0.85, ...}        (new
                #     RGB-D Procrustes output from spark_calibrate.py)
                T_key = (
                    "T_cam_to_base_4x4"
                    if "T_cam_to_base_4x4" in data
                    else "transform_4x4"
                )
                T = np.array(data[T_key], dtype=np.float64)
                if T.shape != (4, 4):
                    raise ValueError(f"{T_key} must be 4x4, got {T.shape}")
                cal.extrinsic = T
                depth_scale = float(data.get("depth_scale_correction", 1.0))
                depth_offset = float(data.get("depth_offset_correction", 0.0))
                cal.depth_scale = depth_scale
                cal.depth_offset = depth_offset
                logger.info(
                    "hand-eye: applied %s extrinsic from %s "
                    "(residual=%.1fmm, depth_scale=%.4f, depth_offset=%.1fmm, "
                    "poses/corners=%d)",
                    name,
                    fname,
                    float(data.get("rmse_mm", data.get("residual_mm", 0.0))),
                    depth_scale,
                    depth_offset * 1000.0,
                    int(data.get("num_pairs", data.get("num_poses", 0))),
                )
            except Exception as exc:
                logger.warning(
                    "hand-eye: failed to load %s from %s: %s", name, fname, exc
                )

    def _build_unified_camera_registry(self):
        # Wrap the already-open single-arm cameras in one role-keyed registry.
        # The dual-Kinect open in _init_kinects() carries master/subordinate
        # sync, depth-mode, retry, and stagger semantics that depend on
        # runtime device enumeration, so do NOT re-open through the
        # config-driven builder here: wrap the live devices and their
        # calibrated CameraCalibration objects so the registry entries and the
        # _kinect / _kinect2 / _realsense slots reference the same objects.
        # The bimanual rig opens its whole roster from config instead.
        if (getattr(self.config, "robot_family", "") or "").lower() == "bimanual_franka":
            self._init_bimanual_cameras()
            return

        entries = []
        # sideview = master Kinect (_kinect); birdview = subordinate (_kinect2);
        # wrist = RealSense (_realsense). This is the slot<->role mapping from
        # _init_kinects (camera_registry.assign_kinect_roles).
        slot_roles = [
            ("sideview", self._kinect, self._kinect_cal, "kinect"),
            ("birdview", self._kinect2, self._kinect2_cal, "kinect"),
            ("wrist", self._realsense, self._realsense_cal, "realsense"),
        ]
        for role, device, cal, kind in slot_roles:
            if device is None or cal is None:
                continue
            entries.append(
                CameraEntry(
                    role=role,
                    kind=kind,
                    device=device,
                    calibration=cal,
                    serial=getattr(device, "serial", "") or "",
                )
            )

        # `sideview` (master Kinect) is the workspace anchor for single-arm
        # rigs, matching camera_registry._resolve_primary_role.
        primary = (
            "sideview"
            if any(e.role == "sideview" for e in entries)
            else (entries[0].role if entries else "")
        )
        self._camera_registry = registry_from_open_devices(
            entries, primary_role=primary
        )

        # Re-point the legacy slots FROM the registry (identity reassignments;
        # the entries wrap the same device + cal objects).
        sv = self._camera_registry.get("sideview")
        bv = self._camera_registry.get("birdview")
        wr = self._camera_registry.get("wrist")
        if sv is not None:
            self._kinect = sv.device
            self._kinect_cal = sv.calibration
        if bv is not None:
            self._kinect2 = bv.device
            self._kinect2_cal = bv.calibration
        if wr is not None:
            self._realsense = wr.device
            self._realsense_cal = wr.calibration
        # Every other roster entry (e.g. a ZED or a webcam on this rig) opens
        # through the camera driver registry and joins the same registry.
        if self.profile is not None:
            extra = [
                dict(e, type=camera_type(e)) for e in camera_spec(self.profile).values()
                if (e["role"], camera_type(e)) not in PIPELINE_OPENED and e.get("enabled", True) is not False
            ]
            build_camera_registry(extra, registry=self._camera_registry)
        logger.info(
            "Unified camera registry built: roles=%s primary=%s",
            self._camera_registry.roles,
            self._camera_registry.primary_role,
        )

    def _init_robot(self):
        """
        Connect the robot through ``make_robot_driver`` with the family
        config's gripper/robot/collision blocks, then build the executor.

        ur10e: SafeRobot (CBF filter on, force off) around the driver.
        franka: the bare driver; the CBF clamps direct joint moves
        unpredictably, so the libfranka collision thresholds and the
        executor's workspace box protect the arm. bimanual_franka:
        pipeline_init_bimanual. A connection failure leaves the pipeline
        perception-only; a config error (unknown family, gripper type or
        backend) raises.
        """
        # Tear down a previous driver before reassigning self._robot so a
        # reconnect never leaves two control sockets open.
        if getattr(self, "_robot", None) is not None:
            try:
                self._robot.disconnect()
            except Exception as exc:
                logger.warning(
                    "previous robot driver disconnect raised: %s "
                    "(continuing with reconnect)",
                    exc,
                )
            self._robot = None

        family = (getattr(self.config, "robot_family", "ur10e") or "ur10e").lower()
        try:
            if family == "bimanual_franka":
                self._init_bimanual_franka()
                return
            blocks = self.profile.driver_blocks() if self.profile is not None else {}
            self._robot = make_robot_driver(family, self.config.robot_ip, frequency=None, **blocks)
            self._robot.connect()
            if family == "ur10e":
                # The SafeRobot box comes from the same control: block the
                # ScoreExecutor reads, so the two boxes stay one source.
                ctrl = (self.profile.control() if self.profile is not None else {}) or {}
                ws_min = ctrl.get("workspace_min", [-1.1, -0.5, -0.27])
                ws_max = ctrl.get("workspace_max", [-0.5, 0.7, 0.50])
                safety_cfg = SafetyConfig(
                    vel_lin_max=self.config.velocity + 0.05,
                    ws_min=np.array([float(v) for v in ws_min]),
                    ws_max=np.array([float(v) for v in ws_max]),
                )
                self._robot = SafeRobot(
                    self._robot,
                    enable_obstacles=False,  # obstacles set from detection map
                    enable_force=False,  # noisy F/T causes jiggling
                    config=safety_cfg,
                )
            self._executor = ScoreExecutor(
                robot=self._robot,
                velocity=self.config.velocity,
                safe_height=self.config.safe_height,
                pipeline=self,
                strict_placement_verify=self.config.strict_placement_verify,
                grasp_calibration=self._grasp_calibration,
            )
            self._executor._profile = self.profile
            self._executor._apply_family_workspace_bounds()
            logger.info(
                "%s connected at %s (gripper %s, SafeRobot %s)",
                family,
                self.config.robot_ip,
                self._robot.GRIPPER_TYPE,
                "on" if family == "ur10e" else "off",
            )
        except ValueError:
            self._robot = None
            self._executor = None
            raise
        except Exception as e:
            logger.warning(
                "Robot connection failed (%s): %s (running perception-only)",
                family,
                e,
            )
            self._robot = None
            self._executor = None



BIMANUAL_CALIBRATION = "calibration_bimanual.json"


def load_bimanual_calibration(path=None):
    """(T_zed_to_right_base, T_right_to_left) from the calibration file, or None."""
    path = path or calibrations_dir() / BIMANUAL_CALIBRATION
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    arms = data["arms"]
    T_r = np.asarray(arms["right"]["T_zed_to_base"], dtype=float).reshape(4, 4)
    T_l = np.asarray(arms["left"]["T_zed_to_base"], dtype=float).reshape(4, 4)
    return T_r, T_l @ np.linalg.inv(T_r)


def _arm_safety(ws: dict, max_linear_velocity: float) -> SafetyConfig:
    return SafetyConfig(
        vel_lin_max=max_linear_velocity + 0.05,
        ws_min=np.array([ws.get("x_min", -0.85), ws.get("y_min", -0.85), ws.get("z_min", 0.0)], dtype=float),
        ws_max=np.array([ws.get("x_max", 0.85), ws.get("y_max", 0.85), ws.get("z_max", 1.20)], dtype=float),
        reach_max=float(ws.get("reach_max", 0.855)),
        eps_singularity=0.0,  # the UR elbow check does not apply to a Franka
        eta_kinematic=float(ws.get("eta_kinematic", 5.0)),
    )


class BimanualInitMixin:

    def _init_bimanual_cameras(self):
        """Open the config roster; map external/wrist roles onto the slots."""
        cfg = family_block(self.profile, "bimanual_franka")
        self.camera_registry = build_camera_registry(cfg.get("cameras") or [], primary_role="external")
        self._T_right_to_left = None
        ext = self.camera_registry.get("external")
        try:
            cal = load_bimanual_calibration()
        except Exception as exc:  # noqa: BLE001 - a bad file must not stop bring-up
            logger.error("bimanual calibration unreadable: %s", exc)
            cal = None
        if cal is None:
            logger.warning(
                "no %s in %s: detections stay in the camera frame and the bimanual "
                "executor refuses to run",
                BIMANUAL_CALIBRATION, calibrations_dir(),
            )
        else:
            T_zed_to_right, self._T_right_to_left = cal
            if ext is not None:
                ext.calibration.extrinsic = T_zed_to_right
            logger.info("bimanual calibration loaded: external camera -> right arm base")
        # Slots read by the single-arm capture path: the external camera is
        # the scene camera; the right wrist stands in for "wrist".
        wl, wr = self.camera_registry.get("wrist_left"), self.camera_registry.get("wrist_right")
        wrist = wr or wl
        self._kinect = ext.device if ext else None
        self._kinect_cal = ext.calibration if ext else None
        self._kinect2 = None
        self._kinect2_cal = None
        self._realsense = wrist.device if wrist else None
        self._realsense_cal = wrist.calibration if wrist else None
        self._wrist_for_arm = dict(self.camera_registry.wrist_for_arm)

    def _init_bimanual_franka(self):
        """Driver + BimanualSafeRobot + BimanualScoreExecutor from the profile."""
        cfg = family_block(self.profile, "bimanual_franka")
        robot_cfg = cfg.get("robot") or {}
        blocks = self.profile.driver_blocks() if self.profile is not None else {
            "gripper": cfg.get("grippers") or {}, "robot": robot_cfg, "collision": cfg.get("collision_behavior") or {},
        }
        driver = make_robot_driver("bimanual_franka", self.config.robot_ip, **blocks)
        driver.connect()
        self._raw_driver = driver

        vmax = float(robot_cfg.get("max_linear_velocity", 0.15))
        inter = cfg.get("inter_arm") or {}
        T_r2l = getattr(self, "_T_right_to_left", None)
        if T_r2l is not None:
            # Calibrated frames: the right arm's base is the shared frame.
            bases = BaseFrames(T_left_world=np.linalg.inv(T_r2l), T_right_world=np.eye(4))
        else:
            logger.warning("no bimanual calibration: the inter-arm gate uses robot.base_left/right "
                           "from the config, and the executor refuses scores until calibrated")
            try:
                bases = BaseFrames.from_yaml_block(robot_cfg)
            except ValueError as exc:
                logger.warning("bimanual base offsets missing (%s); identity bases", exc)
                bases = None
        self._safe = BimanualSafeRobot(
            driver,
            _arm_safety(cfg.get("workspace_left") or {}, vmax),
            _arm_safety(cfg.get("workspace_right") or {}, vmax),
            InterArmConfig(
                dist_hard_m=float(inter.get("dist_hard_m", 0.12)),
                dist_soft_m=float(inter.get("dist_soft_m", 0.20)),
                eta_kinematic=float(inter.get("eta_kinematic", 8.0)),
            ),
            bases=bases,
        )
        self._robot = self._safe
        self._executor = BimanualScoreExecutor(
            self._safe,
            velocity=self.config.velocity,
            pipeline=self,
            strict_placement_verify=self.config.strict_placement_verify,
            T_right_to_left=T_r2l,
            motion_cfg=cfg.get("bimanual_motion") or {},
        )
        self._executor._profile = self.profile
        # The per-arm executors applied bounds in __init__ before _profile
        # existed; re-apply so each reads its workspace_<arm> box, the table
        # floor and the grasp orientation from the profile.
        for ex in self._executor._arm_executors.values():
            ex._profile = self.profile
            ex._apply_family_workspace_bounds()
        logger.info(
            "Bimanual Franka connected (backend=%s, grippers=%s, left=%s, right=%s)",
            driver.backend, driver.GRIPPER_TYPE, driver.left_ip, driver.right_ip,
        )
