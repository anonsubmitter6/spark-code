"""FairConfig: the LIBERO-PRO runner's CLI (tyro) and the fields the executor and LIBERO-Dyn read."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class FairConfig:
    suite: str = "object"
    """LIBERO suite: object, goal, spatial."""
    perturbation: Literal['position', 'task', 'language', 'object'] = "position"
    """LIBERO-PRO perturbation."""
    num_trials: int = 5
    """Trials per task; trial i uses init state (i + trial_offset)."""
    cam_width: int = 640
    cam_height: int = 480
    no_gemini: bool = False
    """No LLM: scripted plan from the BDDL pick/place hints (smoke and fixed-plan runs)."""
    verbose: bool = False
    output_dir: str = "results/libero_pro"
    """Result JSON goes to <output_dir>/<suite>.json."""
    only_task_id: int = -1
    """If >= 0, run only this task."""
    trial_offset: int = 0
    """Added to the trial index for the env seed and the init state."""
    dev_placements: bool = False
    """Development seeds: reset again after loading the init state, so the placements are re-sampled from the
    seeded RNG instead of the official ones. Needs trial_offset >= 1000 (seeds 0-999 are kept for scored runs)."""
    trace_dir: str = ''
    """Save a per-step trace of each trial here (dyn/trace.py: gripper, aim, true object positions); analysis only."""
    trace_frames: int = 0
    """With trace_dir: also save the rendered images every N env steps."""
    freshness_timeout_s: Optional[float] = 0.5
    """Re-run SAM3 between primitives when the cached detections are older than this. None disables."""

    # perception
    use_gemini_prompts: bool = False
    """The planner VLM proposes the SAM3 prompts from the instruction and the scene image."""
    no_bddl_hints: bool = False
    """Pass only the instruction to the planner, not the BDDL pick/place names."""
    adaptive_prompts: bool = False
    """The VLM proposes adaptive_k phrasings per object; SAM3 keeps one per object."""
    adaptive_k: int = 3
    language_only: bool = False
    """Nothing from the BDDL goal or the curated phrase table reaches perception: prompts come from the
    VLM and the instruction (forces use_gemini_prompts)."""
    wrist_verify: bool = False
    """Re-ground a weakly detected pick target from the home-pose wrist image (one VLM call)."""
    relation_bind: bool = False
    """With twin pick targets, take the instance that satisfies the instruction's spatial relation, from
    3D positions only."""

    # verify-and-adapt arm of the fixed-plan displacement result
    pre_close_retarget_min_cm: float = 4.0
    """Apparent target moves below this at the pre-close capture are treated as occlusion bias."""
    event_captures: bool = False
    """Capture and re-detect at pre-close and post-release."""
    scene_diff: bool = False
    """Diff detections against the plan-time binding before each keypoint primitive."""
    retarget_max_cm: float = 10.0
    """Targets that moved less than this are retargeted in place; larger moves go to recovery."""
    scene_diff_moved_cm: float = 1.5
    scene_diff_self_radius_cm: float = 12.0
    """Changes within this radius of the gripper are attributed to the robot."""
    scene_diff_unknown: bool = False
    """Report a label with no fresh evidence as 'unknown', not 'ok' at delta 0. Reporting only: the same code runs."""
    freshness_steps: int = 0
    """Run the freshness gate on env steps, not wall time, so a seed repeats: this many steps stand for 50 ms. 0: wall time."""
    post_conditions: bool = False
    """Run the plan as a tree: verify_grasp, verify_placed, search_keypoint and tilt_wrist have handlers, and a failed
    verify acts through the plan's retry and fallback, then the recovery loop. Off: they are skipped."""
    rebind_fresh: bool = False
    """At re-bind or local retry, a pick phrase missing from the fresh frame swaps to the object's best fresh phrase."""
    validator_object_snap: bool = False
    """A plan label naming an object snaps to that object's detected phrase; no snap between unrelated labels."""
    layered_recovery: bool = False
    """Repair at the lowest responsible layer (perception re-bind, local retry, replan)."""
    blind_replan: bool = True
    """Without layered recovery, a failed attempt retracts, re-detects, re-plans and re-executes (up to 2 times)."""
    max_local_retries: int = 1

    # perception noise (robustness sweep, libero_pro/perception/noise.py); all off by default, seeded per trial
    depth_noise_k: float = 0.0
    """Gaussian depth noise with sigma = k * z on every depth frame the robot uses (agentview and wrist)."""
    depth_dropout: float = 0.0
    """Fraction of depth pixels set to 0 (invalid) on every depth frame."""
    seg_noise_px: int = 0
    """Erode or dilate (random sign per mask) every SAM3 mask by this many pixels before its geometry."""
    calib_noise_cm: float = 0.0
    """Fixed per-trial extrinsic error of each camera: a translation of this length, random direction."""
    calib_rot_deg_per_cm: float = 0.5
    """Rotation error of the extrinsic offset, in degrees per cm of calib_noise_cm (random axis)."""
    views: Literal['both', 'agentview', 'wrist'] = "both"
    """Cameras the robot has: both, agentview only (wrist off), or wrist only (agentview off; the home-pose
    wrist frame binds the scene and the recovery re-detects from the wrist)."""
