"""Perception noise for the robustness sweep: depth noise, mask erosion/dilation, extrinsic error, partial views.

Every op is off by default and then returns its input object untouched. ``start_trial`` arms the ops for one
trial from the config and the trial seed (trial + trial_offset); each op draws from its own generator seeded
by that index, never from the global numpy RNG, so the simulator's draws are unchanged and paired runs repeat.
The noise reaches only what the robot perceives: the simulator's cameras and renders stay true.
"""
from __future__ import annotations

import zlib
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

WRIST_CAM = 'robot0_eye_in_hand'
_WRIST_KEYS = ('robot0_eye_in_hand_image', 'robot0_eye_in_hand_depth')

_cfg = None
_rng_depth: Optional[np.random.Generator] = None
_rng_mask: Optional[np.random.Generator] = None
_seed = 0
_calib: dict = {}


def start_trial(cfg, idx: int) -> dict:
    """Arm the ops for one trial; returns the settings and the drawn extrinsic offsets (for trial_meta).
    Empty dict when every op is off."""
    global _cfg, _rng_depth, _rng_mask, _seed, _calib
    _cfg, _seed, _calib = cfg, int(idx), {}
    _rng_depth = np.random.default_rng([_seed, 1])
    _rng_mask = np.random.default_rng([_seed, 2])
    if not active(cfg):
        return {}
    out = {k: getattr(cfg, k) for k in ('depth_noise_k', 'depth_dropout', 'seg_noise_px', 'calib_noise_cm',
                                         'calib_rot_deg_per_cm', 'views')}
    if cfg.calib_noise_cm > 0:
        out['calib'] = {}
        for cam in ('agentview', WRIST_CAM):
            dt, rv = _calib_offset(cam)
            out['calib'][cam] = {'dt_m': dt.round(5).tolist(), 'rotvec_rad': rv.round(5).tolist()}
    return out


def active(cfg) -> bool:
    return bool(cfg is not None and (getattr(cfg, 'depth_noise_k', 0) > 0 or getattr(cfg, 'depth_dropout', 0) > 0
                                     or getattr(cfg, 'seg_noise_px', 0) > 0 or getattr(cfg, 'calib_noise_cm', 0) > 0
                                     or getattr(cfg, 'views', 'both') != 'both'))


# depth ---------------------------------------------------------------------------------------------------

def depth(d: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """sigma = k * z Gaussian noise per pixel, then a ``depth_dropout`` fraction of pixels set to 0 (invalid:
    backprojection keeps only depths in (0.01, 10) m). A fresh draw on every frame."""
    if d is None or _cfg is None:
        return d
    k, p = float(_cfg.depth_noise_k), float(_cfg.depth_dropout)
    if k <= 0 and p <= 0:
        return d
    out = d.astype(np.float64, copy=True)
    if k > 0:
        out = np.clip(out + k * out * _rng_depth.standard_normal(out.shape), 0.0, None)
    if p > 0:
        out[_rng_depth.random(out.shape) < p] = 0.0
    return out.astype(d.dtype, copy=False)


# segmentation --------------------------------------------------------------------------------------------

def mask_fn():
    """The per-mask op for sim_detect.detect_rendered, or None when segmentation noise is off."""
    return _perturb_mask if _cfg is not None and int(_cfg.seg_noise_px) > 0 else None


def detect_kw() -> dict:
    """Extra detect_rendered kwargs: {} when off, so the call is the unperturbed one."""
    f = mask_fn()
    return {} if f is None else {'mask_fn': f}


def perturb_mask(mask: np.ndarray, r: int, sign: int) -> np.ndarray:
    """Binary erosion (sign < 0) or dilation (sign > 0) by a disc of radius r pixels."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = (np.asarray(mask) > 0).astype(np.uint8)
    return (cv2.dilate(m, k) if sign > 0 else cv2.erode(m, k)).astype(bool)


def _perturb_mask(mask_tensor):
    """SAM3 mask tensor (1, H, W) or (H, W) -> the same shape, eroded or dilated with a random sign."""
    import torch
    m = mask_tensor.detach().cpu().numpy()
    shape = m.shape
    sign = 1 if _rng_mask.random() < 0.5 else -1
    out = perturb_mask(m.squeeze(), int(_cfg.seg_noise_px), sign)
    return torch.from_numpy(out.reshape(shape))


# calibration ---------------------------------------------------------------------------------------------

def _calib_offset(cam: str) -> tuple[np.ndarray, np.ndarray]:
    """(translation m, rotation vector rad) for one camera: |t| = calib_noise_cm, rotation angle
    calib_rot_deg_per_cm * that, both in uniform random directions. Fixed per trial and camera."""
    if cam not in _calib:
        rng = np.random.default_rng([_seed, 3, zlib.crc32(cam.encode())])
        u = rng.standard_normal(3); u /= np.linalg.norm(u)
        a = rng.standard_normal(3); a /= np.linalg.norm(a)
        t_cm = float(_cfg.calib_noise_cm)
        _calib[cam] = (u * t_cm / 100.0, a * np.deg2rad(float(_cfg.calib_rot_deg_per_cm) * t_cm))
    return _calib[cam]


def extrinsics(cam: str, pos: np.ndarray, mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The camera pose the robot believes: the true pose composed with the trial's fixed offset in the
    camera frame (an extrinsic / hand-eye calibration error)."""
    if _cfg is None or float(_cfg.calib_noise_cm) <= 0:
        return pos, mat
    dt, rv = _calib_offset(cam)
    return pos + mat @ dt, mat @ Rotation.from_rotvec(rv).as_matrix()


# partial views -------------------------------------------------------------------------------------------

def views(cfg) -> str:
    return getattr(cfg, 'views', 'both')


def primary_cam(cfg) -> str:
    """The camera that binds the scene: agentview, or the wrist camera when the agentview is off."""
    return WRIST_CAM if views(cfg) == 'wrist' else 'agentview'


def view_obs(obs: dict, cfg) -> dict:
    """The observation the robot gets. 'agentview': the wrist images removed. 'wrist': the wrist images in the
    agentview slots (the binding frame), the agentview gone. 'both': ``obs`` itself."""
    v = views(cfg)
    if v == 'both' or obs is None:
        return obs
    out = {k: x for k, x in obs.items() if k not in _WRIST_KEYS and not k.startswith('agentview_')}
    if v == 'agentview':
        out.update({k: x for k, x in obs.items() if k.startswith('agentview_')})
    else:
        out['agentview_image'] = obs.get('robot0_eye_in_hand_image')
        out['agentview_depth'] = obs.get('robot0_eye_in_hand_depth')
    return out
