"""Perception-noise ops (libero_pro/perception/noise.py): off is a pass-through, on has the intended statistics."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from spark_bench.fair.config import FairConfig
from spark_bench.libero_pro.perception import detect, noise

H, W = 48, 64


def _cfg(**kw):
    return dataclasses.replace(FairConfig(cam_width=W, cam_height=H), **kw)


class _Env:
    """Two cameras with fixed poses; enough of env.sim for detect_scene."""
    def __init__(self):
        names = ['agentview', 'robot0_eye_in_hand']
        self.data = SimpleNamespace(cam_xpos=np.array([[1.0, 0.0, 1.5], [0.2, 0.1, 0.9]]),
                                    cam_xmat=np.stack([np.eye(3).ravel()] * 2))
        self.model = SimpleNamespace(cam_fovy=np.array([45.0, 75.0]))
        self.sim = SimpleNamespace(model=SimpleNamespace(_model=self.model, camera_name2id=names.index),
                                   data=SimpleNamespace(_data=self.data))


def _obs(seed=0):
    r = np.random.default_rng(seed)
    return {'agentview_image': r.integers(0, 255, (H, W, 3), np.uint8),
            'agentview_depth': r.uniform(0.5, 1.5, (H, W, 1)),
            'robot0_eye_in_hand_image': r.integers(0, 255, (H, W, 3), np.uint8),
            'robot0_eye_in_hand_depth': r.uniform(0.2, 0.6, (H, W, 1))}


@pytest.fixture(autouse=True)
def _disarm(monkeypatch):
    monkeypatch.setattr(noise, '_cfg', None)   # restored to None after each test


@pytest.fixture
def stub(monkeypatch):
    """detect_scene with SAM3 stubbed out: records every detect_rendered call."""
    calls = []
    monkeypatch.setattr(detect.mujoco, 'mj_forward', lambda m, d: None)
    monkeypatch.setattr(detect, 'get_real_depth_map', lambda sim, d: d * 1.0)
    monkeypatch.setattr(detect.sim_detect, 'detect_rendered',
                        lambda sam3, rgb, depth, prompts, pos, mat, fovy, w, h, **kw:
                        calls.append(dict(rgb=rgb.copy(), depth=depth.copy(), prompts=list(prompts),
                                          pos=pos.copy(), mat=mat.copy(), fovy=fovy, **kw)) or [])
    return calls


def _run(calls, cfg, trial=None):
    env = _Env()
    if trial is not None:
        noise.start_trial(cfg, trial)
    detect.detect_scene(env, None, ['red block', 'bowl'], _obs(), cfg)
    out = list(calls)
    calls.clear()
    return out, env


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.keys() == y.keys()
        for k in x:
            assert np.array_equal(x[k], y[k]) if isinstance(x[k], np.ndarray) else x[k] == y[k], k


def test_off_is_a_pass_through(stub):
    base, _ = _run(stub, _cfg())                 # never armed
    armed, _ = _run(stub, _cfg(), trial=1000)    # armed with every op off
    assert len(base) == 2 and base[1]['fovy'] == 75.0   # agentview + wrist, true poses
    _same(base, armed)
    assert 'mask_fn' not in armed[0]
    d = np.ones((4, 4))
    o = {'agentview_image': 1}
    assert noise.depth(d) is d and noise.view_obs(o, _cfg()) is o and noise.detect_kw() == {}
    p, m = np.zeros(3), np.eye(3)
    assert noise.extrinsics('agentview', p, m) == (p, m)
    assert noise.start_trial(_cfg(), 5) == {}


def test_depth_noise_statistics():
    noise.start_trial(_cfg(depth_noise_k=0.02, depth_dropout=0.1), 1000)
    z = np.full((400, 400), 0.8, np.float32)
    n = noise.depth(z)
    assert n.dtype == z.dtype and np.all(z == 0.8)   # input untouched
    kept = n > 0
    assert abs((~kept).mean() - 0.1) < 0.005
    rel = (n[kept] - 0.8) / 0.8
    assert abs(rel.std() - 0.02) < 0.001 and abs(rel.mean()) < 0.001
    noise.start_trial(_cfg(depth_noise_k=0.02, depth_dropout=0.1), 1000)
    assert np.array_equal(noise.depth(z), n)          # same trial seed, same draw
    noise.start_trial(_cfg(depth_noise_k=0.02, depth_dropout=0.1), 1001)
    assert not np.array_equal(noise.depth(z), n)


def test_seg_noise_erodes_or_dilates_by_r():
    torch = pytest.importorskip('torch')
    yy, xx = np.mgrid[:200, :200]
    disc = (yy - 100) ** 2 + (xx - 100) ** 2 <= 40 ** 2
    for r in (2, 4, 8):
        assert abs(noise.perturb_mask(disc, r, +1).sum() / (np.pi * (40 + r) ** 2) - 1) < 0.03
        assert abs(noise.perturb_mask(disc, r, -1).sum() / (np.pi * (40 - r) ** 2) - 1) < 0.03
    noise.start_trial(_cfg(seg_noise_px=4), 1000)
    f = noise.mask_fn()
    t = torch.from_numpy(disc[None])
    outs = [f(t) for _ in range(400)]
    assert all(o.shape == t.shape for o in outs)
    grew = np.mean([int(o.sum()) > int(t.sum()) for o in outs])
    assert 0.4 < grew < 0.6                             # random sign per mask


def test_calibration_offset_is_fixed_per_trial_and_sized(stub):
    cfg = _cfg(calib_noise_cm=2.0)
    rec = noise.start_trial(cfg, 1000)
    for cam in ('agentview', noise.WRIST_CAM):
        dt, rv = np.array(rec['calib'][cam]['dt_m']), np.array(rec['calib'][cam]['rotvec_rad'])
        assert abs(np.linalg.norm(dt) - 0.02) < 1e-4
        assert abs(np.rad2deg(np.linalg.norm(rv)) - 1.0) < 1e-3   # 0.5 deg per cm
    assert rec['calib']['agentview'] != rec['calib'][noise.WRIST_CAM]
    assert noise.start_trial(cfg, 1000) == rec and noise.start_trial(cfg, 1001) != rec
    calls, env = _run(stub, cfg, trial=1000)
    a = calls[0]
    assert abs(np.linalg.norm(a['pos'] - [1.0, 0.0, 1.5]) - 0.02) < 1e-9
    assert np.allclose(a['mat'] @ a['mat'].T, np.eye(3)) and not np.allclose(a['mat'], np.eye(3))
    assert np.array_equal(env.data.cam_xpos[0], [1.0, 0.0, 1.5])   # the simulator's camera is unchanged
    calls2, _ = _run(stub, cfg, trial=1000)
    assert np.array_equal(calls2[0]['pos'], a['pos'])


def test_partial_views(stub):
    agent, _ = _run(stub, _cfg(views='agentview'), trial=1000)
    assert len(agent) == 1 and agent[0]['fovy'] == 45.0                 # no wrist detection
    assert detect.redetect_wrist(_Env(), None, ['bowl'], _cfg(views='agentview')) is None
    wrist, _ = _run(stub, _cfg(views='wrist'), trial=1000)
    assert len(wrist) == 1 and wrist[0]['fovy'] == 75.0                 # the wrist frame binds
    assert np.array_equal(wrist[0]['rgb'], _obs()['robot0_eye_in_hand_image'][::-1])
    assert detect.redetect_agentview(_Env(), None, ['bowl'], _cfg(views='wrist')) is None


def test_runner_refuses_noise_on_scored_seeds():
    from spark_bench.run_libero_pro import run_suite
    with pytest.raises(ValueError, match='dev'):
        run_suite(_cfg(seg_noise_px=2))
