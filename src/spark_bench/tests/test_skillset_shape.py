"""SPARK_SKILLSET=shape / ask / askw: the rim-pinch choice on synthetic point sets, and the arms' requests."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spark_bench.libero_pro.primitives import keypoint
from spark_core import skillset

PROMPTS = Path(skillset.__file__).parent / 'prompts'
# sha256[:16] of each arm's franka prompt.
FROZEN = {'generic': 'cdbbd78fff267853', 'base': 'fb9fc072bc3541e2', 'plain': 'fb9fc072bc3541e2',
          'shape': 'fb9fc072bc3541e2', 'ask': 'd4d47b033b2c5439'}
TABLE = 0.90


def _dish(radius, height, n=4000):
    """Points on a paraboloid dish: floor at TABLE, rim at TABLE + height."""
    rng = np.random.default_rng(0)
    rho, phi = radius * np.sqrt(rng.random(n)), 2 * np.pi * rng.random(n)
    return np.c_[rho * np.cos(phi), rho * np.sin(phi), TABLE + height * (rho / radius) ** 2]


def _block(side=0.04, n=2000):
    rng = np.random.default_rng(0)
    return np.c_[side * (rng.random(n) - 0.5), side * (rng.random(n) - 0.5), np.full(n, TABLE + side)]


PLATE, BOWL, BLOCK = _dish(0.0625, 0.01), _dish(0.05, 0.05), _block()


def test_rim_shape_on_synthetic_points():
    plate, bowl, block = (keypoint._rim_shape(p) for p in (PLATE, BOWL, BLOCK))
    assert plate['wide'] and plate['flat']               # wide and lower than the fingertips: regulated close
    assert bowl['wide'] and not bowl['flat']             # wide and hollow: compliant close
    assert not block['wide']                             # fits between the jaws: top-down pick
    # Offsets come from the points: a quarter aperture inside the +Y edge (98th percentile, 0.9 R on a disc);
    # under the rim or just above the base.
    assert plate['y'] == pytest.approx(0.9 * 0.0625 - 0.02, abs=0.004)
    assert plate['z'] == pytest.approx(TABLE + keypoint.RIM_CLEAR, abs=0.002)
    assert bowl['y'] == pytest.approx(0.9 * 0.05 - 0.02, abs=0.004)
    assert bowl['z'] == pytest.approx(TABLE + 0.05 - keypoint.FINGER_DEPTH, abs=0.004)
    assert bowl['dy'] == pytest.approx(0.025, abs=0.004) and bowl['hang'] == pytest.approx(0.015, abs=0.004)
    # Place target: the plate is a flat support, the bowl is a container.
    assert keypoint._flat_top(PLATE) and not keypoint._flat_top(BOWL) and not keypoint._flat_top(None)


@pytest.mark.parametrize('arm,label,pts,style,expect', [
    ('shape', 'bowl', BLOCK, None, 'top'),       # the label is never read
    ('shape', 'block', BOWL, None, 'rim'),
    ('shape', 'block', PLATE, None, 'rim'),
    ('ask', 'block', BOWL, None, 'top'),         # the plan did not ask
    ('ask', 'block', BOWL, 'rim', 'rim'),
    ('ask', 'bowl', BOWL, 'center', 'top'),
    ('askw', 'block', BOWL, 'rim', 'rim'),       # the executor runs askw as ask
    ('askw', 'bowl', BOWL, None, 'top'),
    ('plain', 'bowl', BOWL, 'rim', 'top'),
    ('base', 'block', BOWL, 'rim', 'top'),
    ('base', 'bowl', BLOCK, None, 'bowl'),       # label rule
    ('', 'plate', BLOCK, None, 'plate'),
])
def test_pick_branch_per_arm(monkeypatch, arm, label, pts, style, expect):
    monkeypatch.setenv('SPARK_SKILLSET', arm)
    calls = []
    for name, tag in (('_do_rim_pinch', 'rim'), ('_do_bowl_rim_pinch', 'bowl'), ('_do_plate_rim_pinch', 'plate'),
                      ('_do_top_down_pick', 'top')):
        monkeypatch.setattr(keypoint, name, lambda *a, _t=tag, **k: calls.append(_t))
    det = SimpleNamespace(mask=np.ones((2, 2)), position_3d=np.array([0.0, 0.0, TABLE]))
    ex = SimpleNamespace(det_map={label: det}, holding=False, last_pick_label=None, last_pick_shape=None,
                         sam3=SimpleNamespace(_backproject_mask=lambda *a: pts), depth=np.ones((2, 2)),
                         cam_pos=None, cam_mat=None, cam_fovy=None, cam_w=2, cam_h=2,
                         env=SimpleNamespace(), trial_meta={}, instruction='')
    keypoint.handle(ex, {'keypoint_label': label, **({'grasp_style': style} if style else {})})
    assert calls == [expect]
    assert (ex.last_pick_shape is not None) == (expect == 'rim')
    if expect == 'rim':    # a pre-close re-approach carries no grasp_style: the first choice stands
        keypoint.handle(ex, {'keypoint_label': label})
        assert calls == ['rim', 'rim']


@pytest.mark.parametrize('arm,expect', [('shape', 'rim'), ('base', 'top')])
def test_wrist_bound_pick_uses_its_own_points(monkeypatch, arm, expect):
    """A pick bound from the wrist camera has no agentview mask; the geometry arms read det.pts_world."""
    monkeypatch.setenv('SPARK_SKILLSET', arm)
    calls = []
    for name, tag in (('_do_rim_pinch', 'rim'), ('_do_top_down_pick', 'top')):
        monkeypatch.setattr(keypoint, name, lambda *a, _t=tag, **k: calls.append(_t))
    det = SimpleNamespace(mask=None, pts_world=BOWL, position_3d=np.array([0.0, 0.0, TABLE]))
    ex = SimpleNamespace(det_map={'block': det}, holding=False, last_pick_label=None, last_pick_shape=None,
                         sam3=SimpleNamespace(), depth=np.ones((2, 2)), cam_pos=None, cam_mat=None, cam_fovy=None,
                         cam_w=2, cam_h=2, env=SimpleNamespace(), trial_meta={}, instruction='')
    keypoint.handle(ex, {'keypoint_label': 'block'})
    assert calls == [expect]
    if arm == 'shape':
        assert ex.trial_meta['pick_shape'][0]['view'] == 'wrist'


def test_prompts(monkeypatch):
    text = (PROMPTS / 'system_franka.txt').read_text()

    def prompt(arm):
        monkeypatch.setenv('SPARK_SKILLSET', arm)
        return skillset.restrict_prompt(text)

    for arm, digest in FROZEN.items():     # the arms that already ran keep their prompt byte for byte
        assert hashlib.sha256(prompt(arm).encode()).hexdigest()[:16] == digest, arm
    base, ask = prompt('base'), prompt('ask')
    assert prompt('shape') == base
    assert 'grasp_style' not in base and not any(w in skillset._ASK_STYLE.lower() for w in ('bowl', 'plate', 'dish'))
    assert ask.replace(', grasp_style ("rim" | "center", optional)', '').replace(skillset._ASK_STYLE, '') == base
    assert ask.count('grasp_style') == 2 and ask.endswith(skillset._ASK_STYLE)
    askw = prompt('askw')
    assert askw.replace(', grasp_style ("rim" | "center")', '').replace(skillset._ASKW_STYLE, '') == base
    assert askw.endswith(skillset._ASKW_STYLE) and 'optional' not in skillset._ASKW_STYLE
    assert not any(w in skillset._ASKW_STYLE.lower() for w in ('bowl', 'plate', 'dish'))


@pytest.mark.parametrize('arm', ['', 'base', 'plain', 'shape', 'ask', 'askw'])
def test_width_reaches_the_planner_in_askw_only(monkeypatch, arm):
    """The request of every other arm has no width."""
    from spark_bench.libero_pro.planning import llm
    from spark_core import sim_detect
    from spark_core.planner import SPARKPlanner
    monkeypatch.setenv('SPARK_SKILLSET', arm)
    seen = {}

    class Planner:
        def generate_score(self, hint, **kw):
            seen['details'] = kw['detection_details']
            return {'tree': {'type': 'sequence', 'children': [{'type': 'grasp', 'params': {}}]}}

    monkeypatch.setattr(llm, 'SPARKPlanner', Planner)
    width = sim_detect.width_cm(BOWL)
    det = SimpleNamespace(confidence=0.9, position_3d=np.array([0.0, 0.0, TABLE]), width_cm=width)
    llm.plan_llm('pick up the block', ['block'], det_map={'block': det}, no_bddl_hints=True)
    expect = {'label': 'block', 'confidence': 0.9, 'position_3d': [0.0, 0.0, TABLE]}
    assert seen['details'] == [dict(expect, width_cm=width) if arm == 'askw' else expect]
    # The number is the shape arm's measure: above the opening exactly when the shape arm takes the rim.
    for pts in (PLATE, BOWL, BLOCK):
        assert (sim_detect.width_cm(pts) > 100 * keypoint.FRANKA_OPEN_APERTURE) == keypoint._rim_shape(pts)['wide']
    assert sim_detect.width_cm(BOWL[:50]) is None
    assert SPARKPlanner._format_geometry({'width_cm': 9.83}) == ', width=9.8cm'
    assert SPARKPlanner._format_geometry({}) == ''


@pytest.mark.parametrize('arm', ['ask', 'askw'])
def test_detections_carry_their_width_in_askw_only(monkeypatch, arm):
    pytest.importorskip('robosuite')
    from spark_bench.libero_pro.perception import detect
    monkeypatch.setenv('SPARK_SKILLSET', arm)
    monkeypatch.setattr(detect.sim_detect, 'backproject_mask', lambda *a: BOWL)
    masked, wrist_only = SimpleNamespace(mask=np.ones((2, 2))), SimpleNamespace(mask=None)
    detect.tag_widths([masked, wrist_only], np.ones((2, 2)), SimpleNamespace(pos=None, mat=None, fovy=45.0),
                       SimpleNamespace(cam_width=2, cam_height=2))
    assert not hasattr(wrist_only, 'width_cm')
    assert (getattr(masked, 'width_cm', None) == detect.sim_detect.width_cm(BOWL)) == (arm == 'askw')
    assert hasattr(masked, 'width_cm') == (arm == 'askw')
