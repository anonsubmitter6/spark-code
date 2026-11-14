"""Pointed slots: the planner's image point becomes the placement target; off means no change."""

from types import SimpleNamespace

import numpy as np
import pytest

from spark_real import recording as run_timing
from spark_real.control import place as sp
from spark_real.routes import state
from spark_real.skills import place_in_slot as pis

FX, H, W = 600.0, 720, 1280
U, V = np.meshgrid(np.arange(W), np.arange(H))
X, Y = (U - W / 2) / FX, -(V - H / 2) / FX  # world metres on the table, overhead camera 1 m up
CAL = SimpleNamespace(fx=FX, fy=FX, cx=W / 2, cy=H / 2, position=np.array([0.0, 0.0, 1.0]),
                      rotation_matrix=np.diag([1.0, -1.0, -1.0]))


def rect(yaw_deg, half_long, half_short, cx=0.0, cy=0.0):
    t = np.deg2rad(yaw_deg)
    p, q = (X - cx) * np.cos(t) + (Y - cy) * np.sin(t), -(X - cx) * np.sin(t) + (Y - cy) * np.cos(t)
    return (abs(p) < half_long) & (abs(q) < half_short)


def pixel(x, y):
    return W / 2 + x * FX, H / 2 - y * FX


class Ex:
    GRASP_ORIENTATION = [np.pi, 0.0, 0.0]
    MAX_YAW_OFFSET = np.deg2rad(190)

    def __init__(self, tray_yaw, pointed, replies=()):
        """A 34 x 21 cm tray at `tray_yaw`; SAM3 returns a 6 x 18 cm compartment across it under the point.

        `replies`: the planner's answers in order; a callable is asked for each one.
        """
        self.moves, self.asked, self.clicks = [], [], []
        self._holding, self._last_pick_label, self._tool_heads = True, "spoon 1", {}
        self.robot = SimpleNamespace(open_gripper=lambda: None)
        self.detection_map = {"tray": {"position_3d": [0.0, 0.0, 0.0], "obb_minor_m": 0.21, "aspect_ratio": 1.6,
                                       "world_major_axis_rad": 0.0}}
        self.tray = rect(tray_yaw, 0.17, 0.105)
        self.comp_yaw = tray_yaw + 90
        frame = {"rgb": np.zeros((H, W, 3), np.uint8), "depth": np.ones((H, W), np.float32), "calibration": CAL}
        ex, answers = self, iter(replies)

        class Planner:
            def generate(self, contents, **k):
                ex.asked.append((contents, k))
                return _next(answers)

        def _next(it):
            r = next(it)
            return r(ex) if callable(r) else r

        self._pipeline = SimpleNamespace(
            profile=SimpleNamespace(raw={"placement": {"pointed_slots": pointed}, "perception": {"rebind": {"enabled": False}}}),
            _planner=Planner(),
            capture=lambda: {"birdview": frame},
            # the tray for the pointing request; nothing for the post-place check
            detect=lambda caps, prompts, **k: (
                [SimpleNamespace(mask=self.tray, position_3d=[0.0, 0.0, 0.0])] if prompts == ["tray"] else []),
        )

    def sam3(self, pipeline, rgb, depth, px, py, cam):
        self.clicks.append((px, py))
        return rect(self.comp_yaw, 0.09, 0.03, (px - W / 2) / FX, -(py - H / 2) / FX)

    def current_orientation(self, fallback=None):
        return list(self.GRASP_ORIENTATION)

    def _oriented_grasp(self, yaw):
        from scipy.spatial.transform import Rotation as R

        return list((R.from_euler("z", yaw) * R.from_rotvec(self.GRASP_ORIENTATION)).as_rotvec())

    def _move_to(self, pos, orient):
        self.moves.append((np.round(pos, 4).tolist(), np.round(orient, 4).tolist()))


def reply_for(ex, x, y, margin=40):
    """The planner's reply (0..1000 over the crop) that points at world (x, y)."""
    ys, xs = np.where(ex.tray)
    x0, y0 = max(xs.min() - margin, 0), max(ys.min() - margin, 0)
    x1, y1 = min(xs.max() + margin + 1, W), min(ys.max() + margin + 1, H)
    u, v = pixel(x, y)
    return '```json\n{"point": [%.1f, %.1f]}\n```' % ((u - x0) * 1000 / (x1 - x0), (v - y0) * 1000 / (y1 - y0))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(state, "pointed_slots", None)
    run_timing.reset()


def run(ex, monkeypatch):
    monkeypatch.setattr(sp, "_sam3_point", ex.sam3)
    return pis.place_in_slot(ex, {"container_label": "tray", "slot_idx": 1})


@pytest.mark.parametrize("reply, want", [
    ('{"point": [120, 340]}', (120, 340)),
    ('Sure:\n```json\n{"point": [12.5, 7], "reason": "slot 2 is free"}\n```', (12.5, 7)),
    ('[{"point_2d": [500, 250], "label": "free compartment"}]', (500, 250)),
    ('{"x": 3, "y": 4}', (3, 4)),
    ("(410, 220)", (410, 220)),
    ('{"point": null}', None),
    ("the left one", None),
    ('{"point": [1, 2, 3]}', None),
])
def test_parse_point(reply, want):
    assert sp.parse_point(reply) == want


def test_point_accepted_and_recorded(monkeypatch):
    ex = Ex(0, True, [lambda e: reply_for(e, 0.08, 0.02)])
    run_timing._wrap(type(ex._pipeline._planner), "generate", "llm")  # as run_timing.install() does for LLMClient
    res = run(ex, monkeypatch)
    call = run_timing.snapshot()["calls"][0]
    assert (call["fn"], call["by"], call["tag"]) == ("llm", "ask_slot_point", "point_slot")
    assert res.success and len(ex.asked) == 1
    text, image = ex.asked[0][0]
    assert ex.asked[0][1] == {"tag": "point_slot", "json_only": True}
    assert "free compartment" in text and "spoon 1" in text and "tray" not in text  # task-agnostic
    assert image.shape[0] < H and (image[..., 1] == 255).any()  # cropped, container outlined
    assert np.allclose(ex.moves[0][0][:2], [0.08, 0.02], atol=0.004)
    ev = run_timing.snapshot()["events"][0]
    assert ev["kind"] == "pointed_slot" and ev["ok"] and ev["source"] == "sam3_mask"


def test_point_outside_is_reasked_then_falls_back_to_the_index(monkeypatch):
    off = Ex(0, False)
    run(off, monkeypatch)
    ex = Ex(0, True, ['{"point": [1, 1]}'] * 2)
    assert run(ex, monkeypatch).success
    assert len(ex.asked) == 2 and "outside the green outline" in ex.asked[1][0][0]
    assert ex.clicks == [] and ex.moves == off.moves
    assert run_timing.snapshot()["events"][-1]["ok"] is False


def test_reask_can_recover(monkeypatch):
    ex = Ex(0, True, ["no idea", lambda e: reply_for(e, -0.08, 0.0)])
    run(ex, monkeypatch)
    assert len(ex.asked) == 2 and np.allclose(ex.moves[0][0][:2], [-0.08, 0.0], atol=0.004)


@pytest.mark.parametrize("tray_yaw", [0, 35, 90])
def test_yaw_follows_a_rotated_tray(monkeypatch, tray_yaw):
    """The plan-time shape says 0 deg; the compartment under the point says where the slot runs now."""
    from spark_real.control.grasp_strategy import base_orientation, measured_yaw_offset

    ex = Ex(tray_yaw, True, [lambda e: reply_for(e, 0.0, 0.0)])
    run(ex, monkeypatch)
    yaw = np.rad2deg(measured_yaw_offset(ex.moves[0][1], base_orientation(ex)))
    assert abs((yaw - (tray_yaw + 90) + 90) % 180 - 90) < 3


def test_refused_sam3_mask_uses_the_pixel_and_keeps_the_direction(monkeypatch):
    off = Ex(0, False)
    run(off, monkeypatch)
    ex = Ex(0, True, [lambda e: reply_for(e, 0.08, 0.02)])
    monkeypatch.setattr(sp, "_sam3_point", lambda *a: ex.tray)  # SAM3 returned the whole tray
    pis.place_in_slot(ex, {"container_label": "tray", "slot_idx": 1})
    assert np.allclose(ex.moves[0][0][:2], [0.08, 0.02], atol=0.004)
    assert ex.moves[0][1] == off.moves[0][1]
    assert run_timing.snapshot()["events"][0]["source"] == "pixel"


def test_toggle_off_commands_the_same_moves_and_asks_nothing(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("pointing ran with the toggle off")

    monkeypatch.setattr(sp, "pointed_slot", boom)
    moves = []
    for raw in ({}, {"placement": {"pointed_slots": False}}):
        ex = Ex(0, False)
        ex._pipeline.profile.raw = {**raw, "perception": {"rebind": {"enabled": False}}}
        pis.place_in_slot(ex, {"container_label": "tray", "slot_idx": 1})
        assert ex.asked == []
        moves.append(ex.moves)
    assert moves[0] == moves[1] and len(moves[0]) == 3
    # the index path: slot 1 is one 3 cm step across the slot direction from the centroid
    assert np.allclose(moves[0][0][0][:2], [0.0, 0.03], atol=1e-6)


def test_ui_toggle_overrides_the_profile(monkeypatch):
    pipe = SimpleNamespace(profile=SimpleNamespace(raw={"placement": {"pointed_slots": False}}))
    assert sp.enabled(pipe) is False and sp.enabled(None) is False
    monkeypatch.setattr(state, "pointed_slots", True)
    assert sp.enabled(pipe) is True
    monkeypatch.setattr(state, "pointed_slots", False)
    pipe.profile.raw["placement"]["pointed_slots"] = True
    assert sp.enabled(pipe) is False


def test_frontend_toggle_starts_off_and_switches_without_a_restart(monkeypatch):
    import asyncio
    from pathlib import Path

    from spark_real.config import family_block
    from spark_real.routes import detection as routes

    monkeypatch.setattr(state, "pipeline", None)
    assert asyncio.run(routes.pointed_slots_status()) == {"enabled": False, "source": "config"}
    on = asyncio.run(routes.set_pointed_slots(routes.PointedSlotsBody(enabled=True)))
    assert on == {"enabled": True, "source": "ui"}
    cleared = asyncio.run(routes.set_pointed_slots(routes.PointedSlotsBody()))
    assert cleared == {"enabled": False, "source": "config"}

    html = (Path(routes.__file__).parents[1] / "frontend" / "index.html").read_text()
    box = html[html.index('id="opt-pointed-slots"') - 30:][:120]
    assert "_pointedSlotsToggle(this)" in box and "checked" not in box
    for family in ("ur10e", "franka", "bimanual_franka"):
        assert not family_block(None, family, "placement").get("pointed_slots", False)
