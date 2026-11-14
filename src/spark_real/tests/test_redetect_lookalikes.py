"""A numbered look-alike is re-bound to a piece on the table, never to one already in the tray."""
from types import SimpleNamespace

from spark_real.control import recovery as redetect


def _run(monkeypatch, found):
    monkeypatch.setattr(redetect, "sticky_settings", lambda e: {"enabled": False})
    calls = {}

    def detect(captures, prompts, multi_instance=False):
        calls["multi"] = multi_instance
        return [{"label": "silverware", "position_3d": [*xy, -0.27]} for xy in found]

    ex = SimpleNamespace(
        detection_map={
            "silverware 2": {"position_3d": [0.50, 0.0, -0.27]},
            "tray": {"position_3d": [0.0, 0.0, -0.27], "obb_minor_m": 0.25, "slots": [{"world_xyz": [0.0, 0.05, -0.27]}]},
        },
        _destination_labels={"tray"},
        _holding=False,
        _pipeline=SimpleNamespace(capture=lambda: {}, detect=detect, merge_detections=lambda d: d),
    )
    redetect.redetect_single(ex, "silverware 2")
    return ex.detection_map["silverware 2"]["position_3d"][:2], calls["multi"]


def test_piece_in_tray_is_not_a_pick_candidate(monkeypatch):
    xy, multi = _run(monkeypatch, [(0.02, 0.0), (0.30, 0.10)])   # the tray piece scores first
    assert multi is True and xy == [0.30, 0.10]


def test_only_tray_pieces_seen_keeps_the_binding(monkeypatch):
    xy, _ = _run(monkeypatch, [(0.02, 0.0)])
    assert xy == [0.50, 0.0]


def test_moved_tray_gets_a_fresh_slot_direction():
    """A tray that moved 10 cm is measured again: new slot axes, no slot index; a 1 cm shift keeps the old reading."""
    import numpy as np

    from spark_real.perception.masks import resolve_slot_direction

    def enrich(dets, captures):
        dets[0].slots = [{"mode": "normals", "slot_idx": i, "axis_rad": 1.2, "axis_ar": 4.0, "n_pixels": 500,
                          "world_xyz": np.array([0.1 * i, 0.0, -0.27])} for i in range(3)]
        dets[0].world_major_axis_rad = 1.2

    ex = SimpleNamespace(_pipeline=SimpleNamespace(_enrich_with_slots=enrich))
    slot = {"mode": "normals", "slot_idx": 0, "axis_ar": 4.0, "n_pixels": 500, "world_xyz": [0, 0, 0]}
    det = SimpleNamespace(mask=object(), orientation_angle=1.2, aspect_ratio=1.2)
    same = {"slots": [{**slot, "axis_rad": 1.15}]}          # 3 degrees off, centre still: not adopted
    assert redetect._moved_container_axes(ex, "tray", same, det, 0.01, {}) == {}
    old = {"slots": [{**slot, "axis_rad": 0.0}]}             # turned in place: adopted without a move
    assert redetect._moved_container_axes(ex, "tray", old, det, 0.01, {})["world_major_axis_rad"] == 1.2
    new = redetect._moved_container_axes(ex, "tray", old, det, 0.10, {})
    assert {s["slot_idx"] for s in new["slots"]} == {-1} and new["world_major_axis_rad"] == 1.2
    axis, source = resolve_slot_direction({**old, **new}, "tray", 1)
    assert abs(axis - 1.2) < 1e-6 and source.startswith("slot_floors")
