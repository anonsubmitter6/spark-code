import time
from types import SimpleNamespace

from spark_real.control import recovery as redetect


def test_fresh_detection_is_kept(monkeypatch):
    calls = []
    monkeypatch.setattr(redetect, "redetect_single", lambda ex, label: calls.append(label))
    ex = SimpleNamespace(detection_map={})
    redetect.rebind_before_approach(ex, "block")      # no earlier re-detect: runs
    redetect.rebind_before_approach(ex, "block")      # younger than 0.5 s: kept
    assert calls == ["block"]
    ex._rebind_seen["block"] = time.monotonic() - 1.0  # stale: runs again
    redetect.rebind_before_approach(ex, "block")
    assert calls == ["block", "block"]


def test_height_jump_keeps_earlier_z(monkeypatch):
    """A re-detect that lifts the tray 22 cm (depth read off a hand) keeps the earlier z; a 6 cm rise passes."""
    from types import SimpleNamespace

    from spark_real.control import recovery as redetect

    for new_z, want_z in ((-0.05, -0.27), (-0.21, -0.21)):
        ex = SimpleNamespace(detection_map={"tray": {"position_3d": [0.0, 0.0, -0.27]}}, _pipeline=None)
        monkeypatch.setattr(redetect, "rebind_settings", lambda e: {"enabled": True, "log_drift_cm": 2.0, "max_age_s": 0, "max_dz_m": 0.12})
        monkeypatch.setattr(redetect, "redetect_single",
                            lambda e, label, z=new_z: e.detection_map.__setitem__(label, {"position_3d": [0.1, 0.0, z]}))
        redetect.rebind_before_approach(ex, "tray")
        assert ex.detection_map["tray"]["position_3d"] == [0.1, 0.0, want_z]


def test_placed_instance_is_not_rebound(monkeypatch):
    """'silverware 1' already placed keeps its binding; an unplaced or unique label is still re-detected."""
    from types import SimpleNamespace

    from spark_real.control import recovery as redetect

    calls = []
    monkeypatch.setattr(redetect, "rebind_settings", lambda e: {"enabled": True, "log_drift_cm": 2.0, "max_age_s": 0, "max_dz_m": 0.12})
    monkeypatch.setattr(redetect, "redetect_single", lambda e, label: calls.append(label))
    ex = SimpleNamespace(detection_map={}, _placed_labels={"silverware 1", "blue block"})
    for label in ("silverware 1", "silverware 2", "blue block"):
        redetect.rebind_before_approach(ex, label)
    assert calls == ["silverware 2", "blue block"]
