"""Stack's second look adopts a clear move of the base and ignores occlusion-sized shifts."""
from types import SimpleNamespace

from spark_real.control import recovery as execution_recovery
from spark_real.skills import stack as stack_mod


def _run(monkeypatch, new_xy):
    ex = SimpleNamespace(detection_map={"base": {"position_3d": [0.0, 0.0, 0.1]}})
    monkeypatch.setattr(execution_recovery, "rebind_before_approach",
                        lambda e, label: e.detection_map.__setitem__(label, {"position_3d": [*new_xy, 0.1]}))
    return ex, stack_mod._redetect_if_moved(ex, "base", [0.0, 0.0, 0.1])


def test_clear_move_is_adopted(monkeypatch):
    ex, det = _run(monkeypatch, (0.20, 0.0))
    assert det["position_3d"][0] == 0.20 and ex.detection_map["base"]["position_3d"][0] == 0.20


def test_small_shift_is_ignored_and_map_restored(monkeypatch):
    ex, det = _run(monkeypatch, (0.02, 0.0))
    assert det is None and ex.detection_map["base"]["position_3d"][0] == 0.0


def test_the_registered_stack_skill_is_the_stack_function():
    """The skill decorator must sit on ``stack`` itself, not on a helper defined above it."""
    from spark_real.control import bt as bt_dispatch

    for name in ("stack", "place_in_slot"):
        assert bt_dispatch.skill_registry.get(name).fn.__name__ == name
