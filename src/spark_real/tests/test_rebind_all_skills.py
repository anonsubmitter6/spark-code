"""The dispatcher re-detects every scene label a skill names, except the held object."""
from types import SimpleNamespace

from spark_real.control import bt as bt_dispatch, recovery as execution_recovery


def test_labels_rebound_except_held(monkeypatch):
    seen = []
    monkeypatch.setattr(execution_recovery, "rebind_before_approach", lambda ex, label: seen.append(label))
    ex = SimpleNamespace(_holding=True, _last_pick_label="blue block")
    mixin = next(c for c in vars(bt_dispatch).values()
                 if isinstance(c, type) and hasattr(c, "_rebind_action_labels"))
    mixin._rebind_action_labels(ex, {"target_label": "grey block", "keypoint_label": "blue block", "force": 10})
    assert seen == ["grey block"]
