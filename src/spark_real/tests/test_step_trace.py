"""Every executed step leaves one trace record with its outcome and times."""
import json
from types import SimpleNamespace

from spark_real.control import bt as bt_dispatch


def test_one_record_per_step(tmp_path):
    mixin = next(c for c in vars(bt_dispatch).values() if isinstance(c, type) and hasattr(c, "_trace_step"))
    ex = SimpleNamespace(
        _pipeline=SimpleNamespace(profile=SimpleNamespace(raw={}), config=SimpleNamespace(output_dir=str(tmp_path))),
        detection_map={"tray": {"position_3d": [0.1, 0.2, -0.27], "confidence": 0.9}}, _holding=True)
    for name, ok in (("place_in_slot", True), ("grasp", False)):
        mixin._trace_step(ex, name, {"container_label": "tray"}, SimpleNamespace(success=ok, message="m"), 100.0)
    files = sorted((tmp_path / "trace").glob("*.json"))
    assert [f.name for f in files] == ["001_place_in_slot.json", "002_grasp.json"]
    rec = json.loads(files[0].read_text())
    assert rec["success"] is True and rec["detections_used"][0]["label"] == "tray" and rec["t_end"] >= rec["t_start"]
