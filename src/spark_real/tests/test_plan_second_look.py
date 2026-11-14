from types import SimpleNamespace as NS

import numpy as np

import spark_real.pipeline_plan as pp

CLS = next(v for k, v in vars(pp).items() if isinstance(v, type) and hasattr(v, "_redetect_missing_instances"))


def _det(label, x):
    return NS(label=label, position_3d=[x, 0.0, -0.2], confidence=0.9, centroid_2d=(100 + int(x * 100), 100),
              bbox=(90, 90, 110, 110), mask=None, camera="birdview", aspect_ratio=1.0, orientation_angle=0.0)


class Fake(CLS):
    def __init__(self, fresh):
        self._fresh = fresh
        self.asked = None

    def capture(self):
        return {"birdview": {"rgb": np.zeros((240, 320, 3), np.uint8)}}

    def detect(self, captures, prompts, multi_instance_prompts=None, **kw):
        self.asked = (list(prompts), list(multi_instance_prompts or []))
        return list(self._fresh)

    def merge_detections(self, dets, spec=None):
        return list(dets)


def test_named_instances_trigger_a_second_look(monkeypatch):
    monkeypatch.setattr("spark_real.pipeline_plan.annotate_for_planner", lambda rgb, dets, **k: rgb)
    monkeypatch.setattr("spark_real.pipeline_plan.build_detection_details", lambda dets: [{"label": d.label} for d in dets])
    dets = [_det("bowl", 0.0), _det("spoons", 0.05)]
    p = Fake([_det("spoons 1", 0.05), _det("spoons 2", 0.4)])
    out = p._redetect_missing_instances({"spoons:A", "spoons:B"}, dets, np.zeros((2, 2, 3), np.uint8), "birdview")
    assert out is not None and p.asked == (["spoons"], ["spoons"])
    labels, details, _img = out
    assert labels == ["bowl", "spoons 1", "spoons 2"] == [d.label for d in dets]   # same list, in place
    assert [d["label"] for d in details] == labels


def test_no_second_look_for_an_unrelated_or_unchanged_label(monkeypatch):
    monkeypatch.setattr("spark_real.pipeline_plan.annotate_for_planner", lambda rgb, dets, **k: rgb)
    dets = [_det("bowl", 0.0), _det("spoons", 0.05)]
    assert Fake([])._redetect_missing_instances({"fork"}, dets, None, "birdview") is None
    assert Fake([_det("spoons", 0.05)])._redetect_missing_instances({"spoons:A"}, dets, None, "birdview") is None
    assert [d.label for d in dets] == ["bowl", "spoons"]
