"""Wrist check: the frozen question and crops when SPARK_PROMPT is unset, the outlined ones under v2."""
from types import SimpleNamespace

import numpy as np
import pytest

from spark_bench.libero_pro.perception import wrist_verify as wv

FROZEN_ASK = ("The image is a grid of numbered close-up crops of objects on a table. "
              "Which number is the akita black bowl? Read printed labels on the packages. "
              "Finish with a line ANSWER: <number>, using 0 if none of them is the akita black bowl.")
MAGENTA = (255, 0, 255)


def _det(label, x):
    mask = np.zeros((128, 128), bool)
    mask[40:80, x:x + 30] = True
    return SimpleNamespace(label=label, confidence=0.4, mask=mask, position_3d=np.array([x / 100, 0.0, 0.9]),
                           centroid_2d=(x + 15, 60), mask_area=1200, depth_meters=0.5)


def _run(monkeypatch):
    rgb = np.random.default_rng(0).integers(0, 200, (128, 128, 3), dtype=np.uint8)
    det = SimpleNamespace(det_map={'black bowl': _det('black bowl', 20)}, dets=[], rgb=rgb)
    fresh = SimpleNamespace(dets=[_det('container', 70)], rgb=rgb)
    seen = {}
    monkeypatch.setattr(wv, 'redetect_wrist', lambda env, sam3, prompts, cfg, all_instances: seen.update(prompts=prompts) or fresh)
    monkeypatch.setattr(wv.SPARKPlanner, '__init__', lambda self: None)
    monkeypatch.setattr(wv.SPARKPlanner, 'generate',
                        lambda self, parts, **kw: seen.update(ask=parts[0], images=parts[1:], grid=np.asarray(parts[-1])) or 'x' * 700 + ' ANSWER: 1')
    meta = {}
    wv.wrist_verify_picks(None, None, det, ['akita_black_bowl'], SimpleNamespace(verbose=False), trial_meta=meta)
    return det, fresh, seen, meta['wrist_verify'][0]


def test_unset_keeps_the_frozen_question_and_crops(monkeypatch):
    monkeypatch.delenv('SPARK_PROMPT', raising=False)
    monkeypatch.delenv('SPARK_SKILLSET', raising=False)
    det, fresh, seen, rec = _run(monkeypatch)
    assert seen['ask'] == FROZEN_ASK and seen['prompts'] == list(wv._VERIFY_GENERIC) and len(seen['images']) == 1
    assert np.array_equal(seen['grid'], np.asarray(wv._crop_grid([(fresh.rgb, fresh.dets[0].mask)])))
    assert not (seen['grid'] == MAGENTA).all(-1).any()
    assert rec['n_cands'] == 1 and rec['dropped'] == ['black bowl'] and list(det.det_map) == ['akita black bowl']
    assert len(rec['reply']) == 600 and rec['reply'].endswith('ANSWER: 1')


def test_v2_outlines_the_candidate_and_lets_a_related_detection_compete(monkeypatch):
    monkeypatch.setenv('SPARK_PROMPT', 'v2')
    monkeypatch.delenv('SPARK_SKILLSET', raising=False)
    det, fresh, seen, rec = _run(monkeypatch)
    assert 'outlined in magenta' in seen['ask'] and 'do not count' in seen['ask'] and 'packages' not in seen['ask']
    assert 'image 2 and image 3 refer to the same candidates' in seen['ask'] and 'is candidate 1' in seen['ask']
    # three images: the external view with the binding outlined, the wrist view with candidate 2, the crops
    agent, wrist = (np.asarray(i) for i in seen['images'][:2])
    assert len(seen['images']) == 3 and agent.shape == wrist.shape == det.rgb.shape
    for img, x in ((agent, 20), (wrist, 70)):
        ys, xs = np.where((img == MAGENTA).all(-1))
        assert len(xs) > 100 and x - 3 <= xs.min() and xs.max() <= x + 32
    assert seen['ask'].endswith('using 0 when no outlined object is the akita black bowl.')
    assert seen['prompts'] == list(wv._VERIFY_GENERIC) + ['akita black bowl']
    crops = [(det.rgb, det.det_map['black bowl'].mask), (fresh.rgb, fresh.dets[0].mask)]
    plain = np.asarray(wv._crop_grid(crops))
    changed = (seen['grid'] != plain).any(-1)                     # the outline is the only difference
    assert changed.sum() > 200 and (seen['grid'][changed] == MAGENTA).all()
    # crop 1 is the agent-view detection under the related label; the reply keeps it
    assert rec['n_cands'] == 2 and rec['agent_label'] == 'black bowl' and rec['kept_agentview']
    assert list(det.det_map) == ['black bowl'] and rec['reply'].endswith('ANSWER: 1')
