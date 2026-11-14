"""bind_across_views: every phrase of the pick lands on the place both cameras agree on, not a one-view look-alike."""
import numpy as np

from spark_core.sam3 import ObjectDetection
from spark_bench.libero_pro.perception.detect import bind_across_views


def _d(label, conf, xy):
    return ObjectDetection(label=label, confidence=conf, centroid_2d=(0, 0), mask_area=100,
                           position_3d=np.array([xy[0], xy[1], 0.05]), mask=np.ones((2, 2), bool))


def test_views_outvote_one_view_lookalike():
    target, lookalike = (-0.12, -0.24), (0.16, 0.03)
    agent = [_d('Campbell soup can', 0.95, lookalike), _d('Campbell soup can', 0.55, target)]
    wrist = [_d('can with alphabet letters', 0.87, target), _d('Campbell soup can', 0.31, lookalike)]
    phrases = ['alphabet soup', 'can with alphabet letters', 'Campbell soup can']
    out = bind_across_views(agent, wrist, phrases)
    assert [d.label for d in out] == phrases
    for d in out:
        assert np.allclose(d.position_3d[:2], target)
        assert d.mask is not None                       # the agentview mask is kept for grasping
    assert np.allclose(out[0].position_agentview[:2], target)
    # each phrase keeps its own score there: the name, never detected, stays weak
    assert [round(d.confidence, 2) for d in out] == [0.05, 0.87, 0.55]
    assert bind_across_views([], [], phrases) == []


def test_plan_tracks_the_phrase_sam3_sees():
    from spark_bench.fair.planning import track_strongest_phrase
    xy = (0.05, -0.10)
    agent = [_d('small white rectangular box', 0.72, xy), _d('cream cheese box', 0.14, xy)]
    phrases = ['cream cheese', 'cream cheese box', 'small white rectangular box']
    det_map = {d.label: d for d in bind_across_views(agent, [], phrases)}
    det_map['basket'] = _d('basket', 0.9, (0.0, 0.26))
    score = {'tree': {'type': 'sequence', 'children': [
        {'type': 'move_to_keypoint', 'params': {'keypoint_label': 'cream cheese'}},
        {'type': 'grasp', 'params': {}},
        {'type': 'move_to_keypoint', 'params': {'keypoint_label': 'basket'}}]}}
    acts = track_strongest_phrase(score, det_map)
    labels = [c['params'].get('keypoint_label') for c in score['tree']['children']]
    assert labels == ['small white rectangular box', None, 'basket']
    assert acts == [{'event': 'track_strongest_phrase', 'from': 'cream cheese',
                     'to': 'small white rectangular box', 'key': 'keypoint_label'}]
    assert track_strongest_phrase(score, det_map) == []     # already the strongest: no change
