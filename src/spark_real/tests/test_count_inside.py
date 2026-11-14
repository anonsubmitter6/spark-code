"""Numbered look-alikes are verified by how many lie inside the container, not by identity."""
from spark_real.control import predicates as sp
from spark_real.control.common import Predicate

TRAY = {"label": "tray", "position_3d": [0.0, 0.0, 0.0], "obb_minor_m": 0.20, "aspect_ratio": 2.0, "confidence": 0.9}


def _piece(x, conf=0.4, z=0.01):
    return {"label": "silverware", "position_3d": [x, 0.0, z], "confidence": conf}


def _vote(dets, need, total):
    view = sp.SceneView(camera="birdview", lookup=lambda label: TRAY if label == "tray" else None)
    pred = Predicate("inside", {"obj": "silverware 1", "container": "tray"})
    return sp.count_inside_vote(view, pred, dets, "silverware", need, total, sp.EvalConfig()).vote


def test_final_check_counts_pieces():
    assert _vote([_piece(0.0), _piece(0.05), _piece(-0.05)], 3, 3) == sp.PASS
    assert _vote([_piece(0.0), _piece(0.05), _piece(0.60)], 3, 3) == sp.FAIL      # one on the table
    assert _vote([_piece(0.0), _piece(0.05)], 3, 3) == sp.PASS                     # two seen inside, none left outside
    assert _vote([], 3, 3) == sp.ABSTAIN                                              # nothing seen at all
    assert _vote([_piece(0.0), _piece(0.05), _piece(0.0, z=0.30)], 3, 3) == sp.FAIL  # far above the tray


def test_per_placement_check_ignores_pieces_still_on_the_table():
    assert _vote([_piece(0.0), _piece(0.60), _piece(0.70)], 1, 3) == sp.PASS
    assert _vote([_piece(0.50), _piece(0.60), _piece(0.70)], 1, 3) == sp.FAIL    # all seen, none inside
    assert _vote([_piece(0.60)], 1, 3) == sp.ABSTAIN                               # placed piece hidden
    assert _vote([_piece(0.0, conf=0.1), _piece(0.60)], 1, 3) == sp.ABSTAIN        # too weak to count
