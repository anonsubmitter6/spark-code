from types import SimpleNamespace as NS
from spark_real.pipeline_plan import _planner_saw_not_done


def test_planner_no_on_an_abstain_opens_the_last_rung():
    assert _planner_saw_not_done("unverified", NS(reason="no 3d pass vote; llm rung (no): on the edge"))
    assert not _planner_saw_not_done("unverified", NS(reason="no 3d pass vote"))
    assert not _planner_saw_not_done("pass", NS(reason="llm rung (no): x"))
    assert not _planner_saw_not_done("unverified", None)
