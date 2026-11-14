from types import SimpleNamespace as NS
from spark_real.pipeline_plan import _drop_placed_instances


def test_instance_in_the_bowl_is_not_offered_for_a_pick():
    bowl = NS(label="bowl", position_3d=[0.0, 0.0, -0.22], obb_minor_m=0.18, aspect_ratio=1.0,
              world_major_axis_rad=0.0, slots=None)
    inside = NS(label="spoons 1", position_3d=[0.03, 0.02, -0.21])
    outside = NS(label="spoons 2", position_3d=[0.30, -0.20, -0.27])
    kept = _drop_placed_instances([bowl, inside, outside], {"bowl"})
    assert [d.label for d in kept] == ["bowl", "spoons 2"]
