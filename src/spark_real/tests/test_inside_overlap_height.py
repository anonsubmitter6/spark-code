import numpy as np
from spark_real.control import predicates as sp

BOWL = ((np.array([0.0, 0.0])), 0.09, 0.09, 0.0)


def _obj(x, y, minor=0.03, ar=5.0, theta=0.0, top=None, bottom=None):
    d = {"position_3d": [x, y, 0.0], "obb_minor_m": minor, "aspect_ratio": ar, "world_major_axis_rad": theta}
    if top is not None:
        d.update(rim_z_m=top, interior_z_m=bottom)
    return d


def test_spoon_leaning_on_the_rim_counts_as_inside():
    # 15 cm spoon, centre 2 cm outside the shrunk box edge but most of it over the bowl
    spoon = _obj(0.08, 0.0)
    assert not sp._obb_contains(spoon["position_3d"][:2], BOWL, -0.02)
    assert sp.footprint_overlap(spoon, BOWL) >= sp.INSIDE_MIN_OVERLAP


def test_object_beside_the_bowl_is_not_inside():
    assert sp.footprint_overlap(_obj(0.20, 0.0), BOWL) < sp.INSIDE_MIN_OVERLAP
    assert sp.footprint_overlap({"position_3d": [0, 0, 0]}, BOWL) is None


def test_own_height():
    assert abs(sp._own_height(_obj(0, 0, top=-0.10, bottom=-0.19)) - 0.09) < 1e-9
    assert sp._own_height(_obj(0, 0)) == 0.0 and sp._own_height(_obj(0, 0, top=-0.2, bottom=-0.1)) == 0.0


def test_thin_object_on_the_floor_of_a_deep_bowl_is_within_height():
    bowl = {"rim_z_m": -0.226, "interior_z_m": -0.267}
    assert abs(sp._own_height(bowl) - 0.041) < 1e-9
    dz = -0.046                       # spoon on the bowl floor, measured against the rim
    assert not (-sp.INSIDE_Z_BELOW_M <= dz)
    assert -(sp.INSIDE_Z_BELOW_M + sp._own_height(bowl)) <= dz


def test_object_wider_than_the_bowl_sitting_on_it_counts():
    big = _obj(0.03, 0.02, minor=0.24, ar=1.2)      # 24 x 29 cm plushie over an 18 cm bowl
    assert sp.footprint_overlap(big, BOWL) >= sp.INSIDE_MIN_OVERLAP
    assert sp.footprint_overlap(_obj(0.30, 0.0, minor=0.24, ar=1.2), BOWL) < sp.INSIDE_MIN_OVERLAP
