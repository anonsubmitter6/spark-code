import numpy as np
from spark_real.perception_steps import rim_centre_xy


def test_rim_centre_ignores_uneven_interior():
    """A ring at the rim height plus interior points on one side only: the centre is the ring's."""
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    T = np.eye(4); T[:3, :3] = np.diag([1, -1, -1]); T[2, 3] = 1.0   # camera 1 m up, looking down
    depth = np.zeros((480, 640)); mask = np.zeros((480, 640), bool)
    yy, xx = np.mgrid[0:480, 0:640]; r = np.hypot(xx - 320, yy - 240)
    ring = (r > 50) & (r < 60); depth[ring] = 0.80; mask |= ring            # rim at z = 0.20
    inside = (r < 50) & (xx > 330); depth[inside] = 0.84; mask |= inside     # floor seen on one side only
    c = rim_centre_xy(mask, depth, K, T, rim_z=0.20)
    assert c is not None and abs(c[0]) < 0.004 and abs(c[1]) < 0.004
    med_x = np.median(((xx[mask] - 320) * depth[mask] / 600))
    assert med_x > 0.02          # the per-axis median, by contrast, is pulled to the visible side
