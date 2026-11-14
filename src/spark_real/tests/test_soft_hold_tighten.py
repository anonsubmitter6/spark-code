import spark_real.control.grasp as g

CLS = next(v for v in vars(g).values() if isinstance(v, type) and hasattr(v, "_tighten_soft_hold"))


class Fake(CLS):
    ROBOTIQ_POS_MAX = 255

    def __init__(self, widths, tw):
        self._w = list(widths); self._last_grasp_target_width = tw; self.forces = []
        self.robot = type("R", (), {"get_gripper_width": lambda s: self._w[0]})()

    def _robotiq_resqueeze(self, speed=50, force=100, settle=1.0):
        self.forces.append(force)
        if len(self._w) > 1:
            self._w.pop(0)
        return True

    def _robotiq_travel_settle_s(self, *a, **k):
        return 0.0

    def _read_gripper_state(self):
        return True, 200


def test_wide_soft_hold_is_squeezed_until_it_stops_shrinking():
    f = Fake([0.084, 0.050, 0.049], tw=0.035)          # big plushie: 84 mm -> 50 -> 49 (compressed)
    assert f._tighten_soft_hold(20, True, 127) == (True, 200)
    assert f.forces == [40, 80]


def test_hold_at_the_planned_width_is_left_alone():
    f = Fake([0.035], tw=0.035)                        # a block
    assert f._tighten_soft_hold(60, True, 150) == (True, 150) and f.forces == []
    f = Fake([0.019], tw=0.035)                        # small plushie, already narrower than planned
    f._tighten_soft_hold(20, True, 199); assert f.forces == []
