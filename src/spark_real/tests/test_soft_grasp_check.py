"""A compressed soft object counts as held; a skipped pick approach is not followed by a close."""
from types import SimpleNamespace

from spark_real.control.grasp import GraspExecMixin
from spark_real.control.grasp import GraspVerifyMixin


class _Ex(GraspVerifyMixin):
    GRIPPER_EMPTY_THRESHOLD, GRIPPER_FULLY_CLOSED = 235, 250

    def __init__(self, width, flag, pos):
        self._last_grasp_target_width = 0.035
        self.robot = SimpleNamespace(get_gripper_width=lambda: width)
        self._state = (flag, pos)
        self._pre_grasp_force = None
        self._last_keypoint_label = "plushie"

    def _read_gripper_state(self):
        return self._state

    def _vision_says_still_on_table(self, label):
        return True   # the held plushie is still seen at its spot before the lift

    def _gripper_type(self):
        return "robotiq_2f85"


def test_plushie_compressed_to_19mm_with_the_object_flag_is_held():
    assert _Ex(0.019, True, 198)._verify_grasp() is True


def test_jaws_at_their_closed_on_nothing_width_are_not_held():
    assert _Ex(0.009, True, 227)._verify_grasp() is not True


def test_no_close_after_a_skipped_pick_approach():
    ex = SimpleNamespace(_pick_approach_skipped=True, _check_abort=lambda: None)
    result = GraspExecMixin._grasp(ex, {}, 0.0)
    assert result.success is False and "already in its container" in result.message
    assert ex._pick_approach_skipped is False
