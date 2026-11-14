"""A spoon-shaped mask (solidity 0.73-0.79) is not rejected by the solidity check."""
from spark_real.perception.detection_fusion import MaskQuality, QualityThresholds, _verdict


def test_spoon_solidity_passes_and_ragged_mask_fails():
    for solidity, flagged in ((0.75, False), (0.65, True)):
        q = MaskQuality(pixels=5000)
        q.fill, q.solidity, q.component_frac, q.angle_swing_deg, q.mask_aspect_ratio = 0.9, solidity, 1.0, 0.0, 4.0
        _verdict(q, QualityThresholds(), None)
        assert any("solidity" in r for r in q.reasons) == flagged
