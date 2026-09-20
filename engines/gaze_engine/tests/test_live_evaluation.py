from __future__ import annotations

import pytest

from mekasa_gaze.live_evaluation import PredictionSmoother


def test_prediction_smoother_resets_after_invalid_frame() -> None:
    smoother = PredictionSmoother(alpha=0.25)

    assert smoother.update((0.0, 1.0)) == (0.0, 1.0)
    assert smoother.update((1.0, 0.0)) == pytest.approx((0.25, 0.75))
    assert smoother.update(None) is None
    assert smoother.update((0.8, 0.2)) == (0.8, 0.2)
