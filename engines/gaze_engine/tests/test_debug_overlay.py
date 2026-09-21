from __future__ import annotations

import numpy as np

from mekasa_gaze.core.domain import FeatureObservation, HeadPose, TrackingState
from mekasa_gaze.tracking.landmarker import draw_debug_overlay


def test_debug_overlay_draws_eye_point_and_status_without_mutating_frame() -> None:
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    observation = FeatureObservation(
        frame_sequence=1,
        timestamp_ns=1,
        tracking_state=TrackingState.GOOD,
        head_pose=HeadPose(yaw_degrees=1.0, pitch_degrees=2.0, roll_degrees=3.0),
        evidence="synthetic_test",
        feature_latency_ms=4.5,
        debug_eye_points=((0.5, 0.5),),
    )

    output = draw_debug_overlay(frame, observation)

    assert output.shape == frame.shape
    assert np.count_nonzero(output) > 0
    assert np.count_nonzero(frame) == 0
