"""Blink-robust smoothing and the wider distance range of the board pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mekasa_gaze.core.config import GuidedValidationConfig
from mekasa_gaze.core.eye_mode import EyeMode
from mekasa_gaze.runners.board_simulation import synthetic_model, synthetic_observation
from mekasa_gaze.runners.communication_board import GazePipeline
from mekasa_gaze.tracking.smoothing import RobustGazeSmoother, lid_in_transition

ROOT = Path(__file__).resolve().parents[3]
FPS = 30.0


def test_a_blink_does_not_restart_the_smoother_and_post_blink_outliers_are_outvoted() -> None:
    smoother = RobustGazeSmoother(alpha=0.25)
    t = 0.0
    for _ in range(15):
        t += 1 / FPS
        smoother.update(t, (0.3, 0.3))
    for _ in range(6):  # eyes closed for 0.2 s
        t += 1 / FPS
        assert smoother.update(t, None) == (None, False)
    t += 1 / FPS
    first, _ = smoother.update(t, (0.3, 0.9))  # lid still opening: estimate jumps down
    t += 1 / FPS
    second, _ = smoother.update(t, (0.3, 0.85))
    assert first is not None and second is not None
    assert first[1] < 0.35 and second[1] < 0.35  # the jump does not reach the output


def test_unreliable_frames_hold_and_long_gaps_reset() -> None:
    smoother = RobustGazeSmoother()
    smoother.update(0.0, (0.2, 0.2))
    assert smoother.update(0.03, (0.9, 0.9), reliable=False) == ((0.2, 0.2), True)
    smoother.update(1.0, None)  # a real loss of signal (> 0.6 s)
    value, held = smoother.update(1.03, (0.8, 0.8))
    assert value == pytest.approx((0.8, 0.8)) and not held


def test_lid_transition_band_stays_below_downward_looks() -> None:
    eyes = ("left", "right")
    # 2026-11-16, session A: threshold 0.082, bottom-row apertures 0.12-0.13.
    assert not lid_in_transition(
        {"left_eye_aspect_ratio": 0.12, "right_eye_aspect_ratio": 0.13}, eyes, 0.082
    )
    assert lid_in_transition(
        {"left_eye_aspect_ratio": 0.09, "right_eye_aspect_ratio": 0.13}, eyes, 0.082
    )
    assert not lid_in_transition({"left_eye_aspect_ratio": 0.09}, ("right",), 0.082)


def _pipeline(**kwargs: object) -> GazePipeline:
    validity = GuidedValidationConfig.load(ROOT / "config" / "experiment-001.toml")
    model = replace(synthetic_model(EyeMode.BOTH), closed_ear_threshold=0.09)
    return GazePipeline(
        extractor=None,  # type: ignore[arg-type]
        validity_config=validity,
        model=model,
        eye_mode=EyeMode.BOTH,
        smoothing_alpha=1.0,
        area=(0.0, 0.0, 1.0, 1.0),
        **kwargs,  # type: ignore[arg-type]
    )


def test_pipeline_holds_the_estimate_while_the_lid_is_half_closed() -> None:
    pipeline = _pipeline()
    steady = synthetic_observation((0.3, 0.3), "ok", 1, 1.0)
    assert pipeline.process_observation(steady, frame_timestamp_seconds=1.0).screen_point == (
        pytest.approx((0.3, 0.3))
    )
    closing = synthetic_observation((0.3, 0.9), "ok", 2, 1.033)
    features = dict(closing.features, left_eye_aspect_ratio=0.10, right_eye_aspect_ratio=0.10)
    result = pipeline.process_observation(
        replace(closing, features=features), frame_timestamp_seconds=1.033
    )
    assert result.held and result.screen_point == pytest.approx((0.3, 0.3))


def test_pipeline_accepts_a_wider_distance_range() -> None:
    narrow = _pipeline()  # research protocol band 0.25-0.45
    wide = _pipeline(face_width_band=(0.16, 0.62))
    for width in (0.18, 0.58):
        observation = synthetic_observation((0.3, 0.3), "ok", 1, 1.0)
        observation = replace(observation, features=dict(observation.features, face_width=width))
        assert not narrow.process_observation(observation, frame_timestamp_seconds=1.0).tracking_ok
        assert wide.process_observation(observation, frame_timestamp_seconds=1.0).tracking_ok
