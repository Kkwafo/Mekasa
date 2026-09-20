from __future__ import annotations

from pathlib import Path

from mekasa_gaze.config import GuidedValidationConfig
from mekasa_gaze.domain import (
    ExperimentPhase,
    FeatureObservation,
    GazeValidity,
    GuidedObservation,
    MeasurementStage,
    TrackingState,
)
from mekasa_gaze.validity import assess_gaze_validity, mark_possible_blinks


def _config() -> GuidedValidationConfig:
    return GuidedValidationConfig.load(Path("config/experiment-001.toml"))


def _features() -> dict[str, float]:
    return {
        "left_eye_aspect_ratio": 0.3,
        "right_eye_aspect_ratio": 0.3,
        "left_eye_vertical_span": 0.03,
        "right_eye_vertical_span": 0.03,
        "left_iris_x": 0.5,
        "left_iris_y": 0.5,
        "right_iris_x": 0.5,
        "right_iris_y": 0.5,
    }


def _feature_observation(
    *,
    state: TrackingState = TrackingState.GOOD,
    changes: dict[str, float] | None = None,
) -> FeatureObservation:
    features = _features()
    features.update(changes or {})
    return FeatureObservation(1, 1, state, features=features)


def _guided(timestamp: float, validity: GazeValidity) -> GuidedObservation:
    assessment = assess_gaze_validity(_feature_observation(), _config())
    return GuidedObservation(
        relative_timestamp_seconds=timestamp,
        phase=ExperimentPhase.CENTER_STABILITY,
        stage=MeasurementStage.MEASUREMENT,
        target=None,
        tracking_state=TrackingState.GOOD,
        gaze_validity=validity,
        left_eye_state=assessment.left_eye_state,
        right_eye_state=assessment.right_eye_state,
    )


def test_open_finite_eye_geometry_is_gaze_valid() -> None:
    assessment = assess_gaze_validity(_feature_observation(), _config())
    assert assessment.gaze_validity is GazeValidity.VALID


def test_tracking_loss_is_distinct_from_gaze_geometry() -> None:
    assessment = assess_gaze_validity(_feature_observation(state=TrackingState.LOST), _config())
    assert assessment.gaze_validity is GazeValidity.TRACKING_INVALID


def test_low_ear_is_classified_and_not_silently_dropped() -> None:
    assessment = assess_gaze_validity(
        _feature_observation(changes={"left_eye_aspect_ratio": 0.05}), _config()
    )
    assert assessment.gaze_validity is GazeValidity.EYE_CLOSED


def test_degenerate_vertical_span_is_invalid() -> None:
    assessment = assess_gaze_validity(
        _feature_observation(changes={"right_eye_vertical_span": 0.001}), _config()
    )
    assert assessment.gaze_validity is GazeValidity.INVALID


def test_extreme_iris_coordinate_is_invalid() -> None:
    assessment = assess_gaze_validity(
        _feature_observation(changes={"left_iris_y": 50.0}), _config()
    )
    assert assessment.gaze_validity is GazeValidity.INVALID


def test_short_bounded_eye_closure_becomes_possible_blink() -> None:
    observations = [
        _guided(0.00, GazeValidity.VALID),
        _guided(0.10, GazeValidity.EYE_CLOSED),
        _guided(0.20, GazeValidity.EYE_CLOSED),
        _guided(0.30, GazeValidity.VALID),
    ]
    marked, events = mark_possible_blinks(observations, max_blink_seconds=0.45)
    assert events == 1
    assert marked[1].gaze_validity is GazeValidity.BLINK
    assert marked[2].gaze_validity is GazeValidity.BLINK


def test_unbounded_or_long_closure_remains_eye_closed() -> None:
    unbounded, unbounded_events = mark_possible_blinks(
        [_guided(0.0, GazeValidity.EYE_CLOSED), _guided(0.1, GazeValidity.VALID)],
        max_blink_seconds=0.45,
    )
    long, long_events = mark_possible_blinks(
        [
            _guided(0.0, GazeValidity.VALID),
            _guided(0.1, GazeValidity.EYE_CLOSED),
            _guided(0.7, GazeValidity.EYE_CLOSED),
            _guided(0.8, GazeValidity.VALID),
        ],
        max_blink_seconds=0.45,
    )
    assert unbounded_events == 0
    assert unbounded[0].gaze_validity is GazeValidity.EYE_CLOSED
    assert long_events == 0
    assert long[1].gaze_validity is GazeValidity.EYE_CLOSED
