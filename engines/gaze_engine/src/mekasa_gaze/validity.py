from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from mekasa_gaze.config import GuidedValidationConfig
from mekasa_gaze.domain import (
    EyeState,
    FeatureObservation,
    GazeValidity,
    GuidedObservation,
    TrackingState,
)

_REQUIRED_FEATURES = (
    "left_eye_aspect_ratio",
    "right_eye_aspect_ratio",
    "left_eye_vertical_span",
    "right_eye_vertical_span",
    "left_iris_x",
    "left_iris_y",
    "right_iris_x",
    "right_iris_y",
)


@dataclass(frozen=True, slots=True)
class ValidityAssessment:
    gaze_validity: GazeValidity
    left_eye_state: EyeState
    right_eye_state: EyeState
    reason: str


def assess_gaze_validity(
    observation: FeatureObservation, config: GuidedValidationConfig
) -> ValidityAssessment:
    if observation.tracking_state is not TrackingState.GOOD:
        return ValidityAssessment(
            GazeValidity.TRACKING_INVALID,
            EyeState.INVALID,
            EyeState.INVALID,
            "tracking_not_good",
        )

    features = observation.features
    if any(name not in features or not isfinite(features[name]) for name in _REQUIRED_FEATURES):
        return ValidityAssessment(
            GazeValidity.INVALID,
            EyeState.INVALID,
            EyeState.INVALID,
            "missing_or_non_finite_feature",
        )

    left_state = _eye_state(
        features["left_eye_aspect_ratio"],
        features["left_eye_vertical_span"],
        config,
    )
    right_state = _eye_state(
        features["right_eye_aspect_ratio"],
        features["right_eye_vertical_span"],
        config,
    )
    if EyeState.INVALID in (left_state, right_state):
        return ValidityAssessment(
            GazeValidity.INVALID,
            left_state,
            right_state,
            "degenerate_eye_geometry",
        )
    if EyeState.CLOSED in (left_state, right_state):
        return ValidityAssessment(
            GazeValidity.EYE_CLOSED,
            left_state,
            right_state,
            "eye_aspect_ratio_below_experimental_threshold",
        )

    iris_values = (features[name] for name in _REQUIRED_FEATURES if "iris_" in name)
    if any(abs(value) > config.max_abs_iris_coordinate for value in iris_values):
        return ValidityAssessment(
            GazeValidity.INVALID,
            left_state,
            right_state,
            "iris_coordinate_outlier",
        )
    return ValidityAssessment(
        GazeValidity.VALID,
        left_state,
        right_state,
        "valid_experimental_geometry",
    )


def _eye_state(
    aspect_ratio: float,
    vertical_span: float,
    config: GuidedValidationConfig,
) -> EyeState:
    if vertical_span < config.min_eye_vertical_span:
        return EyeState.INVALID
    if aspect_ratio < config.eye_closed_ear_threshold:
        return EyeState.CLOSED
    return EyeState.OPEN


def mark_possible_blinks(
    observations: list[GuidedObservation], max_blink_seconds: float
) -> tuple[list[GuidedObservation], int]:
    """Relabel short, bounded eye-closure runs as possible blinks."""
    output = list(observations)
    events = 0
    index = 0
    while index < len(output):
        if output[index].gaze_validity is not GazeValidity.EYE_CLOSED:
            index += 1
            continue
        start = index
        while (
            index + 1 < len(output) and output[index + 1].gaze_validity is GazeValidity.EYE_CLOSED
        ):
            index += 1
        end = index
        bounded_by_open = (
            start > 0
            and end + 1 < len(output)
            and output[start - 1].gaze_validity is GazeValidity.VALID
            and output[end + 1].gaze_validity is GazeValidity.VALID
        )
        duration = output[end].relative_timestamp_seconds - output[start].relative_timestamp_seconds
        if bounded_by_open and duration <= max_blink_seconds:
            events += 1
            for blink_index in range(start, end + 1):
                output[blink_index] = replace(output[blink_index], gaze_validity=GazeValidity.BLINK)
        index += 1
    return output, events
