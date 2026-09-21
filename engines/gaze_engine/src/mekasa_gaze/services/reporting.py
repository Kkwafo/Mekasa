from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any

import numpy as np

from mekasa_gaze.core.config import GuidedValidationConfig
from mekasa_gaze.core.domain import (
    CameraFrame,
    CameraSettings,
    ExperimentPhase,
    EyeState,
    FeatureObservation,
    GazeValidity,
    GuidedObservation,
    MeasurementStage,
    ProtocolSnapshot,
    TargetName,
    TrackingState,
)
from mekasa_gaze.services.metrics import ExperimentMetrics
from mekasa_gaze.tracking.validity import ValidityAssessment, mark_possible_blinks

_PERSISTED_FEATURES = (
    "left_eye_aspect_ratio",
    "right_eye_aspect_ratio",
    "left_eye_vertical_span",
    "right_eye_vertical_span",
    "left_iris_x",
    "left_iris_y",
    "right_iris_x",
    "right_iris_y",
    "face_center_x",
    "face_center_y",
    "face_width",
    "face_height",
)

_ANALYZED_FEATURES = (
    "left_iris_x",
    "left_iris_y",
    "right_iris_x",
    "right_iris_y",
    "head_yaw_degrees",
    "head_pitch_degrees",
    "head_roll_degrees",
)


class GuidedReportBuilder:
    """Builder for structured experimental validation reports.

    Responsibility:
        Aggregates frame observations during a guided validation run, tracks pipeline
        latencies, relabels physiological blinks, calculates per-target statistics, and compiles
        a comprehensive JSON report adhering to Schema 5.
    """

    def __init__(self) -> None:
        """Initialize empty observation accumulator and performance tracker."""
        self._performance: ExperimentMetrics | None = None
        self._observations: list[GuidedObservation] = []
        self._first_timestamp_ns: int | None = None

    def start_measurement(self) -> None:
        """Begin performance metrics tracking if not already active."""
        if self._performance is None:
            self._performance = ExperimentMetrics.start()

    def add(
        self,
        frame: CameraFrame,
        feature_observation: FeatureObservation,
        snapshot: ProtocolSnapshot,
        assessment: ValidityAssessment,
        loop_latency_ms: float,
        *,
        render_latency_ms: float = 0.0,
        display_latency_ms: float = 0.0,
        iteration_latency_ms: float = 0.0,
    ) -> None:
        """Record an observation tick from the guided validation loop.

        Args:
            frame: Camera frame with capture timing.
            feature_observation: Landmark and feature data.
            snapshot: Active protocol state machine snapshot.
            assessment: Sample validity qualification.
            loop_latency_ms: Core processing loop latency in milliseconds.
            render_latency_ms: UI rendering latency in milliseconds.
            display_latency_ms: Window presentation latency in milliseconds.
            iteration_latency_ms: Frame-to-frame period in milliseconds.

        Side Effects:
            Appends a new GuidedObservation to internal memory and updates metrics.
        """
        self.start_measurement()
        assert self._performance is not None
        if self._first_timestamp_ns is None:
            self._first_timestamp_ns = feature_observation.timestamp_ns
        self._performance.add(
            frame,
            feature_observation,
            loop_latency_ms,
            render_latency_ms=render_latency_ms,
            display_latency_ms=display_latency_ms,
            iteration_latency_ms=iteration_latency_ms,
        )
        persisted_features = {
            name: float(feature_observation.features[name])
            for name in _PERSISTED_FEATURES
            if name in feature_observation.features and isfinite(feature_observation.features[name])
        }
        self._observations.append(
            GuidedObservation(
                relative_timestamp_seconds=(
                    feature_observation.timestamp_ns - self._first_timestamp_ns
                )
                / 1_000_000_000,
                phase=snapshot.phase,
                stage=snapshot.stage,
                target=snapshot.target,
                tracking_state=feature_observation.tracking_state,
                gaze_validity=assessment.gaze_validity,
                left_eye_state=assessment.left_eye_state,
                right_eye_state=assessment.right_eye_state,
                block_index=snapshot.block_index,
                segment_index=snapshot.segment_index,
                validity_reason=assessment.reason,
                features=persisted_features,
                head_pose=feature_observation.head_pose,
                capture_latency_ms=frame.capture_latency_ms,
                feature_latency_ms=feature_observation.feature_latency_ms,
                loop_latency_ms=loop_latency_ms,
                render_latency_ms=render_latency_ms,
                display_latency_ms=display_latency_ms,
                iteration_latency_ms=iteration_latency_ms,
            )
        )

    def build(
        self,
        *,
        camera: CameraSettings,
        environment: dict[str, Any],
        config: GuidedValidationConfig,
        experiment: dict[str, Any],
        completed: bool,
        aborted: bool,
    ) -> dict[str, Any]:
        """Compile recorded observations, latencies, and statistics into a Schema 5 report.

        Args:
            camera: Hardware camera settings.
            environment: Runtime platform metadata.
            config: Protocol configuration parameters.
            experiment: Experiment provenance metadata.
            completed: Whether the protocol ran to normal completion.
            aborted: Whether the operator aborted the protocol early.

        Returns:
            dict[str, Any]: Complete guided validation report dictionary.

        Raises:
            ValueError: If called before any observations were recorded.
        """
        if self._performance is None:
            raise ValueError("Cannot build a guided report without observations")
        observations, blink_events = mark_possible_blinks(
            self._observations, config.max_blink_seconds
        )
        report = self._performance.report(camera, environment)
        measurement = report["measurement"]
        measurement.pop("feature_stability", None)
        measurement.update(summarize_guided_observations(observations, config, blink_events))
        report.update(
            {
                "schema_version": 5,
                "experiment": experiment,
                "protocol": {
                    "completed": completed,
                    "aborted_by_operator": aborted,
                    "target_coordinate_space": "normalized within the TEST AREA",
                    "configuration": _guided_configuration(config),
                },
                "observations": [_observation_dict(item) for item in observations],
                "privacy": {
                    "frames_persisted": False,
                    "video_uploaded": False,
                    "raw_landmarks_persisted": False,
                    "derived_features_persisted": True,
                },
                "limitations": [
                    "Targets are prompts, not proof that the participant fixated them.",
                    "Eye-validity thresholds are experimental heuristics, not clinical cutoffs.",
                    (
                        "Possible blinks are short bounded EAR events, "
                        "not a validated blink classifier."
                    ),
                    "Head pose is an unvalidated diagnostic approximation.",
                    "Separability metrics are descriptive and are not classification accuracy.",
                    "This technical run is not clinical validation.",
                ],
            }
        )
        return report


def summarize_guided_observations(
    observations: list[GuidedObservation],
    config: GuidedValidationConfig,
    blink_events: int,
) -> dict[str, Any]:
    """Calculate aggregated validity, target distributions, and separability statistics.

    Responsibility:
        Processes the sequence of observations to compute sample validity ratios, breakdown
        by eye and rejection cause, target-by-target medians and MADs, and feature separability.

    Args:
        observations: Sequential list of recorded guided observations.
        config: Guided validation experiment configuration.
        blink_events: Total count of detected physiological blink runs.

    Returns:
        dict[str, Any]: Consolidated dictionary of summary metrics.
    """
    validity_counts = Counter(item.gaze_validity for item in observations)
    tracking_frames = sum(item.tracking_state is TrackingState.GOOD for item in observations)
    fully_valid_frames = validity_counts[GazeValidity.VALID]
    partial_frames = validity_counts[GazeValidity.PARTIAL]
    valid_frames = fully_valid_frames + partial_frames
    total_frames = len(observations)
    measurement_records = [item for item in observations if item.is_valid_measurement]

    discrimination_records = [
        item for item in measurement_records if item.phase is ExperimentPhase.GAZE_DISCRIMINATION
    ]
    target_statistics = {
        target.value: _statistics_for_records(
            [
                item
                for item in discrimination_records
                if item.target is not None and item.target.name is target
            ],
            config.min_samples_per_target,
        )
        for target in TargetName
    }

    block_target_statistics = {
        f"BLOCK_{block_index}:{target.value}": _statistics_for_records(
            [
                item
                for item in discrimination_records
                if item.block_index == block_index
                and item.target is not None
                and item.target.name is target
            ],
            config.min_samples_per_target,
        )
        for block_index in range(1, config.discrimination_blocks + 1)
        for target in TargetName
    }

    phase_target_statistics: dict[str, Any] = {}
    for phase in (
        ExperimentPhase.CENTER_STABILITY,
        ExperimentPhase.GAZE_DISCRIMINATION,
        ExperimentPhase.HEAD_MOVEMENT,
    ):
        for target in TargetName:
            records = [
                item
                for item in measurement_records
                if item.phase is phase and item.target is not None and item.target.name is target
            ]
            if records or (
                target is TargetName.CENTER
                and phase in (ExperimentPhase.CENTER_STABILITY, ExperimentPhase.HEAD_MOVEMENT)
            ):
                phase_target_statistics[f"{phase.value}:{target.value}"] = _statistics_for_records(
                    records, config.min_samples_per_target
                )

    return {
        "frames_total": total_frames,
        "frames_with_tracking": tracking_frames,
        "frames_gaze_valid": valid_frames,
        "frames_gaze_fully_valid": fully_valid_frames,
        "frames_gaze_partial": partial_frames,
        "frames_rejected": total_frames - valid_frames,
        "valid_gaze_availability": valid_frames / total_frames if total_frames else None,
        "valid_gaze_availability_when_tracked": (
            valid_frames / tracking_frames if tracking_frames else None
        ),
        "gaze_validity_counts": {state.value: validity_counts[state] for state in GazeValidity},
        "rejection_reason_counts": dict(
            sorted(
                Counter(
                    item.validity_reason
                    for item in observations
                    if item.gaze_validity not in (GazeValidity.VALID, GazeValidity.PARTIAL)
                ).items()
            )
        ),
        "left_eye_state_counts": _enum_counts(observations, "left_eye_state", EyeState),
        "right_eye_state_counts": _enum_counts(observations, "right_eye_state", EyeState),
        "possible_blink_events": blink_events,
        "phase_counts": _enum_counts(observations, "phase", ExperimentPhase),
        "measurement_stage_counts": _enum_counts(observations, "stage", MeasurementStage),
        "target_statistics": target_statistics,
        "block_target_statistics": block_target_statistics,
        "phase_target_statistics": phase_target_statistics,
        "separability": _separability(discrimination_records, config),
    }


def write_guided_report(report: dict[str, Any], path: Path) -> None:
    """Serialize and write a guided experiment report dictionary to a JSON file.

    Args:
        report: Validated report dictionary conforming to Schema 5.
        path: Destination filesystem path.

    Side Effects:
        Creates parent directories if necessary and writes UTF-8 encoded JSON.
    """
    serialized = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def _guided_configuration(config: GuidedValidationConfig) -> dict[str, Any]:
    raw = asdict(config)
    raw["targets"] = [
        {
            "name": target.name.value,
            "normalized_x": target.normalized_x,
            "normalized_y": target.normalized_y,
        }
        for target in config.targets
    ]
    return raw


def _observation_dict(item: GuidedObservation) -> dict[str, Any]:
    return {
        "relative_timestamp_seconds": item.relative_timestamp_seconds,
        "phase": item.phase.value,
        "measurement_stage": item.stage.value,
        "target": (
            {
                "name": item.target.name.value,
                "normalized_x": item.target.normalized_x,
                "normalized_y": item.target.normalized_y,
            }
            if item.target is not None
            else None
        ),
        "tracking_state": item.tracking_state.value,
        "gaze_validity": item.gaze_validity.value,
        "left_eye_state": item.left_eye_state.value,
        "right_eye_state": item.right_eye_state.value,
        "block_index": item.block_index,
        "segment_index": item.segment_index,
        "validity_reason": item.validity_reason,
        "features": dict(item.features),
        "head_pose": asdict(item.head_pose) if item.head_pose is not None else None,
        "capture_latency_ms": item.capture_latency_ms,
        "feature_latency_ms": item.feature_latency_ms,
        "loop_latency_ms": item.loop_latency_ms,
        "render_latency_ms": item.render_latency_ms,
        "display_latency_ms": item.display_latency_ms,
        "iteration_latency_ms": item.iteration_latency_ms,
    }


def _statistics_for_records(records: list[GuidedObservation], min_samples: int) -> dict[str, Any]:
    if len(records) < min_samples:
        return {
            "status": "INSUFFICIENT_SAMPLES",
            "sample_count": len(records),
            "minimum_required": min_samples,
            "features": None,
        }
    return {
        "status": "AVAILABLE",
        "sample_count": len(records),
        "minimum_required": min_samples,
        "features": {
            name: _numeric_statistics(
                [value for item in records if (value := _value(item, name)) is not None]
            )
            for name in _ANALYZED_FEATURES
        },
    }


def _numeric_statistics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "population_stddev": None,
            "median_absolute_deviation": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    center = float(np.median(array))
    return {
        "count": len(values),
        "mean": fmean(values),
        "median": center,
        "population_stddev": pstdev(values) if len(values) > 1 else 0.0,
        "median_absolute_deviation": float(np.median(np.abs(array - center))),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": min(values),
        "max": max(values),
    }


def _separability(
    records: list[GuidedObservation], config: GuidedValidationConfig
) -> dict[str, Any]:
    comparisons = (
        ("LEFT_vs_CENTER", TargetName.LEFT, "left_iris_x"),
        ("LEFT_vs_CENTER_right_eye", TargetName.LEFT, "right_iris_x"),
        ("RIGHT_vs_CENTER", TargetName.RIGHT, "left_iris_x"),
        ("RIGHT_vs_CENTER_right_eye", TargetName.RIGHT, "right_iris_x"),
        ("UP_vs_CENTER", TargetName.UP, "left_iris_y"),
        ("UP_vs_CENTER_right_eye", TargetName.UP, "right_iris_y"),
        ("DOWN_vs_CENTER", TargetName.DOWN, "left_iris_y"),
        ("DOWN_vs_CENTER_right_eye", TargetName.DOWN, "right_iris_y"),
    )
    return {
        name: _comparison(records, target, feature, config.min_samples_per_target)
        for name, target, feature in comparisons
    }


def _comparison(
    records: list[GuidedObservation],
    target: TargetName,
    feature: str,
    min_samples: int,
) -> dict[str, Any]:
    center_values = _target_values(records, TargetName.CENTER, feature)
    target_values = _target_values(records, target, feature)
    if len(center_values) < min_samples or len(target_values) < min_samples:
        return {
            "status": "INSUFFICIENT_SAMPLES",
            "feature": feature,
            "target": target.value,
            "center_sample_count": len(center_values),
            "target_sample_count": len(target_values),
            "minimum_required_per_group": min_samples,
            "target_minus_center_median": None,
            "absolute_median_delta_over_sum_mad": None,
        }
    center_median = median(center_values)
    target_median = median(target_values)
    center_mad = _mad(center_values)
    target_mad = _mad(target_values)
    scale = center_mad + target_mad
    delta = target_median - center_median
    return {
        "status": "AVAILABLE",
        "feature": feature,
        "target": target.value,
        "center_sample_count": len(center_values),
        "target_sample_count": len(target_values),
        "minimum_required_per_group": min_samples,
        "target_minus_center_median": delta,
        "absolute_median_delta_over_sum_mad": (abs(delta) / scale if scale > 0.0 else None),
    }


def _target_values(
    records: list[GuidedObservation], target: TargetName, feature: str
) -> list[float]:
    return [
        value
        for item in records
        if item.target is not None
        and item.target.name is target
        and (value := _value(item, feature)) is not None
    ]


def _value(item: GuidedObservation, feature: str) -> float | None:
    if feature.startswith("left_iris_") and item.left_eye_state is not EyeState.OPEN:
        return None
    if feature.startswith("right_iris_") and item.right_eye_state is not EyeState.OPEN:
        return None
    if feature in item.features:
        value = float(item.features[feature])
        return value if isfinite(value) else None
    if item.head_pose is None:
        return None
    pose_values = {
        "head_yaw_degrees": item.head_pose.yaw_degrees,
        "head_pitch_degrees": item.head_pose.pitch_degrees,
        "head_roll_degrees": item.head_pose.roll_degrees,
    }
    pose_value = pose_values.get(feature)
    return pose_value if pose_value is not None and isfinite(pose_value) else None


def _mad(values: list[float]) -> float:
    center = median(values)
    return median([abs(value - center) for value in values])


def _enum_counts(
    observations: list[GuidedObservation], attribute: str, enum_type: Any
) -> dict[str, int]:
    counts = Counter(getattr(item, attribute) for item in observations)
    return {member.value: counts[member] for member in enum_type}


__all__ = [
    "GuidedReportBuilder",
    "summarize_guided_observations",
    "write_guided_report",
]
