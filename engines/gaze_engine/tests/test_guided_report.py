from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mekasa_gaze.core.config import GuidedValidationConfig
from mekasa_gaze.core.domain import (
    ExperimentPhase,
    FeatureObservation,
    GazeValidity,
    GuidedObservation,
    HeadPose,
    MeasurementStage,
    ProtocolSnapshot,
    TargetName,
    TrackingState,
)
from mekasa_gaze.services.reporting import (
    GuidedReportBuilder,
    summarize_guided_observations,
    write_guided_report,
)
from mekasa_gaze.tracking.camera import CameraFrame, CameraSettings
from mekasa_gaze.tracking.validity import assess_gaze_validity


def _config() -> GuidedValidationConfig:
    return replace(
        GuidedValidationConfig.load(Path("config/experiment-001.toml")),
        min_samples_per_target=2,
    )


def _observation(
    target_name: TargetName,
    iris_value: float,
    *,
    phase: ExperimentPhase = ExperimentPhase.GAZE_DISCRIMINATION,
) -> GuidedObservation:
    config = _config()
    return GuidedObservation(
        relative_timestamp_seconds=iris_value,
        phase=phase,
        stage=MeasurementStage.MEASUREMENT,
        target=config.target(target_name),
        tracking_state=TrackingState.GOOD,
        gaze_validity=GazeValidity.VALID,
        left_eye_state=assess_gaze_validity(_feature(), config).left_eye_state,
        right_eye_state=assess_gaze_validity(_feature(), config).right_eye_state,
        features={
            "left_iris_x": iris_value,
            "left_iris_y": iris_value,
            "right_iris_x": iris_value,
            "right_iris_y": iris_value,
        },
        head_pose=HeadPose(iris_value, iris_value * 2, iris_value * 3),
    )


def _feature() -> FeatureObservation:
    return FeatureObservation(
        frame_sequence=1,
        timestamp_ns=1,
        tracking_state=TrackingState.GOOD,
        features={
            "left_eye_aspect_ratio": 0.3,
            "right_eye_aspect_ratio": 0.3,
            "left_eye_vertical_span": 0.03,
            "right_eye_vertical_span": 0.03,
            "left_iris_x": 0.5,
            "left_iris_y": 0.5,
            "right_iris_x": 0.5,
            "right_iris_y": 0.5,
        },
        feature_latency_ms=3.0,
    )


def test_target_statistics_and_separation_use_only_valid_measurement_records() -> None:
    observations = [
        _observation(TargetName.CENTER, 0.50),
        _observation(TargetName.CENTER, 0.52),
        _observation(TargetName.LEFT, 0.20),
        _observation(TargetName.LEFT, 0.22),
        replace(
            _observation(TargetName.LEFT, 99.0),
            stage=MeasurementStage.ACQUISITION,
        ),
        replace(
            _observation(TargetName.LEFT, 99.0),
            gaze_validity=GazeValidity.INVALID,
        ),
    ]
    summary = summarize_guided_observations(observations, _config(), blink_events=0)

    assert summary["target_statistics"]["CENTER"]["sample_count"] == 2
    assert summary["target_statistics"]["LEFT"]["sample_count"] == 2
    comparison = summary["separability"]["LEFT_vs_CENTER"]
    assert comparison["status"] == "AVAILABLE"
    assert comparison["target_minus_center_median"] == pytest.approx(-0.3)


def test_insufficient_target_samples_are_explicit() -> None:
    summary = summarize_guided_observations(
        [_observation(TargetName.CENTER, 0.5)], _config(), blink_events=0
    )
    right = summary["target_statistics"]["RIGHT"]
    assert right == {
        "status": "INSUFFICIENT_SAMPLES",
        "sample_count": 0,
        "minimum_required": 2,
        "features": None,
    }
    assert summary["separability"]["RIGHT_vs_CENTER"]["status"] == ("INSUFFICIENT_SAMPLES")


def test_builder_omits_non_finite_features_and_counts_tracking_invalid() -> None:
    config = _config()
    builder = GuidedReportBuilder()
    feature = replace(
        _feature(),
        features={**_feature().features, "left_iris_y": float("nan")},
    )
    assessment = assess_gaze_validity(feature, config)
    frame = CameraFrame(
        sequence=1,
        captured_at_ns=1,
        capture_latency_ms=2.0,
        bgr=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    builder.add(
        frame,
        feature,
        ProtocolSnapshot(
            phase=ExperimentPhase.PREPARATION,
            stage=MeasurementStage.WAITING_TRACKING,
            target=None,
            instruction="test",
            stage_remaining_seconds=None,
            protocol_elapsed_seconds=0.0,
        ),
        assessment,
        loop_latency_ms=5.0,
    )
    report = builder.build(
        camera=CameraSettings(0, "test", 2, 2, 30.0),
        environment={},
        config=config,
        experiment={},
        completed=False,
        aborted=True,
    )
    assert "left_iris_y" not in report["observations"][0]["features"]
    assert report["measurement"]["frames_rejected"] == 1
    json.dumps(report, allow_nan=False)


def test_json_writer_rejects_nan_and_serializes_finite_report(tmp_path: Path) -> None:
    output = tmp_path / "guided.json"
    with pytest.raises(ValueError, match="Out of range float"):
        write_guided_report({"invalid": float("nan")}, output)

    write_guided_report({"valid": 1.25, "missing": None}, output)
    written = output.read_text(encoding="utf-8")
    assert "NaN" not in written
    assert "Infinity" not in written
    assert json.loads(written) == {"missing": None, "valid": 1.25}
