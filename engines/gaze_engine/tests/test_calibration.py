from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mekasa_gaze.core.config import CalibrationConfig
from mekasa_gaze.core.domain import TargetName
from mekasa_gaze.core.eye_mode import EyeMode
from mekasa_gaze.tracking.calibration import CalibrationModel, calibrate_from_report


def _config() -> CalibrationConfig:
    return CalibrationConfig(
        ridge_alpha=0.0001,
        validation_block_index=3,
        min_train_samples_per_target=2,
        min_validation_samples_per_target=2,
        min_head_movement_train_samples=2,
        min_head_movement_validation_samples=2,
        head_movement_training_weight=1,
        min_accepted_face_width=0.30,
        max_accepted_face_width=0.45,
        require_audio_prompts=True,
        live_smoothing_alpha=0.25,
        feature_names=("left_iris_x", "right_iris_y"),
    )


def _report() -> dict[str, object]:
    positions = {
        TargetName.CENTER: (0.5, 0.5),
        TargetName.LEFT: (0.15, 0.5),
        TargetName.RIGHT: (0.85, 0.5),
        TargetName.UP: (0.5, 0.15),
        TargetName.DOWN: (0.5, 0.85),
        TargetName.TOP_LEFT: (0.25, 0.25),
        TargetName.TOP_RIGHT: (0.75, 0.25),
        TargetName.BOTTOM_LEFT: (0.25, 0.75),
        TargetName.BOTTOM_RIGHT: (0.75, 0.75),
    }
    observations: list[dict[str, object]] = []
    for block in (1, 2, 3):
        for target, (x, y) in positions.items():
            for repeat in range(3):
                offset = (repeat - 1) * 0.001
                observations.append(
                    {
                        "phase": "GAZE_DISCRIMINATION",
                        "measurement_stage": "MEASUREMENT",
                        "gaze_validity": "GAZE_VALID",
                        "left_eye_state": "OPEN",
                        "right_eye_state": "OPEN",
                        "block_index": block,
                        "target": {
                            "name": target.value,
                            "normalized_x": x,
                            "normalized_y": y,
                        },
                        "features": {
                            "left_iris_x": x + offset,
                            "right_iris_y": y - offset,
                            "face_width": 0.35,
                        },
                        "head_pose": None,
                    }
                )
        for repeat in range(3):
            offset = (repeat - 1) * 0.001
            observations.append(
                {
                    "phase": "HEAD_MOVEMENT",
                    "measurement_stage": "MEASUREMENT",
                    "gaze_validity": "GAZE_VALID",
                    "left_eye_state": "OPEN",
                    "right_eye_state": "OPEN",
                    "block_index": block,
                    "target": {
                        "name": TargetName.CENTER.value,
                        "normalized_x": 0.5,
                        "normalized_y": 0.5,
                    },
                    "features": {
                        "left_iris_x": 0.5 + offset,
                        "right_iris_y": 0.5 - offset,
                        "face_width": 0.35,
                    },
                    "head_pose": None,
                }
            )
    return {
        "schema_version": 5,
        "experiment": {
            "audio": {"active": True},
            "source": {"commit": "test"},
        },
        "protocol": {
            "completed": True,
            "aborted_by_operator": False,
            "configuration": {
                "canvas_width": 1280,
                "canvas_height": 720,
                "debug_panel_fraction": 0.3,
                "min_face_width": 0.30,
                "max_face_width": 0.45,
                "targets": [
                    {"name": target.value, "normalized_x": x, "normalized_y": y}
                    for target, (x, y) in positions.items()
                ],
            },
        },
        "observations": observations,
    }


def test_ridge_calibration_holds_out_complete_block() -> None:
    artifact = calibrate_from_report(_report(), _config(), report_sha256="abc")

    assert artifact["split"]["training_blocks"] == [1, 2]
    assert artifact["split"]["validation_block"] == 3
    metrics = artifact["metrics"]["held_out_validation"]
    assert metrics["closest_target_hit_rate"] == 1.0
    assert metrics["normalized_error"]["median"] < 0.002
    assert artifact["metrics"]["held_out_head_movement_center"]["sample_count"] == 3
    assert artifact["split"]["head_movement_training_sample_count"] == 6
    assert artifact["split"]["head_movement_validation_sample_count"] == 3
    model = CalibrationModel.from_artifact(artifact)
    assert set(model.targets) == set(TargetName)


def test_calibration_accepts_one_usable_eye_and_imputes_the_other() -> None:
    report = _report()
    observations = report["observations"]
    assert isinstance(observations, list)
    partial = observations[0]
    assert isinstance(partial, dict)
    partial["gaze_validity"] = "GAZE_PARTIAL"
    partial["left_eye_state"] = "CLOSED"

    artifact = calibrate_from_report(report, _config(), report_sha256="abc")

    assert artifact["split"]["training_sample_counts"]["CENTER"] == 6
    assert (
        artifact["preprocessing"]["closed_or_unusable_eye_iris_imputation"]
        == "training_feature_mean"
    )


def test_single_eye_mode_uses_only_that_eye_and_frames_where_it_is_open() -> None:
    report = _report()
    observations = report["observations"]
    assert isinstance(observations, list)
    left_closed, right_closed = observations[0], observations[1]
    assert isinstance(left_closed, dict) and isinstance(right_closed, dict)
    left_closed.update(gaze_validity="GAZE_PARTIAL", left_eye_state="CLOSED")
    right_closed.update(gaze_validity="GAZE_PARTIAL", right_eye_state="CLOSED")

    left = calibrate_from_report(report, _config(), eye_mode=EyeMode.LEFT)
    both = calibrate_from_report(report, _config(), eye_mode=EyeMode.BOTH)
    legacy = calibrate_from_report(report, _config())

    assert left["feature_names"] == ["left_iris_x"]
    assert left["eye_mode"] == "LEFT"
    assert left["split"]["training_sample_counts"]["CENTER"] == 5
    assert both["eye_mode"] == "BOTH"
    assert both["split"]["training_sample_counts"]["CENTER"] == 4
    assert legacy["split"]["training_sample_counts"]["CENTER"] == 6
    assert CalibrationModel.from_artifact(left).eye_mode is EyeMode.LEFT


def test_calibration_excludes_observations_outside_configured_acceptance_band() -> None:
    report = _report()
    observations = report["observations"]
    assert isinstance(observations, list)
    outside = observations[0]
    assert isinstance(outside, dict)
    features = outside["features"]
    assert isinstance(features, dict)
    features["face_width"] = 0.20

    artifact = calibrate_from_report(report, _config(), report_sha256="abc")

    assert artifact["split"]["training_sample_counts"]["CENTER"] == 5


def test_calibration_can_reanalyze_report_with_explicit_acceptance_band() -> None:
    report = _report()
    observations = report["observations"]
    assert isinstance(observations, list)
    sample = observations[0]
    assert isinstance(sample, dict)
    features = sample["features"]
    assert isinstance(features, dict)
    features["face_width"] = 0.27

    artifact = calibrate_from_report(
        report,
        replace(_config(), min_accepted_face_width=0.25),
    )

    assert artifact["split"]["training_sample_counts"]["CENTER"] == 6
    assert artifact["preprocessing"]["accepted_face_width"]["min"] == 0.25


def test_calibration_rejects_legacy_single_block_report() -> None:
    report = _report()
    report["schema_version"] = 4

    try:
        calibrate_from_report(report, _config())
    except ValueError as error:
        assert "schema 5" in str(error)
    else:
        raise AssertionError("legacy report should not be accepted for calibration")


def test_calibration_config_uses_head_robust_features_and_thresholds() -> None:
    config = CalibrationConfig.load(Path("config/experiment-002.toml"))

    assert config.min_head_movement_train_samples == 40
    assert config.min_head_movement_validation_samples == 20
    assert config.head_movement_training_weight == 8
    assert config.min_accepted_face_width == pytest.approx(0.25)
    assert "head_yaw_degrees" in config.feature_names
    assert "face_center_x" not in config.feature_names


def test_calibration_model_rejects_pre_head_movement_artifact() -> None:
    artifact = calibrate_from_report(_report(), _config())
    artifact["schema_version"] = 1

    with pytest.raises(ValueError, match="predates blocked head-movement"):
        CalibrationModel.from_artifact(artifact)


def test_ear_reanalysis_reopens_downward_gaze_but_keeps_blinks_excluded() -> None:
    report = _report()
    protocol = report["protocol"]
    assert isinstance(protocol, dict)
    protocol["configuration"].update(
        eye_closed_ear_threshold=0.15, min_eye_vertical_span=0.008, max_abs_iris_coordinate=3.0
    )
    observations = report["observations"]
    assert isinstance(observations, list)
    for item in observations:
        item["features"].update(
            left_eye_aspect_ratio=0.12,
            right_eye_aspect_ratio=0.12,
            left_eye_vertical_span=0.01,
            right_eye_vertical_span=0.01,
            left_iris_y=0.3,
            right_iris_x=0.5,
        )
        item.update(gaze_validity="EYE_CLOSED", left_eye_state="CLOSED", right_eye_state="CLOSED")
    observations[0]["gaze_validity"] = "POSSIBLE_BLINK"

    with pytest.raises(ValueError):
        calibrate_from_report(report, _config())
    artifact = calibrate_from_report(report, _config(), reclassify_ear_threshold=0.09)

    reanalysis = artifact["eye_state_reanalysis"]
    assert reanalysis["reanalysis_ear_threshold"] == 0.09
    assert reanalysis["validity_changes"] == {"EYE_CLOSED->GAZE_VALID": len(observations) - 1}
    assert artifact["split"]["training_sample_counts"]["CENTER"] == 5
