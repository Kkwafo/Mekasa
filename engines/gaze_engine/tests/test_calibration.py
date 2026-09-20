from __future__ import annotations

from mekasa_gaze.calibration import CalibrationModel, calibrate_from_report
from mekasa_gaze.config import CalibrationConfig
from mekasa_gaze.domain import TargetName


def _config() -> CalibrationConfig:
    return CalibrationConfig(
        ridge_alpha=0.0001,
        validation_block_index=3,
        min_train_samples_per_target=2,
        min_validation_samples_per_target=2,
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
                        },
                        "head_pose": None,
                    }
                )
    return {
        "schema_version": 3,
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


def test_calibration_rejects_legacy_single_block_report() -> None:
    report = _report()
    report["schema_version"] = 2

    try:
        calibrate_from_report(report, _config())
    except ValueError as error:
        assert "schema 3" in str(error)
    else:
        raise AssertionError("legacy report should not be accepted for calibration")
