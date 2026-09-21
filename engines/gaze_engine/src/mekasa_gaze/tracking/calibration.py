from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.config import CalibrationConfig
from mekasa_gaze.core.domain import EyeState, FeatureObservation, TargetName
from mekasa_gaze.services.runtime import environment, source_revision


@dataclass(frozen=True, slots=True)
class CalibrationModel:
    feature_names: tuple[str, ...]
    feature_mean: npt.NDArray[np.float64]
    feature_scale: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    output_intercept: npt.NDArray[np.float64]
    targets: dict[TargetName, tuple[float, float]]

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> CalibrationModel:
        if artifact.get("kind") != "PERSONAL_GAZE_RIDGE_EXPERIMENTAL":
            raise ValueError("Unsupported calibration model kind")
        if int(artifact.get("schema_version", 0)) < 2:
            raise ValueError(
                "Calibration model predates blocked head-movement validation; "
                "run guided validation and calibration again"
            )
        feature_names = tuple(str(name) for name in artifact["feature_names"])
        preprocessing = artifact["preprocessing"]
        model = artifact["model"]
        feature_mean = np.asarray(preprocessing["feature_mean"], dtype=np.float64)
        feature_scale = np.asarray(preprocessing["feature_scale"], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
        output_intercept = np.asarray(model["output_intercept"], dtype=np.float64)
        if feature_mean.shape != (len(feature_names),):
            raise ValueError("Calibration feature mean has an invalid shape")
        if feature_scale.shape != (len(feature_names),):
            raise ValueError("Calibration feature scale has an invalid shape")
        if weights.shape != (len(feature_names), 2) or output_intercept.shape != (2,):
            raise ValueError("Calibration regression parameters have invalid shapes")
        if not all(
            np.all(np.isfinite(array))
            for array in (feature_mean, feature_scale, weights, output_intercept)
        ):
            raise ValueError("Calibration model contains non-finite values")
        if np.any(feature_scale <= 0.0):
            raise ValueError("Calibration feature scales must be positive")
        targets = {
            TargetName(item["name"]): (
                float(item["normalized_x"]),
                float(item["normalized_y"]),
            )
            for item in artifact["targets"]
        }
        if set(targets) != set(TargetName):
            raise ValueError("Calibration model does not contain all required targets")
        return cls(
            feature_names,
            feature_mean,
            feature_scale,
            weights,
            output_intercept,
            targets,
        )

    def predict_observation(
        self,
        observation: FeatureObservation,
        *,
        left_eye_state: EyeState = EyeState.OPEN,
        right_eye_state: EyeState = EyeState.OPEN,
    ) -> tuple[float, float]:
        vector = observation_vector(
            observation,
            self.feature_names,
            left_eye_state=left_eye_state,
            right_eye_state=right_eye_state,
        )
        imputed = np.where(np.isfinite(vector), vector, self.feature_mean)
        prediction = ((imputed - self.feature_mean) / self.feature_scale) @ self.weights
        prediction += self.output_intercept
        return float(prediction[0]), float(prediction[1])

    def nearest_target(self, prediction: tuple[float, float]) -> TargetName:
        return min(
            self.targets,
            key=lambda name: (
                (prediction[0] - self.targets[name][0]) ** 2
                + (prediction[1] - self.targets[name][1]) ** 2
            ),
        )


def calibrate_from_report(
    report: dict[str, Any],
    config: CalibrationConfig,
    *,
    report_sha256: str | None = None,
) -> dict[str, Any]:
    _validate_report(report, config)
    gaze_records = _calibration_records(
        report,
        config.feature_names,
        phase="GAZE_DISCRIMINATION",
        min_face_width=config.min_accepted_face_width,
        max_face_width=config.max_accepted_face_width,
    )
    head_movement_records = _calibration_records(
        report,
        config.feature_names,
        phase="HEAD_MOVEMENT",
        min_face_width=config.min_accepted_face_width,
        max_face_width=config.max_accepted_face_width,
    )
    if any(record[1] is not TargetName.CENTER for record in head_movement_records):
        raise ValueError("Head-movement calibration records must use the CENTER target")
    validation_block = config.validation_block_index
    gaze_training = [record for record in gaze_records if record[0] != validation_block]
    validation = [record for record in gaze_records if record[0] == validation_block]
    head_movement_training = [
        record for record in head_movement_records if record[0] != validation_block
    ]
    head_movement_validation = [
        record for record in head_movement_records if record[0] == validation_block
    ]
    _validate_sample_counts(
        gaze_training,
        config.min_train_samples_per_target,
        "training",
    )
    _validate_sample_counts(
        validation,
        config.min_validation_samples_per_target,
        "validation",
    )
    _validate_record_count(
        head_movement_training,
        config.min_head_movement_train_samples,
        "head-movement training",
    )
    _validate_record_count(
        head_movement_validation,
        config.min_head_movement_validation_samples,
        "head-movement validation",
    )

    training = [
        *gaze_training,
        *(head_movement_training * config.head_movement_training_weight),
    ]
    train_x, train_y, train_targets = _arrays(training)
    validation_x, validation_y, validation_targets = _arrays(validation)
    head_validation_x, head_validation_y, head_validation_targets = _arrays(
        head_movement_validation
    )
    if np.any(np.all(~np.isfinite(train_x), axis=0)):
        raise ValueError("A calibration feature is unavailable in every training sample")
    feature_mean = np.nanmean(train_x, axis=0)
    imputed_train_x = np.where(np.isfinite(train_x), train_x, feature_mean)
    imputed_validation_x = np.where(np.isfinite(validation_x), validation_x, feature_mean)
    imputed_head_validation_x = np.where(
        np.isfinite(head_validation_x), head_validation_x, feature_mean
    )
    feature_scale = np.std(imputed_train_x, axis=0)
    feature_scale = np.where(feature_scale > 1e-9, feature_scale, 1.0)
    standardized = (imputed_train_x - feature_mean) / feature_scale
    output_intercept = np.mean(train_y, axis=0)
    centered_y = train_y - output_intercept
    regularized_gram = standardized.T @ standardized + config.ridge_alpha * np.eye(
        standardized.shape[1]
    )
    weights = np.linalg.solve(regularized_gram, standardized.T @ centered_y)

    targets = _targets_from_report(report)
    train_predictions = standardized @ weights + output_intercept
    validation_predictions = ((imputed_validation_x - feature_mean) / feature_scale) @ weights
    validation_predictions += output_intercept
    head_validation_predictions = (
        (imputed_head_validation_x - feature_mean) / feature_scale
    ) @ weights
    head_validation_predictions += output_intercept
    canvas_width, canvas_height = _test_area_dimensions(report)

    artifact: dict[str, Any] = {
        "schema_version": 2,
        "kind": "PERSONAL_GAZE_RIDGE_EXPERIMENTAL",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": source_revision(),
        "environment": environment(),
        "training_report": {
            "sha256": report_sha256,
            "source_commit": report.get("experiment", {}).get("source", {}).get("commit"),
            "schema_version": report.get("schema_version"),
        },
        "feature_names": list(config.feature_names),
        "preprocessing": {
            "feature_mean": feature_mean.tolist(),
            "feature_scale": feature_scale.tolist(),
            "closed_or_unusable_eye_iris_imputation": "training_feature_mean",
            "accepted_face_width": {
                "min": config.min_accepted_face_width,
                "max": config.max_accepted_face_width,
                "source": "calibration_configuration",
            },
        },
        "model": {
            "type": "ridge_regression_closed_form",
            "ridge_alpha": config.ridge_alpha,
            "head_movement_training_weight": config.head_movement_training_weight,
            "weights": weights.tolist(),
            "output_intercept": output_intercept.tolist(),
        },
        "targets": [
            {"name": name.value, "normalized_x": point[0], "normalized_y": point[1]}
            for name, point in targets.items()
        ],
        "split": {
            "strategy": "held_out_complete_protocol_block_with_head_movement_center",
            "training_blocks": sorted({record[0] for record in gaze_training}),
            "validation_block": validation_block,
            "training_sample_counts": _sample_counts(gaze_training),
            "validation_sample_counts": _sample_counts(validation),
            "head_movement_training_sample_count": len(head_movement_training),
            "head_movement_effective_training_sample_count": (
                len(head_movement_training) * config.head_movement_training_weight
            ),
            "head_movement_validation_sample_count": len(head_movement_validation),
        },
        "metrics": {
            "training": _prediction_metrics(
                train_y,
                train_predictions,
                train_targets,
                targets,
                canvas_width,
                canvas_height,
            ),
            "held_out_validation": _prediction_metrics(
                validation_y,
                validation_predictions,
                validation_targets,
                targets,
                canvas_width,
                canvas_height,
            ),
            "held_out_head_movement_center": _prediction_metrics(
                head_validation_y,
                head_validation_predictions,
                head_validation_targets,
                targets,
                canvas_width,
                canvas_height,
            ),
        },
        "privacy": {
            "video_persisted": False,
            "frames_persisted": False,
            "raw_landmarks_persisted": False,
            "personal_derived_model_persisted": True,
            "intended_storage": "local ignored work directory",
        },
        "limitations": [
            "This is a personal experimental mapping, not a generic gaze model.",
            (
                "Held-out blocks reduce adjacent-frame leakage but do not establish "
                "session-to-session repeatability."
            ),
            (
                "Head-movement validation measures only maintaining gaze on CENTER while "
                "moving naturally; it does not validate every gaze target at every pose."
            ),
            "Displayed targets are prompts and do not prove fixation.",
            (
                "When only one eye is usable, iris features from the other eye are replaced "
                "by training-set means rather than treated as observed gaze evidence."
            ),
            (
                "Observations outside the explicit calibration face-size acceptance band "
                "are excluded; that band may differ from the preserved source protocol."
            ),
            "Closest-target hit rate is technical validation, not clinical validation.",
            (
                "The model must not control a cursor, dwell selection, AAC, "
                "or safety-critical actions."
            ),
        ],
    }
    json.dumps(artifact, allow_nan=False)
    return artifact


def observation_vector(
    observation: FeatureObservation,
    feature_names: tuple[str, ...],
    *,
    left_eye_state: EyeState = EyeState.OPEN,
    right_eye_state: EyeState = EyeState.OPEN,
) -> npt.NDArray[np.float64]:
    values: list[float] = []
    for name in feature_names:
        if name.startswith("left_iris_") and left_eye_state is not EyeState.OPEN:
            values.append(float("nan"))
            continue
        if name.startswith("right_iris_") and right_eye_state is not EyeState.OPEN:
            values.append(float("nan"))
            continue
        value = _observation_value(observation, name)
        if value is None or not isfinite(value):
            raise ValueError(f"Observation is missing finite calibration feature {name}")
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def load_calibration_model(path: Path) -> CalibrationModel:
    with path.open(encoding="utf-8") as model_file:
        artifact: dict[str, Any] = json.load(model_file)
    return CalibrationModel.from_artifact(artifact)


def write_calibration_artifact(artifact: dict[str, Any], path: Path) -> None:
    serialized = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def _validate_report(report: dict[str, Any], config: CalibrationConfig) -> None:
    if int(report.get("schema_version", 0)) < 5:
        raise ValueError(
            "Calibration requires a schema 5 guided report with repeated randomized "
            "nine-target and head-movement blocks"
        )
    protocol = report.get("protocol", {})
    if not protocol.get("completed") or protocol.get("aborted_by_operator"):
        raise ValueError("Calibration requires a completed, non-aborted guided report")
    audio_active = report.get("experiment", {}).get("audio", {}).get("active") is True
    if config.require_audio_prompts and not audio_active:
        raise ValueError("Calibration configuration requires active audio prompts")


def _calibration_records(
    report: dict[str, Any],
    feature_names: tuple[str, ...],
    *,
    phase: str,
    min_face_width: float,
    max_face_width: float,
) -> list[tuple[int, TargetName, npt.NDArray[np.float64], tuple[float, float]]]:
    records: list[tuple[int, TargetName, npt.NDArray[np.float64], tuple[float, float]]] = []
    for item in report["observations"]:
        if (
            item["phase"] != phase
            or item["measurement_stage"] != "MEASUREMENT"
            or item.get("block_index") is None
            or item.get("target") is None
            or not _report_observation_has_usable_eye(item)
            or not _report_observation_has_valid_placement(
                item,
                min_face_width=min_face_width,
                max_face_width=max_face_width,
            )
        ):
            continue
        values = [_report_value(item, name) for name in feature_names]
        if any(value is None or (not isfinite(value) and not np.isnan(value)) for value in values):
            continue
        target = item["target"]
        records.append(
            (
                int(item["block_index"]),
                TargetName(target["name"]),
                np.asarray(values, dtype=np.float64),
                (float(target["normalized_x"]), float(target["normalized_y"])),
            )
        )
    if not records:
        raise ValueError(f"Guided report has no usable {phase} calibration observations")
    return records


def _report_value(item: dict[str, Any], name: str) -> float | None:
    if name.startswith("left_iris_") and item["left_eye_state"] != EyeState.OPEN.value:
        return float("nan")
    if name.startswith("right_iris_") and item["right_eye_state"] != EyeState.OPEN.value:
        return float("nan")
    if name in item["features"]:
        return float(item["features"][name])
    pose = item.get("head_pose")
    pose_key = {
        "head_yaw_degrees": "yaw_degrees",
        "head_pitch_degrees": "pitch_degrees",
        "head_roll_degrees": "roll_degrees",
    }.get(name)
    return float(pose[pose_key]) if pose is not None and pose_key is not None else None


def _report_observation_has_usable_eye(item: dict[str, Any]) -> bool:
    if item["gaze_validity"] not in ("GAZE_VALID", "GAZE_PARTIAL", "EYE_CLOSED"):
        return False
    states = (item["left_eye_state"], item["right_eye_state"])
    return EyeState.OPEN.value in states


def _report_observation_has_valid_placement(
    item: dict[str, Any],
    *,
    min_face_width: object,
    max_face_width: object,
) -> bool:
    if min_face_width is None or max_face_width is None:
        return True
    if not isinstance(min_face_width, int | float) or not isinstance(max_face_width, int | float):
        return False
    face_width = item["features"].get("face_width")
    if not isinstance(face_width, int | float):
        return False
    return float(min_face_width) <= float(face_width) <= float(max_face_width)


def _observation_value(observation: FeatureObservation, name: str) -> float | None:
    if name in observation.features:
        return float(observation.features[name])
    if observation.head_pose is None:
        return None
    return {
        "head_yaw_degrees": observation.head_pose.yaw_degrees,
        "head_pitch_degrees": observation.head_pose.pitch_degrees,
        "head_roll_degrees": observation.head_pose.roll_degrees,
    }.get(name)


def _validate_sample_counts(
    records: list[tuple[int, TargetName, npt.NDArray[np.float64], tuple[float, float]]],
    minimum: int,
    split_name: str,
) -> None:
    counts = Counter(record[1] for record in records)
    missing = {target.value: counts[target] for target in TargetName if counts[target] < minimum}
    if missing:
        raise ValueError(
            f"Insufficient {split_name} samples per target; need {minimum}, observed {missing}"
        )


def _validate_record_count(
    records: list[tuple[int, TargetName, npt.NDArray[np.float64], tuple[float, float]]],
    minimum: int,
    split_name: str,
) -> None:
    if len(records) < minimum:
        raise ValueError(
            f"Insufficient {split_name} samples; need {minimum}, observed {len(records)}"
        )


def _arrays(
    records: list[tuple[int, TargetName, npt.NDArray[np.float64], tuple[float, float]]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[TargetName]]:
    return (
        np.vstack([record[2] for record in records]),
        np.asarray([record[3] for record in records], dtype=np.float64),
        [record[1] for record in records],
    )


def _targets_from_report(report: dict[str, Any]) -> dict[TargetName, tuple[float, float]]:
    targets = {
        TargetName(item["name"]): (float(item["normalized_x"]), float(item["normalized_y"]))
        for item in report["protocol"]["configuration"]["targets"]
    }
    if set(targets) != set(TargetName):
        raise ValueError("Guided report does not contain all required targets")
    return targets


def _test_area_dimensions(report: dict[str, Any]) -> tuple[float, float]:
    rendering = report["protocol"]["configuration"]
    width = float(rendering["canvas_width"]) * (1.0 - float(rendering["debug_panel_fraction"]))
    return width, float(rendering["canvas_height"])


def _prediction_metrics(
    expected: npt.NDArray[np.float64],
    predicted: npt.NDArray[np.float64],
    expected_targets: list[TargetName],
    targets: dict[TargetName, tuple[float, float]],
    width: float,
    height: float,
) -> dict[str, Any]:
    normalized_errors = np.linalg.norm(predicted - expected, axis=1)
    pixel_delta = (predicted - expected) * np.asarray([width, height])
    pixel_errors = np.linalg.norm(pixel_delta, axis=1)
    nearest = [_nearest_target(tuple(point), targets, width, height) for point in predicted]
    hit_rate = sum(
        actual is estimated for actual, estimated in zip(expected_targets, nearest, strict=True)
    ) / len(expected_targets)
    per_target = {
        target.value: _error_summary(
            normalized_errors[
                np.asarray([name is target for name in expected_targets], dtype=np.bool_)
            ],
            pixel_errors[np.asarray([name is target for name in expected_targets], dtype=np.bool_)],
            [
                estimated
                for actual, estimated in zip(expected_targets, nearest, strict=True)
                if actual is target
            ],
            target,
        )
        for target in TargetName
    }
    return {
        **_error_summary(normalized_errors, pixel_errors, nearest, expected_targets),
        "closest_target_hit_rate": hit_rate,
        "predictions_outside_normalized_screen_fraction": float(
            np.mean(np.any((predicted < 0.0) | (predicted > 1.0), axis=1))
        ),
        "per_target": per_target,
    }


def _error_summary(
    normalized_errors: npt.NDArray[np.float64],
    pixel_errors: npt.NDArray[np.float64],
    predicted_targets: list[TargetName],
    expected_target: TargetName | list[TargetName],
) -> dict[str, Any]:
    if isinstance(expected_target, list):
        target_hits = sum(
            actual is predicted
            for actual, predicted in zip(expected_target, predicted_targets, strict=True)
        )
    else:
        target_hits = sum(predicted is expected_target for predicted in predicted_targets)
    count = len(normalized_errors)
    return {
        "sample_count": count,
        "normalized_error": _array_summary(normalized_errors),
        "pixel_error": _array_summary(pixel_errors),
        "closest_target_hit_rate": target_hits / count if count else None,
    }


def _array_summary(values: npt.NDArray[np.float64]) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _nearest_target(
    point: tuple[float, float],
    targets: dict[TargetName, tuple[float, float]],
    width: float,
    height: float,
) -> TargetName:
    return min(
        targets,
        key=lambda name: (
            ((point[0] - targets[name][0]) * width) ** 2
            + ((point[1] - targets[name][1]) * height) ** 2
        ),
    )


def _sample_counts(
    records: list[tuple[int, TargetName, npt.NDArray[np.float64], tuple[float, float]]],
) -> dict[str, int]:
    counts = Counter(record[1] for record in records)
    return {target.value: counts[target] for target in TargetName}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate an experimental personal gaze calibration"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "research/experiments/001-eye-feature-extraction/results/guided-validation-result.json"
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("config/experiment-002.toml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/calibration/personal-gaze-model.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with args.report.open(encoding="utf-8") as report_file:
            report: dict[str, Any] = json.load(report_file)
        artifact = calibrate_from_report(
            report,
            CalibrationConfig.load(args.config),
            report_sha256=_sha256(args.report),
        )
        write_calibration_artifact(artifact, args.output)
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Calibration could not run: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "model": str(args.output),
                "split": artifact["split"],
                "metrics": artifact["metrics"],
                "warning": "Experimental technical result; not clinical validation.",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
