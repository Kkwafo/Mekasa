from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mekasa_gaze.domain import TargetName, VisualTarget


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    camera_index: int
    backend: str
    width: int
    height: int
    target_fps: float
    min_face_detection_confidence: float
    min_face_presence_confidence: float
    min_tracking_confidence: float
    warmup_seconds: float
    measure_seconds: float
    display: bool

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        with path.open("rb") as handle:
            raw: dict[str, Any] = tomllib.load(handle)
        camera = raw["camera"]
        tracking = raw["tracking"]
        experiment = raw["experiment"]
        return cls(
            camera_index=int(camera["index"]),
            backend=str(camera["backend"]),
            width=int(camera["width"]),
            height=int(camera["height"]),
            target_fps=float(camera["target_fps"]),
            min_face_detection_confidence=float(tracking["min_face_detection_confidence"]),
            min_face_presence_confidence=float(tracking["min_face_presence_confidence"]),
            min_tracking_confidence=float(tracking["min_tracking_confidence"]),
            warmup_seconds=float(experiment["warmup_seconds"]),
            measure_seconds=float(experiment["measure_seconds"]),
            display=bool(experiment["display"]),
        )


@dataclass(frozen=True, slots=True)
class GuidedValidationConfig:
    camera_index: int
    backend: str
    width: int
    height: int
    target_fps: float
    min_face_detection_confidence: float
    min_face_presence_confidence: float
    min_tracking_confidence: float
    min_face_width: float
    max_face_width: float
    eye_closed_ear_threshold: float
    min_eye_vertical_span: float
    max_abs_iris_coordinate: float
    max_blink_seconds: float
    tracking_stable_seconds: float
    countdown_seconds: float
    transition_seconds: float
    acquisition_seconds: float
    measurement_seconds: float
    center_stability_seconds: float
    head_movement_seconds: float
    completion_display_seconds: float
    min_samples_per_target: int
    discrimination_blocks: int
    target_order_seed: int
    canvas_width: int
    canvas_height: int
    debug_panel_fraction: float
    target_radius_pixels: int
    fullscreen: bool
    show_face_box: bool
    show_landmarks: bool
    audio_enabled: bool
    audio_required: bool
    audio_language: str
    audio_rate: int
    audio_volume: int
    hide_visual_instructions_when_audio: bool
    targets: tuple[VisualTarget, ...]

    @classmethod
    def load(cls, path: Path) -> GuidedValidationConfig:
        with path.open("rb") as handle:
            raw: dict[str, Any] = tomllib.load(handle)
        camera = raw["camera"]
        tracking = raw["tracking"]
        placement = raw["placement"]
        validity = raw["eye_validity"]
        guided = raw["guided"]
        rendering = raw["rendering"]
        audio = raw["audio"]
        raw_targets = raw["targets"]
        targets = tuple(
            VisualTarget(
                name=name,
                normalized_x=float(raw_targets[name.value]["x"]),
                normalized_y=float(raw_targets[name.value]["y"]),
            )
            for name in TargetName
        )
        config = cls(
            camera_index=int(camera["index"]),
            backend=str(camera["backend"]),
            width=int(camera["width"]),
            height=int(camera["height"]),
            target_fps=float(camera["target_fps"]),
            min_face_detection_confidence=float(tracking["min_face_detection_confidence"]),
            min_face_presence_confidence=float(tracking["min_face_presence_confidence"]),
            min_tracking_confidence=float(tracking["min_tracking_confidence"]),
            min_face_width=float(placement["min_face_width"]),
            max_face_width=float(placement["max_face_width"]),
            eye_closed_ear_threshold=float(validity["closed_ear_threshold"]),
            min_eye_vertical_span=float(validity["min_vertical_span"]),
            max_abs_iris_coordinate=float(validity["max_abs_iris_coordinate"]),
            max_blink_seconds=float(validity["max_blink_seconds"]),
            tracking_stable_seconds=float(guided["tracking_stable_seconds"]),
            countdown_seconds=float(guided["countdown_seconds"]),
            transition_seconds=float(guided["transition_seconds"]),
            acquisition_seconds=float(guided["acquisition_seconds"]),
            measurement_seconds=float(guided["measurement_seconds"]),
            center_stability_seconds=float(guided["center_stability_seconds"]),
            head_movement_seconds=float(guided["head_movement_seconds"]),
            completion_display_seconds=float(guided["completion_display_seconds"]),
            min_samples_per_target=int(guided["min_samples_per_target"]),
            discrimination_blocks=int(guided["discrimination_blocks"]),
            target_order_seed=int(guided["target_order_seed"]),
            canvas_width=int(rendering["canvas_width"]),
            canvas_height=int(rendering["canvas_height"]),
            debug_panel_fraction=float(rendering["debug_panel_fraction"]),
            target_radius_pixels=int(rendering["target_radius_pixels"]),
            fullscreen=bool(rendering["fullscreen"]),
            show_face_box=bool(rendering["show_face_box"]),
            show_landmarks=bool(rendering["show_landmarks"]),
            audio_enabled=bool(audio["enabled"]),
            audio_required=bool(audio["required"]),
            audio_language=str(audio["language"]),
            audio_rate=int(audio["rate"]),
            audio_volume=int(audio["volume"]),
            hide_visual_instructions_when_audio=bool(audio["hide_visual_instructions_when_active"]),
            targets=targets,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.0 < self.debug_panel_fraction < 0.5:
            raise ValueError("debug_panel_fraction must be between 0 and 0.5")
        if not 0.0 < self.min_face_width < self.max_face_width < 1.0:
            raise ValueError("face-width placement bounds must satisfy 0 < min < max < 1")
        if self.min_samples_per_target < 1:
            raise ValueError("min_samples_per_target must be positive")
        if self.discrimination_blocks < 3:
            raise ValueError("discrimination_blocks must be at least 3 for blocked validation")
        durations = (
            self.tracking_stable_seconds,
            self.countdown_seconds,
            self.transition_seconds,
            self.acquisition_seconds,
            self.measurement_seconds,
            self.center_stability_seconds,
            self.head_movement_seconds,
        )
        if any(value <= 0.0 for value in durations):
            raise ValueError("guided phase durations must be positive")
        if not 0.0 < self.eye_closed_ear_threshold < 1.0:
            raise ValueError("closed_ear_threshold must be between 0 and 1")
        if self.min_eye_vertical_span <= 0.0 or self.max_abs_iris_coordinate <= 0.0:
            raise ValueError("eye geometry thresholds must be positive")
        if not -10 <= self.audio_rate <= 10:
            raise ValueError("audio rate must be between -10 and 10")
        if not 0 <= self.audio_volume <= 100:
            raise ValueError("audio volume must be between 0 and 100")
        for target in self.targets:
            if not 0.0 <= target.normalized_x <= 1.0:
                raise ValueError(f"target {target.name} x must be normalized")
            if not 0.0 <= target.normalized_y <= 1.0:
                raise ValueError(f"target {target.name} y must be normalized")

    def target(self, name: TargetName) -> VisualTarget:
        return next(target for target in self.targets if target.name is name)


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    ridge_alpha: float
    validation_block_index: int
    min_train_samples_per_target: int
    min_validation_samples_per_target: int
    min_head_movement_train_samples: int
    min_head_movement_validation_samples: int
    head_movement_training_weight: int
    min_accepted_face_width: float
    max_accepted_face_width: float
    require_audio_prompts: bool
    live_smoothing_alpha: float
    feature_names: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> CalibrationConfig:
        with path.open("rb") as handle:
            raw: dict[str, Any] = tomllib.load(handle)
        calibration = raw["calibration"]
        live = raw["live_evaluation"]
        config = cls(
            ridge_alpha=float(calibration["ridge_alpha"]),
            validation_block_index=int(calibration["validation_block_index"]),
            min_train_samples_per_target=int(calibration["min_train_samples_per_target"]),
            min_validation_samples_per_target=int(calibration["min_validation_samples_per_target"]),
            min_head_movement_train_samples=int(calibration["min_head_movement_train_samples"]),
            min_head_movement_validation_samples=int(
                calibration["min_head_movement_validation_samples"]
            ),
            head_movement_training_weight=int(calibration["head_movement_training_weight"]),
            min_accepted_face_width=float(calibration["min_accepted_face_width"]),
            max_accepted_face_width=float(calibration["max_accepted_face_width"]),
            require_audio_prompts=bool(calibration["require_audio_prompts"]),
            live_smoothing_alpha=float(live["smoothing_alpha"]),
            feature_names=tuple(str(name) for name in calibration["feature_names"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative")
        if self.validation_block_index < 1:
            raise ValueError("validation_block_index must be positive")
        if self.min_train_samples_per_target < 1:
            raise ValueError("min_train_samples_per_target must be positive")
        if self.min_validation_samples_per_target < 1:
            raise ValueError("min_validation_samples_per_target must be positive")
        if self.min_head_movement_train_samples < 1:
            raise ValueError("min_head_movement_train_samples must be positive")
        if self.min_head_movement_validation_samples < 1:
            raise ValueError("min_head_movement_validation_samples must be positive")
        if self.head_movement_training_weight < 1:
            raise ValueError("head_movement_training_weight must be positive")
        if not 0.0 < self.min_accepted_face_width < self.max_accepted_face_width < 1.0:
            raise ValueError("accepted face-width bounds must satisfy 0 < min < max < 1")
        if not 0.0 < self.live_smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class InteractionConfig:
    dwell_seconds: float
    cooldown_seconds: float
    hysteresis_margin: float
    validation_blocks: int
    practice_blocks: int
    target_order_seed: int
    tracking_stable_seconds: float
    countdown_seconds: float
    cue_seconds: float
    practice_measurement_seconds: float
    response_timeout_seconds: float
    feedback_seconds: float
    correction_ridge_alpha: float
    min_correction_samples_per_region: int

    @classmethod
    def load(cls, path: Path) -> InteractionConfig:
        with path.open("rb") as handle:
            raw: dict[str, Any] = tomllib.load(handle)
        selection = raw["selection"]
        validation = raw["controlled_validation"]
        config = cls(
            dwell_seconds=float(selection["dwell_seconds"]),
            cooldown_seconds=float(selection["cooldown_seconds"]),
            hysteresis_margin=float(selection["hysteresis_margin"]),
            validation_blocks=int(validation["blocks"]),
            practice_blocks=int(validation["practice_blocks"]),
            target_order_seed=int(validation["target_order_seed"]),
            tracking_stable_seconds=float(validation["tracking_stable_seconds"]),
            countdown_seconds=float(validation["countdown_seconds"]),
            cue_seconds=float(validation["cue_seconds"]),
            practice_measurement_seconds=float(validation["practice_measurement_seconds"]),
            response_timeout_seconds=float(validation["response_timeout_seconds"]),
            feedback_seconds=float(validation["feedback_seconds"]),
            correction_ridge_alpha=float(validation["correction_ridge_alpha"]),
            min_correction_samples_per_region=int(validation["min_correction_samples_per_region"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.dwell_seconds <= 0.0:
            raise ValueError("dwell_seconds must be positive")
        if self.cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds must be non-negative")
        if not 0.0 <= self.hysteresis_margin < 0.25:
            raise ValueError("hysteresis_margin must be in [0, 0.25)")
        if self.validation_blocks < 2:
            raise ValueError("controlled validation requires at least two blocks")
        if not 1 <= self.practice_blocks < self.validation_blocks:
            raise ValueError("practice_blocks must be positive and less than total blocks")
        durations = (
            self.tracking_stable_seconds,
            self.countdown_seconds,
            self.cue_seconds,
            self.practice_measurement_seconds,
            self.response_timeout_seconds,
            self.feedback_seconds,
        )
        if any(value <= 0.0 for value in durations):
            raise ValueError("controlled validation durations must be positive")
        if self.response_timeout_seconds <= self.dwell_seconds:
            raise ValueError("response timeout must exceed dwell duration")
        if self.correction_ridge_alpha < 0.0:
            raise ValueError("correction_ridge_alpha must be non-negative")
        if self.min_correction_samples_per_region < 1:
            raise ValueError("min_correction_samples_per_region must be positive")
