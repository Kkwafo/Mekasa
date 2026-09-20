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
    canvas_width: int
    canvas_height: int
    debug_panel_fraction: float
    target_radius_pixels: int
    fullscreen: bool
    show_face_box: bool
    show_landmarks: bool
    targets: tuple[VisualTarget, ...]

    @classmethod
    def load(cls, path: Path) -> GuidedValidationConfig:
        with path.open("rb") as handle:
            raw: dict[str, Any] = tomllib.load(handle)
        camera = raw["camera"]
        tracking = raw["tracking"]
        validity = raw["eye_validity"]
        guided = raw["guided"]
        rendering = raw["rendering"]
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
            canvas_width=int(rendering["canvas_width"]),
            canvas_height=int(rendering["canvas_height"]),
            debug_panel_fraction=float(rendering["debug_panel_fraction"]),
            target_radius_pixels=int(rendering["target_radius_pixels"]),
            fullscreen=bool(rendering["fullscreen"]),
            show_face_box=bool(rendering["show_face_box"]),
            show_landmarks=bool(rendering["show_landmarks"]),
            targets=targets,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.0 < self.debug_panel_fraction < 0.5:
            raise ValueError("debug_panel_fraction must be between 0 and 0.5")
        if self.min_samples_per_target < 1:
            raise ValueError("min_samples_per_target must be positive")
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
        for target in self.targets:
            if not 0.0 <= target.normalized_x <= 1.0:
                raise ValueError(f"target {target.name} x must be normalized")
            if not 0.0 <= target.normalized_y <= 1.0:
                raise ValueError(f"target {target.name} y must be normalized")

    def target(self, name: TargetName) -> VisualTarget:
        return next(target for target in self.targets if target.name is name)
