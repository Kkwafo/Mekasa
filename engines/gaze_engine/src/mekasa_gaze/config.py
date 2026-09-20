from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
