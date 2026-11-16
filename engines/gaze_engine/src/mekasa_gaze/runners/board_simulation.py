"""Scripted, camera-free replay of the full board pipeline.

Synthetic :class:`FeatureObservation` values go through the *real* validity rules, placement
gate, calibrated estimator, smoothing, board mapping, communication session and speech
adapter, with an injected clock. It verifies software behaviour only (safety gates, timing,
eye modes); it says nothing about real-world gaze accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mekasa_gaze.core.board import CommunicationConfig
from mekasa_gaze.core.config import GuidedValidationConfig
from mekasa_gaze.core.domain import FeatureObservation, TrackingState
from mekasa_gaze.core.eye_mode import EyeMode, feature_names_for_mode
from mekasa_gaze.runners.communication_board import GazePipeline, gaze_input, speak_safely
from mekasa_gaze.tracking.calibration import CalibrationModel
from mekasa_gaze.tracking.communication import CommunicationSession
from mekasa_gaze.ui.board_view import board_area, board_to_screen

FPS = 30.0
_ALL_FEATURES = ("left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y")

# Conditions a segment can impose on the synthetic signal.
CONDITIONS = (
    "ok",
    "no_frame",  # camera disconnected: no frames arrive
    "frozen",  # camera stalls: the last frame is reused and gets old
    "no_face",
    "too_far",
    "left_closed",
    "right_closed",
    "paused",
)


@dataclass(frozen=True, slots=True)
class Segment:
    """``seconds`` of gaze at ``target`` (an option key, or None for the board centre gap)."""

    seconds: float
    target: str | None
    condition: str = "ok"
    expect_emissions: bool = True


@dataclass(slots=True)
class FakeSpeech:
    """Speech double that records phrases; can be told to fail."""

    fail: bool = False
    spoken: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return True

    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": "fake"}

    def speak(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("synthetic speech failure")
        self.spoken.append(text)

    def close(self) -> None:
        return None


class _NoExtractor:
    """Placeholder: simulation feeds observations directly, never images."""

    def extract(self, *_: object, **__: object) -> FeatureObservation:  # pragma: no cover
        raise RuntimeError("simulation does not process images")

    def close(self) -> None:
        return None


def synthetic_model(mode: EyeMode) -> CalibrationModel:
    """Linear model whose iris features *are* the screen coordinates (averaged per eye)."""
    names = feature_names_for_mode(_ALL_FEATURES, mode)
    weights = np.zeros((len(names), 2))
    eyes = 2 if mode is EyeMode.BOTH else 1
    for index, name in enumerate(names):
        weights[index, 0 if name.endswith("_x") else 1] = 1.0 / eyes
    return CalibrationModel(
        feature_names=names,
        feature_mean=np.zeros(len(names)),
        feature_scale=np.ones(len(names)),
        weights=weights,
        output_intercept=np.zeros(2),
        targets={},
    )


def synthetic_observation(
    point: tuple[float, float], condition: str, sequence: int, timestamp_s: float
) -> FeatureObservation:
    """Build an observation looking at screen ``point`` under ``condition``."""
    if condition == "no_face":
        return FeatureObservation(sequence, int(timestamp_s * 1e9), TrackingState.LOST)
    left_ear = 0.05 if condition == "left_closed" else 0.30
    right_ear = 0.05 if condition == "right_closed" else 0.30
    features = {
        "left_eye_aspect_ratio": left_ear,
        "right_eye_aspect_ratio": right_ear,
        "left_eye_vertical_span": 0.02,
        "right_eye_vertical_span": 0.02,
        "left_iris_x": point[0],
        "left_iris_y": point[1],
        "right_iris_x": point[0],
        "right_iris_y": point[1],
        "face_width": 0.10 if condition == "too_far" else 0.35,
    }
    return FeatureObservation(
        sequence, int(timestamp_s * 1e9), TrackingState.GOOD, features=features
    )


def simulate(
    segments: list[Segment],
    communication: CommunicationConfig,
    validity_config: GuidedValidationConfig,
    *,
    calibrated_modes: tuple[EyeMode, ...] | None = None,
    speech: FakeSpeech | None = None,
    smoothing_alpha: float = 0.5,
) -> dict[str, Any]:
    """Run ``segments`` through the pipeline; report emissions per segment."""
    mode = communication.eye_mode
    calibrated = (mode,) if calibrated_modes is None else calibrated_modes
    model = synthetic_model(mode) if mode in calibrated else None
    session = CommunicationSession(
        communication, calibration_mode=mode if model is not None else None
    )
    width, height = validity_config.canvas_width, validity_config.canvas_height
    area = board_area(width, height)
    pipeline = GazePipeline(
        extractor=_NoExtractor(),
        validity_config=validity_config,
        model=model,
        eye_mode=mode,
        smoothing_alpha=smoothing_alpha,
        area=area,
    )
    voice = speech or FakeSpeech()
    layout = session.layout
    now = 1000.0
    sequence = 0
    last = None
    results: list[dict[str, Any]] = []
    unsafe_emissions = 0
    for segment in segments:
        if segment.condition not in CONDITIONS:
            raise ValueError(f"Unknown condition {segment.condition!r}")
        if segment.target is None:
            screen_point = board_to_screen((0.5, 0.5), area)
        else:
            option = layout.option(segment.target)
            left, top, right, bottom = option.bounds
            screen_point = board_to_screen(((left + right) / 2, (top + bottom) / 2), area)
        if segment.condition == "paused":
            session.pause()
        elif session.paused:
            session.resume()
        emitted: list[str] = []
        spoken_ok = 0
        for _ in range(round(segment.seconds * FPS)):
            now += 1.0 / FPS
            connected = segment.condition != "no_frame"
            if connected and segment.condition != "frozen":
                sequence += 1
                observation = synthetic_observation(
                    screen_point, segment.condition, sequence, now - 0.03
                )
                last = pipeline.process_observation(observation, frame_timestamp_seconds=now - 0.03)
            update = session.update(gaze_input(last, now, connected))
            if update.confirmed is not None:
                emitted.append(update.confirmed.key)
                spoken_ok += int(speak_safely(voice, update.confirmed.phrase))
        if not segment.expect_emissions:
            unsafe_emissions += len(emitted)
        results.append(
            {
                "seconds": segment.seconds,
                "target": segment.target,
                "condition": segment.condition,
                "expect_emissions": segment.expect_emissions,
                "emitted": emitted,
                "spoken": spoken_ok,
            }
        )
    return {
        "kind": "SYNTHETIC_BOARD_REPLAY",
        "status": "verified-in-replay (software only; not a gaze accuracy measurement)",
        "eye_mode": mode.value,
        "dwell_seconds": communication.dwell_seconds,
        "segments": results,
        "session": session.stats.as_dict(include_options=True),
        "message_on_screen": session.message,
        "safety": {
            "unsafe_segment_emissions": unsafe_emissions,
            "passed": unsafe_emissions == 0,
        },
    }


def default_script(mode: EyeMode) -> list[Segment]:
    """Communication tasks interleaved with every zero-emission condition."""
    closed_other = "right_closed" if mode is EyeMode.LEFT else "left_closed"
    closed_own = "left_closed" if mode is EyeMode.LEFT else "right_closed"
    script = [
        Segment(2.0, "YES"),
        Segment(1.0, None),
        Segment(2.0, "WATER"),
        Segment(5.0, "WATER", expect_emissions=False),  # staring does not repeat
        Segment(3.0, "HELP", "no_frame", expect_emissions=False),
        Segment(3.0, "HELP", "frozen", expect_emissions=False),
        Segment(3.0, "HELP", "no_face", expect_emissions=False),
        Segment(3.0, "HELP", "too_far", expect_emissions=False),
        Segment(3.0, "HELP", "paused", expect_emissions=False),
        Segment(3.0, "NO", closed_own, expect_emissions=False),
        Segment(2.0, "HELP"),
    ]
    if mode is not EyeMode.BOTH:
        # A single-eye mode keeps working when the *other* eye is closed.
        script.append(Segment(2.0, "NO", closed_other))
    return script


def run_default_simulation(
    communication: CommunicationConfig, validity_config: GuidedValidationConfig
) -> dict[str, Any]:
    """Run the default script for every eye mode and aggregate the verdict."""
    from dataclasses import replace

    reports = []
    for mode in EyeMode:
        config = replace(communication, eye_mode=mode, confirmation_required=False)
        reports.append(simulate(default_script(mode), config, validity_config))
    uncalibrated = simulate(
        [Segment(5.0, "YES", expect_emissions=False)],
        communication,
        validity_config,
        calibrated_modes=(),
    )
    passed = all(report["safety"]["passed"] for report in [*reports, uncalibrated])
    return {
        "kind": "SYNTHETIC_BOARD_REPLAY_SUITE",
        "status": "verified-in-replay (software only)",
        "modes": reports,
        "uncalibrated": uncalibrated,
        "safety": {"passed": passed},
    }


__all__ = [
    "CONDITIONS",
    "FakeSpeech",
    "Segment",
    "default_script",
    "run_default_simulation",
    "simulate",
    "synthetic_model",
    "synthetic_observation",
]
