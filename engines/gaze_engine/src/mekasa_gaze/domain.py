from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class TrackingState(StrEnum):
    """Tracking evidence exposed without inventing a gaze coordinate."""

    GOOD = "TRACKING_GOOD"
    UNCERTAIN = "TRACKING_UNCERTAIN"
    LOST = "TRACKING_LOST"


class GazeValidity(StrEnum):
    """Whether derived eye features are suitable for gaze analysis."""

    VALID = "GAZE_VALID"
    EYE_CLOSED = "EYE_CLOSED"
    BLINK = "POSSIBLE_BLINK"
    INVALID = "GAZE_INVALID"
    TRACKING_INVALID = "TRACKING_INVALID"


class EyeState(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    INVALID = "INVALID"


class ExperimentPhase(StrEnum):
    PREPARATION = "PREPARATION"
    CENTER_STABILITY = "CENTER_STABILITY"
    GAZE_DISCRIMINATION = "GAZE_DISCRIMINATION"
    HEAD_MOVEMENT = "HEAD_MOVEMENT"
    COMPLETE = "COMPLETE"


class MeasurementStage(StrEnum):
    WAITING_TRACKING = "WAITING_TRACKING"
    COUNTDOWN = "COUNTDOWN"
    TRANSITION = "TRANSITION"
    ACQUISITION = "ACQUISITION"
    MEASUREMENT = "MEASUREMENT"
    COMPLETE = "COMPLETE"


class TargetName(StrEnum):
    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"


@dataclass(frozen=True, slots=True)
class HeadPose:
    """Approximate Euler orientation derived from the MediaPipe transform matrix.

    Values are diagnostics, not validated physical gaze angles.
    """

    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float


@dataclass(frozen=True, slots=True)
class VisualTarget:
    name: TargetName
    normalized_x: float
    normalized_y: float


@dataclass(frozen=True, slots=True)
class ProtocolSnapshot:
    phase: ExperimentPhase
    stage: MeasurementStage
    target: VisualTarget | None
    instruction: str
    stage_remaining_seconds: float | None
    protocol_elapsed_seconds: float
    phase_remaining_seconds: float | None = None
    completed: bool = False


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    frame_sequence: int
    timestamp_ns: int
    tracking_state: TrackingState
    features: Mapping[str, float] = field(default_factory=dict)
    head_pose: HeadPose | None = None
    confidence: float | None = None
    evidence: str = ""
    feature_latency_ms: float = 0.0
    # Ephemeral normalized points used only for the local debug overlay.
    debug_eye_points: tuple[tuple[float, float], ...] = ()
    debug_left_iris_points: tuple[tuple[float, float], ...] = ()
    debug_right_iris_points: tuple[tuple[float, float], ...] = ()
    debug_face_box: tuple[float, float, float, float] | None = None

    @property
    def tracking_available(self) -> bool:
        return self.tracking_state is TrackingState.GOOD


@dataclass(frozen=True, slots=True)
class GuidedObservation:
    relative_timestamp_seconds: float
    phase: ExperimentPhase
    stage: MeasurementStage
    target: VisualTarget | None
    tracking_state: TrackingState
    gaze_validity: GazeValidity
    left_eye_state: EyeState
    right_eye_state: EyeState
    validity_reason: str = ""
    features: Mapping[str, float] = field(default_factory=dict)
    head_pose: HeadPose | None = None
    capture_latency_ms: float = 0.0
    feature_latency_ms: float = 0.0
    loop_latency_ms: float = 0.0

    @property
    def is_valid_measurement(self) -> bool:
        return (
            self.stage is MeasurementStage.MEASUREMENT
            and self.gaze_validity is GazeValidity.VALID
            and self.target is not None
        )
