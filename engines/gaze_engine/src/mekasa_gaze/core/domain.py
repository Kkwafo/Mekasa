from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt


class TrackingState(StrEnum):
    """Tracking evidence exposed without inventing a gaze coordinate."""

    GOOD = "TRACKING_GOOD"
    UNCERTAIN = "TRACKING_UNCERTAIN"
    LOST = "TRACKING_LOST"


class GazeValidity(StrEnum):
    """Whether derived eye features are suitable for gaze analysis."""

    VALID = "GAZE_VALID"
    PARTIAL = "GAZE_PARTIAL"
    EYE_CLOSED = "EYE_CLOSED"
    BLINK = "POSSIBLE_BLINK"
    INVALID = "GAZE_INVALID"
    TRACKING_INVALID = "TRACKING_INVALID"


class EyeState(StrEnum):
    """Operational state of an individual eye."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    INVALID = "INVALID"


class ExperimentPhase(StrEnum):
    """High-level phase of the experimental validation workflow."""

    PREPARATION = "PREPARATION"
    CENTER_STABILITY = "CENTER_STABILITY"
    GAZE_DISCRIMINATION = "GAZE_DISCRIMINATION"
    HEAD_MOVEMENT = "HEAD_MOVEMENT"
    COMPLETE = "COMPLETE"


class MeasurementStage(StrEnum):
    """Micro-stage within an active protocol segment."""

    WAITING_TRACKING = "WAITING_TRACKING"
    COUNTDOWN = "COUNTDOWN"
    TRANSITION = "TRANSITION"
    ACQUISITION = "ACQUISITION"
    MEASUREMENT = "MEASUREMENT"
    COMPLETE = "COMPLETE"


class TargetName(StrEnum):
    """Names of standardized visual targets presented to the participant."""

    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"


class RegionName(StrEnum):
    """Names of the four quadrant communication targets."""

    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"



@dataclass(frozen=True, slots=True)
class CameraSettings:
    """Hardware and backend acquisition configuration of a camera."""

    index: int
    backend: str
    width: int
    height: int
    fps: float


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """A timestamped video frame captured from an acquisition device."""

    sequence: int
    captured_at_ns: int
    capture_latency_ms: float
    bgr: npt.NDArray[np.uint8]


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
    """Normalized coordinates of a visual target stimulus on screen."""

    name: TargetName
    normalized_x: float
    normalized_y: float


@dataclass(frozen=True, slots=True)
class ProtocolSnapshot:
    """Point-in-time state of the experimental protocol state machine."""

    phase: ExperimentPhase
    stage: MeasurementStage
    target: VisualTarget | None
    instruction: str
    stage_remaining_seconds: float | None
    protocol_elapsed_seconds: float
    phase_remaining_seconds: float | None = None
    block_index: int | None = None
    segment_index: int | None = None
    completed: bool = False


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    """Raw geometric landmark features extracted from a single video frame."""

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
        """Indicate whether face and eye tracking are currently reliable."""
        return self.tracking_state is TrackingState.GOOD


@dataclass(frozen=True, slots=True)
class GuidedObservation:
    """Synchronized observation combining protocol state, features, and validity assessment."""

    relative_timestamp_seconds: float
    phase: ExperimentPhase
    stage: MeasurementStage
    target: VisualTarget | None
    tracking_state: TrackingState
    gaze_validity: GazeValidity
    left_eye_state: EyeState
    right_eye_state: EyeState
    block_index: int | None = None
    segment_index: int | None = None
    validity_reason: str = ""
    features: Mapping[str, float] = field(default_factory=dict)
    head_pose: HeadPose | None = None
    capture_latency_ms: float = 0.0
    feature_latency_ms: float = 0.0
    loop_latency_ms: float = 0.0
    render_latency_ms: float = 0.0
    display_latency_ms: float = 0.0
    iteration_latency_ms: float = 0.0

    @property
    def is_valid_measurement(self) -> bool:
        """Check if this sample was gathered during a valid measurement stage with usable eyes."""
        return (
            self.stage is MeasurementStage.MEASUREMENT
            and self.gaze_validity in (GazeValidity.VALID, GazeValidity.PARTIAL)
            and self.target is not None
        )
