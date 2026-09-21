from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt


class TrackingState(StrEnum):
    """Categorical evidence of landmark tracking quality from the vision pipeline.

    Responsibility:
        Represents the presence and geometric plausibility of facial landmarks
        in the video frame without inventing or assuming a calibrated gaze coordinate.
    """

    GOOD = "TRACKING_GOOD"
    UNCERTAIN = "TRACKING_UNCERTAIN"
    LOST = "TRACKING_LOST"


class GazeValidity(StrEnum):
    """Categorical assessment of eye usability for gaze estimation and fixation analysis.

    Responsibility:
        Encodes whether derived eye features meet anatomical and experimental thresholds,
        differentiating usable open eyes, partial single-eye tracking, blinks, and invalid frames.
    """

    VALID = "GAZE_VALID"
    PARTIAL = "GAZE_PARTIAL"
    EYE_CLOSED = "EYE_CLOSED"
    BLINK = "POSSIBLE_BLINK"
    INVALID = "GAZE_INVALID"
    TRACKING_INVALID = "TRACKING_INVALID"


class EyeState(StrEnum):
    """Operational state of an individual ocular aperture.

    Responsibility:
        Indicates whether a specific eye (left or right) is open, closed, or geometrically invalid.
    """

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    INVALID = "INVALID"


class ExperimentPhase(StrEnum):
    """High-level phase of the experimental validation workflow.

    Responsibility:
        Defines the macro-stages of structured scientific experiments (e.g. baseline center
        stability, multi-target gaze discrimination, or controlled head movement).
    """

    PREPARATION = "PREPARATION"
    CENTER_STABILITY = "CENTER_STABILITY"
    GAZE_DISCRIMINATION = "GAZE_DISCRIMINATION"
    HEAD_MOVEMENT = "HEAD_MOVEMENT"
    COMPLETE = "COMPLETE"


class MeasurementStage(StrEnum):
    """Micro-stage within an active protocol segment.

    Responsibility:
        Controls the precise real-time progression of a trial, from tracking acquisition,
        countdown, target transition, fixation acquisition, active measurement, to completion.
    """

    WAITING_TRACKING = "WAITING_TRACKING"
    COUNTDOWN = "COUNTDOWN"
    TRANSITION = "TRANSITION"
    ACQUISITION = "ACQUISITION"
    MEASUREMENT = "MEASUREMENT"
    COMPLETE = "COMPLETE"


class TargetName(StrEnum):
    """Identities of the nine standardized calibration and validation visual targets.

    Responsibility:
        Provides unambiguous semantic names for the center, orthogonal, and diagonal visual targets.
    """

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
    """Identities of the four quadrant communication targets.

    Responsibility:
        Defines the abstract quadrant regions used for AAC gaze-dwell selection.
    """

    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"


@dataclass(frozen=True, slots=True)
class CameraSettings:
    """Hardware and backend acquisition configuration of a camera device.

    Responsibility:
        Encapsulates negotiated hardware properties (resolution, backend, framerate).

    Attributes:
        index: Hardware device index enumerated by the operating system.
        backend: Backend API driver name (e.g. 'MSMF', 'DSHOW', 'ANY').
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Target acquisition framerate in frames per second.
    """

    index: int
    backend: str
    width: int
    height: int
    fps: float


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """A timestamped video frame captured from an acquisition device.

    Responsibility:
        Holds an ephemeral raw image array and precise monotonic timing telemetry.

    Attributes:
        sequence: Monotonic sequential counter incremented on every read.
        captured_at_ns: High-resolution timestamp from `perf_counter_ns` upon capture completion.
        capture_latency_ms: Milliseconds elapsed while waiting for the hardware frame.
        bgr: Raw image array in OpenCV standard BGR uint8 format.
    """

    sequence: int
    captured_at_ns: int
    capture_latency_ms: float
    bgr: npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class HeadPose:
    """Approximate 3D Euler orientation derived from facial transformation matrix.

    Responsibility:
        Provides diagnostic head yaw, pitch, and roll angles to measure user posture.
        Note: Values are diagnostic approximations, not calibrated physical angles.

    Attributes:
        yaw_degrees: Horizontal rotation angle (left/right).
        pitch_degrees: Vertical rotation angle (up/down).
        roll_degrees: Lateral head tilt angle.
    """

    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float


@dataclass(frozen=True, slots=True)
class VisualTarget:
    """Normalized 2D coordinates of a visual target stimulus presented on screen.

    Responsibility:
        Defines the target identifier and unit coordinates within the active test area.

    Attributes:
        name: Target semantic identity (e.g. CENTER, LEFT, TOP_RIGHT).
        normalized_x: Horizontal coordinate normalized to [0.0, 1.0].
        normalized_y: Vertical coordinate normalized to [0.0, 1.0].
    """

    name: TargetName
    normalized_x: float
    normalized_y: float


@dataclass(frozen=True, slots=True)
class ProtocolSnapshot:
    """Point-in-time snapshot of the experimental protocol state machine.

    Responsibility:
        Communicates the active phase, micro-stage, prompt, and remaining countdowns
        to visualization renderers and audio controllers.

    Attributes:
        phase: Active high-level experiment phase.
        stage: Current micro-stage within the active segment.
        target: Target stimulus currently displayed, or None if in transition/preparation.
        instruction: Human-readable instructional message in Spanish.
        stage_remaining_seconds: Seconds remaining in the current micro-stage.
        protocol_elapsed_seconds: Total wall time elapsed since protocol start.
        phase_remaining_seconds: Estimated seconds remaining in the current macro-phase.
        block_index: 1-based repetition block counter.
        segment_index: 0-based sequential segment index.
        completed: True when all protocol segments have finished.
    """

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
    """Raw geometric landmark features extracted from a single video frame.

    Responsibility:
        Provides resolution-independent geometric ratios (EAR, normalized iris coordinates,
        face bounding box) and tracking evidence derived from face mesh landmarks.

    Attributes:
        frame_sequence: Sequence index of the corresponding video frame.
        timestamp_ns: Capture timestamp of the frame in nanoseconds.
        tracking_state: Assessment of landmark detection quality (GOOD, UNCERTAIN, LOST).
        features: Key-value map of normalized floating-point geometric features.
        head_pose: Estimated head Euler angles, or None if unavailable.
        confidence: Uncalibrated confidence score, if provided by the model.
        evidence: Descriptive diagnostic text describing tracking validity or failure reason.
        feature_latency_ms: Milliseconds elapsed during landmark inference and feature derivation.
        debug_eye_points: Ephemeral normalized points for eye perimeter visualization.
        debug_left_iris_points: Ephemeral normalized points for left iris visualization.
        debug_right_iris_points: Ephemeral normalized points for right iris visualization.
        debug_face_box: Normalized (min_x, min_y, max_x, max_y) face bounding box.
    """

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
        """Indicate whether face and eye tracking are currently reliable.

        Returns:
            bool: True if tracking state is TrackingState.GOOD.
        """
        return self.tracking_state is TrackingState.GOOD


@dataclass(frozen=True, slots=True)
class GuidedObservation:
    """Synchronized observation combining protocol state, features, and validity assessment.

    Responsibility:
        Represents a fully reconciled record linking experimental stimuli, participant
        ocular response, validity classification, and latency telemetry for reporting.

    Attributes:
        relative_timestamp_seconds: Seconds elapsed since the first protocol observation.
        phase: Active experiment phase during sample capture.
        stage: Current measurement stage during sample capture.
        target: Active visual target presented during capture.
        tracking_state: Raw tracking state from feature extraction.
        gaze_validity: Validity classification (VALID, PARTIAL, EYE_CLOSED, BLINK, INVALID).
        left_eye_state: State of the anatomical left eye.
        right_eye_state: State of the anatomical right eye.
        block_index: Repetition block index.
        segment_index: Sequential segment index.
        validity_reason: Diagnostic reason for validity classification.
        features: Map of persisted normalized geometric features.
        head_pose: Diagnostic Euler head orientation.
        capture_latency_ms: Time in ms spent capturing the camera frame.
        feature_latency_ms: Time in ms spent extracting landmarks.
        loop_latency_ms: Time in ms elapsed across the entire acquisition loop iteration.
        render_latency_ms: Time in ms spent drawing the visual canvas.
        display_latency_ms: Time in ms spent rendering to the window display.
        iteration_latency_ms: Total iteration wall time.
    """

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
        """Check if this sample was gathered during a valid measurement stage with usable eyes.

        Returns:
            bool: True if stage is MEASUREMENT, validity is VALID or PARTIAL, and target is active.
        """
        return (
            self.stage is MeasurementStage.MEASUREMENT
            and self.gaze_validity in (GazeValidity.VALID, GazeValidity.PARTIAL)
            and self.target is not None
        )
