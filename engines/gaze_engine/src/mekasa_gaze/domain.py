from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class TrackingState(StrEnum):
    """Tracking evidence exposed without inventing a gaze coordinate."""

    GOOD = "TRACKING_GOOD"
    UNCERTAIN = "TRACKING_UNCERTAIN"
    LOST = "TRACKING_LOST"


@dataclass(frozen=True, slots=True)
class HeadPose:
    """Approximate Euler orientation derived from the MediaPipe transform matrix.

    Values are diagnostics, not validated physical gaze angles.
    """

    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float


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

    @property
    def tracking_available(self) -> bool:
        return self.tracking_state is TrackingState.GOOD
