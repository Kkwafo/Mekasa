from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from mekasa_gaze.core.domain import FeatureObservation, TrackingState


class PlacementState(StrEnum):
    """Categorical distance/placement classification states for participant positioning."""

    NO_FACE = "NO_FACE"
    TOO_FAR = "TOO_FAR"
    READY = "READY"
    TOO_CLOSE = "TOO_CLOSE"


@dataclass(frozen=True, slots=True)
class PlacementAssessment:
    """Evaluation result of participant camera distance and alignment.

    Responsibility:
        Encapsulates placement state category, localized instructional string,
        and measured normalized face width ratio.
    """

    state: PlacementState
    instruction: str
    face_width: float | None

    @property
    def ready(self) -> bool:
        """Indicate whether participant position is within experimental calibration bounds."""
        return self.state is PlacementState.READY


def assess_face_placement(
    observation: FeatureObservation,
    *,
    min_face_width: float,
    max_face_width: float,
) -> PlacementAssessment:
    """Evaluate participant distance from camera using normalized facial bounding box width.

    Responsibility:
        Compares observed face bounding box width against minimum and maximum distance thresholds
        to classify user position (TOO_FAR, TOO_CLOSE, READY, NO_FACE) and generate Spanish prompts.

    Args:
        observation: Instantaneous facial feature observation from the tracker.
        min_face_width: Minimum normalized face width required (too far below this).
        max_face_width: Maximum normalized face width allowed (too close above this).

    Returns:
        PlacementAssessment: Evaluated placement state, prompt text, and observed width.
    """
    if observation.tracking_state is not TrackingState.GOOD:
        return PlacementAssessment(PlacementState.NO_FACE, "UBICATE FRENTE A LA CAMARA", None)
    raw_width = observation.features.get("face_width")
    if raw_width is None or not isfinite(raw_width):
        return PlacementAssessment(PlacementState.NO_FACE, "ROSTRO NO MEDIBLE", None)
    face_width = float(raw_width)
    if face_width < min_face_width:
        return PlacementAssessment(PlacementState.TOO_FAR, "ACERCATE UN POCO", face_width)
    if face_width > max_face_width:
        return PlacementAssessment(PlacementState.TOO_CLOSE, "ALEJATE UN POCO", face_width)
    return PlacementAssessment(PlacementState.READY, "DISTANCIA CORRECTA", face_width)


def placement_tracking_state(
    observation: FeatureObservation, placement: PlacementAssessment
) -> TrackingState:
    """Derive effective operational tracking state combining feature and placement quality.

    Responsibility:
        Demotes TrackingState.GOOD to TrackingState.UNCERTAIN if participant placement is outside
        acceptable bounding box dimensions, while preserving LOST or UNCERTAIN tracker states.

    Args:
        observation: Feature observation with raw landmark tracking state.
        placement: Evaluated distance and alignment assessment.

    Returns:
        TrackingState: Consolidated operational tracking status.
    """
    if observation.tracking_state is not TrackingState.GOOD:
        return observation.tracking_state
    return TrackingState.GOOD if placement.ready else TrackingState.UNCERTAIN
