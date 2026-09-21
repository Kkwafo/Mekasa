from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from mekasa_gaze.core.domain import FeatureObservation, TrackingState


class PlacementState(StrEnum):
    NO_FACE = "NO_FACE"
    TOO_FAR = "TOO_FAR"
    READY = "READY"
    TOO_CLOSE = "TOO_CLOSE"


@dataclass(frozen=True, slots=True)
class PlacementAssessment:
    state: PlacementState
    instruction: str
    face_width: float | None

    @property
    def ready(self) -> bool:
        return self.state is PlacementState.READY


def assess_face_placement(
    observation: FeatureObservation,
    *,
    min_face_width: float,
    max_face_width: float,
) -> PlacementAssessment:
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
    if observation.tracking_state is not TrackingState.GOOD:
        return observation.tracking_state
    return TrackingState.GOOD if placement.ready else TrackingState.UNCERTAIN
