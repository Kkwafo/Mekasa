from __future__ import annotations

from mekasa_gaze.core.domain import FeatureObservation, TrackingState
from mekasa_gaze.tracking.placement import (
    PlacementState,
    assess_face_placement,
    placement_tracking_state,
)


def _observation(width: float, state: TrackingState = TrackingState.GOOD) -> FeatureObservation:
    return FeatureObservation(1, 1, state, features={"face_width": width})


def test_face_placement_guides_distance_and_gates_protocol_readiness() -> None:
    too_far = assess_face_placement(_observation(0.25), min_face_width=0.30, max_face_width=0.45)
    ready = assess_face_placement(_observation(0.34), min_face_width=0.30, max_face_width=0.45)
    too_close = assess_face_placement(_observation(0.50), min_face_width=0.30, max_face_width=0.45)

    assert too_far.state is PlacementState.TOO_FAR
    assert ready.state is PlacementState.READY
    assert too_close.state is PlacementState.TOO_CLOSE
    assert placement_tracking_state(_observation(0.25), too_far) is TrackingState.UNCERTAIN
    assert placement_tracking_state(_observation(0.34), ready) is TrackingState.GOOD


def test_tracking_loss_is_not_reported_as_distance_error() -> None:
    observation = _observation(0.34, TrackingState.LOST)
    placement = assess_face_placement(observation, min_face_width=0.30, max_face_width=0.45)

    assert placement.state is PlacementState.NO_FACE
    assert placement_tracking_state(observation, placement) is TrackingState.LOST
