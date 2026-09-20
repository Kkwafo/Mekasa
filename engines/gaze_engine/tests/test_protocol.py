from __future__ import annotations

from pathlib import Path

import pytest

from mekasa_gaze.config import GuidedValidationConfig
from mekasa_gaze.domain import (
    ExperimentPhase,
    MeasurementStage,
    TargetName,
    TrackingState,
)
from mekasa_gaze.protocol import GuidedProtocol


def _protocol() -> GuidedProtocol:
    config = GuidedValidationConfig.load(Path("config/experiment-001.toml"))
    return GuidedProtocol(config)


def _config() -> GuidedValidationConfig:
    return GuidedValidationConfig.load(Path("config/experiment-001.toml"))


def test_preparation_requires_stable_tracking_then_countdown() -> None:
    protocol = _protocol()
    waiting = protocol.update(0.0, TrackingState.LOST)
    assert waiting.stage is MeasurementStage.WAITING_TRACKING
    assert waiting.stage_remaining_seconds is None

    acquiring = protocol.update(1.0, TrackingState.GOOD)
    assert acquiring.stage is MeasurementStage.WAITING_TRACKING
    assert acquiring.stage_remaining_seconds == pytest.approx(1.5)

    countdown = protocol.update(2.5, TrackingState.GOOD)
    assert countdown.stage is MeasurementStage.COUNTDOWN
    assert countdown.stage_remaining_seconds == pytest.approx(3.0)

    first_segment = protocol.update(5.5, TrackingState.GOOD)
    assert first_segment.phase is ExperimentPhase.CENTER_STABILITY
    assert first_segment.stage is MeasurementStage.TRANSITION
    assert first_segment.target is not None
    assert first_segment.target.name is TargetName.CENTER


def test_tracking_loss_before_start_resets_preparation() -> None:
    protocol = _protocol()
    protocol.update(0.0, TrackingState.GOOD)
    protocol.update(1.5, TrackingState.GOOD)
    reset = protocol.update(2.0, TrackingState.LOST)
    assert reset.stage is MeasurementStage.WAITING_TRACKING
    restarted = protocol.update(3.0, TrackingState.GOOD)
    assert restarted.stage_remaining_seconds == pytest.approx(1.5)


def test_segment_separates_transition_acquisition_and_measurement() -> None:
    config = _config()
    protocol = _protocol()
    protocol.update(0.0, TrackingState.GOOD)
    protocol.update(1.5, TrackingState.GOOD)
    start = 4.5
    transition = protocol.update(start, TrackingState.GOOD)
    acquisition = protocol.update(start + config.transition_seconds, TrackingState.GOOD)
    measurement = protocol.update(
        start + config.transition_seconds + config.acquisition_seconds,
        TrackingState.GOOD,
    )

    assert transition.stage is MeasurementStage.TRANSITION
    assert acquisition.stage is MeasurementStage.ACQUISITION
    assert measurement.stage is MeasurementStage.MEASUREMENT
    assert measurement.phase is ExperimentPhase.CENTER_STABILITY


def test_protocol_contains_required_discrimination_targets() -> None:
    config = _config()
    protocol = _protocol()
    targets = {
        segment.target.name
        for segment in protocol.planned_measurement_segments
        if segment.phase is ExperimentPhase.GAZE_DISCRIMINATION
    }
    assert targets == set(TargetName)
    assert len(protocol.planned_target_orders) == config.discrimination_blocks
    assert all(set(order) == set(TargetName) for order in protocol.planned_target_orders)
    assert protocol.planned_target_orders == _protocol().planned_target_orders


def test_phase_remaining_covers_all_discrimination_targets() -> None:
    config = _config()
    protocol = _protocol()
    protocol.update(0.0, TrackingState.GOOD)
    protocol.update(1.5, TrackingState.GOOD)
    protocol_start = 4.5
    protocol.update(protocol_start, TrackingState.GOOD)
    center_segment_seconds = (
        config.transition_seconds + config.acquisition_seconds + config.center_stability_seconds
    )

    discrimination = protocol.update(
        protocol_start + center_segment_seconds,
        TrackingState.GOOD,
    )

    assert discrimination.phase is ExperimentPhase.GAZE_DISCRIMINATION
    assert discrimination.block_index == 1
    assert discrimination.phase_remaining_seconds == pytest.approx(
        config.discrimination_blocks
        * len(TargetName)
        * (config.transition_seconds + config.acquisition_seconds + config.measurement_seconds)
    )
