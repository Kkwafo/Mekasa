from __future__ import annotations

from pathlib import Path

import pytest

from mekasa_gaze.config import CalibrationConfig, GuidedValidationConfig, InteractionConfig
from mekasa_gaze.controlled_region_validation import _build_report
from mekasa_gaze.interaction import RegionName
from mekasa_gaze.region_protocol import (
    ControlledRegionProtocol,
    RegionTrialOutcome,
    RegionTrialStage,
)


def _config() -> InteractionConfig:
    return InteractionConfig.load(Path("config/experiment-003.toml"))


def _start_first_response(protocol: ControlledRegionProtocol) -> tuple[float, RegionName]:
    assert protocol.update(0.0, ready=False).stage is RegionTrialStage.PREPARATION
    assert protocol.update(1.0, ready=True).stage is RegionTrialStage.PREPARATION
    assert protocol.update(2.5, ready=True).stage is RegionTrialStage.COUNTDOWN
    cue = protocol.update(5.5, ready=True)
    assert cue.stage is RegionTrialStage.CUE
    assert cue.requested is not None
    response = protocol.update(9.5, ready=True)
    assert response.stage is RegionTrialStage.RESPONSE
    assert response.is_practice
    return 9.5, cue.requested


def _advance_practice_block(protocol: ControlledRegionProtocol) -> tuple[float, RegionName]:
    response_started, _ = _start_first_response(protocol)
    for trial_number in range(4):
        feedback = protocol.update(response_started + 3.0, ready=True)
        assert feedback.completed_trial is not None
        assert feedback.completed_trial.outcome is RegionTrialOutcome.PRACTICE
        cue = protocol.update(response_started + 4.5, ready=True)
        assert cue.stage is RegionTrialStage.CUE
        response = protocol.update(response_started + 8.5, ready=True)
        assert response.stage is RegionTrialStage.RESPONSE
        response_started += 8.5
        if trial_number < 3:
            assert response.is_practice
    assert not response.is_practice
    assert response.requested is not None
    return response_started, response.requested


def test_controlled_protocol_records_practice_without_firing_selection() -> None:
    protocol = ControlledRegionProtocol(_config())
    response_started, requested = _start_first_response(protocol)

    still_practicing = protocol.update(response_started + 1.3, ready=True, selected=requested)
    assert still_practicing.stage is RegionTrialStage.RESPONSE

    feedback = protocol.update(response_started + 3.0, ready=True, selected=requested)

    assert feedback.stage is RegionTrialStage.FEEDBACK
    assert feedback.completed_trial is not None
    assert feedback.completed_trial.outcome is RegionTrialOutcome.PRACTICE
    assert feedback.completed_trial.selected is None
    assert feedback.completed_trial.response_seconds is None
    repeated = protocol.update(response_started + 3.1, ready=True)
    assert repeated.completed_trial == feedback.completed_trial
    next_trial = protocol.update(response_started + 4.5, ready=True)
    assert next_trial.stage is RegionTrialStage.CUE
    assert next_trial.trial_index == 1


def test_controlled_protocol_scores_selection_after_practice() -> None:
    protocol = ControlledRegionProtocol(_config())
    response_started, requested = _advance_practice_block(protocol)

    feedback = protocol.update(response_started + 1.3, ready=True, selected=requested)

    assert feedback.completed_trial is not None
    assert feedback.completed_trial.outcome is RegionTrialOutcome.CORRECT
    assert feedback.completed_trial.response_seconds == pytest.approx(1.3)


def test_controlled_protocol_reports_timeout_without_inventing_selection() -> None:
    protocol = ControlledRegionProtocol(_config())
    response_started, _ = _advance_practice_block(protocol)

    feedback = protocol.update(response_started + 5.0, ready=True)

    assert feedback.completed_trial is not None
    assert feedback.completed_trial.outcome is RegionTrialOutcome.NO_SELECTION
    assert feedback.completed_trial.selected is None
    assert feedback.completed_trial.response_seconds is None


def test_each_seeded_block_contains_every_region_once() -> None:
    protocol = ControlledRegionProtocol(_config())

    assert len(protocol.planned_orders) == 3
    assert all(set(block) == set(RegionName) for block in protocol.planned_orders)


def test_controlled_report_distinguishes_errors_and_no_selection() -> None:
    interaction = _config()
    protocol = ControlledRegionProtocol(interaction)
    results = [
        {
            "requested": "BOTTOM_RIGHT",
            "selected": None,
            "outcome": "PRACTICE",
            "response_seconds": None,
            "dwell_cancellations": 0,
        },
        {
            "requested": "TOP_LEFT",
            "selected": "TOP_LEFT",
            "outcome": "CORRECT",
            "response_seconds": 1.4,
            "dwell_cancellations": 0,
        },
        {
            "requested": "TOP_RIGHT",
            "selected": "BOTTOM_RIGHT",
            "outcome": "INCORRECT",
            "response_seconds": 1.8,
            "dwell_cancellations": 1,
        },
        {
            "requested": "BOTTOM_LEFT",
            "selected": None,
            "outcome": "NO_SELECTION",
            "response_seconds": None,
            "dwell_cancellations": 2,
        },
    ]

    report = _build_report(
        results=results,
        protocol=protocol,
        interaction_config=interaction,
        guided_config=GuidedValidationConfig.load(Path("config/experiment-001.toml")),
        calibration_config=CalibrationConfig.load(Path("config/experiment-002.toml")),
        calibration_sha256="abc",
        audio_metadata={"active": True},
        camera={"index": 0},
        duration_seconds=30.0,
        frames_total=100,
        frames_usable=80,
        placement_invalid_frames=10,
        gaze_unusable_frames=10,
        raw_outside_prediction_frames=4,
        correction_evaluated_frames=50,
        raw_outside_frames_with_correction=4,
        corrected_outside_prediction_frames=1,
        corrector=None,
        completed=False,
        aborted=True,
    )

    measurement = report["measurement"]
    assert measurement["correct_count"] == 1
    assert measurement["incorrect_count"] == 1
    assert measurement["no_selection_count"] == 1
    assert measurement["accuracy_all_trials"] == pytest.approx(1 / 3)
    assert measurement["accuracy_when_selected"] == pytest.approx(0.5)
    assert measurement["dwell_cancellation_count"] == 3
    assert measurement["practice_trial_count"] == 1
    assert measurement["scored_trial_count"] == 3
    assert measurement["correction_comparison"]["corrected_outside_fraction"] == 0.02
    assert measurement["confusion_matrix"]["TOP_RIGHT"]["BOTTOM_RIGHT"] == 1
