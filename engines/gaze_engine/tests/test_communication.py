"""Deterministic safety tests for the communication session (no camera, injected clock)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mekasa_gaze.core.board import CommunicationConfig, binary_board, quadrant_board
from mekasa_gaze.core.domain import EyeState
from mekasa_gaze.core.eye_mode import (
    EyeMode,
    eye_mode_of_features,
    eye_signal_usable,
    feature_names_for_mode,
)
from mekasa_gaze.tracking.communication import (
    AbstainReason,
    CommunicationSession,
    GazeInput,
    SessionState,
)

YES = (0.2, 0.2)
NO = (0.8, 0.2)
WATER = (0.2, 0.8)
HELP = (0.8, 0.8)
GAP = (0.5, 0.2)  # between YES and NO, inside YES' hysteresis margin
CENTER = (0.55, 0.5)  # in the cross-shaped gap, outside every option's margin
DT = 1.0 / 30.0
OPEN = EyeState.OPEN


def _config(**overrides: object) -> CommunicationConfig:
    base = CommunicationConfig(
        dwell_seconds=0.9,
        cooldown_seconds=0.5,
        hysteresis_margin=0.03,
        recovery_seconds=0.3,
        max_frame_age_seconds=0.25,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def _session(**overrides: object) -> CommunicationSession:
    config = _config(**overrides)
    return CommunicationSession(config, calibration_mode=config.eye_mode)


def _tick(
    now: float,
    point: tuple[float, float] | None,
    *,
    frame_age: float | None = 0.01,
    tracking_ok: bool = True,
    left: EyeState = OPEN,
    right: EyeState = OPEN,
) -> GazeInput:
    return GazeInput(
        now_seconds=now,
        frame_timestamp_seconds=None if frame_age is None else now - frame_age,
        prediction=point,
        tracking_ok=tracking_ok,
        left_eye=left,
        right_eye=right,
    )


class Clock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def run(
        self,
        session: CommunicationSession,
        point: tuple[float, float] | None,
        seconds: float,
        **kwargs: object,
    ) -> list[str]:
        """Feed ``seconds`` of 30 fps ticks; return keys of confirmed options."""
        confirmed: list[str] = []
        steps = round(seconds / DT)
        for _ in range(steps):
            self.now += DT
            update = session.update(_tick(self.now, point, **kwargs))  # type: ignore[arg-type]
            if update.confirmed is not None:
                confirmed.append(update.confirmed.key)
        return confirmed


# ---------------------------------------------------------------- happy path


def test_dwell_confirms_once_and_requires_leaving_before_rearming() -> None:
    session = _session()
    clock = Clock()

    assert clock.run(session, YES, 0.3) == []  # recovery after start-up
    assert clock.run(session, YES, 1.0) == ["YES"]
    # Staring at the same option never repeats the message.
    assert clock.run(session, YES, 5.0) == []
    assert session.update(_tick(clock.now + DT, YES)).state is SessionState.AWAITING_EXIT
    clock.now += DT
    # Leaving re-arms; a second deliberate dwell confirms again.
    assert clock.run(session, NO, 1.0) == ["NO"]
    assert session.message == "No"
    assert dict(session.stats.confirmations_by_source) == {"gaze": 2}


def test_focus_and_progress_are_reported_before_confirmation() -> None:
    session = _session()
    clock = Clock()
    clock.run(session, YES, 0.3)
    clock.now += 0.45
    update = session.update(_tick(clock.now, YES))
    assert update.state is SessionState.FOCUSING
    assert update.focused is not None and update.focused.key == "YES"
    assert update.confirmed is None
    assert 0.0 <= update.progress < 1.0


def test_hysteresis_keeps_focus_on_boundary_jitter() -> None:
    session = _session()
    clock = Clock()
    clock.run(session, YES, 0.3)
    confirmed: list[str] = []
    for step in range(40):
        clock.now += DT
        point = (0.47, 0.2) if step % 2 else (0.45, 0.2)  # 0.47 is in the gap, within margin
        update = session.update(_tick(clock.now, point))
        if update.confirmed:
            confirmed.append(update.confirmed.key)
    assert confirmed == ["YES"]


def test_gap_between_options_never_selects() -> None:
    session = _session()
    assert Clock().run(session, GAP, 10.0) == []


@pytest.mark.parametrize("dwell", [0.45, 0.9, 1.2])
def test_dwell_hypotheses_fire_at_configured_time(dwell: float) -> None:
    session = _session(dwell_seconds=dwell, recovery_seconds=0.0)
    clock = Clock()
    fired_at: float | None = None
    for _ in range(200):
        clock.now += DT
        if session.update(_tick(clock.now, HELP)).confirmed is not None:
            fired_at = clock.now
            break
    assert fired_at is not None
    assert fired_at - (100.0 + DT) == pytest.approx(dwell, abs=DT + 1e-9)


# ---------------------------------------------------------------- zero-emission gates


def test_no_calibration_never_emits_from_gaze() -> None:
    session = CommunicationSession(_config(), calibration_mode=None)
    clock = Clock()
    assert clock.run(session, YES, 10.0) == []
    assert session.update(_tick(clock.now + DT, YES)).state is SessionState.NO_CALIBRATION


def test_calibration_for_another_eye_mode_never_emits() -> None:
    session = CommunicationSession(_config(eye_mode=EyeMode.LEFT), calibration_mode=EyeMode.BOTH)
    assert Clock().run(session, YES, 10.0) == []


def test_disconnected_camera_never_emits() -> None:
    session = _session()
    clock = Clock()
    assert clock.run(session, YES, 10.0, frame_age=None) == []
    update = session.update(_tick(clock.now + DT, YES, frame_age=None))
    assert update.abstain_reason is AbstainReason.NO_FRAME


def test_stale_frames_never_emit() -> None:
    session = _session()
    assert Clock().run(session, YES, 10.0, frame_age=0.5) == []
    assert session.stats.abstentions_by_reason["STALE_FRAME"] > 0


def test_invalid_tracking_never_emits() -> None:
    session = _session()
    assert Clock().run(session, YES, 10.0, tracking_ok=False) == []


def test_pause_never_emits_and_resume_requires_recovery() -> None:
    session = _session()
    clock = Clock()
    clock.run(session, YES, 0.6)  # mid-dwell
    session.pause()
    assert clock.run(session, YES, 10.0) == []
    session.resume()
    update = session.update(_tick(clock.now + DT, YES))
    clock.now += DT
    assert update.state is SessionState.RECOVERING
    assert clock.run(session, YES, 1.3) == ["YES"]


def test_signal_loss_cancels_dwell_in_progress() -> None:
    session = _session()
    clock = Clock()
    clock.run(session, YES, 0.3 + 0.8)  # 0.8 s of a 0.9 s dwell
    clock.run(session, YES, 0.1, tracking_ok=False)
    # After recovery the dwell restarts from zero: 0.5 s is not enough.
    assert clock.run(session, YES, 0.3 + 0.5) == []
    assert session.stats.focus_cancellations >= 1
    assert session.stats.signal_loss_events == 1


def test_mode_change_flushes_and_disables_until_matching_calibration() -> None:
    session = _session()
    clock = Clock()
    clock.run(session, YES, 1.0)  # 0.7 s into the dwell
    session.set_eye_mode(EyeMode.RIGHT)
    assert clock.run(session, YES, 5.0) == []
    session.set_calibration(EyeMode.RIGHT)
    assert clock.run(session, YES, 1.3, left=EyeState.CLOSED) == ["YES"]


# ---------------------------------------------------------------- eye modes


def test_both_mode_abstains_when_one_eye_is_missing() -> None:
    session = _session(eye_mode=EyeMode.BOTH)
    assert Clock().run(session, YES, 5.0, right=EyeState.INVALID) == []
    assert session.stats.abstentions_by_reason["EYE_SIGNAL"] > 0


@pytest.mark.parametrize(
    ("mode", "left", "right", "expected"),
    [
        (EyeMode.LEFT, OPEN, EyeState.CLOSED, ["YES"]),
        (EyeMode.LEFT, EyeState.INVALID, OPEN, []),
        (EyeMode.RIGHT, EyeState.CLOSED, OPEN, ["YES"]),
        (EyeMode.RIGHT, OPEN, EyeState.CLOSED, []),
    ],
)
def test_single_eye_modes_depend_only_on_their_eye(
    mode: EyeMode, left: EyeState, right: EyeState, expected: list[str]
) -> None:
    session = _session(eye_mode=mode)
    assert Clock().run(session, YES, 1.3, left=left, right=right) == expected


def test_feature_selection_per_eye_mode() -> None:
    names = (
        "left_iris_x",
        "left_iris_y",
        "right_iris_x",
        "right_iris_y",
        "head_yaw_degrees",
    )
    assert feature_names_for_mode(names, EyeMode.BOTH) == names
    assert feature_names_for_mode(names, EyeMode.LEFT) == (
        "left_iris_x",
        "left_iris_y",
        "head_yaw_degrees",
    )
    assert eye_mode_of_features(feature_names_for_mode(names, EyeMode.RIGHT)) is EyeMode.RIGHT
    with pytest.raises(ValueError):
        feature_names_for_mode(("head_yaw_degrees",), EyeMode.LEFT)
    assert not eye_signal_usable(EyeMode.BOTH, OPEN, EyeState.CLOSED)


# ---------------------------------------------------------------- confirmation & direct access


def test_optional_confirmation_requires_second_dwell_on_same_option() -> None:
    session = _session(confirmation_required=True, confirmation_window_seconds=4.0)
    clock = Clock()
    assert clock.run(session, YES, 1.3) == []
    update = session.update(_tick(clock.now + DT, YES))
    clock.now += DT
    assert update.pending is not None and update.pending.key == "YES"
    # Staying inside the hysteresis margin is not "leaving".
    assert clock.run(session, GAP, 0.3) == []
    assert session.update(_tick(clock.now + DT, YES)).state is SessionState.AWAITING_EXIT
    clock.now += DT
    assert clock.run(session, CENTER, 0.3) == []
    assert clock.run(session, YES, 1.0) == ["YES"]


def test_pending_confirmation_expires_and_is_flushed_by_signal_loss() -> None:
    session = _session(confirmation_required=True, confirmation_window_seconds=1.0)
    clock = Clock()
    clock.run(session, YES, 1.3)
    clock.run(session, CENTER, 1.5)  # window expires
    assert clock.run(session, YES, 1.0) == []  # first dwell again, not a confirmation
    clock.run(session, GAP, 0.2, frame_age=None)  # signal loss discards the new pending
    assert clock.run(session, YES, 1.3) == []
    assert session.stats.pending_discarded >= 2


def test_direct_access_works_without_calibration_and_holds_gaze() -> None:
    session = CommunicationSession(_config(), calibration_mode=None)
    update = session.select_direct("HELP", 50.0, source="keyboard")
    assert update.confirmed is not None and update.confirmed.key == "HELP"
    assert update.message == "Necesito ayuda"
    assert session.message == "Necesito ayuda"


def test_message_survives_pause_and_signal_loss() -> None:
    session = _session()
    clock = Clock()
    clock.run(session, WATER, 1.3)
    session.pause()
    clock.run(session, None, 1.0, frame_age=None)
    assert session.message == "Quiero agua"


# ---------------------------------------------------------------- layouts and config


def test_layouts_have_gaps_and_binary_variants() -> None:
    quadrant = quadrant_board()
    assert [option.label for option in quadrant.options] == ["SÍ", "NO", "AGUA", "AYUDA"]
    assert quadrant.option_at(GAP) is None
    vertical = binary_board(orientation="vertical")
    top = vertical.option_at((0.5, 0.2))
    bottom = vertical.option_at((0.5, 0.8))
    assert top is not None and top.key == "YES"
    assert bottom is not None and bottom.key == "NO"


def test_config_file_loads_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "communication.toml"
    path.write_text(
        '[communication]\ndwell_seconds = 1.2\neye_mode = "left"\nlayout = "binary-vertical"\n'
        '[phrases.YES]\nlabel = "SI"\nphrase = "Si, gracias"\n',
        encoding="utf-8",
    )
    config = CommunicationConfig.load(path)
    assert config.eye_mode is EyeMode.LEFT
    assert config.board().option("YES").phrase == "Si, gracias"
    path.write_text("[communication]\ndwell_seconds = 0.05\n", encoding="utf-8")
    with pytest.raises(ValueError):
        CommunicationConfig.load(path)


def test_repository_communication_config_is_valid() -> None:
    root = Path(__file__).resolve().parents[3]
    config = CommunicationConfig.load(root / "config" / "communication.toml")
    assert config.dwell_seconds == pytest.approx(0.9)


def test_stats_omit_per_option_counts_by_default() -> None:
    session = _session()
    Clock().run(session, YES, 1.3)
    assert "confirmations_by_option" not in session.stats.as_dict()
    assert session.stats.as_dict(include_options=True)["confirmations_by_option"] == {"YES": 1}
