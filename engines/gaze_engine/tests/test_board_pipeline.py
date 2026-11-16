"""Integration: synthetic features -> validity -> estimator -> dwell -> message -> speech."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from time import sleep

import numpy as np
import pytest

from mekasa_gaze.core.board import CommunicationConfig
from mekasa_gaze.core.config import GuidedValidationConfig
from mekasa_gaze.core.domain import CameraFrame, CameraSettings
from mekasa_gaze.core.eye_mode import EyeMode
from mekasa_gaze.runners.board_simulation import (
    FakeSpeech,
    Segment,
    default_script,
    run_default_simulation,
    simulate,
)
from mekasa_gaze.runners.communication_board import calibration_paths
from mekasa_gaze.tracking.communication import SessionState, SessionUpdate
from mekasa_gaze.tracking.frame_source import LatestFrameGrabber
from mekasa_gaze.ui.board_view import render_board, screen_to_board, status_text

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def validity_config() -> GuidedValidationConfig:
    return GuidedValidationConfig.load(ROOT / "config" / "experiment-001.toml")


@pytest.fixture(scope="module")
def communication() -> CommunicationConfig:
    return CommunicationConfig.load(ROOT / "config" / "communication.toml")


@pytest.mark.parametrize("mode", list(EyeMode))
def test_default_replay_communicates_and_never_emits_under_unsafe_conditions(
    mode: EyeMode,
    communication: CommunicationConfig,
    validity_config: GuidedValidationConfig,
) -> None:
    speech = FakeSpeech()
    report = simulate(
        default_script(mode),
        replace(communication, eye_mode=mode),
        validity_config,
        speech=speech,
    )
    emitted = [segment["emitted"] for segment in report["segments"]]
    assert emitted[0] == ["YES"]
    assert emitted[2] == ["WATER"]
    assert report["safety"]["passed"], report
    assert emitted[10] == ["HELP"]
    if mode is not EyeMode.BOTH:
        assert emitted[11] == ["NO"]
    assert speech.spoken[:2] == ["Sí", "Quiero agua"]


def test_uncalibrated_board_is_silent(
    communication: CommunicationConfig, validity_config: GuidedValidationConfig
) -> None:
    report = simulate(
        [Segment(10.0, "YES", expect_emissions=False)],
        communication,
        validity_config,
        calibrated_modes=(),
    )
    assert report["safety"]["passed"]
    assert report["session"]["ticks_by_state"] == {"NO_CALIBRATION": 300}


def test_speech_failure_keeps_message_visible(
    communication: CommunicationConfig, validity_config: GuidedValidationConfig
) -> None:
    report = simulate(
        [Segment(2.0, "HELP")],
        communication,
        validity_config,
        speech=FakeSpeech(fail=True),
    )
    assert report["segments"][0]["emitted"] == ["HELP"]
    assert report["segments"][0]["spoken"] == 0
    assert report["message_on_screen"] == "Necesito ayuda"


def test_suite_passes(
    communication: CommunicationConfig, validity_config: GuidedValidationConfig
) -> None:
    assert run_default_simulation(communication, validity_config)["safety"]["passed"]


def test_status_and_message_bars_never_select() -> None:
    area = (0.02, 0.09, 0.98, 0.85)
    status_point = screen_to_board((0.2, 0.04), area)
    message_point = screen_to_board((0.2, 0.95), area)
    assert status_point is not None and status_point[1] < 0
    assert message_point is not None and message_point[1] > 1


def test_render_smoke(communication: CommunicationConfig) -> None:
    layout = communication.board()
    for state in SessionState:
        update = SessionUpdate(
            state=state, focused=layout.options[0], progress=0.5, message="Quiero agua"
        )
        canvas = render_board(640, 360, layout, update, eye_mode=EyeMode.LEFT, speech_ok=False)
        assert canvas.shape == (360, 640, 3)
        assert status_text(update, EyeMode.LEFT).startswith("[ojo izq.]")


def test_calibration_paths_prefer_mode_files_with_legacy_fallback(tmp_path: Path) -> None:
    (tmp_path / "personal-gaze-model.json").write_text("{}", encoding="utf-8")
    paths = calibration_paths(tmp_path)
    assert paths[EyeMode.BOTH].name == "personal-gaze-model.json"
    assert paths[EyeMode.LEFT].name == "personal-gaze-model-left.json"


# ------------------------------------------------------------------ capture thread


class _FlakyCamera:
    """Delivers ``frames`` frames, then raises like a disconnected USB camera."""

    opened = 0

    def __init__(self, frames: int) -> None:
        self._frames = frames
        self._sequence = 0

    @property
    def actual_settings(self) -> CameraSettings:
        return CameraSettings(0, "FAKE", 4, 4, 30.0)

    def open(self) -> None:
        type(self).opened += 1

    def read(self) -> CameraFrame:
        if self._sequence >= self._frames:
            raise RuntimeError("device lost")
        self._sequence += 1
        sleep(0.002)
        return CameraFrame(self._sequence, 0, 0.0, np.zeros((4, 4, 3), dtype=np.uint8))

    def close(self) -> None:
        return None


def test_grabber_keeps_latest_frame_and_reconnects() -> None:
    _FlakyCamera.opened = 0
    grabber = LatestFrameGrabber(lambda: _FlakyCamera(5), retry_seconds=0.01)
    seen: set[int] = set()
    done = threading.Event()
    with grabber:
        for _ in range(400):
            frame = grabber.take_new()
            if frame is not None:
                assert frame.sequence not in seen  # never the same frame twice
                seen.add(frame.sequence)
            if _FlakyCamera.opened >= 3:
                done.set()
                break
            sleep(0.005)
    status = grabber.status()
    assert done.is_set()
    assert status.reconnect_attempts >= 2
    assert status.last_error is not None and "device lost" in status.last_error
    assert status.frames_captured >= 10
