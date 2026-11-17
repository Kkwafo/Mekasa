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
from mekasa_gaze.ui.board_view import board_area, render_board, screen_to_board, status_text

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def validity_config() -> GuidedValidationConfig:
    return GuidedValidationConfig.load(ROOT / "config" / "experiment-001.toml")


@pytest.fixture(scope="module")
def communication() -> CommunicationConfig:
    # The scripted replay is timed for the 0.9 s dwell path.
    return replace(
        CommunicationConfig.load(ROOT / "config" / "communication.toml"), selection_mode="dwell"
    )


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
    area = board_area(1280, 720)
    status_point = screen_to_board((0.2, 0.03), area)
    message_point = screen_to_board((0.2, 0.12), area)
    assert status_point is not None and status_point[1] < 0
    assert message_point is not None and message_point[1] < 0
    # The board reaches the bottom of the screen (largest vertical angle between rows).
    assert area[3] >= 0.97


def test_render_smoke(communication: CommunicationConfig) -> None:
    layout = communication.board()
    for state in SessionState:
        update = SessionUpdate(
            state=state, focused=layout.options[0], progress=0.5, message="Quiero agua"
        )
        canvas = render_board(640, 360, layout, update, eye_mode=EyeMode.LEFT, speech_ok=False)
        assert canvas.shape == (360, 640, 3)
        assert status_text(update, EyeMode.LEFT)


def test_calibration_paths_prefer_session_models_with_fallbacks(tmp_path: Path) -> None:
    (tmp_path / "personal-gaze-model.json").write_text("{}", encoding="utf-8")
    (tmp_path / "session-gaze-model-left.json").write_text("{}", encoding="utf-8")
    paths = calibration_paths(tmp_path)
    assert paths[EyeMode.BOTH].name == "personal-gaze-model.json"
    assert paths[EyeMode.LEFT].name == "session-gaze-model-left.json"
    assert paths[EyeMode.RIGHT].name == "personal-gaze-model-right.json"


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


# ------------------------------------------------------------------ session alignment & UI


def test_alignment_learns_offset_and_corrects_predictions() -> None:
    from mekasa_gaze.tracking.session_alignment import AlignmentProcedure

    targets = [(0.25, 0.3), (0.75, 0.3), (0.25, 0.7), (0.75, 0.7)]
    procedure = AlignmentProcedure(targets, settle_seconds=0.3, collect_seconds=0.5)
    now = 0.0
    snapshot = procedure.update(now, None)
    while not snapshot.done:
        now += 1 / 30
        target = snapshot.target or targets[-1]
        # Drifted model: shifted right/down and compressed.
        snapshot = procedure.update(now, (0.1 + 0.6 * target[0], 0.2 + 0.6 * target[1]))
    assert snapshot.correction is not None
    corrected = snapshot.correction.predict((0.1 + 0.6 * 0.25, 0.2 + 0.6 * 0.7))
    assert corrected == pytest.approx((0.25, 0.7), abs=0.03)
    assert now < 6.0  # ~ 4 x (0.3 + 0.5) s


def test_alignment_without_signal_gives_no_correction() -> None:
    from mekasa_gaze.tracking.session_alignment import AlignmentProcedure

    procedure = AlignmentProcedure([(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)], max_collect_seconds=1.0)
    now, snapshot = 0.0, procedure.update(0.0, None)
    while not snapshot.done:
        now += 1 / 30
        snapshot = procedure.update(now, None)
    assert snapshot.correction is None
    assert "no disponible" in snapshot.message


def test_board_and_alignment_render_with_unicode(communication: CommunicationConfig) -> None:
    from mekasa_gaze.tracking.session_alignment import AlignmentSnapshot
    from mekasa_gaze.ui.board_view import render_alignment

    layout = communication.board()
    update = SessionUpdate(
        state=SessionState.FOCUSING,
        focused=layout.options[0],
        progress=0.6,
        message="Sí, quiero agua",
    )
    canvas = render_board(
        1280,
        720,
        layout,
        update,
        eye_mode=EyeMode.BOTH,
        eyes=(True, False),
        hint="Acérquese un poco",
        aligned=True,
    )
    assert canvas.shape == (720, 1280, 3)
    assert canvas.std() > 10  # something was drawn
    snap = AlignmentSnapshot(1, 4, (0.25, 0.3), True, 0.5, False)
    assert render_alignment(1280, 720, snap, signal_ok=False).shape == (720, 1280, 3)


def test_pipeline_applies_personal_closed_eye_threshold(
    validity_config: GuidedValidationConfig,
) -> None:
    from dataclasses import replace as dc_replace

    from mekasa_gaze.runners.board_simulation import synthetic_model, synthetic_observation
    from mekasa_gaze.runners.communication_board import GazePipeline

    observation = synthetic_observation((0.3, 0.8), "ok", 1, 10.0)
    features = dict(observation.features)
    features.update(left_eye_aspect_ratio=0.12, right_eye_aspect_ratio=0.12)  # looking down
    lowered = dc_replace(observation, features=features)

    def predict(threshold: float | None) -> tuple[float, float] | None:
        model = dc_replace(synthetic_model(EyeMode.BOTH), closed_ear_threshold=threshold)
        pipeline = GazePipeline(
            extractor=None,  # type: ignore[arg-type]
            validity_config=validity_config,
            model=model,
            eye_mode=EyeMode.BOTH,
            smoothing_alpha=1.0,
            area=(0.0, 0.0, 1.0, 1.0),
        )
        return pipeline.process_observation(lowered, frame_timestamp_seconds=10.0).screen_point

    assert predict(None) is None  # population threshold 0.15 calls this "closed"
    assert predict(0.09) == pytest.approx((0.3, 0.8))
