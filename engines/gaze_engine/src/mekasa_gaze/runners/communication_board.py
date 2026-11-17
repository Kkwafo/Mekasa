"""Mekasa communication board: camera -> quality -> estimator -> dwell -> message -> speech.

Run ``uv run mekasa-board`` (see README). A session normally starts with a ~35 s calibration
(a calibration from another sitting does not transfer; see tracking/gaze_model.py), shows the
quality per eye mode, then opens the board. While the board is in use it compensates for
viewing distance, re-aligns automatically after a clear change of position and learns lightly
from each gaze selection.

Without a usable calibration the board still opens and works through direct access (keys
1-4, mouse click) with gaze selection disabled, so a person is never left without a way to
communicate while setup is incomplete.

The default interface is the conversation board (tracking/conversation.py): SÍ / NO /
HABLAR / URGENTE at home, menus from config/vocabulario.txt to compose a full message, and
DECIR to speak it. ``--interface simple`` keeps the original SÍ / NO / AGUA / AYUDA board.

Keys: 1-4 select directly - ENTER confirm pending option - BACKSPACE back - H home -
SPACE pause/resume - A re-align - K recalibrate - M cycle eye mode - G show/hide gaze point -
C clear message - ESC/Q quit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import deque
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

import cv2
import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.board import BoardOption, CommunicationConfig
from mekasa_gaze.core.config import GuidedValidationConfig
from mekasa_gaze.core.domain import CameraFrame, EyeState, FeatureObservation, TrackingState
from mekasa_gaze.core.eye_mode import EyeMode, eye_signal_usable
from mekasa_gaze.core.interfaces import FeatureExtractor, SpeechBackend
from mekasa_gaze.core.vocabulary import load_vocabulary
from mekasa_gaze.runners.live_evaluation import PredictionSmoother
from mekasa_gaze.services.alarm import play_alarm
from mekasa_gaze.services.runtime import environment, source_revision
from mekasa_gaze.services.speech import create_speech_backend
from mekasa_gaze.tracking.calibration import load_calibration_model
from mekasa_gaze.tracking.communication import CommunicationSession, GazeInput, SessionUpdate
from mekasa_gaze.tracking.conversation import Navigator
from mekasa_gaze.tracking.gaze_model import (
    FitResult,
    fit_personal_model,
    load_personal_model,
    observation_features,
    save_personal_model,
)
from mekasa_gaze.tracking.placement import (
    PlacementAssessment,
    PlacementState,
    assess_face_placement,
)
from mekasa_gaze.tracking.placement_monitor import (
    DRIFT_HINTS,
    Drift,
    PlacementMonitor,
    PlacementStatus,
    compensate_distance,
)
from mekasa_gaze.tracking.session_alignment import (
    AlignmentProcedure,
    AlignmentSnapshot,
    OnlineCorrection,
)
from mekasa_gaze.tracking.session_calibration import (
    CalibrationProcedure,
    calibration_steps,
    estimated_seconds,
)
from mekasa_gaze.tracking.validity import ValidityAssessment, assess_gaze_validity
from mekasa_gaze.ui.board_view import (
    board_area,
    board_to_screen,
    option_centers,
    render_alignment,
    render_board,
    render_calibration,
    render_summary,
    screen_to_board,
)

_WINDOW_NAME = "Mekasa - Tablero de comunicacion"
_PLACEMENT_HINTS = {
    PlacementState.TOO_FAR: "Acérquese un poco a la cámara",
    PlacementState.TOO_CLOSE: "Aléjese un poco de la cámara",
    PlacementState.NO_FACE: "Ubíquese frente a la cámara",
}
_MODE_NAMES = {
    EyeMode.BOTH: "Ambos ojos",
    EyeMode.LEFT: "Ojo izquierdo",
    EyeMode.RIGHT: "Ojo derecho",
}
_KIND_NAMES = {"landmarks": "puntos", "appearance": "imagen", "hybrid": "imagen + puntos"}
_GOOD_QUALITY = 0.9
_AUTO_SWITCH_MARGIN = 0.10
_REALIGN_AFTER_SECONDS = 2.5
_REALIGN_MIN_INTERVAL = 20.0
_TEST_TIMEOUT_SECONDS = 8.0
_TEST_PAUSE_SECONDS = 1.5
_SUMMARY_AUTO_CONTINUE = 6.0
# Calibration collection accepts low lids (downward gaze); personal limits are fitted later.
_COLLECTION_CLOSED_EAR = 0.06
_COLLECTION_MIN_SPAN = 0.003
_NO_DRIFT = PlacementStatus(Drift.OK, 1.0, 0.0)


class GazeEstimator(Protocol):
    """Anything that maps an observation to a screen point for one eye mode."""

    @property
    def eye_mode(self) -> EyeMode: ...

    @property
    def closed_ear_threshold(self) -> float | None: ...

    def predict_observation(
        self,
        observation: FeatureObservation,
        *,
        left_eye_state: EyeState,
        right_eye_state: EyeState,
    ) -> tuple[float, float] | None: ...


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Everything derived from one camera frame. Holds no image data."""

    frame_timestamp_seconds: float
    observation: FeatureObservation
    validity: ValidityAssessment
    placement: PlacementAssessment
    screen_point: tuple[float, float] | None
    raw_screen_point: tuple[float, float] | None
    board_point: tuple[float, float] | None
    tracking_ok: bool
    processing_ms: float
    placement_status: PlacementStatus = _NO_DRIFT

    @property
    def any_eye_open(self) -> bool:
        return EyeState.OPEN in (self.validity.left_eye_state, self.validity.right_eye_state)


class GazePipeline:
    """Frame -> features -> quality gates -> estimate -> distance compensation -> smoothing ->
    session correction -> board coordinates."""

    def __init__(
        self,
        *,
        extractor: FeatureExtractor,
        validity_config: GuidedValidationConfig,
        model: GazeEstimator | None,
        eye_mode: EyeMode,
        smoothing_alpha: float,
        area: tuple[float, float, float, float],
        distance_compensation: bool = False,
        camera_position: tuple[float, float] = (0.5, -0.02),
    ) -> None:
        self._extractor = extractor
        self._base_validity_config = validity_config
        self._validity_config = _personal_validity(validity_config, model)
        self._model = model
        self._eye_mode = eye_mode
        self._smoother = PredictionSmoother(smoothing_alpha)
        self._area = area
        self._correction: Any = None
        self._distance_compensation = distance_compensation
        self._camera = camera_position
        self.monitor = PlacementMonitor(reference=_reference_of(model))

    @property
    def closed_ear_threshold(self) -> float:
        """Closed-eye EAR threshold in effect (personal if the calibration carries one)."""
        return self._validity_config.eye_closed_ear_threshold

    @property
    def eye_mode(self) -> EyeMode:
        """Eye mode used to gate estimation."""
        return self._eye_mode

    @property
    def model(self) -> GazeEstimator | None:
        return self._model

    def set_correction(self, correction: Any) -> None:
        """Install (or clear) the session correction applied after smoothing."""
        self._correction = correction

    def set_model(self, mode: EyeMode, model: GazeEstimator | None) -> None:
        """Switch eye mode and model; resets smoothing, correction and placement reference."""
        self._eye_mode = mode
        self._model = model
        self._validity_config = _personal_validity(self._base_validity_config, model)
        self._correction = None
        self._smoother.update(None)
        self.monitor.rebase(_reference_of(model))

    # Backwards-compatible name used by older callers and tests.
    def set_eye_mode(self, mode: EyeMode, model: GazeEstimator | None) -> None:
        self.set_model(mode, model)

    def process(self, frame: CameraFrame) -> PipelineResult:
        """Run one frame through the pipeline."""
        started = perf_counter()
        rgb = cast(npt.NDArray[np.uint8], cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB))
        observation = self._extractor.extract(
            rgb, sequence=frame.sequence, timestamp_ns=frame.captured_at_ns
        )
        return self.process_observation(
            observation,
            frame_timestamp_seconds=frame.captured_at_ns / 1e9,
            started=started,
        )

    def process_observation(
        self,
        observation: FeatureObservation,
        *,
        frame_timestamp_seconds: float,
        started: float | None = None,
    ) -> PipelineResult:
        """Run the post-landmark stages (also used by replay and simulation)."""
        t0 = perf_counter() if started is None else started
        validity = assess_gaze_validity(observation, self._validity_config)
        placement = assess_face_placement(
            observation,
            min_face_width=self._validity_config.min_face_width,
            max_face_width=self._validity_config.max_face_width,
        )
        tracking_ok = observation.tracking_state is TrackingState.GOOD and placement.ready
        status = self.monitor.update(
            frame_timestamp_seconds, observation_features(observation) if tracking_ok else None
        )
        raw: tuple[float, float] | None = None
        if (
            tracking_ok
            and self._model is not None
            and self._model.eye_mode is self._eye_mode
            and eye_signal_usable(self._eye_mode, validity.left_eye_state, validity.right_eye_state)
        ):
            try:
                raw = self._model.predict_observation(
                    observation,
                    left_eye_state=validity.left_eye_state,
                    right_eye_state=validity.right_eye_state,
                )
            except ValueError:
                raw = None
        if raw is not None and self._distance_compensation:
            raw = compensate_distance(raw, status.distance_scale, self._camera)
        smoothed = self._smoother.update(raw)
        screen = smoothed
        if smoothed is not None and self._correction is not None:
            screen = self._correction.predict(smoothed)
        return PipelineResult(
            frame_timestamp_seconds=frame_timestamp_seconds,
            observation=observation,
            validity=validity,
            placement=placement,
            screen_point=screen,
            raw_screen_point=smoothed,
            board_point=screen_to_board(screen, self._area),
            tracking_ok=tracking_ok,
            processing_ms=(perf_counter() - t0) * 1000.0,
            placement_status=status,
        )


def _reference_of(model: GazeEstimator | None) -> Any:
    return getattr(model, "reference", None)


def _personal_validity(
    config: GuidedValidationConfig, model: GazeEstimator | None
) -> GuidedValidationConfig:
    """Personal eye-closure limits from the calibration; lenient limits while collecting."""
    if model is None:
        return replace(
            config,
            eye_closed_ear_threshold=min(config.eye_closed_ear_threshold, _COLLECTION_CLOSED_EAR),
            min_eye_vertical_span=min(config.min_eye_vertical_span, _COLLECTION_MIN_SPAN),
        )
    updated = config
    if model.closed_ear_threshold is not None:
        updated = replace(updated, eye_closed_ear_threshold=model.closed_ear_threshold)
    span = getattr(model, "min_vertical_span", None)
    if span is not None:
        updated = replace(updated, min_eye_vertical_span=float(span))
    return updated


def gaze_input(result: PipelineResult | None, now: float, camera_connected: bool) -> GazeInput:
    """Reduce the latest pipeline result to a session tick.

    When no new frame arrived, the previous result is reused with its *original* timestamp,
    so the session's frame-age gate stops selection as the frame gets old.
    """
    if result is None or not camera_connected:
        return GazeInput(now, None, None, False, EyeState.INVALID, EyeState.INVALID)
    return GazeInput(
        now_seconds=now,
        frame_timestamp_seconds=result.frame_timestamp_seconds,
        prediction=result.board_point,
        tracking_ok=result.tracking_ok,
        left_eye=result.validity.left_eye_state,
        right_eye=result.validity.right_eye_state,
    )


def speak_safely(backend: SpeechBackend, phrase: str) -> bool:
    """Speak ``phrase``; never raise. Returns False if speech is unavailable or failed."""
    if not backend.active:
        return False
    try:
        backend.speak(phrase)
    except Exception:  # a speech failure must never block or hide the message
        return False
    return True


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values), q)) if values else None


def load_estimator(path: Path) -> GazeEstimator:
    """Load a session model (current format) or a research calibration (legacy format)."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and str(data.get("schema", "")).startswith("mekasa.personal_gaze"):
        return load_personal_model(path)
    return cast(GazeEstimator, load_calibration_model(path))


class _Models:
    """Per-eye-mode estimators (only valid ones are kept)."""

    def __init__(self, paths: dict[EyeMode, Path]) -> None:
        self.models: dict[EyeMode, GazeEstimator] = {}
        self.errors: dict[str, str] = {}
        for mode, path in paths.items():
            if not path.exists():
                continue
            try:
                model = load_estimator(path)
            except (KeyError, ValueError, TypeError, json.JSONDecodeError, OSError) as error:
                self.errors[str(path)] = str(error)
                continue
            if model.eye_mode is not mode:
                self.errors[str(path)] = (
                    f"trained for {model.eye_mode.value}, not {mode.value}; ignored"
                )
                continue
            self.models[mode] = model

    def get(self, mode: EyeMode) -> GazeEstimator | None:
        return self.models.get(mode)


def calibration_paths(directory: Path) -> dict[EyeMode, Path]:
    """Map each eye mode to its model file: session model first, then research calibrations."""
    paths: dict[EyeMode, Path] = {}
    for mode in EyeMode:
        session = directory / f"session-gaze-model-{mode.value.lower()}.json"
        research = directory / f"personal-gaze-model-{mode.value.lower()}.json"
        paths[mode] = session if session.exists() else research
    legacy = directory / "personal-gaze-model.json"
    if not paths[EyeMode.BOTH].exists() and legacy.exists():
        paths[EyeMode.BOTH] = legacy
    return paths


def session_model_path(directory: Path, mode: EyeMode) -> Path:
    return directory / f"session-gaze-model-{mode.value.lower()}.json"


# ----------------------------------------------------------------------------- calibration


def choose_eye_mode(
    configured: EyeMode, results: dict[EyeMode, FitResult]
) -> tuple[EyeMode, str | None]:
    """Keep the configured mode unless it is weak and another mode is clearly better."""

    def quality(mode: EyeMode) -> float:
        result = results.get(mode)
        if result is None or result.model is None:
            return -1.0
        value = result.model.metrics.get("quadrant_accuracy")
        return float(value) if value is not None else -1.0

    current = quality(configured)
    best = max(EyeMode, key=quality)
    if (
        best is not configured
        and current < _GOOD_QUALITY
        and quality(best) >= current + _AUTO_SWITCH_MARGIN
    ):
        return best, f"Se usa {_MODE_NAMES[best].lower()}: funcionó mejor en la calibración"
    return configured, None


def calibration_summary(
    results: dict[EyeMode, FitResult], active: EyeMode
) -> list[tuple[str, bool | None]]:
    """Lines for the summary screen."""
    lines: list[tuple[str, bool | None]] = []
    for mode in EyeMode:
        result = results.get(mode)
        mark = "→ " if mode is active else "   "
        if result is None or result.model is None:
            reason = result.error if result is not None and result.error else "sin datos"
            lines.append((f"{mark}{_MODE_NAMES[mode]}: no disponible ({reason})", False))
            continue
        metrics = result.model.metrics
        quadrant = metrics.get("quadrant_accuracy")
        per = metrics.get("per_quadrant", {})
        if quadrant is None:
            lines.append((f"{mark}{_MODE_NAMES[mode]}: entrenado sin validación", None))
            continue
        top = _mean([per.get("TOP_LEFT"), per.get("TOP_RIGHT")])
        bottom = _mean([per.get("BOTTOM_LEFT"), per.get("BOTTOM_RIGHT")])
        text = (
            f"{mark}{_MODE_NAMES[mode]}: {quadrant:.0%} "
            f"(arriba {top:.0%}, abajo {bottom:.0%}) · {_KIND_NAMES[result.model.kind]}"
        )
        lines.append((text, quadrant >= _GOOD_QUALITY))
    return lines


def _mean(values: list[Any]) -> float:
    numbers = [float(value) for value in values if value is not None]
    return float(np.mean(numbers)) if numbers else 0.0


def _fit_report(results: dict[EyeMode, FitResult]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for mode, result in results.items():
        entry: dict[str, Any] = {"usable_frames": result.usable_samples, "error": result.error}
        if result.model is not None:
            entry.update(
                kind=result.model.kind,
                alpha=result.model.alpha,
                closed_ear_threshold=result.model.closed_ear_threshold,
                min_vertical_span=result.model.min_vertical_span,
                metrics=dict(result.model.metrics),
            )
        entry["candidates"] = [
            {k: v for k, v in candidate.items() if k != "per_quadrant"}
            for candidate in result.candidates
        ]
        report[mode.value] = entry
    return report


# ----------------------------------------------------------------------------- test mode


@dataclass(slots=True)
class Trial:
    """One instructed prompt in the test."""

    requested: str
    started_at: float
    selected: str | None = None
    response_seconds: float | None = None
    timed_out: bool = False


def instructed_order(keys: list[str], repeats: int, rng: random.Random) -> list[str]:
    """Each option ``repeats`` times, shuffled, never the same option twice in a row."""
    order = [key for key in keys for _ in range(repeats)]
    for _ in range(200):
        rng.shuffle(order)
        if all(a != b for a, b in itertools.pairwise(order)):
            break
    return order


@dataclass(slots=True)
class InstructedTest:
    """Clock-injected prompt sequence: prompt, wait for a gaze selection or a timeout, pause.

    Only gaze confirmations answer a prompt (keyboard or mouse selections do not count).
    """

    order: list[str]
    timeout_seconds: float = _TEST_TIMEOUT_SECONDS
    pause_seconds: float = _TEST_PAUSE_SECONDS
    lead_in_seconds: float = 1.0
    trials: list[Trial] = field(default_factory=list)
    _current: Trial | None = None
    _next_at: float | None = None
    _total: int = 0

    def __post_init__(self) -> None:
        self._total = len(self.order)

    @property
    def current(self) -> Trial | None:
        return self._current

    @property
    def total(self) -> int:
        return self._total

    @property
    def done(self) -> bool:
        return not self.order and self._current is None and len(self.trials) == self._total

    def update(
        self, now: float, confirmed_key: str | None = None, source: str | None = None
    ) -> str | None:
        """Advance; returns the key of a prompt that starts on this tick (to announce it)."""
        if self._next_at is None:
            self._next_at = now + self.lead_in_seconds
        current = self._current
        if current is not None:
            if confirmed_key is not None and source == "gaze":
                current.selected = confirmed_key
                current.response_seconds = now - current.started_at
            elif now - current.started_at > self.timeout_seconds:
                current.timed_out = True
            else:
                return None
            self._current = None
            self._next_at = now + self.pause_seconds
            return None
        if self.order and now >= self._next_at:
            trial = Trial(self.order.pop(0), now)
            self.trials.append(trial)
            self._current = trial
            return trial.requested
        return None


def instructed_report(trials: list[Trial], keys: list[str]) -> dict[str, Any]:
    """Accuracy of instructed selections on the real board (prospective measure)."""
    correct = [trial for trial in trials if trial.selected == trial.requested]
    confusion = {key: {other: 0 for other in [*keys, "NONE"]} for key in keys}
    for trial in trials:
        confusion[trial.requested][trial.selected or "NONE"] += 1
    times = [trial.response_seconds for trial in correct if trial.response_seconds is not None]
    return {
        "trials": len(trials),
        "correct": len(correct),
        "accuracy": len(correct) / len(trials) if trials else None,
        "no_selection": sum(1 for trial in trials if trial.selected is None),
        "confusion": confusion,
        "per_option_accuracy": {
            key: (
                sum(1 for t in trials if t.requested == key and t.selected == key)
                / max(1, sum(1 for t in trials if t.requested == key))
            )
            for key in keys
        },
        "correct_response_seconds_median": float(np.median(times)) if times else None,
        "note": "Instructed prompts are technical ground truth, not communication; implicit "
        "learning is disabled during the test.",
    }


# ----------------------------------------------------------------------------- runner


class Phase(StrEnum):
    CALIBRATING = "CALIBRATING"
    SUMMARY = "SUMMARY"
    ALIGNING = "ALIGNING"
    TEST = "TEST"
    TEST_SUMMARY = "TEST_SUMMARY"
    BOARD = "BOARD"


def run_board(
    *,
    communication: CommunicationConfig,
    guided_config: GuidedValidationConfig,
    calibrations: dict[EyeMode, Path],
    face_model_path: Path,
    smoothing_alpha: float,
    speech_settings: dict[str, Any],
    fullscreen: bool,
    align: bool = True,
    calibration_dir: Path | None = None,
    test_repeats: int = 0,
) -> dict[str, Any]:
    """Open the camera and run calibration, optional test and the board until the user quits."""
    # Imported lazily so replay/simulation and tests do not need MediaPipe or a camera.
    from mekasa_gaze.tracking.camera import OpenCVCameraProvider
    from mekasa_gaze.tracking.frame_source import LatestFrameGrabber
    from mekasa_gaze.tracking.landmarker import FaceFeatureExtractor

    models = _Models(calibrations)
    eye_mode = communication.eye_mode
    session = CommunicationSession(communication, calibration_mode=None)
    width, height = guided_config.canvas_width, guided_config.canvas_height
    area = board_area(width, height)
    split = board_to_screen((0.5, 0.5), area)
    keys = [option.key for option in session.layout.options]
    navigator: Navigator | None = None
    if communication.interface == "conversation":
        layout = session.layout
        navigator = Navigator(
            load_vocabulary(Path(communication.vocabulary), gender=communication.gender),
            yes_phrase=layout.option("YES").phrase,
            no_phrase=layout.option("NO").phrase,
            yes_label=layout.option("YES").label,
            no_label=layout.option("NO").label,
        )
    conversation_stats = {"selections": 0, "messages_spoken": 0}
    speech = create_speech_backend(
        enabled=bool(speech_settings.get("enabled", True)),
        required=bool(speech_settings.get("required", False)),
        language=str(speech_settings.get("language", "es-ES")),
        rate=int(speech_settings.get("rate", 0)),
        volume=int(speech_settings.get("volume", 100)),
    )
    grabber = LatestFrameGrabber(
        lambda: OpenCVCameraProvider(
            index=guided_config.camera_index,
            backend=guided_config.backend,
            width=guided_config.width,
            height=guided_config.height,
            target_fps=guided_config.target_fps,
        )
    )
    online = OnlineCorrection()
    stats: dict[str, Any] = {
        "calibrations": 0,
        "alignments_completed": 0,
        "auto_realignments": 0,
        "implicit_samples_added": 0,
        "implicit_samples_retracted": 0,
        "eye_mode_auto_switched": None,
    }
    calibration_report: dict[str, Any] | None = None
    test_results: dict[str, Any] | None = None
    latencies: list[float] = []
    processing: list[float] = []
    speech_failures = 0
    speech_ok = True
    show_gaze = False
    clicks: list[tuple[float, float]] = []
    eye_modes = list(EyeMode)
    focus_trace: deque[tuple[float, tuple[float, float], str]] = deque(maxlen=120)
    rng = random.Random()

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(_WINDOW_NAME, width, height)

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x / width, y / height))

    cv2.setMouseCallback(_WINDOW_NAME, on_mouse)
    started = perf_counter()
    last: PipelineResult | None = None
    flash_until = 0.0
    flash: BoardOption | None = None
    try:
        with (
            FaceFeatureExtractor(
                face_model_path,
                min_face_detection_confidence=guided_config.min_face_detection_confidence,
                min_face_presence_confidence=guided_config.min_face_presence_confidence,
                min_tracking_confidence=guided_config.min_tracking_confidence,
            ) as extractor,
            grabber,
        ):
            pipeline = GazePipeline(
                extractor=extractor,
                validity_config=guided_config,
                model=None,
                eye_mode=eye_mode,
                smoothing_alpha=smoothing_alpha,
                area=area,
                distance_compensation=communication.distance_compensation,
                camera_position=communication.camera_position,
            )
            calibration: CalibrationProcedure | None = None
            alignment: AlignmentProcedure | None = None
            alignment_reason: str | None = None
            fit_results: dict[EyeMode, FitResult] = {}
            summary_since = 0.0
            summary_lines: list[tuple[str, bool | None]] = []
            summary_notice: str | None = None
            test: InstructedTest | None = None
            last_realign = -1e9

            def install(mode: EyeMode, model: GazeEstimator | None) -> None:
                pipeline.set_model(mode, model)
                session.set_eye_mode(mode)
                session.set_calibration(model.eye_mode if model else None)
                online.reset()

            def start_calibration() -> CalibrationProcedure:
                install(eye_mode, None)
                session.pause()
                speak_safely(
                    speech,
                    "Calibración. Mire cada punto hasta que se complete el círculo.",
                )
                return CalibrationProcedure(calibration_steps(area))

            def begin_alignment(reason: str | None = None) -> AlignmentProcedure | None:
                nonlocal alignment_reason
                if not (align and session.gaze_enabled):
                    return None
                pipeline.set_correction(None)
                pipeline.monitor.rebase()
                session.pause()
                alignment_reason = reason
                speak_safely(speech, "Ajuste rápido. Mire cada punto.")
                return AlignmentProcedure(option_centers(session.layout, area))

            def end_alignment(snapshot: AlignmentSnapshot | None, now: float) -> None:
                if snapshot is not None and snapshot.correction is not None:
                    online.set_anchors(snapshot.samples)
                    stats["alignments_completed"] += 1
                pipeline.set_correction(online.correction(now))
                session.resume()

            phase = Phase.BOARD
            if communication.calibrate_at_start:
                calibration = start_calibration()
                phase = Phase.CALIBRATING
            else:
                install(eye_mode, models.get(eye_mode))
                alignment = begin_alignment()
                phase = Phase.ALIGNING if alignment is not None else Phase.BOARD

            while True:
                frame = grabber.take_new()
                if frame is not None:
                    last = pipeline.process(frame)
                    processing.append(last.processing_ms)
                now = perf_counter()
                connected = grabber.status().connected
                fresh = frame is not None and last is not None and connected
                key = cv2.waitKey(1 if frame is not None else 5) & 0xFF
                if key in (27, ord("q")):
                    break

                # ------------------------------------------------------------ calibration
                if phase is Phase.CALIBRATING and calibration is not None:
                    usable = (
                        last.observation
                        if fresh and last is not None and last.tracking_ok and last.any_eye_open
                        else None
                    )
                    snapshot = calibration.update(now, usable)
                    if snapshot.done:
                        fit_results = {
                            mode: fit_personal_model(calibration.samples, mode, split)
                            for mode in EyeMode
                        }
                        for mode, result in fit_results.items():
                            if result.model is not None:
                                models.models[mode] = result.model
                                if calibration_dir is not None:
                                    save_personal_model(
                                        result.model, session_model_path(calibration_dir, mode)
                                    )
                        stats["calibrations"] += 1
                        chosen, summary_notice = choose_eye_mode(eye_mode, fit_results)
                        if chosen is not eye_mode:
                            stats["eye_mode_auto_switched"] = chosen.value
                            eye_mode = chosen
                        install(eye_mode, models.get(eye_mode))
                        calibration_report = {
                            "duration_seconds": estimated_seconds(calibration.steps),
                            "steps": len(calibration.steps),
                            "short_steps": calibration.short_steps,
                            "frames_collected": len(calibration.samples),
                            "fits": _fit_report(fit_results),
                            "active_eye_mode": eye_mode.value,
                        }
                        summary_lines = calibration_summary(fit_results, eye_mode)
                        if summary_notice:
                            summary_lines.append((summary_notice, None))
                        summary_since = now
                        calibration = None
                        phase = Phase.SUMMARY
                        active = fit_results.get(eye_mode)
                        good = (
                            active is not None
                            and active.model is not None
                            and (active.model.metrics.get("quadrant_accuracy") or 0.0)
                            >= _GOOD_QUALITY
                        )
                        speak_safely(
                            speech,
                            "Calibración lista." if good else "Calibración con poca precisión.",
                        )
                        continue
                    signal_ok = last is not None and last.tracking_ok and last.any_eye_open
                    cv2.imshow(
                        _WINDOW_NAME,
                        render_calibration(
                            width,
                            height,
                            snapshot,
                            high_contrast=communication.high_contrast,
                            signal_ok=signal_ok,
                        ),
                    )
                    continue

                # ------------------------------------------------------------ summary
                if phase is Phase.SUMMARY:
                    active = fit_results.get(eye_mode)
                    quality = (
                        float(active.model.metrics.get("quadrant_accuracy") or 0.0)
                        if active is not None and active.model is not None
                        else 0.0
                    )
                    auto = quality >= _GOOD_QUALITY and now - summary_since > _SUMMARY_AUTO_CONTINUE
                    if key in (ord("r"), ord("R"), ord("k"), ord("K")):
                        calibration = start_calibration()
                        phase = Phase.CALIBRATING
                        continue
                    if key in (13, 10, ord(" ")) or auto:
                        session.resume()
                        if test_repeats > 0 and session.gaze_enabled:
                            test = InstructedTest(instructed_order(keys, test_repeats, rng))
                            phase = Phase.TEST
                            speak_safely(speech, "Prueba. Mire la opción que se indica.")
                        else:
                            phase = Phase.BOARD
                        continue
                    footer = "ENTER = continuar · R = repetir calibración · ESC = salir"
                    if quality >= _GOOD_QUALITY:
                        footer = "Continúa solo en unos segundos · " + footer
                    cv2.imshow(
                        _WINDOW_NAME,
                        render_summary(
                            width,
                            height,
                            "Resultado de la calibración",
                            summary_lines,
                            high_contrast=communication.high_contrast,
                            footer=footer,
                        ),
                    )
                    continue

                # ------------------------------------------------------------ alignment
                if phase is Phase.ALIGNING and alignment is not None:
                    raw = last.raw_screen_point if fresh and last is not None else None
                    snapshot_a = alignment.update(now, raw)
                    if key in (ord("s"), ord("S")):
                        end_alignment(None, now)
                        alignment = None
                        phase = Phase.BOARD
                        continue
                    if snapshot_a.done:
                        end_alignment(snapshot_a, now)
                        alignment = None
                        phase = Phase.BOARD
                        continue
                    signal_ok = last is not None and last.raw_screen_point is not None
                    cv2.imshow(
                        _WINDOW_NAME,
                        render_alignment(
                            width,
                            height,
                            snapshot_a,
                            high_contrast=communication.high_contrast,
                            signal_ok=signal_ok,
                            reason=alignment_reason,
                        ),
                    )
                    continue

                # ------------------------------------------------------------ test summary
                if phase is Phase.TEST_SUMMARY:
                    if key in (13, 10, ord(" ")):
                        phase = Phase.BOARD
                        continue
                    if key in (ord("r"), ord("R"), ord("k"), ord("K")):
                        calibration = start_calibration()
                        phase = Phase.CALIBRATING
                        continue
                    continue

                # ------------------------------------------------------------ board / test
                tick = gaze_input(last, now, connected)
                update: SessionUpdate = session.update(tick)
                if fresh and last is not None:
                    latencies.append((now - last.frame_timestamp_seconds) * 1000.0)
                    if update.focused is not None and last.raw_screen_point is not None:
                        focus_trace.append((now, last.raw_screen_point, update.focused.key))

                for click in clicks:
                    board_point = screen_to_board(click, area)
                    option = session.layout.option_at(board_point) if board_point else None
                    if option is not None:
                        update = session.select_direct(option.key, now, source="pointer")
                clicks.clear()

                if key == ord(" "):
                    session.toggle_pause()
                elif key in (ord("a"), ord("A")):
                    alignment = begin_alignment()
                    if alignment is not None:
                        phase = Phase.ALIGNING
                        continue
                elif key in (ord("k"), ord("K")):
                    calibration = start_calibration()
                    phase = Phase.CALIBRATING
                    continue
                elif key == 8 and navigator is not None and phase is Phase.BOARD:
                    navigator.back()  # BACKSPACE: one level up
                elif key in (ord("h"), ord("H")) and navigator is not None:
                    navigator.home()
                elif key in (ord("c"), ord("C")):
                    session.clear_message()
                    if navigator is not None:
                        navigator.last_said = None
                    if online.retract_recent(now):
                        stats["implicit_samples_retracted"] += 1
                        pipeline.set_correction(online.correction(now))
                elif key in (ord("g"), ord("G")):
                    show_gaze = not show_gaze
                elif key in (ord("m"), ord("M")):
                    eye_mode = eye_modes[(eye_modes.index(eye_mode) + 1) % len(eye_modes)]
                    install(eye_mode, models.get(eye_mode))
                    alignment = begin_alignment()
                    if alignment is not None:
                        phase = Phase.ALIGNING
                        continue
                elif key in (13, 10) and update.pending is not None:
                    update = session.select_direct(update.pending.key, now, source="keyboard")
                elif ord("1") <= key <= ord("9"):
                    index = key - ord("1")
                    if index < len(session.layout.options):
                        update = session.select_direct(
                            session.layout.options[index].key, now, source="keyboard"
                        )

                prompt: str | None = None
                if phase is Phase.TEST and test is not None:
                    confirmed = update.confirmed.key if update.confirmed is not None else None
                    started_key = test.update(now, confirmed, update.confirmed_source)
                    if started_key is not None:
                        label = session.layout.option(started_key).label
                        speak_safely(speech, f"Mire {label.lower()}")
                    if test.done:
                        test_results = instructed_report(test.trials, keys)
                        test = None
                        phase = Phase.TEST_SUMMARY
                        cv2.imshow(
                            _WINDOW_NAME,
                            _test_summary_canvas(
                                width, height, test_results, session, communication
                            ),
                        )
                        continue
                    if test.current is not None:
                        label = session.layout.option(test.current.requested).label
                        prompt = f"Prueba {len(test.trials)} de {test.total}: mire {label}"
                    else:
                        prompt = "Prueba: espere la próxima consigna"

                if update.confirmed is not None:
                    spoken: str | None = None
                    if phase is not Phase.TEST and navigator is not None:
                        outcome = navigator.select(keys.index(update.confirmed.key))
                        conversation_stats["selections"] += 1
                        spoken = outcome.speak
                        if outcome.alarm and communication.urgent_alarm:
                            play_alarm()
                    elif phase is not Phase.TEST:
                        spoken = update.confirmed.phrase
                    if spoken:
                        speech_ok = speak_safely(speech, spoken)
                        speech_failures += 0 if speech_ok else 1
                        conversation_stats["messages_spoken"] += 1
                    flash, flash_until = update.confirmed, now + 0.8
                    if (
                        phase is Phase.BOARD
                        and update.confirmed_source == "gaze"
                        and communication.implicit_recalibration
                    ):
                        points = [
                            point
                            for at, point, focused in focus_trace
                            if focused == update.confirmed.key
                            and now - at <= communication.dwell_seconds + 0.3
                        ]
                        if len(points) >= 5:
                            median = np.median(np.asarray(points), axis=0)
                            option = update.confirmed
                            center = board_to_screen(
                                (
                                    (option.bounds[0] + option.bounds[2]) / 2,
                                    (option.bounds[1] + option.bounds[3]) / 2,
                                ),
                                area,
                            )
                            online.add_selection((float(median[0]), float(median[1])), center, now)
                            stats["implicit_samples_added"] += 1
                            pipeline.set_correction(online.correction(now))
                    focus_trace.clear()

                notice: str | None = None
                status = last.placement_status if last is not None else None
                if status is not None and status.drift is not Drift.OK and session.gaze_enabled:
                    notice = DRIFT_HINTS[status.drift]
                    if (
                        communication.auto_realign
                        and align
                        and phase is Phase.BOARD
                        and status.sustained_seconds >= _REALIGN_AFTER_SECONDS
                        and now - last_realign >= _REALIGN_MIN_INTERVAL
                        and update.focused is None
                        and update.pending is None
                    ):
                        alignment = begin_alignment("Cambió de posición")
                        if alignment is not None:
                            last_realign = now
                            stats["auto_realignments"] += 1
                            phase = Phase.ALIGNING
                            continue

                eyes = None
                hint = None
                if last is not None and connected:
                    eyes = (
                        last.validity.left_eye_state is EyeState.OPEN,
                        last.validity.right_eye_state is EyeState.OPEN,
                    )
                    hint = _PLACEMENT_HINTS.get(last.placement.state)
                cells = None
                location = None
                shown = update
                if navigator is not None and phase is Phase.BOARD:
                    cells = [(cell.label, cell.kind.value) for cell in navigator.cells()]
                    location = navigator.breadcrumb()
                    shown = replace(update, message=navigator.message_line())
                canvas = render_board(
                    width,
                    height,
                    session.layout,
                    shown,
                    eye_mode=eye_mode,
                    high_contrast=communication.high_contrast,
                    gaze_point=last.screen_point if (show_gaze and last) else None,
                    speech_ok=speech_ok,
                    flash=flash if now < flash_until else None,
                    eyes=eyes,
                    hint=hint,
                    aligned=(online.anchored or online.implicit_samples > 0)
                    if session.gaze_enabled
                    else None,
                    notice=notice,
                    prompt=prompt,
                    cells=cells,
                    location=location,
                )
                cv2.imshow(_WINDOW_NAME, canvas)
    finally:
        cv2.destroyWindow(_WINDOW_NAME)
        speech.close()

    capture = grabber.status()
    settings = grabber.camera_settings
    duration = perf_counter() - started
    return {
        "schema_version": 2,
        "kind": "COMMUNICATION_BOARD_SESSION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": source_revision(),
        "environment": environment(),
        "camera": asdict(settings)
        if is_dataclass(settings) and not isinstance(settings, type)
        else None,
        "configuration": {
            **{k: v for k, v in asdict(communication).items() if k != "phrases"},
            "eye_mode_at_start": communication.eye_mode.value,
            "eye_mode_at_end": eye_mode.value,
            "calibrated_eye_modes": sorted(mode.value for mode in models.models),
            "calibration_errors": models.errors,
            "session_alignment_enabled": align,
            "test_repeats": test_repeats,
            "speech": speech.metadata,
        },
        "calibration": calibration_report,
        "test": test_results,
        "measurement": {
            "duration_seconds": duration,
            "capture": asdict(capture),
            "frames_processed": len(processing),
            "processing_ms": {
                "p50": _percentile(processing, 50),
                "p95": _percentile(processing, 95),
            },
            "capture_to_decision_ms": {
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
            },
            "speech_failures": speech_failures,
            "adaptation": stats,
            "conversation": conversation_stats,
            "session": session.stats.as_dict(),
        },
        "privacy": {
            "video_persisted": False,
            "frames_persisted": False,
            "eye_patches_persisted": False,
            "raw_landmarks_persisted": False,
            "message_history_persisted": False,
            "per_option_counts_persisted": False,
        },
        "limitations": [
            "Calibration accuracy is cross-validated within the same sitting and the model "
            "choice used the same rounds; only the instructed test measures the board itself.",
            "Free-use selections have no ground truth; this report does not measure accuracy "
            "outside the instructed test.",
            "Timing, distance compensation, auto re-alignment and implicit learning are "
            "engineering hypotheses, not clinical thresholds.",
        ],
    }


def _test_summary_canvas(
    width: int,
    height: int,
    results: dict[str, Any],
    session: CommunicationSession,
    communication: CommunicationConfig,
) -> npt.NDArray[np.uint8]:
    lines: list[tuple[str, bool | None]] = []
    accuracy = results.get("accuracy") or 0.0
    lines.append(
        (f"Acierto: {results['correct']} de {results['trials']} ({accuracy:.0%})", accuracy >= 0.85)
    )
    for key, value in results["per_option_accuracy"].items():
        lines.append((f"{session.layout.option(key).label}: {value:.0%}", value >= 0.85))
    if results.get("no_selection"):
        lines.append((f"Sin selección a tiempo: {results['no_selection']}", False))
    return render_summary(
        width,
        height,
        "Resultado de la prueba",
        lines,
        high_contrast=communication.high_contrast,
        footer="ENTER = usar el tablero · R = recalibrar · ESC = salir",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mekasa local gaze communication board")
    parser.add_argument("--config", type=Path, default=Path("config/communication.toml"))
    parser.add_argument("--guided-config", type=Path, default=Path("config/experiment-001.toml"))
    parser.add_argument(
        "--calibration-config", type=Path, default=Path("config/experiment-002.toml")
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("work/calibration"),
        help="Session models are saved here as session-gaze-model-{both,left,right}.json; "
        "research calibrations (personal-gaze-model-*.json) are used as a fallback.",
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        default=Path("models/gaze/mediapipe-face-landmarker/face_landmarker.task"),
    )
    parser.add_argument(
        "--eye-mode", type=lambda v: EyeMode(v.upper()), choices=list(EyeMode), default=None
    )
    parser.add_argument("--dwell", type=float, default=None, help="Override dwell seconds")
    parser.add_argument("--confirm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--interface",
        choices=("conversation", "simple"),
        default=None,
        help="conversation: menus to compose messages (default); simple: SÍ/NO/AGUA/AYUDA",
    )
    parser.add_argument(
        "--gender", choices=("m", "f"), default=None, help="Grammatical gender of phrases"
    )
    parser.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Calibrate at start (~35 s, default) or reuse the last model with --no-calibrate.",
    )
    parser.add_argument(
        "--align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow 4-point re-alignment (start without calibration, key A, auto re-align).",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=0,
        metavar="N",
        help="After calibration, run an instructed test with each option N times.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("work/communication/board-session-latest.json")
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run the scripted no-camera safety replay and print its report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``mekasa-board``)."""
    import tomllib

    from mekasa_gaze.core.config import CalibrationConfig

    args = _parser().parse_args(argv)
    try:
        communication = CommunicationConfig.load(args.config)
        overrides: dict[str, Any] = {}
        if args.eye_mode is not None:
            overrides["eye_mode"] = args.eye_mode
        if args.dwell is not None:
            overrides["dwell_seconds"] = args.dwell
        if args.confirm is not None:
            overrides["confirmation_required"] = args.confirm
        if args.calibrate is not None:
            overrides["calibrate_at_start"] = args.calibrate
        if args.interface is not None:
            overrides["interface"] = args.interface
        if args.gender is not None:
            overrides["gender"] = args.gender
        if args.test < 0 or args.test > 5:
            raise ValueError("--test must be between 0 and 5")
        if args.test > 0:
            overrides["calibrate_at_start"] = True
        communication = replace(communication, **overrides)
        communication.validate()
        if args.simulate:
            from mekasa_gaze.runners.board_simulation import run_default_simulation

            report = run_default_simulation(
                communication, GuidedValidationConfig.load(args.guided_config)
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if report["safety"]["passed"] else 1
        with args.config.open("rb") as handle:
            speech_settings = tomllib.load(handle).get("speech", {})
        report = run_board(
            communication=communication,
            guided_config=GuidedValidationConfig.load(args.guided_config),
            calibrations=calibration_paths(args.calibration_dir),
            face_model_path=args.face_model,
            smoothing_alpha=CalibrationConfig.load(args.calibration_config).live_smoothing_alpha,
            speech_settings=speech_settings,
            fullscreen=args.fullscreen,
            align=args.align,
            calibration_dir=args.calibration_dir,
            test_repeats=args.test,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"Communication board could not run: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "report": str(args.output),
        "calibration": (report.get("calibration") or {}).get("fits"),
        "test": report.get("test"),
        "adaptation": report["measurement"]["adaptation"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
