"""Mekasa communication board: camera -> quality -> estimator -> dwell -> message -> speech.

Run ``uv run mekasa-board`` (see README). Without a calibration for the active eye mode the
board still opens and works through direct access (keys 1-4, mouse click), and gaze selection
stays disabled; this is deliberate, so a person is never left without a way to communicate
while setup is incomplete.

Keys: 1-4 select an option directly - ENTER confirm pending option - SPACE pause/resume -
M cycle eye mode - G show/hide gaze point (setup) - C clear message - ESC/Q quit.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.board import CommunicationConfig
from mekasa_gaze.core.config import GuidedValidationConfig
from mekasa_gaze.core.domain import CameraFrame, EyeState, FeatureObservation, TrackingState
from mekasa_gaze.core.eye_mode import EyeMode, eye_signal_usable
from mekasa_gaze.core.interfaces import FeatureExtractor, SpeechBackend
from mekasa_gaze.runners.live_evaluation import PredictionSmoother
from mekasa_gaze.services.runtime import environment, source_revision
from mekasa_gaze.services.speech import create_speech_backend
from mekasa_gaze.tracking.calibration import CalibrationModel, load_calibration_model
from mekasa_gaze.tracking.communication import CommunicationSession, GazeInput, SessionUpdate
from mekasa_gaze.tracking.placement import PlacementAssessment, assess_face_placement
from mekasa_gaze.tracking.validity import ValidityAssessment, assess_gaze_validity
from mekasa_gaze.ui.board_view import board_area, render_board, screen_to_board

_WINDOW_NAME = "Mekasa - Tablero de comunicacion"


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Everything derived from one camera frame. Holds no image data."""

    frame_timestamp_seconds: float
    observation: FeatureObservation
    validity: ValidityAssessment
    placement: PlacementAssessment
    screen_point: tuple[float, float] | None
    board_point: tuple[float, float] | None
    tracking_ok: bool
    processing_ms: float


class GazePipeline:
    """Frame -> features -> quality gates -> calibrated estimate -> board coordinates."""

    def __init__(
        self,
        *,
        extractor: FeatureExtractor,
        validity_config: GuidedValidationConfig,
        model: CalibrationModel | None,
        eye_mode: EyeMode,
        smoothing_alpha: float,
        area: tuple[float, float, float, float],
    ) -> None:
        self._extractor = extractor
        self._validity_config = validity_config
        self._model = model
        self._eye_mode = eye_mode
        self._smoother = PredictionSmoother(smoothing_alpha)
        self._area = area

    @property
    def eye_mode(self) -> EyeMode:
        """Eye mode used to gate estimation."""
        return self._eye_mode

    def set_eye_mode(self, mode: EyeMode, model: CalibrationModel | None) -> None:
        """Switch eye mode and model; resets smoothing so no estimate carries over."""
        self._eye_mode = mode
        self._model = model
        self._smoother.update(None)

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
        screen = self._smoother.update(raw)
        return PipelineResult(
            frame_timestamp_seconds=frame_timestamp_seconds,
            observation=observation,
            validity=validity,
            placement=placement,
            screen_point=screen,
            board_point=screen_to_board(screen, self._area),
            tracking_ok=tracking_ok,
            processing_ms=(perf_counter() - t0) * 1000.0,
        )


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


class _Models:
    """Per-eye-mode calibrations loaded at start-up (only valid ones are kept)."""

    def __init__(self, paths: dict[EyeMode, Path]) -> None:
        self.models: dict[EyeMode, CalibrationModel] = {}
        self.errors: dict[str, str] = {}
        for mode, path in paths.items():
            if not path.exists():
                continue
            try:
                model = load_calibration_model(path)
            except (KeyError, ValueError, json.JSONDecodeError, OSError) as error:
                self.errors[str(path)] = str(error)
                continue
            if model.eye_mode is not mode:
                self.errors[str(path)] = (
                    f"trained for {model.eye_mode.value}, not {mode.value}; ignored"
                )
                continue
            self.models[mode] = model

    def get(self, mode: EyeMode) -> CalibrationModel | None:
        return self.models.get(mode)


def run_board(
    *,
    communication: CommunicationConfig,
    guided_config: GuidedValidationConfig,
    calibrations: dict[EyeMode, Path],
    face_model_path: Path,
    smoothing_alpha: float,
    speech_settings: dict[str, Any],
    fullscreen: bool,
) -> dict[str, Any]:
    """Open the camera and run the interactive board until the user quits."""
    # Imported lazily so replay/simulation and tests do not need MediaPipe or a camera.
    from mekasa_gaze.tracking.camera import OpenCVCameraProvider
    from mekasa_gaze.tracking.frame_source import LatestFrameGrabber
    from mekasa_gaze.tracking.landmarker import FaceFeatureExtractor

    models = _Models(calibrations)
    eye_mode = communication.eye_mode
    model = models.get(eye_mode)
    session = CommunicationSession(
        communication, calibration_mode=model.eye_mode if model else None
    )
    width, height = guided_config.canvas_width, guided_config.canvas_height
    area = board_area(width, height)
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
    latencies: list[float] = []
    processing: list[float] = []
    speech_failures = 0
    speech_ok = True
    show_gaze = False
    clicks: list[tuple[float, float]] = []
    eye_modes = list(EyeMode)

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
    flash = None
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
                model=model,
                eye_mode=eye_mode,
                smoothing_alpha=smoothing_alpha,
                area=area,
            )
            while True:
                frame = grabber.take_new()
                if frame is not None:
                    last = pipeline.process(frame)
                    processing.append(last.processing_ms)
                now = perf_counter()
                tick = gaze_input(last, now, grabber.status().connected)
                update: SessionUpdate = session.update(tick)
                if frame is not None and last is not None:
                    latencies.append((now - last.frame_timestamp_seconds) * 1000.0)

                for click in clicks:
                    board_point = screen_to_board(click, area)
                    option = session.layout.option_at(board_point) if board_point else None
                    if option is not None:
                        update = session.select_direct(option.key, now, source="pointer")
                clicks.clear()

                key = cv2.waitKey(1 if frame is not None else 5) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord(" "):
                    session.toggle_pause()
                elif key in (ord("c"), ord("C")):
                    session.clear_message()
                elif key in (ord("g"), ord("G")):
                    show_gaze = not show_gaze
                elif key in (ord("m"), ord("M")):
                    eye_mode = eye_modes[(eye_modes.index(eye_mode) + 1) % len(eye_modes)]
                    model = models.get(eye_mode)
                    session.set_eye_mode(eye_mode)
                    session.set_calibration(model.eye_mode if model else None)
                    pipeline.set_eye_mode(eye_mode, model)
                elif key in (13, 10) and update.pending is not None:
                    update = session.select_direct(update.pending.key, now, source="keyboard")
                elif ord("1") <= key <= ord("9"):
                    index = key - ord("1")
                    if index < len(session.layout.options):
                        update = session.select_direct(
                            session.layout.options[index].key, now, source="keyboard"
                        )

                if update.confirmed is not None:
                    speech_ok = speak_safely(speech, update.confirmed.phrase)
                    speech_failures += 0 if speech_ok else 1
                    flash, flash_until = update.confirmed, now + 0.8
                canvas = render_board(
                    width,
                    height,
                    session.layout,
                    update,
                    eye_mode=eye_mode,
                    high_contrast=communication.high_contrast,
                    gaze_point=last.screen_point if (show_gaze and last) else None,
                    speech_ok=speech_ok,
                    flash=flash if now < flash_until else None,
                )
                cv2.imshow(_WINDOW_NAME, canvas)
    finally:
        cv2.destroyWindow(_WINDOW_NAME)
        speech.close()

    capture = grabber.status()
    settings = grabber.camera_settings
    duration = perf_counter() - started
    return {
        "schema_version": 1,
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
            "speech": speech.metadata,
        },
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
            "session": session.stats.as_dict(),
        },
        "privacy": {
            "video_persisted": False,
            "frames_persisted": False,
            "raw_landmarks_persisted": False,
            "message_history_persisted": False,
            "per_option_counts_persisted": False,
        },
        "limitations": [
            "Selections are real user actions but no instructed ground truth is recorded; "
            "this report does not measure accuracy.",
            "Timing values are engineering hypotheses, not clinical thresholds.",
        ],
    }


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
        help="Holds personal-gaze-model-{both,left,right}.json; the legacy "
        "personal-gaze-model.json is used for BOTH if present.",
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
    parser.add_argument(
        "--layout",
        choices=("quadrant", "binary-horizontal", "binary-vertical"),
        default=None,
    )
    parser.add_argument("--confirm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output", type=Path, default=Path("work/communication/board-session-latest.json")
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run the scripted no-camera safety replay and print its report",
    )
    return parser


def calibration_paths(directory: Path) -> dict[EyeMode, Path]:
    """Map each eye mode to its calibration file, with the legacy BOTH fallback."""
    paths = {mode: directory / f"personal-gaze-model-{mode.value.lower()}.json" for mode in EyeMode}
    legacy = directory / "personal-gaze-model.json"
    if not paths[EyeMode.BOTH].exists() and legacy.exists():
        paths[EyeMode.BOTH] = legacy
    return paths


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``mekasa-board``)."""
    import tomllib
    from dataclasses import replace

    from mekasa_gaze.core.config import CalibrationConfig

    args = _parser().parse_args(argv)
    try:
        communication = CommunicationConfig.load(args.config)
        overrides: dict[str, Any] = {}
        if args.eye_mode is not None:
            overrides["eye_mode"] = args.eye_mode
        if args.dwell is not None:
            overrides["dwell_seconds"] = args.dwell
        if args.layout is not None:
            overrides["layout"] = args.layout
        if args.confirm is not None:
            overrides["confirmation_required"] = args.confirm
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
    print(json.dumps({"report": str(args.output), "measurement": report["measurement"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
