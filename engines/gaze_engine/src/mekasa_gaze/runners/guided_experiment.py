from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, perf_counter_ns
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.config import GuidedValidationConfig
from mekasa_gaze.core.protocol import GuidedProtocol
from mekasa_gaze.services.reporting import GuidedReportBuilder, write_guided_report
from mekasa_gaze.services.runtime import environment, model_identity, source_revision
from mekasa_gaze.services.speech import AudioInstructionController, create_speech_backend
from mekasa_gaze.tracking.camera import CameraUnavailableError, OpenCVCameraProvider
from mekasa_gaze.tracking.landmarker import FaceFeatureExtractor
from mekasa_gaze.tracking.placement import assess_face_placement, placement_tracking_state
from mekasa_gaze.tracking.validity import assess_gaze_validity
from mekasa_gaze.ui.visualization import render_guided_screen, render_loading_screen

_WINDOW_NAME = "Mekasa Experiment 001 — Guided Visual Validation"


def run_guided_validation(
    *,
    config: GuidedValidationConfig,
    model_path: Path,
    report_path: Path,
    fullscreen: bool | None = None,
) -> dict[str, Any]:
    """Execute the full guided validation experiment protocol with GUI and audio.

    Responsibility:
        Coordinates full hardware setup, window creation, MediaPipe landmark extraction,
        monotonic state machine transitions across calibration targets, audio prompts,
        live telemetry rendering, and report compilation upon completion or abort.

    Args:
        config: Experiment and protocol configuration.
        model_path: Path to the MediaPipe face landmarker task binary.
        report_path: Output file path for writing the Schema 5 report.
        fullscreen: Optional boolean flag to force fullscreen display mode.

    Returns:
        dict[str, Any]: Compiled guided validation report dictionary.
    """
    startup_started = perf_counter()
    use_fullscreen = config.fullscreen if fullscreen is None else fullscreen
    camera = OpenCVCameraProvider(
        index=config.camera_index,
        backend=config.backend,
        width=config.width,
        height=config.height,
        target_fps=config.target_fps,
    )
    protocol = GuidedProtocol(config)
    builder = GuidedReportBuilder()
    speech = create_speech_backend(
        enabled=config.audio_enabled,
        required=config.audio_required,
        language=config.audio_language,
        rate=config.audio_rate,
        volume=config.audio_volume,
    )
    audio_instructions = AudioInstructionController(speech)
    frame_times: deque[float] = deque(maxlen=60)
    completed = False
    aborted = False
    completion_started: float | None = None
    startup_seconds = 0.0

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    if use_fullscreen:
        cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(_WINDOW_NAME, config.canvas_width, config.canvas_height)
    cv2.imshow(
        _WINDOW_NAME,
        render_loading_screen(
            config.canvas_width,
            config.canvas_height,
            title="MEKASA",
            status="Iniciando camara y modelo de mirada...",
        ),
    )
    cv2.waitKey(1)

    try:
        with (
            camera,
            FaceFeatureExtractor(
                model_path,
                min_face_detection_confidence=config.min_face_detection_confidence,
                min_face_presence_confidence=config.min_face_presence_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            ) as extractor,
        ):
            startup_seconds = perf_counter() - startup_started
            builder.start_measurement()
            while True:
                iteration_started_ns = perf_counter_ns()
                frame = camera.read()
                rgb = cast(
                    npt.NDArray[np.uint8],
                    cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB),
                )
                feature_observation = extractor.extract(
                    rgb,
                    sequence=frame.sequence,
                    timestamp_ns=frame.captured_at_ns,
                )
                assessment = assess_gaze_validity(feature_observation, config)
                placement = assess_face_placement(
                    feature_observation,
                    min_face_width=config.min_face_width,
                    max_face_width=config.max_face_width,
                )
                now = perf_counter()
                snapshot = protocol.update(
                    now,
                    placement_tracking_state(feature_observation, placement),
                )
                loop_latency_ms = (perf_counter_ns() - iteration_started_ns) / 1_000_000
                audio_instructions.update(snapshot, placement)

                frame_times.append(now)
                rolling_fps = _rolling_fps(frame_times)
                render_started_ns = perf_counter_ns()
                screen = render_guided_screen(
                    frame.bgr,
                    feature_observation,
                    assessment,
                    snapshot,
                    config,
                    rolling_fps,
                    audio_active=speech.active,
                    placement=placement,
                )
                render_latency_ms = (perf_counter_ns() - render_started_ns) / 1_000_000
                display_started_ns = perf_counter_ns()
                cv2.imshow(_WINDOW_NAME, screen)
                key = cv2.waitKey(1) & 0xFF
                display_latency_ms = (perf_counter_ns() - display_started_ns) / 1_000_000
                iteration_latency_ms = (perf_counter_ns() - iteration_started_ns) / 1_000_000
                builder.add(
                    frame,
                    feature_observation,
                    snapshot,
                    assessment,
                    loop_latency_ms,
                    render_latency_ms=render_latency_ms,
                    display_latency_ms=display_latency_ms,
                    iteration_latency_ms=iteration_latency_ms,
                )
                if key in (ord("q"), 27):
                    aborted = True
                    break

                if snapshot.completed:
                    completed = True
                    if completion_started is None:
                        completion_started = now
                    elif now - completion_started >= config.completion_display_seconds:
                        break
    finally:
        speech.close()
        cv2.destroyWindow(_WINDOW_NAME)

    experiment = {
        "name": "Experiment 001 — Guided Visual Validation",
        "kind": "GUIDED_VISUAL_VALIDATION",
        "model": model_identity(model_path),
        "source": source_revision(),
        "startup_seconds": startup_seconds,
        "audio": speech.metadata,
        "target_order_seed": config.target_order_seed,
        "target_orders": [
            [target.value for target in block] for block in protocol.planned_target_orders
        ],
        "display": {
            "fullscreen": use_fullscreen,
            "canvas_width": config.canvas_width,
            "canvas_height": config.canvas_height,
        },
    }
    report = builder.build(
        camera=camera.actual_settings,
        environment=environment(),
        config=config,
        experiment=experiment,
        completed=completed,
        aborted=aborted,
    )
    write_guided_report(report, report_path)
    return report


def _rolling_fps(frame_times: deque[float]) -> float | None:
    if len(frame_times) < 2:
        return None
    duration = frame_times[-1] - frame_times[0]
    return (len(frame_times) - 1) / duration if duration > 0.0 else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Experiment 001 guided visual validation locally"
    )
    parser.add_argument("--config", type=Path, default=Path("config/experiment-001.toml"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/gaze/mediapipe-face-landmarker/face_landmarker.task"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "research/experiments/001-eye-feature-extraction/results/guided-validation-result.json"
        ),
    )
    parser.add_argument(
        "--fullscreen",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the fullscreen setting from the configuration file",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("work/guided-validation/history"),
        help="Local ignored directory that preserves every completed CLI run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the command-line entry point for guided validation experiments.

    Args:
        argv: Optional list of command-line argument strings (defaults to sys.argv[1:]).

    Returns:
        int: 0 on completed protocol, 2 on initialization error, 3 if aborted, 130 on SIGINT.
    """
    args = _parser().parse_args(argv)
    try:
        report = run_guided_validation(
            config=GuidedValidationConfig.load(args.config),
            model_path=args.model,
            report_path=args.report,
            fullscreen=args.fullscreen,
        )
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        history_path = args.history_dir / f"guided-validation-{timestamp}.json"
        write_guided_report(report, history_path)
    except (CameraUnavailableError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Guided validation could not run: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Guided validation interrupted; no completed report was produced.", file=sys.stderr)
        return 130
    measurement = report["measurement"]
    protocol = report["protocol"]
    print(
        json.dumps(
            {
                "report": str(args.report),
                "history_report": str(history_path),
                "protocol": protocol,
                "measurement": measurement,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if bool(protocol["completed"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
