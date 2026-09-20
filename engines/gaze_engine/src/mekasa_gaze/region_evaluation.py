from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, deque
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from mekasa_gaze.calibration import CalibrationModel, load_calibration_model
from mekasa_gaze.camera import CameraUnavailableError, OpenCVCameraProvider
from mekasa_gaze.config import CalibrationConfig, GuidedValidationConfig, InteractionConfig
from mekasa_gaze.domain import GazeValidity
from mekasa_gaze.guided_experiment import _rolling_fps
from mekasa_gaze.interaction import DwellSelector
from mekasa_gaze.landmarker import FaceFeatureExtractor
from mekasa_gaze.live_evaluation import PredictionSmoother
from mekasa_gaze.placement import assess_face_placement
from mekasa_gaze.runtime import environment, source_revision
from mekasa_gaze.validity import assess_gaze_validity
from mekasa_gaze.visualization import render_loading_screen, render_region_selection_screen

_WINDOW_NAME = "Mekasa Experiment 003 — Region Selection"


def run_region_evaluation(
    *,
    guided_config: GuidedValidationConfig,
    calibration_config: CalibrationConfig,
    interaction_config: InteractionConfig,
    calibration_model: CalibrationModel,
    face_model_path: Path,
    calibration_sha256: str,
    fullscreen: bool | None = None,
) -> dict[str, Any]:
    use_fullscreen = guided_config.fullscreen if fullscreen is None else fullscreen
    camera = OpenCVCameraProvider(
        index=guided_config.camera_index,
        backend=guided_config.backend,
        width=guided_config.width,
        height=guided_config.height,
        target_fps=guided_config.target_fps,
    )
    smoother = PredictionSmoother(calibration_config.live_smoothing_alpha)
    selector = DwellSelector(
        dwell_seconds=interaction_config.dwell_seconds,
        cooldown_seconds=interaction_config.cooldown_seconds,
        hysteresis_margin=interaction_config.hysteresis_margin,
    )
    frame_times: deque[float] = deque(maxlen=60)
    events: list[dict[str, Any]] = []
    frames_total = 0
    frames_usable = 0
    started = perf_counter()

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    if use_fullscreen:
        cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(
            _WINDOW_NAME,
            guided_config.canvas_width,
            guided_config.canvas_height,
        )
    cv2.imshow(
        _WINDOW_NAME,
        render_loading_screen(
            guided_config.canvas_width,
            guided_config.canvas_height,
            title="MEKASA",
            status="Preparando seleccion por mirada...",
        ),
    )
    cv2.waitKey(1)

    try:
        with (
            camera,
            FaceFeatureExtractor(
                face_model_path,
                min_face_detection_confidence=guided_config.min_face_detection_confidence,
                min_face_presence_confidence=guided_config.min_face_presence_confidence,
                min_tracking_confidence=guided_config.min_tracking_confidence,
            ) as extractor,
        ):
            started = perf_counter()
            while True:
                frame = camera.read()
                frames_total += 1
                rgb = cast(
                    npt.NDArray[np.uint8],
                    cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB),
                )
                observation = extractor.extract(
                    rgb,
                    sequence=frame.sequence,
                    timestamp_ns=frame.captured_at_ns,
                )
                assessment = assess_gaze_validity(observation, guided_config)
                placement = assess_face_placement(
                    observation,
                    min_face_width=guided_config.min_face_width,
                    max_face_width=guided_config.max_face_width,
                )
                raw_prediction: tuple[float, float] | None = None
                if placement.ready and assessment.gaze_validity in (
                    GazeValidity.VALID,
                    GazeValidity.PARTIAL,
                ):
                    try:
                        raw_prediction = calibration_model.predict_observation(
                            observation,
                            left_eye_state=assessment.left_eye_state,
                            right_eye_state=assessment.right_eye_state,
                        )
                    except ValueError:
                        raw_prediction = None
                prediction = smoother.update(raw_prediction)
                if prediction is not None:
                    frames_usable += 1
                now = perf_counter()
                selection = selector.update(prediction, now)
                if selection.selection_fired and selection.selected is not None:
                    events.append(
                        {
                            "elapsed_seconds": now - started,
                            "region": selection.selected.value,
                        }
                    )
                frame_times.append(now)
                screen = render_region_selection_screen(
                    frame.bgr,
                    observation,
                    assessment,
                    prediction,
                    selection,
                    guided_config,
                    _rolling_fps(frame_times),
                    placement,
                )
                cv2.imshow(_WINDOW_NAME, screen)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cv2.destroyWindow(_WINDOW_NAME)

    duration = perf_counter() - started
    counts = Counter(event["region"] for event in events)
    return {
        "schema_version": 1,
        "kind": "ABSTRACT_REGION_DWELL_EVALUATION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": source_revision(),
        "environment": environment(),
        "camera": asdict(camera.actual_settings),
        "calibration_sha256": calibration_sha256,
        "configuration": {
            "dwell_seconds": interaction_config.dwell_seconds,
            "cooldown_seconds": interaction_config.cooldown_seconds,
            "hysteresis_margin": interaction_config.hysteresis_margin,
            "smoothing_alpha": calibration_config.live_smoothing_alpha,
            "min_face_width": guided_config.min_face_width,
            "max_face_width": guided_config.max_face_width,
        },
        "measurement": {
            "duration_seconds": duration,
            "frames_total": frames_total,
            "frames_usable": frames_usable,
            "usable_fraction": frames_usable / frames_total if frames_total else None,
            "selection_count": len(events),
            "selection_counts_by_region": {region: counts[region] for region in sorted(counts)},
            "events": events,
        },
        "privacy": {
            "video_persisted": False,
            "frames_persisted": False,
            "raw_landmarks_persisted": False,
            "derived_selection_events_persisted": True,
        },
        "limitations": [
            "Selections are abstract technical events, not AAC intents.",
            (
                "No instructed ground truth is recorded, so this run does not measure "
                "selection accuracy."
            ),
            (
                "Dwell, hysteresis and cooldown timings are engineering defaults, "
                "not clinical thresholds."
            ),
            "Tracking loss or out-of-range placement clears dwell progress immediately.",
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate four abstract regions with dwell and tracking-loss safety"
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("work/calibration/personal-gaze-model.json"),
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        default=Path("models/gaze/mediapipe-face-landmarker/face_landmarker.task"),
    )
    parser.add_argument(
        "--guided-config",
        type=Path,
        default=Path("config/experiment-001.toml"),
    )
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=Path("config/experiment-002.toml"),
    )
    parser.add_argument(
        "--interaction-config",
        type=Path,
        default=Path("config/experiment-003.toml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/interaction/region-evaluation-latest.json"),
    )
    parser.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_region_evaluation(
            guided_config=GuidedValidationConfig.load(args.guided_config),
            calibration_config=CalibrationConfig.load(args.calibration_config),
            interaction_config=InteractionConfig.load(args.interaction_config),
            calibration_model=load_calibration_model(args.calibration),
            face_model_path=args.face_model,
            calibration_sha256=_sha256(args.calibration),
            fullscreen=args.fullscreen,
        )
        _write_report(report, args.output)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        history_path = args.output.parent / "history" / f"region-evaluation-{timestamp}.json"
        _write_report(report, history_path)
    except (CameraUnavailableError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Region evaluation could not run: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Region evaluation interrupted.", file=sys.stderr)
        return 130
    print(
        json.dumps(
            {
                "report": str(args.output),
                "history_report": str(history_path),
                "measurement": report["measurement"],
                "warning": "Technical experiment only; not AAC or clinical validation.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
