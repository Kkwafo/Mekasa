from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, perf_counter_ns
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.config import ExperimentConfig
from mekasa_gaze.services.metrics import ExperimentMetrics
from mekasa_gaze.services.runtime import environment, model_identity, source_revision
from mekasa_gaze.tracking.camera import CameraUnavailableError, OpenCVCameraProvider
from mekasa_gaze.tracking.landmarker import FaceFeatureExtractor, draw_debug_overlay


def run(
    *,
    config: ExperimentConfig,
    model_path: Path,
    report_path: Path,
    display: bool | None = None,
) -> dict[str, Any]:
    should_display = config.display if display is None else display
    camera = OpenCVCameraProvider(
        index=config.camera_index,
        backend=config.backend,
        width=config.width,
        height=config.height,
        target_fps=config.target_fps,
    )
    with (
        camera,
        FaceFeatureExtractor(
            model_path,
            min_face_detection_confidence=config.min_face_detection_confidence,
            min_face_presence_confidence=config.min_face_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        ) as extractor,
    ):
        warmup_until = perf_counter() + config.warmup_seconds
        while perf_counter() < warmup_until:
            frame = camera.read()
            rgb = cast(
                npt.NDArray[np.uint8],
                cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB),
            )
            extractor.extract(
                rgb,
                sequence=frame.sequence,
                timestamp_ns=frame.captured_at_ns,
            )

        metrics = ExperimentMetrics.start()
        measure_until = perf_counter() + config.measure_seconds
        while perf_counter() < measure_until:
            loop_started_ns = perf_counter_ns()
            frame = camera.read()
            rgb = cast(
                npt.NDArray[np.uint8],
                cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB),
            )
            observation = extractor.extract(
                rgb,
                sequence=frame.sequence,
                timestamp_ns=frame.captured_at_ns,
            )
            loop_latency_ms = (perf_counter_ns() - loop_started_ns) / 1_000_000
            metrics.add(frame, observation, loop_latency_ms)
            if should_display:
                cv2.imshow(
                    "Mekasa Experiment 001 - press Q to stop",
                    draw_debug_overlay(frame.bgr, observation),
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if should_display:
            cv2.destroyAllWindows()
        report = metrics.report(camera.actual_settings, environment())
        report["experiment"] = {
            "name": "Experiment 001 — Eye Feature Extraction",
            "configuration": asdict(config),
            "debug_display_enabled": should_display,
            "model": model_identity(model_path),
            "source": source_revision(),
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Mekasa Experiment 001 on a local webcam")
    parser.add_argument("--config", type=Path, default=Path("config/experiment-001.toml"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/gaze/mediapipe-face-landmarker/face_landmarker.task"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("research/experiments/001-eye-feature-extraction/results/latest.json"),
    )
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(
            config=ExperimentConfig.load(args.config),
            model_path=args.model,
            report_path=args.report,
            display=args.display,
        )
    except (CameraUnavailableError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Experiment 001 could not run: {error}", file=sys.stderr)
        return 2
    measurement = report["measurement"]
    print(json.dumps({"report": str(args.report), **measurement}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
