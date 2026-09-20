from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from mekasa_gaze.calibration import CalibrationModel, load_calibration_model
from mekasa_gaze.camera import CameraUnavailableError, OpenCVCameraProvider
from mekasa_gaze.config import CalibrationConfig, GuidedValidationConfig
from mekasa_gaze.domain import GazeValidity
from mekasa_gaze.guided_experiment import _rolling_fps
from mekasa_gaze.landmarker import FaceFeatureExtractor
from mekasa_gaze.placement import assess_face_placement
from mekasa_gaze.validity import assess_gaze_validity
from mekasa_gaze.visualization import render_live_evaluation_screen, render_loading_screen

_WINDOW_NAME = "Mekasa Experiment 002 — Live Calibration Evaluation"


@dataclass(slots=True)
class PredictionSmoother:
    alpha: float
    _value: tuple[float, float] | None = None

    def update(self, prediction: tuple[float, float] | None) -> tuple[float, float] | None:
        if prediction is None:
            self._value = None
            return None
        if self._value is None:
            self._value = prediction
        else:
            self._value = (
                self.alpha * prediction[0] + (1.0 - self.alpha) * self._value[0],
                self.alpha * prediction[1] + (1.0 - self.alpha) * self._value[1],
            )
        return self._value


def run_live_evaluation(
    *,
    guided_config: GuidedValidationConfig,
    calibration_config: CalibrationConfig,
    calibration_model: CalibrationModel,
    face_model_path: Path,
    fullscreen: bool | None = None,
) -> None:
    use_fullscreen = guided_config.fullscreen if fullscreen is None else fullscreen
    camera = OpenCVCameraProvider(
        index=guided_config.camera_index,
        backend=guided_config.backend,
        width=guided_config.width,
        height=guided_config.height,
        target_fps=guided_config.target_fps,
    )
    smoother = PredictionSmoother(calibration_config.live_smoothing_alpha)
    frame_times: deque[float] = deque(maxlen=60)

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
            status="Cargando calibracion, camara y modelo...",
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
            while True:
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
                nearest = (
                    calibration_model.nearest_target(prediction) if prediction is not None else None
                )
                now = perf_counter()
                frame_times.append(now)
                screen = render_live_evaluation_screen(
                    frame.bgr,
                    observation,
                    assessment,
                    prediction,
                    nearest,
                    guided_config,
                    _rolling_fps(frame_times),
                    placement,
                )
                cv2.imshow(_WINDOW_NAME, screen)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cv2.destroyWindow(_WINDOW_NAME)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show an experimental live response from a personal calibration"
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
        "--fullscreen",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_live_evaluation(
            guided_config=GuidedValidationConfig.load(args.guided_config),
            calibration_config=CalibrationConfig.load(args.calibration_config),
            calibration_model=load_calibration_model(args.calibration),
            face_model_path=args.face_model,
            fullscreen=args.fullscreen,
        )
    except (CameraUnavailableError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Live calibration evaluation could not run: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Live calibration evaluation interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
