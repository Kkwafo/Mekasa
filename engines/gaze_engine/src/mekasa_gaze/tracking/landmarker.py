from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter_ns

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.domain import FeatureObservation, TrackingState
from mekasa_gaze.tracking.geometry import (
    euclidean_2d,
    euler_from_transform,
    eye_aspect_ratio,
    normalized_iris_position,
)

# MediaPipe Face Mesh topology. Names follow the person's anatomical side;
# each six-point list runs from image-left corner to image-right corner so the
# normalized horizontal iris coordinate has the same image-space direction.
_LEFT_EYE = (362, 385, 387, 263, 373, 380)
_RIGHT_EYE = (33, 160, 158, 133, 153, 144)
_LEFT_IRIS = (473, 474, 475, 476, 477)
_RIGHT_IRIS = (468, 469, 470, 471, 472)


def features_from_landmarks(
    landmarks: Sequence[tuple[float, float, float]],
) -> dict[str, float]:
    """Calculate resolution-independent Experiment 001 features."""
    left_eye = [landmarks[index] for index in _LEFT_EYE]
    right_eye = [landmarks[index] for index in _RIGHT_EYE]
    left_x, left_y = normalized_iris_position(
        [landmarks[index] for index in _LEFT_IRIS],
        landmarks[_LEFT_EYE[0]],
        landmarks[_LEFT_EYE[3]],
        landmarks[_LEFT_EYE[2]],
        landmarks[_LEFT_EYE[4]],
    )
    right_x, right_y = normalized_iris_position(
        [landmarks[index] for index in _RIGHT_IRIS],
        landmarks[_RIGHT_EYE[0]],
        landmarks[_RIGHT_EYE[3]],
        landmarks[_RIGHT_EYE[2]],
        landmarks[_RIGHT_EYE[4]],
    )
    coordinates = np.asarray(landmarks, dtype=np.float64)
    minimum = np.min(coordinates[:, :2], axis=0)
    maximum = np.max(coordinates[:, :2], axis=0)
    center = (minimum + maximum) / 2.0
    size = maximum - minimum
    return {
        "left_eye_aspect_ratio": eye_aspect_ratio(left_eye),
        "right_eye_aspect_ratio": eye_aspect_ratio(right_eye),
        "left_eye_vertical_span": euclidean_2d(landmarks[_LEFT_EYE[2]], landmarks[_LEFT_EYE[4]]),
        "right_eye_vertical_span": euclidean_2d(landmarks[_RIGHT_EYE[2]], landmarks[_RIGHT_EYE[4]]),
        "left_iris_x": left_x,
        "left_iris_y": left_y,
        "right_iris_x": right_x,
        "right_iris_y": right_y,
        "face_center_x": float(center[0]),
        "face_center_y": float(center[1]),
        "face_width": float(size[0]),
        "face_height": float(size[1]),
    }


class FaceFeatureExtractor:
    def __init__(
        self,
        model_path: Path,
        *,
        min_face_detection_confidence: float,
        min_face_presence_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face Landmarker model not found at {model_path}. "
                "Run scripts/fetch-face-landmarker.ps1 explicitly first."
            )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def extract(
        self,
        rgb_frame: npt.NDArray[np.uint8],
        *,
        sequence: int,
        timestamp_ns: int,
    ) -> FeatureObservation:
        started_ns = perf_counter_ns()
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        result = self._landmarker.detect_for_video(image, timestamp_ns // 1_000_000)

        if len(result.face_landmarks) != 1:
            return FeatureObservation(
                frame_sequence=sequence,
                timestamp_ns=timestamp_ns,
                tracking_state=TrackingState.LOST,
                evidence="face_count_not_one",
                feature_latency_ms=(perf_counter_ns() - started_ns) / 1_000_000,
            )

        landmarks = [(float(p.x), float(p.y), float(p.z)) for p in result.face_landmarks[0]]
        try:
            features = features_from_landmarks(landmarks)
            debug_points = tuple(
                (landmarks[index][0], landmarks[index][1]) for index in (*_LEFT_EYE, *_RIGHT_EYE)
            )
            debug_left_iris = tuple(
                (landmarks[index][0], landmarks[index][1]) for index in _LEFT_IRIS
            )
            debug_right_iris = tuple(
                (landmarks[index][0], landmarks[index][1]) for index in _RIGHT_IRIS
            )
            coordinates = np.asarray(landmarks, dtype=np.float64)
            minimum = np.min(coordinates[:, :2], axis=0)
            maximum = np.max(coordinates[:, :2], axis=0)
            debug_face_box = (
                float(minimum[0]),
                float(minimum[1]),
                float(maximum[0]),
                float(maximum[1]),
            )
            pose = None
            evidence = "face_and_required_eye_landmarks"
            matrices = result.facial_transformation_matrixes
            if matrices:
                pose = euler_from_transform(matrices[0])
            else:
                evidence += ";head_pose_unavailable"
            state = TrackingState.GOOD
        except (IndexError, ValueError, FloatingPointError) as error:
            features = {}
            debug_points = ()
            debug_left_iris = ()
            debug_right_iris = ()
            debug_face_box = None
            pose = None
            evidence = f"invalid_landmark_geometry:{type(error).__name__}"
            state = TrackingState.UNCERTAIN

        return FeatureObservation(
            frame_sequence=sequence,
            timestamp_ns=timestamp_ns,
            tracking_state=state,
            features=features,
            head_pose=pose,
            # Face Landmarker does not expose a calibrated gaze-confidence score.
            confidence=None,
            evidence=evidence,
            feature_latency_ms=(perf_counter_ns() - started_ns) / 1_000_000,
            debug_eye_points=debug_points,
            debug_left_iris_points=debug_left_iris,
            debug_right_iris_points=debug_right_iris,
            debug_face_box=debug_face_box,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceFeatureExtractor:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def draw_debug_overlay(
    bgr_frame: npt.NDArray[np.uint8], observation: FeatureObservation
) -> npt.NDArray[np.uint8]:
    output = bgr_frame.copy()
    height, width = output.shape[:2]
    for normalized_x, normalized_y in observation.debug_eye_points:
        point = (round(normalized_x * width), round(normalized_y * height))
        cv2.circle(output, point, 2, (255, 190, 30), -1, cv2.LINE_AA)
    state_color = {
        TrackingState.GOOD: (40, 190, 40),
        TrackingState.UNCERTAIN: (0, 180, 255),
        TrackingState.LOST: (30, 30, 220),
    }[observation.tracking_state]
    cv2.rectangle(output, (8, 8), (475, 102), (15, 15, 15), -1)
    cv2.putText(
        output,
        observation.tracking_state.value,
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        state_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"feature latency {observation.feature_latency_ms:.1f} ms",
        (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    if observation.head_pose is not None:
        pose = observation.head_pose
        text = (
            "head approx y/p/r "
            f"{pose.yaw_degrees:.1f}/{pose.pitch_degrees:.1f}/{pose.roll_degrees:.1f}"
        )
        cv2.putText(
            output,
            text,
            (18, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return output
