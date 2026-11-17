"""Synthetic eyes for camera-free tests of the personal gaze model and the board runner.

The drawing is deliberately crude: sclera, iris and an upper lid that drops with downward gaze
(as real eyelids do). It tests software behaviour, not real-world accuracy.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.domain import FeatureObservation, HeadPose, TrackingState
from mekasa_gaze.tracking.eye_patches import eye_patches_from_landmarks

SKIN = (150, 170, 200)


def draw_eye(
    image: npt.NDArray[np.uint8],
    corner_a: tuple[float, float],
    corner_b: tuple[float, float],
    gaze: tuple[float, float],
) -> None:
    h, w = image.shape[:2]
    ax, ay = corner_a[0] * w, corner_a[1] * h
    bx, by = corner_b[0] * w, corner_b[1] * h
    cx, cy = (ax + bx) / 2, (ay + by) / 2
    width = abs(bx - ax)
    cv2.ellipse(
        image, (int(cx), int(cy)), (int(width / 2), int(width / 4)), 0, 0, 360, (235, 235, 235), -1
    )
    iris_x = cx + (gaze[0] - 0.5) * width * 0.45
    iris_y = cy + (gaze[1] - 0.5) * width * 0.18
    cv2.circle(image, (int(iris_x), int(iris_y)), int(width * 0.18), (40, 30, 20), -1)
    lid = cy - width / 4 + gaze[1] * width * 0.22
    cv2.rectangle(image, (int(ax) - 4, int(cy - width / 2)), (int(bx) + 4, int(lid)), SKIN, -1)


def face_image(
    gaze: tuple[float, float], scale: float = 1.0, shift: float = 0.0
) -> tuple[npt.NDArray[np.uint8], list[tuple[float, float, float]]]:
    """RGB frame with two eyes looking at ``gaze`` (0..1 each axis) and matching landmarks."""
    image = np.full((480, 640, 3), SKIN, dtype=np.uint8)
    half = 0.06 * scale
    right = ((0.40 + shift - half, 0.45), (0.40 + shift + half, 0.45))  # image-left eye
    left = ((0.60 + shift - half, 0.45), (0.60 + shift + half, 0.45))
    draw_eye(image, *right, gaze)
    draw_eye(image, *left, gaze)
    landmarks: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 478
    landmarks[33] = (*right[0], 0.0)
    landmarks[133] = (*right[1], 0.0)
    landmarks[362] = (*left[0], 0.0)
    landmarks[263] = (*left[1], 0.0)
    return image, landmarks


def observation_for(
    gaze: tuple[float, float],
    *,
    sequence: int,
    timestamp_ns: int,
    vertical_in_landmarks: bool = False,
    face_width: float = 0.35,
    eyes_closed: bool = False,
    rng: np.random.Generator | None = None,
) -> FeatureObservation:
    """Observation of a person looking at ``gaze``.

    With ``vertical_in_landmarks=False`` the landmark iris features carry no vertical signal
    (the 2026-11-16 session B situation); only the eye images and a weak eye aperture do.
    """
    rng = rng or np.random.default_rng(sequence)
    image, landmarks = face_image(gaze, scale=face_width / 0.35)
    left_patch, right_patch = eye_patches_from_landmarks(image, landmarks)
    noise = rng.normal(0, 0.2, 200).astype(np.float32)
    patches = (
        None if left_patch is None else left_patch + noise,
        None if right_patch is None else right_patch + noise,
    )
    iris_y = gaze[1] if vertical_in_landmarks else 0.5
    features = {
        "face_width": face_width,
        "face_height": face_width * 1.3,
        "face_center_x": 0.5,
        "face_center_y": 0.5,
    }
    for eye in ("left", "right"):
        features.update(
            {
                f"{eye}_iris_x": gaze[0] + float(rng.normal(0, 0.02)),
                f"{eye}_iris_y": iris_y + float(rng.normal(0, 0.02)),
                f"{eye}_iris_v": iris_y + float(rng.normal(0, 0.02)),
                f"{eye}_eye_aspect_ratio": 0.03
                if eyes_closed
                else 0.30 - 0.05 * gaze[1] + float(rng.normal(0, 0.02)),
                f"{eye}_eye_vertical_span": 0.02,
            }
        )
    return FeatureObservation(
        sequence,
        timestamp_ns,
        TrackingState.GOOD,
        features=features,
        head_pose=HeadPose(
            yaw_degrees=float(rng.normal(0, 1)),
            pitch_degrees=float(rng.normal(8, 1)),
            roll_degrees=float(rng.normal(0, 1)),
        ),
        eye_patches=patches,
    )
