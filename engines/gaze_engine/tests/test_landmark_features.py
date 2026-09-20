from __future__ import annotations

import pytest

from mekasa_gaze.landmarker import features_from_landmarks


def test_feature_extraction_uses_matching_anatomical_eye_and_iris_indices() -> None:
    landmarks = [(0.5, 0.5, 0.0) for _ in range(478)]
    # Subject's left eye: image-space x 0.6 -> 0.8.
    left_eye = {
        362: (0.6, 0.5, 0.0),
        385: (0.65, 0.45, 0.0),
        387: (0.75, 0.45, 0.0),
        263: (0.8, 0.5, 0.0),
        373: (0.75, 0.55, 0.0),
        380: (0.65, 0.55, 0.0),
    }
    # Subject's right eye: image-space x 0.2 -> 0.4.
    right_eye = {
        33: (0.2, 0.5, 0.0),
        160: (0.25, 0.45, 0.0),
        158: (0.35, 0.45, 0.0),
        133: (0.4, 0.5, 0.0),
        153: (0.35, 0.55, 0.0),
        144: (0.25, 0.55, 0.0),
    }
    for index, point in {**left_eye, **right_eye}.items():
        landmarks[index] = point
    for index in (473, 474, 475, 476, 477):
        landmarks[index] = (0.7, 0.5, 0.0)
    for index in (468, 469, 470, 471, 472):
        landmarks[index] = (0.3, 0.5, 0.0)
    landmarks[10] = (0.5, 0.2, 0.0)
    landmarks[152] = (0.5, 0.8, 0.0)
    landmarks[234] = (0.1, 0.5, 0.0)
    landmarks[454] = (0.9, 0.5, 0.0)

    features = features_from_landmarks(landmarks)

    assert features["left_eye_aspect_ratio"] == pytest.approx(0.5)
    assert features["right_eye_aspect_ratio"] == pytest.approx(0.5)
    assert features["left_iris_x"] == pytest.approx(0.5)
    assert features["right_iris_x"] == pytest.approx(0.5)
    assert features["left_iris_y"] == pytest.approx(0.5)
    assert features["right_iris_y"] == pytest.approx(0.5)
    assert features["face_center_x"] == pytest.approx(0.5)
    assert features["face_center_y"] == pytest.approx(0.5)
