from __future__ import annotations

import numpy as np
import pytest

from mekasa_gaze.tracking.geometry import (
    euler_from_transform,
    eye_aspect_ratio,
    normalized_iris_position,
)


def test_eye_aspect_ratio_uses_scale_invariant_geometry() -> None:
    eye = [
        (0.0, 0.0, 0.0),
        (0.25, -0.1, 0.0),
        (0.75, -0.1, 0.0),
        (1.0, 0.0, 0.0),
        (0.75, 0.1, 0.0),
        (0.25, 0.1, 0.0),
    ]
    scaled = [(x * 4, y * 4, z) for x, y, z in eye]
    assert eye_aspect_ratio(eye) == pytest.approx(0.2)
    assert eye_aspect_ratio(scaled) == pytest.approx(eye_aspect_ratio(eye))


def test_normalized_iris_position_is_centered() -> None:
    position = normalized_iris_position(
        [(0.5, 0.5, 0.0)],
        (0.0, 0.5, 0.0),
        (1.0, 0.5, 0.0),
        (0.5, 0.0, 0.0),
        (0.5, 1.0, 0.0),
    )
    assert position == pytest.approx((0.5, 0.5))


def test_euler_identity_is_zero() -> None:
    pose = euler_from_transform(np.identity(4))
    assert pose.yaw_degrees == pytest.approx(0.0)
    assert pose.pitch_degrees == pytest.approx(0.0)
    assert pose.roll_degrees == pytest.approx(0.0)


def test_degenerate_transform_is_rejected() -> None:
    with pytest.raises(ValueError, match="Degenerate"):
        euler_from_transform(np.zeros((4, 4)))


def test_canthal_vertical_offset_is_signed_and_scale_invariant() -> None:
    from mekasa_gaze.tracking.geometry import canthal_vertical_offset

    start, end = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    assert canthal_vertical_offset([(0.5, 0.1, 0.0)], start, end) == pytest.approx(0.1)
    assert canthal_vertical_offset([(0.5, -0.1, 0.0)], start, end) == pytest.approx(-0.1)
    scaled = canthal_vertical_offset([(1.0, 0.2, 0.0)], (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert scaled == pytest.approx(0.1)
    with pytest.raises(ValueError):
        canthal_vertical_offset([(0.5, 0.1, 0.0)], start, start)
