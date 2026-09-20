from __future__ import annotations

from collections.abc import Sequence
from math import asin, atan2, degrees, sqrt

import numpy as np
import numpy.typing as npt

from mekasa_gaze.domain import HeadPose

Point3 = tuple[float, float, float]


def euclidean_2d(a: Point3, b: Point3) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def eye_aspect_ratio(points: Sequence[Point3]) -> float:
    """Return EAR for [outer, upper_outer, upper_inner, inner, lower_inner, lower_outer]."""
    if len(points) != 6:
        raise ValueError("Eye aspect ratio requires exactly six points")
    horizontal = euclidean_2d(points[0], points[3])
    if horizontal <= 1e-9:
        raise ValueError("Eye corner distance is zero")
    vertical = euclidean_2d(points[1], points[5]) + euclidean_2d(points[2], points[4])
    return vertical / (2.0 * horizontal)


def normalized_iris_position(
    iris_points: Sequence[Point3],
    horizontal_start: Point3,
    horizontal_end: Point3,
    upper_lid: Point3,
    lower_lid: Point3,
) -> tuple[float, float]:
    if not iris_points:
        raise ValueError("At least one iris point is required")
    iris = np.mean(np.asarray(iris_points, dtype=np.float64)[:, :2], axis=0)
    start = np.asarray(horizontal_start[:2], dtype=np.float64)
    end = np.asarray(horizontal_end[:2], dtype=np.float64)
    axis = end - start
    axis_norm_sq = float(np.dot(axis, axis))
    if axis_norm_sq <= 1e-12:
        raise ValueError("Eye corner distance is zero")
    horizontal = float(np.dot(iris - start, axis) / axis_norm_sq)

    upper = np.asarray(upper_lid[:2], dtype=np.float64)
    lower = np.asarray(lower_lid[:2], dtype=np.float64)
    lid_axis = lower - upper
    lid_norm_sq = float(np.dot(lid_axis, lid_axis))
    if lid_norm_sq <= 1e-12:
        raise ValueError("Eyelid distance is zero")
    vertical = float(np.dot(iris - upper, lid_axis) / lid_norm_sq)
    return horizontal, vertical


def euler_from_transform(matrix: npt.ArrayLike) -> HeadPose:
    """Extract diagnostic XYZ Euler angles from a 4x4 facial transform."""
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {transform.shape}")
    rotation = transform[:3, :3]
    # Remove uniform/non-uniform scale before reading rotation.
    norms = np.linalg.norm(rotation, axis=0)
    if np.any(norms <= 1e-12):
        raise ValueError("Degenerate head-pose rotation")
    rotation = rotation / norms

    value = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    pitch = asin(value)
    cosine = sqrt(max(0.0, 1.0 - value * value))
    if cosine > 1e-6:
        roll = atan2(rotation[2, 1], rotation[2, 2])
        yaw = atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return HeadPose(
        yaw_degrees=degrees(yaw),
        pitch_degrees=degrees(pitch),
        roll_degrees=degrees(roll),
    )
