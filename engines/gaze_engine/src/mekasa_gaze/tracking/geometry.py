from __future__ import annotations

from collections.abc import Sequence
from math import asin, atan2, degrees, sqrt

import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.domain import HeadPose

Point3 = tuple[float, float, float]


def euclidean_2d(a: Point3, b: Point3) -> float:
    """Compute 2D Euclidean distance between two 3D points projected on the XY plane.

    Responsibility:
        Calculates planar distance between facial coordinates, ignoring depth (Z).

    Args:
        a: First point tuple (x, y, z).
        b: Second point tuple (x, y, z).

    Returns:
        float: Planar Euclidean distance hypot(dx, dy).
    """
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def eye_aspect_ratio(points: Sequence[Point3]) -> float:
    """Calculate the Eye Aspect Ratio (EAR) from six eyelid landmark coordinates.

    Responsibility:
        Computes the Soukupová & Čech (2016) ratio of vertical eyelid opening to horizontal
        palpebral fissure width to detect blinks and measure eye openness.

    Args:
        points: Ordered sequence of 6 landmarks:
            [outer_corner, upper_outer, upper_inner, inner_corner, lower_inner, lower_outer].

    Returns:
        float: Scalar aspect ratio; lower values indicate eye closure / blinking.

    Raises:
        ValueError: If landmark sequence length is not 6 or eye corner distance is zero.
    """
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
    """Calculate invariant normalized horizontal and vertical iris position ratios.

    Responsibility:
        Projects the iris center centroid onto anatomical reference vectors defined by
        palpebral fissure corners and eyelid centers, yielding normalized coordinates
        in the range [0.0, 1.0] invariant to head translation and scaling.

    Args:
        iris_points: Sequence of 3D landmarks delineating the iris boundary or center.
        horizontal_start: Eye reference corner (outer corner).
        horizontal_end: Eye reference corner (inner corner).
        upper_lid: Landmark representing central upper eyelid margin.
        lower_lid: Landmark representing central lower eyelid margin.

    Returns:
        tuple[float, float]: Normalized (horizontal_ratio, vertical_ratio) where ~0.5 is centered.

    Raises:
        ValueError: If iris_points is empty or either corner/eyelid distance is zero.
    """
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


def canthal_vertical_offset(
    iris_points: Sequence[Point3],
    corner_start: Point3,
    corner_end: Point3,
) -> float:
    """Signed vertical offset of the iris centre from the line joining both eye corners.

    The eye corners (canthi) are anchored to the skull, unlike the eyelids, which follow
    vertical gaze and partly cancel the lid-normalized vertical ratio. The offset is the 2D
    cross product divided by the squared corner distance, so it is scale invariant; it is
    positive when the iris is below the corner line in image coordinates (corners must be
    ordered image-left to image-right, as in the landmarker topology).

    Raises:
        ValueError: If no iris point is given or the corner distance is zero.
    """
    if not iris_points:
        raise ValueError("At least one iris point is required")
    iris = np.mean(np.asarray(iris_points, dtype=np.float64)[:, :2], axis=0)
    start = np.asarray(corner_start[:2], dtype=np.float64)
    axis = np.asarray(corner_end[:2], dtype=np.float64) - start
    norm_sq = float(np.dot(axis, axis))
    if norm_sq <= 1e-12:
        raise ValueError("Eye corner distance is zero")
    offset = iris - start
    return float((axis[0] * offset[1] - axis[1] * offset[0]) / norm_sq)


def euler_from_transform(matrix: npt.ArrayLike) -> HeadPose:
    """Extract diagnostic pitch, roll, and yaw Euler angles from a 4x4 facial transform.

    Responsibility:
        Normalizes column vectors of the 3x3 rotation component to remove scaling artifacts,
        then extracts diagnostic XYZ rotation angles in degrees.

    Args:
        matrix: 4x4 facial transformation matrix returned by MediaPipe face geometry pipeline.

    Returns:
        HeadPose: Euler angles in degrees (yaw_degrees, pitch_degrees, roll_degrees).

    Raises:
        ValueError: If input shape is not (4, 4) or rotation matrix is degenerate.
    """
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
