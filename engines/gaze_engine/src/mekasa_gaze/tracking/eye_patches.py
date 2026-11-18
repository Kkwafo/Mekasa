"""Eye-aligned appearance patches for personal gaze regression.

Why: the landmark iris coordinates carried almost no vertical gaze signal in one
of two sessions (lid-normalized ``iris_y`` was flat; only eye aperture moved). Low-resolution
eye images capture what landmarks drop: how much sclera shows above and below the iris and the
shape of the upper lid, which follows vertical gaze. Ridge regression on small normalized eye
patches is the published WebGazer approach (Papoutsaki et al., IJCAI 2016), which reports
webcam gaze estimation from roughly 6x10-pixel eye patches with per-user calibration.

Each patch is cut along the line between the two eye corners (canthi), so it scales with the
eye (distance-invariant) and rotates with head roll. Patches exist only in memory: they are
never written to reports. A fitted model stores per-pixel means, which amount to an average
low-resolution eye image; that file stays local (see docs/PRIVACY in README).
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt

# Corner landmarks in image order (image-left corner first), matching landmarker.py.
LEFT_EYE_CORNERS = (362, 263)  # person's left eye: inner, outer
RIGHT_EYE_CORNERS = (33, 133)  # person's right eye: outer, inner

PATCH_WIDTH = 20
PATCH_HEIGHT = 10
_WIDTH_FACTOR = 1.5  # crop width relative to corner distance
_HEIGHT_FACTOR = 0.8  # crop height relative to corner distance (covers both lids)
_OVERSAMPLE = 2  # warp at 2x, then area-downsample to avoid aliasing

Patch = npt.NDArray[np.float32]


def eye_patch(
    rgb_frame: npt.NDArray[np.uint8],
    corner_a: tuple[float, float],
    corner_b: tuple[float, float],
) -> Patch | None:
    """Cut one eye patch between two normalized corner points.

    Returns:
        A flattened ``PATCH_HEIGHT * PATCH_WIDTH`` float32 vector with zero mean and unit
        standard deviation (illumination normalization), or None if the eye is too small
        or the crop is degenerate.
    """
    height, width = rgb_frame.shape[:2]
    a = np.array([corner_a[0] * width, corner_a[1] * height], dtype=np.float64)
    b = np.array([corner_b[0] * width, corner_b[1] * height], dtype=np.float64)
    axis = b - a
    distance = float(np.hypot(axis[0], axis[1]))
    if distance < 6.0:  # fewer than ~6 px across the eye: no usable appearance
        return None
    unit_x = axis / distance
    unit_y = np.array([-unit_x[1], unit_x[0]])  # image "down" for an upright face
    center = (a + b) / 2.0
    half_w = unit_x * distance * _WIDTH_FACTOR / 2.0
    half_h = unit_y * distance * _HEIGHT_FACTOR / 2.0
    out_w, out_h = PATCH_WIDTH * _OVERSAMPLE, PATCH_HEIGHT * _OVERSAMPLE
    source = np.asarray(
        [center - half_w - half_h, center + half_w - half_h, center - half_w + half_h],
        dtype=np.float32,
    )
    target = np.asarray([[0, 0], [out_w, 0], [0, out_h]], dtype=np.float32)
    matrix = cv2.getAffineTransform(source, target)
    warped = cv2.warpAffine(
        rgb_frame, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    small = cv2.resize(gray, (PATCH_WIDTH, PATCH_HEIGHT), interpolation=cv2.INTER_AREA)
    vector = small.astype(np.float32).reshape(-1)
    spread = float(vector.std())
    if spread < 1e-3:
        return None
    return ((vector - float(vector.mean())) / spread).astype(np.float32)


def eye_patches_from_landmarks(
    rgb_frame: npt.NDArray[np.uint8],
    landmarks: Sequence[tuple[float, float, float]],
) -> tuple[Patch | None, Patch | None]:
    """(left eye patch, right eye patch) using the anatomical naming of landmarker.py."""

    def corner(index: int) -> tuple[float, float]:
        return (landmarks[index][0], landmarks[index][1])

    left = eye_patch(rgb_frame, corner(LEFT_EYE_CORNERS[0]), corner(LEFT_EYE_CORNERS[1]))
    right = eye_patch(rgb_frame, corner(RIGHT_EYE_CORNERS[0]), corner(RIGHT_EYE_CORNERS[1]))
    return left, right


__all__ = [
    "PATCH_HEIGHT",
    "PATCH_WIDTH",
    "Patch",
    "eye_patch",
    "eye_patches_from_landmarks",
]
