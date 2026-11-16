"""Ocular access modes: both eyes, anatomical left only, anatomical right only.

The mode is an explicit operating decision made with the person and their support team.
It decides which eye features a calibration may use and which eye must be usable before a
gaze prediction is allowed to drive selection. The absent eye is represented explicitly
(it is never silently imputed in a single-eye mode).
"""

from __future__ import annotations

from enum import StrEnum

from mekasa_gaze.core.domain import EyeState


class EyeMode(StrEnum):
    """Which eye(s) may drive gaze selection.

    Names follow the person's anatomical side, independent of any mirrored preview.
    """

    BOTH = "BOTH"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


_EYE_PREFIXES = ("left_", "right_")


def feature_names_for_mode(names: tuple[str, ...], mode: EyeMode) -> tuple[str, ...]:
    """Restrict a calibration feature list to the eye(s) allowed by ``mode``.

    Non-ocular features (for example head pose) are always kept.

    Args:
        names: Ordered candidate feature names.
        mode: Ocular access mode.

    Returns:
        tuple[str, ...]: Ordered subset of ``names``.

    Raises:
        ValueError: If no feature of the required eye remains.
    """
    if mode is EyeMode.BOTH:
        selected = names
    else:
        excluded = "right_" if mode is EyeMode.LEFT else "left_"
        selected = tuple(name for name in names if not name.startswith(excluded))
    required = _required_prefixes(mode)
    for prefix in required:
        if not any(name.startswith(prefix) for name in selected):
            raise ValueError(f"Eye mode {mode.value} needs at least one {prefix}* feature")
    return selected


def eye_mode_of_features(names: tuple[str, ...]) -> EyeMode:
    """Infer the eye mode a calibration artifact was trained for from its feature names."""
    has_left = any(name.startswith("left_") for name in names)
    has_right = any(name.startswith("right_") for name in names)
    if has_left and has_right:
        return EyeMode.BOTH
    if has_left:
        return EyeMode.LEFT
    if has_right:
        return EyeMode.RIGHT
    raise ValueError("Calibration features contain no eye feature")


def eye_signal_usable(mode: EyeMode, left: EyeState, right: EyeState) -> bool:
    """Return whether the eye(s) required by ``mode`` are open and geometrically valid.

    In ``BOTH`` mode a single usable eye is not enough: the binocular calibration would have
    to invent the missing eye, so the session abstains instead.
    """
    if mode is EyeMode.LEFT:
        return left is EyeState.OPEN
    if mode is EyeMode.RIGHT:
        return right is EyeState.OPEN
    return left is EyeState.OPEN and right is EyeState.OPEN


def _required_prefixes(mode: EyeMode) -> tuple[str, ...]:
    if mode is EyeMode.LEFT:
        return ("left_",)
    if mode is EyeMode.RIGHT:
        return ("right_",)
    return _EYE_PREFIXES


__all__ = ["EyeMode", "eye_mode_of_features", "eye_signal_usable", "feature_names_for_mode"]
