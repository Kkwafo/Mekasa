"""Where the person sits now, compared with where they sat during calibration.

Two uses:

* **Distance compensation.** For a fixed eye rotation, the on-screen displacement from the
  camera grows with viewing distance (displacement = distance x tan(angle)). Face width in the
  image is inversely proportional to distance, so ``scale = reference_width / current_width``
  approximates ``current_distance / calibration_distance``. Predictions are scaled about the
  camera's position on the screen by that factor. This is textbook projective geometry, but
  its benefit for Mekasa is an engineering hypothesis until measured.
* **Drift detection.** When the face moves or turns well beyond the calibration pose and stays
  there, the board can run a 4-point re-alignment automatically, and the reference is rebased
  to the new pose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import log

import numpy as np


@dataclass(frozen=True, slots=True)
class PlacementReference:
    """Median face placement during calibration (image-normalized units, degrees)."""

    face_width: float
    center_x: float
    center_y: float
    yaw: float
    pitch: float

    def to_dict(self) -> dict[str, float]:
        return {
            "face_width": self.face_width,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "yaw": self.yaw,
            "pitch": self.pitch,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> PlacementReference:
        return cls(
            float(data["face_width"]),
            float(data["center_x"]),
            float(data["center_y"]),
            float(data["yaw"]),
            float(data["pitch"]),
        )


def placement_vector(features: Mapping[str, float]) -> tuple[float, ...] | None:
    """(face_width, center_x, center_y, yaw, pitch) from a feature map, or None if incomplete."""
    names = (
        "face_width",
        "face_center_x",
        "face_center_y",
        "head_yaw_degrees",
        "head_pitch_degrees",
    )
    try:
        values = tuple(float(features[name]) for name in names)
    except KeyError:
        return None
    if not all(np.isfinite(values)) or values[0] <= 0.0:
        return None
    return values


def reference_from(vectors: Sequence[tuple[float, ...]]) -> PlacementReference | None:
    """Median placement over calibration frames."""
    if not vectors:
        return None
    median = np.median(np.asarray(vectors, dtype=np.float64), axis=0)
    return PlacementReference(*(float(value) for value in median))


class Drift(StrEnum):
    """Coarse placement change relative to the reference."""

    OK = "OK"
    CLOSER = "CLOSER"
    FARTHER = "FARTHER"
    MOVED = "MOVED"
    TURNED = "TURNED"


@dataclass(frozen=True, slots=True)
class PlacementStatus:
    """Current placement relative to the reference."""

    drift: Drift
    distance_scale: float  # ~ current distance / reference distance
    sustained_seconds: float  # how long the current non-OK drift has lasted


@dataclass(slots=True)
class PlacementMonitor:
    """Smooths placement and reports drift; clock-injected.

    Attributes:
        reference: Calibration placement (None disables compensation and drift).
        alpha: EMA weight per frame (0.08 ~ 0.4 s time constant at 30 FPS).
        distance_tolerance: |log(scale)| beyond this counts as closer/farther (0.22 ~ 25 %;
            smaller changes are left to the distance compensation, no re-alignment).
        shift_tolerance: Face-centre shift (image fraction) counted as moved.
        angle_tolerance: Yaw/pitch change in degrees counted as turned (the calibration's
            head-movement segment already teaches the model some head rotation).
        scale_limits: Clamp on the compensation factor (about 0.65x to 1.55x the distance).
    """

    reference: PlacementReference | None = None
    alpha: float = 0.08
    distance_tolerance: float = 0.22
    shift_tolerance: float = 0.10
    angle_tolerance: float = 14.0
    scale_limits: tuple[float, float] = (0.65, 1.55)
    _state: np.ndarray | None = None
    _drift_since: float | None = None
    _last_drift: Drift = Drift.OK

    def rebase(self, reference: PlacementReference | None = None) -> None:
        """Make ``reference`` (or the current smoothed placement) the new reference."""
        if reference is None and self._state is not None:
            reference = PlacementReference(*(float(value) for value in self._state))
        self.reference = reference
        self._drift_since = None
        self._last_drift = Drift.OK

    def current(self) -> PlacementReference | None:
        """Smoothed current placement."""
        if self._state is None:
            return None
        return PlacementReference(*(float(value) for value in self._state))

    def update(self, now: float, features: Mapping[str, float] | None) -> PlacementStatus:
        """Feed one frame (None when no face); returns the status."""
        vector = placement_vector(features) if features is not None else None
        if vector is not None:
            current = np.asarray(vector, dtype=np.float64)
            self._state = (
                current
                if self._state is None
                else (1.0 - self.alpha) * self._state + self.alpha * current
            )
        return self._status(now)

    def _status(self, now: float) -> PlacementStatus:
        if self.reference is None or self._state is None:
            return PlacementStatus(Drift.OK, 1.0, 0.0)
        width, center_x, center_y, yaw, pitch = self._state
        reference = self.reference
        ratio = reference.face_width / width
        low, high = self.scale_limits
        scale = min(high, max(low, ratio))
        drift = Drift.OK
        if abs(log(ratio)) > self.distance_tolerance:
            drift = Drift.CLOSER if ratio < 1.0 else Drift.FARTHER
        elif (
            abs(center_x - reference.center_x) > self.shift_tolerance
            or abs(center_y - reference.center_y) > self.shift_tolerance
        ):
            drift = Drift.MOVED
        elif (
            abs(yaw - reference.yaw) > self.angle_tolerance
            or abs(pitch - reference.pitch) > self.angle_tolerance
        ):
            drift = Drift.TURNED
        if drift is Drift.OK:
            self._drift_since = None
        elif self._drift_since is None or drift is not self._last_drift:
            self._drift_since = now
        self._last_drift = drift
        sustained = 0.0 if self._drift_since is None else now - self._drift_since
        return PlacementStatus(drift, scale, sustained)


def compensate_distance(
    point: tuple[float, float], scale: float, camera: tuple[float, float]
) -> tuple[float, float]:
    """Scale a screen-normalized prediction about the camera position by ``scale``."""
    return (
        camera[0] + (point[0] - camera[0]) * scale,
        camera[1] + (point[1] - camera[1]) * scale,
    )


DRIFT_HINTS = {
    Drift.CLOSER: "Está más cerca que al calibrar",
    Drift.FARTHER: "Está más lejos que al calibrar",
    Drift.MOVED: "Se movió respecto de la calibración",
    Drift.TURNED: "Giró la cabeza respecto de la calibración",
}


__all__ = [
    "DRIFT_HINTS",
    "Drift",
    "PlacementMonitor",
    "PlacementReference",
    "PlacementStatus",
    "compensate_distance",
    "placement_vector",
    "reference_from",
]
