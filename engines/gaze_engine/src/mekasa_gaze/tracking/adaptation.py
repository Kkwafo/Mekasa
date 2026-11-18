"""Keep adjusting to the person while the board is used, safely.

What re-analyses of sessions A/B showed (simulated drift, see docs/STATUS.md):

* Learning a full affine correction from a few selections can rotate or shear the board when
  the selections come from only two options (86 % -> 52 % with every selection correct).
* Learning only from selections cannot fix a row that does not register: with top-row
  selections alone the correction moved everything *up* and made the bottom row worse.
* Unsupervised learning from fixations amplified those errors.

So this module keeps only what did not hurt, with explicit safeguards:

* **Per-axis** stretch and shift (no rotation/shear), shrunk toward "no correction"; an axis
  with evidence from one side only keeps its scale and may shift at most a third of the limit.
* A selection becomes evidence only after ``confirm_seconds`` without being undone (clearing
  the message, going back or deleting right after choosing retracts it).
* Every sample remembers the face placement at which it was taken and is weighted by how close
  that placement is to the current one, so after the person moves, evidence from the previous
  posture fades instead of lingering. Samples also decay with age and are capped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import exp, log

import numpy as np

from mekasa_gaze.tracking.placement_monitor import PlacementReference
from mekasa_gaze.tracking.session_alignment import AffineCorrection

Point = tuple[float, float]

# Placement differences that count as "one unit" of posture change in the kernel.
_POSTURE_SCALES = (0.10, 0.05, 0.05, 8.0, 8.0)  # log face width, centre x, centre y, yaw, pitch


def posture_distance(a: PlacementReference | None, b: PlacementReference | None) -> float:
    """Normalized distance between two face placements (0 = same posture)."""
    if a is None or b is None:
        return 0.0
    deltas = (
        log(max(a.face_width, 1e-6) / max(b.face_width, 1e-6)),
        a.center_x - b.center_x,
        a.center_y - b.center_y,
        a.yaw - b.yaw,
        a.pitch - b.pitch,
    )
    return float(np.sqrt(sum((d / s) ** 2 for d, s in zip(deltas, _POSTURE_SCALES, strict=True))))


def fit_axis_correction(
    samples: Sequence[tuple[Point, Point, float]],
    *,
    center: Point = (0.5, 0.5),
    slope_prior: float = 0.3,
    offset_prior: float = 3.0,
    slope_limits: tuple[float, float] = (0.75, 1.4),
    max_offset: float = 0.12,
) -> AffineCorrection:
    """Per-axis scale and shift ``p' = c + a (p - c) + b`` from (target, predicted, weight).

    Each axis needs evidence on both sides of the centre before its scale moves; with one-sided
    evidence only a small shift (a third of ``max_offset``) is learned. ``slope_prior`` is in
    weight x squared screen units, ``offset_prior`` in weight.
    """
    origin = np.asarray(center, dtype=np.float64)
    slopes, offsets = np.ones(2), np.zeros(2)
    if samples:
        targets = np.asarray([t for t, _, _ in samples], dtype=np.float64)
        inputs = np.asarray([p for _, p, _ in samples], dtype=np.float64) - origin
        w = np.asarray([weight for _, _, weight in samples], dtype=np.float64)
        for axis in range(2):
            x, y = inputs[:, axis], targets[:, axis] - origin[axis]
            both_sides = bool(np.any(y < -0.05) and np.any(y > 0.05))
            prior = slope_prior if both_sides else 1e6
            limit = max_offset if both_sides else max_offset / 3.0
            gram = np.array(
                [[w @ (x * x) + prior, w @ x], [w @ x, float(np.sum(w)) + offset_prior]]
            )
            rhs = np.array([w @ (x * y) + prior, w @ y])
            slope, offset = np.linalg.solve(gram, rhs)
            slopes[axis] = float(np.clip(slope, *slope_limits))
            offsets[axis] = float(np.clip(offset, -limit, limit))
    return AffineCorrection(origin, np.ones(2), np.diag(slopes), origin + offsets, 0.0)


@dataclass(slots=True)
class _Sample:
    target: Point
    predicted: Point
    weight: float
    at: float
    posture: PlacementReference | None
    anchor: bool


@dataclass(slots=True)
class AdaptiveCorrection:
    """Posture-aware per-axis correction from alignment anchors and confirmed selections.

    Attributes:
        selection_weight: Weight of one confirmed selection.
        confirm_seconds: A selection counts only after this long without being retracted.
        half_life_seconds: Age at which a selection sample has lost half its weight.
        posture_width: Kernel width in posture units (1 = 10 % distance, 0.05 shift, 8 deg).
        max_selections: Cap on selection samples (oldest dropped first).
    """

    selection_weight: float = 6.0
    confirm_seconds: float = 4.0
    half_life_seconds: float = 600.0
    posture_width: float = 1.5
    max_selections: int = 40
    _samples: list[_Sample] = field(default_factory=list)

    @property
    def anchored(self) -> bool:
        return any(s.anchor for s in self._samples)

    @property
    def implicit_samples(self) -> int:
        return sum(1 for s in self._samples if not s.anchor)

    def reset(self) -> None:
        self._samples.clear()

    def clear_implicit(self) -> None:
        """Drop selection samples (e.g. after the model itself was re-fitted with them)."""
        self._samples = [s for s in self._samples if s.anchor]

    def set_anchors(
        self,
        samples: Sequence[tuple[Point, Point]],
        now: float = 0.0,
        posture: PlacementReference | None = None,
    ) -> None:
        """Replace the explicit alignment (selection samples keep their own posture)."""
        self._samples = [s for s in self._samples if not s.anchor]
        self._samples.extend(
            _Sample(target, predicted, 1.0, now, posture, True) for target, predicted in samples
        )

    def add_selection(
        self,
        predicted: Point,
        target: Point,
        now: float,
        posture: PlacementReference | None = None,
    ) -> None:
        self._samples.append(_Sample(target, predicted, self.selection_weight, now, posture, False))
        selections = [s for s in self._samples if not s.anchor]
        for old in selections[: max(0, len(selections) - self.max_selections)]:
            self._samples.remove(old)

    def retract_recent(self, now: float, window_seconds: float | None = None) -> bool:
        """Drop the last selection if it is recent (default: still unconfirmed)."""
        window = self.confirm_seconds if window_seconds is None else window_seconds
        for sample in reversed(self._samples):
            if not sample.anchor:
                if now - sample.at <= window:
                    self._samples.remove(sample)
                    return True
                return False
        return False

    def correction(
        self, now: float, posture: PlacementReference | None = None
    ) -> AffineCorrection | None:
        """Correction for the current posture, or None without (confirmed) evidence."""
        weighted: list[tuple[Point, Point, float]] = []
        for sample in self._samples:
            weight = sample.weight
            if not sample.anchor:
                age = now - sample.at
                if age < self.confirm_seconds:
                    continue  # not confirmed yet
                weight *= 0.5 ** (age / self.half_life_seconds)
            distance = posture_distance(sample.posture, posture)
            weight *= exp(-0.5 * (distance / self.posture_width) ** 2)
            if weight > 1e-4:
                weighted.append((sample.target, sample.predicted, weight))
        if not weighted:
            return None
        return fit_axis_correction(weighted)


@dataclass(slots=True)
class SelectionTelemetry:
    """Aggregate gaze-vs-option offsets at selection time, per row (no content, no screens)."""

    _offsets: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: {"top": [], "bottom": []}
    )
    retracted: int = 0

    def add(self, predicted: Point, target: Point, split_y: float) -> None:
        row = "top" if target[1] < split_y else "bottom"
        self._offsets[row].append((predicted[0] - target[0], predicted[1] - target[1]))

    def summary(self) -> dict[str, object]:
        result: dict[str, object] = {"retracted": self.retracted}
        for row, values in self._offsets.items():
            if values:
                array = np.asarray(values)
                result[row] = {
                    "selections": len(values),
                    "median_dx": float(np.median(array[:, 0])),
                    "median_dy": float(np.median(array[:, 1])),
                }
            else:
                result[row] = {"selections": 0}
        return result


__all__ = [
    "AdaptiveCorrection",
    "SelectionTelemetry",
    "fit_axis_correction",
    "posture_distance",
]
