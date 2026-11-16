"""Calibration recorded at the start of each board session (about 35 s).

Sequence: a 3x3 grid of points inside the board area, shown in random order (round 1); then
the centre point while the person slowly moves the head (teaches the model that head motion
alone does not move the gaze; in Mekasa data this raised 2x2 accuracy from 81 % to 95 % in
session A and from 94 % to 98 % in session B); then the grid again in a new order (round 2).
Round 2 is the held-out fold for round 1 and vice versa.

The procedure is clock-injected and only collects; fitting happens in ``gaze_model``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from mekasa_gaze.core.domain import FeatureObservation
from mekasa_gaze.tracking.gaze_model import CalibrationSample, sample_from_observation

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class CalibrationStep:
    """One point to look at."""

    target: Point
    round_index: int
    grid: bool  # False for the head-movement step
    settle_seconds: float
    collect_seconds: float


@dataclass(frozen=True, slots=True)
class CalibrationSnapshot:
    """What to draw now."""

    index: int
    total: int
    target: Point | None
    collecting: bool
    progress: float
    done: bool
    head_movement: bool = False
    seconds_left: float = 0.0


def grid_targets(
    area: tuple[float, float, float, float], fractions: tuple[float, ...] = (0.12, 0.5, 0.88)
) -> list[Point]:
    """3x3 grid in board coordinates, mapped to the screen."""
    left, top, right, bottom = area
    return [
        (left + fx * (right - left), top + fy * (bottom - top))
        for fy in fractions
        for fx in fractions
    ]


def calibration_steps(
    area: tuple[float, float, float, float],
    *,
    rounds: int = 2,
    settle_seconds: float = 0.7,
    collect_seconds: float = 1.0,
    head_movement_seconds: float = 4.0,
    seed: int = 7,
) -> list[CalibrationStep]:
    """Round 1, head-movement segment at the centre, round 2 (fixed seed for repeatability)."""
    rng = random.Random(seed)
    targets = grid_targets(area)
    center = targets[4]
    steps: list[CalibrationStep] = []
    for round_index in range(rounds):
        order = targets[:]
        rng.shuffle(order)
        steps.extend(
            CalibrationStep(target, round_index, True, settle_seconds, collect_seconds)
            for target in order
        )
        if round_index == 0 and head_movement_seconds > 0:
            steps.append(CalibrationStep(center, 0, False, 1.2, head_movement_seconds))
    return steps


def estimated_seconds(steps: list[CalibrationStep]) -> float:
    return sum(step.settle_seconds + step.collect_seconds for step in steps)


@dataclass(slots=True)
class CalibrationProcedure:
    """Walks the steps; each step settles, then collects until enough frames or a timeout.

    Attributes:
        steps: Sequence from :func:`calibration_steps`.
        min_samples: Valid frames needed before a grid step may end.
        max_extra_seconds: Extra collection time allowed when frames are missing (blinks,
            lost tracking) before moving on regardless.
    """

    steps: list[CalibrationStep]
    min_samples: int = 12
    max_extra_seconds: float = 1.5
    samples: list[CalibrationSample] = field(default_factory=list)
    _index: int = 0
    _started_at: float | None = None
    _step_samples: int = 0
    _short_steps: int = 0

    @property
    def short_steps(self) -> int:
        """Steps that ended without ``min_samples`` valid frames."""
        return self._short_steps

    def update(self, now: float, observation: FeatureObservation | None) -> CalibrationSnapshot:
        """Advance one tick. Pass an observation only when it is usable for calibration."""
        total = len(self.steps)
        if self._index >= total:
            return CalibrationSnapshot(total, total, None, False, 1.0, True)
        if self._started_at is None:
            self._started_at = now
        step = self.steps[self._index]
        elapsed = now - self._started_at
        collecting = elapsed >= step.settle_seconds
        if collecting and observation is not None:
            self.samples.append(
                sample_from_observation(
                    observation, step.target, step.round_index, now, grid=step.grid
                )
            )
            self._step_samples += 1
        collected_for = max(0.0, elapsed - step.settle_seconds)
        enough = self._step_samples >= self.min_samples
        if (collected_for >= step.collect_seconds and enough) or (
            collected_for >= step.collect_seconds + self.max_extra_seconds
        ):
            self._short_steps += 0 if enough else 1
            self._index += 1
            self._started_at = now
            self._step_samples = 0
            if self._index >= total:
                return CalibrationSnapshot(total, total, None, False, 1.0, True)
            step = self.steps[self._index]
            collecting, collected_for, elapsed = False, 0.0, 0.0
        remaining = sum(s.settle_seconds + s.collect_seconds for s in self.steps[self._index :])
        return CalibrationSnapshot(
            index=self._index,
            total=total,
            target=step.target,
            collecting=collecting,
            progress=min(1.0, collected_for / step.collect_seconds) if collecting else 0.0,
            done=False,
            head_movement=not step.grid,
            seconds_left=max(0.0, remaining - elapsed),
        )


__all__ = [
    "CalibrationProcedure",
    "CalibrationSnapshot",
    "CalibrationStep",
    "calibration_steps",
    "estimated_seconds",
    "grid_targets",
]
