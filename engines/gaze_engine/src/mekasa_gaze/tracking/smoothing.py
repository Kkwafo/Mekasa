"""Blink-robust gaze smoothing.

The original smoother (an exponential moving average) restarted from scratch every time the
eyes closed. The first frames after a blink are the worst of all: the lid is still half closed,
the eye aperture (a model feature) is low and the iris is partly covered, so the estimate jumps,
typically downwards.

This smoother:

* keeps its state across short gaps (up to ``hold_seconds``), so a blink does not restart it;
* filters with the median of the last ``window`` valid samples before the moving average, so a
  few outlier frames are outvoted by the samples from just before the blink;
* lets the caller mark frames as *unreliable* (lid moving fast around a blink): they never
  enter the filter and the last value is held, but never longer than ``max_hold_seconds``.

Correction: the first version flagged frames by a fixed aperture *level*. Looking
further down than during calibration (e.g. after raising the chair) keeps the lid low for a long
time, so the estimate stayed frozen on the previous (upper) option: "it did not detect the
bottom row". Re-analysis of sessions A/B with bottom-row apertures scaled by 0.8 reproduced it
(no usable bottom-row output in A) and the velocity rule below fixed it (bottom row 100 %,
<= 1 % of frames held). See docs/STATUS.md.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]


@dataclass(slots=True)
class RobustGazeSmoother:
    """Median-then-EMA smoother that survives blinks.

    Attributes:
        alpha: EMA weight of the newest median.
        window: Valid samples in the median (7 ~ 0.23 s at 30 FPS).
        hold_seconds: Gap length after which the state is discarded (a real loss of signal).
        max_hold_seconds: Longest time an unreliable stretch may hold the previous value.
    """

    alpha: float = 0.25
    window: int = 7
    hold_seconds: float = 0.6
    max_hold_seconds: float = 0.3
    _samples: deque[tuple[float, Point]] = field(default_factory=deque)
    _value: Point | None = None
    _last_valid: float | None = None
    _held_since: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._value = None
        self._last_valid = None
        self._held_since = None

    def update(
        self, now: float, point: Point | None, *, reliable: bool = True
    ) -> tuple[Point | None, bool]:
        """Feed one frame. Returns (smoothed point or None, held).

        ``held`` is True when the returned point is the previous estimate kept through an
        unreliable frame. A ``None`` point (eyes closed, no prediction) returns None but keeps
        the state for ``hold_seconds``.
        """
        if self._last_valid is not None and now - self._last_valid > self.hold_seconds:
            self.reset()
        if point is None:
            return None, False
        if not reliable and self._value is not None:
            if self._held_since is None:
                self._held_since = now
            if now - self._held_since <= self.max_hold_seconds:
                return self._value, True
            # Past the limit: accept frames until the stretch ends (no new hold inside it).
        if reliable:
            self._held_since = None
        self._samples.append((now, point))
        while len(self._samples) > self.window:
            self._samples.popleft()
        while self._samples and now - self._samples[0][0] > self.hold_seconds:
            self._samples.popleft()
        median = np.median(np.asarray([p for _, p in self._samples]), axis=0)
        current = (float(median[0]), float(median[1]))
        if self._value is None:
            self._value = current
        else:
            self._value = (
                (1.0 - self.alpha) * self._value[0] + self.alpha * current[0],
                (1.0 - self.alpha) * self._value[1] + self.alpha * current[1],
            )
        self._last_valid = now
        return self._value, False


@dataclass(slots=True)
class LidTransitionGate:
    """Flags frames where the eye aperture is *dropping fast* (a lid closing for a blink).

    A sustained low aperture (looking down) is not a transition: after ``window_seconds`` the
    recent peak follows the new level and frames are reliable again.

    Attributes:
        drop: Aperture below this fraction of the recent peak counts as a fast drop.
        window_seconds: How far back the recent peak is taken (0.15 s ~ 4-5 frames).
    """

    drop: float = 0.75
    window_seconds: float = 0.15
    _history: deque[tuple[float, float]] = field(default_factory=deque)

    def reset(self) -> None:
        self._history.clear()

    def in_transition(self, now: float, aperture: float | None) -> bool:
        if aperture is None:
            return False
        while self._history and now - self._history[0][0] > self.window_seconds:
            self._history.popleft()
        peak = max((value for _, value in self._history), default=aperture)
        self._history.append((now, aperture))
        return aperture < self.drop * peak


def min_aperture(features: dict[str, float] | None, eyes: tuple[str, ...]) -> float | None:
    """Smallest eye aspect ratio among the eyes in use."""
    if not features:
        return None
    values = [features.get(f"{eye}_eye_aspect_ratio") for eye in eyes]
    usable = [float(value) for value in values if value is not None]
    return min(usable) if usable else None


__all__ = ["LidTransitionGate", "RobustGazeSmoother", "min_aperture"]
