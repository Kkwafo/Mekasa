"""Blink-robust gaze smoothing.

The previous smoother (an exponential moving average) restarted from scratch every time the eyes
closed. The first frames after a blink are the worst of all: the lid is still half closed, the
eye aperture (a model feature) is low and the iris is partly covered, so the estimate jumps,
typically downwards. Restarting the average from those frames is what made the board look
"decalibrated" after blinking.

This smoother:

* keeps its state across short gaps (up to ``hold_seconds``), so a blink does not restart it;
* filters with the median of the last ``window`` valid samples before the moving average, so a
  few outlier frames are outvoted by the samples from just before the blink;
* lets the caller mark frames as *unreliable* (lid in transition): they never enter the filter,
  and the last smoothed value is held instead.
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
        alpha: EMA weight of the newest median (as the previous smoother).
        window: Valid samples in the median (7 ~ 0.23 s at 30 FPS).
        hold_seconds: Gap length after which the state is discarded (a real loss of signal).
    """

    alpha: float = 0.25
    window: int = 7
    hold_seconds: float = 0.6
    _samples: deque[tuple[float, Point]] = field(default_factory=deque)
    _value: Point | None = None
    _last_valid: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._value = None
        self._last_valid = None

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
        if not reliable:
            return self._value, self._value is not None
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


def lid_in_transition(
    features: dict[str, float] | None,
    eyes: tuple[str, ...],
    closed_threshold: float,
    *,
    factor: float = 1.35,
) -> bool:
    """True when a used eye is open but barely (closing or reopening around a blink).

    Downward gaze lowers the aperture too, but the personal closed threshold is fitted at 60 %
    of the bottom-row aperture, so ``1.35 x threshold`` (~81 %) stays below normal downward
    looks (2026-11-16 data: bottom-row EAR 0.12-0.13 with threshold 0.077-0.082).
    """
    if not features:
        return False
    limit = factor * closed_threshold
    for eye in eyes:
        value = features.get(f"{eye}_eye_aspect_ratio")
        if value is not None and value < limit:
            return True
    return False


__all__ = ["RobustGazeSmoother", "lid_in_transition"]
