from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RegionName(StrEnum):
    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"


_BOUNDS: dict[RegionName, tuple[float, float, float, float]] = {
    RegionName.TOP_LEFT: (0.0, 0.0, 0.5, 0.5),
    RegionName.TOP_RIGHT: (0.5, 0.0, 1.0, 0.5),
    RegionName.BOTTOM_LEFT: (0.0, 0.5, 0.5, 1.0),
    RegionName.BOTTOM_RIGHT: (0.5, 0.5, 1.0, 1.0),
}


@dataclass(frozen=True, slots=True)
class SelectionUpdate:
    candidate: RegionName | None
    dwell_progress: float
    selected: RegionName | None
    selection_fired: bool
    cooldown_remaining_seconds: float
    tracking_available: bool


class DwellSelector:
    """Four-region dwell selector with hysteresis, cooldown and loss reset."""

    def __init__(
        self,
        *,
        dwell_seconds: float,
        cooldown_seconds: float,
        hysteresis_margin: float,
    ) -> None:
        if dwell_seconds <= 0.0:
            raise ValueError("dwell_seconds must be positive")
        if cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds must be non-negative")
        if not 0.0 <= hysteresis_margin < 0.25:
            raise ValueError("hysteresis_margin must be in [0, 0.25)")
        self._dwell_seconds = dwell_seconds
        self._cooldown_seconds = cooldown_seconds
        self._hysteresis_margin = hysteresis_margin
        self._candidate: RegionName | None = None
        self._candidate_since: float | None = None
        self._cooldown_until = 0.0
        self._last_selected: RegionName | None = None

    def update(
        self,
        prediction: tuple[float, float] | None,
        now_seconds: float,
    ) -> SelectionUpdate:
        if prediction is None:
            self._reset_candidate()
            return SelectionUpdate(None, 0.0, None, False, 0.0, False)

        cooldown_remaining = max(0.0, self._cooldown_until - now_seconds)
        if cooldown_remaining > 0.0:
            self._reset_candidate()
            return SelectionUpdate(
                None,
                0.0,
                self._last_selected,
                False,
                cooldown_remaining,
                True,
            )
        self._last_selected = None

        region = self._candidate_region(prediction)
        if region is None:
            self._reset_candidate()
            return SelectionUpdate(None, 0.0, None, False, 0.0, True)
        if region is not self._candidate:
            self._candidate = region
            self._candidate_since = now_seconds

        assert self._candidate_since is not None
        elapsed = max(0.0, now_seconds - self._candidate_since)
        progress = min(1.0, elapsed / self._dwell_seconds)
        if progress < 1.0:
            return SelectionUpdate(region, progress, None, False, 0.0, True)

        self._last_selected = region
        self._cooldown_until = now_seconds + self._cooldown_seconds
        self._reset_candidate()
        return SelectionUpdate(
            None,
            0.0,
            region,
            True,
            self._cooldown_seconds,
            True,
        )

    def _candidate_region(self, point: tuple[float, float]) -> RegionName | None:
        if self._candidate is not None and _inside(
            point,
            _expanded(_BOUNDS[self._candidate], self._hysteresis_margin),
        ):
            return self._candidate
        return region_at(point)

    def _reset_candidate(self) -> None:
        self._candidate = None
        self._candidate_since = None


def region_at(point: tuple[float, float]) -> RegionName | None:
    x, y = point
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        return None
    if y < 0.5:
        return RegionName.TOP_LEFT if x < 0.5 else RegionName.TOP_RIGHT
    return RegionName.BOTTOM_LEFT if x < 0.5 else RegionName.BOTTOM_RIGHT


def region_bounds(region: RegionName) -> tuple[float, float, float, float]:
    return _BOUNDS[region]


def _expanded(
    bounds: tuple[float, float, float, float], margin: float
) -> tuple[float, float, float, float]:
    left, top, right, bottom = bounds
    return (
        max(0.0, left - margin),
        max(0.0, top - margin),
        min(1.0, right + margin),
        min(1.0, bottom + margin),
    )


def _inside(point: tuple[float, float], bounds: tuple[float, float, float, float]) -> bool:
    x, y = point
    left, top, right, bottom = bounds
    return left <= x <= right and top <= y <= bottom
