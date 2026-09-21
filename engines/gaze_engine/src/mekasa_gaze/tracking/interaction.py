from __future__ import annotations

from dataclasses import dataclass

from mekasa_gaze.core.domain import RegionName

_BOUNDS: dict[RegionName, tuple[float, float, float, float]] = {
    RegionName.TOP_LEFT: (0.0, 0.0, 0.5, 0.5),
    RegionName.TOP_RIGHT: (0.5, 0.0, 1.0, 0.5),
    RegionName.BOTTOM_LEFT: (0.0, 0.5, 0.5, 1.0),
    RegionName.BOTTOM_RIGHT: (0.5, 0.5, 1.0, 1.0),
}


@dataclass(frozen=True, slots=True)
class SelectionUpdate:
    """State transition outcome of a dwell selector update tick.

    Responsibility:
        Encapsulates active candidate region, normalized dwell progress ratio [0.0, 1.0],
        confirmed selection region, single-frame selection trigger flag, post-selection
        cooldown countdown in seconds, and tracking availability status.
    """

    candidate: RegionName | None
    dwell_progress: float
    selected: RegionName | None
    selection_fired: bool
    cooldown_remaining_seconds: float
    tracking_available: bool


class DwellSelector:
    """Four-region dwell selector with hysteresis, cooldown, and tracking loss reset.

    Responsibility:
        Addresses the Midas Touch problem by requiring sustained gaze fixation within a region
        for `dwell_seconds`, enforcing spatial hysteresis margins to prevent jitter at boundaries,
        applying refractory cooldown periods post-selection, and resetting timers upon tracking
        loss.
    """

    def __init__(
        self,
        *,
        dwell_seconds: float,
        cooldown_seconds: float,
        hysteresis_margin: float,
    ) -> None:
        """Initialize the dwell selector engine.

        Args:
            dwell_seconds: Duration in seconds gaze must remain in a region to trigger selection.
            cooldown_seconds: Refractory cooldown duration in seconds after a confirmed selection.
            hysteresis_margin: Normalized boundary expansion ratio to prevent edge jitter [0, 0.25).

        Raises:
            ValueError: If dwell_seconds <= 0, cooldown_seconds < 0, or hysteresis is out of range.
        """
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
        """Advance the dwell selection state machine given instantaneous gaze prediction.

        Args:
            prediction: Normalized gaze coordinates (x, y), or None if tracking is unavailable.
            now_seconds: Monotonic timestamp in seconds (from `time.perf_counter()`).

        Returns:
            SelectionUpdate: Current candidate, dwell progress [0.0, 1.0], and confirmed selection.

        Side Effects:
            Mutates candidate tracking state, timer timestamps, and refractory cooldown clocks.
        """
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
        """Determine whether point falls in candidate region (with hysteresis) or a new region."""
        if self._candidate is not None and _inside(
            point,
            _expanded(_BOUNDS[self._candidate], self._hysteresis_margin),
        ):
            return self._candidate
        return region_at(point)

    def _reset_candidate(self) -> None:
        """Clear the current dwell candidate and reset acquisition timer."""
        self._candidate = None
        self._candidate_since = None


def region_at(point: tuple[float, float]) -> RegionName | None:
    """Determine which of the four quadrants contains the given normalized coordinate.

    Responsibility:
        Maps (normalized_x, normalized_y) in [0.0, 1.0] x [0.0, 1.0] to a canonical RegionName.

    Args:
        point: Tuple of (x, y) normalized coordinates.

    Returns:
        RegionName | None: Identified quadrant enum, or None if outside screen bounds.
    """
    x, y = point
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        return None
    if y < 0.5:
        return RegionName.TOP_LEFT if x < 0.5 else RegionName.TOP_RIGHT
    return RegionName.BOTTOM_LEFT if x < 0.5 else RegionName.BOTTOM_RIGHT


def region_bounds(region: RegionName) -> tuple[float, float, float, float]:
    """Return the normalized bounding box coordinates for a screen region.

    Args:
        region: Target screen region enum.

    Returns:
        tuple[float, float, float, float]: (left, top, right, bottom) normalized coordinates.
    """
    return _BOUNDS[region]


def _expanded(
    bounds: tuple[float, float, float, float], margin: float
) -> tuple[float, float, float, float]:
    """Expand bounding box outward by margin while clamping to [0.0, 1.0]."""
    left, top, right, bottom = bounds
    return (
        max(0.0, left - margin),
        max(0.0, top - margin),
        min(1.0, right + margin),
        min(1.0, bottom + margin),
    )


def _inside(point: tuple[float, float], bounds: tuple[float, float, float, float]) -> bool:
    """Check whether a point is contained within the bounding box."""
    x, y = point
    left, top, right, bottom = bounds
    return left <= x <= right and top <= y <= bottom


__all__ = [
    "DwellSelector",
    "RegionName",
    "SelectionUpdate",
    "region_at",
    "region_bounds",
]
