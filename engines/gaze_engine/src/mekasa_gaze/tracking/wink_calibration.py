"""Wink step of the calibration: learn each eye's closure and check whether winks are possible.

After the gaze grid, the person is asked to close only the right eye three times, then only the
left eye. From the eye aperture (EAR) of those frames and of the grid frames, a closure
threshold is fitted per eye:

    threshold = midpoint between the lowest normal aperture of that eye during the grid (which
                includes looking down, when the lid drops) and its aperture while winking.

A wink with an eye is *enabled* only if, with those thresholds, at least 2 of the 3 requested
winks are recognized as "this eye closed while the other stayed open". Many people cannot close
one eye without narrowing the other; then winks stay disabled for that eye and the board keeps
using two-eye blinks and the 3 s look. Thresholds and rules are engineering hypotheses.

Nothing here is persisted except the fitted thresholds and counts (no frames).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mekasa_gaze.core.domain import EyeState

EYES = ("right", "left")  # order asked during calibration (anatomical sides)
REQUESTED_WINKS = 3
_MIN_DETECTED = 2
_SEPARATION = 0.7  # wink aperture must be below 70 % of the lowest normal aperture
_MIN_CLOSED = 0.08
_MAX_CLOSED = 0.8
_FALLBACK_FRACTION = 0.6


def _ear(features: Mapping[str, float], eye: str) -> float | None:
    value = features.get(f"{eye}_eye_aspect_ratio")
    return float(value) if value is not None and np.isfinite(value) else None


@dataclass(frozen=True, slots=True)
class WinkProfile:
    """Per-eye closure thresholds and which eyes can wink."""

    left_threshold: float
    right_threshold: float
    left_wink: bool
    right_wink: bool
    left_detected: int = 0
    right_detected: int = 0

    @property
    def wink_eyes(self) -> tuple[str, ...]:
        return tuple(eye for eye in ("left", "right") if getattr(self, f"{eye}_wink"))

    def threshold(self, eye: str) -> float:
        return float(getattr(self, f"{eye}_threshold"))

    def eye_states(
        self, features: Mapping[str, float], left: EyeState, right: EyeState
    ) -> tuple[EyeState, EyeState]:
        """Re-classify open/closed per eye with the personal thresholds.

        Eyes the quality rules marked INVALID stay INVALID (unusable geometry).
        """
        result = []
        for eye, state in (("left", left), ("right", right)):
            ear = _ear(features, eye)
            if state is EyeState.INVALID and ear is None:
                result.append(EyeState.INVALID)
            elif ear is None:
                result.append(state)
            else:
                result.append(EyeState.CLOSED if ear < self.threshold(eye) else EyeState.OPEN)
        return result[0], result[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mekasa.wink_profile.v1",
            "left_threshold": self.left_threshold,
            "right_threshold": self.right_threshold,
            "left_wink": self.left_wink,
            "right_wink": self.right_wink,
            "left_detected": self.left_detected,
            "right_detected": self.right_detected,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WinkProfile:
        if data.get("schema") != "mekasa.wink_profile.v1":
            raise ValueError("not a Mekasa wink profile")
        return cls(
            float(data["left_threshold"]),
            float(data["right_threshold"]),
            bool(data["left_wink"]),
            bool(data["right_wink"]),
            int(data.get("left_detected", 0)),
            int(data.get("right_detected", 0)),
        )


def save_wink_profile(profile: WinkProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_wink_profile(path: Path) -> WinkProfile:
    return WinkProfile.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))


# ----------------------------------------------------------------------------- live step


@dataclass(frozen=True, slots=True)
class WinkSnapshot:
    """What to draw during the wink step."""

    eye: str | None  # eye being asked ("right"/"left"), None when done
    index: int
    total: int
    detected: int  # winks recognized live for the current eye (provisional thresholds)
    seconds_left: float
    done: bool


@dataclass(slots=True)
class WinkCalibration:
    """Clock-injected step: ``seconds`` per eye, or until 3 winks + a short pause.

    ``baseline`` is each eye's median aperture during the grid; it gives provisional thresholds
    for live feedback only. The fitted profile comes from :func:`fit_wink_profile`.
    """

    baseline: Mapping[str, float]
    seconds: float = 7.0
    records: list[tuple[str, float, float, float]] = field(default_factory=list)
    _index: int = 0
    _started_at: float | None = None
    _detected: int = 0
    _closed_since: float | None = None
    _done_at: float | None = None

    def update(self, now: float, features: Mapping[str, float] | None) -> WinkSnapshot:
        if self._index >= len(EYES):
            return WinkSnapshot(None, len(EYES), len(EYES), 0, 0.0, True)
        if self._started_at is None:
            self._started_at = now
        eye = EYES[self._index]
        other = "left" if eye == "right" else "right"
        if features is not None:
            ear, other_ear = _ear(features, eye), _ear(features, other)
            if ear is not None and other_ear is not None:
                left = ear if eye == "left" else other_ear
                right = ear if eye == "right" else other_ear
                self.records.append((eye, now, left, right))
                closed = ear < _FALLBACK_FRACTION * self.baseline[eye]
                other_open = other_ear >= _FALLBACK_FRACTION * self.baseline[other]
                if closed and other_open and self._closed_since is None:
                    self._closed_since = now
                elif not (closed and other_open) and self._closed_since is not None:
                    if _MIN_CLOSED <= now - self._closed_since <= _MAX_CLOSED:
                        self._detected += 1
                        if self._detected >= REQUESTED_WINKS and self._done_at is None:
                            self._done_at = now + 0.8
                    self._closed_since = None
        elapsed = now - self._started_at
        finished = elapsed >= self.seconds or (self._done_at is not None and now >= self._done_at)
        if finished:
            self._index += 1
            self._started_at, self._detected, self._closed_since, self._done_at = (
                now,
                0,
                None,
                None,
            )
            if self._index >= len(EYES):
                return WinkSnapshot(None, len(EYES), len(EYES), 0, 0.0, True)
            eye, elapsed = EYES[self._index], 0.0
        return WinkSnapshot(
            eye, self._index, len(EYES), self._detected, max(0.0, self.seconds - elapsed), False
        )


# ----------------------------------------------------------------------------- fitting


def _count_winks(
    records: Sequence[tuple[float, float, float]], eye: str, thresholds: Mapping[str, float]
) -> int:
    other = "left" if eye == "right" else "right"
    count, since = 0, None
    for now, left, right in records:
        values = {"left": left, "right": right}
        winking = values[eye] < thresholds[eye] and values[other] >= thresholds[other]
        if winking and since is None:
            since = now
        elif not winking and since is not None:
            if _MIN_CLOSED <= now - since <= _MAX_CLOSED:
                count += 1
            since = None
    return count


def fit_wink_profile(
    records: Sequence[tuple[str, float, float, float]],
    grid_ears: Mapping[str, Sequence[float]],
) -> WinkProfile:
    """Fit per-eye thresholds from the wink step and the grid apertures."""
    thresholds: dict[str, float] = {}
    separated: dict[str, bool] = {}
    for eye in ("left", "right"):
        grid = np.asarray([value for value in grid_ears.get(eye, []) if np.isfinite(value)])
        open_low = float(np.percentile(grid, 5)) if grid.size else 0.2
        fallback = _FALLBACK_FRACTION * (float(np.median(grid)) if grid.size else 0.25)
        asked = np.asarray(
            [
                (left if eye == "left" else right)
                for step_eye, _, left, right in records
                if step_eye == eye
            ]
        )
        closed_level = float(np.percentile(asked, 10)) if asked.size else open_low
        separated[eye] = closed_level < _SEPARATION * open_low
        thresholds[eye] = (open_low + closed_level) / 2.0 if separated[eye] else fallback
    detected = {
        eye: _count_winks(
            [(now, left, right) for step_eye, now, left, right in records if step_eye == eye],
            eye,
            thresholds,
        )
        if separated[eye]
        else 0
        for eye in ("left", "right")
    }
    return WinkProfile(
        left_threshold=thresholds["left"],
        right_threshold=thresholds["right"],
        left_wink=detected["left"] >= _MIN_DETECTED,
        right_wink=detected["right"] >= _MIN_DETECTED,
        left_detected=detected["left"],
        right_detected=detected["right"],
    )


__all__ = [
    "EYES",
    "REQUESTED_WINKS",
    "WinkCalibration",
    "WinkProfile",
    "WinkSnapshot",
    "fit_wink_profile",
    "load_wink_profile",
    "save_wink_profile",
]
