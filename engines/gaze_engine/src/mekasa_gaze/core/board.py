"""Communication board model and configuration.

A board is a set of large rectangular options in normalized screen coordinates. Each option
has a short visual label and the phrase that is spoken locally when it is confirmed. Layouts
are data, so options can be resized, relocated or reduced to two for binary access.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mekasa_gaze.core.eye_mode import EyeMode

Bounds = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class BoardOption:
    """One selectable communication option.

    Attributes:
        key: Stable identifier used in reports and keyboard shortcuts.
        label: Short text shown on the card.
        phrase: Phrase spoken (and shown) after confirmation.
        bounds: Normalized (left, top, right, bottom) rectangle of the card.
    """

    key: str
    label: str
    phrase: str
    bounds: Bounds

    def contains(self, point: tuple[float, float], margin: float = 0.0) -> bool:
        """Return whether ``point`` lies inside the card, optionally expanded by ``margin``."""
        x, y = point
        left, top, right, bottom = self.bounds
        return left - margin <= x <= right + margin and top - margin <= y <= bottom + margin


@dataclass(frozen=True, slots=True)
class BoardLayout:
    """An ordered set of non-overlapping options."""

    name: str
    options: tuple[BoardOption, ...]

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("A board needs at least one option")
        keys = [option.key for option in self.options]
        if len(set(keys)) != len(keys):
            raise ValueError("Board option keys must be unique")
        for option in self.options:
            left, top, right, bottom = option.bounds
            if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
                raise ValueError(f"Option {option.key} bounds must be inside the unit square")
        for index, first in enumerate(self.options):
            for second in self.options[index + 1 :]:
                if _overlap(first.bounds, second.bounds):
                    raise ValueError(f"Options {first.key} and {second.key} overlap")

    def option_at(self, point: tuple[float, float]) -> BoardOption | None:
        """Return the option strictly containing ``point`` (no hysteresis)."""
        for option in self.options:
            if option.contains(point):
                return option
        return None

    def option(self, key: str) -> BoardOption:
        """Return an option by key."""
        for option in self.options:
            if option.key == key:
                return option
        raise KeyError(key)


def _overlap(first: Bounds, second: Bounds) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


# Gap between cards so a boundary prediction is never ambiguous between two options.
_GAP = 0.04

DEFAULT_PHRASES: dict[str, tuple[str, str]] = {
    "YES": ("SÍ", "Sí"),
    "NO": ("NO", "No"),
    "WATER": ("AGUA", "Quiero agua"),
    "HELP": ("AYUDA", "Necesito ayuda"),
}


def quadrant_board(
    phrases: dict[str, tuple[str, str]] | None = None, gap: float = _GAP
) -> BoardLayout:
    """Default 2x2 board: SÍ / NO on top, AGUA / AYUDA below."""
    labels = phrases or DEFAULT_PHRASES
    half = gap / 2.0
    cells = {
        "YES": (0.0, 0.0, 0.5 - half, 0.5 - half),
        "NO": (0.5 + half, 0.0, 1.0, 0.5 - half),
        "WATER": (0.0, 0.5 + half, 0.5 - half, 1.0),
        "HELP": (0.5 + half, 0.5 + half, 1.0, 1.0),
    }
    return BoardLayout(
        "quadrant",
        tuple(
            BoardOption(key, labels[key][0], labels[key][1], bounds)
            for key, bounds in cells.items()
        ),
    )


@dataclass(frozen=True, slots=True)
class CommunicationConfig:
    """Timing, safety and presentation settings for the communication board.

    All timing values are engineering hypotheses to be compared in guided tests
    (see ``config/communication.toml``); none is a clinical threshold.
    """

    dwell_seconds: float = 0.9
    cooldown_seconds: float = 1.0
    hysteresis_margin: float = 0.03
    recovery_seconds: float = 0.5
    max_frame_age_seconds: float = 0.25
    eye_dropout_grace_seconds: float = 0.2
    confirmation_required: bool = False
    confirmation_window_seconds: float = 4.0
    eye_mode: EyeMode = EyeMode.BOTH
    high_contrast: bool = True
    phrases: tuple[tuple[str, str, str], ...] = ()
    # Precision features (see config/communication.toml for the evidence behind each).
    calibrate_at_start: bool = True
    auto_realign: bool = True
    implicit_recalibration: bool = True
    distance_compensation: bool = True
    camera_position: tuple[float, float] = (0.5, -0.02)
    # Conversation board (menus to compose full messages) or the simple 4-phrase board.
    interface: str = "conversation"
    vocabulary: str = "config/vocabulario.txt"
    gender: str = "m"
    urgent_alarm: bool = True
    # How a looked-at option is chosen: "dwell" (look for dwell_seconds), "blink" (look
    # until it is ready, then blink ``blink_select_count`` times) or "combined" (either:
    # look for combined_dwell_seconds or blink).
    selection_mode: str = "dwell"
    combined_dwell_seconds: float = 3.0
    blink_toggle: bool = True
    blink_select_count: int = 3
    blink_toggle_count: int = 4
    blink_speak_count: int = 5
    blink_urgent_count: int = 7
    blink_min_focus_seconds: float = 0.4
    blink_selection_window_seconds: float = 3.0
    blink_min_closed_seconds: float = 0.08
    blink_max_closed_seconds: float = 0.8
    blink_max_gap_seconds: float = 0.8

    def validate(self) -> None:
        """Raise ``ValueError`` for unsafe or inconsistent settings."""
        if not 0.2 <= self.dwell_seconds <= 5.0:
            raise ValueError("dwell_seconds must be within [0.2, 5.0]")
        if self.cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds must be non-negative")
        if not 0.0 <= self.hysteresis_margin < 0.1:
            raise ValueError("hysteresis_margin must be within [0, 0.1)")
        if self.recovery_seconds < 0.0:
            raise ValueError("recovery_seconds must be non-negative")
        if not 0.0 < self.max_frame_age_seconds <= 2.0:
            raise ValueError("max_frame_age_seconds must be within (0, 2]")
        if not 0.0 <= self.eye_dropout_grace_seconds <= 0.5:
            raise ValueError("eye_dropout_grace_seconds must be within [0, 0.5]")
        if self.confirmation_window_seconds <= 0.0:
            raise ValueError("confirmation_window_seconds must be positive")
        if not (0.0 <= self.camera_position[0] <= 1.0 and -0.5 <= self.camera_position[1] <= 1.5):
            raise ValueError("camera_position must be a screen-normalized point near the screen")
        if self.interface not in ("conversation", "simple"):
            raise ValueError("interface must be 'conversation' or 'simple'")
        if self.gender not in ("m", "f"):
            raise ValueError("gender must be 'm' or 'f'")
        if self.selection_mode not in ("dwell", "blink", "combined"):
            raise ValueError("selection_mode must be 'dwell', 'blink' or 'combined'")
        if not 0.5 <= self.combined_dwell_seconds <= 10.0:
            raise ValueError("combined_dwell_seconds must be within [0.5, 10]")
        counts = self.blink_counts
        if len(set(counts.values())) != len(counts) or not all(
            2 <= count <= 9 for count in counts.values()
        ):
            raise ValueError("blink gesture counts must be distinct and within [2, 9]")
        if not 0.1 <= self.blink_min_focus_seconds <= 3.0:
            raise ValueError("blink_min_focus_seconds must be within [0.1, 3]")
        if not 0.5 <= self.blink_selection_window_seconds <= 10.0:
            raise ValueError("blink_selection_window_seconds must be within [0.5, 10]")
        if not 0.0 < self.blink_min_closed_seconds < self.blink_max_closed_seconds <= 2.0:
            raise ValueError("blink closure limits must satisfy 0 < min < max <= 2 s")
        if not 0.2 <= self.blink_max_gap_seconds <= 2.0:
            raise ValueError("blink_max_gap_seconds must be within [0.2, 2]")

    @property
    def effective_dwell_seconds(self) -> float:
        """Dwell time that confirms a selection in the active mode."""
        if self.selection_mode == "combined":
            return self.combined_dwell_seconds
        return self.dwell_seconds

    @property
    def blink_counts(self) -> dict[str, int]:
        """Blink gesture -> number of blinks in a row."""
        return {
            "select": self.blink_select_count,
            "toggle": self.blink_toggle_count,
            "speak": self.blink_speak_count,
            "urgent": self.blink_urgent_count,
        }

    def board(self) -> BoardLayout:
        """Build the configured board layout."""
        phrases = dict(DEFAULT_PHRASES)
        for key, label, phrase in self.phrases:
            phrases[key] = (label, phrase)
        return quadrant_board(phrases)

    @classmethod
    def load(cls, path: Path) -> CommunicationConfig:
        """Load settings from the ``[communication]`` and ``[phrases.*]`` TOML sections."""
        with path.open("rb") as handle:
            raw: dict[str, Any] = tomllib.load(handle)
        section = raw.get("communication", {})
        phrases = tuple(
            (str(key), str(value["label"]), str(value["phrase"]))
            for key, value in raw.get("phrases", {}).items()
        )
        defaults = cls()
        config = cls(
            dwell_seconds=float(section.get("dwell_seconds", defaults.dwell_seconds)),
            cooldown_seconds=float(section.get("cooldown_seconds", defaults.cooldown_seconds)),
            hysteresis_margin=float(section.get("hysteresis_margin", defaults.hysteresis_margin)),
            recovery_seconds=float(section.get("recovery_seconds", defaults.recovery_seconds)),
            max_frame_age_seconds=float(
                section.get("max_frame_age_seconds", defaults.max_frame_age_seconds)
            ),
            eye_dropout_grace_seconds=float(
                section.get("eye_dropout_grace_seconds", defaults.eye_dropout_grace_seconds)
            ),
            confirmation_required=bool(
                section.get("confirmation_required", defaults.confirmation_required)
            ),
            confirmation_window_seconds=float(
                section.get("confirmation_window_seconds", defaults.confirmation_window_seconds)
            ),
            eye_mode=EyeMode(str(section.get("eye_mode", defaults.eye_mode.value)).upper()),
            high_contrast=bool(section.get("high_contrast", defaults.high_contrast)),
            phrases=phrases,
            calibrate_at_start=bool(section.get("calibrate_at_start", defaults.calibrate_at_start)),
            auto_realign=bool(section.get("auto_realign", defaults.auto_realign)),
            implicit_recalibration=bool(
                section.get("implicit_recalibration", defaults.implicit_recalibration)
            ),
            distance_compensation=bool(
                section.get("distance_compensation", defaults.distance_compensation)
            ),
            camera_position=_point(section.get("camera_position", defaults.camera_position)),
            interface=str(section.get("interface", defaults.interface)),
            vocabulary=str(section.get("vocabulary", defaults.vocabulary)),
            gender=str(section.get("gender", defaults.gender)).lower(),
            urgent_alarm=bool(section.get("urgent_alarm", defaults.urgent_alarm)),
            selection_mode=str(section.get("selection_mode", defaults.selection_mode)).lower(),
            blink_toggle=bool(section.get("blink_toggle", defaults.blink_toggle)),
            combined_dwell_seconds=float(
                section.get("combined_dwell_seconds", defaults.combined_dwell_seconds)
            ),
            blink_select_count=int(section.get("blink_select_count", defaults.blink_select_count)),
            blink_toggle_count=int(section.get("blink_toggle_count", defaults.blink_toggle_count)),
            blink_speak_count=int(section.get("blink_speak_count", defaults.blink_speak_count)),
            blink_urgent_count=int(section.get("blink_urgent_count", defaults.blink_urgent_count)),
            blink_min_focus_seconds=float(
                section.get("blink_min_focus_seconds", defaults.blink_min_focus_seconds)
            ),
            blink_selection_window_seconds=float(
                section.get(
                    "blink_selection_window_seconds", defaults.blink_selection_window_seconds
                )
            ),
            blink_min_closed_seconds=float(
                section.get("blink_min_closed_seconds", defaults.blink_min_closed_seconds)
            ),
            blink_max_closed_seconds=float(
                section.get("blink_max_closed_seconds", defaults.blink_max_closed_seconds)
            ),
            blink_max_gap_seconds=float(
                section.get("blink_max_gap_seconds", defaults.blink_max_gap_seconds)
            ),
        )
        config.validate()
        return config


def _point(value: Any) -> tuple[float, float]:
    x, y = value
    return float(x), float(y)


__all__ = [
    "DEFAULT_PHRASES",
    "BoardLayout",
    "BoardOption",
    "CommunicationConfig",
    "quadrant_board",
]
