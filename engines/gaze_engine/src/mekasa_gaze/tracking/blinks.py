"""Deliberate blink sequences as commands (by default 3 = select, 4 = start/stop reading,
5 = say the message, 7 = urgent call; see ``config/communication.toml``).

A *blink* is a closure of the required eyes (both in BOTH mode, the active eye in LEFT/RIGHT
mode) lasting between ``min_closed_seconds`` and ``max_closed_seconds``. Blinks separated by
open intervals of at most ``max_gap_seconds`` form a *sequence*. A sequence is reported only
once it is over (eyes open for longer than ``max_gap_seconds``), so 3 can be told apart from
4 or 5. Anything else is ignored on purpose:

* a single blink (spontaneous blinking, ~15-20 per minute at rest);
* any count that is not a configured command;
* a long closure (resting the eyes), which also cancels a sequence in progress;
* losing the face, which cancels a sequence in progress.

All durations are engineering hypotheses, configurable in ``config/communication.toml``;
spontaneous double blinks do occur, which is why a selection additionally needs a stable gaze
on one option right before the blinks (see ``CommunicationSession.select_by_blink``).
"""

from __future__ import annotations

from dataclasses import dataclass

from mekasa_gaze.core.domain import EyeState
from mekasa_gaze.core.eye_mode import EyeMode

DEFAULT_COMMANDS = (3, 4, 5, 7)


@dataclass(frozen=True, slots=True)
class BlinkGesture:
    """A completed sequence of ``count`` deliberate blinks."""

    count: int
    started_at: float
    ended_at: float


@dataclass(slots=True)
class BlinkDetector:
    """Clock-injected detector fed with per-frame eye states."""

    eye_mode: EyeMode = EyeMode.BOTH
    min_closed_seconds: float = 0.08
    max_closed_seconds: float = 0.8
    max_gap_seconds: float = 0.8
    commands: tuple[int, ...] = DEFAULT_COMMANDS
    _closed_since: float | None = None
    _count: int = 0
    _sequence_start: float | None = None
    _last_blink_end: float | None = None
    _long_closure: bool = False
    _ignored_long_closures: int = 0
    _ignored_sequences: int = 0

    @property
    def count_in_progress(self) -> int:
        """Blinks counted in the sequence that is still open (for on-screen feedback)."""
        return self._count

    @property
    def eyes_closed(self) -> bool:
        return self._closed_since is not None

    @property
    def stats(self) -> dict[str, int]:
        return {
            "ignored_long_closures": self._ignored_long_closures,
            "ignored_sequences": self._ignored_sequences,
        }

    def reset(self) -> None:
        self._closed_since = None
        self._count = 0
        self._sequence_start = None
        self._last_blink_end = None

    def update(
        self,
        now: float,
        left: EyeState | None,
        right: EyeState | None,
        *,
        tracking_ok: bool = True,
    ) -> BlinkGesture | None:
        """Feed one frame; returns a gesture when a sequence has just finished."""
        if not tracking_ok or left is None or right is None:
            self.reset()
            return None
        closed = self._closed(left, right)
        if closed is None:  # eye geometry unusable (not open, not clearly closed)
            return None
        if closed:
            if self._closed_since is None:
                self._closed_since = now
                self._long_closure = False
            elif not self._long_closure and now - self._closed_since > self.max_closed_seconds:
                # Resting the eyes: not a command, and it cancels a sequence in progress.
                self._long_closure = True
                self._ignored_long_closures += 1
                if self._count:
                    self._ignored_sequences += 1
                self._count, self._sequence_start, self._last_blink_end = 0, None, None
            return None
        if self._closed_since is not None:
            duration = now - self._closed_since
            started = self._closed_since
            self._closed_since = None
            if self.min_closed_seconds <= duration <= self.max_closed_seconds:
                if self._count == 0:
                    self._sequence_start = started
                self._count += 1
                self._last_blink_end = now
            return None
        if not self._count or self._last_blink_end is None:
            return None
        if now - self._last_blink_end <= self.max_gap_seconds:
            return None
        count, started_at = self._count, self._sequence_start or now
        self._count, self._sequence_start, self._last_blink_end = 0, None, None
        if count not in self.commands:
            self._ignored_sequences += 1
            return None
        return BlinkGesture(count, started_at, now)

    def _closed(self, left: EyeState, right: EyeState) -> bool | None:
        states = {EyeMode.BOTH: (left, right), EyeMode.LEFT: (left,), EyeMode.RIGHT: (right,)}[
            self.eye_mode
        ]
        if all(state is EyeState.CLOSED for state in states):
            return True
        if all(state is EyeState.OPEN for state in states):
            return False
        return None


__all__ = ["DEFAULT_COMMANDS", "BlinkDetector", "BlinkGesture"]
