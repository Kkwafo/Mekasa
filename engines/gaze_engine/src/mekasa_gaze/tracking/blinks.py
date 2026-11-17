"""Deliberate blink and wink sequences as commands.

Both eyes (defaults in ``config/communication.toml``): 3 = select, 4 = start/stop reading,
5 = say the message, 7 = urgent call. One eye (*wink*, only after the calibration confirmed the
person can wink with that eye): 3 = select.

A *closure* lasts between ``min_closed_seconds`` and ``max_closed_seconds``. Its kind is the
majority pattern of its frames: both eyes closed, only the left or only the right. Closures of
the same kind separated by open intervals of at most ``max_gap_seconds`` form a *sequence*,
reported only once it is over, so 3 can be told apart from 4 or 5. A closure of a different
kind starts a new sequence. Ignored on purpose:

* a single blink (spontaneous blinking, ~15-20 per minute at rest, is bilateral; a wink almost
  never happens by accident, which is why winks are a good selection gesture);
* any count that is not a configured command for that kind;
* a long closure (resting the eyes), which also cancels a sequence in progress;
* losing the face, which cancels a sequence in progress.

All durations are engineering hypotheses. A selection additionally needs a stable gaze on one
option right before the gesture (see ``CommunicationSession.select_by_blink``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from mekasa_gaze.core.domain import EyeState
from mekasa_gaze.core.eye_mode import EyeMode

DEFAULT_COMMANDS = (3, 4, 5, 7)
DEFAULT_WINK_COMMANDS = (3,)
BOTH = "both"
LEFT = "left"
RIGHT = "right"
_OPEN = "open"


@dataclass(frozen=True, slots=True)
class BlinkGesture:
    """A completed sequence of ``count`` deliberate closures of ``eyes`` (both/left/right)."""

    count: int
    started_at: float
    ended_at: float
    eyes: str = BOTH


@dataclass(slots=True)
class BlinkDetector:
    """Clock-injected detector fed with per-frame eye states.

    ``winks`` lists the eyes ("left", "right") whose winks are recognized; in single-eye modes
    winks never apply (the other eye is not usable).
    """

    eye_mode: EyeMode = EyeMode.BOTH
    min_closed_seconds: float = 0.08
    max_closed_seconds: float = 0.8
    max_gap_seconds: float = 0.8
    commands: tuple[int, ...] = DEFAULT_COMMANDS
    winks: tuple[str, ...] = ()
    wink_commands: tuple[int, ...] = DEFAULT_WINK_COMMANDS
    _closed_since: float | None = None
    _votes: Counter[str] = field(default_factory=Counter)
    _kind: str | None = None
    _count: int = 0
    _sequence_start: float | None = None
    _last_blink_end: float | None = None
    _long_closure: bool = False
    _ignored_long_closures: int = 0
    _ignored_sequences: int = 0

    @property
    def count_in_progress(self) -> int:
        """Closures counted in the sequence that is still open (for on-screen feedback)."""
        return self._count

    @property
    def kind_in_progress(self) -> str | None:
        """Kind of the sequence in progress: both, left or right."""
        return self._kind if self._count else None

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
        self._votes.clear()
        self._clear_sequence()

    def _clear_sequence(self) -> None:
        self._kind, self._count, self._sequence_start, self._last_blink_end = None, 0, None, None

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
        pattern = self._pattern(left, right)
        if pattern is None:  # eye geometry unusable (not open, not clearly closed)
            return None
        if pattern != _OPEN:
            if self._closed_since is None:
                self._closed_since = now
                self._long_closure = False
                self._votes.clear()
            self._votes[pattern] += 1
            if not self._long_closure and now - self._closed_since > self.max_closed_seconds:
                # Resting the eyes: not a command, and it cancels a sequence in progress.
                self._long_closure = True
                self._ignored_long_closures += 1
                if self._count:
                    self._ignored_sequences += 1
                self._clear_sequence()
            return None
        if self._closed_since is not None:
            duration = now - self._closed_since
            started = self._closed_since
            kind = self._votes.most_common(1)[0][0] if self._votes else BOTH
            self._closed_since = None
            self._votes.clear()
            if self.min_closed_seconds <= duration <= self.max_closed_seconds:
                if self._count and kind != self._kind:  # another kind: a new sequence
                    self._ignored_sequences += 1
                    self._clear_sequence()
                if self._count == 0:
                    self._sequence_start, self._kind = started, kind
                self._count += 1
                self._last_blink_end = now
            return None
        if not self._count or self._last_blink_end is None:
            return None
        if now - self._last_blink_end <= self.max_gap_seconds:
            return None
        count, kind, started_at = self._count, self._kind or BOTH, self._sequence_start or now
        self._clear_sequence()
        allowed = self.commands if kind == BOTH else self.wink_commands
        if count not in allowed:
            self._ignored_sequences += 1
            return None
        return BlinkGesture(count, started_at, now, kind)

    def _pattern(self, left: EyeState, right: EyeState) -> str | None:
        if self.eye_mode is not EyeMode.BOTH:
            state = left if self.eye_mode is EyeMode.LEFT else right
            if state is EyeState.CLOSED:
                return BOTH
            return _OPEN if state is EyeState.OPEN else None
        if left is EyeState.CLOSED and right is EyeState.CLOSED:
            return BOTH
        if left is EyeState.OPEN and right is EyeState.OPEN:
            return _OPEN
        if LEFT in self.winks and left is EyeState.CLOSED and right is EyeState.OPEN:
            return LEFT
        if RIGHT in self.winks and right is EyeState.CLOSED and left is EyeState.OPEN:
            return RIGHT
        return None


__all__ = [
    "BOTH",
    "DEFAULT_COMMANDS",
    "DEFAULT_WINK_COMMANDS",
    "LEFT",
    "RIGHT",
    "BlinkDetector",
    "BlinkGesture",
]
