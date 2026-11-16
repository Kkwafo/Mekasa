"""Communication session: safe dwell selection over a semantic board.

This state machine is the only component allowed to turn gaze into a confirmed message.
It is deterministic and clock-injected so every safety rule is testable without a camera:

* no calibration valid for the active eye mode -> no gaze selection;
* pause, eye-mode change, missing frame, stale frame, invalid tracking or unusable eye
  -> no selection, and focus progress *and* any pending confirmation are discarded;
* after any signal loss a stable signal is required for ``recovery_seconds`` before re-arming;
* after a confirmation the same option is not re-armed until gaze leaves it (plus cooldown);
* focus, (optional) confirmation and emission are separate, observable states.

Direct access (keyboard, pointer or switch) bypasses gaze gates by design: it is a deliberate
action and the fallback when gaze is not usable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from mekasa_gaze.core.board import BoardLayout, BoardOption, CommunicationConfig
from mekasa_gaze.core.domain import EyeState
from mekasa_gaze.core.eye_mode import EyeMode, eye_signal_usable


class SessionState(StrEnum):
    """Observable session state, rendered to the person and their support."""

    NO_CALIBRATION = "NO_CALIBRATION"
    PAUSED = "PAUSED"
    SIGNAL_LOST = "SIGNAL_LOST"
    RECOVERING = "RECOVERING"
    READY = "READY"
    FOCUSING = "FOCUSING"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    COOLDOWN = "COOLDOWN"
    AWAITING_EXIT = "AWAITING_EXIT"


class AbstainReason(StrEnum):
    """Why a tick could not drive selection."""

    NO_FRAME = "NO_FRAME"
    STALE_FRAME = "STALE_FRAME"
    TRACKING = "TRACKING"
    EYE_SIGNAL = "EYE_SIGNAL"
    NO_PREDICTION = "NO_PREDICTION"


@dataclass(frozen=True, slots=True)
class GazeInput:
    """One pipeline tick, already reduced to what the session needs.

    Attributes:
        now_seconds: Monotonic time of this tick (same clock as the frame timestamp).
        frame_timestamp_seconds: Capture time of the frame used, or ``None`` if no frame
            arrived (camera disconnected, busy or failed read).
        prediction: Smoothed normalized gaze point, or ``None``.
        tracking_ok: Exactly one face tracked and placement within the accepted band.
        left_eye: Anatomical left eye state.
        right_eye: Anatomical right eye state.
    """

    now_seconds: float
    frame_timestamp_seconds: float | None
    prediction: tuple[float, float] | None
    tracking_ok: bool
    left_eye: EyeState
    right_eye: EyeState


@dataclass(frozen=True, slots=True)
class SessionUpdate:
    """Result of one tick. ``confirmed`` is set on exactly one tick per confirmation."""

    state: SessionState
    focused: BoardOption | None = None
    progress: float = 0.0
    pending: BoardOption | None = None
    confirmed: BoardOption | None = None
    confirmed_source: str | None = None
    message: str | None = None
    abstain_reason: AbstainReason | None = None
    recovery_progress: float = 0.0


@dataclass(slots=True)
class SessionStats:
    """Aggregate counters for reports. Contains no message history."""

    ticks: int = 0
    ticks_by_state: Counter[str] = field(default_factory=Counter)
    abstentions_by_reason: Counter[str] = field(default_factory=Counter)
    confirmations_by_source: Counter[str] = field(default_factory=Counter)
    confirmations_by_option: Counter[str] = field(default_factory=Counter)
    focus_cancellations: int = 0
    pending_discarded: int = 0
    signal_loss_events: int = 0

    def as_dict(self) -> dict[str, object]:
        """Serialize counters for JSON reports."""
        return {
            "ticks": self.ticks,
            "ticks_by_state": dict(sorted(self.ticks_by_state.items())),
            "abstentions_by_reason": dict(sorted(self.abstentions_by_reason.items())),
            "confirmations_by_source": dict(sorted(self.confirmations_by_source.items())),
            "confirmations_by_option": dict(sorted(self.confirmations_by_option.items())),
            "focus_cancellations": self.focus_cancellations,
            "pending_discarded": self.pending_discarded,
            "signal_loss_events": self.signal_loss_events,
        }


class CommunicationSession:
    """Deterministic, clock-injected dwell selection over a :class:`BoardLayout`."""

    def __init__(
        self,
        config: CommunicationConfig,
        *,
        layout: BoardLayout | None = None,
        calibration_mode: EyeMode | None = None,
    ) -> None:
        """Create a session.

        Args:
            config: Validated communication settings (dwell, safety, eye mode).
            layout: Board layout; defaults to ``config.board()``.
            calibration_mode: Eye mode of the loaded calibration, or ``None`` if no valid
                calibration is loaded. Gaze selection requires it to equal the active mode.
        """
        config.validate()
        self._config = config
        self._layout = layout or config.board()
        self._eye_mode = config.eye_mode
        self._calibration_mode = calibration_mode
        self._paused = False
        self._signal_since: float | None = None
        self._was_lost = True
        self._focused: BoardOption | None = None
        self._focus_since: float | None = None
        self._pending: BoardOption | None = None
        self._pending_until = 0.0
        self._cooldown_until = float("-inf")
        self._rearm: BoardOption | None = None
        self._message: str | None = None
        self.stats = SessionStats()

    # ------------------------------------------------------------------ properties
    @property
    def layout(self) -> BoardLayout:
        """Active board layout."""
        return self._layout

    @property
    def eye_mode(self) -> EyeMode:
        """Active ocular mode."""
        return self._eye_mode

    @property
    def paused(self) -> bool:
        """Whether gaze selection is paused."""
        return self._paused

    @property
    def gaze_enabled(self) -> bool:
        """Whether a calibration valid for the active eye mode is loaded."""
        return self._calibration_mode is not None and self._calibration_mode is self._eye_mode

    @property
    def message(self) -> str | None:
        """Last confirmed phrase. Stays visible even if speech fails."""
        return self._message

    # ------------------------------------------------------------------ controls
    def pause(self) -> None:
        """Pause gaze selection and discard any focus or pending confirmation."""
        self._paused = True
        self._flush()

    def resume(self) -> None:
        """Resume gaze selection; a fresh stable-signal recovery is required."""
        self._paused = False
        self._flush()

    def toggle_pause(self) -> None:
        """Toggle pause."""
        if self._paused:
            self.resume()
        else:
            self.pause()

    def set_eye_mode(self, mode: EyeMode) -> None:
        """Switch ocular mode. Flushes all progress; selection stays disabled until a
        calibration for the new mode is set with :meth:`set_calibration`."""
        if mode is not self._eye_mode:
            self._eye_mode = mode
            self._flush()

    def set_calibration(self, mode: EyeMode | None) -> None:
        """Declare which eye mode the active calibration supports (``None`` = none)."""
        self._calibration_mode = mode
        self._flush()

    def clear_message(self) -> None:
        """Clear the displayed message (explicit user/caregiver action only)."""
        self._message = None

    def select_direct(self, key: str, now_seconds: float, *, source: str) -> SessionUpdate:
        """Confirm an option through direct access (keyboard, pointer or switch).

        Gaze progress is discarded and gaze is held in cooldown so the same gesture does not
        also trigger a gaze confirmation.
        """
        option = self._layout.option(key)
        self._flush()
        return self._confirm(option, now_seconds, source)

    # ------------------------------------------------------------------ tick
    def update(self, tick: GazeInput) -> SessionUpdate:
        """Advance the session by one pipeline tick."""
        update = self._update(tick)
        self.stats.ticks += 1
        self.stats.ticks_by_state[update.state.value] += 1
        if update.abstain_reason is not None:
            self.stats.abstentions_by_reason[update.abstain_reason.value] += 1
        return update

    def _update(self, tick: GazeInput) -> SessionUpdate:
        now = tick.now_seconds
        if not self.gaze_enabled:
            self._flush()
            return self._snapshot(SessionState.NO_CALIBRATION)
        if self._paused:
            self._flush()
            return self._snapshot(SessionState.PAUSED)

        reason = self._abstain_reason(tick)
        if reason is not None:
            if not self._was_lost:
                self.stats.signal_loss_events += 1
            self._was_lost = True
            self._flush()
            return self._snapshot(SessionState.SIGNAL_LOST, abstain_reason=reason)
        assert tick.prediction is not None
        point = tick.prediction

        if self._signal_since is None:
            self._signal_since = now
        if self._was_lost:
            stable_for = now - self._signal_since
            if stable_for < self._config.recovery_seconds:
                return self._snapshot(
                    SessionState.RECOVERING,
                    recovery_progress=stable_for / self._config.recovery_seconds,
                )
            self._was_lost = False

        if self._pending is not None and now > self._pending_until:
            self._discard_pending()

        if now < self._cooldown_until:
            self._clear_focus()
            return self._snapshot(SessionState.COOLDOWN)

        margin = self._config.hysteresis_margin
        if self._rearm is not None:
            if self._rearm.contains(point, margin):
                self._clear_focus()
                return self._snapshot(SessionState.AWAITING_EXIT)
            self._rearm = None

        candidate = self._candidate(point)
        idle_state = SessionState.AWAITING_CONFIRMATION if self._pending else SessionState.READY
        if candidate is None:
            self._cancel_focus()
            return self._snapshot(idle_state)
        if candidate is not self._focused:
            self._cancel_focus()
            self._focused = candidate
            self._focus_since = now
        assert self._focus_since is not None
        progress = min(1.0, (now - self._focus_since) / self._config.dwell_seconds)
        if progress < 1.0:
            state = idle_state if self._pending else SessionState.FOCUSING
            return self._snapshot(state, progress=progress)

        self._clear_focus()
        if not self._config.confirmation_required:
            return self._confirm(candidate, now, "gaze")
        if self._pending is candidate:
            return self._confirm(candidate, now, "gaze")
        # First dwell (or a change of mind): hold as pending, require leaving and
        # returning to the same option within the confirmation window.
        self._pending = candidate
        self._pending_until = now + self._config.confirmation_window_seconds
        self._rearm = candidate
        return self._snapshot(SessionState.AWAITING_CONFIRMATION)

    # ------------------------------------------------------------------ helpers
    def _abstain_reason(self, tick: GazeInput) -> AbstainReason | None:
        if tick.frame_timestamp_seconds is None:
            return AbstainReason.NO_FRAME
        age = tick.now_seconds - tick.frame_timestamp_seconds
        if age > self._config.max_frame_age_seconds or age < 0.0:
            return AbstainReason.STALE_FRAME
        if not tick.tracking_ok:
            return AbstainReason.TRACKING
        if not eye_signal_usable(self._eye_mode, tick.left_eye, tick.right_eye):
            return AbstainReason.EYE_SIGNAL
        if tick.prediction is None:
            return AbstainReason.NO_PREDICTION
        return None

    def _candidate(self, point: tuple[float, float]) -> BoardOption | None:
        if self._focused is not None and self._focused.contains(
            point, self._config.hysteresis_margin
        ):
            return self._focused
        return self._layout.option_at(point)

    def _confirm(self, option: BoardOption, now: float, source: str) -> SessionUpdate:
        self._pending = None
        self._rearm = option if source == "gaze" else None
        self._cooldown_until = now + self._config.cooldown_seconds
        self._message = option.phrase
        self.stats.confirmations_by_source[source] += 1
        self.stats.confirmations_by_option[option.key] += 1
        return self._snapshot(
            SessionState.COOLDOWN,
            confirmed=option,
            confirmed_source=source,
        )

    def _cancel_focus(self) -> None:
        if self._focused is not None:
            self.stats.focus_cancellations += 1
        self._clear_focus()

    def _clear_focus(self) -> None:
        self._focused = None
        self._focus_since = None

    def _discard_pending(self) -> None:
        if self._pending is not None:
            self.stats.pending_discarded += 1
        self._pending = None
        self._rearm = None

    def _flush(self) -> None:
        """Discard everything that could lead to an emission. Keeps the visible message."""
        self._cancel_focus()
        self._discard_pending()
        self._signal_since = None
        self._was_lost = True

    def _snapshot(
        self,
        state: SessionState,
        *,
        progress: float = 0.0,
        confirmed: BoardOption | None = None,
        confirmed_source: str | None = None,
        abstain_reason: AbstainReason | None = None,
        recovery_progress: float = 0.0,
    ) -> SessionUpdate:
        return SessionUpdate(
            state=state,
            focused=self._focused,
            progress=progress,
            pending=self._pending,
            confirmed=confirmed,
            confirmed_source=confirmed_source,
            message=self._message,
            abstain_reason=abstain_reason,
            recovery_progress=recovery_progress,
        )


__all__ = [
    "AbstainReason",
    "CommunicationSession",
    "GazeInput",
    "SessionState",
    "SessionStats",
    "SessionUpdate",
]
