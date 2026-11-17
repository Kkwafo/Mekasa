"""Blink commands (2 = select, 4 = start/stop) and the blink selection mode of the session."""

from __future__ import annotations

from dataclasses import replace

from mekasa_gaze.core.board import CommunicationConfig
from mekasa_gaze.core.domain import EyeState
from mekasa_gaze.core.eye_mode import EyeMode
from mekasa_gaze.tracking.blinks import BlinkDetector, BlinkGesture
from mekasa_gaze.tracking.communication import (
    CommunicationSession,
    GazeInput,
    SessionState,
    SessionUpdate,
)

FPS = 30.0
OPEN, CLOSED = EyeState.OPEN, EyeState.CLOSED


class Feed:
    """Drives a detector frame by frame with a scripted eye state."""

    def __init__(self, detector: BlinkDetector) -> None:
        self.detector = detector
        self.t = 0.0
        self.gestures: list[BlinkGesture] = []

    def run(
        self, seconds: float, left: EyeState, right: EyeState | None = None, tracking: bool = True
    ) -> None:
        right = left if right is None else right
        for _ in range(round(seconds * FPS)):
            self.t += 1 / FPS
            gesture = self.detector.update(self.t, left, right, tracking_ok=tracking)
            if gesture is not None:
                self.gestures.append(gesture)

    def blinks(self, count: int, closed: float = 0.2, gap: float = 0.3) -> None:
        for index in range(count):
            self.run(closed, CLOSED)
            if index < count - 1:
                self.run(gap, OPEN)


def _counts(feed: Feed) -> list[int]:
    return [gesture.count for gesture in feed.gestures]


def test_two_and_four_blinks_are_commands_reported_after_the_sequence_ends() -> None:
    feed = Feed(BlinkDetector())
    feed.run(1.0, OPEN)
    feed.blinks(2)
    feed.run(0.5, OPEN)
    assert _counts(feed) == []  # still within the gap: could become 4
    assert feed.detector.count_in_progress == 2
    feed.run(0.5, OPEN)
    assert _counts(feed) == [2]
    feed.blinks(4)
    feed.run(1.5, OPEN)
    assert _counts(feed) == [2, 4]


def test_single_triple_and_long_closures_are_ignored() -> None:
    feed = Feed(BlinkDetector())
    feed.run(1.0, OPEN)
    feed.blinks(1)  # spontaneous blink
    feed.run(1.5, OPEN)
    feed.blinks(3)
    feed.run(1.5, OPEN)
    feed.blinks(1)
    feed.run(0.3, OPEN)
    feed.run(2.0, CLOSED)  # resting the eyes cancels the sequence in progress
    feed.run(0.3, OPEN)
    feed.blinks(1)
    feed.run(1.5, OPEN)
    assert _counts(feed) == []
    assert feed.detector.stats["ignored_long_closures"] == 1


def test_noise_and_face_loss_do_not_count() -> None:
    feed = Feed(BlinkDetector(min_closed_seconds=0.1))
    feed.run(0.5, OPEN)
    feed.run(1 / FPS, CLOSED)  # one-frame dip: noise
    feed.run(0.3, OPEN)
    feed.blinks(1)
    feed.run(0.3, OPEN, tracking=False)  # face lost mid-sequence resets it
    feed.blinks(1)
    feed.run(1.5, OPEN)
    assert _counts(feed) == []


def test_single_eye_mode_uses_only_that_eye() -> None:
    feed = Feed(BlinkDetector(eye_mode=EyeMode.LEFT))
    feed.run(0.5, OPEN)
    for _ in range(2):
        feed.run(0.2, CLOSED, EyeState.INVALID)  # right eye unusable, left blinks
        feed.run(0.3, OPEN, EyeState.INVALID)
    feed.run(1.0, OPEN, EyeState.INVALID)
    assert _counts(feed) == [2]


# ---------------------------------------------------------------- session in blink mode

YES_POINT = (0.25, 0.25)
GAP = (0.5, 0.5)


class Clock:
    def __init__(self, session: CommunicationSession) -> None:
        self.session = session
        self.t = 100.0

    def run(
        self,
        point: tuple[float, float] | None,
        seconds: float,
        *,
        eyes: EyeState = OPEN,
        tracking: bool = True,
    ) -> list[SessionUpdate]:
        updates = []
        for _ in range(round(seconds * FPS)):
            self.t += 1 / FPS
            tick = GazeInput(self.t, self.t, point if eyes is OPEN else None, tracking, eyes, eyes)
            updates.append(self.session.update(tick))
        return updates


def _blink_session() -> tuple[CommunicationSession, Clock]:
    config = replace(CommunicationConfig(), selection_mode="blink")
    session = CommunicationSession(config, calibration_mode=EyeMode.BOTH)
    return session, Clock(session)


def test_blink_mode_starts_on_standby_and_never_selects_by_time() -> None:
    session, clock = _blink_session()
    assert clock.run(YES_POINT, 2.0)[-1].state is SessionState.PAUSED
    session.resume()
    updates = clock.run(YES_POINT, 10.0)
    assert not any(update.confirmed for update in updates)
    assert updates[-1].armed is not None and updates[-1].armed.key == "YES"
    assert updates[-1].progress == 1.0


def test_double_blink_selects_the_armed_option_and_hard_loss_disarms() -> None:
    session, clock = _blink_session()
    session.resume()
    clock.run(YES_POINT, 1.5)
    clock.run(None, 0.2, eyes=EyeState.CLOSED)  # the blinks close the eyes...
    clock.run(YES_POINT, 0.3)
    clock.run(None, 0.2, eyes=EyeState.CLOSED)
    clock.run(YES_POINT, 0.8)
    update = session.select_by_blink(clock.t)  # ...but the option stays armed
    assert update is not None and update.confirmed is not None
    assert update.confirmed.key == "YES" and update.confirmed_source == "blink"
    # Without a new fixation the same blinks do nothing.
    clock.run(None, 1.2, eyes=EyeState.CLOSED)
    assert session.select_by_blink(clock.t) is None
    # Selecting the same option again only needs a new fixation (no exit rule for blinks).
    clock.run(YES_POINT, 1.5)
    clock.run(YES_POINT, 0.5, tracking=False)  # face lost: disarm
    assert session.select_by_blink(clock.t) is None
    assert session.stats.blink_selections_rejected == 2


def test_blink_selection_needs_recent_focus_and_active_reading() -> None:
    session, clock = _blink_session()
    session.resume()
    clock.run(GAP, 2.0)  # looking between options arms nothing
    assert session.select_by_blink(clock.t) is None
    clock.run(YES_POINT, 1.0)
    clock.run(None, 3.5, eyes=EyeState.CLOSED)  # too long since the fixation
    assert session.select_by_blink(clock.t) is None
    clock.run(YES_POINT, 1.0)
    session.pause()
    assert session.select_by_blink(clock.t) is None


def test_dwell_mode_ignores_blink_selection() -> None:
    session = CommunicationSession(CommunicationConfig(), calibration_mode=EyeMode.BOTH)
    clock = Clock(session)
    assert not session.paused
    clock.run(YES_POINT, 0.3)
    assert session.select_by_blink(clock.t) is None
