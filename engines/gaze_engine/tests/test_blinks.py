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
    feed = Feed(BlinkDetector(commands=(2, 4)))
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
    feed = Feed(BlinkDetector(commands=(2, 4)))
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
    feed = Feed(BlinkDetector(min_closed_seconds=0.1, commands=(2, 4)))
    feed.run(0.5, OPEN)
    feed.run(1 / FPS, CLOSED)  # one-frame dip: noise
    feed.run(0.3, OPEN)
    feed.blinks(1)
    feed.run(0.3, OPEN, tracking=False)  # face lost mid-sequence resets it
    feed.blinks(1)
    feed.run(1.5, OPEN)
    assert _counts(feed) == []


def test_single_eye_mode_uses_only_that_eye() -> None:
    feed = Feed(BlinkDetector(eye_mode=EyeMode.LEFT, commands=(2, 4)))
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


def test_default_commands_are_three_four_five_and_seven() -> None:
    feed = Feed(BlinkDetector())
    feed.run(0.5, OPEN)
    for count in (2, 3, 4, 5, 6, 7, 8):
        feed.blinks(count, closed=0.15, gap=0.25)
        feed.run(1.2, OPEN)
    assert _counts(feed) == [3, 4, 5, 7]


# ---------------------------------------------------------------- combined mode


def _combined() -> tuple[CommunicationSession, Clock]:
    config = replace(CommunicationConfig(), selection_mode="combined")
    session = CommunicationSession(config, calibration_mode=EyeMode.BOTH)
    return session, Clock(session)


def test_combined_mode_selects_after_three_seconds_or_by_blinking() -> None:
    session, clock = _combined()
    assert not session.paused and session.dwell_seconds == 3.0
    updates = clock.run(YES_POINT, 3.0)  # 0.5 s recovery at start + 2.5 s looking
    assert not any(update.confirmed for update in updates)
    assert updates[-1].armed is not None  # blinking now would choose it
    updates = clock.run(YES_POINT, 0.8)
    confirmed = [update.confirmed for update in updates if update.confirmed]
    assert [option.key for option in confirmed] == ["YES"]  # the 3 s look chose it
    assert session.select_by_blink(clock.t) is None  # the dwell choice disarmed it
    clock.run((0.75, 0.25), 1.5)  # leave YES (rearm-on-exit) by looking at NO
    clock.run(YES_POINT, 1.0)
    blink = session.select_by_blink(clock.t)
    assert blink is not None and blink.confirmed_source == "blink"


def test_dwell_freezes_while_a_blink_gesture_is_in_progress() -> None:
    session, clock = _combined()
    clock.run(YES_POINT, 2.0)
    held: list[SessionUpdate] = []
    for _ in range(round(3.0 * FPS)):  # 3 s of "blink gesture in progress"
        clock.t += 1 / FPS
        held.append(session.update(GazeInput(clock.t, clock.t, YES_POINT, True, OPEN, OPEN, True)))
    assert not any(update.confirmed for update in held)
    assert held[-1].progress < 1.0
    rest = clock.run(YES_POINT, 1.8)  # 1.5 s before the freeze + 1.5 s after = 3 s
    assert any(update.confirmed for update in rest)  # resumes where it was frozen


def test_combined_after_a_dwell_choice_blinks_may_choose_the_same_cell_but_dwell_may_not() -> None:
    session, clock = _combined()
    updates = clock.run(YES_POINT, 3.8)
    assert sum(1 for update in updates if update.confirmed) == 1
    # Still looking at the same cell (the next menu puts a new option there).
    updates = clock.run(YES_POINT, 6.0)
    assert not any(update.confirmed for update in updates)  # no dwell cascade
    assert updates[-1].state is SessionState.AWAITING_EXIT
    assert updates[-1].armed is not None  # ...but deliberate blinks can choose it
    blink = session.select_by_blink(clock.t)
    assert blink is not None and blink.confirmed_source == "blink"


# ---------------------------------------------------------------- winks


def test_winks_of_an_enabled_eye_form_their_own_sequence() -> None:
    feed = Feed(BlinkDetector(winks=("right",), wink_commands=(3,)))
    feed.run(0.5, OPEN)
    for _ in range(3):
        feed.run(0.2, OPEN, CLOSED)  # right eye closed, left open
        feed.run(0.3, OPEN)
    feed.run(1.0, OPEN)
    assert [(g.count, g.eyes) for g in feed.gestures] == [(3, "right")]
    # Left winks are not enabled: those frames are ignored (no closure, no count).
    for _ in range(3):
        feed.run(0.2, CLOSED, OPEN)
        feed.run(0.3, OPEN)
    feed.run(1.0, OPEN)
    assert len(feed.gestures) == 1
    # A both-eyes blink in the middle starts a new sequence of another kind.
    feed.run(0.2, OPEN, CLOSED)
    feed.run(0.3, OPEN)
    feed.blinks(3)
    feed.run(1.0, OPEN)
    assert [(g.count, g.eyes) for g in feed.gestures] == [(3, "right"), (3, "both")]


def test_single_eye_mode_never_reports_winks() -> None:
    feed = Feed(BlinkDetector(eye_mode=EyeMode.RIGHT, winks=("right",)))
    feed.run(0.5, OPEN)
    for _ in range(3):
        feed.run(0.2, OPEN, CLOSED)
        feed.run(0.3, OPEN)
    feed.run(1.0, OPEN)
    assert [(g.count, g.eyes) for g in feed.gestures] == [(3, "both")]  # the active eye


def test_wink_calibration_learns_per_eye_and_detects_who_can_wink() -> None:
    from mekasa_gaze.tracking.wink_calibration import WinkCalibration, fit_wink_profile

    grid = {"left": [0.30] * 200 + [0.18] * 40, "right": [0.29] * 200 + [0.17] * 40}
    step = WinkCalibration({"left": 0.29, "right": 0.28}, seconds=4.0)
    t, snapshots = 0.0, []

    def frame(left: float, right: float, seconds: float) -> None:
        nonlocal t
        for _ in range(round(seconds * FPS)):
            t += 1 / FPS
            snapshots.append(
                step.update(t, {"left_eye_aspect_ratio": left, "right_eye_aspect_ratio": right})
            )

    # Right eye: three clean winks (left stays open).
    for _ in range(3):
        frame(0.29, 0.29, 0.4)
        frame(0.28, 0.05, 0.25)
    frame(0.29, 0.29, 1.5)
    assert max(s.detected for s in snapshots if s.eye == "right") == 3
    # Left eye: the person cannot isolate it, both eyes close.
    while not snapshots[-1].done:
        frame(0.29, 0.29, 0.4)
        frame(0.05, 0.07, 0.25)
    profile = fit_wink_profile(step.records, grid)
    assert profile.right_wink and not profile.left_wink
    assert profile.right_detected == 3 and profile.left_detected == 0
    assert 0.05 < profile.right_threshold < 0.17  # below the lowest normal aperture (looking down)
    states = profile.eye_states(
        {"left_eye_aspect_ratio": 0.28, "right_eye_aspect_ratio": 0.06}, OPEN, OPEN
    )
    assert states == (OPEN, CLOSED)


def test_interrupt_resets_dwell_when_the_screen_changes_under_the_gaze() -> None:
    session, clock = _combined()
    clock.run(YES_POINT, 2.8)  # 2.3 s of a 3 s look
    session.interrupt(clock.t)  # e.g. 7 blinks opened the urgent screen
    updates = clock.run(YES_POINT, 1.5)
    assert not any(update.confirmed for update in updates)
    assert updates[0].state is SessionState.COOLDOWN
    assert updates[-1].progress < 0.3  # the 3 s look restarted from zero after the cooldown


# ---------------------------------------------------------------- blink first, look after


def test_urgent_fires_at_the_seventh_blink_and_repeats_while_blinking_continues() -> None:
    feed = Feed(BlinkDetector(commands=(3, 4, 5), urgent_count=7, urgent_repeat_seconds=2.5))
    feed.run(0.5, OPEN)
    feed.blinks(7, closed=0.15, gap=0.25)
    feed.run(1 / FPS, OPEN)
    assert [(g.count, g.urgent) for g in feed.gestures] == [(7, True)]  # no waiting
    feed.run(0.25, OPEN)
    feed.blinks(8, closed=0.15, gap=0.25)  # keeps blinking (same sequence)
    feed.run(1.5, OPEN)
    assert [(g.count, g.urgent) for g in feed.gestures] == [(7, True), (14, True)]


def test_blink_then_look_chooses_what_is_looked_at_after_the_blinks() -> None:
    session, clock = _combined()
    clock.run(YES_POINT, 1.0)
    clock.run((0.75, 0.75), 0.6)  # the estimate jumped elsewhere during the blinks
    assert session.start_selection_intent(clock.t)
    updates = clock.run((0.75, 0.25), 0.9)  # looks at NO
    assert updates[-1].state is SessionState.SELECTING and not updates[-1].confirmed
    updates = clock.run((0.75, 0.25), 0.3)
    confirmed = [u.confirmed.key for u in updates if u.confirmed]
    assert confirmed == ["NO"]
    assert [u.confirmed_source for u in updates if u.confirmed] == ["blink"]


def test_selection_window_expires_and_survives_a_blink() -> None:
    session, clock = _combined()
    clock.run(GAP, 1.0)
    assert session.start_selection_intent(clock.t, source="wink")
    clock.run(GAP, 0.5)
    clock.run(None, 0.3, eyes=EyeState.CLOSED)  # another blink: the window stays open
    assert session.selecting
    clock.run(GAP, 5.0)  # nothing looked at in time
    assert not session.selecting and session.stats.intents_expired == 1


def test_selection_window_lifts_the_exit_rule_and_dwell_mode_rejects_it() -> None:
    session, clock = _combined()
    clock.run(YES_POINT, 3.8)  # dwell choice; the exit rule now blocks YES
    assert session.start_selection_intent(clock.t)
    updates = clock.run(YES_POINT, 1.5)
    assert [u.confirmed.key for u in updates if u.confirmed] == ["YES"]
    dwell = CommunicationSession(CommunicationConfig(), calibration_mode=EyeMode.BOTH)
    assert not dwell.start_selection_intent(1.0)
