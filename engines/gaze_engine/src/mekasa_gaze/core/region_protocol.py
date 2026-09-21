from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

from mekasa_gaze.core.config import InteractionConfig
from mekasa_gaze.core.domain import RegionName


class RegionTrialStage(StrEnum):
    """Execution stages of a single region trial within the controlled protocol."""

    PREPARATION = "PREPARATION"
    COUNTDOWN = "COUNTDOWN"
    CUE = "CUE"
    RESPONSE = "RESPONSE"
    FEEDBACK = "FEEDBACK"
    COMPLETE = "COMPLETE"


class RegionTrialOutcome(StrEnum):
    """Categorical outcomes for completed region selection trials."""

    PRACTICE = "PRACTICE"
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    NO_SELECTION = "NO_SELECTION"


@dataclass(frozen=True, slots=True)
class RegionTrialResult:
    """Outcome record of a completed region selection trial.

    Responsibility:
        Encapsulates trial identification, target region, participant selection,
        evaluated outcome, and response latency in seconds.
    """

    trial_index: int
    block_index: int
    requested: RegionName
    selected: RegionName | None
    outcome: RegionTrialOutcome
    response_seconds: float | None


@dataclass(frozen=True, slots=True)
class RegionProtocolSnapshot:
    """Point-in-time state of the controlled region selection protocol.

    Responsibility:
        Provides instantaneous trial stage, requested target region, active indices,
        time remaining in current phase, completed trial result, and completion status.
    """

    stage: RegionTrialStage
    requested: RegionName | None
    trial_index: int | None
    block_index: int | None
    stage_remaining_seconds: float | None
    completed_trial: RegionTrialResult | None = None
    completed: bool = False
    is_practice: bool = False


class ControlledRegionProtocol:
    """Deterministic four-region prompt/response protocol driven by monotonic time.

    Responsibility:
        Coordinates the timed progression of randomized region selection trials,
        managing stability countdowns, cue presentation, user response capture,
        feedback intervals, and block randomization.
    """

    def __init__(self, config: InteractionConfig) -> None:
        """Initialize the controlled region selection protocol.

        Args:
            config: Interaction configuration defining phase durations, block counts, and seed.
        """
        self._config = config
        self._trials = self._build_trials(config)
        self._stage = RegionTrialStage.PREPARATION
        self._stable_since: float | None = None
        self._countdown_since: float | None = None
        self._stage_started_at: float | None = None
        self._response_started_at: float | None = None
        self._trial_index = 0
        self._feedback_result: RegionTrialResult | None = None

    @staticmethod
    def _build_trials(config: InteractionConfig) -> tuple[tuple[int, RegionName], ...]:
        """Generate randomized target sequences across all validation blocks.

        Args:
            config: Interaction configuration containing validation block count and seed.

        Returns:
            tuple[tuple[int, RegionName], ...]: Tuple of (block_index, requested_region) pairs.
        """
        generator = random.Random(config.target_order_seed)
        trials: list[tuple[int, RegionName]] = []
        for block_index in range(1, config.validation_blocks + 1):
            regions = list(RegionName)
            generator.shuffle(regions)
            trials.extend((block_index, region) for region in regions)
        return tuple(trials)

    @property
    def planned_orders(self) -> tuple[tuple[RegionName, ...], ...]:
        """Return planned region prompt presentation sequences partitioned by block."""
        return tuple(
            tuple(region for block, region in self._trials if block == block_index)
            for block_index in range(1, self._config.validation_blocks + 1)
        )

    def update(
        self,
        now_seconds: float,
        *,
        ready: bool,
        selected: RegionName | None = None,
    ) -> RegionProtocolSnapshot:
        """Advance the protocol state machine based on monotonic time and region selection.

        Args:
            now_seconds: Monotonic timestamp in seconds (from `time.perf_counter()`).
            ready: Whether tracking and head placement satisfy operational readiness.
            selected: Region confirmed/selected by dwell interaction during this tick, if any.

        Returns:
            RegionProtocolSnapshot: Active trial state, target prompt, remaining time, or feedback.

        Side Effects:
            Mutates internal trial index, stage state, and timing checkpoints.
        """
        if self._stage in (RegionTrialStage.PREPARATION, RegionTrialStage.COUNTDOWN):
            return self._update_preparation(now_seconds, ready)
        if self._stage is RegionTrialStage.COMPLETE:
            return RegionProtocolSnapshot(
                RegionTrialStage.COMPLETE,
                None,
                None,
                None,
                None,
                completed=True,
            )

        assert self._stage_started_at is not None
        block_index, requested = self._trials[self._trial_index]
        elapsed = now_seconds - self._stage_started_at
        if self._stage is RegionTrialStage.CUE:
            if elapsed >= self._config.cue_seconds:
                self._stage = RegionTrialStage.RESPONSE
                self._stage_started_at = now_seconds
                self._response_started_at = now_seconds
                elapsed = 0.0
            else:
                return self._snapshot(requested, block_index, self._config.cue_seconds - elapsed)

        if self._stage is RegionTrialStage.RESPONSE:
            assert self._response_started_at is not None
            response_elapsed = now_seconds - self._response_started_at
            if block_index <= self._config.practice_blocks:
                if response_elapsed >= self._config.practice_measurement_seconds:
                    result = RegionTrialResult(
                        self._trial_index,
                        block_index,
                        requested,
                        None,
                        RegionTrialOutcome.PRACTICE,
                        None,
                    )
                    return self._start_feedback(now_seconds, result)
                return self._snapshot(
                    requested,
                    block_index,
                    self._config.practice_measurement_seconds - response_elapsed,
                )
            if selected is not None:
                outcome = (
                    RegionTrialOutcome.CORRECT
                    if selected is requested
                    else RegionTrialOutcome.INCORRECT
                )
                result = RegionTrialResult(
                    self._trial_index,
                    block_index,
                    requested,
                    selected,
                    outcome,
                    response_elapsed,
                )
                return self._start_feedback(now_seconds, result)
            if response_elapsed >= self._config.response_timeout_seconds:
                result = RegionTrialResult(
                    self._trial_index,
                    block_index,
                    requested,
                    None,
                    RegionTrialOutcome.NO_SELECTION,
                    None,
                )
                return self._start_feedback(now_seconds, result)
            return self._snapshot(
                requested,
                block_index,
                self._config.response_timeout_seconds - response_elapsed,
            )

        assert self._stage is RegionTrialStage.FEEDBACK
        if elapsed < self._config.feedback_seconds:
            return self._snapshot(requested, block_index, self._config.feedback_seconds - elapsed)
        self._trial_index += 1
        if self._trial_index >= len(self._trials):
            self._stage = RegionTrialStage.COMPLETE
            return RegionProtocolSnapshot(
                RegionTrialStage.COMPLETE,
                None,
                None,
                None,
                None,
                completed=True,
            )
        self._stage = RegionTrialStage.CUE
        self._stage_started_at = now_seconds
        self._feedback_result = None
        next_block, next_region = self._trials[self._trial_index]
        return self._snapshot(next_region, next_block, self._config.cue_seconds)

    def _update_preparation(self, now_seconds: float, ready: bool) -> RegionProtocolSnapshot:
        if not ready:
            self._stage = RegionTrialStage.PREPARATION
            self._stable_since = None
            self._countdown_since = None
            return RegionProtocolSnapshot(RegionTrialStage.PREPARATION, None, None, None, None)
        if self._stable_since is None:
            self._stable_since = now_seconds
        stable_elapsed = now_seconds - self._stable_since
        if stable_elapsed < self._config.tracking_stable_seconds:
            return RegionProtocolSnapshot(
                RegionTrialStage.PREPARATION,
                None,
                None,
                None,
                self._config.tracking_stable_seconds - stable_elapsed,
            )
        self._stage = RegionTrialStage.COUNTDOWN
        if self._countdown_since is None:
            self._countdown_since = now_seconds
        countdown_elapsed = now_seconds - self._countdown_since
        if countdown_elapsed < self._config.countdown_seconds:
            return RegionProtocolSnapshot(
                RegionTrialStage.COUNTDOWN,
                None,
                None,
                None,
                self._config.countdown_seconds - countdown_elapsed,
            )
        self._stage = RegionTrialStage.CUE
        self._stage_started_at = now_seconds
        block_index, requested = self._trials[0]
        return self._snapshot(requested, block_index, self._config.cue_seconds)

    def _start_feedback(
        self, now_seconds: float, result: RegionTrialResult
    ) -> RegionProtocolSnapshot:
        self._stage = RegionTrialStage.FEEDBACK
        self._stage_started_at = now_seconds
        self._feedback_result = result
        return RegionProtocolSnapshot(
            self._stage,
            result.requested,
            result.trial_index,
            result.block_index,
            self._config.feedback_seconds,
            completed_trial=result,
            is_practice=result.block_index <= self._config.practice_blocks,
        )

    def _snapshot(
        self, requested: RegionName, block_index: int, remaining: float
    ) -> RegionProtocolSnapshot:
        return RegionProtocolSnapshot(
            self._stage,
            requested,
            self._trial_index,
            block_index,
            max(0.0, remaining),
            completed_trial=(
                self._feedback_result if self._stage is RegionTrialStage.FEEDBACK else None
            ),
            is_practice=block_index <= self._config.practice_blocks,
        )
