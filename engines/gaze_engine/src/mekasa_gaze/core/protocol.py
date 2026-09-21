from __future__ import annotations

import random
from dataclasses import dataclass, replace

from mekasa_gaze.core.config import GuidedValidationConfig
from mekasa_gaze.core.domain import (
    ExperimentPhase,
    MeasurementStage,
    ProtocolSnapshot,
    TargetName,
    TrackingState,
    VisualTarget,
)


@dataclass(frozen=True, slots=True)
class ProtocolSegment:
    """Scheduled segment in the guided validation protocol.

    Responsibility:
        Associates an experiment phase, visual target stimulus, measurement duration,
        and block counter into an immutable scheduled segment.
    """

    phase: ExperimentPhase
    target: VisualTarget
    measurement_seconds: float
    block_index: int | None = None


class GuidedProtocol:
    """Deterministic guided-validation state machine driven by monotonic time.

    Responsibility:
        Coordinates the timed progression of multi-target calibration and validation
        experiments, managing preparation, stability, target presentation, pause on tracking
        loss, and resumption upon face re-acquisition.
    """

    def __init__(self, config: GuidedValidationConfig) -> None:
        """Initialize the protocol state machine.

        Args:
            config: Guided validation configuration defining phase durations and targets.
        """
        self._config = config
        self._segments = self._build_segments(config)
        self._stable_since: float | None = None
        self._countdown_since: float | None = None
        self._protocol_started_at: float | None = None
        self._segment_started_at: float | None = None
        self._segment_index = 0
        self._pause_started_at: float | None = None
        self._paused_snapshot: ProtocolSnapshot | None = None

    @staticmethod
    def _build_segments(config: GuidedValidationConfig) -> tuple[ProtocolSegment, ...]:
        """Construct the deterministic sequence of protocol segments.

        Args:
            config: Guided validation configuration.

        Returns:
            tuple[ProtocolSegment, ...]: Complete planned segment schedule.
        """
        center = config.target(TargetName.CENTER)
        generator = random.Random(config.target_order_seed)
        segments: list[ProtocolSegment] = [
            ProtocolSegment(
                ExperimentPhase.CENTER_STABILITY,
                center,
                config.center_stability_seconds,
            )
        ]
        for block_index in range(1, config.discrimination_blocks + 1):
            names = list(TargetName)
            generator.shuffle(names)
            segments.extend(
                ProtocolSegment(
                    ExperimentPhase.GAZE_DISCRIMINATION,
                    config.target(name),
                    config.measurement_seconds,
                    block_index=block_index,
                )
                for name in names
            )
        segments.extend(
            ProtocolSegment(
                ExperimentPhase.HEAD_MOVEMENT,
                center,
                config.head_movement_seconds,
                block_index=block_index,
            )
            for block_index in range(1, config.discrimination_blocks + 1)
        )
        return tuple(segments)

    @property
    def planned_measurement_segments(self) -> tuple[ProtocolSegment, ...]:
        """Return all planned protocol segments in execution order."""
        return self._segments

    @property
    def planned_target_orders(self) -> tuple[tuple[TargetName, ...], ...]:
        """Return target presentation orders partitioned by discrimination block."""
        return tuple(
            tuple(
                segment.target.name
                for segment in self._segments
                if segment.block_index == block_index
                and segment.phase is ExperimentPhase.GAZE_DISCRIMINATION
            )
            for block_index in range(1, self._config.discrimination_blocks + 1)
        )

    def update(self, now_seconds: float, tracking_state: TrackingState) -> ProtocolSnapshot:
        """Advance the protocol state machine based on current time and tracking quality.

        Args:
            now_seconds: Monotonic timestamp in seconds (from `time.perf_counter()`).
            tracking_state: Current tracking assessment (TrackingState.GOOD, UNCERTAIN, LOST).

        Returns:
            ProtocolSnapshot: Current state, active target, remaining countdown, and instruction.

        Side Effects:
            Mutates internal timing checkpoints, advances segment indices, or triggers pause/resume.
        """
        if self._protocol_started_at is None:
            return self._update_preparation(now_seconds, tracking_state)
        if tracking_state is not TrackingState.GOOD:
            return self._pause_running(now_seconds)
        if self._pause_started_at is not None:
            paused_seconds = now_seconds - self._pause_started_at
            self._protocol_started_at += paused_seconds
            assert self._segment_started_at is not None
            self._segment_started_at += paused_seconds
            self._pause_started_at = None
            self._paused_snapshot = None
        return self._update_running(now_seconds)

    def _pause_running(self, now_seconds: float) -> ProtocolSnapshot:
        """Freeze protocol progression when tracking is lost."""
        if self._pause_started_at is None:
            frozen = self._update_running(now_seconds)
            if frozen.completed:
                return frozen
            self._pause_started_at = now_seconds
            self._paused_snapshot = replace(
                frozen,
                stage=MeasurementStage.WAITING_TRACKING,
                instruction="Recuperá una posición válida frente a la cámara.",
            )
        assert self._paused_snapshot is not None
        return self._paused_snapshot

    def _update_preparation(
        self, now_seconds: float, tracking_state: TrackingState
    ) -> ProtocolSnapshot:
        """Handle initial tracking acquisition and stability countdown before starting."""
        if tracking_state is not TrackingState.GOOD:
            self._stable_since = None
            self._countdown_since = None
            return ProtocolSnapshot(
                phase=ExperimentPhase.PREPARATION,
                stage=MeasurementStage.WAITING_TRACKING,
                target=None,
                instruction="Ubicá tu rostro frente a la cámara.",
                stage_remaining_seconds=None,
                protocol_elapsed_seconds=0.0,
                phase_remaining_seconds=None,
            )

        if self._stable_since is None:
            self._stable_since = now_seconds
        stable_elapsed = now_seconds - self._stable_since
        if stable_elapsed < self._config.tracking_stable_seconds:
            return ProtocolSnapshot(
                phase=ExperimentPhase.PREPARATION,
                stage=MeasurementStage.WAITING_TRACKING,
                target=None,
                instruction="Mantené el rostro visible y cómodo.",
                stage_remaining_seconds=max(
                    0.0, self._config.tracking_stable_seconds - stable_elapsed
                ),
                protocol_elapsed_seconds=0.0,
                phase_remaining_seconds=None,
            )

        if self._countdown_since is None:
            self._countdown_since = now_seconds
        countdown_elapsed = now_seconds - self._countdown_since
        if countdown_elapsed < self._config.countdown_seconds:
            return ProtocolSnapshot(
                phase=ExperimentPhase.PREPARATION,
                stage=MeasurementStage.COUNTDOWN,
                target=None,
                instruction="La prueba comienza en…",
                stage_remaining_seconds=max(
                    0.0, self._config.countdown_seconds - countdown_elapsed
                ),
                protocol_elapsed_seconds=0.0,
                phase_remaining_seconds=max(
                    0.0, self._config.countdown_seconds - countdown_elapsed
                ),
            )

        self._protocol_started_at = now_seconds
        self._segment_started_at = now_seconds
        return self._update_running(now_seconds)

    def _update_running(self, now_seconds: float) -> ProtocolSnapshot:
        """Advance through active segments (transition, acquisition, measurement)."""
        assert self._protocol_started_at is not None
        assert self._segment_started_at is not None
        while self._segment_index < len(self._segments):
            segment = self._segments[self._segment_index]
            segment_total = (
                self._config.transition_seconds
                + self._config.acquisition_seconds
                + segment.measurement_seconds
            )
            segment_elapsed = now_seconds - self._segment_started_at
            if segment_elapsed < segment_total:
                return self._segment_snapshot(segment, segment_elapsed, now_seconds)
            self._segment_index += 1
            self._segment_started_at += segment_total

        return ProtocolSnapshot(
            phase=ExperimentPhase.COMPLETE,
            stage=MeasurementStage.COMPLETE,
            target=None,
            instruction="Prueba finalizada. Podés descansar.",
            stage_remaining_seconds=None,
            protocol_elapsed_seconds=now_seconds - self._protocol_started_at,
            phase_remaining_seconds=None,
            completed=True,
        )

    def _segment_snapshot(
        self,
        segment: ProtocolSegment,
        elapsed: float,
        now_seconds: float,
    ) -> ProtocolSnapshot:
        """Create a snapshot corresponding to the current point within an active segment."""
        if elapsed < self._config.transition_seconds:
            stage = MeasurementStage.TRANSITION
            remaining = self._config.transition_seconds - elapsed
            instruction = "Preparáte para el siguiente punto."
        elif elapsed < self._config.transition_seconds + self._config.acquisition_seconds:
            stage = MeasurementStage.ACQUISITION
            remaining = self._config.transition_seconds + self._config.acquisition_seconds - elapsed
            instruction = "Llevá la mirada al punto."
        else:
            stage = MeasurementStage.MEASUREMENT
            measurement_elapsed = elapsed - (
                self._config.transition_seconds + self._config.acquisition_seconds
            )
            remaining = segment.measurement_seconds - measurement_elapsed
            instruction = self._measurement_instruction(segment.phase)

        assert self._protocol_started_at is not None
        phase_remaining = self._phase_remaining_seconds(segment, elapsed)
        return ProtocolSnapshot(
            phase=segment.phase,
            stage=stage,
            target=segment.target,
            instruction=instruction,
            stage_remaining_seconds=max(0.0, remaining),
            protocol_elapsed_seconds=now_seconds - self._protocol_started_at,
            phase_remaining_seconds=phase_remaining,
            block_index=segment.block_index,
            segment_index=self._segment_index,
        )

    def _phase_remaining_seconds(self, current_segment: ProtocolSegment, elapsed: float) -> float:
        """Estimate remaining time for the current macro-phase across subsequent segments."""
        current_total = (
            self._config.transition_seconds
            + self._config.acquisition_seconds
            + current_segment.measurement_seconds
        )
        remaining = max(0.0, current_total - elapsed)
        for segment in self._segments[self._segment_index + 1 :]:
            if segment.phase is not current_segment.phase:
                break
            remaining += (
                self._config.transition_seconds
                + self._config.acquisition_seconds
                + segment.measurement_seconds
            )
        return remaining

    @staticmethod
    def _measurement_instruction(phase: ExperimentPhase) -> str:
        """Return Spanish instructional text for the given measurement phase."""
        if phase is ExperimentPhase.CENTER_STABILITY:
            return "Mirá el punto central. Cabeza quieta; parpadeá normalmente."
        if phase is ExperimentPhase.HEAD_MOVEMENT:
            return "Mirá el centro y mové la cabeza suavemente, sin forzar."
        return "Mantené la mirada sobre el punto."
