from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from statistics import fmean, median, pstdev
from time import perf_counter_ns
from typing import Any

import numpy as np
import psutil

from mekasa_gaze.core.domain import (
    CameraFrame,
    CameraSettings,
    FeatureObservation,
    TrackingState,
)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": len(values),
        "mean": fmean(values),
        "median": median(values),
        "p90": float(np.percentile(values, 90)),
        "max": max(values),
    }


@dataclass(slots=True)
class ExperimentMetrics:
    """Real-time performance and tracking quality metrics aggregator.

    Responsibility:
        Records frame-by-frame latencies (video capture, ML inference, rendering, display),
        tracks participant facial tracking availability runs and loss transitions, samples
        process memory (RSS) and CPU utilization, and aggregates feature stability distributions.
    """

    started_ns: int
    _capture_latencies: list[float]
    _feature_latencies: list[float]
    _loop_latencies: list[float]
    _render_latencies: list[float]
    _display_latencies: list[float]
    _iteration_latencies: list[float]
    _features: dict[str, list[float]]
    _states: Counter[TrackingState]
    _rss_samples_mb: list[float]
    _previous_available: bool | None
    _loss_events: int
    _current_loss_frames: int
    _longest_loss_frames: int
    _frames: int
    _first_frame_timestamp_ns: int | None
    _last_frame_timestamp_ns: int | None
    _process: psutil.Process
    _process_cpu_seconds_started: float

    @classmethod
    def start(cls) -> ExperimentMetrics:
        """Initialize metrics collection baseline using current process CPU and timestamp.

        Returns:
            ExperimentMetrics: Ready-to-record metrics accumulator instance.
        """
        process = psutil.Process()
        cpu_times = process.cpu_times()
        return cls(
            started_ns=perf_counter_ns(),
            _capture_latencies=[],
            _feature_latencies=[],
            _loop_latencies=[],
            _render_latencies=[],
            _display_latencies=[],
            _iteration_latencies=[],
            _features={},
            _states=Counter(),
            _rss_samples_mb=[],
            _previous_available=None,
            _loss_events=0,
            _current_loss_frames=0,
            _longest_loss_frames=0,
            _frames=0,
            _first_frame_timestamp_ns=None,
            _last_frame_timestamp_ns=None,
            _process=process,
            _process_cpu_seconds_started=cpu_times.user + cpu_times.system,
        )

    def add(
        self,
        frame: CameraFrame,
        observation: FeatureObservation,
        loop_latency_ms: float,
        *,
        render_latency_ms: float | None = None,
        display_latency_ms: float | None = None,
        iteration_latency_ms: float | None = None,
    ) -> None:
        """Record instantaneous telemetry metrics for a single processing iteration.

        Args:
            frame: Camera frame with capture timestamp and latency.
            observation: Extracted facial feature observation.
            loop_latency_ms: Total pipeline iteration wall-clock duration in milliseconds.
            render_latency_ms: UI rendering time in milliseconds, if applicable.
            display_latency_ms: Window presentation time in milliseconds, if applicable.
            iteration_latency_ms: Complete frame-to-frame period in milliseconds, if applicable.

        Side Effects:
            Appends latency samples, updates tracking loss counters, and samples process memory.
        """
        self._frames += 1
        if self._first_frame_timestamp_ns is None:
            self._first_frame_timestamp_ns = frame.captured_at_ns
        self._last_frame_timestamp_ns = frame.captured_at_ns
        self._capture_latencies.append(frame.capture_latency_ms)
        self._feature_latencies.append(observation.feature_latency_ms)
        self._loop_latencies.append(loop_latency_ms)
        if render_latency_ms is not None:
            self._render_latencies.append(render_latency_ms)
        if display_latency_ms is not None:
            self._display_latencies.append(display_latency_ms)
        if iteration_latency_ms is not None:
            self._iteration_latencies.append(iteration_latency_ms)
        self._states[observation.tracking_state] += 1

        available = observation.tracking_available
        if self._previous_available is True and not available:
            self._loss_events += 1
        if available:
            self._longest_loss_frames = max(self._longest_loss_frames, self._current_loss_frames)
            self._current_loss_frames = 0
            for name, value in observation.features.items():
                self._features.setdefault(name, []).append(float(value))
            if observation.head_pose is not None:
                self._features.setdefault("head_yaw_degrees", []).append(
                    observation.head_pose.yaw_degrees
                )
                self._features.setdefault("head_pitch_degrees", []).append(
                    observation.head_pose.pitch_degrees
                )
                self._features.setdefault("head_roll_degrees", []).append(
                    observation.head_pose.roll_degrees
                )
        else:
            self._current_loss_frames += 1
        self._previous_available = available

        self._rss_samples_mb.append(self._process.memory_info().rss / (1024 * 1024))

    def report(self, camera: CameraSettings, environment: dict[str, Any]) -> dict[str, Any]:
        """Compile aggregated telemetry, latencies, CPU/memory stats, and feature stability.

        Args:
            camera: Camera settings used during the run.
            environment: Host runtime and package environment metadata.

        Returns:
            dict[str, Any]: Complete experiment measurement report dictionary (Schema 1).
        """
        completed_ns = perf_counter_ns()
        duration_s = (completed_ns - self.started_ns) / 1_000_000_000
        cpu_times = self._process.cpu_times()
        process_cpu_seconds = cpu_times.user + cpu_times.system - self._process_cpu_seconds_started
        logical_cpu_count = psutil.cpu_count(logical=True) or 1
        one_core_cpu_percent = 100.0 * process_cpu_seconds / duration_s if duration_s else None
        stream_duration_s = (
            (self._last_frame_timestamp_ns - self._first_frame_timestamp_ns) / 1_000_000_000
            if self._frames > 1
            and self._first_frame_timestamp_ns is not None
            and self._last_frame_timestamp_ns is not None
            else None
        )
        self._longest_loss_frames = max(self._longest_loss_frames, self._current_loss_frames)
        good = self._states[TrackingState.GOOD]
        stability = {
            name: {
                "sample_count": len(values),
                "population_stddev": pstdev(values) if len(values) > 1 else 0.0,
                "median_absolute_deviation": float(
                    np.median(np.abs(np.asarray(values) - np.median(values)))
                ),
            }
            for name, values in sorted(self._features.items())
            if values
        }
        return {
            "schema_version": 1,
            "measurement": {
                "duration_seconds": duration_s,
                "stream_duration_seconds": stream_duration_s,
                "frames_attempted": self._frames,
                "effective_fps": (
                    (self._frames - 1) / stream_duration_s
                    if stream_duration_s is not None and stream_duration_s > 0.0
                    else None
                ),
                "tracking_availability": good / self._frames if self._frames else None,
                "tracking_loss_events": self._loss_events,
                "longest_non_good_run_frames": self._longest_loss_frames,
                "tracking_state_counts": {
                    state.value: self._states[state] for state in TrackingState
                },
                "capture_latency_ms": _summary(self._capture_latencies),
                "feature_latency_ms": _summary(self._feature_latencies),
                "loop_latency_ms": _summary(self._loop_latencies),
                "render_latency_ms": _summary(self._render_latencies),
                "display_latency_ms": _summary(self._display_latencies),
                "iteration_latency_ms": _summary(self._iteration_latencies),
                "process_cpu": {
                    "process_cpu_seconds": process_cpu_seconds,
                    "utilization_percent_one_core_basis": one_core_cpu_percent,
                    "utilization_percent_total_capacity": (
                        one_core_cpu_percent / logical_cpu_count
                        if one_core_cpu_percent is not None
                        else None
                    ),
                    "logical_cpu_count": logical_cpu_count,
                    "semantics": (
                        "Process user+system CPU time divided by measured wall time. "
                        "One-core basis may exceed 100%; total-capacity basis divides "
                        "by logical CPU count."
                    ),
                },
                "resident_memory_mb": _summary(self._rss_samples_mb),
                "feature_stability": stability,
            },
            "camera": asdict(camera),
            "environment": environment,
            "privacy": {
                "frames_persisted": False,
                "video_uploaded": False,
                "raw_landmarks_persisted": False,
            },
            "limitations": [
                "Feature stability is meaningful only if the participant attempted a steady pose.",
                "MediaPipe tracking evidence is not calibrated gaze confidence.",
                "Head pose is an unvalidated diagnostic approximation.",
                "This technical run is not clinical validation.",
            ],
        }
