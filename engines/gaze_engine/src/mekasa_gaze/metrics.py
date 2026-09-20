from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from statistics import fmean, median, pstdev
from time import perf_counter_ns
from typing import Any

import numpy as np
import psutil

from mekasa_gaze.camera import CameraFrame, CameraSettings
from mekasa_gaze.domain import FeatureObservation, TrackingState


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
    started_ns: int
    _capture_latencies: list[float]
    _feature_latencies: list[float]
    _loop_latencies: list[float]
    _features: dict[str, list[float]]
    _states: Counter[TrackingState]
    _cpu_samples: list[float]
    _rss_samples_mb: list[float]
    _previous_available: bool | None
    _loss_events: int
    _current_loss_frames: int
    _longest_loss_frames: int
    _frames: int
    _process: psutil.Process

    @classmethod
    def start(cls) -> ExperimentMetrics:
        process = psutil.Process()
        process.cpu_percent(interval=None)
        return cls(
            started_ns=perf_counter_ns(),
            _capture_latencies=[],
            _feature_latencies=[],
            _loop_latencies=[],
            _features={},
            _states=Counter(),
            _cpu_samples=[],
            _rss_samples_mb=[],
            _previous_available=None,
            _loss_events=0,
            _current_loss_frames=0,
            _longest_loss_frames=0,
            _frames=0,
            _process=process,
        )

    def add(
        self,
        frame: CameraFrame,
        observation: FeatureObservation,
        loop_latency_ms: float,
    ) -> None:
        self._frames += 1
        self._capture_latencies.append(frame.capture_latency_ms)
        self._feature_latencies.append(observation.feature_latency_ms)
        self._loop_latencies.append(loop_latency_ms)
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

        self._cpu_samples.append(self._process.cpu_percent(interval=None))
        self._rss_samples_mb.append(self._process.memory_info().rss / (1024 * 1024))

    def report(self, camera: CameraSettings, environment: dict[str, Any]) -> dict[str, Any]:
        completed_ns = perf_counter_ns()
        duration_s = (completed_ns - self.started_ns) / 1_000_000_000
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
                "frames_attempted": self._frames,
                "effective_fps": self._frames / duration_s if duration_s else None,
                "tracking_availability": good / self._frames if self._frames else None,
                "tracking_loss_events": self._loss_events,
                "longest_non_good_run_frames": self._longest_loss_frames,
                "tracking_state_counts": {
                    state.value: self._states[state] for state in TrackingState
                },
                "capture_latency_ms": _summary(self._capture_latencies),
                "feature_latency_ms": _summary(self._feature_latencies),
                "loop_latency_ms": _summary(self._loop_latencies),
                "process_cpu_percent": {
                    **_summary(self._cpu_samples),
                    "semantics": "psutil process utilization; may exceed 100% on multicore systems",
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
