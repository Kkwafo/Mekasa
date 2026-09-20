from __future__ import annotations

from mekasa_gaze.camera import CameraFrame, CameraSettings
from mekasa_gaze.domain import FeatureObservation, TrackingState
from mekasa_gaze.metrics import ExperimentMetrics


def _frame(sequence: int) -> CameraFrame:
    import numpy as np

    return CameraFrame(
        sequence,
        sequence * 100_000_000,
        2.0,
        np.zeros((2, 2, 3), dtype=np.uint8),
    )


def _observation(sequence: int, state: TrackingState) -> FeatureObservation:
    return FeatureObservation(
        frame_sequence=sequence,
        timestamp_ns=sequence,
        tracking_state=state,
        features={"left_iris_x": 0.5} if state is TrackingState.GOOD else {},
        evidence="test",
        feature_latency_ms=4.0,
    )


def test_tracking_loss_transitions_are_counted_once_per_run() -> None:
    metrics = ExperimentMetrics.start()
    states = [
        TrackingState.GOOD,
        TrackingState.LOST,
        TrackingState.LOST,
        TrackingState.GOOD,
        TrackingState.UNCERTAIN,
    ]
    for index, state in enumerate(states):
        metrics.add(_frame(index), _observation(index, state), 6.0)
    report = metrics.report(CameraSettings(0, "test", 2, 2, 30.0), {})
    measurement = report["measurement"]
    assert measurement["tracking_loss_events"] == 2
    assert measurement["longest_non_good_run_frames"] == 2
    assert measurement["tracking_availability"] == 2 / 5
    assert measurement["tracking_state_counts"]["TRACKING_LOST"] == 2
    assert measurement["process_cpu"]["logical_cpu_count"] >= 1
    assert measurement["process_cpu"]["process_cpu_seconds"] >= 0.0
    assert measurement["stream_duration_seconds"] == 0.4
    assert measurement["effective_fps"] == 10.0


def test_optional_render_and_iteration_latencies_are_reported() -> None:
    metrics = ExperimentMetrics.start()
    metrics.add(
        _frame(0),
        _observation(0, TrackingState.GOOD),
        6.0,
        render_latency_ms=2.0,
        display_latency_ms=3.0,
        iteration_latency_ms=11.0,
    )
    measurement = metrics.report(CameraSettings(0, "test", 2, 2, 30.0), {})["measurement"]
    assert measurement["render_latency_ms"]["median"] == 2.0
    assert measurement["display_latency_ms"]["median"] == 3.0
    assert measurement["iteration_latency_ms"]["median"] == 11.0
