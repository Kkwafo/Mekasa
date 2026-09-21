from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from mekasa_gaze.core.config import GuidedValidationConfig
from mekasa_gaze.core.domain import (
    ExperimentPhase,
    FeatureObservation,
    HeadPose,
    MeasurementStage,
    ProtocolSnapshot,
    RegionName,
    TargetName,
    TrackingState,
)
from mekasa_gaze.runners.guided_experiment import _rolling_fps
from mekasa_gaze.tracking.interaction import SelectionUpdate
from mekasa_gaze.tracking.placement import PlacementAssessment, PlacementState
from mekasa_gaze.tracking.validity import assess_gaze_validity
from mekasa_gaze.ui.visualization import (
    render_guided_screen,
    render_loading_screen,
    render_region_selection_screen,
)


def _config() -> GuidedValidationConfig:
    return GuidedValidationConfig.load(Path("config/experiment-001.toml"))


def _observation() -> FeatureObservation:
    return FeatureObservation(
        frame_sequence=1,
        timestamp_ns=1,
        tracking_state=TrackingState.GOOD,
        features={
            "left_eye_aspect_ratio": 0.3,
            "right_eye_aspect_ratio": 0.31,
            "left_eye_vertical_span": 0.03,
            "right_eye_vertical_span": 0.031,
            "left_iris_x": 0.5,
            "left_iris_y": 0.5,
            "right_iris_x": 0.49,
            "right_iris_y": 0.51,
        },
        head_pose=HeadPose(yaw_degrees=1.0, pitch_degrees=-2.0, roll_degrees=0.5),
        debug_eye_points=((0.4, 0.4), (0.6, 0.4)),
        debug_left_iris_points=((0.44, 0.4),),
        debug_right_iris_points=((0.56, 0.4),),
        debug_face_box=(0.25, 0.15, 0.75, 0.85),
    )


def test_guided_screen_renders_target_and_debug_without_mutating_camera_frame() -> None:
    config = _config()
    camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    observation = _observation()
    snapshot = ProtocolSnapshot(
        phase=ExperimentPhase.GAZE_DISCRIMINATION,
        stage=MeasurementStage.MEASUREMENT,
        target=config.target(TargetName.CENTER),
        instruction="Mirá el punto central sin mover la cabeza.",
        stage_remaining_seconds=2.5,
        protocol_elapsed_seconds=10.0,
    )

    output = render_guided_screen(
        camera_frame,
        observation,
        assess_gaze_validity(observation, config),
        snapshot,
        config,
        rolling_fps=29.8,
    )

    assert output.shape == (config.canvas_height, config.canvas_width, 3)
    assert np.count_nonzero(output) > 0
    assert np.count_nonzero(camera_frame) == 0
    debug_start = config.canvas_width - round(config.canvas_width * config.debug_panel_fraction)
    assert np.count_nonzero(output[:, :debug_start]) > 0
    assert np.count_nonzero(output[:, debug_start:]) > 0


def test_rolling_fps_uses_elapsed_intervals() -> None:
    assert _rolling_fps(deque([0.0], maxlen=60)) is None
    assert _rolling_fps(deque([0.0, 0.5, 1.0], maxlen=60)) == 2.0


def test_loading_screen_has_requested_size_and_visible_content() -> None:
    output = render_loading_screen(1280, 720, title="MEKASA", status="Cargando...")

    assert output.shape == (720, 1280, 3)
    assert np.count_nonzero(output) > 0


def test_region_screen_animates_dwell_progress_without_mutating_camera_frame() -> None:
    config = _config()
    camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    observation = _observation()
    assessment = assess_gaze_validity(observation, config)
    placement = PlacementAssessment(PlacementState.READY, "DISTANCIA CORRECTA", 0.3)
    idle = SelectionUpdate(None, 0.0, None, False, 0.0, True)
    dwelling = SelectionUpdate(RegionName.TOP_LEFT, 0.65, None, False, 0.0, True)

    idle_frame = render_region_selection_screen(
        camera_frame,
        observation,
        assessment,
        (0.25, 0.25),
        idle,
        config,
        30.0,
        placement,
    )
    dwelling_frame = render_region_selection_screen(
        camera_frame,
        observation,
        assessment,
        (0.25, 0.25),
        dwelling,
        config,
        30.0,
        placement,
    )

    assert idle_frame.shape == (config.canvas_height, config.canvas_width, 3)
    assert np.count_nonzero(idle_frame != dwelling_frame) > 1_000
    assert np.count_nonzero(camera_frame) == 0
