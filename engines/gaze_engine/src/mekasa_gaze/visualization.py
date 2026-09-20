from __future__ import annotations

import textwrap
import unicodedata
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from mekasa_gaze.config import GuidedValidationConfig
from mekasa_gaze.domain import (
    EyeState,
    FeatureObservation,
    GazeValidity,
    MeasurementStage,
    ProtocolSnapshot,
    TrackingState,
)
from mekasa_gaze.validity import ValidityAssessment

_BACKGROUND = (24, 27, 31)
_PANEL = (35, 39, 45)
_TEXT = (235, 238, 242)
_MUTED = (165, 172, 181)
_ACCENT = (245, 190, 55)
_GOOD = (80, 205, 110)
_WARN = (40, 180, 245)
_BAD = (70, 80, 225)


def render_guided_screen(
    camera_bgr: npt.NDArray[np.uint8],
    feature_observation: FeatureObservation,
    assessment: ValidityAssessment,
    snapshot: ProtocolSnapshot,
    config: GuidedValidationConfig,
    rolling_fps: float | None,
) -> npt.NDArray[np.uint8]:
    canvas = np.full(
        (config.canvas_height, config.canvas_width, 3),
        _BACKGROUND,
        dtype=np.uint8,
    )
    debug_width = round(config.canvas_width * config.debug_panel_fraction)
    test_width = config.canvas_width - debug_width
    cv2.rectangle(
        canvas,
        (test_width, 0),
        (config.canvas_width - 1, config.canvas_height - 1),
        _PANEL,
        -1,
    )
    cv2.line(canvas, (test_width, 0), (test_width, config.canvas_height), _MUTED, 1)

    _draw_test_area(canvas, test_width, snapshot, config)
    _draw_debug_panel(
        canvas,
        test_width,
        camera_bgr,
        feature_observation,
        assessment,
        snapshot,
        config,
        rolling_fps,
    )
    return canvas


def _draw_test_area(
    canvas: npt.NDArray[np.uint8],
    test_width: int,
    snapshot: ProtocolSnapshot,
    config: GuidedValidationConfig,
) -> None:
    _text(canvas, "MEKASA — GUIDED VISUAL VALIDATION", (36, 46), 0.75, _MUTED, 2)
    _text(
        canvas,
        f"FASE: {snapshot.phase.value}  |  ESTADO: {snapshot.stage.value}",
        (36, 82),
        0.58,
        _TEXT,
        1,
    )
    instruction_lines = textwrap.wrap(_ascii(snapshot.instruction), width=58) or [""]
    for index, line in enumerate(instruction_lines[:3]):
        _text(canvas, line, (36, 130 + index * 34), 0.72, _TEXT, 2)

    if snapshot.target is not None:
        target_x = round(snapshot.target.normalized_x * (test_width - 1))
        target_y = round(snapshot.target.normalized_y * (config.canvas_height - 1))
        color = _target_color(snapshot.stage)
        radius = config.target_radius_pixels
        cv2.circle(canvas, (target_x, target_y), radius + 12, color, 3, cv2.LINE_AA)
        cv2.circle(canvas, (target_x, target_y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (target_x, target_y), 7, _BACKGROUND, -1, cv2.LINE_AA)
        label = snapshot.target.name.value
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
        _text(
            canvas,
            label,
            (target_x - label_size[0] // 2, target_y + radius + 42),
            0.72,
            color,
            2,
        )

    if snapshot.stage_remaining_seconds is not None:
        remaining = max(0.0, snapshot.stage_remaining_seconds)
        if snapshot.stage is MeasurementStage.COUNTDOWN:
            countdown = str(max(1, int(np.ceil(remaining))))
            size = cv2.getTextSize(countdown, cv2.FONT_HERSHEY_SIMPLEX, 3.2, 7)[0]
            _text(
                canvas,
                countdown,
                ((test_width - size[0]) // 2, config.canvas_height // 2 + 70),
                3.2,
                _ACCENT,
                7,
            )
        else:
            _text(
                canvas,
                f"Etapa: {remaining:0.1f} s",
                (36, config.canvas_height - 42),
                0.65,
                _MUTED,
                1,
            )
            if snapshot.phase_remaining_seconds is not None:
                _text(
                    canvas,
                    f"Fase: {max(0.0, snapshot.phase_remaining_seconds):0.1f} s",
                    (260, config.canvas_height - 42),
                    0.65,
                    _MUTED,
                    1,
                )

    if snapshot.completed:
        cv2.rectangle(canvas, (0, 0), (test_width - 1, config.canvas_height - 1), _BACKGROUND, -1)
        message = "PRUEBA FINALIZADA — PODES DESCANSAR"
        size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        _text(
            canvas,
            message,
            ((test_width - size[0]) // 2, config.canvas_height // 2),
            0.9,
            _GOOD,
            2,
        )

    _text(
        canvas,
        "Q o ESC: cancelar la prueba",
        (test_width - 260, config.canvas_height - 16),
        0.45,
        _MUTED,
        1,
    )


def _draw_debug_panel(
    canvas: npt.NDArray[np.uint8],
    panel_x: int,
    camera_bgr: npt.NDArray[np.uint8],
    observation: FeatureObservation,
    assessment: ValidityAssessment,
    snapshot: ProtocolSnapshot,
    config: GuidedValidationConfig,
    rolling_fps: float | None,
) -> None:
    margin = 16
    panel_width = canvas.shape[1] - panel_x
    _text(canvas, "DEBUG CAMERA — LOCAL", (panel_x + margin, 30), 0.58, _TEXT, 2)
    preview_x = panel_x + margin
    preview_y = 48
    preview_width = panel_width - margin * 2
    preview_height = min(round(preview_width * 0.75), round(canvas.shape[0] * 0.38))
    preview = cast(
        npt.NDArray[np.uint8],
        cv2.resize(camera_bgr, (preview_width, preview_height)),
    )
    _draw_preview_geometry(preview, observation, config)
    canvas[
        preview_y : preview_y + preview_height,
        preview_x : preview_x + preview_width,
    ] = preview
    cv2.rectangle(
        canvas,
        (preview_x, preview_y),
        (preview_x + preview_width, preview_y + preview_height),
        _MUTED,
        1,
    )

    y = preview_y + preview_height + 30
    rows = (
        ("Tracking", observation.tracking_state.value, _tracking_color(observation.tracking_state)),
        ("Gaze frame", assessment.gaze_validity.value, _validity_color(assessment.gaze_validity)),
        ("Ojo izquierdo", assessment.left_eye_state.value, _eye_color(assessment.left_eye_state)),
        ("Ojo derecho", assessment.right_eye_state.value, _eye_color(assessment.right_eye_state)),
        ("FPS", f"{rolling_fps:.1f}" if rolling_fps is not None else "—", _TEXT),
        ("Target", snapshot.target.name.value if snapshot.target else "—", _TEXT),
    )
    for label, value, color in rows:
        _text(canvas, f"{label}: {value}", (panel_x + margin, y), 0.48, color, 1)
        y += 24

    if observation.head_pose is None:
        pose_text = ("Yaw: —", "Pitch: —", "Roll: —")
    else:
        pose = observation.head_pose
        pose_text = (
            f"Yaw: {pose.yaw_degrees:+.1f} deg",
            f"Pitch: {pose.pitch_degrees:+.1f} deg",
            f"Roll: {pose.roll_degrees:+.1f} deg",
        )
    for text in pose_text:
        _text(canvas, text, (panel_x + margin, y), 0.48, _TEXT, 1)
        y += 24

    _text(canvas, "Iris izq.", (panel_x + margin, y + 6), 0.43, (255, 210, 50), 1)
    _text(canvas, "Iris der.", (panel_x + 112, y + 6), 0.43, (255, 100, 220), 1)
    _text(canvas, "NO SE GUARDA VIDEO", (panel_x + margin, canvas.shape[0] - 22), 0.5, _GOOD, 2)


def _draw_preview_geometry(
    preview: npt.NDArray[np.uint8],
    observation: FeatureObservation,
    config: GuidedValidationConfig,
) -> None:
    height, width = preview.shape[:2]
    if config.show_face_box and observation.debug_face_box is not None:
        x1, y1, x2, y2 = observation.debug_face_box
        cv2.rectangle(
            preview,
            (round(x1 * width), round(y1 * height)),
            (round(x2 * width), round(y2 * height)),
            _GOOD,
            2,
        )
    if not config.show_landmarks:
        return
    for point in observation.debug_eye_points:
        cv2.circle(preview, _pixel(point, width, height), 2, _ACCENT, -1, cv2.LINE_AA)
    for point in observation.debug_left_iris_points:
        cv2.circle(preview, _pixel(point, width, height), 3, (255, 210, 50), -1, cv2.LINE_AA)
    for point in observation.debug_right_iris_points:
        cv2.circle(preview, _pixel(point, width, height), 3, (255, 100, 220), -1, cv2.LINE_AA)


def _pixel(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    return round(point[0] * width), round(point[1] * height)


def _target_color(stage: MeasurementStage) -> tuple[int, int, int]:
    return {
        MeasurementStage.TRANSITION: _MUTED,
        MeasurementStage.ACQUISITION: _ACCENT,
        MeasurementStage.MEASUREMENT: (245, 245, 245),
    }.get(stage, _MUTED)


def _tracking_color(state: TrackingState) -> tuple[int, int, int]:
    return {
        TrackingState.GOOD: _GOOD,
        TrackingState.UNCERTAIN: _WARN,
        TrackingState.LOST: _BAD,
    }[state]


def _validity_color(state: GazeValidity) -> tuple[int, int, int]:
    return _GOOD if state is GazeValidity.VALID else _WARN if state is GazeValidity.BLINK else _BAD


def _eye_color(state: EyeState) -> tuple[int, int, int]:
    return _GOOD if state is EyeState.OPEN else _WARN if state is EyeState.CLOSED else _BAD


def _text(
    canvas: npt.NDArray[np.uint8],
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(
        canvas,
        _ascii(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _ascii(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
