from __future__ import annotations

import textwrap
import unicodedata
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.config import GuidedValidationConfig
from mekasa_gaze.core.domain import (
    ExperimentPhase,
    EyeState,
    FeatureObservation,
    GazeValidity,
    MeasurementStage,
    ProtocolSnapshot,
    RegionName,
    TargetName,
    TrackingState,
)
from mekasa_gaze.core.region_protocol import (
    RegionProtocolSnapshot,
    RegionTrialOutcome,
    RegionTrialStage,
)
from mekasa_gaze.tracking.interaction import SelectionUpdate, region_bounds
from mekasa_gaze.tracking.placement import PlacementAssessment, PlacementState
from mekasa_gaze.tracking.validity import ValidityAssessment

_BACKGROUND = (24, 27, 31)
_PANEL = (35, 39, 45)
_TEXT = (235, 238, 242)
_MUTED = (165, 172, 181)
_ACCENT = (245, 190, 55)
_GOOD = (80, 205, 110)
_WARN = (40, 180, 245)
_BAD = (70, 80, 225)


def render_loading_screen(
    width: int,
    height: int,
    *,
    title: str,
    status: str,
) -> npt.NDArray[np.uint8]:
    canvas = np.full((height, width, 3), _BACKGROUND, dtype=np.uint8)
    center_x = width // 2
    center_y = height // 2
    cv2.circle(canvas, (center_x, center_y - 70), 34, _PANEL, 8, cv2.LINE_AA)
    cv2.ellipse(
        canvas,
        (center_x, center_y - 70),
        (34, 34),
        0,
        -80,
        45,
        _ACCENT,
        8,
        cv2.LINE_AA,
    )
    _centered_text(canvas, title, center_y + 10, 0.9, _TEXT, 2)
    _centered_text(canvas, status, center_y + 55, 0.6, _MUTED, 1)
    _centered_text(canvas, "Procesamiento local — no se guarda video", height - 48, 0.48, _GOOD, 1)
    return canvas


def render_guided_screen(
    camera_bgr: npt.NDArray[np.uint8],
    feature_observation: FeatureObservation,
    assessment: ValidityAssessment,
    snapshot: ProtocolSnapshot,
    config: GuidedValidationConfig,
    rolling_fps: float | None,
    *,
    audio_active: bool = False,
    placement: PlacementAssessment | None = None,
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

    _draw_test_area(canvas, test_width, snapshot, config, audio_active, placement)
    _draw_debug_panel(
        canvas,
        test_width,
        camera_bgr,
        feature_observation,
        assessment,
        snapshot,
        config,
        rolling_fps,
        audio_active,
        placement,
    )
    return canvas


def render_live_evaluation_screen(
    camera_bgr: npt.NDArray[np.uint8],
    feature_observation: FeatureObservation,
    assessment: ValidityAssessment,
    prediction: tuple[float, float] | None,
    nearest_target: TargetName | None,
    config: GuidedValidationConfig,
    rolling_fps: float | None,
    placement: PlacementAssessment | None = None,
) -> npt.NDArray[np.uint8]:
    snapshot = ProtocolSnapshot(
        phase=ExperimentPhase.PREPARATION,
        stage=MeasurementStage.WAITING_TRACKING,
        target=None,
        instruction="",
        stage_remaining_seconds=None,
        protocol_elapsed_seconds=0.0,
    )
    canvas = render_guided_screen(
        camera_bgr,
        feature_observation,
        assessment,
        snapshot,
        config,
        rolling_fps,
        placement=placement,
    )
    debug_width = round(config.canvas_width * config.debug_panel_fraction)
    test_width = config.canvas_width - debug_width
    cv2.rectangle(canvas, (0, 0), (test_width - 1, config.canvas_height - 1), _BACKGROUND, -1)
    _text(canvas, "EVALUACION EN VIVO — EXPERIMENTAL", (36, 48), 0.76, _MUTED, 2)
    _text(
        canvas,
        "Mira distintos puntos. La cruz muestra la prediccion, no controla el cursor.",
        (36, 84),
        0.54,
        _TEXT,
        1,
    )
    for target in config.targets:
        point = (
            round(target.normalized_x * (test_width - 1)),
            round(target.normalized_y * (config.canvas_height - 1)),
        )
        cv2.circle(canvas, point, 9, _MUTED, 2, cv2.LINE_AA)
        _text(canvas, target.name.value, (point[0] + 13, point[1] + 5), 0.4, _MUTED, 1)

    if prediction is None:
        message = "SIN PREDICCION — FRAME NO VALIDO"
        color = _BAD
    else:
        clipped_x = min(1.0, max(0.0, prediction[0]))
        clipped_y = min(1.0, max(0.0, prediction[1]))
        point = (
            round(clipped_x * (test_width - 1)),
            round(clipped_y * (config.canvas_height - 1)),
        )
        cv2.circle(canvas, point, 25, _ACCENT, 3, cv2.LINE_AA)
        cv2.line(canvas, (point[0] - 34, point[1]), (point[0] + 34, point[1]), _ACCENT, 2)
        cv2.line(canvas, (point[0], point[1] - 34), (point[0], point[1] + 34), _ACCENT, 2)
        target_label = nearest_target.value if nearest_target is not None else "—"
        message = (
            f"PREDICCION EXPERIMENTAL: {target_label}  ({prediction[0]:.3f}, {prediction[1]:.3f})"
        )
        color = _ACCENT
    _text(canvas, message, (36, config.canvas_height - 44), 0.66, color, 2)
    _text(
        canvas,
        "Q o ESC: finalizar evaluacion",
        (test_width - 285, config.canvas_height - 16),
        0.45,
        _MUTED,
        1,
    )
    return canvas


def render_region_selection_screen(
    camera_bgr: npt.NDArray[np.uint8],
    feature_observation: FeatureObservation,
    assessment: ValidityAssessment,
    prediction: tuple[float, float] | None,
    selection: SelectionUpdate,
    config: GuidedValidationConfig,
    rolling_fps: float | None,
    placement: PlacementAssessment,
    *,
    show_region_labels: bool = True,
) -> npt.NDArray[np.uint8]:
    snapshot = ProtocolSnapshot(
        phase=ExperimentPhase.PREPARATION,
        stage=MeasurementStage.WAITING_TRACKING,
        target=None,
        instruction="",
        stage_remaining_seconds=None,
        protocol_elapsed_seconds=0.0,
    )
    canvas = render_guided_screen(
        camera_bgr,
        feature_observation,
        assessment,
        snapshot,
        config,
        rolling_fps,
        placement=placement,
    )
    debug_width = round(config.canvas_width * config.debug_panel_fraction)
    test_width = config.canvas_width - debug_width
    cv2.rectangle(canvas, (0, 0), (test_width - 1, config.canvas_height - 1), _BACKGROUND, -1)

    labels = {
        RegionName.TOP_LEFT: "REGION 1",
        RegionName.TOP_RIGHT: "REGION 2",
        RegionName.BOTTOM_LEFT: "REGION 3",
        RegionName.BOTTOM_RIGHT: "REGION 4",
    }
    header_height = 72
    footer_height = 64
    region_height = config.canvas_height - header_height - footer_height
    for region in RegionName:
        left, top, right, bottom = region_bounds(region)
        x1 = round(left * test_width)
        x2 = round(right * test_width) - 1
        y1 = header_height + round(top * region_height)
        y2 = header_height + round(bottom * region_height) - 1
        active = selection.candidate is region
        selected = selection.selected is region
        fill = (42, 57, 64) if active else _PANEL
        if selected:
            fill = (38, 78, 50)
        border = _GOOD if selected else _ACCENT if active else _MUTED
        cv2.rectangle(canvas, (x1 + 8, y1 + 8), (x2 - 8, y2 - 8), fill, -1)
        cv2.rectangle(canvas, (x1 + 8, y1 + 8), (x2 - 8, y2 - 8), border, 3)
        if show_region_labels:
            label = labels[region]
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2)[0]
            _text(
                canvas,
                label,
                ((x1 + x2 - size[0]) // 2, (y1 + y2) // 2),
                0.82,
                border,
                2,
            )

    _text(canvas, "EXPERIMENTO 003 — SELECCION POR PERMANENCIA", (24, 42), 0.68, _TEXT, 2)
    if prediction is not None:
        x = round(min(1.0, max(0.0, prediction[0])) * (test_width - 1))
        y = header_height + round(min(1.0, max(0.0, prediction[1])) * max(1, region_height - 1))
        cv2.circle(canvas, (x, y), 14, _ACCENT, 2, cv2.LINE_AA)

    if not placement.ready:
        status = placement.instruction
        color = _WARN
    elif not selection.tracking_available:
        status = "SIN SELECCION — MIRADA NO DISPONIBLE"
        color = _BAD
    elif selection.cooldown_remaining_seconds > 0.0:
        status = f"SELECCION REGISTRADA — pausa {selection.cooldown_remaining_seconds:.1f} s"
        color = _GOOD
    elif selection.candidate is not None:
        status = f"MANTENE LA MIRADA — {selection.dwell_progress * 100:.0f}%"
        color = _ACCENT
    else:
        status = "MIRA UNA REGION PARA INICIAR"
        color = _TEXT
    _text(canvas, status, (24, config.canvas_height - 26), 0.56, color, 2)
    progress_width = round((test_width - 48) * selection.dwell_progress)
    cv2.rectangle(
        canvas,
        (24, config.canvas_height - 15),
        (24 + progress_width, config.canvas_height - 8),
        _ACCENT,
        -1,
    )
    return canvas


def render_controlled_region_screen(
    camera_bgr: npt.NDArray[np.uint8],
    feature_observation: FeatureObservation,
    assessment: ValidityAssessment,
    prediction: tuple[float, float] | None,
    selection: SelectionUpdate,
    protocol: RegionProtocolSnapshot,
    config: GuidedValidationConfig,
    rolling_fps: float | None,
    placement: PlacementAssessment,
) -> npt.NDArray[np.uint8]:
    canvas = render_region_selection_screen(
        camera_bgr,
        feature_observation,
        assessment,
        prediction,
        selection,
        config,
        rolling_fps,
        placement,
        show_region_labels=False,
    )
    debug_width = round(config.canvas_width * config.debug_panel_fraction)
    test_width = config.canvas_width - debug_width
    header_height = 72
    footer_height = 64
    region_height = config.canvas_height - header_height - footer_height
    cv2.rectangle(canvas, (0, 0), (test_width - 1, header_height - 1), _BACKGROUND, -1)
    cv2.rectangle(
        canvas,
        (0, config.canvas_height - footer_height),
        (test_width - 1, config.canvas_height - 1),
        _BACKGROUND,
        -1,
    )
    _text(canvas, "EXPERIMENTO 003 — VALIDACION CONTROLADA", (24, 40), 0.7, _TEXT, 2)

    if protocol.requested is not None and protocol.stage in (
        RegionTrialStage.CUE,
        RegionTrialStage.RESPONSE,
    ):
        left, top, right, bottom = region_bounds(protocol.requested)
        center = (
            round((left + right) * 0.5 * test_width),
            header_height + round((top + bottom) * 0.5 * region_height),
        )
        cv2.circle(canvas, center, 34, (245, 245, 245), 4, cv2.LINE_AA)
        cv2.circle(canvas, center, 8, (245, 245, 245), -1, cv2.LINE_AA)

    status, color = _controlled_region_status(protocol, selection, placement)
    _text(canvas, status, (24, config.canvas_height - 30), 0.58, color, 2)
    if protocol.stage_remaining_seconds is not None:
        _text(
            canvas,
            f"{protocol.stage_remaining_seconds:.1f} s",
            (test_width - 105, config.canvas_height - 30),
            0.55,
            _MUTED,
            1,
        )
    return canvas


def _controlled_region_status(
    protocol: RegionProtocolSnapshot,
    selection: SelectionUpdate,
    placement: PlacementAssessment,
) -> tuple[str, tuple[int, int, int]]:
    if protocol.stage is RegionTrialStage.PREPARATION:
        return placement.instruction, _GOOD if placement.ready else _WARN
    if protocol.stage is RegionTrialStage.COUNTDOWN:
        return "PREPARATE — LA PRUEBA ESTA POR COMENZAR", _ACCENT
    if protocol.stage is RegionTrialStage.CUE:
        if protocol.is_practice:
            return "PRACTICA — ESCUCHA Y LUEGO MIRA EL PUNTO", _ACCENT
        return "ESCUCHA LA REGION INDICADA", _TEXT
    if protocol.stage is RegionTrialStage.RESPONSE:
        if not placement.ready:
            return placement.instruction, _WARN
        if not selection.tracking_available:
            return "MIRADA NO DISPONIBLE — EL PROGRESO SE REINICIO", _BAD
        if protocol.is_practice:
            return "PRACTICA — MANTENE LA MIRADA EN EL PUNTO", _ACCENT
        if selection.candidate is not None:
            return f"MANTENE LA MIRADA — {selection.dwell_progress * 100:.0f}%", _ACCENT
        return "MIRA LA REGION INDICADA", _TEXT
    if protocol.stage is RegionTrialStage.FEEDBACK and protocol.completed_trial is not None:
        labels = {
            RegionTrialOutcome.PRACTICE: ("PRACTICA REGISTRADA", _ACCENT),
            RegionTrialOutcome.CORRECT: ("CORRECTO", _GOOD),
            RegionTrialOutcome.INCORRECT: ("REGION INCORRECTA", _BAD),
            RegionTrialOutcome.NO_SELECTION: ("SIN SELECCION", _WARN),
        }
        return labels[protocol.completed_trial.outcome]
    if protocol.stage is RegionTrialStage.COMPLETE:
        return "PRUEBA FINALIZADA — PODES DESCANSAR", _GOOD
    return "PREPARANDO SIGUIENTE REGION", _MUTED


def _draw_test_area(
    canvas: npt.NDArray[np.uint8],
    test_width: int,
    snapshot: ProtocolSnapshot,
    config: GuidedValidationConfig,
    audio_active: bool,
    placement: PlacementAssessment | None,
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
    hide_instruction = audio_active and config.hide_visual_instructions_when_audio
    if not hide_instruction:
        instruction_lines = textwrap.wrap(_ascii(snapshot.instruction), width=58) or [""]
        for index, line in enumerate(instruction_lines[:3]):
            _text(canvas, line, (36, 130 + index * 34), 0.72, _TEXT, 2)

    if snapshot.stage is MeasurementStage.WAITING_TRACKING and placement is not None:
        color = _GOOD if placement.ready else _WARN
        _centered_text(
            canvas[:, :test_width],
            placement.instruction,
            config.canvas_height // 2,
            1.05,
            color,
            3,
        )
        if placement.face_width is not None:
            _centered_text(
                canvas[:, :test_width],
                f"Tamano detectado: {placement.face_width:.3f}",
                config.canvas_height // 2 + 48,
                0.55,
                _MUTED,
                1,
            )

    if snapshot.target is not None:
        target_x = round(snapshot.target.normalized_x * (test_width - 1))
        target_y = round(snapshot.target.normalized_y * (config.canvas_height - 1))
        color = _target_color(snapshot.stage)
        radius = config.target_radius_pixels
        cv2.circle(canvas, (target_x, target_y), radius + 12, color, 3, cv2.LINE_AA)
        cv2.circle(canvas, (target_x, target_y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (target_x, target_y), 7, _BACKGROUND, -1, cv2.LINE_AA)
        if not audio_active:
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
    audio_active: bool,
    placement: PlacementAssessment | None,
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
    placement_value = placement.state.value if placement is not None else "—"
    placement_color = (
        _GOOD if placement is not None and placement.state is PlacementState.READY else _WARN
    )
    rows = (
        ("Tracking", observation.tracking_state.value, _tracking_color(observation.tracking_state)),
        ("Distancia", placement_value, placement_color),
        ("Gaze frame", assessment.gaze_validity.value, _validity_color(assessment.gaze_validity)),
        ("Ojo izquierdo", assessment.left_eye_state.value, _eye_color(assessment.left_eye_state)),
        ("Ojo derecho", assessment.right_eye_state.value, _eye_color(assessment.right_eye_state)),
        ("FPS", f"{rolling_fps:.1f}" if rolling_fps is not None else "—", _TEXT),
        ("Fase", snapshot.phase.value, _TEXT),
        ("Bloque", str(snapshot.block_index) if snapshot.block_index else "—", _TEXT),
        ("Target", snapshot.target.name.value if snapshot.target else "—", _TEXT),
        ("Audio", "ACTIVO" if audio_active else "INACTIVO", _GOOD if audio_active else _WARN),
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
    if state is GazeValidity.VALID:
        return _GOOD
    if state in (GazeValidity.PARTIAL, GazeValidity.BLINK):
        return _WARN
    return _BAD


def _eye_color(state: EyeState) -> tuple[int, int, int]:
    return _GOOD if state is EyeState.OPEN else _WARN if state is EyeState.CLOSED else _BAD


def _centered_text(
    canvas: npt.NDArray[np.uint8],
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    ascii_text = _ascii(text)
    width = cv2.getTextSize(ascii_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    _text(canvas, ascii_text, ((canvas.shape[1] - width) // 2, y), scale, color, thickness)


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


__all__ = [
    "render_controlled_region_screen",
    "render_guided_screen",
    "render_live_evaluation_screen",
    "render_loading_screen",
    "render_region_selection_screen",
]
