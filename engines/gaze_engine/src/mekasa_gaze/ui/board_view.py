"""High-contrast renderer for the communication board and its quick alignment step.

Both information bands sit at the top: a status line (what the system is doing, why it is not
selecting, which eyes are usable, alignment and distance) and a message bar that keeps the last
confirmed phrase visible even if speech fails. The board fills the rest of the screen down to
the bottom edge, which maximizes the vertical angle between the two rows (vertical gaze is the
weakest axis with a webcam above the screen). Gaze is mapped into the board area only, so
looking at the bands never selects anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.board import BoardLayout, BoardOption
from mekasa_gaze.core.eye_mode import EyeMode
from mekasa_gaze.tracking.communication import AbstainReason, SessionState, SessionUpdate
from mekasa_gaze.tracking.session_alignment import AlignmentSnapshot
from mekasa_gaze.tracking.session_calibration import CalibrationSnapshot
from mekasa_gaze.ui.text import draw_text
from mekasa_gaze.ui.visualization import _filled_rounded_rectangle

STATUS_FRACTION = 0.07
MESSAGE_FRACTION = 0.11
_MARGIN = 0.02
_BOARD_TOP = STATUS_FRACTION + MESSAGE_FRACTION + 0.015

Color = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Palette:
    """BGR colours. The default is high contrast (white on near-black, yellow focus)."""

    background: Color
    card: Color
    card_text: Color
    focus: Color
    progress: Color
    confirmed: Color
    pending: Color
    status_text: Color
    muted: Color
    warning: Color
    good: Color
    message_background: Color
    message_text: Color


HIGH_CONTRAST = Palette(
    background=(10, 10, 10),
    card=(70, 38, 20),
    card_text=(255, 255, 255),
    focus=(0, 220, 255),
    progress=(0, 120, 170),
    confirmed=(70, 175, 50),
    pending=(255, 160, 40),
    status_text=(255, 255, 255),
    muted=(150, 150, 150),
    warning=(60, 140, 255),
    good=(90, 210, 90),
    message_background=(35, 35, 35),
    message_text=(255, 255, 255),
)

SOFT = Palette(
    background=(36, 34, 29),
    card=(65, 62, 52),
    card_text=(239, 241, 239),
    focus=(116, 185, 199),
    progress=(80, 130, 140),
    confirmed=(119, 191, 142),
    pending=(200, 160, 110),
    status_text=(239, 241, 239),
    muted=(158, 171, 171),
    warning=(92, 184, 232),
    good=(142, 191, 119),
    message_background=(43, 50, 53),
    message_text=(239, 241, 239),
)


def board_area(width: int, height: int) -> tuple[float, float, float, float]:
    """Normalized screen rectangle (left, top, right, bottom) occupied by the board."""
    del width, height
    return (_MARGIN, _BOARD_TOP, 1.0 - _MARGIN, 1.0 - _MARGIN)


def screen_to_board(
    point: tuple[float, float] | None, area: tuple[float, float, float, float]
) -> tuple[float, float] | None:
    """Map a full-screen normalized gaze point into board-normalized coordinates."""
    if point is None:
        return None
    left, top, right, bottom = area
    return ((point[0] - left) / (right - left), (point[1] - top) / (bottom - top))


def board_to_screen(
    point: tuple[float, float], area: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Inverse of :func:`screen_to_board` (used for pointer clicks and alignment points)."""
    left, top, right, bottom = area
    return (left + point[0] * (right - left), top + point[1] * (bottom - top))


def option_centers(
    layout: BoardLayout, area: tuple[float, float, float, float]
) -> list[tuple[float, float]]:
    """Screen-normalized centre of every option (alignment targets)."""
    return [
        board_to_screen(
            ((option.bounds[0] + option.bounds[2]) / 2, (option.bounds[1] + option.bounds[3]) / 2),
            area,
        )
        for option in layout.options
    ]


_ABSTAIN_TEXT = {
    AbstainReason.NO_FRAME: "Cámara sin imagen: revise la conexión",
    AbstainReason.STALE_FRAME: "Imagen demorada: selección detenida",
    AbstainReason.TRACKING: "No se ve el rostro de forma estable",
    AbstainReason.EYE_SIGNAL: "El ojo configurado no se ve bien",
    AbstainReason.NO_PREDICTION: "Mirada fuera de la pantalla",
}

_MODE_TEXT = {
    EyeMode.BOTH: "Ambos ojos",
    EyeMode.LEFT: "Ojo izquierdo",
    EyeMode.RIGHT: "Ojo derecho",
}


def status_text(
    update: SessionUpdate,
    eye_mode: EyeMode,
    hint: str | None = None,
    notice: str | None = None,
) -> str:
    """Short Spanish status line for the person and their support.

    ``hint`` explains a tracking loss (e.g. move closer); ``notice`` replaces the idle "ready"
    line (e.g. a placement change since calibration).
    """
    state = update.state
    if state is SessionState.NO_CALIBRATION:
        return "Sin calibración para este modo · use las teclas 1 a 4"
    if state is SessionState.PAUSED:
        return "En pausa · ESPACIO para continuar"
    if state is SessionState.SIGNAL_LOST:
        reason = update.abstain_reason or AbstainReason.TRACKING
        if reason is AbstainReason.TRACKING and hint:
            return hint
        return _ABSTAIN_TEXT.get(reason, "Sin señal")
    if state is SessionState.RECOVERING:
        return "Recuperando señal…"
    if state is SessionState.FOCUSING and update.focused is not None:
        return f"Mirando {update.focused.label}…"
    if state is SessionState.AWAITING_CONFIRMATION and update.pending is not None:
        return f"Para confirmar {update.pending.label}: mire afuera y vuelva"
    if state is SessionState.COOLDOWN:
        return "Confirmado"
    if state is SessionState.AWAITING_EXIT:
        return "Confirmado · aparte la mirada para elegir otra vez"
    if notice:
        return notice
    return f"Listo · {_MODE_TEXT[eye_mode]}: mire una opción"


def render_board(
    width: int,
    height: int,
    layout: BoardLayout,
    update: SessionUpdate,
    *,
    eye_mode: EyeMode,
    high_contrast: bool = True,
    gaze_point: tuple[float, float] | None = None,
    speech_ok: bool = True,
    flash: BoardOption | None = None,
    eyes: tuple[bool, bool] | None = None,
    hint: str | None = None,
    aligned: bool | None = None,
    notice: str | None = None,
    prompt: str | None = None,
) -> npt.NDArray[np.uint8]:
    """Render one frame of the board.

    Args:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        layout: Board layout (board-normalized coordinates).
        update: Latest session update.
        eye_mode: Active ocular mode.
        high_contrast: Use the high-contrast palette.
        gaze_point: Optional screen-normalized gaze point (setup aid only).
        speech_ok: False shows that the last phrase could not be spoken.
        flash: Option to highlight as just confirmed.
        eyes: (left usable, right usable) for the eye indicators, or None if unknown.
        hint: Placement instruction (e.g. move closer) shown when tracking is the issue.
        aligned: Whether a session alignment is active (None hides the indicator).
        notice: Replaces the idle status line (placement change since calibration, ...).
        prompt: Test-mode instruction shown in the message bar instead of the message.
    """
    palette = HIGH_CONTRAST if high_contrast else SOFT
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = palette.background
    area = board_area(width, height)
    status_h = int(height * STATUS_FRACTION)
    margin_px = int(width * _MARGIN)

    lost = update.state in (
        SessionState.SIGNAL_LOST,
        SessionState.NO_CALIBRATION,
        SessionState.PAUSED,
    )
    indicators_w = int(width * 0.30)
    draw_text(
        canvas,
        status_text(update, eye_mode, hint, notice),
        (margin_px, 0, width - indicators_w, status_h),
        palette.warning
        if (lost or (notice and update.state is SessionState.READY))
        else palette.status_text,
        max_size=max(14, status_h // 2),
        bold=False,
        align="left",
    )
    _draw_indicators(canvas, palette, eye_mode, eyes, aligned, width - indicators_w, status_h)
    if update.state is SessionState.RECOVERING:
        bar_w = int((width - 2 * margin_px) * min(1.0, update.recovery_progress))
        cv2.rectangle(
            canvas, (margin_px, status_h - 6), (margin_px + bar_w, status_h - 2), palette.good, -1
        )

    for option in layout.options:
        _draw_option(canvas, option, update, area, palette, flash)

    message_top = status_h
    message_bottom = status_h + int(height * MESSAGE_FRACTION)
    cv2.rectangle(canvas, (0, message_top), (width, message_bottom), palette.message_background, -1)
    text = prompt if prompt else update.message
    if text:
        draw_text(
            canvas,
            text,
            (margin_px, message_top, width - margin_px, message_bottom),
            palette.focus if prompt else palette.message_text,
            max_size=int(height * MESSAGE_FRACTION * 0.65),
        )
    if not speech_ok and not prompt:
        draw_text(
            canvas,
            "sin voz",
            (width - int(width * 0.14), message_top, width - margin_px, message_bottom),
            palette.warning,
            max_size=24,
            bold=False,
        )

    if gaze_point is not None:
        x = int(gaze_point[0] * width)
        y = int(gaze_point[1] * height)
        cv2.circle(canvas, (x, y), 16, palette.focus, 3, cv2.LINE_AA)
    return canvas


def render_alignment(
    width: int,
    height: int,
    snapshot: AlignmentSnapshot,
    *,
    high_contrast: bool = True,
    signal_ok: bool = True,
    reason: str | None = None,
) -> npt.NDArray[np.uint8]:
    """Quick-alignment screen: one point at a time, a ring that fills while collecting."""
    title = f"Ajuste rápido · mire el punto ({min(snapshot.index + 1, snapshot.total)}/"
    title += f"{snapshot.total})"
    if reason:
        title = f"{reason} · {title}"
    if not signal_ok:
        title = "Ajuste rápido · no se ven bien los ojos"
    return _target_screen(
        width,
        height,
        title,
        snapshot.target,
        snapshot.collecting,
        snapshot.progress,
        high_contrast=high_contrast,
        warning=not signal_ok,
        footer="S = saltar ajuste · ESC = salir",
    )


def render_calibration(
    width: int,
    height: int,
    snapshot: CalibrationSnapshot,
    *,
    high_contrast: bool = True,
    signal_ok: bool = True,
) -> npt.NDArray[np.uint8]:
    """Calibration screen (~35 s): grid points, then head movement on the centre point."""
    seconds = round(snapshot.seconds_left)
    if snapshot.head_movement:
        title = "Siga mirando el punto y mueva despacio la cabeza (sí y no)"
    else:
        title = f"Calibración · mire el punto · quedan {seconds} s"
    if not signal_ok:
        title = "Calibración · no se ven bien los ojos: acérquese o mejore la luz"
    return _target_screen(
        width,
        height,
        title,
        snapshot.target,
        snapshot.collecting,
        snapshot.progress,
        high_contrast=high_contrast,
        warning=not signal_ok,
        footer=f"Punto {min(snapshot.index + 1, snapshot.total)} de {snapshot.total} · ESC = salir",
    )


def render_summary(
    width: int,
    height: int,
    title: str,
    lines: list[tuple[str, bool | None]],
    *,
    high_contrast: bool = True,
    footer: str = "",
) -> npt.NDArray[np.uint8]:
    """Full-screen summary (calibration quality, test results). ``lines``: (text, good?)."""
    palette = HIGH_CONTRAST if high_contrast else SOFT
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = palette.background
    margin = int(width * 0.06)
    draw_text(
        canvas,
        title,
        (margin, int(height * 0.06), width - margin, int(height * 0.2)),
        palette.status_text,
        max_size=int(height * 0.08),
    )
    row_h = int(height * 0.09)
    top = int(height * 0.25)
    for index, (text, good) in enumerate(lines):
        color = palette.status_text if good is None else palette.good if good else palette.warning
        draw_text(
            canvas,
            text,
            (margin, top + index * row_h, width - margin, top + (index + 1) * row_h),
            color,
            max_size=int(row_h * 0.55),
            bold=False,
            align="left",
        )
    if footer:
        draw_text(
            canvas,
            footer,
            (0, int(height * 0.88), width, height),
            palette.muted,
            max_size=int(height * 0.04),
            bold=False,
        )
    return canvas


def _target_screen(
    width: int,
    height: int,
    title: str,
    target: tuple[float, float] | None,
    collecting: bool,
    progress: float,
    *,
    high_contrast: bool,
    warning: bool,
    footer: str,
) -> npt.NDArray[np.uint8]:
    palette = HIGH_CONTRAST if high_contrast else SOFT
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = palette.background
    status_h = int(height * STATUS_FRACTION)
    draw_text(
        canvas,
        title,
        (int(width * _MARGIN), 0, width - int(width * _MARGIN), status_h),
        palette.warning if warning else palette.status_text,
        max_size=max(14, int(status_h * 0.55)),
        bold=False,
        align="left",
    )
    if target is not None:
        x, y = int(target[0] * width), int(target[1] * height)
        radius = max(14, min(width, height) // 36)
        cv2.circle(canvas, (x, y), radius * 3, palette.card, -1, cv2.LINE_AA)
        if collecting:
            angle = int(360 * progress)
            cv2.ellipse(
                canvas,
                (x, y),
                (radius * 2, radius * 2),
                -90,
                0,
                angle,
                palette.good,
                8,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(canvas, (x, y), radius * 2, palette.focus, 4, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius, palette.card_text, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), max(4, radius // 3), palette.background, -1, cv2.LINE_AA)
    draw_text(
        canvas,
        footer,
        (0, height - status_h, width, height),
        palette.muted,
        max_size=max(12, status_h // 3),
        bold=False,
    )
    return canvas


def _draw_indicators(
    canvas: npt.NDArray[np.uint8],
    palette: Palette,
    eye_mode: EyeMode,
    eyes: tuple[bool, bool] | None,
    aligned: bool | None,
    left: int,
    height: int,
) -> None:
    width = canvas.shape[1]
    radius = max(6, height // 6)
    y = height // 2
    labels = []
    if eyes is not None:
        labels = [
            ("Izq", eyes[0], eye_mode is not EyeMode.RIGHT),
            ("Der", eyes[1], eye_mode is not EyeMode.LEFT),
        ]
    x = left
    slot = (width - left) // 3
    for label, usable, required in labels:
        color = palette.good if usable else (palette.warning if required else palette.muted)
        cv2.circle(canvas, (x + radius + 4, y), radius, color, -1 if usable else 3, cv2.LINE_AA)
        draw_text(
            canvas,
            label,
            (x + 2 * radius + 10, 0, x + slot, height),
            palette.muted,
            max_size=int(height * 0.42),
            bold=False,
            align="left",
        )
        x += slot
    if aligned is not None:
        draw_text(
            canvas,
            "Ajustado" if aligned else "Sin ajuste",
            (x, 0, width - 8, height),
            palette.good if aligned else palette.muted,
            max_size=int(height * 0.42),
            bold=False,
            align="left",
        )


def _draw_option(
    canvas: npt.NDArray[np.uint8],
    option: BoardOption,
    update: SessionUpdate,
    area: tuple[float, float, float, float],
    palette: Palette,
    flash: BoardOption | None,
) -> None:
    height, width = canvas.shape[:2]
    left, top = board_to_screen((option.bounds[0], option.bounds[1]), area)
    right, bottom = board_to_screen((option.bounds[2], option.bounds[3]), area)
    box = (int(left * width), int(top * height), int(right * width), int(bottom * height))
    focused = update.focused is not None and update.focused.key == option.key
    pending = update.pending is not None and update.pending.key == option.key
    confirmed = (update.confirmed is not None and update.confirmed.key == option.key) or (
        flash is not None and flash.key == option.key
    )
    border = palette.card
    border_width = 4
    fill = palette.card
    if confirmed:
        border, border_width, fill = palette.confirmed, 14, palette.confirmed
    elif pending:
        border, border_width = palette.pending, 12
    elif focused:
        border, border_width = palette.focus, 10
    _filled_rounded_rectangle(canvas, (box[0], box[1]), (box[2], box[3]), border, 28)
    inner = (
        box[0] + border_width,
        box[1] + border_width,
        box[2] - border_width,
        box[3] - border_width,
    )
    _filled_rounded_rectangle(canvas, (inner[0], inner[1]), (inner[2], inner[3]), fill, 22)
    if focused and update.progress > 0.0 and not confirmed:
        fill_height = int((inner[3] - inner[1]) * min(1.0, update.progress))
        _filled_rounded_rectangle(
            canvas,
            (inner[0], inner[3] - fill_height),
            (inner[2], inner[3]),
            palette.progress,
            min(22, max(1, fill_height // 2)),
        )
    label_box = (inner[0], inner[1], inner[2], inner[3])
    draw_text(
        canvas,
        option.label,
        label_box,
        palette.card_text,
        max_size=int((inner[3] - inner[1]) * 0.5),
        fill=0.75,
    )
    if pending:
        draw_text(
            canvas,
            "mire otra vez para confirmar",
            (inner[0], inner[3] - (inner[3] - inner[1]) // 5, inner[2], inner[3]),
            palette.pending,
            max_size=28,
            bold=False,
        )


__all__ = [
    "HIGH_CONTRAST",
    "SOFT",
    "board_area",
    "board_to_screen",
    "option_centers",
    "render_alignment",
    "render_board",
    "render_calibration",
    "render_summary",
    "screen_to_board",
    "status_text",
]
