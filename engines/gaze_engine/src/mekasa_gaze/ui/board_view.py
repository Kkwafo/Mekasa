"""High-contrast renderer for the communication board.

The screen has three bands: a status bar (what the system is doing and why it is not
selecting), the board, and a message bar that keeps the last confirmed phrase visible even if
speech fails. Gaze coordinates are mapped into the board band, so looking at the status or
message bar never selects anything.

OpenCV's Hershey fonts only cover ASCII, so accents are transliterated on screen (the spoken
phrase keeps them). A Unicode text renderer is a tracked follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.board import BoardLayout, BoardOption
from mekasa_gaze.core.eye_mode import EyeMode
from mekasa_gaze.tracking.communication import AbstainReason, SessionState, SessionUpdate
from mekasa_gaze.ui.visualization import _ascii, _filled_rounded_rectangle

STATUS_FRACTION = 0.09
MESSAGE_FRACTION = 0.15
_MARGIN = 0.02

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
    warning: Color
    message_background: Color
    message_text: Color


HIGH_CONTRAST = Palette(
    background=(0, 0, 0),
    card=(64, 32, 16),
    card_text=(255, 255, 255),
    focus=(0, 230, 255),
    progress=(0, 160, 200),
    confirmed=(80, 200, 60),
    pending=(255, 170, 60),
    status_text=(255, 255, 255),
    warning=(60, 120, 255),
    message_background=(30, 30, 30),
    message_text=(255, 255, 255),
)

SOFT = Palette(
    background=(36, 34, 29),
    card=(65, 62, 52),
    card_text=(239, 241, 239),
    focus=(116, 185, 199),
    progress=(90, 150, 160),
    confirmed=(119, 191, 142),
    pending=(200, 160, 110),
    status_text=(239, 241, 239),
    warning=(92, 184, 232),
    message_background=(43, 50, 53),
    message_text=(239, 241, 239),
)


def board_area(width: int, height: int) -> tuple[float, float, float, float]:
    """Normalized screen rectangle (left, top, right, bottom) occupied by the board."""
    del width, height
    return (_MARGIN, STATUS_FRACTION, 1.0 - _MARGIN, 1.0 - MESSAGE_FRACTION)


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
    """Inverse of :func:`screen_to_board` (used for pointer clicks)."""
    left, top, right, bottom = area
    return (left + point[0] * (right - left), top + point[1] * (bottom - top))


_ABSTAIN_TEXT = {
    AbstainReason.NO_FRAME: "Camara sin imagen: revisar conexion",
    AbstainReason.STALE_FRAME: "Imagen demorada: seleccion detenida",
    AbstainReason.TRACKING: "Rostro no estable o fuera de distancia",
    AbstainReason.EYE_SIGNAL: "El ojo configurado no se ve bien",
    AbstainReason.NO_PREDICTION: "Mirada fuera de la pantalla",
}


def status_text(update: SessionUpdate, eye_mode: EyeMode) -> str:
    """Short Spanish status line for the person and their support."""
    mode = {EyeMode.BOTH: "ambos ojos", EyeMode.LEFT: "ojo izq.", EyeMode.RIGHT: "ojo der."}[
        eye_mode
    ]
    state = update.state
    if state is SessionState.NO_CALIBRATION:
        text = "Sin calibracion para este modo: use teclas 1-4 o calibre"
    elif state is SessionState.PAUSED:
        text = "En pausa - ESPACIO para continuar"
    elif state is SessionState.SIGNAL_LOST:
        text = _ABSTAIN_TEXT.get(update.abstain_reason or AbstainReason.TRACKING, "Sin senal")
    elif state is SessionState.RECOVERING:
        text = "Recuperando senal..."
    elif state is SessionState.FOCUSING and update.focused is not None:
        text = f"Mirando: {update.focused.label}"
    elif state is SessionState.AWAITING_CONFIRMATION and update.pending is not None:
        text = f"Para confirmar {update.pending.label}: mire afuera y vuelva"
    elif state is SessionState.COOLDOWN:
        text = "Confirmado"
    elif state is SessionState.AWAITING_EXIT:
        text = "Confirmado - aparte la mirada para elegir otra vez"
    else:
        text = "Listo: mire una opcion"
    return f"[{mode}] {text}"


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
) -> npt.NDArray[np.uint8]:
    """Render one frame of the board.

    Args:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        layout: Board layout (board-normalized coordinates).
        update: Latest session update.
        eye_mode: Active ocular mode (shown in the status bar).
        high_contrast: Use the high-contrast palette.
        gaze_point: Optional screen-normalized gaze point (setup aid only).
        speech_ok: False shows that the last phrase could not be spoken.
        flash: Option to highlight as just confirmed.
    """
    palette = HIGH_CONTRAST if high_contrast else SOFT
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = palette.background
    area = board_area(width, height)

    lost = update.state in (
        SessionState.SIGNAL_LOST,
        SessionState.NO_CALIBRATION,
        SessionState.PAUSED,
    )
    status_color = palette.warning if lost else palette.status_text
    _fit_text(
        canvas,
        status_text(update, eye_mode),
        (int(width * _MARGIN), 0, int(width * (1 - _MARGIN)), int(height * STATUS_FRACTION)),
        status_color,
        max_scale=1.2,
    )
    if update.state is SessionState.RECOVERING:
        bar_w = int(width * (1 - 2 * _MARGIN) * min(1.0, update.recovery_progress))
        y = int(height * STATUS_FRACTION) - 6
        cv2.rectangle(
            canvas,
            (int(width * _MARGIN), y),
            (int(width * _MARGIN) + bar_w, y + 4),
            palette.progress,
            -1,
        )

    for option in layout.options:
        _draw_option(canvas, option, update, area, palette, flash)

    message_top = int(height * (1.0 - MESSAGE_FRACTION))
    cv2.rectangle(canvas, (0, message_top), (width, height), palette.message_background, -1)
    message = update.message or ""
    if message and not speech_ok:
        message = f"{message}   (sin voz)"
    _fit_text(
        canvas,
        message,
        (int(width * _MARGIN), message_top, int(width * (1 - _MARGIN)), height),
        palette.message_text,
        max_scale=2.4,
    )

    if gaze_point is not None:
        x = int(gaze_point[0] * width)
        y = int(gaze_point[1] * height)
        cv2.circle(canvas, (x, y), 14, palette.focus, 3, cv2.LINE_AA)
    return canvas


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
    if confirmed:
        border, border_width = palette.confirmed, 14
    elif pending:
        border, border_width = palette.pending, 12
    elif focused:
        border, border_width = palette.focus, 10
    _filled_rounded_rectangle(canvas, (box[0], box[1]), (box[2], box[3]), border, 24)
    inner = (
        box[0] + border_width,
        box[1] + border_width,
        box[2] - border_width,
        box[3] - border_width,
    )
    _filled_rounded_rectangle(canvas, (inner[0], inner[1]), (inner[2], inner[3]), palette.card, 18)
    if focused and update.progress > 0.0:
        fill_height = int((inner[3] - inner[1]) * min(1.0, update.progress))
        cv2.rectangle(
            canvas,
            (inner[0], inner[3] - fill_height),
            (inner[2], inner[3]),
            palette.progress,
            -1,
        )
    _fit_text(canvas, option.label, inner, palette.card_text, max_scale=5.0, thickness=8)


def _fit_text(
    canvas: npt.NDArray[np.uint8],
    text: str,
    bounds: tuple[int, int, int, int],
    color: Color,
    *,
    max_scale: float,
    thickness: int | None = None,
) -> None:
    rendered = _ascii(text)
    if not rendered:
        return
    left, top, right, bottom = bounds
    box_w, box_h = max(1, right - left), max(1, bottom - top)
    scale = max_scale
    stroke = thickness or max(2, int(scale * 2))
    while scale > 0.4:
        stroke = thickness or max(2, int(scale * 2))
        (text_w, text_h), _ = cv2.getTextSize(rendered, cv2.FONT_HERSHEY_SIMPLEX, scale, stroke)
        if text_w <= box_w * 0.9 and text_h <= box_h * 0.6:
            break
        scale *= 0.9
    (text_w, text_h), _ = cv2.getTextSize(rendered, cv2.FONT_HERSHEY_SIMPLEX, scale, stroke)
    origin = (left + (box_w - text_w) // 2, top + (box_h + text_h) // 2)
    cv2.putText(
        canvas, rendered, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, stroke, cv2.LINE_AA
    )


__all__ = [
    "HIGH_CONTRAST",
    "SOFT",
    "board_area",
    "board_to_screen",
    "render_board",
    "screen_to_board",
    "status_text",
]
