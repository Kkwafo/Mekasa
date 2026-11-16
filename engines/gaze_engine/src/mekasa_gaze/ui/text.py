"""Unicode text for OpenCV canvases (accents and ñ), rendered with Pillow and cached.

OpenCV's Hershey fonts are ASCII-only, which turned "SÍ" into "SI". Text is rasterized once
per (text, size, colour) and alpha-blended onto the BGR canvas, so per-frame cost stays low.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

Color = tuple[int, int, int]

_FONT_CANDIDATES = {
    True: (
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
    False: (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ),
}

FontType = ImageFont.FreeTypeFont | ImageFont.ImageFont


@lru_cache(maxsize=64)
def _font(size: int, bold: bool) -> FontType:
    for candidate in _FONT_CANDIDATES[bold]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=512)
def _text_size(text: str, size: int, bold: bool) -> tuple[int, int, int, int]:
    left, top, right, bottom = _font(size, bold).getbbox(text)
    return int(left), int(top), int(right), int(bottom)


@lru_cache(maxsize=256)
def _fitted_size(text: str, width: int, height: int, max_size: int, bold: bool) -> int:
    low, high = 8, max(8, max_size)
    while low < high:
        mid = (low + high + 1) // 2
        left, top, right, bottom = _text_size(text, mid, bold)
        if right - left <= width and bottom - top <= height:
            low = mid
        else:
            high = mid - 1
    return low


@lru_cache(maxsize=256)
def _glyphs(text: str, size: int, bold: bool) -> npt.NDArray[np.float32]:
    left, top, right, bottom = _text_size(text, size, bold)
    image = Image.new("L", (max(1, right - left), max(1, bottom - top)), 0)
    ImageDraw.Draw(image).text((-left, -top), text, fill=255, font=_font(size, bold))
    return np.asarray(image, dtype=np.float32) / 255.0


def draw_text(
    canvas: npt.NDArray[np.uint8],
    text: str,
    box: tuple[int, int, int, int],
    color: Color,
    *,
    max_size: int,
    bold: bool = True,
    fill: float = 0.9,
    align: str = "center",
) -> None:
    """Draw ``text`` as large as fits in ``box`` (left, top, right, bottom), BGR ``color``."""
    if not text:
        return
    left, top, right, bottom = box
    width, height = int((right - left) * fill), int((bottom - top) * fill)
    if width <= 0 or height <= 0:
        return
    size = _fitted_size(text, width, height, max_size, bold)
    alpha = _glyphs(text, size, bold)
    glyph_h, glyph_w = alpha.shape
    x = left + (right - left - glyph_w) // 2 if align == "center" else left + 4
    y = top + (bottom - top - glyph_h) // 2
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.shape[1], x + glyph_w), min(canvas.shape[0], y + glyph_h)
    if x1 <= x0 or y1 <= y0:
        return
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    mask = alpha[y0 - y : y1 - y, x0 - x : x1 - x, None]
    blended = region * (1.0 - mask) + np.asarray(color, dtype=np.float32) * mask
    canvas[y0:y1, x0:x1] = blended.astype(np.uint8)


__all__ = ["draw_text"]
