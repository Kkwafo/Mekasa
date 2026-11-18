from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mekasa_gaze.core.board import CommunicationConfig
from mekasa_gaze.core.domain import FeatureObservation, TrackingState
from mekasa_gaze.ui.board_view import (
    CAMERA_VIEWS,
    board_area,
    camera_view_box,
    draw_camera_view,
)

W, H = 1280, 720


def _observation() -> FeatureObservation:
    return FeatureObservation(
        frame_sequence=1,
        timestamp_ns=1,
        tracking_state=TrackingState.GOOD,
        debug_eye_points=((0.4, 0.4), (0.6, 0.4)),
        debug_left_iris_points=((0.6, 0.4),),
        debug_right_iris_points=((0.4, 0.4),),
    )


def test_off_draws_nothing() -> None:
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    draw_camera_view(canvas, frame, _observation(), view="off", tracking_ok=True)
    assert camera_view_box(W, H, "off") is None
    assert np.count_nonzero(canvas) == 0


def test_corner_view_stays_in_the_top_bands_and_never_covers_an_option() -> None:
    left, top, right, bottom = camera_view_box(W, H, "corner") or (0, 0, 0, 0)
    assert 0 <= left < right <= W and 0 <= top < bottom
    assert bottom <= board_area(W, H)[1] * H  # above the board: no option underneath
    assert abs((right - left) / (bottom - top) - 4 / 3) < 0.02


def test_center_view_is_inside_the_board_and_centred() -> None:
    left, top, right, bottom = camera_view_box(W, H, "center") or (0, 0, 0, 0)
    a_left, a_top, a_right, a_bottom = board_area(W, H)
    assert a_left * W < left < right < a_right * W
    assert a_top * H < top < bottom < a_bottom * H
    assert abs((left + right) / 2 - (a_left + a_right) / 2 * W) <= 1


@pytest.mark.parametrize("view", ["corner", "center"])
def test_draws_mirrored_image_with_landmarks_and_keeps_frame(view: str) -> None:
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :320] = (0, 0, 255)  # left half red in the camera image
    original = frame.copy()
    draw_camera_view(canvas, frame, _observation(), view=view, tracking_ok=True)
    left, top, right, bottom = camera_view_box(W, H, view) or (0, 0, 0, 0)
    middle = (top + bottom) // 2
    # Mirrored: the camera's left half appears on the right of the view.
    assert tuple(canvas[middle, right - 10]) == (0, 0, 255)
    assert tuple(canvas[middle, left + 10]) == (0, 0, 0)
    assert np.array_equal(frame, original)  # the camera frame is not modified


def test_no_frame_shows_placeholder() -> None:
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    draw_camera_view(canvas, None, None, view="corner", tracking_ok=False)
    assert np.count_nonzero(canvas) > 0


def test_config_validates_camera_view() -> None:
    config = CommunicationConfig.load(Path(__file__).parents[3] / "config" / "communication.toml")
    assert config.camera_view == "off"  # shipped default: no camera image on screen
    assert set(CAMERA_VIEWS) == {"off", "corner", "center"}
    replace(config, camera_view="center").validate()
    with pytest.raises(ValueError):
        replace(config, camera_view="grande").validate()
