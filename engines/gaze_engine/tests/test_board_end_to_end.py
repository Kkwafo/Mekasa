"""End-to-end run of the real board runner with a synthetic person, camera and clock.

Covers: calibration at start -> model fitting (landmarks lack vertical signal, as in the
2026-11-16 session B, so eye images must carry it) -> summary -> instructed test -> free use of
the bottom row -> moving closer triggers automatic re-alignment -> implicit learning.
It verifies software behaviour only; real accuracy needs a person (``mekasa.ps1 prueba``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2 as real_cv2
import numpy as np
import pytest

from mekasa_gaze.core.board import CommunicationConfig
from mekasa_gaze.core.config import GuidedValidationConfig
from mekasa_gaze.core.domain import CameraFrame
from mekasa_gaze.core.eye_mode import EyeMode
from mekasa_gaze.runners import communication_board as board
from mekasa_gaze.runners.board_simulation import FakeSpeech
from mekasa_gaze.tracking import frame_source, landmarker
from mekasa_gaze.tracking.frame_source import CaptureStatus
from mekasa_gaze.ui import board_view
from synthetic_face import observation_for

ROOT = Path(__file__).resolve().parents[3]
FPS = 30.0


@dataclass
class World:
    """Shared state of the fake clock, camera, person and keyboard."""

    t: float = 1000.0
    gaze: tuple[float, float] = (0.5, 0.6)
    face_width: float = 0.35
    closed: bool = False
    closed_eyes: tuple[str, ...] = ()
    keys: list[int] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    alignment_reasons: list[str | None] = field(default_factory=list)
    board_started_at: float | None = None
    summary_seen_at: float | None = None
    seq: int = 0


def _install_fakes(monkeypatch: pytest.MonkeyPatch, world: World, speech: FakeSpeech) -> None:
    class Grabber:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def __enter__(self) -> Grabber:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def take_new(self) -> CameraFrame:
            world.seq += 1
            return CameraFrame(world.seq, int(world.t * 1e9), 0.0, np.zeros((4, 4, 3), np.uint8))

        def status(self) -> CaptureStatus:
            return CaptureStatus(True, world.seq, 0, 0, None)

        @property
        def camera_settings(self) -> None:
            return None

    class Extractor:
        def __init__(self, *_: object, **__: object) -> None:
            self._rng = np.random.default_rng(5)

        def __enter__(self) -> Extractor:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def extract(self, _rgb: object, *, sequence: int, timestamp_ns: int) -> Any:
            return observation_for(
                world.gaze,
                sequence=sequence,
                timestamp_ns=timestamp_ns,
                face_width=world.face_width,
                eyes_closed=world.closed,
                closed_eyes=world.closed_eyes,
                rng=self._rng,
            )

    class CV2:
        def __getattr__(self, name: str) -> Any:
            return getattr(real_cv2, name)

        def namedWindow(self, *_: object) -> None: ...
        def setWindowProperty(self, *_: object) -> None: ...
        def resizeWindow(self, *_: object) -> None: ...
        def setMouseCallback(self, *_: object) -> None: ...
        def imshow(self, *_: object) -> None: ...
        def destroyWindow(self, *_: object) -> None: ...

        def waitKey(self, _delay: int) -> int:
            world.t += 1 / FPS
            return world.keys.pop(0) if world.keys else 255

    def wrap(name: str, hook: Any) -> None:
        original = getattr(board, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            hook(*args, **kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(board, name, wrapper)

    def on_calibration(_w: int, _h: int, snapshot: Any, **_: Any) -> None:
        if snapshot.target is not None:
            world.gaze = snapshot.target

    def on_alignment(_w: int, _h: int, snapshot: Any, **kwargs: Any) -> None:
        if snapshot.index == 0 and not snapshot.collecting:
            world.alignment_reasons.append(kwargs.get("reason"))
        if snapshot.target is not None:
            world.gaze = snapshot.target

    def on_summary(_w: int, _h: int, title: str, *_: Any, **__: Any) -> None:
        if not world.summaries or world.summaries[-1] != title:
            world.summaries.append(title)
            world.summary_seen_at = world.t
        if title == "Resultado de la prueba":
            world.keys.append(13)  # ENTER: go to the board
        elif world.summary_seen_at is not None and world.t - world.summary_seen_at > 8.0:
            world.keys.append(13)  # quality below "good": continue anyway

    monkeypatch.setattr(board, "perf_counter", lambda: world.t)
    monkeypatch.setattr(board, "cv2", CV2())
    monkeypatch.setattr(board, "create_speech_backend", lambda **_: speech)
    monkeypatch.setattr(frame_source, "LatestFrameGrabber", Grabber)
    monkeypatch.setattr(landmarker, "FaceFeatureExtractor", Extractor)
    wrap("render_calibration", on_calibration)
    wrap("render_alignment", on_alignment)
    wrap("render_summary", on_summary)


def _center(area: tuple[float, float, float, float], key: str) -> tuple[float, float]:
    option = CommunicationConfig().board().option(key)
    bounds = option.bounds
    return board_view.board_to_screen(
        ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2), area
    )


def test_board_session_calibrates_tests_and_adapts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = World()
    speech = FakeSpeech()
    _install_fakes(monkeypatch, world, speech)
    guided = replace(
        GuidedValidationConfig.load(ROOT / "config" / "experiment-001.toml"),
        canvas_width=640,
        canvas_height=360,
    )
    area = board_view.board_area(640, 360)
    communication = replace(
        CommunicationConfig.load(ROOT / "config" / "communication.toml"),
        interface="simple",
        selection_mode="dwell",
    )
    gap = board_view.board_to_screen((0.5, 0.5), area)

    # Free-use script after the test: (start, end, target or None for the gap, face width).
    # Starts by looking away: after a gaze selection the same option needs an exit first
    # (rearm-on-exit), and the test may have ended on WATER.
    script = [
        (0.0, 1.5, None, 0.35),
        (1.5, 4.0, "WATER", 0.35),
        (4.0, 5.5, None, 0.35),
        (5.5, 8.0, "HELP", 0.35),
        (8.0, 9.5, None, 0.35),
        (9.5, 14.5, None, 0.46),  # moves ~25 % closer and stays -> auto re-alignment
        (14.5, 30.0, "WATER", 0.46),
    ]

    def on_board(*args: Any, **kwargs: Any) -> None:
        prompt = kwargs.get("prompt")
        layout = args[2]
        if prompt and "mire" in prompt:
            label = prompt.rsplit("mire ", 1)[1]
            key = next(option.key for option in layout.options if option.label == label)
            world.gaze = _center(area, key)
            return
        if prompt:
            world.gaze = gap
            return
        if world.board_started_at is None:
            world.board_started_at = world.t
        elapsed = world.t - world.board_started_at
        for start, end, target, width in script:
            if start <= elapsed < end:
                world.face_width = width
                world.gaze = gap if target is None else _center(area, target)
                break
        else:
            world.keys.append(27)

    wrap_original = board.render_board

    def render_board_hook(*args: Any, **kwargs: Any) -> Any:
        on_board(*args, **kwargs)
        return wrap_original(*args, **kwargs)

    monkeypatch.setattr(board, "render_board", render_board_hook)
    countdown: list[int] = []
    original_countdown = board.render_countdown

    def countdown_hook(*args: Any, **kwargs: Any) -> Any:
        countdown.append(args[2])
        return original_countdown(*args, **kwargs)

    monkeypatch.setattr(board, "render_countdown", countdown_hook)

    report = board.run_board(
        communication=communication,
        guided_config=guided,
        calibrations=board.calibration_paths(tmp_path),
        face_model_path=tmp_path / "unused.task",
        smoothing_alpha=0.25,
        speech_settings={},
        fullscreen=False,
        calibration_dir=tmp_path,
        test_repeats=2,
    )

    fits = report["calibration"]["fits"]
    both = fits["BOTH"]
    assert both["kind"] in ("appearance", "hybrid"), both
    assert both["metrics"]["quadrant_accuracy"] >= 0.9
    assert (tmp_path / "session-gaze-model-both.json").exists()
    results = [title for title in world.summaries if title != "Cómo funciona"]
    assert world.summaries[0] == "Cómo funciona"  # brief explanation first
    assert results[:2] == ["Resultado de la calibración", "Resultado de la prueba"]
    assert speech.spoken[0].startswith("Hola. Primero vamos a calibrar")
    assert sorted(set(countdown)) == [1, 2, 3]
    assert not any(text.startswith("Calibración. Mire") for text in speech.spoken)

    test = report["test"]
    assert test["trials"] == 8
    assert test["accuracy"] >= 0.75, test["confusion"]

    assert "Quiero agua" in speech.spoken and "Necesito ayuda" in speech.spoken
    adaptation = report["measurement"]["adaptation"]
    assert adaptation["auto_realignments"] >= 1
    assert "Cambió de posición" in world.alignment_reasons
    assert adaptation["implicit_samples_added"] >= 2
    assert adaptation["model_refits"] >= 1  # weights re-fitted with confirmed choices
    assert adaptation["selection_offsets"]["bottom"]["selections"] >= 1
    assert speech.spoken.count("Quiero agua") >= 2  # bottom row still works after moving
    assert report["privacy"]["eye_patches_persisted"] is False
    assert report["configuration"]["eye_mode_at_end"] == EyeMode.BOTH.value


def test_conversation_board_composes_and_speaks_a_full_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Navigate HABLAR > CÓMO ESTOY > ME DUELE > CABEZA > MUCHO > DECIR, then URGENTE."""
    world = World()
    speech = FakeSpeech()
    _install_fakes(monkeypatch, world, speech)
    alarms: list[float] = []
    monkeypatch.setattr(board, "play_alarm", lambda: alarms.append(world.t) or True)
    guided = replace(
        GuidedValidationConfig.load(ROOT / "config" / "experiment-001.toml"),
        canvas_width=640,
        canvas_height=360,
    )
    area = board_view.board_area(640, 360)
    gap = board_view.board_to_screen((0.5, 0.5), area)
    communication = replace(
        CommunicationConfig.load(ROOT / "config" / "communication.toml"), selection_mode="dwell"
    )
    assert communication.interface == "conversation"
    plan = ["HABLAR", "CÓMO ESTOY \u203a", "ME DUELE \u203a", "CABEZA", "MUCHO", "DECIR", "URGENTE"]
    state = {"step": 0, "cells": None, "away_until": 0.0}
    seen_cells: list[list[str]] = []

    def render_board_hook(*args: Any, **kwargs: Any) -> Any:
        cells = kwargs.get("cells")
        labels = [label for label, _ in cells] if cells else None
        if labels is not None and labels != state["cells"]:
            if state["cells"] is not None:
                state["step"] += 1  # the screen changed: the last choice was taken
                state["away_until"] = world.t + 0.8  # look away (rearm) before the next one
            state["cells"] = labels
            seen_cells.append(labels)
        if state["step"] >= len(plan) or world.t > 2000.0:
            world.keys.append(27)
        elif labels is not None and world.t >= state["away_until"]:
            target = plan[state["step"]]
            if target in labels:
                key = ("YES", "NO", "WATER", "HELP")[labels.index(target)]
                world.gaze = _center(area, key)
        else:
            world.gaze = gap
        return original(*args, **kwargs)

    original = board.render_board
    monkeypatch.setattr(board, "render_board", render_board_hook)
    report = board.run_board(
        communication=communication,
        guided_config=guided,
        calibrations=board.calibration_paths(tmp_path),
        face_model_path=tmp_path / "unused.task",
        smoothing_alpha=0.25,
        speech_settings={},
        fullscreen=False,
        calibration_dir=tmp_path,
    )
    assert "Me duele la cabeza, mucho." in speech.spoken, (speech.spoken, seen_cells)
    assert "¡Necesito ayuda urgente! Vengan, por favor." in speech.spoken
    assert alarms, "urgent must raise the local alarm"
    assert seen_cells[0] == ["SÍ", "NO", "HABLAR", "URGENTE"]
    assert ["UN POCO", "BASTANTE", "MUCHO", "SIGUIENTE"] in seen_cells
    conversation = report["measurement"]["conversation"]
    assert conversation["messages_spoken"] == 2
    assert conversation["selections"] == len(plan)
    assert report["privacy"]["message_history_persisted"] is False
    assert "cabeza" not in str(report)  # no message content in the report


def test_blink_mode_starts_selects_and_stops_with_blinks_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Standby -> 4 blinks start -> long stare selects nothing -> look + 3 blinks compose
    "Necesito agua." -> DECIR -> 4 blinks pause."""
    world = World()
    speech = FakeSpeech()
    _install_fakes(monkeypatch, world, speech)
    guided = replace(
        GuidedValidationConfig.load(ROOT / "config" / "experiment-001.toml"),
        canvas_width=640,
        canvas_height=360,
    )
    area = board_view.board_area(640, 360)
    gap = board_view.board_to_screen((0.5, 0.5), area)
    communication = replace(
        CommunicationConfig.load(ROOT / "config" / "communication.toml"), selection_mode="blink"
    )
    four = [(0.2, None, True), (0.3, None, False)] * 3 + [(0.2, None, True), (1.5, None, False)]

    def choose(label: str, stare: float = 1.0) -> list[tuple[float, str | None, bool]]:
        return [
            (stare, label, False),
            (0.2, label, True),
            (0.3, label, False),
            (0.2, label, True),
            (0.3, label, False),
            (0.2, label, True),
            (1.5, label, False),
        ]

    timeline = [
        (1.0, None, False),
        *four,
        *choose("HABLAR", stare=5.0),
        *choose("NECESITO \u203a"),
        *choose("COMER Y BEBER \u203a"),
        *choose("AGUA"),
        *choose("DECIR"),
        (1.0, None, False),
        *four,
    ]
    ends = np.cumsum([duration for duration, _, _ in timeline])
    state: dict[str, Any] = {"spoken_before_stare_end": None}

    def render_board_hook(*args: Any, **kwargs: Any) -> Any:
        if world.board_started_at is None:
            world.board_started_at = world.t
        elapsed = world.t - world.board_started_at
        index = int(np.searchsorted(ends, elapsed, side="right"))
        if index >= len(timeline):
            world.keys.append(27)
            world.closed = False
            return original(*args, **kwargs)
        _, label, closed = timeline[index]
        world.closed = closed
        labels = [cell for cell, _ in kwargs["cells"]] if kwargs.get("cells") else []
        if label is not None and label in labels:
            world.gaze = _center(area, ("YES", "NO", "WATER", "HELP")[labels.index(label)])
        else:
            world.gaze = gap
        if state["spoken_before_stare_end"] is None and elapsed >= float(ends[10]) - 0.1:
            state["spoken_before_stare_end"] = list(speech.spoken)
        return original(*args, **kwargs)

    original = board.render_board
    monkeypatch.setattr(board, "render_board", render_board_hook)
    report = board.run_board(
        communication=communication,
        guided_config=guided,
        calibrations=board.calibration_paths(tmp_path),
        face_model_path=tmp_path / "unused.task",
        smoothing_alpha=0.25,
        speech_settings={},
        fullscreen=False,
        calibration_dir=tmp_path,
    )
    spoken = speech.spoken
    assert "Para empezar a elegir, parpadee cuatro veces." in spoken
    assert "Empiezo a leer." in spoken
    # Staring at HABLAR for 5 s (well past the 0.9 s dwell) chose nothing.
    assert state["spoken_before_stare_end"] is not None
    assert not any("Necesito" in text for text in state["spoken_before_stare_end"])
    assert "Necesito agua." in spoken, (
        spoken,
        report["measurement"]["session"],
        report["measurement"]["blinks"],
        report["measurement"]["conversation"],
    )
    assert spoken[-1] == "Pausa."
    blinks = report["measurement"]["blinks"]
    assert blinks["toggle_gestures"] == 2
    assert blinks["selections"] == 5
    assert report["measurement"]["session"]["confirmations_by_source"] == {"blink": 5}


def test_combined_mode_dwell_or_blinks_and_blink_shortcuts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default config (combined, blink first): HABLAR by a 3 s look; 3 blinks (gaze wandering)
    then a short look choose NECESITO; COMER Y BEBER by look; AGUA by blinks + look; 5 blinks
    say the message; the 7th blink raises the urgent call at once and 3 s more blinking
    repeats it."""
    world = World()
    speech = FakeSpeech()
    _install_fakes(monkeypatch, world, speech)
    alarms: list[float] = []
    monkeypatch.setattr(board, "play_alarm", lambda: alarms.append(world.t) or True)
    guided = replace(
        GuidedValidationConfig.load(ROOT / "config" / "experiment-001.toml"),
        canvas_width=640,
        canvas_height=360,
    )
    area = board_view.board_area(640, 360)
    gap = board_view.board_to_screen((0.5, 0.5), area)
    communication = CommunicationConfig.load(ROOT / "config" / "communication.toml")
    assert communication.selection_mode == "combined"

    def blinks(count: int, last_open: float = 0.3) -> list[tuple[float, str | None, bool]]:
        actions: list[tuple[float, str | None, bool]] = []
        for index in range(count):
            actions.append((0.2, None, True))  # gaze wanders while blinking
            actions.append((0.3 if index < count - 1 else last_open, None, False))
        return actions

    sep = "\u203a"
    timeline = [
        (1.0, None, False),
        (3.8, "HABLAR", False),  # chosen by the 3 s look
        *blinks(3),
        (2.5, f"NECESITO {sep}", False),  # ...then looked at: chosen
        (3.8, f"COMER Y BEBER {sep}", False),
        *blinks(3),
        (2.5, "AGUA", False),
        (1.0, None, False),
        *blinks(5, last_open=1.5),  # say it now
        (1.0, None, False),
        *blinks(14, last_open=1.5),  # the 7th blink calls; still blinking 2.5 s later: again
        (1.0, None, False),
    ]
    ends = np.cumsum([duration for duration, _, _ in timeline])

    def render_board_hook(*args: Any, **kwargs: Any) -> Any:
        if world.board_started_at is None:
            world.board_started_at = world.t
        elapsed = world.t - world.board_started_at
        index = int(np.searchsorted(ends, elapsed, side="right"))
        if index >= len(timeline):
            world.keys.append(27)
            world.closed = False
            return original(*args, **kwargs)
        _, label, closed = timeline[index]
        world.closed = closed
        labels = [cell for cell, _ in kwargs["cells"]] if kwargs.get("cells") else []
        if label is not None and label in labels:
            world.gaze = _center(area, ("YES", "NO", "WATER", "HELP")[labels.index(label)])
        else:
            world.gaze = gap
        return original(*args, **kwargs)

    original = board.render_board
    monkeypatch.setattr(board, "render_board", render_board_hook)
    report = board.run_board(
        communication=communication,
        guided_config=guided,
        calibrations=board.calibration_paths(tmp_path),
        face_model_path=tmp_path / "unused.task",
        smoothing_alpha=0.25,
        speech_settings={},
        fullscreen=False,
        calibration_dir=tmp_path,
    )
    spoken = speech.spoken
    assert "Necesito agua." in spoken, (spoken, report["measurement"]["session"])
    urgent = "¡Necesito ayuda urgente! Vengan, por favor."
    assert spoken[-2:] == [urgent, urgent], spoken
    assert len(alarms) == 2
    sources = report["measurement"]["session"]["confirmations_by_source"]
    assert sources == {"blink": 2, "gaze": 2}, sources
    blink_stats = report["measurement"]["blinks"]
    assert blink_stats["speak_gestures"] == 1 and blink_stats["urgent_gestures"] == 2
    assert report["measurement"]["session"]["intents_started"] == 2


def test_calibration_learns_winks_and_three_right_winks_choose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The wink step finds that the right eye can wink and the left cannot; on the board three
    right winks and then a look choose HABLAR, while left 'winks' are ignored and the 3 s look
    chooses NECESITO."""
    world = World()
    speech = FakeSpeech()
    _install_fakes(monkeypatch, world, speech)
    guided = replace(
        GuidedValidationConfig.load(ROOT / "config" / "experiment-001.toml"),
        canvas_width=640,
        canvas_height=360,
    )
    area = board_view.board_area(640, 360)
    gap = board_view.board_to_screen((0.5, 0.5), area)
    communication = CommunicationConfig.load(ROOT / "config" / "communication.toml")
    assert communication.calibrate_winks and communication.selection_mode == "combined"
    summaries: list[list[str]] = []

    original_wink = board.render_wink_step

    def wink_hook(*args: Any, **kwargs: Any) -> Any:
        eye = args[2]
        phase = (world.t * 1000) % 600  # close for 0.2 s every 0.6 s
        if eye == "right":
            world.closed, world.closed_eyes = False, ("right",) if phase < 200 else ()
        else:  # cannot isolate the left eye: both close
            world.closed, world.closed_eyes = phase < 200, ()
        world.gaze = gap
        return original_wink(*args, **kwargs)

    original_summary = board.render_summary

    def summary_hook(*args: Any, **kwargs: Any) -> Any:
        world.closed, world.closed_eyes = False, ()
        lines = [text for text, _ in args[3]]
        if not summaries or summaries[-1] != lines:
            summaries.append(lines)
        return original_summary(*args, **kwargs)

    def winks(eye: str, label: str | None) -> list[tuple[float, str | None, tuple[str, ...]]]:
        actions: list[tuple[float, str | None, tuple[str, ...]]] = []
        for _ in range(3):
            actions.append((0.2, label, (eye,)))
            actions.append((0.3, label, ()))
        return actions

    sep = "\u203a"
    timeline = [
        (1.0, None, ()),
        *winks("right", None),  # gaze wanders while winking
        (2.5, "HABLAR", ()),  # ...then the look chooses
        (1.0, f"NECESITO {sep}", ()),
        *winks("left", f"NECESITO {sep}"),  # ignored; the look keeps counting
        (2.5, f"NECESITO {sep}", ()),
        (1.0, None, ()),
    ]
    ends = np.cumsum([duration for duration, _, _ in timeline])

    def render_board_hook(*args: Any, **kwargs: Any) -> Any:
        if world.board_started_at is None:
            world.board_started_at = world.t
        elapsed = world.t - world.board_started_at
        index = int(np.searchsorted(ends, elapsed, side="right"))
        if index >= len(timeline):
            world.keys.append(27)
            world.closed_eyes = ()
            return original(*args, **kwargs)
        _, label, closed_eyes = timeline[index]
        world.closed, world.closed_eyes = False, closed_eyes
        labels = [cell for cell, _ in kwargs["cells"]] if kwargs.get("cells") else []
        if label is not None and label in labels:
            world.gaze = _center(area, ("YES", "NO", "WATER", "HELP")[labels.index(label)])
        else:
            world.gaze = gap
        return original(*args, **kwargs)

    original = board.render_board
    monkeypatch.setattr(board, "render_wink_step", wink_hook)
    monkeypatch.setattr(board, "render_summary", summary_hook)
    monkeypatch.setattr(board, "render_board", render_board_hook)
    report = board.run_board(
        communication=communication,
        guided_config=guided,
        calibrations=board.calibration_paths(tmp_path),
        face_model_path=tmp_path / "unused.task",
        smoothing_alpha=0.25,
        speech_settings={},
        fullscreen=False,
        calibration_dir=tmp_path,
    )
    winks_report = report["calibration"]["winks"]
    assert winks_report["right_wink"] and not winks_report["left_wink"], winks_report
    result = next(lines for lines in summaries if any("Guiño" in line for line in lines))
    assert any("Guiño ojo derecho: 3 de 3" in line for line in result), summaries
    assert any("izquierdo: no se reconoció" in line for line in result)
    assert (tmp_path / "session-wink-profile.json").exists()
    assert "Cierre solo el ojo derecho, tres veces." in speech.spoken
    sources = report["measurement"]["session"]["confirmations_by_source"]
    assert sources == {"wink": 1, "gaze": 1}, (sources, report["measurement"]["blinks"])
    # The wink step makes the person close the eyes again and again: never an urgent call.
    assert report["measurement"]["blinks"]["urgent_gestures"] == 0
