"""Blink-robust smoothing and the wider distance range of the board pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mekasa_gaze.core.config import GuidedValidationConfig
from mekasa_gaze.core.eye_mode import EyeMode
from mekasa_gaze.runners.board_simulation import synthetic_model, synthetic_observation
from mekasa_gaze.runners.communication_board import GazePipeline, profile_name
from mekasa_gaze.tracking.smoothing import LidTransitionGate, RobustGazeSmoother

ROOT = Path(__file__).resolve().parents[3]
FPS = 30.0


def test_a_blink_does_not_restart_the_smoother_and_post_blink_outliers_are_outvoted() -> None:
    smoother = RobustGazeSmoother(alpha=0.25)
    t = 0.0
    for _ in range(15):
        t += 1 / FPS
        smoother.update(t, (0.3, 0.3))
    for _ in range(6):  # eyes closed for 0.2 s
        t += 1 / FPS
        assert smoother.update(t, None) == (None, False)
    t += 1 / FPS
    first, _ = smoother.update(t, (0.3, 0.9))  # lid still opening: estimate jumps down
    t += 1 / FPS
    second, _ = smoother.update(t, (0.3, 0.85))
    assert first is not None and second is not None
    assert first[1] < 0.35 and second[1] < 0.35  # the jump does not reach the output


def test_unreliable_frames_hold_and_long_gaps_reset() -> None:
    smoother = RobustGazeSmoother()
    smoother.update(0.0, (0.2, 0.2))
    assert smoother.update(0.03, (0.9, 0.9), reliable=False) == ((0.2, 0.2), True)
    smoother.update(1.0, None)  # a real loss of signal (> 0.6 s)
    value, held = smoother.update(1.03, (0.8, 0.8))
    assert value == pytest.approx((0.8, 0.8)) and not held


def test_holding_never_lasts_more_than_max_hold() -> None:
    smoother = RobustGazeSmoother(alpha=1.0, window=1, max_hold_seconds=0.3)
    smoother.update(0.0, (0.3, 0.3))
    t, outputs = 0.0, []
    for _ in range(20):  # a long stretch flagged unreliable (e.g. looking further down)
        t += 1 / FPS
        outputs.append(smoother.update(t, (0.3, 0.8), reliable=False))
    assert outputs[0] == ((0.3, 0.3), True)
    assert outputs[-1] == ((0.3, 0.8), False)  # after 0.3 s the new gaze is accepted


def test_lid_gate_flags_fast_drops_but_not_a_sustained_low_lid() -> None:
    gate = LidTransitionGate()
    t = 0.0
    for _ in range(10):
        t += 1 / FPS
        assert not gate.in_transition(t, 0.26)
    t += 1 / FPS
    assert gate.in_transition(t, 0.12)  # closing for a blink
    flagged = []
    for _ in range(15):  # but if it stays low (looking down), it is reliable again
        t += 1 / FPS
        flagged.append(gate.in_transition(t, 0.12))
    assert flagged[-1] is False and sum(flagged) <= 5


def test_profile_names_are_short_safe_labels() -> None:
    assert profile_name(" Lentes 2 ") == "Lentes2"
    assert profile_name("papá") == "papá"
    with pytest.raises(ValueError):
        profile_name("../")


def _pipeline(**kwargs: object) -> GazePipeline:
    validity = GuidedValidationConfig.load(ROOT / "config" / "experiment-001.toml")
    model = replace(synthetic_model(EyeMode.BOTH), closed_ear_threshold=0.09)
    return GazePipeline(
        extractor=None,  # type: ignore[arg-type]
        validity_config=validity,
        model=model,
        eye_mode=EyeMode.BOTH,
        smoothing_alpha=1.0,
        area=(0.0, 0.0, 1.0, 1.0),
        **kwargs,  # type: ignore[arg-type]
    )


def test_pipeline_holds_the_estimate_while_the_lid_is_half_closed() -> None:
    pipeline = _pipeline()
    steady = synthetic_observation((0.3, 0.3), "ok", 1, 1.0)
    assert pipeline.process_observation(steady, frame_timestamp_seconds=1.0).screen_point == (
        pytest.approx((0.3, 0.3))
    )
    closing = synthetic_observation((0.3, 0.9), "ok", 2, 1.033)
    # Aperture falls from 0.30 to 0.12 in one frame: a lid closing, not a downward look.
    features = dict(closing.features, left_eye_aspect_ratio=0.12, right_eye_aspect_ratio=0.12)
    result = pipeline.process_observation(
        replace(closing, features=features), frame_timestamp_seconds=1.033
    )
    assert result.held and result.screen_point == pytest.approx((0.3, 0.3))


def test_pipeline_accepts_a_wider_distance_range() -> None:
    narrow = _pipeline()  # research protocol band 0.25-0.45
    wide = _pipeline(face_width_band=(0.16, 0.62))
    for width in (0.18, 0.58):
        observation = synthetic_observation((0.3, 0.3), "ok", 1, 1.0)
        observation = replace(observation, features=dict(observation.features, face_width=width))
        assert not narrow.process_observation(observation, frame_timestamp_seconds=1.0).tracking_ok
        assert wide.process_observation(observation, frame_timestamp_seconds=1.0).tracking_ok


def test_main_keeps_history_and_separates_profiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from mekasa_gaze.runners import communication_board as board

    seen: dict[str, object] = {}

    def fake_run_board(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "measurement": {"adaptation": {}},
            "calibration": None,
            "test": None,
            "configuration": {"profile": kwargs["profile"]},
        }

    monkeypatch.setattr(board, "run_board", fake_run_board)
    output = tmp_path / "communication" / "board-session-latest.json"
    for _ in range(2):
        code = board.main(
            [
                "--profile",
                "lentes",
                "--calibration-dir",
                str(tmp_path / "calibration"),
                "--output",
                str(output),
            ]
        )
        assert code == 0
    assert seen["calibration_dir"] == tmp_path / "calibration" / "profiles" / "lentes"
    assert seen["profile"] == "lentes"
    assert output.exists()
    history = sorted((output.parent / "history").glob("board-session-*-lentes.json"))
    assert len(history) == 2  # every session kept, none overwritten
