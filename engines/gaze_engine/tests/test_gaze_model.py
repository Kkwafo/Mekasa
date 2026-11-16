"""Personal gaze model, eye patches, session calibration, placement and adaptation."""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from mekasa_gaze.core.config import GuidedValidationConfig
from mekasa_gaze.core.domain import EyeState, FeatureObservation, HeadPose, TrackingState
from mekasa_gaze.core.eye_mode import EyeMode
from mekasa_gaze.runners.communication_board import (
    GazePipeline,
    InstructedTest,
    choose_eye_mode,
    instructed_order,
    instructed_report,
    load_estimator,
)
from mekasa_gaze.tracking.eye_patches import eye_patch, eye_patches_from_landmarks
from mekasa_gaze.tracking.gaze_model import (
    CalibrationSample,
    FitResult,
    PersonalGazeModel,
    fit_personal_model,
    personal_thresholds,
    save_personal_model,
    usable_samples,
)
from mekasa_gaze.tracking.placement_monitor import (
    Drift,
    PlacementMonitor,
    PlacementReference,
    compensate_distance,
)
from mekasa_gaze.tracking.session_alignment import (
    OnlineCorrection,
    fit_affine_toward_identity,
)
from mekasa_gaze.tracking.session_calibration import (
    CalibrationProcedure,
    calibration_steps,
    estimated_seconds,
    grid_targets,
)
from mekasa_gaze.ui.board_view import board_area, render_calibration, render_summary

ROOT = Path(__file__).resolve().parents[3]
AREA = (0.02, 0.2, 0.98, 0.98)
SPLIT = (0.5, 0.59)


# ------------------------------------------------------------------ synthetic eyes


def _draw_eye(
    image: npt.NDArray[np.uint8],
    corner_a: tuple[float, float],
    corner_b: tuple[float, float],
    gaze: tuple[float, float],
) -> None:
    """Sclera, iris and an upper lid that drops with downward gaze (like real eyelids)."""
    h, w = image.shape[:2]
    ax, ay = corner_a[0] * w, corner_a[1] * h
    bx, by = corner_b[0] * w, corner_b[1] * h
    cx, cy = (ax + bx) / 2, (ay + by) / 2
    width = abs(bx - ax)
    cv2.ellipse(
        image, (int(cx), int(cy)), (int(width / 2), int(width / 4)), 0, 0, 360, (235, 235, 235), -1
    )
    iris_x = cx + (gaze[0] - 0.5) * width * 0.45
    iris_y = cy + (gaze[1] - 0.5) * width * 0.18
    cv2.circle(image, (int(iris_x), int(iris_y)), int(width * 0.18), (40, 30, 20), -1)
    lid = cy - width / 4 + (gaze[1]) * width * 0.22  # lid edge lowers with gaze y
    cv2.rectangle(
        image, (int(ax) - 4, int(cy - width / 2)), (int(bx) + 4, int(lid)), (150, 170, 200), -1
    )


def _face_image(gaze: tuple[float, float], scale: float = 1.0, shift: float = 0.0):
    image = np.full((480, 640, 3), (150, 170, 200), dtype=np.uint8)
    half = 0.06 * scale
    right = ((0.40 + shift - half, 0.45), (0.40 + shift + half, 0.45))  # image-left eye
    left = ((0.60 + shift - half, 0.45), (0.60 + shift + half, 0.45))
    _draw_eye(image, *right, gaze)
    _draw_eye(image, *left, gaze)
    landmarks = [(0.0, 0.0, 0.0)] * 478
    landmarks = list(landmarks)
    landmarks[33] = (*right[0], 0.0)
    landmarks[133] = (*right[1], 0.0)
    landmarks[362] = (*left[0], 0.0)
    landmarks[263] = (*left[1], 0.0)
    return image, landmarks


def test_eye_patch_is_normalized_scale_invariant_and_tracks_vertical_gaze() -> None:
    up_image, landmarks = _face_image((0.5, 0.2))
    down_image, _ = _face_image((0.5, 0.8))
    big_image, big_landmarks = _face_image((0.5, 0.2), scale=1.6)
    up_left, up_right = eye_patches_from_landmarks(up_image, landmarks)
    down_left, _ = eye_patches_from_landmarks(down_image, landmarks)
    big_left, _ = eye_patches_from_landmarks(big_image, big_landmarks)
    assert up_left is not None and up_right is not None and down_left is not None
    assert big_left is not None
    assert up_left.shape == (200,)
    assert abs(float(up_left.mean())) < 1e-4 and float(up_left.std()) == pytest.approx(1, abs=1e-3)
    same_gaze_bigger_face = float(np.corrcoef(up_left, big_left)[0, 1])
    different_gaze = float(np.corrcoef(up_left, down_left)[0, 1])
    assert same_gaze_bigger_face > 0.85  # closer to the camera, same look -> same patch
    assert different_gaze < same_gaze_bigger_face - 0.2


def test_eye_patch_rejects_tiny_eyes() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert eye_patch(image, (0.5, 0.5), (0.505, 0.5)) is None


# ------------------------------------------------------------------ fitting


def _samples(
    *, vertical_in_landmarks: bool, rounds: int = 2, seed: int = 3
) -> list[CalibrationSample]:
    """Grid fixations; landmarks carry horizontal gaze, optionally vertical; patches both."""
    rng = np.random.default_rng(seed)
    samples: list[CalibrationSample] = []
    t = 0.0
    for round_index in range(rounds):
        for target in grid_targets(AREA):
            gx = (target[0] - AREA[0]) / (AREA[2] - AREA[0])
            gy = (target[1] - AREA[1]) / (AREA[3] - AREA[1])
            image, landmarks = _face_image((gx, gy))
            patches = eye_patches_from_landmarks(image, landmarks)
            for _ in range(15):
                t += 1 / 30
                noisy = (
                    patches[0] + rng.normal(0, 0.3, 200).astype(np.float32),
                    patches[1] + rng.normal(0, 0.3, 200).astype(np.float32),
                )
                iris_y = gy if vertical_in_landmarks else 0.5
                features = {
                    "face_width": 0.35,
                    "face_center_x": 0.5,
                    "face_center_y": 0.5,
                    "head_yaw_degrees": rng.normal(0, 1),
                    "head_pitch_degrees": rng.normal(8, 1),
                    "head_roll_degrees": rng.normal(0, 1),
                }
                for eye in ("left", "right"):
                    features.update(
                        {
                            f"{eye}_iris_x": gx + rng.normal(0, 0.03),
                            f"{eye}_iris_y": iris_y + rng.normal(0, 0.03),
                            f"{eye}_iris_v": iris_y + rng.normal(0, 0.03),
                            f"{eye}_eye_aspect_ratio": 0.30 - 0.04 * gy + rng.normal(0, 0.05),
                            f"{eye}_eye_vertical_span": 0.02,
                        }
                    )
                samples.append(CalibrationSample(target, round_index, t, features, noisy))
            t += 0.5
    return samples


def test_appearance_rescues_vertical_gaze_when_landmarks_lack_it() -> None:
    samples = _samples(vertical_in_landmarks=False)
    landmarks_only = fit_personal_model(samples, EyeMode.BOTH, SPLIT, kinds=("landmarks",))
    full = fit_personal_model(samples, EyeMode.BOTH, SPLIT)
    assert landmarks_only.model is not None and full.model is not None
    assert landmarks_only.model.metrics["vertical_accuracy"] < 0.85
    assert full.model.kind in ("appearance", "hybrid")
    assert full.model.metrics["quadrant_accuracy"] >= 0.95
    assert full.model.metrics["fixation_quadrant_accuracy"] == 1.0
    assert full.model.metrics["cross_validated"] is True
    assert {c["kind"] for c in full.candidates if "skipped" not in c} == {
        "landmarks",
        "appearance",
        "hybrid",
    }


def test_single_eye_models_use_only_that_eye() -> None:
    result = fit_personal_model(_samples(vertical_in_landmarks=True), EyeMode.LEFT, SPLIT)
    assert result.model is not None
    names = result.model.spec.scalar_names
    assert all(not name.startswith("right_") for name in names)
    assert result.model.spec.patch_eyes in ((), ("left",))


def test_model_round_trip_and_prediction(tmp_path: Path) -> None:
    result = fit_personal_model(_samples(vertical_in_landmarks=False), EyeMode.BOTH, SPLIT)
    model = result.model
    assert model is not None and model.reference is not None
    path = tmp_path / "session-gaze-model-both.json"
    save_personal_model(model, path)
    loaded = load_estimator(path)
    assert isinstance(loaded, PersonalGazeModel)
    image, landmarks = _face_image((0.15, 0.85))
    patches = eye_patches_from_landmarks(image, landmarks)
    features = {
        "left_iris_x": 0.15,
        "right_iris_x": 0.15,
        "left_iris_y": 0.5,
        "right_iris_y": 0.5,
        "left_iris_v": 0.5,
        "right_iris_v": 0.5,
        "left_eye_aspect_ratio": 0.27,
        "right_eye_aspect_ratio": 0.27,
        "face_width": 0.35,
    }
    observation = FeatureObservation(
        1,
        0,
        TrackingState.GOOD,
        features=features,
        head_pose=HeadPose(yaw_degrees=0.0, pitch_degrees=8.0, roll_degrees=0.0),
        eye_patches=patches,
    )
    point = loaded.predict_observation(
        observation, left_eye_state=EyeState.OPEN, right_eye_state=EyeState.OPEN
    )
    assert point is not None
    assert point[0] < SPLIT[0] and point[1] > SPLIT[1]  # bottom-left quadrant
    no_patch = FeatureObservation(1, 0, TrackingState.GOOD, features=features)
    if model.spec.patch_eyes:
        assert (
            loaded.predict_observation(
                no_patch, left_eye_state=EyeState.OPEN, right_eye_state=EyeState.OPEN
            )
            is None
        )


def test_too_few_samples_returns_a_spanish_reason() -> None:
    samples = _samples(vertical_in_landmarks=True)[:20]
    result = fit_personal_model(samples, EyeMode.BOTH, SPLIT)
    assert result.model is None and result.error and "muestras" in result.error


def test_personal_thresholds_follow_downward_lids_and_blinks_are_guarded() -> None:
    def sample(t: float, y: float, ear: float) -> CalibrationSample:
        features = {
            "left_eye_aspect_ratio": ear,
            "right_eye_aspect_ratio": ear,
            "left_eye_vertical_span": 0.01,
            "right_eye_vertical_span": 0.01,
        }
        return CalibrationSample((0.5, y), 0, t, features, (None, None))

    bottom = [sample(i / 30, 0.9, 0.12) for i in range(30)]
    thresholds = personal_thresholds(bottom, split_y=0.59)
    assert thresholds.closed_ear_threshold == pytest.approx(0.072)
    with_blink = [*bottom[:10], sample(10 / 30, 0.9, 0.04), *bottom[11:]]
    kept = usable_samples(with_blink, EyeMode.BOTH, thresholds)
    assert len(kept) < 30 - 1  # the blink frame and its neighbours are dropped
    assert all(s.features["left_eye_aspect_ratio"] >= 0.072 for s in kept)


# ------------------------------------------------------------------ session calibration


def test_calibration_procedure_runs_grid_head_movement_and_second_round() -> None:
    steps = calibration_steps(AREA)
    assert len(steps) == 19 and sum(not step.grid for step in steps) == 1
    assert 30 <= estimated_seconds(steps) <= 40
    assert {step.round_index for step in steps if step.grid} == {0, 1}
    procedure = CalibrationProcedure(steps)
    observation = FeatureObservation(
        1, 0, TrackingState.GOOD, features={"left_eye_aspect_ratio": 0.3}
    )
    now, snapshot = 0.0, procedure.update(0.0, observation)
    while not snapshot.done:
        now += 1 / 30
        snapshot = procedure.update(now, observation)
    assert now == pytest.approx(estimated_seconds(steps), abs=1.0)
    assert procedure.short_steps == 0
    head = [sample for sample in procedure.samples if not sample.grid]
    assert head and all(sample.round_index == 0 for sample in head)


def test_calibration_waits_for_missing_frames_but_not_forever() -> None:
    steps = calibration_steps(AREA, rounds=1, head_movement_seconds=0)
    procedure = CalibrationProcedure(steps, max_extra_seconds=1.0)
    now, snapshot = 0.0, procedure.update(0.0, None)
    while not snapshot.done:
        now += 1 / 30
        snapshot = procedure.update(now, None)
    assert procedure.short_steps == len(steps)
    assert now == pytest.approx(estimated_seconds(steps) + len(steps) * 1.0, abs=1.0)


# ------------------------------------------------------------------ adaptation


def test_identity_prior_moves_partially_with_one_sample_and_fully_with_anchors() -> None:
    one = fit_affine_toward_identity([((0.3, 0.8), (0.3, 0.6), 6.0)])
    shifted = one.predict((0.3, 0.6))
    assert 0.6 < shifted[1] < 0.8  # partial, not a jump
    anchors = [
        ((tx, ty), (0.1 + 0.6 * tx, 0.2 + 0.6 * ty), 1.0)
        for tx, ty in [(0.25, 0.3), (0.75, 0.3), (0.25, 0.8), (0.75, 0.8)]
        for _ in range(30)
    ]
    full = fit_affine_toward_identity(anchors)
    assert full.predict((0.1 + 0.6 * 0.25, 0.2 + 0.6 * 0.8)) == pytest.approx((0.25, 0.8), abs=0.03)


def test_online_correction_learns_decays_and_retracts() -> None:
    online = OnlineCorrection()
    assert online.correction(0.0) is None
    for i in range(6):
        online.add_selection((0.3, 0.6), (0.3, 0.8), float(i))
    learned = online.correction(6.0)
    assert learned is not None and learned.predict((0.3, 0.6))[1] > 0.74
    older = online.correction(6.0 + 3 * 600.0)
    assert older is not None and older.predict((0.3, 0.6))[1] < learned.predict((0.3, 0.6))[1]
    assert online.retract_recent(5.5) and online.implicit_samples == 5
    assert not online.retract_recent(100.0)


def test_placement_monitor_detects_distance_and_rebases() -> None:
    reference = PlacementReference(0.35, 0.5, 0.5, 0.0, 8.0)
    monitor = PlacementMonitor(reference=reference, alpha=1.0)
    near = {
        "face_width": 0.42,
        "face_center_x": 0.5,
        "face_center_y": 0.5,
        "head_yaw_degrees": 0.0,
        "head_pitch_degrees": 8.0,
    }
    status = monitor.update(0.0, near)
    assert status.drift is Drift.CLOSER and status.distance_scale == pytest.approx(0.35 / 0.42)
    assert monitor.update(3.0, near).sustained_seconds == pytest.approx(3.0)
    monitor.rebase()
    assert monitor.update(3.1, near).drift is Drift.OK
    turned = dict(near, head_yaw_degrees=15.0)
    assert monitor.update(3.2, turned).drift is Drift.TURNED
    assert compensate_distance((0.5, 0.8), 0.8, (0.5, -0.02)) == pytest.approx((0.5, 0.636))


def test_pipeline_compensates_distance_about_the_camera() -> None:
    from dataclasses import replace

    from mekasa_gaze.runners.board_simulation import synthetic_model, synthetic_observation

    validity = GuidedValidationConfig.load(ROOT / "config" / "experiment-001.toml")
    model = synthetic_model(EyeMode.BOTH)

    class _WithReference:
        eye_mode = model.eye_mode
        closed_ear_threshold = None
        reference = PlacementReference(0.35, 0.5, 0.5, 0.0, 0.0)

        def predict_observation(self, observation, **kwargs):  # type: ignore[no-untyped-def]
            return model.predict_observation(observation, **kwargs)

    pipeline = GazePipeline(
        extractor=None,  # type: ignore[arg-type]
        validity_config=validity,
        model=_WithReference(),
        eye_mode=EyeMode.BOTH,
        smoothing_alpha=1.0,
        area=(0.0, 0.0, 1.0, 1.0),
        distance_compensation=True,
        camera_position=(0.5, 0.0),
    )
    pipeline.monitor.alpha = 1.0
    base = synthetic_observation((0.5, 0.8), "ok", 1, 1.0)
    features = dict(
        base.features,
        face_width=0.40,
        face_center_x=0.5,
        face_center_y=0.5,
        head_yaw_degrees=0.0,
        head_pitch_degrees=0.0,
    )
    closer = replace(
        base,
        features=features,
        head_pose=HeadPose(yaw_degrees=0.0, pitch_degrees=0.0, roll_degrees=0.0),
    )
    result = pipeline.process_observation(closer, frame_timestamp_seconds=1.0)
    assert result.screen_point == pytest.approx((0.5, 0.8 * 0.35 / 0.40))


# ------------------------------------------------------------------ test mode & UI


def test_instructed_test_counts_gaze_only_and_times_out() -> None:
    test = InstructedTest(["YES", "WATER"], timeout_seconds=3.0, pause_seconds=0.5)
    assert test.update(0.0) is None
    assert test.update(1.0) == "YES"
    test.update(1.5, "YES", "keyboard")  # keyboard does not answer a prompt
    assert test.current is not None
    test.update(2.0, "YES", "gaze")
    assert test.current is None
    assert test.update(2.6) == "WATER"
    test.update(6.0)  # timeout
    assert test.done
    report = instructed_report(test.trials, ["YES", "NO", "WATER", "HELP"])
    assert report["correct"] == 1 and report["no_selection"] == 1
    assert report["confusion"]["WATER"]["NONE"] == 1
    assert report["correct_response_seconds_median"] == pytest.approx(1.0)


def test_test_order_has_no_immediate_repeats() -> None:
    order = instructed_order(["YES", "NO", "WATER", "HELP"], 2, random.Random(1))
    assert sorted(order) == sorted(["YES", "NO", "WATER", "HELP"] * 2)
    assert all(a != b for a, b in itertools.pairwise(order))


def test_choose_eye_mode_switches_only_when_clearly_better() -> None:
    def result(quadrant: float) -> FitResult:
        model = fit_personal_model(_samples(vertical_in_landmarks=True), EyeMode.BOTH, SPLIT).model
        assert model is not None
        from dataclasses import replace

        return FitResult(replace(model, metrics={"quadrant_accuracy": quadrant}), [], 100)

    results = {EyeMode.BOTH: result(0.7), EyeMode.LEFT: result(0.95), EyeMode.RIGHT: result(0.6)}
    assert choose_eye_mode(EyeMode.BOTH, results)[0] is EyeMode.LEFT
    results[EyeMode.BOTH] = result(0.92)
    assert choose_eye_mode(EyeMode.BOTH, results) == (EyeMode.BOTH, None)


def test_calibration_and_summary_screens_render() -> None:
    from mekasa_gaze.tracking.session_calibration import CalibrationSnapshot

    area = board_area(1280, 720)
    target = grid_targets(area)[0]
    snapshot = CalibrationSnapshot(0, 19, target, True, 0.4, False, False, 30.0)
    canvas = render_calibration(1280, 720, snapshot)
    assert canvas.shape == (720, 1280, 3) and canvas.std() > 5
    head = CalibrationSnapshot(9, 19, target, True, 0.4, False, True, 18.0)
    assert render_calibration(1280, 720, head, signal_ok=False).shape == (720, 1280, 3)
    summary = render_summary(
        1280, 720, "Resultado", [("→ Ambos ojos: 98 %", True), ("Ojo derecho: 60 %", False)]
    )
    assert summary.std() > 5
