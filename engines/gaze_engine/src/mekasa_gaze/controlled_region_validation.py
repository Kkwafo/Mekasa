from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from time import perf_counter
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from mekasa_gaze.calibration import CalibrationModel, load_calibration_model
from mekasa_gaze.camera import CameraUnavailableError, OpenCVCameraProvider
from mekasa_gaze.config import CalibrationConfig, GuidedValidationConfig, InteractionConfig
from mekasa_gaze.domain import GazeValidity
from mekasa_gaze.guided_experiment import _rolling_fps
from mekasa_gaze.interaction import DwellSelector, RegionName, region_at
from mekasa_gaze.landmarker import FaceFeatureExtractor
from mekasa_gaze.live_evaluation import PredictionSmoother
from mekasa_gaze.placement import assess_face_placement
from mekasa_gaze.region_evaluation import _sha256, _write_report
from mekasa_gaze.region_protocol import (
    ControlledRegionProtocol,
    RegionProtocolSnapshot,
    RegionTrialOutcome,
    RegionTrialStage,
)
from mekasa_gaze.runtime import environment, source_revision
from mekasa_gaze.speech import SpeechBackend, create_speech_backend
from mekasa_gaze.validity import assess_gaze_validity
from mekasa_gaze.visualization import render_controlled_region_screen, render_loading_screen

_WINDOW_NAME = "Mekasa Experiment 003 — Controlled Region Validation"


def run_controlled_region_validation(
    *,
    guided_config: GuidedValidationConfig,
    calibration_config: CalibrationConfig,
    interaction_config: InteractionConfig,
    calibration_model: CalibrationModel,
    face_model_path: Path,
    calibration_sha256: str,
    fullscreen: bool | None = None,
) -> dict[str, Any]:
    use_fullscreen = guided_config.fullscreen if fullscreen is None else fullscreen
    camera = OpenCVCameraProvider(
        index=guided_config.camera_index,
        backend=guided_config.backend,
        width=guided_config.width,
        height=guided_config.height,
        target_fps=guided_config.target_fps,
    )
    speech = create_speech_backend(
        enabled=guided_config.audio_enabled,
        required=guided_config.audio_required,
        language=guided_config.audio_language,
        rate=guided_config.audio_rate,
        volume=guided_config.audio_volume,
    )
    protocol = ControlledRegionProtocol(interaction_config)
    selector = DwellSelector(
        dwell_seconds=interaction_config.dwell_seconds,
        cooldown_seconds=interaction_config.cooldown_seconds,
        hysteresis_margin=interaction_config.hysteresis_margin,
    )
    smoother = PredictionSmoother(calibration_config.live_smoothing_alpha)
    frame_times: deque[float] = deque(maxlen=60)
    results: list[dict[str, Any]] = []
    seen_results: set[int] = set()
    trial_metrics: dict[int, dict[str, int]] = defaultdict(_empty_trial_metrics)
    frames_total = 0
    frames_usable = 0
    placement_invalid_frames = 0
    gaze_unusable_frames = 0
    outside_prediction_frames = 0
    last_candidate: RegionName | None = None
    last_progress = 0.0
    last_audio_key: str | None = None
    completed = False
    aborted = False
    completion_started: float | None = None
    started = perf_counter()
    snapshot = protocol.update(started, ready=False)

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    if use_fullscreen:
        cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(
            _WINDOW_NAME,
            guided_config.canvas_width,
            guided_config.canvas_height,
        )
    cv2.imshow(
        _WINDOW_NAME,
        render_loading_screen(
            guided_config.canvas_width,
            guided_config.canvas_height,
            title="MEKASA",
            status="Preparando validacion controlada...",
        ),
    )
    cv2.waitKey(1)

    try:
        with (
            camera,
            FaceFeatureExtractor(
                face_model_path,
                min_face_detection_confidence=guided_config.min_face_detection_confidence,
                min_face_presence_confidence=guided_config.min_face_presence_confidence,
                min_tracking_confidence=guided_config.min_tracking_confidence,
            ) as extractor,
        ):
            started = perf_counter()
            snapshot = protocol.update(started, ready=False)
            while True:
                frame = camera.read()
                frames_total += 1
                rgb = cast(
                    npt.NDArray[np.uint8],
                    cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB),
                )
                observation = extractor.extract(
                    rgb,
                    sequence=frame.sequence,
                    timestamp_ns=frame.captured_at_ns,
                )
                assessment = assess_gaze_validity(observation, guided_config)
                placement = assess_face_placement(
                    observation,
                    min_face_width=guided_config.min_face_width,
                    max_face_width=guided_config.max_face_width,
                )
                gaze_usable = assessment.gaze_validity in (
                    GazeValidity.VALID,
                    GazeValidity.PARTIAL,
                )
                ready = placement.ready and gaze_usable
                raw_prediction: tuple[float, float] | None = None
                if ready:
                    try:
                        raw_prediction = calibration_model.predict_observation(
                            observation,
                            left_eye_state=assessment.left_eye_state,
                            right_eye_state=assessment.right_eye_state,
                        )
                    except ValueError:
                        raw_prediction = None
                prediction = smoother.update(raw_prediction)
                if prediction is not None:
                    frames_usable += 1
                if not placement.ready:
                    placement_invalid_frames += 1
                elif not gaze_usable:
                    gaze_unusable_frames += 1
                if prediction is not None and region_at(prediction) is None:
                    outside_prediction_frames += 1

                now = perf_counter()
                previous_stage = snapshot.stage
                snapshot = protocol.update(now, ready=ready)
                if (
                    previous_stage is RegionTrialStage.RESPONSE
                    and snapshot.stage is RegionTrialStage.FEEDBACK
                    and snapshot.completed_trial is not None
                    and snapshot.completed_trial.outcome is RegionTrialOutcome.NO_SELECTION
                    and last_candidate is not None
                    and last_progress > 0.0
                ):
                    trial_metrics[snapshot.completed_trial.trial_index]["dwell_cancellations"] += 1
                if snapshot.stage is RegionTrialStage.RESPONSE:
                    selection = selector.update(prediction, now)
                    assert snapshot.trial_index is not None
                    metrics = trial_metrics[snapshot.trial_index]
                    metrics["frames_total"] += 1
                    if prediction is not None:
                        metrics["frames_usable"] += 1
                    else:
                        metrics["frames_unusable"] += 1
                    if (
                        last_candidate is not None
                        and last_progress > 0.0
                        and not selection.selection_fired
                        and selection.candidate is not last_candidate
                    ):
                        metrics["dwell_cancellations"] += 1
                    if selection.selection_fired and selection.selected is not None:
                        snapshot = protocol.update(now, ready=ready, selected=selection.selected)
                else:
                    selection = selector.update(None, now)
                last_candidate = selection.candidate
                last_progress = selection.dwell_progress

                if (
                    snapshot.completed_trial is not None
                    and snapshot.completed_trial.trial_index not in seen_results
                ):
                    result = snapshot.completed_trial
                    seen_results.add(result.trial_index)
                    results.append(
                        {
                            **asdict(result),
                            "requested": result.requested.value,
                            "selected": result.selected.value if result.selected else None,
                            "outcome": result.outcome.value,
                            **trial_metrics[result.trial_index],
                        }
                    )

                last_audio_key = _update_audio(snapshot, speech, last_audio_key)
                frame_times.append(now)
                screen = render_controlled_region_screen(
                    frame.bgr,
                    observation,
                    assessment,
                    prediction,
                    selection,
                    snapshot,
                    guided_config,
                    _rolling_fps(frame_times),
                    placement,
                )
                cv2.imshow(_WINDOW_NAME, screen)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    aborted = True
                    break
                if snapshot.completed:
                    completed = True
                    if completion_started is None:
                        completion_started = now
                    elif now - completion_started >= guided_config.completion_display_seconds:
                        break
    finally:
        speech.close()
        cv2.destroyWindow(_WINDOW_NAME)

    duration = perf_counter() - started
    return _build_report(
        results=results,
        protocol=protocol,
        interaction_config=interaction_config,
        guided_config=guided_config,
        calibration_config=calibration_config,
        calibration_sha256=calibration_sha256,
        audio_metadata=speech.metadata,
        camera=asdict(camera.actual_settings),
        duration_seconds=duration,
        frames_total=frames_total,
        frames_usable=frames_usable,
        placement_invalid_frames=placement_invalid_frames,
        gaze_unusable_frames=gaze_unusable_frames,
        outside_prediction_frames=outside_prediction_frames,
        completed=completed,
        aborted=aborted,
    )


def _empty_trial_metrics() -> dict[str, int]:
    return {
        "frames_total": 0,
        "frames_usable": 0,
        "frames_unusable": 0,
        "dwell_cancellations": 0,
    }


def _update_audio(
    snapshot: RegionProtocolSnapshot,
    speech: SpeechBackend,
    last_key: str | None,
) -> str | None:
    key: str | None = None
    text: str | None = None
    if snapshot.stage is RegionTrialStage.PREPARATION:
        key, text = "preparation", "Ubicá tu rostro frente a la cámara."
    elif snapshot.stage is RegionTrialStage.COUNTDOWN:
        if snapshot.stage_remaining_seconds is not None:
            number = max(1, min(3, int(np.ceil(snapshot.stage_remaining_seconds))))
            key, text = f"countdown:{number}", {3: "Tres", 2: "Dos", 1: "Uno"}[number]
    elif snapshot.stage is RegionTrialStage.CUE and snapshot.requested is not None:
        key = f"trial:{snapshot.trial_index}"
        text = {
            RegionName.TOP_LEFT: "Arriba a la izquierda",
            RegionName.TOP_RIGHT: "Arriba a la derecha",
            RegionName.BOTTOM_LEFT: "Abajo a la izquierda",
            RegionName.BOTTOM_RIGHT: "Abajo a la derecha",
        }[snapshot.requested]
    elif snapshot.stage is RegionTrialStage.COMPLETE:
        key, text = "complete", "Prueba finalizada. Podés descansar."
    if key is not None and key != last_key and text is not None:
        speech.speak(text)
    return key if key is not None else last_key


def _build_report(
    *,
    results: list[dict[str, Any]],
    protocol: ControlledRegionProtocol,
    interaction_config: InteractionConfig,
    guided_config: GuidedValidationConfig,
    calibration_config: CalibrationConfig,
    calibration_sha256: str,
    audio_metadata: dict[str, Any],
    camera: dict[str, Any],
    duration_seconds: float,
    frames_total: int,
    frames_usable: int,
    placement_invalid_frames: int,
    gaze_unusable_frames: int,
    outside_prediction_frames: int,
    completed: bool,
    aborted: bool,
) -> dict[str, Any]:
    outcome_counts = Counter(item["outcome"] for item in results)
    correct = outcome_counts[RegionTrialOutcome.CORRECT.value]
    incorrect = outcome_counts[RegionTrialOutcome.INCORRECT.value]
    no_selection = outcome_counts[RegionTrialOutcome.NO_SELECTION.value]
    selected_trials = correct + incorrect
    response_times = [
        float(item["response_seconds"]) for item in results if item["response_seconds"] is not None
    ]
    correct_response_times = [
        float(item["response_seconds"])
        for item in results
        if item["outcome"] == RegionTrialOutcome.CORRECT.value
        and item["response_seconds"] is not None
    ]
    confusion: dict[str, dict[str, int]] = {}
    per_region: dict[str, dict[str, Any]] = {}
    for region in RegionName:
        region_results = [item for item in results if item["requested"] == region.value]
        selected_counts = Counter(item["selected"] or "NONE" for item in region_results)
        confusion[region.value] = {
            name: selected_counts[name]
            for name in (*[member.value for member in RegionName], "NONE")
        }
        region_correct = sum(
            item["outcome"] == RegionTrialOutcome.CORRECT.value for item in region_results
        )
        per_region[region.value] = {
            "trial_count": len(region_results),
            "correct_count": region_correct,
            "accuracy": region_correct / len(region_results) if region_results else None,
            "correct_response_seconds": _summary(
                [
                    float(item["response_seconds"])
                    for item in region_results
                    if item["outcome"] == RegionTrialOutcome.CORRECT.value
                    and item["response_seconds"] is not None
                ]
            ),
        }
    total = len(results)
    return {
        "schema_version": 1,
        "kind": "CONTROLLED_ABSTRACT_REGION_VALIDATION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": source_revision(),
        "environment": environment(),
        "camera": camera,
        "calibration_sha256": calibration_sha256,
        "audio": audio_metadata,
        "protocol": {
            "completed": completed,
            "aborted_by_operator": aborted,
            "planned_trial_count": interaction_config.validation_blocks * len(RegionName),
            "target_order_seed": interaction_config.target_order_seed,
            "target_orders": [
                [region.value for region in block] for block in protocol.planned_orders
            ],
        },
        "configuration": {
            "dwell_seconds": interaction_config.dwell_seconds,
            "cooldown_seconds": interaction_config.cooldown_seconds,
            "hysteresis_margin": interaction_config.hysteresis_margin,
            "cue_seconds": interaction_config.cue_seconds,
            "response_timeout_seconds": interaction_config.response_timeout_seconds,
            "feedback_seconds": interaction_config.feedback_seconds,
            "smoothing_alpha": calibration_config.live_smoothing_alpha,
            "min_face_width": guided_config.min_face_width,
            "max_face_width": guided_config.max_face_width,
        },
        "measurement": {
            "duration_seconds": duration_seconds,
            "frames_total": frames_total,
            "frames_usable": frames_usable,
            "usable_fraction": frames_usable / frames_total if frames_total else None,
            "placement_invalid_frames": placement_invalid_frames,
            "gaze_unusable_frames": gaze_unusable_frames,
            "outside_prediction_frames": outside_prediction_frames,
            "completed_trial_count": total,
            "correct_count": correct,
            "incorrect_count": incorrect,
            "no_selection_count": no_selection,
            "accuracy_all_trials": correct / total if total else None,
            "accuracy_when_selected": correct / selected_trials if selected_trials else None,
            "selection_completion_rate": selected_trials / total if total else None,
            "dwell_cancellation_count": sum(int(item["dwell_cancellations"]) for item in results),
            "selected_response_seconds": _summary(response_times),
            "correct_response_seconds": _summary(correct_response_times),
            "per_requested_region": per_region,
            "confusion_matrix": confusion,
            "trials": results,
        },
        "privacy": {
            "video_persisted": False,
            "frames_persisted": False,
            "raw_landmarks_persisted": False,
            "derived_trial_results_persisted": True,
        },
        "limitations": [
            "Requested regions are technical ground truth prompts, not communication intents.",
            "One local session does not establish cross-session or clinical reliability.",
            "Dwell and timing values are experimental engineering defaults.",
            "The test does not control a cursor, AAC board or external system.",
        ],
    }


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(np.ceil(0.9 * len(ordered))) - 1)
    return {
        "count": len(values),
        "mean": fmean(values),
        "median": median(values),
        "p90": ordered[p90_index],
        "max": max(values),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run audio-guided controlled validation of four abstract gaze regions"
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("work/calibration/personal-gaze-model.json"),
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        default=Path("models/gaze/mediapipe-face-landmarker/face_landmarker.task"),
    )
    parser.add_argument(
        "--guided-config",
        type=Path,
        default=Path("config/experiment-001.toml"),
    )
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=Path("config/experiment-002.toml"),
    )
    parser.add_argument(
        "--interaction-config",
        type=Path,
        default=Path("config/experiment-003.toml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/interaction/controlled-region-validation-latest.json"),
    )
    parser.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_controlled_region_validation(
            guided_config=GuidedValidationConfig.load(args.guided_config),
            calibration_config=CalibrationConfig.load(args.calibration_config),
            interaction_config=InteractionConfig.load(args.interaction_config),
            calibration_model=load_calibration_model(args.calibration),
            face_model_path=args.face_model,
            calibration_sha256=_sha256(args.calibration),
            fullscreen=args.fullscreen,
        )
        _write_report(report, args.output)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        history_path = (
            args.output.parent / "history" / (f"controlled-region-validation-{timestamp}.json")
        )
        _write_report(report, history_path)
    except (CameraUnavailableError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Controlled region validation could not run: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Controlled region validation interrupted.", file=sys.stderr)
        return 130
    print(
        json.dumps(
            {
                "report": str(args.output),
                "history_report": str(history_path),
                "protocol": report["protocol"],
                "measurement": report["measurement"],
                "warning": "Technical experiment only; not AAC or clinical validation.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
