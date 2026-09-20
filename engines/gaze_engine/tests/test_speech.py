from __future__ import annotations

from dataclasses import dataclass, field

from mekasa_gaze.domain import (
    ExperimentPhase,
    MeasurementStage,
    ProtocolSnapshot,
    TargetName,
    VisualTarget,
)
from mekasa_gaze.speech import AudioInstructionController


@dataclass
class FakeSpeech:
    spoken: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return True

    @property
    def metadata(self) -> dict[str, object]:
        return {"active": True, "backend": "fake"}

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def close(self) -> None:
        return


def _snapshot(
    stage: MeasurementStage,
    *,
    target: TargetName | None = None,
    segment_index: int | None = None,
) -> ProtocolSnapshot:
    visual_target = VisualTarget(target, 0.5, 0.5) if target is not None else None
    return ProtocolSnapshot(
        phase=ExperimentPhase.GAZE_DISCRIMINATION,
        stage=stage,
        target=visual_target,
        instruction="ignored",
        stage_remaining_seconds=2.5,
        protocol_elapsed_seconds=1.0,
        segment_index=segment_index,
    )


def test_audio_target_is_announced_once_per_segment() -> None:
    speech = FakeSpeech()
    controller = AudioInstructionController(speech)
    snapshot = _snapshot(MeasurementStage.TRANSITION, target=TargetName.LEFT, segment_index=3)

    controller.update(snapshot)
    controller.update(snapshot)
    controller.update(_snapshot(MeasurementStage.MEASUREMENT, target=TargetName.LEFT))

    assert speech.spoken == ["Izquierda"]


def test_countdown_announces_each_number_once() -> None:
    speech = FakeSpeech()
    controller = AudioInstructionController(speech)
    for remaining in (2.9, 2.5, 1.9, 1.2, 0.9):
        snapshot = _snapshot(MeasurementStage.COUNTDOWN)
        object.__setattr__(snapshot, "stage_remaining_seconds", remaining)
        controller.update(snapshot)

    assert speech.spoken == ["Tres", "Dos", "Uno"]
