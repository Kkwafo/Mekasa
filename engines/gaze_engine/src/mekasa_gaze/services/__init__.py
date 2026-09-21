"""Mekasa Services: Text-to-speech, hardware diagnostics, metrics telemetry, and reporting."""

from mekasa_gaze.services.metrics import ExperimentMetrics
from mekasa_gaze.services.reporting import (
    GuidedReportBuilder,
    summarize_guided_observations,
    write_guided_report,
)
from mekasa_gaze.services.runtime import (
    environment,
    model_identity,
    package_version,
    source_revision,
)
from mekasa_gaze.services.speech import (
    AudioInstructionController,
    DisabledSpeech,
    SpeechBackend,
    WindowsSapiSpeech,
    create_speech_backend,
)

__all__ = [
    "AudioInstructionController",
    "DisabledSpeech",
    "ExperimentMetrics",
    "GuidedReportBuilder",
    "SpeechBackend",
    "WindowsSapiSpeech",
    "create_speech_backend",
    "environment",
    "model_identity",
    "package_version",
    "source_revision",
    "summarize_guided_observations",
    "write_guided_report",
]
