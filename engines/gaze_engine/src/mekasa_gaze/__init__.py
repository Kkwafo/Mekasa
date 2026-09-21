"""Mekasa: Local-first gaze research and communication engine.

Structured under Clean Architecture principles:
- core: System rules, base data models, configuration schemas, protocol state machines.
- tracking: Physical camera capture, facial landmarks, geometry, calibration, dwell selection.
- ui: Visual presentation, target stimuli, boards, and interactive visual feedback.
- services: Speech synthesis (TTS), hardware/runtime inspection, telemetry metrics, reporting.
- runners: Automated experimental and validation CLI workflows.
"""

from mekasa_gaze import core, runners, services, tracking, ui
from mekasa_gaze.core.domain import FeatureObservation, HeadPose, TrackingState

__all__ = [
    "FeatureObservation",
    "HeadPose",
    "TrackingState",
    "core",
    "runners",
    "services",
    "tracking",
    "ui",
]
__version__ = "0.0.1"
