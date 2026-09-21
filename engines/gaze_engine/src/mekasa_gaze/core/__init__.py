"""Mekasa Core: Domain entities, configurations, state machines, and interfaces."""

from mekasa_gaze.core.config import (
    CalibrationConfig,
    ExperimentConfig,
    GuidedValidationConfig,
    InteractionConfig,
)
from mekasa_gaze.core.domain import (
    CameraFrame,
    CameraSettings,
    ExperimentPhase,
    EyeState,
    FeatureObservation,
    GazeValidity,
    GuidedObservation,
    HeadPose,
    MeasurementStage,
    ProtocolSnapshot,
    RegionName,
    TargetName,
    TrackingState,
    VisualTarget,
)
from mekasa_gaze.core.interfaces import (
    CameraProvider,
    FeatureExtractor,
    SpeechBackend,
)
from mekasa_gaze.core.protocol import GuidedProtocol, ProtocolSegment
from mekasa_gaze.core.region_protocol import (
    ControlledRegionProtocol,
    RegionProtocolSnapshot,
    RegionTrialOutcome,
    RegionTrialResult,
    RegionTrialStage,
)

__all__ = [
    "CalibrationConfig",
    "CameraFrame",
    "CameraProvider",
    "CameraSettings",
    "ControlledRegionProtocol",
    "ExperimentConfig",
    "ExperimentPhase",
    "EyeState",
    "FeatureExtractor",
    "FeatureObservation",
    "GazeValidity",
    "GuidedObservation",
    "GuidedProtocol",
    "GuidedValidationConfig",
    "HeadPose",
    "InteractionConfig",
    "MeasurementStage",
    "ProtocolSegment",
    "ProtocolSnapshot",
    "RegionName",
    "RegionProtocolSnapshot",
    "RegionTrialOutcome",
    "RegionTrialResult",
    "RegionTrialStage",
    "SpeechBackend",
    "TargetName",
    "TrackingState",
    "VisualTarget",
]
