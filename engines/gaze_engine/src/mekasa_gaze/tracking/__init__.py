"""Mekasa Tracking: Hardware acquisition, landmark extraction,
calibration, and dwell interaction.
"""

from mekasa_gaze.tracking.calibration import (
    CalibrationModel,
    calibrate_from_report,
    load_calibration_model,
    observation_vector,
    write_calibration_artifact,
)
from mekasa_gaze.tracking.camera import (
    CameraFrame,
    CameraProvider,
    CameraSettings,
    CameraUnavailableError,
    OpenCVCameraProvider,
)
from mekasa_gaze.tracking.geometry import (
    euclidean_2d,
    euler_from_transform,
    eye_aspect_ratio,
    normalized_iris_position,
)
from mekasa_gaze.tracking.interaction import (
    DwellSelector,
    RegionName,
    SelectionUpdate,
    region_at,
    region_bounds,
)
from mekasa_gaze.tracking.landmarker import (
    FaceFeatureExtractor,
    draw_debug_overlay,
    features_from_landmarks,
)
from mekasa_gaze.tracking.placement import (
    PlacementAssessment,
    PlacementState,
    assess_face_placement,
    placement_tracking_state,
)
from mekasa_gaze.tracking.probe import probe
from mekasa_gaze.tracking.region_correction import (
    RegionPredictionCorrector,
    fit_region_prediction_corrector,
)
from mekasa_gaze.tracking.validity import (
    ValidityAssessment,
    assess_gaze_validity,
    mark_possible_blinks,
)

__all__ = [
    "CalibrationModel",
    "CameraFrame",
    "CameraProvider",
    "CameraSettings",
    "CameraUnavailableError",
    "DwellSelector",
    "FaceFeatureExtractor",
    "OpenCVCameraProvider",
    "PlacementAssessment",
    "PlacementState",
    "RegionName",
    "RegionPredictionCorrector",
    "SelectionUpdate",
    "ValidityAssessment",
    "assess_face_placement",
    "assess_gaze_validity",
    "calibrate_from_report",
    "draw_debug_overlay",
    "euclidean_2d",
    "euler_from_transform",
    "eye_aspect_ratio",
    "features_from_landmarks",
    "fit_region_prediction_corrector",
    "load_calibration_model",
    "mark_possible_blinks",
    "normalized_iris_position",
    "observation_vector",
    "placement_tracking_state",
    "probe",
    "region_at",
    "region_bounds",
    "write_calibration_artifact",
]
