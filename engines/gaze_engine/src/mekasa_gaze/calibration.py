from __future__ import annotations

from mekasa_gaze.tracking.calibration import (
    CalibrationModel,
    calibrate_from_report,
    load_calibration_model,
    main,
    observation_vector,
    write_calibration_artifact,
)

__all__ = [
    "CalibrationModel",
    "calibrate_from_report",
    "load_calibration_model",
    "main",
    "observation_vector",
    "write_calibration_artifact",
]

if __name__ == "__main__":
    raise SystemExit(main())
