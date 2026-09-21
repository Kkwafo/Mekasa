from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from statistics import fmean, median
from typing import Any

import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.domain import RegionName
from mekasa_gaze.tracking.interaction import region_bounds


@dataclass(frozen=True, slots=True)
class RegionPredictionCorrector:
    """Session-local affine ridge correction learned from quadrant practice points."""

    input_mean: npt.NDArray[np.float64]
    input_scale: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    output_intercept: npt.NDArray[np.float64]
    ridge_alpha: float
    sample_counts: dict[RegionName, int]
    training_error: dict[str, float | int | None]

    def predict(self, point: tuple[float, float]) -> tuple[float, float]:
        vector = np.asarray(point, dtype=np.float64)
        if vector.shape != (2,) or not np.all(np.isfinite(vector)):
            raise ValueError("Correction input must contain two finite coordinates")
        prediction = ((vector - self.input_mean) / self.input_scale) @ self.weights
        prediction += self.output_intercept
        return float(prediction[0]), float(prediction[1])

    def artifact(self) -> dict[str, Any]:
        return {
            "kind": "SESSION_REGION_RIDGE_CORRECTION_EXPERIMENTAL",
            "input": "smoothed_baseline_normalized_prediction",
            "preprocessing": {
                "input_mean": self.input_mean.tolist(),
                "input_scale": self.input_scale.tolist(),
            },
            "model": {
                "type": "ridge_regression_closed_form",
                "ridge_alpha": self.ridge_alpha,
                "weights": self.weights.tolist(),
                "output_intercept": self.output_intercept.tolist(),
            },
            "sample_counts": {region.value: self.sample_counts[region] for region in RegionName},
            "training_normalized_euclidean_error": self.training_error,
        }


def fit_region_prediction_corrector(
    samples: list[tuple[RegionName, tuple[float, float]]],
    *,
    ridge_alpha: float,
    min_samples_per_region: int,
) -> RegionPredictionCorrector:
    if ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be non-negative")
    if min_samples_per_region < 1:
        raise ValueError("min_samples_per_region must be positive")

    counts = Counter(region for region, _ in samples)
    insufficient = {
        region.value: counts[region]
        for region in RegionName
        if counts[region] < min_samples_per_region
    }
    if insufficient:
        raise ValueError(
            "Insufficient practice samples per region; "
            f"need {min_samples_per_region}, observed {insufficient}"
        )

    inputs = np.asarray([point for _, point in samples], dtype=np.float64)
    targets = np.asarray([_region_center(region) for region, _ in samples], dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[1] != 2 or not np.all(np.isfinite(inputs)):
        raise ValueError("Practice samples must contain finite two-dimensional predictions")

    input_mean = np.mean(inputs, axis=0)
    input_scale = np.std(inputs, axis=0)
    input_scale = np.where(input_scale > 1e-9, input_scale, 1.0)
    standardized = (inputs - input_mean) / input_scale
    output_intercept = np.mean(targets, axis=0)
    centered_targets = targets - output_intercept
    regularized_gram = standardized.T @ standardized + ridge_alpha * np.eye(2)
    weights = np.linalg.solve(regularized_gram, standardized.T @ centered_targets)
    predictions = standardized @ weights + output_intercept
    errors = np.linalg.norm(predictions - targets, axis=1).tolist()

    return RegionPredictionCorrector(
        input_mean=input_mean,
        input_scale=input_scale,
        weights=weights,
        output_intercept=output_intercept,
        ridge_alpha=ridge_alpha,
        sample_counts={region: counts[region] for region in RegionName},
        training_error=_summary(errors),
    )


def _region_center(region: RegionName) -> tuple[float, float]:
    left, top, right, bottom = region_bounds(region)
    return (left + right) * 0.5, (top + bottom) * 0.5


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = [value for value in values if isfinite(value)]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    ordered = sorted(finite)
    p90_index = min(len(ordered) - 1, int(np.ceil(0.9 * len(ordered))) - 1)
    return {
        "count": len(ordered),
        "mean": fmean(ordered),
        "median": median(ordered),
        "p90": ordered[p90_index],
        "max": max(ordered),
    }
