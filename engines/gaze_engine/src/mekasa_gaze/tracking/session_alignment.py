"""Quick per-session alignment: a few seconds looking at known points before using the board.

Posture, distance and lighting shift between sessions, so a calibrated model drifts. The
controlled region validation already showed that a session-local affine correction removes
most of that drift. This procedure collects predictions while the person looks at the centre
of each board option (an independent, instructed label), then fits a 2D affine ridge map
``predicted -> instructed`` that is applied to every later prediction of the session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class AffineCorrection:
    """2D affine map fitted with ridge regularization on standardized inputs."""

    input_mean: npt.NDArray[np.float64]
    input_scale: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    intercept: npt.NDArray[np.float64]
    training_error_median: float

    def predict(self, point: tuple[float, float]) -> tuple[float, float]:
        """Correct one screen-normalized point."""
        vector = (np.asarray(point, dtype=np.float64) - self.input_mean) / self.input_scale
        corrected = vector @ self.weights + self.intercept
        return float(corrected[0]), float(corrected[1])


def fit_affine_correction(
    samples: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    ridge_alpha: float = 1.0,
) -> AffineCorrection:
    """Fit ``predicted -> target`` from (target, predicted) pairs.

    Raises:
        ValueError: With fewer than 3 distinct targets (an affine map would be undetermined)
            or non-finite values.
    """
    if ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be non-negative")
    if len({target for target, _ in samples}) < 3:
        raise ValueError("Alignment needs samples from at least 3 distinct points")
    targets = np.asarray([target for target, _ in samples], dtype=np.float64)
    inputs = np.asarray([predicted for _, predicted in samples], dtype=np.float64)
    if not (np.all(np.isfinite(inputs)) and np.all(np.isfinite(targets))):
        raise ValueError("Alignment samples must be finite")
    mean = inputs.mean(axis=0)
    scale = inputs.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    standardized = (inputs - mean) / scale
    intercept = targets.mean(axis=0)
    gram = standardized.T @ standardized + ridge_alpha * np.eye(2)
    weights = np.linalg.solve(gram, standardized.T @ (targets - intercept))
    errors = np.linalg.norm(standardized @ weights + intercept - targets, axis=1)
    return AffineCorrection(mean, scale, weights, intercept, float(np.median(errors)))


@dataclass(frozen=True, slots=True)
class AlignmentSnapshot:
    """What to show during alignment."""

    index: int
    total: int
    target: tuple[float, float] | None
    collecting: bool
    progress: float
    done: bool
    correction: AffineCorrection | None = None
    message: str = ""


@dataclass(slots=True)
class AlignmentProcedure:
    """Clock-injected sequence: for each point, settle (eyes move), then collect samples.

    Attributes:
        targets: Screen-normalized points to look at, in order.
        settle_seconds: Time to move the eyes before collecting.
        collect_seconds: Collection window once enough valid samples may arrive.
        max_collect_seconds: Give up on a point after this long without enough samples.
        min_samples: Valid predictions needed per point.
    """

    targets: list[tuple[float, float]]
    settle_seconds: float = 0.8
    collect_seconds: float = 1.2
    max_collect_seconds: float = 3.0
    min_samples: int = 12
    ridge_alpha: float = 1.0
    _index: int = 0
    _started_at: float | None = None
    _samples: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    _point_samples: int = 0
    _failed: list[int] = field(default_factory=list)
    _result: AlignmentSnapshot | None = None

    def update(self, now: float, prediction: tuple[float, float] | None) -> AlignmentSnapshot:
        """Advance with one tick; ``prediction`` is the uncorrected smoothed estimate."""
        if self._result is not None:
            return self._result
        if self._started_at is None:
            self._started_at = now
        elapsed = now - self._started_at
        target = self.targets[self._index]
        collecting = elapsed >= self.settle_seconds
        if collecting and prediction is not None:
            self._samples.append((target, prediction))
            self._point_samples += 1
        collected_for = max(0.0, elapsed - self.settle_seconds)
        enough = self._point_samples >= self.min_samples
        if (collected_for >= self.collect_seconds and enough) or (
            collected_for >= self.max_collect_seconds
        ):
            if not enough:
                self._failed.append(self._index)
            self._index += 1
            self._started_at = now
            self._point_samples = 0
            if self._index >= len(self.targets):
                return self._finish()
            target = self.targets[self._index]
            collecting, collected_for = False, 0.0
        progress = min(1.0, collected_for / self.collect_seconds) if collecting else 0.0
        return AlignmentSnapshot(
            self._index, len(self.targets), target, collecting, progress, False
        )

    def _finish(self) -> AlignmentSnapshot:
        correction: AffineCorrection | None = None
        message = ""
        usable = [
            sample for sample in self._samples if self.targets.index(sample[0]) not in self._failed
        ]
        try:
            correction = fit_affine_correction(usable, ridge_alpha=self.ridge_alpha)
            message = f"Ajuste listo ({len(self.targets) - len(self._failed)} puntos)"
        except ValueError:
            message = "Ajuste no disponible: poca senal ocular"
        self._result = AlignmentSnapshot(
            len(self.targets), len(self.targets), None, False, 1.0, True, correction, message
        )
        return self._result


__all__ = [
    "AffineCorrection",
    "AlignmentProcedure",
    "AlignmentSnapshot",
    "fit_affine_correction",
]
