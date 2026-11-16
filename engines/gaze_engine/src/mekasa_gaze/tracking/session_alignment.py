"""Quick per-session alignment: a few seconds looking at known points before using the board.

Posture, distance and lighting shift between sessions, so a calibrated model drifts. The
controlled region validation already showed that a session-local affine correction removes
most of that drift. This procedure collects predictions while the person looks at the centre
of each board option (an independent, instructed label), then fits a 2D affine ridge map
``predicted -> instructed`` that is applied to every later prediction of the session.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    samples: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()


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
            message = "Ajuste no disponible: poca señal ocular"
        self._result = AlignmentSnapshot(
            len(self.targets),
            len(self.targets),
            None,
            False,
            1.0,
            True,
            correction,
            message,
            tuple(usable) if correction is not None else (),
        )
        return self._result


def fit_affine_toward_identity(
    samples: Sequence[tuple[tuple[float, float], tuple[float, float], float]],
    *,
    matrix_prior: float = 0.3,
    offset_prior: float = 3.0,
    center: tuple[float, float] = (0.5, 0.5),
) -> AffineCorrection:
    """Weighted affine fit ``predicted -> target`` shrunk toward "no correction".

    ``samples`` are (target, predicted, weight). The map is ``A (p - center) + b``; priors pull
    ``A`` toward identity and ``b`` toward ``center`` (no shift). Because inputs are centred on
    the screen, a lone implicit sample is explained mostly by a partial shift, not by a
    rotation or scaling about the screen corner, while a 4-point alignment (about 120 frames
    spread over the board) is dominated by its data. ``matrix_prior`` is in squared screen
    units (0.3 ~ 13 frames at 0.15 from the centre); ``offset_prior`` is in frames.
    """
    if not samples:
        return identity_correction()
    targets = np.asarray([target for target, _, _ in samples], dtype=np.float64)
    inputs = np.asarray([predicted for _, predicted, _ in samples], dtype=np.float64)
    weights = np.asarray([weight for _, _, weight in samples], dtype=np.float64)
    if not (np.all(np.isfinite(inputs)) and np.all(np.isfinite(targets))) or np.any(weights < 0):
        raise ValueError("samples must be finite with non-negative weights")
    origin = np.asarray(center, dtype=np.float64)
    design = np.column_stack([inputs - origin, np.ones(len(inputs))])
    prior = np.diag([matrix_prior, matrix_prior, offset_prior])
    gram = design.T @ (design * weights[:, None]) + prior
    solution = np.zeros((3, 2))
    for axis in range(2):
        prior_mean = np.array(
            [1.0 if axis == 0 else 0.0, 1.0 if axis == 1 else 0.0, float(origin[axis])]
        )
        rhs = design.T @ (weights * targets[:, axis]) + prior @ prior_mean
        solution[:, axis] = np.linalg.solve(gram, rhs)
    corrected = design @ solution
    errors = np.linalg.norm(corrected - targets, axis=1)
    return AffineCorrection(
        origin, np.ones(2), solution[:2, :], solution[2, :], float(np.median(errors))
    )


def identity_correction() -> AffineCorrection:
    """A correction that changes nothing."""
    return AffineCorrection(np.zeros(2), np.ones(2), np.eye(2), np.zeros(2), 0.0)


@dataclass(slots=True)
class OnlineCorrection:
    """Session correction that keeps learning from confirmed gaze selections.

    Anchors come from an explicit alignment (instructed points). Each selection confirmed by
    gaze adds one implicit sample: the median uncorrected estimate during the dwell, labelled
    with the centre of the chosen option, i.e. the assumption that the person looked at what
    they selected (the implicit-calibration idea used by WebGazer with clicks). Wrong selections
    would teach the wrong thing, so implicit samples are light, decay with age, are capped, and
    clearing the message soon after a selection retracts its sample.
    """

    selection_weight: float = 6.0
    half_life_seconds: float = 600.0
    max_selections: int = 30
    retract_window_seconds: float = 5.0
    _anchors: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    _selections: list[tuple[tuple[float, float], tuple[float, float], float]] = field(
        default_factory=list
    )

    @property
    def implicit_samples(self) -> int:
        return len(self._selections)

    @property
    def anchored(self) -> bool:
        return bool(self._anchors)

    def reset(self) -> None:
        """Forget everything (new calibration or eye mode)."""
        self._anchors.clear()
        self._selections.clear()

    def set_anchors(
        self, samples: Sequence[tuple[tuple[float, float], tuple[float, float]]]
    ) -> None:
        """Replace the explicit alignment; implicit samples from the old pose are dropped."""
        self._anchors = list(samples)
        self._selections.clear()

    def add_selection(
        self, predicted: tuple[float, float], target: tuple[float, float], now: float
    ) -> None:
        self._selections.append((target, predicted, now))
        del self._selections[: -self.max_selections]

    def retract_recent(self, now: float) -> bool:
        """Drop the last implicit sample if it is recent (the person cleared the message)."""
        if self._selections and now - self._selections[-1][2] <= self.retract_window_seconds:
            self._selections.pop()
            return True
        return False

    def correction(self, now: float) -> AffineCorrection | None:
        """Current correction, or None when there is nothing to correct with."""
        if not self._anchors and not self._selections:
            return None
        weighted = [(target, predicted, 1.0) for target, predicted in self._anchors]
        for target, predicted, at in self._selections:
            decay = 0.5 ** (max(0.0, now - at) / self.half_life_seconds)
            weighted.append((target, predicted, self.selection_weight * decay))
        return fit_affine_toward_identity(weighted)


__all__ = [
    "AffineCorrection",
    "AlignmentProcedure",
    "AlignmentSnapshot",
    "OnlineCorrection",
    "fit_affine_correction",
    "fit_affine_toward_identity",
    "identity_correction",
]
