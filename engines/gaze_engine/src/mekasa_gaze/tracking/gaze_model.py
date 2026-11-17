"""Personal gaze model trained in the same sitting it is used.

Evidence that drove this design (measured in Mekasa, 2026-11-16, two guided sessions A and B
recorded about 4 hours apart, 2x2 quadrant frame accuracy):

* Within a session a ridge model on landmark features reached 96-100 % (A) on a held-out block.
* Across sessions it collapsed: trained on A and tested on B 26 %, B on A 15-36 %. Even a
  4-point affine correction only reached 72 % (A->B). One block of 9 points recorded in the
  same sitting recovered 100 % on A.
* In session B the landmark features carried little vertical signal (vertical 63-79 %), while
  horizontal stayed at 96-100 %. That is the "bottom row does not work" symptom.

So: calibrate at the start of every use (about 35 s), add eye-appearance features that capture
the lid and sclera (see eye_patches.py), and let leave-one-round-out validation choose between
landmark, appearance and hybrid models per eye mode. Selection uses the same rounds that are
reported, so the reported cross-validated accuracy is mildly optimistic; prospective board
tests (``--prueba``) are the honest measure.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.domain import EyeState, FeatureObservation
from mekasa_gaze.core.eye_mode import EyeMode
from mekasa_gaze.tracking.eye_patches import PATCH_HEIGHT, PATCH_WIDTH, Patch
from mekasa_gaze.tracking.placement_monitor import (
    PlacementReference,
    placement_vector,
    reference_from,
)

SCHEMA = "mekasa.personal_gaze_model.v1"
KINDS = ("landmarks", "appearance", "hybrid")
ALPHAS = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
POSE_FEATURES = ("head_yaw_degrees", "head_pitch_degrees", "head_roll_degrees")
_EYE_FEATURES = ("iris_x", "iris_y", "iris_v", "eye_aspect_ratio")
_PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT
_MIN_PATCH_COVERAGE = 0.8
_BLINK_GUARD_SECONDS = 0.12
_FRAME_GAP_SECONDS = 0.08

Point = tuple[float, float]


def observation_features(observation: FeatureObservation) -> dict[str, float]:
    """Feature map including head pose under the ``head_*_degrees`` names."""
    features = {name: float(value) for name, value in observation.features.items()}
    pose = observation.head_pose
    if pose is not None:
        features["head_yaw_degrees"] = pose.yaw_degrees
        features["head_pitch_degrees"] = pose.pitch_degrees
        features["head_roll_degrees"] = pose.roll_degrees
    return features


def _eyes(mode: EyeMode) -> tuple[str, ...]:
    return {EyeMode.BOTH: ("left", "right"), EyeMode.LEFT: ("left",), EyeMode.RIGHT: ("right",)}[
        mode
    ]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Which features a model uses."""

    kind: str
    eye_mode: EyeMode

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")

    @property
    def scalar_names(self) -> tuple[str, ...]:
        eye_names: tuple[str, ...] = ()
        if self.kind in ("landmarks", "hybrid"):
            eye_names = tuple(
                f"{eye}_{name}" for eye in _eyes(self.eye_mode) for name in _EYE_FEATURES
            )
        return eye_names + POSE_FEATURES

    @property
    def patch_eyes(self) -> tuple[str, ...]:
        return _eyes(self.eye_mode) if self.kind in ("appearance", "hybrid") else ()

    @property
    def blocks(self) -> tuple[int, ...]:
        """Sizes of the feature blocks, in vector order (scalars, then each patch)."""
        return (len(self.scalar_names), *tuple(_PATCH_SIZE for _ in self.patch_eyes))

    def vectorize(
        self,
        features: Mapping[str, float],
        patches: tuple[Patch | None, Patch | None],
    ) -> npt.NDArray[np.float64] | None:
        """Feature vector, or None if anything required is missing."""
        values: list[float] = []
        for name in self.scalar_names:
            value = features.get(name)
            if value is None or not isfinite(value):
                return None
            values.append(float(value))
        parts = [np.asarray(values, dtype=np.float64)]
        for eye in self.patch_eyes:
            patch = patches[0] if eye == "left" else patches[1]
            if patch is None or patch.shape != (_PATCH_SIZE,):
                return None
            parts.append(patch.astype(np.float64))
        return np.concatenate(parts)


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """One frame recorded while the person looked at a known point (memory only)."""

    target: Point
    round_index: int  # validation fold; a head-movement segment belongs to the round it follows
    timestamp: float
    features: Mapping[str, float]
    patches: tuple[Patch | None, Patch | None]
    grid: bool = True  # False for head-movement frames: used for training, never scored


def sample_from_observation(
    observation: FeatureObservation,
    target: Point,
    round_index: int,
    timestamp: float,
    *,
    grid: bool = True,
) -> CalibrationSample:
    """Wrap an observation as a calibration sample."""
    return CalibrationSample(
        target,
        round_index,
        timestamp,
        observation_features(observation),
        observation.eye_patches,
        grid,
    )


# ----------------------------------------------------------------------------- ridge


@dataclass(frozen=True, slots=True)
class _Fit:
    mean: npt.NDArray[np.float64]
    scale: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    intercept: npt.NDArray[np.float64]

    def predict(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        result: npt.NDArray[np.float64] = ((matrix - self.mean) / self.scale) @ self.weights
        return result + self.intercept


class _RidgePath:
    """Ridge solutions for many alphas from one eigendecomposition.

    Per-dimension standardization plus per-block equalization: each block (scalar landmarks,
    each eye patch) is scaled to unit total variance so a 200-pixel patch does not drown 11
    landmark values. ``alpha`` is per sample, so the grid does not depend on frame count.
    ``eigh`` is used instead of ``solve``: it is reused for every alpha, and multithreaded
    OpenBLAS ``solve`` was ~100x slower than single-threaded on the development machine.
    """

    def __init__(
        self,
        matrix: npt.NDArray[np.float64],
        targets: npt.NDArray[np.float64],
        blocks: Sequence[int],
    ) -> None:
        self.mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std > 1e-9, std, 1.0)
        block_factor = np.concatenate([np.full(size, np.sqrt(size)) for size in blocks])
        self.scale = std * block_factor
        standardized = (matrix - self.mean) / self.scale
        self.intercept = targets.mean(axis=0)
        count = matrix.shape[0]
        eigenvalues, self._vectors = np.linalg.eigh(standardized.T @ standardized / count)
        self._eigenvalues = np.clip(eigenvalues, 0.0, None)
        self._projected = self._vectors.T @ (standardized.T @ (targets - self.intercept) / count)

    def fit(self, alpha: float) -> _Fit:
        weights = self._vectors @ (self._projected / (self._eigenvalues + alpha)[:, None])
        return _Fit(self.mean, self.scale, weights, self.intercept)


def _fit_ridge(
    matrix: npt.NDArray[np.float64],
    targets: npt.NDArray[np.float64],
    blocks: Sequence[int],
    alpha: float,
) -> _Fit:
    return _RidgePath(matrix, targets, blocks).fit(alpha)


# ----------------------------------------------------------------------------- metrics


def evaluate_predictions(
    predictions: npt.NDArray[np.float64],
    targets: npt.NDArray[np.float64],
    split: Point,
    groups: Sequence[tuple[int, Point]] | None = None,
) -> dict[str, Any]:
    """Board-relevant accuracy of screen predictions against instructed targets.

    ``split`` is the screen point that divides the 2x2 board. Targets on a midline count only
    for the axis they are not on. ``groups`` (round, target) enables fixation-level accuracy:
    the median prediction of each fixation, closer to what dwell selection does.
    """
    eps = 0.02
    dx = targets[:, 0] - split[0]
    dy = targets[:, 1] - split[1]
    horizontal = np.abs(dx) > eps
    vertical = np.abs(dy) > eps
    both = horizontal & vertical
    right_ok = (predictions[:, 0] > split[0]) == (dx > 0)
    down_ok = (predictions[:, 1] > split[1]) == (dy > 0)
    quadrant_ok = right_ok & down_ok
    errors = np.linalg.norm(predictions - targets, axis=1)

    def rate(mask: npt.NDArray[np.bool_], ok: npt.NDArray[np.bool_]) -> float | None:
        return float(ok[mask].mean()) if mask.any() else None

    per_quadrant: dict[str, float | None] = {}
    for name, sx, sy in (
        ("TOP_LEFT", -1, -1),
        ("TOP_RIGHT", 1, -1),
        ("BOTTOM_LEFT", -1, 1),
        ("BOTTOM_RIGHT", 1, 1),
    ):
        mask = both & (np.sign(dx) == sx) & (np.sign(dy) == sy)
        per_quadrant[name] = rate(mask, quadrant_ok)
    result: dict[str, Any] = {
        "frames": len(targets),
        "quadrant_accuracy": rate(both, quadrant_ok),
        "horizontal_accuracy": rate(horizontal, right_ok),
        "vertical_accuracy": rate(vertical, down_ok),
        "median_error": float(np.median(errors)) if len(errors) else None,
        "per_quadrant": per_quadrant,
    }
    if groups is not None:
        hits: list[bool] = []
        for key in sorted(set(groups)):
            mask = np.array([group == key for group in groups])
            target = np.asarray(key[1])
            if abs(target[0] - split[0]) <= eps or abs(target[1] - split[1]) <= eps:
                continue
            median = np.median(predictions[mask], axis=0)
            hits.append(
                bool(
                    ((median[0] > split[0]) == (target[0] > split[0]))
                    and ((median[1] > split[1]) == (target[1] > split[1]))
                )
            )
        result["fixation_quadrant_accuracy"] = float(np.mean(hits)) if hits else None
        result["fixations"] = len(hits)
    return result


def _score(metrics: Mapping[str, Any]) -> tuple[float, float]:
    quadrant = metrics.get("quadrant_accuracy")
    error = metrics.get("median_error")
    return (
        float(quadrant) if quadrant is not None else 0.0,
        -float(error) if error is not None else -1.0,
    )


# ----------------------------------------------------------------------------- model


@dataclass(frozen=True, slots=True)
class PersonalGazeModel:
    """Fitted personal model. ``predict_observation`` matches the board pipeline's protocol."""

    spec: FeatureSpec
    mean: npt.NDArray[np.float64]
    scale: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    intercept: npt.NDArray[np.float64]
    alpha: float
    closed_ear_threshold: float
    min_vertical_span: float
    reference: PlacementReference | None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    created_at_utc: str = ""

    @property
    def eye_mode(self) -> EyeMode:
        return self.spec.eye_mode

    @property
    def kind(self) -> str:
        return self.spec.kind

    def predict_features(
        self, features: Mapping[str, float], patches: tuple[Patch | None, Patch | None]
    ) -> Point | None:
        vector = self.spec.vectorize(features, patches)
        if vector is None:
            return None
        point = ((vector - self.mean) / self.scale) @ self.weights + self.intercept
        if not np.all(np.isfinite(point)):
            return None
        return float(point[0]), float(point[1])

    def predict_observation(
        self,
        observation: FeatureObservation,
        *,
        left_eye_state: EyeState,
        right_eye_state: EyeState,
    ) -> Point | None:
        del left_eye_state, right_eye_state  # gating happens in the pipeline
        return self.predict_features(observation_features(observation), observation.eye_patches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "created_at_utc": self.created_at_utc,
            "kind": self.spec.kind,
            "eye_mode": self.spec.eye_mode.value,
            "scalar_features": list(self.spec.scalar_names),
            "patch_eyes": list(self.spec.patch_eyes),
            "patch_shape": [PATCH_HEIGHT, PATCH_WIDTH],
            "alpha": self.alpha,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "intercept": self.intercept.tolist(),
            "closed_ear_threshold": self.closed_ear_threshold,
            "min_vertical_span": self.min_vertical_span,
            "reference_placement": self.reference.to_dict() if self.reference else None,
            "metrics": dict(self.metrics),
            "privacy": {
                "frames_persisted": False,
                "eye_patches_persisted": False,
                "note": "mean/scale of patch features describe an average 20x10 normalized eye "
                "image of the calibrated person. Local file; delete it to remove it.",
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PersonalGazeModel:
        if data.get("schema") != SCHEMA:
            raise ValueError("not a Mekasa personal gaze model")
        spec = FeatureSpec(str(data["kind"]), EyeMode(str(data["eye_mode"])))
        if list(data["scalar_features"]) != list(spec.scalar_names) or list(
            data["patch_eyes"]
        ) != list(spec.patch_eyes):
            raise ValueError("model features do not match this version of Mekasa")
        if list(data["patch_shape"]) != [PATCH_HEIGHT, PATCH_WIDTH]:
            raise ValueError("model patch size does not match this version of Mekasa")
        size = sum(spec.blocks)
        mean = np.asarray(data["mean"], dtype=np.float64)
        scale = np.asarray(data["scale"], dtype=np.float64)
        weights = np.asarray(data["weights"], dtype=np.float64)
        intercept = np.asarray(data["intercept"], dtype=np.float64)
        if mean.shape != (size,) or scale.shape != (size,) or weights.shape != (size, 2):
            raise ValueError("model arrays have unexpected shapes")
        if intercept.shape != (2,) or np.any(scale <= 0):
            raise ValueError("model arrays are invalid")
        reference = data.get("reference_placement")
        return cls(
            spec,
            mean,
            scale,
            weights,
            intercept,
            float(data["alpha"]),
            float(data["closed_ear_threshold"]),
            float(data["min_vertical_span"]),
            PlacementReference.from_dict(reference) if reference else None,
            dict(data.get("metrics", {})),
            str(data.get("created_at_utc", "")),
        )


def save_personal_model(model: PersonalGazeModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")


def load_personal_model(path: Path) -> PersonalGazeModel:
    return PersonalGazeModel.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))


# ----------------------------------------------------------------------------- fitting


@dataclass(frozen=True, slots=True)
class PersonalThresholds:
    """Eye-closure limits learned from the person's own downward gaze."""

    closed_ear_threshold: float
    min_vertical_span: float


def personal_thresholds(samples: Sequence[CalibrationSample], split_y: float) -> PersonalThresholds:
    """Closed = below 45 % of the median aperture seen while looking at the lowest row.

    Eyelids follow downward gaze (2026-11-16: EAR 0.26 at the top row, 0.12 at the bottom), so
    a population threshold of 0.15 calls downward fixations "closed". The first rule (60 %)
    still called 11-13 % of bottom-row frames "closed" when the gaze went 30 % lower than during
    calibration (raised chair, 2026-11-17); 45 % lowers that to 1-2 % while still catching 94 %
    of real blinks in session B (blink minima median 0.024). Clamped to plausible ranges.
    """
    bottom = [sample for sample in samples if sample.target[1] > split_y + 0.02]
    ears = [
        min(
            sample.features.get("left_eye_aspect_ratio", np.nan),
            sample.features.get("right_eye_aspect_ratio", np.nan),
        )
        for sample in bottom
    ]
    ears = [value for value in ears if isfinite(value)]
    spans = [
        min(
            sample.features.get("left_eye_vertical_span", np.nan),
            sample.features.get("right_eye_vertical_span", np.nan),
        )
        for sample in bottom
    ]
    spans = [value for value in spans if isfinite(value)]
    ear = float(np.clip(0.45 * np.median(ears), 0.05, 0.16)) if ears else 0.1
    span = float(np.clip(0.5 * np.percentile(spans, 10), 0.003, 0.008)) if spans else 0.006
    return PersonalThresholds(ear, span)


def usable_samples(
    samples: Sequence[CalibrationSample], mode: EyeMode, thresholds: PersonalThresholds
) -> list[CalibrationSample]:
    """Samples whose required eyes are open, minus frames near blinks."""
    eyes = _eyes(mode)

    def low(sample: CalibrationSample) -> bool:
        return any(
            not sample.features.get(f"{eye}_eye_aspect_ratio", 0.0)
            >= thresholds.closed_ear_threshold
            or not sample.features.get(f"{eye}_eye_vertical_span", 0.0)
            >= thresholds.min_vertical_span
            for eye in eyes
        )

    ordered = sorted(samples, key=lambda sample: sample.timestamp)
    suspicious = [sample.timestamp for sample in ordered if low(sample)]
    # Gaps in the frame stream inside a fixation are blinks the collector already dropped.
    for previous, current in itertools.pairwise(ordered):
        if (
            previous.target == current.target
            and previous.round_index == current.round_index
            and current.timestamp - previous.timestamp > _FRAME_GAP_SECONDS
        ):
            suspicious.extend((previous.timestamp, current.timestamp))
    marks = np.asarray(sorted(suspicious), dtype=np.float64)
    kept: list[CalibrationSample] = []
    for sample in ordered:
        if low(sample):
            continue
        if marks.size:
            index = np.searchsorted(marks, sample.timestamp)
            near = [
                abs(marks[i] - sample.timestamp) for i in (index - 1, index) if 0 <= i < marks.size
            ]
            if near and min(near) < _BLINK_GUARD_SECONDS:
                continue
        kept.append(sample)
    return kept


@dataclass(frozen=True, slots=True)
class FitResult:
    """Chosen model plus the evidence behind the choice."""

    model: PersonalGazeModel | None
    candidates: list[dict[str, Any]]
    usable_samples: int
    error: str | None = None


def _matrix(
    samples: Sequence[CalibrationSample], spec: FeatureSpec
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[CalibrationSample]]:
    rows: list[npt.NDArray[np.float64]] = []
    kept: list[CalibrationSample] = []
    for sample in samples:
        vector = spec.vectorize(sample.features, sample.patches)
        if vector is not None:
            rows.append(vector)
            kept.append(sample)
    if not rows:
        return np.empty((0, sum(spec.blocks))), np.empty((0, 2)), []
    targets = np.asarray([sample.target for sample in kept], dtype=np.float64)
    return np.vstack(rows), targets, kept


def fit_personal_model(
    samples: Sequence[CalibrationSample],
    mode: EyeMode,
    split: Point,
    *,
    kinds: Sequence[str] = KINDS,
    alphas: Sequence[float] = ALPHAS,
    min_samples_per_target: int = 8,
) -> FitResult:
    """Choose kind and alpha by leave-one-round-out validation, then refit on everything."""
    thresholds = personal_thresholds(samples, split[1])
    usable = usable_samples(samples, mode, thresholds)
    grid_samples = [sample for sample in usable if sample.grid]
    rounds = sorted({sample.round_index for sample in grid_samples})
    targets = {sample.target for sample in grid_samples}
    counts = {target: 0 for target in targets}
    for sample in grid_samples:
        counts[sample.target] += 1
    if len(targets) < 4 or min(counts.values(), default=0) < min_samples_per_target:
        return FitResult(
            None,
            [],
            len(usable),
            f"pocas muestras válidas ({len(usable)}); revise luz, distancia y que se vean los ojos",
        )
    candidates: list[dict[str, Any]] = []
    best: tuple[tuple[float, float], FeatureSpec, float, dict[str, Any]] | None = None
    for kind in kinds:
        spec = FeatureSpec(kind, mode)
        matrix, target_matrix, kept = _matrix(usable, spec)
        if len(kept) < _MIN_PATCH_COVERAGE * len(usable):
            candidates.append({"kind": kind, "skipped": "features missing in too many frames"})
            continue
        round_of = np.asarray([sample.round_index for sample in kept])
        scored = np.asarray([sample.grid for sample in kept])
        folds: list[tuple[int, _RidgePath, npt.NDArray[np.bool_]]] = []
        for held in rounds if len(rounds) >= 2 else []:
            train = round_of != held
            test = (round_of == held) & scored
            if test.any() and train.sum() >= 10:
                folds.append(
                    (held, _RidgePath(matrix[train], target_matrix[train], spec.blocks), test)
                )
        for alpha in alphas:
            if not folds:
                metrics: dict[str, Any] = {"cross_validated": False}
            else:
                predictions: list[npt.NDArray[np.float64]] = []
                truths: list[npt.NDArray[np.float64]] = []
                groups: list[tuple[int, Point]] = []
                for held, path, test in folds:
                    predictions.append(path.fit(alpha).predict(matrix[test]))
                    truths.append(target_matrix[test])
                    groups.extend((held, kept[i].target) for i in np.flatnonzero(test))
                metrics = evaluate_predictions(
                    np.vstack(predictions), np.vstack(truths), split, groups
                )
                metrics["cross_validated"] = True
            candidates.append({"kind": kind, "alpha": alpha, **metrics})
            score = _score(metrics)
            if best is None or score > best[0]:
                best = (score, spec, alpha, metrics)
    if best is None:
        return FitResult(None, candidates, len(usable), "ningún modelo pudo entrenarse")
    _, spec, alpha, metrics = best
    matrix, target_matrix, kept = _matrix(usable, spec)
    fit = _fit_ridge(matrix, target_matrix, spec.blocks, alpha)
    vectors = [vector for vector in (placement_vector(s.features) for s in kept) if vector]
    model = PersonalGazeModel(
        spec=spec,
        mean=fit.mean,
        scale=fit.scale,
        weights=fit.weights,
        intercept=fit.intercept,
        alpha=alpha,
        closed_ear_threshold=thresholds.closed_ear_threshold,
        min_vertical_span=thresholds.min_vertical_span,
        reference=reference_from(vectors),
        metrics={
            **metrics,
            "training_frames": len(kept),
            "rounds": len(rounds),
            "selection_note": "kind and alpha chosen on the same rounds that are reported; "
            "measure prospectively with --prueba",
        },
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    return FitResult(model, candidates, len(usable))


__all__ = [
    "ALPHAS",
    "KINDS",
    "CalibrationSample",
    "FeatureSpec",
    "FitResult",
    "PersonalGazeModel",
    "PersonalThresholds",
    "evaluate_predictions",
    "fit_personal_model",
    "load_personal_model",
    "observation_features",
    "personal_thresholds",
    "sample_from_observation",
    "save_personal_model",
    "usable_samples",
]
