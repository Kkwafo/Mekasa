from __future__ import annotations

import json

import pytest

from mekasa_gaze.interaction import RegionName
from mekasa_gaze.region_correction import fit_region_prediction_corrector


def test_region_corrector_learns_session_affine_distortion() -> None:
    centers = {
        RegionName.TOP_LEFT: (0.25, 0.25),
        RegionName.TOP_RIGHT: (0.75, 0.25),
        RegionName.BOTTOM_LEFT: (0.25, 0.75),
        RegionName.BOTTOM_RIGHT: (0.75, 0.75),
    }
    samples: list[tuple[RegionName, tuple[float, float]]] = []
    for region, (target_x, target_y) in centers.items():
        for offset in (-0.01, -0.005, 0.0, 0.005, 0.01):
            raw = (
                1.2 * target_x + 0.1 * target_y - 0.2 + offset,
                -0.05 * target_x + 1.1 * target_y + 0.15 - offset,
            )
            samples.append((region, raw))

    corrector = fit_region_prediction_corrector(
        samples,
        ridge_alpha=0.01,
        min_samples_per_region=5,
    )

    for target in centers.values():
        raw = (
            1.2 * target[0] + 0.1 * target[1] - 0.2,
            -0.05 * target[0] + 1.1 * target[1] + 0.15,
        )
        assert corrector.predict(raw) == pytest.approx(target, abs=0.01)
    artifact = corrector.artifact()
    assert artifact["sample_counts"] == {
        "TOP_LEFT": 5,
        "TOP_RIGHT": 5,
        "BOTTOM_LEFT": 5,
        "BOTTOM_RIGHT": 5,
    }
    json.dumps(artifact, allow_nan=False)


def test_region_corrector_rejects_incomplete_practice_data() -> None:
    with pytest.raises(ValueError, match="Insufficient practice samples"):
        fit_region_prediction_corrector(
            [(RegionName.TOP_LEFT, (0.1, 0.2))],
            ridge_alpha=1.0,
            min_samples_per_region=1,
        )
