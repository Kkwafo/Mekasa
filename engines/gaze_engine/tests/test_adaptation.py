"""Safe continuous adaptation: per-axis, confirmed selections, posture-aware fading."""

from __future__ import annotations

import pytest

from mekasa_gaze.tracking.adaptation import (
    AdaptiveCorrection,
    SelectionTelemetry,
    fit_axis_correction,
    posture_distance,
)
from mekasa_gaze.tracking.placement_monitor import PlacementReference

POSTURE = PlacementReference(0.40, 0.5, 0.45, 0.0, 5.0)
MOVED = PlacementReference(0.40, 0.5, 0.33, 0.0, 12.0)  # chair raised: face higher, head tilted


def test_axis_fit_never_rotates_and_needs_both_sides_to_stretch() -> None:
    # Two options on one diagonal: a full affine map could flip the other diagonal.
    samples = [((0.25, 0.25), (0.3, 0.3), 6.0), ((0.75, 0.75), (0.7, 0.7), 6.0)]
    correction = fit_axis_correction(samples)
    assert correction.weights[0, 1] == 0 and correction.weights[1, 0] == 0
    x, y = correction.predict((0.7, 0.3))  # the other diagonal stays where it is
    assert x > 0.5 and y < 0.5
    # Bottom row squeezed up: evidence from both rows stretches the vertical axis.
    squeezed = [((0.25, 0.3), (0.25, 0.35), 6.0), ((0.25, 0.8), (0.25, 0.6), 6.0)]
    assert fit_axis_correction(squeezed).weights[1, 1] > 1.1


def test_one_sided_evidence_only_shifts_a_little() -> None:
    # Only top-row selections, estimated too low: the old version moved everything up, which
    # pushed the failing bottom row further away. Now the shift is capped at a third.
    top_only = [((0.25, 0.25), (0.25, 0.40), 6.0)] * 5
    correction = fit_axis_correction(top_only)
    assert correction.weights[1, 1] == pytest.approx(1.0)
    assert abs(correction.predict((0.5, 0.6))[1] - 0.6) <= 0.04 + 1e-9


def test_selections_count_only_after_confirmation_and_can_be_retracted() -> None:
    adaptive = AdaptiveCorrection(confirm_seconds=4.0)
    adaptive.add_selection((0.25, 0.40), (0.25, 0.25), 10.0, POSTURE)
    adaptive.add_selection((0.25, 0.70), (0.25, 0.80), 11.0, POSTURE)
    assert adaptive.correction(12.0, POSTURE) is None  # not confirmed yet
    assert adaptive.retract_recent(13.0)  # the person went back right after choosing
    assert adaptive.implicit_samples == 1
    later = adaptive.correction(20.0, POSTURE)
    assert later is not None
    assert not adaptive.retract_recent(20.0)  # confirmed choices are not retracted


def test_evidence_from_another_posture_fades() -> None:
    assert posture_distance(POSTURE, POSTURE) == 0.0
    assert posture_distance(POSTURE, MOVED) > 2.0
    adaptive = AdaptiveCorrection(confirm_seconds=0.0)
    adaptive.add_selection((0.25, 0.35), (0.25, 0.25), 0.0, POSTURE)
    adaptive.add_selection((0.25, 0.65), (0.25, 0.80), 0.0, POSTURE)
    same = adaptive.correction(1.0, POSTURE)
    moved = adaptive.correction(1.0, MOVED)
    assert same is not None
    assert moved is None or abs(moved.predict((0.25, 0.65))[1] - 0.65) < abs(
        same.predict((0.25, 0.65))[1] - 0.65
    )


def test_telemetry_keeps_only_row_offsets() -> None:
    telemetry = SelectionTelemetry()
    telemetry.add((0.26, 0.31), (0.25, 0.25), 0.59)
    telemetry.add((0.24, 0.66), (0.25, 0.80), 0.59)
    summary = telemetry.summary()
    assert summary["top"] == {
        "selections": 1,
        "median_dx": pytest.approx(0.01),
        "median_dy": pytest.approx(0.06),
    }
    assert summary["bottom"]["median_dy"] == pytest.approx(-0.14)  # type: ignore[index]
