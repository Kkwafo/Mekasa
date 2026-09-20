from __future__ import annotations

import pytest

from mekasa_gaze.interaction import DwellSelector, RegionName, region_at


def _selector() -> DwellSelector:
    return DwellSelector(dwell_seconds=1.0, cooldown_seconds=0.5, hysteresis_margin=0.06)


def test_region_mapping_rejects_predictions_outside_screen() -> None:
    assert region_at((0.2, 0.2)) is RegionName.TOP_LEFT
    assert region_at((0.8, 0.2)) is RegionName.TOP_RIGHT
    assert region_at((0.2, 0.8)) is RegionName.BOTTOM_LEFT
    assert region_at((0.8, 0.8)) is RegionName.BOTTOM_RIGHT
    assert region_at((-0.1, 0.5)) is None


def test_dwell_fires_once_then_enforces_cooldown() -> None:
    selector = _selector()

    first = selector.update((0.2, 0.2), 10.0)
    middle = selector.update((0.2, 0.2), 10.5)
    fired = selector.update((0.2, 0.2), 11.0)
    cooling = selector.update((0.2, 0.2), 11.2)
    restarted = selector.update((0.2, 0.2), 11.6)

    assert first.dwell_progress == 0.0
    assert middle.dwell_progress == pytest.approx(0.5)
    assert fired.selection_fired
    assert fired.selected is RegionName.TOP_LEFT
    assert not cooling.selection_fired
    assert cooling.cooldown_remaining_seconds == pytest.approx(0.3)
    assert restarted.candidate is RegionName.TOP_LEFT
    assert restarted.dwell_progress == 0.0


def test_hysteresis_tolerates_boundary_jitter_but_tracking_loss_resets() -> None:
    selector = _selector()

    selector.update((0.49, 0.25), 1.0)
    jitter = selector.update((0.53, 0.25), 1.5)
    lost = selector.update(None, 1.6)
    after_loss = selector.update((0.49, 0.25), 1.7)

    assert jitter.candidate is RegionName.TOP_LEFT
    assert jitter.dwell_progress == pytest.approx(0.5)
    assert not lost.tracking_available
    assert after_loss.dwell_progress == 0.0
