from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mekasa_gaze.config import GuidedValidationConfig
from mekasa_gaze.domain import TargetName


def _config() -> GuidedValidationConfig:
    return GuidedValidationConfig.load(Path("config/experiment-001.toml"))


def test_guided_targets_are_normalized_and_complete() -> None:
    config = _config()
    assert {target.name for target in config.targets} == set(TargetName)
    assert config.target(TargetName.CENTER).normalized_x == pytest.approx(0.5)
    assert config.target(TargetName.LEFT).normalized_x == pytest.approx(0.15)
    assert config.target(TargetName.RIGHT).normalized_x == pytest.approx(0.85)
    assert config.target(TargetName.UP).normalized_y == pytest.approx(0.15)
    assert config.target(TargetName.DOWN).normalized_y == pytest.approx(0.85)


def test_invalid_target_coordinate_is_rejected() -> None:
    config = _config()
    invalid_target = replace(config.targets[0], normalized_x=1.1)
    invalid = replace(config, targets=(invalid_target, *config.targets[1:]))
    with pytest.raises(ValueError, match="normalized"):
        invalid.validate()
