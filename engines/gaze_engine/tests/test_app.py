from __future__ import annotations

import os
from pathlib import Path

import pytest

from mekasa_gaze import app
from mekasa_gaze.services.storage import directory_size, enforce_budget


def _write(path: Path, size: int, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))


def test_budget_removes_oldest_history_first_and_keeps_current_files(tmp_path: Path) -> None:
    work = tmp_path / "work"
    current = work / "calibration" / "session-gaze-model-both.json"
    latest = work / "communication" / "board-session-latest.json"
    _write(current, 100, 1)  # oldest of all, but never a candidate
    _write(latest, 100, 2)
    old = work / "communication" / "history" / "board-session-a.json"
    middle = work / "calibration" / "history" / "s1" / "session-gaze-model-both.json"
    new = work / "communication" / "history" / "board-session-c.json"
    _write(old, 100, 10)
    _write(middle, 100, 20)
    _write(new, 100, 30)

    report = enforce_budget(work, 350)

    assert report.used_bytes == 500
    assert report.removed_files == 2
    assert report.used_after_bytes == 300
    assert not old.exists() and not middle.exists()
    assert new.exists() and current.exists() and latest.exists()
    assert not middle.parent.exists()  # empty history folder cleaned up
    assert directory_size(work) == 300


def test_budget_never_deletes_outside_history_even_if_over(tmp_path: Path) -> None:
    work = tmp_path / "work"
    keep = work / "calibration" / "session-gaze-model-left.json"
    _write(keep, 1000, 1)
    report = enforce_budget(work, 10)
    assert report.removed_files == 0
    assert keep.exists()


def test_budget_within_limit_touches_nothing(tmp_path: Path) -> None:
    work = tmp_path / "work"
    log = work / "logs" / "mekasa-1.log"
    _write(log, 10, 1)
    assert enforce_budget(work, 1000).removed_files == 0
    assert log.exists()
    with pytest.raises(ValueError):
        enforce_budget(work, 0)


def test_prepare_data_dir_copies_defaults_once_and_keeps_user_edits(tmp_path: Path) -> None:
    resources, data = tmp_path / "bundle", tmp_path / "data"
    (resources / "config").mkdir(parents=True)
    (resources / "config" / "vocabulario.txt").write_text("default", encoding="utf-8")
    (resources / "config" / "communication.toml").write_text("", encoding="utf-8")
    model = resources / "models/gaze/mediapipe-face-landmarker/face_landmarker.task"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")

    copied = app.prepare_data_dir(resources, data)
    assert "config/vocabulario.txt" in copied
    assert (data / "models/gaze/mediapipe-face-landmarker/face_landmarker.task").exists()
    assert (data / "work").is_dir()

    (data / "config" / "vocabulario.txt").write_text("editado por la familia", encoding="utf-8")
    assert app.prepare_data_dir(resources, data) == []
    assert (data / "config" / "vocabulario.txt").read_text(encoding="utf-8") == (
        "editado por la familia"
    )


def test_storage_settings_defaults_and_section(tmp_path: Path) -> None:
    config = tmp_path / "communication.toml"
    assert app.storage_settings(config) == (512 * 1024 * 1024, 200 * 1024 * 1024)
    config.write_text("[storage]\nmax_work_mb = 2\nmin_free_disk_mb = 0\n", encoding="utf-8")
    assert app.storage_settings(config) == (2 * 1024 * 1024, 0)
    config.write_text("[storage]\nmax_work_mb = 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        app.storage_settings(config)


def test_shipped_config_has_storage_section() -> None:
    config = app.resource_root() / "config" / "communication.toml"
    budget, _ = app.storage_settings(config)
    assert budget == 512 * 1024 * 1024


def test_data_root_honours_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEKASA_HOME", str(tmp_path / "home"))
    assert app.data_root() == tmp_path / "home"


def test_main_runs_simulation_in_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("MEKASA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def fake_board(argv: list[str]) -> int:
        calls.append(argv)
        assert Path.cwd() == home
        return 0

    monkeypatch.setattr("mekasa_gaze.runners.communication_board.main", fake_board)
    assert app.main(["simular"]) == 0
    assert app.main(["--dwell", "2"]) == 0
    assert app.main(["prueba"]) == 0
    assert calls == [["--simulate"], ["--dwell", "2"], ["--test", "2"]]
    assert (home / "config" / "communication.toml").exists()
    assert app.main(["desconocido"]) == 2
