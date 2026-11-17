"""Local storage budget for session logs, reports and calibration history.

Mekasa keeps every session report and every calibration under ``work/`` (nothing leaves the
computer). Over months that grows without bound, so the installed program reserves a fixed
amount of disk for it: when the budget is exceeded, the oldest files in ``history`` and
``logs`` folders are removed first. The files the board needs to start (the latest session
models, wink profile and latest report) are never removed.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

# Only files inside a ``history`` or ``logs`` folder are candidates for deletion.
_PRUNABLE_DIRS = frozenset({"history", "logs"})


@dataclass(frozen=True, slots=True)
class StorageReport:
    """What the budget check found and did."""

    used_bytes: int
    budget_bytes: int
    removed_files: int
    removed_bytes: int
    free_disk_bytes: int

    @property
    def used_after_bytes(self) -> int:
        return self.used_bytes - self.removed_bytes

    def to_dict(self) -> dict[str, int]:
        return {
            "used_bytes": self.used_bytes,
            "budget_bytes": self.budget_bytes,
            "removed_files": self.removed_files,
            "removed_bytes": self.removed_bytes,
            "used_after_bytes": self.used_after_bytes,
            "free_disk_bytes": self.free_disk_bytes,
        }


def _files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()] if root.exists() else []


def directory_size(root: Path) -> int:
    """Total bytes of all files below ``root`` (0 if it does not exist)."""
    return sum(path.stat().st_size for path in _files(root))


def enforce_budget(root: Path, budget_bytes: int) -> StorageReport:
    """Delete the oldest history files below ``root`` until it fits in ``budget_bytes``.

    Files outside ``history`` folders (current models, latest report) are kept even if the
    budget cannot be met with history alone. Empty history folders left behind are removed.
    """
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    files = _files(root)
    used = sum(path.stat().st_size for path in files)
    removed_files = 0
    removed_bytes = 0
    if used > budget_bytes:
        history = sorted(
            (path for path in files if _PRUNABLE_DIRS & set(path.relative_to(root).parts[:-1])),
            key=lambda path: path.stat().st_mtime,
        )
        for path in history:
            if used - removed_bytes <= budget_bytes:
                break
            size = path.stat().st_size
            try:
                path.unlink()
            except OSError:
                continue
            removed_files += 1
            removed_bytes += size
        _remove_empty_history_dirs(root)
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    return StorageReport(used, budget_bytes, removed_files, removed_bytes, free)


def _remove_empty_history_dirs(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        if not _PRUNABLE_DIRS & set(directory.relative_to(root).parts):
            continue
        try:
            directory.rmdir()  # only succeeds when empty
        except OSError:
            continue


__all__ = ["StorageReport", "directory_size", "enforce_budget"]
