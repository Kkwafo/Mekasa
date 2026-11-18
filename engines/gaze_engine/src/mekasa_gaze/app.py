"""Entry point of the installed program (``Mekasa.exe``) and of ``mekasa``.

The research runners read ``config/``, ``models/`` and write ``work/`` relative to the current
directory (the repository). An installed program cannot write next to itself (Program Files is
read-only), so this launcher:

1. picks a per-user data folder (``%LOCALAPPDATA%\\Mekasa``, or ``MEKASA_HOME``);
2. on first run copies the default configuration and the face model there (later runs keep
   the family's edits to ``config/vocabulario.txt`` and ``config/communication.toml``);
3. keeps ``work/`` (reports, calibrations, logs) within the disk budget of ``[storage]``;
4. changes to that folder and starts the board, fully offline.

Commands (first argument): ``tablero`` (default), ``prueba``, ``demo`` (board with the camera
view, to film how it works), ``camaras``, ``simular``, ``carpeta`` (open the data folder).
Other arguments go to the board unchanged.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from mekasa_gaze import __version__
from mekasa_gaze.services.storage import StorageReport, enforce_budget

_FACE_MODEL = Path("models/gaze/mediapipe-face-landmarker/face_landmarker.task")
_CONFIG_FILES = (
    "communication.toml",
    "vocabulario.txt",
    "experiment-001.toml",
    "experiment-002.toml",
)
_DEFAULT_BUDGET_MB = 512
_DEFAULT_MIN_FREE_MB = 200
_LOG_KEEP = 30  # program logs kept besides the storage budget


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """Folder with the shipped ``config/`` and ``models/`` (bundle or repository)."""
    if frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[4]


def data_root() -> Path:
    """Per-user folder for configuration, calibrations, reports and logs."""
    override = os.environ.get("MEKASA_HOME")
    if override:
        return Path(override).expanduser()
    local = os.environ.get("LOCALAPPDATA")
    return Path(local) / "Mekasa" if local else Path.home() / ".mekasa"


def prepare_data_dir(resources: Path, data: Path) -> list[str]:
    """Copy missing defaults into ``data``. Never overwrites a file the user may have edited."""
    copied: list[str] = []
    (data / "config").mkdir(parents=True, exist_ok=True)
    (data / "work").mkdir(parents=True, exist_ok=True)
    for name in _CONFIG_FILES:
        source, target = resources / "config" / name, data / "config" / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
            copied.append(f"config/{name}")
    source, target = resources / _FACE_MODEL, data / _FACE_MODEL
    if source.exists() and (not target.exists() or target.stat().st_size != source.stat().st_size):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(str(_FACE_MODEL.as_posix()))
    return copied


def storage_settings(config: Path) -> tuple[int, int]:
    """``(budget_bytes, min_free_bytes)`` from the ``[storage]`` section, with defaults."""
    section: dict[str, Any] = {}
    try:
        with config.open("rb") as handle:
            section = tomllib.load(handle).get("storage", {})
    except (OSError, tomllib.TOMLDecodeError):
        section = {}
    budget = float(section.get("max_work_mb", _DEFAULT_BUDGET_MB))
    min_free = float(section.get("min_free_disk_mb", _DEFAULT_MIN_FREE_MB))
    if budget <= 0 or min_free < 0:
        raise ValueError("[storage] max_work_mb must be > 0 and min_free_disk_mb >= 0")
    return int(budget * 1024 * 1024), int(min_free * 1024 * 1024)


def check_storage(data: Path) -> tuple[StorageReport, str | None]:
    budget, min_free = storage_settings(data / "config" / "communication.toml")
    report = enforce_budget(data / "work", budget)
    warning = None
    if report.free_disk_bytes < min_free:
        warning = (
            f"Queda poco espacio en disco ({report.free_disk_bytes // (1024 * 1024)} MB). "
            "Mekasa sigue funcionando, pero puede no poder guardar calibraciones ni reportes."
        )
    return report, warning


def _open_log(data: Path) -> TextIO:
    logs = data / "work" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for old in sorted(logs.glob("mekasa-*.log"))[:-_LOG_KEEP]:
        with contextlib.suppress(OSError):
            old.unlink()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (logs / f"mekasa-{stamp}.log").open("a", encoding="utf-8", buffering=1)


def _message_box(text: str, title: str = "Mekasa") -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)
    except (AttributeError, OSError):
        pass


def _dispatch(args: list[str]) -> int:
    command = args[0].lower() if args and not args[0].startswith("-") else "tablero"
    rest = args[1:] if args and not args[0].startswith("-") else args
    if command == "camaras":
        from mekasa_gaze.tracking.probe import main as probe

        return probe(rest)
    if command == "carpeta":
        if sys.platform == "win32":
            os.startfile(Path.cwd())  # type: ignore[attr-defined,unused-ignore]
        print(Path.cwd())
        return 0
    from mekasa_gaze.runners.communication_board import main as board

    if command == "simular":
        return board(["--simulate", *rest])
    if command == "prueba":
        return board(["--test", "2", *rest])
    if command == "demo":  # shows the camera view for filming; V cycles corner/center/off
        return board(["--camera-view", "corner", *rest])
    if command == "tablero":
        return board(rest)
    print(f"Comando desconocido: {command}. Use tablero, prueba, demo, camaras, simular o carpeta.")
    return 2


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # MediaPipe imports matplotlib.pyplot at start-up; no plotting window is ever used, and the
    # bundle has no tkinter, so pin the non-interactive backend.
    os.environ.setdefault("MPLBACKEND", "Agg")
    data = data_root()
    data.mkdir(parents=True, exist_ok=True)
    log: TextIO | None = None
    if frozen() and (sys.stdout is None or sys.stderr is None):  # windowed build: no console
        log = _open_log(data)
        sys.stdout = sys.stderr = log
    try:
        copied = prepare_data_dir(resource_root(), data)
        os.chdir(data)
        report, warning = check_storage(data)
        print(f"Mekasa {__version__} · datos en {data}")
        if copied:
            print("Configuración inicial copiada: " + ", ".join(copied))
        if report.removed_files:
            print(
                f"Espacio reservado: se borraron {report.removed_files} archivos antiguos "
                f"({report.removed_bytes // 1024} KB) para no superar "
                f"{report.budget_bytes // (1024 * 1024)} MB."
            )
        if warning:
            print(warning)
        code = _dispatch(args)
        check_storage(data)  # the session just added reports and calibrations
    except Exception as error:
        print(f"Mekasa no pudo iniciar: {error!r}", file=sys.stderr)
        if log is not None:
            _message_box(
                f"Mekasa no pudo iniciar:\n{error}\n\nDetalles en {data / 'work' / 'logs'}"
            )
        return 2
    finally:
        if log is not None:
            log.flush()
    if code == 2 and log is not None:
        _message_box(f"Mekasa terminó con un error. Detalles en {data / 'work' / 'logs'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
