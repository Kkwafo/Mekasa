# PyInstaller spec for the offline Windows build of Mekasa.
# Build with scripts/build-release.ps1 (it fetches the face model and writes build-info.json).
# Output: dist/Mekasa/ with Mekasa.exe (no console window) and Mekasa-consola.exe (console,
# for "camaras", "simular" and troubleshooting). Both run mekasa_gaze.app.
# matplotlib must stay: mediapipe.tasks.python.vision imports it at start-up.
# ruff: noqa
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent
SRC = ROOT / "engines" / "gaze_engine" / "src"
MODEL = ROOT / "models" / "gaze" / "mediapipe-face-landmarker" / "face_landmarker.task"
BUILD_INFO = ROOT / "build" / "build-info.json"

for required in (MODEL, BUILD_INFO):
    if not required.exists():
        raise SystemExit(f"Missing {required}; run scripts/build-release.ps1")

datas = [
    (str(MODEL), "models/gaze/mediapipe-face-landmarker"),
    (str(MODEL.with_name("metadata.json")), "models/gaze/mediapipe-face-landmarker"),
    (str(BUILD_INFO), "."),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "NOTICE"), "."),
    (str(ROOT / "THIRD_PARTY_LICENSES.md"), "."),
]
for name in ("communication.toml", "vocabulario.txt", "experiment-001.toml", "experiment-002.toml"):
    datas.append((str(ROOT / "config" / name), "config"))

binaries = []
hiddenimports = ["mekasa_gaze.runners.board_simulation"]
for package in ("mediapipe", "numpy"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "pytest", "IPython", "jax", "torch", "tensorflow"],
    noarchive=False,
)
pyz = PYZ(a.pure)

common = dict(
    exclude_binaries=True,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    version=None,
)
gui = EXE(pyz, a.scripts, [], name="Mekasa", console=False, **common)
console = EXE(pyz, a.scripts, [], name="Mekasa-consola", console=True, **common)

coll = COLLECT(gui, console, a.binaries, a.datas, strip=False, upx=False, name="Mekasa")
