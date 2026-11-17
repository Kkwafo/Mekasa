"""PyInstaller entry script for Mekasa.exe (see packaging/mekasa.spec)."""

from mekasa_gaze.app import main

if __name__ == "__main__":
    raise SystemExit(main())
