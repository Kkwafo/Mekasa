from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def package_version(name: str) -> str | None:
    """Retrieve installed distribution version for a Python package name.

    Responsibility:
        Safely inspects importlib metadata without raising PackageNotFoundError if absent.

    Args:
        name: Distribution package name (e.g. 'mediapipe', 'numpy').

    Returns:
        str | None: Installed version string if found, None otherwise.
    """
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment() -> dict[str, Any]:
    """Capture host runtime environment details for experiment provenance.

    Responsibility:
        Collects Python version, OS platform string, CPU processor name, logical core count,
        and pinned versions of critical computer vision / ML libraries.

    Returns:
        dict[str, Any]: Environment metadata dictionary.
    """
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "packages": {
            name: package_version(name)
            for name in ("mediapipe", "numpy", "opencv-contrib-python", "psutil")
        },
    }


def source_revision() -> dict[str, str | bool | None]:
    """Query git repository state for current commit hash and uncommitted changes flag.

    Responsibility:
        Executes local git rev-parse and status commands to capture version provenance.

    Returns:
        dict[str, str | bool | None]: Mapping with keys 'commit' (SHA or None) and 'dirty' (bool).
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def model_identity(model_path: Path) -> dict[str, str | int]:
    """Compute cryptographic hash and file size of a model binary file.

    Responsibility:
        Reads model file in 1MB chunks to compute SHA-256 digest and file size for integrity.

    Args:
        model_path: Filesystem path to the model binary.

    Returns:
        dict[str, str | int]: Mapping with 'filename', 'size_bytes', and 'sha256'.
    """
    digest = hashlib.sha256()
    with model_path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "filename": model_path.name,
        "size_bytes": model_path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
