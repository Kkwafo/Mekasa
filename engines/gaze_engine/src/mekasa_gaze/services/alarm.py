"""Local audible alert for urgent messages (never raises, never blocks the board loop)."""

from __future__ import annotations

import sys
import threading


def _terminal_bell() -> None:
    if sys.stdout is not None:  # the windowed build may have no console
        sys.stdout.write("\a")
        sys.stdout.flush()


def _beep_pattern() -> None:
    # winsound exists only on Windows; the platform check also lets mypy type-check on Linux.
    if sys.platform != "win32":
        _terminal_bell()
        return
    try:
        import winsound

        for _ in range(3):
            winsound.Beep(1200, 180)
            winsound.Beep(880, 180)
    except Exception:  # no sound device: the spoken message still plays
        _terminal_bell()


def play_alarm() -> bool:
    """Start the alert on a background thread. Returns False if it could not start."""
    try:
        threading.Thread(target=_beep_pattern, name="mekasa-alarm", daemon=True).start()
    except RuntimeError:
        return False
    return True


__all__ = ["play_alarm"]
