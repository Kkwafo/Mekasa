"""Local audible alert for urgent messages (never raises, never blocks the board loop)."""

from __future__ import annotations

import sys
import threading


def _beep_pattern() -> None:
    try:
        import winsound

        for _ in range(3):
            winsound.Beep(1200, 180)
            winsound.Beep(880, 180)
    except Exception:  # no sound device or not Windows: the spoken message still plays
        sys.stdout.write("\a")
        sys.stdout.flush()


def play_alarm() -> bool:
    """Start the alert on a background thread. Returns False if it could not start."""
    try:
        threading.Thread(target=_beep_pattern, name="mekasa-alarm", daemon=True).start()
    except RuntimeError:
        return False
    return True


__all__ = ["play_alarm"]
