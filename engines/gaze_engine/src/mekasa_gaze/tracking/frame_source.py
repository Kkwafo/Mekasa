"""Background capture with a single-slot "latest frame" buffer and reconnection.

Capture runs on its own thread so inference and UI never block on the camera driver.
Only the most recent frame is kept (a bounded queue of one): older frames are dropped and
counted instead of accumulating latency. A failed read marks the camera as disconnected and
the thread retries opening it with a bounded backoff, so unplugging and re-plugging a USB
camera recovers without restarting the application.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic, sleep

from mekasa_gaze.core.domain import CameraFrame
from mekasa_gaze.core.interfaces import CameraProvider


@dataclass(frozen=True, slots=True)
class CaptureStatus:
    """Snapshot of capture health for diagnostics and reports."""

    connected: bool
    frames_captured: int
    frames_dropped: int
    reconnect_attempts: int
    last_error: str | None


class LatestFrameGrabber:
    """Owns a :class:`CameraProvider` on a background thread and exposes the newest frame."""

    def __init__(
        self,
        camera_factory: Callable[[], CameraProvider],
        *,
        retry_seconds: float = 1.0,
        max_retry_seconds: float = 5.0,
    ) -> None:
        """Create a grabber.

        Args:
            camera_factory: Builds a fresh (unopened) camera provider for every (re)connect.
            retry_seconds: Initial delay before reopening after a failure.
            max_retry_seconds: Upper bound for the exponential reconnect backoff.
        """
        self._factory = camera_factory
        self._retry_seconds = retry_seconds
        self._max_retry_seconds = max_retry_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: CameraFrame | None = None
        self._consumed_sequence: int | None = None
        self._connected = False
        self._captured = 0
        self._dropped = 0
        self._reconnects = 0
        self._last_error: str | None = None
        self._settings: object | None = None

    def start(self) -> None:
        """Start the capture thread (idempotent)."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mekasa-capture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Stop capturing and release the camera."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def __enter__(self) -> LatestFrameGrabber:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def take_new(self) -> CameraFrame | None:
        """Return the newest frame if it has not been returned before, else ``None``."""
        with self._lock:
            frame = self._latest
            if frame is None or frame.sequence == self._consumed_sequence:
                return None
            self._consumed_sequence = frame.sequence
            return frame

    @property
    def camera_settings(self) -> object | None:
        """Negotiated settings of the currently open camera, if any."""
        with self._lock:
            return self._settings

    def status(self) -> CaptureStatus:
        """Return a thread-safe snapshot of capture health."""
        with self._lock:
            return CaptureStatus(
                connected=self._connected,
                frames_captured=self._captured,
                frames_dropped=self._dropped,
                reconnect_attempts=self._reconnects,
                last_error=self._last_error,
            )

    def _run(self) -> None:
        delay = self._retry_seconds
        first = True
        while not self._stop.is_set():
            if not first:
                with self._lock:
                    self._reconnects += 1
            first = False
            camera = self._factory()
            try:
                camera.open()
                with self._lock:
                    # last_error is kept after a successful reopen: it is diagnostic history.
                    self._connected = True
                    self._settings = camera.actual_settings
                delay = self._retry_seconds
                while not self._stop.is_set():
                    frame = camera.read()
                    with self._lock:
                        if (
                            self._latest is not None
                            and self._latest.sequence != self._consumed_sequence
                        ):
                            self._dropped += 1
                        # Sequence numbers restart per provider; offset keeps them unique.
                        self._latest = CameraFrame(
                            sequence=self._captured,
                            captured_at_ns=frame.captured_at_ns,
                            capture_latency_ms=frame.capture_latency_ms,
                            bgr=frame.bgr,
                        )
                        self._captured += 1
            except Exception as error:  # driver errors vary by backend; never crash the UI
                with self._lock:
                    self._last_error = f"{type(error).__name__}: {error}"
            finally:
                with self._lock:
                    self._connected = False
                try:
                    camera.close()
                except Exception as error:
                    with self._lock:
                        self._last_error = f"close failed: {error}"
            if self._stop.is_set():
                break
            deadline = monotonic() + delay
            while not self._stop.is_set() and monotonic() < deadline:
                sleep(0.05)
            delay = min(self._max_retry_seconds, delay * 2.0)


__all__ = ["CaptureStatus", "LatestFrameGrabber"]
