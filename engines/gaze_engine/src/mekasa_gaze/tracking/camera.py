from __future__ import annotations

from time import perf_counter_ns
from typing import Any

import cv2

from mekasa_gaze.core.domain import CameraFrame, CameraSettings
from mekasa_gaze.core.interfaces import CameraProvider


class CameraUnavailableError(RuntimeError):
    """Raised when a requested camera cannot provide frames."""


_BACKEND_NAMES = {
    "auto": cv2.CAP_ANY,
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
}


class OpenCVCameraProvider:
    """Acquires video frames from physical or virtual camera devices using OpenCV.

    Responsibility:
        Interfaces directly with OS video subsystems (MSMF, DirectShow, V4L2) via
        OpenCV VideoCapture to capture real-time BGR image frames, track hardware latency,
        and query actual sensor dimensions and frame rates.
    """

    def __init__(
        self,
        *,
        index: int,
        backend: str,
        width: int,
        height: int,
        target_fps: float,
    ) -> None:
        """Initialize camera provider configuration without opening the device.

        Args:
            index: Zero-indexed integer identifier of the video capture device.
            backend: Backend driver identifier ('auto', 'msmf', or 'dshow').
            width: Desired horizontal pixel resolution.
            height: Desired vertical pixel resolution.
            target_fps: Desired capture frame rate in frames per second.

        Raises:
            ValueError: If the requested backend string is not supported.
        """
        if backend not in _BACKEND_NAMES:
            choices = ", ".join(sorted(_BACKEND_NAMES))
            raise ValueError(f"Unknown camera backend {backend!r}; choose one of: {choices}")
        self._requested = CameraSettings(index, backend, width, height, target_fps)
        self._capture: Any | None = None
        self._sequence = 0
        self._actual: CameraSettings | None = None

    @property
    def actual_settings(self) -> CameraSettings:
        """Return the actual hardware capture settings negotiated with the camera driver.

        Returns:
            CameraSettings: Negotiated resolution, framerate, and active backend driver.

        Raises:
            CameraUnavailableError: If accessed before `open()` has succeeded.
        """
        if self._actual is None:
            raise CameraUnavailableError("Camera has not been opened")
        return self._actual

    def open(self) -> None:
        """Open the physical video device and negotiate capture properties.

        Side Effects:
            Allocates OS video device handles and locks the camera resource.

        Raises:
            CameraUnavailableError: If the device cannot be opened or initialized.
        """
        capture = cv2.VideoCapture(self._requested.index, _BACKEND_NAMES[self._requested.backend])
        if not capture.isOpened():
            capture.release()
            raise CameraUnavailableError(
                f"Could not open camera index {self._requested.index} "
                f"with backend {self._requested.backend}"
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested.height)
        capture.set(cv2.CAP_PROP_FPS, self._requested.fps)

        backend_name = str(capture.getBackendName())
        self._actual = CameraSettings(
            index=self._requested.index,
            backend=backend_name,
            width=round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
        )
        self._capture = capture

    def read(self) -> CameraFrame:
        """Capture the next synchronous video frame from the video stream.

        Returns:
            CameraFrame: Sequence number, monotonic capture timestamp, latency, and BGR matrix.

        Side Effects:
            Increments the internal frame sequence counter.

        Raises:
            CameraUnavailableError: If the camera is not opened or returns an empty frame buffer.
        """
        if self._capture is None:
            raise CameraUnavailableError("Camera must be opened before reading")
        started_ns = perf_counter_ns()
        ok, frame = self._capture.read()
        completed_ns = perf_counter_ns()
        if not ok or frame is None or frame.size == 0:
            raise CameraUnavailableError("Camera opened but returned no frame")
        result = CameraFrame(
            sequence=self._sequence,
            captured_at_ns=completed_ns,
            capture_latency_ms=(completed_ns - started_ns) / 1_000_000,
            bgr=frame,
        )
        self._sequence += 1
        return result

    def close(self) -> None:
        """Release OS video capture handles and reset provider state.

        Side Effects:
            Closes the OpenCV VideoCapture device and unlocks hardware resources.
        """
        if self._capture is not None:
            self._capture.release()
        self._capture = None

    def __enter__(self) -> OpenCVCameraProvider:
        """Context manager entry point; opens the camera."""
        self.open()
        return self

    def __exit__(self, *_args: object) -> None:
        """Context manager exit point; releases camera resources."""
        self.close()


__all__ = [
    "CameraFrame",
    "CameraProvider",
    "CameraSettings",
    "CameraUnavailableError",
    "OpenCVCameraProvider",
]
