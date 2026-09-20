from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt


class CameraUnavailableError(RuntimeError):
    """Raised when a requested camera cannot provide frames."""


@dataclass(frozen=True, slots=True)
class CameraSettings:
    index: int
    backend: str
    width: int
    height: int
    fps: float


@dataclass(frozen=True, slots=True)
class CameraFrame:
    sequence: int
    captured_at_ns: int
    capture_latency_ms: float
    bgr: npt.NDArray[np.uint8]


class CameraProvider(Protocol):
    @property
    def actual_settings(self) -> CameraSettings: ...

    def open(self) -> None: ...

    def read(self) -> CameraFrame: ...

    def close(self) -> None: ...


_BACKEND_NAMES = {
    "auto": cv2.CAP_ANY,
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
}


class OpenCVCameraProvider:
    def __init__(
        self,
        *,
        index: int,
        backend: str,
        width: int,
        height: int,
        target_fps: float,
    ) -> None:
        if backend not in _BACKEND_NAMES:
            choices = ", ".join(sorted(_BACKEND_NAMES))
            raise ValueError(f"Unknown camera backend {backend!r}; choose one of: {choices}")
        self._requested = CameraSettings(index, backend, width, height, target_fps)
        self._capture: Any | None = None
        self._sequence = 0
        self._actual: CameraSettings | None = None

    @property
    def actual_settings(self) -> CameraSettings:
        if self._actual is None:
            raise CameraUnavailableError("Camera has not been opened")
        return self._actual

    def open(self) -> None:
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
        if self._capture is not None:
            self._capture.release()
        self._capture = None

    def __enter__(self) -> OpenCVCameraProvider:
        self.open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
