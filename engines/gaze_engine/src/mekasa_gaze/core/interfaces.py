from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.domain import CameraFrame, CameraSettings, FeatureObservation


class CameraProvider(Protocol):
    """Protocol for hardware camera acquisition devices.

    Defines the contract for initializing, capturing timestamped video frames,
    and releasing underlying hardware resources without coupling to OpenCV.
    """

    @property
    def actual_settings(self) -> CameraSettings:
        """Return the actual camera properties negotiated with the hardware backend."""
        ...

    def open(self) -> None:
        """Initialize and open the physical camera stream."""
        ...

    def read(self) -> CameraFrame:
        """Capture and return a single synchronized frame with high-resolution timestamp."""
        ...

    def close(self) -> None:
        """Release camera hardware resources."""
        ...


class FeatureExtractor(Protocol):
    """Protocol for extracting facial and eye geometry features from image frames.

    Maps a raw visual image frame to a structured FeatureObservation without
    coupling callers to specific deep-learning landmark providers (e.g. MediaPipe).
    """

    def extract(
        self,
        rgb_frame: npt.NDArray[np.uint8],
        *,
        sequence: int,
        timestamp_ns: int,
    ) -> FeatureObservation:
        """Process an RGB frame and extract normalized geometric facial and eye features."""
        ...

    def close(self) -> None:
        """Release extractor model and accelerator resources."""
        ...


class SpeechBackend(Protocol):
    """Protocol for local text-to-speech (TTS) audio feedback services.

    Provides auditory instructions to the participant without relying on
    external network services or cloud APIs.
    """

    @property
    def active(self) -> bool:
        """Indicate whether the local speech engine is initialized and available."""
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        """Return descriptive metadata of the speech engine, voice, and configuration."""
        ...

    def speak(self, text: str) -> None:
        """Synthesize and vocalize the provided text asynchronously."""
        ...

    def close(self) -> None:
        """Terminate ongoing audio playback and release process resources."""
        ...
