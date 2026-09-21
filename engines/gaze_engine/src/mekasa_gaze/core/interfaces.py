from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from mekasa_gaze.core.domain import CameraFrame, CameraSettings, FeatureObservation


class CameraProvider(Protocol):
    """Protocol for hardware camera acquisition devices.

    Responsibility:
        Defines the contract for initializing, capturing timestamped video frames,
        and releasing physical or virtual camera resources without coupling to OpenCV.
    """

    @property
    def actual_settings(self) -> CameraSettings:
        """Return the actual camera properties negotiated with the hardware backend.

        Returns:
            CameraSettings: Negotiated hardware resolution, backend name, and framerate.
        """
        ...

    def open(self) -> None:
        """Initialize and open the physical camera stream.

        Side Effects:
            Allocates OS video capture handles and negotiates capture properties.

        Raises:
            CameraUnavailableError: If the device cannot be opened or is busy.
        """
        ...

    def read(self) -> CameraFrame:
        """Capture and return a single synchronized frame with high-resolution timestamp.

        Returns:
            CameraFrame: Captured BGR frame with sequence counter and timestamp.

        Side Effects:
            Advances the internal frame sequence counter and captures current time.

        Raises:
            CameraUnavailableError: If the camera is not opened or fails to yield a frame.
        """
        ...

    def close(self) -> None:
        """Release camera hardware resources.

        Side Effects:
            Closes the video capture handle and frees underlying OS resources.
        """
        ...


class FeatureExtractor(Protocol):
    """Protocol for extracting facial and eye geometry features from image frames.

    Responsibility:
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
        """Process an RGB frame and extract normalized geometric facial and eye features.

        Args:
            rgb_frame: RGB image array of shape (height, width, 3) and uint8 dtype.
            sequence: Monotonic sequence index of the video frame.
            timestamp_ns: High-resolution capture timestamp in nanoseconds.

        Returns:
            FeatureObservation: Structured observation containing normalized eye/face
                metrics, diagnostic head pose, tracking state, and debug points.
        """
        ...

    def close(self) -> None:
        """Release extractor model and accelerator resources.

        Side Effects:
            Frees neural network graph memory and ML inference pipeline resources.
        """
        ...


class SpeechBackend(Protocol):
    """Protocol for local text-to-speech (TTS) audio feedback services.

    Responsibility:
        Provides auditory instructions to the participant without relying on
        external network services, cloud APIs, or telemetries.
    """

    @property
    def active(self) -> bool:
        """Indicate whether the local speech engine is initialized and operational.

        Returns:
            bool: True if an active voice is available for vocalization, False otherwise.
        """
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        """Return descriptive metadata of the speech engine, voice, and configuration.

        Returns:
            dict[str, Any]: Key-value metadata including voice name, culture, rate, and volume.
        """
        ...

    def speak(self, text: str) -> None:
        """Synthesize and vocalize the provided text asynchronously.

        Args:
            text: Human-readable prompt text to vocalize in the configured language.

        Side Effects:
            Spawns background speech playback, aborting any prior in-flight speech cue.
        """
        ...

    def close(self) -> None:
        """Terminate ongoing audio playback and release process resources.

        Side Effects:
            Terminates child synthesis processes and disposes audio handles.
        """
        ...
