from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from mekasa_gaze.core.domain import ExperimentPhase, MeasurementStage, ProtocolSnapshot, TargetName
from mekasa_gaze.core.interfaces import SpeechBackend
from mekasa_gaze.tracking.placement import PlacementAssessment, PlacementState

_PROBE_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$language = $env:MEKASA_SPEECH_LANGUAGE
$voice = $synth.GetInstalledVoices() |
  Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -eq $language } |
  Select-Object -First 1
if ($null -eq $voice) {
  $synth.Dispose()
  exit 4
}
[PSCustomObject]@{
  name = $voice.VoiceInfo.Name
  culture = $voice.VoiceInfo.Culture.Name
} | ConvertTo-Json -Compress
$synth.Dispose()
"""

_SPEAK_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$language = $env:MEKASA_SPEECH_LANGUAGE
$voice = $synth.GetInstalledVoices() |
  Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -eq $language } |
  Select-Object -First 1
if ($null -eq $voice) {
  $synth.Dispose()
  exit 4
}
$synth.SelectVoice($voice.VoiceInfo.Name)
$synth.Rate = [int]$env:MEKASA_SPEECH_RATE
$synth.Volume = [int]$env:MEKASA_SPEECH_VOLUME
$synth.Speak($env:MEKASA_SPEECH_TEXT)
$synth.Dispose()
"""


class DisabledSpeech:
    """Null-object speech synthesizer used when audio feedback is disabled or unavailable.

    Responsibility:
        Satisfies the SpeechBackend protocol safely without performing any I/O or audio output.
    """

    @property
    def active(self) -> bool:
        """Indicate whether the local speech engine is active (always False)."""
        return False

    @property
    def metadata(self) -> dict[str, Any]:
        """Return descriptive metadata indicating disabled audio feedback."""
        return {"active": False, "backend": "disabled"}

    def speak(self, text: str) -> None:
        """No-op vocalization call."""
        del text

    def close(self) -> None:
        """No-op resource disposal."""
        return


class WindowsSapiSpeech:
    """Asynchronous local speech synthesizer using installed Windows System.Speech voices.

    Responsibility:
        Leverages the built-in Windows System.Speech API via background PowerShell process
        spawning to generate offline auditory instructions without external network calls.
    """

    def __init__(self, *, language: str, rate: int, volume: int) -> None:
        """Initialize and probe Windows SAPI speech synthesizer for the requested language.

        Args:
            language: BCP-47 / Windows culture identifier string (e.g. 'es-ES', 'es-MX', 'es-AR').
            rate: Speech playback speed offset (-10 to 10).
            volume: Audio volume percentage (0 to 100).

        Raises:
            RuntimeError: If no enabled local Windows voice is found matching the language.
        """
        self._powershell = _windows_powershell_path()
        self._language = language
        self._rate = rate
        self._volume = volume
        self._process: subprocess.Popen[bytes] | None = None
        probe = subprocess.run(
            self._command(_PROBE_SCRIPT),
            check=False,
            capture_output=True,
            env=self._environment(),
            creationflags=_no_window_flag(),
        )
        if probe.returncode != 0:
            raise RuntimeError(f"No enabled local Windows voice found for {language}")
        try:
            voice = json.loads(probe.stdout.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Windows speech voice probe returned invalid output") from error
        self._voice_name = str(voice["name"])
        self._voice_culture = str(voice["culture"])

    @property
    def active(self) -> bool:
        """Indicate whether the local speech engine is operational."""
        return True

    @property
    def metadata(self) -> dict[str, Any]:
        """Return descriptive metadata of the active Windows voice."""
        return {
            "active": True,
            "backend": "windows_system_speech",
            "voice": self._voice_name,
            "culture": self._voice_culture,
            "rate": self._rate,
            "volume": self._volume,
            "local_only": True,
        }

    def speak(self, text: str) -> None:
        """Synthesize and vocalize the prompt text asynchronously in the background.

        Args:
            text: Spanish instruction text to speak.

        Side Effects:
            Aborts any in-flight vocalization process and spawns a new synthesis process.
        """
        self._stop_current()
        environment = self._environment()
        environment["MEKASA_SPEECH_TEXT"] = text
        self._process = subprocess.Popen(
            self._command(_SPEAK_SCRIPT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            creationflags=_no_window_flag(),
        )

    def close(self) -> None:
        """Terminate ongoing audio playback process and clean up subprocess handles.

        Side Effects:
            Terminates or kills active PowerShell synthesis processes.
        """
        if self._process is None:
            return
        try:
            self._process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=2.0)
        finally:
            self._process = None

    def _stop_current(self) -> None:
        """Terminate active playback subprocess immediately."""
        if self._process is None or self._process.poll() is not None:
            self._process = None
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=1.0)
        self._process = None

    def _environment(self) -> dict[str, str]:
        """Construct environment variables for the speech synthesis process."""
        return {
            **os.environ,
            "MEKASA_SPEECH_LANGUAGE": self._language,
            "MEKASA_SPEECH_RATE": str(self._rate),
            "MEKASA_SPEECH_VOLUME": str(self._volume),
        }

    def _command(self, script: str) -> list[str]:
        """Construct PowerShell CLI invocation argument list."""
        return [
            str(self._powershell),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ]


class AudioInstructionController:
    """Manages event-triggered auditory instruction prompts during protocol execution.

    Responsibility:
        Monitors protocol snapshot transitions and participant placement states to dispatch
        spoken instructions through the configured SpeechBackend without repeating identical cues.
    """

    def __init__(self, backend: SpeechBackend) -> None:
        """Initialize the audio instruction controller with a speech backend.

        Args:
            backend: Speech synthesis backend implementation.
        """
        self._backend = backend
        self._last_cue: str | None = None

    def update(
        self,
        snapshot: ProtocolSnapshot,
        placement: PlacementAssessment | None = None,
    ) -> None:
        """Evaluate current protocol state and placement to speak new audio instructions.

        Args:
            snapshot: Current guided protocol snapshot.
            placement: Optional participant camera placement assessment.

        Side Effects:
            Invokes backend.speak() whenever the evaluated cue identifier changes.
        """
        cue = _cue(snapshot, placement)
        if cue is None or cue[0] == self._last_cue:
            return
        self._last_cue = cue[0]
        self._backend.speak(cue[1])


def create_speech_backend(
    *,
    enabled: bool,
    required: bool,
    language: str,
    rate: int,
    volume: int,
) -> SpeechBackend:
    """Factory function for creating and validating a local speech backend instance.

    Responsibility:
        Selects between WindowsSapiSpeech and DisabledSpeech based on platform and configuration.

    Args:
        enabled: Whether audio feedback is enabled by configuration.
        required: Whether failure to find a local voice should raise an exception.
        language: Desired voice culture identifier.
        rate: Speech rate adjustment (-10 to 10).
        volume: Speech volume (0 to 100).

    Returns:
        SpeechBackend: Initialized speech synthesizer instance.

    Raises:
        RuntimeError: If required is True but no compatible local voice engine is available.
    """
    if not enabled:
        return DisabledSpeech()
    try:
        if platform.system() != "Windows":
            raise RuntimeError("Local speech is currently implemented for Windows only")
        return WindowsSapiSpeech(language=language, rate=rate, volume=volume)
    except (FileNotFoundError, OSError, RuntimeError) as error:
        if required:
            raise RuntimeError(
                f"Required local audio instructions are unavailable: {error}"
            ) from error
        return DisabledSpeech()


def _cue(
    snapshot: ProtocolSnapshot,
    placement: PlacementAssessment | None = None,
) -> tuple[str, str] | None:
    if snapshot.stage is MeasurementStage.WAITING_TRACKING:
        if placement is not None:
            return {
                PlacementState.NO_FACE: (
                    "placement:no-face",
                    "Ubicá tu rostro frente a la cámara.",
                ),
                PlacementState.TOO_FAR: (
                    "placement:too-far",
                    "Estás demasiado lejos. Acercate un poco.",
                ),
                PlacementState.TOO_CLOSE: (
                    "placement:too-close",
                    "Estás demasiado cerca. Alejate un poco.",
                ),
                PlacementState.READY: (
                    "placement:ready",
                    "Distancia correcta. Mantené esta posición.",
                ),
            }[placement.state]
        return "preparation", "Ubicá tu rostro frente a la cámara."
    if snapshot.stage is MeasurementStage.COUNTDOWN:
        if snapshot.stage_remaining_seconds is None:
            return None
        number = max(1, min(3, int(snapshot.stage_remaining_seconds + 0.999)))
        return f"countdown:{number}", {3: "Tres", 2: "Dos", 1: "Uno"}[number]
    if snapshot.stage is MeasurementStage.TRANSITION and snapshot.target is not None:
        if snapshot.phase is ExperimentPhase.CENTER_STABILITY:
            text = "Centro. Mantené la cabeza quieta y parpadeá normalmente."
        elif snapshot.phase is ExperimentPhase.HEAD_MOVEMENT:
            text = "Centro. Mové la cabeza suavemente, sin forzar."
        else:
            text = {
                TargetName.CENTER: "Centro",
                TargetName.LEFT: "Izquierda",
                TargetName.RIGHT: "Derecha",
                TargetName.UP: "Arriba",
                TargetName.DOWN: "Abajo",
                TargetName.TOP_LEFT: "Arriba a la izquierda",
                TargetName.TOP_RIGHT: "Arriba a la derecha",
                TargetName.BOTTOM_LEFT: "Abajo a la izquierda",
                TargetName.BOTTOM_RIGHT: "Abajo a la derecha",
            }[snapshot.target.name]
        return f"segment:{snapshot.segment_index}", text
    if snapshot.completed:
        return "complete", "Prueba finalizada. Podés descansar."
    return None


def _windows_powershell_path() -> Path:
    windows_directory = os.environ.get("WINDIR", r"C:\Windows")
    executable = Path(windows_directory) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    if not executable.is_file():
        raise FileNotFoundError(f"Windows PowerShell was not found at {executable}")
    return executable


def _no_window_flag() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


__all__ = [
    "AudioInstructionController",
    "DisabledSpeech",
    "SpeechBackend",
    "WindowsSapiSpeech",
    "create_speech_backend",
]
