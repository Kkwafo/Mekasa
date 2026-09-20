# Experiment 001 — Eye Feature Extraction

## Question

Can a conventional webcam on an ordinary laptop produce a real-time, locally processed stream of face/eye landmarks, normalized eye features and an approximate head pose with measurable latency and stability?

This experiment does **not** estimate screen gaze and does not validate communication, accessibility or clinical suitability.

## Pipeline

```text
webcam -> OpenCV capture -> RGB frame -> MediaPipe Face Landmarker (VIDEO)
       -> normalized eye/iris features -> approximate head pose -> metrics/debug overlay
```

Frames remain in memory and are neither saved nor uploaded. Aggregate technical metrics are written to JSON. Raw landmarks are not persisted.

## Reproduce

From the repository root:

```powershell
uv python install 3.12
uv sync --locked
./scripts/fetch-face-landmarker.ps1
uv run mekasa-probe-cameras --backend msmf
uv run mekasa-experiment-001 --display
```

Press `Q` to end a displayed run early. Use `--no-display` for an unattended measurement. The committed default requests 640×480 at 30 FPS, warms up for 3 seconds and measures for 30 seconds; negotiated camera settings are read back and reported.

## Metrics

- effective FPS;
- capture, feature and full-loop latency distributions;
- tracking state counts and availability;
- loss transitions and longest consecutive non-good run;
- per-feature population standard deviation and median absolute deviation;
- process CPU utilization and resident memory.

CPU follows psutil process semantics and can exceed 100% when multiple CPU cores are used. Stability metrics only describe the recorded run; the operator should attempt a steady pose if interpreting them as jitter.

## Current result

One real unattended no-visible-face run is committed. It verifies camera access, the no-face MediaPipe path and measurement instrumentation, but it cannot provide tracking or eye-feature stability evidence. Experiment 001 remains incomplete until a participant-present run is performed under stated conditions.

See `RESULTS.md` for measured runs and blockers.
