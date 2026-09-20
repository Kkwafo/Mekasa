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

No result is committed until the real dependency environment, model, camera and subject-present run complete. A camera listed by Windows Device Manager is not proof that the current process can open or measure it.

See `RESULTS.md` for measured runs and blockers.

