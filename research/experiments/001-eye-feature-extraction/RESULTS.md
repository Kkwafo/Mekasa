# Experiment 001 — Results

## 2026-09-19 environment audit

- Host OS: Windows 10 Pro, 64-bit.
- CPU: Intel Core i5-8250U, 4 physical cores / 8 logical processors.
- Memory: 12,278,231,040 bytes reported by Windows (approximately 11.4 GiB).
- Camera enumerated by Windows: `Integrated Camera`, device status `OK`.
- Git: 2.54.0.windows.1.
- At initial inspection, Python, uv, Flutter/Dart and CMake were unavailable from the shell.

These are observed environment facts, not pipeline results.

Python 3.12.14 was subsequently located in the workspace runtime. `uv` 0.12.17 resolved and installed the committed lock after transient DNS/time-out retries. The explicit model fetch was verified at 3,758,596 bytes and SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.

Windows' legacy product-name query reported Windows 10 Pro, while Python identified the running kernel as `Windows-11-10.0.26200-SP0`. Both observations are retained instead of choosing one silently.

## Measured run: 2026-09-19, no visible face

Source commit: `29f1e54bae5854de2e5090a78608b30fa5e60ef1` (clean).

Configuration: Media Foundation camera 0, negotiated 640×480 at 30 FPS, 3 s warm-up, 30 s measurement, debug display disabled.

Packages: MediaPipe 1.0.1, OpenCV contrib 5.0.0.93, NumPy 2.5.3, psutil 7.2.2

| Metric | Observed value |
| --- | ---: |
| Duration | 30.0687 s |
| Attempted frames | 901 |
| Effective FPS | 29.9647 |
| Capture latency median / P90 / max | 2.8547 / 83.0195 / 121.7992 ms |
| Feature latency median / P90 / max | 3.2987 / 4.1631 / 13.4431 ms |
| Capture-to-observation latency median / P90 / max | 6.8188 / 86.5461 / 127.0265 ms |
| Process CPU time | 9.96875 s |
| Process utilization, one-core basis | 33.1533% |
| Process utilization, total 8-logical-CPU capacity | 4.1442% |
| Resident memory median / max | 184.6133 / 184.7148 MiB |
| Tracking availability | 0 / 901 frames (0%) |
| Longest non-good run | 901 frames |

The camera was accessible, but no face was visible during this unattended run. Therefore this run measures capture plus the no-face inference path only. It provides **no eye-feature stability, head-pose stability or tracked-face latency evidence**. `tracking_loss_events` is zero because that metric counts transitions from good tracking to non-good tracking; the run never acquired tracking. The continuous loss is represented by the 901-frame longest non-good run.

The observed 29.96 FPS meets the frame-rate target for this no-face path, and median feature latency is below the 33 ms hypothesis. The P90 capture/loop latencies show strong camera-delivery variability and must not be hidden by the medians. A participant-present run under stated pose/lighting conditions is still required before Experiment 001 can be considered complete.

Aggregate raw metrics are committed in `results/2026-09-19-windows-i5-8250u-no-visible-face.json`. No frames, video or raw landmarks were persisted.
