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

The observed 29.96 FPS meets the frame-rate target for this no-face path, and median feature latency is below the 33 ms hypothesis. The P90 capture/loop latencies show strong camera-delivery variability and must not be hidden by the medians. A controlled guided run is still required before the new target-conditioned statistics can be interpreted.

Aggregate raw metrics are committed in `results/2026-09-19-windows-i5-8250u-no-visible-face.json`. No frames, video or raw landmarks were persisted.

## Measured run: 2026-09-20, dynamic / uncontrolled gaze

Source commit: `2c5ecc9fc665ad0834dcbbca9ccab58f7e6afc43` (clean).

Configuration: Media Foundation camera 0, negotiated 640×480 at 30 FPS, 3 s warm-up, 30 s measurement, debug display disabled. A person was in front of the camera and intentionally looked in different directions while attempting to keep their head approximately still. The software did not display or record target instructions.

| Metric | Observed value |
| --- | ---: |
| Duration | 30.1538 s |
| Attempted frames | 901 |
| Effective FPS | 29.8802 |
| Tracking good / uncertain / lost | 901 / 0 / 0 |
| Tracking availability | 100% |
| Tracking loss events | 0 |
| Feature latency median / P90 / max | 10.8983 / 14.2248 / 31.9564 ms |
| Capture-to-observation latency median / P90 / max | 32.0634 / 47.8066 / 85.9729 ms |
| Process utilization, one-core basis | 61.6631% |
| Process utilization, total 8-logical-CPU capacity | 7.7079% |
| Resident memory median / max | 182.9609 / 184.8359 MiB |

This run is valid evidence that face/eye tracking remained continuously available for this participant, camera, pose range and environment. It is **not** a gaze-accuracy test, a controlled stability test or a clinical validation. There was no temporal ground truth for intended gaze direction. The legacy `feature_stability` section is therefore only a dispersion summary over changing, uncontrolled gaze and head behavior; it must not be interpreted as natural gaze jitter or stability.

The unusually large population standard deviations for normalized iris Y (`left ≈ 1.764`, `right ≈ 0.846`) are preserved rather than discarded. Possible contributors include blinks, partially closed eyes, small vertical eyelid denominators, transient landmark geometry and actual gaze changes. The original report cannot distinguish them because it did not classify eye openness or persist per-observation derived features.

Aggregate raw metrics are committed in `results/2026-09-20-windows-i5-8250u-dynamic-uncontrolled-gaze.json`. No frames, video or raw landmarks were persisted.

## Guided validation readiness: 2026-09-20

The guided runner, protocol, validity classification, visualization and report analysis are implemented. A hardware readiness check reopened camera 0 through Media Foundation at 640×480 and 30 FPS and obtained a 480×640 frame; the first-frame capture call measured 332.4362 ms during that probe. A separate non-fullscreen smoke start loaded the Face Landmarker model, created the CPU inference delegate and entered the camera-processing loop. It was deliberately interrupted before the protocol could complete. No guided JSON, video, image or human result was produced by the smoke check.

This readiness check establishes that the camera, model initialization and application loop can start on this host. It does not establish sustained guided performance, gaze discrimination, participant compliance, target accuracy, accessibility or clinical validity. Those observations require Kofi to run the guided protocol while physically present and following the targets.
