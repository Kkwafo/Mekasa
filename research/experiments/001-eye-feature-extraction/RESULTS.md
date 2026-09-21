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

## Measured run: 2026-09-20, guided visual validation 01

Source commit: `70bf39b8ca1dea4f5d0b4c4e5ae9007ef2560671` (clean). The protocol completed without operator cancellation on Media Foundation camera 0 at 640×480 and 30 requested/negotiated FPS.

| Metric | Observed value |
| --- | ---: |
| Frames | 653 |
| Tracking good / uncertain / lost | 653 / 0 / 0 |
| Tracking availability | 100% |
| Tracking loss events | 0 |
| Gaze-valid frames | 616 (94.33%) |
| Rejected frames | 37 (5.67%) |
| Valid target samples, CENTER / LEFT / RIGHT / UP / DOWN | 41 / 44 / 42 / 42 / 38 |
| Feature latency median / P90 / max | 37.4073 / 41.2209 / 90.9776 ms |
| Full processing-loop latency median / P90 / max | 40.7719 / 45.1171 / 810.7669 ms |
| Process CPU, one-core / total-capacity basis | 134.974% / 16.872% |
| Resident memory median / max | 209.2930 / 209.8125 MiB |

All five target groups exceeded the configured 20-sample descriptive minimum. The strongest robust median separations were right-eye iris X for RIGHT versus CENTER (4.491) and left-eye iris Y for DOWN versus CENTER (3.181). LEFT was clearer in right-eye iris X (1.628) than left-eye iris X (0.529). UP was only moderate in left-eye iris Y (1.045) and weak in right-eye iris Y (0.524). These are within-run descriptive ratios, not accuracy.

The report's 9.339 FPS value is not a valid sustained-stream rate: its 69.921-second denominator began before camera/model initialization, while observation timestamps span 44.940 seconds. The timestamp-derived stream rate is approximately 14.508 FPS, still below the 24 FPS engineering hypothesis. The difference between the approximately 68.9 ms frame interval and 40.8 ms median processing loop also shows that render/display work needs separate instrumentation.

All 37 rejected observations were labelled `degenerate_eye_geometry`. They formed nine runs; eight were bounded by valid observations and lasted no more than the configured 0.45 seconds. Their low EAR values make blink-like closures a plausible contributor, but the implementation checked minimum vertical span before eye closure, so the reported zero possible blinks is not reliable. This must be corrected prospectively rather than retroactively changing the recorded JSON.

Head-pose and repeated-center medians changed during the sequential protocol. For example, median pitch was approximately 8.11° at discrimination CENTER, 11.75° at LEFT and 6.92° at RIGHT. Because every target occurred in one fixed block, target effects cannot yet be separated cleanly from posture drift or time order. Repeated, seeded-randomized blocks and audio prompts are required before fitting a calibration model.

The exact report is preserved in `results/2026-09-20-windows-i5-8250u-guided-validation-01.json` with SHA-256 `d9808f9ab8485d1bdd9fb700a4bddd4f1ffc3fce87c2e628fbcc73c946b27a94`. It contains derived features but no video, images, raw facial landmark arrays or participant name. It supports continued engineering; it does not validate gaze accuracy, calibration, AAC performance or clinical suitability.

## Measured run: 2026-09-20, audio-guided randomized validation 01

Source commit: `c04933cb0d59170ce7a4494008ef39ab4d4ae493` (clean). The three-block protocol completed without operator cancellation. Spoken instructions used the local Windows System.Speech backend with the `Microsoft Helena Desktop` Spanish voice; no network speech service was used. Target order was seeded and randomized independently in each block.

| Metric | Observed value |
| --- | ---: |
| Startup | 15.1521 s |
| Frames / stream duration | 2,808 / 130.6676 s |
| Effective FPS | 21.4820 |
| Tracking good / uncertain / lost | 2,808 / 0 / 0 |
| Original full-valid / rejected frames | 2,318 / 490 |
| Original possible-blink frames / events | 112 / 28 |
| Feature latency median / P90 | 20.0060 / 31.0283 ms |
| Full iteration latency median / P90 | 45.0117 / 53.8782 ms |
| Process CPU, one-core / total-capacity basis | 190.7911% / 23.8489% |
| Resident memory maximum | 219.6484 MiB |

The original report classified every DOWN measurement in blocks 2 and 3 as unusable, so the first calibration attempt correctly stopped with `DOWN: 0` held-out samples. The raw derived evidence shows why: block 2 contained 64 `OPEN/CLOSED` observations and one `CLOSED/CLOSED` observation; block 3 contained 67 `OPEN/CLOSED` observations. Block 1 still had 42 `OPEN/OPEN` DOWN observations. In other words, this was not a missing-sample or tracking-loss problem: looking down consistently narrowed the participant's right eye below the experimental EAR threshold while the left eye remained open.

The pipeline was corrected prospectively by introducing `GAZE_PARTIAL`. For backward-compatible calibration of this preserved schema-3 report, observations previously labelled `EYE_CLOSED` are accepted only when their explicit per-eye state still contains one `OPEN` eye. Iris fields from the closed or invalid eye are masked and replaced by training-set means; bilateral closures, possible blinks, invalid geometry and tracking-invalid frames remain excluded. The source report itself was not rewritten.

Recalibration then used 651 samples from blocks 1–2 and held out all 338 samples from block 3. All five validation targets had 67–69 samples. Held-out median normalized error was 0.1320 (99.87 px on the configured canvas), P90 was 0.2864 (208.62 px), and closest-target hit rate was 79.88%. Results varied materially by target: LEFT, RIGHT and UP each had 100% closest-target hit rate, CENTER had 95.52%, but DOWN had only 2/67 correct (2.99%) and median error 0.2596 (187.62 px). RIGHT also had a relatively large 197.53 px median error despite remaining closest to the RIGHT target.

This calibration is sufficient to exercise the local live visualization, but not to claim reliable gaze control. The held-out DOWN failure and block-to-block displacement require another guided run after the partial-eye correction, followed by comparison with the same complete-block holdout. No clinical or AAC-use conclusion follows from these measurements.

The exact report is preserved in `results/2026-09-20-windows-i5-8250u-audio-guided-blocked-01.json` with SHA-256 `1e01913e8003d99ad2cb5c0630212d325ad62a9752a49dec10b69a71d3152fd5`. It contains derived features only and no video, images, raw facial landmark arrays or participant name.

## Measured run: 2026-09-20, audio-guided corrected validation 02

Source commit: `53481addbdbb990a11817da895a63480dbe10d06` (clean). This is the latest report still present after the operator ran the workflow twice; the conventional output path had overwritten the preceding unpreserved trial. Future CLI runs now also receive a unique copy under the ignored `work/guided-validation/history/` directory.

| Metric | Observed value |
| --- | ---: |
| Startup | 10.8864 s |
| Frames | 2,957 |
| Effective FPS | 22.2720 |
| Tracking availability | 99.9662% |
| Usable gaze availability | 97.5313% |
| Fully valid / partial / rejected frames | 2,653 / 231 / 73 |
| Possible-blink events | 16 |
| Full iteration latency median | 43.1657 ms |
| Process CPU, total-capacity basis | 26.4316% |
| Resident memory maximum | 220.6641 MiB |

Every block/target group had 60–70 usable samples, including DOWN. This verifies that the partial-eye correction removed the prior `DOWN: 0` calibration blocker. The resulting model trained on 691 samples and held out 329 samples from block 3. Held-out closest-target hit rate was 95.14%, but median error was 118.10 px, P90 error was 283.19 px, and 24.01% of raw predictions fell outside normalized screen bounds. Per-target closest-region results were 100% for CENTER, DOWN, LEFT and RIGHT and 75.38% for UP. Large median errors remained for DOWN (167.62 px) and RIGHT (152.87 px). Thus the coarse direction often remained correct while the coordinate was imprecise.

Camera-relative face width drifted from a block-1 median of 0.3192 to 0.3007 in block 2 and 0.2909 in block 3; face height moved from 0.5508 to 0.5175 and 0.5053. Over the same blocks, fully valid discrimination frames declined from 348 to 306 and 259, while partial frames increased from 0 to 37 and 70. Head pose also shifted. This supports the operator's observation that moving closer improved capture and shows that consistency, not merely getting detected once, is important. It does not prove a universal viewing distance.

The preparation UI now gates protocol start on an experimental detected-face-width band of 0.30–0.45 and displays `ACERCATE UN POCO`, `DISTANCIA CORRECTA` or `ALEJATE UN POCO`. This band is camera-relative engineering guidance derived to catch the observed drift, not a clinical or ergonomic recommendation.

The exact latest report is preserved in `results/2026-09-20-windows-i5-8250u-audio-guided-corrected-02.json` with SHA-256 `8bfd186d99c13e5f57bb1a035f1cb651f8213ef861eebed027e6dd971373850f`. It contains derived features only and no video, images, raw facial landmarks or participant name.

## Measured run: 2026-09-21, nine-target validation 03

Source commit: `fc453996409bb76306bba3851b3085777bf69ada` (clean). Schema 4 directly added the four diagonal centers required by Experiment 003. The model trained on target blocks 1–2 and held out target block 3.

| Metric | Observed value |
| --- | ---: |
| Held-out target samples | 806 |
| Closest-target hit rate | 88.9578% |
| Pixel error median / P90 | 59.9926 / 131.2800 px |
| Predictions outside the screen | 1.8610% |
| Best target median | CENTER, 23.6687 px |
| Weakest target median | BOTTOM_RIGHT, 128.5040 px |
| TOP_LEFT hit rate / median error | 47.0588% / 110.4793 px |

This materially improved the earlier five-target global median and added direct diagonal evidence, but it did not solve natural head movement. During target collection the observed yaw range was only approximately −1.72° to 0.04°. During the center-fixation head-movement phase it expanded to approximately −8.93° to 5.04°. Schema 4 recorded those frames but the calibrator did not use them.

A retrospective evaluation of the produced full-feature model on 149 placement-valid head-movement center samples measured a 1,182.80 px median error, 1,524.19 px P90 and 81.88% predictions outside screen bounds. This directly supports the operator's report that changing head direction caused the live mapping to become misconfigured. The fixed-head held-out score must not be interpreted as head-movement robustness.

Schema 5 therefore replaces the single diagnostic movement phase with three blocked center-fixation movement phases. Blocks 1–2 contribute empirical movement anchors; block 3 remains held out and receives a separate error report. Pose angles can only be regression inputs once those movement anchors cover their range; the movement observations are explicitly weighted during fitting. This is still limited to center fixation and is not evidence for every gaze/pose combination.

The exact local report is `work/guided-validation/history/guided-validation-20260921T190555.628492Z.json`, SHA-256 `ebc87f00045d90ffdf0fba8617d5810504b2eed464f1bab95c0bc02a6f2c2ea1`. The derived model SHA-256 is `aedd884728ce2f7a56db1b7f0e3211b8cc28acf0fc512ac5c8a25f8223fc94ef`. Neither artifact contains video, images or raw facial landmarks.

## Measured run: 2026-09-21, blocked head-movement validation 04

Source commit: `93ad0d9053906b294ebb723aa620ad4948973a08` (clean). The schema-5 protocol completed all nine-target and three head-movement blocks, but calibration initially rejected the run because its recorded minimum face width was 0.30.

| Metric | Observed value |
| --- | ---: |
| Frames / effective FPS | 7,160 / 29.9989 |
| Tracking availability | 100.0000% |
| Usable gaze availability | 98.8128% |
| Target-measurement usable-eye samples | 2,418 |
| Accepted at original 0.30 face-width floor | 632 |
| Accepted at evidence-based 0.25 floor | 2,418 |
| Head-movement usable-eye samples | 444 |
| Accepted at original 0.30 floor | 14 |
| Accepted at evidence-based 0.25 floor | 444 |
| Target-phase face width, min / median / max | 0.2770 / 0.2876 / 0.3143 |
| Movement-phase face width, min / median / max | 0.2643 / 0.2861 / 0.3043 |

The failure was therefore caused by an over-strict engineering placement rule, not loss of face tracking or generally unusable eye evidence. The observed run supports lowering the face-width floor to 0.25: its smallest observed value was 0.2643, leaving measured margin. It provides no evidence for reliable operation below 0.25, so a still-greater distance must be evaluated rather than assumed.

Reanalysis at the 0.25–0.45 acceptance band made every block eligible. An exploratory ridge candidate selected using development block 2 uses iris coordinates plus head-pose diagnostics, ridge alpha 10 and an eightfold weight for center head-movement anchors. On block 3 it measured 117.60 px target median error, 191.88 px P90, 59.98% closest-target hit rate and 0.25% outside-screen predictions. The separately held-out movement-center block measured 65.03 px median, 152.16 px P90, 88.89% closest-to-center rate and 1.39% outside-screen predictions.

This is a real improvement in head-movement robustness but is not point-accurate gaze. Because the model/configuration iteration was chosen after examining this completed session, these values are retrospective engineering evidence, not a prospective independent validation. Another session is required before treating the improvement as repeatable.

Future guided runs now pause measurement whenever placement leaves the accepted band and announce `ACERCATE` or `ALEJATE` through local audio. They can no longer silently finish with too few placement-valid samples. The exact local report is `work/guided-validation/history/guided-validation-20260921T210436.062735Z.json`, SHA-256 `d6cc04de976ca797a43a14b23c6baae71240f98fe514ceca60f6bbab660a9e39`. The experimental schema-2 calibration generated from it at source commit `621f81cff2197efc2a543c3749ff745ff6e92360` has SHA-256 `fefd0b3f403fcf4c14eefb66b9b4bbf837b6c36f7c2eaa065dad4be5bbaf0abf`.

## Measured run: 2026-09-21, prospective farther-distance validation 05

Source commit: `386cc3614d14375a79ef7df444029f3dd7fe07b3` (clean). This run prospectively exercised the 0.25–0.45 placement band and the previously selected ridge configuration rather than selecting them after seeing this report. The protocol and calibration both completed.

| Metric | Observed value |
| --- | ---: |
| Frames / effective FPS | 7,203 / 30.0046 |
| Tracking availability / losses | 99.9861% / 1 |
| Usable gaze availability | 94.4329% |
| Full-loop latency median / P90 | 22.3793 / 32.5455 ms |
| Process CPU, one-core / total-capacity basis | 111.8800% / 13.9850% |
| Resident memory median / maximum | 212.6563 / 222.0117 MiB |
| Held-out fixed-target samples | 783 |
| Fixed-target hit rate | 63.0907% |
| Fixed-target pixel error median / P90 | 125.6154 / 174.0415 px |
| Held-out head-movement center samples | 140 |
| Head-movement center hit rate | 83.5714% |
| Head-movement pixel error median / P90 | 57.2415 / 163.8433 px |

During active fixed-target collection, face width was 0.2640 minimum, 0.2734 median and 0.3072 maximum. The preceding run's target-phase median was 0.2876, so the new run provides real evidence at a somewhat greater camera distance. Preparation paused for 264 frames; 155 of those frames, approximately 5.2 seconds at the measured rate, were below the 0.25 floor. After the operator moved closer, all 27 fixed block/target groups had enough samples and calibration completed. The distance warning therefore prevented an unusable run without forcing the earlier 0.30 minimum.

Compared with validation 04, the fixed-target median worsened from 117.60 to 125.62 px while P90 improved from 191.88 to 174.04 px and hit rate rose from 59.98% to 63.09%. For held-out head movement, median improved from 65.03 to 57.24 px while P90 worsened from 152.16 to 163.84 px and center hit rate declined from 88.89% to 83.57%. Usable gaze availability fell from 98.8128% to 94.4329%, consistent with weaker eye evidence at the greater distance, although this single run cannot establish distance as the cause.

Fixed-target performance was asymmetric: BOTTOM_RIGHT had an 8.54% hit rate and 165.18 px median error; LEFT had 25.56% and 166.05 px; UP had 36.67% and 138.62 px; TOP_LEFT had 41.11% and 149.73 px. In contrast, RIGHT, CENTER and TOP_RIGHT had at least 96.63% hit rates. This is not point-accurate gaze and is not suitable for AAC control.

A three-fold block sensitivity check using only this report showed that the configured model's fixed-target median varied from 96.67 to 125.62 px and head-movement median from 57.24 to 124.04 px depending on the held-out block. Reducing movement-anchor weight improved fixed-target cross-fold performance but degraded head-movement performance; adding face geometry did not remove the trade-off. The evidence therefore points to a protocol/model identifiability limitation rather than a single threshold problem: center-only movement anchors do not show the model how each non-center gaze target behaves across head poses.

The next experiment should collect pose-diverse evidence at multiple gaze targets, or explicitly factor eye-in-head direction from head pose, and compare it prospectively against this baseline. It should not tune another global linear weight against this same held-out block. The exact report is `work/guided-validation/history/guided-validation-20260921T221343.930358Z.json`, SHA-256 `fed157a87da97d9dbfd6b9b9ea5d214374acdf04622aff54e02025be56fca53f`. The derived model SHA-256 is `91cdb1527b5825c007c25420fd77bac4ef0413891ba7a87855927a6a5ce6ee50`. Neither artifact contains video, images or raw facial landmarks.
