# Experiment 001 — Eye Feature Extraction

## Purpose and boundary

Experiment 001 asks whether a conventional webcam can produce a real-time, local stream of face/eye landmarks, normalized eye and iris features, and approximate head pose with measurable performance. Its guided visual validation adds known on-screen stimuli so feature distributions can be compared under a reproducible prompt.

It does **not** estimate screen gaze, train a predictor, calibrate a person, measure gaze accuracy, validate communication, or establish clinical suitability. A displayed target records what the software asked the participant to look at; it does not prove fixation.

```text
camera -> face/eye feature extraction -> domain observation
       -> guided protocol -> metrics -> visualization -> JSON report
```

The experiment remains in Python and in-process so UI or transport overhead does not contaminate the measurement. Flutter and the future local gRPC/Protobuf boundary remain outside this experiment.

## Guided protocol

All durations, target positions, validity thresholds, camera settings and overlay options are centralized in `config/experiment-001.toml`.

1. **Preparation** — shows the local camera debug view and waits for `TRACKING_GOOD` to remain stable for 1.5 seconds, then presents a 3-second countdown. Tracking loss before the start resets this preparation.
2. **Center stability** — shows the center target. After a 0.75-second transition and 1-second acquisition interval, it measures for 5 seconds while asking for a quiet head and normal blinking.
3. **Gaze discrimination** — presents three blocks containing CENTER, LEFT, RIGHT, UP and DOWN exactly once in a seeded-randomized order. Every target has a 1-second transition, 3-second audio/acquisition guard and 3-second measurement interval. Only gaze-valid frames from the measurement interval contribute to target statistics.
4. **Head movement robustness** — shows the center target and asks for small, comfortable head motion while approximately maintaining the center intention. This is exploratory; it does not claim that head-pose compensation works.
5. **Completion** — clearly reports that the test ended and the participant may rest.

Targets use normalized coordinates within the test area: CENTER `(0.50, 0.50)`, LEFT `(0.15, 0.50)`, RIGHT `(0.85, 0.50)`, UP `(0.50, 0.15)` and DOWN `(0.50, 0.85)`. The renderer maps them to the selected canvas resolution.

## Run the guided validation

From `C:\Projects\Mekasa`, after dependencies and the explicit MediaPipe model fetch are available:

```powershell
C:\Projects\Mekasa\work\bootstrap\Scripts\uv.exe run --locked mekasa-guided-validation
```

The default is fullscreen with required local Spanish audio. The installed Windows voice announces preparation, countdown, every target, head movement and completion. Speech is asynchronous and begins during transition; target labels and long visual instructions are hidden while audio is active so reading does not intentionally shift gaze. No speech audio is recorded. Use `--no-fullscreen` only if the display environment cannot open fullscreen. Press `Q` or `Esc` to cancel; an aborted report is marked accordingly and the command exits nonzero.

The screen has a large test area and a smaller debug panel. The target and participant instruction take priority. The debug panel shows the live webcam, face box, relevant eye landmarks, left and right iris markers, tracking state, gaze-validity state, both eye states, rolling FPS, target, and approximate yaw/pitch/roll. It also shows the remaining time for the current stage and whole phase.

During the run:

- sit in a comfortable, repeatable position with the face visible;
- wait for stable tracking and the countdown;
- follow the displayed point with the eyes during discrimination;
- allow the acquisition interval to finish, then hold the prompted direction as comfortably as possible;
- blink normally;
- during head movement, keep looking approximately at center while making only small, comfortable movements;
- stop with `Q` or `Esc` if rest is needed.

The result is written to:

```text
C:\Projects\Mekasa\research\experiments\001-eye-feature-extraction\results\guided-validation-result.json
```

Do not commit that participant-specific working result by default. Preserve a reviewed, de-identified measurement under a dated filename only when its conditions and limitations are documented.

## Tracking, eye state and gaze validity

Tracking and gaze validity are separate:

- `TRACKING_GOOD`, `TRACKING_UNCERTAIN`, and `TRACKING_LOST` describe face/landmark evidence;
- `GAZE_VALID`, `EYE_CLOSED`, `POSSIBLE_BLINK`, `GAZE_INVALID`, and `TRACKING_INVALID` describe whether derived geometry is usable for gaze-feature analysis;
- each eye is also marked `OPEN`, `CLOSED`, or `INVALID`.

Eye Aspect Ratio (EAR) is used as an eye-openness signal. Soukupová and Čech introduced EAR as a scalar related to eye opening and used temporal evidence for blink detection in [Real-Time Eye Blink Detection using Facial Landmarks](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf). Mekasa's current cutoff and possible-blink duration are configurable engineering heuristics, not clinically validated thresholds or a validated blink classifier.

Frames are explicitly rejected and counted when tracking is not good, required values are missing/non-finite, the vertical eye span is degenerate, EAR is below the experimental cutoff, or normalized iris coordinates exceed the configured guardrail. Short eye-closure runs bounded by valid frames are relabeled `POSSIBLE_BLINK`; other closures remain `EYE_CLOSED`. Rejected frames are never silently folded into gaze statistics.

## Report contents and metrics

Each stored observation contains only derived data needed for technical analysis:

- relative timestamp, phase, target and normalized target position;
- transition/acquisition/measurement stage;
- tracking, gaze-validity and per-eye state;
- normalized iris position, EAR, vertical eye span and coarse face geometry;
- approximate head yaw, pitch and roll;
- capture, feature and full-loop latency.

The summary includes timestamp-derived effective FPS, capture/feature/processing/render/display/full-iteration latency distributions, tracking availability and losses, process CPU, resident memory, valid-gaze availability, rejected-frame and eye-state counts, possible-blink events, and statistics by phase, block and target. Camera/model startup is reported separately rather than contaminating sustained FPS. A target needs the configured minimum sample count before statistics are marked `AVAILABLE`; otherwise it is explicitly `INSUFFICIENT_SAMPLES`.

Per-target features include count, mean, median, population standard deviation, median absolute deviation (MAD), P10, P90, minimum and maximum. CENTER stability and HEAD_MOVEMENT center data remain separate from the GAZE_DISCRIMINATION CENTER data.

Exploratory separability compares horizontal iris features for LEFT/RIGHT against CENTER and vertical iris features for UP/DOWN against CENTER. It reports:

```text
target_minus_center_median = median(target) - median(center)
absolute_median_delta_over_sum_mad = |median(target) - median(center)|
                                     / (MAD(target) + MAD(center))
```

The second value is omitted when the denominator is zero. Larger values mean the observed medians were farther apart relative to within-group robust spread in this run. It is not accuracy, angular error, a trained model score, or clinical evidence.

## Vertical-iris anomaly from the uncontrolled run

The 2026-09-20 uncontrolled run reported much larger iris-Y dispersion than iris-X dispersion (`left_iris_y` population standard deviation about 1.764 and `right_iris_y` about 0.846). The original aggregate report did not preserve time-aligned derived observations or eye-openness classifications, so it cannot determine whether individual outliers came from blinks, partial closure, a small vertical normalization denominator, transient landmarks, actual gaze changes, or a combination. The result is preserved without retroactive filtering.

Guided validation addresses the observability gap prospectively: it persists derived per-frame EAR and vertical span, classifies invalid/closed-eye frames, records every exclusion reason, and computes target statistics only from gaze-valid measurement frames. This can identify associations in a new run; it does not prove the cause of the old anomaly.

## Privacy and interpretation limits

- Processing is local and no cloud service is used.
- Video and images are not saved or uploaded.
- Raw facial landmark arrays are not persisted.
- The debug overlay uses ephemeral in-memory points.
- The JSON stores derived technical features and environment metadata, but no participant name.
- Target prompts are not eye-tracker ground truth and do not prove compliance or fixation.
- Head pose is an unvalidated diagnostic approximation.
- Results apply only to the observed person, device, setup and conditions.
- This experiment is not a medical device evaluation or clinical validation.

See `RESULTS.md` for the preserved no-visible-face and dynamic/uncontrolled-gaze runs.
