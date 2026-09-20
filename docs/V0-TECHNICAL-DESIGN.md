# Mekasa V0 — Technical Design

Status: accepted for Experiment 001 implementation  
Date: 2026-09-19  
Scope: experimental assistive-technology research prototype; not a medical device

## 1. V0 question and non-goals

Mekasa V0 asks whether a person can reliably select four large communication targets using voluntary gaze and a conventional camera. Reliability must be demonstrated by measured experiments; this document does not claim that the system is clinically effective or suitable for any person or diagnosis.

V0 does not diagnose, make clinical decisions, control safety-critical equipment, use an LLM, upload camera data, or silently collect a research dataset. MediaPipe is an initial landmark extractor, not a gaze estimator.

## 2. System architecture

The monorepo separates three domains and their contracts:

```text
CameraProvider -> FeatureExtractor -> GazeEstimator -> GazePoint
                                                     |
                                                     v
                                           InteractionEngine
                                                     |
                                                     v
                                         CommunicationIntent
                                                     |
                                                     v
                                            Flutter UI / TTS
```

- `engines/gaze_engine`: Python research and real-time computer-vision engine.
- `protocols/gaze`: transport-neutral Protobuf messages and local service definition.
- `apps/mekasa_app`: Flutter presentation, accessibility, board, interaction feedback and local TTS. It is intentionally deferred until Experiment 001 establishes a viable signal.
- `research/experiments`: reproducible experiment configurations and measured reports.
- `models`: model metadata, checksums and licensing; no silent downloads at runtime.

The UI does not import or understand MediaPipe types. The engine emits domain objects with normalized coordinates, a monotonic timestamp, confidence/evidence fields, tracking state and optional head-pose diagnostics.

## 3. Technology choices

### Python engine

- Python 3.12 or newer, constrained by actual binary-wheel compatibility.
- `uv` for interpreter acquisition, isolated environments and a committed cross-platform lockfile.
- OpenCV `VideoCapture` behind `CameraProvider` for webcam and recorded sources.
- MediaPipe Tasks Face Landmarker in VIDEO mode for synchronous, timestamped experiment measurements. This mode uses tracking between frames; its blocking behavior makes per-frame latency observable.
- NumPy for normalized geometry and head-pose calculations.
- `psutil` for process CPU and memory measurements when available.
- `pytest`, Ruff and mypy for initial quality gates.

SciPy, scikit-learn, PyTorch, ONNX and ONNX Runtime are deferred until an experiment uses them. Adding heavyweight packages before a need would enlarge the environment and license surface without improving Experiment 001.

### Flutter application

Flutter/Dart remains the V0 presentation choice because it provides a single accessible UI codebase for desktop and a path to mobile. Flutter setup and code begin after the vision signal and local contract are testable. This sequencing avoids building a UI around fabricated gaze data.

### Sources checked

- Google Face Landmarker for Python: <https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python>
- OpenCV `VideoCapture`: <https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html>
- uv project locking: <https://docs.astral.sh/uv/concepts/projects/sync/>
- Python 3.12 on Windows: <https://docs.python.org/3.12/using/windows.html>

Exact package versions are selected by resolution on the real host and recorded in `uv.lock` and the experiment report, not copied from a prompt.

## 4. AI architecture

The initial interfaces are:

- `CameraProvider.open/read/close`: owns capture and produces in-memory frames plus capture timestamps.
- `FeatureExtractor.start/extract/stop`: maps a frame to `FeatureObservation` without estimating screen gaze.
- `GazeEstimator.start/calibrate/predict/stop`: later maps feature vectors to `GazePoint`.
- `GazeModel.load/predict/calibrate/reset/metadata`: later provides traceable model lifecycle.
- `GazeFilter.update/reset`: later compares EMA, One Euro and Kalman filters.

Experiment 001 extracts normalized eye and iris geometry, eye-aspect ratios, relative face position and approximate yaw/pitch/roll. All ratios use inter-landmark distances so that features are less sensitive to resolution and face scale. Tracking evidence and gaze-frame validity are modeled separately. Invalid or insufficient landmark evidence produces `TRACKING_UNCERTAIN` or `TRACKING_LOST`; good tracking can still produce `GAZE_PARTIAL`, `EYE_CLOSED`, `POSSIBLE_BLINK`, `GAZE_INVALID` or `TRACKING_INVALID` rather than `GAZE_VALID`. `GAZE_PARTIAL` preserves evidence from one usable eye while masking and mean-imputing the unusable eye's iris fields for calibration. It does not invent an observation for the unusable eye. No gaze coordinate is produced when both eyes are unusable.

The facial transformation matrix is treated as a diagnostic head-pose approximation. Its coordinate convention and relationship to physical degrees must be validated before any angular-error claim.

## 5. IPC design

The target boundary is gRPC with Protocol Buffers over a local-only transport. The schema carries gaze samples, tracking state, confidence/evidence, optional head pose, calibration commands and diagnostic metrics. It does not carry a continuous video stream.

Initial research code remains in-process to avoid IPC contaminating feature-extraction measurements. The future server must bind only to loopback. Unix domain sockets on Linux/macOS and named pipes or loopback TCP on Windows will be benchmarked before a production choice. Authentication, peer identity, permissions and socket lifetime remain required security work.

The detailed decision is recorded in `docs/adr/ADR-005-protobuf-contracts.md` when the schema is introduced.

## 6. Camera pipeline

```text
Open camera -> negotiate width/height/FPS -> capture BGR frame
-> convert BGR to RGB -> MediaPipe Face Landmarker
-> validate one face and required landmarks
-> eye/iris geometry + approximate head pose
-> structured observation -> optional local visual debug overlay
```

Frames are ephemeral. The normal and benchmark paths do not write frames, faces or video. The debug window displays the current in-memory frame only. A recorded input must be explicitly supplied and is treated as research/test data.

Requested camera settings are not reported as achieved settings. The provider reads back negotiated width, height, FPS and backend name and records those values.

## 7. Calibration algorithm (Experiment 002)

Calibration is deferred until Experiment 001 shows stable features. The planned procedure is:

1. Present a configurable grid without requiring clicks.
2. Use a countdown and collect multiple timestamped feature vectors per target.
3. Reject missing-face, closed-eye, low-evidence and obvious outlier frames using documented rules.
4. Split by target/time into calibration and held-out validation samples.
5. Compare linear, ridge, polynomial, random-forest, gradient-boosting and small-MLP baselines only when supported by sufficient data.
6. Report mean, median and P90 pixel error, target hit rate, jitter, prediction latency and tracking availability.
7. Mark poor calibration without blaming the user; do not enable dwell selection when evidence is insufficient.

The split must prevent adjacent frames from leaking nearly identical observations between training and validation.

## 8. Baseline gaze model

There is no gaze model in Experiment 001. Experiment 002 starts with a closed-form ridge regression over normalized iris, coarse face and diagnostic head-pose features, with independent normalized X/Y outputs and a saved preprocessing pipeline. Three seeded-randomized protocol blocks prevent a fixed target order; the final complete block is held out from both fitting and standardization. The artifact reports normalized/pixel error and nearest-target hit rate without calling those metrics clinical accuracy. More complex models are accepted only if later held-out and cross-session evidence improves communication-relevant performance without an unacceptable latency or calibration burden.

The first live evaluation draws a predicted crosshair and nearest target only inside the experiment window. Invalid observations clear the prediction. It cannot control the operating-system cursor, dwell selection, AAC intents or external systems.

Model metadata must include version, type, input/output schemas, dataset provenance, creation time, metrics, dependency environment and source commit. Artifacts require checksums and license/provenance documentation.

## 9. Experiment 001 guided measurement design

Measured separately:

- capture latency: blocking duration of frame acquisition;
- feature latency: RGB conversion plus Face Landmarker plus feature computation;
- loop latency: capture through completed observation;
- effective FPS: completed iterations divided by measured wall time;
- tracking availability: good observations / attempted frames;
- tracking loss events: transitions from a tracked state to lost;
- consecutive loss lengths and longest loss;
- valid-gaze availability and explicit rejection/eye-state counts;
- per-target robust feature distributions from measurement windows only;
- descriptive LEFT/RIGHT versus CENTER iris-X and UP/DOWN versus CENTER iris-Y separability;
- process CPU: process user+system CPU time divided by measured wall time, reported on a one-core basis and divided by logical CPU count for total-capacity context;
- resident memory when available.

The state machine waits for stable tracking, counts down, and then separates transition, acquisition and measurement windows for CENTER stability, three randomized five-direction discrimination blocks and CENTER head movement. A local installed Spanish voice announces short instructions before measurement so reading does not intentionally shift gaze. Reports include observations with derived features, block/target/sample state, hardware, OS, Python/package versions, camera backend and negotiated settings. A run with no accessible camera is a failed/blocked measurement, not zero FPS. Rendering and audio cue selection are tested with deterministic synthetic data; a real guided result still requires a person to follow the targets.

## 10. Testing strategy

- Pure unit tests for geometry, feature validation, tracking transitions and metric aggregation.
- Synthetic landmarks for deterministic edge cases; these tests validate mathematics, not real-world accuracy.
- Contract tests for camera/provider lifecycle with fakes.
- A no-camera smoke path that fails with an actionable message.
- An explicit hardware experiment, never part of ordinary CI.
- Later property-based sequence tests for dwell, cooldown, hysteresis and tracking loss.
- Flutter unit/widget/integration tests after the application exists.

CI must not claim webcam performance because hosted runners do not represent target hardware.

## 11. Privacy and logging model

- Local processing only; no network endpoint is needed for Experiment 001.
- No frame/video persistence or upload by default.
- No face images, raw landmarks tied to identity, communication contents or personal identifiers in logs.
- The legacy metric JSON contains aggregate technical measurements. Guided-validation JSON also contains time-aligned derived eye/iris, coarse face and head-pose features needed for target analysis.
- Neither format contains images, video, participant names or raw landmark arrays. Guided result files are ignored by default and require deliberate review before preservation or sharing.
- Any later research collection requires a separate opt-in workflow, purpose, retention/deletion policy and ethics review as applicable.
- Model download is an explicit setup action with URL, license and checksum; production startup never downloads silently.

## 12. Known risks

- Ordinary RGB webcams may not provide sufficient gaze precision for reliable 2x2 selection across users, lighting, eyewear, eye conditions and head movement.
- MediaPipe landmark confidence is not the same as gaze confidence; Experiment 001 must not mislabel it.
- Camera drivers may ignore requested resolution/FPS or add buffering and auto-exposure latency.
- Head-pose matrix conventions may differ from the desired physical interpretation.
- A development-machine result does not generalize to AAC users or clinical contexts.
- Eye-only recovery, pause/cancel, fatigue and caregiver setup need participatory design before product claims.
- Loopback TCP is local but is not automatically secure against other local processes.

## 13. Performance targets

Targets are hypotheses, not results:

- requested capture: 640×480 at 30 FPS initially;
- median feature-processing latency: at most 33 ms on a representative ordinary laptop CPU;
- effective processing: at least 24 completed FPS for Experiment 001;
- tracking availability: reported, with no pass threshold until test conditions are defined;
- memory and CPU: measured and reported, with optimization based on evidence.

If the target is missed, measure image size, model mode, frame dropping and visual-debug overhead before changing algorithms.

## 14. V0 milestones

1. Experiment 001: reproducible webcam -> landmarks -> valid features -> guided targets -> head pose -> visual debug, with honest per-target and performance metrics. Human guided execution remains required before interpreting results.
2. Experiment 002: click-free personal calibration and held-out validation of baseline regressors.
3. Experiment 003: four abstract target regions plus smoothing, dwell, hysteresis, cooldown and tracking-loss safety.
4. Experiment 004: semantic intents YES/NO/WATER/HELP, localized presentation and offline TTS.
5. Integrated V0: caregiver setup, eye-only communication path, diagnostics, tests, build and documented limitations.

Every milestone follows research, analysis, plan, implementation, test, measurement, correction, verification, documentation and commit. A milestone is not complete when its hardware or human validation was not performed.
