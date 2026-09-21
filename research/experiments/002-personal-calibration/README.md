# Experiment 002 — Personal Calibration and Live Evaluation

## Purpose

Experiment 002 fits a local, personal mapping from derived eye/face/head features to normalized screen coordinates. It then measures that mapping on a complete protocol block that was not used for training and can display a live experimental prediction.

This is technical research tooling. It is not a generic gaze model, clinical validation, an AAC interface, cursor control or dwell selection.

## Required input

Calibration accepts only a completed schema-4 Guided Visual Validation report with:

- three seeded-randomized discrimination blocks;
- all nine targets in every block, including the four diagonal region centers;
- separate transition, acquisition and measurement intervals;
- gaze-valid derived features;
- active local audio prompts when required by configuration.

The first two blocks train the default model. The complete third block remains held out for validation. Frames from the held-out block never participate in fitting or feature standardization. This reduces adjacent-frame leakage but does not measure repeatability across different sessions or days.

## End-to-end local workflow

Run all commands from `C:\Projects\Mekasa`.

### 1. Collect an audio-guided report

```powershell
C:\Projects\Mekasa\work\bootstrap\Scripts\uv.exe run --locked mekasa-guided-validation
```

The installed `Microsoft Helena Desktop` Spanish voice announces preparation, countdown, direction changes, head-movement instructions and completion. Direction words are spoken during the transition interval, before measurement. When audio is active, long visual instructions and target-name labels are hidden from the test area so reading them does not intentionally contaminate eye position. Text remains available as a fallback when audio is disabled.

The three blocks contain CENTER, LEFT, RIGHT, UP, DOWN and the four diagonal region centers exactly once each, in a deterministic randomized order recorded in the report. A measured 3.15-second worst-case local speech duration motivated a 4-second transition-plus-acquisition guard before measurement. The default sequence takes roughly 3 minutes 35 seconds after camera/model startup.

### 2. Fit and evaluate the personal calibration

```powershell
C:\Projects\Mekasa\work\bootstrap\Scripts\uv.exe run --locked mekasa-calibrate
```

The baseline is a closed-form ridge regression with independent normalized X/Y outputs. Inputs are both iris coordinates, coarse face position/size and approximate head pose. Feature standardization is learned only from training blocks.

For reports produced with placement guidance, frames whose detected face width falls outside the recorded preparation band are excluded from fitting and held-out evaluation. The command fails honestly if that leaves too few samples for any target.

The command prints and stores:

- sample counts by target for training and held-out validation;
- mean, median, P90 and maximum normalized Euclidean error;
- the same error converted to test-area pixels;
- closest-target hit rate globally and per target;
- fraction of raw predictions outside normalized screen bounds;
- source report hash, configuration, source revision and limitations.

The private local model is written to:

```text
C:\Projects\Mekasa\work\calibration\personal-gaze-model.json
```

`work/` is ignored by Git. The model contains derived personal coefficients and must not be committed or shared by default.

### 3. View the live experimental response

```powershell
C:\Projects\Mekasa\work\bootstrap\Scripts\uv.exe run --locked mekasa-live-evaluation
```

The window shows nine reference positions, a smoothed predicted crosshair and the nearest target label. Invalid/closed-eye/tracking-invalid frames clear the prediction rather than carrying an old value forward. Press `Q` or `Esc` to stop.

A loading screen is shown while the camera and landmark model initialize. Camera-relative face-size guidance remains visible: keep the placement in `DISTANCIA CORRECTA`; changing distance after calibration can shift predictions even though face size is one of the model inputs.

The response is explicitly labelled experimental. It does not move the operating-system cursor, make a selection, speak an AAC message or trigger any external action.

## Interpretation

Closest-target hit rate is meaningful only for the held-out block and setup that produced it. Random adjacent-frame splitting is deliberately not used. A strong result on one session may still fail after posture, camera, lighting, eyewear or fatigue changes.

Before any interaction work, repeat the full collection/calibration process in separate sessions and compare target-level errors. Do not hide failed targets or excluded frames. No threshold in this experiment is clinically validated.

The guided CLI keeps the conventional latest report and also archives each future run under `work/guided-validation/history/`, preventing another trial from silently replacing the only local copy.

## Privacy

- camera processing, speech and calibration are local;
- audio uses an installed Windows voice and is not uploaded;
- speech audio is not recorded or persisted;
- video, frames and raw facial landmark arrays are not saved;
- the guided report contains time-aligned derived features;
- the calibration artifact contains personal derived coefficients and evaluation metrics;
- neither artifact should contain a participant name.
