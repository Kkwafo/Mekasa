# Experiment 003 — Abstract Region Selection

## Purpose

Experiment 003 tests the interaction mechanics needed after personal calibration: four large abstract regions, prediction smoothing, dwell progress, boundary hysteresis, post-selection cooldown and immediate cancellation on tracking loss. It deliberately does not attach communication meaning to a region, move the operating-system cursor or speak a message.

## Run locally

First create a current guided report and calibration as described in Experiment 002. Then run:

```powershell
C:\Projects\Mekasa\work\bootstrap\Scripts\uv.exe run --locked mekasa-region-evaluation
```

The four regions are labelled `REGION 1` through `REGION 4`. Looking continuously at one region fills the progress indicator. The default dwell is 1.20 seconds, followed by a 1.00-second cooldown. A 0.06 normalized hysteresis margin prevents small boundary jitter from constantly switching the active region.

The screen also checks camera-relative face size. `ACERCATE UN POCO` means the detected face width is below the configured preparation band; `ALEJATE UN POCO` means it is above it. Moving outside the band, losing tracking, closing both eyes or producing invalid geometry clears dwell immediately. It never carries an old selection through missing evidence.

Press `Q` or `Esc` to finish. A local summary is written to `work/interaction/region-evaluation-latest.json`, and every run is also retained under `work/interaction/history/`. These ignored files contain derived selection event times and regions, but no video, images or raw facial landmarks.

## Controlled validation

The controlled runner measures intended-versus-selected regions:

```powershell
C:\Projects\Mekasa\work\bootstrap\Scripts\uv.exe run --locked mekasa-controlled-region-validation
```

After placement and countdown, the local Spanish voice announces one region and a visual marker appears inside it. Selection remains disabled during the four-second cue guard. The response window then accepts the first completed dwell or records `NO_SELECTION` after five seconds. Feedback is followed by the next trial.

Three seeded-randomized blocks request every region exactly once, producing 12 trials in roughly two minutes. The first complete block is practice: selection is disabled, raw predictions are collected against the four diagonal region centers, and a session-local ridge correction is fitted. Practice is never included in accuracy. The remaining two blocks measure the corrected response. The report records:

- requested and selected region for every trial;
- `PRACTICE`, `CORRECT`, `INCORRECT`, or `NO_SELECTION` outcome;
- accuracy across all trials and accuracy when a selection occurred;
- completion rate and response-time distributions;
- per-region accuracy and the full confusion matrix;
- interrupted dwell attempts, unusable frames, placement failures and predictions outside the screen;
- raw and corrected out-of-screen frames per scored attempt;
- diagonal-correction sample counts and training residuals.

The latest report and unique history copies remain under `work/interaction/`. Region names are not shown as reading prompts during the controlled trial; audio and the target marker provide the instruction.

## Configuration

Engineering defaults are in `config/experiment-003.toml`:

- `dwell_seconds`: continuous evidence required before a selection;
- `cooldown_seconds`: period during which another selection cannot fire;
- `hysteresis_margin`: extra boundary retained around the active candidate.
- `controlled_validation.blocks`: complete randomized blocks;
- `controlled_validation.practice_blocks`: initial complete blocks excluded from scoring;
- `practice_measurement_seconds`: fixed gaze collection interval per practice region;
- `correction_ridge_alpha`: regularization applied to the session-local correction;
- `min_correction_samples_per_region`: minimum usable practice evidence required;
- `cue_seconds`: audio/acquisition guard before selection is enabled;
- `response_timeout_seconds`: maximum selection window;
- `feedback_seconds`: result display before the next trial.

These timings and the face-size band are technical starting points, not accessibility or clinical recommendations.

## Required evaluation

The free runner cannot calculate accuracy because it has no requested target. The controlled runner supplies that missing technical ground truth, but still requires a person to complete the protocol. Semantic intents remain out of scope until controlled results are measured, corrected and repeated in another session.
