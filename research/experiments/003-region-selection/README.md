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

## Configuration

Engineering defaults are in `config/experiment-003.toml`:

- `dwell_seconds`: continuous evidence required before a selection;
- `cooldown_seconds`: period during which another selection cannot fire;
- `hysteresis_margin`: extra boundary retained around the active candidate.

These timings and the face-size band are technical starting points, not accessibility or clinical recommendations.

## Required evaluation

This first runner records free selections and therefore cannot calculate selection accuracy. The next controlled measurement must announce or display a randomized requested region and record intended-versus-selected region, time to selection, false activations, cancellations and no-selection trials. Semantic intents remain out of scope until the abstract mechanism is measured and corrected.
