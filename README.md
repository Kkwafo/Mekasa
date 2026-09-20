# Mekasa

**Communication through gaze**

Mekasa is an open-source, local-first research project exploring eye-gaze access for augmentative and alternative communication (AAC). V0 asks a deliberately narrow question: can a person reliably select four large communication targets using voluntary gaze and a conventional camera?

Mekasa is an experimental assistive-technology research prototype. It is not a medical device, is not intended for diagnosis or clinical decision-making, and is not clinically validated.

## Current status

The repository is at **Experiment 001 — Eye Feature Extraction**. It implements a Python pipeline boundary for webcam capture, MediaPipe face landmarks, normalized eye/iris features, approximate head pose, an optional visual debug overlay and honest aggregate performance metrics. It does not yet estimate gaze coordinates or provide the four-item communication board.

The first committed hardware run measured the no-visible-face path at 29.96 FPS. It is not evidence of successful face/eye tracking; see the [measured results](research/experiments/001-eye-feature-extraction/RESULTS.md).

## Architecture

```text
Camera
  -> Computer Vision
  -> Feature Extraction
  -> Personal Calibration       (planned Experiment 002)
  -> Gaze Model                 (planned Experiment 002)
  -> Gaze Coordinates
  -> Interaction Engine         (planned Experiment 003)
  -> Communication Board
  -> Local Speech               (planned Experiment 004)
```

Flutter owns presentation and accessibility. Python owns research and computer vision. A transport-neutral Protobuf contract prevents the UI from depending on MediaPipe. Continuous video is not sent to the UI.

## Privacy

- processing is local;
- frames are not saved or uploaded by default;
- aggregate experiment metrics do not include images or raw landmarks;
- application use and any future opt-in research collection are separate;
- no API key or Internet connection is required at runtime once dependencies and the explicitly selected model are installed.

## Quick start: Experiment 001

Requirements: Windows, Linux or macOS; a webcam; and `uv`.

```powershell
uv python install 3.12
uv sync --locked
./scripts/fetch-face-landmarker.ps1
uv run mekasa-probe-cameras
uv run mekasa-experiment-001 --display
```

The `uv.lock` file must exist before `--locked` can succeed. Maintainers create/update it with `uv lock` after checking actual package compatibility. See the [experiment protocol](research/experiments/001-eye-feature-extraction/README.md) and [technical design](docs/V0-TECHNICAL-DESIGN.md).

## Safety and limitations

Ordinary webcams, lighting, eyewear, eye conditions, head motion and individual anatomy may all affect tracking. Face landmarks are not gaze estimation, technical metrics are not communication success, and developer testing is not evidence that the system suits an AAC user. When tracking evidence is insufficient, later interaction logic must prevent selection rather than guess.

Mekasa V0 only communicates. It must not control medical equipment, mobility devices, doors, emergency services, financial systems or other safety-critical systems.

## Collaboration

The design requires AAC users as design participants, plus relevant SLP/fonoaudiology, occupational therapy, PM&R, vision, rehabilitation engineering, accessibility/HCI and engineering input. Open questions are recorded in [V0 clinical questions](docs/V0-CLINICAL-QUESTIONS.md); the document intentionally does not invent clinical answers.

## License

Project source is intended for Apache-2.0 licensing. Third-party models, datasets and assets require their own verified provenance and compatible licenses before redistribution. The current MediaPipe model binary is downloaded explicitly and is not committed.
