# Mekasa Flutter application

The Flutter presentation application is intentionally not scaffolded yet.

Experiment 001 must first establish a participant-present, measurable feature stream. Building the communication UI before that evidence would encourage coupling the board to simulated or assumed gaze behavior. Flutter and Dart were also unavailable on the inspected Windows host on 2026-09-19.

When work starts here, the application will own:

- accessible rendering and localization;
- caregiver/setup mode;
- eye-only communication flow;
- dwell progress and interaction feedback;
- the semantic YES/NO/WATER/HELP board;
- local operating-system text-to-speech.

It will consume generated types from `protocols/gaze` and will not import MediaPipe or camera implementation details. Initial tests must cover tracking loss, dwell, cooldown, target hysteresis, pause/cancel and one-selection-per-dwell behavior before connecting real gaze coordinates.

