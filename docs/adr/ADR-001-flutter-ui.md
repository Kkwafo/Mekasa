# ADR-001: Flutter for the presentation layer

- Status: Accepted
- Date: 2026-09-19

## Context

Mekasa needs a large, calm and accessible communication UI across desktop platforms with a later mobile path. The UI must not own camera or vision-library details.

## Decision

Use Flutter/Dart for presentation, interaction feedback, localization, caregiver setup and local TTS. Begin the Flutter application after Experiment 001 establishes a usable feature stream and protocol boundary.

## Alternatives

Native applications would improve platform-specific control but multiply implementation effort. React/WebGazer is rejected as the core because browser gaze tooling would hide the experiment's landmark, calibration and signal-processing boundary.

## Consequences

The team must validate desktop accessibility semantics and packaging on each OS. Flutter remains a client of domain contracts and never imports MediaPipe types.

