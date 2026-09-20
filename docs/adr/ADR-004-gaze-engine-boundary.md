# ADR-004: Gaze engine boundary

- Status: Accepted
- Date: 2026-09-19

## Context

Camera, landmark extraction, gaze estimation, interaction and communication have different rates of change and evidence requirements.

## Decision

Expose `CameraProvider`, `FeatureExtractor`, `GazeEstimator`, `GazeFilter` and `GazeModel` boundaries. Emit domain-level tracking/gaze data to a separate interaction engine. Never expose MediaPipe result objects to Flutter.

## Alternatives

A single UI-integrated pipeline is initially shorter but prevents provider replacement, deterministic tests and a future ONNX/mobile transition.

## Consequences

More interfaces and mapping code are required. The benefit is explicit uncertainty handling and interchangeable cameras, recorded sources and external trackers.

