# ADR-007: Personal, click-free calibration

- Status: Accepted for Experiment 002 design
- Date: 2026-09-19

## Context

Camera geometry, anatomy, posture and eye behavior vary. A generic landmark mapping cannot be assumed to predict screen gaze for every person.

## Decision

Use a click-free, multi-target personal calibration with multiple samples per target and held-out validation. Reject frames with insufficient evidence and compare simple regressors before complex models.

## Alternatives

One generic model is lower-friction but unsupported as a universal solution. Click-confirmed calibration violates the eye-only access goal.

## Consequences

Calibration time and failure handling are part of product quality. The system must report poor calibration without blame and must not enable selection when evidence is insufficient.

