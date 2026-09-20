# ADR-003: Local-first processing and ephemeral video

- Status: Accepted
- Date: 2026-09-19

## Context

Face video is sensitive, and communication access must not depend on an Internet connection.

## Decision

Capture and process frames on the local device. Do not record, upload or stream video by default. Normal technical reports contain aggregate metrics, not frames or raw landmarks.

## Alternatives

Cloud inference is rejected for V0 because it adds privacy exposure, connectivity dependence and network latency. Automatic research capture is rejected because ordinary use is not consent to research data collection.

## Consequences

Models and dependencies need an explicit installation path for offline runtime. Future research export requires a separate opt-in, governance and ethics workflow.

