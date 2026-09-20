# ADR-005: Protobuf contracts for local IPC

- Status: Accepted for schema; transport under evaluation
- Date: 2026-09-19

## Context

Flutter and the research engine need a typed, versionable boundary. Continuous video should not cross it.

## Decision

Define gaze samples, tracking state, optional evidence fields, calibration commands and aggregate diagnostics in Protocol Buffers. Target gRPC on a local-only transport. Experiment 001 remains in-process so IPC does not contaminate its feature latency measurement.

## Alternatives

JSON is easy to inspect but weaker for evolution and generated clients. FFI reduces serialization but tightly couples runtime and ABI. Ad-hoc stdout is unsuitable for a long-lived bidirectional service.

## Consequences

Code generation and compatibility tests will be required. Loopback TCP, Unix domain sockets and Windows named pipes must be benchmarked and threat-modelled before selecting production transport.

