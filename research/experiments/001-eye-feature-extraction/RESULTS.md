# Experiment 001 — Results

## 2026-09-19 environment audit

- Host OS: Windows 10 Pro, 64-bit.
- CPU: Intel Core i5-8250U, 4 physical cores / 8 logical processors.
- Memory: 12,278,231,040 bytes reported by Windows (approximately 11.4 GiB).
- Camera enumerated by Windows: `Integrated Camera`, device status `OK`.
- Git: 2.54.0.windows.1.
- At initial inspection, Python, uv, Flutter/Dart and CMake were unavailable from the shell.

These are observed environment facts, not pipeline results.

## Measurement status

Pending. The first `uv` installation attempt through Windows Package Manager stalled while downloading dependencies over the available network. A Chocolatey fallback did not install without elevation. No Python pipeline was runnable at that point, so FPS, latency, tracking, stability and CPU values are intentionally absent.

The repository contains a reproducible environment definition, tests and an explicit checksum-verified model acquisition path. This section must be updated only after the command emits a real JSON report.

