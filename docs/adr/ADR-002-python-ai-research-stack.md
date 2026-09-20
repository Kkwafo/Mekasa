# ADR-002: Python for AI and computer-vision research

- Status: Accepted
- Date: 2026-09-19

## Context

V0 needs reproducible computer-vision experiments, numerical analysis and a future train-in-Python/export-to-ONNX path.

## Decision

Use Python 3.12+ with `uv`, OpenCV, MediaPipe Tasks and NumPy for Experiment 001. Add SciPy, scikit-learn, PyTorch, ONNX and ONNX Runtime only when an experiment requires them.

## Alternatives

Dart-only vision would reduce process count but has a weaker research ecosystem. A permanently embedded Python server would impede future mobile/on-device inference.

## Consequences

Python is the research and initial inference host, not a permanent product constraint. Exact compatible dependencies are committed in `uv.lock`.

