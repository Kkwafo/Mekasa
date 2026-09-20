# ADR-006: Train in Python and prepare for ONNX deployment

- Status: Accepted as future direction
- Date: 2026-09-19

## Context

Research benefits from PyTorch, while a product should not require a Python service forever.

## Decision

Keep training/evaluation in Python and define traceable model metadata. When a learned model is justified, test export to ONNX and parity in ONNX Runtime before considering on-device integration.

## Alternatives

Shipping PyTorch is simpler for research but heavy for device deployment. Writing a neural model directly in platform code would make research and reproduction harder.

## Consequences

No PyTorch/ONNX dependency is added before Experiment 002 or later needs it. Export parity, operator support, quantization effects and licensing become explicit gates.

