# Experiment 003 — Results

## Measured run: 2026-09-20, free abstract-region evaluation 01

Source commit: `b1c058338bab8f7b29ff8c82551e1a6111e6af5a` (clean). Calibration SHA-256: `0bb5dc038940e84ea2ff6667f0e5dcf4d87420f1384cb9ae511dfcb1c26c4af1`.

| Metric | Observed value |
| --- | ---: |
| Duration | 119.1101 s |
| Frames total / usable | 3,545 / 3,253 |
| Usable fraction | 91.7630% |
| Selection events | 31 |
| TOP_LEFT / TOP_RIGHT | 8 / 7 |
| BOTTOM_LEFT / BOTTOM_RIGHT | 8 / 8 |
| First selection | 27.1550 s |
| Median / P90 inter-selection interval | 2.2424 / 3.3884 s |
| Minimum / maximum inter-selection interval | 2.2048 / 16.6782 s |

The balanced counts and mostly regular event spacing establish that the dwell/cooldown state machine produced repeatable events during this operator-driven run. They do not establish accuracy, because the free runner did not record an intended region. The 292 unusable frames also cannot be separated retrospectively into placement, eye-validity and out-of-screen causes because schema 1 recorded only the aggregate usable count.

The exact report is preserved in `results/2026-09-20-windows-i5-8250u-free-region-evaluation-01.json` with SHA-256 `aa9cb33f9ed9d27155fd2a6053fb19ef887136c78160a05e31fc723001589514`. It contains derived selection events and environment metadata, but no video, images or raw facial landmarks.

## Controlled validation readiness

The audio-guided controlled runner is implemented prospectively. It randomizes all four requested regions in three complete blocks, separates cue and response intervals, records a confusion matrix and response timing, and distinguishes incorrect selections from timeouts. Loading, placement gating, tracking-loss reset, dwell, hysteresis and cooldown remain active. No controlled human result exists yet, so no selection-accuracy value is reported here.
