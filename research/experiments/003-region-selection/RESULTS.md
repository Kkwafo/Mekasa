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

## Measured run: 2026-09-21, controlled validation baseline 01

Source commit: `7ded07c4ad8a26bb29c1137c95d77dad0b696c0c` (clean). Calibration SHA-256: `ed72c770b92bae240d96f8157b44518d6f943f1abdafc8a75bbebec590f2a959`.

| Metric | Observed value |
| --- | ---: |
| Protocol completed | 12 / 12 trials |
| Frames total / usable | 3,271 / 3,168 |
| Usable fraction | 96.8511% |
| Correct / incorrect / no selection | 7 / 1 / 4 |
| Accuracy across all trials | 58.3333% |
| Accuracy when a selection occurred | 87.5000% |
| Selection completion rate | 66.6667% |
| Median correct response | 1.2058 s |
| Dwell cancellations | 14 |
| Baseline predictions outside the screen | 1,117 / 3,271 frames (34.1486%) |

Performance changed sharply by block: block 1 produced one correct selection, one incorrect selection and two timeouts; block 2 produced two correct selections and two timeouts; block 3 produced four correct selections. The only wrong selection was `BOTTOM_LEFT` requested and `BOTTOM_RIGHT` selected. This learning/adaptation pattern means the aggregate 58.33% must not be treated as a stable accuracy estimate.

The run established that tracking and placement were mostly available, but the axial five-point calibration did not reliably cover the diagonal centers used by the four-region interface. Schema 1 counted outside-screen predictions across the whole run and could not compare raw versus corrected response frames. It also scored the initial adaptation block. These limitations motivated schema 2: the first complete block is now practice, learns a session-local diagonal correction, is excluded from accuracy, and records raw/corrected outside-screen counts per trial.

The exact local schema-1 report remains at `work/interaction/history/controlled-region-validation-20260921T182551.619633Z.json` (and, until another run, `work/interaction/controlled-region-validation-latest.json`). Its SHA-256 is `f15f2ecd146996b9152ba82fd1a55faa9ca409cd863e264a742c6d47f8ad3770`. It contains derived trial results and environment metadata, but no video, images or raw facial landmarks.

This is one technical operator session. It is not clinical validation, does not establish communication performance and does not support a medical claim.
