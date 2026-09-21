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

## Measured run: 2026-09-21, session-corrected controlled validation 02

Source commit: `1085781c2d29d40e8e6a7eb771b426fd73e59a74` (clean). The operator reported substantial visible deviation between the intended point and the rendered continuous prediction even while fixating the marker; the measurements support that report.

| Metric | Observed value |
| --- | ---: |
| Protocol completed | 4 practice + 8 scored trials |
| Frames total / usable | 2,915 / 2,859 |
| Usable fraction | 98.0789% |
| Correct / incorrect / no selection, scored only | 7 / 1 / 0 |
| Scored region accuracy | 87.5000% |
| Selection completion rate | 100.0000% |
| Median correct response | 1.2157 s |
| Raw outside-screen fraction after practice | 30.1297% |
| Corrected outside-screen fraction after practice | 0.1769% |
| Correction training error, median / P90 normalized | 0.2504 / 0.4182 |

Block 2 produced four correct selections. Block 3 produced three correct selections and one `TOP_LEFT` to `BOTTOM_LEFT` error. The correction therefore made the signal much more usable for coarse four-region selection: it removed all timeouts and nearly all outside-screen predictions. It did **not** produce point-accurate gaze. A median correction residual of 0.2504 normalized screen units is large and is consistent with the visible offset reported by the operator.

The upstream five-point calibration also had a held-out median error of 92.22 px, P90 of 169.01 px and especially weak `UP` behavior (143.52 px median, 37.35% closest-target hit rate). The next calibration iteration must directly sample diagonal targets and evaluate their held-out errors. Region-level success must not be presented as cursor-level precision.

The exact local schema-2 report is `work/interaction/history/controlled-region-validation-20260921T185334.324080Z.json`, SHA-256 `5e429fa4a61834e456cb94bad7fda67aa034e8dee83ad553b273acc253b4f2f9`. It contains derived metrics only, with no video, images or raw facial landmarks.
