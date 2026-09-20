# Midas touch and unintended selection

In gaze interaction, looking at an item can mean reading or exploring it rather than intending to activate it. Mekasa therefore treats target entry as evidence of attention, not immediate communication intent.

## V0 design direction

Experiment 003 will evaluate a state machine with:

```text
OUTSIDE -> CANDIDATE -> DWELLING -> SELECTED -> COOLDOWN
                    \-> CANCELLED
```

- Dwell delays activation and exposes progress.
- Tracking loss pauses or resets progress according to measured behavior; it never completes selection.
- Target hysteresis keeps a candidate stable near a border without enlarging the actual activation region invisibly.
- Cooldown prevents one sustained look from repeatedly selecting.
- A gaze-accessible pause/cancel path must be present.
- Low confidence/uncertain tracking cannot produce selection.

## Values are configuration, not facts

The initial 800 ms dwell mentioned in the V0 specification is a hypothesis. Dwell, hysteresis margin, grace period and cooldown belong in centralized configuration and require synthetic sequence tests plus participatory evaluation. The system must not optimize only for fewer accidental selections if doing so makes intentional communication unacceptably slow or tiring.

## Metrics

Experiment 003 should report false selections, missed selections, time to selection, corrections, selections per intent, target transitions, cancellation use and tracking-loss behavior. `Effort Per Intent` is a research concept, not yet a clinically validated measure.

