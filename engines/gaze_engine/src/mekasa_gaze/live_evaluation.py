from __future__ import annotations

from mekasa_gaze.runners.live_evaluation import (
    PredictionSmoother,
    main,
    run_live_evaluation,
)

__all__ = [
    "PredictionSmoother",
    "main",
    "run_live_evaluation",
]

if __name__ == "__main__":
    raise SystemExit(main())
