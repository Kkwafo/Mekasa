from __future__ import annotations

from mekasa_gaze.runners.region_evaluation import (
    _sha256,
    _write_report,
    main,
    run_region_evaluation,
)

__all__ = [
    "_sha256",
    "_write_report",
    "main",
    "run_region_evaluation",
]

if __name__ == "__main__":
    raise SystemExit(main())
