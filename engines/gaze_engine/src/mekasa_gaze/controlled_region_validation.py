from __future__ import annotations

from mekasa_gaze.runners.controlled_region_validation import (
    _build_report,
    main,
    run_controlled_region_validation,
)

__all__ = [
    "_build_report",
    "main",
    "run_controlled_region_validation",
]

if __name__ == "__main__":
    raise SystemExit(main())
