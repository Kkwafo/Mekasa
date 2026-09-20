from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from mekasa_gaze.camera import CameraUnavailableError, OpenCVCameraProvider


def probe(max_index: int, backend: str) -> list[dict[str, object]]:
    cameras: list[dict[str, object]] = []
    for index in range(max_index + 1):
        provider = OpenCVCameraProvider(
            index=index,
            backend=backend,
            width=640,
            height=480,
            target_fps=30.0,
        )
        try:
            with provider:
                frame = provider.read()
                cameras.append(
                    {
                        **asdict(provider.actual_settings),
                        "first_frame_shape": list(frame.bgr.shape),
                        "first_frame_capture_latency_ms": frame.capture_latency_ms,
                    }
                )
        except CameraUnavailableError:
            continue
    return cameras


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe local cameras without saving frames")
    parser.add_argument("--max-index", type=int, default=4)
    parser.add_argument("--backend", choices=("auto", "msmf", "dshow"), default="auto")
    args = parser.parse_args(argv)
    cameras = probe(args.max_index, args.backend)
    print(json.dumps({"cameras": cameras}, indent=2))
    return 0 if cameras else 2


if __name__ == "__main__":
    raise SystemExit(main())
