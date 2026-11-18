# Third-party licenses

The Python environment is exactly recorded in `uv.lock`, but this file is not yet a complete redistribution notice. Before distributing binaries, generate and review a dependency license report and copy all required notices.

## Experiment 001 model

The MediaPipe Face Landmarker task bundle is fetched from Google's documented model URL and verified by checksum. It is not committed or redistributed by Mekasa. The model page displayed during the review did not state an artifact license in its model table, so redistribution remains blocked pending explicit license verification. See `models/gaze/mediapipe-face-landmarker/metadata.json`.

## Source dependencies

Experiment 001 uses MediaPipe, OpenCV contrib, NumPy and psutil, with test/build tools recorded in `uv.lock`. Package metadata is not a substitute for reviewing upstream license texts and bundled notices. No statement here changes an upstream license.

