# Models

Model binaries are not committed automatically. Every model requires explicit acquisition, provenance, checksum and license review.

Experiment 001 uses Google's MediaPipe Face Landmarker bundle. Fetch it explicitly from the repository root:

```powershell
./scripts/fetch-face-landmarker.ps1
```

The script verifies the exact byte length and SHA-256 recorded in the model metadata. Application startup never downloads a model.

