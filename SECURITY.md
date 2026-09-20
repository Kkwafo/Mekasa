# Security policy

## Supported versions

Mekasa is pre-release research software. Only the current `main` branch receives security fixes.

## Reporting

Do not open a public issue for a vulnerability that could expose camera data, local communication content or code execution. Until a private reporting address is published, contact a repository maintainer privately through the hosting platform.

Do not include real face images, videos, health information, secrets or personal communication content in a report. Use synthetic reproduction data where possible.

## Current security boundaries

- Experiment 001 opens a local camera and does not start a network listener.
- Frames remain in process memory and are not written by the experiment.
- The model download is explicit and checksum-verified.
- Future IPC must be local-only and must consider hostile local processes; loopback alone is not authentication.

This policy does not claim that the prototype has completed a security audit.

