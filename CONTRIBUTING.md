# Contributing to Mekasa

Mekasa welcomes contributions from AAC users, caregivers, SLPs/fonoaudiólogos, occupational therapists, PM&R specialists, vision professionals, rehabilitation engineers, accessibility/HCI practitioners, researchers and software engineers.

## Principles

- Treat AAC users as design participants and decision-makers, not passive test subjects.
- Do not make clinical claims or infer suitability from a diagnosis.
- Do not commit personal video, face images, health data or communication content.
- Distinguish synthetic tests, developer experiments, benchmarks and validated evidence.
- When tracking evidence is insufficient, prefer no selection over a guessed selection.
- Keep MediaPipe, models and camera providers behind domain contracts.

## Development

```powershell
uv sync --locked
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest
```

Hardware experiments are explicit and are not CI tests. Include the configuration, environment, negotiated camera settings, raw aggregate report and limitations with a performance claim.

By submitting a contribution, you agree that it is licensed under Apache-2.0 unless clearly stated otherwise and accepted by the maintainers.

