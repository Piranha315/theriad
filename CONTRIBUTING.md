# Contributing to Theriad

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

## Linting

```bash
ruff check theriad tests
```

## Principles

1. **Clean interfaces first** — every module boundary must be a stable abstraction.
2. **Testability** — no untestable global state; inject dependencies.
3. **Logging** — every routing decision must be logged at DEBUG level.
4. **Simplicity over cleverness** — if you need a comment to explain it, simplify it.

## Adding a routing layer

1. Create `theriad/layers/<your_layer>/layer.py` implementing `RoutingLayer`.
2. Register it in `theriad/engine/pipeline.py`.
3. Add unit tests in `tests/unit/layers/`.
4. Update `ROADMAP.md`.

## Commit messages

`<type>(<scope>): <subject>` — e.g. `feat(layer0): add JSON detection rule`

Types: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`
