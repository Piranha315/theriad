"""Config loader — supports YAML files and env-var overrides."""
from __future__ import annotations

from pathlib import Path

import yaml

from .schema import TheriadConfig

_DEFAULT_PATHS = [
    Path("theriad.yaml"),
    Path("theriad.yml"),
    Path.home() / ".theriad" / "config.yaml",
]


def load_config(path: str | Path | None = None) -> TheriadConfig:
    """
    Load TheriadConfig from a YAML file.

    Decision: YAML chosen over TOML/JSON because it is human-friendly for
    nested structures and is the de-facto standard for ML tooling configs.
    """
    candidate = Path(path) if path else _locate_default()
    raw: dict = {}
    if candidate and candidate.exists():
        with open(candidate) as fh:
            raw = yaml.safe_load(fh) or {}
    return TheriadConfig.model_validate(raw)


def _locate_default() -> Path | None:
    for p in _DEFAULT_PATHS:
        if p.exists():
            return p
    return None
