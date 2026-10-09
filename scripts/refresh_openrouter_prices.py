#!/usr/bin/env python3
"""Refresh the checked-in OpenRouter pricing snapshot.

Run from the repository root:
    python scripts/refresh_openrouter_prices.py

The runtime never performs this network request automatically. This keeps
routing deterministic and startup available when OpenRouter is down.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CATALOG_URL = "https://openrouter.ai/api/v1/models"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "theriad" / "accounting" / "openrouter_prices.json"
sys.path.insert(0, str(ROOT))


def registered_openrouter_models() -> dict[str, dict[str, float]]:
    from theriad.models.registry import DEFAULT_REGISTRY

    return {
        candidate.model: {
            "input_per_1m": candidate.input_cost_per_million,
            "output_per_1m": candidate.output_cost_per_million,
        }
        for tier in DEFAULT_REGISTRY.all_tiers()
        for candidate in DEFAULT_REGISTRY.candidates_for_tier(tier)
        if candidate.provider == "openrouter"
    }


def main() -> None:
    request = urllib.request.Request(CATALOG_URL, headers={"User-Agent": "Theriad/price-refresh"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)

    registered = registered_openrouter_models()
    wanted = set(registered)
    try:
        previous = json.loads(OUTPUT.read_text(encoding="utf-8")).get("models", {})
    except (OSError, ValueError, TypeError):
        previous = {}
    # Preserve the last known price when a model is transiently omitted from
    # the public catalog. A partial refresh must not silently turn pricing into
    # "unknown" for a production candidate.
    models: dict[str, dict[str, float]] = dict(registered)
    models.update({model: values for model, values in previous.items() if model in wanted})
    for model in payload.get("data", []):
        model_id = model.get("id")
        if model_id not in wanted:
            continue
        pricing = model.get("pricing") or {}
        try:
            models[model_id] = {
                "input_per_1m": float(pricing["prompt"]) * 1_000_000,
                "output_per_1m": float(pricing["completion"]) * 1_000_000,
            }
        except (KeyError, TypeError, ValueError):
            continue

    snapshot = {
        "refreshed_at": datetime.now(timezone.utc).date().isoformat(),
        "source": CATALOG_URL,
        "models": dict(sorted(models.items())),
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(f"Updated {OUTPUT} with {len(models)} models")


if __name__ == "__main__":
    main()
