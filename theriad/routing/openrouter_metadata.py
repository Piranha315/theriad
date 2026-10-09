"""Provider-free OpenRouter catalog verification for Router V2 plans.

Only the public ``/api/v1/models`` catalog is read here.  This module never
constructs a completion request and never reads a provider credential.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CATALOG_URL = "https://openrouter.ai/api/v1/models"
METADATA_SCHEMA_VERSION = 2


class OpenRouterMetadataError(RuntimeError):
    """Raised when public model metadata cannot authorize a study plan."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _per_million(value: Any) -> float:
    try:
        return float(value) * 1_000_000.0
    except (TypeError, ValueError) as exc:
        raise OpenRouterMetadataError(f"catalog contains a non-numeric price: {value!r}") from exc


def _model_record(row: dict[str, Any]) -> dict[str, Any]:
    pricing = row.get("pricing") or {}
    parameters = row.get("supported_parameters") or []
    if not isinstance(parameters, list):
        raise OpenRouterMetadataError(f"catalog supported_parameters is malformed for {row.get('id')}")
    return {
        "id": str(row.get("id") or ""),
        "name": str(row.get("name") or ""),
        "input_per_1m": _per_million(pricing.get("prompt")),
        "output_per_1m": _per_million(pricing.get("completion")),
        "supported_parameters": sorted(str(item) for item in parameters),
        "architecture": row.get("architecture") or {},
        "top_provider": row.get("top_provider") or {},
    }


def _relevant_metadata_sha256(
    *,
    url: str,
    requested_model_ids: tuple[str, ...],
    tool_model_ids: set[str],
    models: dict[str, dict[str, Any]],
) -> str:
    """Hash only metadata that can change eligibility or study economics.

    OpenRouter's catalog contains hundreds of unrelated models and changes
    frequently. Binding execution to the raw response made unrelated catalog
    edits invalidate an otherwise byte-identical Phase A model snapshot.
    """

    relevant = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "url": url,
        "requested_model_ids": list(requested_model_ids),
        "tool_model_ids": sorted(tool_model_ids),
        "models": models,
    }
    return hashlib.sha256(_canonical_json(relevant).encode("utf-8")).hexdigest()


def fetch_openrouter_metadata(
    model_ids: Iterable[str],
    *,
    tool_model_ids: Iterable[str] = (),
    timeout: float = 20.0,
    url: str = CATALOG_URL,
) -> dict[str, Any]:
    """Fetch and validate the exact public catalog rows needed by a phase."""

    requested = tuple(dict.fromkeys(str(value) for value in model_ids if str(value)))
    tool_ids = set(str(value) for value in tool_model_ids)
    if not requested:
        raise OpenRouterMetadataError("metadata preflight received no model IDs")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Theriad/router-v2-metadata-preflight"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_bytes = response.read()
    except (OSError, ValueError) as exc:
        raise OpenRouterMetadataError(f"OpenRouter public metadata is unavailable: {exc}") from exc
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenRouterMetadataError("OpenRouter public metadata is not valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise OpenRouterMetadataError("OpenRouter public metadata has no data list")
    by_id = {
        str(row.get("id")): row
        for row in payload["data"]
        if isinstance(row, dict) and row.get("id")
    }
    missing = [model for model in requested if model not in by_id]
    if missing:
        raise OpenRouterMetadataError("catalog is missing Phase A model(s): " + ", ".join(missing))
    models: dict[str, dict[str, Any]] = {}
    for model in requested:
        record = _model_record(by_id[model])
        parameters = set(record["supported_parameters"])
        if model in tool_ids and not {"tools", "tool_choice"}.issubset(parameters):
            raise OpenRouterMetadataError(
                f"{model} does not advertise both tools and tool_choice support"
            )
        models[model] = record
    fetched_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "url": url,
        "fetched_at": fetched_at,
        "response_sha256": _relevant_metadata_sha256(
            url=url,
            requested_model_ids=requested,
            tool_model_ids=tool_ids,
            models=models,
        ),
        "catalog_response_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "requested_model_ids": list(requested),
        "tool_model_ids": sorted(tool_ids),
        "models": models,
    }


def compare_metadata_to_plan(
    metadata: dict[str, Any],
    price_snapshot: dict[str, dict[str, Any]],
    *,
    expected_response_sha256: str | None = None,
) -> None:
    """Fail closed when relevant public metadata differs from a reviewed plan."""

    if metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
        raise OpenRouterMetadataError("unsupported metadata snapshot schema")
    if expected_response_sha256 and metadata.get("response_sha256") != expected_response_sha256:
        raise OpenRouterMetadataError(
            "relevant OpenRouter model metadata changed; regenerate and re-review the executable plan"
        )
    models = metadata.get("models")
    if not isinstance(models, dict):
        raise OpenRouterMetadataError("metadata snapshot has no model records")
    for model, expected in price_snapshot.items():
        actual = models.get(model)
        if not isinstance(actual, dict):
            raise OpenRouterMetadataError(f"metadata snapshot is missing {model}")
        for key in ("input_per_1m", "output_per_1m"):
            if abs(float(actual.get(key)) - float(expected.get(key))) > 1e-12:
                raise OpenRouterMetadataError(
                    f"public price drift for {model} {key}: "
                    f"plan={expected.get(key)!r}, catalog={actual.get(key)!r}"
                )
        if model.endswith(":free") and (
            float(actual.get("input_per_1m")) != 0.0
            or float(actual.get("output_per_1m")) != 0.0
        ):
            raise OpenRouterMetadataError("the frozen free Nemotron endpoint is no longer zero-priced")
    for model in metadata.get("tool_model_ids", []):
        actual = models.get(model) or {}
        if not {"tools", "tool_choice"}.issubset(set(actual.get("supported_parameters") or [])):
            raise OpenRouterMetadataError(
                f"{model} no longer advertises both tools and tool_choice support"
            )


def write_metadata_snapshot(metadata: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "CATALOG_URL",
    "METADATA_SCHEMA_VERSION",
    "OpenRouterMetadataError",
    "compare_metadata_to_plan",
    "fetch_openrouter_metadata",
    "write_metadata_snapshot",
]
