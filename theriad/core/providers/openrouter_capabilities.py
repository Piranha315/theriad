"""Explicit, snapshot-bound request compatibility; no inference-time discovery.

Sampling knobs may use the endpoint's native defaults. Required task semantics
(tools, JSON formats, stops and output bounds) must never be silently removed.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from .interfaces import ProviderError

LUNA = "openai/gpt-5.6-luna"
SNAPSHOT_KEY = "openrouter_capability_snapshot"
_SAMPLING_DEFAULTS = {"temperature", "top_p", "frequency_penalty", "presence_penalty"}
_ENVELOPE = {"model", "messages", "provider"}


class CapabilityPreflightError(ProviderError):
    """Local rejection with proof that no billable request was dispatched."""


def catalog_digest(catalog: dict) -> str:
    return hashlib.sha256(json.dumps(catalog, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def compatible_payload(payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Resolve one exact endpoint, or reject before any HTTP dispatch.

    Other models retain their existing payloads unless explicitly opted in.
    Luna requires an archived, fresh snapshot even outside the study runner.
    """
    snapshot = metadata.get(SNAPSHOT_KEY)
    models = payload.get("models") or [payload.get("model")]
    if snapshot is None and LUNA not in models:
        return payload

    def fail(reason: str) -> None:
        raise CapabilityPreflightError("openrouter", f"capability preflight: {reason}", status_code=400)

    if not isinstance(snapshot, dict):
        fail("a fresh archived endpoint snapshot is required")
    if payload.get("models") or not isinstance(payload.get("model"), str):
        fail("capability amendments require one explicit model, without model fallbacks")
    try:
        fetched = datetime.fromisoformat(snapshot["retrieved_at"])
        age = (datetime.now(timezone.utc) - fetched).total_seconds()
        catalog = snapshot["catalog"]
        data = catalog["data"]
        tag = snapshot["endpoint_tag"]
        if catalog_digest(catalog) != snapshot["catalog_digest"]:
            fail("endpoint catalog digest mismatch")
    except (KeyError, TypeError, ValueError):
        fail("invalid endpoint snapshot")
    if not 0 <= age <= 86400:
        fail("endpoint snapshot is stale or future dated; refresh before collection")
    if data.get("id") != payload["model"] or not isinstance(tag, str) or "/" not in tag:
        fail("model mismatch or ambiguous base provider slug")
    endpoints = [e for e in data.get("endpoints", []) if e.get("tag") == tag]
    if len(endpoints) != 1:
        fail("exact provider endpoint absent or duplicated")
    endpoint = endpoints[0]
    if type(endpoint.get("status")) is not int or endpoint["status"] != 0:
        fail("selected provider endpoint is unavailable")
    supported = endpoint.get("supported_parameters")
    if not isinstance(supported, list) or not all(isinstance(p, str) for p in supported):
        fail("missing endpoint capabilities")
    result = deepcopy(payload)
    provider = result.setdefault("provider", {})
    if not isinstance(provider, dict):
        fail("invalid provider preferences")
    if provider.get("require_parameters", True) is not True:
        fail("require_parameters must remain true")
    for key in ("only", "order"):
        if key in provider and provider[key] != [tag]:
            fail("provider preferences conflict with the exact endpoint")
    if tag in provider.get("ignore", []) or tag.split("/")[0] in provider.get("ignore", []):
        fail("selected endpoint is ignored")
    provider.update(only=[tag], order=[tag], allow_fallbacks=False, require_parameters=True)
    for key in set(result) - _ENVELOPE:
        if key not in supported:
            if key in _SAMPLING_DEFAULTS:
                result.pop(key)
            else:
                fail(f"selected endpoint does not support required parameter {key}")
    if "max_tokens" in result:
        limit = endpoint.get("max_completion_tokens")
        requested = result["max_tokens"]
        if type(limit) is not int or type(requested) is not int or not 0 < requested <= limit:
            fail("output bound exceeds the endpoint capability")
    if "tool_choice" in result:
        choice = result["tool_choice"]
        mode = "function" if isinstance(choice, dict) and choice.get("type") == "function" else choice
        if not isinstance(mode, str) or endpoint.get("supports_tool_choice", {}).get(mode) is not True:
            fail("unsupported tool choice mode")
    return result
