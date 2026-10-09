"""Provider-reported cost and cache details with a pricing-table fallback."""

from __future__ import annotations

from typing import Any

from theriad.core.providers.interfaces import CompletionResponse

from .pricing import PricingTable


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and value >= 0:
        return float(value)
    return None


def provider_reported_cost(response: CompletionResponse) -> float | None:
    """Read a billed cost when an upstream API includes it in usage metadata."""
    direct = _number(getattr(response, "cost_usd", None))
    if direct is not None:
        return direct
    raw = response.raw if isinstance(response.raw, dict) else {}
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    for value in (
        usage.get("cost"),
        usage.get("total_cost"),
        raw.get("cost"),
        raw.get("total_cost"),
    ):
        parsed = _number(value)
        if parsed is not None:
            return parsed
    return None


def cached_prompt_tokens(response: CompletionResponse) -> int:
    """Return provider-reported prompt cache reads across common response shapes."""
    raw = response.raw if isinstance(response.raw, dict) else {}
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        details = usage.get("input_tokens_details")
    details = details if isinstance(details, dict) else {}
    values = (
        details.get("cached_tokens"),
        usage.get("cache_read_input_tokens"),
        usage.get("cached_tokens"),
    )
    return max((int(value) for value in values if isinstance(value, int) and value > 0), default=0)


def usage_is_available(response: CompletionResponse) -> bool:
    """Return whether the provider supplied both actual token counters.

    Provider adapters normalize missing usage to zero for compatibility with
    older callers, so the dataclass fields alone cannot distinguish a real
    zero from an omitted usage block. Synthetic/local responses with no raw
    payload keep the historical token fields; real provider payloads must
    explicitly contain prompt/input and completion/output counters.
    """
    raw = response.raw if isinstance(response.raw, dict) else {}
    # The router annotates synthetic/test responses with private ``_theriad_*``
    # metadata after the provider call. That metadata is not evidence that a
    # provider payload omitted its usage block.
    provider_payload = {key: value for key, value in raw.items() if not key.startswith("_theriad_")}
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else None
    if usage is None:
        return not provider_payload and response.prompt_tokens >= 0 and response.completion_tokens >= 0
    prompt_key = next((key for key in ("prompt_tokens", "input_tokens") if key in usage), None)
    completion_key = next((key for key in ("completion_tokens", "output_tokens") if key in usage), None)
    if prompt_key is None or completion_key is None:
        return False
    return all(
        isinstance(usage[key], (int, float))
        and not isinstance(usage[key], bool)
        and usage[key] >= 0
        for key in (prompt_key, completion_key)
    )


def realized_cost(response: CompletionResponse, pricing: PricingTable) -> float | None:
    """Prefer provider billing; otherwise calculate only from actual usage."""
    reported = provider_reported_cost(response)
    if reported is not None:
        return reported
    if not usage_is_available(response):
        return None
    entry = pricing.get(response.model)
    if entry is None:
        return None
    return (
        response.prompt_tokens * entry.input_per_1m
        + response.completion_tokens * entry.output_per_1m
    ) / 1_000_000
