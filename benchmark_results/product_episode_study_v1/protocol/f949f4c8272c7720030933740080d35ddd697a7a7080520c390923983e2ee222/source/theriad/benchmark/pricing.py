"""
Pricing tables for benchmark providers.

OpenRouter pricing comes from the main theriad.accounting.pricing module.
Groq, Cerebras, and Ollama are defined here since they are not part of the
main routing registry.

Prices are in USD per 1M tokens (input, output).
Sources: provider pricing pages as of 2025-06.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPrice:
    input_per_1m: float
    output_per_1m: float


# ── Groq ─────────────────────────────────────────────────────────────────────
# https://console.groq.com/docs/models
GROQ_PRICES: dict[str, ProviderPrice] = {
    "llama-3.3-70b-versatile":       ProviderPrice(0.59,  0.79),
    "llama-3.1-70b-versatile":       ProviderPrice(0.59,  0.79),
    "llama-3.1-8b-instant":          ProviderPrice(0.05,  0.08),
    "llama3-70b-8192":               ProviderPrice(0.59,  0.79),
    "llama3-8b-8192":                ProviderPrice(0.05,  0.08),
    "gemma2-9b-it":                  ProviderPrice(0.20,  0.20),
    "mixtral-8x7b-32768":            ProviderPrice(0.24,  0.24),
}

# ── Cerebras ──────────────────────────────────────────────────────────────────
# https://cerebras.ai/blog/cerebras-inference
CEREBRAS_PRICES: dict[str, ProviderPrice] = {
    "llama3.3-70b":  ProviderPrice(0.60, 0.60),
    "llama3.1-70b":  ProviderPrice(0.60, 0.60),
    "llama3.1-8b":   ProviderPrice(0.10, 0.10),
}

# ── Ollama ────────────────────────────────────────────────────────────────────
# Local inference — always free.
OLLAMA_PRICE = ProviderPrice(0.0, 0.0)


def estimate(
    price: ProviderPrice | None,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Return cost in USD or None if pricing is unavailable."""
    if price is None:
        return None
    return (
        prompt_tokens * price.input_per_1m
        + completion_tokens * price.output_per_1m
    ) / 1_000_000
