"""Runtime model selection for Theriad and direct assistant providers.

Theriad retains its router-generated provider/model candidates. Direct OpenAI
and Anthropic modes intentionally replace that chain with provider-native
models. Legacy Groq and DeepInfra mappings remain for benchmark compatibility,
but are not selectable in the live product.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from theriad.models.tiers import (
    TIER_CHEAP_BULK,
    TIER_CHEAP_CODE,
    TIER_CODE_STRONG,
    TIER_FREE_CODE,
    TIER_FREE_GENERAL,
    TIER_FREE_OR_FLOOR,
    TIER_FREE_SUMMARIZE,
    TIER_FREE_VERIFY,
    TIER_PREMIUM,
    TIER_STRONG_REASONING,
)

ProviderCandidate = tuple[str, str]

OPENAI_CHEAP_MODEL: Final = "gpt-5.4-mini"
OPENAI_STRONG_MODEL: Final = "gpt-5.6-terra"
OPENAI_PREMIUM_MODEL: Final = "gpt-5.6-sol"

ANTHROPIC_CHEAP_MODEL: Final = "claude-haiku-4-5-20251001"
ANTHROPIC_STRONG_MODEL: Final = "claude-sonnet-5"
ANTHROPIC_PREMIUM_MODEL: Final = "claude-opus-4-8"

GROQ_CHEAP_MODEL: Final = "openai/gpt-oss-20b"
GROQ_STRONG_MODEL: Final = "openai/gpt-oss-120b"
DEEPINFRA_CHEAP_MODEL: Final = "deepseek-ai/DeepSeek-V4-Flash"
DEEPINFRA_PREMIUM_MODEL: Final = "deepseek-ai/DeepSeek-V4-Pro"

_CHEAP_OR_FREE_TIERS = frozenset(
    {
        TIER_FREE_OR_FLOOR,
        TIER_CHEAP_BULK,
        TIER_CHEAP_CODE,
        TIER_FREE_GENERAL,
        TIER_FREE_CODE,
        TIER_FREE_VERIFY,
        TIER_FREE_SUMMARIZE,
    }
)
_STRONG_TIERS = frozenset({TIER_CODE_STRONG, TIER_STRONG_REASONING})

DIRECT_MODEL_BY_PROVIDER_AND_TIER: Final[dict[str, dict[str, str]]] = {
    "openai": {
        **{tier: OPENAI_CHEAP_MODEL for tier in _CHEAP_OR_FREE_TIERS},
        **{tier: OPENAI_STRONG_MODEL for tier in _STRONG_TIERS},
        TIER_PREMIUM: OPENAI_PREMIUM_MODEL,
    },
    "anthropic": {
        **{tier: ANTHROPIC_CHEAP_MODEL for tier in _CHEAP_OR_FREE_TIERS},
        **{tier: ANTHROPIC_STRONG_MODEL for tier in _STRONG_TIERS},
        TIER_PREMIUM: ANTHROPIC_PREMIUM_MODEL,
    },
    "groq": {
        **{tier: GROQ_CHEAP_MODEL for tier in _CHEAP_OR_FREE_TIERS},
        **{tier: GROQ_STRONG_MODEL for tier in _STRONG_TIERS},
        TIER_PREMIUM: GROQ_STRONG_MODEL,
    },
    "deepinfra": {
        **{tier: DEEPINFRA_CHEAP_MODEL for tier in _CHEAP_OR_FREE_TIERS},
        **{tier: DEEPINFRA_CHEAP_MODEL for tier in _STRONG_TIERS},
        TIER_PREMIUM: DEEPINFRA_PREMIUM_MODEL,
    },
}


def get_provider_model_for_tier(assistant_provider: str, tier: str) -> str:
    """Return a bare, provider-native model ID for a direct assistant mode."""
    provider = assistant_provider.strip().lower()
    if provider not in DIRECT_MODEL_BY_PROVIDER_AND_TIER:
        raise ValueError(f"Direct model mapping is unavailable for provider: {assistant_provider}")
    try:
        return DIRECT_MODEL_BY_PROVIDER_AND_TIER[provider][tier]
    except KeyError as exc:
        raise ValueError(f"Unsupported routing tier: {tier}") from exc


def get_provider_candidates(
    assistant_provider: str,
    tier: str,
    routed_candidates: Iterable[ProviderCandidate] | None = None,
) -> list[ProviderCandidate]:
    """Resolve execution candidates while honoring the selected provider boundary."""
    provider = assistant_provider.strip().lower()
    if provider in {"theriad", "openrouter"}:
        return list(routed_candidates or ())
    model = get_provider_model_for_tier(provider, tier)
    return [(provider, model)]


def normalize_direct_model_id(assistant_provider: str, model: str) -> str:
    """Strip only the selected direct provider's OpenRouter-style prefix."""
    provider = assistant_provider.strip().lower()
    prefix = f"{provider}/"
    if provider in {"openai", "anthropic", "groq", "deepinfra"} and model.startswith(prefix):
        return model[len(prefix) :]
    return model
