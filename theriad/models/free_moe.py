"""
free_moe_experimental — router-level Mixture-of-Experts policy.

Routes each request to the most appropriate free-tier expert pool using a
deterministic, in-process classifier. The policy is opt-in: existing routing
is unchanged unless ``policy="free_moe_experimental"`` is selected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from theriad.core.logging.logger import get_logger

from .policy import PolicyResolution, PolicyResolver
from .registry import DEFAULT_REGISTRY, ModelRegistry
from .tiers import (
    TIER_FREE_CODE,
    TIER_FREE_GENERAL,
    TIER_FREE_SUMMARIZE,
    TIER_FREE_VERIFY,
)

if TYPE_CHECKING:
    from theriad.routing.context import ContextEstimate
    from theriad.routing.continuity import ContinuitySignal
    from .unified import RoutingConstraints

logger = get_logger("models.free_moe")

_CODE_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("code_fence", re.compile(r"```")),
    ("def", re.compile(r"\bdef\s+[a-zA-Z_]")),
    ("class", re.compile(r"\bclass\s+[a-zA-Z_]")),
    ("function", re.compile(r"\bfunction\b")),
    ("import", re.compile(r"\bimport\s+[a-zA-Z_]")),
    ("bug", re.compile(r"\bbug\b")),
    ("error", re.compile(r"\berror\b")),
    ("debug", re.compile(r"\bdebug(?:ging)?\b")),
    ("typeerror", re.compile(r"\btypeerror\b")),
    ("syntax", re.compile(r"\bsyntax\b")),
    ("pytest", re.compile(r"\bpytest\b")),
    ("compile", re.compile(r"\bcompil(?:e|er|ing|ation)\b")),
    ("runtime", re.compile(r"\bruntime\b")),
    ("segfault", re.compile(r"\bsegfault\b")),
)

_VERIFY_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fact-check", re.compile(r"\bfact[ -]?check(?:ing)?\b")),
    ("is it true", re.compile(r"\bis\s+it\s+true\b")),
    ("verify", re.compile(r"\bverif(?:y|ies|ied|ication)\b")),
    ("claim", re.compile(r"\bclaims?\b")),
    ("source", re.compile(r"\bsources?\b")),
    ("citation", re.compile(r"\bcitations?\b")),
    ("according to", re.compile(r"\baccording\s+to\b")),
    ("does-cause", re.compile(r"\bdoes\b.{1,120}\bcause\b")),
)

_SUMMARIZE_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("summarize", re.compile(r"\bsummari[sz](?:e|es|ed|ing|ation)\b")),
    ("tldr", re.compile(r"\btl\s*;?\s*dr\b")),
    ("extract", re.compile(r"\bextract(?:ion|ed|ing|s)?\b")),
    ("key points", re.compile(r"\bkey\s+points?\b")),
    ("bullet points", re.compile(r"\bbullet\s+points?\b")),
    ("main idea", re.compile(r"\bmain\s+idea\b")),
)


@dataclass(frozen=True)
class FreeMoeDecision:
    """Deterministic expert-pool selection for an opt-in free MoE request."""

    tier: str
    pool: str
    confidence: float
    matched_signal: str
    is_moe: bool = True


def _first_match(
    message: str, signals: tuple[tuple[str, re.Pattern[str]], ...]
) -> str | None:
    for name, pattern in signals:
        if pattern.search(message):
            return name
    return None


def route_free_moe(message: str) -> FreeMoeDecision:
    """Classify ``message`` into one free expert pool without I/O or model calls."""
    normalized = (message or "").lower()

    matched = _first_match(normalized, _CODE_SIGNALS)
    if matched is not None:
        decision = FreeMoeDecision(TIER_FREE_CODE, "code_free", 0.9, matched)
    else:
        matched = _first_match(normalized, _VERIFY_SIGNALS)
        if matched is not None:
            decision = FreeMoeDecision(TIER_FREE_VERIFY, "verify_free", 0.9, matched)
        else:
            matched = _first_match(normalized, _SUMMARIZE_SIGNALS)
            if matched is not None:
                decision = FreeMoeDecision(
                    TIER_FREE_SUMMARIZE, "summarize_free", 0.9, matched
                )
            elif len(message or "") > 800:
                decision = FreeMoeDecision(
                    TIER_FREE_SUMMARIZE,
                    "summarize_free",
                    0.9,
                    "long_message",
                )
            else:
                decision = FreeMoeDecision(
                    TIER_FREE_GENERAL,
                    "general_free",
                    0.7,
                    "default",
                )

    logger.debug(
        "free_moe_experimental routing",
        extra={
            "tier": decision.tier,
            "pool": decision.pool,
            "confidence": decision.confidence,
            "matched_signal": decision.matched_signal,
        },
    )
    return decision


class FreeMoeResolver:
    """Select a free expert pool, then reuse PolicyResolver candidate scoring."""

    def __init__(self, registry: ModelRegistry | None = None):
        self._registry = registry or DEFAULT_REGISTRY
        self._resolver = PolicyResolver(self._registry)

    def resolve(
        self,
        message: str,
        confidence: float = 1.0,
        context_estimate: ContextEstimate | None = None,
        continuity_signal: ContinuitySignal | None = None,
        constraints: RoutingConstraints | None = None,
    ) -> PolicyResolution:
        decision = route_free_moe(message)
        resolution = self._resolver.resolve(
            decision.tier,
            policy="aggressive_savings",
            confidence=confidence,
            context_estimate=context_estimate,
            continuity_signal=continuity_signal,
            constraints=constraints,
        )
        return replace(resolution, policy="free_moe_experimental")


_DEFAULT_FREE_MOE_RESOLVER: FreeMoeResolver | None = None


def get_free_moe_resolver() -> FreeMoeResolver:
    """Return the process-wide default resolver singleton."""
    global _DEFAULT_FREE_MOE_RESOLVER
    if _DEFAULT_FREE_MOE_RESOLVER is None:
        _DEFAULT_FREE_MOE_RESOLVER = FreeMoeResolver()
    return _DEFAULT_FREE_MOE_RESOLVER
