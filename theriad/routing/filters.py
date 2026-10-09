"""
Project-level model filter.

Applies per-project allow/block pattern lists and budget constraints to a
list of model candidates before execution, so project owners can prevent
specific models (or families) from being called on their behalf.

Pattern matching uses fnmatch-style globs:
    "ollama/*"            — any Ollama model
    "cohere/*"            — any Cohere model
    "xiaomi/mimo-v2.5*"   — MiMo v2.5 and all versioned releases
    "qwen/*"              — any Qwen model
    "anthropic/*"         — any Anthropic model
    "*:free"              — any free-tier model (suffix match)

Design decisions:
- Allowed patterns are an allowlist: if non-empty, only matching models pass.
  An empty allowlist means "all models allowed" (no restriction).
- Blocked patterns are a denylist: matching models are always excluded, even
  if they also match an allowed pattern.
- local_only excludes all non-Ollama models regardless of patterns.
- max_cost_per_call_usd blocks models whose per-call cost estimate exceeds
  the budget. Candidates with unknown cost (None) are passed through.
- allow_premium=False blocks models in the "premium" tier.

Usage:
    f = ModelFilter(
        allowed_model_patterns=["qwen/*", "ollama/*"],
        blocked_model_patterns=["anthropic/*"],
        max_cost_per_call_usd=0.01,
        allow_premium=False,
        local_only=False,
    )
    allowed, reason = f.is_allowed("qwen/qwen3-8b", "openrouter")
    candidates = f.filter_candidates([(prov, model, cost, tier), ...])
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field

from theriad.core.logging.logger import get_logger

logger = get_logger("routing.filters")

# Local provider identifiers — used for local_only enforcement
_LOCAL_PROVIDERS = frozenset({"ollama", "local"})


@dataclass
class ModelFilterSettings:
    """
    Project-level model filtering settings.

    Fields mirror the project_settings DB columns so callers can pass
    settings loaded from the database directly.
    """
    allowed_model_patterns: list[str] = field(default_factory=list)
    blocked_model_patterns: list[str] = field(default_factory=list)
    max_cost_per_call_usd: float | None = None
    allow_premium: bool = True
    local_only: bool = False


@dataclass
class FilterResult:
    """Outcome of a single model/provider filter check."""
    allowed: bool
    reason: str | None = None   # human-readable explanation when allowed=False


class ModelFilter:
    """
    Apply project-level allow/block patterns to a model candidate.

    Instantiate once per project settings load; call is_allowed() per candidate.
    """

    def __init__(self, settings: ModelFilterSettings):
        self._settings = settings

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_allowed(
        self,
        model: str,
        provider: str,
        *,
        tier: str | None = None,
        estimated_cost_usd: float | None = None,
    ) -> FilterResult:
        """
        Return FilterResult(allowed, reason).

        Check order (first failure wins):
          1. local_only — reject non-local providers
          2. blocked_model_patterns — reject if any pattern matches
          3. allowed_model_patterns — reject if patterns set and none match
          4. allow_premium=False — reject premium tier
          5. max_cost_per_call_usd — reject if cost estimate exceeds budget
        """
        s = self._settings

        # 1. local_only
        if s.local_only and provider not in _LOCAL_PROVIDERS:
            return FilterResult(
                False,
                f"project local_only=True: model '{model}' on provider '{provider}' is not local",
            )

        # 2. blocked patterns
        for pattern in s.blocked_model_patterns:
            if self._match(model, pattern):
                return FilterResult(
                    False,
                    f"model '{model}' matches blocked pattern '{pattern}'",
                )

        # 3. allowed patterns (non-empty = restrict to this list)
        if s.allowed_model_patterns:
            if not any(self._match(model, p) for p in s.allowed_model_patterns):
                return FilterResult(
                    False,
                    f"model '{model}' does not match any allowed pattern "
                    f"({s.allowed_model_patterns})",
                )

        # 4. premium gate
        if not s.allow_premium and tier == "premium":
            return FilterResult(
                False,
                f"model '{model}' is in premium tier and allow_premium=False",
            )

        # 5. cost cap (skip when cost is unknown)
        if (
            s.max_cost_per_call_usd is not None
            and estimated_cost_usd is not None
            and estimated_cost_usd > s.max_cost_per_call_usd
        ):
            return FilterResult(
                False,
                f"estimated cost ${estimated_cost_usd:.6f} exceeds project cap "
                f"${s.max_cost_per_call_usd:.6f}",
            )

        return FilterResult(True)

    def filter_candidates(
        self,
        candidates: list[tuple[str, str, float | None, str | None]],
    ) -> list[tuple[str, str, float | None, str | None]]:
        """
        Filter a list of (provider, model, cost, tier) tuples.

        Returns only candidates that pass is_allowed(). Logs each rejection.

        Args:
            candidates: list of (provider, model, estimated_cost_usd, tier)

        Returns:
            Filtered list in the same order.
        """
        allowed = []
        for provider, model, cost, tier in candidates:
            result = self.is_allowed(
                model, provider, tier=tier, estimated_cost_usd=cost
            )
            if result.allowed:
                allowed.append((provider, model, cost, tier))
            else:
                logger.info(
                    "model filtered by project settings",
                    extra={"model": model, "provider": provider, "reason": result.reason},
                )
        return allowed

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _match(model: str, pattern: str) -> bool:
        """fnmatch-style wildcard match. Case-sensitive (model slugs are case-sensitive)."""
        return fnmatch.fnmatch(model, pattern)
