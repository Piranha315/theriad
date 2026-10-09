"""
Tier constants and ModelCandidate definition.

Design decisions:
- Tiers are plain string constants (not enums) for easy serialisation and config
  interop. Using an Enum would require conversion boilerplate everywhere tiers
  appear in config files and log output.
- ModelCandidate is a stdlib dataclass (not Pydantic) because it lives on the
  hot path; runtime validation is a separate config-load concern.
- Cost fields are per-million tokens, matching the industry standard used by
  OpenRouter and Anthropic pricing pages.
- quality_prior is [0, 1] — initialised from LMArena/public leaderboards but
  overridable by Theriad-specific eval results (ADR D-017).
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Tier constants ────────────────────────────────────────────────────────────

TIER_FREE_OR_FLOOR = "free_or_floor"
"""Lowest-cost execution: free models or zero-cost local inference."""

TIER_CHEAP_BULK = "cheap_bulk"
"""Cheap reliable model for summaries, extraction, rewriting, bulk transforms."""

TIER_CHEAP_CODE = "cheap_code"
"""Cost-efficient code / webdev tier — good LMArena code rankings at low price."""

TIER_CODE_STRONG = "code_strong"
"""Harder code tasks: multi-file changes, debugging, repo-level reasoning."""

TIER_STRONG_REASONING = "strong_reasoning"
"""Complex analysis and high-confidence reasoning, cheaper than premium frontier."""

TIER_PREMIUM = "premium"
"""Premium fallback only. Use when cheaper candidates fail or policy is quality_first."""

TIER_FRONTIER = "frontier"
"""Highest-quality tier available. Entered deliberately rather than as a routine
escalation target; its quality prior and cost profile are calibrated separately
from the premium tier and must be validated by the eval grid."""

TIER_FREE_GENERAL = "free_moe_general"
"""Experimental free-model pool for general chat and open-ended questions."""

TIER_FREE_CODE = "free_moe_code"
"""Experimental free-model pool for coding and debugging tasks."""

TIER_FREE_VERIFY = "free_moe_verify"
"""Experimental free-model pool for verification and fact-checking tasks."""

TIER_FREE_SUMMARIZE = "free_moe_summarize"
"""Experimental free-model pool for summarisation, extraction, and rewriting."""

TIER_FREE_MOE_TIERS: frozenset[str] = frozenset({
    TIER_FREE_GENERAL,
    TIER_FREE_CODE,
    TIER_FREE_VERIFY,
    TIER_FREE_SUMMARIZE,
})

VALID_TIERS: frozenset[str] = frozenset({
    TIER_FREE_OR_FLOOR,
    TIER_CHEAP_BULK,
    TIER_CHEAP_CODE,
    TIER_CODE_STRONG,
    TIER_STRONG_REASONING,
    TIER_PREMIUM,
    TIER_FRONTIER,
    *TIER_FREE_MOE_TIERS,
})

# ── Per-tier provider limits ───────────────────────────────────────────────────

TIER_MAX_TOKENS: dict[str, int] = {
    TIER_FREE_OR_FLOOR:    1200,
    TIER_CHEAP_BULK:       1200,
    TIER_CHEAP_CODE:       3500,
    TIER_CODE_STRONG:      6000,
    TIER_STRONG_REASONING: 3000,
    TIER_PREMIUM:          8000,
    TIER_FRONTIER:        16000,
    TIER_FREE_GENERAL:     1200,
    TIER_FREE_CODE:        1200,
    TIER_FREE_VERIFY:      1200,
    TIER_FREE_SUMMARIZE:   1200,
}
"""
Max output tokens to request per tier. Prevents cheap models from producing
runaway 20k-token responses that inflate cost and latency unexpectedly.
Config-driven: modify here to change limits project-wide.
"""

TIER_TIMEOUT_SECONDS: dict[str, float] = {
    TIER_FREE_OR_FLOOR:    12.0,
    TIER_CHEAP_BULK:       20.0,
    TIER_CHEAP_CODE:       45.0,
    TIER_CODE_STRONG:      90.0,
    TIER_STRONG_REASONING: 90.0,
    TIER_PREMIUM:          120.0,
    TIER_FRONTIER:         240.0,
    TIER_FREE_GENERAL:     12.0,
    TIER_FREE_CODE:        12.0,
    TIER_FREE_VERIFY:      12.0,
    TIER_FREE_SUMMARIZE:   12.0,
}
"""
HTTP timeout (seconds) for provider calls per tier. Larger timeouts for
reasoning/premium tiers that have longer time-to-first-token. Smaller
timeouts for cheap tiers catch runaway requests early and trigger fallback.
"""


# ── ModelCandidate ─────────────────────────────────────────────────────────────

@dataclass
class ModelCandidate:
    """
    A single concrete model that can be selected for a tier.

    Fields:
        tier:                   Logical tier this candidate belongs to.
        provider:               Provider name (openrouter, ollama, …).
        model:                  Provider-specific model slug.
        role:                   "primary" | "fallback" | "premium_fallback" | "experimental"
        input_cost_per_million: USD per 1M input tokens (None = unknown).
        output_cost_per_million:USD per 1M output tokens (None = unknown).
        context_window:         Maximum context in tokens (None = unknown).
        quality_prior:          [0, 1] aggregate quality estimate from public
                                benchmarks. Used when no per-axis score applies.
        source:                 Where quality_prior came from (e.g. "lmarena_2025-06").
        notes:                  Free-text rationale / caveats.
        quality_profile:        Optional {axis: [0,1]} per-capability scores; see
                                theriad.models.capabilities. Absent axes fall back
                                to quality_prior.
        origin:                 ISO-ish country code for the vendor ("US", "CN",
                                "CA"). Drives the one user-facing option Theriad
                                exposes: restricting routing to non-China vendors.
        quality_profile_source: Provenance for quality_profile. Required whenever
                                quality_profile is set — an unsourced axis score
                                routes real traffic on a guess.
    """
    tier: str
    provider: str
    model: str
    role: str = "primary"
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    context_window: int | None = None
    quality_prior: float | None = None
    source: str | None = None
    notes: str | None = None
    # Latency priors (ms) — groundwork for latency-aware policy selection.
    # Populated from benchmark data or empirical logs; None = unknown.
    # Future policies can use these for "cheapest viable under latency budget."
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    # Per-capability quality (ADR D-051). None = no per-axis data; the resolver
    # then behaves exactly as it did before profiles existed.
    quality_profile: dict[str, float] | None = None
    quality_profile_source: str | None = None
    # Vendor origin. Unset means unknown, which the origin filter treats as
    # non-compliant: a model we cannot attribute must not silently satisfy a
    # constraint the user set deliberately.
    origin: str | None = None

    def quality_for(self, axis: str | None) -> float | None:
        """Return this candidate's quality on ``axis``, falling back to the scalar.

        Resolution order:
          1. ``quality_profile[axis]`` when the axis is present
          2. ``quality_prior`` otherwise (including axis=None)

        Falling back rather than returning None keeps a partially-populated
        registry usable: a candidate with no measurement on this axis competes on
        its aggregate score instead of being silently scored as zero.
        """
        if axis and self.quality_profile:
            score = self.quality_profile.get(axis)
            if score is not None:
                return score
        return self.quality_prior

    def estimated_cost(self, prompt_tokens: int, output_tokens: int) -> float | None:
        """Return estimated USD cost, or None if pricing is unknown."""
        if self.input_cost_per_million is None or self.output_cost_per_million is None:
            return None
        return (
            prompt_tokens * self.input_cost_per_million
            + output_tokens * self.output_cost_per_million
        ) / 1_000_000
