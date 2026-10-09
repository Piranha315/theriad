"""
Model registry: maps tiers to ordered lists of ModelCandidate.

Design decisions:
- Default config is code, not YAML, so it ships without file-system dependencies.
  Users can override via theriad.yaml model_tiers section (loaded by config loader).
- Candidate order within a tier is significant: primary first, then fallbacks.
- Unknown model slugs do NOT hard-fail at registry construction time — providers
  validate slugs at call time so that config errors surface with useful context.
- Registry is intentionally read-only after construction (no add/remove at runtime)
  to avoid concurrent-modification surprises in async execution contexts.
"""
from __future__ import annotations

from .calibration import load_calibration
from .capabilities import index_to_quality
from .tiers import (
    TIER_CHEAP_BULK,
    TIER_CHEAP_CODE,
    TIER_CODE_STRONG,
    TIER_FREE_CODE,
    TIER_FREE_GENERAL,
    TIER_FREE_OR_FLOOR,
    TIER_FREE_SUMMARIZE,
    TIER_FREE_VERIFY,
    TIER_FRONTIER,
    TIER_PREMIUM,
    TIER_STRONG_REASONING,
    ModelCandidate,
)


def _c(**kw) -> ModelCandidate:
    """Shorthand constructor for the default config below."""
    return ModelCandidate(**kw)


DEFAULT_BASELINE_MODEL = "openai/gpt-6-luna"
"""Canonical premium comparison baseline used by current API and CLI defaults."""


# ── Default tier configuration ────────────────────────────────────────────────
#
# Model slugs are verified against OpenRouter at config authoring time.
# If a slug becomes unavailable, the provider will return a 4xx and the
# TheriadPipeline will try the next candidate in the fallback chain.
#
# quality_profile (ADR D-051) is populated ONLY where a per-axis measurement
# exists. Absent axes fall back to the aggregate quality_prior via
# ModelCandidate.quality_for(), so a partially-populated registry is correct —
# just less discriminating. Do not fill an axis in to make the table look
# complete: an invented axis score routes real traffic on a guess, whereas an
# absent one degrades to the aggregate we already trusted.
#
# NO PROFILES ARE POPULATED YET, deliberately. Because quality_for() falls back
# to quality_prior per-axis, profile scores and the existing priors are compared
# against each other inside a single ranking — so they must share a scale. The
# quality_prior values below are hand-authored on a 0.68–0.92 band; raw benchmark
# numbers (e.g. GLM-5.2's AA Coding Index of 68.8 → 0.688) are NOT on that scale,
# and mixing them would demote every profiled candidate against every unprofiled
# one for reasons that have nothing to do with capability.
#
# Populating this therefore requires a normalization decision first: pick one
# benchmark family per axis, score every candidate in a tier on it, and convert
# to the [0,1] band with a documented transform. Partial population is safe only
# *within* a tier where every candidate is profiled on the same axis.
#
# The prices below are SEEDS, not a record. Cost estimation resolves prices from
# the live OpenRouter catalog (see theriad.accounting.live_prices); these values
# only matter for a fully cold, fully offline start and for tier ordering that is
# authored by hand. Reconciled against the live catalog on 2026-07-28 —
# deepseek-v4-flash had been understated by ~43%, which made the router prefer it
# on a price advantage that did not exist. Re-reconcile when editing, and do not
# treat a stale seed here as evidence of what a model costs.

_DEFAULT_TIER_CONFIG: dict[str, list[ModelCandidate]] = {

    TIER_FREE_OR_FLOOR: [
        _c(
            tier=TIER_FREE_OR_FLOOR,
            provider="openrouter",
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            origin="US",
            role="primary",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=131072,
            quality_prior=0.68,
            source="openrouter_free_tier",
            notes="Free Nemotron Nano 30B; zero cost for trivial tasks.",
        ),
        _c(
            tier=TIER_FREE_OR_FLOOR,
            provider="openrouter",
            model="nvidia/nemotron-3-super-120b-a12b:free",
            origin="US",
            role="fallback",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=131072,
            quality_prior=0.78,
            source="openrouter_free_tier",
            notes="Free Nemotron Super 120B fallback; higher quality at still zero cost.",
        ),
    ],

    TIER_CHEAP_BULK: [
        _c(
            tier=TIER_CHEAP_BULK,
            provider="openrouter",
            model="nvidia/nemotron-3-super-120b-a12b:free",
            origin="US",
            role="primary",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=1_000_000,
            quality_prior=0.78,
            source="openrouter_free_tier",
            notes=(
                "Strictly better than the MiMo it replaced: higher prior (0.78 "
                "vs 0.76) at zero cost. The tradeoff is availability, not "
                "quality — free endpoints are rate-limited, so the paid "
                "candidates below exist to absorb overflow rather than to be "
                "better."
            ),
        ),
        _c(
            tier=TIER_CHEAP_BULK,
            provider="openrouter",
            model="openai/gpt-oss-20b",
            origin="US",
            role="fallback",
            input_cost_per_million=0.03,
            output_cost_per_million=0.13,
            context_window=131_072,
            quality_prior=0.76,
            source="openai_release_2026",
            notes=(
                "US-origin paid fallback. Under us_only(), the Nemotron free "
                "primary has no paid option when rate-limited — this fills that "
                "gap. Verify slug and pricing before committing; live OpenRouter "
                "catalog values were checked on 2026-08-09."
            ),
        ),
        _c(
            tier=TIER_CHEAP_BULK,
            provider="openrouter",
            model="xiaomi/mimo-v2.6-flash",
            origin="CN",
            role="fallback",
            input_cost_per_million=0.14,
            output_cost_per_million=0.28,
            context_window=1_048_576,
            quality_prior=0.76,
            source="openrouter_catalog_2026-09-24",
            notes="Repointed from xiaomi/mimo-v2.5 (2026-09-24) to its same-family successor, live OpenRouter catalog pricing. Quality prior carried over as a floor; measured eval outcomes carry over the same way (theriad/eval/successors.py). Cheap bulk model for summarization, extraction, rewrite.",
        ),
        _c(
            tier=TIER_CHEAP_BULK,
            provider="openrouter",
            model="deepseek/deepseek-v4.1-flash",
            origin="CN",
            role="fallback",
            # OpenRouter list price, served by the deepinfra/fp8 endpoint that
            # routing.py orders first. DeepSeek-hosted serving costs $0.60 out
            # and v4-flash carried peak/off-peak pricing there; unverified for 4.1.
            input_cost_per_million=0.14,
            output_cost_per_million=0.42,
            context_window=1_048_576,
            quality_prior=index_to_quality(50),
            source="artificial_analysis_2026-08",
            notes="Repointed from deepseek/deepseek-v4-flash-0731 (2026-09-24) to its same-family successor, live OpenRouter catalog pricing. Quality prior carried over as a floor; measured eval outcomes carry over the same way (theriad/eval/successors.py). Higher-quality cheap fallback; reasoning can increase response time.",
        ),
    ],

    TIER_CHEAP_CODE: [
        _c(
            tier=TIER_CHEAP_CODE,
            provider="openrouter",
            model="poolside/laguna-xs-2.1:free",
            origin="US",
            role="primary",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=262_144,
            quality_prior=0.74,
            source="openrouter_free_tier_2026-07-10",
            notes=(
                "Purpose-built coding model at zero cost. Prior is 0.06 below "
                "the qwen3.6-plus it replaced, which is a real quality "
                "concession — but qwen carried a $1.95/1M output price that made "
                "this tier cost 5.4x premium, inverting the cost ladder. Free "
                "first, paid on overflow."
            ),
        ),
        _c(
            tier=TIER_CHEAP_CODE,
            provider="openrouter",
            model="nvidia/nemotron-3-super-120b-a12b:free",
            origin="US",
            role="fallback",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=1_000_000,
            quality_prior=0.78,
            source="openrouter_free_tier",
            notes="Second free option before any paid call; general but capable.",
        ),
        _c(
            tier=TIER_CHEAP_CODE,
            provider="openrouter",
            model="openai/gpt-oss-20b",
            origin="US",
            role="fallback",
            input_cost_per_million=0.03,
            output_cost_per_million=0.13,
            context_window=131_072,
            quality_prior=0.76,
            source="openai_release_2026",
            notes=(
                "US-origin paid fallback for when free-tier rate limits are hit "
                "under RoutingConstraints.us_only(). Open-weight Apache 2.0, "
                "MoE architecture, tool-use capable. Replaces the role Qwen and "
                "MiMo play on unconstrained traffic. Verify slug and pricing on "
                "OpenRouter before committing; live catalog values were checked "
                "on 2026-08-09."
            ),
        ),
        _c(
            tier=TIER_CHEAP_CODE,
            provider="openrouter",
            model="qwen/qwen3.7-plus",
            origin="CN",
            role="fallback",
            input_cost_per_million=0.32,
            output_cost_per_million=1.28,
            context_window=1_000_000,
            quality_prior=0.80,
            source="openrouter_catalog_2026-09-24",
            notes="Repointed from qwen/qwen3.6-plus (2026-09-24) to its same-family successor, live OpenRouter catalog pricing. Quality prior carried over as a floor; measured eval outcomes carry over the same way (theriad/eval/successors.py). Cost-efficient code tier; short generation, refactoring.",
        ),
        _c(
            tier=TIER_CHEAP_CODE,
            provider="openrouter",
            model="xiaomi/mimo-v2.6-flash",
            origin="CN",
            role="fallback",
            input_cost_per_million=0.14,
            output_cost_per_million=0.28,
            context_window=1_048_576,
            quality_prior=0.76,
            source="openrouter_catalog_2026-09-24",
            notes="Repointed from xiaomi/mimo-v2.5 (2026-09-24) to its same-family successor, live OpenRouter catalog pricing. Quality prior carried over as a floor; measured eval outcomes carry over the same way (theriad/eval/successors.py). Cheap fallback from bulk tier.",
        ),
    ],

    TIER_CODE_STRONG: [
        _c(
            tier=TIER_CODE_STRONG,
            provider="openrouter",
            model="deepseek/deepseek-v4.1-flash",
            origin="CN",
            role="primary",
            # OpenRouter list price, served by the deepinfra/fp8 endpoint that
            # routing.py orders first. DeepSeek-hosted serving costs $0.60 out
            # and v4-flash carried peak/off-peak pricing there; unverified for 4.1.
            input_cost_per_million=0.14,
            output_cost_per_million=0.42,
            context_window=1_048_576,
            quality_prior=index_to_quality(50),
            source="artificial_analysis_2026-08",
            notes=(
                "Repointed from deepseek/deepseek-v4-flash-0731 (2026-09-24) to its same-family successor, live OpenRouter catalog pricing. Quality prior carried over as a floor; measured eval outcomes carry over the same way (theriad/eval/successors.py). "
                "Code-strong primary after the Kimi K2.7 removal. DeepSeek V4 "
                "Flash 0731 brings a 1M-token context and materially lower cost. "
                "Its live OpenRouter code-category arena score trails Kimi, and "
                "it is text-only, so this is a cost/context substitution rather "
                "than proof of universal superiority; the replacement study "
                "should validate it on Theriad's code workload."
            ),
        ),
        _c(
            tier=TIER_CODE_STRONG,
            provider="openrouter",
            model="qwen/qwen3.7-plus",
            origin="CN",
            role="fallback",
            input_cost_per_million=0.32,
            output_cost_per_million=1.28,
            context_window=1_000_000,
            quality_prior=0.80,
            source="openrouter_catalog_2026-09-24",
            notes="Repointed from qwen/qwen3.6-plus (2026-09-24) to its same-family successor, live OpenRouter catalog pricing. Quality prior carried over as a floor; measured eval outcomes carry over the same way (theriad/eval/successors.py). Cheaper fallback when DeepSeek 0731 is unavailable.",
        ),
        _c(
            tier=TIER_CODE_STRONG,
            provider="openrouter",
            model="poolside/laguna-xs-2.1:free",
            origin="US",
            role="fallback",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=262_144,
            quality_prior=0.74,
            source="openrouter_free_tier_2026-07-10",
            notes=(
                "Present so the non-China constraint is satisfiable on this "
                "tier at all — DeepSeek and Qwen are both CN, and a constraint the "
                "router cannot honour is worse than one it honours poorly. "
                "A real quality step down (0.74 against DeepSeek's 0.834), so "
                "US-only mode gives up meaningful ground on hard code. Closing "
                "that gap needs a paid US code model with a published index to "
                "convert from; none is in the registry yet."
            ),
        ),
        _c(
            tier=TIER_CODE_STRONG,
            provider="openrouter",
            model="poolside/laguna-s-2.1",
            origin="US",
            role="fallback",
            input_cost_per_million=0.09,
            output_cost_per_million=0.18,
            context_window=1_048_576,
            quality_prior=0.82,
            source="poolside_release_2026-07",
            notes=(
                "Paid 118B US-origin sibling of laguna-xs-2.1. Provides a viable "
                "paid option under RoutingConstraints.us_only(), where DeepSeek "
                "and Qwen are both excluded. quality_prior is provisional — run "
                "eval grid before promoting above XS in the unconstrained path. "
                "The OpenRouter slug and current pricing were verified on "
                "2026-08-09; recheck before production use."
            ),
        ),
    ],

    TIER_STRONG_REASONING: [
        _c(
            tier=TIER_STRONG_REASONING,
            provider="openrouter",
            model="nvidia/nemotron-3-super-120b-a12b:free",
            origin="US",
            role="primary",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=131072,
            quality_prior=0.82,
            source="openrouter_free_tier",
            notes="Free Nemotron Super 120B; strong reasoning at zero cost.",
        ),
        _c(
            tier=TIER_STRONG_REASONING,
            provider="openrouter",
            model="openai/gpt-oss-120b",
            origin="US",
            role="fallback",
            input_cost_per_million=0.037,
            output_cost_per_million=0.17,
            context_window=131_072,
            quality_prior=0.82,
            source="openai_release_2026",
            notes=(
                "US-origin paid fallback for strong reasoning under us_only(). "
                "Before this addition, DeepSeek was the only paid option in this "
                "tier and is CN-origin. GPT-OSS 120B is the open-weight US "
                "alternative at a "
                "quality level appropriate for the tier. Verify slug and pricing; "
                "live OpenRouter catalog values were checked on 2026-08-09."
            ),
        ),
        _c(
            tier=TIER_STRONG_REASONING,
            provider="openrouter",
            model="deepseek/deepseek-v4.1-flash",
            origin="CN",
            role="fallback",
            # OpenRouter list price, served by the deepinfra/fp8 endpoint that
            # routing.py orders first. DeepSeek-hosted serving costs $0.60 out
            # and v4-flash carried peak/off-peak pricing there; unverified for 4.1.
            input_cost_per_million=0.14,
            output_cost_per_million=0.42,
            context_window=1_048_576,
            quality_prior=index_to_quality(50),
            source="artificial_analysis_2026-08",
            notes="Repointed from deepseek/deepseek-v4-flash-0731 (2026-09-24) to its same-family successor, live OpenRouter catalog pricing. Quality prior carried over as a floor; measured eval outcomes carry over the same way (theriad/eval/successors.py). First paid reasoning fallback; strong price-quality result.",
        ),
    ],

    TIER_PREMIUM: [
        _c(
            tier=TIER_PREMIUM,
            provider="openrouter",
            model="deepseek/deepseek-v4.1-flash",
            origin="CN",
            role="primary",
            # OpenRouter list price, served by the deepinfra/fp8 endpoint that
            # routing.py orders first. DeepSeek-hosted serving costs $0.60 out
            # and v4-flash carried peak/off-peak pricing there; unverified for 4.1.
            input_cost_per_million=0.14,
            output_cost_per_million=0.42,
            context_window=1_048_576,
            quality_prior=index_to_quality(50),
            source="artificial_analysis_2026-08",
            notes=(
                "Repointed from deepseek/deepseek-v4-flash-0731 (2026-09-24) to its same-family successor, live OpenRouter catalog pricing. Quality prior carried over as a floor; measured eval outcomes carry over the same way (theriad/eval/successors.py). "
                "AA Intelligence Index 50, one point behind Luna, at roughly a "
                "third of Luna's price ($0.48/1k against $1.40 on this tier's "
                "profile; $0.31 at OpenRouter's rate). On AA's Pareto frontier "
                "for intelligence vs cost per task. "
                "MEASURED 2026-08-16: the eval grid, at full 200-item coverage, "
                "puts this at 0.825 against Luna's 0.940 and the FREE "
                "nemotron-3-super's 0.870, and its OpenRouter rate rose to "
                "0.14/0.28. Primacy was retained anyway because selection is by "
                "expected value, and at premium's V (12.0 hand-set, 3.21 "
                "calibrated) the +0.115 measured gap does not cover Luna's 2.6x "
                "cost — Luna needs V>=17 to win. Revisit when V is re-derived "
                "from a sweep that does not bottom out at its endpoints. "
                "CHINA-ORIGIN: this reverses the premise of "
                "docs/non-chinese-model-costs.md, which exists to keep "
                "China-origin models off the paid path. Retained deliberately "
                "as a price/quality decision that overrides that default — see "
                "the doc's postscript. Luna is the non-China head of the chain "
                "and is one fallback away."
            ),
        ),
        _c(
            tier=TIER_PREMIUM,
            provider="openrouter",
            model="openai/gpt-6-luna",
            origin="US",
            role="fallback",
            input_cost_per_million=0.10,
            output_cost_per_million=0.50,
            context_window=1_050_000,
            quality_prior=index_to_quality(51),   # AA Intelligence Index, max effort
            source="artificial_analysis_2026-08",
            notes=(
                "Repointed from openai/gpt-5.6-luna (2026-09-24) to its same-family successor, live OpenRouter catalog pricing. Quality prior carried over as a floor; measured eval outcomes carry over the same way (theriad/eval/successors.py). "
                "Ties GLM-5.2 at index 51 and costs 0.41x at list price "
                "(0.21x at OpenRouter's promotional rate), so this is a cost "
                "improvement rather than a concession. Caveat: ~121s end-to-end "
                "at max reasoning effort against GLM-5.2's ~13s. Lower effort "
                "trades index for speed (xhigh 49 at ~40s, high 46 at ~10s); if "
                "premium traffic is user-facing, set effort per request. "
                "MEASURED 2026-08-16: 0.940 on the eval grid at full coverage, "
                "the best premium candidate by 11.5 points over the primary. "
                "The index prior (0.845) understates that badly."
            ),
        ),
        _c(
            tier=TIER_PREMIUM,
            provider="openrouter",
            model="meta/muse-spark-1.3",
            origin="US",
            role="fallback",
            input_cost_per_million=1.25,
            output_cost_per_million=4.25,
            context_window=1_048_576,
            quality_prior=index_to_quality(51),
            source="artificial_analysis_2026-08",
            notes=(
                "Repointed from meta/muse-spark-1.1 (2026-09-24), skipping 1.2, "
                "at the same price. Availability is now confirmed: muse-spark-1.3 "
                "resolves on OpenRouter's meta endpoint and solved 6/10 HLE items "
                "as 1.2 in the eval pilot. Prior carried over as a floor. "
                "Ties Luna and GLM-5.2 at index 51, and is markedly less verbose "
                "(94M output tokens across the AA suite against GLM-5.2's 141M), "
                "so its per-task cost beats its per-token price. Cached input is "
                "0.15/1M, a ~9x discount. "
                "AVAILABILITY UNVERIFIED: a closed Meta model that may require a "
                "direct endpoint rather than OpenRouter. Carried as a deep "
                "fallback so a bad slug degrades to the next candidate instead "
                "of failing a request outright; promote only after a live call "
                "confirms it resolves."
            ),
        ),
        _c(
            tier=TIER_PREMIUM,
            provider="openrouter",
            model="x-ai/grok-4.7",
            origin="US",
            role="fallback",
            input_cost_per_million=1.60,
            output_cost_per_million=4.80,
            context_window=500_000,
            # Kept at 4.5's prior deliberately. 4.6 benchmarks at or above 4.5,
            # so this understates it; raising it on vendor claims would let an
            # unmeasured model outrank measured ones, which is the D-053 failure.
            # The grid corrects it either way.
            quality_prior=index_to_quality(54),
            source="artificial_analysis_2026-08",
            notes=(
                "Repointed from x-ai/grok-4.6 (2026-09-24) to its same-family successor, live OpenRouter catalog pricing. Quality prior carried over as a floor; measured eval outcomes carry over the same way (theriad/eval/successors.py). "
                "Repointed from grok-4.5 (2026-08-12): 4.6 supersedes it at "
                "identical $2.00/$6.00 and the same 500K window, so the swap is "
                "free. Three index points above Luna and GLM-5.2 at ~17s "
                "end-to-end, so this is the latency-sensitive choice. 2.53x "
                "GLM-5.2 per token, though AA cost-per-task puts it near parity "
                "because it is less verbose — the grid will settle which "
                "applies here. Cached input is 0.50/1M, a 75% discount. "
                "WARNING: past 200K tokens xAI doubles to $4.00/$12.00 and "
                "charges the higher rate on *every* token in the request, so "
                "the flat rate above understates long-context calls by 2x."
            ),
        ),
        _c(
            tier=TIER_PREMIUM,
            provider="openrouter",
            model="z-ai/glm-5.3",
            origin="CN",
            role="fallback",
            input_cost_per_million=0.84,
            output_cost_per_million=2.64,
            context_window=1_310_720,
            # Was 0.92, hand-authored. Recomputed from its published index of 51
            # via index_to_quality so it sits on the same scale as everything
            # above it. The demotion is a scale correction, not new evidence
            # that GLM-5.2 got worse.
            quality_prior=index_to_quality(51),
            source="artificial_analysis_2026-08",
            notes=(
                "Repointed from z-ai/glm-5.2 (2026-09-24) to its same-family successor, live OpenRouter catalog pricing. Quality prior carried over as a floor; measured eval outcomes carry over the same way (theriad/eval/successors.py). "
                "Retained as a fallback rather than removed: it ties Luna on "
                "the index and is the incumbent, so keeping it costs nothing "
                "and preserves a working path if Luna's latency proves "
                "unacceptable. China-origin — see docs/non-chinese-model-costs.md."
            ),
        ),
        # openai/gpt-4o was removed here (D-053). Its hand-authored 0.89 prior
        # was never benchmarked, and once the measured candidates above were
        # converted onto the published scale that stale guess outranked all of
        # them — so quality_first selected the tier's most expensive model
        # ($13.00/1k vs Luna's $1.40) on its weakest evidence. It is a 2024-era
        # model dominated on both price and published quality by everything that
        # remains, which is exactly the "remove on evidence" half of D-053.
        # Still the savings baseline in /api/stats; that is a separate use and
        # is unaffected.
    ],

    TIER_FRONTIER: [
        _c(
            tier=TIER_FRONTIER,
            provider="openrouter",
            model="z-ai/glm-5.3-flash",
            origin="CN",
            role="primary",
            # OpenRouter currently advertises a temporary 50% Z.AI discount at
            # $0.075/$0.25. Keep the offline seed at the non-promotional
            # $0.15/$0.50 list rate so a discount expiry cannot make the budget
            # guard or value ranking silently optimistic; live pricing overlays
            # pick up the current catalog rate at runtime.
            input_cost_per_million=0.15,
            output_cost_per_million=0.50,
            context_window=1_048_576,
            quality_prior=index_to_quality(57),
            source="artificial_analysis_2026-08",
            notes=(
                "Default frontier primary for unrestricted model_origin_policy=any "
                "routing. Artificial Analysis reports an Intelligence Index of "
                "57 (57.46 unrounded), slightly above Muse Spark 1.2's 56.76, "
                "and about $0.09 per Intelligence Index task. The OpenRouter "
                "catalog reports text, image, and video input, mandatory "
                "reasoning, a 1M-token context, and a 128K maximum completion. "
                "The public index is a prior, not a Theriad quality result; run "
                "the replacement study before treating the swap as validated."
            ),
        ),
        _c(
            tier=TIER_FRONTIER,
            provider="openrouter",
            model="meta/muse-spark-1.3",
            origin="US",
            role="fallback",
            input_cost_per_million=1.25,
            output_cost_per_million=4.25,
            context_window=1_048_576,
            quality_prior=0.892,
            source="openrouter_design_arena_2026-08",
            notes=(
                "Repointed from meta/muse-spark-1.2 (2026-09-24) to its same-family successor, live OpenRouter catalog pricing. Quality prior carried over as a floor; measured eval outcomes carry over the same way (theriad/eval/successors.py). "
                "US-origin fallback retained for model_origin_policy=us_only: "
                "the origin filter excludes the GLM primary and leaves Muse as "
                "the best U.S. frontier candidate. The live OpenRouter Design "
                "Arena code-category signal is 60.3% (1351) versus Opus at "
                "60.8% (1359), so this prior is provisional and not a Theriad "
                "quality result. Live pricing is $1.25/$4.25 per 1M tokens; "
                "re-run the replacement study before treating either route as "
                "quality-validated."
            ),
        ),
        _c(
            tier=TIER_FRONTIER,
            provider="openrouter",
            model="x-ai/grok-4.7",
            origin="US",
            role="fallback",
            input_cost_per_million=1.60,
            output_cost_per_million=4.80,
            context_window=500_000,
            quality_prior=index_to_quality(54),
            source="artificial_analysis_2026-08",
            notes=(
                "Repointed from x-ai/grok-4.6 (2026-09-24) to its same-family successor, live OpenRouter catalog pricing. Quality prior carried over as a floor; measured eval outcomes carry over the same way (theriad/eval/successors.py). "
                "Repointed from grok-4.5 (2026-08-12); same price, same window. "
                "Vendor diversity: a different lab from the primary, so the tier "
                "degrades to a genuinely independent model rather than another "
                "endpoint of the same one. Note this is the *smallest* context "
                "in the tier (500K vs muse-spark-1.2's 1M) and the dearest on "
                "output, and it carries a 2x surcharge above 200K tokens."
            ),
        ),
    ],

    TIER_FREE_GENERAL: [
        _c(
            tier=TIER_FREE_GENERAL,
            provider="openrouter",
            model="nvidia/nemotron-3-super-120b-a12b:free",
            origin="US",
            role="primary",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=1_000_000,
            quality_prior=0.78,
            source="openrouter_free_tier_2026-07-10",
            notes="MoE general pool primary; strong general reasoning at zero cost.",
        ),
        _c(
            tier=TIER_FREE_GENERAL,
            provider="openrouter",
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            origin="US",
            role="fallback",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=256_000,
            quality_prior=0.68,
            source="openrouter_free_tier_2026-07-10",
            notes="MoE general pool fallback.",
        ),
    ],

    TIER_FREE_CODE: [
        _c(
            tier=TIER_FREE_CODE,
            provider="openrouter",
            model="poolside/laguna-xs-2.1:free",
            origin="US",
            role="primary",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=262_144,
            quality_prior=0.74,
            source="openrouter_free_tier_2026-07-10",
            notes="MoE code pool primary; purpose-built coding model at zero cost.",
        ),
        _c(
            tier=TIER_FREE_CODE,
            provider="openrouter",
            model="cohere/north-mini-code:free",
            origin="CA",
            role="fallback",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=256_000,
            quality_prior=0.71,
            source="openrouter_free_tier_2026-07-10",
            notes="MoE code pool fallback; agentic coding specialist.",
        ),
        _c(
            tier=TIER_FREE_CODE,
            provider="openrouter",
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            origin="US",
            role="fallback",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=256_000,
            quality_prior=0.68,
            source="openrouter_free_tier",
            notes=(
                "US-origin free fallback. Cohere North Mini is CA-origin and is "
                "excluded under RoutingConstraints.us_only(), leaving Laguna XS "
                "as the only option. Nemotron Nano is already in other free-tier "
                "pools so the slug is verified; adding it here gives us_only() a "
                "second option when Laguna is rate-limited."
            ),
        ),
    ],

    TIER_FREE_VERIFY: [
        _c(
            tier=TIER_FREE_VERIFY,
            provider="openrouter",
            model="nvidia/nemotron-3-super-120b-a12b:free",
            origin="US",
            role="primary",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=1_000_000,
            quality_prior=0.78,
            source="openrouter_free_tier_2026-07-10",
            notes="MoE verify pool primary; strong reasoning for claim analysis.",
        ),
        _c(
            tier=TIER_FREE_VERIFY,
            provider="openrouter",
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            origin="US",
            role="fallback",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=256_000,
            quality_prior=0.68,
            source="openrouter_free_tier_2026-07-10",
            notes="MoE verify pool fallback.",
        ),
    ],

    TIER_FREE_SUMMARIZE: [
        _c(
            tier=TIER_FREE_SUMMARIZE,
            provider="openrouter",
            model="nvidia/nemotron-3-super-120b-a12b:free",
            origin="US",
            role="primary",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=1_000_000,
            quality_prior=0.78,
            source="openrouter_free_tier_2026-07-10",
            notes="MoE summarize pool primary; large context for extraction and rewriting.",
        ),
        _c(
            tier=TIER_FREE_SUMMARIZE,
            provider="openrouter",
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            origin="US",
            role="fallback",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            context_window=1_000_000,
            quality_prior=0.78,
            source="openrouter_free_tier_2026-07-10",
            notes="MoE summarize pool fallback for harder long-context work.",
        ),
    ],
}


# Measured calibration, when one has been generated, is merged over the table
# above at import time (see theriad.models.calibration). It only ever *adds* or
# *overrides* per-axis scores, never deletes them, and it stamps
# quality_profile_source with the grid it came from — so a score in this file
# that is still a judgement call remains visibly a judgement call.
_APPLIED_CALIBRATION = load_calibration()
_CALIBRATED_CANDIDATES = _APPLIED_CALIBRATION.apply_to_tiers(_DEFAULT_TIER_CONFIG)


class ModelRegistry:
    """
    Read-only registry mapping tiers to ordered ModelCandidate lists.

    Usage:
        registry = ModelRegistry()
        candidates = registry.candidates_for_tier("cheap_code")
        primary = registry.primary_for_tier("cheap_code")
    """

    def __init__(self, tier_config: dict[str, list[ModelCandidate]] | None = None):
        self._tiers: dict[str, list[ModelCandidate]] = tier_config or _DEFAULT_TIER_CONFIG

    def candidates_for_tier(self, tier: str) -> list[ModelCandidate]:
        """Return all candidates for a tier, in priority order. Empty list if tier unknown."""
        return list(self._tiers.get(tier, []))

    def primary_for_tier(self, tier: str) -> ModelCandidate | None:
        """Return the primary (role='primary') candidate for a tier, or first if none tagged."""
        candidates = self._tiers.get(tier, [])
        primary = next((c for c in candidates if c.role == "primary"), None)
        return primary or (candidates[0] if candidates else None)

    def fallback_chain_for_tier(self, tier: str) -> list[ModelCandidate]:
        """Return all non-primary candidates in registration order."""
        return [c for c in self._tiers.get(tier, []) if c.role != "primary"]

    def all_tiers(self) -> list[str]:
        return list(self._tiers.keys())


# Singleton default registry — avoids re-constructing the dict on every call.
DEFAULT_REGISTRY = ModelRegistry()
