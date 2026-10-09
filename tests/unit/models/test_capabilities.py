"""Tests for capability-axis routing and epsilon-greedy exploration."""
from __future__ import annotations

import random

import pytest

from theriad.models.capabilities import (
    AXIS_BULK,
    AXIS_CODE,
    AXIS_KNOWLEDGE,
    AXIS_REASONING,
    VALID_AXES,
    axis_for_task_type,
)
from theriad.models.exploration import NO_EXPLORATION, ExplorationConfig
from theriad.models.policy import PolicyResolver
from theriad.models.registry import ModelRegistry
from theriad.models.tiers import TIER_CHEAP_BULK, ModelCandidate


def _make_registry(*candidates: ModelCandidate) -> ModelRegistry:
    by_tier: dict[str, list[ModelCandidate]] = {}
    for c in candidates:
        by_tier.setdefault(c.tier, []).append(c)
    return ModelRegistry(by_tier)


# A coder and a generalist with mirrored strengths on the same aggregate prior,
# so any difference in selection is attributable to the axis alone.
CODER = ModelCandidate(
    tier=TIER_CHEAP_BULK, provider="openrouter", model="vendor/coder",
    role="primary", input_cost_per_million=1.0, output_cost_per_million=1.0,
    quality_prior=0.80,
    quality_profile={AXIS_CODE: 0.95, AXIS_KNOWLEDGE: 0.60},
    quality_profile_source="test_fixture",
)
SCHOLAR = ModelCandidate(
    tier=TIER_CHEAP_BULK, provider="openrouter", model="vendor/scholar",
    role="fallback", input_cost_per_million=1.0, output_cost_per_million=1.0,
    quality_prior=0.80,
    quality_profile={AXIS_CODE: 0.60, AXIS_KNOWLEDGE: 0.95},
    quality_profile_source="test_fixture",
)


# ── axis mapping ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "task_type,expected",
    [
        ("code", AXIS_CODE),
        ("code_hard", AXIS_CODE),
        ("debugging", AXIS_CODE),
        ("qa", AXIS_KNOWLEDGE),
        ("analysis", AXIS_REASONING),
        ("architecture", AXIS_REASONING),
        ("summarize", AXIS_BULK),
        ("extraction", AXIS_BULK),
    ],
)
def test_task_type_maps_to_axis(task_type, expected):
    assert axis_for_task_type(task_type) == expected


def test_axis_lookup_is_case_and_space_insensitive():
    assert axis_for_task_type("  CODE  ") == AXIS_CODE


def test_unmapped_and_empty_task_types_return_none():
    # None is meaningful: it tells the resolver to use the aggregate prior.
    assert axis_for_task_type("interpretive_dance") is None
    assert axis_for_task_type("") is None
    assert axis_for_task_type(None) is None


def test_every_mapped_axis_is_a_valid_axis():
    for task in ["code", "qa", "analysis", "summarize", "debugging", "planning"]:
        assert axis_for_task_type(task) in VALID_AXES


# ── ModelCandidate.quality_for ────────────────────────────────────────────────

def test_quality_for_reads_the_profile_when_axis_present():
    assert CODER.quality_for(AXIS_CODE) == 0.95


def test_quality_for_falls_back_to_prior_on_unmeasured_axis():
    # REASONING is absent from the profile; must degrade to the aggregate, not 0.
    assert CODER.quality_for(AXIS_REASONING) == 0.80


def test_quality_for_falls_back_to_prior_when_axis_is_none():
    assert CODER.quality_for(None) == 0.80


def test_quality_for_on_unprofiled_candidate_returns_prior():
    bare = ModelCandidate(
        tier=TIER_CHEAP_BULK, provider="p", model="m", quality_prior=0.42
    )
    assert bare.quality_for(AXIS_CODE) == 0.42


# ── Capability-aware selection ────────────────────────────────────────────────

def test_quality_first_routes_code_to_the_coder():
    resolver = PolicyResolver(_make_registry(CODER, SCHOLAR))
    r = resolver.resolve(TIER_CHEAP_BULK, "quality_first", task_type="code")
    assert r.model == "vendor/coder"
    assert r.capability_axis == AXIS_CODE


def test_quality_first_routes_knowledge_to_the_scholar():
    """The same tier, same policy, same priors — only the task type differs."""
    resolver = PolicyResolver(_make_registry(CODER, SCHOLAR))
    r = resolver.resolve(TIER_CHEAP_BULK, "quality_first", task_type="qa")
    assert r.model == "vendor/scholar"
    assert r.capability_axis == AXIS_KNOWLEDGE


def test_without_task_type_selection_falls_back_to_aggregate():
    """Equal aggregates → the pre-capability tie-break (registration order) holds."""
    resolver = PolicyResolver(_make_registry(CODER, SCHOLAR))
    r = resolver.resolve(TIER_CHEAP_BULK, "quality_first")
    assert r.capability_axis is None
    assert r.model == "vendor/coder"


def test_unmapped_task_type_does_not_engage_capability_routing():
    resolver = PolicyResolver(_make_registry(CODER, SCHOLAR))
    r = resolver.resolve(TIER_CHEAP_BULK, "quality_first", task_type="nonsense")
    assert r.capability_axis is None


def test_axis_shifts_selection_only_when_the_quality_gain_pays():
    """Capability is a quality signal, and under the unified rule quality only
    wins when its gain outweighs the price of that gain."""
    cheap_scholar = ModelCandidate(
        tier=TIER_CHEAP_BULK, provider="openrouter", model="vendor/cheap-scholar",
        role="fallback", input_cost_per_million=0.01, output_cost_per_million=0.01,
        quality_prior=0.50, quality_profile={AXIS_CODE: 0.10},
        quality_profile_source="test_fixture",
    )
    # cheap_scholar is ~100x cheaper and 0.85 worse on the code axis. On
    # cheap_bulk (V=2.0) that gap is worth ~1.7 against a price gap of ~0.99,
    # so the coder wins; the point is that the comparison is now made rather
    # than decided by a flag.
    resolver = PolicyResolver(_make_registry(CODER, cheap_scholar))
    r = resolver.resolve(TIER_CHEAP_BULK, "balanced", task_type="code")
    assert r.model == "vendor/coder"
    assert r.capability_axis == AXIS_CODE


# ── Exploration ───────────────────────────────────────────────────────────────

def test_exploration_is_off_by_default():
    resolver = PolicyResolver(_make_registry(CODER, SCHOLAR))
    r = resolver.resolve(TIER_CHEAP_BULK, "quality_first", task_type="code")
    assert r.exploration_applied is False
    assert r.exploit_model == "vendor/coder"


def test_epsilon_zero_never_explores():
    resolver = PolicyResolver(_make_registry(CODER, SCHOLAR))
    for _ in range(50):
        r = resolver.resolve(
            TIER_CHEAP_BULK, "quality_first", task_type="code",
            exploration=ExplorationConfig(epsilon=0.0),
        )
        assert r.exploration_applied is False


def test_epsilon_one_always_explores_and_records_the_counterfactual():
    resolver = PolicyResolver(_make_registry(CODER, SCHOLAR))
    r = resolver.resolve(
        TIER_CHEAP_BULK, "quality_first", task_type="code",
        exploration=ExplorationConfig(epsilon=1.0, rng=random.Random(0)),
    )
    assert r.exploration_applied is True
    assert r.model == "vendor/scholar"      # the non-top candidate
    assert r.exploit_model == "vendor/coder"  # what we would have picked


def test_exploration_rate_is_approximately_epsilon():
    resolver = PolicyResolver(_make_registry(CODER, SCHOLAR))
    cfg = ExplorationConfig(epsilon=0.3, rng=random.Random(1234))
    explored = sum(
        resolver.resolve(
            TIER_CHEAP_BULK, "quality_first", task_type="code", exploration=cfg
        ).exploration_applied
        for _ in range(2000)
    )
    assert 0.25 < explored / 2000 < 0.35


def test_exploration_is_a_noop_for_single_candidate_tiers():
    resolver = PolicyResolver(_make_registry(CODER))
    r = resolver.resolve(
        TIER_CHEAP_BULK, "quality_first", task_type="code",
        exploration=ExplorationConfig(epsilon=1.0, rng=random.Random(0)),
    )
    assert r.exploration_applied is False
    assert r.model == "vendor/coder"


def test_exploration_stays_within_the_tier():
    """Exploration answers 'which candidate', never 'which cost band'."""
    resolver = PolicyResolver(_make_registry(CODER, SCHOLAR))
    tier_models = {"vendor/coder", "vendor/scholar"}
    for _ in range(100):
        r = resolver.resolve(
            TIER_CHEAP_BULK, "quality_first", task_type="code",
            exploration=ExplorationConfig(epsilon=0.5, rng=random.Random()),
        )
        assert r.model in tier_models
        assert r.tier == TIER_CHEAP_BULK


def test_exploration_is_reproducible_under_a_seeded_rng():
    resolver = PolicyResolver(_make_registry(CODER, SCHOLAR))

    def run(seed: int) -> list[bool]:
        cfg = ExplorationConfig(epsilon=0.5, rng=random.Random(seed))
        return [
            resolver.resolve(
                TIER_CHEAP_BULK, "quality_first", task_type="code", exploration=cfg
            ).exploration_applied
            for _ in range(20)
        ]

    assert run(7) == run(7)


def test_invalid_epsilon_is_rejected():
    with pytest.raises(ValueError):
        ExplorationConfig(epsilon=1.5)
    with pytest.raises(ValueError):
        ExplorationConfig(epsilon=-0.1)


def test_no_exploration_singleton_is_disabled():
    assert NO_EXPLORATION.enabled is False
    assert NO_EXPLORATION.should_explore() is False


# ── Benchmark normalization (D-053) ───────────────────────────────────────────

from theriad.models.capabilities import (  # noqa: E402
    AA_INDEX_CEILING,
    AA_INDEX_FLOOR,
    QUALITY_CEILING,
    QUALITY_FLOOR,
    index_to_quality,
)
from theriad.models.registry import DEFAULT_REGISTRY  # noqa: E402
from theriad.models.tiers import TIER_FRONTIER, TIER_PREMIUM  # noqa: E402


def test_transform_endpoints_land_on_the_band():
    assert index_to_quality(AA_INDEX_FLOOR) == QUALITY_FLOOR
    assert index_to_quality(AA_INDEX_CEILING) == QUALITY_CEILING


def test_transform_is_monotone():
    scores = [index_to_quality(i) for i in range(40, 65)]
    assert scores == sorted(scores)


def test_transform_clips_outside_the_measured_range():
    assert 0.0 <= index_to_quality(0) <= 1.0
    assert 0.0 <= index_to_quality(200) <= 1.0


def test_converted_scores_stay_inside_the_registry_band():
    """Converted priors must be comparable against candidates that have no
    published index, or partial population reorders the registry for reasons
    unrelated to capability."""
    for i in (42, 51, 54, 56, 61):
        assert QUALITY_FLOOR <= index_to_quality(i) <= QUALITY_CEILING


# ── The new candidates are actually routable ──────────────────────────────────

def test_luna_leads_the_non_china_path():
    """DeepSeek 0731 took the primary slot on price; Luna is the first
    non-China candidate and is one fallback away, so reverting the origin
    decision is a role swap and nothing more."""
    cands = [c.model for c in DEFAULT_REGISTRY.candidates_for_tier(TIER_PREMIUM)]
    luna = next(c for c in DEFAULT_REGISTRY.candidates_for_tier(TIER_PREMIUM)
                if c.model == "openai/gpt-6-luna")
    assert cands.index("openai/gpt-6-luna") == 1
    assert luna.quality_prior == index_to_quality(51)


def test_premium_is_cheaper_than_the_incumbent_it_replaced():
    cands = {c.model: c for c in DEFAULT_REGISTRY.candidates_for_tier(TIER_PREMIUM)}
    luna, glm = cands["openai/gpt-6-luna"], cands["z-ai/glm-5.3"]
    profile = (1600, 900)  # premium tier token profile
    cost = lambda c: (profile[0] * c.input_cost_per_million
                      + profile[1] * c.output_cost_per_million) / 1e6
    assert cost(luna) < cost(glm)


def test_glm_retained_as_a_fallback_rather_than_removed():
    """Promote on published evidence, demote on measurement — the incumbent
    stays reachable until the grid says otherwise."""
    models = [c.model for c in DEFAULT_REGISTRY.candidates_for_tier(TIER_PREMIUM)]
    assert "z-ai/glm-5.3" in models
    assert models.index("openai/gpt-6-luna") < models.index("z-ai/glm-5.3")


def test_frontier_tier_uses_glm_flash_for_unconstrained_routing():
    frontier = DEFAULT_REGISTRY.primary_for_tier(TIER_FRONTIER)
    premium = DEFAULT_REGISTRY.primary_for_tier(TIER_PREMIUM)
    assert frontier.model == "z-ai/glm-5.3-flash"
    assert (frontier.input_cost_per_million, frontier.output_cost_per_million) == (0.15, 0.50)
    assert frontier.quality_prior == index_to_quality(57)
    assert frontier.quality_prior > premium.quality_prior


def test_us_only_frontier_routing_keeps_muse():
    from theriad.models.unified import RoutingConstraints

    resolution = PolicyResolver().resolve(
        TIER_FRONTIER,
        "balanced",
        constraints=RoutingConstraints.us_only(),
    )
    assert resolution.model == "meta/muse-spark-1.3"


def test_frontier_fallback_is_a_different_vendor():
    """A fallback to another endpoint of the same model is not resilience."""
    cands = DEFAULT_REGISTRY.candidates_for_tier(TIER_FRONTIER)
    vendors = {c.model.split("/")[0] for c in cands}
    assert len(vendors) > 1


def test_every_converted_prior_records_its_source():
    for tier in (TIER_PREMIUM, TIER_FRONTIER):
        for c in DEFAULT_REGISTRY.candidates_for_tier(tier):
            assert c.source, f"{c.model} has no source"


def test_new_candidates_are_priceable():
    """An unpriced model resolves to $0.00 and would look free to the router."""
    from theriad.accounting.pricing import PricingTable

    pt = PricingTable(use_live=False)
    for tier in (TIER_PREMIUM, TIER_FRONTIER):
        for c in DEFAULT_REGISTRY.candidates_for_tier(tier):
            assert pt.get(c.model) is not None, f"{c.model} is unpriced"


def test_code_specialist_uses_deepseek_flash():
    """DeepSeek's published general index is now the code-strong prior."""
    from theriad.models.tiers import TIER_CODE_STRONG

    code_model = DEFAULT_REGISTRY.primary_for_tier(TIER_CODE_STRONG)
    assert code_model.model == "deepseek/deepseek-v4.1-flash"
    assert code_model.quality_prior == index_to_quality(50)


def test_kimi_is_not_in_the_active_default_registry():
    routed = {
        c.model
        for tier in DEFAULT_REGISTRY.all_tiers()
        for c in DEFAULT_REGISTRY.candidates_for_tier(tier)
    }
    assert "moonshotai/kimi-k2.7-code" not in routed


# ── DeepSeek V4 Flash 0731 in premium ─────────────────────────────────────────

def test_premium_primary_is_deepseek_flash():
    c = DEFAULT_REGISTRY.primary_for_tier(TIER_PREMIUM)
    assert c.model == "deepseek/deepseek-v4.1-flash"
    assert c.quality_prior == index_to_quality(50)


def test_premium_chain_is_ordered_by_value_not_registration():
    """The unified router orders fallbacks by expected value, so a failing
    primary degrades to the next best trade rather than the next line in the
    registry.

    Luna leads as of the 2026-08-22 repricing. DeepSeek led while its flat rate
    was $0.14/$0.28 and the 0.011 quality gap could not overcome a 0.275 cost
    gap. DeepSeek's 2026-08-16 move to peak/off-peak pricing cut that cost gap
    to 0.024 on the time-weighted average, at which the quality difference
    decides and Luna wins. DeepSeek is *still the cheaper model* — see
    ``test_deepseek_0731_is_cheaper_than_luna`` — so this is expected value
    changing hands, not a price inversion.
    """
    from theriad.models.policy import PolicyResolver
    from theriad.models.unified import RoutingConstraints, expected_value

    resolution = PolicyResolver().resolve(TIER_PREMIUM, "balanced")
    assert resolution.model == "openai/gpt-6-luna"
    evs = [expected_value(c, None, TIER_PREMIUM) for c in resolution.fallback_chain]
    assert evs == sorted(evs, reverse=True)

    # DeepSeek is now the first fallback rather than the primary.
    assert resolution.fallback_chain[0].model == "deepseek/deepseek-v4.1-flash"

    # Luna is the first non-China option, so excluding China does not move the
    # winner now that it already is the winner.
    us_only = PolicyResolver().resolve(
        TIER_PREMIUM, "balanced", constraints=RoutingConstraints.exclude_china()
    )
    assert us_only.model == "openai/gpt-6-luna"


def test_deepseek_flash_is_cheaper_than_luna():
    """Still true after the 2026-08-16 hike, but the margin is now thin.

    Guarding the *direction* matters because the premium primary flipped to Luna
    on expected value: if a future reprice inverts the cost order too, the
    reasoning recorded in ``test_premium_chain_is_ordered_by_value_not_registration``
    stops applying and both tests should be revisited together.
    """
    cands = {c.model: c for c in DEFAULT_REGISTRY.candidates_for_tier(TIER_PREMIUM)}
    cost = lambda c: (1600 * c.input_cost_per_million
                      + 900 * c.output_cost_per_million) / 1e6
    assert cost(cands["deepseek/deepseek-v4.1-flash"]) < cost(cands["openai/gpt-6-luna"])


def test_stale_april_deepseek_is_no_longer_routed():
    """The April release is index 40 at the same price as 0731 — strictly
    dominated, so nothing should still point at it."""
    routed = {
        c.model
        for t in DEFAULT_REGISTRY.all_tiers()
        for c in DEFAULT_REGISTRY.candidates_for_tier(t)
    }
    assert "deepseek/deepseek-v4-flash" not in routed
    assert "deepseek/deepseek-v4.1-flash" in routed


def test_muse_spark_is_a_deep_fallback_not_a_primary():
    """The prior 1.1 release remains a legacy premium fallback."""
    cands = DEFAULT_REGISTRY.candidates_for_tier(TIER_PREMIUM)
    muse = next(c for c in cands if c.model == "meta/muse-spark-1.3")
    assert muse.role == "fallback"
    assert [c.model for c in cands].index("meta/muse-spark-1.3") >= 2


def test_all_premium_candidates_priceable_exactly():
    """An alias match would price a new slug from an older model's entry."""
    from theriad.accounting.pricing import PricingTable

    pt = PricingTable(use_live=False)
    for c in DEFAULT_REGISTRY.candidates_for_tier(TIER_PREMIUM):
        r = pt.lookup(c.model)
        assert r is not None and r.source == "exact", f"{c.model}: {r}"
