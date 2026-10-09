"""
Routers under test, plus baselines to bracket them.

``theriad.eval.regret`` scores any callable of ``EvalItem -> Action``. This module
supplies the ones worth scoring:

- :func:`theriad_router` — the real production routing path (layer0 → layer1 →
  PolicyResolver). This is the thing the harness exists to measure.
- :func:`fixed_router` — always one model. Two of these bracket the result:
  always-cheapest is the floor on spend and the ceiling on accuracy loss;
  always-strongest is the reverse.
- :func:`oracle_router` — plays the oracle's own choice, so regret is 0 by
  construction. Included as a self-test: a non-zero result means the harness
  disagrees with itself.

A router's regret is only meaningful when every model it can select is present in
the grid. A model the grid never measured falls back to the uniform Beta prior
(p=0.5) and, if unpriced, to a cost of 0 — which would flatter it enormously.
:func:`unmeasured_models` reports that before it silently becomes a result.
"""
from __future__ import annotations

from collections.abc import Callable

from .corpus import EvalItem
from .grid import OutcomeGrid
from .oracle import Action, ExpectedValueOracle

# Tier used when every routing layer abstains, matching TheriadPipeline's default.
FALLBACK_TIER = "cheap_bulk"


def theriad_router(
    policy: str = "balanced",
    *,
    registry=None,
    use_capability_axis: bool = True,
    fallback_tier: str = FALLBACK_TIER,
) -> Callable[[EvalItem], Action]:
    """The production router: routing layers → tier → PolicyResolver → model.

    Uses layer0 (rules) and layer1 (features) only, mirroring
    ``theriad.api.server._build_layers``. Those need no ML dependencies and no
    network, so scoring stays offline. Layer 2's embedding router would add a
    model-loading dependency to what is otherwise a pure-analysis path; when it
    is in play in production its effect shows up as a different task_type, which
    this router still honours via ``use_capability_axis``.

    Args:
        policy: Policy name passed to PolicyResolver.
        registry: Optional ModelRegistry override; defaults to the shipped one.
        use_capability_axis: Pass the layer's task_type through to the resolver
            so capability profiles apply (D-051). Set False to measure the
            router as it behaved before capability routing.
    """
    from theriad.layers.layer0.layer import build_default_layer
    from theriad.layers.layer1_features.layer import FeatureRoutingLayer
    from theriad.models.policy import PolicyResolver

    layer0 = build_default_layer()
    layer1 = FeatureRoutingLayer()
    resolver = PolicyResolver(registry)

    def _route(item: EvalItem) -> Action:
        ctx: dict = {}
        result = layer0.route(item.prompt, ctx) or layer1.route(item.prompt, ctx)
        if result is None:
            tier, confidence, task_type = fallback_tier, 0.0, None
        else:
            tier = result.decision.tier
            confidence = result.decision.confidence
            task_type = result.decision.signals.get("task_type")
        resolution = resolver.resolve(
            tier,
            policy,
            confidence,
            message=item.prompt,
            task_type=task_type if use_capability_axis else None,
        )
        return Action(model=resolution.model, n_samples=1)

    return _route


def fixed_router(model: str, n_samples: int = 1) -> Callable[[EvalItem], Action]:
    """Always play the same action. Use two of these to bracket the real router."""

    def _route(item: EvalItem) -> Action:
        return Action(model=model, n_samples=n_samples)

    return _route


def oracle_router(oracle: ExpectedValueOracle, models: list[str] | None = None):
    """Play the oracle's own choice. Regret must be 0 — a harness self-test."""

    def _route(item: EvalItem) -> Action:
        return oracle.evaluate(item, models).best.action

    return _route


def routed_models(
    router: Callable[[EvalItem], Action], items: list[EvalItem]
) -> set[str]:
    """Every model ``router`` selects across ``items``."""
    return {router(item).model for item in items}


def unmeasured_models(
    router: Callable[[EvalItem], Action],
    items: list[EvalItem],
    grid: OutcomeGrid,
) -> set[str]:
    """Models the router selects that the grid never measured.

    These are scored on the uniform Beta prior (p=0.5) rather than evidence, so
    any regret number that includes them is partly fiction. Extend grid coverage
    to cover them, or restrict the registry, before reading the result.
    """
    measured = set(grid.models())
    return {m for m in routed_models(router, items) if m not in measured}
