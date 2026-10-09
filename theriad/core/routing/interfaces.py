"""
Core routing abstractions.

Design rationale:
- RoutingDecision emits an abstract TIER label (e.g. "cheap_code"), never a
  concrete model slug. Model selection happens later via PolicyResolver (ADR D-016).
- Confidence is [0, 1]; callers compare against RouterConfig.min_confidence.
- signals carries structured evidence from the routing rule (keyword counts,
  feature values, etc.) so that 'theriad explain' can show why a tier was chosen.
- RoutingLayer is the extension point: add a new layer without touching core.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RoutingDecision:
    """Single atomic routing verdict — always names a tier, never a model slug."""
    tier: str              # abstract tier label, e.g. "cheap_bulk", "cheap_code"
    confidence: float      # [0.0, 1.0]
    reason: str            # human-readable explanation
    signals: dict = field(default_factory=dict)  # structured evidence


@dataclass
class RoutingResult:
    """Aggregated result produced by the routing pipeline."""
    decision: RoutingDecision
    layer_name: str                          # which layer produced the final decision
    all_decisions: list[RoutingDecision] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)


class RoutingRule(ABC):
    """
    A single deterministic routing rule (used by Layer 0).

    Each rule is stateless and side-effect-free.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique rule identifier."""

    @abstractmethod
    def evaluate(self, prompt: str) -> RoutingDecision | None:
        """
        Evaluate the prompt.

        Returns a RoutingDecision if the rule fires, None if it abstains.
        """


class RoutingLayer(ABC):
    """
    An ordered layer in the routing pipeline.

    Layers are tried in order; a layer that produces a decision with
    confidence >= threshold short-circuits the remaining layers.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Layer identifier matching RouterConfig.layers entries."""

    @abstractmethod
    def route(self, prompt: str, context: dict[str, Any]) -> RoutingResult | None:
        """
        Attempt to route the prompt.

        Returns None to signal "pass to next layer".
        Context may carry features extracted by prior layers.
        """


class RouterProtocol(ABC):
    """Top-level router that orchestrates layers."""

    @abstractmethod
    def route(self, prompt: str) -> RoutingResult:
        """Return the best routing decision for the prompt."""
