"""
DeterministicLayer — orchestrates all Layer 0 rules.

Design:
- Rules are evaluated in registration order; first rule with confidence >=
  min_confidence short-circuits (Principle 2: terminate routing early).
- All fired decisions are recorded in RoutingResult.all_decisions for
  auditability (Principle 4: every decision must be explainable).
- min_confidence is injected at construction time, not read from global state,
  making this layer fully testable without config side effects.
"""
from __future__ import annotations

from typing import Any

from theriad.core.logging.logger import get_logger
from theriad.core.routing.interfaces import (
    RoutingDecision,
    RoutingLayer,
    RoutingResult,
    RoutingRule,
)

logger = get_logger("layer0")


class DeterministicLayer(RoutingLayer):
    """Runs registered rules in order; returns on first high-confidence hit."""

    def __init__(self, rules: list[RoutingRule], min_confidence: float = 0.75):
        self._rules = rules
        self._min_confidence = min_confidence

    @property
    def name(self) -> str:
        return "layer0"

    def route(self, prompt: str, context: dict[str, Any]) -> RoutingResult | None:
        fired: list[RoutingDecision] = []

        for rule in self._rules:
            decision = rule.evaluate(prompt)
            if decision is None:
                continue

            fired.append(decision)
            logger.debug(
                "rule fired",
                extra={
                    "rule": rule.name,
                    "tier": decision.tier,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                },
            )

            if decision.confidence >= self._min_confidence:
                logger.info(
                    "layer0 routed",
                    extra={"rule": rule.name, "tier": decision.tier},
                )
                return RoutingResult(
                    decision=decision,
                    layer_name=self.name,
                    all_decisions=fired,
                    features=context,
                )

        # No rule hit min_confidence — pass to next layer
        if fired:
            logger.debug("layer0 pass-through", extra={"fired_rules": len(fired)})
        return None


def build_default_layer(min_confidence: float = 0.75) -> DeterministicLayer:
    """
    Factory that returns a DeterministicLayer with the default rule set.

    Rule order matters: content-type rules first (most specific signal),
    long-context check last (only fires when content type is ambiguous).
    Rationale: prompt shortness is not evidence that a model will succeed.
    """
    from .rules import (
        BuildProjectRule,
        CodeDetectionRule,
        JsonDetectionRule,
        LanguageDetectionRule,
        MarkdownDetectionRule,
        PromptLengthRule,
    )

    rules: list[RoutingRule] = [
        JsonDetectionRule(),       # cheap regex + json.loads — unambiguous content type
        BuildProjectRule(),        # full artifact requests must outrank generic code signals
        CodeDetectionRule(),       # regex — unambiguous content type
        MarkdownDetectionRule(),   # regex — unambiguous content type
        LanguageDetectionRule(),   # langdetect — non-English needs explicit routing
        PromptLengthRule(),        # length last: only fires when content type is ambiguous
    ]
    return DeterministicLayer(rules=rules, min_confidence=min_confidence)
