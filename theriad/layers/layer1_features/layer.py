"""
Layer 1: Feature-based routing layer.

Uses numerical features to make routing decisions without any model inference.
Sits after Layer 0 and produces a RoutingResult when features are conclusive.

Decision logic (in priority order):
1. JSON probability ≥ 0.9  → cheap_bulk tier
2. code_ratio ≥ 0.3        → cheap_code tier
3. code_intent_count ≥ 2    → cheap_code tier (natural-language coding request)
4. code_keyword_count ≥ 5   → cheap_code tier
5. token_count > 3000       → strong_reasoning tier (long context needed)
6. markdown_ratio ≥ 1.5     → cheap_bulk tier (doc editing)

All thresholds are constants, not learned. Simpler, debuggable, auditable.
Tiers are emitted, not model slugs (ADR D-016).
"""
from __future__ import annotations

from typing import Any

from theriad.core.logging.logger import get_logger
from theriad.core.routing.interfaces import RoutingDecision, RoutingLayer, RoutingResult
from theriad.models.tiers import TIER_CHEAP_BULK, TIER_CHEAP_CODE, TIER_STRONG_REASONING

from .extractor import FeatureExtractor

logger = get_logger("layer1_features")


class FeatureRoutingLayer(RoutingLayer):
    """Routes prompts using extracted numerical features."""

    def __init__(self, min_confidence: float = 0.75):
        self._extractor = FeatureExtractor()
        self._min_confidence = min_confidence

    @property
    def name(self) -> str:
        return "layer1_features"

    def route(self, prompt: str, context: dict[str, Any]) -> RoutingResult | None:
        features = self._extractor.extract(prompt)
        context["features"] = features.to_dict()

        decision = self._decide(features)
        if decision is None or decision.confidence < self._min_confidence:
            return None

        logger.info(
            "layer1 routed",
            extra={"tier": decision.tier, "reason": decision.reason},
        )
        return RoutingResult(
            decision=decision,
            layer_name=self.name,
            features=context,
        )

    def _decide(self, f) -> RoutingDecision | None:  # noqa: F821

        if f.json_probability >= 0.9:
            return RoutingDecision(
                tier=TIER_CHEAP_BULK,
                confidence=0.88,
                reason=f"feature: json_probability={f.json_probability:.2f}",
                signals={"json_probability": round(f.json_probability, 3)},
            )
        if f.code_ratio >= 0.3:
            return RoutingDecision(
                tier=TIER_CHEAP_CODE,
                confidence=0.85,
                reason=f"feature: code_ratio={f.code_ratio:.2f}",
                signals={"code_ratio": round(f.code_ratio, 3)},
            )
        if f.code_intent_count >= 2:
            return RoutingDecision(
                tier=TIER_CHEAP_CODE,
                confidence=0.80,
                reason=f"feature: code_intent_terms={f.code_intent_count}",
                signals={"code_intent_count": f.code_intent_count},
            )
        if f.code_keyword_count >= 5:
            return RoutingDecision(
                tier=TIER_CHEAP_CODE,
                confidence=0.80,
                reason=f"feature: code_keywords={f.code_keyword_count}",
                signals={"code_keyword_count": f.code_keyword_count},
            )
        if f.token_count > 3000:
            return RoutingDecision(
                tier=TIER_STRONG_REASONING,
                confidence=0.78,
                reason=f"feature: token_count={f.token_count}",
                signals={"token_count": f.token_count},
            )
        if f.markdown_ratio >= 1.5:
            return RoutingDecision(
                tier=TIER_CHEAP_BULK,
                confidence=0.78,
                reason=f"feature: markdown_ratio={f.markdown_ratio:.2f}",
                signals={"markdown_ratio": round(f.markdown_ratio, 3)},
            )
        return None
