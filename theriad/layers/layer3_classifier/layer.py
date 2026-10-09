"""
ClassifierRoutingLayer — wraps LocalClassifier as a RoutingLayer.

Invoked ONLY if embedding confidence (from Layer 2) is below the threshold
configured in RouterConfig. This is enforced by the pipeline: layers run in
order and this layer is last.

Removable: exclude 'layer3_classifier' from RouterConfig.layers and do not
instantiate this class. Zero coupling to core.
"""
from __future__ import annotations

from typing import Any

from theriad.core.logging.logger import get_logger
from theriad.core.routing.interfaces import RoutingDecision, RoutingLayer, RoutingResult

from .classifier import LocalClassifier

logger = get_logger("layer3")


class ClassifierRoutingLayer(RoutingLayer):
    """Optional last-resort routing layer using local text classification."""

    def __init__(
        self,
        classifier: LocalClassifier | None = None,
        min_confidence: float = 0.60,
        auto_fit: bool = True,
    ):
        self._clf = classifier or LocalClassifier()
        self._min_confidence = min_confidence
        self._fitted = False
        self._auto_fit = auto_fit

    @property
    def name(self) -> str:
        return "layer3_classifier"

    def _ensure_fitted(self) -> None:
        if not self._fitted and self._auto_fit:
            self._clf.fit()
            self._fitted = True

    def route(self, prompt: str, context: dict[str, Any]) -> RoutingResult | None:
        try:
            self._ensure_fitted()
            result = self._clf.predict(prompt)
        except ImportError:
            logger.warning("layer3 unavailable: scikit-learn not installed")
            return None
        except Exception as exc:
            logger.error("layer3 error", extra={"error": str(exc)})
            return None

        context["classifier_task"] = result.task_type
        context["classifier_confidence"] = result.confidence

        if result.confidence < self._min_confidence:
            logger.debug("layer3 low confidence", extra={"confidence": result.confidence})
            return None

        logger.info("layer3 routed", extra={"task": result.task_type, "tier": result.tier})
        return RoutingResult(
            decision=RoutingDecision(
                tier=result.tier,
                confidence=result.confidence,
                reason=result.reason,
                signals={"task_type": result.task_type},
            ),
            layer_name=self.name,
            features=context,
        )
