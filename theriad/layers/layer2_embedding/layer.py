"""
EmbeddingRoutingLayer — wraps EmbeddingRouter as a RoutingLayer.

Lazy initialization: prototypes are built on the first call to route()
so that the model is not downloaded at import time.
"""
from __future__ import annotations

from typing import Any

from theriad.core.logging.logger import get_logger
from theriad.core.routing.interfaces import RoutingLayer, RoutingResult

from .backends import EmbeddingBackend, SentenceTransformerBackend
from .router import EmbeddingRouter

logger = get_logger("layer2")


class EmbeddingRoutingLayer(RoutingLayer):
    """Routing layer backed by local embedding similarity."""

    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        min_confidence: float = 0.75,
        prototypes: list[tuple[str, str, str]] | None = None,
    ):
        self._backend = backend or SentenceTransformerBackend()
        self._min_confidence = min_confidence
        self._custom_prototypes = prototypes
        self._router: EmbeddingRouter | None = None

    @property
    def name(self) -> str:
        return "layer2_embedding"

    def _ensure_ready(self) -> EmbeddingRouter:
        if self._router is None:
            self._router = EmbeddingRouter(self._backend)
            self._router.build_prototypes(self._custom_prototypes)
        return self._router

    def route(self, prompt: str, context: dict[str, Any]) -> RoutingResult | None:
        router = self._ensure_ready()
        decision = router.route(prompt)

        context["embedding_confidence"] = decision.confidence
        context["embedding_model"] = self._backend.model_name

        if decision.confidence < self._min_confidence:
            logger.debug(
                "layer2 pass-through",
                extra={"confidence": decision.confidence},
            )
            return None

        logger.info("layer2 routed", extra={"tier": decision.tier, "confidence": decision.confidence})
        return RoutingResult(
            decision=decision,
            layer_name=self.name,
            features=context,
        )
