"""
EmbeddingRouter — kNN-based task classifier using embeddings.

Design:
- Stores labeled prototype embeddings (one per task type).
- At routing time, embeds the prompt and finds the nearest prototype via
  cosine similarity (= dot product when vectors are unit-norm).
- Confidence = cosine similarity of nearest neighbour (range [0, 1]).
- If confidence < threshold, callers should escalate to next layer.
- Prototypes store TIER LABELS, not model slugs (ADR D-016). The policy
  resolver maps the returned tier to a concrete model at execution time.

Alternatives considered:
- SVM classifier: requires labeled dataset up front. Rejected — no training data
  available at v1. Prototypes can be hand-crafted from representative examples.
- k>1 nearest neighbours with voting: adds complexity for marginal gain. Deferred.

Future: prototype embeddings could be persisted to disk and loaded at startup.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from theriad.core.logging.logger import get_logger
from theriad.core.routing.interfaces import RoutingDecision
from theriad.models.tiers import (
    TIER_CHEAP_BULK,
    TIER_CHEAP_CODE,
    TIER_CODE_STRONG,
    TIER_STRONG_REASONING,
)

from .backends import EmbeddingBackend

logger = get_logger("layer2.router")


@dataclass
class TaskPrototype:
    task_type: str          # e.g. "code", "qa", "summarize", "translate"
    tier: str               # tier label for this task type
    embedding: np.ndarray   # unit-norm prototype vector


# Default prototype prompts and their associated tiers.
# These are representative examples that anchor each task cluster.
# Format: (task_type, tier, example_prompt)
_DEFAULT_PROTOTYPES: list[tuple[str, str, str]] = [
    ("code",      TIER_CHEAP_CODE,
     "Write a Python function to sort a list of dictionaries by a key."),
    ("code",      TIER_CHEAP_CODE,
     "Debug this JavaScript: function add(a,b){return a-b;}"),
    ("qa",        TIER_CHEAP_BULK,
     "What is the capital of France?"),
    ("qa",        TIER_CHEAP_BULK,
     "How many planets are in the solar system?"),
    ("summarize", TIER_CHEAP_BULK,
     "Summarize the following article in three bullet points."),
    ("summarize", TIER_CHEAP_BULK,
     "Give me a TL;DR of this document."),
    ("translate", TIER_CHEAP_BULK,
     "Translate the following text from English to Spanish."),
    ("analysis",  TIER_STRONG_REASONING,
     "Analyze the competitive landscape of cloud providers and provide strategic recommendations."),
    ("analysis",  TIER_STRONG_REASONING,
     "Write a detailed technical architecture document for a distributed system."),
    ("code_hard", TIER_CODE_STRONG,
     "Refactor this 500-line Python module to use async/await and add comprehensive type hints."), #USING ANCHOR QUERIES IS RIDICULOUS AND NOT WHAT I WANTED.
]


class EmbeddingRouter:
    """
    Nearest-prototype embedding router.

    Usage:
        backend = SentenceTransformerBackend()
        router = EmbeddingRouter(backend)
        router.build_prototypes()   # embed default prototypes
        decision = router.route("Write a sorting algorithm")
    """

    def __init__(self, backend: EmbeddingBackend):
        self._backend = backend
        self._prototypes: list[TaskPrototype] = []

    @property
    def is_ready(self) -> bool:
        return len(self._prototypes) > 0

    def build_prototypes(
        self,
        prototypes: list[tuple[str, str, str]] | None = None,
    ) -> None:
        """
        Embed all prototype prompts and store them.

        Args:
            prototypes: list of (task_type, tier, example_prompt).
                        Defaults to _DEFAULT_PROTOTYPES.
        """
        source = prototypes or _DEFAULT_PROTOTYPES
        prompts = [p[2] for p in source]
        embeddings = self._backend.embed_batch(prompts)

        self._prototypes = [
            TaskPrototype(task_type=task, tier=tier, embedding=emb)
            for (task, tier, _), emb in zip(source, embeddings, strict=False)
        ]
        logger.info("prototypes built", extra={"count": len(self._prototypes)})

    def route(self, prompt: str) -> RoutingDecision:
        """
        Return a RoutingDecision for the prompt.

        Confidence = cosine similarity of nearest prototype.
        Callers should compare against RouterConfig.min_confidence.
        """
        if not self._prototypes:
            raise RuntimeError("EmbeddingRouter: call build_prototypes() first")

        query = self._backend.embed(prompt)
        prototype_matrix = np.stack([p.embedding for p in self._prototypes])
        similarities = prototype_matrix @ query  # cosine sim (unit-norm vectors)

        best_idx = int(np.argmax(similarities))
        best_proto = self._prototypes[best_idx]
        confidence = float(np.clip(similarities[best_idx], 0.0, 1.0))

        logger.debug(
            "embedding route",
            extra={
                "task": best_proto.task_type,
                "confidence": confidence,
                "tier": best_proto.tier,
            },
        )

        return RoutingDecision(
            tier=best_proto.tier,
            confidence=confidence,
            reason=(
                f"embedding: task_type={best_proto.task_type} "
                f"confidence={confidence:.3f} backend={self._backend.model_name}"
            ),
            signals={
                "task_type": best_proto.task_type,
                "similarity": round(confidence, 4),
                "backend": self._backend.model_name,
            },
        )
