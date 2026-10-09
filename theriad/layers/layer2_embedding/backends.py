"""
Pluggable embedding backends for Layer 2.

Design decisions:
- EmbeddingBackend is an ABC; swap models by passing a different backend.
- SentenceTransformerBackend supports BERT, MiniLM, and Matryoshka-compatible
  encoders (any model loadable by sentence-transformers).
- Matryoshka support: truncate embedding to first N dims. No special handling
  needed — sentence-transformers returns full vectors; callers slice as needed.
- Model is loaded lazily (first call) to keep import time fast.
- Alternative considered: use raw transformers library directly. Rejected because
  sentence-transformers provides a clean encode() API and handles pooling.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256

import numpy as np

from theriad.core.logging.logger import get_logger

logger = get_logger("layer2.backends")


class EmbeddingBackend(ABC):
    """Abstract embedding backend. Accepts text, returns unit-norm vector."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier for this backend's model."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a normalized 1-D float32 embedding vector."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Return a 2-D float32 array of shape (N, dim)."""


class SentenceTransformerBackend(EmbeddingBackend):
    """
    Backend using the sentence-transformers library.

    Compatible with:
      - BERT-based models (e.g. 'bert-base-nli-mean-tokens')
      - MiniLM (e.g. 'all-MiniLM-L6-v2')
      - Matryoshka encoders (e.g. 'nomic-ai/nomic-embed-text-v1.5')
        — caller can slice returned vector to desired dimension.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", matryoshka_dim: int | None = None):
        self._model_name = model_name
        self._matryoshka_dim = matryoshka_dim
        self._model = None  # lazy load

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("loading embedding model", extra={"model": self._model_name})
            self._model = SentenceTransformer(self._model_name)

    def embed(self, text: str) -> np.ndarray:
        self._load()
        vec = self._model.encode(text, normalize_embeddings=True)
        return self._maybe_truncate(vec)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self._load()
        vecs = self._model.encode(texts, normalize_embeddings=True)
        if self._matryoshka_dim:
            vecs = vecs[:, : self._matryoshka_dim]
        return vecs

    def _maybe_truncate(self, vec: np.ndarray) -> np.ndarray:
        if self._matryoshka_dim and len(vec) > self._matryoshka_dim:
            vec = vec[: self._matryoshka_dim]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec


class MockEmbeddingBackend(EmbeddingBackend):
    """
    Deterministic mock backend for testing — no model download required.

    Returns a fixed-dimension vector based on text length hash, ensuring
    tests are fast, offline, and reproducible.
    """

    def __init__(self, dim: int = 64):
        self._dim = dim
        self._model_name_str = f"mock-{dim}d"

    @property
    def model_name(self) -> str:
        return self._model_name_str

    def embed(self, text: str) -> np.ndarray:
        # ``hash()`` is intentionally randomized between Python processes, so
        # it makes classifier tests depend on the interpreter's hash seed.
        # Derive the seed from a stable digest instead.
        seed = int.from_bytes(sha256(text.encode("utf-8")).digest()[:8], "little")
        rng = np.random.default_rng(seed % (2**32))
        vec = rng.standard_normal(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])
