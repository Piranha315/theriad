"""Layer 2 — local embedding-based routing."""
from .backends import EmbeddingBackend, SentenceTransformerBackend
from .layer import EmbeddingRoutingLayer
from .router import EmbeddingRouter

__all__ = [
    "EmbeddingBackend",
    "SentenceTransformerBackend",
    "EmbeddingRouter",
    "EmbeddingRoutingLayer",
]
