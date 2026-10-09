"""
Tests for Layer 2 embedding router.

All tests use MockEmbeddingBackend — no model download required.
The mock produces deterministic embeddings via hash-seeded RNG.
"""
from __future__ import annotations

import numpy as np
import pytest

from theriad.layers.layer2_embedding.backends import MockEmbeddingBackend
from theriad.layers.layer2_embedding.layer import EmbeddingRoutingLayer
from theriad.layers.layer2_embedding.router import EmbeddingRouter
from theriad.models.tiers import TIER_CHEAP_BULK, TIER_CHEAP_CODE

# ── MockEmbeddingBackend ──────────────────────────────────────────────────────

def test_mock_backend_returns_unit_vector():
    backend = MockEmbeddingBackend(dim=64)
    vec = backend.embed("hello world")
    assert vec.shape == (64,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_mock_backend_deterministic():
    backend = MockEmbeddingBackend(dim=64)
    v1 = backend.embed("test prompt")
    v2 = backend.embed("test prompt")
    np.testing.assert_array_equal(v1, v2)


def test_mock_backend_different_prompts_differ():
    backend = MockEmbeddingBackend(dim=64)
    v1 = backend.embed("hello")
    v2 = backend.embed("world")
    assert not np.allclose(v1, v2)


def test_mock_backend_batch_shape():
    backend = MockEmbeddingBackend(dim=32)
    batch = backend.embed_batch(["a", "b", "c"])
    assert batch.shape == (3, 32)


# ── EmbeddingRouter ───────────────────────────────────────────────────────────

def test_router_not_ready_before_build():
    router = EmbeddingRouter(MockEmbeddingBackend())
    assert not router.is_ready


def test_router_ready_after_build():
    router = EmbeddingRouter(MockEmbeddingBackend())
    router.build_prototypes()
    assert router.is_ready


def test_router_raises_if_not_built():
    router = EmbeddingRouter(MockEmbeddingBackend())
    with pytest.raises(RuntimeError, match="build_prototypes"):
        router.route("hello")


def test_router_returns_decision():
    router = EmbeddingRouter(MockEmbeddingBackend(dim=64))
    router.build_prototypes()
    decision = router.route("Write me a Python function")
    assert decision.tier  # non-empty tier string
    assert 0.0 <= decision.confidence <= 1.0
    assert "embedding:" in decision.reason


def test_router_confidence_in_range():
    backend = MockEmbeddingBackend(dim=64)
    router = EmbeddingRouter(backend)
    router.build_prototypes()
    for prompt in ["hello", "def foo(): pass", "translate to French", "analyze market trends"]:
        d = router.route(prompt)
        assert 0.0 <= d.confidence <= 1.0


def test_router_decision_has_signals():
    router = EmbeddingRouter(MockEmbeddingBackend(dim=64))
    router.build_prototypes()
    decision = router.route("Write a Python function")
    assert "task_type" in decision.signals
    assert "similarity" in decision.signals


def test_custom_prototypes():
    backend = MockEmbeddingBackend(dim=32)
    router = EmbeddingRouter(backend)
    custom = [
        ("greeting", TIER_CHEAP_BULK, "Hello, how are you?"),
        ("farewell", TIER_CHEAP_BULK, "Goodbye, see you later!"),
    ]
    router.build_prototypes(custom)
    assert len(router._prototypes) == 2
    d = router.route("Hi there!")
    assert d.tier == TIER_CHEAP_BULK


def test_custom_prototypes_mixed_tiers():
    backend = MockEmbeddingBackend(dim=32)
    router = EmbeddingRouter(backend)
    custom = [
        ("greeting", TIER_CHEAP_BULK, "Hello, how are you?"),
        ("code_task", TIER_CHEAP_CODE, "Write a sorting algorithm"),
    ]
    router.build_prototypes(custom)
    assert len(router._prototypes) == 2
    # Both tiers should be accessible
    tiers = {p.tier for p in router._prototypes}
    assert TIER_CHEAP_BULK in tiers
    assert TIER_CHEAP_CODE in tiers


# ── EmbeddingRoutingLayer ─────────────────────────────────────────────────────

def test_layer_routes_high_confidence():
    """With a mock backend that produces same-hash vectors, verify routing works."""
    backend = MockEmbeddingBackend(dim=64)
    layer = EmbeddingRoutingLayer(backend=backend, min_confidence=0.0)  # accept any confidence
    result = layer.route("Write a Python function", {})
    assert result is not None
    assert result.layer_name == "layer2_embedding"


def test_layer_injects_confidence_into_context():
    backend = MockEmbeddingBackend(dim=64)
    layer = EmbeddingRoutingLayer(backend=backend, min_confidence=0.0)
    ctx: dict = {}
    layer.route("test prompt", ctx)
    assert "embedding_confidence" in ctx
    assert "embedding_model" in ctx


def test_layer_abstains_below_min_confidence():
    backend = MockEmbeddingBackend(dim=64)
    layer = EmbeddingRoutingLayer(backend=backend, min_confidence=1.01)  # impossible threshold
    ctx: dict = {}
    result = layer.route("any prompt", ctx)
    assert result is None
    # Context must still be populated
    assert "embedding_confidence" in ctx


def test_layer_lazy_init():
    """Router should not be initialized until first route() call."""
    backend = MockEmbeddingBackend(dim=64)
    layer = EmbeddingRoutingLayer(backend=backend)
    assert layer._router is None
    layer.route("hello", {})
    assert layer._router is not None
