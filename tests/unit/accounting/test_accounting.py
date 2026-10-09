"""Tests for cost accounting and pricing."""
from __future__ import annotations

import json

import pytest

from theriad.accounting.pricing import (
    PLATFORM_FEE_PER_1M_TOKENS,
    ModelPrice,
    PricingTable,
    estimate_cost,
    platform_fee,
)
from theriad.accounting.tracker import CostTracker, InferenceRecord
from theriad.models.registry import DEFAULT_BASELINE_MODEL

# ── PricingTable ──────────────────────────────────────────────────────────────

def test_pricing_exact_match():
    table = PricingTable()
    price = table.get("openai/gpt-4o-mini")
    assert price is not None
    assert price.input_per_1m == 0.15


def test_pricing_fallback_suffix():
    table = PricingTable()
    price = table.get("gpt-4o-mini")  # without org prefix
    assert price is not None


def test_pricing_unknown_model():
    table = PricingTable()
    price = table.get("unknown/model-xyz")
    assert price is None


def test_pricing_update():
    table = PricingTable()
    table.update("custom/model", ModelPrice(1.0, 2.0))
    price = table.get("custom/model")
    assert price is not None
    assert price.input_per_1m == 1.0


# ── estimate_cost ─────────────────────────────────────────────────────────────

def test_estimate_cost_known_model():
    # gpt-4o-mini: $0.15/1M input, $0.60/1M output
    cost = estimate_cost("openai/gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
    expected = (1000 * 0.15 + 500 * 0.60) / 1_000_000
    assert abs(cost - expected) < 1e-9


def test_estimate_cost_zero_for_unknown():
    cost = estimate_cost("unknown/model", 1000, 500)
    assert cost == 0.0


def test_estimate_cost_local_model_is_free():
    cost = estimate_cost("llama3", 5000, 2000)
    assert cost == 0.0


# ── platform fee ──────────────────────────────────────────────────────────────

def test_platform_fee_is_one_tenth_cent_per_million_tokens():
    # 0.001 USD per 1M tokens = one tenth of a cent per million tokens.
    assert PLATFORM_FEE_PER_1M_TOKENS == 0.001
    fee = platform_fee(600_000, 400_000)  # 1M tokens total
    assert abs(fee - 0.001) < 1e-12


def test_platform_fee_ignores_negative_counts():
    assert platform_fee(-100, -50) == 0.0


def test_estimate_cost_excludes_fee_by_default():
    # Default behaviour is unchanged: pure model cost, no fee.
    assert estimate_cost("llama3", 5000, 2000) == 0.0


def test_estimate_cost_can_include_platform_fee():
    tokens_in, tokens_out = 1000, 500
    model = estimate_cost("openai/gpt-4o-mini", tokens_in, tokens_out)
    total = estimate_cost(
        "openai/gpt-4o-mini", tokens_in, tokens_out, include_platform_fee=True
    )
    assert abs(total - (model + platform_fee(tokens_in, tokens_out))) < 1e-12


def test_platform_fee_applies_to_free_models():
    # Free/unknown models still incur the platform fee when requested.
    total = estimate_cost("llama3", 1000, 500, include_platform_fee=True)
    assert total == platform_fee(1000, 500)
    assert total > 0


# ── CostTracker ───────────────────────────────────────────────────────────────

@pytest.fixture
def tracker(tmp_path):
    return CostTracker(metrics_path=str(tmp_path / "metrics.jsonl"))


def test_tracker_record_returns_record(tracker):
    rec = tracker.record(
        provider="openrouter",
        model="openai/gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=250.0,
        routing_layer="layer0",
        routing_reason="json_detection",
    )
    assert isinstance(rec, InferenceRecord)
    assert rec.provider == "openrouter"
    assert rec.total_tokens == 150
    assert rec.estimated_cost_usd > 0


def test_tracker_persists_to_disk(tracker, tmp_path):
    tracker.record(
        provider="openrouter", model="openai/gpt-4o-mini",
        prompt_tokens=10, completion_tokens=5, latency_ms=100.0,
    )
    path = tmp_path / "metrics.jsonl"
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["model"] == "openai/gpt-4o-mini"
    assert "timestamp" in data


def test_tracker_multiple_records(tracker, tmp_path):
    for i in range(5):
        tracker.record(
            provider="openrouter", model="openai/gpt-4o-mini",
            prompt_tokens=10 * i, completion_tokens=5, latency_ms=float(100 + i),
        )
    assert len(tracker.records()) == 5
    path = tmp_path / "metrics.jsonl"
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 5


def test_tracker_summary_empty(tmp_path):
    tracker = CostTracker(metrics_path=str(tmp_path / "m.jsonl"))
    s = tracker.summary()
    assert s["total_requests"] == 0
    assert s["total_cost_usd"] == 0.0


def test_tracker_summary_aggregates(tracker):
    tracker.record(provider="or", model="poolside/laguna-xs-2.1:free", prompt_tokens=100,
                   completion_tokens=50, latency_ms=200.0)
    tracker.record(provider="or", model="poolside/laguna-xs-2.1:free", prompt_tokens=200,
                   completion_tokens=100, latency_ms=400.0)
    s = tracker.summary()
    assert s["total_requests"] == 2
    assert s["total_tokens"] == 450
    assert s["avg_latency_ms"] == 300.0
    assert s["models_used"]["poolside/laguna-xs-2.1:free"] == 2
    assert s["baseline_model"] == DEFAULT_BASELINE_MODEL
    assert s["baseline_cost_usd"] > s["total_cost_usd"]
    assert s["estimated_savings_usd"] > 0
    assert s["estimated_savings_pct"] > 0


def test_tracker_fallback_rate(tracker):
    tracker.record(provider="p", model="m", prompt_tokens=1, completion_tokens=1,
                   latency_ms=1.0, used_fallback=True)
    tracker.record(provider="p", model="m", prompt_tokens=1, completion_tokens=1,
                   latency_ms=1.0, used_fallback=False)
    assert tracker.summary()["fallback_rate"] == 0.5


def test_load_from_disk(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rec = {"model": "m", "provider": "p", "total_tokens": 10}
    path.write_text(json.dumps(rec) + "\n")
    loaded = CostTracker.load_from_disk(str(path))
    assert len(loaded) == 1
    assert loaded[0]["model"] == "m"


def test_load_from_disk_nonexistent(tmp_path):
    result = CostTracker.load_from_disk(str(tmp_path / "nope.jsonl"))
    assert result == []
