"""Exercise real API fallback loops with fake provider dispatches and billed costs."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from theriad.api import server
from theriad.core.providers.interfaces import CompletionResponse
from theriad.models.policy import PolicyResolution
from theriad.models.tiers import ModelCandidate
from theriad.storage import database as db


@pytest.fixture
def api(monkeypatch, tmp_path):
    from theriad.accounting.pricing import PricingTable
    from theriad.layers.layer2_embedding.backends import MockEmbeddingBackend
    from theriad.workflow.detector import WorkflowDetector

    db.close_db()
    db.configure(tmp_path / "accounting.db")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(server, "_create_workflow_detector", lambda: WorkflowDetector(embedding_backend=MockEmbeddingBackend(dim=64)))
    monkeypatch.setattr("theriad.agents.chat_agent.classify_chat_intent", lambda _message: SimpleNamespace(intent="normal_chat"))
    monkeypatch.setattr("theriad.accounting.pricing.PricingTable", lambda *args, **kwargs: PricingTable(use_live=False))
    candidates = [
        ModelCandidate(
            tier="cheap_bulk", provider="openrouter", model=name, role="primary",
            input_cost_per_million=0.1, output_cost_per_million=0.1,
            context_window=128000, quality_prior=0.8,
        )
        for name in ("first", "second")
    ]
    monkeypatch.setattr("theriad.models.policy.PolicyResolver.resolve", lambda *_args, **_kwargs: PolicyResolution(
        tier="cheap_bulk", selected=candidates[0], fallback_chain=candidates[1:], policy="balanced",
    ))
    usage_ledger = []
    monkeypatch.setattr(server, "_record_routing_usage", lambda _request, **kwargs: usage_ledger.append(kwargs))
    provider = SimpleNamespace(name="openrouter", complete=AsyncMock())
    monkeypatch.setattr("theriad.core.providers.factory.build_provider", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr("theriad.accounting.tracker.CostTracker.record", lambda *_args, **_kwargs: None)
    with TestClient(server.app) as client:
        yield client, provider, usage_ledger
    db.close_db()


def completion(model, content, cost):
    return CompletionResponse(
        model=model, content=content, prompt_tokens=10, completion_tokens=20,
        latency_ms=5, provider="openrouter", finish_reason="stop", cost_usd=cost,
    )


def send(client, surface):
    if surface == "chat":
        return client.post("/api/chat/send", json={"message": "Hello", "mode": "execute", "allow_paid_calls": True})
    return client.post("/v1/chat/completions", headers={"Authorization": "Bearer test-key"}, json={
        "model": "theriad", "messages": [{"role": "user", "content": "Hello"}],
    })


def accounting(response, surface):
    data = response.json()
    return data["run"] if surface == "chat" else data["x_theriad"]


def total_cost(metadata, surface):
    return metadata["actual_cost_usd" if surface == "chat" else "estimated_cost_usd"]


@pytest.mark.parametrize("surface", ["chat", "harness"])
def test_charged_empty_response_and_final_response_both_count(api, surface):
    client, provider, usage_ledger = api
    provider.complete.side_effect = [completion("first", "", 0.02), completion("second", "Hello!", 0.03)]
    response = send(client, surface)
    assert response.status_code == 200, response.text
    meta = accounting(response, surface)
    assert total_cost(meta, surface) == pytest.approx(0.05)
    assert meta["cost_known"] is True
    assert meta["known_attempt_cost_usd"] == pytest.approx(0.05)
    assert meta["attempt_input_tokens"] == 20
    assert meta["attempt_output_tokens"] == 40
    assert len(meta["attempt_usage"]) == provider.complete.await_count == 2
    assert usage_ledger[-1]["provider_cost_usd"] == pytest.approx(0.05)
    assert usage_ledger[-1]["input_tokens"] == 20
    persisted = db.get_db().execute("SELECT actual_cost_usd FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
    assert persisted[0] == pytest.approx(0.05)
    # The established final-response usage fields retain their original meaning.
    final_input = meta["input_tokens"] if surface == "chat" else response.json()["usage"]["prompt_tokens"]
    assert final_input == 10


@pytest.mark.parametrize("surface", ["chat", "harness"])
def test_failed_dispatch_makes_total_cost_unknown_after_successful_fallback(api, surface):
    client, provider, usage_ledger = api
    provider.complete.side_effect = [TimeoutError("provider timeout"), completion("second", "Hello!", 0.03)]
    response = send(client, surface)
    assert response.status_code == 200, response.text
    meta = accounting(response, surface)
    assert total_cost(meta, surface) is None
    assert meta["cost_known"] is False
    assert meta["known_attempt_cost_usd"] == pytest.approx(0.03)
    assert meta["attempt_usage"][0]["error_type"] == "TimeoutError"
    assert meta["attempt_usage"][0]["cost_usd"] is None
    assert usage_ledger[-1]["provider_cost_usd"] is None
    assert usage_ledger[-1]["usage_known"] is False
    persisted = db.get_db().execute("SELECT actual_cost_usd FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
    assert persisted[0] is None


@pytest.mark.parametrize("surface", ["chat", "harness"])
def test_exhausted_chain_keeps_failure_and_paid_attempt_accounting(api, surface):
    client, provider, _ = api
    provider.complete.side_effect = [completion("first", "", 0.02), TimeoutError("provider timeout")]
    response = send(client, surface)
    assert response.status_code == 502
    meta = response.json()["accounting" if surface == "chat" else "x_theriad"]
    assert total_cost(meta, surface) is None
    assert meta["cost_known"] is False
    assert meta["known_attempt_cost_usd"] == pytest.approx(0.02)
    assert len(meta["attempt_usage"]) == 2


@pytest.mark.parametrize("surface", ["chat", "harness"])
def test_successful_response_with_unknown_price_is_not_reported_free(api, surface):
    client, provider, usage_ledger = api
    provider.complete.side_effect = [completion("unpriced-model", "Hello!", None)]
    response = send(client, surface)
    assert response.status_code == 200, response.text
    meta = accounting(response, surface)
    assert total_cost(meta, surface) is None
    assert meta["cost_known"] is False
    assert meta["known_attempt_cost_usd"] == 0.0
    assert usage_ledger[-1]["provider_cost_usd"] is None


def test_chat_objectively_rejected_paid_answer_is_included(api, monkeypatch):
    client, provider, usage_ledger = api
    from theriad.quality.objective_verifier import ObjectiveVerification

    def grade(_prompt, output):
        passed = output == "42"
        return ObjectiveVerification(applicable=True, passed=passed, source="test", score=float(passed), details="checked")

    monkeypatch.setattr("theriad.quality.objective_verifier.verify_objective_response", grade)
    provider.complete.side_effect = [completion("first", "41", 0.02), completion("second", "42", 0.03)]
    response = send(client, "chat")
    assert response.status_code == 200, response.text
    assert response.json()["assistant_message"]["content"] == "42"
    assert response.json()["run"]["actual_cost_usd"] == pytest.approx(0.05)
    assert usage_ledger[-1]["provider_cost_usd"] == pytest.approx(0.05)


def test_legacy_cost_log_failure_does_not_drop_harness_run_accounting(api, monkeypatch):
    client, provider, usage_ledger = api

    def unavailable_log(*_args, **_kwargs):
        raise OSError("log unavailable")

    monkeypatch.setattr("theriad.accounting.tracker.CostTracker.record", unavailable_log)
    provider.complete.side_effect = [completion("first", "Hello!", 0.03)]
    response = send(client, "harness")
    assert response.status_code == 200
    assert response.json()["x_theriad"]["actual_cost_usd"] == pytest.approx(0.03)
    assert usage_ledger[-1]["provider_cost_usd"] == pytest.approx(0.03)
    persisted = db.get_db().execute("SELECT actual_cost_usd FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
    assert persisted[0] == pytest.approx(0.03)
