"""
Tests for Phase 6 production fixes:
- TIER_MAX_TOKENS and TIER_TIMEOUT_SECONDS correctness
- /api/execute response includes per-attempt diagnostics and separated latency
- /api/execute returns null cost when pricing is unknown (not $0.00000)
- Workflow warning present in execution note field
- /api/route includes BuildProjectRule in routing path

All provider calls are mocked — no real API key required.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

import theriad.api.server as server_module
from theriad.layers.layer2_embedding.backends import MockEmbeddingBackend
from theriad.models.registry import DEFAULT_BASELINE_MODEL
from theriad.workflow.detector import WorkflowDetector


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _mock_wf_detector():
    return WorkflowDetector(embedding_backend=MockEmbeddingBackend(dim=64))


@pytest.fixture()
def client(monkeypatch):
    """TestClient with workflow detector mocked (no model download)."""
    monkeypatch.setattr(server_module, "_create_workflow_detector", _mock_wf_detector)
    return TestClient(server_module.app)


def _make_mock_response(model="qwen/qwen3-14b", content="Hello", prompt_tokens=10,
                        completion_tokens=20, latency_ms=500.0):
    """Return a fake CompletionResponse."""
    from theriad.core.providers.interfaces import CompletionResponse
    return CompletionResponse(
        model=model,
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        provider="openrouter",
    )


# ── TIER_MAX_TOKENS and TIER_TIMEOUT_SECONDS ──────────────────────────────────

class TestTierLimits:

    def test_all_tiers_have_max_tokens(self):
        from theriad.models.tiers import TIER_MAX_TOKENS, VALID_TIERS
        for tier in VALID_TIERS:
            assert tier in TIER_MAX_TOKENS, f"TIER_MAX_TOKENS missing tier: {tier}"
            assert TIER_MAX_TOKENS[tier] > 0

    def test_all_tiers_have_timeout(self):
        from theriad.models.tiers import TIER_TIMEOUT_SECONDS, VALID_TIERS
        for tier in VALID_TIERS:
            assert tier in TIER_TIMEOUT_SECONDS, f"TIER_TIMEOUT_SECONDS missing tier: {tier}"
            assert TIER_TIMEOUT_SECONDS[tier] > 0

    def test_cheap_bulk_max_tokens(self):
        from theriad.models.tiers import TIER_CHEAP_BULK, TIER_MAX_TOKENS
        assert TIER_MAX_TOKENS[TIER_CHEAP_BULK] == 1200

    def test_cheap_code_max_tokens(self):
        from theriad.models.tiers import TIER_CHEAP_CODE, TIER_MAX_TOKENS
        assert TIER_MAX_TOKENS[TIER_CHEAP_CODE] == 3500

    def test_code_strong_max_tokens(self):
        from theriad.models.tiers import TIER_CODE_STRONG, TIER_MAX_TOKENS
        assert TIER_MAX_TOKENS[TIER_CODE_STRONG] == 6000

    def test_premium_max_tokens(self):
        from theriad.models.tiers import TIER_MAX_TOKENS, TIER_PREMIUM
        assert TIER_MAX_TOKENS[TIER_PREMIUM] == 8000

    def test_cheap_bulk_timeout(self):
        from theriad.models.tiers import TIER_CHEAP_BULK, TIER_TIMEOUT_SECONDS
        assert TIER_TIMEOUT_SECONDS[TIER_CHEAP_BULK] == 20.0

    def test_code_strong_timeout(self):
        from theriad.models.tiers import TIER_CODE_STRONG, TIER_TIMEOUT_SECONDS
        assert TIER_TIMEOUT_SECONDS[TIER_CODE_STRONG] == 90.0

    def test_frontier_has_the_largest_timeout(self):
        """frontier supersedes premium as the slowest tier: Muse Spark 1.2's
        mandatory reasoning can run well past premium's 120s ceiling."""
        from theriad.models.tiers import (
            TIER_FRONTIER, TIER_PREMIUM, TIER_TIMEOUT_SECONDS,
        )
        max_timeout = max(TIER_TIMEOUT_SECONDS.values())
        assert TIER_TIMEOUT_SECONDS[TIER_FRONTIER] == max_timeout
        assert TIER_TIMEOUT_SECONDS[TIER_FRONTIER] > TIER_TIMEOUT_SECONDS[TIER_PREMIUM]

    def test_max_tokens_ordering(self):
        """cheap_bulk < cheap_code < code_strong <= premium."""
        from theriad.models.tiers import (
            TIER_CHEAP_BULK, TIER_CHEAP_CODE, TIER_CODE_STRONG,
            TIER_MAX_TOKENS, TIER_PREMIUM,
        )
        assert TIER_MAX_TOKENS[TIER_CHEAP_BULK] < TIER_MAX_TOKENS[TIER_CHEAP_CODE]
        assert TIER_MAX_TOKENS[TIER_CHEAP_CODE] < TIER_MAX_TOKENS[TIER_CODE_STRONG]
        assert TIER_MAX_TOKENS[TIER_CODE_STRONG] <= TIER_MAX_TOKENS[TIER_PREMIUM]


def test_route_api_threads_prompt_into_free_moe_policy(client):
    response = client.post(
        "/api/route",
        json={
            "prompt": "Debug this TypeError in my Python function",
            "policy": "free_moe_experimental",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["tier"] == "free_moe_code"
    assert data["model"].endswith(":free")
    assert data["policy"] == "free_moe_experimental"


def test_stats_reports_savings_against_current_baseline(client):
    records = [{
        "timestamp": "2026-07-10T12:00:00+00:00",
        "model": "poolside/laguna-xs-2.1:free",
        "provider": "openrouter",
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "estimated_cost_usd": 0.00005,
        "latency_ms": 250,
        "routing_layer": "cheap_bulk",
        "routing_reason": "test request",
        "used_fallback": False,
    }]
    with patch(
        "theriad.accounting.tracker.CostTracker.load_from_disk",
        return_value=records,
    ):
        response = client.get("/api/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["baseline_model"] == DEFAULT_BASELINE_MODEL
    assert data["baseline_cost_usd"] > data["total_cost_usd"]
    assert data["estimated_savings_usd"] > 0
    assert data["estimated_savings_pct"] > 0


def test_stats_uses_hosted_run_records_and_reports_recorded_zero(monkeypatch):
    from types import SimpleNamespace

    records = [{
        "timestamp": "2026-09-04T12:00:00+00:00",
        "model": "poolside/laguna-xs-2.1:free",
        "provider": "openrouter",
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "estimated_cost_usd": 0.0,
        "latency_ms": 25,
        "routing_layer": "cheap_bulk",
        "routing_reason": "free request",
        "used_fallback": False,
    }]
    monkeypatch.setattr(server_module, "hosted_data_plane_enabled", lambda: True)
    monkeypatch.setattr(server_module, "user_scope_id", lambda _request: "hosted-user")
    monkeypatch.setattr(
        "theriad.settings.deployment.is_hosted_mode", lambda: True
    )
    monkeypatch.setattr(
        "theriad.api.auth_context.get_auth_context",
        lambda _request: SimpleNamespace(principal_type="registered"),
    )
    monkeypatch.setattr(
        server_module,
        "_hosted_inference_records",
        lambda _request, _limit: records,
    )

    data = server_module.get_stats(object())

    assert data["has_data"] is True
    assert data["total_requests"] == 1
    assert data["total_cost_usd"] == 0.0
    assert data["cost_status"] == "recorded_zero"
    assert data["costs_complete"] is True
    assert data["unknown_cost_requests"] == 0


def test_stats_does_not_turn_unknown_cost_into_free(monkeypatch, client):
    records = [{
        "timestamp": "2026-09-04T12:00:00+00:00",
        "model": "provider/model-without-usage",
        "provider": "openrouter",
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "estimated_cost_usd": None,
        "latency_ms": 25,
        "routing_layer": "cheap_bulk",
        "routing_reason": "usage omitted",
        "used_fallback": False,
    }]
    with patch(
        "theriad.accounting.tracker.CostTracker.load_from_disk",
        return_value=records,
    ):
        response = client.get("/api/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["has_data"] is True
    assert data["total_cost_usd"] == 0.0
    assert data["cost_status"] == "unavailable"
    assert data["costs_complete"] is False
    assert data["unknown_cost_requests"] == 1
    assert data["estimated_savings_usd"] is None
    assert data["blended_cost_per_mtok"] is None


# ── /api/execute diagnostics ──────────────────────────────────────────────────

class TestExecuteDiagnostics:

    def _exec(self, client, prompt="Say hello.", mock_response=None):
        """POST /api/execute with provider mocked."""
        if mock_response is None:
            mock_response = _make_mock_response()
        with patch(
            "theriad.core.providers.openrouter.OpenRouterProvider.complete",
            new=AsyncMock(return_value=mock_response),
        ):
            import os
            os.environ.setdefault("OPENROUTER_API_KEY", "sk-test-fake")
            return client.post(
                "/api/execute",
                json={"prompt": prompt, "allow_paid_calls": True},
            )

    def test_execution_attempts_present(self, client):
        resp = self._exec(client)
        assert resp.status_code == 200
        data = resp.json()
        assert "execution_attempts" in data
        assert isinstance(data["execution_attempts"], list)
        assert len(data["execution_attempts"]) >= 1

    def test_attempt_has_required_fields(self, client):
        resp = self._exec(client)
        data = resp.json()
        attempt = data["execution_attempts"][0]
        required = {
            "attempt", "model", "tier", "provider", "status",
            "error", "latency_ms", "input_tokens", "output_tokens",
            "timeout_triggered", "fallback_reason",
        }
        assert required.issubset(attempt.keys()), (
            f"Missing fields: {required - attempt.keys()}"
        )

    def test_successful_attempt_has_status_success(self, client):
        resp = self._exec(client)
        data = resp.json()
        attempt = data["execution_attempts"][0]
        assert attempt["status"] == "success"

    def test_successful_attempt_has_token_counts(self, client):
        mock = _make_mock_response(prompt_tokens=15, completion_tokens=30)
        resp = self._exec(client, mock_response=mock)
        data = resp.json()
        attempt = data["execution_attempts"][0]
        assert attempt["input_tokens"] == 15
        assert attempt["output_tokens"] == 30

    def test_separated_latency_fields_present(self, client):
        resp = self._exec(client)
        data = resp.json()
        for field in ("routing_latency_ms", "provider_latency_ms",
                      "fallback_latency_ms", "total_latency_ms"):
            assert field in data, f"Missing field: {field}"
            assert isinstance(data[field], (int, float))

    def test_routing_latency_is_nonnegative(self, client):
        resp = self._exec(client)
        data = resp.json()
        assert data["routing_latency_ms"] >= 0

    def test_total_latency_is_largest(self, client):
        resp = self._exec(client)
        data = resp.json()
        assert data["total_latency_ms"] >= data["routing_latency_ms"]
        assert data["total_latency_ms"] >= data["provider_latency_ms"]

    def test_execution_note_present(self, client):
        resp = self._exec(client)
        data = resp.json()
        assert "execution_note" in data
        assert "single-prompt" in data["execution_note"].lower()

    def test_attempt_has_no_raw_monotonic_times(self, client):
        """start_time / end_time (raw floats) must be stripped before returning."""
        resp = self._exec(client)
        data = resp.json()
        for attempt in data["execution_attempts"]:
            assert "start_time" not in attempt
            assert "end_time" not in attempt


# ── Cost display: null when pricing unknown ───────────────────────────────────

class TestCostDisplay:

    def _exec_with_unknown_model(self, client):
        """Use a model slug not in PricingTable → cost should be null."""
        mock_response = _make_mock_response(model="unknown/nonexistent-model-xyz")
        with patch(
            "theriad.core.providers.openrouter.OpenRouterProvider.complete",
            new=AsyncMock(return_value=mock_response),
        ):
            import os
            os.environ.setdefault("OPENROUTER_API_KEY", "sk-test-fake")
            return client.post(
                "/api/execute",
                json={"prompt": "Say hello.", "allow_paid_calls": True},
            )

    def test_unknown_model_cost_is_null_not_zero(self, client):
        resp = self._exec_with_unknown_model(client)
        assert resp.status_code == 200
        data = resp.json()
        # actual_cost_usd should be null (Python None → JSON null), NOT 0.00000
        assert data["actual_cost_usd"] is None, (
            f"Expected null for unknown model pricing, got: {data['actual_cost_usd']}"
        )

    def test_known_model_cost_is_float(self, client):
        """A model that IS in PricingTable returns a float cost."""
        mock = _make_mock_response(model="openai/gpt-4o", prompt_tokens=100, completion_tokens=50)
        with patch(
            "theriad.core.providers.openrouter.OpenRouterProvider.complete",
            new=AsyncMock(return_value=mock),
        ):
            import os
            os.environ.setdefault("OPENROUTER_API_KEY", "sk-test-fake")
            resp = client.post(
                "/api/execute",
                json={"prompt": "Say hello.", "allow_paid_calls": True},
            )
        data = resp.json()
        if data["actual_cost_usd"] is not None:
            assert isinstance(data["actual_cost_usd"], float)
            assert data["actual_cost_usd"] >= 0


# ── /api/execute safety gate ──────────────────────────────────────────────────

class TestExecuteSafetyGate:

    def test_returns_403_without_allow_paid_calls(self, client):
        resp = client.post("/api/execute", json={"prompt": "Hello."})
        assert resp.status_code == 403

    def test_returns_503_without_api_key(self, client, monkeypatch):
        import os
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Also make config load fail to ensure env fallback
        monkeypatch.setattr(server_module, "_load_config", lambda: (_ for _ in ()).throw(Exception("no config")))
        # Also patch secrets file lookup so a saved key doesn't bypass the gate
        monkeypatch.setattr("theriad.settings.secrets.get_openrouter_api_key", lambda: None)
        resp = client.post("/api/execute", json={"prompt": "Hello.", "allow_paid_calls": True})
        assert resp.status_code == 503


# ── /api/route includes BuildProjectRule ─────────────────────────────────────

class TestRouteIncludesBuildProjectRule:

    def test_build_app_prompt_routes_to_code_strong(self, client):
        resp = client.post("/api/route", json={
            "prompt": "Build a React app with auth and a database."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "code_strong", (
            f"Expected code_strong, got {data['tier']!r} (reason: {data['reason']!r})"
        )

    def test_build_prompt_routing_path_includes_build_project(self, client):
        resp = client.post("/api/route", json={
            "prompt": "Create a backend service with authentication."
        })
        data = resp.json()
        layer_names = [r["layer"] for r in data["routing_path"]]
        assert "L0:build_project" in layer_names, (
            f"build_project not in routing path: {layer_names}"
        )

    def test_plain_summarize_does_not_hit_code_strong(self, client):
        resp = client.post("/api/route", json={"prompt": "Summarize this paragraph."})
        data = resp.json()
        assert data["tier"] != "code_strong"
