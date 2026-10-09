"""
Integration tests for POST /api/workflow/analyze.

Uses FastAPI TestClient. Monkeypatches _create_workflow_detector to inject
MockEmbeddingBackend so no model download is required.
"""
from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

import theriad.api.server as server_module
from theriad.layers.layer2_embedding.backends import MockEmbeddingBackend
from theriad.workflow.detector import WorkflowDetector


def _mock_detector():
    return WorkflowDetector(embedding_backend=MockEmbeddingBackend(dim=64))


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server_module, "_create_workflow_detector", _mock_detector)
    return TestClient(server_module.app)


class TestWorkflowAnalyzeEndpoint:

    def test_endpoint_exists(self, client):
        resp = client.post("/api/workflow/analyze", json={"prompt": "Summarize this."})
        assert resp.status_code == 200

    def test_no_api_key_required(self, client):
        """Endpoint must work without OPENROUTER_API_KEY set."""
        import os
        env_backup = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            resp = client.post("/api/workflow/analyze", json={"prompt": "Summarize this."})
            assert resp.status_code == 200
        finally:
            if env_backup is not None:
                os.environ["OPENROUTER_API_KEY"] = env_backup

    def test_single_prompt_returns_workflow_selected_false(self, client):
        resp = client.post("/api/workflow/analyze", json={"prompt": "Summarize this paragraph."})
        data = resp.json()
        assert data["workflow_selected"] is False

    def test_workflow_prompt_returns_workflow_selected_true(self, client):
        resp = client.post("/api/workflow/analyze", json={
            "prompt": "Design, build, test, and deploy a React app."
        })
        data = resp.json()
        assert data["workflow_selected"] is True

    def test_workflow_response_has_steps(self, client):
        resp = client.post("/api/workflow/analyze", json={
            "prompt": "Build, test, and deploy a new backend service."
        })
        data = resp.json()
        if data["workflow_selected"]:
            assert len(data["steps"]) == 6
            assert len(data["step_costs"]) == len(data["steps"])
            assert all("model" in step for step in data["step_costs"])
            assert data["estimated_total_cost_usd"] >= 0.0
            assert data["estimated_latency_ms"] > 0

    def test_response_schema_keys(self, client):
        resp = client.post("/api/workflow/analyze", json={"prompt": "Explain this."})
        data = resp.json()
        required = {
            "workflow_selected", "reason", "signals", "workflow_type",
            "decision_source", "confidence", "semantic_margin", "embedding_model",
            "embedding_dims_used", "steps", "estimated_total_cost_usd",
            "step_costs",
            "baseline_estimated_cost_usd", "estimated_savings_pct",
            "estimated_latency_ms", "policy", "baseline",
        }
        assert required.issubset(data.keys())

    def test_confidence_is_float_between_0_and_1(self, client):
        resp = client.post("/api/workflow/analyze", json={"prompt": "Summarize this."})
        data = resp.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_decision_source_is_valid(self, client):
        resp = client.post("/api/workflow/analyze", json={"prompt": "Translate this."})
        data = resp.json()
        assert data["decision_source"] in ("rule_guardrail", "mrl_embedding", "default")

    def test_policy_echoed_in_response(self, client):
        resp = client.post("/api/workflow/analyze", json={
            "prompt": "Summarize this.",
            "policy": "quality_first",
        })
        data = resp.json()
        assert data["policy"] == "quality_first"

    def test_baseline_echoed_in_response(self, client):
        resp = client.post("/api/workflow/analyze", json={
            "prompt": "Summarize this.",
            "baseline": "anthropic/claude-3-haiku",
        })
        data = resp.json()
        assert data["baseline"] == "anthropic/claude-3-haiku"

    def test_non_workflow_has_empty_steps(self, client):
        resp = client.post("/api/workflow/analyze", json={"prompt": "Translate this sentence."})
        data = resp.json()
        if not data["workflow_selected"]:
            assert data["steps"] == []

    def test_workflow_step_has_required_fields(self, client):
        resp = client.post("/api/workflow/analyze", json={
            "prompt": "Design, build, test, and deploy a React app."
        })
        data = resp.json()
        if data["workflow_selected"] and data["steps"]:
            step = data["steps"][0]
            required_step_keys = {
                "name", "task_type", "description", "input_refs",
                "output_key", "routing_tier", "uses_model",
            }
            assert required_step_keys.issubset(step.keys())
