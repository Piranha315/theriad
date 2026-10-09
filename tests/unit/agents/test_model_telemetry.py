"""Tests for user-facing model and cost attribution."""
from __future__ import annotations

from theriad.agents.browser_schemas import BrowserAgentStartRequest
from theriad.agents.browser_session_store import BrowserSession
from theriad.agents.model_telemetry import record_model_usage
from theriad.core.providers.interfaces import CompletionResponse


def test_browser_status_reports_actual_model_and_nonzero_cost():
    session = BrowserSession(BrowserAgentStartRequest(goal="check a public page"))
    record_model_usage(CompletionResponse(
        model="xiaomi/mimo-v2.5",
        content="done",
        prompt_tokens=1_000,
        completion_tokens=500,
        latency_ms=125.0,
        provider="test-provider",
        finish_reason="stop",
    ), session.model_usage)

    status = session.to_status().model_dump()

    assert status["models_used"][0]["model"] == "xiaomi/mimo-v2.5"
    assert status["models_used"][0]["calls"] == 1
    assert status["estimated_cost_usd"] == 0.00028
    assert status["cost_known"] is True
    assert status["models_used"][0]["finish_reasons"] == ["stop"]
    assert status["models_used"][0]["truncated"] is False
