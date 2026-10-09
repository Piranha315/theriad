"""Regression tests for evidence acquisition and freshness-safe synthesis."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from theriad.core.providers.interfaces import CompletionResponse
from theriad.search.service import SearchResult, SearchService


def test_extract_visible_text_omits_non_user_content():
    from theriad.search.evidence import extract_visible_text

    text = extract_visible_text(
        "<html><head><title>Hidden</title></head><body>"
        "<script>temperature = 50</script><h1>Current conditions</h1>"
        "<p>90 degrees and sunny.</p></body></html>"
    )

    assert "Current conditions" in text
    assert "90 degrees and sunny" in text
    assert "temperature = 50" not in text
    assert "Hidden" not in text


@pytest.mark.asyncio
async def test_fetch_blocks_private_dns_answer_before_request(monkeypatch):
    import theriad.search.evidence as evidence

    requests: list[str] = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url):
            requests.append(url)
            raise AssertionError("private DNS targets must be rejected before HTTP")

    monkeypatch.setattr(evidence.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(
        evidence.socket,
        "getaddrinfo",
        lambda _host, port, **_kwargs: [
            (evidence.socket.AF_INET, evidence.socket.SOCK_STREAM, 6, "", ("10.0.0.7", port))
        ],
    )

    result = await evidence.fetch_page_evidence("https://public.example/page")

    assert result["status"] == "blocked_url"
    assert requests == []


@pytest.mark.asyncio
async def test_fetch_validates_each_redirect_target_before_request(monkeypatch):
    import theriad.search.evidence as evidence

    requests: list[str] = []

    class FakeResponse:
        def __init__(self, status_code, *, headers=None, text=""):
            self.status_code = status_code
            self.headers = headers or {}
            self.text = text

    class FakeClient:
        instances: list["FakeClient"] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.__class__.instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url):
            requests.append(url)
            return FakeResponse(
                302,
                headers={"location": "http://127.0.0.1:8080/metadata"},
            )

    monkeypatch.setattr(evidence.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(
        evidence.socket,
        "getaddrinfo",
        lambda _host, port, **_kwargs: [
            (evidence.socket.AF_INET, evidence.socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )

    result = await evidence.fetch_page_evidence("https://public.example/start")

    assert result["status"] == "blocked_url"
    assert requests == ["https://public.example/start"]
    assert FakeClient.instances[0].kwargs["follow_redirects"] is False
    assert FakeClient.instances[0].kwargs["trust_env"] is False


@pytest.mark.asyncio
async def test_current_question_synthesizes_when_only_search_snippets_are_available():
    from theriad.agents.search_synthesis import synthesize_web_search
    from theriad.agents.model_telemetry import record_model_usage

    class FakeService:
        async def search(self, _query):
            return [SearchResult(
                "Weather source",
                "https://weather.example/current",
                "Philadelphia is 77°F and overcast, according to a recent update.",
                "test-api",
            )]

    telemetry: dict = {}

    async def fake_model(*_args, **kwargs):
        record_model_usage(CompletionResponse(
            model="xiaomi/mimo-v2.5",
            content=(
                "A recent search result reports 77°F and overcast conditions in "
                "Philadelphia, but the page could not be independently verified."
            ),
            prompt_tokens=100,
            completion_tokens=30,
            latency_ms=42.0,
            provider="test-provider",
        ), kwargs.get("usage_sink"))
        return (
            "A recent search result reports 77°F and overcast conditions in "
            "Philadelphia, but the page could not be independently verified."
        )

    with patch(
        "theriad.agents.search_synthesis._research_queries",
        new=AsyncMock(return_value=["current weather in Philadelphia"]),
    ), patch(
        "theriad.agents.llm_planner._call_model",
        new=fake_model,
    ):
        answer, _results, _queries = await synthesize_web_search(
            "What is the current weather in Philadelphia?",
            service=FakeService(),
            telemetry=telemetry,
        )

    assert "77°F" in answer
    assert "independently verified" in answer
    assert telemetry["evidence_quality"] == "search_snippets_only"
    assert telemetry["model_calls"] == 1
    assert telemetry["cost_known"] is True
    assert telemetry["estimated_cost_usd"] > 0


@pytest.mark.asyncio
async def test_underspecified_weather_asks_for_location_before_searching():
    from theriad.agents.search_synthesis import synthesize_web_search

    class UnusedService:
        async def search(self, _query):
            raise AssertionError("an ambiguous local-weather request should clarify first")

    telemetry: dict = {}
    answer, results, queries = await synthesize_web_search(
        "Can you look up the weather for me?",
        service=UnusedService(),
        telemetry=telemetry,
    )

    assert answer == "Weather is location-specific. What city, region, or ZIP code should I use?"
    assert results == []
    assert queries == []
    assert telemetry["synthesis_status"] == "clarification_required"
    assert telemetry["model_calls"] == 0


@pytest.mark.asyncio
async def test_current_question_uses_fetched_page_and_records_model_usage():
    from theriad.agents.search_synthesis import synthesize_web_search
    from theriad.agents.model_telemetry import record_model_usage

    class FakeProvider:
        name = "test-provider"

        async def search(self, _query):
            return [SearchResult(
                "Current source",
                "https://weather.example/current",
                "A discovery snippet",
                "test-api",
            )]

    telemetry: dict = {}
    captured: dict = {}

    async def fake_model(*_args, **kwargs):
        captured["max_tokens"] = kwargs.get("max_tokens")
        record_model_usage(CompletionResponse(
            model="xiaomi/mimo-v2.5",
            content="It is 90°F and sunny.",
            prompt_tokens=1_000,
            completion_tokens=100,
            latency_ms=42.0,
            provider="test-provider",
        ), kwargs.get("usage_sink"))
        return "It is 90°F and sunny."

    with patch(
        "theriad.agents.search_synthesis._research_queries",
        new=AsyncMock(return_value=["current weather in Philadelphia"]),
    ), patch(
        "theriad.agents.search_synthesis.fetch_results_evidence",
        new=AsyncMock(return_value=[{
            "url": "https://weather.example/current",
            "retrieved_at": "2026-08-09T21:00:00+00:00",
            "status": "fetched",
            "status_code": 200,
            "text": "Current conditions: 90°F, sunny, updated at 17:00 EDT.",
        }]),
    ), patch(
        "theriad.agents.llm_planner._call_model",
        new=fake_model,
    ):
        answer, _results, _queries = await synthesize_web_search(
            "What is the current weather in Philadelphia?",
            service=SearchService(provider=FakeProvider()),
            telemetry=telemetry,
        )

    assert "90°F" in answer
    assert telemetry["evidence_quality"] == "grounded"
    assert telemetry["model_calls"] == 1
    assert telemetry["models_used"][0]["model"] == "xiaomi/mimo-v2.5"
    assert telemetry["estimated_cost_usd"] > 0
    assert telemetry["cost_known"] is True
    assert telemetry["synthesis_finish_reason"] is None
    assert captured["max_tokens"] >= 8_192


@pytest.mark.asyncio
async def test_free_only_search_synthesis_uses_verified_zero_cost_models():
    from theriad.agents.search_synthesis import synthesize_web_search

    class FakeService:
        async def search(self, _query):
            return [SearchResult(
                "Public source",
                "https://example.org/source",
                "A useful public-web result.",
                "test-api",
            )]

    calls: list[dict] = []

    async def fake_model(*_args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("json_mode"):
            return '{"queries": ["independent source"]}'
        return "The source supports this answer: [1]."

    telemetry: dict = {}
    with patch(
        "theriad.agents.llm_planner._call_model",
        new=fake_model,
    ):
        answer, _results, _queries = await synthesize_web_search(
            "Find public information about this topic",
            service=FakeService(),
            telemetry=telemetry,
            free_only=True,
        )

    assert answer
    assert len(calls) == 2
    assert all(call["assistant_provider"] == "theriad" for call in calls)
    assert all(call["model"].endswith(":free") for call in calls)
    assert telemetry["free_only"] is True
    assert telemetry["estimated_cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_truncated_synthesis_falls_back_instead_of_returning_partial_answer():
    from theriad.agents.search_synthesis import synthesize_web_search
    from theriad.agents.model_telemetry import record_model_usage

    class FakeProvider:
        name = "test-provider"

        async def search(self, _query):
            return [SearchResult(
                "Current source",
                "https://weather.example/current",
                "A discovery snippet",
                "test-api",
            )]

    partial = "Based on today's data, Philadelphia will be hot.\n\n## Key Weather"

    async def truncated_model(*_args, **kwargs):
        record_model_usage(CompletionResponse(
            model="xiaomi/mimo-v2.5",
            content=partial,
            prompt_tokens=1_000,
            completion_tokens=100,
            latency_ms=42.0,
            provider="test-provider",
            finish_reason="length",
        ), kwargs.get("usage_sink"))
        return partial

    telemetry: dict = {}
    with patch(
        "theriad.agents.search_synthesis._research_queries",
        new=AsyncMock(return_value=["current weather in Philadelphia"]),
    ), patch(
        "theriad.agents.search_synthesis.fetch_results_evidence",
        new=AsyncMock(return_value=[{
            "url": "https://weather.example/current",
            "retrieved_at": "2026-08-09T21:00:00+00:00",
            "status": "fetched",
            "status_code": 200,
            "text": "Current conditions: 90°F, sunny, updated at 17:00 EDT.",
        }]),
    ), patch(
        "theriad.agents.llm_planner._call_model",
        new=truncated_model,
    ):
        answer, _results, _queries = await synthesize_web_search(
            "What is the current weather in Philadelphia?",
            service=SearchService(provider=FakeProvider()),
            telemetry=telemetry,
        )

    assert "I found these live web results" in answer
    assert partial not in answer
    assert telemetry["synthesis_finish_reason"] == "length"


@pytest.mark.asyncio
async def test_truncated_synthesis_retries_before_using_digest_fallback():
    from theriad.agents.search_synthesis import synthesize_web_search
    from theriad.agents.model_telemetry import record_model_usage

    class FakeProvider:
        name = "test-provider"

        async def search(self, _query):
            return [SearchResult(
                "Moon phases",
                "https://space.example/moon",
                "The Moon's phases are caused by changing sunlight angles.",
                "test-api",
            )]

    calls = 0

    async def recovering_model(*_args, **kwargs):
        nonlocal calls
        calls += 1
        truncated = calls == 1
        content = "partial answer" if truncated else "The Moon's phases are caused by the changing angle of sunlight."
        record_model_usage(CompletionResponse(
            model="xiaomi/mimo-v2.5",
            content=content,
            prompt_tokens=100,
            completion_tokens=30,
            latency_ms=10.0,
            provider="test-provider",
            finish_reason="length" if truncated else "stop",
        ), kwargs.get("usage_sink"))
        return content

    telemetry: dict = {}
    with patch(
        "theriad.agents.search_synthesis._research_queries",
        new=AsyncMock(return_value=["why does the moon have phases"]),
    ), patch(
        "theriad.agents.llm_planner._call_model",
        new=recovering_model,
    ):
        answer, _results, _queries = await synthesize_web_search(
            "Explain why the Moon has phases.",
            service=FakeProvider(),
            telemetry=telemetry,
        )

    assert calls == 2
    assert answer == "The Moon's phases are caused by the changing angle of sunlight."
    assert telemetry["synthesis_status"] == "synthesized"


def test_digest_fallback_does_not_truncate_evidence():
    from theriad.agents.search_synthesis import _fallback_answer

    long_snippet = "A complete source passage " + ("with additional context. " * 40)
    answer = _fallback_answer(
        "Summarize this source",
        [{"title": "Source", "url": "https://example.org", "snippet": long_snippet}],
    )

    assert " ".join(long_snippet.split()) in answer
    assert "…" not in answer


@pytest.mark.asyncio
async def test_guest_search_uses_verified_free_models_through_theriad_provider():
    from theriad.agents.search_synthesis import synthesize_web_search

    class FakeProvider:
        name = "test-provider"

        async def search(self, _query):
            return [SearchResult(
                "Current source",
                "https://weather.example/current",
                "A discovery snippet",
                "test-api",
            )]

    calls: list[dict] = []

    async def fake_model(*_args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("json_mode"):
            return '{"queries": ["current weather in Philadelphia"]}'
        return "It is 90°F and sunny."

    with patch(
        "theriad.agents.search_synthesis.fetch_results_evidence",
        new=AsyncMock(return_value=[{
            "url": "https://weather.example/current",
            "retrieved_at": "2026-08-09T21:00:00+00:00",
            "status": "fetched",
            "status_code": 200,
            "text": "Current conditions: 90°F, sunny.",
        }]),
    ), patch(
        "theriad.agents.llm_planner._call_model",
        new=fake_model,
    ):
        answer, _results, _queries = await synthesize_web_search(
            "Find today's weather in Philadelphia",
            service=SearchService(provider=FakeProvider()),
            free_only=True,
        )

    assert "90°F" in answer
    assert len(calls) == 2
    assert all(call["assistant_provider"] == "theriad" for call in calls)
    assert all(call["model"].endswith(":free") for call in calls)
    assert all(call["model"] != "xiaomi/mimo-v2.5" for call in calls)
    assert all(call["model"] == "nvidia/nemotron-3-super-120b-a12b:free" for call in calls)
