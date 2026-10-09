"""
Tests for D-050: LLM synthesis of the browser agent's final answer.

Coverage:
1. Synthesis is skipped (returns None) when allow_paid_calls=False (default).
2. A fake provider's answer replaces the template and updates accounting.
3. Provider failure returns None — deterministic fallback is kept.
4. Empty provider content returns None.
5. Placeholder search results are excluded from evidence.
6. Deterministic fallback never presents the placeholder as a finding.
7. Budget exhaustion (max_model_calls) skips synthesis.
"""
from __future__ import annotations

from theriad.agents.answer_synthesis import (
    build_browser_evidence,
    synthesize_browser_answer,
)
from theriad.agents.browser_schemas import ActionLogEntry, BrowserAgentStartRequest
from theriad.agents.browser_session_store import BrowserSession
from theriad.core.providers.interfaces import (
    CompletionRequest,
    CompletionResponse,
    ProviderError,
    ProviderProtocol,
)
from theriad.search.service import PLACEHOLDER_TITLE_PREFIX, is_placeholder_result


def _make_session(**kwargs) -> BrowserSession:
    defaults = dict(goal="Where can I find thrifting events in Philly?")
    defaults.update(kwargs)
    return BrowserSession(BrowserAgentStartRequest(**defaults))


class _FakeProvider(ProviderProtocol):
    def __init__(
        self,
        content: str = "Natural answer.",
        raise_error: bool = False,
        finish_reason: str | None = None,
    ):
        self.content = content
        self.raise_error = raise_error
        self.finish_reason = finish_reason
        self.requests: list[CompletionRequest] = []

    @property
    def name(self) -> str:
        return "fake"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if self.raise_error:
            raise ProviderError("fake", "boom")
        return CompletionResponse(
            model="fake/model",
            content=self.content,
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=1.0,
            provider="fake",
            finish_reason=self.finish_reason,
        )

    async def health_check(self) -> bool:
        return True


def _real_result() -> dict:
    return {
        "title": "Philly Flea Markets",
        "url": "https://phillyflea.example/events",
        "snippet": "Weekly thrift and vintage markets across Philadelphia.",
        "source": "brave",
    }


def _placeholder_result() -> dict:
    return {
        "title": f"{PLACEHOLDER_TITLE_PREFIX}thrifting events in Philly",
        "url": "https://html.duckduckgo.com/html/?q=thrifting",
        "snippet": (
            "Browser search via duckduckgo. Configure Brave Search credentials "
            "for API-native result extraction."
        ),
        "source": "duckduckgo",
    }


# ── Gating ────────────────────────────────────────────────────────────────────

async def test_synthesis_skipped_without_allow_paid_calls():
    session = _make_session()  # allow_paid_calls defaults to False
    session.search_results = [_real_result()]
    provider = _FakeProvider()

    answer = await synthesize_browser_answer(session, provider=provider, model="fake/model")

    assert answer is None
    assert provider.requests == []
    assert session.model_calls == 0


async def test_synthesis_skipped_when_model_call_budget_exhausted():
    session = _make_session(allow_paid_calls=True, max_model_calls=1)
    session.model_calls = 1
    session.search_results = [_real_result()]
    provider = _FakeProvider()

    answer = await synthesize_browser_answer(session, provider=provider, model="fake/model")

    assert answer is None
    assert provider.requests == []


# ── Happy path ────────────────────────────────────────────────────────────────

async def test_synthesis_returns_model_answer_and_updates_accounting():
    session = _make_session(allow_paid_calls=True)
    session.search_results = [_real_result()]
    provider = _FakeProvider(content="Check out the weekly Philly Flea Markets.")

    answer = await synthesize_browser_answer(session, provider=provider, model="fake/model")

    assert answer == "Check out the weekly Philly Flea Markets."
    assert session.model_calls == 1
    # Evidence includes the real snippet and the user goal.
    sent = provider.requests[0]
    user_content = sent.messages[-1].content
    assert "thrifting events in Philly" in user_content
    assert "Weekly thrift and vintage markets" in user_content
    assert sent.max_tokens is None


async def test_truncated_provider_answer_is_rejected_for_fallback():
    session = _make_session(allow_paid_calls=True)
    session.search_results = [_real_result()]
    provider = _FakeProvider(content="The answer starts but stops here.", finish_reason="length")

    answer = await synthesize_browser_answer(session, provider=provider, model="fake/model")

    assert answer is None
    assert session.model_calls == 1
    assert provider.requests[0].max_tokens is None


# ── Failure paths keep the deterministic fallback ─────────────────────────────

async def test_provider_error_returns_none():
    session = _make_session(allow_paid_calls=True)
    session.search_results = [_real_result()]

    answer = await synthesize_browser_answer(
        session, provider=_FakeProvider(raise_error=True), model="fake/model"
    )

    assert answer is None


async def test_empty_model_content_returns_none():
    session = _make_session(allow_paid_calls=True)
    session.search_results = [_real_result()]

    answer = await synthesize_browser_answer(
        session, provider=_FakeProvider(content="   "), model="fake/model"
    )

    assert answer is None


async def test_no_evidence_returns_none_without_provider_call():
    session = _make_session(allow_paid_calls=True)
    provider = _FakeProvider()

    answer = await synthesize_browser_answer(session, provider=provider, model="fake/model")

    assert answer is None
    assert provider.requests == []


# ── Placeholder handling ──────────────────────────────────────────────────────

def test_is_placeholder_result_detection():
    assert is_placeholder_result(_placeholder_result())
    assert not is_placeholder_result(_real_result())


def test_placeholder_excluded_from_evidence():
    session = _make_session()
    session.search_results = [_placeholder_result(), _real_result()]

    evidence = build_browser_evidence(session)

    assert "Configure Brave Search" not in evidence
    assert "Weekly thrift and vintage markets" in evidence


def test_fallback_never_presents_placeholder_as_finding():
    from theriad.agents.browser_runtime import _synthesize_browser_final_answer

    session = _make_session()
    session.search_results = [_placeholder_result()]
    session.action_log.append(ActionLogEntry(
        step=1,
        action_type="navigate",
        target="https://html.duckduckgo.com/html/?q=thrifting",
        url="https://html.duckduckgo.com/html/?q=thrifting",
        result="success",
        summary="Navigated",
    ))

    answer = _synthesize_browser_final_answer(session)

    assert "Configure Brave Search" not in answer
    assert "Here’s what I found" not in answer
    assert "https://html.duckduckgo.com/html/?q=thrifting" in answer


def test_fallback_still_lists_real_results():
    from theriad.agents.browser_runtime import _synthesize_browser_final_answer

    session = _make_session()
    session.search_results = [_placeholder_result(), _real_result()]

    answer = _synthesize_browser_final_answer(session)

    assert "Philly Flea Markets" in answer
    assert "https://phillyflea.example/events" in answer
    assert "Configure Brave Search" not in answer
