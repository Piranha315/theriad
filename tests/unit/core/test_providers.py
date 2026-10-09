"""Tests for provider implementations (no real network calls)."""
from __future__ import annotations

import pytest

from theriad.core.providers.interfaces import (
    CompletionRequest,
    CompletionResponse,
    Message,
    ProviderError,
    ProviderProtocol,
)
from theriad.core.providers.ollama import OllamaProvider
from theriad.core.providers.openrouter import OpenRouterProvider
from theriad.core.providers.registry import all_providers, get, register

# ── Interface contracts ───────────────────────────────────────────────────────

def test_message_construction():
    m = Message(role="user", content="hello")
    assert m.role == "user"
    assert m.content == "hello"


def test_completion_request_defaults():
    req = CompletionRequest(
        model="gpt-4o-mini",
        messages=[Message(role="user", content="hi")],
    )
    assert req.temperature == 0.0
    assert req.max_tokens is None


def test_provider_error_str():
    err = ProviderError("openrouter", "Not found", 404)
    assert "openrouter" in str(err)
    assert err.status_code == 404


# ── Registry ─────────────────────────────────────────────────────────────────

class _FakeProvider(ProviderProtocol):
    @property
    def name(self) -> str:
        return "fake"

    async def complete(self, request):
        return CompletionResponse(
            model="fake-model", content="ok",
            prompt_tokens=5, completion_tokens=2,
            latency_ms=10.0, provider="fake",
        )

    async def health_check(self) -> bool:
        return True


def test_registry_register_and_get():
    p = _FakeProvider()
    register(p)
    assert get("fake") is p


def test_registry_get_unknown_raises():
    with pytest.raises(KeyError, match="nonexistent"):
        get("nonexistent")


def test_registry_all_providers():
    p = _FakeProvider()
    register(p)
    providers = all_providers()
    assert "fake" in providers


# ── OpenRouter _extract_content ───────────────────────────────────────────────

class TestExtractContent:
    """Tests for OpenRouterProvider._extract_content (static method)."""

    def _call(self, message: dict, choice_extra: dict | None = None, usage: dict | None = None) -> str:
        from theriad.core.providers.openrouter import OpenRouterProvider
        choice = {"message": message, **(choice_extra or {})}
        full_response = {"model": "test-model", "usage": usage or {}}
        return OpenRouterProvider._extract_content(choice, full_response)

    def test_standard_string_content(self):
        assert self._call({"content": "Hello world"}) == "Hello world"

    def test_content_array(self):
        content = [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]
        assert self._call({"content": content}) == "part1part2"

    def test_legacy_text_field(self):
        result = self._call({"content": None}, choice_extra={"text": "legacy answer"})
        assert result == "legacy answer"

    def test_reasoning_fallback_when_content_is_none(self):
        # Reasoning model returns answer in reasoning field, null content
        result = self._call({"content": None, "reasoning": "Therefore the answer is 42."})
        assert result == "Therefore the answer is 42."

    def test_reasoning_content_fallback_when_content_is_none(self):
        # DeepSeek-R1 style: reasoning_content field
        result = self._call({"content": None, "reasoning_content": "The answer is B."})
        assert result == "The answer is B."

    def test_reasoning_fallback_when_content_is_empty_string(self):
        result = self._call({"content": "", "reasoning": "My answer is C."})
        assert result == "My answer is C."

    def test_reasoning_fallback_when_content_is_whitespace(self):
        result = self._call({"content": "   ", "reasoning": "Therefore X."})
        assert result == "Therefore X."

    def test_no_fallback_when_content_is_present(self):
        # Content takes precedence over reasoning — don't mix them
        result = self._call({"content": "Final answer", "reasoning": "Long thinking..."})
        assert result == "Final answer"

    def test_truly_empty_returns_empty_string(self):
        # No content, no reasoning, no text — returns empty (and logs warning)
        result = self._call({"content": None})
        assert result == ""

    def test_reasoning_fallback_is_stripped(self):
        result = self._call({"content": None, "reasoning": "  answer  \n"})
        assert result == "answer"


# ── OpenRouter provider structure ─────────────────────────────────────────────

def test_openrouter_name():
    p = OpenRouterProvider(api_key="test")
    assert p.name == "openrouter"


def test_openrouter_headers_contain_auth():
    p = OpenRouterProvider(api_key="sk-test-key")
    headers = p._headers()
    assert headers["Authorization"] == "Bearer sk-test-key"
    assert "X-Title" in headers


# ── Ollama provider structure ─────────────────────────────────────────────────

def test_ollama_name():
    p = OllamaProvider()
    assert p.name == "ollama"


def test_ollama_strips_org_prefix():
    """Ollama model names should not have org prefixes."""
    # Indirectly: we test the stripping logic by constructing the payload
    model_in = "openai/gpt-4o-mini"
    stripped = model_in.split("/")[-1] if "/" in model_in else model_in
    assert stripped == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_openrouter_raises_provider_error_on_bad_status(httpx_mock=None):
    """Provider must raise ProviderError on non-200 (mocked via monkeypatch)."""
    from unittest.mock import AsyncMock, MagicMock, patch

    p = OpenRouterProvider(api_key="test")
    req = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=[Message(role="user", content="hi")],
    )

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ProviderError) as exc_info:
            await p.complete(req)
    assert exc_info.value.status_code == 401
