"""Focused tests for direct OpenAI/Anthropic provider runtime behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from theriad.core.providers.anthropic import AnthropicProvider
from theriad.core.providers.factory import (
    ProviderConfigurationError,
    build_provider,
    get_effective_provider_name,
)
from theriad.core.providers.interfaces import CompletionRequest, Message, ProviderError
from theriad.core.providers.openai import OpenAIProvider
from theriad.core.providers.openai_compatible import OpenAICompatibleProvider
from theriad.core.providers.openrouter import OpenRouterProvider
from theriad.core.providers.runtime import (
    get_provider_candidates,
    get_provider_model_for_tier,
    normalize_direct_model_id,
)
from theriad.models.tiers import (
    TIER_CHEAP_BULK,
    TIER_CODE_STRONG,
    TIER_FREE_GENERAL,
    TIER_PREMIUM,
    TIER_STRONG_REASONING,
)


def _response(status_code: int, data: object, text: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.json = MagicMock(return_value=data)
    return response


def _client(response: MagicMock) -> AsyncMock:
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(return_value=response)
    client.get = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_openai_responses_request_and_output_mapping():
    response = _response(
        200,
        {
            "id": "resp_test",
            "model": "gpt-5.6-terra",
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": [{"text": "private reasoning"}]},
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Hello "},
                        {"type": "output_text", "text": "world"},
                    ],
                },
            ],
            "usage": {"input_tokens": 12, "output_tokens": 3},
        },
    )
    client = _client(response)
    provider = OpenAIProvider(api_key="sk-openai-private", timeout=9.0)
    request = CompletionRequest(
        model="gpt-5.6-terra",
        messages=[
            Message(role="system", content="Be concise."),
            Message(role="user", content="Say hello."),
        ],
        max_tokens=250,
    )

    with patch("theriad.core.providers.openai.httpx.AsyncClient", return_value=client):
        result = await provider.complete(request)

    call = client.post.await_args
    assert call.args[0] == "https://api.openai.com/v1/responses"
    assert call.kwargs["headers"]["Authorization"] == "Bearer sk-openai-private"
    payload = call.kwargs["json"]
    assert payload == {
        "model": "gpt-5.6-terra",
        "input": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say hello."},
        ],
        "store": False,
        "max_output_tokens": 250,
    }
    assert result.content == "Hello world"
    assert "private reasoning" not in result.content
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 3
    assert result.finish_reason == "stop"
    assert result.provider == "openai"


@pytest.mark.asyncio
async def test_openai_error_does_not_echo_key_or_upstream_body():
    key = "sk-openai-should-never-escape"
    response = _response(401, {}, text=f"Unauthorized credential {key}")
    client = _client(response)
    provider = OpenAIProvider(api_key=key)
    request = CompletionRequest(
        model="gpt-5.4-mini",
        messages=[Message(role="user", content="hello")],
    )

    with patch("theriad.core.providers.openai.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete(request)

    assert exc_info.value.status_code == 401
    assert key not in str(exc_info.value)
    assert response.text not in str(exc_info.value)


@pytest.mark.asyncio
async def test_openai_timeout_is_safe():
    client = _client(_response(200, {}))
    client.post.side_effect = httpx.TimeoutException("timed out")
    provider = OpenAIProvider(api_key="sk-openai-private", timeout=3.0)
    request = CompletionRequest(
        model="gpt-5.4-mini",
        messages=[Message(role="user", content="hello")],
    )

    with patch("theriad.core.providers.openai.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError, match="timeout after 3.0s"):
            await provider.complete(request)


@pytest.mark.asyncio
async def test_openai_health_check_handles_request_failure():
    client = _client(_response(200, {}))
    client.get.side_effect = RuntimeError("offline")
    provider = OpenAIProvider(api_key="sk-openai-private")

    with patch("theriad.core.providers.openai.httpx.AsyncClient", return_value=client):
        assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_anthropic_messages_request_and_output_mapping():
    response = _response(
        200,
        {
            "id": "msg_test",
            "model": "claude-sonnet-5",
            "content": [
                {"type": "thinking", "thinking": "private reasoning"},
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4},
            "stop_reason": "end_turn",
        },
    )
    client = _client(response)
    provider = AnthropicProvider(api_key="sk-ant-private", timeout=11.0)
    request = CompletionRequest(
        model="claude-sonnet-5",
        messages=[
            Message(role="system", content="Be accurate."),
            Message(role="system", content="Be concise."),
            Message(role="user", content="Say hello."),
        ],
        max_tokens=300,
    )

    with patch("theriad.core.providers.anthropic.httpx.AsyncClient", return_value=client):
        result = await provider.complete(request)

    call = client.post.await_args
    assert call.args[0] == "https://api.anthropic.com/v1/messages"
    assert call.kwargs["headers"]["x-api-key"] == "sk-ant-private"
    assert call.kwargs["headers"]["anthropic-version"] == "2023-06-01"
    payload = call.kwargs["json"]
    assert payload["system"] == "Be accurate.\n\nBe concise."
    assert payload["messages"] == [{"role": "user", "content": "Say hello."}]
    assert payload["max_tokens"] == 300
    assert result.content == "Hello world"
    assert "private reasoning" not in result.content
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 4
    assert result.finish_reason == "end_turn"
    assert result.provider == "anthropic"


@pytest.mark.asyncio
async def test_anthropic_default_max_tokens_and_safe_error():
    key = "sk-ant-should-never-escape"
    response = _response(403, {}, text=f"Rejected credential {key}")
    client = _client(response)
    provider = AnthropicProvider(api_key=key)
    request = CompletionRequest(
        model="claude-haiku-4-5-20251001",
        messages=[Message(role="user", content="hello")],
    )

    assert provider._payload(request)["max_tokens"] == 8192
    with patch("theriad.core.providers.anthropic.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete(request)

    assert exc_info.value.status_code == 403
    assert key not in str(exc_info.value)
    assert response.text not in str(exc_info.value)


@pytest.mark.asyncio
async def test_anthropic_health_check_uses_models_endpoint():
    client = _client(_response(200, {"data": []}))
    provider = AnthropicProvider(api_key="sk-ant-private")

    with patch("theriad.core.providers.anthropic.httpx.AsyncClient", return_value=client):
        assert await provider.health_check() is True

    call = client.get.await_args
    assert call.args[0] == "https://api.anthropic.com/v1/models"
    assert call.kwargs["headers"]["x-api-key"] == "sk-ant-private"


def test_factory_builds_direct_provider_with_explicit_key():
    provider = build_provider("openai", api_key="sk-openai-private", timeout=7.0)
    assert isinstance(provider, OpenAIProvider)
    assert provider.name == "openai"


@pytest.mark.asyncio
async def test_openrouter_multi_model_request_caches_stable_system_prompt():
    response = _response(
        200,
        {
            "model": "provider/chosen",
            "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1},
        },
    )
    client = _client(response)
    provider = OpenRouterProvider(api_key="sk-or-private")
    request = CompletionRequest(
        model="anthropic/claude-sonnet-4-6",
        messages=[
            Message(role="system", content="Stable instructions"),
            Message(role="user", content="Hello"),
        ],
        extra={
            "models": ["anthropic/claude-sonnet-4-6", "anthropic/claude-haiku-4-5"],
            "cache_control": {"type": "ephemeral"},
        },
    )

    with patch("theriad.core.providers.openrouter.httpx.AsyncClient", return_value=client):
        await provider.complete(request)

    payload = client.post.await_args.kwargs["json"]
    assert "model" not in payload
    assert payload["models"] == [
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-haiku-4-5",
    ]
    assert payload["messages"][0]["content"] == [
        {
            "type": "text",
            "text": "Stable instructions",
            "cache_control": {"type": "ephemeral"},
        }
    ]


@pytest.mark.asyncio
async def test_openai_compatible_provider_maps_groq_payload_and_usage():
    response = _response(
        200,
        {
            "model": "openai/gpt-oss-20b",
            "choices": [{"message": {"content": "fast"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2},
        },
    )
    client = _client(response)
    provider = OpenAICompatibleProvider(
        name="groq",
        api_key="gsk_private",
        base_url="https://api.groq.com/openai/v1",
        default_extra={"service_tier": "auto"},
    )

    with patch("theriad.core.providers.openai_compatible.httpx.AsyncClient", return_value=client):
        result = await provider.complete(
            CompletionRequest(
                model="openai/gpt-oss-20b",
                messages=[Message(role="user", content="Hello")],
            )
        )

    assert client.post.await_args.args[0] == "https://api.groq.com/openai/v1/chat/completions"
    assert client.post.await_args.kwargs["json"]["service_tier"] == "auto"
    assert result.provider == "groq"
    assert result.prompt_tokens == 8


def test_factory_resolves_theriad_to_dynamic_openrouter_import(monkeypatch):
    import theriad.core.providers.openrouter as openrouter_module

    captured = {}

    class FakeOpenRouter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openrouter_module, "OpenRouterProvider", FakeOpenRouter)
    provider = build_provider("theriad", api_key="sk-or-private", timeout=4.0)

    assert isinstance(provider, FakeOpenRouter)
    assert captured == {"api_key": "sk-or-private", "timeout": 4.0}
    assert get_effective_provider_name("theriad") == "openrouter"


def test_factory_missing_key_error_is_safe(monkeypatch):
    import theriad.settings.secrets as secrets_module

    monkeypatch.setattr(secrets_module, "get_provider_api_key", lambda provider: None)
    with pytest.raises(ProviderConfigurationError) as exc_info:
        build_provider("anthropic")

    assert exc_info.value.provider == "anthropic"
    assert "not configured" in str(exc_info.value)


@pytest.mark.parametrize(
    ("provider", "tier", "expected"),
    [
        ("openai", TIER_FREE_GENERAL, "gpt-5.4-mini"),
        ("openai", TIER_CODE_STRONG, "gpt-5.6-terra"),
        ("openai", TIER_PREMIUM, "gpt-5.6-sol"),
        ("anthropic", TIER_CHEAP_BULK, "claude-haiku-4-5-20251001"),
        ("anthropic", TIER_STRONG_REASONING, "claude-sonnet-5"),
        ("anthropic", TIER_PREMIUM, "claude-opus-4-8"),
        ("groq", TIER_CHEAP_BULK, "openai/gpt-oss-20b"),
        ("groq", TIER_STRONG_REASONING, "openai/gpt-oss-120b"),
        ("deepinfra", TIER_CHEAP_BULK, "deepseek-ai/DeepSeek-V4-Flash"),
    ],
)
def test_direct_model_tier_mapping(provider, tier, expected):
    assert get_provider_model_for_tier(provider, tier) == expected
    assert get_provider_candidates(provider, tier, [("openrouter", "other/model")]) == [
        (provider, expected)
    ]


def test_theriad_preserves_routed_candidates():
    routed = [
        ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
        ("openrouter", "openai/gpt-4o-mini"),
    ]
    assert get_provider_candidates("theriad", TIER_CHEAP_BULK, routed) == routed


def test_direct_model_prefix_normalization_is_provider_specific():
    assert normalize_direct_model_id("openai", "openai/gpt-5.6-terra") == "gpt-5.6-terra"
    assert normalize_direct_model_id("anthropic", "anthropic/claude-sonnet-5") == "claude-sonnet-5"
    assert normalize_direct_model_id("openai", "anthropic/claude-sonnet-5") == (
        "anthropic/claude-sonnet-5"
    )
