from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from theriad.core.providers.headroom import with_headroom
from theriad.core.providers.interfaces import CompletionRequest, Message, ProviderError
from theriad.core.providers.openrouter import OpenRouterProvider
from theriad.core.providers.readiness import with_provider_readiness
from theriad.routing.study_adapters import StructuredCompletionRequest


def _request() -> StructuredCompletionRequest:
    return StructuredCompletionRequest(
        model="fixture/model",
        messages=(Message(role="user", content="Use both tools."),),
        tool_schemas=(
            {"name": "first", "description": "first tool", "parameters": {"type": "object"}},
            {"name": "second", "description": "second tool", "parameters": {"type": "object"}},
        ),
        max_tokens=128,
        metadata={"tool_choice": "required", "parallel_tool_calls": True},
    )


def _transport(seen: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "fixture-response",
                "model": "fixture/model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "first", "arguments": "{}"},
                                },
                                {
                                    "id": "call-2",
                                    "type": "function",
                                    "function": {"name": "second", "arguments": "{\"x\": 2}"},
                                },
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "cost": 0.00125},
            },
        )

    return httpx.MockTransport(handler)


def test_openrouter_native_tools_preserve_schema_choice_order_parallel_and_usage():
    seen: list[dict] = []
    provider = OpenRouterProvider(
        api_key="fixture",
        base_url="https://fixture.invalid/v1",
        transport=_transport(seen),
    )
    response = asyncio.run(provider.complete_with_tools(_request()))

    assert provider.supports_tool_calls is True
    assert provider.supports_agent_tools is True
    assert [call.name for call in response.tool_calls] == ["first", "second"]
    assert [call.order for call in response.tool_calls] == [0, 1]
    assert {call.parallel_group for call in response.tool_calls} == {"response-0"}
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 8
    assert response.cost_usd == 0.00125
    assert seen[0]["tool_choice"] == "required"
    assert seen[0]["parallel_tool_calls"] is True
    assert [tool["function"]["name"] for tool in seen[0]["tools"]] == ["first", "second"]


def test_openrouter_native_length_preserves_billed_truncated_tool_response():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "truncated-response",
                "model": "fixture/model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "native_finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "first",
                                        "arguments": '{"incomplete":',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 128,
                    "cost": 0.0015,
                },
            },
        )

    provider = OpenRouterProvider(
        api_key="fixture",
        base_url="https://fixture.invalid/v1",
        transport=httpx.MockTransport(handler),
    )
    response = asyncio.run(provider.complete_with_tools(_request()))

    assert response.finish_reason == "length"
    assert response.tool_calls == ()
    assert response.cost_usd == 0.0015
    assert response.completion_tokens == 128


def test_headroom_and_readiness_preserve_structured_capability_without_rewriting_tools():
    seen: list[dict] = []
    provider = OpenRouterProvider(
        api_key="fixture",
        base_url="https://fixture.invalid/v1",
        transport=_transport(seen),
    )
    wrapped = with_provider_readiness(with_headroom(provider))

    assert wrapped.supports_tool_calls is True
    assert wrapped.supports_agent_tools is True
    response = asyncio.run(wrapped.complete_with_tools(_request()))
    assert [call.name for call in response.tool_calls] == ["first", "second"]
    assert seen[0]["messages"] == [{"role": "user", "content": "Use both tools."}]


@pytest.mark.asyncio
async def test_openrouter_normalizes_http_200_embedded_upstream_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": {"message": "upstream overloaded", "code": 502}},
        )

    provider = OpenRouterProvider(
        api_key="fixture",
        base_url="https://fixture.invalid/v1",
        transport=httpx.MockTransport(handler),
    )
    request = CompletionRequest(
        model="fixture/model",
        messages=[Message(role="user", content="hello")],
    )

    with pytest.raises(ProviderError, match="upstream overloaded") as exc_info:
        await provider.complete(request)

    assert exc_info.value.status_code == 502
