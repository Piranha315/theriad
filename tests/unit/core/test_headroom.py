"""Tests for the default-on Headroom provider decorator."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from theriad.core.providers.headroom import HeadroomProvider
from theriad.core.providers.interfaces import (
    CompletionRequest,
    CompletionResponse,
    Message,
    ProviderProtocol,
)


class _FakeProvider(ProviderProtocol):
    def __init__(self):
        self.requests: list[CompletionRequest] = []
        self.complete_mock = AsyncMock(side_effect=self._complete)

    @property
    def name(self) -> str:
        return "openrouter"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return await self.complete_mock(request)

    async def _complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        return CompletionResponse(
            model=request.model,
            content="ok",
            prompt_tokens=10,
            completion_tokens=2,
            latency_ms=1.0,
            provider=self.name,
            raw={},
        )

    async def health_check(self) -> bool:
        return True


def _long_json_context() -> str:
    return "\n".join(
        f'{{"id": {index}, "status": "complete", "detail": "repeated tool result"}}'
        for index in range(80)
    )


@pytest.mark.asyncio
async def test_headroom_compresses_context_but_preserves_latest_turn(monkeypatch):
    monkeypatch.setenv("THERIAD_HEADROOM_ENABLED", "true")
    monkeypatch.setenv("THERIAD_HEADROOM_MODE", "optimize")
    monkeypatch.setenv("THERIAD_HEADROOM_KOMPRESS_MODEL", "disabled")
    provider = _FakeProvider()
    wrapped = HeadroomProvider(provider)

    response = await wrapped.complete(
        CompletionRequest(
            model="openai/gpt-4o-mini",
            messages=[
                Message(role="system", content="Keep the system policy exact."),
                Message(role="user", content=_long_json_context()),
                Message(role="user", content="What is the important pattern?"),
            ],
        )
    )

    sent = provider.requests[0]
    assert sent.messages[0].content == "Keep the system policy exact."
    assert sent.messages[-1].content == "What is the important pattern?"
    assert sent.messages[1].content != _long_json_context()
    assert response.raw["_theriad_headroom"]["tokens_saved"] > 0


@pytest.mark.asyncio
async def test_headroom_fails_open_to_original_request(monkeypatch):
    monkeypatch.setenv("THERIAD_HEADROOM_ENABLED", "true")
    provider = _FakeProvider()
    wrapped = HeadroomProvider(provider)

    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=[
            Message(role="user", content=_long_json_context()),
            Message(role="user", content="Answer this."),
        ],
    )
    # asyncio.to_thread invokes the patched callable synchronously in the
    # worker; a regular raising function is the failure path we want to test.
    def raise_now(*args, **kwargs):
        raise RuntimeError("compressor unavailable")

    monkeypatch.setattr("theriad.core.providers.headroom._compress_messages", raise_now)
    response = await wrapped.complete(request)

    assert provider.requests[0] is request
    assert response.raw["_theriad_headroom"]["reason"] == "middleware_failure"


@pytest.mark.asyncio
async def test_headroom_bypasses_multimodal_requests(monkeypatch):
    monkeypatch.setenv("THERIAD_HEADROOM_ENABLED", "true")
    provider = _FakeProvider()
    wrapped = HeadroomProvider(provider)
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=[
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                ],
            )
        ],
    )

    response = await wrapped.complete(request)

    assert provider.requests[0] is request
    assert response.raw["_theriad_headroom"]["reason"] == "multimodal_request"
