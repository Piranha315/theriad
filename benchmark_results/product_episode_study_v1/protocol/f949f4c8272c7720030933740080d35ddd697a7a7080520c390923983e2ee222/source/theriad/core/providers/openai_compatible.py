"""Reusable adapter for OpenAI-compatible chat-completions providers."""

from __future__ import annotations

import time
from typing import Any

import httpx

from .interfaces import CompletionRequest, CompletionResponse, ProviderError, ProviderProtocol


class OpenAICompatibleProvider(ProviderProtocol):
    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        default_extra: dict[str, Any] | None = None,
    ):
        self._name = name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._default_extra = dict(default_extra or {})

    @property
    def name(self) -> str:
        return self._name

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "temperature": request.temperature,
            **self._default_extra,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        for field in (
            "tools",
            "tool_choice",
            "response_format",
            "reasoning_effort",
            "service_tier",
        ):
            if field in request.extra:
                payload[field] = request.extra[field]

        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, f"timeout after {self._timeout}s") from exc
        except httpx.RequestError as exc:
            raise ProviderError(self.name, "network request failed") from exc

        latency_ms = (time.monotonic() - started) * 1000
        if not 200 <= response.status_code < 300:
            raise ProviderError(
                self.name,
                f"API request failed with status {response.status_code}",
                status_code=response.status_code,
            )
        try:
            data = response.json()
            choice = data["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(self.name, "unexpected response shape") from exc
        usage = data.get("usage") or {}
        message = choice.get("message") or {}
        content = message.get("content") or choice.get("text") or ""
        return CompletionResponse(
            model=str(data.get("model") or request.model),
            content=str(content),
            prompt_tokens=max(0, int(usage.get("prompt_tokens") or 0)),
            completion_tokens=max(0, int(usage.get("completion_tokens") or 0)),
            latency_ms=latency_ms,
            provider=self.name,
            raw=data,
            finish_reason=choice.get("finish_reason"),
            reasoning=message.get("reasoning_content") or message.get("reasoning"),
        )

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=min(self._timeout, 5.0)) as client:
                response = await client.get(f"{self._base_url}/models", headers=self._headers())
            return 200 <= response.status_code < 300
        except Exception:
            return False
