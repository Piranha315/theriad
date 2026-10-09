"""Direct Anthropic Messages API provider.

The Messages API represents system instructions separately from user/assistant
turns, so this adapter performs that translation at the provider boundary.

API reference: https://platform.claude.com/docs/en/api/messages/create
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from .interfaces import CompletionRequest, CompletionResponse, ProviderError, ProviderProtocol

_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
_DEFAULT_API_VERSION = "2023-06-01"
# Anthropic requires an explicit max_tokens value. This is a provider-boundary
# fallback used only when a caller intentionally omits Theriad's application
# cap; user-facing synthesis requests otherwise omit max_tokens entirely.
_DEFAULT_MAX_TOKENS = 8192
_ALLOWED_EXTRA_FIELDS = frozenset(
    {
        "metadata",
        "stop_sequences",
        "thinking",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
    }
)


class AnthropicProvider(ProviderProtocol):
    """Execute text completions directly through Anthropic's Messages API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 30.0,
        api_version: str = _DEFAULT_API_VERSION,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._api_version = api_version

    @property
    def name(self) -> str:
        return "anthropic"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": self._api_version,
            "Content-Type": "application/json",
        }

    def _payload(self, request: CompletionRequest) -> dict[str, Any]:
        system_parts: list[str] = []
        messages: list[dict[str, str]] = []
        for message in request.messages:
            if message.role == "system":
                system_parts.append(message.text)
            else:
                messages.append({"role": message.role, "content": message.content})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": (
                request.max_tokens if request.max_tokens is not None else _DEFAULT_MAX_TOKENS
            ),
            "temperature": request.temperature,
        }
        if system_parts:
            system_text = "\n\n".join(system_parts)
            if "cache_control" in request.extra:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": request.extra["cache_control"],
                    }
                ]
            else:
                payload["system"] = system_text
        for key in _ALLOWED_EXTRA_FIELDS:
            if key in request.extra:
                payload[key] = request.extra[key]
        return payload

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        started_at = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/messages",
                    headers=self._headers(),
                    json=self._payload(request),
                )
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, f"timeout after {self._timeout}s") from exc
        except httpx.RequestError as exc:
            raise ProviderError(self.name, "network request failed") from exc

        latency_ms = (time.monotonic() - started_at) * 1000
        if response.status_code < 200 or response.status_code >= 300:
            raise ProviderError(
                self.name,
                f"API request failed with status {response.status_code}",
                status_code=response.status_code,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(self.name, "invalid JSON response") from exc
        if not isinstance(data, dict):
            raise ProviderError(self.name, "unexpected response shape")

        raw_usage = data.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        raw_model = data.get("model")
        model = raw_model if isinstance(raw_model, str) else request.model
        return CompletionResponse(
            model=model,
            content=self._extract_text(data),
            prompt_tokens=_safe_int(usage.get("input_tokens")),
            completion_tokens=_safe_int(usage.get("output_tokens")),
            latency_ms=latency_ms,
            provider=self.name,
            raw=data,
            finish_reason=(
                data.get("stop_reason") if isinstance(data.get("stop_reason"), str) else None
            ),
        )

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Flatten visible text blocks while excluding thinking metadata."""
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise ProviderError("anthropic", "unexpected response shape")
        parts: list[str] = []
        for block in blocks:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
        return "".join(parts)

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=min(self._timeout, 5.0)) as client:
                response = await client.get(
                    f"{self._base_url}/models",
                    headers=self._headers(),
                )
            return 200 <= response.status_code < 300
        except Exception:
            # Health checks are diagnostic and must never break request handling.
            return False


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return 0
