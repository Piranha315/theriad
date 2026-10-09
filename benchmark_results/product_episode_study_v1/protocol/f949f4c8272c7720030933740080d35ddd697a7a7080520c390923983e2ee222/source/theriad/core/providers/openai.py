"""Direct OpenAI Responses API provider.

This adapter deliberately uses ``httpx`` instead of an SDK so the provider
contract remains small, async, and consistent with the existing OpenRouter and
Ollama adapters.

API reference: https://developers.openai.com/api/reference/resources/responses
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from .interfaces import CompletionRequest, CompletionResponse, ProviderError, ProviderProtocol

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_ALLOWED_EXTRA_FIELDS = frozenset(
    {
        "metadata",
        "parallel_tool_calls",
        "reasoning",
        "text",
        "tool_choice",
        "tools",
        "top_p",
        "truncation",
    }
)


class OpenAIProvider(ProviderProtocol):
    """Execute text completions directly through OpenAI's Responses API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "openai"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, request: CompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "input": [{"role": message.role, "content": message.content} for message in request.messages],
            # Responses are self-contained in Theriad; do not create provider-side
            # application state unless a future feature explicitly opts into it.
            "store": False,
        }
        # Current reasoning-family models may reject sampling parameters. Keep
        # the default request free of them, while retaining an explicit opt-in
        # for callers targeting a model that supports temperature.
        if request.temperature != 0.0:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_output_tokens"] = request.max_tokens
        for key in _ALLOWED_EXTRA_FIELDS:
            if key in request.extra:
                payload[key] = request.extra[key]
        return payload

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        started_at = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/responses",
                    headers=self._headers(),
                    json=self._payload(request),
                )
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, f"timeout after {self._timeout}s") from exc
        except httpx.RequestError as exc:
            # Avoid copying request/header details from the upstream exception.
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

        content = self._extract_output_text(data)
        raw_usage = data.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        raw_model = data.get("model")
        model = raw_model if isinstance(raw_model, str) else request.model
        return CompletionResponse(
            model=model,
            content=content,
            prompt_tokens=_safe_int(usage.get("input_tokens")),
            completion_tokens=_safe_int(usage.get("output_tokens")),
            latency_ms=latency_ms,
            provider=self.name,
            raw=data,
            finish_reason=self._finish_reason(data),
        )

    @staticmethod
    def _extract_output_text(data: dict[str, Any]) -> str:
        """Flatten user-facing output text while excluding reasoning blocks."""
        top_level_text = data.get("output_text")
        if isinstance(top_level_text, str):
            return top_level_text

        parts: list[str] = []
        output = data.get("output")
        if not isinstance(output, list):
            raise ProviderError("openai", "unexpected response shape")

        for item in output:
            if not isinstance(item, dict):
                continue
            blocks = item.get("content")
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "output_text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block_type == "refusal" and isinstance(block.get("refusal"), str):
                    parts.append(block["refusal"])
        return "".join(parts)

    @staticmethod
    def _finish_reason(data: dict[str, Any]) -> str | None:
        status = data.get("status")
        if status == "completed":
            return "stop"
        if status == "incomplete":
            details = data.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, dict) else None
            if reason == "max_output_tokens":
                return "length"
            return reason if isinstance(reason, str) else "incomplete"
        return status if isinstance(status, str) else None

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
    """Convert provider usage fields to non-negative integers."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return 0
