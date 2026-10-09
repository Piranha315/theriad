"""
OpenRouter provider implementation.

OpenRouter exposes an OpenAI-compatible /chat/completions endpoint that routes
requests to many underlying models. This makes it ideal for Theriad's cost
optimization — we pick the cheapest suitable model, OpenRouter handles delivery.

Decision: use httpx (async) rather than the openai SDK so we control the
request lifecycle (timeout, headers) directly without SDK version drift.

API reference: https://openrouter.ai/docs
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import httpx

from theriad.core.logging.logger import get_logger

from .interfaces import CompletionRequest, CompletionResponse, ProviderError, ProviderProtocol

logger = get_logger("providers.openrouter")

_BASE_URL = "https://openrouter.ai/api/v1"
_HEALTH_URL = "https://openrouter.ai/api/v1/models"
_EXPLICIT_CACHE_MODELS = frozenset(
    {
        "qwen/qwen3-max",
        "qwen/qwen-plus",
        "qwen/qwen3.6-plus",
        "qwen/qwen3-coder-plus",
        "qwen/qwen3-coder-flash",
        "deepseek/deepseek-v3.2",
    }
)


def _supports_explicit_cache(model: str) -> bool:
    return model.startswith("anthropic/") or model in _EXPLICIT_CACHE_MODELS


def _raise_embedded_provider_error(data: Any) -> None:
    """Normalize OpenRouter's occasional HTTP-200 error envelope."""

    if not isinstance(data, Mapping) or not isinstance(data.get("error"), Mapping):
        return
    error = data["error"]
    message = str(error.get("message") or "provider returned an embedded error")
    raw_code = error.get("code")
    try:
        status_code = int(raw_code) if raw_code is not None else None
    except (TypeError, ValueError):
        status_code = None
    raise ProviderError("openrouter", message, status_code=status_code, request_id=data.get("id") if isinstance(data.get("id"), str) else None)


class OpenRouterProvider(ProviderProtocol):
    """
    OpenRouter inference provider.

    Requires OPENROUTER_API_KEY to be set in config or environment.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _BASE_URL,
        timeout: float = 30.0,
        http_referer: str = "https://github.com/theriad",
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._http_referer = http_referer
        # A transport injection is strictly for provider-free HTTP fixtures
        # and local contract tests.  Production construction leaves it None.
        self._transport = transport

    supports_tool_calls = True
    # The provider exposes a native, repeatable structured-tool interface.
    # Stateful agent loops (such as tau2) are owned by the routing adapter;
    # each turn is sent through ``complete_with_tools`` and the adapter feeds
    # trusted tool results back into the next request.
    supports_agent_tools = True

    @property
    def name(self) -> str:
        return "openrouter"

    def _headers(self, session_id: str | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": self._http_referer,
            "X-Title": "Theriad",
            "Content-Type": "application/json",
        }
        # Pass conversation session ID so OpenRouter can route follow-up messages
        # to the same model/provider (relevant when using openrouter/auto).
        if session_id:
            headers["X-Session-Id"] = session_id
        return headers

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        messages: list[dict] = [
            {"role": message.role, "content": message.content} for message in request.messages
        ]
        cache_control = request.extra.get("cache_control")
        candidate_models = request.extra.get("models")
        cache_models = (
            candidate_models
            if isinstance(candidate_models, list) and candidate_models
            else [request.model]
        )
        if cache_control and all(_supports_explicit_cache(model) for model in cache_models):
            # OpenRouter's explicit cache breakpoint belongs on a message
            # content block. Other providers cache automatically; sending an
            # unsupported explicit parameter would shrink the endpoint pool.
            cache_index = next(
                (index for index, message in enumerate(messages) if message["role"] == "system"),
                None,
            )
            if cache_index is not None:
                text = messages[cache_index]["content"]
                # Only wrap a plain string. A system message that is already a
                # block list has been built deliberately and must not be flattened.
                if isinstance(text, str):
                    messages[cache_index]["content"] = [
                        {"type": "text", "text": text, "cache_control": cache_control}
                    ]
        payload = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.extra:
            for key in (
                "response_format",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "provider",
                "reasoning",
                "include_reasoning",
                "seed",
                "stop",
                "tools",
                "tool_choice",
                "models",
            ):
                if key in request.extra:
                    payload[key] = request.extra[key]
        if payload.get("models"):
            payload.pop("model", None)

        t0 = time.monotonic()
        try:
            client_kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
            async with httpx.AsyncClient(**client_kwargs) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(session_id=request.session_id),
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, f"timeout after {self._timeout}s") from exc
        except httpx.RequestError as exc:
            raise ProviderError(self.name, f"request error: {exc}") from exc

        latency_ms = (time.monotonic() - t0) * 1000

        if resp.status_code != 200:
            raise ProviderError(self.name, resp.text, status_code=resp.status_code, request_id=resp.headers.get("x-request-id"))

        data = resp.json()
        _raise_embedded_provider_error(data)
        try:
            choice = data["choices"][0]
            usage = data.get("usage", {})
            finish_reason: str | None = choice.get("finish_reason")

            content = self._extract_content(choice, data)
            reasoning = self._extract_reasoning(choice)

            return CompletionResponse(
                model=data.get("model", request.model),
                content=content,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                latency_ms=latency_ms,
                provider=self.name,
                raw=data,
                finish_reason=finish_reason,
                reasoning=reasoning,
                cost_usd=self._reported_cost(data),
            )
        except (KeyError, IndexError) as exc:
            raise ProviderError(self.name, f"unexpected response shape: {data}") from exc

    async def complete_with_tools(self, request: Any) -> Any:
        """Execute one native OpenAI-compatible structured-tool request.

        ``StructuredCompletionRequest`` lives in the routing layer, so this
        provider intentionally accepts its small duck-typed contract here.
        The response is normalized before it crosses back into the adapter;
        no prose-to-tool coercion is performed.
        """
        from theriad.routing.study_adapters import (  # local import avoids a cycle
            StructuredCompletionResponse,
            StructuredToolCall,
        )

        messages: list[dict[str, Any]] = []
        for message in request.messages:
            value: dict[str, Any] = {
                "role": message.role,
                "content": message.content,
            }
            if getattr(message, "tool_call_id", None):
                value["tool_call_id"] = message.tool_call_id
            if getattr(message, "tool_calls", None):
                value["tool_calls"] = message.tool_calls
            messages.append(value)
        tools: list[dict[str, Any]] = []
        for schema in request.tool_schemas:
            value = dict(schema)
            if value.get("type") == "function" and isinstance(value.get("function"), Mapping):
                tools.append(value)
            else:
                tools.append({"type": "function", "function": value})
        metadata = dict(getattr(request, "metadata", {}) or {})
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "tools": tools,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        for key in (
            "tool_choice",
            "parallel_tool_calls",
            "provider",
            "seed",
            "reasoning",
            "models",
        ):
            if key in metadata:
                payload[key] = metadata[key]
        if payload.get("models"):
            payload.pop("model", None)
        started = time.monotonic()
        try:
            client_kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(session_id=getattr(request, "session_id", None)),
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, f"timeout after {self._timeout}s") from exc
        except httpx.RequestError as exc:
            raise ProviderError(self.name, f"request error: {exc}") from exc
        latency_ms = (time.monotonic() - started) * 1000.0
        if response.status_code != 200:
            raise ProviderError(self.name, response.text, status_code=response.status_code, request_id=response.headers.get("x-request-id"))
        data = response.json()
        _raise_embedded_provider_error(data)
        try:
            choice = data["choices"][0]
            message = choice.get("message", {})
            raw_calls = message.get("tool_calls") or []
            # OpenRouter can report the OpenAI-compatible finish reason as
            # ``tool_calls`` while the provider-native reason is ``length``.
            # In that case the final tool-call arguments may be truncated JSON.
            # Preserve the billed response as a normal truncated completion so
            # the study collector can settle it and mark it unusable, instead
            # of raising after a successful (and billable) provider response.
            native_finish_reason = choice.get("native_finish_reason")
            finish_reason = native_finish_reason or choice.get("finish_reason")
            truncated = finish_reason in {"length", "max_tokens", "MAX_TOKENS"}
            calls: list[StructuredToolCall] = []
            for order, raw_call in enumerate(() if truncated else raw_calls):
                if not isinstance(raw_call, Mapping):
                    raise ValueError("OpenRouter tool_calls entries must be objects")
                function = raw_call.get("function")
                function = function if isinstance(function, Mapping) else raw_call
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    import json

                    arguments = json.loads(arguments)
                if not isinstance(arguments, Mapping):
                    raise ValueError("OpenRouter tool-call arguments must be an object")
                parallel_group = str(raw_call.get("parallel_group") or "")
                if not parallel_group and len(raw_calls) > 1 and metadata.get("parallel_tool_calls"):
                    parallel_group = "response-0"
                calls.append(
                    StructuredToolCall(
                        name=str(function.get("name") or ""),
                        arguments=dict(arguments),
                        order=order,
                        call_id=str(raw_call.get("id") or ""),
                        parallel_group=parallel_group,
                    )
                )
            usage = data.get("usage", {}) or {}
            return StructuredCompletionResponse(
                model=str(data.get("model") or request.model),
                content=self._extract_content(choice, data),
                tool_calls=tuple(calls),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                latency_ms=latency_ms,
                provider=self.name,
                raw=data,
                finish_reason=finish_reason,
                cost_usd=self._reported_cost(data),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(self.name, f"unexpected structured response shape: {data}") from exc

    @staticmethod
    def _reported_cost(data: Mapping[str, Any]) -> float | None:
        usage = data.get("usage")
        candidates = (
            data.get("cost"),
            usage.get("cost") if isinstance(usage, Mapping) else None,
            usage.get("total_cost") if isinstance(usage, Mapping) else None,
        )
        for candidate in candidates:
            try:
                if candidate is not None:
                    value = float(candidate)
                    if value >= 0:
                        return value
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_content(choice: dict, full_response: dict) -> str:
        """
        Robustly extract text from an OpenRouter/OpenAI-compatible choice.

        Handles:
        - choices[0].message.content as a plain string  (standard)
        - choices[0].message.content as a content array (multi-modal / structured)
        - choices[0].text                               (legacy completions API)
        - choices[0].message.reasoning only             (kept separate)
        - None content with output tokens               (logged as warning)
        """
        message = choice.get("message", {})

        # Primary: message.content
        content = message.get("content")

        if isinstance(content, list):
            # OpenAI structured content array: [{"type": "text", "text": "..."}, ...]
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text") or item.get("content") or "")
                else:
                    parts.append(str(item))
            content = "".join(parts)

        elif content is None:
            # Reasoning is metadata, not answer content. Keep it separate so a
            # reasoning trace cannot cross a user-facing answer boundary.
            content = choice.get("text")

        if not isinstance(content, str):
            content = str(content) if content is not None else ""

        if not content.strip():
            # Preserve the legacy adapter behavior for response shapes that
            # omit finish_reason entirely. A completed response with a null
            # content field keeps reasoning separate; callers can use the
            # structured reasoning field instead of displaying it as the
            # answer.
            reasoning_fallback = (
                message.get("reasoning") or message.get("reasoning_content") or ""
            )
            if choice.get("finish_reason") is None and isinstance(reasoning_fallback, str):
                if reasoning_fallback.strip():
                    return reasoning_fallback.strip()

            logger.warning(
                "extracted content is empty",
                extra={
                    "model": full_response.get("model", "unknown"),
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": full_response.get("usage", {}).get("completion_tokens", 0),
                    "has_message_content": "content" in message,
                    "has_reasoning": bool(
                        message.get("reasoning") or message.get("reasoning_content")
                    ),
                    "has_text": "text" in choice,
                },
            )

        return content

    @staticmethod
    def _extract_reasoning(choice: dict) -> str | None:
        """Extract provider reasoning without promoting it to answer text."""
        message = choice.get("message", {})
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        return reasoning.strip() if isinstance(reasoning, str) and reasoning.strip() else None

    async def health_check(self) -> bool:
        try:
            client_kwargs: dict[str, Any] = {"timeout": 5.0}
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
            async with httpx.AsyncClient(**client_kwargs) as client:
                resp = await client.get(_HEALTH_URL, headers=self._headers())
            return resp.status_code == 200
        except Exception:
            return False
