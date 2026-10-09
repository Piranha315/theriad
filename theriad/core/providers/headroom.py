"""Default-on Headroom middleware for provider requests.

Headroom belongs at the provider boundary rather than in the model registry:
every newly registered model then inherits the same context policy.  The
middleware is deliberately fail-open.  A missing optional package, a model
tokenizer mismatch, or a compressor error must never turn a working provider
call into an application outage.

The deterministic compressor is the default (``THERIAD_HEADROOM_KOMPRESS_MODEL``
is ``disabled``) so a fresh deployment does not block its first request while
downloading an ML model.  Operators can set that variable to ``default`` to
enable Headroom's text compressor after explicitly warming/monitoring it.
"""

from __future__ import annotations

import asyncio
import copy
import os
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from theriad.core.logging.logger import get_logger

from .interfaces import CompletionRequest, CompletionResponse, Message, ProviderProtocol

logger = get_logger("providers.headroom")

_DEFAULT_MODEL_LIMIT = 200_000
_MIN_INPUT_CHARS = 1_000
_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", "disabled"})
_IMPORT_WARNING_EMITTED = False


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def _mode() -> str:
    """Return ``optimize``, ``audit``, or ``disabled`` from environment."""
    if not _env_flag("THERIAD_HEADROOM_ENABLED", default=True):
        return "disabled"
    value = os.environ.get("THERIAD_HEADROOM_MODE", "optimize").strip().lower()
    if value in {"audit", "observe"}:
        return "audit"
    if value in {"disabled", "off", "passthrough"}:
        return "disabled"
    return "optimize"


def _kompress_model() -> str | None:
    """Resolve the optional text-compressor model setting.

    ``disabled`` keeps Headroom's deterministic JSON/structured transforms
    without downloading an ML model.  ``default``/``auto`` delegates to the
    package default model.
    """
    value = os.environ.get("THERIAD_HEADROOM_KOMPRESS_MODEL", "disabled").strip()
    if not value or value.lower() in {"default", "auto"}:
        return None
    if value.lower() in {"disabled", "off", "none"}:
        return "disabled"
    return value


def _model_limit(model: str) -> int:
    """Use the catalogued model context limit when available."""
    try:
        from theriad.models.registry import DEFAULT_REGISTRY

        for tier in DEFAULT_REGISTRY.all_tiers():
            for candidate in DEFAULT_REGISTRY.candidates_for_tier(tier):
                if candidate.model == model and candidate.context_window:
                    return max(1, int(candidate.context_window))
    except Exception:
        # Context metadata is advisory; compression must remain available for
        # custom models and provider-direct slugs.
        pass
    return _DEFAULT_MODEL_LIMIT


def _contains_image(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in {
                "image_url",
                "input_image",
                "image",
            }:
                return True
    return False


def _message_dicts(request: CompletionRequest) -> list[dict[str, Any]]:
    """Copy provider messages so Headroom cannot mutate the caller's request."""
    return [
        {
            "role": message.role,
            "content": copy.deepcopy(message.content),
        }
        for message in request.messages
    ]


def _should_bypass(request: CompletionRequest, messages: list[dict[str, Any]]) -> str | None:
    if request.extra.get("_theriad_headroom_bypass"):
        return "request_override"
    if not messages:
        return "empty_messages"
    if isinstance(request.extra.get("response_format"), dict) and request.extra["response_format"].get("type") in {"json_object", "json_schema"}:
        return "structured_output_request"
    if _contains_image(messages):
        return "multimodal_request"
    text_size = sum(len(str(message.get("content", ""))) for message in messages)
    if text_size < _MIN_INPUT_CHARS:
        return "short_request"
    return None


def _compress_messages(
    messages: list[dict[str, Any]],
    *,
    model: str,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the synchronous Headroom pipeline off the async event loop."""
    global _IMPORT_WARNING_EMITTED

    try:
        from headroom import CompressConfig, compress
    except Exception as exc:  # pragma: no cover - exercised in dependency-light installs
        if not _IMPORT_WARNING_EMITTED:
            logger.warning(
                "Headroom unavailable; forwarding provider requests unchanged",
                extra={"error_type": type(exc).__name__},
            )
            _IMPORT_WARNING_EMITTED = True
        return messages, {
            "enabled": False,
            "mode": "disabled",
            "reason": "dependency_unavailable",
            "tokens_saved": 0,
        }

    config = CompressConfig(
        # Context prefixes are user-role messages in the provider contract so
        # they remain compatible with Anthropic/OpenAI/OpenRouter adapters.
        compress_user_messages=True,
        # Theriad's system prompt contains policy/tool instructions; preserve
        # it byte-for-byte for correctness and provider prefix caching.
        compress_system_messages=False,
        # The final user message is the current request and must remain exact.
        protect_recent=1,
        protect_analysis_context=True,
        min_tokens_to_compress=250,
        kompress_model=_kompress_model(),
    )
    result = compress(
        messages,
        model=model,
        model_limit=_model_limit(model),
        optimize=mode == "optimize",
        config=config,
    )
    stats: dict[str, Any] = {
        "enabled": mode != "disabled",
        "mode": mode,
        "tokens_before": int(getattr(result, "tokens_before", 0) or 0),
        "tokens_after": int(getattr(result, "tokens_after", 0) or 0),
        "tokens_saved": int(getattr(result, "tokens_saved", 0) or 0),
        "compression_ratio": float(getattr(result, "compression_ratio", 0.0) or 0.0),
        "transforms": list(getattr(result, "transforms_applied", []) or []),
    }
    return list(result.messages), stats


def _with_stats(response: CompletionResponse, stats: dict[str, Any]) -> CompletionResponse:
    """Attach non-sensitive compression diagnostics to the provider response."""
    if isinstance(response.raw, dict):
        response.raw = {**response.raw, "_theriad_headroom": stats}
    return response


async def _complete_with_headroom(
    provider_name: str,
    complete: Callable[[CompletionRequest], Awaitable[CompletionResponse]],
    request: CompletionRequest,
) -> CompletionResponse:
    """Compress a request, then invoke the underlying provider exactly once."""
    messages = _message_dicts(request)
    mode = _mode()
    bypass_reason = "disabled" if mode == "disabled" else _should_bypass(request, messages)
    if bypass_reason:
        stats = {
            "enabled": mode != "disabled",
            "mode": mode,
            "bypassed": True,
            "reason": bypass_reason,
            "tokens_saved": 0,
        }
        response = await complete(request)
        return _with_stats(response, stats)

    try:
        compressed_messages, stats = await asyncio.to_thread(
            _compress_messages,
            messages,
            model=request.model,
            mode=mode,
        )
        # Package-level protect_recent does not preserve nested JSON strings
        # under every transform. Enforce the active instruction boundary here.
        if len(compressed_messages) != len(messages) or any(
            a.get("role") != b.get("role") for a, b in zip(messages, compressed_messages)
        ):
            compressed_messages = copy.deepcopy(messages)
            stats.update(tokens_saved=0, protection_restored=True)
        else:
            last_user = next((i for i in range(len(messages)-1, -1, -1)
                              if messages[i].get("role") == "user"), None)
            restored = False
            for i, original in enumerate(messages):
                if i == last_user or original.get("role") in {"system", "developer"}:
                    restored |= compressed_messages[i] != original
                    compressed_messages[i] = copy.deepcopy(original)
            if restored:
                # Do not report pre-restoration token savings as actual savings.
                stats.update(tokens_saved=0, protection_restored=True)
        stats["provider"] = provider_name
        stats["model"] = request.model
        if stats.get("tokens_saved", 0):
            logger.info("Headroom compressed provider context", extra=stats)
        forwarded = replace(
            request,
            messages=[
                replace(original, content=message.get("content", ""))
                for original, message in zip(request.messages, compressed_messages)
            ],
        )
    except Exception as exc:
        # The wrapper is an optimization layer, never a reliability gate. Do
        # not catch exceptions from the actual provider call below: provider
        # retry/fallback policy must see those errors unchanged.
        logger.warning(
            "Headroom middleware failed; forwarding original provider request",
            extra={"provider": provider_name, "model": request.model, "error_type": type(exc).__name__},
        )
        response = await complete(request)
        return _with_stats(
            response,
            {
                "enabled": True,
                "mode": mode,
                "bypassed": True,
                "reason": "middleware_failure",
                "tokens_saved": 0,
            },
        )

    response = await complete(forwarded)
    return _with_stats(response, stats)


class HeadroomProvider(ProviderProtocol):
    """Provider decorator that applies Headroom before each completion."""

    def __init__(self, provider: ProviderProtocol):
        self._provider = provider

    @property
    def name(self) -> str:
        return self._provider.name

    async def health_check(self) -> bool:
        return await self._provider.health_check()

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return await _complete_with_headroom(
            self.name,
            self._provider.complete,
            request,
        )

    @property
    def supports_tool_calls(self) -> bool:
        return bool(getattr(self._provider, "supports_tool_calls", False))

    @property
    def supports_agent_tools(self) -> bool:
        return bool(getattr(self._provider, "supports_agent_tools", False))

    async def complete_with_tools(self, request: Any) -> Any:
        """Forward native structured requests without text compression."""
        method = getattr(self._provider, "complete_with_tools", None)
        if not callable(method):
            raise AttributeError("wrapped provider has no structured tool interface")
        return await method(request)

    def __getattr__(self, name: str) -> Any:
        # Preserve provider-specific capability metadata on the explicit
        # decorator used by dependency-injection tests and local harnesses.
        return getattr(self._provider, name)


async def _decorated_complete(
    provider: ProviderProtocol,
    original_complete: Callable[[CompletionRequest], Awaitable[CompletionResponse]],
    request: CompletionRequest,
) -> CompletionResponse:
    return await _complete_with_headroom(
        str(getattr(provider, "name", "unknown")),
        original_complete,
        request,
    )


def with_headroom(provider: ProviderProtocol) -> ProviderProtocol:
    """Decorate a provider exactly once while preserving its concrete type.

    Preserving the original instance matters to callers that use provider
    subclasses for diagnostics or dependency injection.
    """
    if isinstance(provider, HeadroomProvider) or getattr(provider, "_theriad_headroom_wrapped", False):
        return provider
    original_complete = getattr(provider, "complete", None)
    if not callable(original_complete):
        return provider

    async def decorated(request: CompletionRequest) -> CompletionResponse:
        return await _decorated_complete(provider, original_complete, request)

    provider.complete = decorated  # type: ignore[method-assign]
    original_tools = getattr(provider, "complete_with_tools", None)
    if callable(original_tools):
        async def decorated_tools(request: Any) -> Any:
            # Tool schemas/arguments are part of the graded observation.  Do
            # not pass them through a text compressor that may reorder or
            # rewrite structured messages.
            return await original_tools(request)

        provider.complete_with_tools = decorated_tools  # type: ignore[attr-defined]
    provider._theriad_headroom_wrapped = True  # type: ignore[attr-defined]
    return provider
