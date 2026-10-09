"""
Provider-agnostic LLM completion interfaces.

Decision: Use dataclasses (not Pydantic) for the request/response types that
flow at inference time to keep hot-path overhead minimal.  Config types remain
Pydantic because they are only parsed once at startup.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A single chat message.

    ``content`` is normally a plain string. For multimodal requests it may be a
    list of OpenAI-style content blocks — ``{"type": "text", ...}`` and
    ``{"type": "image_url", "image_url": {"url": ...}}`` — which providers pass
    through untouched. Keeping both shapes on one field avoids rewriting every
    text-only call site, which is the overwhelming majority.
    """
    role: str    # "system" | "user" | "assistant"
    content: str | list[dict[str, Any]]
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The textual part of this message, ignoring any non-text blocks.

        Routing, token estimation, and logging all want a string; none of them
        should have to know whether an image happens to be attached.
        """
        if isinstance(self.content, str):
            return self.content
        return "\n".join(
            str(block.get("text", ""))
            for block in self.content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    @property
    def has_images(self) -> bool:
        if isinstance(self.content, str):
            return False
        return any(
            isinstance(block, dict) and block.get("type") == "image_url"
            for block in self.content
        )


@dataclass
class CompletionRequest:
    model: str
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    # Optional session identifier passed to the provider for conversation stickiness.
    # OpenRouter uses this as X-Session-Id so its auto-router or provider selection
    # can stick to the same underlying model across turns in a conversation.
    session_id: str | None = None


@dataclass
class CompletionResponse:
    model: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    provider: str
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    # Provider reasoning is kept separate from user-facing answer content.
    reasoning: str | None = None
    # Some OpenAI-compatible gateways return an exact usage charge.  Keep it
    # optional so existing providers remain source-compatible and accounting
    # can prefer the provider's reported value when it is present.
    cost_usd: float | None = None
    """
    Provider-reported reason the generation stopped.
    Common values: "stop" (natural end), "length" (hit max_tokens cap),
    "content_filter", "error". None = not reported or unknown.
    """


class ProviderProtocol(ABC):
    """Abstract base for all LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'openrouter', 'ollama')."""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Execute a completion request.

        Must raise ProviderError on non-retryable failures.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable."""


class ProviderError(Exception):
    """Raised when a provider call fails in a non-retryable way."""

    def __init__(self, provider: str, message: str, status_code: int | None = None, *, request_id: str | None = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.status_code = status_code
        self.request_id = request_id
