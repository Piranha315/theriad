"""Provider construction from backend-managed credentials.

Imports remain inside ``build_provider`` so existing tests and callers can
monkeypatch concrete provider modules without import-time instances escaping
their patches.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from .interfaces import ProviderProtocol

ProviderBuilder = Callable[..., ProviderProtocol]


@dataclass(frozen=True)
class ProviderAdapterSpec:
    """Registration contract for a direct inference provider adapter."""

    name: str
    builder: ProviderBuilder
    env_key: str
    aliases: tuple[str, ...] = ()


_PROVIDER_ADAPTERS: dict[str, ProviderAdapterSpec] = {}
_PROVIDER_ALIASES: dict[str, str] = {"theriad": "openrouter"}


def register_provider_adapter(
    name: str,
    builder: ProviderBuilder,
    *,
    env_key: str | None = None,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> None:
    """Register a provider without editing this factory's dispatch logic."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Provider name cannot be empty")
    if normalized in _PROVIDER_ADAPTERS and not replace:
        raise ValueError(f"Provider adapter already registered: {normalized}")
    spec = ProviderAdapterSpec(
        name=normalized,
        builder=builder,
        env_key=env_key or f"{normalized.upper().replace('-', '_')}_API_KEY",
        aliases=tuple(alias.strip().lower() for alias in aliases),
    )
    _PROVIDER_ADAPTERS[normalized] = spec
    for alias in spec.aliases:
        _PROVIDER_ALIASES[alias] = normalized


def registered_provider_adapters() -> tuple[ProviderAdapterSpec, ...]:
    return tuple(_PROVIDER_ADAPTERS.values())


def _build_openrouter(**kwargs) -> ProviderProtocol:
    from .openrouter import OpenRouterProvider

    return OpenRouterProvider(**kwargs)


def _build_openai(**kwargs) -> ProviderProtocol:
    from .openai import OpenAIProvider

    return OpenAIProvider(**kwargs)


def _build_anthropic(**kwargs) -> ProviderProtocol:
    from .anthropic import AnthropicProvider

    return AnthropicProvider(**kwargs)


def _build_groq(**kwargs) -> ProviderProtocol:
    from .openai_compatible import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        default_extra={"service_tier": "auto"},
        **kwargs,
    )


def _build_deepinfra(**kwargs) -> ProviderProtocol:
    from .openai_compatible import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        name="deepinfra",
        base_url="https://api.deepinfra.com/v1/openai",
        **kwargs,
    )


register_provider_adapter("openrouter", _build_openrouter, env_key="OPENROUTER_API_KEY")
register_provider_adapter("openai", _build_openai, env_key="OPENAI_API_KEY")
register_provider_adapter("anthropic", _build_anthropic, env_key="ANTHROPIC_API_KEY")
register_provider_adapter("groq", _build_groq, env_key="GROQ_API_KEY")
register_provider_adapter("deepinfra", _build_deepinfra, env_key="DEEPINFRA_API_KEY")


class ProviderConfigurationError(RuntimeError):
    """A safe, credential-free provider configuration failure."""

    def __init__(self, provider: str, message: str | None = None):
        self.provider = provider
        super().__init__(message or f"{provider} API key is not configured")


def get_effective_provider_name(assistant_provider: str) -> str:
    """Map user-facing Theriad selection to its OpenRouter implementation."""
    normalized = assistant_provider.strip().lower()
    normalized = _PROVIDER_ALIASES.get(normalized, normalized)
    if normalized not in _PROVIDER_ADAPTERS:
        raise ValueError(f"Unsupported provider: {assistant_provider}")
    return normalized


def build_provider(
    assistant_provider: str,
    *,
    api_key: str | None = None,
    timeout: float = 30.0,
    base_url: str | None = None,
) -> ProviderProtocol:
    """Build a configured provider without logging or returning its raw key."""
    provider_name = get_effective_provider_name(assistant_provider)

    spec = _PROVIDER_ADAPTERS[provider_name]
    if api_key is None:
        if provider_name in {"openrouter", "openai", "anthropic", "groq", "deepinfra"}:
            from theriad.settings.secrets import get_provider_api_key

            api_key = get_provider_api_key(provider_name)
        else:
            api_key = os.environ.get(spec.env_key)
    resolved_key = api_key.strip() if isinstance(api_key, str) else ""
    if not resolved_key:
        raise ProviderConfigurationError(provider_name)

    kwargs: dict[str, object] = {"api_key": resolved_key, "timeout": timeout}
    if base_url is not None:
        kwargs["base_url"] = base_url
    provider = spec.builder(**kwargs)
    # Headroom is a provider-independent default. Decorating here means new
    # model slugs inherit the context policy automatically instead of requiring
    # a per-model registry flag.
    from .headroom import with_headroom
    from .readiness import with_provider_readiness

    return with_provider_readiness(with_headroom(provider))


def build_active_provider(
    *,
    timeout: float = 30.0,
    base_url: str | None = None,
) -> ProviderProtocol:
    """Build the provider selected in local assistant settings."""
    from theriad.settings.secrets import get_active_assistant_provider

    return build_provider(
        get_active_assistant_provider(),
        timeout=timeout,
        base_url=base_url,
    )
