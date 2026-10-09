"""
Provider registry — maps provider name strings to ProviderProtocol instances.

Design: simple dict registry, no metaclass magic. Providers register themselves
via register(). Callers look up by name. Supports lazy instantiation.
"""
from __future__ import annotations

from theriad.core.logging.logger import get_logger

from .interfaces import ProviderProtocol

logger = get_logger("providers.registry")

_registry: dict[str, ProviderProtocol] = {}


def register(provider: ProviderProtocol) -> None:
    """Register a provider instance."""
    _registry[provider.name] = provider
    logger.debug("provider registered", extra={"provider": provider.name})


def get(name: str) -> ProviderProtocol:
    if name not in _registry:
        raise KeyError(f"Provider '{name}' not registered. Available: {list(_registry)}")
    return _registry[name]


def all_providers() -> dict[str, ProviderProtocol]:
    return dict(_registry)
