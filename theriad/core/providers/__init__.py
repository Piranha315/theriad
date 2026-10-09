"""Provider abstraction interfaces and registry."""
from .factory import (
    ProviderConfigurationError,
    build_active_provider,
    build_provider,
    get_effective_provider_name,
    register_provider_adapter,
    registered_provider_adapters,
)
from .interfaces import (
    CompletionRequest,
    CompletionResponse,
    Message,
    ProviderError,
    ProviderProtocol,
)
from .registry import all_providers, get, register
from .runtime import get_provider_candidates, get_provider_model_for_tier

__all__ = [
    "Message",
    "CompletionRequest",
    "CompletionResponse",
    "ProviderProtocol",
    "ProviderError",
    "ProviderConfigurationError",
    "build_provider",
    "build_active_provider",
    "get_effective_provider_name",
    "register_provider_adapter",
    "registered_provider_adapters",
    "get_provider_candidates",
    "get_provider_model_for_tier",
    "register",
    "get",
    "all_providers",
]
