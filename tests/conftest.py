"""Shared pytest fixtures."""
import pytest

from theriad.core.config.schema import TheriadConfig


@pytest.fixture
def default_config() -> TheriadConfig:
    return TheriadConfig()


@pytest.fixture(autouse=True)
def isolate_router_v2_env(monkeypatch):
    """Keep a developer .env from switching Router V2 on for unrelated tests.

    Importing theriad.api.server loads the project .env into os.environ, so a
    local THERIAD_ROUTER_V2_MODE=on would otherwise leak into every test that
    runs afterwards. Tests that exercise V2 set these variables themselves.
    """
    import os

    from theriad.routing.v2 import clear_router_cache

    for name in [key for key in os.environ if key.startswith("THERIAD_ROUTER_V2_")]:
        monkeypatch.delenv(name, raising=False)
    clear_router_cache()
    yield


@pytest.fixture(autouse=True)
def reset_process_circuit_state():
    """Keep process-wide provider circuits isolated between unit tests."""
    from theriad.runtime.circuit_breaker import get_circuit_registry

    registry = get_circuit_registry()
    registry.reset()
    yield
    registry.reset()
