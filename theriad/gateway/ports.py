"""Provider-neutral seams for an organization-owned gateway deployment."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Protocol

from .contracts import Deployment

CREDENTIAL_SCOPE = ContextVar("gateway_credential_scope", default=None)
LOOPBACK_BYOK = ContextVar("gateway_loopback_byok", default=None)


class Credentials(Protocol):
    async def token(self, deployment: Deployment) -> str: ...


class Policy(Protocol):
    def allows(self, deployment: Deployment) -> bool: ...


class AtomicStorage(Protocol):
    def get(self, key: str) -> Any: ...
    def put(self, key: str, value: Any, ttl: float | None = None) -> None: ...
    def mutate(self, key: str, change: Callable[[Any], Any]) -> Any: ...


class Accounting(Protocol):
    def reserve(self, maximum: float, phase: str, identity: str) -> tuple[str, str]: ...
    def settle(self, ledger: str, reservation: str, actual: float | None) -> None: ...


class Audit(Protocol):
    def emit(self, event: dict) -> None: ...


class StoreAudit:
    def __init__(self, store: AtomicStorage):
        self.store = store

    def emit(self, event):
        import uuid

        self.store.put("audit/" + uuid.uuid4().hex, event)


class LocalAndGoogleCredentials:
    """Endpoint-specific env references or refreshable attached/ADC identity."""

    def __init__(self, google=None):
        self.google = google

    async def token(self, deployment):
        import asyncio
        import os

        if deployment.provider == "openrouter" and LOOPBACK_BYOK.get():
            return LOOPBACK_BYOK.get()
        if deployment.provider != "vertex":
            return os.environ.get(deployment.credential_env or "", "")
        if self.google is None:
            from .gcp import GoogleCredentials

            self.google = GoogleCredentials()
        try:
            return await asyncio.to_thread(self.google.token)
        except Exception as exc:
            from .contracts import GatewayError

            # Authentication failures may include endpoint/credential details;
            # expose a stable code, never the provider's exception text.
            raise GatewayError(
                "provider_credentials_missing", "Google identity credentials are unavailable", 503
            ) from exc


def evaluation_store(default):
    """One configured budget authority, independent of a client's release store."""
    import os

    if os.environ.get("THERIAD_EVALUATION_STORE") == "gcp":
        from .gcp import GoogleStore

        return GoogleStore(
            os.environ["THERIAD_EVALUATION_PROJECT"], os.environ["THERIAD_EVALUATION_BUCKET"]
        )
    if os.environ.get("THERIAD_EVALUATION_DB"):
        from .store import LocalStore

        return LocalStore(os.environ["THERIAD_EVALUATION_DB"])
    from .store import LocalStore

    if isinstance(default, LocalStore):
        # Private release databases must not silently mint another monthly allowance.
        return LocalStore("~/.theriad/gateway/state.sqlite")
    return default
