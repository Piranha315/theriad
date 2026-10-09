"""Request identity context and the hosted authentication boundary."""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from theriad.settings.deployment import (
    DeploymentConfigurationError,
    get_deployment_mode,
    is_hosted_mode,
)
from theriad.storage.organizations import ensure_personal_organization
from theriad.storage.hosted import HostedStorageUnavailable

PrincipalType = Literal["anonymous", "guest", "registered"]


@dataclass(frozen=True)
class AuthContext:
    """The identity and deployment facts routes must not infer ad hoc."""

    principal_type: PrincipalType
    user_id: str | None
    organization_id: str | None
    role: str | None
    deployment_mode: str
    legacy_local_scope: bool = False


current_auth_context: ContextVar[AuthContext | None] = ContextVar(
    "theriad_current_auth_context", default=None
)


def get_auth_context(request: Request) -> AuthContext:
    """Build the request identity from the Theriad bearer session."""
    # Import lazily: auth.py owns token lookup and must not import this module
    # during its own initialization.
    from theriad.api.auth import get_request_user

    mode = get_deployment_mode()
    user = get_request_user(request)
    if user is None:
        return AuthContext(
            principal_type="anonymous",
            user_id=None,
            organization_id=None,
            role=None,
            deployment_mode=mode,
            legacy_local_scope=mode != "hosted",
        )

    is_guest = bool(user.get("is_guest"))
    if is_guest:
        organization_id = None
        role = None
        from theriad.storage.hosted import hosted_data_plane_enabled

        if hosted_data_plane_enabled():
            from theriad.storage.organizations import ensure_guest_organization

            membership = ensure_guest_organization(user["id"])
            organization_id = membership["organization_id"]
            role = membership["role"]
        return AuthContext(
            principal_type="guest",
            user_id=user["id"],
            organization_id=organization_id,
            role=role,
            deployment_mode=mode,
        )

    membership = ensure_personal_organization(
        user["id"], user.get("display_name") or user.get("username")
    )
    return AuthContext(
        principal_type="registered",
        user_id=user["id"],
        organization_id=membership["organization_id"],
        role=membership["role"],
        deployment_mode=mode,
    )


def require_auth_context(request: Request) -> AuthContext:
    """Return a non-anonymous context or raise the canonical 401."""
    context = get_auth_context(request)
    if context.principal_type == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required.")
    return context


def set_auth_context(context: AuthContext) -> Token:
    return current_auth_context.set(context)


def reset_auth_context(token: Token) -> None:
    current_auth_context.reset(token)


_PUBLIC_EXACT_PATHS = frozenset(
    {
        "/api/health",
        "/api/models",
        "/api/pricing",
        "/api/pricing/plans",
        "/v1/models",
        "/api/billing/webhook",
    }
)


def is_public_api_path(path: str, method: str) -> bool:
    """Return whether an API path may be used before account creation."""
    if method.upper() == "OPTIONS":
        return True
    if path in _PUBLIC_EXACT_PATHS:
        return True
    if path == "/api/auth" or path.startswith("/api/auth/"):
        return True
    # Landing-page measurement is intentionally anonymous. The event payload
    # contains no account data and the collector has its own validation.
    if path == "/api/analytics/collect" and method.upper() == "POST":
        return True
    # The OAuth callback is authenticated by a one-time state value stored by
    # the authenticated start request. The callback must not fall back to the
    # process-global secret file; the state carries the owning user id.
    if path.startswith("/api/settings/providers/openrouter/oauth/callback/"):
        return True
    return False


def requires_hosted_auth(path: str, method: str) -> bool:
    """All HTTP API surfaces are private except the explicit public allowlist."""
    if not (path.startswith("/api/") or path.startswith("/v1/")):
        return False
    return not is_public_api_path(path, method)


def is_hosted_real_agent_path(path: str) -> bool:
    """Real agent actions stay disabled until hosted isolation exists."""
    # ``/api/chat/agent`` contains a body-aware, read-only API-search path.
    # Let that route reach its handler; the handler blocks browser, terminal,
    # and multi-tool execution after it classifies the request.
    if path == "/api/chat/agent":
        return False
    if path.startswith("/api/agent/") and not path.endswith("/capabilities"):
        return True
    return path.startswith("/api/browser/session/")


def hosted_agent_unavailable_response() -> JSONResponse:
    """Return the stable fail-closed response for hosted agent execution."""
    return JSONResponse(
        status_code=503,
        content={
            "error": "hosted_agent_execution_disabled",
            "message": (
                "Hosted real terminal and browser execution is disabled until tenant-isolated "
                "sandboxing, secrets, budgets, approvals, cancellation, and audit controls are deployed."
            ),
        },
        headers={"Cache-Control": "no-store"},
    )


async def hosted_auth_middleware(request: Request, call_next):
    """Reject anonymous private API requests when running in hosted mode."""
    try:
        get_deployment_mode()
        context = get_auth_context(request)
    except DeploymentConfigurationError as exc:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc)},
            headers={"Cache-Control": "no-store"},
        )
    except HostedStorageUnavailable as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": "hosted_storage_unavailable",
                "message": str(exc),
            },
            headers={"Cache-Control": "no-store"},
        )

    context_token = set_auth_context(context)
    try:
        if is_hosted_mode() and requires_hosted_auth(request.url.path, request.method):
            if context.principal_type == "anonymous":
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authentication required in hosted mode."},
                    headers={"Cache-Control": "no-store"},
                )
            from theriad.storage.hosted import (
                assert_hosted_storage_ready,
                hosted_data_plane_enabled,
            )

            if hosted_data_plane_enabled():
                try:
                    assert_hosted_storage_ready(probe=True)
                except HostedStorageUnavailable as exc:
                    return JSONResponse(
                        status_code=503,
                        content={
                            "error": "hosted_storage_unavailable",
                            "message": str(exc),
                        },
                        headers={"Cache-Control": "no-store"},
                    )
        if is_hosted_mode() and is_hosted_real_agent_path(request.url.path):
            try:
                from theriad.storage.audit import record_event
                record_event(
                    "hosted_agent_blocked",
                    organization_id=context.organization_id,
                    user_id=context.user_id,
                    resource_type="http_path",
                    resource_id=request.url.path,
                    metadata={"method": request.method, "reason": "sandbox_not_deployed"},
                )
            except Exception:
                pass
            return hosted_agent_unavailable_response()
        return await call_next(request)
    finally:
        reset_auth_context(context_token)
