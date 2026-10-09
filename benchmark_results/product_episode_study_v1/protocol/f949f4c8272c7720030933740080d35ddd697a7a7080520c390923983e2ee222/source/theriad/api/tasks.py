"""Public API for unified tasks and the OpenSandbox browser boundary."""
from __future__ import annotations

import hmac
import os
import secrets
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from theriad.agents.task_engine import TaskEngineError, ToolCall, get_task_engine
from theriad.agents.task_store import (
    TaskStoreError,
    supervisor_lease_seconds,
    task_store_configuration,
)
router = APIRouter(prefix="/api/agent", tags=["unified-agent"])


def _local_api_token() -> str:
    configured = os.environ.get("THERIAD_LOCAL_API_TOKEN", "").strip()
    if configured:
        return configured
    token_path = Path(os.environ.get(
        "THERIAD_LOCAL_API_TOKEN_FILE", "~/.theriad/local-api-token"
    )).expanduser()
    token_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(32))
    os.chmod(token_path, 0o600)
    return token_path.read_text(encoding="utf-8").strip()


def require_local_agent_token(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    expected = _local_api_token()
    supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(401, "A valid local agent bearer token is required.")


Protected = Depends(require_local_agent_token)


class TaskCreateBody(BaseModel):
    goal: str
    permission_profile: str = "strict"
    workspace_root: str | None = None
    requested_capabilities: list[str] = Field(default_factory=list)
    allowed_domains: list[str] = Field(default_factory=list)
    allowed_applications: list[str] = Field(default_factory=list)
    budgets: dict[str, Any] = Field(default_factory=dict)


class ToolCallBody(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class ApprovalBody(BaseModel):
    approval_id: str
    approved: bool
    # Only used when a browser approval was restored after restart.  The
    # original call is intentionally persisted without sensitive arguments;
    # the caller must submit a fresh canonical call instead of replaying a
    # redacted snapshot.
    replacement: "ToolCallBody | None" = None


class TaskInputBody(BaseModel):
    process_id: str
    text: str


def _task(task_id: str):
    try:
        return get_task_engine().get(task_id)
    except TaskEngineError as exc:
        raise HTTPException(404, str(exc)) from exc
    except TaskStoreError as exc:
        raise HTTPException(503, str(exc)) from exc


def _task_owner(request: Request) -> str | None:
    """Resolve an account owner without mistaking the local harness token for a session."""
    from theriad.api.auth import get_request_user

    user = get_request_user(request)
    return user["id"] if user else None


def _task_principal(request: Request) -> str | None:
    """Stable shared-store owner key; never expose it in task payloads."""
    from theriad.api.auth import get_request_user

    user = get_request_user(request)
    if not user:
        return None
    if user.get("supabase_user_id"):
        return f"supabase:{user['supabase_user_id']}"
    return f"local:{user['id']}"


def _owned_task(task_id: str, request: Request):
    task = _task(task_id)
    scope_id = _task_owner(request)
    owner_id = getattr(task, "owner_user_id", None)
    if owner_id != scope_id and (owner_id is not None or scope_id is not None):
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/tasks", dependencies=[Protected])
def create_task(body: TaskCreateBody, request: Request) -> dict:
    try:
        from theriad.api.auth_context import get_auth_context
        context = get_auth_context(request)
        return get_task_engine().create(
            **body.model_dump(),
            owner_user_id=_task_owner(request),
            owner_organization_id=context.organization_id,
            principal_id=_task_principal(request),
        ).safe_dict()
    except TaskEngineError as exc:
        raise HTTPException(400, str(exc)) from exc
    except TaskStoreError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/tasks/{task_id}", dependencies=[Protected])
def task_status(task_id: str, request: Request) -> dict:
    return _owned_task(task_id, request).safe_dict()


@router.get("/tasks/{task_id}/events", dependencies=[Protected])
def task_events(task_id: str, request: Request, after: int = 0) -> dict:
    _owned_task(task_id, request)
    return {"events": get_task_engine().events(task_id, after=after)}


@router.post("/tasks/{task_id}/run", dependencies=[Protected])
async def task_run(task_id: str, request: Request) -> dict:
    _owned_task(task_id, request)
    try:
        return await get_task_engine().supervise(task_id)
    except TaskEngineError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/tasks/{task_id}/tools", dependencies=[Protected])
async def task_tool(task_id: str, body: ToolCallBody, request: Request) -> dict:
    _owned_task(task_id, request)
    call = ToolCall(name=body.name, arguments=body.arguments)
    if body.call_id:
        call.call_id = body.call_id
    result = await get_task_engine().execute_async(task_id, call)
    return result.__dict__


@router.post("/tasks/{task_id}/approve", dependencies=[Protected])
async def task_approve(task_id: str, body: ApprovalBody, request: Request) -> dict:
    _owned_task(task_id, request)
    try:
        replacement = None
        if body.replacement is not None:
            replacement = ToolCall(
                name=body.replacement.name,
                arguments=body.replacement.arguments,
                call_id=body.replacement.call_id or str(uuid.uuid4()),
            )
        result = await get_task_engine().approve_async(
            task_id,
            body.approval_id,
            approved=body.approved,
            replacement_call=replacement,
        )
        return result.__dict__
    except TaskEngineError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/tasks/{task_id}/input", dependencies=[Protected])
async def task_input(task_id: str, body: TaskInputBody, request: Request) -> dict:
    _owned_task(task_id, request)
    result = await get_task_engine().execute_async(
        task_id,
        ToolCall("terminal.write", {"process_id": body.process_id, "text": body.text}),
    )
    return result.__dict__


@router.post("/tasks/{task_id}/stop", dependencies=[Protected])
async def task_stop(task_id: str, request: Request) -> dict:
    _owned_task(task_id, request)
    return (await get_task_engine().stop_async(task_id)).safe_dict()


@router.post("/tasks/{task_id}/extension/pair", dependencies=[Protected])
def extension_pair(task_id: str, request: Request) -> JSONResponse:
    _owned_task(task_id, request)
    return _deprecated_extension_response()


@router.get("/tasks/{task_id}/extension/status", dependencies=[Protected])
def extension_status(task_id: str, request: Request) -> JSONResponse:
    _owned_task(task_id, request)
    return _deprecated_extension_response()


@router.websocket("/extension/ws")
async def extension_websocket(websocket: WebSocket, code: str) -> None:
    del code
    await websocket.close(
        code=4409,
        reason="Extension browser execution is deprecated; use OpenSandbox task APIs.",
    )


def _deprecated_extension_response() -> JSONResponse:
    return JSONResponse(
        status_code=410,
        content={
            "error": "Extension browser execution is deprecated.",
            "replacement": {
                "backend": "opensandbox",
                "task_tools": [
                    "browser.opensandbox.start",
                    "browser.opensandbox.observe",
                    "browser.opensandbox.screenshot",
                    "browser.opensandbox.navigate",
                    "browser.opensandbox.action",
                    "browser.opensandbox.stop",
                ],
            },
        },
        headers={"Deprecation": "true", "Sunset": "0"},
    )


@router.get("/capabilities")
async def capability_manifest() -> dict:
    from theriad.api.auth import deployment_mode, is_hosted_deployment
    from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime
    from theriad.runtime.profile import get_runtime_config
    from theriad.settings.deployment import get_deployment_mode, is_hosted_mode
    from theriad.agents.hosted_runtime import hosted_agent_capability
    from theriad.storage.hosted import storage_status
    from theriad.settings.secrets import (
        get_active_assistant_provider,
        get_all_provider_key_statuses,
    )

    # Ensure the documented local token file exists before reporting setup.
    _local_api_token()
    rc = get_runtime_config()
    browser_runtime = get_opensandbox_browser_runtime()
    browser_status = await browser_runtime.availability()
    provider_statuses = get_all_provider_key_statuses()
    assistant_provider = get_active_assistant_provider()
    key_provider = "openrouter" if assistant_provider == "theriad" else assistant_provider
    selected_provider_configured = provider_statuses[key_provider]["configured"]
    from theriad.agents.desktop_control import availability as desktop_availability
    hosted = is_hosted_mode()
    hosted_controls = hosted_agent_capability(probe=True) if hosted else {
        "available": False,
        "sandboxed": False,
        "tenant_isolated": False,
        "secret_isolation": False,
        "network_policy": "local_only",
        "cancellation": True,
        "audit_support": True,
        "observation_backend": "local",
        "action_backend": "guarded_host",
        "reason": "Hosted isolation controls apply only to hosted deployments.",
    }
    return {
        "contract_version": "1.0",
        "production_mock_success_allowed": False,
        "task_supervisor": {
            **task_store_configuration(),
            "canonical": True,
            "lease_seconds": supervisor_lease_seconds(),
        },
        "deployment_mode": get_deployment_mode(),
        "hosted_execution": {
            "available": bool(not hosted),
            "observation_backend": "disabled" if hosted else "local",
            "action_backend": "disabled" if hosted else "guarded_host",
            "sandboxed": False,
            "approval_mode": "per_sensitive_action",
            "audit_logging": True,
            "reason": (
                "Hosted execution is disabled until a tenant-isolated OS/container sandbox is deployed."
                if hosted else None
            ),
        },
        "hosted_controls": hosted_controls,
        "hosted_storage": storage_status(probe=False),
        "authentication": {
            "task_api": "bearer",
            "token_file": os.environ.get(
                "THERIAD_LOCAL_API_TOKEN_FILE", "~/.theriad/local-api-token"
            ),
        },
        "browser": {
            **browser_status,
            "canonical": True,
            "backend": "opensandbox",
            "deprecated_backends": [
                "mock", "playwright", "playwright_mcp", "browserbase",
                "electron", "chrome_mcp", "extension",
            ],
        },
        "web": {
            "available": True,
            "mode": "api_first_with_bounded_fallback",
            "tools": ["web.search", "web.fetch"],
            "live_verified": False,
            "note": (
                "Capability is installed; provider/network availability is reported "
                "by each tool call. Stateful browser automation is separate."
            ),
        },
        "terminal": {
            "available": not hosted,
            "mode": "disabled_hosted" if hosted else "guarded_host",
            "sandboxed": False,
            "isolation": "none",
            "supports": ["pty", "stream", "stdin", "background", "stop", "process_tree_cleanup"],
            "warning": (
                "Hosted real execution is disabled; no sandbox is deployed."
                if hosted else "Commands execute on the host; policy gates are not OS isolation."
            ),
        },
        "project_tools": {
            "available": True,
            "tools": ["read", "search", "write", "patch", "replace", "move", "delete"],
        },
        "git_tools": {
            "available": bool(shutil.which("git")),
            "tools": ["status", "diff", "log", "branch", "add", "commit", "push"],
        },
        "desktop": {
            **desktop_availability(),
            "setup": "Install the desktop extra, set THERIAD_DESKTOP_CONTROL_ENABLED=true, and grant macOS Accessibility/Screen Recording permissions.",
        },
        "runtime_profile": rc.to_safe_dict(),
        "model_provider": {
            "openrouter_configured": provider_statuses["openrouter"]["configured"],
            "active_provider": assistant_provider,
            "selected_provider_configured": selected_provider_configured,
            "agent_model": os.environ.get("THERIAD_AGENT_MODEL", "deepseek/deepseek-v4-flash-0731"),
            "setup": (
                None
                if selected_provider_configured
                else f"Connect {key_provider.title()} in Settings."
            ),
        },
        "provider_readiness": _provider_readiness_payload(provider_statuses),
    }


def _provider_readiness_payload(configured: dict[str, dict]) -> dict:
    """Return secret-free configured-vs-live provider state."""
    from theriad.runtime.circuit_breaker import get_circuit_registry

    circuits = get_circuit_registry().snapshot()
    by_key = {row["key"]: row for row in circuits}
    providers = []
    for name, status in sorted(configured.items()):
        circuit = by_key.get(f"llm:{name}")
        providers.append({
            "provider": name,
            "configured": bool(status.get("configured")),
            "readiness": circuit.get("status") if circuit else "unknown",
            "circuit": circuit,
        })
    return {
        "providers": providers,
        "search_circuits": [row for row in circuits if row["key"].startswith("search:")],
    }
@router.get("/providers/readiness")
def provider_readiness() -> dict:
    from theriad.settings.secrets import get_all_provider_key_statuses

    return _provider_readiness_payload(get_all_provider_key_statuses())


@router.get("/storage")
def agent_storage_status() -> dict:
    """Return secret-free shared-storage readiness for operators."""
    try:
        return {
            "ready": True,
            **task_store_configuration(),
            **get_task_engine().storage_status(),
        }
    except TaskStoreError as exc:
        return JSONResponse(status_code=503, content={
            "ready": False,
            **task_store_configuration(),
            "configured": False,
            "error": str(exc),
        })
@router.post("/providers/{provider_name}/probe", dependencies=[Protected])
async def probe_provider(provider_name: str) -> dict:
    """Perform an explicit live probe; no credential value is returned."""
    from theriad.core.providers.factory import (
        ProviderConfigurationError,
        UnsupportedProviderError,
        build_provider,
    )

    try:
        provider = build_provider(provider_name, timeout=8)
        ready = await provider.health_check()
    except (ProviderConfigurationError, UnsupportedProviderError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"provider": provider_name, "ready": bool(ready)}
