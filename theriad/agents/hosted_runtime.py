"""Capability and transport boundary for hosted agent workers.

Hosted agent work is never executed in the web process. This module only
describes the two external prerequisites (Browserbase plus a disposable
container runner) and provides a narrow authenticated transport for a future
worker service. Until the worker health check and the explicit enable flag
pass, callers receive ``available: false`` and the API keeps refusing agent
execution.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class HostedRunnerConfig:
    enabled: bool
    runner_url: str | None
    runner_token: str | None
    browserbase_configured: bool
    browserbase_sdk_available: bool

    @property
    def runner_configured(self) -> bool:
        return bool(self.runner_url and self.runner_token)


def get_hosted_runner_config() -> HostedRunnerConfig:
    browserbase_key = os.environ.get("BROWSERBASE_API_KEY", "").strip()
    try:
        from theriad.settings.secrets import get_browserbase_api_key

        browserbase_key = browserbase_key or (get_browserbase_api_key() or "")
    except Exception:
        pass
    import importlib.util

    return HostedRunnerConfig(
        enabled=_truthy(os.environ.get("THERIAD_HOSTED_AGENT_ENABLED")),
        runner_url=os.environ.get("THERIAD_AGENT_RUNNER_URL", "").strip().rstrip("/") or None,
        runner_token=os.environ.get("THERIAD_AGENT_RUNNER_TOKEN", "").strip() or None,
        browserbase_configured=bool(browserbase_key),
        browserbase_sdk_available=importlib.util.find_spec("browserbase") is not None,
    )


def runner_health(config: HostedRunnerConfig | None = None) -> bool:
    config = config or get_hosted_runner_config()
    if not config.runner_configured:
        return False
    try:
        response = httpx.get(
            f"{config.runner_url}/health",
            headers={"Authorization": f"Bearer {config.runner_token}"},
            timeout=3.0,
        )
        return response.status_code == 200 and response.json().get("status") == "ok"
    except (httpx.HTTPError, ValueError, AttributeError):
        return False


def hosted_agent_capability(*, probe: bool = True) -> dict[str, Any]:
    config = get_hosted_runner_config()
    runner_healthy = runner_health(config) if probe else config.runner_configured
    browserbase_ready = config.browserbase_configured and config.browserbase_sdk_available
    # The web process deliberately does not invoke the worker protocol until
    # the task/event adapter is enabled in a reviewed release. This prevents a
    # configuration-only flag from ever making the current host executor look
    # tenant-isolated.
    adapter_integrated = False
    ready = bool(config.enabled and browserbase_ready and runner_healthy and adapter_integrated)
    reason = None
    if not config.enabled:
        reason = "Hosted agents are disabled until the operator enables the isolated worker."
    elif not browserbase_ready:
        reason = "Browserbase SDK and server-side credentials are not both configured."
    elif not config.runner_configured:
        reason = "The disposable per-task runner URL and server token are not configured."
    elif not runner_healthy:
        reason = "The isolated runner did not pass its startup health check."
    elif not adapter_integrated:
        reason = "The isolated runner is provisioned, but the hosted task adapter is not enabled."
    return {
        "available": ready,
        "sandboxed": ready,
        "tenant_isolated": ready,
        "secret_isolation": ready,
        "network_policy": "default_deny_allowlist" if ready else "not_deployed",
        "cancellation": ready,
        "audit_support": ready,
        "observation_backend": "browserbase" if ready else "disabled",
        "action_backend": "isolated_container" if ready else "disabled",
        "browserbase": {
            "configured": config.browserbase_configured,
            "sdk_available": config.browserbase_sdk_available,
            "startup_check": browserbase_ready,
        },
        "runner": {
            "configured": config.runner_configured,
            "startup_check": runner_healthy,
            "protocol": "theriad-agent-runner-v1",
        },
        "adapter_integrated": adapter_integrated,
        "reason": reason,
    }


class HostedRunnerClient:
    """Authenticated HTTP client; command execution belongs to the worker."""

    def __init__(self, config: HostedRunnerConfig | None = None):
        self.config = config or get_hosted_runner_config()
        if not self.config.runner_configured:
            raise RuntimeError("Hosted agent runner is not configured.")

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.runner_token}"}

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{self.config.runner_url}{path}",
                    json=payload,
                    headers=self._headers,
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError("Hosted agent runner is unavailable.") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Hosted agent runner returned an invalid response.")
        return data

    async def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/v1/tasks", payload)

    async def approve(self, task_id: str, approval_id: str, approved: bool) -> dict[str, Any]:
        return await self._post(f"/v1/tasks/{task_id}/approvals", {"approval_id": approval_id, "approved": approved})

    async def stop(self, task_id: str) -> dict[str, Any]:
        return await self._post(f"/v1/tasks/{task_id}/cancel", {})

    async def send_input(self, task_id: str, process_id: str, text: str) -> dict[str, Any]:
        return await self._post(f"/v1/tasks/{task_id}/input", {"process_id": process_id, "text": text})
