"""Isolation and truthful-capability contracts for the separate agent worker."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from theriad.agents.hosted_runtime import hosted_agent_capability
from theriad.agents.runner_service import StartRequest, _docker_command


def test_hosted_agents_stay_disabled_until_both_backends_and_adapter_are_ready(monkeypatch):
    monkeypatch.delenv("THERIAD_HOSTED_AGENT_ENABLED", raising=False)
    monkeypatch.delenv("THERIAD_AGENT_RUNNER_URL", raising=False)
    monkeypatch.delenv("THERIAD_AGENT_RUNNER_TOKEN", raising=False)

    capability = hosted_agent_capability(probe=False)

    assert capability["available"] is False
    assert capability["sandboxed"] is False
    assert capability["adapter_integrated"] is False


def test_runner_command_is_non_root_network_default_deny_and_argv_only(tmp_path):
    body = StartRequest(
        task_id="task-1",
        organization_id="org-1",
        user_id="user-1",
        argv=["python", "-c", "print('ok')"],
        approved=True,
    )
    command = _docker_command(body, tmp_path)

    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--user") + 1] == "65532:65532"
    assert "--read-only" in command
    assert "--privileged" not in command
    assert command[-3:] == body.argv


def test_runner_rejects_domain_allowlist_without_proxy(tmp_path):
    body = StartRequest(
        task_id="task-2",
        organization_id="org-1",
        user_id="user-1",
        argv=["curl", "https://example.com"],
        allowed_domains=["example.com"],
        approved=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        _docker_command(body, tmp_path)
    assert exc_info.value.status_code == 409
