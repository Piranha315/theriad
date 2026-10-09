"""Profile-aware MCP broker primitives with safe audit logging."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from time import time
from typing import Any

from .runtime_config import RuntimeConfig, current_config

SECRET_KEYS = {"api_key", "token", "cookie", "authorization", "password", "secret", "env"}


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if any(s in k.lower() for s in SECRET_KEYS) else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


@dataclass
class MCPAuditEntry:
    runtime_profile: str
    browser_profile_mode: str
    session_id: str
    mcp_server: str
    tool_name: str
    redacted_args: dict[str, Any]
    action_type: str
    approval_required: bool
    approval_received: bool
    status: str
    latency_ms: int
    error: str | None = None
    timestamp: float = 0.0


class MCPBroker:
    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or current_config()
        self.audit_log: list[MCPAuditEntry] = []

    def is_allowed(self, tool_name: str) -> bool:
        allowlist = self.config.mcp_allowlist
        return tool_name in allowlist or any(item.endswith(".*") and tool_name.startswith(item[:-1]) for item in allowlist)

    def record_call(
        self,
        session_id: str,
        mcp_server: str,
        tool_name: str,
        args: dict[str, Any],
        action_type: str,
        approval_received: bool = False,
        latency_ms: int = 0,
        error: str | None = None,
    ) -> MCPAuditEntry:
        approval_required = action_type in self.config.approval_required_actions
        status = "blocked" if not self.is_allowed(tool_name) else ("approval_required" if approval_required and not approval_received else "ok")
        entry = MCPAuditEntry(
            runtime_profile=self.config.runtime_profile.value,
            browser_profile_mode=self.config.browser_profile_mode.value,
            session_id=session_id,
            mcp_server=mcp_server,
            tool_name=tool_name,
            redacted_args=redact(args),
            action_type=action_type,
            approval_required=approval_required,
            approval_received=approval_received,
            status=status,
            latency_ms=latency_ms,
            error=error,
            timestamp=time(),
        )
        self.audit_log.append(entry)
        return entry

    def last_safe_log(self) -> dict[str, Any] | None:
        return asdict(self.audit_log[-1]) if self.audit_log else None
