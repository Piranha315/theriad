"""
MCPBroker — controlled gateway for all MCP tool calls.

Every call to an external MCP tool must go through this broker.
The broker enforces:
    - Subtask allowlists (tool must be permitted for the current subtask)
    - Server allowlists (only the subtask's designated server may be called)
    - Risk-based approval gates (high/critical tools require human sign-off)
    - Per-session call budgets (hard cap on calls per tool per session)
    - Full diagnostics logging (every attempt, regardless of outcome)

Adapters (e.g. ChromeDevToolsAdapter) are injected at construction time so
the broker itself has no transport knowledge and is easy to test with stubs.

Usage
-----
    from theriad.tools.chrome_devtools_adapter import ChromeDevToolsAdapter
    from theriad.tools.mcp_broker import MCPBroker

    adapter = ChromeDevToolsAdapter()
    broker  = MCPBroker(session_id="abc123", adapters={"chrome_devtools": adapter})

    result = broker.call(
        subtask="website_preview_qa",
        server_name="chrome_devtools",
        tool_name="take_screenshot",
        arguments={"page_id": "p1"},
    )
    if result.ok:
        screenshot_data = result.result
"""
from __future__ import annotations

import logging
import time
from typing import Any

from theriad.runtime.profile import (
    RuntimeConfig,
    get_runtime_config,
)
from theriad.tools.mcp_registry import get_allowlist, get_tool_spec
from theriad.tools.mcp_schemas import (
    CallStatus,
    MCPBrokerStatus,
    MCPCallLog,
    MCPCallResult,
    MCPServerStatus,
    RiskLevel,
)

logger = logging.getLogger(__name__)


class MCPBroker:
    """
    Controlled gateway for MCP tool calls.

    Parameters
    ----------
    session_id:
        Opaque identifier for the current agent/preview session; included in
        every log entry.
    adapters:
        Map of server_name → adapter instance. Adapters must expose:
            .available: bool
            .call_tool(tool_name: str, arguments: dict) -> dict
        Adapters should never be called directly by external code.
    """

    def __init__(
        self,
        session_id: str,
        adapters: dict[str, Any] | None = None,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        self.session_id = session_id
        self._adapters: dict[str, Any] = adapters or {}
        self._call_log: list[MCPCallLog] = []
        # (subtask, server_name, tool_name) → number of calls this session
        self._call_counts: dict[tuple[str, str, str], int] = {}
        # Runtime profile — defaults to process-wide singleton
        self._runtime_config: RuntimeConfig = runtime_config or get_runtime_config()

    # ── Public API ────────────────────────────────────────────────────────────

    def call(
        self,
        subtask: str,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        approval_received: bool = False,
    ) -> MCPCallResult:
        """
        Invoke a tool through the broker.

        Never raises. Returns MCPCallResult with ok=False if the call was
        blocked, requires approval, or the adapter raised an exception.

        When approval_required=True in the result, the caller must obtain
        explicit human approval before re-invoking.
        """
        arguments = arguments or {}
        start = time.monotonic()

        # ── 1. Subtask allowlist ──────────────────────────────────────────────
        allowlist = get_allowlist(subtask)
        if allowlist is None:
            return self._blocked(
                subtask, server_name, tool_name, start,
                f"Unknown subtask '{subtask}'. "
                "Register it in theriad/tools/mcp_registry.py.",
            )

        if allowlist.allowed_server != server_name:
            return self._blocked(
                subtask, server_name, tool_name, start,
                f"Server '{server_name}' is not the designated server for "
                f"subtask '{subtask}' (expected '{allowlist.allowed_server}').",
            )

        if tool_name not in allowlist.allowed_tools:
            return self._blocked(
                subtask, server_name, tool_name, start,
                f"Tool '{tool_name}' is not in the allowlist for subtask "
                f"'{subtask}'. Allowed: {allowlist.allowed_tools}.",
            )

        # ── 2. Risk / approval gate ───────────────────────────────────────────
        spec = get_tool_spec(server_name, tool_name)
        risk = spec.risk_level if spec else RiskLevel.medium
        approval_needed = (
            (spec.requires_approval if spec else False)
            or allowlist.requires_approval_for_any
            # consumer_secure escalates: any high/critical tool requires approval
            or (
                self._runtime_config.is_consumer_secure
                and risk in (RiskLevel.high, RiskLevel.critical)
            )
            # profile-level action check (e.g. login, submit_form for consumer_secure)
            or self._runtime_config.requires_approval_for(tool_name)
        )

        if approval_needed and not approval_received:
            latency = (time.monotonic() - start) * 1000
            self._append_log(
                subtask, server_name, tool_name, risk,
                approval_required=True,
                approval_received=False,
                status=CallStatus.approval_pending,
                latency_ms=latency,
            )
            logger.info(
                "[mcp_broker] approval_pending session=%s subtask=%s tool=%s risk=%s",
                self.session_id, subtask, tool_name, risk.value,
            )
            return MCPCallResult(
                ok=False,
                approval_required=True,
                blocked_reason="Human approval required before this action.",
                latency_ms=round(latency, 2),
            )

        # ── 3. Budget check ───────────────────────────────────────────────────
        budget_key = (subtask, server_name, tool_name)
        current = self._call_counts.get(budget_key, 0)
        # Use the tighter of the per-tool and per-subtask caps
        spec_limit    = spec.max_calls_per_session if spec else 20
        subtask_limit = allowlist.max_calls_per_session or 20
        limit = min(spec_limit, subtask_limit)
        if current >= limit:
            return self._blocked(
                subtask, server_name, tool_name, start,
                f"Per-session call budget exceeded: {limit} calls allowed "
                f"for '{tool_name}' under subtask '{subtask}'.",
            )

        # ── 4. Adapter availability ───────────────────────────────────────────
        adapter = self._adapters.get(server_name)
        if adapter is None:
            return self._blocked(
                subtask, server_name, tool_name, start,
                f"No adapter registered for server '{server_name}'. "
                "Instantiate one and pass it to MCPBroker(adapters=...).",
            )

        if not getattr(adapter, "available", False):
            return self._blocked(
                subtask, server_name, tool_name, start,
                f"Adapter '{server_name}' reports unavailable. "
                f"Error: {getattr(adapter, 'error', 'unknown')}",
            )

        # ── 5. Execute ────────────────────────────────────────────────────────
        try:
            result = adapter.call_tool(tool_name, arguments)
            latency = (time.monotonic() - start) * 1000
            self._call_counts[budget_key] = current + 1
            summary = str(result)[:120] if result else None
            self._append_log(
                subtask, server_name, tool_name, risk,
                approval_required=False,
                approval_received=approval_received,
                status=CallStatus.ok,
                latency_ms=latency,
                result_summary=summary,
            )
            logger.info(
                "[mcp_broker] ok session=%s subtask=%s tool=%s latency=%.1fms",
                self.session_id, subtask, tool_name, latency,
            )
            return MCPCallResult(
                ok=True,
                result=result,
                latency_ms=round(latency, 2),
            )

        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            self._append_log(
                subtask, server_name, tool_name, risk,
                approval_required=False,
                approval_received=approval_received,
                status=CallStatus.error,
                latency_ms=latency,
                error=str(exc),
            )
            logger.warning(
                "[mcp_broker] error session=%s subtask=%s tool=%s err=%s",
                self.session_id, subtask, tool_name, exc,
            )
            return MCPCallResult(
                ok=False,
                error=str(exc),
                latency_ms=round(latency, 2),
            )

    def get_call_log(self) -> list[MCPCallLog]:
        """Return a copy of all call log entries for this session."""
        return list(self._call_log)

    def close(self) -> None:
        """Close transports owned by the broker and release session resources."""
        for adapter in self._adapters.values():
            close = getattr(adapter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    logger.debug("[mcp_broker] adapter close failed: %s", exc)

    def get_status(self) -> MCPBrokerStatus:
        """Return current broker and adapter status."""
        from theriad.tools.mcp_registry import SUBTASK_ALLOWLISTS
        servers = [
            MCPServerStatus(
                server_name=name,
                available=getattr(adapter, "available", False),
                connected=getattr(adapter, "connected", False),
                error=getattr(adapter, "error", None),
                tool_count=len(getattr(adapter, "tools", [])),
            )
            for name, adapter in self._adapters.items()
        ]
        rc = self._runtime_config
        return MCPBrokerStatus(
            broker_ready=True,
            servers=servers,
            subtask_allowlists=list(SUBTASK_ALLOWLISTS.keys()),
            total_calls_this_session=sum(self._call_counts.values()),
            runtime_profile=rc.runtime_profile.value,
            browser_profile_mode=rc.browser_profile_mode.value,
            profile_badge=rc.profile_badge,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _blocked(
        self,
        subtask: str,
        server_name: str,
        tool_name: str,
        start: float,
        reason: str,
    ) -> MCPCallResult:
        latency = (time.monotonic() - start) * 1000
        spec = get_tool_spec(server_name, tool_name)
        risk = spec.risk_level if spec else RiskLevel.medium
        self._append_log(
            subtask, server_name, tool_name, risk,
            approval_required=False,
            status=CallStatus.blocked,
            latency_ms=latency,
            error=reason,
        )
        logger.info(
            "[mcp_broker] blocked session=%s tool=%s reason=%s",
            self.session_id, tool_name, reason,
        )
        return MCPCallResult(
            ok=False,
            blocked_reason=reason,
            latency_ms=round(latency, 2),
        )

    def _append_log(
        self,
        subtask: str,
        server_name: str,
        tool_name: str,
        risk: RiskLevel,
        *,
        approval_required: bool,
        approval_received: bool = False,
        status: CallStatus,
        latency_ms: float,
        error: str | None = None,
        result_summary: str | None = None,
    ) -> None:
        rc = self._runtime_config
        self._call_log.append(
            MCPCallLog(
                session_id=self.session_id,
                subtask=subtask,
                server_name=server_name,
                tool_name=tool_name,
                risk_level=risk,
                approval_required=approval_required,
                approval_received=approval_received,
                status=status,
                latency_ms=round(latency_ms, 2),
                error=error,
                result_summary=result_summary,
                runtime_profile=rc.runtime_profile.value,
                browser_profile_mode=rc.browser_profile_mode.value,
                action_type=tool_name,
            )
        )
