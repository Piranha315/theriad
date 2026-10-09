"""
Pydantic v2 schemas for the MCP broker layer.

All fields follow the exact names that will be returned by the future
browser-agent backend so the frontend contract is stable.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    low      = "low"
    medium   = "medium"
    high     = "high"
    critical = "critical"


class CallStatus(str, Enum):
    ok               = "ok"
    blocked          = "blocked"
    error            = "error"
    approval_pending = "approval_pending"


# ── Tool / subtask specs ──────────────────────────────────────────────────────

class MCPToolSpec(BaseModel):
    """Per-tool capability entry stored in the tool registry."""
    server_name:           str
    tool_name:             str
    risk_level:            RiskLevel = RiskLevel.low
    requires_approval:     bool      = False
    max_calls_per_session: int       = 20
    timeout_seconds:       float     = 10.0


class SubtaskAllowlist(BaseModel):
    """
    Tools permitted for a named subtask.

    Only tools explicitly listed in ``allowed_tools`` may be called under
    this subtask. The broker rejects anything not on the list before it
    reaches any adapter or MCP server.
    """
    subtask:                   str
    allowed_server:            str
    allowed_tools:             list[str]
    requires_approval_for_any: bool = False
    max_calls_per_session:     int  = 20


# ── Call I/O ──────────────────────────────────────────────────────────────────

class MCPCallRequest(BaseModel):
    """Input to MCPBroker.call()."""
    session_id:  str
    subtask:     str
    server_name: str
    tool_name:   str
    arguments:   dict[str, Any] = Field(default_factory=dict)


class MCPCallResult(BaseModel):
    """
    Output from MCPBroker.call().

    ``ok`` is True only when the adapter executed the tool successfully.
    When ``approval_required`` is True the broker did not execute the tool;
    the caller must obtain human approval and re-invoke.
    """
    ok:               bool
    result:           dict[str, Any] | None = None
    error:            str | None            = None
    blocked_reason:   str | None            = None
    approval_required: bool                 = False
    latency_ms:       float                 = 0.0


# ── Diagnostics ───────────────────────────────────────────────────────────────

class MCPCallLog(BaseModel):
    """
    Diagnostic record written for every broker call attempt regardless of
    outcome. Sensitive arguments are never logged here.

    Per-call fields mandated by the profile spec:
        runtime_profile      — active RuntimeProfile value
        browser_profile_mode — active BrowserProfileMode value
        action_type          — logical action category (mirrors tool_name unless overridden)
        approval_received    — True when human explicitly approved this action
    """
    session_id:           str
    subtask:              str
    server_name:          str
    tool_name:            str
    risk_level:           RiskLevel
    approval_required:    bool
    approval_received:    bool      = False
    status:               CallStatus
    latency_ms:           float
    error:                str | None = None
    result_summary:       str | None = None   # ≤ 120 chars, safe to log
    # Runtime context (safe to log — no secrets)
    runtime_profile:      str = "consumer_secure"
    browser_profile_mode: str = "ephemeral"
    action_type:          str | None = None


# ── Status ────────────────────────────────────────────────────────────────────

class MCPServerStatus(BaseModel):
    """Runtime availability of one MCP server."""
    server_name: str
    available:   bool
    connected:   bool
    error:       str | None = None
    tool_count:  int        = 0


class MCPBrokerStatus(BaseModel):
    """
    Returned by GET /api/tools/status.

    Frontend uses this to show "Chrome MCP: Connected / Not configured" and
    "Browser runtime: Mock / MCP available / Real session active".
    """
    broker_ready:              bool                  = True
    servers:                   list[MCPServerStatus] = Field(default_factory=list)
    subtask_allowlists:        list[str]             = Field(default_factory=list)
    total_calls_this_session:  int                   = 0
    # Runtime profile (safe fields only — no secrets)
    runtime_profile:           str                   = "consumer_secure"
    browser_profile_mode:      str                   = "ephemeral"
    profile_badge:             str                   = (
        "Secure Mode — isolated browser and approval gates enabled."
    )


class ChromeStatusResponse(BaseModel):
    """
    Returned by GET /api/tools/chrome/status.

    Reports Chrome MCP availability and the active runtime profile/browser mode.
    Never includes secrets, cookies, tokens, or API keys.
    """
    configured:               bool
    available:                bool
    runtime_profile:          str
    browser_profile_mode:     str
    allow_real_chrome_profile: bool
    headless:                 bool
    mode:                     str     # "mock" | "chrome_mcp"
    message:                  str
    # Optional: filled when using dedicated_persistent or real_profile
    chrome_user_data_dir:     Optional[str] = None
