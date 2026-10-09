"""
Pydantic schemas for browser-agent API surface.

Design:
- All request/response shapes live here so server.py stays thin.
- Status responses never include secrets, API keys, cookies, or auth tokens.
- Enums are plain strings for easy JSON round-tripping.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Policy presets ────────────────────────────────────────────────────────────

POLICY_AGGRESSIVE_SAVINGS = "aggressive_savings"
POLICY_BALANCED = "balanced"
POLICY_QUALITY_FIRST = "quality_first"
POLICY_FAST = "fast"

VALID_POLICIES = frozenset({
    POLICY_AGGRESSIVE_SAVINGS,
    POLICY_BALANCED,
    POLICY_QUALITY_FIRST,
    POLICY_FAST,
})

# ── Session statuses ──────────────────────────────────────────────────────────

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_WAITING_FOR_APPROVAL = "waiting_for_approval"
STATUS_NEEDS_APPROVAL = "needs_approval"
# D-045: distinct from waiting_for_approval — there is no pending yes/no
# decision here, the action cannot proceed because a required input
# (currently: login credentials) was never provided and no credential
# capture mechanism exists yet. Durable pause, same as waiting_for_approval:
# run_until_pause() and run_step()'s guard both treat it as a stop condition.
STATUS_WAITING_FOR_INPUT = "waiting_for_input"
# Part 4: agent pauses for the user to log in manually in the controlled
# browser, then click "Continue". Distinct from waiting_for_input (which is
# a dead end) — this is a recoverable pause expecting human action.
STATUS_MANUAL_LOGIN_PAUSE = "manual_login_pause"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_STOPPED = "stopped"
STATUS_STOPPED_BUDGET = "stopped_budget_exceeded"

# ── Default approval-required action types ───────────────────────────────────
# Legacy constant (D-040 era). Kept for backward compatibility — some callers
# still pass this explicitly as a session-level `approval_required_actions`
# override. It is NOT the schema default anymore (see D-044): permission_mode
# + the HARD_STOP_ACTIONS / HIGH_RISK_ACTIONS classification below is now the
# primary mechanism, so behavior can vary by user-selected permission mode
# instead of being a single fixed list for every session.

DEFAULT_APPROVAL_REQUIRED: list[str] = [
    "login",
    "submit_form",
    "send_message",
    "purchase",
    "checkout",
    "delete",
    "edit_account_data",
    "upload_file",
    "download_file",
    "enter_sensitive_info",
    "grant_permission",
    "access_financial_portal",
    "access_medical_portal",
    "access_legal_portal",
]

# ── Permission modes (D-044) ──────────────────────────────────────────────────
# User-facing permission policy that controls how often the agent pauses for
# approval. See DECISIONS.md D-044 for the full behavior spec.

PERMISSION_ALWAYS_ASK = "always_ask"
PERMISSION_ASK_HIGH_RISK = "ask_high_risk"
PERMISSION_NEVER_ASK = "never_ask"

VALID_PERMISSION_MODES = frozenset({
    PERMISSION_ALWAYS_ASK,
    PERMISSION_ASK_HIGH_RISK,
    PERMISSION_NEVER_ASK,
})

# Recommended default for both consumer_secure and personal_trusted (Part 6).
DEFAULT_PERMISSION_MODE = PERMISSION_ASK_HIGH_RISK

# Actions that are always safe to auto-allow, in every permission mode,
# because they never change page/account state or leave the current page
# context. This is the "Can allow without approval" list from always_ask,
# which is also a subset of ask_high_risk's low-risk list.
ALWAYS_AGNOSTIC_SAFE_ACTIONS: frozenset[str] = frozenset({
    "navigate",
    "search",
    "scroll",
    "read_page",
    "extract_text",
    "screenshot",
    "summarize",
    "go_back",
    "go_forward",
    "reload",
    "dismiss_popup",
})

# Normal link/button clicks — safe to auto-allow under ask_high_risk and
# never_ask, but NOT under always_ask (spec: "Ask before: click, ...").
CLICK_ACTIONS: frozenset[str] = frozenset({"click_link", "click_button"})

# High-risk actions: ask_high_risk mode asks before these; always_ask asks
# before these too (they're outside ALWAYS_AGNOSTIC_SAFE_ACTIONS); never_ask
# allows them automatically UNLESS they are also hard-stop actions.
HARD_STOP_ACTIONS: frozenset[str] = frozenset({
    "purchase",
    "checkout",
    "send_message",
    "delete",
    "delete_account_data",
    "edit_account_data",
    "submit_legal_form",
    "submit_financial_form",
    "submit_medical_form",
    "submit_school_form",
    "submit_job_form",
    "trade",
    "invest",
    "expose_secrets",
    "install_software",
    "grant_permission",
    "upload_sensitive_file",
})

HIGH_RISK_ACTIONS: frozenset[str] = frozenset({
    "login",
    "submit_form",
    "upload_file",
    "download_file",
    "enter_sensitive_info",
    "access_financial_portal",
    "access_medical_portal",
    "access_legal_portal",
} | HARD_STOP_ACTIONS)

# ── Subtask types ─────────────────────────────────────────────────────────────

SUBTASK_OBSERVE_DOM = "agent_observe_dom"
SUBTASK_OBSERVE_SCREENSHOT = "agent_observe_screenshot"
SUBTASK_EXTRACT_DATA = "agent_extract_data"
SUBTASK_PLAN_NEXT_ACTION = "agent_plan_next_action"
SUBTASK_VALIDATE_ACTION = "agent_validate_action"
SUBTASK_SUMMARIZE_LOG = "agent_summarize_log"
SUBTASK_FINAL_REPORT = "agent_final_report"
SUBTASK_ORCHESTRATE = "agent_orchestrate"
SUBTASK_EVALUATE_GOAL = "agent_evaluate_goal"


# ── Request schemas ───────────────────────────────────────────────────────────

class BrowserAgentStartRequest(BaseModel):
    goal: str
    # Server-built reusable instructions. Kept separate from ``goal`` so skill
    # text never changes intent detection, URL resolution, or policy matching.
    skill_context: str = Field(default="", max_length=12_000)
    start_url: Optional[str] = None
    policy: str = POLICY_BALANCED
    speed_preference: int | None = Field(default=None, ge=0, le=100)
    quality_preference: int | None = Field(default=None, ge=0, le=100)
    max_steps: int = 20
    max_runtime_seconds: int = 300
    max_model_calls: int = 10
    max_cost_usd: Optional[float] = None
    allow_paid_calls: bool = False
    allow_external_navigation: bool = True
    allow_scripts: bool = False
    # Extra, caller-specified action types to gate behind approval, on top of
    # the built-in hard-stop list and permission_mode policy (D-044). This is
    # now an opt-in *addition*, not the primary safety mechanism — defaults to
    # empty because permission_mode + HARD_STOP_ACTIONS already cover the
    # default-safe behavior for every mode.
    approval_required_actions: list[str] = Field(default_factory=list)
    # Permission policy (D-044): controls how often the agent pauses for
    # approval. Hard-stop actions always require approval regardless of this
    # setting, in every runtime profile. Default matches the recommended
    # setting for both consumer_secure and personal_trusted (Part 6).
    permission_mode: str = DEFAULT_PERMISSION_MODE
    # Observation mode flags
    # real_observation_enabled=False → safe default: always use synthetic mock data.
    # Set True (+ Playwright installed) to observe a real webpage.
    real_observation_enabled: bool = False
    # capture_screenshot=True → include a screenshot on each observation.
    # Disabled by default; only activated when the user or DOM-insufficient flag
    # requires it. Never sent to the browser page context.
    capture_screenshot: bool = False
    # Real action backend flag (Part 7, updated).
    # real_action_enabled=False → all actions remain mocked (default, always safe).
    # Set True to opt in per-session; the server honors it only when
    # Playwright is installed (otherwise it downgrades to mock with a warning
    # — the toggle is never silently ignored).
    # Only low-risk actions are executed for real; approval-gated actions are
    # never auto-executed regardless of this flag.
    real_action_enabled: bool = False
    # Preferred live backend. "extension" is available through the unified
    # task API because it requires an explicit task-bound pairing handshake.
    browser_backend: Optional[str] = None


class BrowserAgentStepRequest(BaseModel):
    """
    Manually advance a session by N internal steps.

    Debug/developer use only (Part 2) — normal sessions run autonomously via
    /start, /approve, and /deny, which each drive the loop internally until
    a pause or terminal condition. This endpoint remains for developer
    tooling and tests; it is not part of the normal user-facing UX.
    """
    session_id: str
    steps: int = 1


class BrowserAgentStopRequest(BaseModel):
    session_id: str


class BrowserAgentApprovalRequest(BaseModel):
    session_id: str
    approved: bool = True
    reason: Optional[str] = None


class BrowserAgentDenyRequest(BaseModel):
    session_id: str
    reason: Optional[str] = None


# ── Action / observation schemas ──────────────────────────────────────────────

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PendingApproval(BaseModel):
    """
    Durable, blocking checkpoint (D-044 / Part 1). While a session has a
    PendingApproval, the runtime must not run observe/plan/validate again —
    see theriad/agents/browser_runtime.py run_step()'s guard at the top of
    the function and run_until_pause().
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str
    proposed_action: str
    target: Optional[str] = None
    current_url: Optional[str] = None
    reason: str
    risk_level: str  # low | medium | high | critical
    confidence: Optional[float] = None
    data_to_submit: Optional[dict[str, Any]] = None
    alternatives: Optional[list[str]] = None
    created_at: str = Field(default_factory=_utcnow_iso)
    originating_step_id: Optional[int] = None
    # D-045: set when the action cannot complete without user-provided input
    # that this release has no capture mechanism for (currently: login
    # credentials). When set, the UI must show a distinct "credentials
    # required" affordance instead of plain Approve/Deny — approving alone
    # cannot make the action succeed. See browser_runtime.resolve_pending_approval().
    required_input: Optional[str] = None       # e.g. "credentials"
    input_fields: Optional[list[str]] = None   # e.g. ["email", "password"]
    # D-045: fingerprint of the observation that produced this proposal,
    # captured at creation time. Used by the no-progress guard in
    # browser_runtime.py to detect "this identical action was already
    # executed against this identical page state" without re-observing.
    observation_fingerprint: Optional[str] = None


class StepDiagnostic(BaseModel):
    step_id: int
    session_id: str
    subtask_type: str
    tier_used: Optional[str] = None          # None when deterministic / no-model
    model_used: Optional[str] = None
    deterministic: bool = False
    input_tokens_estimate: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    observation_type: str = "none"           # dom | screenshot | both | none
    current_url: Optional[str] = None
    action_proposed: Optional[str] = None
    action_taken: Optional[str] = None
    confidence: Optional[float] = None
    risk_level: Optional[str] = None
    approval_required: bool = False
    blocked_by_policy: bool = False
    reason: Optional[str] = None
    error: Optional[str] = None
    # Part 7 additions
    pending_approval_id: Optional[str] = None
    approval_state: str = "none"             # none | pending | approved | denied
    permission_mode: Optional[str] = None
    loop_stop_reason: Optional[str] = None


class ActionLogEntry(BaseModel):
    """
    result values (D-044 + D-045):
      success              — auto-allowed action executed synchronously
                              (never used for an *approved* pending action —
                              see "executed" below)
      approval_pending     — paused, waiting for user decision
      blocked              — blocked by policy, or denied by user
      error                — execution raised/failed
      executed             — a previously-*approved* action was carried out
                              AND the mock/real world state actually changed
                              as a result. Never logged unless state changed.
      blocked_missing_input — approval was given but the action cannot
                              complete without input this release cannot
                              capture yet (e.g. login credentials)
      no_progress (as action_type, result="failed") — an approved+executed
                              action was immediately re-proposed against an
                              unchanged page state; the session stops rather
                              than looping.
    """
    step: int
    action_type: str
    target: Optional[str] = None
    url: Optional[str] = None
    result: str
    summary: Optional[str] = None


# ── Status response ───────────────────────────────────────────────────────────

class BrowserAgentStatus(BaseModel):
    session_id: str
    status: str
    goal: str
    speed_preference: int = 50
    quality_preference: int = 50
    current_url: Optional[str] = None
    steps_taken: int = 0
    model_calls: int = 0
    estimated_cost_usd: float = 0.0
    # ``cost_known`` prevents the UI from rendering an unknown price as $0.
    cost_known: bool = True
    # Grouped records of models that actually returned a response.
    models_used: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    last_observation: Optional[str] = None
    action_log: list[ActionLogEntry] = Field(default_factory=list)
    pending_approval: Optional[PendingApproval] = None
    final_result: Optional[str] = None
    final_answer: Optional[str] = None
    error: Optional[str] = None
    page_title: Optional[str] = None
    captcha_detected: bool = False
    captcha_reason: Optional[str] = None
    search_results: list[dict[str, Any]] = Field(default_factory=list)
    original_user_prompt: Optional[str] = None
    resolved_initial_url: Optional[str] = None
    navigation_reason: Optional[str] = None
    planner_next_action: Optional[str] = None
    # User-facing progress update for chat polling. This deliberately uses
    # plain language rather than exposing internal logger messages.
    progress_message: Optional[str] = None
    # Observation mode — safe to expose to the UI (no secrets)
    real_observation_enabled: bool = False
    # Part 6: action_backend and observation_backend replace the static
    # "Actions are mocked" label. action_backend is one of:
    #   "mock"       — all actions are simulated (safe default)
    #   "playwright" — low-risk actions execute via Playwright
    #   "chrome_mcp" — actions route through Chrome MCP (not yet implemented)
    action_backend: str = "mock"
    observation_backend: str = "mock"
    # Kept for backward compatibility; derived from action_backend.
    actions_mocked: bool = True
    # Diagnostics (per-step detail)
    step_diagnostics: list[StepDiagnostic] = Field(default_factory=list)
    # Runtime profile — always shown in agent panel (no secrets)
    runtime_profile: str = "consumer_secure"
    browser_profile_mode: str = "ephemeral"
    profile_badge: str = "Secure Mode — isolated browser and approval gates enabled."
    # Permission policy (D-044)
    permission_mode: str = DEFAULT_PERMISSION_MODE
    # Set when the autonomous loop stops for a reason other than a normal
    # approval pause (e.g. "max_internal_steps_exceeded", "no_progress",
    # "no_safe_alternative_after_denial"). None while running or paused
    # normally for approval.
    loop_stop_reason: Optional[str] = None
    # True when a visible, task-scoped Theriad browser is ready for the user
    # to complete a CAPTCHA or login before continuing.
    browser_handoff_ready: bool = False
