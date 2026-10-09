"""Mock browser-agent session runtime for Theriad.

The public control model is start/approve/deny/stop. Step counters remain an
internal budget and diagnostics detail, never the normal user-facing UX.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from time import monotonic, time
from typing import Any

from .observation import BrowserObservation, is_captcha_observation
from .runtime_config import RuntimeConfig, current_config
from theriad.search import SearchService
from uuid import uuid4


class AgentStatus(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    # Kept as an enum alias for callers that imported the former name.  The
    # public contract has one approval status, regardless of its cause.
    NEEDS_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    BUDGET_EXCEEDED = "budget_exceeded"


APPROVAL_REQUIRED_ACTIONS = {
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
}


@dataclass
class BrowserAgentStartRequest:
    goal: str
    start_url: str | None = None
    policy: dict[str, Any] = field(default_factory=dict)
    max_steps: int = 25
    max_runtime_seconds: int = 120
    max_model_calls: int = 10
    max_cost_usd: float | None = None
    allow_paid_calls: bool = False
    allow_external_navigation: bool = False
    allow_scripts: bool = False
    approval_required_actions: list[str] = field(
        default_factory=lambda: sorted(APPROVAL_REQUIRED_ACTIONS)
    )


@dataclass
class ActivityItem:
    timestamp: float
    status: str
    message: str
    tool: str | None = None
    latency_ms: int | None = None
    cost_usd: float | None = None


@dataclass
class DiagnosticStep:
    internal_step_id: int
    subtask_type: str
    model_tier: str | None
    deterministic: bool
    observation_type: str
    proposed_action: str | None
    action_taken: str | None
    confidence: float
    risk_level: str
    approval_required: bool
    blocked_by_policy: bool
    latency_ms: int
    cost_usd: float
    runtime_profile: str = "consumer_secure"
    browser_profile_mode: str = "ephemeral"
    approvals_bypassed: bool = False
    error: str | None = None


@dataclass
class PermissionRequest:
    proposed_action: str
    target_site_page: str
    reason: str
    submitted_data: dict[str, Any] | None
    risk_level: str
    model_confidence: float
    alternatives: list[str]


@dataclass
class BrowserAgentSession:
    session_id: str
    request: BrowserAgentStartRequest
    runtime_config: RuntimeConfig = field(default_factory=current_config)
    status: AgentStatus = AgentStatus.STARTING
    current_url: str | None = None
    current_page_summary: str = "Browser-agent runtime is not wired yet. This is a mock session preview."
    activity_log: list[ActivityItem] = field(default_factory=list)
    diagnostics: list[DiagnosticStep] = field(default_factory=list)
    pending_approval: PermissionRequest | None = None
    final_result: str | None = None
    page_title: str | None = None
    screenshot: str | None = None
    captcha_detected: bool = False
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    cost_usd: float = 0.0
    internal_steps: int = 0
    started_monotonic: float = field(default_factory=monotonic)
    stopped: bool = False
    _last_tick: float = field(default_factory=monotonic)
    _phase: int = 0

    @property
    def elapsed_seconds(self) -> float:
        return round(monotonic() - self.started_monotonic, 2)

    def add_activity(self, status: str, message: str, **kwargs: Any) -> None:
        self.activity_log.append(ActivityItem(time(), status, message, **kwargs))
        # A live card needs recent context, not an unbounded session transcript.
        del self.activity_log[:-50]

    def add_diagnostic(
        self,
        subtask_type: str,
        observation_type: str,
        proposed_action: str | None,
        action_taken: str | None,
        confidence: float = 0.88,
        risk_level: str = "low",
        approval_required: bool = False,
        blocked_by_policy: bool = False,
        latency_ms: int = 120,
        cost_usd: float = 0.0,
        error: str | None = None,
    ) -> None:
        self.internal_steps += 1
        self.diagnostics.append(
            DiagnosticStep(
                internal_step_id=self.internal_steps,
                subtask_type=subtask_type,
                model_tier="mock-preview" if cost_usd else None,
                deterministic=cost_usd == 0,
                observation_type=observation_type,
                proposed_action=proposed_action,
                action_taken=action_taken,
                confidence=confidence,
                risk_level=risk_level,
                approval_required=approval_required,
                blocked_by_policy=blocked_by_policy,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                runtime_profile=self.runtime_config.runtime_profile.value,
                browser_profile_mode=self.runtime_config.browser_profile_mode.value,
                approvals_bypassed=not approval_required and (proposed_action not in self.runtime_config.approval_required_actions if proposed_action else False),
                error=error,
            )
        )
        self.cost_usd = round(self.cost_usd + cost_usd, 6)
        if cost_usd:
            self.model_calls += 1

    def enforce_budgets(self) -> bool:
        if self.internal_steps >= self.request.max_steps:
            self.status = AgentStatus.BUDGET_EXCEEDED
            self.final_result = "Stopped because the internal step budget was reached."
        elif self.elapsed_seconds >= self.request.max_runtime_seconds:
            self.status = AgentStatus.BUDGET_EXCEEDED
            self.final_result = "Stopped because the runtime budget was reached."
        elif self.model_calls >= self.request.max_model_calls:
            self.status = AgentStatus.BUDGET_EXCEEDED
            self.final_result = "Stopped because the model-call budget was reached."
        elif self.request.max_cost_usd is not None and self.cost_usd >= self.request.max_cost_usd:
            self.status = AgentStatus.BUDGET_EXCEEDED
            self.final_result = "Stopped because the cost budget was reached."
        else:
            return False
        self.add_activity(self.status.value, self.final_result)
        return True

    def tick(self) -> None:
        if self.status not in {AgentStatus.STARTING, AgentStatus.RUNNING}:
            return
        if self.stopped:
            self.status = AgentStatus.STOPPED
            self.final_result = "Session stopped by user."
            return
        if monotonic() - self._last_tick < 0.6:
            return
        self._last_tick = monotonic()
        if self.enforce_budgets():
            return
        self.status = AgentStatus.RUNNING
        script = [
            ("planning", "Planning task", "goal"),
            ("navigation", "Opening page", "page"),
            ("observation", "Reading visible page content", "visible_text"),
            ("analysis", "Found relevant result", "summary"),
        ]
        if self._phase < len(script):
            subtask, message, obs = script[self._phase]
            self.add_activity("running", message, tool="mock-browser")
            self.add_diagnostic(subtask, obs, None, message.lower(), cost_usd=0.0002 if subtask == "analysis" else 0.0)
            if self._phase == 1:
                self.current_url = self.request.start_url or "about:blank"
            self._phase += 1
            return
        if "submit_form" in self.runtime_config.approval_required_actions:
            self.status = AgentStatus.WAITING_FOR_APPROVAL
            self.pending_approval = PermissionRequest(
                proposed_action="submit_form",
                target_site_page=self.current_url or self.request.start_url or "current page",
                reason="Submitting information is a sensitive browser action and requires explicit user approval.",
                submitted_data={"query": self.request.goal},
                risk_level="medium",
                model_confidence=0.81,
                alternatives=["Summarize findings without submitting the form", "Stop the session"],
            )
            self.add_activity("waiting_for_approval", "Permission required before submitting form")
            self.add_diagnostic("action", "form", "submit_form", None, risk_level="medium", approval_required=True)
            return
        self.complete()

    def complete(self) -> None:
        self.status = AgentStatus.COMPLETED
        self.final_result = "Mock browser-agent session completed. Browser-agent runtime is not wired yet. This is a mock session preview."
        self.pending_approval = None
        self.add_activity("completed", "Completed mock browser-agent session")

    def observe_page(self, observation: BrowserObservation) -> bool:
        """Record a page observation and hard-stop if it is a CAPTCHA."""

        self.current_url = observation.current_url
        self.page_title = observation.page_title
        self.screenshot = observation.screenshot
        if not is_captcha_observation(observation):
            self.current_page_summary = observation.visible_text[:500]
            return False

        self.captcha_detected = True
        self.status = AgentStatus.WAITING_FOR_APPROVAL
        self.final_result = (
            "CAPTCHA encountered. Complete it manually or switch to API search. "
            f"Page: {observation.page_title or 'Untitled page'} "
            f"({observation.current_url})."
        )
        self.pending_approval = PermissionRequest(
            proposed_action="captcha_handoff",
            target_site_page=observation.current_url,
            reason="A CAPTCHA requires human interaction and cannot be bypassed autonomously.",
            submitted_data=None,
            risk_level="medium",
            model_confidence=1.0,
            alternatives=["Use API search instead", "Stop task"],
        )
        self.add_activity("waiting_for_approval", self.final_result)
        self.add_diagnostic(
            "observation",
            "captcha",
            "captcha_handoff",
            None,
            approval_required=True,
            blocked_by_policy=True,
        )
        return True

    def approve(self) -> None:
        if self.status != AgentStatus.WAITING_FOR_APPROVAL:
            return
        if self.captcha_detected:
            self.captcha_detected = False
            self.pending_approval = None
            self.status = AgentStatus.RUNNING
            self.final_result = None
            self.add_activity("running", "User completed the CAPTCHA; continuing execution")
            return
        self.add_activity("running", "User approved sensitive action; continuing execution")
        self.add_diagnostic("approval", "user_decision", self.pending_approval.proposed_action if self.pending_approval else None, "approved")
        self.pending_approval = None
        self.status = AgentStatus.RUNNING
        self.complete()

    def deny(self) -> None:
        if self.status != AgentStatus.WAITING_FOR_APPROVAL:
            return
        self.add_activity("stopped", "User denied sensitive action; no sensitive action was performed")
        self.add_diagnostic("approval", "user_decision", self.pending_approval.proposed_action if self.pending_approval else None, "denied", blocked_by_policy=True)
        self.pending_approval = None
        self.status = AgentStatus.STOPPED
        self.final_result = "User denied permission. The agent did not perform the sensitive action and stopped safely."

    def stop(self) -> None:
        self.stopped = True
        self.status = AgentStatus.STOPPED
        self.pending_approval = None
        self.final_result = "Session stopped by user."
        self.add_activity("stopped", self.final_result)

    def use_api_search(self) -> None:
        """Leave the CAPTCHA page and continue via the API-first search path."""

        if self.status != AgentStatus.WAITING_FOR_APPROVAL or not self.captcha_detected:
            return
        self.captcha_detected = False
        self.pending_approval = None
        self.add_activity(
            "running",
            "Switched to API search; the CAPTCHA page will not be retried.",
            tool="search-service",
        )
        try:
            results = SearchService().search(self.request.goal)
        except Exception as exc:
            self.status = AgentStatus.FAILED
            self.final_result = f"API search failed: {str(exc).strip() or exc.__class__.__name__}"
            self.add_activity("failed", self.final_result, tool="search-service")
            return

        self.artifacts = [
            {"type": "search_result", **result.to_dict()} for result in results
        ]
        self.status = AgentStatus.COMPLETED
        useful = [result for result in results if result.url]
        if useful:
            summary = "; ".join(
                f"{result.title}: {result.url}" for result in useful[:5]
            )
            self.final_result = f"API search results: {summary}"
        else:
            self.final_result = (
                results[0].snippet if results else "API search returned no results."
            )
        self.add_activity("completed", self.final_result, tool="search-service")

    def to_dict(self, include_diagnostics: bool = True) -> dict[str, Any]:
        self.tick()
        data = asdict(self)
        data["status"] = self.status.value
        data["final_answer"] = self.final_result
        data["elapsed_seconds"] = self.elapsed_seconds
        data["runtime_profile"] = self.runtime_config.runtime_profile.value
        data["browser_profile_mode"] = self.runtime_config.browser_profile_mode.value
        data["real_chrome_profile_enabled"] = self.runtime_config.allow_real_chrome_profile and self.runtime_config.browser_profile_mode.value == "real_profile"
        data["mcp_tools_enabled"] = self.runtime_config.mcp_allowlist
        data.pop("started_monotonic", None)
        data.pop("stopped", None)
        data.pop("_last_tick", None)
        data.pop("_phase", None)
        if not include_diagnostics:
            data.pop("diagnostics", None)
        return data


class BrowserAgentStore:
    def __init__(self) -> None:
        self._sessions: dict[str, BrowserAgentSession] = {}

    def start(self, request: BrowserAgentStartRequest, runtime_config: RuntimeConfig | None = None) -> BrowserAgentSession:
        runtime_config = runtime_config or current_config()
        request.approval_required_actions = sorted(runtime_config.approval_required_actions)
        session = BrowserAgentSession(str(uuid4()), request, runtime_config, current_url=request.start_url)
        session.add_activity("starting", "Started browser-agent session")
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> BrowserAgentSession:
        return self._sessions[session_id]

    @property
    def sessions(self) -> dict[str, BrowserAgentSession]:
        """Read-only-by-convention compatibility view for older chat callers."""
        return self._sessions


STORE = BrowserAgentStore()
