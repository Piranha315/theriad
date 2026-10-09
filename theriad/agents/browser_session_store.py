"""
Durable browser-agent session store.

Design:
- JSON checkpoints restore pauses, budgets, trace, and final state after restart.
- Thread-safe for async FastAPI handlers via a simple dict + asyncio.Lock.
- Sessions are never deleted mid-run; they transition to terminal statuses.
- Secrets are never stored on the session object.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .browser_schemas import (
    ActionLogEntry,
    BrowserAgentStartRequest,
    BrowserAgentStatus,
    PendingApproval,
    STATUS_IDLE,
    STATUS_RUNNING,
    StepDiagnostic,
)
from .jspace import JSpace
from .model_telemetry import summarize_model_usage, usage_totals


class BrowserSession:
    """
    Mutable session state.  All fields are plain Python — no Pydantic overhead
    on the hot path.  Converted to BrowserAgentStatus only for API responses.
    """

    def __init__(
        self,
        req: BrowserAgentStartRequest,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> None:
        self.session_id: str = str(uuid.uuid4())
        # Authorization state is persisted but never returned in BrowserAgentStatus.
        self.owner_user_id = owner_user_id
        self.owner_organization_id = owner_organization_id
        self.goal: str = req.goal
        # Skills guide model-backed planning/synthesis only. Deterministic
        # permission and browser policy checks never read this field.
        self.skill_context: str = req.skill_context
        self.start_url: Optional[str] = req.start_url
        self.policy: str = req.policy
        from theriad.models.preferences import RoutingPreferences

        routing_preferences = RoutingPreferences.resolve(
            speed=req.speed_preference,
            quality=req.quality_preference,
            legacy_policy=req.policy,
        )
        self.speed_preference: int = routing_preferences.speed
        self.quality_preference: int = routing_preferences.quality
        self.max_steps: int = req.max_steps
        self.max_runtime_seconds: int = req.max_runtime_seconds
        self.max_model_calls: int = req.max_model_calls
        self.max_cost_usd: Optional[float] = req.max_cost_usd
        self.allow_paid_calls: bool = req.allow_paid_calls
        self.allow_external_navigation: bool = req.allow_external_navigation
        self.allow_scripts: bool = req.allow_scripts
        self.approval_required_actions: list[str] = list(req.approval_required_actions)
        self.permission_mode: str = req.permission_mode
        self.real_observation_enabled: bool = req.real_observation_enabled
        self.capture_screenshot: bool = req.capture_screenshot

        # ── Real action backend (Part 7) ──────────────────────────────────────
        # Resolved at creation time from runtime config + request flag.
        # The server enforces that real_action_enabled can only be True when
        # THERIAD_BROWSER_ACTION_BACKEND=playwright and Playwright is available.
        # Tests may set this directly on the session object to exercise the
        # real-action execution path without going through the API.
        self.real_action_enabled: bool = req.real_action_enabled
        self.requested_browser_backend: Optional[str] = req.browser_backend
        # Resolved backend label for UI display (Part 6).
        # Set after __init__ by the session factory (server.py) once Playwright
        # availability is confirmed. Defaults to "mock" (safe).
        self.action_backend: str = "mock"
        self.observation_backend: str = "mock"
        self.browser_bridge = None
        # A task-scoped Theriad-owned profile lets Scrapling's challenge
        # cookies flow back into the live browser without touching personal
        # Chrome. The path is internal and never returned in API status.
        self.browser_profile_dir: Optional[str] = None
        self.browser_visible: bool = False
        self.browser_handoff_ready: bool = False

        # URL history for go_back support (Part 1).
        # Populated every time current_url changes via a real action.
        self.url_history: list[str] = []

        # ── J-Space: shared context for multi-model agent coordination ────────
        self.jspace: JSpace = JSpace()

        self.status: str = STATUS_IDLE
        self.current_url: Optional[str] = req.start_url
        self.last_observation: Optional[str] = None
        self.action_log: list[ActionLogEntry] = []
        self.step_diagnostics: list[StepDiagnostic] = []
        self.steps_taken: int = 0
        self.model_calls: int = 0
        self.estimated_cost_usd: float = 0.0
        # One entry per provider response. This is separate from
        # ``model_calls`` because the latter also counts planned/retried calls
        # for budget safety, while the UI should report only actual usage.
        self.model_usage: list[dict[str, Any]] = []
        self.start_time: float = time.monotonic()
        self.pending_approval: Optional[PendingApproval] = None
        self.final_result: Optional[str] = None
        self.final_answer: Optional[str] = None
        self.error: Optional[str] = None
        self.page_title: Optional[str] = None
        self.captcha_detected: bool = False
        self.captcha_reason: Optional[str] = None
        self.search_results: list[dict] = []
        self.original_user_prompt: str = req.goal
        self.resolved_initial_url: Optional[str] = req.start_url
        self.navigation_reason: Optional[str] = None
        self.planner_next_action: Optional[str] = None
        self.progress_message: str = "Planning your task…"
        # D-044: durable approval bookkeeping and autonomous-loop guards.
        # approval_state mirrors the most recent resolution ("none" until a
        # PendingApproval has been created at least once).
        self.approval_state: str = "none"    # none | pending | approved | denied
        # (action_type, target) pairs the user has denied this session — used
        # to detect "no safe alternative" instead of re-proposing forever.
        self.denied_actions: list[tuple[str, Optional[str]]] = []
        # Set when the autonomous loop stops for a reason other than a normal
        # approval pause (Part 1/7).
        self.loop_stop_reason: Optional[str] = None

        # ── D-045: state-transition fixes for the approval-loop bug ──────────
        # Test-only simulation flag: this release has no real credential
        # capture UI/endpoint, so this is never set True by any live API path
        # — only tests set it directly on the session object to exercise the
        # "credentials were provided" branch. Approving a login action with
        # this False (the honest default) never pretends to succeed.
        self.credentials_available: bool = False
        # One-shot override consumed by the next observe_page() call in
        # run_step — lets an approved/denied action deterministically steer
        # the next mock observation (e.g. to "authenticated_home" after a
        # simulated-successful login) instead of relying on URL-hash luck.
        self.mock_page_override: Optional[str] = None
        # Signatures of (action_type, target, current_url, observation_fingerprint)
        # for actions that were actually EXECUTED (state-changing) after
        # approval. If the identical signature is proposed again, the mock
        # world state provably didn't change — stop instead of looping.
        self.executed_action_signatures: set[tuple[str, Optional[str], Optional[str], str]] = set()

        # ── Repeated-empty-read guard (about:blank loop fix, Part 3) ─────────
        # Tracks consecutive read_page steps that returned 0 chars on the
        # same URL with the same observation hash. run_step stops the session
        # (failed / no_progress) when this exceeds 2.
        self.consecutive_empty_reads: int = 0
        self.last_read_page_url: Optional[str] = None
        self.last_read_page_fingerprint: Optional[str] = None

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def to_status(self) -> BrowserAgentStatus:
        """Convert to API-safe status response (no secrets)."""
        from theriad.runtime.profile import get_runtime_config
        rc = get_runtime_config()
        action_backend = self.action_backend
        observation_backend = self.observation_backend
        usage_cost, cost_known = usage_totals(self.model_usage)
        if not self.model_usage:
            usage_cost = self.estimated_cost_usd
            cost_known = True
        return BrowserAgentStatus(
            session_id=self.session_id,
            status=self.status,
            goal=self.goal,
            speed_preference=self.speed_preference,
            quality_preference=self.quality_preference,
            current_url=self.current_url,
            steps_taken=self.steps_taken,
            model_calls=self.model_calls,
            estimated_cost_usd=round(usage_cost, 8) if usage_cost is not None else 0.0,
            cost_known=cost_known,
            models_used=summarize_model_usage(self.model_usage),
            elapsed_seconds=round(self.elapsed_seconds, 2),
            last_observation=self.last_observation,
            action_log=list(self.action_log),
            pending_approval=self.pending_approval,
            final_result=self.final_result,
            final_answer=self.final_answer,
            error=self.error,
            page_title=self.page_title,
            captcha_detected=self.captcha_detected,
            captcha_reason=self.captcha_reason,
            search_results=list(self.search_results),
            original_user_prompt=self.original_user_prompt,
            resolved_initial_url=self.resolved_initial_url,
            navigation_reason=self.navigation_reason,
            planner_next_action=self.planner_next_action,
            progress_message=self.progress_message,
            real_observation_enabled=self.real_observation_enabled,
            # Part 6: expose resolved backend labels.
            action_backend=action_backend,
            observation_backend=observation_backend,
            # actions_mocked: backward-compat derived field.
            actions_mocked=(action_backend == "mock"),
            step_diagnostics=list(self.step_diagnostics),
            runtime_profile=rc.runtime_profile.value,
            browser_profile_mode=rc.browser_profile_mode.value,
            profile_badge=rc.profile_badge,
            permission_mode=self.permission_mode,
            loop_stop_reason=self.loop_stop_reason,
            browser_handoff_ready=self.browser_handoff_ready,
        )

    def budget_exceeded(self) -> Optional[str]:
        """Return a reason string if any budget is exceeded, else None."""
        if self.steps_taken >= self.max_steps:
            return f"max_steps ({self.max_steps}) reached"
        if self.elapsed_seconds > self.max_runtime_seconds:
            return f"max_runtime_seconds ({self.max_runtime_seconds}s) exceeded"
        if self.model_calls >= self.max_model_calls:
            return f"max_model_calls ({self.max_model_calls}) reached"
        if self.max_cost_usd is not None and self.estimated_cost_usd >= self.max_cost_usd:
            return f"max_cost_usd (${self.max_cost_usd}) exceeded"
        return None


class BrowserSessionStore:
    """Thread-safe cache with optional atomic JSON checkpoints."""

    def __init__(self, session_dir: str | Path | None = None) -> None:
        self._sessions: dict[str, BrowserSession] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self.session_dir = Path(session_dir).expanduser() if session_dir else None

    def _path(self, session_id: str) -> Path:
        if self.session_dir is None:
            raise RuntimeError("browser session persistence is disabled")
        return self.session_dir / f"{session_id}.json"

    @staticmethod
    def _request_payload(session: BrowserSession) -> dict[str, Any]:
        return {
            "goal": session.goal,
            "skill_context": session.skill_context,
            "start_url": session.start_url,
            "policy": session.policy,
            "speed_preference": session.speed_preference,
            "quality_preference": session.quality_preference,
            "max_steps": session.max_steps,
            "max_runtime_seconds": session.max_runtime_seconds,
            "max_model_calls": session.max_model_calls,
            "max_cost_usd": session.max_cost_usd,
            "allow_paid_calls": session.allow_paid_calls,
            "allow_external_navigation": session.allow_external_navigation,
            "allow_scripts": session.allow_scripts,
            "approval_required_actions": session.approval_required_actions,
            "permission_mode": session.permission_mode,
            "real_observation_enabled": session.real_observation_enabled,
            "capture_screenshot": session.capture_screenshot,
            "real_action_enabled": session.real_action_enabled,
            "browser_backend": session.requested_browser_backend,
        }

    @staticmethod
    def _state_payload(session: BrowserSession) -> dict[str, Any]:
        def dump(value: Any) -> Any:
            if hasattr(value, "model_dump"):
                return value.model_dump()
            if isinstance(value, dict):
                return {str(key): dump(item) for key, item in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [dump(item) for item in value]
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            return None

        excluded = {"browser_bridge", "start_time", "jspace"}
        state = {
            key: dump(value)
            for key, value in session.__dict__.items()
            if key not in excluded
        }
        state["elapsed_seconds"] = session.elapsed_seconds
        state["jspace"] = session.jspace.to_dict()
        return state

    @classmethod
    def export_checkpoint(cls, session: BrowserSession) -> dict[str, Any]:
        return {
            "version": 1,
            "request": cls._request_payload(session),
            "state": cls._state_payload(session),
        }

    def checkpoint_sync(self, session: BrowserSession) -> None:
        if self._sessions.get(session.session_id) is not session:
            return
        payload = self.export_checkpoint(session)
        if self.session_dir is not None:
            self.session_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            target = self._path(session.session_id)
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        supervisor_task_id = getattr(session, "supervisor_task_id", None)
        if supervisor_task_id:
            from .task_engine import get_task_engine

            get_task_engine().checkpoint(
                str(supervisor_task_id),
                metadata={"browser_checkpoint": payload},
            )

    @staticmethod
    def _restore_payload(payload: dict[str, Any], session_id: str | None = None) -> Optional[BrowserSession]:
        try:
            if payload.get("version") != 1:
                return None
            session = BrowserSession(BrowserAgentStartRequest(**payload["request"]))
            state = payload["state"]
            for key, value in state.items():
                if key in {"elapsed_seconds", "jspace", "action_log", "step_diagnostics", "pending_approval"}:
                    continue
                setattr(session, key, value)
            session.session_id = session_id or str(state["session_id"])
            session.start_time = time.monotonic() - max(0.0, float(state.get("elapsed_seconds", 0)))
            session.jspace = JSpace.from_dict(dict(state.get("jspace") or {}))
            session.action_log = [ActionLogEntry(**row) for row in state.get("action_log", [])]
            session.step_diagnostics = [StepDiagnostic(**row) for row in state.get("step_diagnostics", [])]
            pending = state.get("pending_approval")
            session.pending_approval = PendingApproval(**pending) if pending else None
            session.denied_actions = [tuple(row) for row in state.get("denied_actions", [])]
            session.executed_action_signatures = {
                tuple(row) for row in state.get("executed_action_signatures", [])
            }
            session.browser_bridge = None
            if session.status == STATUS_RUNNING:
                session.status = STATUS_IDLE
                session.loop_stop_reason = "recovered_after_restart"
                session.progress_message = "Recovered from checkpoint; ready to resume."
            return session
        except (ValueError, TypeError, KeyError):
            return None

    def _load_sync(self, session_id: str) -> Optional[BrowserSession]:
        if self.session_dir is None:
            return None
        path = self._path(session_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return self._restore_payload(payload, session_id)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    async def restore_checkpoint(self, payload: dict[str, Any]) -> Optional[BrowserSession]:
        session = self._restore_payload(payload)
        if session is None:
            return None
        async with self._lock:
            self._sessions[session.session_id] = session
        return session

    async def create(
        self,
        req: BrowserAgentStartRequest,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> BrowserSession:
        session = BrowserSession(
            req,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
        async with self._lock:
            self._sessions[session.session_id] = session
            self.checkpoint_sync(session)
        return session

    async def get(
        self, session_id: str, *, owner_user_id: str | None = None
    ) -> Optional[BrowserSession]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = self._load_sync(session_id)
                if session is not None:
                    self._sessions[session_id] = session
            if session is not None and session.owner_user_id != owner_user_id and (
                session.owner_user_id is not None or owner_user_id is not None
            ):
                return None
            return session

    async def all_ids(self) -> list[str]:
        async with self._lock:
            ids = set(self._sessions)
            if self.session_dir and self.session_dir.exists():
                ids.update(path.stem for path in self.session_dir.glob("*.json"))
            return sorted(ids)

    def get_sync(self, session_id: str) -> Optional[BrowserSession]:
        """Synchronous get for use inside the execution loop (already locked)."""
        return self._sessions.get(session_id)


# Module-level singleton — shared across all API handlers in the same process.
_root = Path(__file__).resolve().parents[2]
_store = BrowserSessionStore(
    os.environ.get("THERIAD_BROWSER_SESSION_DIR", str(_root / "data" / "browser-sessions"))
)


def get_session_store() -> BrowserSessionStore:
    return _store
