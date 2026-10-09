"""
Browser-agent execution loop.

In mock mode (default), all browser actions are simulated — no real browser,
no browser-use dependency.  The full pipeline (observation → planning →
policy → execution → logging) runs with synthetic data so sessions, step
logs, approval gates, and diagnostics all work without installing anything.

To switch to a real browser:
  1. pip install browser-use
  2. Pass mock_mode=False when calling run_step().
  3. Replace the stubs in browser_observation._real_observe().

Design:
- One step = one full pipeline pass (observe → plan → validate → execute).
- Screenshot/vision is never called by default (dom_was_insufficient required).
- Log summarization happens every N steps, not every step.
- Budget is checked before each step starts; session is stopped if exceeded.
- Sensitive actions pause the session with status=waiting_for_approval.
- Page content is treated as untrusted throughout.
"""
from __future__ import annotations

import asyncio
import re
import shutil
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from theriad.core.logging.logger import get_logger

from .browser_events import ActionProposal, ActionResult, ObservationEvent
from .browser_observation import observe_page
from .browser_policy import PolicyDecision, validate_action
from .browser_router import (
    SUMMARIZE_EVERY_N_STEPS,
    route_subtask,
    should_summarize,
)
from .browser_schemas import (
    ActionLogEntry,
    BrowserAgentStartRequest,
    PendingApproval,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_MANUAL_LOGIN_PAUSE,
    STATUS_NEEDS_APPROVAL,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_STOPPED_BUDGET,
    STATUS_WAITING_FOR_APPROVAL,
    STATUS_WAITING_FOR_INPUT,
    ALWAYS_AGNOSTIC_SAFE_ACTIONS,
    SUBTASK_OBSERVE_DOM,
    SUBTASK_OBSERVE_SCREENSHOT,
    SUBTASK_PLAN_NEXT_ACTION,
    SUBTASK_SUMMARIZE_LOG,
    SUBTASK_VALIDATE_ACTION,
    SUBTASK_FINAL_REPORT,
    SUBTASK_ORCHESTRATE,
    SUBTASK_EVALUATE_GOAL,
    StepDiagnostic,
)
from .browser_session_store import BrowserSession
from .llm_planner import (
    evaluate_goal as llm_evaluate_goal,
    execute_step as llm_execute_step,
    orchestrate_plan,
    summarize_observations as llm_summarize_observations,
    synthesize_answer as llm_synthesize_answer,
)

_rt_logger = get_logger("agents.browser_runtime")

# ── Autonomous-loop safety valves (D-044 / Part 1-2) ──────────────────────────
# Defensive ceiling on internal steps run inside a single run_until_pause()
# call, independent of the user-configured max_steps. Guarantees a single
# HTTP request (start/approve/deny) always returns even if a future bug in
# budget_exceeded() fails to terminate the loop.
MAX_INTERNAL_STEPS_PER_CALL = 200

# If this many consecutive action_log entries are blocked/error with no
# success in between, the loop is not making progress — stop instead of
# spinning silently.
NO_PROGRESS_THRESHOLD = 5

_TERMINAL_STATUSES = frozenset({STATUS_COMPLETED, STATUS_FAILED, STATUS_STOPPED, STATUS_STOPPED_BUDGET})

# D-045: statuses where the loop is durably paused waiting on something only
# a human (or a future credential-capture feature) can resolve. Both must be
# treated as stop conditions by run_until_pause() and as no-op guards at the
# top of run_step() — see the note on STATUS_WAITING_FOR_INPUT in
# browser_schemas.py for why this is a separate status from
# STATUS_WAITING_FOR_APPROVAL rather than reusing it.
# STATUS_MANUAL_LOGIN_PAUSE (Part 4): recoverable pause while the user logs in
# manually in the controlled browser. continue_after_manual_login() resumes it.
_PAUSE_STATUSES = frozenset({
    STATUS_WAITING_FOR_APPROVAL,
    STATUS_WAITING_FOR_INPUT,
    STATUS_MANUAL_LOGIN_PAUSE,
    STATUS_NEEDS_APPROVAL,
})

# Actions safe to execute for real (without approval) when real_action_enabled.
# Subset of ALWAYS_AGNOSTIC_SAFE_ACTIONS that are meaningful to execute with
# Playwright (i.e. have observable side effects on current_url / page state).
_REAL_EXECUTABLE_ACTIONS = frozenset({
    "navigate",
    "scroll",
    "click_link",
    "click_button",
    "go_back",
    "wait",
})


def _check_browser_use_available() -> bool:
    """Return True if browser-use is installed."""
    try:
        import importlib
        importlib.import_module("browser_use")
        return True
    except ImportError:
        return False


BROWSER_USE_AVAILABLE = _check_browser_use_available()


# ── CAPTCHA auto-solve helpers ─────────────────────────────────────────────────

def _html_has_captcha(html: str, url: str = "") -> bool:
    try:
        from theriad.search.captcha_solver import detect_captcha as _sc_detect
        return _sc_detect(html, url) is not None
    except ImportError:
        return False


def _stealth_bypass_fetch(
    url: str,
    user_data_dir: str | None = None,
) -> "str | None":
    """Fetch via Scrapling's native stealth/Cloudflare path in a worker thread."""
    try:
        from theriad.search.scrapling_provider import fetch_stealth_html
        return fetch_stealth_html(url, user_data_dir=user_data_dir)
    except Exception as exc:
        _rt_logger.debug("Scrapling stealth bypass failed: %s", exc)
    return None


def _task_browser_profile_dir(session: BrowserSession) -> str:
    """Create the private task-scoped profile shared by Playwright and Scrapling."""
    if session.browser_profile_dir:
        return session.browser_profile_dir
    profile_dir = Path.home() / ".theriad" / "browser-sessions" / session.session_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    session.browser_profile_dir = str(profile_dir)
    return session.browser_profile_dir


async def _close_and_cleanup_task_browser(session: BrowserSession) -> None:
    """Close a task browser and remove only its Theriad-owned profile."""
    if session.browser_bridge is not None:
        try:
            await session.browser_bridge.close_session(session.session_id)
        except Exception as exc:
            _rt_logger.debug("Could not close task browser: %s", exc)

    profile_dir = session.browser_profile_dir
    if not profile_dir:
        return
    profile_path = Path(profile_dir).expanduser().resolve()
    profile_root = (Path.home() / ".theriad" / "browser-sessions").resolve()
    # Never delete an arbitrary configured path. The task browser profile is
    # intentionally created under this fixed root and named by the UUID.
    if profile_path.parent != profile_root or profile_path.name != session.session_id:
        _rt_logger.warning("Refusing to clean unexpected browser profile path: %s", profile_path)
        return
    await asyncio.to_thread(shutil.rmtree, profile_path, ignore_errors=True)
    session.browser_profile_dir = None
    session.browser_visible = False
    session.browser_handoff_ready = False


async def _restart_live_browser(
    session: BrowserSession,
    *,
    visible: bool,
) -> "ObservationEvent | None":
    """Reopen the task browser after Scrapling updates its shared profile."""
    bridge = session.browser_bridge
    if bridge is not None:
        try:
            await bridge.close_session(session.session_id)
        except Exception as exc:
            _rt_logger.debug("Could not close browser before profile refresh: %s", exc)
    session.browser_visible = visible
    session.browser_handoff_ready = visible
    try:
        bridge = await _get_browser_bridge(session, mock_mode=False)
        result = await bridge.observe(session.session_id)
        if result.ok and result.observation is not None:
            return result.observation
    except Exception as exc:
        _rt_logger.debug("Could not reopen browser after CAPTCHA handling: %s", exc)
    session.browser_handoff_ready = False
    return None


async def _attempt_captcha_solve(
    observation: "ObservationEvent",
    session: "BrowserSession",
) -> "ObservationEvent | None":
    """Try to bypass a detected CAPTCHA without human intervention.

    Two-tier strategy:
    1. Stealth evasion — Scrapling camoufox / PlayWright stealth (no API key).
    2. CapSolver token injection (requires CAPTCHA_SOLVER_API_KEY).

    Returns a clean ObservationEvent on success, None when both tiers fail.
    """
    # Mock runs must remain deterministic and must never make optional network
    # calls merely because a fixture contains CAPTCHA-like text.
    if not session.real_observation_enabled:
        return None

    url = observation.url
    if not url:
        return None
    task_profile_dir = _task_browser_profile_dir(session)

    # Scrapling and the live Playwright browser must not hold the same profile
    # open concurrently. Close the live context, let Scrapling update the
    # task-scoped profile, then reopen and verify the actual page before
    # returning a solved observation.
    if session.browser_bridge is not None:
        try:
            await session.browser_bridge.close_session(session.session_id)
        except Exception as exc:
            _rt_logger.debug("Could not prepare browser profile for CAPTCHA solve: %s", exc)

    # ── Tier 1: stealth browser evasion ────────────────────────────────────────
    stealth_html = await asyncio.to_thread(
        _stealth_bypass_fetch,
        url,
        task_profile_dir,
    )
    if stealth_html and not _html_has_captcha(stealth_html, url):
        resumed = await _restart_live_browser(session, visible=False)
        if resumed is not None and not resumed.captcha_detected:
            _rt_logger.info("CAPTCHA bypassed via stealth browser at %s", url)
            return resumed

    # The verification reopen above may have confirmed that the current
    # browser still shows the challenge. Close it again before the optional
    # token solver reuses the same profile directory.
    if session.browser_bridge is not None:
        try:
            await session.browser_bridge.close_session(session.session_id)
        except Exception as exc:
            _rt_logger.debug("Could not prepare profile for token solver: %s", exc)

    # ── Tier 2: CapSolver ──────────────────────────────────────────────────────
    try:
        from theriad.search.captcha_solver import (
            CaptchaChallenge,
            CaptchaType,
            CaptchaSolver,
            detect_captcha as _sc_detect,
            solve_with_playwright,
        )
    except ImportError:
        _rt_logger.debug("theriad[captcha-solver] not installed; skipping CapSolver tier")
        await _restart_live_browser(session, visible=True)
        return None

    solver = CaptchaSolver.from_environment()
    if solver is None:
        await _restart_live_browser(session, visible=True)
        return None

    # Prefer HTML from the stealth fetch (richer) for challenge detection;
    # fall back to the observation's visible_text if stealth failed.
    html_for_detection = stealth_html or (observation.visible_text or "")
    challenge = _sc_detect(html_for_detection, url)
    if challenge is None:
        # Type unrecognised — try Cloudflare Turnstile (most common on web pages).
        challenge = CaptchaChallenge(
            type=CaptchaType.CLOUDFLARE_TURNSTILE, url=url, site_key=None,
        )

    _rt_logger.info(
        "Attempting CapSolver for %s (type=%s site_key=%s)",
        url, challenge.type.name, challenge.site_key,
    )
    injected_html = await asyncio.to_thread(
        solve_with_playwright,
        url,
        challenge,
        solver,
        user_data_dir=task_profile_dir,
    )
    if injected_html and not _html_has_captcha(injected_html, url):
        resumed = await _restart_live_browser(session, visible=False)
        if resumed is not None and not resumed.captcha_detected:
            _rt_logger.info("CAPTCHA solved via CapSolver at %s", url)
            return resumed

    # No clean live page was verified. Reopen a visible task-scoped browser so
    # the user can solve the remaining challenge without connecting personal
    # Chrome or exposing unrelated tabs.
    await _restart_live_browser(session, visible=True)
    return None


async def _get_browser_bridge(session: BrowserSession, *, mock_mode: bool):
    """Create and synchronize the session's browser bridge lazily."""
    from theriad.browser.bridge import BrowserBridge

    backend = "mock"
    for candidate in (session.action_backend, session.observation_backend):
        if candidate and candidate != "mock":
            backend = candidate
            break
    if mock_mode:
        backend = "mock"
    bridge = session.browser_bridge
    if bridge is None or bridge.backend != backend:
        bridge = BrowserBridge(backend=backend, observer=observe_page)
        session.browser_bridge = bridge
    user_data_dir = (
        _task_browser_profile_dir(session)
        if backend == "playwright" and not mock_mode
        else None
    )
    started = await bridge.ensure_session(
        session.session_id,
        start_url=session.current_url or session.start_url or "about:blank",
        user_prompt=session.goal,
        visible=session.browser_visible if backend == "playwright" else False,
        user_data_dir=user_data_dir,
    )
    if not started.ok:
        raise RuntimeError(
            started.error or (
                f"Browser backend '{backend}' could not start. "
                "For Playwright, run 'playwright install chromium'. "
                "For Electron, ensure the desktop app is running. "
                "For BrowserBase, verify BROWSERBASE_API_KEY is set."
            )
        )
    return bridge


def _prepare_episode_routing(session: BrowserSession, episode) -> None:
    if episode is None:
        return
    from theriad.routing.episode_hooks import baseline_models_for_provider
    from theriad.settings.secrets import get_active_assistant_provider

    provider = get_active_assistant_provider()
    episode.prepare("agents", baseline_models_for_provider({
        "browser_orchestrator": ("z-ai/glm-5.3", "premium"),
        "browser_executor": ("deepseek/deepseek-v4.1-flash", "cheap_code"),
        "browser_evaluator": ("nvidia/nemotron-3-super-120b-a12b:free", "strong_reasoning"),
        "browser_synthesizer": ("xiaomi/mimo-v2.6-flash", "cheap_bulk"),
        "browser_summarizer": ("nvidia/nemotron-3-nano-30b-a3b:free", "cheap_bulk"),
    }, provider), provider, session.goal)


async def run_step(session: BrowserSession, *, mock_mode: bool = True, routing_session=None) -> None:
    """Execute one step while recording any provider responses."""
    from .model_telemetry import capture_model_usage
    from theriad.routing.episode_hooks import episode_routing

    try:
        with capture_model_usage(session.model_usage), episode_routing(routing_session) as episode:
            _prepare_episode_routing(session, episode)
            if episode is not None:
                await episode.run_tool("browser.step", lambda: _run_step(session, mock_mode=mock_mode))
            else:
                await _run_step(session, mock_mode=mock_mode)
    finally:
        from .browser_session_store import get_session_store
        get_session_store().checkpoint_sync(session)


async def _run_step(session: BrowserSession, *, mock_mode: bool = True) -> None:
    """
    Execute one step of the browser-agent loop.

    Async because real observation uses Playwright's async API — the sync
    API cannot run inside FastAPI's event loop. Mock-only steps simply never
    hit an await point that blocks.

    Mutates session in place.  Raises no exceptions — errors are captured
    on the session object so the API can return them cleanly.

    Step pipeline:
      1. Budget check
      2. DOM observation (real Playwright or mock; never actions)
      3. Screenshot observation (only if DOM insufficient + capture_screenshot)
      4. Action planning (cheap_reasoning by default)
      5. Action validation (deterministic policy)
      6. Approval gate (pause if sensitive action)
      7. Action execution (live adapter when real_action_enabled; otherwise mock)
      8. Logging
      9. Periodic log summarization
     10. Goal-completion check

    Observation mode matrix:
      mock_mode=True  → always mock (default, no deps)
      mock_mode=False + session.real_observation_enabled=True + Playwright → real DOM read
      mock_mode=False + session.real_observation_enabled=False → mock
      mock_mode=False + no Playwright → STATUS_FAILED with clear error
    """
    from .browser_observation import PLAYWRIGHT_AVAILABLE

    # ── 0. Durable blocking checkpoint (Part 1) ───────────────────────────────
    # A session with an unresolved pending_approval must NEVER run
    # observe/plan/validate again. This is the fix for the permission-loop
    # bug: previously nothing stopped a caller from invoking run_step (or an
    # autonomous loop) again while paused, which would silently re-run the
    # full pipeline and re-propose the same sensitive action.
    #
    # Loop guard: if this is called again while still paused, do not append
    # a duplicate log entry every time — only append the terse waiting
    # message once, and only if it isn't already the most recent log line.
    if session.status == STATUS_WAITING_FOR_APPROVAL and session.pending_approval is not None:
        waiting_msg = f"Waiting for approval for {session.pending_approval.action_type} action."
        last = session.action_log[-1] if session.action_log else None
        if last is None or last.summary != waiting_msg:
            session.action_log.append(ActionLogEntry(
                step=session.steps_taken,
                action_type=session.pending_approval.action_type,
                target=session.pending_approval.target,
                url=session.current_url,
                result="approval_pending",
                summary=waiting_msg,
            ))
        return

    # D-045: same durable-checkpoint principle as above, for the "we cannot
    # proceed without user-provided input this release cannot capture yet"
    # dead end (currently: login credentials). pending_approval is already
    # None here (cleared in resolve_pending_approval) — status alone is the
    # guard. Without this, run_until_pause()/repeated run_step() calls would
    # re-run observe/plan and re-detect the same login form, recreating the
    # exact loop this fix exists to stop.
    if session.status == STATUS_WAITING_FOR_INPUT:
        waiting_msg = "Waiting for user-provided input; cannot continue without it."
        last = session.action_log[-1] if session.action_log else None
        if last is None or last.summary != waiting_msg:
            session.action_log.append(ActionLogEntry(
                step=session.steps_taken,
                action_type="login",
                target=None,
                url=session.current_url,
                result="blocked_missing_input",
                summary=waiting_msg,
            ))
        return

    # Part 4: Manual-login pause guard. The session is waiting for the user to
    # log in manually in the controlled browser and click "Continue". Do not
    # re-observe/plan while paused — continue_after_manual_login() clears this.
    if session.status == STATUS_MANUAL_LOGIN_PAUSE:
        waiting_msg = "Paused for manual login. Click 'Continue' after logging in."
        last = session.action_log[-1] if session.action_log else None
        if last is None or last.summary != waiting_msg:
            session.action_log.append(ActionLogEntry(
                step=session.steps_taken,
                action_type="login",
                target=None,
                url=session.current_url,
                result="manual_login_pause",
                summary=waiting_msg,
            ))
        return

    if session.status == STATUS_NEEDS_APPROVAL and session.captcha_detected:
        # CAPTCHA is a durable human handoff, never an autonomous retry signal.
        return

    step_start = time.monotonic()
    step_id = session.steps_taken + 1

    # Resolve effective observation mode for this step
    use_real_obs = (
        not mock_mode
        and session.real_observation_enabled
    )

    # ── 1. Budget check ───────────────────────────────────────────────────────
    budget_reason = session.budget_exceeded()
    if budget_reason:
        session.status = STATUS_STOPPED_BUDGET
        session.error = f"Budget exceeded: {budget_reason}"
        return

    # Dependency guard: real observation needs Playwright
    if use_real_obs and not PLAYWRIGHT_AVAILABLE:
        session.status = STATUS_FAILED
        session.error = (
            "Real browser observation requires Playwright. "
            "Install with: pip install playwright && playwright install chromium"
        )
        return

    # Legacy guard: old mock_mode=False path that tested for browser-use
    if not mock_mode and not session.real_observation_enabled and not BROWSER_USE_AVAILABLE:
        # Fall through to mock observation — do not fail
        pass

    session.status = STATUS_RUNNING

    # ── 2. DOM observation ────────────────────────────────────────────────────
    obs_route = route_subtask(SUBTASK_OBSERVE_DOM)
    obs_start = time.monotonic()

    effective_mock = not use_real_obs

    # D-045: one-shot override set by resolve_pending_approval() after a
    # state-changing action (e.g. simulated-successful login) so the next
    # observation deterministically reflects the new state instead of
    # relying on the new URL's hash happening to land on a different mock
    # template. Consumed (cleared) here regardless of outcome — it must
    # never leak into a later, unrelated observation.
    forced_page_type = session.mock_page_override
    session.mock_page_override = None

    try:
        bridge = await _get_browser_bridge(session, mock_mode=effective_mock)
        bridge_observation = await bridge.observe(
            session.session_id,
            forced_page_type=forced_page_type,
        )
        if not bridge_observation.ok or bridge_observation.observation is None:
            raise RuntimeError(
                bridge_observation.error or
                "The browser could not read the current page state — the browser process "
                "may have crashed or lost connection. Try restarting the session."
            )
        observation = bridge_observation.observation
    except Exception as exc:
        session.status = STATUS_FAILED
        session.error = (
            f"Page observation failed ({type(exc).__name__}: {exc}). "
            "The browser backend may be unreachable or the page took too long to respond."
        )
        return

    session.current_url = observation.url
    session.page_title = observation.title
    if observation.captcha_detected:
        solved_observation = await _attempt_captcha_solve(observation, session)
        if solved_observation is not None:
            # CAPTCHA cleared — swap in the fresh observation and continue this step.
            observation = solved_observation
            session.current_url = observation.url
            session.page_title = observation.title
            session.last_observation = _compress_observation(observation)
            _rt_logger.info(
                "CAPTCHA auto-solved; continuing agent loop for session %s",
                session.session_id,
            )
            session.action_log.append(ActionLogEntry(
                step=step_id,
                action_type="captcha",
                target=None,
                url=observation.url,
                result="success",
                summary="CAPTCHA bypassed automatically; continuing task.",
            ))
            # Fall through to action planning with the cleared observation.
        else:
            # Auto-solve failed — hand off to human.
            message = (
                "CAPTCHA encountered. Theriad tried automatic protection handling and opened a temporary "
                "browser window. Complete the CAPTCHA there, then click Continue."
                if session.browser_handoff_ready else
                "CAPTCHA encountered. Theriad tried automatic protection handling but could not clear this "
                "challenge. Use API search instead or complete it manually."
            )
            session.status = STATUS_NEEDS_APPROVAL
            session.captcha_detected = True
            session.captcha_reason = observation.captcha_reason
            session.final_answer = message
            session.final_result = message
            session.error = None
            session.last_observation = _compress_observation(observation)
            session.action_log.append(ActionLogEntry(
                step=step_id,
                action_type="captcha",
                target=None,
                url=observation.url,
                result="needs_approval",
                summary=message,
            ))
            session.steps_taken += 1
            return

    obs_latency = (time.monotonic() - obs_start) * 1000
    dom_text = _compress_observation(observation)
    # Prefix real observations so the UI can distinguish them
    if use_real_obs:
        dom_text = "[LIVE] " + dom_text
    session.last_observation = dom_text

    diag_obs = StepDiagnostic(
        step_id=step_id,
        session_id=session.session_id,
        subtask_type=SUBTASK_OBSERVE_DOM,
        tier_used=obs_route.tier,
        deterministic=True,
        observation_type="dom",
        current_url=observation.url,
        latency_ms=round(obs_latency, 1),
        reason="real" if use_real_obs else "mock",
    )
    session.step_diagnostics.append(diag_obs)

    # ── 3. Screenshot (only if DOM insufficient AND capture_screenshot) ───────
    dom_sufficient = _dom_is_sufficient(observation)
    want_screenshot = session.capture_screenshot or not dom_sufficient

    if want_screenshot:
        scr_route = route_subtask(
            SUBTASK_OBSERVE_SCREENSHOT,
            dom_was_insufficient=not dom_sufficient,
            policy=session.policy,
        )
        try:
            bridge = await _get_browser_bridge(session, mock_mode=effective_mock)
            bridge_screenshot = await bridge.screenshot(session.session_id)
            if not bridge_screenshot.ok or bridge_screenshot.observation is None:
                raise RuntimeError(bridge_screenshot.error or "Browser bridge screenshot failed")
            observation = bridge_screenshot.observation
            if observation.screenshot_description:
                dom_text = dom_text + "\n[Visual] " + observation.screenshot_description
                session.last_observation = dom_text
        except Exception:
            pass  # Screenshot optional; continue with DOM-only

        session.step_diagnostics.append(StepDiagnostic(
            step_id=step_id,
            session_id=session.session_id,
            subtask_type=SUBTASK_OBSERVE_SCREENSHOT,
            tier_used=scr_route.tier,
            observation_type="screenshot",
            current_url=observation.url,
        ))

    # ── 3b. Orchestrator: generate plan on first step (or replan) ──────────
    if not session.jspace.plan or session.jspace.replan_reason:
        session.progress_message = "Planning your task…"
        try:
            orch_start = time.monotonic()
            plan_steps = await orchestrate_plan(
                session.goal,
                session.jspace,
                skill_context=session.skill_context,
            )
            orch_latency = (time.monotonic() - orch_start) * 1000
            session.jspace.plan = plan_steps
            session.jspace.current_step_index = 0
            session.jspace.replan_reason = None
            session.model_calls += 1
            session.step_diagnostics.append(StepDiagnostic(
                step_id=step_id,
                session_id=session.session_id,
                subtask_type=SUBTASK_ORCHESTRATE,
                tier_used="premium",
                deterministic=False,
                latency_ms=round(orch_latency, 1),
                reason=f"Orchestrator generated {len(plan_steps)}-step plan.",
            ))
            _rt_logger.info("orchestrator plan ready", extra={
                "steps": len(plan_steps), "session": session.session_id,
            })
            session.progress_message = "Plan ready. Working through the task…"
        except Exception as exc:
            _rt_logger.warning(
                "Task planner encountered an error (%s: %s); using a simpler built-in "
                "fallback plan for this step.",
                type(exc).__name__, exc,
            )

    # Update J-Space with current observation
    session.jspace.add_observation(
        observation.url or "about:blank",
        (observation.visible_text or "")[:200],
    )

    # ── 4. Action planning ────────────────────────────────────────────────────
    prior_failures = sum(
        1 for e in session.action_log
        if e.result in ("error", "blocked")
    )
    plan_route = route_subtask(
        SUBTASK_PLAN_NEXT_ACTION,
        policy=session.policy,
        prior_failures=prior_failures,
        risk_level="low",
    )
    plan_start = time.monotonic()

    try:
        proposal = await llm_execute_step(
            observation,
            session.jspace,
            session.goal,
        )
    except Exception as exc:
        _rt_logger.warning(
            "AI action planner encountered an error (%s: %s); using the built-in "
            "fallback planner for this step.",
            type(exc).__name__, exc,
        )
        proposal = _mock_plan_action(observation, session)
    plan_latency = (time.monotonic() - plan_start) * 1000

    # Planner returned None: blank page, no content, and the goal names no
    # navigable target. Fail cleanly instead of reading about:blank forever.
    if proposal is None:
        session.status = STATUS_FAILED
        session.loop_stop_reason = "no_progress"
        session.error = (
            "I could not infer an initial website or search query from this "
            "request. Try naming a website or making the search target more specific."
        )
        session.final_result = f"Failed: {session.error}"
        session.action_log.append(ActionLogEntry(
            step=step_id,
            action_type="no_progress",
            target=None,
            url=observation.url,
            result="failed",
            summary=session.error,
        ))
        session.steps_taken += 1
        return

    session.planner_next_action = f"{proposal.action_type}({proposal.target or ''})"

    if proposal.action_type == "navigate":
        from theriad.search import (
            is_placeholder_url,
            is_unresolved_search_url,
            resolve_initial_navigation,
        )
        if is_placeholder_url(proposal.target) or is_unresolved_search_url(proposal.target):
            blocked_target = proposal.target
            navigation = resolve_initial_navigation(session.goal)
            proposal.target = navigation.resolved_url
            session.resolved_initial_url = navigation.resolved_url
            placeholder_blocked = is_placeholder_url(blocked_target)
            rewrite_prefix = (
                "Blocked placeholder navigation to example.com"
                if placeholder_blocked
                else "Replaced unresolved search homepage"
            )
            session.navigation_reason = (
                "Blocked placeholder navigation to example.com. "
                if placeholder_blocked
                else "Replaced unresolved search homepage with a query URL. "
            ) + navigation.reason
            session.action_log.append(ActionLogEntry(
                step=step_id,
                action_type="navigate",
                target=blocked_target,
                url=observation.url,
                result="blocked",
                summary=(
                    f"{rewrite_prefix}; corrected URL: {navigation.resolved_url}"
                ),
            ))
            session.planner_next_action = f"navigate({navigation.resolved_url})"

    # Planning counts as a model call (even in mock, tracks budget)
    session.model_calls += 1

    session.step_diagnostics.append(StepDiagnostic(
        step_id=step_id,
        session_id=session.session_id,
        subtask_type=SUBTASK_PLAN_NEXT_ACTION,
        tier_used=plan_route.tier,
        resolved_tier=plan_route.resolved_tier if hasattr(plan_route, 'resolved_tier') else None,
        deterministic=False,
        latency_ms=round(plan_latency, 1),
        action_proposed=f"{proposal.action_type}:{proposal.target}",
        confidence=proposal.confidence,
        risk_level=proposal.risk_level,
        approval_required=proposal.requires_approval,
        current_url=observation.url,
        reason=proposal.reason,
    ))

    # ── 5. Action validation (deterministic policy) ───────────────────────────
    policy_decision = validate_action(
        proposal,
        session.approval_required_actions,
        page_content=observation.visible_text,
        permission_mode=session.permission_mode,
    )

    val_diag = StepDiagnostic(
        step_id=step_id,
        session_id=session.session_id,
        subtask_type=SUBTASK_VALIDATE_ACTION,
        tier_used="no_model",
        deterministic=True,
        blocked_by_policy=policy_decision.blocked_by_policy,
        approval_required=policy_decision.requires_approval,
        reason=policy_decision.reason,
        current_url=observation.url,
        permission_mode=session.permission_mode,
        approval_state="pending" if policy_decision.requires_approval else "none",
    )
    session.step_diagnostics.append(val_diag)

    if policy_decision.blocked_by_policy:
        session.action_log.append(ActionLogEntry(
            step=step_id,
            action_type=proposal.action_type,
            target=proposal.target,
            url=observation.url,
            result="blocked",
            summary=policy_decision.reason,
        ))
        session.steps_taken += 1
        return

    # ── 6. Approval gate ──────────────────────────────────────────────────────
    if policy_decision.requires_approval:
        fingerprint = _observation_fingerprint(observation)
        no_progress_signature = (proposal.action_type, proposal.target, observation.url, fingerprint)

        # D-045, requirement 4: if this exact action was already EXECUTED
        # (state-changing, per resolve_pending_approval) against this exact
        # page state and is now being proposed again unchanged, the world
        # state provably did not move. This is defense-in-depth beyond the
        # mock_page_override mechanism that normally prevents this — it
        # catches the case where an action was marked "executed" but the
        # observation didn't actually change (a bug elsewhere, or a future
        # real-execution path that silently no-ops).
        if no_progress_signature in session.executed_action_signatures:
            session.status = STATUS_FAILED
            session.loop_stop_reason = "no_progress"
            session.error = (
                "The approved login action did not change the page state. "
                "Stopping to avoid a loop."
            )
            session.action_log.append(ActionLogEntry(
                step=step_id,
                action_type="no_progress",
                target=proposal.target,
                url=observation.url,
                result="failed",
                summary=session.error,
            ))
            session.steps_taken += 1
            return

        denial_signature = (proposal.action_type, proposal.target)
        if denial_signature in session.denied_actions:
            # Part 1, item 6b: the user already denied this exact action and
            # the deterministic planner has nothing else to propose given the
            # unchanged page state — there is no safe alternative. Stop
            # instead of re-pausing on the same denied action forever.
            session.status = STATUS_FAILED
            session.loop_stop_reason = "no_safe_alternative_after_denial"
            session.error = (
                f"No safe alternative to '{proposal.action_type}' after the user denied it. "
                f"Stopping instead of re-requesting the same action."
            )
            session.action_log.append(ActionLogEntry(
                step=step_id,
                action_type=proposal.action_type,
                target=proposal.target,
                url=observation.url,
                result="blocked",
                summary=session.error,
            ))
            session.steps_taken += 1
            return

        # D-045, requirement 3: for login specifically, tell the UI up front
        # (before any Approve/Deny click) that this cannot complete on
        # approval alone — there is no credential-capture mechanism yet.
        is_login = proposal.action_type == "login"
        session.pending_approval = PendingApproval(
            action_type=proposal.action_type,
            proposed_action=f"{proposal.action_type} → {proposal.target}",
            target=proposal.target,
            current_url=observation.url,
            reason=proposal.reason,
            risk_level=proposal.risk_level,
            confidence=proposal.confidence,
            originating_step_id=step_id,
            required_input="credentials" if is_login else None,
            input_fields=["email", "password"] if is_login else None,
            observation_fingerprint=fingerprint,
        )
        val_diag.pending_approval_id = session.pending_approval.id
        session.approval_state = "pending"
        session.status = STATUS_WAITING_FOR_APPROVAL
        session.action_log.append(ActionLogEntry(
            step=step_id,
            action_type=proposal.action_type,
            target=proposal.target,
            url=observation.url,
            result="approval_pending",
            summary=f"Paused: approval required for '{proposal.action_type}'",
        ))
        session.steps_taken += 1
        return

    if proposal.action_type == "navigate" and _navigation_target_already_attempted(session, proposal.target):
        session.status = STATUS_FAILED
        session.loop_stop_reason = "repeated_navigation"
        session.error = f"Stopped to avoid repeating navigation to {proposal.target}."
        session.final_result = f"Failed: {session.error}"
        session.action_log.append(ActionLogEntry(
            step=step_id,
            action_type="no_progress",
            target=proposal.target,
            url=observation.url,
            result="failed",
            summary=session.error,
        ))
        session.steps_taken += 1
        return

    # ── 7. Action execution ───────────────────────────────────────────────────
    # Route low-risk actions through the selected live BrowserBridge. A live
    # backend never falls back to a simulated success.
    # Approval-gated actions never reach this point (handled by step 6 above).
    use_real_action = (
        session.real_action_enabled
        and proposal.action_type in _REAL_EXECUTABLE_ACTIONS
    )
    if use_real_action:
        result = await _execute_bridge_action(proposal, observation, session)
    else:
        result = _mock_execute_action(proposal, observation, session)
    exec_latency = (time.monotonic() - step_start) * 1000

    if result.url:
        session.current_url = result.url

    session.step_diagnostics.append(StepDiagnostic(
        step_id=step_id,
        session_id=session.session_id,
        subtask_type="agent_execute_action",
        tier_used="no_model",
        deterministic=True,
        latency_ms=round(exec_latency, 1),
        action_taken=f"{result.action_type}:{result.target}",
        current_url=result.url or observation.url,
        error=result.error,
    ))

    # ── 8. Log action ─────────────────────────────────────────────────────────
    session.action_log.append(ActionLogEntry(
        step=step_id,
        action_type=result.action_type,
        target=result.target,
        url=result.url or observation.url,
        result="success" if result.success else "error",
        summary=result.result_summary,
    ))
    session.steps_taken += 1

    # ── 8b. Repeated-empty-read guard (Part 3 of the about:blank fix) ─────────
    # read_page with 0 chars on the same URL with the same observation hash is
    # not progress. More than 2 consecutive such steps → stop, don't spin to
    # max_steps and don't generate a fake final report.
    obs_fingerprint = _observation_fingerprint(observation)
    is_empty_read = (
        result.action_type == "read_page"
        and len(observation.visible_text or "") == 0
    )
    if (
        is_empty_read
        and observation.url == session.last_read_page_url
        and obs_fingerprint == session.last_read_page_fingerprint
    ):
        session.consecutive_empty_reads += 1
    else:
        session.consecutive_empty_reads = 1 if is_empty_read else 0
    if is_empty_read:
        session.last_read_page_url = observation.url
        session.last_read_page_fingerprint = obs_fingerprint

    if session.consecutive_empty_reads > 2:
        session.status = STATUS_FAILED
        session.loop_stop_reason = "no_progress"
        session.error = "Repeated empty read_page on unchanged page. Stopping to avoid loop."
        session.final_result = f"Failed: {session.error}"
        session.action_log.append(ActionLogEntry(
            step=step_id,
            action_type="no_progress",
            target=observation.url,
            url=observation.url,
            result="failed",
            summary=session.error,
        ))
        return

    # ── 9. Periodic log summarization ─────────────────────────────────────────
    if should_summarize(session.steps_taken):
        try:
            recent = [
                {"action_type": e.action_type, "target": e.target, "result": e.result, "summary": e.summary or ""}
                for e in session.action_log[-SUMMARIZE_EVERY_N_STEPS:]
            ]
            summary_text = await llm_summarize_observations(session.jspace, recent)
            session.jspace.add_finding(summary_text)
            session.step_diagnostics.append(StepDiagnostic(
                step_id=step_id,
                session_id=session.session_id,
                subtask_type=SUBTASK_SUMMARIZE_LOG,
                tier_used="cheap_bulk",
                deterministic=False,
                reason=f"LLM log summarization at step {step_id}.",
            ))
        except Exception:
            _mock_summarize_log(session, step_id)

    # Record action in J-Space
    session.jspace.record_action(
        result.action_type,
        result.target,
        "success" if result.success else (result.error or "error"),
    )

    # Advance J-Space plan step when appropriate
    current_jstep = session.jspace.current_step
    if current_jstep and not current_jstep.done and result.success:
        session.jspace.advance_step(result.result_summary)

    # ── 10. Goal completion check ─────────────────────────────────────────────
    goal_complete = False
    try:
        eval_result = await llm_evaluate_goal(
            session.goal,
            session.jspace,
            skill_context=session.skill_context,
        )
        goal_complete = eval_result.get("complete", False)
        if goal_complete:
            _rt_logger.info("evaluator says goal complete", extra={
                "reasoning": eval_result.get("reasoning", ""),
            })
            session.progress_message = "Task complete. Checking the results…"
        session.model_calls += 1
    except Exception:
        goal_complete = _goal_appears_complete(session)

    # Navigation is progress, not an answer. Evaluators can reasonably mark a
    # navigation step complete when they see a matching URL, but question and
    # research tasks must continue until the browser has produced usable
    # evidence for the final response.
    if (
        goal_complete
        and _goal_requires_answer(session.goal)
        and not _has_answer_evidence(session)
    ):
        goal_complete = False
        session.progress_message = "I reached a page. Reading it for an answer…"

    if goal_complete:
        if not _visited_non_blank_url(session):
            session.status = STATUS_FAILED
            session.loop_stop_reason = "no_progress"
            session.error = "Agent did not navigate away from about:blank."
            session.final_result = "Failed: the agent never navigated away from about:blank."
            session.final_answer = (
                "I couldn’t complete the browser task because the browser never "
                "reached a usable webpage."
            )
            session.action_log.append(ActionLogEntry(
                step=step_id,
                action_type="no_progress",
                target=session.current_url,
                url=session.current_url,
                result="failed",
                summary=session.error,
            ))
            return
        _mock_final_report(session, step_id)
        # D-050: opt-in LLM synthesis of the user-facing answer. Requires
        # allow_paid_calls=True + an OpenRouter key; any failure keeps the
        # deterministic template written by _mock_final_report above.
        if session.allow_paid_calls:
            session.progress_message = "Preparing your answer…"
            from .answer_synthesis import synthesize_browser_answer
            llm_answer = await synthesize_browser_answer(session)
            if llm_answer:
                session.final_answer = llm_answer
                session.step_diagnostics.append(StepDiagnostic(
                    step_id=step_id,
                    session_id=session.session_id,
                    subtask_type=SUBTASK_FINAL_REPORT,
                    tier_used="cheap_bulk",
                    deterministic=False,
                    reason="Final answer synthesized by model (D-050).",
                ))
                session.progress_message = "Answer ready."
        session.status = STATUS_COMPLETED


def resolve_pending_approval(session: BrowserSession, *, approved: bool, reason: str = "") -> None:
    """
    Resolve a session's pending_approval (Part 1, items 5-6; D-045 state-
    transition fix).

    Mutates session in place:
    - approved=True + action_type == "login": see _resolve_login_approval().
      Approving a login is NOT automatic success — see that function for why.
    - approved=True + any other action_type: execute the previously-proposed
      action now, log it as "executed" (D-045: never "success" for an
      approval-resolution outcome — see ActionLogEntry docstring), clear
      pending_approval, set status back to running.
    - approved=False: record the denial (so a future identical proposal is
                       recognized as "no safe alternative" instead of pausing
                       forever), log it, clear pending_approval, set status
                       back to running so the caller can resume the loop via
                       run_until_pause() and let the planner try again.

    This function does NOT itself continue the autonomous loop — callers
    (API handlers) are responsible for calling run_until_pause() afterward.
    Kept separate so tests can resolve an approval without also driving
    further steps.
    """
    pending = session.pending_approval
    if pending is None:
        return
    session.pending_approval = None

    if approved:
        if pending.action_type == "login":
            _resolve_login_approval(session, pending)
            return

        obs = ObservationEvent(
            url=pending.current_url or session.current_url or "about:blank",
            title="",
            observation_type="dom",
        )
        result = _mock_execute_action(
            ActionProposal(
                action_type=pending.action_type,
                target=pending.target,
                reason=pending.reason,
                confidence=pending.confidence or 0.5,
                risk_level=pending.risk_level,
                requires_approval=False,
            ),
            obs,
            session,
        )
        if result.url:
            session.current_url = result.url
        session.action_log.append(ActionLogEntry(
            step=session.steps_taken,
            action_type=pending.action_type,
            target=pending.target,
            url=result.url or obs.url,
            # D-045: "executed", never "success" — this is an approval
            # resolution, not an auto-allowed synchronous action.
            result="executed" if result.success else "error",
            summary=f"Approved by user: {result.result_summary}",
        ))
        session.approval_state = "approved"
        session.status = STATUS_RUNNING
    else:
        session.denied_actions.append((pending.action_type, pending.target))
        # D-045, requirement 5 explicitly allows either outcome after a
        # denied login ("login_blocked OR stopped"). We deliberately do NOT
        # force mock_page_override to "login_blocked" here: that template
        # has a clickable "Return to previous page" link, which would hand
        # the deterministic planner a seemingly-safe alternative action and
        # mask genuine no-progress loops instead of surfacing them. Denying
        # leaves the page/observation as-is; the existing denied_actions
        # guard below (run_step, Part 1 item 6b) is what actually stops the
        # session with "no_safe_alternative_after_denial" when the planner
        # has nothing else to propose against the unchanged page.
        session.action_log.append(ActionLogEntry(
            step=session.steps_taken,
            action_type=pending.action_type,
            target=pending.target,
            url=session.current_url,
            result="blocked",
            summary=f"Denied by user: {reason or 'no reason given'}",
        ))
        session.approval_state = "denied"
        # Resume to running — run_until_pause() will replan on the next
        # iteration. If the planner has nothing else to propose, run_step's
        # denied_actions check (Part 1, item 6b) will stop the session with
        # a clear explanation rather than re-pausing on the same action.
        session.status = STATUS_RUNNING


def _resolve_login_approval(session: BrowserSession, pending: PendingApproval) -> None:
    """
    D-045 core fix: approving a login action must not be treated as
    automatic success. This is the root of the original bug report —
    approval execution never mutated session/world state, so the next
    observation kept returning the same login page and the same login
    approval kept re-appearing.

    Login can only truly "execute" here if session.credentials_available is
    True. That flag is never set by any live API path in this release —
    there is no credential-capture endpoint yet — it exists solely so tests
    can exercise the "credentials were provided" branch. The honest default
    (False) means approving alone never pretends to succeed; per the user's
    explicit instruction, we stop cleanly instead.
    """
    login_url = pending.current_url or session.current_url or "about:blank"
    fingerprint = pending.observation_fingerprint or ""

    if session.credentials_available:
        new_url = _post_login_url(login_url, pending.target)
        session.current_url = new_url
        # One-shot: force the *next* observation to the authenticated-home
        # template instead of relying on new_url's hash landing somewhere
        # that happens to lack a login form.
        session.mock_page_override = "authenticated_home"
        # Record that this exact (action, target, url, page-state) was
        # actually executed, so the no-progress guard in run_step() can
        # detect a regression if it somehow gets proposed again unchanged.
        session.executed_action_signatures.add(
            (pending.action_type, pending.target, login_url, fingerprint)
        )
        session.action_log.append(ActionLogEntry(
            step=session.steps_taken,
            action_type="login",
            target=pending.target,
            url=new_url,
            result="executed",
            summary="Mock login completed; transitioned to authenticated page.",
        ))
        session.approval_state = "approved"
        session.status = STATUS_RUNNING
        return

    # No credentials available: do not execute, do not move current_url,
    # do not log success. Stop cleanly with an honest, specific reason.
    session.action_log.append(ActionLogEntry(
        step=session.steps_taken,
        action_type="login",
        target=pending.target,
        url=login_url,
        result="blocked_missing_input",
        summary="Credentials required; cannot continue without credentials.",
    ))
    session.approval_state = "blocked_missing_input"
    session.status = STATUS_WAITING_FOR_INPUT
    session.error = "Credentials required to continue."


def _post_login_url(login_url: str, target: Optional[str]) -> str:
    """
    D-045: deterministic "moved away from the login page" URL for the
    simulated-successful-login path. Strips a trailing /auth (the mock
    login form's submit target — see _mock_login_form in
    browser_observation.py) from whichever of login_url/target has it, then
    appends /home so the new URL is guaranteed to differ from the login
    page URL that produced the pending approval.
    """
    base = login_url
    if base.endswith("/auth"):
        base = base[: -len("/auth")]
    elif target and target.endswith("/auth"):
        base = target[: -len("/auth")]
    return base.rstrip("/") + "/home"


async def run_until_pause(session: BrowserSession, *, mock_mode: bool = True, routing_session=None) -> None:
    """Advance a session while capturing the models that actually respond."""
    from .model_telemetry import capture_model_usage
    from theriad.routing.episode_hooks import episode_routing

    try:
        with capture_model_usage(session.model_usage), episode_routing(routing_session) as episode:
            _prepare_episode_routing(session, episode)
            await _run_until_pause(session, mock_mode=mock_mode)
    finally:
        from .browser_session_store import get_session_store
        get_session_store().checkpoint_sync(session)


async def _run_until_pause(session: BrowserSession, *, mock_mode: bool = True) -> None:
    """
    Run the agent loop autonomously until a terminal state or an approval
    checkpoint is reached (Part 2/5). Async — see run_step().

    This is the ONLY way normal sessions should advance multiple steps.
    Manual per-step calls to run_step() remain available for developer/debug
    tooling (see BrowserAgentStepRequest) but are not part of the normal
    user-facing flow: Start Agent / Approve / Deny each call this function
    once to drive the session to its next natural stopping point.

    Stopping conditions (Part 2):
      completed, waiting_for_approval, stopped, failed, budget_exceeded
      (max_steps / max_runtime_seconds / max_model_calls / max_cost_usd —
      see BrowserSession.budget_exceeded()), max_internal_steps exceeded,
      no_progress detected.
    """
    internal_steps = 0
    while internal_steps < MAX_INTERNAL_STEPS_PER_CALL:
        if session.status in _TERMINAL_STATUSES or session.status in _PAUSE_STATUSES:
            return
        await run_step(session, mock_mode=mock_mode)
        internal_steps += 1

        if _no_progress_detected(session):
            session.status = STATUS_FAILED
            session.loop_stop_reason = "no_progress"
            session.error = (
                f"No progress detected: {NO_PROGRESS_THRESHOLD} consecutive "
                f"blocked/error actions with no successful action in between."
            )
            return

    # Safety valve: hit the internal loop cap without reaching a natural
    # pause/terminal state. Should not happen given max_steps/runtime/
    # model-call budgets, but this guarantees the request always returns.
    if session.status not in _TERMINAL_STATUSES and session.status not in _PAUSE_STATUSES:
        session.status = STATUS_STOPPED_BUDGET
        session.loop_stop_reason = "max_internal_steps_exceeded"
        session.error = f"max_internal_steps ({MAX_INTERNAL_STEPS_PER_CALL}) exceeded"


def _no_progress_detected(session: BrowserSession) -> bool:
    """
    True when the most recent NO_PROGRESS_THRESHOLD action_log entries are
    all blocked/error with no success in between — the loop is spinning
    without making progress toward the goal.
    """
    if len(session.action_log) < NO_PROGRESS_THRESHOLD:
        return False
    recent = session.action_log[-NO_PROGRESS_THRESHOLD:]
    return all(e.result in ("blocked", "error") for e in recent)


# ── Real action executor (Playwright, low-risk only) ─────────────────────────

async def _execute_bridge_action(
    proposal: ActionProposal,
    obs: ObservationEvent,
    session: BrowserSession,
) -> ActionResult:
    """Execute through a stateful live adapter and return verified state."""
    try:
        bridge = await _get_browser_bridge(session, mock_mode=False)
        if proposal.action_type == "navigate":
            verified = await bridge.navigate(session.session_id, proposal.target or obs.url)
        elif proposal.action_type in {"click_link", "click_button"}:
            verified = await bridge.click(
                session.session_id,
                proposal.target or "",
                action_type=proposal.action_type,
            )
        elif proposal.action_type == "scroll":
            verified = await bridge.scroll(session.session_id, proposal.target or 700)
        elif proposal.action_type == "wait":
            await asyncio.sleep(0.5)
            verified = await bridge.observe(session.session_id)
        else:
            return ActionResult(
                action_type=proposal.action_type,
                target=proposal.target,
                url=obs.url,
                success=False,
                error=f"Live action '{proposal.action_type}' is not implemented by BrowserBridge.",
                result_summary=f"Live backend did not execute {proposal.action_type}.",
            )
        if not verified.ok:
            return ActionResult(
                action_type=proposal.action_type,
                target=proposal.target,
                url=verified.url or obs.url,
                success=False,
                error=verified.error or "Live browser action failed.",
                result_summary=verified.error or "Live browser action failed.",
            )
        result = ActionResult(
            action_type=proposal.action_type,
            target=proposal.target,
            url=verified.url or obs.url,
            success=True,
            result_summary=f"Executed with verified {verified.backend} state at {verified.url or obs.url}.",
        )
        # Keep the legacy helper observable for compatibility tests, but only
        # provide it a result that came back from a real adapter.
        session._verified_bridge_action_result = result
        return _real_execute_action(proposal, obs, session)
    except Exception as exc:
        return ActionResult(
            action_type=proposal.action_type,
            target=proposal.target,
            url=obs.url,
            success=False,
            error=f"Live browser action failed: {exc}",
            result_summary=f"Live browser action failed: {exc}",
        )

def _real_execute_action(
    proposal: ActionProposal,
    obs: ObservationEvent,
    session: BrowserSession,
) -> ActionResult:
    """
    Execute a low-risk browser action for real (Part 1).

    Design constraints:
    - Only called for actions in _REAL_EXECUTABLE_ACTIONS.
    - Never submits forms, sends messages, purchases, uploads, or downloads.
    - Approval-gated actions never reach this function (run_step step 6).
    - Does not bypass browser_policy.py — policy runs before this function.
    - Falls back to mock result + logs a warning on any unexpected error.

    Implementation strategy (stateless-Playwright approach):
    - For navigate/click_link: update session.current_url; the next
      observe_page() call will load the new URL in a fresh Playwright context.
    - For scroll/wait: no URL change; return success with no side effects.
    - For go_back: pop session.url_history; update session.current_url.
    - For click_button: if a href is inferable, treat as navigate; otherwise
      return success with "no-op" (button state without persistent page is
      unknowable).

    This approach avoids shared persistent browser state between actions and
    observations, keeping the implementation conservative and side-effect-free.
    Chrome MCP integration (when fully wired through MCPBroker) would replace
    this with a richer, state-aware execution path.
    """
    verified = getattr(session, "_verified_bridge_action_result", None)
    if verified is not None:
        delattr(session, "_verified_bridge_action_result")
        return verified

    action_type = proposal.action_type
    target = proposal.target or ""
    return ActionResult(
        action_type=action_type,
        target=target,
        url=obs.url,
        success=False,
        error="No verified live-adapter result was supplied.",
        result_summary=f"Did not execute {action_type}; live verification is required.",
    )


# ── Manual-login pause/continue (Part 4) ──────────────────────────────────────

def pause_for_manual_login(session: BrowserSession) -> None:
    """
    Pause the session for manual login (Part 4).

    Sets STATUS_MANUAL_LOGIN_PAUSE so the autonomous loop stops and waits
    for the user to log in manually in the controlled browser, then call
    continue_after_manual_login() to resume.

    This is a user-controlled recoverable pause — distinct from
    STATUS_WAITING_FOR_INPUT (which is a dead end requiring credentials
    this release cannot capture).
    """
    session.status = STATUS_MANUAL_LOGIN_PAUSE
    session.action_log.append(ActionLogEntry(
        step=session.steps_taken,
        action_type="login",
        target=session.current_url,
        url=session.current_url,
        result="manual_login_pause",
        summary=(
            "Agent paused for manual login. "
            "Log in in the browser, then click 'Continue'."
        ),
    ))


def continue_after_manual_login(session: BrowserSession) -> None:
    """
    Resume the session after the user has logged in manually (Part 4).

    Clears STATUS_MANUAL_LOGIN_PAUSE, re-observes the page (via the
    normal run_step path on next call), and resumes the autonomous loop.
    The next run_step() will observe the post-login page state and plan
    accordingly — no assumption is made about what the page looks like now.
    """
    if session.status != STATUS_MANUAL_LOGIN_PAUSE:
        return
    session.action_log.append(ActionLogEntry(
        step=session.steps_taken,
        action_type="login",
        target=session.current_url,
        url=session.current_url,
        result="manual_login_continued",
        summary="User indicated login is complete. Resuming observation.",
    ))
    # Clear the pause; run_until_pause() caller drives the next steps.
    session.status = STATUS_RUNNING


# ── Goal → navigation bootstrap (Part 1 of the about:blank fix) ──────────────
# A goal like "Go to wikipedia and summarize AI" must produce a navigate
# action before any read_page — otherwise the agent reads about:blank forever
# and "completes" without doing anything.

# Known site names → canonical URLs. Deliberately tiny and deterministic;
# extend as needed. Checked only after explicit URLs and bare domains.
_KNOWN_SITES: dict[str, str] = {
    "wikipedia": "https://www.wikipedia.org/",
    "google": "https://www.google.com/",
    "zillow": "https://www.zillow.com/",
    "redfin": "https://www.redfin.com/",
    "realtor": "https://www.realtor.com/",
    "apartments": "https://www.apartments.com/",
    "github": "https://github.com/",
    "linkedin": "https://www.linkedin.com/",
    "youtube": "https://www.youtube.com/",
    "amazon": "https://www.amazon.com/",
}

_EXPLICIT_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
# Bare domain: at least one dot, TLD of 2+ letters (rejects "e.g", "i.e").
_BARE_DOMAIN_RE = re.compile(
    r"\b([a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)*\.[a-z]{2,})\b",
    re.IGNORECASE,
)

# URLs considered "blank" — the agent has not gone anywhere yet. Exact match
# only: mock URLs like "about:blank/result-1" are real (mock) destinations.
_BLANK_URLS = frozenset({"", "about:blank"})


def _is_navigable_http_url(url: str | None) -> bool:
    """Only absolute HTTP(S) links from page content are valid destinations."""
    parsed = urlsplit((url or "").strip())
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _navigation_target_already_attempted(session: BrowserSession, target: str | None) -> bool:
    """Prevent a successful-looking navigation cycle from consuming a budget."""
    return bool(target) and any(
        entry.action_type == "navigate"
        and entry.result == "success"
        and entry.target == target
        for entry in session.action_log[-8:]
    )

def _is_blank_url(url: Optional[str]) -> bool:
    """True when the URL means 'nowhere yet' (None, empty, about:blank)."""
    return url is None or url.strip() in _BLANK_URLS


def _extract_target_url(goal: str) -> Optional[str]:
    """
    Deterministically extract a navigation target from a user goal.

    Priority:
      1. Explicit http(s):// URL → used as-is (trailing punctuation stripped).
      2. Bare domain (e.g. "example.com") → https://<domain>
      3. Known site name (e.g. "wikipedia") → canonical URL.

    Returns None when the goal names no target — the caller decides whether
    to fail with a clear message or continue with the current page.
    """
    m = _EXPLICIT_URL_RE.search(goal)
    if m:
        return m.group(0).rstrip(".,;:!?)")

    m = _BARE_DOMAIN_RE.search(goal)
    if m:
        return f"https://{m.group(1).lower()}"

    goal_lower = goal.lower()
    for name, url in _KNOWN_SITES.items():
        if name in goal_lower:
            return url

    return None


def _visited_non_blank_url(session: BrowserSession) -> bool:
    """True if the session ever reached a URL other than about:blank."""
    if not _is_blank_url(session.current_url):
        return True
    return any(not _is_blank_url(e.url) for e in session.action_log)


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _compress_observation(obs: ObservationEvent) -> str:
    """Produce a compact text summary of a page observation."""
    lines = [f"URL: {obs.url}", f"Title: {obs.title}"]
    if obs.visible_text:
        lines.append(f"Text: {obs.visible_text[:300]}")
    if obs.clickable_elements:
        elem_strs = [e.get("text", e.get("tag", "?")) for e in obs.clickable_elements[:5]]
        lines.append(f"Elements: {', '.join(elem_strs)}")
    if obs.forms:
        lines.append(f"Forms: {len(obs.forms)} form(s) detected")
    return " | ".join(lines)


def _observation_fingerprint(obs: ObservationEvent) -> str:
    """
    D-045: cheap content fingerprint of an observation, used to detect
    "the page state did not actually change" independent of the URL string
    (a URL can be identical while content differs, or vice versa in real
    browsing — this keys on content, not just obs.url).
    """
    import hashlib
    raw = f"{obs.title}|{obs.visible_text}|{obs.forms}|{obs.clickable_elements}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _dom_is_sufficient(obs: ObservationEvent) -> bool:
    """Return True when DOM observation has enough content to plan from."""
    return bool(obs.visible_text or obs.clickable_elements)


def _mock_plan_action(
    obs: ObservationEvent,
    session: BrowserSession,
) -> Optional[ActionProposal]:
    """
    Simulate cheap_reasoning action planning.

    In real mode, this sends the compressed observation + goal to the
    cheap_reasoning model and parses the JSON response.

    Returns None only when the page is blank (about:blank, no content) and
    the goal names no navigable target — there is nothing sensible to do,
    and proposing read_page would just spin. The caller (run_step) turns
    None into a clean failure asking the user for a start URL.
    """
    # ── Goal → navigation bootstrap ───────────────────────────────────────────
    # On a blank page, a goal that names a site/URL ("go to wikipedia...",
    # "open example.com", "visit https://...") must navigate FIRST — never
    # fall through to read_page on about:blank.
    if _is_blank_url(obs.url):
        target = _extract_target_url(session.goal)
        if target:
            return ActionProposal(
                action_type="navigate",
                target=target,
                reason=f"Goal names a target site; navigating to {target} from a blank page.",
                confidence=0.9,
                risk_level="low",
                requires_approval=False,
            )

    # Check if any clickable element seems goal-relevant
    goal_lower = session.goal.lower()
    for elem in obs.clickable_elements:
        text = elem.get("text", "").lower()
        if any(kw in text for kw in goal_lower.split()[:3]):
            return ActionProposal(
                action_type="click_button",
                target=elem.get("text"),
                reason=f"Clicking '{elem.get('text')}' appears relevant to goal.",
                confidence=0.75,
                risk_level="low",
                requires_approval=False,
            )

    # Forms present? Could be login/signup — needs approval
    if obs.forms:
        form = obs.forms[0]
        inputs = form.get("inputs", [])
        has_password = any(i.get("type") == "password" for i in inputs)
        if has_password:
            return ActionProposal(
                action_type="login",
                target=form.get("action", obs.url),
                reason="Login form detected. Credentials required.",
                confidence=0.85,
                risk_level="high",
                requires_approval=True,
            )

    # Only follow an HTTP(S) link whose visible text overlaps the task. A
    # first-link fallback is unsafe: search pages can put mailto, ads, and
    # unrelated controls before meaningful results.
    keywords = {
        word for word in re.findall(r"[a-z0-9]{3,}", goal_lower)
        if word not in {"can", "you", "the", "and", "for", "with", "this", "that", "find", "good"}
    }
    for element in obs.clickable_elements:
        href = element.get("href", "")
        text = " ".join(str(element.get(key, "")) for key in ("text", "label", "aria")).lower()
        if _is_navigable_http_url(href) and keywords.intersection(re.findall(r"[a-z0-9]{3,}", text)):
            return ActionProposal(
                action_type="navigate",
                target=href,
                reason=f"Following task-relevant link: '{element.get('text', href)}'",
                confidence=0.7,
                risk_level="low",
                requires_approval=False,
            )

    # Preserve deterministic mock bootstrap behavior on about:blank. Real
    # browser sessions begin on a resolved URL, where arbitrary first links
    # remain forbidden by the rule above.
    if _is_blank_url(obs.url):
        for element in obs.clickable_elements:
            href = element.get("href", "")
            # Synthetic observations use relative/about:blank URLs; this path
            # is mock-only and never authorizes those schemes in Playwright.
            if (
                _is_navigable_http_url(href)
                or (href and not urlsplit(href).scheme)
                or href.startswith("about:blank/")
            ):
                return ActionProposal(
                    action_type="navigate",
                    target=href,
                    reason="Following safe mock bootstrap link.",
                    confidence=0.6,
                    risk_level="low",
                    requires_approval=False,
                )

    # ── read_page fallback ────────────────────────────────────────────────────
    # Acceptable on a real page with content. On a blank page with no content
    # and no target in the goal, there is nothing to read and nothing to do —
    # return None so run_step fails with a clear message instead of looping.
    if _is_blank_url(obs.url) and not obs.visible_text and not obs.clickable_elements:
        return None

    return ActionProposal(
        action_type="read_page",
        target=obs.url,
        reason="No clear next action; reading page content.",
        confidence=0.5,
        risk_level="low",
        requires_approval=False,
    )


def _mock_execute_action(
    proposal: ActionProposal,
    obs: ObservationEvent,
    session: BrowserSession,
) -> ActionResult:
    """Simulate executing a browser action."""
    if proposal.action_type == "navigate":
        new_url = proposal.target or obs.url
        return ActionResult(
            action_type="navigate",
            target=proposal.target,
            url=new_url,
            success=True,
            result_summary=f"Navigated to {new_url}",
        )

    if proposal.action_type in ("click_button", "click_link"):
        return ActionResult(
            action_type=proposal.action_type,
            target=proposal.target,
            url=obs.url,
            success=True,
            result_summary=f"Clicked '{proposal.target}'",
        )

    if proposal.action_type == "read_page":
        return ActionResult(
            action_type="read_page",
            target=obs.url,
            url=obs.url,
            success=True,
            result_summary=f"Read page: {obs.title} ({len(obs.visible_text)} chars)",
        )

    return ActionResult(
        action_type=proposal.action_type,
        target=proposal.target,
        url=obs.url,
        success=True,
        result_summary=f"Executed {proposal.action_type}",
    )


def _mock_summarize_log(session: BrowserSession, step_id: int) -> None:
    """Append a log-compression entry (cheap_bulk tier in production)."""
    recent = session.action_log[-SUMMARIZE_EVERY_N_STEPS:]
    actions_taken = [e.action_type for e in recent]
    session.step_diagnostics.append(StepDiagnostic(
        step_id=step_id,
        session_id=session.session_id,
        subtask_type=SUBTASK_SUMMARIZE_LOG,
        tier_used="cheap_bulk",
        deterministic=False,
        reason=f"Log compressed at step {step_id}. Actions: {', '.join(actions_taken)}",
    ))


def _mock_final_report(session: BrowserSession, step_id: int) -> None:
    """
    Generate final report (cheap_bulk tier in production).

    Truthfulness rules (Part 5 of the about:blank fix):
    - Never says "mock run" when the session's action backend is Playwright.
    - Reports the URLs actually visited (about:blank excluded) and the
      actions actually taken.
    - Only called for genuinely completed sessions — run_step's completion
      guard fails the session instead when no non-blank URL was ever visited.
    """
    # Deduplicate while preserving visit order; exclude about:blank.
    seen: set[str] = set()
    urls: list[str] = []
    for e in session.action_log:
        if e.url and not _is_blank_url(e.url) and e.url not in seen:
            seen.add(e.url)
            urls.append(e.url)
    actions = [e.action_type for e in session.action_log]
    backend_label = (
        "live actions via Playwright"
        if session.action_backend == "playwright"
        else "mock actions"
    )
    session.final_result = (
        f"Goal: {session.goal}\n"
        f"Steps taken: {session.steps_taken}\n"
        f"URLs visited: {', '.join(urls) or 'none'}\n"
        f"Actions: {', '.join(actions)}\n"
        f"Goal satisfied: yes — visited {urls[-1] if urls else 'n/a'}\n"
        f"Status: Completed ({backend_label})"
    )
    session.final_answer = _synthesize_browser_final_answer(session, urls)
    session.step_diagnostics.append(StepDiagnostic(
        step_id=step_id,
        session_id=session.session_id,
        subtask_type=SUBTASK_FINAL_REPORT,
        tier_used="cheap_bulk",
        deterministic=False,
        reason="Final report generated at goal completion.",
    ))


def _synthesize_browser_final_answer(
    session: BrowserSession,
    urls: list[str] | None = None,
) -> str:
    """Turn collected browser evidence into a concise user-facing answer."""
    visited = urls or []
    if not visited:
        seen: set[str] = set()
        for entry in session.action_log:
            if entry.url and not _is_blank_url(entry.url) and entry.url not in seen:
                seen.add(entry.url)
                visited.append(entry.url)

    # Placeholder entries (BrowserSearchProvider) are search *launch pages*
    # carrying a developer-facing config hint — never present them as findings.
    from theriad.search.service import is_placeholder_result
    real_results = [r for r in session.search_results if not is_placeholder_result(r)]
    if real_results:
        lines = ["Here are the most relevant results:"]
        for result in real_results[:5]:
            title = (result.get("title") or "Relevant result").strip()
            snippet = (result.get("snippet") or "").strip()
            url = (result.get("url") or "").strip()
            detail = f" — {snippet}" if snippet else ""
            source = f" ({url})" if url else ""
            lines.append(f"- {title}{detail}{source}")
        return "\n".join(lines)

    if session.last_observation:
        observation = session.last_observation.removeprefix("[LIVE] ").strip()
        if len(observation) > 600:
            observation = observation[:597].rstrip() + "…"
        if observation:
            source = visited[-1] if visited else session.current_url
            source_note = f"\n\nSource: {source}" if source else ""
            return f"I completed the browser task. {observation}{source_note}"

    if session.search_results:
        # Only the placeholder was collected and no page content was read:
        # be honest and link the results page instead of faking findings.
        url = (session.search_results[0].get("url") or "").strip()
        link = f" You can review the results here: {url}" if url else ""
        return (
            f"I ran a web search for “{session.goal}” but couldn’t extract "
            f"individual results in this environment.{link}"
        )

    if visited:
        title = session.page_title or "the requested page"
        return f"I completed the browser task and reached {title}.\n\nSource: {visited[-1]}"

    return (
        "I couldn’t complete the browser task because no usable page content "
        "or search results were collected."
    )


def _goal_appears_complete(session: BrowserSession) -> bool:
    """
    Heuristic: goal looks complete after a meaningful number of successful actions.
    Threshold is set above SUMMARIZE_EVERY_N_STEPS so log summarization always
    fires at least once before completion on a typical session.
    In production, the planner model makes this call.
    """
    if _goal_requires_answer(session.goal) and not _has_answer_evidence(session):
        return False

    from .browser_router import SUMMARIZE_EVERY_N_STEPS
    successes = sum(1 for e in session.action_log if e.result == "success")
    # Require more than one summarization cycle before declaring done
    threshold = min(SUMMARIZE_EVERY_N_STEPS + 2, session.max_steps)
    return successes >= threshold


def _goal_requires_answer(goal: str) -> bool:
    """Return whether a goal needs page/search evidence, not just navigation."""
    text = goal.lower().strip()
    if not text:
        return True

    answer_signals = (
        "find ", "search ", "look up", "what ", "what's", "when ", "where ",
        "which ", "how ", "why ", "summarize", "summary", "extract", "read ",
        "compare", "weather", "price", "cost", "list ", "answer", "tell me",
    )
    if any(signal in text for signal in answer_signals):
        return True

    # Explicit navigation-only requests are complete once the requested page
    # is reached. Everything else stays evidence-gated conservatively.
    return not bool(re.match(r"^(please\s+)?(go|open|visit|navigate)\b", text))


def _has_answer_evidence(session: BrowserSession) -> bool:
    """Whether the session has collected content suitable for an answer."""
    from theriad.search.service import is_placeholder_result

    if any(not is_placeholder_result(result) for result in session.search_results):
        return True
    if session.jspace.findings or session.jspace.extracted_data:
        return True

    # _compress_observation always includes URL/title, so require actual page
    # text rather than treating a successful observation shell as evidence.
    observation = (session.last_observation or "").removeprefix("[LIVE] ")
    text_marker = "Text:"
    if text_marker in observation:
        page_text = observation.split(text_marker, 1)[1].split(" | Elements:", 1)[0].strip()
        return bool(page_text)

    # A successful action log entry alone is not evidence: read_page can
    # succeed while extracting zero characters from a shell/blank page.
    return False
