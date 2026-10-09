"""
Theriad API Server — FastAPI bridge between the Theriad UI and the Theriad engine.

Run with:
    uvicorn theriad.api.server:app --reload --port 8000

Endpoints:
    GET  /api/status    Provider connectivity + active config (no live ping)
    POST /api/route     Dry-run route analysis — returns tier, model, cost, routing path (no provider calls)
    POST /api/execute   Real execution — routes + calls provider + records metrics
    POST /api/simulate  Offline batch simulation (no provider calls ever)
    GET  /api/models    Live tier registry from DEFAULT_REGISTRY
    GET  /api/pricing   Per-model prices from the live OpenRouter catalog + freshness
    GET  /api/pricing/plans  OpenRouter-first self-serve and enterprise pricing policy
    GET  /api/stats     Aggregate stats from ~/.theriad/metrics.jsonl
    GET  /api/logs      Recent inference records (last N, most recent first)

OpenAI-compatible endpoints (usable as a custom provider in Codex, Continue, etc.):
    GET  /v1/models              Lists theriad/* virtual models
    POST /v1/chat/completions    Routes + executes; returns OpenAI-format response

Browser-agent endpoints (separate from /api/execute):
    POST /api/agent/browser/start    Start a session; runs autonomously until
                                      pause/terminal (D-044, Part 5)
    POST /api/agent/browser/approve  Approve a pending action; resumes the loop
    POST /api/agent/browser/deny     Deny a pending action; resumes the loop
    POST /api/agent/browser/stop     Stop a session
    GET  /api/agent/browser/status   Get session status
    POST /api/agent/browser/step     Developer/debug: manually advance N
                                      internal steps (not normal UX)

Design:
- CORS enabled for Vite dev server (localhost:5173).
- All heavy imports are deferred inside handlers to keep startup fast.
- /api/route is strictly dry-run — no provider calls, ever.
- /api/execute requires allow_paid_calls=true; refuses otherwise with HTTP 403.
- Graceful empty-state responses when metrics.jsonl doesn't exist yet.
- Browser-agent is fully isolated from /api/execute — normal execution is
  unchanged. The router may recommend browser_agent but never auto-runs it.
"""
from __future__ import annotations

from math import isfinite
from typing import Any

from pathlib import Path as _Path
from dotenv import load_dotenv
_project_root = _Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")  # always finds the project-root .env regardless of cwd

from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from theriad.core.logging.logger import get_logger
from theriad.models.registry import DEFAULT_BASELINE_MODEL
from theriad.storage.hosted import hosted_data_plane_enabled, hosted_repositories
from theriad.accounting.pricing import platform_fee

app = FastAPI(title="Theriad API", version="0.1.0")
logger = get_logger("api.server")

# ── Module-level sticky route cache (per-process, in-memory) ─────────────────
# Exposed at module level so tests can monkeypatch or inspect it directly.
from theriad.routing.sticky import StickyRouteCache  # noqa: E402
_sticky_cache = StickyRouteCache()

# Short-lived, one-time OpenRouter PKCE verifiers. Production multi-worker
# deployments should move these ephemeral records to a shared TTL store.
_openrouter_oauth_flows: dict[str, tuple[str, float, str | None]] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-user auth (guest trial accounts + registered users)
from theriad.api.auth import (  # noqa: E402
    auth_router,
    get_request_user,
    is_hosted_deployment,
    is_public_api_path,
    reset_request_scope,
    set_request_scope,
    user_scope_id,
)
app.include_router(auth_router)

# Hosted deployments use one canonical middleware boundary for identity,
# tenant storage readiness, and disabled real-agent execution. Keep the small
# response helper local because body-aware API search is admitted by its route
# handler before it reaches the real-agent guard.
def _hosted_agent_unavailable_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": "hosted_agent_executor_unavailable",
            "detail": (
                "Agent execution is not enabled for this hosted deployment. "
                "Configure an isolated executor before enabling browser or terminal actions."
            ),
        },
        headers={"Cache-Control": "no-store"},
    )


# Hosted deployments must opt into a strict boundary explicitly. Local and
# development mode retain the legacy anonymous scope for the CLI and existing
# local workflows; hosted mode exposes only the public allowlist without a
# valid Theriad bearer session.
from theriad.api.auth_context import hosted_auth_middleware  # noqa: E402
app.middleware("http")(hosted_auth_middleware)
from theriad.settings.runtime import (  # noqa: E402
    execution_budget_error as _execution_budget_error,
    get_runtime_settings as _get_runtime_settings,
    update_runtime_settings as _update_runtime_settings,
)

# Anonymous collection is public so the landing page can be measured; the
# summary route inside this router is owner-gated.
from theriad.api.analytics import router as analytics_router  # noqa: E402
app.include_router(analytics_router)

# Stripe billing is optional. The router exposes an explicit policy-only state
# until real Stripe credentials, prices, and webhook verification are present.
from theriad.api.billing import (  # noqa: E402
    entitlements_router,
    router as billing_router,
)
app.include_router(billing_router)
app.include_router(entitlements_router)

from theriad.api.audit import router as audit_router  # noqa: E402
app.include_router(audit_router)

# Mount project/conversation/file/run routes
from theriad.api.projects import projects_router  # noqa: E402
app.include_router(projects_router)

# Reusable prompt-skill CRUD and safe package import routes.
from theriad.api.skills import router as skills_router  # noqa: E402
app.include_router(skills_router)

# Unified task/tool/extension contract. Existing browser and chat routes remain
# compatibility surfaces while clients migrate to /api/agent/tasks.
from theriad.api.tasks import router as agent_tasks_router  # noqa: E402
app.include_router(agent_tasks_router)

from theriad.agents.task_engine import ToolCall, get_task_engine  # noqa: E402
from theriad.agents.task_store import TaskStoreError  # noqa: E402


@app.exception_handler(TaskStoreError)
async def _task_store_unavailable(_request: Request, exc: TaskStoreError):
    return JSONResponse(status_code=503, content={"error": str(exc)})
# The old ``theriad_browser`` mock session router is intentionally not mounted.
# Canonical browser routes are defined below and backed by OpenSandbox.

# Harness integration API — OpenAI-compatible /v1 endpoint
from theriad.api.harness import router as _harness_router  # noqa: E402
app.include_router(_harness_router)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _estimate_prompt_tokens(prompt: str) -> int:
    return max(1, int(len(prompt.split()) * 1.33))


def _record_routing_usage(
    request: Request,
    *,
    run_id: str | None,
    provider: str | None,
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    provider_cost_usd: float | None,
    paid_request: bool,
    usage_known: bool | None = None,
) -> None:
    """Write the tenant usage ledger without making billing a request blocker."""
    if not run_id:
        return
    try:
        from theriad.api.auth_context import get_auth_context
        from theriad.billing.service import record_usage_event

        context = get_auth_context(request)
        record_usage_event(
            organization_id=context.organization_id,
            user_id=context.user_id,
            run_id=run_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_cost_usd=provider_cost_usd,
            billable=context.principal_type == "registered" and paid_request,
            usage_known=(
                input_tokens is not None and output_tokens is not None
                if usage_known is None else usage_known
            ),
            metadata={"deployment_mode": context.deployment_mode},
        )
    except Exception as exc:  # pragma: no cover - billing must not break a response
        logger.warning("routing usage ledger write failed", extra={"error": str(exc)})


def _user_content(prompt: str, images: list[tuple[str, str]]):
    """Build the user message content, multimodal only when images are attached.

    Returns the plain string when there are no images, so the overwhelmingly
    common text-only path produces byte-identical payloads to before.

    Each image is preceded by a text block naming it, so a prompt like "compare
    these two" still has something to refer to; without the filenames the model
    sees an unlabelled pile of pixels.
    """
    if not images:
        return prompt
    blocks: list[dict] = [{"type": "text", "text": prompt}]
    for name, data_uri in images:
        blocks.append({"type": "text", "text": f"Attached image: {name}"})
        blocks.append({"type": "image_url", "image_url": {"url": data_uri}})
    return blocks


def _format_price_per_mtok(value: float | None) -> str:
    """Format a USD-per-1M-token price without collapsing cheap models together.

    A flat ``:.2f`` renders $0.098 as "$0.10" and $0.14 as "$0.14" — erasing
    exactly the distinctions this engine exists to make. Sub-dollar prices get
    the precision they need; larger ones stay readable.
    """
    if value is None:
        return "?"
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.4f}"
    if value < 1:
        return f"${value:.3f}".rstrip("0").rstrip(".")
    return f"${value:.2f}"


def _effective_prices(candidate) -> tuple[float | None, float | None, bool]:
    """Resolve a candidate's real prices, preferring the live catalog.

    Returns ``(input_per_1m, output_per_1m, is_live)``. Falls back to the
    registry seed values when the catalog does not carry the slug.
    """
    from theriad.accounting.pricing import PricingTable

    seed_in = candidate.input_cost_per_million
    seed_out = candidate.output_cost_per_million
    try:
        table = PricingTable()
        result = table.lookup(candidate.model)
    except Exception:                                # pragma: no cover - defensive
        return seed_in, seed_out, False
    if result is None:
        return seed_in, seed_out, False
    return result.price.input_per_1m, result.price.output_per_1m, not table.live_stale


def _format_ctx(ctx: int | None) -> str:
    if ctx is None:
        return "?"
    if ctx >= 1_000_000:
        return f"{ctx // 1_000_000}M"
    return f"{ctx // 1024}K"


_SYSTEM_PROMPT = (
    "You are Theriad, an AI assistant powered by the Theriad cost-optimization engine. "
    "When Theriad routing is selected, Theriad routes each query to the cheapest capable LLM using "
    "a multi-layer classification pipeline. When OpenAI or Anthropic direct mode is selected, stay "
    "within that provider and choose a model tier appropriate to the task. "
    "Live web access is provided only by Theriad's server-side web-research harness; never claim to "
    "have browsed or cite current information unless retrieved web evidence is present in the prompt. "
    "Answer the user's question directly and helpfully. If the user asks what you are or how you work, "
    "explain that you are Theriad and clearly distinguish routed and direct-provider modes."
)

_FOLLOW_UP_QUESTIONS_PROMPT = (
    "The user has enabled follow-up questions. If the request is materially ambiguous or "
    "missing information needed to give a useful answer, ask one concise clarifying question "
    "instead of guessing. Otherwise answer directly. Do not ask a question merely to prolong "
    "the conversation, and do not ask for information you can reasonably infer."
)


def _chat_system_prompt(
    project_instructions: str = "",
    enabled_skills: str = "",
    *,
    allow_follow_up_questions: bool = False,
) -> str:
    """Build the shared chat system prompt for every provider implementation."""
    from theriad.storage.prompt_cache import cached_system_prompt

    base_prompt = _SYSTEM_PROMPT
    if allow_follow_up_questions:
        base_prompt = f"{base_prompt}\n\n{_FOLLOW_UP_QUESTIONS_PROMPT}"
    return cached_system_prompt(base_prompt, project_instructions, enabled_skills)


def _tier_badges(primary) -> list[str]:
    badges: list[str] = []
    inp = primary.input_cost_per_million or 0.0
    if inp == 0.0:
        badges.append("Free")
    elif inp < 0.50:
        badges.append("Cheap")
    else:
        badges.append("Premium")
    if "openai" in primary.model:
        badges.append("OpenAI Fallback")
    return badges


_TIER_USE: dict[str, str] = {
    "free_or_floor":    "Handles quick facts and simple formatting when a free model is sufficient.",
    "cheap_bulk":       "Used for summarizing, extracting, or rewriting text where speed and low cost matter most.",
    "cheap_code":       "Handles short code generation and focused refactoring without paying for a large coding model.",
    "code_strong":      "Used for complex debugging, architecture, and multi-file coding where stronger reasoning is worth the cost.",
    "strong_reasoning": "Handles difficult analysis, logic, and math that smaller models may not solve reliably.",
    "premium":          "Reserved for the hardest or multimodal requests and as a fallback when cheaper models fail.",
    "free_moe_general": "Routes everyday questions to the best available free general-purpose model.",
    "free_moe_code": "Routes coding questions to a free model specialized for software tasks.",
    "free_moe_verify": "Routes fact-checking and verification work to a free reasoning model.",
    "free_moe_summarize": "Routes long summaries and extraction work to a free large-context model.",
}

_TIER_RISK: dict[str, str] = {
    "free_or_floor":    "Low",
    "cheap_bulk":       "Low",
    "cheap_code":       "Medium",
    "code_strong":      "Medium",
    "strong_reasoning": "High",
    "premium":          "High",
}


def _load_config():
    from theriad.core.config.loader import load_config
    return load_config(None)


def _build_layers():
    from theriad.layers.layer0.layer import build_default_layer
    from theriad.layers.layer1_features.layer import FeatureRoutingLayer
    return build_default_layer(), FeatureRoutingLayer()


def _tier_limits(tier: str) -> tuple[int | None, float]:
    """Return (max_tokens, timeout_seconds) for the given tier."""
    from theriad.models.tiers import TIER_MAX_TOKENS, TIER_TIMEOUT_SECONDS
    return TIER_MAX_TOKENS.get(tier), TIER_TIMEOUT_SECONDS.get(tier, 30.0)


def _resolved_routing_preferences(body):
    from theriad.models.preferences import RoutingPreferences
    return RoutingPreferences.resolve(
        speed=getattr(body, "speed_preference", None),
        quality=getattr(body, "quality_preference", None),
        legacy_policy=getattr(body, "policy", "balanced"),
    )


def _routing_constraints(request: Request | None, assistant_provider: str = "theriad"):
    """Return hard routing constraints for the current account settings.

    Origin policy applies to Theriad's cross-provider route. Direct OpenAI,
    Anthropic, Groq, and DeepInfra modes are already explicit provider choices
    and should not be silently rewritten by a catalog-origin filter.
    """
    from theriad.models.unified import NO_CONSTRAINTS, RoutingConstraints

    if request is None or assistant_provider != "theriad":
        return NO_CONSTRAINTS
    settings = _get_runtime_settings(request)
    if settings.get("model_origin_policy") == "us_only":
        return RoutingConstraints.us_only()
    return NO_CONSTRAINTS


def _provider_request_extra(
    provider_name: str,
    model: str,
    preferences,
    candidate_models: list[str] | None = None,
) -> dict:
    """Serialize serving preferences only for aggregators that support them."""
    if provider_name != "openrouter":
        return {}
    from theriad.core.providers.routing import build_provider_routing_preferences

    # OpenRouter accepts at most three entries in its top-level ``models``
    # request hint.  When Theriad has a longer fallback chain, omit the hint
    # and keep the selected ``model`` field so the server-side fallback loop
    # can try every candidate one at a time.
    models = list(dict.fromkeys(candidate_models or []))
    use_model_hint = 1 < len(models) <= 3
    route = build_provider_routing_preferences(
        model,
        speed=preferences.speed,
        quality=preferences.quality,
    )
    provider_payload = route.to_openrouter()
    if use_model_hint and "order" not in provider_payload:
        provider_payload["sort"] = {
            "by": route.strategy,
            "partition": "none",
        }
    payload: dict = {"provider": provider_payload}
    if use_model_hint:
        payload["models"] = models
    return payload


def _resolve_assistant_provider(requested: str | None) -> str:
    """Resolve the user-facing assistant mode without crossing providers."""
    from theriad.settings.secrets import get_active_assistant_provider

    selected = (requested or get_active_assistant_provider()).strip().lower()
    if selected == "openrouter":
        selected = "theriad"
    if selected not in {"theriad", "openai", "anthropic", "groq", "deepinfra"}:
        raise ValueError(f"Unsupported assistant provider: {selected}")
    return selected


class _ProjectScopeError(ValueError):
    """A conversation/project pair is missing or does not describe one scope."""

    def __init__(self, message: str, *, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def _resolve_project_scope(
    conversation_id: str | None,
    requested_project_id: str | None,
    *,
    scope_id: str | None = None,
) -> str | None:
    """Derive project scope from a conversation and reject caller mismatches.

    Conversation ownership is canonical.  This keeps project instructions,
    files, and enabled skills from one project out of a different conversation's
    provider request.
    """
    from theriad.api.auth import current_scope_id
    from theriad.storage.database import get_db
    from theriad.storage.hosted import hosted_data_plane_enabled, hosted_repositories

    conversation_id = str(conversation_id or "").strip() or None
    requested_project_id = str(requested_project_id or "").strip() or None
    hosted_context = None
    hosted_repos = None
    if hosted_data_plane_enabled() and scope_id:
        from theriad.api.auth_context import current_auth_context

        hosted_context = current_auth_context.get()
        if hosted_context and hosted_context.principal_type in {"registered", "guest"} and hosted_context.organization_id:
            hosted_repos = hosted_repositories()
    db = get_db()
    scope = current_scope_id()
    if scope is None:
        scope = scope_id
    scope_clause = "user_id = ?" if scope else "user_id IS NULL"
    scope_value = (scope,) if scope else ()

    if conversation_id:
        if hosted_repos and hosted_context:
            row = hosted_repos["conversations"].get_owned(
                conversation_id,
                organization_id=hosted_context.organization_id,
                user_id=hosted_context.user_id,
            )
        else:
            db = get_db()
            row = db.execute(
                f"SELECT project_id FROM conversations WHERE id = ? AND {scope_clause}",
                (conversation_id, *scope_value),
            ).fetchone()
        if row is None:
            raise _ProjectScopeError("Conversation not found", status_code=404)
        conversation_project_id = row["project_id"]
        if requested_project_id is not None and requested_project_id != conversation_project_id:
            raise _ProjectScopeError(
                "project_id does not match the conversation's project",
                status_code=409,
            )
        return conversation_project_id

    if requested_project_id:
        if hosted_repos and hosted_context:
            row = hosted_repos["projects"].get_owned(
                requested_project_id,
                organization_id=hosted_context.organization_id,
                user_id=hosted_context.user_id,
            )
        else:
            db = get_db()
            row = db.execute(
                f"SELECT id FROM projects WHERE id = ? AND {scope_clause}",
                (requested_project_id, *scope_value),
            ).fetchone()
        if row is None:
            raise _ProjectScopeError("Project not found", status_code=404)
    return requested_project_id


def _project_scope_error_response(exc: _ProjectScopeError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})


def _resolve_assistant_api_key(assistant_provider: str) -> str:
    """Resolve a backend-only key, retaining the legacy Theriad config fallback."""
    from theriad.core.providers.factory import get_effective_provider_name
    from theriad.settings.secrets import get_openrouter_api_key, get_provider_api_key

    effective_provider = get_effective_provider_name(assistant_provider)
    api_key = (
        get_openrouter_api_key()
        if effective_provider == "openrouter"
        else get_provider_api_key(effective_provider)
    ) or ""
    if api_key or effective_provider != "openrouter":
        return api_key

    # Backward compatibility for pre-Settings installs.
    try:
        cfg = _load_config()
        provider_cfg = cfg.providers.get("openrouter")
        if provider_cfg and provider_cfg.api_key:
            return provider_cfg.api_key
    except Exception:
        pass
    return ""


def _assistant_candidates(
    assistant_provider: str,
    tier: str,
    resolution,
    requested_model: str | None = None,
) -> list[tuple[str, str]]:
    """Build a provider-contained execution order for a routed tier."""
    from theriad.core.providers.factory import get_effective_provider_name
    from theriad.core.providers.runtime import (
        get_provider_candidates,
        normalize_direct_model_id,
    )

    routed = [(resolution.provider, resolution.model), *resolution.fallback_tuples()]
    if assistant_provider == "theriad":
        return routed

    effective_provider = get_effective_provider_name(assistant_provider)
    if requested_model and requested_model.strip():
        return [(
            effective_provider,
            normalize_direct_model_id(effective_provider, requested_model.strip()),
        )]
    return get_provider_candidates(assistant_provider, tier, routed)


# ── Request schemas ───────────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    prompt: str
    policy: str = "balanced"
    baseline: str = DEFAULT_BASELINE_MODEL
    speed_preference: int | None = Field(default=None, ge=0, le=100)
    quality_preference: int | None = Field(default=None, ge=0, le=100)


class ExecuteRequest(BaseModel):
    prompt: str
    policy: str = "balanced"
    baseline: str = DEFAULT_BASELINE_MODEL
    speed_preference: int | None = Field(default=None, ge=0, le=100)
    quality_preference: int | None = Field(default=None, ge=0, le=100)
    # None follows the provider selected in Settings. "theriad" keeps the
    # cross-provider OpenRouter route; direct modes stay within one provider.
    assistant_provider: str | None = None
    assistant_model: str | None = None
    allow_paid_calls: bool = False
    # None follows the account runtime setting; a bool is a per-request override.
    allow_follow_up_questions: bool | None = None
    workflow_selected: bool = False
    """
    Set to True when the client has already identified this as a multi-step
    workflow via /api/workflow/analyze. When True, max_output_tokens is capped
    to min(tier_cap, 4000) so single-prompt preview mode cannot dump a full
    repo worth of output.
    """
    # Optional project workspace context
    project_id: str | None = None
    conversation_id: str | None = None
    file_ids: list[str] = []
    max_context_tokens: int = 4096
    # Required only after the server flags an unusually large request.
    # "paid" keeps the selected route; "free" explicitly switches to a free model.
    cost_decision: str | None = Field(default=None, pattern="^(paid|free)$")


class SimulatePrompt(BaseModel):
    prompt: str
    expected_output_tokens: int = 200
    id: str | None = None


class SimulateRequest(BaseModel):
    prompts: list[SimulatePrompt]
    baseline: str = DEFAULT_BASELINE_MODEL
    policy: str = "balanced"
    speed_preference: int | None = Field(default=None, ge=0, le=100)
    quality_preference: int | None = Field(default=None, ge=0, le=100)


# ── GET /api/health ───────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    """Lightweight health check — confirms the backend is reachable."""
    return {"status": "ok"}


# ── OpenAI-compatible v1 API ──────────────────────────────────────────────────
# Exposes Theriad as a drop-in provider for any OpenAI-compatible client
# (Codex CLI, Continue.dev, LibreChat, …).
#
# Model name encodes the routing policy:
#   theriad/auto    → balanced (default)
#   theriad/cheap   → aggressive_savings
#   theriad/quality → quality_first
#   theriad/free    → free_moe_experimental
#
# Auth: set THERIAD_API_KEY env var to require a Bearer token.
# If the var is unset the server runs in open/local mode (no auth required).

import json as _json
import os as _os
import time as _time
import uuid as _uuid

_V1_MODELS = [
    {"id": "theriad/auto",    "policy": "balanced",              "description": "Balanced cost/quality"},
    {"id": "theriad/cheap",   "policy": "aggressive_savings",    "description": "Minimize cost"},
    {"id": "theriad/quality", "policy": "quality_first",         "description": "Maximize quality"},
    {"id": "theriad/free",    "policy": "free_moe_experimental", "description": "Free models only"},
]
_V1_MODEL_MAP: dict[str, dict] = {m["id"]: m for m in _V1_MODELS}


def _v1_check_auth(request: Request) -> bool:
    expected = _os.environ.get("THERIAD_API_KEY", "").strip()
    if not expected:
        return True  # open / local mode
    auth = request.headers.get("Authorization", "")
    return auth.startswith("Bearer ") and auth[7:] == expected


def _v1_auth_error():
    return JSONResponse(
        status_code=401,
        content={"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}},
    )


@app.get("/v1/models")
async def v1_list_models(request: Request):
    """List available Theriad virtual models (OpenAI-compatible)."""
    if not _v1_check_auth(request):
        return _v1_auth_error()
    created = int(_time.time())
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": created, "owned_by": "theriad"}
            for m in _V1_MODELS
        ],
    }


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    Routes the conversation through Theriad's cost-optimization layers,
    calls the selected provider, and returns an OpenAI-format response.
    Supports both streaming (SSE) and non-streaming modes.
    """
    if not _v1_check_auth(request):
        return _v1_auth_error()

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}})

    model_id = body.get("model", "theriad/auto")
    messages = body.get("messages") or []
    do_stream = bool(body.get("stream", False))
    max_tokens = int(body.get("max_tokens") or 2048)
    temperature = float(body.get("temperature") or 0.7)

    if not messages:
        return JSONResponse(status_code=400, content={"error": {"message": "'messages' is required", "type": "invalid_request_error"}})

    # Map model id → routing policy (unknown ids default to balanced)
    policy = (_V1_MODEL_MAP.get(model_id) or {}).get("policy", "balanced")

    # Extract the last user turn for routing heuristics
    prompt = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    if not prompt:
        return JSONResponse(status_code=400, content={"error": {"message": "No user message found in 'messages'", "type": "invalid_request_error"}})

    # Keep the legacy OpenAI-compatible handler on the same render contract as
    # the harness router.  The harness is registered first today, but this
    # avoids a silent preview regression if the route registration changes.
    from theriad.capabilities.detector import CapabilityDetector
    from theriad.capabilities.rendering import (
        renderer_metadata,
        renderer_system_instruction,
        with_renderer_instruction,
    )

    v1_capability = CapabilityDetector().detect(prompt)
    v1_rendering = renderer_metadata(v1_capability)
    v1_renderer_instruction = renderer_system_instruction(v1_capability)

    # ── Route ──────────────────────────────────────────────────────────────────
    from theriad.layers.layer0.layer import build_default_layer
    from theriad.layers.layer1_features.layer import FeatureRoutingLayer
    from theriad.models.policy import PolicyResolver

    layer0 = build_default_layer()
    layer1 = FeatureRoutingLayer()
    route_ctx: dict = {}
    route_result = layer0.route(prompt, route_ctx) or layer1.route(prompt, route_ctx)

    if route_result is None:
        from theriad.core.routing.interfaces import RoutingDecision, RoutingResult
        route_result = RoutingResult(
            decision=RoutingDecision(tier="cheap_bulk", confidence=0.0, reason="all layers abstained"),
            layer_name="fallback",
        )

    resolver = PolicyResolver()
    resolution = resolver.resolve(route_result.decision.tier, policy, route_result.decision.confidence, message=prompt)

    # ── Execute ────────────────────────────────────────────────────────────────
    from theriad.core.providers.factory import ProviderConfigurationError, build_provider
    from theriad.core.providers.interfaces import CompletionRequest
    from theriad.core.providers.interfaces import Message as ProviderMessage
    from theriad.settings.secrets import get_active_assistant_provider

    try:
        assistant_provider = get_active_assistant_provider()
        provider = build_provider(assistant_provider, timeout=60)
    except (ProviderConfigurationError, Exception) as exc:
        return JSONResponse(status_code=503, content={"error": {"message": f"Provider unavailable: {exc}", "type": "api_error"}})

    provider_messages = with_renderer_instruction(
        [
            ProviderMessage(role=m.get("role", "user"), content=m.get("content") or "")
            for m in messages
        ],
        v1_renderer_instruction,
    )
    req = CompletionRequest(
        model=resolution.model,
        messages=provider_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    try:
        resp = await provider.complete(req)
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": {"message": f"Provider call failed: {exc}", "type": "api_error"}})

    content = resp.content or resp.reasoning or ""
    completion_id = f"chatcmpl-{_uuid.uuid4().hex[:24]}"
    created = int(_time.time())

    # Extra header lets callers see which model was actually used
    x_theriad = {
        "routed_to": resolution.model,
        "provider": resolution.provider,
        "tier": resolution.tier,
        "policy": policy,
        "routing_layer": route_result.layer_name,
    }
    if v1_rendering:
        x_theriad["renderer"] = v1_rendering

    if do_stream:
        # Buffer the full response then deliver as two SSE chunks so the client
        # receives a proper stream without needing provider-level streaming support.
        async def _stream():
            delta_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
                "x_theriad": x_theriad,
            }
            yield f"data: {_json.dumps(delta_chunk)}\n\n"
            done_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {_json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        from starlette.responses import StreamingResponse
        return StreamingResponse(_stream(), media_type="text/event-stream",
                                 headers={"X-Theriad-Route": _json.dumps(x_theriad)})

    return JSONResponse(content={
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "x_theriad": x_theriad,
    }, headers={"X-Theriad-Route": _json.dumps(x_theriad)})


# ── GET /api/tools/status ─────────────────────────────────────────────────────

@app.get("/api/tools/status")
async def tools_status():
    """Return the canonical OpenSandbox browser runtime status."""
    from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime

    state = await get_opensandbox_browser_runtime().availability()
    return {
        "canonical": True,
        "browser_backend": "opensandbox",
        "browser": state,
        "deprecated_backends": [
            "mock", "playwright", "playwright_mcp", "browserbase",
            "electron", "chrome_mcp", "extension",
        ],
    }


# ── GET /api/tools/chrome/status ─────────────────────────────────────────────

@app.get("/api/tools/chrome/status")
def chrome_status():
    """Return a stable deprecation response for the former Chrome MCP API."""
    return JSONResponse(
        status_code=410,
        content={
            "error": "Chrome DevTools MCP is deprecated.",
            "replacement": "/api/tools/status",
            "backend": "opensandbox",
        },
        headers={"Deprecation": "true", "Sunset": "0"},
    )


# ── GET /api/runtime/status ──────────────────────────────────────────────────

@app.get("/api/runtime/status")
async def runtime_status():
    """Return the active runtime profile plus OpenSandbox readiness."""
    from theriad.runtime.profile import get_runtime_config
    from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime

    rc = get_runtime_config()
    return {
        **rc.to_safe_dict(),
        "browser_runtime": "opensandbox",
        "opensandbox_status": await get_opensandbox_browser_runtime().availability(),
        "deprecated_backends": [
            "mock", "playwright", "playwright_mcp", "browserbase",
            "electron", "chrome_mcp", "extension",
        ],
    }


# ── POST /api/preview/qa ──────────────────────────────────────────────────────

class PreviewQARequest(BaseModel):
    html:             str
    css:              str  = ""
    tailwind_enabled: bool = False


@app.post("/api/preview/qa")
async def preview_qa(body: PreviewQARequest):
    """Render and inspect a preview in an OpenSandbox Chromium session."""
    import base64 as _base64
    import time as _time

    from theriad.browser.opensandbox_runtime import (
        _redact_text,
        get_opensandbox_browser_runtime,
    )

    started_at = _time.monotonic()
    runtime = get_opensandbox_browser_runtime()
    availability = await runtime.availability()
    if not availability.get("available"):
        return {
            "status": "unavailable",
            "message": availability.get("error") or "OpenSandbox is unavailable.",
            "screenshot": None,
            "console_errors": [],
            "network_errors": [],
            "dom_summary": None,
            "warnings": [
                "Start OpenSandbox and configure THERIAD_OPENSANDBOX_DOMAIN."
            ],
            "latency_ms": round((_time.monotonic() - started_at) * 1000, 1),
            "backend": "opensandbox",
        }

    session_id: str | None = None
    console_errors: list[str] = []
    network_errors: list[str] = []
    warnings: list[str] = []
    try:
        session = await runtime.start_session(
            user_prompt="Render the generated preview for QA.",
            viewport={"width": 1440, "height": 1000},
        )
        session_id = session.get("session_id")
        if not session_id:
            raise RuntimeError(
                session.get("error") or "OpenSandbox did not return a session id"
            )
        live = runtime.sessions.get(session_id)
        page = live.page if live is not None else None
        if page is None:
            raise RuntimeError("OpenSandbox browser page is unavailable")

        def on_console(message: Any) -> None:
            level = getattr(message, "type", "")
            text = getattr(message, "text", "")
            text = text() if callable(text) else text
            if str(level).lower() in {"error", "warning"}:
                console_errors.append(_redact_text(str(text))[:500])

        def on_request_failed(page_request: Any) -> None:
            url = _redact_text(str(getattr(page_request, "url", "")))[:500]
            failure = getattr(page_request, "failure", None)
            failure = failure() if callable(failure) else failure
            if isinstance(failure, dict):
                detail = failure.get("errorText") or failure.get("error_text") or "request failed"
            else:
                detail = str(failure or "request failed")
            network_errors.append(f"{url} -> {_redact_text(detail)[:300]}")

        try:
            page.on("console", on_console)
            page.on("requestfailed", on_request_failed)
        except Exception:
            warnings.append("OpenSandbox Playwright event hooks are unavailable.")

        html_doc = _build_preview_doc(body.html, body.css, body.tailwind_enabled)
        preview_url = "data:text/html;base64," + _base64.b64encode(
            html_doc.encode("utf-8")
        ).decode("ascii")
        result = await runtime.navigate(session_id, preview_url)
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "Preview navigation failed")
        snapshot = await runtime.observe(session_id, screenshot=True)
        visible_text = str(snapshot.get("visible_text_summary") or "")
        elements = snapshot.get("elements") or []
        dom_summary = (
            f"Title: {snapshot.get('page_title') or '(none)'} | "
            f"Body text: {len(visible_text)} chars | Elements: {len(elements)}"
        )
        return {
            "status": "ok",
            "message": "OpenSandbox preview QA completed.",
            "backend": "opensandbox",
            "screenshot": snapshot.get("screenshot_base64"),
            "console_errors": console_errors[:50],
            "network_errors": network_errors[:50],
            "dom_summary": dom_summary,
            "warnings": warnings,
            "latency_ms": round((_time.monotonic() - started_at) * 1000, 1),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "message": f"OpenSandbox preview QA failed: {_redact_text(exc)}",
            "backend": "opensandbox",
            "screenshot": None,
            "console_errors": console_errors[:50],
            "network_errors": network_errors[:50],
            "dom_summary": None,
            "warnings": warnings,
            "latency_ms": round((_time.monotonic() - started_at) * 1000, 1),
        }
    finally:
        if session_id:
            await runtime.stop_session(session_id)


def _legacy_preview_qa(body: PreviewQARequest):
    """
    Run Chrome DevTools QA on a generated HTML preview.

    Builds the same preview document the iframe renders, opens it in an
    isolated Chrome session via MCPBroker + ChromeDevToolsAdapter, and
    returns a screenshot, console errors, network errors, and DOM summary.

    If Chrome MCP is not configured, returns status="unavailable" cleanly.
    The preview iframe continues to work regardless.

    Security:
      - Uses ephemeral browser profile (consumer_secure forces this).
      - Does not allow forms, uploads, or downloads in the preview page.
      - API keys and secrets are never embedded in the preview document.
      - Page content cannot override browser policy (D-041).

    Response shape:
      {
        "status":         "ok" | "unavailable" | "failed",
        "message":        str,
        "screenshot":     str | null,   # base64 PNG if available
        "console_errors": [],
        "network_errors": [],
        "dom_summary":    str | null,
        "warnings":       [],
        "latency_ms":     float
      }
    """
    import json as _json
    import os
    import tempfile
    import threading
    import time as _time
    import http.server
    import urllib.parse

    from theriad.tools.chrome_devtools_adapter import ChromeDevToolsAdapter
    from theriad.tools.mcp_broker import MCPBroker

    start = _time.monotonic()

    adapter = ChromeDevToolsAdapter()
    if not adapter.available:
        return {
            "status":         "unavailable",
            "message":        (
                "Chrome MCP is not configured. Preview iframe still works. "
                "Set THERIAD_CHROME_MCP_ENABLED=true to enable Chrome QA."
            ),
            "screenshot":     None,
            "console_errors": [],
            "network_errors": [],
            "dom_summary":    None,
            "warnings":       [],
            "latency_ms":     round((_time.monotonic() - start) * 1000, 1),
        }

    broker = MCPBroker(
        session_id=f"preview-qa-{int(_time.monotonic() * 1000)}",
        adapters={"chrome_devtools": adapter},
    )

    html_doc = _build_preview_doc(body.html, body.css, body.tailwind_enabled)

    screenshot:     str | None  = None
    console_errors: list[str]   = []
    network_errors: list[str]   = []
    dom_summary:    str | None  = None
    warnings:       list[str]   = []

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(html_doc)
            tmp_path = tmp.name

        tmp_dir  = os.path.dirname(tmp_path)
        tmp_name = os.path.basename(tmp_path)

        class _QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args):  # suppress access logs
                pass

        httpd         = http.server.HTTPServer(("127.0.0.1", 0), _QuietHandler)
        port          = httpd.server_address[1]
        preview_url   = f"http://127.0.0.1:{port}/{urllib.parse.quote(tmp_name)}"
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)

        orig_dir = os.getcwd()
        try:
            os.chdir(tmp_dir)
            server_thread.start()

            # new_page
            np_res = broker.call(
                subtask="website_preview_qa", server_name="chrome_devtools",
                tool_name="new_page", arguments={},
            )
            if not np_res.ok:
                raise RuntimeError(np_res.blocked_reason or np_res.error or "new_page failed")
            page_id = (np_res.result or {}).get("pageId")

            def _args(**kw):
                d = dict(**kw)
                if page_id:
                    d["pageId"] = page_id
                return d

            # navigate_page
            nav_res = broker.call(
                subtask="website_preview_qa", server_name="chrome_devtools",
                tool_name="navigate_page", arguments=_args(url=preview_url),
            )
            if not nav_res.ok:
                warnings.append(f"Navigation: {nav_res.blocked_reason or nav_res.error}")

            # take_screenshot
            scr_res = broker.call(
                subtask="website_preview_qa", server_name="chrome_devtools",
                tool_name="take_screenshot", arguments=_args(),
            )
            if scr_res.ok and scr_res.result:
                screenshot = (
                    scr_res.result.get("screenshot")
                    or scr_res.result.get("data")
                )
            else:
                warnings.append("Screenshot unavailable.")

            # list_console_messages
            con_res = broker.call(
                subtask="website_preview_qa", server_name="chrome_devtools",
                tool_name="list_console_messages", arguments=_args(),
            )
            if con_res.ok and con_res.result:
                msgs = con_res.result.get("messages") or con_res.result.get("console", [])
                console_errors = [
                    m.get("text", str(m))
                    for m in (msgs if isinstance(msgs, list) else [])
                    if isinstance(m, dict) and m.get("level") in ("error", "warning")
                ]

            # list_network_requests
            net_res = broker.call(
                subtask="website_preview_qa", server_name="chrome_devtools",
                tool_name="list_network_requests", arguments=_args(),
            )
            if net_res.ok and net_res.result:
                reqs = net_res.result.get("requests") or net_res.result.get("network", [])
                network_errors = [
                    f"{r.get('method','?')} {r.get('url','?')} → {r.get('status','?')}"
                    for r in (reqs if isinstance(reqs, list) else [])
                    if isinstance(r, dict)
                    and str(r.get("status", "200")).startswith(("4", "5", "0"))
                ]

            # evaluate_script — DOM summary
            eval_res = broker.call(
                subtask="website_preview_qa", server_name="chrome_devtools",
                tool_name="evaluate_script",
                arguments=_args(script=(
                    'JSON.stringify({'
                    '  title: document.title,'
                    '  bodyLength: document.body ? document.body.innerText.length : 0,'
                    '  elementCount: document.querySelectorAll("*").length'
                    '})'
                )),
            )
            if eval_res.ok and eval_res.result:
                try:
                    raw = eval_res.result.get("result") or eval_res.result.get("value", "{}")
                    dom_data = _json.loads(raw) if isinstance(raw, str) else raw
                    dom_summary = (
                        f"Title: {dom_data.get('title','(none)')} | "
                        f"Body text: {dom_data.get('bodyLength', 0)} chars | "
                        f"Elements: {dom_data.get('elementCount', 0)}"
                    )
                except Exception:
                    dom_summary = str(eval_res.result)[:200]

        finally:
            os.chdir(orig_dir)
            httpd.shutdown()
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        return {
            "status":         "ok",
            "message":        "Chrome QA completed.",
            "screenshot":     screenshot,
            "console_errors": console_errors,
            "network_errors": network_errors,
            "dom_summary":    dom_summary,
            "warnings":       warnings,
            "latency_ms":     round((_time.monotonic() - start) * 1000, 1),
        }

    except Exception as exc:
        return {
            "status":         "failed",
            "message":        f"Chrome QA failed: {exc}",
            "screenshot":     None,
            "console_errors": [],
            "network_errors": [],
            "dom_summary":    None,
            "warnings":       [],
            "latency_ms":     round((_time.monotonic() - start) * 1000, 1),
        }


def _build_preview_doc(html: str, css: str, tailwind_enabled: bool) -> str:
    """
    Build a self-contained HTML preview document.

    Mirrors buildPreviewDoc() in Theriad.jsx — the Chrome QA page is identical
    to what the iframe renders. No API keys or secrets are embedded.
    """
    import re
    is_full_doc = bool(
        re.search(r"<!doctype\s+html", html, re.IGNORECASE)
        or re.search(r"<html[\s>]", html, re.IGNORECASE)
    )
    tw_script = '<script src="https://cdn.tailwindcss.com"></script>'

    if is_full_doc:
        doc = html
        if css:
            doc = doc.replace("</head>", f"<style>\n{css}\n</style>\n</head>")
        if tailwind_enabled:
            doc = doc.replace("</head>", f"{tw_script}\n</head>")
        return doc

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8" />\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        + (f"  {tw_script}\n" if tailwind_enabled else "")
        + (f"  <style>\n{css}\n  </style>\n" if css else "")
        + "</head>\n"
        '<body style="margin:0;padding:1rem;">\n'
        f"{html}\n"
        "</body>\n"
        "</html>"
    )


# ── GET /api/settings/provider-status ────────────────────────────────────────

@app.get("/api/settings/provider-status")
def get_provider_status():
    """
    Return OpenRouter key configuration status (masked key only — never full key).
    """
    from theriad.settings.secrets import get_openrouter_key_status
    return JSONResponse(
        content=get_openrouter_key_status(),
        headers={"Cache-Control": "no-store"},
    )


# ── POST /api/settings/openrouter-key ────────────────────────────────────────

class SaveKeyRequest(BaseModel):
    api_key: str


def _provider_account_required(request: Request) -> JSONResponse | None:
    """Keep guest mode free-only until a visitor has a durable account."""
    user = get_request_user(request)
    if user and user.get("is_guest"):
        return JSONResponse(
            status_code=403,
            content={
                "error": "account_required",
                "detail": "Create an account before connecting a paid provider.",
            },
            headers={"Cache-Control": "no-store"},
        )
    return None


@app.post("/api/settings/openrouter-key")
def save_openrouter_key(body: SaveKeyRequest, request: Request):
    """
    Validate and save the OpenRouter API key to the current request's scoped
    settings file (or the legacy global file for anonymous local use).

    Returns the masked status — never echoes the full key.
    """
    account_error = _provider_account_required(request)
    if account_error:
        return account_error
    from theriad.settings.secrets import get_openrouter_key_status, save_openrouter_api_key
    try:
        save_openrouter_api_key(body.api_key)
        _sync_hosted_provider_metadata(request, "openrouter", get_openrouter_key_status())
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    return JSONResponse(
        content=get_openrouter_key_status(),
        headers={"Cache-Control": "no-store"},
    )


# ── DELETE /api/settings/openrouter-key ──────────────────────────────────────

@app.delete("/api/settings/openrouter-key")
def delete_openrouter_key(request: Request):
    """
    Remove the saved OpenRouter API key from the current request's scoped
    settings file (or the legacy global file for anonymous local use).

    Has no effect if the key is sourced from an environment variable.
    Returns updated status.
    """
    from theriad.settings.secrets import delete_openrouter_api_key, get_openrouter_key_status
    delete_openrouter_api_key()
    _sync_hosted_provider_metadata(request, "openrouter", get_openrouter_key_status())
    return JSONResponse(
        content=get_openrouter_key_status(),
        headers={"Cache-Control": "no-store"},
    )


# ── Multi-provider assistant settings ────────────────────────────────────────

class ActiveAssistantProviderRequest(BaseModel):
    provider: str


def _provider_settings_payload() -> dict:
    from theriad.settings.secrets import (
        get_provider_settings_status,
    )
    return dict(get_provider_settings_status())


def _sync_hosted_provider_metadata(request: Request, provider: str, status: dict) -> None:
    """Project masked provider metadata without putting a key in Postgres."""
    from theriad.storage.hosted import hosted_data_plane_enabled, hosted_repositories

    if not hosted_data_plane_enabled():
        return
    from theriad.api.auth_context import get_auth_context

    context = get_auth_context(request)
    if context.principal_type != "registered" or not context.organization_id or not context.user_id:
        return
    repos = hosted_repositories()
    repos["provider_metadata"].client.upsert_row(
        "provider_connections",
        {
            "organization_id": context.organization_id,
            "user_id": context.user_id,
            "provider": provider,
            "secret_ref": f"theriad://provider/{context.organization_id}/{provider}",
            "active": bool(status.get("configured")),
            "masked_key": status.get("masked_key"),
            "updated_at": _now(),
        },
        conflict="organization_id,provider",
    )


@app.get("/api/settings/providers")
def get_assistant_providers():
    """Return masked BYOK status for every supported assistant provider."""
    return JSONResponse(
        content=_provider_settings_payload(),
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/settings/providers/openrouter/oauth/start")
def start_openrouter_oauth(request: Request):
    """Create a short-lived OpenRouter PKCE authorization URL."""
    account_error = _provider_account_required(request)
    if account_error:
        return account_error
    import base64
    import hashlib
    import os
    import secrets
    import time
    from urllib.parse import urlencode

    now = time.monotonic()
    for pending_state, (_, expires_at, _) in list(_openrouter_oauth_flows.items()):
        if expires_at <= now:
            _openrouter_oauth_flows.pop(pending_state, None)

    state = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    owner = get_request_user(request)
    owner_id = owner["id"] if owner else None
    _openrouter_oauth_flows[state] = (verifier, now + 10 * 60, owner_id)

    public_base = os.environ.get("THERIAD_PUBLIC_URL", "").strip().rstrip("/")
    base_url = public_base or str(request.base_url).rstrip("/")
    callback_url = (
        f"{base_url}/api/settings/providers/openrouter/oauth/callback/{state}"
    )
    authorization_url = "https://openrouter.ai/auth?" + urlencode(
        {
            "callback_url": callback_url,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    return JSONResponse(
        content={"authorization_url": authorization_url, "expires_in_seconds": 600},
        headers={"Cache-Control": "no-store"},
    )


def _oauth_result_page(title: str, message: str, *, success: bool) -> HTMLResponse:
    color = "#67d8a1" if success else "#ff8f82"
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title}</title><style>
body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#0d1117;color:#e8edf2;
font:14px/1.55 system-ui,sans-serif}}main{{max-width:420px;margin:24px;padding:28px;border:1px solid #29313b;
border-radius:16px;background:#141a21}}h1{{margin:0 0 10px;color:{color};font-size:20px}}p{{margin:0;color:#aab4bf}}
</style></head><body><main><h1>{title}</h1><p>{message}</p></main>
<script>setTimeout(()=>window.close(),900)</script></body></html>"""
    return HTMLResponse(
        content=html,
        status_code=200 if success else 400,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'",
        },
    )


@app.get("/api/settings/providers/openrouter/oauth/callback/{state}")
async def finish_openrouter_oauth(state: str, code: str = ""):
    """Exchange a one-time PKCE code and save only the resulting API key."""
    import time

    flow = _openrouter_oauth_flows.pop(state, None)
    if flow is None or flow[1] <= time.monotonic() or not code.strip():
        return _oauth_result_page(
            "Connection expired",
            "Return to Theriad and start the OpenRouter connection again.",
            success=False,
        )

    import httpx

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/auth/keys",
                headers={"Content-Type": "application/json"},
                json={
                    "code": code.strip(),
                    "code_verifier": flow[0],
                    "code_challenge_method": "S256",
                },
            )
        payload = response.json() if response.status_code == 200 else {}
        key = payload.get("key") if isinstance(payload, dict) else None
        if not isinstance(key, str) or not key.strip():
            raise ValueError("OpenRouter did not return a key")

        from theriad.settings.secrets import save_provider_api_key

        # The callback is a browser popup and usually cannot carry the
        # original Authorization header. The one-time state is bound to the
        # authenticated starter, so persist the key in that user's scope.
        save_provider_api_key("openrouter", key, scope_id=flow[2])
    except Exception:
        return _oauth_result_page(
            "Could not connect",
            "OpenRouter did not complete the connection. Return to Theriad and try again.",
            success=False,
        )

    return _oauth_result_page(
        "OpenRouter connected",
        "The key was saved by Theriad. You can close this window.",
        success=True,
    )


@app.post("/api/settings/providers/{provider}/key")
def save_assistant_provider_key(provider: str, body: SaveKeyRequest, request: Request):
    account_error = _provider_account_required(request)
    if account_error:
        return account_error
    from theriad.settings.secrets import get_provider_key_status, save_provider_api_key
    try:
        save_provider_api_key(provider, body.api_key)
        status = get_provider_key_status(provider)
        _sync_hosted_provider_metadata(request, provider, status)
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": str(exc)},
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(
        content={"provider": provider, **status},
        headers={"Cache-Control": "no-store"},
    )


@app.delete("/api/settings/providers/{provider}/key")
def delete_assistant_provider_key(provider: str, request: Request):
    from theriad.settings.secrets import delete_provider_api_key, get_provider_key_status
    try:
        delete_provider_api_key(provider)
        status = get_provider_key_status(provider)
        _sync_hosted_provider_metadata(request, provider, status)
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": str(exc)},
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(
        content={"provider": provider, **status},
        headers={"Cache-Control": "no-store"},
    )


@app.put("/api/settings/providers/active")
def set_active_assistant_provider(body: ActiveAssistantProviderRequest, request: Request):
    user = get_request_user(request)
    if user and user.get("is_guest") and body.provider != "theriad":
        return JSONResponse(
            status_code=403,
            content={
                "error": "account_required",
                "detail": "Create an account before selecting a paid provider.",
            },
            headers={"Cache-Control": "no-store"},
        )
    from theriad.settings.secrets import (
        get_provider_key_status,
        save_active_assistant_provider,
    )

    provider_key_name = "openrouter" if body.provider == "theriad" else body.provider
    try:
        status = get_provider_key_status(provider_key_name)
        if not status["configured"]:
            return JSONResponse(
                status_code=409,
                content={
                    "error": (
                        f"Connect {provider_key_name.title()} before selecting it "
                        "as the assistant provider."
                    )
                },
                headers={"Cache-Control": "no-store"},
            )
        save_active_assistant_provider(body.provider)
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": str(exc)},
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(
        content=_provider_settings_payload(),
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/settings/providers/{provider}/test")
async def test_assistant_provider(provider: str):
    """Validate a saved key without returning upstream details or the key."""
    from theriad.core.providers.factory import (
        ProviderConfigurationError,
        build_provider,
    )

    try:
        provider_client = build_provider(provider, timeout=8.0)
        ok = await provider_client.health_check()
    except (ProviderConfigurationError, ValueError) as exc:
        return JSONResponse(
            status_code=409,
            content={"ok": False, "provider": provider, "error": str(exc)},
            headers={"Cache-Control": "no-store"},
        )
    except Exception:
        ok = False

    if not ok:
        return JSONResponse(
            status_code=502,
            content={
                "ok": False,
                "provider": provider,
                "error": "Connection failed. Check the saved key and provider status.",
            },
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(
        content={"ok": True, "provider": provider},
        headers={"Cache-Control": "no-store"},
    )


# ── Browser backend settings ──────────────────────────────────────────────────

@app.get("/api/settings/browser-backend")
async def browser_backend_get():
    """Return the OpenSandbox browser capability contract."""
    from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime

    state = await get_opensandbox_browser_runtime().availability()
    return {
        "current_backend": "opensandbox",
        "web_search": {
            "available": True,
            "mode": "automatic",
            "label": "Automatic web search",
        },
        "options": {"opensandbox": state},
        "deprecated_backends": [
            "mock",
            "playwright",
            "playwright_mcp",
            "browserbase",
            "electron",
            "chrome_mcp",
            "extension",
        ],
    }


@app.post("/api/settings/browser-backend")
async def browser_backend_set(body: dict):
    """Accept only the OpenSandbox backend after the browser migration."""
    backend = (body.get("backend") or "").strip().lower()
    if not backend:
        return JSONResponse(status_code=422, content={"error": "backend is required"})
    if backend != "opensandbox":
        return JSONResponse(
            status_code=410,
            content={
                "error": "Legacy browser backends are deprecated.",
                "replacement": "opensandbox",
            },
            headers={"Deprecation": "true", "Sunset": "0"},
        )
    from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime

    state = await get_opensandbox_browser_runtime().availability()
    return {"backend": "opensandbox", "available": state.get("available", False), "error": state.get("error")}


# ── Runtime safety settings ───────────────────────────────────────────────────

class RuntimeSettingsUpdate(BaseModel):
    """Partial update for account-scoped chat safety controls."""

    regular_chat_returns_answer: bool | None = None
    allow_follow_up_questions: bool | None = None
    route_only_available: bool | None = None
    disable_premium_models: bool | None = None
    model_origin_policy: str | None = Field(default=None, pattern="^(any|us_only)$")
    max_cost_per_request_usd: float | None = Field(default=None, ge=0)
    max_daily_spend_usd: float | None = Field(default=None, ge=0)


@app.get("/api/settings/runtime")
def runtime_settings_get(request: Request):
    return _get_runtime_settings(request)


@app.put("/api/settings/runtime")
def runtime_settings_update(body: RuntimeSettingsUpdate, request: Request):
    updates = body.model_dump(exclude_unset=True)
    if any(
        updates.get(key) is None
        for key in (
            "regular_chat_returns_answer",
            "allow_follow_up_questions",
            "route_only_available",
            "disable_premium_models",
        )
        if key in updates
    ):
        return JSONResponse(
            status_code=422,
            content={"error": "Boolean runtime settings cannot be null."},
        )
    try:
        return _update_runtime_settings(request, updates)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})


# ── GET /api/status ───────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Return provider connectivity and active config (config-based, no live ping)."""
    try:
        cfg = _load_config()
        providers = cfg.providers or {}
        return {
            "openrouter": {
                "connected": "openrouter" in providers,
                "configured": "openrouter" in providers,
                "priority": 1,
            },
            "ollama": {
                "connected": "ollama" in providers,
                "configured": "ollama" in providers,
                "url": "localhost:11434",
            },
            "active_policy": cfg.router.policy,
            "fallback_tier": cfg.router.fallback_tier,
            "dry_run_default": False,
            "max_cost_per_request": None,
        }
    except Exception as exc:
        return {
            "openrouter": {"connected": False, "configured": False, "priority": 1},
            "ollama": {"connected": False, "configured": False, "url": "localhost:11434"},
            "active_policy": "balanced",
            "fallback_tier": "cheap_bulk",
            "dry_run_default": False,
            "max_cost_per_request": None,
            "error": str(exc),
        }


# ── POST /api/route ───────────────────────────────────────────────────────────

@app.post("/api/route")
def route_prompt(body: RouteRequest, request: Request):
    """
    Route a prompt through Theriad layers and return the full decision breakdown.

    Runs every L0 rule + L1 features, resolves tier → model via PolicyResolver,
    estimates cost vs baseline. No provider calls.
    """
    from theriad.accounting.pricing import PricingTable, estimate_cost
    from theriad.layers.layer0.rules import (
        BuildProjectRule,
        CodeDetectionRule,
        JsonDetectionRule,
        LanguageDetectionRule,
        MarkdownDetectionRule,
        PromptLengthRule,
    )
    from theriad.layers.layer1_features.extractor import FeatureExtractor
    from theriad.layers.layer1_features.layer import FeatureRoutingLayer
    from theriad.models.policy import PolicyResolver

    # ── Run every L0 rule individually (for routing_path) ────────────────────
    rules = [
        JsonDetectionRule(),
        CodeDetectionRule(),
        MarkdownDetectionRule(),
        LanguageDetectionRule(),
        BuildProjectRule(),
        PromptLengthRule(),
    ]
    routing_path = []
    winning_decision = None

    for rule in rules:
        d = rule.evaluate(body.prompt)
        fired = d is not None and d.confidence >= 0.75
        routing_path.append({
            "layer": f"L0:{rule.name}",
            "fired": fired,
            "tier": d.tier if d else None,
            "confidence": round(d.confidence, 3) if d else None,
            "reason": d.reason if d else "abstained",
            "signals": d.signals if d else {},
        })
        if fired and winning_decision is None:
            winning_decision = d

    # ── L1 features ───────────────────────────────────────────────────────────
    extractor = FeatureExtractor()
    features = extractor.extract(body.prompt)
    layer1_result = FeatureRoutingLayer().route(body.prompt, {})

    routing_path.append({
        "layer": "L1:features",
        "fired": layer1_result is not None,
        "tier": layer1_result.decision.tier if layer1_result else None,
        "confidence": round(layer1_result.decision.confidence, 3) if layer1_result else None,
        "reason": layer1_result.decision.reason if layer1_result else "abstained",
        "signals": {
            "token_count": features.token_count,
            "code_ratio": round(features.code_ratio, 3),
        },
    })

    if winning_decision is None and layer1_result is not None:
        winning_decision = layer1_result.decision

    # ── Fallback if all layers abstained ─────────────────────────────────────
    if winning_decision is None:
        from theriad.core.routing.interfaces import RoutingDecision
        winning_decision = RoutingDecision(
            tier="cheap_bulk",
            confidence=0.0,
            reason="all layers abstained — using fallback tier",
        )

    # ── Resolve tier → model ──────────────────────────────────────────────────
    resolver = PolicyResolver()
    preferences = _resolved_routing_preferences(body)
    origin_constraints = _routing_constraints(request)
    resolution = resolver.resolve(
        winning_decision.tier,
        body.policy,
        winning_decision.confidence,
        message=body.prompt,
        speed_preference=preferences.speed,
        quality_preference=preferences.quality,
        constraints=origin_constraints,
    )

    # ── Cost estimate ─────────────────────────────────────────────────────────
    pt = PricingTable()
    prompt_tokens = _estimate_prompt_tokens(body.prompt)
    est_output = 200
    theriad_model_cost = estimate_cost(resolution.model, prompt_tokens, est_output, pt) or 0.0
    # Registered paid routes include the transparent Theriad routing fee in
    # the preview. Anonymous/local and guest previews remain fee-exempt.
    request_user = get_request_user(request)
    fee = platform_fee(prompt_tokens, est_output) if request_user and not request_user.get("is_guest") else 0.0
    theriad_cost = theriad_model_cost + fee
    baseline_cost = estimate_cost(body.baseline, prompt_tokens, est_output, pt) or 0.0
    savings_pct = (
        (baseline_cost - theriad_cost) / baseline_cost * 100
        if baseline_cost > 0 else 0.0
    )

    winning_layer = next(
        (item["layer"] for item in routing_path if item["fired"]), "fallback"
    )

    # ── Capability hook (advisory only — never auto-starts a session) ─────────
    from theriad.capabilities.detector import CapabilityDetector
    cap_result = CapabilityDetector().detect(body.prompt)
    browser_agent_recommended = cap_result.recommended_execution_mode == "browser_agent"

    return {
        "tier": resolution.tier,
        "model": resolution.model,
        "provider": resolution.provider,
        "model_origin_policy": "us_only" if origin_constraints.active else "any",
        "policy": body.policy,
        "speed_preference": preferences.speed,
        "quality_preference": preferences.quality,
        "estimated_price_multiplier": preferences.price_multiplier,
        "confidence": round(winning_decision.confidence, 3),
        "reason": winning_decision.reason,
        "signals": winning_decision.signals,
        "winning_layer": winning_layer,
        "routing_path": routing_path,
        "fallback_chain": resolution.fallback_tuples(),
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_output_tokens": est_output,
        "estimated_cost_usd": round(theriad_cost, 8),
        "model_cost_usd": round(theriad_model_cost, 8),
        "platform_fee_usd": round(fee, 8),
        "baseline_cost_usd": round(baseline_cost, 8),
        "savings_pct": round(savings_pct, 1),
        "features": {
            "token_count": features.token_count,
            "code_ratio": round(features.code_ratio, 3),
            "json_probability": round(features.json_probability, 3),
            "markdown_ratio": round(features.markdown_ratio, 3),
        },
        # Advisory flag — caller must explicitly POST /api/agent/browser/start.
        # Routing never auto-starts a browser session.
        "browser_agent_recommended": browser_agent_recommended,
        "capability": cap_result.to_dict(),
    }


# ── POST /api/execute ─────────────────────────────────────────────────────────

@app.post("/api/execute")
async def execute_prompt(body: ExecuteRequest, request: Request):
    """
    Real execution: route prompt → resolve model → call OpenRouter → record metrics.

    Requires allow_paid_calls=True. Returns HTTP 403 otherwise — no silent billing.
    The API key is read from theriad.yaml providers.openrouter.api_key or the
    OPENROUTER_API_KEY environment variable.
    """
    import time

    # ── Safety gate ───────────────────────────────────────────────────────────
    if not body.allow_paid_calls:
        return JSONResponse(
            status_code=403,
            content={
                "error": (
                    "allow_paid_calls must be true to execute. "
                    "This endpoint makes real provider calls that incur cost."
                )
            },
        )

    scope_id = user_scope_id(request)
    request_user = get_request_user(request)
    if request_user and request_user.get("is_guest") and body.cost_decision != "free":
        return JSONResponse(
            status_code=403,
            content={
                "error": "registered_account_required",
                "message": "Guest mode is limited to the free-model chat path. Register before paid execution.",
            },
        )

    # ── Resolve selected assistant + backend-only API key ────────────────────
    try:
        assistant_provider = _resolve_assistant_provider(body.assistant_provider)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})

    try:
        body.project_id = _resolve_project_scope(
            body.conversation_id, body.project_id, scope_id=scope_id
        )
    except _ProjectScopeError as exc:
        return _project_scope_error_response(exc)

    execution_provider = "theriad" if body.cost_decision == "free" else assistant_provider
    origin_constraints = _routing_constraints(request, execution_provider)
    runtime_settings = _get_runtime_settings(request)
    allow_follow_up_questions = (
        body.allow_follow_up_questions
        if body.allow_follow_up_questions is not None
        else runtime_settings["allow_follow_up_questions"]
    )
    cost_preview = _chat_cost_preview(
        ChatSendRequest(
            project_id=body.project_id,
            conversation_id=body.conversation_id,
            message=body.prompt,
            file_ids=body.file_ids,
            policy=body.policy,
            baseline=body.baseline,
            speed_preference=body.speed_preference,
            quality_preference=body.quality_preference,
            assistant_provider=execution_provider,
            assistant_model=body.assistant_model,
            mode="execute",
            allow_paid_calls=body.allow_paid_calls,
            allow_follow_up_questions=allow_follow_up_questions,
            cost_decision=body.cost_decision,
        ),
        constraints=origin_constraints,
        allow_follow_up_questions=allow_follow_up_questions,
        scope_id=scope_id,
    )
    if body.cost_decision == "free" and not cost_preview["requires_confirmation"]:
        return JSONResponse(
            status_code=422,
            content={"error": "Free fallback is offered only after a high-context warning."},
        )
    if cost_preview["requires_confirmation"] and body.cost_decision not in {"paid", "free"}:
        return JSONResponse(
            status_code=409,
            content={
                "error": "cost_confirmation_required",
                "status": "cost_confirmation_required",
                "cost_preview": cost_preview,
            },
        )
    if body.cost_decision == "free":
        assistant_provider = "theriad"

    api_key = _resolve_assistant_api_key(assistant_provider)
    if scope_id:
        from theriad.api.auth_context import get_auth_context
        from theriad.billing.service import entitlements_for, request_limit_error

        context = get_auth_context(request)
        entitlements = entitlements_for(
            organization_id=context.organization_id,
            user_id=context.user_id,
            guest=context.principal_type == "guest",
        )
        denial = request_limit_error(
            entitlements,
            is_free_request=body.cost_decision == "free",
            provider_key_configured=bool(api_key),
        )
        if denial:
            status_code = 503 if denial["code"] == "provider_key_required" else 403
            return JSONResponse(status_code=status_code, content={"error": denial["code"], "message": denial["message"], "entitlements": entitlements})
    if not api_key:
        label = "OpenRouter" if assistant_provider == "theriad" else assistant_provider.title()
        return JSONResponse(
            status_code=503,
            content={
                "error": (
                    f"No {label} API key found. Connect it in Settings before executing."
                )
            },
        )

    # ── Build project + enabled-skill context ─────────────────────────────────
    context_prefix: str = ""
    instructions: str | None = None
    from theriad.skills import build_skill_context
    skill_ctx = build_skill_context(body.project_id)
    context_meta: dict = skill_ctx.metadata()
    attached_images: list[tuple[str, str]] = []
    has_project_context = bool(
        body.project_id or body.conversation_id or body.file_ids or skill_ctx.text
    )
    if has_project_context:
        from theriad.api.projects import (
            get_file_images,
            get_file_texts,
            get_project_instructions,
            get_recent_messages,
        )
        from theriad.storage.context import build_context

        instructions = (
            get_project_instructions(body.project_id, scope_id)
            if body.project_id else None
        )
        messages = (
            get_recent_messages(body.conversation_id, scope_id=scope_id)
            if body.conversation_id else []
        )
        file_texts = get_file_texts(body.file_ids, scope_id)
        attached_images = get_file_images(body.file_ids, scope_id)

        ctx_result = build_context(
            messages=messages,
            file_texts=file_texts,
            max_context_tokens=body.max_context_tokens,
            include_paid_warning=bool(file_texts),
        )
        context_prefix = ctx_result.text
        context_meta = {
            **skill_ctx.metadata(),
            "context_tokens": ctx_result.total_tokens,
            "context_truncated": ctx_result.truncated,
            "context_truncation_warning": ctx_result.truncation_warning,
            "paid_provider_warning": ctx_result.paid_provider_warning,
            "history_summarized": ctx_result.history_summarized,
            "messages_summarized": ctx_result.messages_summarized,
            "recent_messages_kept": ctx_result.recent_messages_kept,
        }

    system_prompt = _chat_system_prompt(
        instructions or "",
        skill_ctx.text,
        allow_follow_up_questions=allow_follow_up_questions,
    )
    from theriad.capabilities.detector import CapabilityDetector
    from theriad.capabilities.rendering import renderer_metadata, renderer_system_instruction

    execute_capability = CapabilityDetector().detect(body.prompt)
    execute_rendering = renderer_metadata(execute_capability)
    execute_renderer_instruction = renderer_system_instruction(execute_capability)
    if execute_renderer_instruction:
        system_prompt = f"{system_prompt}\n\n{execute_renderer_instruction}"

    # Compose effective prompt (context prefix + user prompt)
    effective_prompt = (
        f"{context_prefix}\n\n{body.prompt}" if context_prefix else body.prompt
    )

    # ── Route the prompt ──────────────────────────────────────────────────────
    from theriad.accounting.pricing import PricingTable, estimate_cost
    from theriad.accounting.tracker import CostTracker
    from theriad.core.providers.interfaces import CompletionRequest, Message, ProviderError
    from theriad.core.providers.factory import build_provider
    from theriad.models.policy import PolicyResolver

    total_start = time.monotonic()

    routing_start = time.monotonic()
    layer0, layer1 = _build_layers()
    ctx: dict = {}
    routing_result = layer0.route(effective_prompt, ctx) or layer1.route(effective_prompt, ctx)

    if routing_result is None:
        from theriad.core.routing.interfaces import RoutingDecision, RoutingResult
        routing_result = RoutingResult(
            decision=RoutingDecision(
                tier="cheap_bulk",
                confidence=0.0,
                reason="all layers abstained — using fallback tier",
            ),
            layer_name="fallback",
        )

    tier = routing_result.decision.tier
    confidence = routing_result.decision.confidence
    routing_source = routing_result.layer_name
    routing_reason_text = routing_result.decision.reason

    # ── Context estimate + soft continuity signal ─────────────────────────────
    # Continuity is a soft bonus only — it never hard-pins a model.
    # The previous model receives a small score boost; the resolver still picks
    # the best candidate on cost/quality grounds.
    from theriad.routing.context import ContextRequirementEstimator
    context_estimate = ContextRequirementEstimator().estimate(effective_prompt)

    continuity_signal = None
    if body.conversation_id:
        continuity_signal = _sticky_cache.get_continuity_signal(
            body.conversation_id, tier
        )

    resolver = PolicyResolver()
    preferences = _resolved_routing_preferences(body)
    resolution = resolver.resolve(
        tier, body.policy, confidence,
        context_estimate=context_estimate,
        continuity_signal=continuity_signal,
        message=effective_prompt,
        speed_preference=preferences.speed,
        quality_preference=preferences.quality,
        constraints=origin_constraints,
    )
    # Existing policies resolve back to the same tier. The opt-in MoE policy
    # returns its selected expert tier so limits and diagnostics are accurate.
    tier = resolution.tier
    sticky_route_used = resolution.continuity_bonus_applied
    cost_quality_tradeoff = resolution.cost_quality_tradeoff
    routing_latency_ms = (time.monotonic() - routing_start) * 1000
    if sticky_route_used:
        routing_source = "continuity_soft"

    # ── Build execution order without crossing the selected provider ─────────
    candidates = _assistant_candidates(
        assistant_provider,
        tier,
        resolution,
        body.assistant_model,
    )
    if body.cost_decision == "free":
        from theriad.models.registry import DEFAULT_REGISTRY

        free_tier = _free_tier_for(resolution.tier, body.prompt)
        free_candidates = DEFAULT_REGISTRY.candidates_for_tier(free_tier)
        candidates = [(candidate.provider, candidate.model) for candidate in free_candidates]
        tier = free_tier
    all_model_slugs = [m for _, m in candidates]

    # ── Apply project model filters ───────────────────────────────────────────
    if body.project_id:
        from theriad.api.projects import load_project_settings
        from theriad.routing.filters import ModelFilter, ModelFilterSettings

        proj_settings = load_project_settings(body.project_id, scope_id)
        if proj_settings:
            pt_for_filter = PricingTable()
            filter_settings = ModelFilterSettings(
                allowed_model_patterns=proj_settings.get("allowed_model_patterns") or [],
                blocked_model_patterns=proj_settings.get("blocked_model_patterns") or [],
                # Kept in storage for compatibility, but no longer used as a
                # hard model cutoff. The token-cost preflight asks the user.
                max_cost_per_call_usd=None,
                allow_premium=proj_settings.get("allow_premium", True),
                local_only=proj_settings.get("local_only", False),
            )
            model_filter = ModelFilter(filter_settings)
            # Estimate costs for filter
            est_input_tokens = _estimate_prompt_tokens(effective_prompt)
            candidates_with_meta = []
            for prov, mdl in candidates:
                est_cost = estimate_cost(mdl, est_input_tokens, 200, pt_for_filter)
                candidates_with_meta.append((prov, mdl, est_cost, tier))

            filtered = model_filter.filter_candidates(candidates_with_meta)
            if not filtered:
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": (
                            "All candidate models are blocked by this project's model filter. "
                            "Update project settings (allowed_model_patterns / blocked_model_patterns) "
                            "or allow a compatible model."
                        ),
                        "blocked_models": all_model_slugs,
                    },
                )
            candidates = [(prov, mdl) for prov, mdl, _, _ in filtered]
            all_model_slugs = [m for _, m in candidates]

            # Optional: enable per-project openrouter/auto fallback
            if (
                assistant_provider == "theriad"
                and proj_settings.get("enable_openrouter_auto_fallback")
                and not sticky_route_used
            ):
                if confidence < 0.5 and "openrouter/auto" not in all_model_slugs:
                    candidates.append(("openrouter", "openrouter/auto"))
                    all_model_slugs.append("openrouter/auto")

    # ── Global openrouter/auto fallback (from config) ─────────────────────────
    try:
        cfg_for_fb = _load_config()
        if (
            assistant_provider == "theriad"
            and
            getattr(cfg_for_fb.router, "enable_openrouter_auto_fallback", False)
            and confidence < 0.5
            and "openrouter/auto" not in all_model_slugs
        ):
            candidates.append(("openrouter", "openrouter/auto"))
            all_model_slugs.append("openrouter/auto")
    except Exception:
        pass

    # ── Vision gate ───────────────────────────────────────────────────────────
    # An image attachment overrides tier routing on capability grounds: the
    # cheapest capable model is meaningless if it cannot see. Applied after the
    # project filter so an explicitly blocked model is not reintroduced here.
    if attached_images:
        from theriad.models.vision import ensure_vision_candidates

        vision_capable = ensure_vision_candidates(candidates)
        if [m for _, m in vision_capable] != all_model_slugs:
            routing_reason_text = (
                f"{routing_reason_text} → vision required "
                f"({len(attached_images)} image attachment"
                f"{'s' if len(attached_images) != 1 else ''})"
            )
        candidates = vision_capable
        all_model_slugs = [m for _, m in candidates]

    # ── Resolve per-tier limits ───────────────────────────────────────────────
    max_tokens, timeout_secs = _tier_limits(tier)
    # Workflow-detected single-prompt preview: apply a stricter output cap so
    # that executing a build prompt in single-call mode doesn't dump 20k tokens.
    _WORKFLOW_SINGLE_PROMPT_CAP = 4000
    if body.workflow_selected and max_tokens is not None:
        max_tokens = min(max_tokens, _WORKFLOW_SINGLE_PROMPT_CAP)
    elif body.workflow_selected:
        max_tokens = _WORKFLOW_SINGLE_PROMPT_CAP

    # ── Execute with fallback + per-attempt diagnostics ───────────────────────
    execution_attempts: list[dict] = []
    response = None
    used_fallback = False
    last_error: Exception | None = None
    provider_latency_ms = 0.0
    fallback_latency_ms = 0.0
    objective_validation = None
    from theriad.runtime.circuit_breaker import get_circuit_registry, model_circuit_key

    for i, (provider_name, model_slug) in enumerate(candidates):
        attempt_start = time.monotonic()
        # New provider per attempt so timeout is applied correctly
        provider = build_provider(
            provider_name,
            api_key=api_key,
            timeout=timeout_secs,
        )
        provider_user_content = _user_content(body.prompt, attached_images)
        if attached_images and context_prefix:
            # Keep multimodal context in the same user message. Some clients
            # (and providers) treat a preceding text-only user turn as the
            # primary prompt and can drop the image block.
            provider_user_content = [
                {"type": "text", "text": context_prefix},
                *provider_user_content,
            ]
        req = CompletionRequest(
            model=model_slug,
            messages=(
                [Message(role="system", content=system_prompt)]
                + (
                    [Message(role="user", content=context_prefix)]
                    if context_prefix and not attached_images else []
                )
                + [Message(role="user", content=provider_user_content)]
            ),
            max_tokens=max_tokens,
            extra={
                **_provider_request_extra(
                    provider.name,
                    model_slug,
                    preferences,
                    [
                        model
                        for candidate_provider, model in candidates
                        if candidate_provider == "openrouter"
                    ],
                ),
                **__import__(
                    "theriad.storage.prompt_cache", fromlist=["prompt_cache_extra"]
                ).prompt_cache_extra(provider.name),
            },
            session_id=body.conversation_id or None,
        )
        attempt: dict = {
            "attempt": i + 1,
            "model": model_slug,
            "tier": tier,
            "provider": provider.name,
            "status": None,
            "error": None,
            "start_time": attempt_start,
            "end_time": None,
            "latency_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "tokens_per_second": None,
            "timeout_triggered": False,
            "fallback_reason": None,
        }
        try:
            candidate_response = await provider.complete(req)
            attempt_end = time.monotonic()
            latency = (attempt_end - attempt_start) * 1000
            tps = (
                candidate_response.completion_tokens / (latency / 1000)
                if latency > 0 and candidate_response.completion_tokens > 0 else None
            )
            finish_reason = candidate_response.finish_reason

            # ── Empty output guard ────────────────────────────────────────────
            # A provider that reports completion_tokens > 0 but returns empty
            # content has failed to deliver usable output. Treat as a soft
            # failure so the fallback chain can try the next model.
            content_is_empty = not candidate_response.content.strip()
            if content_is_empty and candidate_response.completion_tokens > 0:
                get_circuit_registry().record_failure(
                    model_circuit_key(provider.name, model_slug),
                    RuntimeError("empty provider output"),
                    latency_ms=latency,
                )
                attempt.update({
                    "status": "empty_output",
                    "error": (
                        f"Provider reported {candidate_response.completion_tokens} output tokens "
                        f"but returned no renderable content (finish_reason={finish_reason!r})."
                    ),
                    "end_time": attempt_end,
                    "latency_ms": round(latency, 1),
                    "input_tokens": candidate_response.prompt_tokens,
                    "output_tokens": candidate_response.completion_tokens,
                    "tokens_per_second": round(tps, 1) if tps else None,
                    "finish_reason": finish_reason,
                    "timeout_triggered": False,
                    "fallback_reason": "empty_output",
                })
                execution_attempts.append(attempt)
                fallback_latency_ms += latency
                last_error = Exception(attempt["error"])
                continue  # try next model

            get_circuit_registry().record_success(
                model_circuit_key(provider.name, model_slug), latency_ms=latency
            )

            from theriad.quality.objective_verifier import verify_objective_response

            objective_validation = verify_objective_response(
                body.prompt, candidate_response.content
            )
            if (
                objective_validation is not None
                and not objective_validation.passed
                and i < len(candidates) - 1
            ):
                attempt.update({
                    "status": "validation_failed",
                    "error": objective_validation.details,
                    "end_time": attempt_end,
                    "latency_ms": round(latency, 1),
                    "input_tokens": candidate_response.prompt_tokens,
                    "output_tokens": candidate_response.completion_tokens,
                    "tokens_per_second": round(tps, 1) if tps else None,
                    "finish_reason": finish_reason,
                    "fallback_reason": "objective_validation_failed",
                    "objective_validation": objective_validation.to_dict(),
                })
                execution_attempts.append(attempt)
                fallback_latency_ms += latency
                last_error = Exception(objective_validation.details)
                continue

            attempt.update({
                "status": "success",
                "end_time": attempt_end,
                "latency_ms": round(latency, 1),
                "input_tokens": candidate_response.prompt_tokens,
                "output_tokens": candidate_response.completion_tokens,
                "tokens_per_second": round(tps, 1) if tps else None,
                "finish_reason": finish_reason,
                "objective_validation": (
                    objective_validation.to_dict() if objective_validation else None
                ),
            })
            execution_attempts.append(attempt)
            response = candidate_response
            used_fallback = i > 0
            provider_latency_ms = latency
            break
        except ProviderError as exc:
            attempt_end = time.monotonic()
            latency = (attempt_end - attempt_start) * 1000
            timeout_hit = "timeout" in str(exc).lower()
            transient = exc.status_code is None or exc.status_code in {408, 409, 425, 429} or (
                isinstance(exc.status_code, int) and exc.status_code >= 500
            )
            get_circuit_registry().record_failure(
                model_circuit_key(provider.name, model_slug),
                exc,
                latency_ms=latency,
                transient=transient,
            )
            attempt.update({
                "status": "error",
                "error": str(exc),
                "end_time": attempt_end,
                "latency_ms": round(latency, 1),
                "finish_reason": None,
                "timeout_triggered": timeout_hit,
                "fallback_reason": "provider_error",
            })
            execution_attempts.append(attempt)
            fallback_latency_ms += latency
            last_error = exc
            # Non-retryable 4xx (except 429): stop immediately
            if exc.status_code and 400 <= exc.status_code < 500 and exc.status_code != 429:
                break
        except Exception as exc:
            attempt_end = time.monotonic()
            latency = (attempt_end - attempt_start) * 1000
            get_circuit_registry().record_failure(
                model_circuit_key(provider.name, model_slug), exc, latency_ms=latency
            )
            timeout_hit = "timeout" in str(exc).lower() or "TimeoutException" in type(exc).__name__
            attempt.update({
                "status": "error",
                "error": str(exc),
                "end_time": attempt_end,
                "latency_ms": round(latency, 1),
                "finish_reason": None,
                "timeout_triggered": timeout_hit,
                "fallback_reason": "unexpected_error",
            })
            execution_attempts.append(attempt)
            fallback_latency_ms += latency
            last_error = exc

    total_latency_ms = (time.monotonic() - total_start) * 1000

    # ── Update sticky cache ───────────────────────────────────────────────────
    if body.conversation_id:
        if response is not None:
            _sticky_cache.set(
                body.conversation_id,
                selected_model=response.model,
                selected_tier=tier,
                provider=response.provider,
                reason="continuity_retained" if sticky_route_used else "first_call",
            )
        else:
            _sticky_cache.clear(body.conversation_id, reason="provider_error")

    if response is None:
        # Distinguish between "all calls errored" vs "all calls returned empty content"
        statuses = {a.get("status") for a in execution_attempts}
        all_empty = statuses == {"empty_output"} if statuses else False
        error_msg = (
            "The selected provider returned empty output despite reporting completion tokens. "
            "This may indicate a model-specific content encoding issue (e.g. finish_reason=length "
            "with structured content that wasn't extracted). See execution_attempts for details."
            if all_empty
            else f"The selected provider failed. Last error: {last_error}"
        )
        return JSONResponse(
            status_code=502,
            content={
                "status": "empty_output" if all_empty else "error",
                "error": error_msg,
                "fallback_chain": all_model_slugs,
                "execution_attempts": [
                    {k: v for k, v in a.items() if k not in ("start_time", "end_time")}
                    for a in execution_attempts
                ],
                "routing_latency_ms": round(routing_latency_ms, 1),
                "total_latency_ms": round(round(time.monotonic() - total_start, 3) * 1000, 1),
            },
        )

    # ── Compute realized cost from provider billing or real token counts ──
    from theriad.accounting.actuals import cached_prompt_tokens, realized_cost, usage_is_available

    pt = PricingTable()
    actual_cost = realized_cost(response, pt)
    prompt_cache_tokens = cached_prompt_tokens(response)

    # ── Write inference record ────────────────────────────────────────────────
    try:
        cfg2 = _load_config()
        metrics_path = cfg2.metrics_path
    except Exception:
        metrics_path = "~/.theriad/metrics.jsonl"

    if scope_id is None:
        # Preserve the local CLI/desktop compatibility file. Authenticated
        # requests are represented by the tenant-scoped runs table instead.
        tracker = CostTracker(metrics_path=metrics_path)
        tracker.record(
            provider=response.provider,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
            routing_layer=routing_result.layer_name,
            routing_reason=body.prompt[:200],
            used_fallback=used_fallback,
            tags={
                "policy": body.policy,
                "tier": tier,
                "source": "api_execute",
            },
        )

    # Strip raw monotonic timestamps from attempt records before returning
    clean_attempts = [
        {k: v for k, v in a.items() if k not in ("start_time", "end_time")}
        for a in execution_attempts
    ]

    # Models configured but NOT actually called (distinct from execution_attempts)
    actually_tried = {a["model"] for a in execution_attempts}
    fallback_available = [m for m in all_model_slugs if m not in actually_tried]

    # ── Persist messages + run when conversation context is active ────────────
    saved_user_message_id: str | None = None
    saved_assistant_message_id: str | None = None
    saved_run_id: str | None = None
    persistence_status = "not_requested"
    persistence_error: str | None = None
    _routing_observations = None
    if resolution.router_v2_decision is not None:
        from theriad.routing.v2 import observation_rows

        _routing_observations = observation_rows(
            resolution.router_v2_decision,
            prompt=body.prompt,
            actual_model=response.model,
            actual_cost_usd=actual_cost,
            actual_latency_ms=provider_latency_ms,
            input_tokens=response.prompt_tokens,
            output_tokens=response.completion_tokens,
        )
    _run_kwargs = dict(
        model=response.model,
        tier=tier,
        provider=response.provider,
        policy=body.policy,
        input_tokens=response.prompt_tokens,
        output_tokens=response.completion_tokens,
        actual_cost_usd=actual_cost,
        routing_latency_ms=routing_latency_ms,
        provider_latency_ms=provider_latency_ms,
        total_latency_ms=total_latency_ms,
        fallback_used=used_fallback,
        status="success",
        # Extended routing metadata
        routing_source=routing_source,
        routing_reason=routing_reason_text,
        sticky_route_used=sticky_route_used,
        cost_quality_tradeoff=cost_quality_tradeoff,
        fallback_chain=all_model_slugs,
        routing_observations=_routing_observations,
    )
    if body.conversation_id:
        persistence_status = "pending"
        try:
            from theriad.api.projects import save_message, save_run
            saved_user_message_id = save_message(
                conversation_id=body.conversation_id,
                role="user",
                content=body.prompt,  # save original prompt, not with context prefix
            )
            saved_assistant_message_id = save_message(
                conversation_id=body.conversation_id,
                role="assistant",
                content=response.content,
            )
            saved_run_id = save_run(
                project_id=body.project_id,
                conversation_id=body.conversation_id,
                message_id=saved_assistant_message_id,
                **_run_kwargs,
                user_id=scope_id,
            )
            persistence_status = "saved"
        except Exception as exc:
            # The provider has already completed, so preserve its response, but
            # make a durable-history failure explicit to the caller and logs.
            # Returning a normal-looking success here made missing chat records
            # impossible to distinguish from a successful save.
            persistence_status = "failed"
            persistence_error = (
                "Execution completed, but the conversation record could not be saved."
            )
            logger.error(
                "execute conversation persistence failed",
                extra={
                    "conversation_id": body.conversation_id,
                    "project_id": body.project_id,
                    "user_id": scope_id,
                    "error_type": type(exc).__name__,
                },
                exc_info=True,
            )
    elif body.project_id:
        # project_id without conversation_id: still save a run record
        persistence_status = "pending"
        try:
            from theriad.api.projects import save_run
            saved_run_id = save_run(
                project_id=body.project_id,
                conversation_id=None,
                message_id=None,
                **_run_kwargs,
                user_id=scope_id,
            )
            persistence_status = "saved"
        except Exception as exc:
            persistence_status = "failed"
            persistence_error = "Execution completed, but the run record could not be saved."
            logger.error(
                "execute project run persistence failed",
                extra={
                    "project_id": body.project_id,
                    "user_id": scope_id,
                    "error_type": type(exc).__name__,
                },
                exc_info=True,
            )
    elif scope_id is not None:
        persistence_status = "pending"
        try:
            from theriad.api.projects import save_run
            saved_run_id = save_run(
                project_id=None,
                conversation_id=None,
                message_id=None,
                **_run_kwargs,
                user_id=scope_id,
            )
            persistence_status = "saved"
        except Exception as exc:
            persistence_status = "failed"
            persistence_error = "Execution completed, but the run record could not be saved."
            logger.error(
                "execute account run persistence failed",
                extra={
                    "user_id": scope_id,
                    "error_type": type(exc).__name__,
                },
                exc_info=True,
            )

    _record_routing_usage(
        request,
        run_id=saved_run_id,
        provider=response.provider,
        model=response.model,
        input_tokens=response.prompt_tokens,
        output_tokens=response.completion_tokens,
        provider_cost_usd=actual_cost,
        paid_request=body.cost_decision != "free",
        usage_known=usage_is_available(response),
    )

    result: dict = {
        "output": response.content,
        "capability": execute_capability.to_dict(),
        "rendering": execute_rendering,
        "selected_tier": tier,
        "selected_model": response.model,
        "provider": response.provider,
        "assistant_provider": assistant_provider,
        "routing_scope": "cross_provider" if assistant_provider == "theriad" else "provider_direct",
        "policy": body.policy,
        "speed_preference": preferences.speed,
        "quality_preference": preferences.quality,
        "estimated_price_multiplier": preferences.price_multiplier,
        "cost_preview": cost_preview,
        "actual_cost_usd": round(actual_cost, 8) if actual_cost is not None else None,
        "prompt_cache_used": prompt_cache_tokens > 0,
        "input_tokens": response.prompt_tokens,
        "output_tokens": response.completion_tokens,
        "fallback_used": used_fallback,
        # execution_attempts: models actually called (with status + diagnostics)
        "execution_attempts": clean_attempts,
        # fallback_available: configured fallbacks that were NOT needed this run
        "fallback_available": fallback_available,
        # All configured slugs in priority order
        "fallback_chain": all_model_slugs,
        "router_v2": (
            {
                "router_version": resolution.router_version,
                "lane": resolution.lane,
                "knee_threshold": resolution.knee_threshold,
                "ood_score": resolution.ood_score,
                "shadow": resolution.shadow,
                "shadow_challenger": resolution.shadow_challenger,
                "applied": resolution.router_v2_decision.applied,
            }
            if resolution.router_v2_decision is not None
            else None
        ),
        "objective_validation": (
            objective_validation.to_dict() if objective_validation else None
        ),
        "status": "success",
        "error": None,
        # Routing metadata (for Chat Inspector, Runs, Logs)
        "routing_source": routing_source,
        "routing_reason": routing_reason_text,
        "sticky_route_used": sticky_route_used,
        "cost_quality_tradeoff": cost_quality_tradeoff,
        # Context-window + continuity diagnostics
        "selected_context_bucket": resolution.selected_context_bucket,
        "selected_model_context_limit": resolution.selected_model_context_limit,
        "context_fit_reason": resolution.context_fit_reason,
        "compression_or_chunking_recommended": resolution.compression_or_chunking_recommended,
        "continuity_bonus_applied": resolution.continuity_bonus_applied,
        "continuity_bonus_weight": resolution.continuity_bonus_weight,
        "reason_for_stay_or_switch": resolution.reason_for_stay_or_switch,
        # Workflow boundary note — surfaced here so clients can display it
        "execution_note": (
            "Workflow analysis only. Execute currently runs single-prompt mode."
        ),
        "workflow_single_prompt_cap_applied": body.workflow_selected,
        # Separated latency breakdown
        "routing_latency_ms": round(routing_latency_ms, 1),
        "provider_latency_ms": round(provider_latency_ms, 1),
        "fallback_latency_ms": round(fallback_latency_ms, 1),
        "total_latency_ms": round(total_latency_ms, 1),
        # Legacy field — kept for backwards compatibility
        "latency_ms": round(response.latency_ms, 1),
        # Project context metadata (only present when project context was used)
        **context_meta,
        # Persistence IDs (null when not using project context)
        "saved_run_id": saved_run_id,
        "saved_user_message_id": saved_user_message_id,
        "saved_assistant_message_id": saved_assistant_message_id,
        "persistence": {
            "status": persistence_status,
            "error": persistence_error,
        },
    }
    return result


# ── POST /api/simulate ────────────────────────────────────────────────────────

@app.post("/api/simulate")
def simulate(body: SimulateRequest):
    """
    Offline batch cost simulation across a list of prompts. No provider calls.

    Routes each prompt, resolves to a concrete model, estimates token costs,
    and compares against the baseline model.
    """
    from theriad.accounting.pricing import PricingTable, estimate_cost, platform_fee
    from theriad.models.policy import PolicyResolver
    from theriad.storage.hosted import hosted_data_plane_enabled, hosted_repositories

    layer0, layer1 = _build_layers()
    resolver = PolicyResolver()
    pt = PricingTable()

    tier_counts: dict[str, int] = {}
    theriad_total = 0.0
    baseline_total = 0.0
    premium_count = 0
    results = []

    for item in body.prompts:
        prompt_tokens = _estimate_prompt_tokens(item.prompt)
        out_tokens = item.expected_output_tokens

        ctx: dict = {}
        routing_result = layer0.route(item.prompt, ctx) or layer1.route(item.prompt, ctx)

        if routing_result is None:
            from theriad.core.routing.interfaces import RoutingDecision, RoutingResult
            routing_result = RoutingResult(
                decision=RoutingDecision(
                    tier="cheap_bulk",
                    confidence=0.0,
                    reason="all layers abstained",
                ),
                layer_name="fallback",
            )

        tier = routing_result.decision.tier

        resolution = resolver.resolve(
            tier,
            body.policy,
            routing_result.decision.confidence,
            message=item.prompt,
            speed_preference=_resolved_routing_preferences(body).speed,
            quality_preference=_resolved_routing_preferences(body).quality,
        )
        tier = resolution.tier
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if tier == "premium":
            premium_count += 1
        theriad_cost = estimate_cost(resolution.model, prompt_tokens, out_tokens, pt) or 0.0
        baseline_cost = estimate_cost(body.baseline, prompt_tokens, out_tokens, pt) or 0.0

        theriad_total += theriad_cost
        baseline_total += baseline_cost

        results.append({
            "id": item.id,
            "prompt_preview": item.prompt[:80],
            "tier": tier,
            "model": resolution.model,
            "prompt_tokens": prompt_tokens,
            "output_tokens": out_tokens,
            "theriad_cost_usd": round(theriad_cost, 8),
            "baseline_cost_usd": round(baseline_cost, 8),
        })

    n = len(results)
    savings_pct = (
        (baseline_total - theriad_total) / baseline_total * 100
        if baseline_total > 0 else 0.0
    )

    # Sort tier counts descending for display
    top_tiers = sorted(tier_counts.items(), key=lambda x: -x[1])

    # Top expensive prompts (highest theriad_cost first)
    top_expensive = sorted(results, key=lambda r: -r["theriad_cost_usd"])[:5]

    return {
        "prompts": n,
        "baseline_model": body.baseline,
        "policy": body.policy,
        "theriad_estimated_cost_usd": round(theriad_total, 6),
        "baseline_estimated_cost_usd": round(baseline_total, 6),
        "estimated_savings_pct": round(savings_pct, 1),
        "premium_usage_pct": round(premium_count / n * 100, 1) if n > 0 else 0.0,
        "avg_cost_per_prompt": round(theriad_total / n, 6) if n > 0 else 0.0,
        "tier_counts": dict(top_tiers),
        "top_expensive": top_expensive,
        "note": "Dry-run only. No provider calls were made.",
    }


# ── GET /api/models ───────────────────────────────────────────────────────────

@app.get("/api/models")
def get_models(request: Request):
    """Return the live tier registry — one card per tier with primary model + fallback chain."""
    from theriad.models.registry import DEFAULT_REGISTRY
    from theriad.models.unified import NO_CONSTRAINTS

    constraints = _routing_constraints(request)
    result = []
    for tier in DEFAULT_REGISTRY.all_tiers():
        if constraints is NO_CONSTRAINTS:
            primary = DEFAULT_REGISTRY.primary_for_tier(tier)
            fallbacks = DEFAULT_REGISTRY.fallback_chain_for_tier(tier)
        else:
            eligible = [
                candidate
                for candidate in DEFAULT_REGISTRY.candidates_for_tier(tier)
                if constraints.permits(candidate)
            ]
            primary = eligible[0] if eligible else None
            fallbacks = eligible[1:] if eligible else []
        if primary is None:
            continue
        price_in, price_out, _is_live = _effective_prices(primary)
        result.append({
            "tier": tier,
            "model": primary.model,
            "provider": primary.provider.title(),
            "origin": primary.origin,
            "model_origin_policy": "us_only" if constraints.active else "any",
            "p_in": _format_price_per_mtok(price_in),
            "p_out": _format_price_per_mtok(price_out),
            "ctx": _format_ctx(primary.context_window),
            "q_prior": str(round(primary.quality_prior, 2))
                       if primary.quality_prior is not None else "?",
            "use": _TIER_USE.get(tier, "General purpose"),
            "risk": _TIER_RISK.get(tier, "Medium"),
            "badges": _tier_badges(primary),
            "notes": primary.notes or "",
            "fallback_chain": [
                {"model": f.model, "provider": f.provider, "origin": f.origin, "role": f.role}
                for f in fallbacks
            ],
        })
    return result


# ── GET /api/pricing ──────────────────────────────────────────────────────────

@app.get("/api/pricing")
def get_pricing():
    """Return current per-model prices plus honest freshness metadata.

    The UI prices its comparison from this endpoint rather than hardcoding
    constants, so a price change in the catalog cannot leave the marketing
    numbers quietly wrong.

    ``stale: true`` means the catalog was unreachable and these are
    last-known-good values — still the best available, never zeros.
    """
    from theriad.accounting.live_prices import get_live_snapshot
    from theriad.models.registry import DEFAULT_REGISTRY

    try:
        snapshot = get_live_snapshot()
        meta = snapshot.as_dict()
    except Exception as exc:                         # pragma: no cover - defensive
        snapshot = None
        meta = {"stale": True, "origin": "empty", "error": str(exc), "model_count": 0}

    models: dict[str, dict] = {}
    seen: set[str] = set()
    for tier in DEFAULT_REGISTRY.all_tiers():
        for candidate in DEFAULT_REGISTRY.candidates_for_tier(tier):
            if candidate.model in seen:
                continue
            seen.add(candidate.model)
            price_in, price_out, is_live = _effective_prices(candidate)
            models[candidate.model] = {
                "input_per_1m": price_in,
                "output_per_1m": price_out,
                "display_in": _format_price_per_mtok(price_in),
                "display_out": _format_price_per_mtok(price_out),
                "live": is_live,
                "tier": tier,
            }

    # The public landing comparison includes fixed benchmark models that are
    # intentionally not routing candidates. Expose their live catalog prices
    # through the same endpoint so the UI does not silently fall back to stale
    # client-only values for those benchmarks.
    from theriad.accounting.pricing import PricingTable

    seeded_prices = PricingTable(use_live=False)
    for slug in ("openai/gpt-5.6-sol", "anthropic/claude-opus-5"):
        catalog_price = snapshot.prices.get(slug) if snapshot is not None else None
        if catalog_price is not None:
            price_in, price_out = catalog_price
            is_live = snapshot.stale is False
        else:
            seed = seeded_prices.get(slug)
            if seed is None:
                continue
            price_in, price_out = seed.input_per_1m, seed.output_per_1m
            is_live = False
        models[slug] = {
            "input_per_1m": price_in,
            "output_per_1m": price_out,
            "display_in": _format_price_per_mtok(price_in),
            "display_out": _format_price_per_mtok(price_out),
            "live": is_live,
            "tier": "comparison",
        }

    return {"models": models, **meta}


# ── GET /api/pricing/plans ────────────────────────────────────────────────────

@app.get("/api/pricing/plans")
def get_pricing_plans():
    """Return the OpenRouter-first self-serve and enterprise pricing policy.

    This is a public policy surface, not a billing endpoint. Theriad access and
    Paid OpenRouter model usage is capped by the plan allowance first, then
    continues pay-as-you-go through OpenRouter.
    """
    from theriad.pricing_plans import get_pricing_catalog

    return get_pricing_catalog()


# ── GET /api/stats ────────────────────────────────────────────────────────────

def _record_cost(record: dict) -> float | None:
    """Return a finite recorded cost, preserving an unknown cost as None."""
    value = record.get("estimated_cost_usd")
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) and parsed >= 0 else None


@app.get("/api/stats")
def get_stats(request: Request):
    """Aggregate stats from the current account's durable run records."""
    from collections import defaultdict
    from theriad.accounting.scoped_metrics import load_scoped_records
    from theriad.settings.deployment import is_hosted_mode

    scope_id = user_scope_id(request)
    # Anonymous local/dev keeps the legacy JSONL surface for CLI and existing
    # local installs. Authenticated requests always use their durable scope;
    # hosted requests never consult a process-global file.
    if hosted_data_plane_enabled() and is_hosted_mode():
        from theriad.api.auth_context import get_auth_context

        context = get_auth_context(request)
        if context.principal_type in {"registered", "guest"}:
            # Keep Costs on the same tenant-scoped durable source as Logs.
            records = _hosted_inference_records(request, 1000)
        else:
            records = []
    elif scope_id is None and not is_hosted_mode():
        from theriad.accounting.tracker import CostTracker

        try:
            cfg = _load_config()
            metrics_path = cfg.metrics_path
        except Exception:
            metrics_path = "~/.theriad/metrics.jsonl"
        records = CostTracker.load_from_disk(metrics_path)
    else:
        records = load_scoped_records(scope_id)

    if not records:
        return {
            "has_data": False,
            "total_requests": 0,
            "total_cost_usd": 0.0,
            "avg_latency_ms": 0.0,
            "fallback_rate": 0.0,
            "premium_usage_pct": 0.0,
            "models_used": {},
            "daily_costs": [],
            "recent_high_cost": [],
            "baseline_model": DEFAULT_BASELINE_MODEL,
            "baseline_cost_usd": 0.0,
            "estimated_savings_usd": 0.0,
            "estimated_savings_pct": 0.0,
            "total_tokens": 0,
            "blended_cost_per_mtok": None,
            "baseline_blended_cost_per_mtok": None,
            "known_cost_requests": 0,
            "unknown_cost_requests": 0,
            "costs_complete": True,
            "cost_status": "no_records",
        }

    from theriad.accounting.pricing import PricingTable, estimate_cost

    record_costs = [_record_cost(record) for record in records]
    known_costs = [cost for cost in record_costs if cost is not None]
    unknown_cost_requests = len(records) - len(known_costs)
    costs_complete = unknown_cost_requests == 0
    total_cost = sum(known_costs)
    if unknown_cost_requests == 0 and total_cost == 0:
        cost_status = "recorded_zero"
    elif unknown_cost_requests == len(records):
        cost_status = "unavailable"
    elif unknown_cost_requests:
        cost_status = "partial"
    else:
        cost_status = "recorded"
    baseline_model = DEFAULT_BASELINE_MODEL
    pricing = PricingTable()
    baseline_cost = sum(
        estimate_cost(
            baseline_model,
            int(r.get("prompt_tokens", 0) or 0),
            int(r.get("completion_tokens", 0) or 0),
            pricing,
        )
        for r in records
    )
    estimated_savings = baseline_cost - total_cost if costs_complete else None
    estimated_savings_pct = (
        estimated_savings / baseline_cost * 100
        if estimated_savings is not None and baseline_cost > 0
        else None
    )
    # Blended (projected) rate: what a million tokens costs at the actual
    # model mix used so far, vs the same tokens on the premium baseline.
    total_tokens_all = sum(
        int(r.get("prompt_tokens", 0) or 0) + int(r.get("completion_tokens", 0) or 0)
        for r in records
    )
    blended_per_mtok = (
        total_cost / total_tokens_all * 1_000_000
        if costs_complete and total_tokens_all > 0 else None
    )
    baseline_per_mtok = (
        baseline_cost / total_tokens_all * 1_000_000 if total_tokens_all > 0 else None
    )
    avg_latency = sum(r.get("latency_ms", 0) for r in records) / len(records)
    fallbacks = sum(1 for r in records if r.get("used_fallback"))
    models: dict[str, int] = {}
    for r in records:
        m = r.get("model", "unknown")
        models[m] = models.get(m, 0) + 1

    # Estimate savings against the canonical premium comparison baseline.
    from theriad.accounting.pricing import PricingTable, estimate_cost as _est_cost
    _pt = PricingTable()
    baseline_total = 0.0
    for r in records:
        prompt_tok = r.get("prompt_tokens", 0)
        completion_tok = r.get("completion_tokens", 0)
        b = _est_cost(DEFAULT_BASELINE_MODEL, prompt_tok, completion_tok, _pt)
        baseline_total += b if b else 0.0
    savings_usd = baseline_total - total_cost if costs_complete else None
    savings_pct = (
        round(savings_usd / baseline_total * 100, 1)
        if savings_usd is not None and baseline_total > 0 else None
    )

    # Daily cost buckets (last 14 calendar days, including zero-cost days).
    # Returning the complete range lets the UI truthfully label the chart.
    from datetime import date, timedelta
    daily: dict[str, float] = defaultdict(float)
    for r, record_cost in zip(records, record_costs):
        ts = r.get("timestamp", "")
        if ts and record_cost is not None:
            daily[ts[:10]] += record_cost
    today = date.today()
    daily_costs = [
        {"date": (today - timedelta(days=13 - i)).isoformat(),
         "cost": round(daily.get((today - timedelta(days=13 - i)).isoformat(), 0.0), 6)}
        for i in range(14)
    ]

    # Recent high-cost prompts
    by_cost = sorted(
        (record for record, cost in zip(records, record_costs) if cost is not None),
        key=lambda record: _record_cost(record) or 0.0,
        reverse=True,
    )
    recent_high = [
        {
            "prompt": r.get("routing_reason", "(no preview)")[:80],
            "tier": r.get("routing_layer", "unknown"),
            "model": r.get("model", "unknown"),
            "cost_usd": _record_cost(r),
        }
        for r in by_cost[:5]
    ]

    # Premium usage proxy (models with high cost)
    premium = sum(
        1 for r in records
        if any(s in r.get("model", "") for s in ("claude-opus", "claude-sonnet", "premium"))
    )

    return {
        "has_data": True,
        "total_requests": len(records),
        "total_cost_usd": round(total_cost, 4),
        "baseline_cost_usd": round(baseline_total, 4),
        "savings_usd": round(savings_usd, 4) if savings_usd is not None else None,
        "savings_pct": savings_pct,
        "avg_latency_ms": round(avg_latency, 0),
        "fallback_rate": round(fallbacks / len(records) * 100, 1),
        "premium_usage_pct": round(premium / len(records) * 100, 1),
        "models_used": models,
        "daily_costs": daily_costs,
        "recent_high_cost": recent_high,
        "baseline_model": baseline_model,
        "baseline_cost_usd": round(baseline_cost, 6),
        "estimated_savings_usd": (
            round(estimated_savings, 6) if estimated_savings is not None else None
        ),
        "estimated_savings_pct": (
            round(estimated_savings_pct, 1)
            if estimated_savings_pct is not None else None
        ),
        "total_tokens": total_tokens_all,
        "blended_cost_per_mtok": (
            round(blended_per_mtok, 4) if blended_per_mtok is not None else None
        ),
        "baseline_blended_cost_per_mtok": (
            round(baseline_per_mtok, 4) if baseline_per_mtok is not None else None
        ),
        "known_cost_requests": len(known_costs),
        "unknown_cost_requests": unknown_cost_requests,
        "costs_complete": costs_complete,
        "cost_status": cost_status,
    }


# ── POST /api/workflow/analyze ────────────────────────────────────────────────

class WorkflowAnalyzeRequest(BaseModel):
    prompt: str
    policy: str = "balanced"
    baseline: str = DEFAULT_BASELINE_MODEL
    speed_preference: int | None = Field(default=None, ge=0, le=100)
    quality_preference: int | None = Field(default=None, ge=0, le=100)


def _create_workflow_detector():
    """
    Create a WorkflowDetector with the default (SentenceTransformer) backend.
    Override in tests via monkeypatching to avoid model downloads:
        monkeypatch.setattr(server, '_create_workflow_detector', lambda: ...)
    """
    from theriad.workflow.detector import WorkflowDetector
    return WorkflowDetector()


@app.post("/api/workflow/analyze")
def analyze_workflow(body: WorkflowAnalyzeRequest):
    """
    Offline workflow analysis: classify prompt → plan steps → estimate cost.

    No provider calls. No OPENROUTER_API_KEY required.
    Returns workflow plan and cost breakdown when a multi-step workflow is detected,
    or a simple {workflow_selected: false} response for single-call prompts.
    """
    from dataclasses import asdict

    from theriad.workflow.cost_simulator import WorkflowCostSimulator
    from theriad.workflow.planner import WorkflowPlanner

    detector = _create_workflow_detector()
    detection = detector.detect(body.prompt)

    if detection.is_workflow:
        planner = WorkflowPlanner()
        steps = planner.plan(detection, body.prompt, policy=body.policy)
        simulator = WorkflowCostSimulator()
        sim = simulator.simulate(
            steps,
            policy=body.policy,
            baseline=body.baseline,
            speed_preference=body.speed_preference,
            quality_preference=body.quality_preference,
        )
        steps_data = [asdict(s) for s in steps]
    else:
        steps_data = []
        sim = {
            "total_cost_usd": 0.0,
            "baseline_cost_usd": 0.0,
            "savings_pct": 0.0,
            "latency_ms": 0,
            "step_costs": [],
        }

    return {
        "workflow_selected": detection.is_workflow,
        "reason": detection.reason,
        "signals": detection.signals,
        "workflow_type": detection.workflow_type,
        "decision_source": detection.decision_source,
        "confidence": round(detection.confidence, 3),
        "semantic_margin": (
            round(detection.semantic_margin, 4)
            if detection.semantic_margin is not None else None
        ),
        "embedding_model": detection.embedding_model,
        "embedding_dims_used": detection.embedding_dims_used,
        "steps": steps_data,
        # Additive workflow detail for the Analyze UI. Existing clients can
        # ignore this field; it exposes the simulator's concrete model picks
        # without changing routing or execution contracts.
        "step_costs": sim.get("step_costs", []),
        "estimated_total_cost_usd": sim["total_cost_usd"],
        "baseline_estimated_cost_usd": sim["baseline_cost_usd"],
        "estimated_savings_pct": sim["savings_pct"],
        "estimated_latency_ms": sim["latency_ms"],
        "policy": body.policy,
        "speed_preference": _resolved_routing_preferences(body).speed,
        "quality_preference": _resolved_routing_preferences(body).quality,
        "estimated_price_multiplier": _resolved_routing_preferences(body).price_multiplier,
        "baseline": body.baseline,
    }


# ── POST /api/chat/send ───────────────────────────────────────────────────────

# ── POST /api/capabilities/analyze ───────────────────────────────────────────

class CapabilityAnalyzeRequest(BaseModel):
    prompt: str
    policy: str = "balanced"
    conversation_id: str | None = None
    project_id: str | None = None


@app.post("/api/capabilities/analyze")
def analyze_capabilities(body: CapabilityAnalyzeRequest):
    """
    Rule-based capability analysis.

    Returns what execution capabilities the prompt requires, the recommended
    execution mode, risk level, and whether user confirmation is needed before
    starting an agentic runtime.

    No provider calls.  Always free to call.
    """
    from theriad.capabilities.detector import CapabilityDetector
    result = CapabilityDetector().detect(body.prompt)
    return result.to_dict()


# ── POST /api/chat/send ───────────────────────────────────────────────────────

class ChatSendRequest(BaseModel):
    project_id: str | None = None
    conversation_id: str | None = None
    message: str
    file_ids: list[str] = []
    policy: str = "balanced"
    baseline: str = DEFAULT_BASELINE_MODEL
    speed_preference: int | None = Field(default=None, ge=0, le=100)
    quality_preference: int | None = Field(default=None, ge=0, le=100)
    # None follows the provider selected in Settings.
    assistant_provider: str | None = None
    assistant_model: str | None = None
    mode: str = "dry_run"          # "dry_run" | "execute"
    allow_paid_calls: bool = False
    # None follows the account runtime setting; a bool is a per-request override.
    allow_follow_up_questions: bool | None = None
    # Inject the user's named memory collections into the prompt context.
    memory_enabled: bool = True
    # Opt-in quality loop.  When None, policy-driven defaults apply.
    # True forces the loop on; False forces it off regardless of policy.
    quality_loop_enabled: bool | None = None
    # Required only after the server flags an unusually large request.
    # "paid" keeps the selected route; "free" explicitly switches to a free model.
    cost_decision: str | None = Field(default=None, pattern="^(paid|free)$")


def _free_tier_for(tier: str, message: str = "") -> str:
    from theriad.models.tiers import (
        TIER_CHEAP_CODE,
        TIER_CODE_STRONG,
        TIER_FREE_CODE,
        TIER_FREE_GENERAL,
        TIER_FREE_SUMMARIZE,
        TIER_FREE_VERIFY,
        TIER_STRONG_REASONING,
    )

    normalized = message.lower()
    if tier in {TIER_CHEAP_CODE, TIER_CODE_STRONG}:
        return TIER_FREE_CODE
    if any(word in normalized for word in ("summarize", "summary", "extract", "rewrite")):
        return TIER_FREE_SUMMARIZE
    if tier == TIER_STRONG_REASONING or any(
        word in normalized for word in ("verify", "fact-check", "prove", "check this claim")
    ):
        return TIER_FREE_VERIFY
    return TIER_FREE_GENERAL


def _chat_cost_preview(
    body: ChatSendRequest,
    *,
    constraints=None,
    allow_follow_up_questions: bool | None = None,
    free_only: bool = False,
    scope_id: str | None = None,
) -> dict:
    """Estimate a chat request without saving messages or calling a provider."""
    from theriad.accounting.pricing import PricingTable, estimate_cost
    from theriad.accounting.token_guard import assess_token_risk, estimate_tokens
    from theriad.api.projects import (
        get_file_texts,
        get_project_instructions,
        get_recent_messages,
        load_project_settings,
    )
    from theriad.models.policy import PolicyResolver
    from theriad.models.registry import DEFAULT_REGISTRY
    from theriad.skills import build_skill_context
    from theriad.storage.context import build_context
    instructions = (
        get_project_instructions(body.project_id, scope_id)
        if body.project_id else None
    )
    messages = (
        get_recent_messages(body.conversation_id, limit=200, scope_id=scope_id)
        if body.conversation_id else []
    )
    file_texts = get_file_texts(body.file_ids, scope_id)
    skill_ctx = build_skill_context(body.project_id)
    raw_source = "\n".join(
        [
            instructions or "",
            skill_ctx.text,
            *(str(message.get("content", "")) for message in messages),
            *(text for _, text in file_texts),
            body.message,
        ]
    )
    context = build_context(
        messages=messages,
        file_texts=file_texts,
        max_context_tokens=4096,
    )
    follow_up_enabled = (
        body.allow_follow_up_questions
        if allow_follow_up_questions is None and body.allow_follow_up_questions is not None
        else bool(allow_follow_up_questions)
    )
    system_prompt = _chat_system_prompt(
        instructions or "",
        skill_ctx.text,
        allow_follow_up_questions=follow_up_enabled,
    )
    sent_tokens = estimate_tokens(system_prompt + context.text + body.message)

    layer0, layer1 = _build_layers()
    routing_result = layer0.route(body.message, {}) or layer1.route(body.message, {})
    tier = routing_result.decision.tier if routing_result else "cheap_bulk"
    confidence = routing_result.decision.confidence if routing_result else 0.0
    preferences = _resolved_routing_preferences(body)
    resolution = PolicyResolver().resolve(
        tier,
        body.policy,
        confidence,
        message=body.message,
        speed_preference=preferences.speed,
        quality_preference=preferences.quality,
        constraints=constraints,
    )
    assistant_provider = _resolve_assistant_provider(body.assistant_provider)
    selected_model = _assistant_candidates(
        assistant_provider, resolution.tier, resolution, body.assistant_model
    )[0][1]
    free_tier = _free_tier_for(resolution.tier, body.message)
    free_candidate = DEFAULT_REGISTRY.primary_for_tier(free_tier)
    free_model = free_candidate.model if free_candidate else "nvidia/nemotron-3-super-120b-a12b:free"
    if free_only:
        selected_model = free_model
    max_tokens, _ = _tier_limits(resolution.tier)
    expected_output = min(max_tokens or 1200, 4000 if resolution.tier in {"code_strong", "premium"} else 1600)
    estimated_cost = estimate_cost(
        selected_model,
        sent_tokens,
        expected_output,
        PricingTable(),
    ) or 0.0
    routing_fee = platform_fee(sent_tokens, expected_output) if scope_id and not free_only else 0.0
    model_cost = estimated_cost
    estimated_cost += routing_fee
    if free_only:
        estimated_cost = 0.0
    assessment = assess_token_risk(
        source_tokens=estimate_tokens(raw_source),
        sent_tokens=sent_tokens,
        expected_output_tokens=expected_output,
        estimated_cost_usd=estimated_cost,
        selected_model=selected_model,
        free_model=free_model,
        cost_notice_threshold_usd=(
            (load_project_settings(body.project_id, scope_id) or {}).get("max_cost_per_call_usd")
            if body.project_id
            else None
        ),
    )
    assessment_payload = assessment.to_dict()
    assessment_payload.update({
        "model_cost_usd": round(model_cost, 8),
        "routing_fee_usd": round(routing_fee, 8),
        "estimated_total_cost_usd": round(estimated_cost, 8),
    })
    if free_only:
        # Guest mode never opens the paid confirmation path. The provider route
        # is forced to a zero-cost candidate by the chat endpoint as well.
        assessment_payload.update({
            "estimated_cost_usd": 0.0,
            "model_cost_usd": 0.0,
            "routing_fee_usd": 0.0,
            "estimated_total_cost_usd": 0.0,
            "requires_confirmation": False,
            "reason": None,
        })
    return {
        **assessment_payload,
        "history_will_be_summarized": context.history_summarized,
        "messages_summarized": context.messages_summarized,
        "recent_messages_kept": context.recent_messages_kept,
        "free_tier": free_tier,
        "free_available": bool(_resolve_assistant_api_key("theriad")),
        "guest_free_only": free_only,
    }


@app.post("/api/chat/cost-preview")
def chat_cost_preview(body: ChatSendRequest, request: Request):
    try:
        user = get_request_user(request)
        scope_id = user["id"] if user else None
        guest_mode = bool(user and user.get("is_guest"))
        if guest_mode:
            body.cost_decision = "free"
            body.assistant_provider = "theriad"
            body.assistant_model = None
        body.project_id = _resolve_project_scope(
            body.conversation_id, body.project_id, scope_id=scope_id
        )
        assistant_provider = _resolve_assistant_provider(body.assistant_provider)
        execution_provider = "theriad" if body.cost_decision == "free" else assistant_provider
        runtime_settings = _get_runtime_settings(request)
        allow_follow_up_questions = (
            body.allow_follow_up_questions
            if body.allow_follow_up_questions is not None
            else runtime_settings["allow_follow_up_questions"]
        )
        return _chat_cost_preview(
            body,
            constraints=_routing_constraints(request, execution_provider),
            allow_follow_up_questions=allow_follow_up_questions,
            free_only=guest_mode,
            scope_id=scope_id,
        )
    except _ProjectScopeError as exc:
        return _project_scope_error_response(exc)


_THERIAD_ABOUT_ANSWER = (
    "Theriad is a cost-first AI router. It analyzes each request, chooses the "
    "cheapest model that is capable of handling it, and falls back to stronger "
    "models only when the task needs them. You can use Theriad routing or connect "
    "your own OpenAI or Anthropic API key for provider-direct chat. It can also coordinate browser and "
    "local-tool tasks while tracking model choice, cost, latency, and estimated "
    "savings against a fixed premium-model baseline."
)


def _is_about_theriad_browser_prompt(message: str) -> bool:
    normalized = " ".join((message or "").lower().split())
    direct_phrases = (
        "what is theriad",
        "what does theriad do",
        "how does theriad work",
        "tell me about theriad",
        "what is theriad",
        "what does theriad do",
        "how does theriad work",
        "tell me about theriad",
        "what does this app do",
        "what is this app",
        "tell me about this app",
        "what are you",
    )
    return any(phrase in normalized for phrase in direct_phrases)


@app.post("/api/chat/send")
async def chat_send(body: ChatSendRequest, request: Request):
    """
    Unified chat endpoint.

    mode="dry_run"  — route the prompt, save the user message, return route info.
                      No provider call, no assistant message.
    mode="execute"  — requires allow_paid_calls=True.  Routes + calls provider,
                      saves both messages and a run record.

    Always creates a new conversation when conversation_id is omitted.
    Returns HTTP 403 when mode="execute" but allow_paid_calls is False.
    """
    import time

    from theriad.api.projects import (
        _new_id, _now,
        get_file_images, get_file_texts, get_project_instructions, get_recent_messages,
        save_message, save_run,
    )
    from theriad.storage.database import get_db
    from theriad.storage.context import build_context
    from theriad.accounting.pricing import PricingTable, estimate_cost, platform_fee
    from theriad.models.policy import PolicyResolver

    _user = get_request_user(request)
    scope_id = _user["id"] if _user else None
    guest_mode = bool(_user and _user.get("is_guest"))
    if guest_mode:
        # A guest can use the trial without ever entering the paid execution
        # path. Ignore client-supplied paid decisions and provider overrides.
        body.cost_decision = "free"
        body.assistant_provider = "theriad"
        body.assistant_model = None

    try:
        assistant_provider = _resolve_assistant_provider(body.assistant_provider)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})

    try:
        body.project_id = _resolve_project_scope(
            body.conversation_id, body.project_id, scope_id=scope_id
        )
    except _ProjectScopeError as exc:
        return _project_scope_error_response(exc)

    execution_provider = "theriad" if body.cost_decision == "free" else assistant_provider
    origin_constraints = _routing_constraints(request, execution_provider)
    runtime_settings = _get_runtime_settings(request)
    allow_follow_up_questions = (
        body.allow_follow_up_questions
        if body.allow_follow_up_questions is not None
        else runtime_settings["allow_follow_up_questions"]
    )
    cost_preview = _chat_cost_preview(
        body,
        constraints=origin_constraints,
        allow_follow_up_questions=allow_follow_up_questions,
        free_only=guest_mode,
        scope_id=scope_id,
    )
    if body.cost_decision == "free" and not cost_preview["requires_confirmation"] and not guest_mode:
        return JSONResponse(
            status_code=422,
            content={"error": "Free fallback is offered only after a high-context warning."},
        )
    if (
        body.mode == "execute"
        and cost_preview["requires_confirmation"]
        and body.cost_decision not in {"paid", "free"}
    ):
        return JSONResponse(
            status_code=409,
            content={
                "error": "cost_confirmation_required",
                "status": "cost_confirmation_required",
                "cost_preview": cost_preview,
            },
        )
    if body.cost_decision == "free":
        assistant_provider = "theriad"

    hosted_context = None
    hosted_repos = None
    if hosted_data_plane_enabled() and _user:
        from theriad.api.auth_context import get_auth_context

        hosted_context = get_auth_context(request)
        if hosted_context.principal_type in {"registered", "guest"} and hosted_context.organization_id:
            hosted_repos = hosted_repositories()
    db = get_db() if hosted_repos is None else None

    # ── 0b. Per-user scope ────────────────────────────────────────────────────
    _scope = scope_id

    # ── 1. Ensure conversation exists (and belongs to this user) ─────────────
    conversation_id = body.conversation_id
    if conversation_id:
        if hosted_repos and hosted_context:
            owned = hosted_repos["conversations"].get_owned(
                conversation_id,
                organization_id=hosted_context.organization_id,
                user_id=hosted_context.user_id,
            )
        else:
            owned = db.execute(
                "SELECT id FROM conversations WHERE id = ? AND user_id IS ?",
                (conversation_id, _scope),
            ).fetchone()
        if owned is None:
            return JSONResponse(
                status_code=404, content={"error": "Conversation not found"}
            )
    else:
        conv_id = _new_id()
        now_ts = _now()
        title = (body.message[:60] + "…") if len(body.message) > 60 else body.message
        if hosted_repos and hosted_context:
            hosted_repos["conversations"].insert_owned(
                {
                    "id": conv_id,
                    "project_id": body.project_id,
                    "title": title,
                    "created_at": now_ts,
                    "updated_at": now_ts,
                },
                organization_id=hosted_context.organization_id,
                user_id=hosted_context.user_id,
            )
        else:
            db.execute(
                "INSERT INTO conversations (id, project_id, title, created_at, updated_at, user_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (conv_id, body.project_id, title, now_ts, now_ts, _scope),
            )
            db.commit()
        conversation_id = conv_id

    # ── 2. Save user message ──────────────────────────────────────────────────
    user_msg_id = save_message(
        conversation_id=conversation_id,
        role="user",
        content=body.message,
    )

    # ── 3. Build context ──────────────────────────────────────────────────────
    instructions = (
        get_project_instructions(body.project_id, scope_id)
        if body.project_id else None
    )
    prior_messages = get_recent_messages(
        conversation_id, limit=200, scope_id=scope_id
    )
    # Exclude the user message we just saved (it's in body.message already)
    prior_messages = prior_messages[:-1]
    file_texts = get_file_texts(body.file_ids, scope_id)
    attached_images = get_file_images(body.file_ids, scope_id)

    from theriad.skills import build_skill_context
    skill_ctx = build_skill_context(body.project_id)

    ctx_result = build_context(
        messages=prior_messages,
        file_texts=file_texts,
        max_context_tokens=4096,
        include_paid_warning=bool(file_texts),
    )
    # Keep accumulated context separate from the latest user turn. Headroom
    # protects the final message and can safely compress the earlier prefix.
    context_parts = [ctx_result.text] if ctx_result.text else []
    context_prefix = "\n".join(context_parts)
    effective_prompt = (
        f"{context_prefix}\n\n{body.message}" if context_prefix else body.message
    )
    system_prompt = _chat_system_prompt(
        instructions or "",
        skill_ctx.text,
        allow_follow_up_questions=allow_follow_up_questions,
    )

    # ── 3b. Memory injection (optional, scoped to an @mentioned upload) ──────
    # @mentions now select a named memory collection rather than a model.
    memory_injected = 0
    memory_source = None
    memory_store = None
    if body.memory_enabled:
        from theriad.memory.providers import build_memory_context, detect_mentioned_memory
        from theriad.memory.retrieval import retrieve_conversations

        memory_store = _memory_store_for_request(request)
        # Keep model-specific memories out of generic conversation retrieval;
        # the selected concrete model gets its own block below.
        memory_entries = [entry for entry in memory_store.list() if entry.model_id is None]
        memory_source = detect_mentioned_memory(body.message, memory_store)
        memory_block, memory_injected = build_memory_context(
            memory_source,
            store=memory_store,
            query=body.message,
        )
        conversation_block, conversation_sources = retrieve_conversations(
            body.message,
            imported_entries=memory_entries,
            project_id=body.project_id,
        )
        if memory_block:
            context_parts.insert(0, memory_block)
        if conversation_block:
            context_parts.insert(0, conversation_block)
        context_prefix = "\n".join(context_parts)
        effective_prompt = (
            f"{context_prefix}\n\n{body.message}" if context_prefix else body.message
        )
    else:
        conversation_sources = []

    # ── 4a. Capability analysis (free, deterministic, no provider calls) ─────
    from theriad.capabilities.detector import CapabilityDetector
    from theriad.capabilities.rendering import renderer_metadata, renderer_system_instruction
    from theriad.agents.chat_agent import classify_chat_intent
    capability_result = CapabilityDetector().detect(body.message)
    capability_info = capability_result.to_dict()
    chat_rendering = renderer_metadata(capability_result)
    chat_renderer_instruction = renderer_system_instruction(capability_result)
    if chat_renderer_instruction:
        system_prompt = f"{system_prompt}\n\n{chat_renderer_instruction}"
    chat_intent = classify_chat_intent(body.message).intent

    # ── 4. Route the prompt (always free) ─────────────────────────────────────
    routing_start = time.monotonic()
    layer0, layer1 = _build_layers()
    rctx: dict = {}
    # Route the current user request, not the assembled conversation context.
    # Including prior messages in the classifier made the inspector report a
    # stale/incorrect prompt length for short route-only requests.
    routing_result = layer0.route(body.message, rctx) or layer1.route(body.message, rctx)

    if routing_result is None:
        from theriad.core.routing.interfaces import RoutingDecision, RoutingResult
        routing_result = RoutingResult(
            decision=RoutingDecision(
                tier="cheap_bulk",
                confidence=0.0,
                reason="all layers abstained — using fallback tier",
            ),
            layer_name="fallback",
        )

    tier = routing_result.decision.tier

    from theriad.routing.context import ContextRequirementEstimator
    _chat_context_estimate = ContextRequirementEstimator().estimate(body.message)
    _chat_continuity_signal = None
    if conversation_id:
        _chat_continuity_signal = _sticky_cache.get_continuity_signal(
            conversation_id, tier
        )

    resolver = PolicyResolver()
    preferences = _resolved_routing_preferences(body)
    resolution = resolver.resolve(
        tier, body.policy, routing_result.decision.confidence,
        context_estimate=_chat_context_estimate,
        continuity_signal=_chat_continuity_signal,
        message=effective_prompt,
        speed_preference=preferences.speed,
        quality_preference=preferences.quality,
        constraints=origin_constraints,
    )
    routing_latency_ms = (time.monotonic() - routing_start) * 1000

    candidates = _assistant_candidates(
        assistant_provider,
        resolution.tier,
        resolution,
        body.assistant_model,
    )
    if body.cost_decision == "free":
        free_tier = _free_tier_for(resolution.tier, body.message)
        tier = free_tier
        resolution = resolver.resolve(
            free_tier,
            "free_moe_experimental",
            routing_result.decision.confidence,
            message=effective_prompt,
            speed_preference=preferences.speed,
            quality_preference=preferences.quality,
            constraints=origin_constraints,
        )
        candidates = _assistant_candidates(
            assistant_provider,
            resolution.tier,
            resolution,
            body.assistant_model,
        )
    # Resolve the final concrete route. An image attachment can replace the
    # routed chain with vision-capable candidates.
    if attached_images:
        from theriad.models.vision import ensure_vision_candidates

        candidates = ensure_vision_candidates(candidates)
    primary_provider, primary_model = candidates[0]

    # Model-scoped context is resolved after routing so a routed request gets
    # the memory belonging to the concrete model that will answer it. Shared
    # memories and explicit @upload memories remain separate from this block.
    model_memory_id = None
    model_memory_injected = 0
    model_memory_profile = None
    if body.memory_enabled:
        from theriad.memory.learning import model_memory_id as build_model_memory_id
        from theriad.memory.providers import build_model_memory_context

        model_memory_id = build_model_memory_id(primary_provider, primary_model)
        model_memory_profile = memory_store.get_model_profile(model_memory_id, create=False)
        model_memory_block, model_memory_injected = build_model_memory_context(
            model_memory_id,
            store=memory_store,
            query=body.message,
        )
        if model_memory_block:
            context_parts.insert(0, model_memory_block)
            context_prefix = "\n".join(context_parts)
            effective_prompt = (
                f"{context_prefix}\n\n{body.message}" if context_prefix else body.message
            )

    pt = PricingTable()
    prompt_tokens = _estimate_prompt_tokens(body.message)
    est_output = 200
    model_cost = estimate_cost(primary_model, prompt_tokens, est_output, pt) or 0.0
    routing_fee = platform_fee(prompt_tokens, est_output) if scope_id and not guest_mode and body.cost_decision != "free" else 0.0
    theriad_cost = model_cost + routing_fee
    baseline_cost = estimate_cost(body.baseline, prompt_tokens, est_output, pt) or 0.0

    route_info = {
        "tier": resolution.tier,
        "model": primary_model,
        "provider": primary_provider,
        "assistant_provider": assistant_provider,
        "speed_preference": preferences.speed,
        "quality_preference": preferences.quality,
        "estimated_price_multiplier": preferences.price_multiplier,
        "routing_scope": "cross_provider" if assistant_provider == "theriad" else "provider_direct",
        "model_origin_policy": "us_only" if origin_constraints.active else "any",
        "guest_free_only": guest_mode,
        "follow_up_questions_enabled": allow_follow_up_questions,
        "memory_model_id": model_memory_id,
        "confidence": round(routing_result.decision.confidence, 3),
        "reason": routing_result.decision.reason,
        "estimated_cost_usd": round(theriad_cost, 8),
        "model_cost_usd": round(model_cost, 8),
        "platform_fee_usd": round(routing_fee, 8),
        "baseline_cost_usd": round(baseline_cost, 8),
        "savings_pct": round(
            (baseline_cost - theriad_cost) / baseline_cost * 100, 1
        ) if baseline_cost > 0 else 0.0,
        "routing_latency_ms": round(routing_latency_ms, 1),
        "fallback_chain": candidates[1:],
        # Context-window + continuity diagnostics
        "selected_context_bucket": resolution.selected_context_bucket,
        "context_fit_reason": resolution.context_fit_reason,
        "continuity_bonus_applied": resolution.continuity_bonus_applied,
        "reason_for_stay_or_switch": resolution.reason_for_stay_or_switch,
        "router_version": resolution.router_version,
        "router_lane": resolution.lane,
        "router_knee_threshold": resolution.knee_threshold,
        "router_ood_score": resolution.ood_score,
        "router_shadow": resolution.shadow,
        "router_shadow_challenger": resolution.shadow_challenger,
    }

    context_info = {
        "tokens": ctx_result.total_tokens,
        "truncated": ctx_result.truncated,
        "truncation_warning": ctx_result.truncation_warning,
        "paid_warning": ctx_result.paid_provider_warning,
        "memory_injected": memory_injected,
        "memory_source": memory_source,
        "model_memory_injected": model_memory_injected,
        "model_memory_profile": model_memory_profile,
        **skill_ctx.metadata(),
        "history_summarized": ctx_result.history_summarized,
        "messages_summarized": ctx_result.messages_summarized,
        "recent_messages_kept": ctx_result.recent_messages_kept,
        "conversation_sources": [source.to_dict() for source in conversation_sources],
    }

    # Account-scoped safety controls apply before any provider or agent call.
    # Explicit model selection and an explicit free fallback are intentional
    # overrides of the premium-default guard.
    if body.mode == "execute" and body.allow_paid_calls:
        if (
            runtime_settings["disable_premium_models"]
            and resolution.tier == "premium"
            and not body.assistant_model
            and body.cost_decision != "free"
        ):
            return JSONResponse(
                status_code=409,
                content={
                    "error": "premium_models_disabled",
                    "message": "This request selected the premium tier, which is disabled by your runtime settings.",
                    "route": route_info,
                    "runtime_settings": runtime_settings,
                },
            )

        estimated_for_guard = 0.0 if body.cost_decision == "free" else cost_preview.get("estimated_cost_usd")
        budget_error = _execution_budget_error(request, estimated_for_guard)
        if budget_error:
            return JSONResponse(
                status_code=409,
                content={
                    "error": budget_error["code"],
                    "message": budget_error["message"],
                    "cost_preview": cost_preview,
                    "route": route_info,
                    "runtime_settings": runtime_settings,
                    "budget": budget_error,
                },
            )

    if not runtime_settings["regular_chat_returns_answer"]:
        return {
            "conversation_id": conversation_id,
            "user_message": {"id": user_msg_id, "role": "user", "content": body.message},
            "assistant_message": None,
            "route": route_info,
            "run": None,
            "context": context_info,
            "capability": capability_info,
            "chat_intent": chat_intent,
            "runtime_settings": runtime_settings,
            "quality_loop": {"loop_enabled": False, "reason": "route_only_setting"},
            "status": "route_only",
        }

    # Product identity is deterministic so prior conversation context or a
    # routed provider cannot redefine what Theriad is.
    if _is_about_theriad_browser_prompt(body.message):
        asst_msg_id = save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=_THERIAD_ABOUT_ANSWER,
        )
        about_route = {
            **route_info,
            "tier": "local_product_info",
            "model": "theriad/product-info",
            "provider": "local",
            "estimated_cost_usd": 0.0,
            "savings_pct": 100.0,
            "reason": "Deterministic Theriad product description.",
        }
        return {
            "conversation_id": conversation_id,
            "user_message": {"id": user_msg_id, "role": "user", "content": body.message},
            "assistant_message": {
                "id": asst_msg_id,
                "role": "assistant",
                "content": _THERIAD_ABOUT_ANSWER,
            },
            "route": about_route,
            "run": None,
            "context": context_info,
            "capability": capability_info,
            "chat_intent": "normal_chat",
            "quality_loop": {"loop_enabled": False, "reason": "deterministic_product_info"},
            "status": "success",
        }

    # Agentic requests must dispatch to the appropriate runtime before any
    # ordinary model call. The chat UI uses this response to start the agent,
    # avoiding a paid text-only answer that would immediately be discarded.
    if chat_intent != "normal_chat" and body.mode == "execute":
        return {
            "conversation_id": conversation_id,
            "user_message": {"id": user_msg_id, "role": "user", "content": body.message},
            "assistant_message": None,
            "route": route_info,
            "run": None,
            "context": context_info,
            "capability": capability_info,
            "chat_intent": chat_intent,
            "runtime_settings": runtime_settings,
            "quality_loop": {"loop_enabled": False, "reason": "agent_runtime_required"},
            "status": "agent_required",
        }

    # ── 5. Safety gate for execute mode ───────────────────────────────────────
    if body.mode == "execute" and not body.allow_paid_calls:
        return JSONResponse(
            status_code=403,
            content={
                "error": (
                    "allow_paid_calls must be true to execute. "
                    "This will make a real provider call that incurs cost."
                ),
                "conversation_id": conversation_id,
                "user_message": {"id": user_msg_id, "role": "user", "content": body.message},
                "route": route_info,
            },
        )

    # ── 6. Dry-run: return route info only ────────────────────────────────────
    if body.mode != "execute":
        # Indicate whether the quality loop would run (advisory, no actual call)
        _ql_preview: dict = {"loop_enabled": False, "reason": "dry_run — no draft output yet"}
        try:
            from theriad.core.config.schema import QualityLoopConfig, QualityLoopPassLimits
            _ql_cfg = QualityLoopConfig()
            _ql_max = getattr(_ql_cfg.max_passes, body.policy.replace("-", "_"), 0)
            if body.quality_loop_enabled is False:
                _ql_preview = {"loop_enabled": False, "reason": "explicitly disabled"}
            elif body.quality_loop_enabled is True or _ql_max > 0:
                _ql_preview = {
                    "loop_enabled": True,
                    "max_passes": _ql_max if body.quality_loop_enabled is None else max(1, _ql_max),
                    "reason": "policy enables loop" if body.quality_loop_enabled is None else "explicitly enabled",
                }
        except Exception:
            pass
        return {
            "conversation_id": conversation_id,
            "user_message": {"id": user_msg_id, "role": "user", "content": body.message},
            "assistant_message": None,
            "route": route_info,
            "run": None,
            "context": context_info,
            "capability": capability_info,
            "chat_intent": chat_intent,
            "quality_loop": _ql_preview,
            "status": "dry_run",
        }

    # ── 7. Execute (mode="execute" + allow_paid_calls=True) ───────────────────
    api_key = _resolve_assistant_api_key(assistant_provider)
    if _scope:
        from theriad.api.auth_context import get_auth_context
        from theriad.billing.service import entitlements_for, request_limit_error

        context = get_auth_context(request)
        entitlements = entitlements_for(
            organization_id=context.organization_id,
            user_id=context.user_id,
            guest=context.principal_type == "guest",
        )
        denial = request_limit_error(
            entitlements,
            is_free_request=body.cost_decision == "free",
            provider_key_configured=bool(api_key),
        )
        if denial:
            status_code = 503 if denial["code"] == "provider_key_required" else 403
            return JSONResponse(status_code=status_code, content={"error": denial["code"], "message": denial["message"], "entitlements": entitlements})
    if not api_key:
        label = "OpenRouter" if assistant_provider == "theriad" else assistant_provider.title()
        return JSONResponse(
            status_code=503,
            content={"error": f"No {label} API key found. Connect it in Settings."},
        )

    from theriad.core.providers.factory import build_provider
    from theriad.core.providers.interfaces import CompletionRequest, Message, ProviderError

    max_tokens, timeout_secs = _tier_limits(tier)

    from theriad.agents.model_telemetry import complete_with_telemetry, usage_totals

    attempt_usage: list[dict] = []
    response = None
    used_fallback = False
    last_error: Exception | None = None
    provider_latency_ms = 0.0
    total_start = time.monotonic()
    objective_validation = None
    from theriad.runtime.circuit_breaker import get_circuit_registry, model_circuit_key

    for i, (provider_name, model_slug) in enumerate(candidates):
        attempt_start = time.monotonic()
        provider = build_provider(
            provider_name,
            api_key=api_key,
            timeout=timeout_secs,
        )
        provider_user_content = _user_content(body.message, attached_images)
        if attached_images and context_prefix:
            # Keep multimodal context in the same user message. Some clients
            # (and providers) treat a preceding text-only user turn as the
            # primary prompt and can drop the image block.
            provider_user_content = [
                {"type": "text", "text": context_prefix},
                *provider_user_content,
            ]
        req = CompletionRequest(
            model=model_slug,
            messages=(
                [Message(role="system", content=system_prompt)]
                + (
                    [Message(role="user", content=context_prefix)]
                    if context_prefix and not attached_images else []
                )
                + [Message(role="user", content=provider_user_content)]
            ),
            max_tokens=max_tokens,
            extra={
                **_provider_request_extra(
                    provider.name,
                    model_slug,
                    preferences,
                    [
                        model
                        for candidate_provider, model in candidates
                        if candidate_provider == "openrouter"
                    ],
                ),
                **__import__(
                    "theriad.storage.prompt_cache", fromlist=["prompt_cache_extra"]
                ).prompt_cache_extra(provider.name),
            },
            session_id=conversation_id,
        )
        try:
            candidate_response = await complete_with_telemetry(
                provider, req, role="chat_completion", usage_sink=attempt_usage,
                pricing_table=pt,
            )
            if not candidate_response.content.strip() and candidate_response.completion_tokens > 0:
                last_error = Exception("empty output")
                get_circuit_registry().record_failure(
                    model_circuit_key(provider.name, model_slug), last_error
                )
                continue
            get_circuit_registry().record_success(
                model_circuit_key(provider.name, model_slug),
                latency_ms=(time.monotonic() - attempt_start) * 1000,
            )
            from theriad.quality.objective_verifier import verify_objective_response

            objective_validation = verify_objective_response(
                body.message, candidate_response.content
            )
            if (
                objective_validation is not None
                and not objective_validation.passed
                and i < len(candidates) - 1
            ):
                last_error = Exception(objective_validation.details)
                continue
            response = candidate_response
            used_fallback = i > 0
            provider_latency_ms = (time.monotonic() - attempt_start) * 1000
            break
        except ProviderError as exc:
            last_error = exc
            transient = exc.status_code is None or exc.status_code in {408, 409, 425, 429} or (
                isinstance(exc.status_code, int) and exc.status_code >= 500
            )
            get_circuit_registry().record_failure(
                model_circuit_key(provider.name, model_slug), exc, transient=transient
            )
            if exc.status_code and 400 <= exc.status_code < 500 and exc.status_code != 429:
                break
        except Exception as exc:
            last_error = exc
            get_circuit_registry().record_failure(
                model_circuit_key(provider.name, model_slug), exc
            )

    total_latency_ms = (time.monotonic() - total_start) * 1000

    actual_cost, cost_known = usage_totals(attempt_usage)
    attempt_input_tokens = sum(entry.get("prompt_tokens") or 0 for entry in attempt_usage)
    attempt_output_tokens = sum(entry.get("completion_tokens") or 0 for entry in attempt_usage)
    attempt_accounting = {
        "actual_cost_usd": actual_cost,
        "cost_known": cost_known,
        "known_attempt_cost_usd": round(sum(entry.get("cost_usd") or 0.0 for entry in attempt_usage), 8),
        "attempt_input_tokens": attempt_input_tokens,
        "attempt_output_tokens": attempt_output_tokens,
        "attempt_usage": attempt_usage,
    }

    if response is None:
        return JSONResponse(
            status_code=502,
            content={
                "error": f"The selected provider failed. Last error: {last_error}",
                "accounting": attempt_accounting,
            },
        )

    # Prefer provider-reported billing, then fall back to the current table.
    from theriad.accounting.actuals import cached_prompt_tokens, realized_cost

    final_response_cost = realized_cost(response, pt)
    prompt_cache_tokens = cached_prompt_tokens(response)

    # ── 8. Quality loop (opt-in; normal execute unchanged when disabled) ─────
    final_content = response.content
    quality_loop_summary: dict = {"loop_enabled": False, "passes_run": 0,
                                  "stop_reason": "loop_not_enabled"}
    try:
        from theriad.quality.loop_engine import run_quality_loop
        from theriad.quality.loop_schemas import QualityLoopRequest, ValidationSignal
        initial_validation = (
            ValidationSignal(
                source=objective_validation.source,
                passed=objective_validation.passed,
                score=objective_validation.score,
                details=objective_validation.details,
                is_objective=True,
            )
            if objective_validation is not None
            else None
        )
        _ql_req = QualityLoopRequest(
            prompt=effective_prompt,
            draft_output=response.content,
            policy=body.policy,
            quality_loop_enabled=body.quality_loop_enabled,
            draft_cost_usd=actual_cost if actual_cost is not None else 0.0,
            initial_validation=initial_validation,
        )
        _ql_result = run_quality_loop(_ql_req)
        if _ql_result.loop_enabled and _ql_result.passes_run > 0:
            final_content = _ql_result.final_output
        quality_loop_summary = _ql_result.summary()
    except Exception as _ql_exc:
        quality_loop_summary = {"loop_enabled": False, "error": str(_ql_exc)}

    # ── 9. Save assistant message + run ───────────────────────────────────────
    asst_msg_id = save_message(
        conversation_id=conversation_id,
        role="assistant",
        content=final_content,
    )
    _routing_observations = None
    if resolution.router_v2_decision is not None:
        from theriad.routing.v2 import observation_rows

        _routing_observations = observation_rows(
            resolution.router_v2_decision,
            prompt=body.message,
            actual_model=response.model,
            actual_cost_usd=final_response_cost,
            actual_latency_ms=provider_latency_ms,
            input_tokens=response.prompt_tokens,
            output_tokens=response.completion_tokens,
        )
    run_id = save_run(
        project_id=body.project_id,
        conversation_id=conversation_id,
        message_id=asst_msg_id,
        model=response.model,
        tier=tier,
        provider=response.provider,
        policy=body.policy,
        input_tokens=attempt_input_tokens,
        output_tokens=attempt_output_tokens,
        actual_cost_usd=actual_cost,
        routing_latency_ms=routing_latency_ms,
        provider_latency_ms=provider_latency_ms,
        total_latency_ms=total_latency_ms,
        fallback_used=used_fallback,
        status="success",
        user_id=_scope,
        routing_observations=_routing_observations,
    )

    _record_routing_usage(
        request,
        run_id=run_id,
        provider=response.provider,
        model=response.model,
        input_tokens=attempt_input_tokens,
        output_tokens=attempt_output_tokens,
        provider_cost_usd=actual_cost,
        paid_request=body.cost_decision != "free",
        usage_known=all(entry.get("usage_available", False) for entry in attempt_usage),
    )

    # Learning is deliberately local and conservative: only explicit
    # first-person facts in the user's message are persisted, and only when
    # the model's Memory-tab policy has opted in. No second provider call is
    # made and the assistant's answer is never treated as a user fact.
    model_memory_learned = 0
    learned_model_id = model_memory_id
    if body.memory_enabled and memory_store is not None:
        from theriad.memory.learning import learn_from_interaction, model_memory_id as build_model_memory_id

        learned_model_id = build_model_memory_id(response.provider, response.model)
        try:
            learned = learn_from_interaction(
                memory_store,
                model_id=learned_model_id,
                user_message=body.message,
            )
            model_memory_learned = len(learned)
            if model_memory_learned:
                model_memory_profile = memory_store.get_model_profile(learned_model_id, create=False)
        except (TypeError, ValueError) as exc:
            logger.warning("model memory learning skipped", extra={"error": str(exc)})
    context_info["model_memory_learned"] = model_memory_learned
    context_info["learned_model_id"] = learned_model_id
    context_info["model_memory_profile"] = model_memory_profile

    if _scope is None:
        # Local/dev keeps the legacy file for CLI compatibility. Account
        # requests are already durable and isolated in the runs table.
        try:
            cfg2 = _load_config()
            metrics_path = cfg2.metrics_path
        except Exception:
            metrics_path = "~/.theriad/metrics.jsonl"

        from theriad.accounting.tracker import CostTracker
        tracker = CostTracker(metrics_path=metrics_path)
        tracker.record(
            provider=response.provider,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
            routing_layer=routing_result.layer_name,
            routing_reason=body.message[:200],
            used_fallback=used_fallback,
            tags={
                "policy": body.policy,
                "speed_preference": preferences.speed,
                "quality_preference": preferences.quality,
                "tier": tier,
                "source": "api_chat_send",
            },
        )

    return {
        "conversation_id": conversation_id,
        "user_message": {"id": user_msg_id, "role": "user", "content": body.message},
        "assistant_message": {"id": asst_msg_id, "role": "assistant", "content": final_content},
        "route": route_info,
        "run": {
            "id": run_id,
            "model": response.model,
            "provider": response.provider,
            "assistant_provider": assistant_provider,
            "tier": tier,
            "input_tokens": response.prompt_tokens,
            "output_tokens": response.completion_tokens,
            **attempt_accounting,
            "prompt_cache_used": prompt_cache_tokens > 0,
            "total_latency_ms": round(total_latency_ms, 1),
            "fallback_used": used_fallback,
        },
        "context": context_info,
        "capability": capability_info,
        "rendering": chat_rendering,
        "chat_intent": chat_intent,
        "quality_loop": quality_loop_summary,
        "objective_validation": (
            objective_validation.to_dict() if objective_validation else None
        ),
        "status": "success",
        "cost_preview": cost_preview,
    }


# ── GET /api/usage/summary ────────────────────────────────────────────────────

@app.get("/api/usage/summary")
def usage_summary(request: Request):
    """
    Per-user spend summary across all recorded runs: total cost, total tokens,
    and the blended (projected) cost per million tokens for this user's actual
    mix of models. The projection answers "at my current model melange, what
    does a million tokens cost me?" — more descriptive than a rounded
    per-request figure.
    """
    from theriad.api.auth_context import get_auth_context
    from theriad.billing.service import usage_summary_for
    from theriad.storage.database import get_db

    context = get_auth_context(request)
    account_summary = usage_summary_for(
        organization_id=context.organization_id,
        user_id=context.user_id,
    )
    from theriad.storage.hosted import hosted_data_plane_enabled

    if hosted_data_plane_enabled() and context.principal_type in {"registered", "guest"}:
        total_tokens = int(account_summary.get("total_tokens") or 0)
        total_cost = float(account_summary.get("provider_cost_usd") or 0.0)
        blended = total_cost / total_tokens * 1_000_000 if total_tokens else None
        return {
            "has_data": bool(account_summary.get("has_data")),
            "total_runs": int(account_summary.get("total_runs") or 0),
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "blended_cost_per_mtok": round(blended, 4) if blended is not None else None,
            "baseline_model": DEFAULT_BASELINE_MODEL,
            "baseline_cost_per_mtok": None,
            "provider_cost_usd": account_summary["provider_cost_usd"],
            "routing_fee_usd": account_summary["routing_fee_usd"],
            "total_billed_usd": account_summary["total_billed_usd"],
            "pending_reconciliation": account_summary["pending_reconciliation"],
            "routing_fee": account_summary["routing_fee"],
            "billing_period": account_summary["billing_period"],
        }
    db = get_db()
    scope = user_scope_id(request)
    row = db.execute(
        """SELECT COUNT(*) AS runs,
                  COALESCE(SUM(actual_cost_usd), 0.0) AS cost,
                  COALESCE(SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)), 0) AS tokens
           FROM runs
           WHERE user_id IS ? AND status = 'success'""",
        (scope,),
    ).fetchone()
    total_runs = int(row["runs"])
    total_cost = float(row["cost"])
    total_tokens = int(row["tokens"])
    blended = (total_cost / total_tokens * 1_000_000) if total_tokens > 0 else None

    # Same tokens priced at the premium baseline, for an at-a-glance contrast.
    baseline_model = DEFAULT_BASELINE_MODEL
    baseline_per_mtok = None
    if total_tokens > 0:
        from theriad.accounting.pricing import PricingTable, estimate_cost
        pt = PricingTable()
        rows = db.execute(
            """SELECT COALESCE(input_tokens, 0) AS tin, COALESCE(output_tokens, 0) AS tout
               FROM runs WHERE user_id IS ? AND status = 'success'""",
            (scope,),
        ).fetchall()
        baseline_cost = sum(
            estimate_cost(baseline_model, r["tin"], r["tout"], pt) or 0.0 for r in rows
        )
        baseline_per_mtok = baseline_cost / total_tokens * 1_000_000

    return {
        "has_data": total_tokens > 0,
        "total_runs": total_runs,
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
        "blended_cost_per_mtok": round(blended, 4) if blended is not None else None,
        "baseline_model": baseline_model,
        "baseline_cost_per_mtok": (
            round(baseline_per_mtok, 4) if baseline_per_mtok is not None else None
        ),
        "provider_cost_usd": account_summary["provider_cost_usd"] if context.user_id else total_cost,
        "routing_fee_usd": account_summary["routing_fee_usd"],
        "total_billed_usd": account_summary["total_billed_usd"],
        "pending_reconciliation": account_summary["pending_reconciliation"],
        "routing_fee": account_summary["routing_fee"],
        "billing_period": account_summary["billing_period"],
    }


# ── GET /api/logs ─────────────────────────────────────────────────────────────

def _hosted_inference_records(request: Request, limit: int) -> list[dict]:
    """Adapt durable hosted runs to the legacy inference-log response shape."""
    from theriad.api.auth_context import get_auth_context
    from theriad.storage.hosted import hosted_repositories

    context = get_auth_context(request)
    repos = hosted_repositories()
    rows = repos["runs"].list_for_owner(
        organization_id=context.organization_id,
        user_id=context.user_id,
        order="created_at.desc",
        limit=limit,
    )

    # Hosted runs keep the assistant message id, not a duplicate raw prompt.
    # Reconstruct a short prompt preview from the preceding user message in
    # one owner-scoped query so the Logs screen remains useful without adding
    # prompt text to the runs schema.
    conversation_ids = [
        str(row.get("conversation_id"))
        for row in rows
        if row.get("conversation_id")
    ]
    prompt_by_assistant_id: dict[str, str] = {}
    if conversation_ids:
        ids = ",".join(conversation_ids)
        message_rows = repos["messages"].client.list_rows(
            "messages",
            filters={
                "conversation_id": f"in.({ids})",
                "organization_id": f"eq.{context.organization_id}",
                "user_id": f"eq.{context.user_id}",
            },
            select="id,conversation_id,role,content,created_at",
            order="created_at.asc",
            limit=1000,
        )
        last_user_by_conversation: dict[str, str] = {}
        for raw_message in message_rows:
            message = dict(raw_message)
            conversation_id = str(message.get("conversation_id") or "")
            if message.get("role") == "user":
                last_user_by_conversation[conversation_id] = str(message.get("content") or "")
            elif message.get("role") == "assistant" and message.get("id"):
                prompt_by_assistant_id[str(message["id"])] = last_user_by_conversation.get(
                    conversation_id, ""
                )

    records = []
    for row in rows:
        input_tokens = int(row.get("input_tokens") or 0)
        output_tokens = int(row.get("output_tokens") or 0)
        records.append({
            "id": row.get("id"),
            "timestamp": row.get("created_at") or "",
            "model": row.get("model") or "unknown",
            "provider": row.get("provider") or "unknown",
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost_usd": (
                float(row["actual_cost_usd"])
                if row.get("actual_cost_usd") is not None else None
            ),
            "latency_ms": float(row.get("total_latency_ms") or 0.0),
            "routing_layer": row.get("tier") or "unknown",
            "routing_reason": (
                prompt_by_assistant_id.get(str(row.get("message_id") or ""))
                or "(no prompt recorded)"
            ),
            "used_fallback": bool(row.get("fallback_used")),
            "status": row.get("status") or "unknown",
            "error": row.get("error"),
        })
    return records


@app.get("/api/logs")
def get_logs(request: Request, limit: int = 100):
    """Return recent inference records for the current account."""
    from theriad.accounting.scoped_metrics import load_scoped_records
    from theriad.settings.deployment import is_hosted_mode
    from theriad.storage.hosted import hosted_data_plane_enabled

    scope = user_scope_id(request)
    records: list[dict]
    context = None
    if hosted_data_plane_enabled() and is_hosted_mode():
        from theriad.api.auth_context import get_auth_context

        context = get_auth_context(request)
    if context and context.principal_type in {"registered", "guest"}:
        records = _hosted_inference_records(request, max(1, min(limit, 500)))
    elif scope is None and not is_hosted_mode():
        from theriad.accounting.tracker import CostTracker

        try:
            cfg = _load_config()
            metrics_path = cfg.metrics_path
        except Exception:
            metrics_path = "~/.theriad/metrics.jsonl"
        records = list(reversed(CostTracker.load_from_disk(metrics_path)))[:max(1, min(limit, 500))]
        records = [{**record, "_display_id": str(index + 1)} for index, record in enumerate(records)]
    else:
        records = list(reversed(load_scoped_records(scope)))[:max(1, min(limit, 500))]

    result = []
    for record in records:
        ts_raw = record.get("timestamp", "") or ""
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            time_str = dt.isoformat()
        except Exception:
            time_str = ts_raw[:8] if ts_raw else "?"

        cost_val = _record_cost(record)
        latency_val = float(record.get("latency_ms", 0) or 0)
        result.append({
            "id": str(record.get("_display_id") or record.get("id") or "?"),
            "time": time_str,
            "timestamp": ts_raw,
            "prompt": str(record.get("routing_reason") or "(no prompt recorded)")[:120],
            "tier": record.get("routing_layer", "unknown"),
            "model": record.get("model", "unknown"),
            "cost": f"${cost_val:.5f}" if cost_val is not None else "Unavailable",
            "latency": f"{latency_val:.0f}ms",
            "conf": "—",
            "status": (
                record.get("status")
                if record.get("status") and record.get("status") != "success"
                else "Fallback" if record.get("used_fallback") else "Routed"
            ),
            "tokens": record.get("total_tokens", 0),
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Browser-agent endpoints
# Completely separate from /api/execute — normal execution is unchanged.
# ══════════════════════════════════════════════════════════════════════════════

class LocalBrowserStartRequest(BaseModel):
    goal: str = ""
    visible: bool = False
    start_url: str | None = None
    user_prompt: str = ""
    viewport_width: int = 1280
    viewport_height: int = 900
    allowed_domains: list[str] = Field(default_factory=list)


class LocalBrowserSessionRequest(BaseModel):
    session_id: str


class LocalBrowserNavigateRequest(LocalBrowserSessionRequest):
    url: str


class LocalBrowserActionRequest(LocalBrowserSessionRequest):
    action: str | None = None
    element_id: str | None = None
    text: str | None = None
    key: str | None = None
    amount: int | None = None
    milliseconds: int | None = None
    url: str | None = None
    approved: bool = False
    deny: bool = False


def _browser_request_context(request: Request) -> tuple[str | None, str | None]:
    owner_override = getattr(request.state, "theriad_browser_owner", None)
    if owner_override is not None:
        return owner_override
    from theriad.api.auth_context import get_auth_context

    return user_scope_id(request), get_auth_context(request).organization_id


def _browser_task_for_session(session_id: str, request: Request):
    """Resolve a browser session through its owner-scoped canonical task."""
    owner_user_id, owner_organization_id = _browser_request_context(request)
    task = get_task_engine().task_for_browser_session(session_id)
    if task is None:
        return None
    if task.owner_user_id != owner_user_id or task.owner_organization_id != owner_organization_id:
        if (
            task.owner_user_id is not None
            or owner_user_id is not None
            or task.owner_organization_id is not None
            or owner_organization_id is not None
        ):
            return None
    return task


def _browser_pending_payload(task, current_url: str | None = None) -> dict[str, Any] | None:
    pending = task.pending_approval
    if pending is None:
        return None
    from theriad.agents.task_engine import _safe_arguments

    action = get_task_engine()._browser_action(pending.call)
    safe_arguments = _safe_arguments(pending.call.arguments)
    target = (
        pending.call.arguments.get("element_id")
        or pending.call.arguments.get("selector")
        or pending.call.arguments.get("url")
    )
    return {
        "id": pending.approval_id,
        "approval_id": pending.approval_id,
        "action_type": action,
        "proposed_action": action,
        "target": str(target)[:500] if target else None,
        "current_url": current_url,
        "reason": pending.reason,
        "risk_level": "critical" if action in {"purchase", "delete", "send_message"} else "high",
        "data_to_submit": None,
        "arguments": safe_arguments,
        "created_at": pending.created_at,
        "resubmission_required": pending.resubmission_required,
    }


async def _browser_task_payload(task, request: Request) -> dict[str, Any]:
    from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime

    owner_user_id, owner_organization_id = _browser_request_context(request)
    session_id = str(task.metadata.get("browser_session_id") or "") or None
    browser: dict[str, Any] = {}
    if session_id:
        browser = await get_opensandbox_browser_runtime().status(
            session_id,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
    safe_task = task.safe_dict()
    payload: dict[str, Any] = {
        **browser,
        "ok": bool(browser.get("ok", task.status not in {"failed", "stopped"})),
        "session_id": session_id,
        "task_id": task.task_id,
        "goal": safe_task["goal"],
        "status": browser.get("status") or task.status,
        "task_status": task.status,
        "backend": "opensandbox",
        "browser_backend": "opensandbox",
        "observation_backend": "opensandbox",
        "action_backend": "opensandbox",
        "actions_mocked": False,
        "original_user_prompt": safe_task["goal"],
        "allowed_domains": task.allowed_domains,
        "task_metadata": safe_task["metadata"],
        "task_event_count": len(task.events),
        "steps_taken": sum(
            1
            for event in task.events
            if event.kind == "tool.completed"
            and str(event.payload.get("tool", "")).startswith("browser.opensandbox.")
        ),
    }
    if task.status in {"completed", "failed", "stopped"}:
        payload["status"] = task.status
    if task.final_answer:
        payload["final_answer"] = task.final_answer
        payload["final_result"] = task.final_answer
    if task.status == "failed" and not payload.get("error"):
        payload["error"] = task.metadata.get("error") or "Browser task failed."
    pending = _browser_pending_payload(task, payload.get("current_url"))
    if pending is not None:
        payload["status"] = "waiting_for_approval"
        payload["pending_approval"] = pending
        payload["pending_action"] = pending
    return payload


async def _runtime_browser_action(
    session_id: str,
    request: Request,
    *,
    approved: bool = False,
    deny: bool = False,
    stop: bool = False,
) -> dict[str, Any]:
    from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime

    owner_user_id, owner_organization_id = _browser_request_context(request)
    runtime = get_opensandbox_browser_runtime()
    if stop:
        return await runtime.stop_session(
            session_id,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
    return await runtime.action(
        session_id,
        approved=approved,
        deny=deny,
        owner_user_id=owner_user_id,
        owner_organization_id=owner_organization_id,
    )


def _browser_owner_request(
    owner_user_id: str | None,
    owner_organization_id: str | None,
) -> Request:
    """Build an owner-scoped request for shared status serialization."""
    synthetic = Request({
        "type": "http",
        "method": "GET",
        "path": "/api/browser/status",
        "headers": [],
    })
    synthetic.state.theriad_browser_owner = (owner_user_id, owner_organization_id)
    return synthetic


def _browser_replacement_call(body: dict[str, Any]) -> ToolCall | None:
    """Build a fresh browser action for a post-restart approval.

    Pending browser snapshots intentionally redact typed values.  A caller may
    therefore resubmit the sensitive action in the approval request, but the
    replacement is never persisted by the server or task engine.
    """
    raw = body.get("replacement_action") or body.get("arguments")
    if raw is None and isinstance(body.get("action"), dict):
        raw = body.get("action")
    if not isinstance(raw, dict):
        return None
    action = {
        str(key): value
        for key, value in raw.items()
        if key not in {"session_id", "approved", "deny"}
    }
    return ToolCall("browser.opensandbox.action", action, approved=True)


async def _resume_browser_task(task_id: str, *, max_calls: int | None = None) -> None:
    from theriad.agents.task_runner import run_task

    await run_task(task_id, engine=get_task_engine(), max_calls=max_calls)


@app.post("/api/browser/session/start")
async def local_browser_session_start(body: LocalBrowserStartRequest, request: Request):
    """Create a task-owned OpenSandbox browser session without planning."""
    from theriad.agents.browser_schemas import BrowserAgentStartRequest
    from theriad.api.auth_context import get_auth_context

    context = get_auth_context(request)
    req = BrowserAgentStartRequest(
        goal=body.user_prompt or body.goal or "Open a browser session.",
        start_url=body.start_url,
        real_observation_enabled=True,
        real_action_enabled=True,
    )
    payload, error = await _start_browser_session(
        req,
        run_now=False,
        owner_user_id=user_scope_id(request),
        owner_organization_id=context.organization_id,
        allowed_domains=body.allowed_domains,
        source="browser-session-api",
        request=request,
    )
    return error if payload is None else payload


@app.get("/api/browser/session/{session_id}")
async def browser_session_status(session_id: str, request: Request, screenshot: bool = True):
    """Return the task-scoped OpenSandbox browser observation."""
    task = _browser_task_for_session(session_id, request)
    if task is None:
        return JSONResponse(status_code=404, content={"error": "Browser session not found."})
    if screenshot:
        await get_task_engine().execute_async(
            task.task_id,
            ToolCall("browser.opensandbox.observe", {"screenshot": True}),
        )
    return await _browser_task_payload(task, request)


@app.post("/api/browser/session/{session_id}/approve")
async def browser_session_approve(session_id: str, request: Request):
    task = _browser_task_for_session(session_id, request)
    if task is None:
        return JSONResponse(status_code=404, content={"error": "Browser session not found."})
    engine = get_task_engine()
    if task.pending_approval is not None:
        await engine.approve_async(
            task.task_id,
            task.pending_approval.approval_id,
            approved=True,
        )
        await _resume_browser_task(task.task_id)
    else:
        browser = await _runtime_browser_action(session_id, request, approved=True)
        if not browser.get("ok"):
            return JSONResponse(status_code=409, content=browser)
    return await _browser_task_payload(task, request)


@app.post("/api/browser/session/{session_id}/deny")
async def browser_session_deny(session_id: str, request: Request):
    task = _browser_task_for_session(session_id, request)
    if task is None:
        return JSONResponse(status_code=404, content={"error": "Browser session not found."})
    engine = get_task_engine()
    if task.pending_approval is not None:
        await engine.approve_async(
            task.task_id,
            task.pending_approval.approval_id,
            approved=False,
        )
        await _resume_browser_task(task.task_id)
    else:
        browser = await _runtime_browser_action(session_id, request, deny=True)
        if not browser.get("ok"):
            return JSONResponse(status_code=409, content=browser)
    return await _browser_task_payload(task, request)


@app.post("/api/browser/session/{session_id}/use-api-search")
async def browser_session_use_api_search(session_id: str, request: Request):
    task = _browser_task_for_session(session_id, request)
    if task is None:
        return JSONResponse(status_code=404, content={"error": "Browser session not found."})
    from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime

    context = _browser_request_context(request)
    result = await get_opensandbox_browser_runtime().use_api_search(
        session_id,
        owner_user_id=context[0],
        owner_organization_id=context[1],
    )
    task.status = "completed" if result.get("ok") else "failed"
    task.final_answer = result.get("final_answer")
    get_task_engine().event(task, "browser.api_search", {
        "status": result.get("status"),
        "ok": bool(result.get("ok")),
    })
    return await _browser_task_payload(task, request)


@app.post("/api/browser/session/stop")
async def local_browser_session_stop(body: LocalBrowserSessionRequest, request: Request):
    task = _browser_task_for_session(body.session_id, request)
    if task is None:
        return JSONResponse(status_code=404, content={"error": "Browser session not found."})
    await get_task_engine().stop_async(task.task_id)
    return await _browser_task_payload(task, request)


@app.post("/api/browser/navigate")
async def local_browser_navigate(body: LocalBrowserNavigateRequest, request: Request):
    task = _browser_task_for_session(body.session_id, request)
    if task is None:
        return JSONResponse(status_code=404, content={"ok": False, "error": "Browser session not found."})
    await get_task_engine().execute_async(
        task.task_id,
        ToolCall("browser.opensandbox.navigate", {"url": body.url}),
    )
    return await _browser_task_payload(task, request)


@app.post("/api/browser/action")
async def local_browser_action(body: LocalBrowserActionRequest, request: Request):
    """
    Execute one deterministic browser action.

    Page text is observation data only and never influences policy. Sensitive
    element metadata can make an action stricter (approval/block), never less
    restrictive. Re-submit with approved=true to execute the pending action,
    or deny=true to discard it.
    """
    action = {
        key: value for key, value in body.model_dump(
            exclude={"session_id", "approved", "deny"}
        ).items()
        if value is not None
    }
    task = _browser_task_for_session(body.session_id, request)
    if task is None:
        return JSONResponse(status_code=404, content={"ok": False, "error": "Browser session not found."})
    engine = get_task_engine()
    if body.deny:
        if task.pending_approval is not None:
            await engine.approve_async(
                task.task_id,
                task.pending_approval.approval_id,
                approved=False,
            )
        else:
            await _runtime_browser_action(body.session_id, request, deny=True)
    else:
        if task.pending_approval is not None and body.approved:
            pending = task.pending_approval
            replacement = ToolCall(
                "browser.opensandbox.action",
                action,
                approved=True,
            )
            await engine.approve_async(
                task.task_id,
                pending.approval_id,
                approved=True,
                replacement_call=replacement if pending.resubmission_required else None,
            )
            await _resume_browser_task(task.task_id)
        else:
            await engine.execute_async(
                task.task_id,
                ToolCall("browser.opensandbox.action", action, approved=body.approved),
            )
    return await _browser_task_payload(task, request)


@app.get("/api/browser/observe")
async def local_browser_observe(
    session_id: str, request: Request, screenshot: bool = True
):
    task = _browser_task_for_session(session_id, request)
    if task is None:
        return JSONResponse(status_code=404, content={"ok": False, "error": "Browser session not found."})
    await get_task_engine().execute_async(
        task.task_id,
        ToolCall("browser.opensandbox.observe", {"screenshot": screenshot}),
    )
    return await _browser_task_payload(task, request)


@app.get("/api/browser/status")
async def local_browser_status(request: Request, session_id: str | None = None):
    from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime

    if session_id:
        task = _browser_task_for_session(session_id, request)
        if task is None:
            return JSONResponse(status_code=404, content={"error": "Browser session not found."})
        return await _browser_task_payload(task, request)
    return await get_opensandbox_browser_runtime().status()


@app.get("/api/agent/browser/capabilities")
async def browser_agent_capabilities():
    """Return the OpenSandbox browser capability contract.

    The path is retained for clients that have not migrated to
    ``/api/browser/status``.  It no longer inspects or advertises any of the
    former mock, local-Playwright, Electron, Browserbase, or Chrome-MCP
    backends.
    """
    from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime

    state = await get_opensandbox_browser_runtime().availability()
    available = bool(state.get("available"))
    return {
        "backend": "opensandbox",
        "opensandbox_available": available,
        "real_observation_supported": available,
        "real_actions_enabled": available,
        "real_actions_supported": available,
        "action_backend": "opensandbox" if available else "unavailable",
        "observation_backend": "opensandbox" if available else "unavailable",
        "available_backends": ["opensandbox"] if available else [],
        "actions_always_mocked": False,
        "status": state.get("status", "available" if available else "unavailable"),
        "error": state.get("error"),
        "server": state.get("server"),
        "image": state.get("image"),
        "deprecated_backends": [
            "mock",
            "playwright",
            "playwright_mcp",
            "browserbase",
            "electron",
            "chrome_mcp",
            "extension",
        ],
    }


async def _start_browser_session(
    req: Any,
    *,
    run_now: bool = True,
    owner_user_id: str | None = None,
    owner_organization_id: str | None = None,
    task_id: str | None = None,
    allowed_domains: list[str] | None = None,
    source: str = "browser-agent",
    request: Request | None = None,
) -> tuple[dict[str, Any] | None, str | JSONResponse | None]:
    """Create and optionally run a task-owned OpenSandbox browser session."""
    from theriad.agents.browser_policy import detect_injection_in_goal
    from theriad.agents.task_runner import run_task

    if detect_injection_in_goal(req.goal):
        return None, JSONResponse(
            status_code=400,
            content={"error": "Goal contains suspicious content and was rejected."},
        )

    requested_backend = str(getattr(req, "browser_backend", None) or "opensandbox").lower()
    if requested_backend != "opensandbox":
        return None, JSONResponse(
            status_code=410,
            content={
                "error": f"Browser backend '{requested_backend}' is deprecated.",
                "replacement": "opensandbox",
            },
            headers={"Deprecation": "true", "Sunset": "0"},
        )

    engine = get_task_engine()
    domain_allowlist = sorted({str(item).strip().lower() for item in (allowed_domains or []) if str(item).strip()})
    if task_id:
        try:
            task = engine.get(task_id)
        except Exception as exc:
            return None, JSONResponse(status_code=404, content={"error": str(exc)})
        if task.owner_user_id != owner_user_id or task.owner_organization_id != owner_organization_id:
            return None, JSONResponse(status_code=404, content={"error": "Task not found"})
        if domain_allowlist:
            task.allowed_domains = domain_allowlist
    else:
        try:
            task = engine.create(
                goal=req.goal,
                permission_profile="strict",
                requested_capabilities=["browser"],
                allowed_domains=domain_allowlist,
                budgets={
                    "max_tool_calls": max(1, min(int(getattr(req, "max_steps", 20)), 100)),
                    "max_runtime_seconds": max(5, min(int(getattr(req, "max_runtime_seconds", 300)), 3600)),
                    "max_model_calls": max(1, min(int(getattr(req, "max_model_calls", 10)), 100)),
                    "max_cost_usd": getattr(req, "max_cost_usd", None),
                },
                owner_user_id=owner_user_id,
                owner_organization_id=owner_organization_id,
                metadata={
                    "source": source,
                    "browser_only": True,
                    "browser_backend": "opensandbox",
                },
            )
        except (TypeError, ValueError) as exc:
            return None, JSONResponse(status_code=422, content={"error": str(exc)})

    task.metadata.update({"browser_only": True, "browser_backend": "opensandbox"})
    engine.checkpoint(task.task_id)
    start_call = ToolCall(
        "browser.opensandbox.start",
        {
            "start_url": getattr(req, "start_url", None),
            "user_prompt": req.goal,
            "visible": bool(getattr(req, "visible", False)),
            "viewport": {
                "width": 1280,
                "height": 900,
            },
        },
    )
    started = await engine.execute_async(task.task_id, start_call)
    if not started.ok:
        error = started.error or "OpenSandbox browser could not start."
        if task.status not in {"failed", "stopped"}:
            engine.fail(task.task_id, error)
        status_code = 503 if "unavailable" in error.lower() or "opensandbox" in error.lower() else 400
        return None, JSONResponse(
            status_code=status_code,
            content={
                "error": error,
                "backend": "opensandbox",
                "setup": "Start OpenSandbox and configure THERIAD_OPENSANDBOX_DOMAIN.",
            },
        )

    observed = await engine.execute_async(
        task.task_id,
        ToolCall(
            "browser.opensandbox.observe",
            {"screenshot": bool(getattr(req, "capture_screenshot", False))},
        ),
    )
    if not observed.ok:
        error = observed.error or "OpenSandbox browser observation failed."
        engine.fail(task.task_id, error)
        return None, JSONResponse(status_code=503, content={"error": error, "backend": "opensandbox"})

    if run_now:
        await run_task(
            task.task_id,
            engine=engine,
            only_capabilities={"browser"},
            max_calls=max(1, min(int(getattr(req, "max_steps", 20)), 100)),
        )
    status_request = request or Request({
        "type": "http",
        "method": "GET",
        "path": "/api/browser/status",
        "headers": [],
    })
    status_request.state.theriad_browser_owner = (owner_user_id, owner_organization_id)
    return await _browser_task_payload(task, status_request), None


@app.post("/api/agent/browser/start")
async def browser_agent_start(body: dict, request: Request):
    """Start and optionally run a task-owned OpenSandbox browser session."""
    from theriad.agents.browser_schemas import BrowserAgentStartRequest
    from theriad.skills import build_skill_context

    try:
        project_id = _resolve_project_scope(
            body.get("conversation_id"),
            body.get("project_id"),
            scope_id=user_scope_id(request),
        )
    except _ProjectScopeError as exc:
        return _project_scope_error_response(exc)

    try:
        req = BrowserAgentStartRequest(
            **{
                **body,
                # Server-selected global/project skills; never trust a client to
                # smuggle arbitrary text into the internal skill channel.
                "skill_context": build_skill_context(project_id).text,
            }
        )
    except Exception as exc:
        return JSONResponse(status_code=422, content={"error": f"Invalid request: {exc}"})

    from theriad.api.auth_context import get_auth_context
    context = get_auth_context(request)
    payload, warning_or_response = await _start_browser_session(
        req,
        owner_user_id=user_scope_id(request),
        owner_organization_id=context.organization_id,
        allowed_domains=body.get("allowed_domains") or [],
        source="browser-agent-api",
        request=request,
    )
    if payload is None:
        return warning_or_response

    payload["observation_warning"] = warning_or_response
    payload["message"] = "OpenSandbox session created and running. Poll status for progress."
    return payload


@app.post("/api/agent/browser/step")
async def browser_agent_step(body: dict, request: Request):
    """Advance the canonical task runner by a bounded number of calls."""
    session_id = body.get("session_id")
    if not session_id:
        return JSONResponse(status_code=422, content={"error": "session_id is required"})
    try:
        steps = max(1, min(int(body.get("steps", 1)), 100))
    except (TypeError, ValueError):
        return JSONResponse(status_code=422, content={"error": "steps must be an integer"})

    task = _browser_task_for_session(str(session_id), request)
    if task is None:
        return JSONResponse(status_code=404, content={"error": f"Session '{session_id}' not found"})

    before_sequence = len(task.events)
    if task.status not in {"completed", "failed", "stopped"} and not task.pending_approval:
        await _resume_browser_task(task.task_id, max_calls=steps)
    result = await _browser_task_payload(task, request)
    result["steps_run_this_call"] = sum(
        1
        for event in task.events
        if event.sequence > before_sequence and event.kind == "tool.started"
    )
    return result


@app.post("/api/agent/browser/stop")
async def browser_agent_stop(body: dict, request: Request):
    """Stop a task and destroy its OpenSandbox browser handles."""
    session_id = body.get("session_id")
    if not session_id:
        return JSONResponse(status_code=422, content={"error": "session_id is required"})

    task = _browser_task_for_session(str(session_id), request)
    if task is None:
        return JSONResponse(status_code=404, content={"error": f"Session '{session_id}' not found"})
    engine = get_task_engine()
    await engine.stop_async(task.task_id)
    return await _browser_task_payload(task, request)


@app.get("/api/agent/browser/status")
async def browser_agent_status(session_id: str, request: Request):
    """Return owner-scoped OpenSandbox/task status without secrets."""
    task = _browser_task_for_session(session_id, request)
    if task is None:
        return JSONResponse(status_code=404, content={"error": f"Session '{session_id}' not found"})
    return await _browser_task_payload(task, request)


async def _resolve_and_resume(
    session_id: str,
    *,
    approved: bool,
    reason: str,
    request: Request,
    replacement_call: ToolCall | None = None,
):
    """Resolve a TaskEngine approval and continue the same browser task."""
    task = _browser_task_for_session(session_id, request)
    if task is None:
        return JSONResponse(status_code=404, content={"error": f"Session '{session_id}' not found"})

    pending = task.pending_approval
    if pending is None:
        return JSONResponse(status_code=400, content={"error": "No pending approval on this session"})

    engine = get_task_engine()
    try:
        result = await engine.approve_async(
            task.task_id,
            pending.approval_id,
            approved=approved,
            replacement_call=replacement_call,
        )
    except Exception as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    if reason:
        engine.event(task, "approval.reason", {"reason": str(reason)[:500]})
    if not result.approval_required and task.status == "running":
        await _resume_browser_task(task.task_id)

    return await _browser_task_payload(task, request)


@app.post("/api/agent/browser/approve")
async def browser_agent_approve(body: dict, request: Request):
    """Approve a pending canonical OpenSandbox action."""
    session_id = body.get("session_id")
    approved = body.get("approved", True)
    reason = body.get("reason", "")

    if not session_id:
        return JSONResponse(status_code=422, content={"error": "session_id is required"})

    return await _resolve_and_resume(
        session_id,
        approved=bool(approved),
        reason=reason,
        request=request,
        replacement_call=_browser_replacement_call(body),
    )


@app.post("/api/agent/browser/deny")
async def browser_agent_deny(body: dict, request: Request):
    """Deny a pending canonical OpenSandbox action and resume safely."""
    session_id = body.get("session_id")
    reason = body.get("reason", "")

    if not session_id:
        return JSONResponse(status_code=422, content={"error": "session_id is required"})

    return await _resolve_and_resume(
        session_id,
        approved=False,
        reason=reason,
        request=request,
    )


@app.post("/api/agent/browser/pause_login")
async def browser_agent_pause_login(body: dict, request: Request):
    """Retire the local-browser login handoff with the legacy systems."""
    return JSONResponse(
        status_code=410,
        content={
            "error": "Manual local-browser login is deprecated.",
            "detail": (
                "OpenSandbox sessions are isolated server-side; this API does not expose a "
                "personal Chrome or extension window for manual login."
            ),
            "replacement": "Use a task-owned OpenSandbox session and explicit browser approvals.",
        },
        headers={"Deprecation": "true", "Sunset": "0"},
    )


@app.post("/api/agent/browser/continue_login")
async def browser_agent_continue_login(body: dict, request: Request):
    """Retire the local-browser login continuation endpoint."""
    return JSONResponse(
        status_code=410,
        content={
            "error": "Manual local-browser login is deprecated.",
            "replacement": "Use a task-owned OpenSandbox session and explicit browser approvals.",
        },
        headers={"Deprecation": "true", "Sunset": "0"},
    )
@app.post("/api/agent/browser/continue_captcha")
async def browser_agent_continue_captcha(body: dict, request: Request):
    """Continue a CAPTCHA handoff through the task-owned OpenSandbox runtime."""
    session_id = body.get("session_id")
    if not session_id:
        return JSONResponse(status_code=422, content={"error": "session_id is required"})
    task = _browser_task_for_session(str(session_id), request)
    if task is None:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime

    owner_user_id, owner_organization_id = _browser_request_context(request)
    browser = await get_opensandbox_browser_runtime().status(
        str(session_id),
        owner_user_id=owner_user_id,
        owner_organization_id=owner_organization_id,
    )
    if not browser.get("captcha_detected"):
        return JSONResponse(status_code=400, content={"error": "Session is not paused on a CAPTCHA"})
    result = await _runtime_browser_action(str(session_id), request, approved=True)
    get_task_engine().event(task, "browser.captcha.continued", {"ok": bool(result.get("ok"))})
    if result.get("ok") and task.status not in {"completed", "failed", "stopped"}:
        task.status = "running"
        await _resume_browser_task(task.task_id)
    return await _browser_task_payload(task, request)


@app.post("/api/agent/browser/use_api_search")
async def browser_agent_use_api_search(body: dict, request: Request):
    """Leave a CAPTCHA page and finish through the public API-search lane."""
    session_id = body.get("session_id")
    if not session_id:
        return JSONResponse(status_code=422, content={"error": "session_id is required"})
    task = _browser_task_for_session(str(session_id), request)
    if task is None:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    owner_user_id, owner_organization_id = _browser_request_context(request)
    from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime

    result = await get_opensandbox_browser_runtime().use_api_search(
        str(session_id),
        owner_user_id=owner_user_id,
        owner_organization_id=owner_organization_id,
    )
    task.status = "completed" if result.get("ok") else "failed"
    task.final_answer = result.get("final_answer") or result.get("final_result")
    get_task_engine().event(task, "browser.api_search", {
        "status": result.get("status"),
        "ok": bool(result.get("ok")),
    })
    return await _browser_task_payload(task, request)


# ══════════════════════════════════════════════════════════════════════════════
# Chat-integrated agents (Part A/B) — main chat → intent → agent runtime.
# The Agent page stays available as the detailed trace/debug console; chat
# runs reuse the exact same browser runtime via _start_browser_session().
# ══════════════════════════════════════════════════════════════════════════════

async def _browser_status_payload(
    session_id: str | None,
    *,
    owner_user_id: str | None = None,
    owner_organization_id: str | None = None,
    task_id: str | None = None,
) -> dict | None:
    if not session_id:
        return None
    engine = get_task_engine()
    task = None
    if task_id:
        try:
            task = engine.get(task_id)
        except Exception:
            task = None
    if task is not None and task.metadata.get("browser_session_id") != str(session_id):
        task = None
    if task is None:
        task = engine.task_for_browser_session(str(session_id))
    if task is None:
        return None
    if task.owner_user_id != owner_user_id or task.owner_organization_id != owner_organization_id:
        if (
            task.owner_user_id is not None
            or owner_user_id is not None
            or task.owner_organization_id is not None
            or owner_organization_id is not None
        ):
            return None
    return await _browser_task_payload(
        task,
        _browser_owner_request(owner_user_id, owner_organization_id),
    )


@app.post("/api/chat/agent")
async def chat_agent_start(body: dict, request: Request):
    """
    Agent-capable chat path (Part A).

    Classifies the message (normal_chat | browser_agent | terminal_agent |
    multi_tool_agent). For agent intents, creates a ChatAgentRun and returns
    a chat-shaped payload with agent_run_id + compact status the chat UI
    renders inline. normal_chat returns intent only — the caller should use
    /api/chat/send as usual.

    Terminal execution is routed through TaskEngine → TerminalTool and is
    policy-gated: mock mode (default) only logs proposed commands.
    """
    import asyncio

    from theriad.agents.chat_agent import (
        ChatAgentRun,
        checkpoint_chat_run,
        classify_chat_intent,
        get_chat_agent_run_store,
        propose_terminal_commands,
    )
    from theriad.agents.browser_schemas import BrowserAgentStartRequest
    from theriad.agents.task_engine import ToolCall, get_task_engine
    from theriad.api.auth_context import get_auth_context
    from theriad.search import is_api_search_request
    from theriad.skills import build_skill_context

    message = (body.get("message") or "").strip()
    if not message:
        return JSONResponse(status_code=422, content={"error": "message is required"})

    _agent_user = get_request_user(request)
    auth_context = get_auth_context(request)
    intent_result = classify_chat_intent(message)
    api_search_request = (
        intent_result.intent in ("browser_agent", "multi_tool_agent")
        and not body.get("start_url")
        and is_api_search_request(message)
    )
    read_only_api_search_request = (
        api_search_request and intent_result.intent == "browser_agent"
    )
    hosted_mode = is_hosted_deployment()
    if hosted_mode and not read_only_api_search_request:
        try:
            from theriad.storage.audit import record_event

            record_event(
                "hosted_agent_blocked",
                organization_id=auth_context.organization_id,
                user_id=auth_context.user_id,
                resource_type="http_path",
                resource_id=request.url.path,
                metadata={
                    "method": request.method,
                    "reason": "sandbox_not_deployed",
                    "intent": intent_result.intent,
                    "api_search_request": api_search_request,
                },
            )
        except Exception:
            # A missing audit store must not turn a safe denial into an agent
            # execution opportunity or leak persistence details to the caller.
            pass
    if _agent_user and _agent_user.get("is_guest") and not read_only_api_search_request:
        return JSONResponse(
            status_code=403,
            content={
                "error": (
                    "Guest mode supports read-only web search and free chat only. "
                    "Create an account to use browser or terminal tasks."
                )
            },
        )
    # Hosted mode permits this route only for a pure, read-only API search. A
    # multi-tool request may contain a safe web leg, but it can also execute a
    # terminal leg and therefore still requires an isolated executor.
    if hosted_mode and not read_only_api_search_request:
        return _hosted_agent_unavailable_response()
    if body.get("conversation_id"):
        from theriad.storage.database import get_db as _get_db
        _owned = _get_db().execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id IS ?",
            (body.get("conversation_id"), _agent_user["id"] if _agent_user else None),
        ).fetchone()
        if _owned is None:
            return JSONResponse(
                status_code=404, content={"error": "Conversation not found"}
            )

    try:
        project_id = _resolve_project_scope(
            body.get("conversation_id"),
            body.get("project_id"),
            scope_id=user_scope_id(request),
        )
    except _ProjectScopeError as exc:
        return _project_scope_error_response(exc)

    if intent_result.intent == "normal_chat":
        return {
            "intent": "normal_chat",
            "agent_run_id": None,
            "classification": intent_result.to_dict(),
            "message": "No agent needed — send through /api/chat/send.",
        }

    requested_capabilities = []
    if intent_result.intent in ("browser_agent", "multi_tool_agent"):
        requested_capabilities.append("web")
        if not api_search_request:
            requested_capabilities.append("browser")
    if intent_result.intent in ("terminal_agent", "multi_tool_agent"):
        requested_capabilities.append("terminal")
    principal_id = None
    if _agent_user:
        principal_id = (
            f"supabase:{_agent_user['supabase_user_id']}"
            if _agent_user.get("supabase_user_id")
            else f"local:{_agent_user['id']}"
        )
    engine = get_task_engine()
    try:
        task = engine.create(
            goal=message,
            permission_profile=str(body.get("task_permission_profile") or "strict"),
            requested_capabilities=requested_capabilities,
            budgets={
                "max_steps": int(body.get("max_steps", 20)),
                "max_runtime_seconds": int(body.get("max_runtime_seconds", 120)),
                "max_model_calls": int(body.get("max_model_calls", 8)),
                "max_cost_usd": body.get("max_cost_usd", 0.05),
            },
            owner_user_id=_agent_user["id"] if _agent_user else None,
            owner_organization_id=auth_context.organization_id,
            principal_id=principal_id,
            metadata={"source": "chat"},
        )
    except (TypeError, ValueError) as exc:
        return JSONResponse(status_code=422, content={"error": f"Invalid task request: {exc}"})
    run = ChatAgentRun(
        message=message,
        intent=intent_result.intent,
        run_id=task.task_id,
        task_id=task.task_id,
        conversation_id=body.get("conversation_id"),
        owner_user_id=_agent_user["id"] if _agent_user else None,
        owner_organization_id=auth_context.organization_id,
    )
    checkpoint_chat_run(run)
    observation_warning: str | None = None
    api_search_completed = False

    # ── Browser leg (browser_agent / multi_tool_agent) ────────────────────────
    if intent_result.intent in ("browser_agent", "multi_tool_agent"):
        # Natural-language web searches use the task engine's web tools. Do not
        # open a search-engine page after successful structured retrieval.
        if api_search_request:
            try:
                from theriad.agents.search_synthesis import synthesize_web_search
                from theriad.search.service import SearchResult, SearchService

                class _TaskWebService(SearchService):
                    def __init__(self, task_id: str) -> None:
                        self.task_id = task_id

                    async def search(self, query: str) -> list[SearchResult]:
                        result = await asyncio.to_thread(
                            engine.execute,
                            self.task_id,
                            ToolCall("web.search", {"query": query, "limit": 8}),
                        )
                        if not result.ok:
                            raise RuntimeError(result.error or "web.search failed")
                        return [
                            SearchResult(**row)
                            for row in (result.output or {}).get("results", [])
                        ]

                    async def fetch(self, url: str) -> dict:
                        result = await asyncio.to_thread(
                            engine.execute,
                            self.task_id,
                            ToolCall("web.fetch", {"url": url}),
                        )
                        if not result.ok:
                            return {"url": url, "status": "unavailable", "text": ""}
                        return result.output or {}

                import inspect as _inspect
                _search_kwargs = {"telemetry": run.search_trace}
                search_parameters = _inspect.signature(synthesize_web_search).parameters
                if "service" in search_parameters:
                    _search_kwargs["service"] = _TaskWebService(task.task_id)
                if "free_only" in search_parameters:
                    # Hosted chat search is an anonymous/public-web surface.
                    # Keep registered hosted users on the verified zero-cost
                    # models too; this route has no paid authorization or
                    # provider-budget settlement boundary.
                    _search_kwargs["free_only"] = bool(
                        hosted_mode
                        or (_agent_user and _agent_user.get("is_guest"))
                    )
                answer, results, _queries = await synthesize_web_search(
                    message, **_search_kwargs,
                )
                run.search_results = results
                run.search_answer = answer
                api_search_completed = True
            except Exception:
                # API search is canonical for read-only requests. Do not
                # silently turn a provider failure into browser navigation.
                run.error = (
                    "Automatic web search is unavailable right now. "
                    "Check the network connection and try again."
                )
                run.search_trace["error"] = "search_unavailable"

        # URL in message: use the same durable web.fetch adapter as agent tasks.
        if not run.error and not api_search_completed and not body.get("start_url"):
            import re as _re
            _url_m = _re.search(r'https?://[^\s<>"\']+', message)
            if _url_m:
                _target_url = _url_m.group(0).rstrip(".,;!?)")
                _page_text: str = ""
                _fetch_result = await asyncio.to_thread(
                    engine.execute,
                    task.task_id,
                    ToolCall("web.fetch", {"url": _target_url}),
                )
                if _fetch_result.ok and isinstance(_fetch_result.output, dict):
                    _page_text = str(_fetch_result.output.get("text") or "").strip()

                if _page_text.strip():
                    import time as _time
                    _url_started = _time.perf_counter()
                    _url_usage: list[dict] = []
                    try:
                        from theriad.agents.llm_planner import _call_model
                        _synthesis = await _call_model(
                            model="xiaomi/mimo-v2.5",
                            tier="cheap_bulk",
                            system_prompt=(
                                "You are a helpful assistant. Answer the user's question "
                                "based on the provided web page content. Be concise and direct."
                            ),
                            user_content=(
                                f"User question: {message}\n\n"
                                f"Web page content from {_target_url}:\n{_page_text[:6000]}"
                            ),
                            # Do not impose an application-level cap on the
                            # user-facing page-fetch answer.
                            max_tokens=None,
                            temperature=0.3,
                            usage_sink=_url_usage,
                        )
                        from theriad.agents.model_telemetry import is_truncated_finish_reason
                        if _url_usage and is_truncated_finish_reason(
                            _url_usage[-1].get("finish_reason")
                        ):
                            _synthesis = ""
                        run.search_answer = _synthesis or (
                            f"[Fetched from {_target_url}]\n\n{_page_text[:4000]}"
                        )
                    except Exception:
                        run.search_answer = f"[Fetched from {_target_url}]\n\n{_page_text[:4000]}"
                    from theriad.agents.model_telemetry import (
                        summarize_model_usage,
                        usage_totals,
                    )
                    _url_cost, _url_cost_known = usage_totals(_url_usage)
                    run.search_trace.update({
                        "execution_kind": "page_fetch",
                        "queries": [message],
                        "queries_used": 1,
                        "source_count": 1,
                        "sources": [{
                            "title": _target_url,
                            "url": _target_url,
                            "snippet": _page_text[:500],
                            "source": "page_fetch",
                        }],
                        "model_calls": len(_url_usage),
                        "models_used": summarize_model_usage(_url_usage),
                        "estimated_cost_usd": _url_cost,
                        "cost_known": _url_cost_known,
                        "elapsed_seconds": round(_time.perf_counter() - _url_started, 3),
                        "retrieved_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                        "evidence_quality": "fetched_page",
                        "evidence_status": "fetched_page",
                        "synthesis_finish_reason": (
                            _url_usage[-1].get("finish_reason") if _url_usage else None
                        ),
                    })
                    api_search_completed = True

        if api_search_completed or run.error:
            pass
        else:
            try:
                req = BrowserAgentStartRequest(
                    goal=message,
                    skill_context=build_skill_context(project_id).text,
                    start_url=body.get("start_url"),
                    policy=body.get("policy", "balanced"),
                    speed_preference=body.get("speed_preference"),
                    quality_preference=body.get("quality_preference"),
                    max_steps=int(body.get("max_steps", 20)),
                    max_runtime_seconds=int(body.get("max_runtime_seconds", 120)),
                    max_model_calls=int(body.get("max_model_calls", 8)),
                    permission_mode=body.get("permission_mode", "ask_high_risk"),
                    real_observation_enabled=True,
                    real_action_enabled=True,
                    browser_backend="opensandbox",
                    max_cost_usd=body.get("max_cost_usd", 0.05),
                    # D-050: enables LLM synthesis of the browser final answer.
                    allow_paid_calls=bool(body.get("allow_paid_calls", False)),
                )
            except Exception as exc:
                await engine.fail_async(task.task_id, f"Invalid browser request: {exc}")
                return JSONResponse(status_code=422, content={"error": f"Invalid request: {exc}"})

            session, warning_or_response = await _start_browser_session(
                req,
                run_now=not bool(body.get("stream_progress", False)),
                owner_user_id=user_scope_id(request),
                owner_organization_id=auth_context.organization_id,
                task_id=task.task_id,
                allowed_domains=body.get("allowed_domains") or [],
                source="chat-agent",
                request=request,
            )
            if session is None:
                await engine.fail_async(task.task_id, "Browser session could not start")
                # Save the browser-start error to history so it isn't silently lost.
                _conv_id = body.get("conversation_id")
                if _conv_id:
                    try:
                        import json as _json
                        from theriad.api.projects import save_message as _sm
                        _resp_body = getattr(warning_or_response, "body", b"{}")
                        _err_data = _json.loads(_resp_body)
                        _err_text = (
                            _err_data.get("setup")
                            or _err_data.get("error")
                            or "Browser agent failed to start."
                        )
                        _sm(conversation_id=_conv_id, role="assistant", content=_err_text)
                    except Exception:
                        pass
                return warning_or_response
            if warning_or_response:
                observation_warning = warning_or_response
            run.browser_session_id = session.get("session_id")
            engine.attach_specialist(
                task.task_id,
                kind="browser",
                specialist_id=run.browser_session_id or "opensandbox",
                details={"backend": "opensandbox"},
            )
            checkpoint_chat_run(run)

    # ── Terminal leg (terminal_agent / multi_tool_agent) ──────────────────────
    if intent_result.intent in ("terminal_agent", "multi_tool_agent") and not run.error:
        engine.attach_specialist(
            task.task_id,
            kind="terminal",
            specialist_id=f"{task.task_id}:terminal",
        )
        commands = propose_terminal_commands(message)
        if not commands:
            run.terminal_commands.append({
                "command": None,
                "verdict": "needs_approval",
                "reason": "No deterministic command matched this request; nothing was run.",
                "executed": False,
            })
            run.terminal_status = "completed"
        else:
            needs_approval = False
            for cmd in commands:
                result = await asyncio.to_thread(
                    engine.execute,
                    task.task_id,
                    ToolCall("terminal.run", {"command": cmd}),
                )
                payload = result.output or {"command": cmd, "verdict": "blocked",
                                            "reason": result.error, "executed": False}
                if result.approval_required:
                    payload["verdict"] = "needs_approval"
                    payload["approval_id"] = (
                        task.pending_approval.approval_id
                        if task.pending_approval else None
                    )
                run.terminal_commands.append(payload)
                if result.approval_required:
                    needs_approval = True
                    break
                if not result.ok:
                    run.error = result.error or f"Command failed: {cmd}"
                    break
            run.terminal_status = (
                "waiting_for_approval" if needs_approval
                else "failed" if run.error
                else "completed"
            )
        engine.sync_specialist(
            task.task_id,
            kind="terminal",
            status=run.terminal_status or "running",
            final_answer=run.search_answer,
            error=run.error,
            details={"command_count": len(run.terminal_commands)},
        )

    checkpoint_chat_run(run)
    get_chat_agent_run_store().add(run)

    if run.browser_session_id and body.get("stream_progress", False):
        # Start the autonomous loop after the run is registered so the first
        # poll can show "Planning your task…" immediately.
        asyncio.create_task(_resume_browser_task(task.task_id))

    browser_payload = await _browser_status_payload(
        run.browser_session_id,
        owner_user_id=run.owner_user_id,
        owner_organization_id=run.owner_organization_id,
        task_id=task.task_id,
    )
    status = run.to_status(browser_payload)
    status["observation_warning"] = observation_warning
    status["classification"] = intent_result.to_dict()
    if run.browser_session_id and browser_payload:
        engine.sync_specialist(
            task.task_id,
            kind="browser",
            status=str(browser_payload.get("status") or "running"),
            final_answer=browser_payload.get("final_answer"),
            error=browser_payload.get("error"),
            details={"steps_taken": browser_payload.get("steps_taken", 0)},
        )
    if status["phase"] == "completed" and task.status != "completed":
        engine.execute(
            task.task_id,
            ToolCall(
                "task.complete",
                {"answer": status.get("trace", {}).get("final_answer") or "Task completed."},
            ),
        )
    elif status["phase"] == "failed" and task.status != "failed":
        await engine.fail_async(task.task_id, status.get("error") or "Chat agent failed")
    checkpoint_chat_run(run)

    # Save the assistant message immediately for runs that have already
    # reached a terminal phase (search runs, synchronous browser runs).
    # Async browser runs will be saved on the first poll that sees "completed".
    _try_save_agent_message(run, status)

    return status


def _try_save_agent_message(run, status: dict) -> None:
    """Persist the assistant message for a terminal agent run (idempotent)."""
    if (
        run.assistant_message_saved
        or not run.conversation_id
        or status.get("phase") not in ("completed", "failed")
    ):
        return
    try:
        from theriad.api.projects import save_message as _save_msg
        trace = status.get("trace", {})
        content = (
            trace.get("final_answer")
            or trace.get("final_result")
            or status.get("error")
            or f"Agent run {status.get('phase_label', status.get('phase', 'finished')).lower()} ({run.intent})."
        )
        _save_msg(conversation_id=run.conversation_id, role="assistant", content=content)
        run.assistant_message_saved = True
        from theriad.agents.chat_agent import checkpoint_chat_run
        checkpoint_chat_run(run)
    except Exception:
        pass  # persistence failure must not break the response


@app.get("/api/chat/agent/status")
async def chat_agent_status(run_id: str, request: Request):
    """Poll a chat agent run — compact phase + expandable trace (Part F)."""
    from theriad.agents.chat_agent import checkpoint_chat_run, get_chat_agent_run_store
    from theriad.agents.task_engine import ToolCall, get_task_engine

    run = get_chat_agent_run_store().get(
        run_id, owner_user_id=user_scope_id(request)
    )
    if run is None:
        return JSONResponse(status_code=404, content={"error": f"Run '{run_id}' not found"})

    engine = get_task_engine()
    task = engine.get(run.task_id or run.run_id)
    if task.principal_id:
        current_user = get_request_user(request)
        current_principal = None
        if current_user:
            current_principal = (
                f"supabase:{current_user['supabase_user_id']}"
                if current_user.get("supabase_user_id")
                else f"local:{current_user['id']}"
            )
        if current_principal != task.principal_id:
            return JSONResponse(
                status_code=404,
                content={"error": f"Run '{run_id}' not found"},
            )
    if run.terminal_status == "waiting_for_approval" and task.pending_approval is None:
        # A completed approval can be present in the durable task context even
        # when the in-memory chat run has not yet seen the corresponding poll.
        terminal_record = next(
            (
                row.get("result")
                for row in reversed(task.context)
                if row.get("tool") == "terminal.run"
            ),
            None,
        )
        if isinstance(terminal_record, dict):
            command = terminal_record.get("command")
            replaced = False
            for index, item in enumerate(run.terminal_commands):
                if item.get("command") == command:
                    run.terminal_commands[index] = terminal_record
                    replaced = True
                    break
            if not replaced:
                run.terminal_commands.append(terminal_record)
            run.terminal_status = "completed"
            engine.sync_specialist(
                task.task_id,
                kind="terminal",
                status="completed",
                final_answer=run.search_answer,
                details={"command_count": len(run.terminal_commands)},
            )
    browser_payload = await _browser_status_payload(
        run.browser_session_id,
        owner_user_id=run.owner_user_id,
        owner_organization_id=run.owner_organization_id,
        task_id=task.task_id,
    )
    status = run.to_status(browser_payload)

    if run.browser_session_id and browser_payload:
        engine.sync_specialist(
            task.task_id,
            kind="browser",
            status=str(browser_payload.get("status") or "running"),
            final_answer=browser_payload.get("final_answer"),
            error=browser_payload.get("error"),
            details={"steps_taken": browser_payload.get("steps_taken", 0)},
        )
    if status["phase"] == "completed" and task.status != "completed":
        engine.execute(
            task.task_id,
            ToolCall(
                "task.complete",
                {"answer": status.get("trace", {}).get("final_answer") or "Task completed."},
            ),
        )
    elif status["phase"] == "failed" and task.status != "failed":
        await engine.fail_async(task.task_id, status.get("error") or "Chat agent failed")
    checkpoint_chat_run(run)
    _try_save_agent_message(run, status)
    return status


# ── Memory layer ──────────────────────────────────────────────────────────────
# A modular, hand-editable memory file (data/memory.json) users can grow via
# the UI, direct file edits, or imports of ChatGPT/Claude conversation exports.
# Local/dev requests use the hand-editable JSON store. When the hosted
# Supabase adapter is enabled, registered users use tenant-scoped Postgres rows
# instead; hosted memory never falls back to data/memory-users/.


def _memory_store_for_request(request: Request):
    """Return tenant-backed memory in hosted mode and SQLite memory locally."""
    from hashlib import sha256
    from os import getenv

    from theriad.memory import get_memory_store
    from theriad.storage.hosted import hosted_data_plane_enabled

    if hosted_data_plane_enabled():
        from fastapi import HTTPException

        from theriad.api.auth_context import get_auth_context
        from theriad.memory.hosted import HostedMemoryStore

        context = get_auth_context(request)
        if context.principal_type not in {"registered", "guest"} or not context.organization_id or not context.user_id:
            raise HTTPException(
                status_code=403,
                detail="Create a registered workspace before saving hosted memory.",
            )
        return HostedMemoryStore(context.organization_id, context.user_id)

    scope_id = user_scope_id(request)
    if scope_id is None:
        return get_memory_store()
    data_dir = _Path(getenv("THERIAD_DATA_DIR") or "data")
    digest = sha256(scope_id.encode("utf-8")).hexdigest()[:32]
    return get_memory_store(data_dir / "memory-users" / f"{digest}.json")

class MemoryAddRequest(BaseModel):
    text: str
    title: str | None = None
    tags: list[str] = []
    source: str = "manual"


class MemoryUpdateRequest(BaseModel):
    text: str | None = None
    title: str | None = None
    tags: list[str] | None = None
    source: str | None = None


class MemoryImportRequest(BaseModel):
    # Parsed JSON of a conversation export or generic {title, messages} data.
    # Sent as a JSON body, not a file upload, so callers can validate the file
    # client-side first.
    # For pre-parsed entries from CSV/TXT/MD, set format="direct".
    data: list
    name: str | None = None
    # Accepted for compatibility with older provider-labelled import clients.
    provider_hint: str | None = None
    format: str | None = None


def _suggest_memory_upload_name(filenames, import_name: str | None = None, fmt: str | None = None) -> str:
    """Turn an uploaded filename into a compact, editable @mention alias."""
    import re

    raw = str(filenames[0] if filenames else "").replace("\\", "/").rsplit("/", 1)[-1]
    stem = raw.rsplit(".", 1)[0] if "." in raw else raw
    if import_name and import_name.strip():
        stem = import_name.strip()
    elif not stem or stem.lower() in {"conversations", "memory"}:
        stem = fmt or "upload"
    alias = re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-_").lower()
    return alias[:40] or "upload"


def _memory_import_response(fmt, entries, skipped, store, *, suggested_name=None):
    upload = store.add_upload(entries, suggested_name=suggested_name)
    return {
        "format": fmt,
        "files_skipped": skipped,
        "conversations_parsed": len(entries),
        "memories_added": upload["count"] if upload else 0,
        "duplicates_skipped": len(entries) - (upload["count"] if upload else 0),
        "upload": upload,
    }


@app.get("/api/memory")
def memory_list(request: Request, source: str | None = None, tag: str | None = None, q: str | None = None):
    store = _memory_store_for_request(request)
    entries = store.list(source=source, tag=tag, query=q)
    return {
        "memories": [e.to_dict() for e in entries],
        "count": len(entries),
        "file_path": str(store.path),
        "name": store.name,
        "uploads": store.upload_sources(),
    }


class MemoryRenameRequest(BaseModel):
    name: str


@app.patch("/api/memory/name")
def memory_rename(body: MemoryRenameRequest, request: Request):
    store = _memory_store_for_request(request)
    try:
        store.set_name(body.name)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    return {"name": store.name}


class MemoryUploadRenameRequest(BaseModel):
    name: str


class ModelMemoryUpdateRequest(BaseModel):
    model_id: str
    context_enabled: bool | None = None
    learning_enabled: bool | None = None
    max_memories: int | None = Field(default=None, ge=10, le=500)


def _known_memory_models() -> list[dict]:
    """Return registry/direct models that can receive their own memory profile."""
    from theriad.core.providers.runtime import DIRECT_MODEL_BY_PROVIDER_AND_TIER
    from theriad.memory.learning import model_memory_id
    from theriad.models.registry import DEFAULT_REGISTRY

    models: dict[str, dict] = {}
    for tier in DEFAULT_REGISTRY.all_tiers():
        for candidate in DEFAULT_REGISTRY.candidates_for_tier(tier):
            model_id = model_memory_id(candidate.provider, candidate.model)
            models[model_id] = {
                "model_id": model_id,
                "provider": candidate.provider,
                "model": candidate.model,
                "label": candidate.model,
            }
    for provider, tiers in DIRECT_MODEL_BY_PROVIDER_AND_TIER.items():
        for model in set(tiers.values()):
            model_id = model_memory_id(provider, model)
            models.setdefault(
                model_id,
                {"model_id": model_id, "provider": provider, "model": model, "label": model},
            )
    return sorted(models.values(), key=lambda item: (item["provider"], item["model"]))


@app.get("/api/memory/models")
def memory_models(request: Request):
    store = _memory_store_for_request(request)
    profiles = {profile["model_id"]: profile for profile in store.model_profiles()}
    models = {model["model_id"]: model for model in _known_memory_models()}
    for model_id in profiles:
        models.setdefault(
            model_id,
            {
                "model_id": model_id,
                "provider": model_id.split(":", 1)[0],
                "model": model_id.split(":", 1)[1] if ":" in model_id else model_id,
                "label": model_id,
            },
        )
    return {
        "models": [
            {**model, "profile": profiles.get(model_id)}
            for model_id, model in sorted(
                models.items(), key=lambda item: (item[1]["provider"], item[1]["model"])
            )
        ],
        "profiles": list(profiles.values()),
    }


@app.patch("/api/memory/models")
def memory_model_update(body: ModelMemoryUpdateRequest, request: Request):
    try:
        profile = _memory_store_for_request(request).update_model_profile(
            body.model_id,
            context_enabled=body.context_enabled,
            learning_enabled=body.learning_enabled,
            max_memories=body.max_memories,
        )
    except (TypeError, ValueError) as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    return {"profile": profile}


@app.get("/api/memory/mentions")
def memory_mentions(request: Request):
    from theriad.memory.providers import memory_mentions as get_memory_mentions

    return {"mentions": get_memory_mentions(_memory_store_for_request(request))}


@app.patch("/api/memory/uploads/{upload_id}")
def memory_upload_rename(upload_id: str, body: MemoryUploadRenameRequest, request: Request):
    try:
        return {"upload": _memory_store_for_request(request).rename_upload(upload_id, body.name)}
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})


@app.post("/api/memory")
def memory_add(body: MemoryAddRequest, request: Request):
    try:
        entry = _memory_store_for_request(request).add(
            body.text, title=body.title, tags=body.tags, source=body.source
        )
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    return {"memory": entry.to_dict()}


@app.put("/api/memory/{entry_id}")
def memory_update(entry_id: str, body: MemoryUpdateRequest, request: Request):
    entry = _memory_store_for_request(request).update(
        entry_id, text=body.text, title=body.title, tags=body.tags, source=body.source
    )
    if entry is None:
        return JSONResponse(status_code=404, content={"error": f"Memory '{entry_id}' not found"})
    return {"memory": entry.to_dict()}


@app.delete("/api/memory/{entry_id}")
def memory_delete(entry_id: str, request: Request):
    if not _memory_store_for_request(request).delete(entry_id):
        return JSONResponse(status_code=404, content={"error": f"Memory '{entry_id}' not found"})
    return {"deleted": entry_id}


@app.post("/api/memory/import")
def memory_import(body: MemoryImportRequest, request: Request):
    from theriad.memory import MemoryEntry
    from theriad.memory.importers import ImportError_, detect_and_import, import_conversation_export

    store = _memory_store_for_request(request)

    if body.format == "direct":
        entries = []
        for item in body.data:
            if not isinstance(item, dict) or not item.get("text", "").strip():
                continue
            entries.append(MemoryEntry(
                text=item["text"].strip(),
                tags=item.get("tags", []),
                source=item.get("source", "import"),
            ))
        return _memory_import_response(
            "direct", entries, 0, store,
            suggested_name=_suggest_memory_upload_name(
                [], body.name or body.provider_hint, "imported-memory"
            ),
        )

    try:
        if body.provider_hint:
            fmt, entries = detect_and_import(body.data, body.provider_hint)
        else:
            fmt, entries = import_conversation_export(body.data)
    except ImportError_ as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    return _memory_import_response(
        fmt, entries, 0, store,
        suggested_name=_suggest_memory_upload_name([], body.name or body.provider_hint, fmt),
    )


@app.post("/api/memory/import/files")
async def memory_import_files(
    request: Request,
    files: list[UploadFile] = File(...),
    name: str | None = None,
    provider_hint: str | None = None,
):
    """Import a directory selection, multiple files, or ZIP archive."""
    from theriad.memory.importers import ImportError_, import_uploaded_files

    try:
        uploaded = [(file.filename or "upload", await file.read()) for file in files]
        fmt, entries, skipped = import_uploaded_files(uploaded, provider_hint)
        return _memory_import_response(
            fmt,
            entries,
            skipped,
            _memory_store_for_request(request),
            suggested_name=_suggest_memory_upload_name(
                [filename for filename, _ in uploaded], name or provider_hint, fmt,
            ),
        )
    except ImportError_ as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})


@app.get("/api/memory/graph")
def memory_graph(request: Request):
    from theriad.memory.graph import build_memory_graph

    return build_memory_graph(_memory_store_for_request(request).list())


@app.get("/api/memory/providers")
def memory_providers():
    """Per-provider memory capacities retained for older clients."""
    from theriad.memory.providers import DEFAULT_INJECT_BUDGET_CHARS, providers_as_dicts

    return {
        "providers": providers_as_dicts(),
        "default_inject_budget_chars": DEFAULT_INJECT_BUDGET_CHARS,
    }
