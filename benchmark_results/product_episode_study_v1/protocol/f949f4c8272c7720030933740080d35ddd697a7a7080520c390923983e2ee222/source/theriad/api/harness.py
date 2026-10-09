"""
Harness Integration API — OpenAI-compatible /v1 endpoint.

Allows any LLM harness (Claude Code, LangChain, CI pipelines, test runners, etc.)
to use Theriad's cost-optimized routing as a drop-in OpenAI replacement:

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="theriad-local")
    response = client.chat.completions.create(
        model="theriad",
        messages=[{"role": "user", "content": "Summarize this log…"}]
    )

Supported model values
───────────────────────
"theriad" / "theriad"        Auto-route to cheapest capable model (balanced).
"theriad/fast"               Speed-preferred routing (speed_preference=80).
"theriad/quality"            Quality-preferred routing (quality_preference=80).
"theriad/free"               Free-models-only pool (free_moe_experimental policy).
"<tier-name>"               Pin to a Theriad tier, e.g. "cheap_code", "code_strong".
"<provider/model>"          Use a specific model slug via OpenRouter, e.g. "qwen/qwen3-14b".

Extension request fields (optional, accepted alongside any model value)
───────────────────────
x_theriad_policy     Override routing policy: "balanced" | "aggressive_savings" |
                    "quality_first" | "free_moe_experimental".
x_theriad_speed      Speed preference 0–100 (overrides virtual model default).
x_theriad_quality    Quality preference 0–100 (overrides virtual model default).

Auth
────
Authorization: Bearer <api-key>   — used as the OpenRouter key for this request.
If omitted, the key configured in the current request's Settings scope is
used (the legacy ~/.theriad/secrets.json file for anonymous local use).
When running locally any non-empty placeholder such as "theriad-local" works
if a key is already configured in Settings.

Response includes an "x_theriad" extension object with routing metadata for
observability (tier, model chosen, routing layer that fired, estimated cost).
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from theriad.api.auth import user_scope_id
from theriad.capabilities.detector import CapabilityDetector
from theriad.capabilities.rendering import (
    renderer_metadata,
    renderer_system_instruction,
    with_renderer_instruction,
)
from theriad.core.providers.interfaces import Message
router = APIRouter(prefix="/v1", tags=["harness"])

# Virtual model names that activate Theriad's routing pipeline
_THERIAD_ROUTING_MODELS: frozenset[str] = frozenset(
    {"theriad", "theriad", "theriad/fast", "theriad/quality", "theriad/free"}
)

# Per-virtual-model routing defaults
_MODEL_SPEED_PREFS: dict[str, int] = {"theriad/fast": 80}
_MODEL_QUALITY_PREFS: dict[str, int] = {"theriad/quality": 80}
_MODEL_POLICIES: dict[str, str] = {"theriad/free": "free_moe_experimental"}


# ── Request / response schemas ────────────────────────────────────────────────

class _ChatMessage(BaseModel):
    role: str
    content: str | list[Any]

    def as_text(self) -> str:
        """Return content as a plain string, handling content-block arrays."""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for block in self.content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)


class _HarnessChatRequest(BaseModel):
    model: str = "theriad"
    messages: list[_ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    # Theriad routing extension fields
    x_theriad_policy: str | None = None
    x_theriad_speed: int | None = Field(default=None, ge=0, le=100)
    x_theriad_quality: int | None = Field(default=None, ge=0, le=100)

    model_config = {"extra": "allow"}  # tolerate unknown OpenAI fields


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_api_key(authorization: str | None, request: Request) -> str:
    """Return the provider API key: bearer token if present, else configured key."""
    # In hosted mode Authorization is a Theriad session, never an OpenRouter
    # credential. Only local/dev clients may use a provider-key bearer token.
    from theriad.settings.deployment import is_hosted_mode
    if authorization and not is_hosted_mode():
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
            if token:
                return token
    from theriad.settings.secrets import get_openrouter_api_key
    return get_openrouter_api_key() or ""


def _routing_prompt(messages: list[_ChatMessage]) -> str:
    """Extract the last user message text for routing classification."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.as_text()
    return messages[-1].as_text() if messages else ""


def _is_valid_tier(name: str) -> bool:
    from theriad.models.tiers import VALID_TIERS
    return name in VALID_TIERS


def _openai_completion_response(
    *,
    completion_id: str,
    model: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str | None,
    x_theriad: dict,
) -> dict:
    return {
        "id": f"theriad-{completion_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "x_theriad": x_theriad,
    }


# ── GET /v1/models ────────────────────────────────────────────────────────────

@router.get("/models")
def list_models():
    """
    List Theriad virtual model IDs and tier names in OpenAI model-list format.

    Returns the full set of valid model values for /v1/chat/completions so that
    harnesses can discover what to pass in the model field.
    """
    from theriad.models.tiers import VALID_TIERS

    now = int(time.time())
    models: list[dict] = []

    for name in ["theriad", "theriad", "theriad/fast", "theriad/quality", "theriad/free"]:
        models.append({"id": name, "object": "model", "created": now, "owned_by": "theriad"})

    for tier in sorted(VALID_TIERS):
        models.append({"id": tier, "object": "model", "created": now, "owned_by": "theriad"})

    return {"object": "list", "data": models}


# ── POST /v1/chat/completions ─────────────────────────────────────────────────

@router.post("/chat/completions")
async def chat_completions(
    body: _HarnessChatRequest,
    request: Request,
    authorization: str | None = Header(default=None),
):
    """
    OpenAI-compatible chat completions backed by Theriad cost-optimized routing.

    The cheapest capable model is selected automatically unless you pin a tier
    or model slug. The x_theriad field in the response shows exactly what was
    chosen and why.
    """
    import time as _time
    from theriad.core.providers.interfaces import CompletionRequest, ProviderError
    from theriad.core.providers.factory import build_provider

    # ── Streaming not yet supported ──────────────────────────────────────────
    if body.stream:
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": (
                        "Streaming is not yet supported by the Theriad harness API. "
                        "Set stream=false (the default) to get a standard response."
                    ),
                    "type": "not_implemented",
                    "code": 501,
                }
            },
        )

    if not body.messages:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": "messages must not be empty",
                    "type": "invalid_request_error",
                    "code": 422,
                }
            },
        )

    # ── API key ───────────────────────────────────────────────────────────────
    # Local OpenAI-compatible clients conventionally put the upstream
    # provider key in Authorization. Do not reject that token as an invalid
    # Theriad session; only hosted mode treats every bearer as a Theriad
    # session. A valid local Theriad session still receives tenant-scoped runs.
    from theriad.settings.deployment import is_hosted_mode
    if is_hosted_mode():
        scope_id = user_scope_id(request)
    else:
        from theriad.api.auth import get_request_user
        local_user = get_request_user(request)
        scope_id = local_user["id"] if local_user else None
    api_key = _resolve_api_key(authorization, request)
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": (
                        "No API key found. Pass 'Authorization: Bearer <key>' or "
                        "configure a key in Theriad Settings."
                    ),
                    "type": "authentication_error",
                    "code": 401,
                }
            },
        )

    # ── Normalise model name + routing parameters ─────────────────────────────
    model_name = (body.model or "theriad").strip().lower()
    policy = body.x_theriad_policy or _MODEL_POLICIES.get(model_name, "balanced")
    speed_pref = (
        body.x_theriad_speed
        if body.x_theriad_speed is not None
        else _MODEL_SPEED_PREFS.get(model_name)
    )
    quality_pref = (
        body.x_theriad_quality
        if body.x_theriad_quality is not None
        else _MODEL_QUALITY_PREFS.get(model_name)
    )

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
            is_free_request=(
                model_name == "theriad/free" or policy == "free_moe_experimental"
            ),
            provider_key_configured=bool(api_key),
        )
        if denial:
            status_code = 503 if denial["code"] == "provider_key_required" else 403
            return JSONResponse(
                status_code=status_code,
                content={
                    "error": {
                        "message": denial["message"],
                        "type": "entitlement_error",
                        "code": denial["code"],
                    },
                    "entitlements": entitlements,
                },
            )

    routing_prompt = _routing_prompt(body.messages)
    total_start = _time.monotonic()

    # Convert to Theriad messages
    theriad_messages = [Message(role=m.role, content=m.as_text()) for m in body.messages]
    capability = CapabilityDetector().detect(routing_prompt)
    rendering = renderer_metadata(capability)
    theriad_messages = with_renderer_instruction(
        theriad_messages, renderer_system_instruction(capability)
    )

    # ── Routing: Theriad pipeline vs. direct model slug ────────────────────────
    use_theriad_routing = model_name in _THERIAD_ROUTING_MODELS or _is_valid_tier(model_name)

    if use_theriad_routing:
        from theriad.layers.layer0.layer import build_default_layer
        from theriad.layers.layer1_features.layer import FeatureRoutingLayer
        from theriad.core.routing.interfaces import RoutingDecision, RoutingResult
        from theriad.models.policy import PolicyResolver
        from theriad.models.preferences import RoutingPreferences

        if _is_valid_tier(model_name):
            # Pinned tier — skip routing layers
            tier = model_name
            routing_result = RoutingResult(
                decision=RoutingDecision(
                    tier=tier, confidence=1.0, reason="tier_pinned_by_caller"
                ),
                layer_name="pinned",
            )
        else:
            layer0 = build_default_layer()
            layer1 = FeatureRoutingLayer()
            ctx: dict = {}
            routing_result = layer0.route(routing_prompt, ctx) or layer1.route(routing_prompt, ctx)
            if routing_result is None:
                routing_result = RoutingResult(
                    decision=RoutingDecision(
                        tier="cheap_bulk",
                        confidence=0.0,
                        reason="all_layers_abstained",
                    ),
                    layer_name="fallback",
                )
            tier = routing_result.decision.tier

        preferences = RoutingPreferences.resolve(
            speed=speed_pref, quality=quality_pref, legacy_policy=policy
        )
        resolution = PolicyResolver().resolve(
            tier,
            policy,
            routing_result.decision.confidence,
            message=routing_prompt,
            speed_preference=preferences.speed,
            quality_preference=preferences.quality,
            task_type=routing_result.decision.signals.get("task_type"),
        )

        provider_name = resolution.provider
        model_slug = resolution.model
        fallback_candidates: list[tuple[str, str]] = [
            (resolution.provider, resolution.model),
            *resolution.fallback_tuples(),
        ]
        x_theriad_meta: dict = {
            "tier": resolution.tier,
            "provider": provider_name,
            "model": model_slug,
            "routing_layer": routing_result.layer_name,
            "routing_reason": routing_result.decision.reason,
            "confidence": round(routing_result.decision.confidence, 3),
            "policy": policy,
        }
    else:
        # Direct model slug (e.g. "qwen/qwen3-14b") → OpenRouter pass-through
        provider_name = "openrouter"
        model_slug = model_name
        fallback_candidates = [("openrouter", model_slug)]
        x_theriad_meta = {
            "tier": "direct",
            "provider": provider_name,
            "model": model_slug,
            "routing_layer": "direct",
            "routing_reason": "direct_model_specified_by_caller",
            "confidence": 1.0,
            "policy": "none",
        }

    if rendering:
        x_theriad_meta["renderer"] = rendering

    # ── Determine effective max_tokens ────────────────────────────────────────
    from theriad.models.tiers import TIER_MAX_TOKENS
    tier_key = x_theriad_meta.get("tier", "cheap_bulk")
    tier_max = TIER_MAX_TOKENS.get(tier_key)
    if body.max_tokens is not None:
        max_tokens: int | None = (
            min(body.max_tokens, tier_max) if tier_max is not None else body.max_tokens
        )
    else:
        max_tokens = tier_max

    # ── Execute with fallback chain ───────────────────────────────────────────
    from theriad.accounting.pricing import PricingTable
    from theriad.agents.model_telemetry import complete_with_telemetry, usage_totals

    attempt_usage: list[dict] = []
    attempt_pricing = PricingTable()
    response = None
    last_error: Exception | None = None

    for attempt_provider, attempt_model in fallback_candidates:
        try:
            provider = build_provider(attempt_provider, api_key=api_key, timeout=30.0)
            req = CompletionRequest(
                model=attempt_model,
                messages=theriad_messages,
                max_tokens=max_tokens,
            )
            candidate = await complete_with_telemetry(
                provider, req, role="harness_completion", usage_sink=attempt_usage,
                pricing_table=attempt_pricing,
            )
            if not candidate.content.strip():
                last_error = Exception(
                    f"Empty response from {attempt_model} (finish_reason={candidate.finish_reason!r})"
                )
                continue
            x_theriad_meta["model"] = candidate.model
            x_theriad_meta["provider"] = candidate.provider
            response = candidate
            break
        except ProviderError as exc:
            last_error = exc
            # Non-retryable 4xx (not 429): stop trying fallbacks
            if exc.status_code and 400 <= exc.status_code < 500 and exc.status_code != 429:
                break
        except Exception as exc:
            last_error = exc

    actual_cost, cost_known = usage_totals(attempt_usage)
    attempt_input_tokens = sum(entry.get("prompt_tokens") or 0 for entry in attempt_usage)
    attempt_output_tokens = sum(entry.get("completion_tokens") or 0 for entry in attempt_usage)
    x_theriad_meta.update({
        "estimated_cost_usd": actual_cost,
        "actual_cost_usd": actual_cost,
        "cost_known": cost_known,
        "known_attempt_cost_usd": round(sum(entry.get("cost_usd") or 0.0 for entry in attempt_usage), 8),
        "attempt_input_tokens": attempt_input_tokens,
        "attempt_output_tokens": attempt_output_tokens,
        "attempt_usage": attempt_usage,
    })

    if response is None:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"All providers exhausted. Last error: {last_error}",
                    "type": "provider_error",
                    "code": 502,
                },
                "x_theriad": x_theriad_meta,
            },
        )

    # ── Record cost and latency ───────────────────────────────────────────────
    total_latency_ms = (_time.monotonic() - total_start) * 1000
    x_theriad_meta["latency_ms"] = round(total_latency_ms, 1)
    try:
        from theriad.accounting.tracker import CostTracker

        if scope_id is None:
            CostTracker(metrics_path="~/.theriad/metrics.jsonl").record(
                provider=response.provider,
                model=response.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=response.latency_ms,
                routing_layer=x_theriad_meta["routing_layer"],
                routing_reason=routing_prompt[:200],
                tags={"policy": policy, "source": "harness_api"},
            )
    except Exception:
        pass
    try:
        from theriad.api.projects import save_run
        harness_run_id = save_run(
            project_id=None,
            conversation_id=None,
            message_id=None,
            model=response.model,
            tier=x_theriad_meta.get("tier"),
            provider=response.provider,
            policy=policy,
            input_tokens=attempt_input_tokens,
            output_tokens=attempt_output_tokens,
            actual_cost_usd=actual_cost,
            routing_latency_ms=None,
            provider_latency_ms=response.latency_ms,
            total_latency_ms=total_latency_ms,
            fallback_used=response.model != model_slug,
            status="success",
            routing_source=x_theriad_meta.get("routing_layer"),
            routing_reason=x_theriad_meta.get("routing_reason"),
            fallback_chain=[model for _, model in fallback_candidates],
            user_id=scope_id,
        )
        from theriad.api.server import _record_routing_usage
        _record_routing_usage(
            request,
            run_id=harness_run_id,
            provider=response.provider,
            model=response.model,
            input_tokens=attempt_input_tokens,
            output_tokens=attempt_output_tokens,
            provider_cost_usd=actual_cost,
            paid_request=not (
                model_name == "theriad/free" or policy == "free_moe_experimental"
            ),
            usage_known=all(entry.get("usage_available", False) for entry in attempt_usage),
        )
    except Exception:
        pass

    return _openai_completion_response(
        completion_id=uuid.uuid4().hex[:16],
        model=response.model,
        content=response.content,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        finish_reason=response.finish_reason,
        x_theriad=x_theriad_meta,
    )
