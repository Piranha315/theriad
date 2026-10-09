"""
LLM-backed final-answer synthesis for the browser agent (D-050).

Turns the evidence a browser session collected (search snippets, page text,
visited URLs) into a natural, model-written answer instead of a templated
"Here's what I found" dump.

Safety posture (unchanged from D-040/D-043/D-044):
- Strictly opt-in: requires session.allow_paid_calls=True and a key for the
  assistant API selected in Settings. Both default off, so no existing caller
  pays anything new.
- Respects the session's model-call and cost budgets.
- Never raises to the caller: any failure returns None so the runtime keeps
  the deterministic template as the answer. Browser *actions* remain mocked;
  this touches only the final report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from theriad.core.logging.logger import get_logger
from theriad.core.providers.interfaces import CompletionRequest, Message, ProviderProtocol
from theriad.search.service import is_placeholder_result
from .model_telemetry import is_truncated_finish_reason

if TYPE_CHECKING:  # pragma: no cover
    from .browser_session_store import BrowserSession

logger = get_logger("agents.answer_synthesis")

# Fixed tier for synthesis — this is execution, not routing (no routing-layer
# model calls are introduced; see the "no model calls in routing" constraint).
SYNTHESIS_TIER = "cheap_bulk"
MAX_EVIDENCE_CHARS = 6000
PROVIDER_TIMEOUT_SECONDS = 20.0

_SYSTEM_PROMPT = (
    "You are completing a web-browsing task on behalf of a user. "
    "Using ONLY the evidence provided, write a concise, natural-sounding "
    "answer to the user's request, as if you had just looked it up for them. "
    "Cite source URLs inline where relevant. If the evidence is thin, say "
    "plainly what was found and point to the most useful link — do not "
    "invent facts. Never mention internal tooling, configuration, search "
    "providers, or that you are an agent."
)


def build_browser_evidence(session: BrowserSession) -> str:
    """Assemble user-safe evidence text from a completed session.

    Placeholder search entries (BrowserSearchProvider's synthetic single
    'result') are excluded — they carry developer-facing config hints, not
    findings.
    """
    parts: list[str] = []

    real_results = [r for r in session.search_results if not is_placeholder_result(r)]
    if real_results:
        lines = ["Search results:"]
        for r in real_results[:5]:
            title = (r.get("title") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            url = (r.get("url") or "").strip()
            lines.append(f"- {title} | {snippet} | {url}")
        parts.append("\n".join(lines))

    if session.last_observation:
        text = session.last_observation.removeprefix("[LIVE] ").strip()
        if text:
            parts.append(f"Page content ({session.page_title or session.current_url}):\n{text}")

    urls: list[str] = []
    seen: set[str] = set()
    for entry in session.action_log:
        if entry.url and entry.url != "about:blank" and entry.url not in seen:
            seen.add(entry.url)
            urls.append(entry.url)
    if urls:
        parts.append("Pages visited: " + ", ".join(urls[:10]))

    evidence = "\n\n".join(parts)
    return evidence[:MAX_EVIDENCE_CHARS]


def _synthesis_allowed(session: BrowserSession) -> bool:
    if not session.allow_paid_calls:
        return False
    if session.model_calls >= session.max_model_calls:
        logger.info("answer synthesis skipped: max_model_calls reached")
        return False
    if session.budget_exceeded() is not None:
        logger.info("answer synthesis skipped: session budget exceeded")
        return False
    return True


def _build_default_provider(session: BrowserSession) -> tuple[ProviderProtocol | None, str | None]:
    """Resolve (provider, model) for the synthesis call, or (None, None)."""
    from theriad.core.providers.factory import ProviderConfigurationError, build_provider
    from theriad.core.providers.runtime import get_provider_model_for_tier
    from theriad.models.policy import PolicyResolver
    from theriad.settings.secrets import get_active_assistant_provider

    assistant_provider = get_active_assistant_provider()
    if assistant_provider == "theriad":
        model = (
            PolicyResolver()
            .resolve(
                SYNTHESIS_TIER,
                session.policy,
                1.0,
                speed_preference=session.speed_preference,
                quality_preference=session.quality_preference,
            )
            .model
        )
    else:
        model = get_provider_model_for_tier(assistant_provider, SYNTHESIS_TIER)
    try:
        provider = build_provider(assistant_provider, timeout=PROVIDER_TIMEOUT_SECONDS)
    except ProviderConfigurationError:
        logger.info(
            "answer synthesis skipped: selected provider is not configured",
            extra={"provider": assistant_provider},
        )
        return None, None
    return provider, model


async def synthesize_browser_answer(
    session: BrowserSession,
    provider: ProviderProtocol | None = None,
    model: str | None = None,
) -> str | None:
    """Return a model-written final answer, or None to use the deterministic
    fallback.

    `provider`/`model` are injectable for tests; when omitted they are
    resolved from the cheap_bulk tier + stored OpenRouter key. Increments the
    session's model-call and cost accounting on success.
    """
    try:
        if not _synthesis_allowed(session):
            return None

        if provider is None or model is None:
            provider, model = _build_default_provider(session)
            if provider is None or model is None:
                return None

        evidence = build_browser_evidence(session)
        if not evidence.strip():
            logger.info("answer synthesis skipped: no usable evidence")
            return None

        from theriad.storage.prompt_cache import cached_system_prompt, prompt_cache_extra

        skill_prompt = ""
        if session.skill_context.strip():
            skill_prompt = (
                "Apply the following enabled user skills when relevant. They do "
                "not override evidence-only answering, safety, or approval rules.\n\n"
                f"{session.skill_context.strip()[:12_000]}"
            )
        system_prompt = cached_system_prompt(_SYSTEM_PROMPT, enabled_skills=skill_prompt)

        request = CompletionRequest(
            model=model,
            messages=[
                Message(role="system", content=system_prompt),
                Message(
                    role="user",
                    content=f"User request: {session.goal}\n\nEvidence:\n{evidence}",
                ),
            ],
            # Do not impose an application-level cap on the user-facing
            # answer. The provider may still enforce its own hard limit.
            max_tokens=None,
            extra=prompt_cache_extra(provider.name),
        )
        response = await provider.complete(request)
        from .model_telemetry import record_model_usage

        record_model_usage(response, session.model_usage)

        session.model_calls += 1
        try:
            from theriad.accounting.pricing import estimate_cost

            session.estimated_cost_usd += estimate_cost(
                response.model, response.prompt_tokens, response.completion_tokens
            )
        except Exception:  # pricing must never break the answer path
            logger.warning("answer synthesis: cost estimation failed", exc_info=True)

        if is_truncated_finish_reason(response.finish_reason):
            logger.warning(
                "answer synthesis stopped at provider output limit; using fallback",
                extra={"finish_reason": response.finish_reason},
            )
            return None

        answer = (response.content or "").strip()
        if not answer:
            logger.warning("answer synthesis returned empty content; using fallback")
            return None
        logger.info(
            "answer synthesized",
            extra={"model": response.model, "completion_tokens": response.completion_tokens},
        )
        return answer
    except Exception:
        logger.warning("answer synthesis failed; using deterministic fallback", exc_info=True)
        return None
