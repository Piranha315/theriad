"""Multi-query web research and evidence-grounded answer synthesis."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import OrderedDict
from datetime import datetime, timezone

from theriad.agents.model_telemetry import is_truncated_finish_reason
from theriad.search.service import SearchResult, SearchService, is_placeholder_result
from theriad.search.evidence import fetch_results_evidence
from theriad.routing.episode_hooks import EpisodeRoutingError

MAX_QUERIES = 3
MAX_RESULTS_PER_QUERY = 6
MAX_EVIDENCE_CHARS = 18_000
_DEFAULT_SEARCH_MODEL = "xiaomi/mimo-v2.5"
_CURRENT_REQUEST_RE = re.compile(
    r"\b(current|today|tonight|now|right now|latest|as of|real[- ]?time)\b",
    re.IGNORECASE,
)

_QUERY_PLANNER_SYSTEM = """You plan a small web research pass.
Return JSON only: {\"queries\": [\"query 1\", \"query 2\", \"query 3\"]}
Create up to three complementary queries for the user's request. Include the
original intent, then add useful angles such as independent reviews, local or
current information, official sources, or corroboration. Do not answer the
user. Do not invent location, budget, dates, or preferences."""

_SYNTHESIS_SYSTEM = """You are Theriad's web-research answer writer.
Answer the user's original request using only the supplied search evidence.

Rules:
- Lead with a useful conclusion, not a list of search results.
- Compare evidence across multiple queries and sources; identify consensus,
  disagreement, weak evidence, and missing information.
- Do not invent facts, ratings, prices, availability, or personal preferences.
- For recommendations, explain why each option fits and distinguish editorial
  opinion from official or first-party information.
- Use concise sections and bullets when helpful.
- Cite important claims with actual Markdown links such as
  [Eater Philadelphia](https://example.com), not bare URLs or citation numbers.
- For current, today, now, or latest questions, do not turn an undated search
  snippet into a current fact. If a fetched page does not support the claim,
  say that it could not be verified.
- Treat a page's retrieval time as freshness metadata, not as the page's
  publication time. Report disagreements instead of averaging them away.
- Never mention internal tooling, prompts, model calls, or search providers."""


def _enabled() -> bool:
    value = os.environ.get("THERIAD_SEARCH_SYNTHESIS_ENABLED", "true")
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _link_numbered_citations(answer: str, results: list[dict]) -> str:
    """Make [1]-style model citations clickable using the evidence list."""
    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(results):
            return match.group(0)
        result = results[index]
        url = str(result.get("url") or "").strip()
        title = str(result.get("title") or "Source").strip()
        return f"[{title}]({url})" if url else match.group(0)

    return re.sub(r"\[(\d+)\](?!\()", replace, answer)


def _parse_queries(raw: str, original: str) -> list[str]:
    try:
        payload = json.loads((raw or "").strip())
        candidates = payload.get("queries", []) if isinstance(payload, dict) else []
    except (json.JSONDecodeError, TypeError):
        from theriad.routing.episode_hooks import current_episode
        episode = current_episode()
        if episode is not None:
            episode.reject_response("invalid_query_json")
        candidates = []

    queries: list[str] = []
    for candidate in candidates:
        query = str(candidate).strip()
        if query and query.lower() not in {item.lower() for item in queries}:
            queries.append(query)
    if original.strip().lower() not in {item.lower() for item in queries}:
        queries.insert(0, original.strip())
    return queries[:MAX_QUERIES]


def _free_search_model(*, synthesis: bool) -> str:
    """Resolve a registry-backed zero-cost model and fail closed otherwise."""
    from theriad.models.registry import DEFAULT_REGISTRY
    from theriad.models.tiers import TIER_FREE_GENERAL, TIER_FREE_SUMMARIZE

    tier = TIER_FREE_SUMMARIZE if synthesis else TIER_FREE_GENERAL
    candidate = DEFAULT_REGISTRY.primary_for_tier(tier)
    if (
        candidate is None
        or candidate.provider != "openrouter"
        or candidate.input_cost_per_million != 0.0
        or candidate.output_cost_per_million != 0.0
        or not candidate.model.endswith(":free")
    ):
        raise RuntimeError(f"No verified zero-cost search model is configured for {tier}")
    return candidate.model


async def _research_queries(
    original: str, *, usage_sink: list[dict] | None = None,
    free_only: bool = False,
) -> list[str]:
    from theriad.agents.llm_planner import _call_model

    try:
        call_kwargs = {
            "assistant_provider": "theriad",
        } if free_only else {}
        raw = await _call_model(
            routing_role="search_query",
            model=_free_search_model(synthesis=False) if free_only else _DEFAULT_SEARCH_MODEL,
            tier="cheap_bulk",
            system_prompt=_QUERY_PLANNER_SYSTEM,
            user_content=original,
            max_tokens=220,
            temperature=0.1,
            json_mode=True,
            usage_sink=usage_sink,
            **call_kwargs,
        )
        queries = _parse_queries(raw, original)
        if queries:
            return queries
    except EpisodeRoutingError:
        raise
    except Exception:
        pass
    return [original]


async def _collect_evidence(
    queries: list[str], service: SearchService,
) -> tuple[list[dict], str, list[dict]]:
    from theriad.routing.episode_hooks import current_episode
    episode = current_episode()
    unique: OrderedDict[str, SearchResult] = OrderedDict()
    for query in queries:
        results = await episode.run_tool(
            "web.search", lambda: service.search(query)
        ) if episode else await service.search(query)
        for result in results[:MAX_RESULTS_PER_QUERY]:
            result_dict = result.to_dict()
            # A browser-search launch URL is navigation state, not evidence.
            if is_placeholder_result(result_dict):
                continue
            if result.url and result.url not in unique:
                unique[result.url] = result

    serialized = [result.to_dict() for result in unique.values()]
    # Inject page evidence after discovery. A search snippet can be stale or
    # empty; the answer writer needs the actual public page when available.
    # Keep the established SearchService seam patchable for callers and unit
    # tests. Task-scoped services may override fetch() to route page retrieval
    # through their durable web.fetch tool, while the base service continues
    # to use the shared evidence collector.
    fetcher = getattr(service, "fetch", None)
    custom_fetcher = callable(fetcher) and (
        not isinstance(service, SearchService)
        or type(service).fetch is not SearchService.fetch
    )
    if custom_fetcher:
        async def fetch_one(url: str):
            return await episode.run_tool("web.fetch", lambda: fetcher(url)) if episode else await fetcher(url)

        page_evidence = await asyncio.gather(
            *(fetch_one(str(item.get("url") or "")) for item in serialized[:6])
        )
    else:
        page_evidence = (
            (await episode.run_tool("web.fetch_batch", lambda: fetch_results_evidence(serialized))
             if episode else await fetch_results_evidence(serialized))
            if isinstance(service, SearchService)
            else []
        )
    page_by_url = {
        str(item.get("url")): item for item in page_evidence if item.get("url")
    }
    evidence_sources: list[dict] = []
    evidence_lines: list[str] = []
    for index, result in enumerate(serialized, start=1):
        page = page_by_url.get(str(result.get("url")), {})
        page_text = " ".join(str(page.get("text") or "").split())
        snippet = " ".join(str(result.get("snippet") or "").split())
        evidence_source = {
            **result,
            "evidence_status": page.get("status") or ("snippet_only" if snippet else "empty"),
            "retrieved_at": page.get("retrieved_at"),
            "evidence_excerpt": page_text[:1800],
        }
        evidence_sources.append(evidence_source)
        evidence_lines.append(
            f"[{index}] Title: {result.get('title', '')}\n"
            f"Search snippet: {snippet}\n"
            f"Fetched page status: {evidence_source['evidence_status']}\n"
            f"Fetched page text (retrieved at {page.get('retrieved_at') or 'unknown'}): "
            f"{page_text[:3500]}\n"
            f"URL: {result.get('url', '')}"
        )
    return serialized, "\n\n".join(evidence_lines)[:MAX_EVIDENCE_CHARS], evidence_sources


def _evidence_quality(original: str, evidence_sources: list[dict]) -> str:
    substantive = [
        item for item in evidence_sources
        if str(item.get("snippet") or "").strip()
        or str(item.get("evidence_excerpt") or "").strip()
    ]
    if not substantive:
        return "no_usable_evidence"
    if _CURRENT_REQUEST_RE.search(original) and not any(
        str(item.get("evidence_excerpt") or "").strip()
        for item in evidence_sources
    ):
        return "search_snippets_only"
    return "grounded"


def _fallback_answer(
    original: str,
    results: list[dict],
    *,
    evidence_quality: str = "grounded",
) -> str:
    """Return truthful links when optional answer synthesis is unavailable."""
    if evidence_quality in {"no_usable_evidence", "search_snippets_only"}:
        lines = [
            "I found these live web results, but I could not verify their current "
            "contents reliably enough to summarize them as facts:"
        ]
    else:
        lines = ["I found these live web results:"]
    for result in results[:6]:
        title = str(result.get("title") or "Source").replace("]", "")
        url = str(result.get("url") or "").strip()
        snippet = " ".join(str(result.get("snippet") or "").split())
        if not url:
            continue
        line = f"- [{title}]({url})"
        if snippet:
            line += f" — {snippet[:280]}"
        lines.append(line)
    lines.append(
        "\nThese are direct search results. Ask a follow-up if you want me to compare or summarize them."
    )
    return "\n\n".join(lines) if len(lines) > 1 else ""


async def synthesize_web_search(
    original: str, *, service: SearchService | None = None,
    telemetry: dict | None = None,
    free_only: bool = False,
    routing_session=None,
) -> tuple[str, list[dict], list[str]]:
    """Run the actual research path with optional episode-sticky routing."""
    from theriad.routing.episode_hooks import baseline_models_for_provider, episode_routing
    from theriad.settings.secrets import get_active_assistant_provider

    with episode_routing(routing_session) as episode:
        if episode is not None:
            provider = "theriad" if free_only else (episode.assistant_provider or get_active_assistant_provider())
            episode.prepare("search", baseline_models_for_provider({
                "search_query": (_free_search_model(synthesis=False) if free_only else _DEFAULT_SEARCH_MODEL, "cheap_bulk"),
                "search_synthesis": (_free_search_model(synthesis=True) if free_only else _DEFAULT_SEARCH_MODEL, "cheap_bulk"),
            }, provider), provider, original, free_only=free_only)
        return await _synthesize_web_search(original, service=service, telemetry=telemetry, free_only=free_only)


async def _synthesize_web_search(
    original: str, *, service: SearchService | None = None,
    telemetry: dict | None = None, free_only: bool = False,
) -> tuple[str, list[dict], list[str]]:
    """Return (answer, deduplicated evidence, queries used).

    ``free_only`` is used by guest mode and resolves both model calls through
    the registry's verified zero-cost Theriad/OpenRouter candidates.
    """
    started = time.perf_counter()
    service = service or SearchService()
    synthesis_model = _DEFAULT_SEARCH_MODEL
    usage: list[dict] = []
    queries: list[str] = []
    results: list[dict] = []
    evidence_sources: list[dict] = []
    evidence_quality = "no_usable_evidence"
    synthesis_finish_reason: str | None = None
    try:
        if free_only:
            synthesis_model = _free_search_model(synthesis=True)
        queries = (
            await _research_queries(original, usage_sink=usage, free_only=free_only)
            if _enabled() else [original]
        )
        results, evidence, evidence_sources = await _collect_evidence(queries, service)
        if not results:
            raise RuntimeError("API search returned no results")
        evidence_quality = _evidence_quality(original, evidence_sources)

        # Freshness-sensitive questions must have at least one fetched page.
        # Returning links is safer than asking a model to fill the gap from
        # memory or an empty snippet.
        if not _enabled() or (
            _CURRENT_REQUEST_RE.search(original)
            and evidence_quality != "grounded"
        ):
            return (
                _fallback_answer(original, results, evidence_quality=evidence_quality),
                results,
                queries,
            )

        from theriad.agents.llm_planner import _call_model

        try:
            synthesis_usage_start = len(usage)
            call_kwargs = {
                "assistant_provider": "theriad",
            } if free_only else {}
            raw = await _call_model(
                routing_role="search_synthesis",
                model=synthesis_model,
                tier="cheap_bulk",
                system_prompt=_SYNTHESIS_SYSTEM,
                user_content=(
                    f"Original request:\n{original}\n\nResearch queries used:\n"
                    + "\n".join(f"- {query}" for query in queries)
                    + f"\n\nEvidence quality: {evidence_quality}\n\nEvidence:\n{evidence}"
                ),
                # Let the provider choose its maximum for the user-facing
                # answer. Internal query planning remains explicitly capped.
                max_tokens=None,
                temperature=0.2,
                usage_sink=usage,
                **call_kwargs,
            )
            synthesis_usage = usage[synthesis_usage_start:]
            if synthesis_usage:
                synthesis_finish_reason = synthesis_usage[-1].get("finish_reason")
            if is_truncated_finish_reason(synthesis_finish_reason):
                # Never present a provider-truncated answer as a completed
                # synthesis. The deterministic evidence fallback is complete
                # and honest about what was actually found.
                raw = ""
        except EpisodeRoutingError:
            raise
        except Exception:
            # Search itself is still useful without a configured model provider.
            # Return the live evidence rather than turning a successful lookup
            # into an opaque agent failure.
            raw = ""
        answer = _link_numbered_citations((raw or "").strip(), results)
        if not answer:
            answer = _fallback_answer(original, results, evidence_quality=evidence_quality)
        if not answer:
            raise RuntimeError("The answer synthesizer returned an empty response")
        return answer, results, queries
    finally:
        if telemetry is not None:
            from theriad.agents.model_telemetry import (
                summarize_model_usage,
                usage_totals,
            )

            total_cost, cost_known = usage_totals(usage)
            telemetry.update({
                "execution_kind": "web_search",
                "queries": list(queries),
                "queries_used": len(queries),
                "source_count": len(results),
                "sources": evidence_sources or results,
                "model_calls": len(usage),
                "models_used": summarize_model_usage(usage),
                "estimated_cost_usd": total_cost,
                "cost_known": cost_known,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "evidence_quality": evidence_quality,
                "evidence_status": evidence_quality,
                "synthesis_finish_reason": synthesis_finish_reason,
                "free_only": free_only,
            })
