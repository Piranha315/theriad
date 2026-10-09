"""Real product adapters for resettable benchmark episodes.

Search uses the production planner/synthesizer against a frozen evidence
service. Tool and agent tasks use the real TaskEngine in a fresh directory.
These are fixture executions, explicitly separate from live user-route smoke.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from theriad.core.providers.interfaces import CompletionRequest, Message

from .episode_hooks import (
    EpisodeRoutingError,
    EpisodeRoutingSession,
    JsonlAttemptSink,
    episode_routing,
)
from .episode_study import Episode, EpisodeOutcome


def _chat_baseline(row: Episode, provider: str) -> dict[str, Any]:
    """Freeze the shared chat core on an explicitly isolated public fixture."""
    from theriad.api.server import (
        _assistant_candidates,
        _build_layers,
        _chat_system_prompt,
        _provider_request_extra,
        _tier_limits,
    )
    from theriad.capabilities.detector import CapabilityDetector
    from theriad.capabilities.rendering import renderer_system_instruction
    from theriad.models.policy import PolicyResolver
    from theriad.models.preferences import RoutingPreferences
    from theriad.models.vision import ensure_vision_candidates
    from theriad.routing.context import ContextRequirementEstimator
    from theriad.storage.context import build_context
    from theriad.storage.prompt_cache import prompt_cache_extra

    layer0, layer1 = _build_layers()
    routing = layer0.route(row.prompt, {}) or layer1.route(row.prompt, {})
    tier = routing.decision.tier if routing else "cheap_bulk"
    built = build_context(messages=row.fixture.get("messages", []),
                          file_texts=row.fixture.get("file_texts", []), max_context_tokens=4096)
    context = built.text
    effective = context + "\n\n" + row.prompt if context else row.prompt
    preferences = RoutingPreferences.resolve(speed=None, quality=None, legacy_policy="balanced")
    resolution = PolicyResolver().resolve(tier, "balanced", routing.decision.confidence if routing else 0,
        context_estimate=ContextRequirementEstimator().estimate(row.prompt),
        message=effective, speed_preference=preferences.speed, quality_preference=preferences.quality)
    chain = _assistant_candidates(provider, resolution.tier, resolution)
    if "vision" in row.capabilities:
        chain = ensure_vision_candidates(chain, use_live=False)
    chain = list(dict.fromkeys(chain))
    system = _chat_system_prompt(str(row.fixture.get("project_instructions", "")),
                                 str(row.fixture.get("skill_context", "")),
                                 allow_follow_up_questions=False)
    renderer = renderer_system_instruction(CapabilityDetector().detect(row.prompt))
    if renderer:
        system += "\n\n" + renderer
    max_tokens, timeout = _tier_limits(resolution.tier)
    request_extras = {
        model: {**_provider_request_extra(name, model, preferences, [m for p, m in chain if p == "openrouter"]),
                **prompt_cache_extra(name)} for name, model in chain
    }
    return {"kind": "current", "assistant_provider": provider, "chain": [list(x) for x in chain],
            "tier": resolution.tier, "context_prefix": context, "system_prompt": system,
            "request_extras": request_extras, "max_output_tokens": max_tokens,
            "timeout_seconds": timeout, "preferences": {"speed": preferences.speed, "quality": preferences.quality},
            "context_truncated": built.truncated, "history_summarized": built.history_summarized,
            "objective_verification": True, "adapter": "shared_chat_core_with_fixture_context",
            "isolated_user_features": ["account_and_entitlement_http_gates", "persistent_memory", "enabled_personal_skills", "conversation_continuity", "live_circuit_state"]}


def current_chat_chain(row: Episode) -> list[tuple[str, str]]:
    from theriad.settings.secrets import get_active_assistant_provider
    return [tuple(item) for item in _chat_baseline(row, get_active_assistant_provider())["chain"]]


def freeze_baseline(row: Episode) -> dict[str, Any]:
    from theriad.settings.secrets import get_active_assistant_provider

    from .episode_hooks import baseline_models_for_provider
    provider = get_active_assistant_provider()
    if row.family in {"chat", "coding"}:
        return _chat_baseline(row, provider)
    if row.family == "search":
        from theriad.agents.search_synthesis import _DEFAULT_SEARCH_MODEL, _enabled
        return {"kind": "current", "assistant_provider": provider,
                "models": baseline_models_for_provider({
                    "search_query": (_DEFAULT_SEARCH_MODEL, "cheap_bulk"),
                    "search_synthesis": (_DEFAULT_SEARCH_MODEL, "cheap_bulk"),
                }, provider), "synthesis_enabled": _enabled(),
                "adapter": "search_synthesis_frozen_evidence"}
    if row.family not in {"tools", "agents"}:
        raise ValueError("Unsupported episode family")
    return {"kind": "current", "assistant_provider": provider,
            "models": baseline_models_for_provider({
                "agent_planner": (os.environ.get("THERIAD_AGENT_MODEL", "deepseek/deepseek-v4-flash-0731"), "code_strong"),
            }, provider), "adapter": "task_engine_project_fixture",
            "capabilities": ["project"], "permission_profile": "autonomous"}


def baseline_primary(plan: Mapping[str, Any]) -> str:
    if plan.get("chain"):
        return str(plan["chain"][0][1])
    return next(iter(plan["models"].values()))


def _image_message(spec: Mapping[str, Any]) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new("RGB", (720, 440), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 26) if Path("/System/Library/Fonts/Menlo.ttc").exists() else ImageFont.load_default(size=26)
    draw.multiline_text((32, 28), str(spec["text"]), fill="black", font=font, spacing=14)
    data = io.BytesIO()
    canvas.save(data, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data.getvalue()).decode(), "detail": "low"}}


def _text_grade(row: Episode, text: str) -> tuple[bool | None, dict[str, Any]]:
    from theriad.benchmark.types import BenchmarkPrompt
    grader = row.grader
    kind = grader["kind"]
    if kind == "numeric":
        result = BenchmarkPrompt(id=row.episode_id, category=row.family, prompt=row.prompt,
            answer=str(grader["answer"]), answer_type="exact_match").grade(text)
        return result.passed, {"grade_status": result.status, "reason": result.reason, "grader_version": result.grader_version}
    if kind == "exact":
        return text.strip() == str(grader["answer"]).strip(), {"grader": "strict_exact"}
    if kind == "json":
        try:
            answer = _strict_json(text.strip())
        except (ValueError, TypeError):
            return False, {"grader": "json", "reason": "invalid_json"}
        return _json_equal(answer, grader["answer"]), {"grader": "strict_json_value_and_type_equality"}
    if kind in {"contains", "evidence"}:
        required = grader.get("required_terms", [])
        urls = grader.get("required_urls", [])
        if not required or (kind == "evidence" and not urls):
            return None, {"reason": "missing_frozen_rubric"}
        lower = text.casefold()
        correct = bool(text.strip()) and all(str(term).casefold() in lower for term in required)
        correct = correct and not any(str(term).casefold() in lower for term in grader.get("forbidden_terms", []))
        if kind == "evidence":
            cited = set(re.findall(r"\]\((https?://[^\s)]+)\)", text))
            correct = correct and set(urls).issubset(cited)
            # The product's deterministic fallback lists documents; that is
            # not evidence that the model completed a synthesis task.
            correct = correct and "direct search results" not in lower
        return correct, {"grader": kind, "required_terms": required, "required_urls": urls}
    return None, {"reason": f"unsupported_grader:{kind}"}


def _strict_json(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("nonfinite JSON number")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)


def _json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_json_equal(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_json_equal(a, b) for a, b in zip(actual, expected, strict=True))
    return actual == expected


class FrozenEvidenceService:
    def __init__(self, documents):
        self.documents = documents

    async def search(self, query):
        from theriad.search.service import SearchResult
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        ordered = sorted(self.documents, key=lambda doc: (-len(terms.intersection(re.findall(r"[a-z0-9]+", (doc["title"] + " " + doc["text"]).lower()))), doc["url"]))
        return [SearchResult(doc["title"], doc["url"], doc["text"][:400], "frozen_benchmark_evidence") for doc in ordered]

    async def fetch(self, url):
        for doc in self.documents:
            if doc["url"] == url:
                return {**doc, "status": "ok", "retrieved_at": doc.get("retrieved_at", "2026-09-05T00:00:00Z")}
        raise ValueError("URL is outside the frozen evidence fixture")


def derived_capabilities(metadata: Mapping[str, Any]) -> set[str]:
    """Code *generation* uses the provider's text output capability.

    Execution belongs to the isolated grader, never to model capability
    metadata. Preserve the provider catalog and derive this alias identically
    during candidate eligibility and runtime validation.
    """
    capabilities = set(metadata.get("capabilities", []))
    if "text" in capabilities:
        capabilities.add("code")
    return capabilities


class EpisodeRuntime:
    def __init__(self, *, directory: Path, metadata: Mapping[str, Mapping[str, Any]], baseline_plans: Mapping[str, Mapping[str, Any]], provider_factory=None):
        from theriad.core.providers.factory import build_provider
        self.directory, self.metadata, self.baseline_plans = Path(directory), deepcopy(dict(metadata)), deepcopy(dict(baseline_plans))
        for quote in self.metadata.values():
            quote["capabilities"] = sorted(derived_capabilities(quote))
        self.provider_factory = provider_factory or build_provider
        self._selections: dict[tuple[str, str], list[str]] = {}
        self._selection_latency_ms: dict[tuple[str, str], float] = {}
        self._prediction_details: dict[tuple[str, str], dict[str, Any]] = {}

    def maximum_cost(self, row: Episode, policy: Mapping[str, Any]) -> float:
        models = self.models_for(row, policy)
        if not models:
            raise ValueError("no eligible frozen models")
        maxima = []
        for model in models:
            quote = self.metadata.get(model)
            if quote is None or any(
                isinstance(quote.get(key), bool) or not isinstance(quote.get(key), (int, float))
                or not math.isfinite(quote[key]) or quote[key] < 0
                for key in ("input_per_1m", "output_per_1m")
            ):
                raise ValueError(f"missing frozen price:{model}")
            if quote.get("request_cost_usd", 0) or quote.get("image_cost_usd", 0):
                raise ValueError("per-request/image charges need an explicit bounded reservation adapter")
            maxima.append((row.max_input_tokens * quote["input_per_1m"] + row.max_output_tokens * quote["output_per_1m"]) / 1e6)
        return max(maxima) * row.max_calls

    def max_cost(self, row: Episode, policy: Mapping[str, Any]) -> float:
        return self.maximum_cost(row, policy)

    @staticmethod
    def _family_policy(row: Episode, policy: Mapping[str, Any]) -> Mapping[str, Any]:
        while policy.get("kind") == "by_family":
            policy = policy["policies"][row.family]
        return policy

    def models_for(self, row, policy):
        policy = self._family_policy(row, policy)
        key = (row.content_hash, json.dumps(policy, sort_keys=True))
        if key not in self._selections:
            selection_started = time.perf_counter()
            models = self._select_models(row, policy)
            for model in models:
                metadata = self.metadata.get(model, {})
                if policy.get("kind") != "current" and not set(row.capabilities).issubset(set(metadata.get("capabilities", []))):
                    raise ValueError("challenger lacks frozen capability metadata")
                provider = self.baseline_plans[row.episode_id].get("assistant_provider", "theriad")
                if metadata.get("assistant_provider", provider) != provider:
                    raise ValueError("challenger crosses frozen assistant-provider boundary")
            self._selections[key] = list(dict.fromkeys(models))
            self._selection_latency_ms[key] = (time.perf_counter() - selection_started) * 1000
        return list(self._selections[key])

    def _select_models(self, row, policy):
        if policy.get("kind") == "static":
            return [str(policy["models"][row.family])]
        if policy.get("kind") == "model":
            return [str(policy["model"])]
        if policy.get("kind") == "surface":
            from theriad.models.tiers import ModelCandidate
            from theriad.routing.v2 import RouterV2, RouterV2Artifact
            from theriad.routing.v2_training import artifact_candidates
            artifact = RouterV2Artifact.from_dict(policy["artifacts"][row.family])
            router = RouterV2(artifact)
            priced = [ModelCandidate(tier="cheap_code" if row.family == "coding" else "cheap_bulk", provider=meta.get("provider", "openrouter"), model=model,
                input_cost_per_million=meta["input_per_1m"], output_cost_per_million=meta["output_per_1m"],
                context_window=meta.get("context_window", meta.get("context_length")))
                for model, meta in self.metadata.items()]
            candidates = [c for c in artifact_candidates(artifact, priced)
                          if set(row.capabilities).issubset(set(self.metadata[c.model].get("capabilities", [])))]
            decision = router.predict(row.prompt, candidates=candidates, lane="code" if row.family == "coding" else "general")
            self._prediction_details[(row.content_hash, json.dumps(policy, sort_keys=True))] = decision.to_dict()
            return [decision.selected_model]
        if policy.get("kind") != "current":
            raise ValueError("unsupported episode policy kind")
        plan = self.baseline_plans[row.episode_id]
        return [str(x[1]) for x in plan["chain"]] if plan.get("chain") else list(plan["models"].values())

    async def __call__(self, row: Episode, policy_id: str, policy: Mapping[str, Any]) -> EpisodeOutcome:
        from theriad.agents.model_telemetry import complete_with_telemetry
        policy = self._family_policy(row, policy)
        routed_models = self.models_for(row, policy)
        selection_latency_ms = self._selection_latency_ms[(row.content_hash, json.dumps(policy, sort_keys=True))]
        self.maximum_cost(row, policy)  # Fail closed on missing prices before paid work.
        selected = None if policy.get("kind") == "current" else {"*": routed_models[0]}
        plan = self.baseline_plans[row.episode_id]
        from theriad.agents.task_runner import _TOOL_HELP
        local_tools = [name for name in _TOOL_HELP if name.startswith(("project.", "git."))]
        session = EpisodeRoutingSession(episode_id=row.episode_id, policy_id=policy_id, group_id=row.group_id,
            fixture_version=row.fixture_version, grader_version=row.grader_version,
            selected_models=selected, model_metadata=self.metadata,
            sink=JsonlAttemptSink(self.directory / "attempts.jsonl"),
            max_output_tokens=row.max_output_tokens, max_model_calls=row.max_calls,
            max_input_tokens=row.max_input_tokens, image_input_token_bound=2048,
            provider_factory=self.provider_factory, assistant_provider=plan.get("assistant_provider", "theriad"),
            free_only=bool(plan.get("free_only", False)),
            tool_prices={name: 0.0 for name in ["web.search", "web.fetch", "web.fetch_batch", "task.complete", *local_tools]})
        started = time.perf_counter()
        evidence: dict[str, Any] = {"adapter_kind": "real_product_fixture", "live_user_route": False,
            "bounded_calls": row.max_calls, "selected_configuration": routed_models,
            "isolated_user_features": plan.get("isolated_user_features", []),
            "tool_surface": "project_files" if row.family in {"tools", "agents"} else None,
            "browser_exercised": False}
        prediction = self._prediction_details.get((row.content_hash, json.dumps(policy, sort_keys=True)))
        evidence["predicted_metrics"] = prediction
        session.emit({"event": "policy_decision", "policy_version": "product-episodes-v1",
                      "configuration": routed_models, "predicted_metrics": prediction,
                      "router_latency_ms": selection_latency_ms})
        status, success, output = "completed", None, ""
        try:
            with episode_routing(session):
              async with asyncio.timeout(row.max_seconds):
                if row.family == "search":
                    from theriad.agents.search_synthesis import _enabled, synthesize_web_search
                    if _enabled() != plan.get("synthesis_enabled", _enabled()):
                        raise EpisodeRoutingError("Search synthesis setting drifted from the baseline snapshot")
                    session.prepare(row.family, plan["models"], session.assistant_provider, row.prompt,
                                    required_capabilities=frozenset(row.capabilities))
                    output, _, queries = await synthesize_web_search(row.prompt, service=FrozenEvidenceService(row.fixture["documents"]))
                    evidence["queries"] = queries
                    success, grade_evidence = _text_grade(row, output)
                    evidence.update(grade_evidence)
                elif row.family in {"tools", "agents"}:
                    from theriad.agents.task_engine import TaskEngine
                    from theriad.agents.task_runner import run_task
                    session.prepare(row.family, plan["models"], session.assistant_provider, row.prompt,
                                    required_capabilities=frozenset(row.capabilities))
                    with tempfile.TemporaryDirectory(prefix="theriad-study-task-") as temporary:
                        root = Path(temporary)
                        workspace = root / "workspace"
                        workspace.mkdir()
                        for relative, content in row.fixture.get("files", {}).items():
                            target = self._workspace_path(workspace, relative)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_text(content)
                        engine = TaskEngine(event_dir=root / "events", allowed_workspace_root=workspace)
                        task = engine.create(goal=row.prompt, workspace_root=str(workspace), permission_profile="autonomous",
                            requested_capabilities=["project"], budgets={"max_tool_calls": row.max_calls, "max_runtime_seconds": row.max_seconds})
                        result = await run_task(task.task_id, engine=engine, only_capabilities={"project"})
                        output = str(result.get("final_answer") or "")
                        success = result.get("status") == "completed"
                        observed = {}
                        for relative, expected in row.fixture.get("expected_files", {}).items():
                            target = self._workspace_path(workspace, relative)
                            actual = target.read_text() if target.is_file() else None
                            observed[relative] = actual
                            success = success and actual == expected
                        for relative, expected in row.fixture.get("expected_json", {}).items():
                            target = self._workspace_path(workspace, relative)
                            try:
                                actual = _strict_json(target.read_text())
                            except (ValueError, OSError):
                                actual = None
                            observed[relative] = actual
                            success = success and _json_equal(actual, expected)
                        changed_paths = set(row.fixture.get("expected_files", {})) | set(row.fixture.get("expected_json", {}))
                        preserved = {}
                        for relative, original in row.fixture.get("files", {}).items():
                            if relative not in changed_paths:
                                target = self._workspace_path(workspace, relative)
                                preserved[relative] = target.is_file() and target.read_text() == original
                                success = success and preserved[relative]
                        if not row.fixture.get("expected_files") and not row.fixture.get("expected_json"):
                            success = None
                        evidence.update({"final_state": observed, "task_status": result.get("status"),
                            "tool_events": len(task.events), "preserved_inputs": preserved,
                            "trajectory": [{"sequence": event.sequence, "kind": event.kind, "payload": event.payload}
                                           for event in task.events]})
                        if result.get("status") != "completed":
                            status = "task_failed"
                else:
                    chain = [tuple(x) for x in plan["chain"]]
                    if selected:
                        from theriad.core.providers.factory import get_effective_provider_name
                        chain = [(get_effective_provider_name(session.assistant_provider), selected["*"])]
                    # Every baseline fallback is part of the frozen configuration;
                    # wildcard selection replaces the entire chain for challengers.
                    session.prepare(row.family, {f"completion_{i}": item[1] for i, item in enumerate(plan["chain"])}, session.assistant_provider, row.prompt,
                        required_capabilities=frozenset(row.capabilities))
                    messages = [Message("system", plan["system_prompt"])] if plan.get("system_prompt") else []
                    context = plan.get("context_prefix", "")
                    content: Any = row.prompt
                    if row.fixture.get("image"):
                        content = [{"type": "text", "text": row.prompt}, _image_message(row.fixture["image"])]
                        if context:
                            content.insert(0, {"type": "text", "text": context})
                    elif context:
                        messages.append(Message("user", context))
                    messages.append(Message("user", content))
                    response = None
                    for index, (provider_name, model) in enumerate(chain):
                        if sum(a.get("kind") == "model" for a in session.attempts) >= row.max_calls:
                            break
                        provider = self.provider_factory(provider_name, timeout=min(row.max_seconds, plan.get("timeout_seconds", 60)))
                        try:
                            extra = self._request_extra(plan, provider_name, model, selected=bool(selected))
                            candidate = await complete_with_telemetry(provider, CompletionRequest(model, messages,
                                max_tokens=min(row.max_output_tokens, plan.get("max_output_tokens") or row.max_output_tokens), extra=extra), role="completion")
                            if not candidate.content.strip() and candidate.completion_tokens > 0:
                                session.reject_response("empty_response")
                                continue
                            from theriad.quality.objective_verifier import verify_objective_response
                            validation = verify_objective_response(row.prompt, candidate.content)
                            evidence["objective_verification"] = validation.to_dict() if validation else None
                            if validation is not None and not validation.passed and index < len(chain) - 1:
                                session.reject_response("objective_validation_failed")
                                continue
                            response = candidate
                            break
                        except EpisodeRoutingError:
                            raise
                        except Exception as exc:
                            if selected:
                                raise
                            from theriad.core.providers.interfaces import ProviderError
                            if isinstance(exc, ProviderError) and exc.status_code and 400 <= exc.status_code < 500 and exc.status_code != 429:
                                break
                    if response is None:
                        status, success = "provider_failed", False
                    else:
                        output = response.content
                        from theriad.agents.model_telemetry import is_truncated_finish_reason
                        if is_truncated_finish_reason(response.finish_reason):
                            status, success = "truncated", False
                        elif row.family == "coding":
                            from .sandbox import (
                                SandboxPolicyError,
                                SandboxUnavailable,
                                run_python_tests,
                            )
                            matches = re.findall(r"```(?:python)?\s*\n(.*?)```", output, re.S)
                            code = matches[0] if len(matches) == 1 else output
                            try:
                                grade = await asyncio.to_thread(run_python_tests, code, row.fixture["tests"], entry_point=row.fixture["entry_point"])
                                success = grade.passed
                                evidence["sandbox"] = grade.to_dict()
                                if grade.status in {"unavailable", "policy_error"}:
                                    success, status = None, "grader_unavailable"
                            except (SandboxUnavailable, SandboxPolicyError) as exc:
                                success, status = None, "grader_unavailable"
                                evidence["sandbox"] = {"status": "unavailable", "error_type": type(exc).__name__}
                        else:
                            success, grade_evidence = _text_grade(row, output)
                            evidence.update(grade_evidence)
        except Exception as exc:
            status, success = "timeout" if isinstance(exc, TimeoutError) else "execution_error", False
            evidence["error_type"] = type(exc).__name__
        session.record_charge("router", 0.0, role="local_cpu", elapsed_ms=sum(
            event.get("router_latency_ms", 0) for event in session.events))
        summary = session.summary()
        evidence["router_latency_ms"] = selection_latency_ms
        evidence.update({"accounting": summary, "output": output, "baseline_plan": self.baseline_plans[row.episode_id]})
        model_attempts = [a for a in session.attempts if a.get("kind") == "model"]
        evidence["input_tokens"] = sum(a["prompt_tokens"] for a in model_attempts) if all(a.get("usage_available") for a in model_attempts) else None
        evidence["output_tokens"] = sum(a["completion_tokens"] for a in model_attempts) if all(a.get("usage_available") for a in model_attempts) else None
        if success is None and status == "completed":
            status = "ungradable"
        return EpisodeOutcome(row.episode_id, row.group_id, row.family, policy_id, row.content_hash,
            success, status, summary["total_cost_usd"], (time.perf_counter() - started) * 1000 + selection_latency_ms,
            len(model_attempts), str(model_attempts[-1].get("model", "")) if model_attempts else "", evidence)

    @staticmethod
    def _request_extra(plan, provider, model, *, selected):
        if not selected:
            return deepcopy(plan.get("request_extras", {}).get(model, {}))
        from theriad.api.server import _provider_request_extra
        from theriad.models.preferences import RoutingPreferences
        from theriad.storage.prompt_cache import prompt_cache_extra
        preferences = RoutingPreferences(**plan.get("preferences", {}))
        return {**_provider_request_extra(provider, model, preferences, [model]),
                **prompt_cache_extra(provider)}

    @staticmethod
    def _workspace_path(root: Path, relative: str) -> Path:
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or path == root.resolve():
            raise ValueError("fixture file escapes its workspace")
        return path
