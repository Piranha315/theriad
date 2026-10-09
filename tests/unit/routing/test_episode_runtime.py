from __future__ import annotations

import json
import sys
from dataclasses import replace

import pytest

from theriad.core.providers.interfaces import CompletionResponse
from theriad.routing.episode_hooks import current_episode
from theriad.routing.episode_runtime import EpisodeRuntime, _json_equal, _text_grade, freeze_baseline
from theriad.routing.episode_study import Episode


FIRST = "fixture/first"
SECOND = "fixture/second"
META = {model: {"capabilities": ["text", "tools", "vision"], "input_per_1m": 1,
                "output_per_1m": 2, "provider": "openrouter", "assistant_provider": "theriad"}
        for model in (FIRST, SECOND, "xiaomi/mimo-v2.6-flash", "deepseek/deepseek-v4.1-flash")}


def episode(family="chat", *, prompt="Return the answer.", fixture=None, grader=None, max_calls=4):
    return Episode("episode", "group", family, "pilot", prompt, fixture or {},
                   grader or {"kind": "exact", "answer": "correct"}, max_calls=max_calls)


def chat_plan():
    return {"kind": "current", "assistant_provider": "theriad", "chain": [["openrouter", FIRST], ["openrouter", SECOND]],
            "system_prompt": "Frozen shared system", "context_prefix": "[user]: earlier turn",
            "request_extras": {FIRST: {"models": [FIRST, SECOND]}, SECOND: {"models": [FIRST, SECOND]}},
            "max_output_tokens": 512, "timeout_seconds": 20, "preferences": {"speed": 50, "quality": 50}}


class Provider:
    name = "openrouter"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        assert self.outcomes, "Unexpected model dispatch"
        output = self.outcomes.pop(0)
        if isinstance(output, BaseException):
            raise output
        return CompletionResponse(model=request.model, content=output, prompt_tokens=100, completion_tokens=30,
            latency_ms=1, provider=self.name, finish_reason="stop",
            raw={"usage": {"prompt_tokens": 100, "completion_tokens": 30}})


def runtime(tmp_path, row, provider, plan=None):
    return EpisodeRuntime(directory=tmp_path, metadata=META, baseline_plans={row.episode_id: plan or chat_plan()},
                          provider_factory=lambda *_args, **_kwargs: provider)


@pytest.fixture(autouse=True)
def isolated_product_provider(monkeypatch):
    monkeypatch.setattr("theriad.settings.secrets.get_active_assistant_provider", lambda: "theriad")
    # All adapters use the injected fake factory. This only removes the guard
    # that normally forbids any planner invocation inside a unit-test process.
    monkeypatch.setenv("THERIAD_TEST_LIVE_MODELS", "true")
    monkeypatch.setenv("THERIAD_API_SEARCH_SYNTHESIS", "true")


def test_baseline_freezes_actual_role_models_and_shared_chat_context(monkeypatch):
    monkeypatch.setenv("THERIAD_AGENT_MODEL", "fixture/configured-agent")
    search = freeze_baseline(episode("search"))
    task = freeze_baseline(episode("tools"))
    assert search["models"] == {"search_query": "xiaomi/mimo-v2.6-flash", "search_synthesis": "xiaomi/mimo-v2.6-flash"}
    assert task["models"] == {"agent_planner": "fixture/configured-agent"}
    row = episode(fixture={"messages": [{"role": "user", "content": "earlier public turn"}]})
    plan = freeze_baseline(row)
    assert plan["chain"]
    assert plan["system_prompt"]
    assert "earlier public turn" in plan["context_prefix"]
    assert plan["objective_verification"] is True
    assert len(plan["request_extras"]) == len(plan["chain"])


def test_numeric_and_json_grades_reject_ambiguous_and_type_wrong_answers():
    numeric = episode(grader={"kind": "numeric", "answer": "42"})
    assert _text_grade(numeric, "ANSWER: 42")[0] is True
    assert _text_grade(numeric, "42 or 41")[0] is False
    structured = episode(grader={"kind": "json", "answer": {"ok": True}})
    assert _text_grade(structured, '{"ok":true}')[0] is True
    assert _text_grade(structured, '{"ok":1}')[0] is False
    assert _text_grade(structured, '{"ok":false,"ok":true}')[0] is False
    assert not _json_equal([1], [True])


@pytest.mark.asyncio
async def test_current_chat_fallback_is_real_charged_and_keeps_same_context(tmp_path):
    row = episode()
    provider = Provider([TimeoutError("upstream"), "correct"])
    result = await runtime(tmp_path, row, provider)(row, "baseline", {"kind": "current"})
    assert result.success is True
    assert result.attempts == 2
    assert result.cost_usd is None  # the timeout may have been billed
    assert [req.model for req in provider.requests] == [FIRST, SECOND]
    assert provider.requests[0].messages == provider.requests[1].messages
    assert [m.role for m in provider.requests[0].messages] == ["system", "user", "user"]
    assert provider.requests[0].extra["models"] == [FIRST, SECOND]
    assert current_episode() is None


@pytest.mark.asyncio
async def test_surface_policy_routes_each_decision_and_reacts_to_unavailable_model(tmp_path):
    row = episode()
    artifact = {
        "schema_version": 1,
        "router_version": "fixture-router",
        "feature_mean": {},
        "feature_stdev": {},
        "lanes": {
            "general": {"knee_threshold": 0.0, "fallback_model": SECOND, "models": [FIRST, SECOND]},
            "code": {"knee_threshold": 0.0, "fallback_model": SECOND, "models": [FIRST, SECOND]},
        },
        "models": {
            FIRST: {"intercept": 1.0, "expected_output_tokens": 30},
            SECOND: {"intercept": 0.5, "expected_output_tokens": 30},
        },
    }
    provider = Provider([TimeoutError("unavailable"), "correct"])
    result = await runtime(tmp_path, row, provider)(
        row, "challenger", {"kind": "surface", "artifacts": {row.family: artifact}}
    )
    assert result.success is True
    assert len(result.evidence["decision_trace"]) >= 2
    assert [request.model for request in provider.requests] == [FIRST, SECOND]
    assert any(event["reason"] in {"cheapest_above_knee", "highest_success_fallback"} for event in result.evidence["decision_trace"])


@pytest.mark.asyncio
async def test_objective_rejection_triggers_baseline_fallback_and_counts_both_costs(tmp_path):
    row = episode(prompt="Return JSON only.", grader={"kind": "json", "answer": {"a": 2}})
    provider = Provider(["bad JSON", '{"a":2}'])
    result = await runtime(tmp_path, row, provider)(row, "baseline", {"kind": "current"})
    assert result.success is True
    assert result.attempts == 2
    assert result.cost_usd == pytest.approx(0.00032)
    events = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert any(event.get("reason") == "objective_validation_failed" for event in events)


@pytest.mark.asyncio
async def test_by_family_challenger_never_inherits_baseline_fallback(tmp_path):
    row = episode()
    provider = Provider([RuntimeError("reject"), "correct"])
    policy = {"kind": "by_family", "policies": {"chat": {"kind": "model", "model": SECOND}}}
    runner = runtime(tmp_path, row, provider)
    assert runner.models_for(row, policy) == [SECOND]
    assert runner.max_cost(row, policy) == runner.maximum_cost(row, policy)
    result = await runner(row, "challenger", policy)
    assert result.success is False
    assert result.attempts == 1
    assert [req.model for req in provider.requests] == [SECOND]
    assert "models" not in provider.requests[0].extra


@pytest.mark.asyncio
async def test_chat_call_bound_counts_failed_attempts(tmp_path):
    row = episode(max_calls=1)
    provider = Provider([RuntimeError("failed"), "correct"])
    result = await runtime(tmp_path, row, provider)(row, "baseline", {"kind": "current"})
    assert result.success is False
    assert result.attempts == 1
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_search_runs_real_query_and_synthesis_against_frozen_service(tmp_path):
    row = episode("search", prompt="Return JSON about the launch year using the evidence.",
        fixture={"documents": [{"url": "https://fixture.invalid/history", "title": "Launch history", "text": "The launch occurred in 2007."}]},
        grader={"kind": "json", "answer": {"year": 2007, "citation": "https://fixture.invalid/history"}}, max_calls=2)
    provider = Provider(['{"queries":[]}', '{"year":2007,"citation":"https://fixture.invalid/history"}'])
    result = await runtime(tmp_path, row, provider, freeze_baseline(row))(row, "baseline", {"kind": "current"})
    assert result.success is True
    assert result.attempts == 2
    assert result.cost_usd == pytest.approx(.00032)
    assert [req.model for req in provider.requests] == ["xiaomi/mimo-v2.6-flash"] * 2
    assert "2007" in provider.requests[1].messages[-1].text
    assert result.evidence["accounting"]["all_costs_known"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["tools", "agents"])
async def test_task_engine_runs_complete_trajectory_and_resets_each_episode(tmp_path, family):
    row = episode(family, prompt="Read input.txt, write output.json with the requested answer, and finish.",
        fixture={"files": {"input.txt": "public source"}, "expected_json": {"output.json": {"answer": 42}}},
        grader={"kind": "state"}, max_calls=4)
    decisions = [
        '{"tool":"project.read","arguments":{"path":"input.txt"}}',
        '{"tool":"project.write","arguments":{"path":"output.json","content":"{\\"answer\\":42}"}}',
        '{"tool":"task.complete","arguments":{"answer":"Done"}}',
    ]
    # Serialize deliberately; malformed escaping must never simulate success.
    decisions[1] = json.dumps({"tool": "project.write", "arguments": {"path": "output.json", "content": '{"answer":42}'}})
    provider = Provider(decisions * 2)
    runner = runtime(tmp_path, row, provider, freeze_baseline(row))
    for policy_id in ("baseline", "challenger"):
        result = await runner(row, policy_id, {"kind": "current"})
        assert result.success is True
        assert result.attempts == 3
        assert result.evidence["preserved_inputs"] == {"input.txt": True}
        assert result.evidence["trajectory"][-1]["kind"]
        assert result.evidence["final_state"] == {"output.json": {"answer": 42}}
        assert result.evidence["browser_exercised"] is False
    assert current_episode() is None


@pytest.mark.asyncio
async def test_expected_boolean_and_preserved_input_are_verified_from_actual_state(tmp_path):
    row = episode("tools", prompt="Write the boolean result and preserve inputs.",
        fixture={"files": {"input.txt": "original"}, "expected_json": {"out.json": {"ok": True}}},
        grader={"kind": "state"})
    provider = Provider([
        json.dumps({"tool": "project.write", "arguments": {"path": "out.json", "content": '{"ok":1}'}}),
        '{"tool":"task.complete","arguments":{"answer":"Done"}}'])
    result = await runtime(tmp_path, row, provider, freeze_baseline(row))(row, "baseline", {"kind": "current"})
    assert result.success is False
    assert result.evidence["task_status"] == "completed"


@pytest.mark.asyncio
async def test_task_retry_cap_stops_extra_model_dispatch(tmp_path):
    row = episode("tools", fixture={"expected_files": {"out.txt": "ok"}}, grader={"kind": "state"}, max_calls=1)
    provider = Provider([RuntimeError("JSON mode refused"), "unused"])
    result = await runtime(tmp_path, row, provider, freeze_baseline(row))(row, "baseline", {"kind": "current"})
    assert result.success is False
    assert result.attempts == 1
    assert len(provider.requests) == 1
    assert result.cost_usd is None


@pytest.mark.asyncio
async def test_missing_code_grader_is_ungradable_after_charged_model_response(monkeypatch, tmp_path):
    monkeypatch.delenv("THERIAD_ROUTER_V2_CODE_GRADER_PYTHON", raising=False)
    row = episode("coding", fixture={"tests": "def check(f): assert f(2) == 3", "entry_point": "increment"},
                  grader={"kind": "code"})
    provider = Provider(["def increment(x): return x + 1"])
    result = await runtime(tmp_path, row, provider)(row, "baseline", {"kind": "current"})
    assert result.success is None
    assert result.status == "grader_unavailable"
    assert result.cost_usd == pytest.approx(.00016)


@pytest.mark.asyncio
async def test_code_fixture_uses_real_pinned_macos_sandbox(monkeypatch, tmp_path):
    from theriad.routing.sandbox import sandbox_available
    if not sandbox_available():
        pytest.skip("macOS sandbox-exec required")
    monkeypatch.setenv("THERIAD_ROUTER_V2_CODE_GRADER_PYTHON", sys.executable)
    row = episode("coding", fixture={"tests": "def check(f): assert f(2) == 3\n", "entry_point": "increment"},
                  grader={"kind": "code"})
    provider = Provider(["```python\ndef increment(x): return x + 1\n```"])
    result = await runtime(tmp_path, row, provider)(row, "baseline", {"kind": "current"})
    assert result.success is True, result.evidence
    assert result.evidence["sandbox"]["executor_id"] == "router-v2.macos.sandbox-exec"


@pytest.mark.asyncio
async def test_vision_fixture_renders_image_without_leaking_answer_metadata(tmp_path):
    row = replace(episode(fixture={"image": {"text": "RECEIPT\nTOTAL 12", "expected": "do-not-send"}},
                          grader={"kind": "numeric", "answer": "12"}), capabilities=("text", "vision"))
    provider = Provider(["ANSWER: 12"])
    result = await runtime(tmp_path, row, provider)(row, "baseline", {"kind": "current"})
    assert result.success is True
    message = provider.requests[0].messages[-1]
    assert message.has_images
    assert "do-not-send" not in json.dumps(message.content)
    assert message.content[-1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_unknown_or_invalid_prices_and_incompatible_challenger_fail_before_call(tmp_path):
    row = episode()
    runner = EpisodeRuntime(directory=tmp_path, metadata={FIRST: {"input_per_1m": float("nan"), "output_per_1m": 1}},
                            baseline_plans={row.episode_id: {"chain": [["openrouter", FIRST]]}},
                            provider_factory=lambda *_args, **_kwargs: pytest.fail("provider called"))
    with pytest.raises(ValueError, match="price"):
        runner.max_cost(row, {"kind": "current"})
    with pytest.raises(ValueError, match="capability"):
        runner.models_for(row, {"kind": "model", "model": FIRST})
