from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from theriad.accounting.pricing import ModelPrice, PricingTable
from theriad.core.providers.interfaces import (
    CompletionRequest,
    CompletionResponse,
    Message,
    ProviderError,
)
from theriad.routing.sandbox import CODE_GRADER_RUNTIME_ENV
from theriad.routing.study import CORE_MODELS
from theriad.routing.study_adapters import (
    BFCLAdapter,
    ExecutableCodeAdapter,
    StructuredCompletionRequest,
    StructuredCompletionResponse,
    StructuredToolCall,
    TauBenchAdapter,
    runtime_collection_eligibility,
)
from theriad.routing.study_budget import StudyBudgetExceeded, StudyBudgetLedger
from theriad.routing.study_collector import (
    PHASE_A,
    PlannedStudyCall,
    StudyCollector,
    StudyCollectorError,
    StudyPlan,
    _canonical_hash,
    _FreeEndpointRetryProvider,
    corpus_fingerprint,
    load_plan,
    write_plan,
)
from theriad.runtime.circuit_breaker import CircuitOpenError

from .test_study import _prompt


@pytest.fixture(autouse=True)
def _pin_active_code_grader_runtime(monkeypatch):
    """Make fixture grading use the same explicitly selected runtime contract."""
    import theriad.routing.sandbox as sandbox_module

    monkeypatch.setenv(CODE_GRADER_RUNTIME_ENV, sys.executable)
    monkeypatch.setattr(sandbox_module, "sandbox_available", lambda: True)
    monkeypatch.setattr(sandbox_module, "runtime_supported", lambda runtime=None: True)


def _code_row():
    return replace(
        _prompt(5, "train"),
        prompt="Implement answer(x) so the pinned tests pass.",
        grader={
            "kind": "sandboxed_code_tests",
            "code_task_type": "function_implementation",
            "entry_point": "answer",
            "executable_tests": "def check(answer):\n    assert answer(2) == 3\n",
            "test_harness_id": "fixture-harness",
            "test_harness_version": "fixture-v1",
        },
        task_format="code_execution",
        lane="code",
    )


class _OverloadedProvider:
    name = "fixture"
    supports_tool_calls = True
    supports_agent_tools = True

    def __init__(self):
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        if self.calls < 3:
            raise ProviderError("fixture", "overloaded", 502)
        return CompletionResponse(request.model, "A", 1, 1, 1.0, "fixture")

    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_free_endpoint_retry_is_bounded_and_never_applies_to_paid_slug():
    base = _OverloadedProvider()
    provider = _FreeEndpointRetryProvider(base, max_attempts=3)
    free_request = CompletionRequest(
        "fixture/model:free", [Message(role="user", content="answer")]
    )

    response = await provider.complete(free_request)

    assert response.model == "fixture/model:free"
    assert base.calls == 3
    assert provider.api_invocations == 3

    paid = _OverloadedProvider()
    paid_provider = _FreeEndpointRetryProvider(paid, max_attempts=3)
    paid_request = CompletionRequest(
        "fixture/model", [Message(role="user", content="answer")]
    )
    with pytest.raises(ProviderError, match="overloaded"):
        await paid_provider.complete(paid_request)
    assert paid.calls == 1


@pytest.mark.asyncio
async def test_free_endpoint_retry_waits_for_open_circuit_without_counting_an_api_call():
    class _CircuitThenSuccess(_OverloadedProvider):
        async def complete(self, request):
            self.calls += 1
            if self.calls == 1:
                raise CircuitOpenError("llm:openrouter", 0.0)
            return CompletionResponse(request.model, "A", 1, 1, 1.0, "fixture")

    base = _CircuitThenSuccess()
    provider = _FreeEndpointRetryProvider(base, max_attempts=3)
    request = CompletionRequest(
        "fixture/model:free", [Message(role="user", content="answer")]
    )

    response = await provider.complete(request)

    assert response.content == "A"
    assert base.calls == 2
    assert provider.api_invocations == 1


def _unavailable_bfcl_row():
    return replace(
        _prompt(6, "train", "tool_state"),
        prompt_id="bfcl-unavailable",
        source="bfcl-v4",
        task_format="tool_use",
        grader={
            "kind": "bfcl_official_checker",
            "executor_id": "bfcl-checker",
            "executor_version": "bfcl-v1",
            "evidence_policy": "official",
            "tool_schemas": [{"name": "lookup", "parameters": {"type": "object"}}],
            "official_checker_id": "bfcl-checker",
            "official_checker_version": "bfcl-v1",
            "official_checker_available": False,
            "task_mode": "simple",
            "ground_truth": [{"name": "lookup", "arguments": {}}],
        },
    )


class _CodeProvider:
    name = "fixture"

    def __init__(self, body: str):
        self.body = body
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return CompletionResponse(
            model=request.model,
            content=self.body,
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=1.0,
            provider=self.name,
            raw={"usage": {"cost": 0.0}},
            cost_usd=0.0,
        )


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec only available on macOS")
def test_executable_code_adapter_uses_real_sandbox_and_never_plain_grader(monkeypatch):
    row = _code_row()
    passing = _CodeProvider("```python\ndef answer(x):\n    return x + 1\n```")
    result = ExecutableCodeAdapter().collect(passing, row, model="fixture")
    assert result.status == "graded"
    assert result.grade is True
    assert result.evidence["sandbox"]["status"] == "passed"
    assert passing.calls == 1

    failing = _CodeProvider("```python\ndef answer(x):\n    return x + 2\n```")
    result = ExecutableCodeAdapter().collect(failing, row, model="fixture")
    assert result.status == "graded"
    assert result.grade is False
    assert "sandbox" in result.evidence

    import theriad.routing.sandbox as sandbox

    original_policy = sandbox.default_policy
    monkeypatch.setattr(
        sandbox,
        "default_policy",
        lambda **kwargs: replace(
            original_policy(**kwargs), wall_clock_seconds=0.25, cpu_seconds=1
        ),
    )
    timeout = _CodeProvider("```python\ndef answer(x):\n    while True: pass\n```")
    result = ExecutableCodeAdapter().collect(timeout, row, model="fixture")
    assert result.status == "timeout"
    assert result.grade is False
    assert result.evidence["sandbox"]["timed_out"] is True


def test_code_prose_is_ungradable_and_sandbox_unavailability_blocks_provider(monkeypatch):
    row = _code_row()
    prose = _CodeProvider("The answer is 3.")
    result = ExecutableCodeAdapter().collect(prose, row, model="fixture")
    assert result.status == "ungradable_code_response"
    assert result.grade is None

    import theriad.routing.sandbox as sandbox

    monkeypatch.setattr(sandbox, "sandbox_available", lambda: False)
    blocked = _CodeProvider("```python\ndef answer(x):\n    return x + 1\n```")
    result = ExecutableCodeAdapter().collect(blocked, row, model="fixture")
    assert result.status == "sandbox_unavailable"
    assert result.eligible is False
    assert blocked.calls == 0


def test_code_adapter_blocks_incompatible_grader_runtime_before_provider(monkeypatch):
    row = _code_row()
    import theriad.routing.sandbox as sandbox

    monkeypatch.setattr(sandbox, "runtime_supported", lambda runtime=None: False)
    blocked = _CodeProvider("```python\ndef answer(x):\n    return x + 1\n```")
    result = ExecutableCodeAdapter().collect(blocked, row, model="fixture")
    assert result.status == "sandbox_unavailable"
    assert result.eligible is False
    assert blocked.calls == 0


def test_code_adapter_blocks_missing_pinned_grader_runtime_before_provider(monkeypatch):
    row = _code_row()
    monkeypatch.delenv(CODE_GRADER_RUNTIME_ENV, raising=False)
    blocked = _CodeProvider("```python\ndef answer(x):\n    return x + 1\n```")
    result = ExecutableCodeAdapter().collect(blocked, row, model="fixture")
    assert result.status == "sandbox_unavailable"
    assert result.eligible is False
    assert blocked.calls == 0


class _OfficialDisagreement:
    available = True

    def check(self, **_kwargs):
        return {
            "official_checker_id": "fixture-official",
            "official_checker_revision": "fixture-v1",
            "official_checker_result": {"valid": False, "error": ["authoritative fixture"]},
            "authoritative": True,
        }


class _StructuredProvider:
    supports_tool_calls = True

    async def complete_with_tools(self, request: StructuredCompletionRequest):
        return StructuredCompletionResponse(
            request.model,
            "",
            (StructuredToolCall("first", {"x": 1}, 0),),
            provider="fixture",
        )


def test_official_bfcl_result_is_authoritative_over_compatibility_result():
    row = replace(
        _prompt(1, "train", "tool_state"),
        source="bfcl-v4",
        task_format="tool_use",
        grader={
            "kind": "bfcl_official_checker",
            "executor_id": "fixture",
            "executor_version": "v1",
            "evidence_policy": "official",
            "tool_schemas": [{"name": "first"}],
            "official_checker_id": "fixture",
            "official_checker_version": "v1",
            "official_checker_available": True,
            "task_mode": "simple",
            "ground_truth": [{"name": "first", "arguments": {"x": 1}}],
        },
    )
    result = BFCLAdapter(official_checker=_OfficialDisagreement()).collect(
        _StructuredProvider(), row, model="fixture"
    )
    assert result.grade is False
    assert result.evidence["official_checker_result"]["valid"] is False
    assert result.evidence["compatibility_agrees"] is False


class _TauEnvironment:
    def __init__(self):
        self._state = {"open": True, "actions": []}

    def reset(self, initial_state):
        self._state = dict(initial_state)
        self._state.setdefault("actions", [])

    def restore_state(self, state):
        self._state = dict(state)

    def restore_trajectory(self, trajectory):
        for record in trajectory:
            for call in record.get("tool_calls") or []:
                self.apply_tool_call(
                    StructuredToolCall(call["name"], call["arguments"], call["order"])
                )

    def tool_schemas(self):
        return ({"name": "close", "parameters": {"type": "object"}},)

    def apply_tool_call(self, call):
        self._state["actions"].append(call.to_dict())
        self._state["open"] = False
        return {"ok": True, "state": dict(self._state)}

    def apply_actions(self, actions):
        for action in actions:
            self.apply_tool_call(StructuredToolCall(action["name"], action.get("arguments", {}), 0))

    def state(self):
        return dict(self._state)

    def evaluate(self, trajectory):
        return {"passed": self._state["open"] is False, "state": dict(self._state)}


class _TauProvider:
    supports_agent_tools = True

    def __init__(self):
        self.requests = []

    async def complete_with_tools(self, request):
        self.requests.append(request)
        if any(message.role == "tool" for message in request.messages):
            assistant = next(
                message for message in reversed(request.messages) if message.role == "assistant"
            )
            tool_message = next(
                message for message in reversed(request.messages) if message.role == "tool"
            )
            assert assistant.tool_calls
            assert assistant.tool_calls[0]["id"]
            assert tool_message.tool_call_id == assistant.tool_calls[0]["id"]
            return StructuredCompletionResponse(request.model, "done", (), provider="fixture")
        return StructuredCompletionResponse(
            request.model,
            "",
            (StructuredToolCall("close", {}, 0, call_id="c1"),),
            provider="fixture",
        )


def test_tau_adapter_runs_interactive_turns_and_persists_state():
    row = replace(
        _prompt(2, "train", "tool_state"),
        source="tau3",
        task_format="agent_state",
        grader={
            "kind": "tau2_official_environment",
            "executor_id": "fixture-tau",
            "executor_version": "v1",
            "evidence_policy": "official",
            "official_environment_id": "fixture",
            "official_environment_available": True,
            "official_evaluator_id": "tau2.evaluator.evaluator.evaluate_simulation",
            "official_evaluator_version": "v1",
            "initial_state": {"open": True},
            "max_turns": 3,
        },
    )
    persisted = []
    result = TauBenchAdapter().collect(
        _TauProvider(),
        row,
        model="fixture",
        environment_factory=lambda _row: _TauEnvironment(),
        persist=persisted.append,
    )
    assert result.status == "graded"
    assert result.grade is True
    assert result.api_call_count == 2
    assert result.evidence["interactive_agent_loop"] is True
    assert persisted[-1]["status"] == "complete"


def test_tau_adapter_keeps_frozen_initial_state_immutable_and_supports_resume():
    row = replace(
        _prompt(3, "train", "tool_state"),
        source="tau3",
        task_format="agent_state",
        grader={
            "kind": "tau2_official_environment",
            "executor_id": "fixture-tau",
            "executor_version": "v1",
            "evidence_policy": "official",
            "official_environment_id": "fixture",
            "official_environment_available": True,
            "official_evaluator_id": "tau2.evaluator.evaluator",
            "official_evaluator_version": "v1",
            "initial_state": {"open": True, "actions": []},
            "max_turns": 3,
        },
    )
    frozen_grader = deepcopy(row.grader)
    result = TauBenchAdapter().collect(
        _TauProvider(),
        row,
        model="fixture",
        environment_factory=lambda _row: _TauEnvironment(),
    )
    assert result.grade is True
    assert row.grader == frozen_grader

    resumed = TauBenchAdapter().collect(
        _TauProvider(),
        row,
        model="fixture",
        environment_factory=lambda _row: _TauEnvironment(),
        resume={
            "state": {"open": False, "actions": [{"name": "close"}]},
            "trajectory": [
                {
                    "turn": 0,
                    "assistant_content": "",
                    "tool_calls": [
                        {"name": "close", "arguments": {}, "order": 0}
                    ],
                    "tool_results": [{"ok": True}],
                }
            ],
        },
    )
    assert resumed.status == "graded"
    assert resumed.grade is True
    assert resumed.api_call_count == 2


def test_runtime_capability_preflight_happens_before_budget_reservation():
    row = _code_row()
    provider = type("NoTools", (), {"name": "fixture", "complete": None})()
    allowed, reason = runtime_collection_eligibility(provider, row)
    assert allowed is False
    assert "completion capability" in reason


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec only available on macOS")
def test_real_collector_executes_code_adapter_path_with_zero_cost(tmp_path: Path):
    row = _code_row()
    models = CORE_MODELS
    calls = tuple(
        PlannedStudyCall(
            prompt_id=row.prompt_id,
            model=model,
            split="train",
            max_output_tokens=64,
            maximum_cost_usd=0.0,
            idempotency_key=f"A:{row.prompt_id}:{model}",
            input_tokens=1,
            expected_output_tokens=16,
            expected_cost_usd=0.0,
            task_format=row.task_format,
            adapter_id="router-v2-executable-code",
        )
        for model in models
    )
    draft = StudyPlan(
        calls=calls,
        price_snapshot={model: {"input_per_1m": 0.0, "output_per_1m": 0.0} for model in models},
        corpus_hash=corpus_fingerprint((row,)),
        phase=PHASE_A,
        status="ready",
        cost_scenarios={
            "free_nemotron": {
                "status": "priced",
                "within_hard_cap": True,
                "untouched_test_reserve_preserved": True,
            }
        },
    )
    plan = replace(draft, plan_hash=_canonical_hash(draft._payload()))
    provider = _CodeProvider("```python\ndef answer(x):\n    return x + 1\n```")
    collector = StudyCollector(
        prompts=(row,),
        plan=plan,
        ledger=StudyBudgetLedger(tmp_path / "budget.json"),
        state_path=tmp_path / "state.json",
        provider_factory=lambda: provider,
    )
    summary = collector.execute(
        reviewed_plan_hash=plan.plan_hash,
        allow_provider_calls=True,
        allow_paid_calls=False,
        yes=True,
    )
    assert provider.calls == 3
    assert summary["graded_outcomes"] == 3
    assert summary["actual_cost_usd"] == 0.0

    persisted = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    persisted_runs = persisted["rows"][row.prompt_id]["runs"]
    persisted_runs[models[0]]["status"] = "failure"
    persisted_runs[models[1]]["status"] = "ungradable_code_response"
    (tmp_path / "state.json").write_text(
        json.dumps(persisted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    resumed = collector.execute(
        reviewed_plan_hash=plan.plan_hash,
        allow_provider_calls=True,
        allow_paid_calls=False,
        yes=True,
    )
    assert provider.calls == 3
    assert resumed["runs"] == 3
    assert resumed["completed_runs"] == 3


def test_collector_retries_only_explicit_settled_error_with_new_reservation(tmp_path: Path):
    row = _prompt(1, "train")
    model = CORE_MODELS[0]
    calls = tuple(
        PlannedStudyCall(
            prompt_id=row.prompt_id,
            model=candidate,
            split="train",
            max_output_tokens=32,
            maximum_cost_usd=0.0,
            idempotency_key=f"A:{row.prompt_id}:{candidate}",
            input_tokens=1,
            expected_output_tokens=1,
            expected_cost_usd=0.0,
            task_format=row.task_format,
            adapter_id="plain-text",
        )
        for candidate in CORE_MODELS
    )
    draft = StudyPlan(
        calls=calls,
        price_snapshot={
            candidate: {"input_per_1m": 0.0, "output_per_1m": 0.0}
            for candidate in CORE_MODELS
        },
        corpus_hash=corpus_fingerprint((row,)),
        phase=PHASE_A,
        status="ready",
        cost_scenarios={
            "free_nemotron": {
                "status": "priced",
                "within_hard_cap": True,
                "untouched_test_reserve_preserved": True,
            }
        },
    )
    plan = replace(draft, plan_hash=_canonical_hash(draft._payload()))

    class _RejectOnceProvider:
        name = "fixture"

        def __init__(self):
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("fixture", "invalid tool history", 400)
            return CompletionResponse(
                request.model,
                "A",
                1,
                1,
                1.0,
                self.name,
                raw={"usage": {"cost": 0.0}},
                cost_usd=0.0,
            )

    provider = _RejectOnceProvider()
    ledger = StudyBudgetLedger(tmp_path / "budget.json")
    state_path = tmp_path / "state.json"
    collector = StudyCollector(
        prompts=(row,),
        plan=plan,
        ledger=ledger,
        state_path=state_path,
        provider_factory=lambda: provider,
    )
    with pytest.raises(ProviderError, match="invalid tool history"):
        collector.execute(
            reviewed_plan_hash=plan.plan_hash,
            allow_provider_calls=True,
            allow_paid_calls=False,
            yes=True,
        )
    failed = json.loads(state_path.read_text(encoding="utf-8"))["rows"][row.prompt_id][
        "runs"
    ][model]
    ledger.settle(failed["reservation_id"], 0.0)

    summary = collector.execute(
        reviewed_plan_hash=plan.plan_hash,
        allow_provider_calls=True,
        allow_paid_calls=False,
        yes=True,
        retry_errors=True,
        retry_error_prompt_ids=frozenset({row.prompt_id}),
    )

    completed = json.loads(state_path.read_text(encoding="utf-8"))["rows"][row.prompt_id][
        "runs"
    ][model]
    ledger_rows = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))[
        "entries"
    ]
    assert provider.calls == 4
    assert summary["completed_runs"] == 3
    assert completed["status"] == "success"
    assert completed["retry_count"] == 1
    assert completed["attempt_history"][0]["status"] == "error"
    assert len(ledger_rows) == 4
    assert ledger_rows[1]["idempotency_key"].endswith(":retry:1")


def test_collector_persists_observed_cost_when_it_exceeds_reservation(tmp_path: Path):
    row = _prompt(1, "train")
    model = CORE_MODELS[1]
    call = PlannedStudyCall(
        prompt_id=row.prompt_id,
        model=model,
        split="train",
        max_output_tokens=32,
        maximum_cost_usd=0.001,
        idempotency_key=f"A:{row.prompt_id}:{model}",
        input_tokens=1,
        expected_output_tokens=1,
        expected_cost_usd=0.0005,
        task_format=row.task_format,
        adapter_id="plain-text",
    )
    draft = StudyPlan(
        calls=(call,),
        price_snapshot={model: {"input_per_1m": 1.0, "output_per_1m": 1.0}},
        corpus_hash=corpus_fingerprint((row,)),
        phase=PHASE_A,
        status="ready",
        cost_scenarios={
            "paid": {
                "status": "priced",
                "within_hard_cap": True,
                "untouched_test_reserve_preserved": True,
            }
        },
    )
    plan = replace(draft, plan_hash=_canonical_hash(draft._payload()))

    class _OverBudgetProvider:
        name = "fixture"

        def __init__(self):
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            return CompletionResponse(
                request.model,
                "A",
                1,
                1,
                1.0,
                self.name,
                raw={"usage": {"cost": 0.002}},
                cost_usd=0.002,
            )

    provider = _OverBudgetProvider()
    ledger = StudyBudgetLedger(tmp_path / "budget.json")
    state_path = tmp_path / "state.json"
    collector = StudyCollector(
        prompts=(row,),
        plan=plan,
        ledger=ledger,
        state_path=state_path,
        provider_factory=lambda: provider,
        pricing=PricingTable(prices={model: ModelPrice(1.0, 1.0)}, use_live=False),
    )

    with pytest.raises(StudyBudgetExceeded, match="exceeded its reservation"):
        collector.execute(
            reviewed_plan_hash=plan.plan_hash,
            allow_provider_calls=True,
            allow_paid_calls=True,
            yes=True,
        )

    run = json.loads(state_path.read_text(encoding="utf-8"))["rows"][row.prompt_id][
        "runs"
    ][model]
    reservation = ledger.reservation(run["reservation_id"])
    assert provider.calls == 1
    assert run["status"] == "error"
    assert run["observed_realized_cost_usd"] == pytest.approx(0.002)
    assert reservation is not None
    assert reservation["actual_cost_usd"] is None


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec only available on macOS")
def test_selected_model_execution_never_completes_phase(tmp_path: Path):
    row = _code_row()
    calls = tuple(
        PlannedStudyCall(
            prompt_id=row.prompt_id,
            model=model,
            split="train",
            max_output_tokens=64,
            maximum_cost_usd=0.0,
            idempotency_key=f"A:{row.prompt_id}:{model}",
            input_tokens=1,
            expected_output_tokens=16,
            expected_cost_usd=0.0,
            task_format=row.task_format,
            adapter_id="router-v2-executable-code",
        )
        for model in CORE_MODELS
    )
    draft = StudyPlan(
        calls=calls,
        price_snapshot={
            model: {"input_per_1m": 0.0, "output_per_1m": 0.0}
            for model in CORE_MODELS
        },
        corpus_hash=corpus_fingerprint((row,)),
        phase=PHASE_A,
        status="ready",
        cost_scenarios={
            "free_nemotron": {
                "status": "priced",
                "within_hard_cap": True,
                "untouched_test_reserve_preserved": True,
            }
        },
    )
    plan = replace(draft, plan_hash=_canonical_hash(draft._payload()))
    provider = _CodeProvider("```python\ndef answer(x):\n    return x + 1\n```")
    state_path = tmp_path / "state.json"
    collector = StudyCollector(
        prompts=(row,),
        plan=plan,
        ledger=StudyBudgetLedger(tmp_path / "budget.json"),
        state_path=state_path,
        provider_factory=lambda: provider,
    )

    summary = collector.execute(
        reviewed_plan_hash=plan.plan_hash,
        allow_provider_calls=True,
        allow_paid_calls=False,
        yes=True,
        selected_models=frozenset({CORE_MODELS[1]}),
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert provider.calls == 1
    assert summary["execution_scope"] == "selected_models"
    assert summary["selected_models"] == [CORE_MODELS[1]]
    assert summary["phase_completion_deferred"] is True
    assert state["phase_status"] == {}


def test_all_call_preflight_blocks_late_ineligible_row_before_reservation(tmp_path: Path):
    code = _code_row()
    bfcl = _unavailable_bfcl_row()
    rows = (code, bfcl)
    calls = tuple(
        PlannedStudyCall(
            prompt_id=row.prompt_id,
            model=CORE_MODELS[0],
            split="train",
            max_output_tokens=64,
            maximum_cost_usd=0.0,
            idempotency_key=f"A:{row.prompt_id}:{CORE_MODELS[0]}",
            input_tokens=1,
            expected_output_tokens=16,
            expected_cost_usd=0.0,
            task_format=row.task_format,
            adapter_id="fixture",
        )
        for row in rows
    )
    draft = StudyPlan(
        calls=calls,
        price_snapshot={CORE_MODELS[0]: {"input_per_1m": 0.0, "output_per_1m": 0.0}},
        corpus_hash=corpus_fingerprint(rows),
        phase=PHASE_A,
        status="ready",
        cost_scenarios={
            "free_nemotron": {
                "status": "priced",
                "within_hard_cap": True,
                "untouched_test_reserve_preserved": True,
            }
        },
    )
    plan = replace(draft, plan_hash=_canonical_hash(draft._payload()))
    provider = _CodeProvider("```python\ndef answer(x):\n    return x + 1\n```")
    ledger = StudyBudgetLedger(tmp_path / "budget.json")
    collector = StudyCollector(
        prompts=rows,
        plan=plan,
        ledger=ledger,
        state_path=tmp_path / "state.json",
        provider_factory=lambda: provider,
    )
    with pytest.raises(StudyCollectorError, match="all-call preflight"):
        collector.execute(
            reviewed_plan_hash=plan.plan_hash,
            allow_provider_calls=True,
            allow_paid_calls=False,
            yes=True,
        )
    assert provider.calls == 0
    assert ledger.status()["committed_usd"] == 0


def test_enforced_price_snapshot_drift_fails_before_provider(tmp_path: Path):
    row = _code_row()
    call = PlannedStudyCall(
        prompt_id=row.prompt_id,
        model=CORE_MODELS[0],
        split="train",
        max_output_tokens=64,
        maximum_cost_usd=0.0,
        idempotency_key=f"A:{row.prompt_id}:{CORE_MODELS[0]}",
        input_tokens=1,
        expected_output_tokens=16,
    )
    draft = StudyPlan(
        calls=(call,),
        price_snapshot={CORE_MODELS[0]: {"input_per_1m": 0.0, "output_per_1m": 0.0}},
        corpus_hash=corpus_fingerprint((row,)),
        phase=PHASE_A,
        status="ready",
        cost_scenarios={
            "free_nemotron": {
                "status": "priced",
                "within_hard_cap": True,
                "untouched_test_reserve_preserved": True,
            }
        },
    )
    plan = replace(draft, plan_hash=_canonical_hash(draft._payload()))
    provider = _CodeProvider("```python\ndef answer(x):\n    return x + 1\n```")
    collector = StudyCollector(
        prompts=(row,),
        plan=plan,
        ledger=StudyBudgetLedger(tmp_path / "budget.json"),
        state_path=tmp_path / "state.json",
        provider_factory=lambda: provider,
        pricing=PricingTable(
            prices={CORE_MODELS[0]: ModelPrice(1.0, 1.0)}, use_live=False
        ),
        adapter_context={"enforce_price_snapshot": True},
    )
    with pytest.raises(StudyCollectorError, match="price snapshot drift"):
        collector.execute(
            reviewed_plan_hash=plan.plan_hash,
            allow_provider_calls=True,
            allow_paid_calls=True,
            yes=True,
        )
    assert provider.calls == 0


def test_serialized_plan_preserves_trajectory_reservation_fields(tmp_path: Path):
    row = _code_row()
    call = PlannedStudyCall(
        prompt_id=row.prompt_id,
        model=CORE_MODELS[0],
        split="train",
        max_output_tokens=2048,
        maximum_cost_usd=0.0,
        idempotency_key="trajectory-fixture",
        expected_turns=2,
        max_turns=8,
        potential_api_calls=8,
        expected_trajectory_tokens=512,
        max_trajectory_tokens=16384,
    )
    draft = StudyPlan(
        calls=(call,),
        price_snapshot={CORE_MODELS[0]: {"input_per_1m": 0.0, "output_per_1m": 0.0}},
        corpus_hash=corpus_fingerprint((row,)),
        phase=PHASE_A,
        status="ready",
    )
    plan = replace(draft, plan_hash=_canonical_hash(draft._payload()))
    path = tmp_path / "trajectory-plan.json"
    write_plan(plan, path)
    loaded = load_plan(path)
    loaded_call = loaded.calls[0]
    assert loaded_call.expected_turns == 2
    assert loaded_call.max_turns == 8
    assert loaded_call.potential_api_calls == 8
    assert loaded_call.max_trajectory_tokens == 16384
