#!/usr/bin/env python3
"""Exercise the real Phase-A collector with provider-free runtime fixtures.

This rehearsal intentionally supplies only in-process fake providers.  It
does not construct OpenRouter, invoke an HTTP transport, unlock a paid plan,
or read any validation/test outcomes.  The collector, adapters, sandbox code,
ledger reservations, persistence, and retry gates are nevertheless exercised
through their production code paths.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.core.providers.interfaces import CompletionResponse  # noqa: E402
from theriad.routing.study import CORE_MODELS, StudyPrompt, corpus_checksum  # noqa: E402
from theriad.routing.study_adapters import (  # noqa: E402
    StructuredCompletionRequest,
    StructuredCompletionResponse,
    StructuredToolCall,
)
from theriad.routing.study_budget import StudyBudgetExceeded, StudyBudgetLedger  # noqa: E402
from theriad.routing.study_collector import (  # noqa: E402
    PHASE_A,
    PlannedStudyCall,
    StudyCollector,
    StudyCollectorError,
    StudyPlan,
    _canonical_hash,
    corpus_fingerprint,
)

REPORT_PATH = ROOT / "reports/router-v2/router-v2-zero-spend-runtime-rehearsal.json"
STATE_PATH = ROOT / "reports/router-v2/router-v2-zero-spend-runtime-rehearsal-state.json"
BUDGET_PATH = ROOT / "reports/router-v2/router-v2-zero-spend-runtime-rehearsal-budget.json"


def _prompt(
    prompt_id: str,
    prompt: str,
    *,
    source: str = "fixture-general",
    lane: str = "general",
    label_type: str = "deterministic",
    task_format: str = "exact_match",
    grader: dict[str, Any] | None = None,
) -> StudyPrompt:
    return StudyPrompt(
        prompt_id=prompt_id,
        prompt=prompt,
        split="train",
        source=source,
        benchmark_family=source,
        domain="rehearsal",
        template_family=source,
        collection_vintage="2026-08-25",
        license_name="internal-fixture",
        license_status="internal",
        source_revision="fixture-v1",
        provenance_url="https://example.invalid/router-v2-rehearsal",
        lane=lane,
        label_type=label_type,
        task_format=task_format,
        leakage_group=f"rehearsal:{prompt_id}",
        tags=("representative",),
        grader=grader or {"answer": "A", "answer_type": "exact_match"},
    )


def rehearsal_rows() -> tuple[StudyPrompt, ...]:
    rows = [
        _prompt("ordinary-1", "Return the letter A.") ,
        _prompt(
            "code-1",
            "Implement increment(x) so the executable tests pass.",
            source="fixture-code",
            lane="code",
            task_format="code_execution",
            grader={
                "kind": "sandboxed_code_tests",
                "code_task_type": "function_implementation",
                "entry_point": "increment",
                "executable_tests": "def check(increment):\n    assert increment(2) == 3\n",
                "test_harness_id": "rehearsal-harness",
                "test_harness_version": "rehearsal-v1",
            },
        ),
    ]
    bfcl_modes = {
        "simple": ([{"name": "lookup", "arguments": {"value": 1}}], [{"name": "lookup"}]),
        "multiple": (
            [
                {"name": "lookup", "arguments": {"value": 1}},
                {"name": "summarize", "arguments": {"value": 2}},
            ],
            [{"name": "lookup"}, {"name": "summarize"}],
        ),
        "parallel": (
            [
                {"name": "lookup", "arguments": {"value": 1}},
                {"name": "lookup", "arguments": {"value": 2}},
            ],
            [{"name": "lookup"}],
        ),
        "irrelevance": ([], [{"name": "lookup"}]),
    }
    for mode, (ground_truth, schemas) in bfcl_modes.items():
        rows.append(
            _prompt(
                f"bfcl-{mode}",
                f"BFCL {mode} structured tool fixture.",
                source="bfcl-v4",
                task_format="tool_use",
                label_type="tool_state",
                grader={
                    "kind": "bfcl_official_checker",
                    "executor_id": "fixture-bfcl-checker",
                    "executor_version": "fixture-v1",
                    "evidence_policy": "provider-free-authoritative-fixture",
                    "tool_schemas": [
                        {
                            "name": schema["name"],
                            "description": "rehearsal tool",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "integer"}},
                            },
                        }
                        for schema in schemas
                    ],
                    "official_checker_id": "fixture-bfcl-checker",
                    "official_checker_version": "fixture-v1",
                    "official_checker_available": True,
                    "task_mode": mode,
                    "source_task_id": f"fixture-{mode}",
                    "ground_truth": ground_truth,
                },
            )
        )
    rows.append(
        _prompt(
            "tau2-agent",
            "Close the ticket using the interactive environment.",
            source="tau3",
            task_format="agent_state",
            label_type="tool_state",
            grader={
                "kind": "tau2_official_environment",
                "executor_id": "fixture-tau2",
                "executor_version": "fixture-v1",
                "evidence_policy": "provider-free-authoritative-fixture",
                "official_environment_id": "fixture-tau2-environment",
                "official_environment_available": True,
                "official_evaluator_id": "fixture-tau2-evaluator",
                "official_evaluator_version": "fixture-v1",
                "initial_state": {"status": "open", "actions": []},
                "expected_turns": 2,
                "max_turns": 3,
            },
        )
    )
    return tuple(rows)


class FakeOfficialBFCLChecker:
    available = True

    def check(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "official_checker_id": "fixture-bfcl-checker",
            "official_checker_revision": "fixture-v1",
            "official_checker_result": {"valid": True, "error": []},
            "authoritative": True,
        }


class FakeTauEnvironment:
    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def reset(self, initial_state: dict[str, Any]) -> None:
        self._state = dict(initial_state)
        self._state.setdefault("actions", [])

    def restore_state(self, state: dict[str, Any]) -> None:
        self._state = dict(state)

    def restore_trajectory(self, trajectory: list[dict[str, Any]]) -> None:
        for record in trajectory:
            for call in record.get("tool_calls") or ():
                self.apply_tool_call(
                    StructuredToolCall(
                        str(call["name"]),
                        dict(call.get("arguments") or {}),
                        int(call.get("order", 0)),
                    )
                )

    def tool_schemas(self) -> tuple[dict[str, Any], ...]:
        return ({"name": "close_ticket", "parameters": {"type": "object"}},)

    def apply_actions(self, actions: tuple[dict[str, Any], ...]) -> None:
        for action in actions:
            self.apply_tool_call(
                StructuredToolCall(str(action["name"]), dict(action.get("arguments") or {}), 0)
            )

    def apply_tool_call(self, call: StructuredToolCall) -> dict[str, Any]:
        self._state.setdefault("actions", []).append(call.to_dict())
        self._state["status"] = "closed"
        return {"ok": True, "state": dict(self._state)}

    def state(self) -> dict[str, Any]:
        return dict(self._state)

    def evaluate(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        return {
            "passed": self._state.get("status") == "closed",
            "evaluator_id": "fixture-tau2-evaluator",
            "trajectory_turns": len(trajectory.get("trajectory") or ()),
            "state": dict(self._state),
        }


class FakeProvider:
    name = "provider-free-rehearsal"
    supports_tool_calls = True
    supports_agent_tools = True

    def __init__(self) -> None:
        self.ordinary_calls = 0
        self.structured_calls = 0

    async def complete(self, request: Any) -> CompletionResponse:
        self.ordinary_calls += 1
        content = (
            "```python\ndef increment(x):\n    return x + 1\n```"
            if "increment" in request.messages[0].text
            else "A"
        )
        return CompletionResponse(
            model=request.model,
            content=content,
            prompt_tokens=4,
            completion_tokens=4,
            latency_ms=0.1,
            provider=self.name,
            raw={"usage": {"cost": 0.0}},
            cost_usd=0.0,
        )

    async def complete_with_tools(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResponse:
        self.structured_calls += 1
        prompt_id = str(request.metadata.get("prompt_id") or "")
        if request.metadata.get("task_format") == "agent_state":
            calls = () if any(message.role == "tool" for message in request.messages) else (
                StructuredToolCall("close_ticket", {}, 0, call_id="tau-call"),
            )
        elif prompt_id.endswith("irrelevance"):
            calls = ()
        else:
            calls = tuple(
                StructuredToolCall(
                    str(schema["name"]),
                    {"value": index + 1},
                    index,
                    call_id=f"bfcl-{index}",
                    parallel_group="parallel-0" if prompt_id.endswith("parallel") else "",
                )
                for index, schema in enumerate(request.tool_schemas)
            )
        return StructuredCompletionResponse(
            model=request.model,
            content="",
            tool_calls=calls,
            prompt_tokens=5,
            completion_tokens=3,
            latency_ms=0.1,
            provider=self.name,
            raw={"usage": {"cost": 0.0}},
            cost_usd=0.0,
        )


class NoToolProvider:
    name = "provider-free-no-tools"
    supports_tool_calls = False
    supports_agent_tools = False
    complete = None


class FailsOnceProvider:
    name = "provider-free-interrupted"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, _request: Any) -> CompletionResponse:
        self.calls += 1
        raise RuntimeError("provider-free interruption fixture")


def _plan(rows: tuple[StudyPrompt, ...]) -> StudyPlan:
    calls: list[PlannedStudyCall] = []
    for row in rows:
        is_tau = row.task_format == "agent_state"
        expected_turns = 2 if is_tau else 1
        max_turns = 3 if is_tau else 1
        for model in CORE_MODELS:
            calls.append(
                PlannedStudyCall(
                    prompt_id=row.prompt_id,
                    model=model,
                    split="train",
                    max_output_tokens=int(row.max_output_tokens or 1),
                    maximum_cost_usd=0.0,
                    idempotency_key=f"rehearsal:{row.prompt_id}:{model}",
                    input_tokens=8,
                    expected_output_tokens=int(row.expected_output_tokens or 1),
                    expected_cost_usd=0.0,
                    task_format=row.task_format,
                    adapter_id="rehearsal",
                    expected_turns=expected_turns,
                    max_turns=max_turns,
                    potential_api_calls=max_turns,
                    expected_trajectory_tokens=expected_turns * int(row.expected_output_tokens or 1),
                    max_trajectory_tokens=max_turns * int(row.max_output_tokens or 1),
                )
            )
    draft = StudyPlan(
        calls=tuple(calls),
        price_snapshot={
            model: {"input_per_1m": 0.0, "output_per_1m": 0.0, "source": "fake"}
            for model in CORE_MODELS
        },
        corpus_hash=corpus_fingerprint(rows),
        phase=PHASE_A,
        status="ready",
        preconditions=("corpus_validated", "no_provider_calls_in_prior_phase"),
        potential_call_count=sum(call.potential_api_calls for call in calls),
        cost_scenarios={
            "free_nemotron": {
                "status": "priced",
                "within_hard_cap": True,
                "untouched_test_reserve_preserved": True,
                "expected_total_cost_usd": 0.0,
                "worst_case_total_cost_usd": 0.0,
            }
        },
    )
    return replace(draft, plan_hash=_canonical_hash(draft._payload()))


def _collector(
    rows: tuple[StudyPrompt, ...],
    plan: StudyPlan,
    ledger_path: Path,
    state_path: Path,
    provider: Any,
    *,
    adapter_context: dict[str, Any] | None = None,
) -> StudyCollector:
    return StudyCollector(
        prompts=rows,
        plan=plan,
        ledger=StudyBudgetLedger(ledger_path),
        state_path=state_path,
        provider_factory=lambda: provider,
        adapter_context=adapter_context or {},
    )


def _failure_case(
    row: StudyPrompt,
    provider: Any,
    root: Path,
    *,
    adapter_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = (row,)
    plan = _plan(rows)
    state_path = root / f"{row.prompt_id}-state.json"
    ledger_path = root / f"{row.prompt_id}-budget.json"
    collector = _collector(
        rows,
        plan,
        ledger_path,
        state_path,
        provider,
        adapter_context=adapter_context,
    )
    try:
        collector.execute(
            reviewed_plan_hash=plan.plan_hash,
            allow_provider_calls=True,
            allow_paid_calls=False,
            yes=True,
            retry_errors=True,
        )
    except Exception as exc:  # expected fixture failure
        state = json.loads(state_path.read_text(encoding="utf-8"))
        row_state = state["rows"].get(row.prompt_id, {})
        return {
            "exception": f"{type(exc).__name__}: {exc}",
            "run": row_state.get("runs", {}).get(CORE_MODELS[0]),
            "ledger": StudyBudgetLedger(ledger_path).status(),
            "provider_calls": int(getattr(provider, "calls", 0)),
        }
    raise AssertionError(f"failure fixture unexpectedly completed: {row.prompt_id}")


def run() -> dict[str, Any]:
    rows = rehearsal_rows()
    plan = _plan(rows)
    provider = FakeProvider()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # These are this script's own provider-free evidence files.  Start a
    # rehearsal from a clean fixture ledger/state so a prior interrupted local
    # rehearsal cannot be mistaken for an external reservation.
    for path in (
        STATE_PATH,
        BUDGET_PATH,
        STATE_PATH.with_suffix(STATE_PATH.suffix + ".lock"),
        BUDGET_PATH.with_suffix(BUDGET_PATH.suffix + ".lock"),
    ):
        path.unlink(missing_ok=True)
    with TemporaryDirectory(prefix="theriad-router-v2-rehearsal-fixtures-") as failure_dir:
        failure_root = Path(failure_dir)
        state_path = STATE_PATH
        budget_path = BUDGET_PATH
        collector = _collector(
            rows,
            plan,
            budget_path,
            state_path,
            provider,
            adapter_context={
                "official_checker": FakeOfficialBFCLChecker(),
                "environment_factory": lambda _row: FakeTauEnvironment(),
            },
        )
        first_summary = collector.execute(
            reviewed_plan_hash=plan.plan_hash,
            allow_provider_calls=True,
            allow_paid_calls=False,
            yes=True,
        )
        calls_after_first = provider.ordinary_calls + provider.structured_calls
        second_summary = collector.execute(
            reviewed_plan_hash=plan.plan_hash,
            allow_provider_calls=True,
            allow_paid_calls=False,
            yes=True,
        )
        calls_after_retry = provider.ordinary_calls + provider.structured_calls

        state = json.loads(state_path.read_text(encoding="utf-8"))
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        runs = [
            run
            for row_state in state["rows"].values()
            for run in (row_state.get("runs") or {}).values()
        ]
        outcomes = [
            outcome
            for row_state in state["rows"].values()
            for outcome in (row_state.get("outcomes") or {}).values()
        ]
        if len(runs) != len(rows) * len(CORE_MODELS):
            raise AssertionError("rehearsal did not persist every task/model run")
        if len(outcomes) != len(runs):
            raise AssertionError("rehearsal did not persist an explicit outcome for every run")
        if calls_after_first != calls_after_retry:
            raise AssertionError("completed Phase-A retry issued duplicate fake provider calls")

        import theriad.routing.sandbox as sandbox

        original_sandbox_available = sandbox.sandbox_available
        sandbox.sandbox_available = lambda: False
        try:
            sandbox_block = _failure_case(
                next(row for row in rows if row.task_format == "code_execution"),
                FakeProvider(),
                failure_root,
            )
        finally:
            sandbox.sandbox_available = original_sandbox_available

        capability_block = _failure_case(
            next(row for row in rows if row.source == "bfcl-v4"),
            NoToolProvider(),
            failure_root,
            adapter_context={"official_checker": FakeOfficialBFCLChecker()},
        )

        interrupted_provider = FailsOnceProvider()
        interrupted_row = next(row for row in rows if row.prompt_id == "ordinary-1")
        interrupted_rows = (replace(interrupted_row, prompt_id="interrupted-1"),)
        interrupted_plan = _plan(interrupted_rows)
        interrupted_state = failure_root / "interrupted-state.json"
        interrupted_budget = failure_root / "interrupted-budget.json"
        interrupted_collector = _collector(
            interrupted_rows,
            interrupted_plan,
            interrupted_budget,
            interrupted_state,
            interrupted_provider,
        )
        try:
            interrupted_collector.execute(
                reviewed_plan_hash=interrupted_plan.plan_hash,
                allow_provider_calls=True,
                allow_paid_calls=False,
                yes=True,
                retry_errors=True,
            )
        except RuntimeError:
            pass
        try:
            interrupted_collector.execute(
                reviewed_plan_hash=interrupted_plan.plan_hash,
                allow_provider_calls=True,
                allow_paid_calls=False,
                yes=True,
                retry_errors=True,
            )
        except StudyCollectorError as exc:
            interrupted_retry = {
                "exception": f"{type(exc).__name__}: {exc}",
                "provider_calls": interrupted_provider.calls,
                "ledger": StudyBudgetLedger(interrupted_budget).status(),
            }
        else:
            raise AssertionError("ambiguous interrupted reservation was retried")

        overflow_ledger = StudyBudgetLedger(failure_root / "overflow-budget.json")
        try:
            overflow_ledger.reserve(
                phase="train",
                model="fixture",
                maximum_cost_usd=16.0,
                idempotency_key="overflow",
            )
        except StudyBudgetExceeded as exc:
            overflow = {"exception": f"{type(exc).__name__}: {exc}", "provider_calls": 0}
        else:
            raise AssertionError("budget overflow fixture unexpectedly reserved")

        mismatch_ledger = StudyBudgetLedger(failure_root / "idempotency-budget.json")
        first_reservation = mismatch_ledger.reserve(
            phase="train",
            model="fixture",
            maximum_cost_usd=0.0,
            idempotency_key="same-key",
            potential_api_calls=1,
            max_turns=1,
        )
        try:
            mismatch_ledger.reserve(
                phase="train",
                model="fixture",
                maximum_cost_usd=0.0,
                idempotency_key="same-key",
                potential_api_calls=2,
                max_turns=2,
            )
        except StudyBudgetExceeded as exc:
            idempotency_mismatch = {
                "first_reservation": first_reservation,
                "exception": f"{type(exc).__name__}: {exc}",
            }
        else:
            raise AssertionError("trajectory idempotency mismatch was accepted")

        report = {
            "schema_version": 1,
            "status": "passed",
            "phase": PHASE_A,
            "corpus": {
                "row_count": len(rows),
                "task_model_runs": len(plan.calls),
                "potential_api_call_count": plan.potential_api_call_count,
                "corpus_sha256": corpus_checksum(rows),
                "task_formats": sorted({row.task_format for row in rows}),
            },
            "collector": {
                "plan_hash": plan.plan_hash,
                "phase_a_usable_graded_coverage": 1.0,
                "runs_persisted": len(runs),
                "explicit_outcomes_persisted": len(outcomes),
                "adapter_statuses": sorted(
                    {str(run.get("adapter_status") or "plain-text") for run in runs}
                ),
                "tau2_actual_api_calls": sum(
                    int(run.get("api_call_count") or 0)
                    for run in runs
                    if run.get("adapter_id") == "router-v2-tau-bench"
                ),
                "fake_provider_method_invocations": calls_after_first,
                "external_provider_calls": 0,
                "provider_spend_usd": 0.0,
                "retry_behavior": {
                    "second_execute_skipped_completed_runs": True,
                    "calls_after_first": calls_after_first,
                    "calls_after_retry": calls_after_retry,
                },
                "summary_first": first_summary,
                "summary_retry": second_summary,
            },
            "fixtures": {
                "bfcl_modes": ["simple", "multiple", "parallel", "irrelevance"],
                "bfcl_official_checker": "injected provider-free authoritative fixture; pinned official smoke is reported separately",
                "tau2": "interactive two-turn fake environment with persisted state and evaluator",
                "code": "strict fenced Python response graded by the real sandbox",
            },
            "safety_cases": {
                "capability_block_before_reservation": capability_block,
                "sandbox_block_before_reservation": sandbox_block,
                "interrupted_retry_refused": interrupted_retry,
                "budget_overflow_refused": overflow,
                "trajectory_idempotency_mismatch_refused": idempotency_mismatch,
            },
            "ledger": {
                "path": str(BUDGET_PATH),
                "entries": len(budget.get("entries", [])),
                "status": StudyBudgetLedger(BUDGET_PATH).status(),
            },
            "provider_calls": 0,
            "provider_spend_usd": 0.0,
            "phase_a_unlocked": False,
            "validation_outcomes_opened": False,
            "test_outcomes_opened": False,
            "production_artifact_created": False,
            "router_v2_enabled": False,
        }
        REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report


def main() -> int:
    report = run()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
