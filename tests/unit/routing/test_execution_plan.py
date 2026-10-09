from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from theriad.routing.execution_plan import (
    ExecutionPlanError,
    assert_matches_locked_plan,
    assert_reviewed_hash,
    canonical_hash,
    load_execution_plan,
    materialize_execution_plan,
    write_execution_plan,
)
from theriad.routing.study import corpus_checksum
from theriad.routing.study_budget import StudyBudgetLedger
from theriad.routing.study_collector import (
    PHASE_A,
    PlannedStudyCall,
    StudyCollector,
    StudyPlan,
    _canonical_hash,
)


def _locked_plan() -> StudyPlan:
    call = PlannedStudyCall(
        prompt_id="train-1",
        model="nvidia/nemotron-3-super-120b-a12b:free",
        split="train",
        max_output_tokens=128,
        maximum_cost_usd=0.0,
        idempotency_key="train-1:model",
        input_tokens=32,
        expected_output_tokens=32,
        expected_cost_usd=0.0,
        task_format="exact_match",
        adapter_id="plain-text",
    )
    draft = StudyPlan(
        calls=(),
        planned_calls=(call,),
        price_snapshot={
            "nvidia/nemotron-3-super-120b-a12b:free": {
                "input_per_1m": 0.0,
                "output_per_1m": 0.0,
            }
        },
        corpus_hash="corpus-hash",
        phase=PHASE_A,
        status="locked_pending_provider_approval",
        cost_scenarios={
            "free_nemotron": {
                "status": "priced",
                "within_hard_cap": True,
                "untouched_test_reserve_preserved": True,
            }
        },
    )
    return replace(draft, plan_hash=_canonical_hash(draft._payload()))


def _executable(locked: StudyPlan):
    return materialize_execution_plan(
        locked,
        parent_phased_plan_hash="phased-parent-hash",
        executor_manifest_hash="manifest-hash",
        anchor_boundary_hash="anchor-hash",
        metadata_snapshot={
            "schema_version": 1,
            "requested_model_ids": ["nvidia/nemotron-3-super-120b-a12b:free"],
            "tool_model_ids": [],
            "models": {},
        },
        metadata_response_sha256="metadata-hash",
        metadata_fetched_at="2026-08-26T00:00:00+00:00",
    )


def _assert_parent(executable, locked):
    assert_matches_locked_plan(
        executable,
        locked,
        parent_phased_plan_hash="phased-parent-hash",
        corpus_hash="corpus-hash",
        executor_manifest_hash="manifest-hash",
        anchor_boundary_hash="anchor-hash",
    )


def test_locked_hash_cannot_authorize_materialized_plan_but_execution_hash_can():
    locked = _locked_plan()
    executable = _executable(locked)
    assert executable.execution_plan_hash != locked.plan_hash
    with pytest.raises(ExecutionPlanError):
        assert_reviewed_hash(executable, locked.plan_hash)
    assert_reviewed_hash(executable, executable.execution_plan_hash)
    _assert_parent(executable, locked)


def test_preflight_and_execution_read_byte_identical_immutable_contents(tmp_path: Path):
    executable = _executable(_locked_plan())
    path = tmp_path / "execution-plan.json"
    written = write_execution_plan(executable, path)
    assert written == path.read_bytes()
    loaded = load_execution_plan(path)
    assert loaded.to_dict() == executable.to_dict()
    assert path.read_bytes() == written


@pytest.mark.parametrize(
    ("field", "mutator"),
    [
        ("calls", lambda payload: payload["calls"][0].update({"prompt_id": "tampered"})),
        ("status", lambda payload: payload.update({"status": "locked"})),
        (
            "price_snapshot",
            lambda payload: payload["price_snapshot"][
                "nvidia/nemotron-3-super-120b-a12b:free"
            ].update({"output_per_1m": 1.0}),
        ),
        ("executor_manifest_hash", lambda payload: payload.update({"executor_manifest_hash": "other"})),
        ("corpus_hash", lambda payload: payload.update({"corpus_hash": "other"})),
    ],
)
def test_any_schedule_or_gate_tamper_fails_before_authorization(
    tmp_path: Path, field: str, mutator
):
    locked = _locked_plan()
    executable = _executable(locked)
    payload = executable.to_dict()
    mutator(payload)
    # Recompute the attacker-visible file hash to prove the parent comparison,
    # rather than only the file checksum, is the gate for every field.
    payload["execution_plan_hash"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "execution_plan_hash"}
    )
    path = tmp_path / f"tampered-{field}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if field == "status":
        with pytest.raises(ExecutionPlanError):
            load_execution_plan(path)
        return
    tampered = load_execution_plan(path)
    with pytest.raises(ExecutionPlanError):
        _assert_parent(tampered, locked)


class _Audit:
    def to_dict(self):
        return {"valid": True}


def _load_phase_cli_module():
    path = Path(__file__).parents[3] / "scripts/execute_router_v2_phase.py"
    spec = importlib.util.spec_from_file_location("router_v2_phase_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_rejects_locked_hash_before_constructing_collector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_phase_cli_module()
    base = _locked_plan()
    locked = replace(
        base,
        corpus_hash=corpus_checksum(()),
    )
    locked = replace(locked, plan_hash=_canonical_hash(locked._payload()))
    executable = _executable(locked)
    execution_path = tmp_path / "execution.json"
    write_execution_plan(executable, execution_path)
    monkeypatch.setattr(module, "load_jsonl", lambda _path: ())
    monkeypatch.setattr(module, "validate_corpus", lambda _prompts: _Audit())
    monkeypatch.setattr(
        module,
        "_load_phase",
        lambda _path, _phase: (locked, locked.plan_hash, "phased-parent-hash"),
    )
    monkeypatch.setattr(
        module,
        "_static_preflight",
        lambda *_args, **_kwargs: {
            "executor_manifest_hash": "manifest-hash",
            "anchor_boundary_hash": "anchor-hash",
        },
    )
    monkeypatch.setattr(module, "_metadata_for_plan", lambda *_args, **_kwargs: executable.metadata_snapshot)
    monkeypatch.setattr(
        module,
        "StudyCollector",
        lambda **_kwargs: pytest.fail("collector constructed before hash authorization"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "execute_router_v2_phase.py",
            "--phase",
            PHASE_A,
            "--corpus",
            str(tmp_path / "corpus.jsonl"),
            "--plan",
            str(tmp_path / "locked.json"),
            "--execution-plan",
            str(execution_path),
            "--state",
            str(tmp_path / "state.json"),
            "--budget-ledger",
            str(tmp_path / "budget.json"),
            "--execute",
            "--allow-provider-calls",
            "--yes",
            "--reviewed-plan-hash",
            locked.plan_hash,
        ],
    )
    assert module.main() == 2


def test_cli_exact_execution_hash_authorizes_without_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_phase_cli_module()
    base = _locked_plan()
    locked = replace(base, corpus_hash=corpus_checksum(()))
    locked = replace(locked, plan_hash=_canonical_hash(locked._payload()))
    executable = _executable(locked)
    execution_path = tmp_path / "execution.json"
    write_execution_plan(executable, execution_path)
    monkeypatch.setattr(module, "load_jsonl", lambda _path: ())
    monkeypatch.setattr(module, "validate_corpus", lambda _prompts: _Audit())
    monkeypatch.setattr(
        module,
        "_load_phase",
        lambda _path, _phase: (locked, locked.plan_hash, "phased-parent-hash"),
    )
    monkeypatch.setattr(
        module,
        "_static_preflight",
        lambda *_args, **_kwargs: {
            "executor_manifest_hash": "manifest-hash",
            "anchor_boundary_hash": "anchor-hash",
        },
    )
    monkeypatch.setattr(module, "_metadata_for_plan", lambda *_args, **_kwargs: executable.metadata_snapshot)
    monkeypatch.setattr(module, "_verify_execution_now", lambda **_kwargs: executable)

    class FakeCollector:
        def __init__(self, **_kwargs):
            self.constructed = True

        def execute(self, **_kwargs):
            return {
                "task_model_runs": 0,
                "provider_api_invocations": 0,
                "trajectory_turns": 0,
                "completed_graded_outcomes": 0,
                "reserved_cost_usd": 0.0,
                "realized_provider_cost_usd": 0.0,
                "actual_cost_usd": 0.0,
            }

    monkeypatch.setattr(module, "StudyCollector", FakeCollector)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "execute_router_v2_phase.py",
            "--phase",
            PHASE_A,
            "--corpus",
            str(tmp_path / "corpus.jsonl"),
            "--plan",
            str(tmp_path / "locked.json"),
            "--execution-plan",
            str(execution_path),
            "--state",
            str(tmp_path / "state.json"),
            "--budget-ledger",
            str(tmp_path / "budget.json"),
            "--execute",
            "--allow-provider-calls",
            "--yes",
            "--reviewed-plan-hash",
            executable.execution_plan_hash,
        ],
    )
    assert module.main() == 0


def test_snapshot_comparison_ignores_fetch_provenance_but_not_relevant_metadata():
    module = _load_phase_cli_module()
    frozen = {
        "schema_version": 2,
        "fetched_at": "before",
        "response_sha256": "relevant-hash",
        "catalog_response_sha256": "catalog-before",
        "models": {"model": {"input_per_1m": 0.0}},
    }
    refreshed = {
        **frozen,
        "fetched_at": "after",
        "catalog_response_sha256": "catalog-after",
    }
    assert module._metadata_matches_frozen_snapshot(refreshed, frozen)

    refreshed["models"] = {"model": {"input_per_1m": 1.0}}
    assert not module._metadata_matches_frozen_snapshot(refreshed, frozen)


def test_summary_keeps_task_runs_api_calls_turns_and_costs_distinct(tmp_path: Path):
    locked = _locked_plan()
    plan = replace(locked, calls=locked.planned_calls, status="ready")
    ledger = StudyBudgetLedger(tmp_path / "budget.json")
    first = ledger.reserve(
        phase="train",
        model=plan.calls[0].model,
        maximum_cost_usd=0.2,
        idempotency_key="summary-plain",
    )
    second = ledger.reserve(
        phase="train",
        model=plan.calls[0].model,
        maximum_cost_usd=0.3,
        idempotency_key="summary-tau",
        potential_api_calls=3,
        expected_turns=3,
        max_turns=3,
    )
    ledger.settle(first, 0.1)
    ledger.settle(second, 0.2)
    collector = StudyCollector(
        prompts=(),
        plan=plan,
        ledger=ledger,
        state_path=tmp_path / "state.json",
    )
    summary = collector.summary(
        {
            "rows": {
                "plain": {
                    "runs": {
                        plan.calls[0].model: {
                            "status": "success",
                            "adapter_id": "plain-text",
                            "api_call_count": 1,
                            "realized_cost_usd": 0.1,
                        }
                    },
                    "outcomes": {plan.calls[0].model: {"success": 1.0}},
                },
                "tau": {
                    "runs": {
                        plan.calls[0].model: {
                            "status": "success",
                            "adapter_id": "router-v2-tau-bench",
                            "api_call_count": 3,
                            "realized_cost_usd": 0.2,
                        }
                    },
                    "outcomes": {plan.calls[0].model: {"success": 1.0}},
                },
            }
        }
    )
    assert summary["task_model_runs"] == 2
    assert summary["provider_api_invocations"] == 4
    assert summary["trajectory_turns"] == 3
    assert summary["completed_graded_outcomes"] == 2
    assert summary["reserved_cost_usd"] == pytest.approx(0.5)
    assert summary["realized_provider_cost_usd"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("field", "mutator"),
    [
        ("call", lambda payload: payload["calls"][0].update({"model": "other"})),
        ("status", lambda payload: payload.update({"status": "locked"})),
        (
            "price",
            lambda payload: payload["price_snapshot"][
                "nvidia/nemotron-3-super-120b-a12b:free"
            ].update({"output_per_1m": 1.0}),
        ),
        ("manifest", lambda payload: payload.update({"executor_manifest_hash": "other"})),
        ("corpus", lambda payload: payload.update({"corpus_hash": "other"})),
    ],
)
def test_cli_tampering_fails_before_provider_construction_or_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, mutator
):
    module = _load_phase_cli_module()
    base = _locked_plan()
    locked = replace(base, corpus_hash=corpus_checksum(()))
    locked = replace(locked, plan_hash=_canonical_hash(locked._payload()))
    executable = _executable(locked)
    payload = executable.to_dict()
    mutator(payload)
    payload["execution_plan_hash"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "execution_plan_hash"}
    )
    execution_path = tmp_path / f"tampered-{field}.json"
    execution_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(module, "load_jsonl", lambda _path: ())
    monkeypatch.setattr(module, "validate_corpus", lambda _prompts: _Audit())
    monkeypatch.setattr(
        module,
        "_load_phase",
        lambda _path, _phase: (locked, locked.plan_hash, "phased-parent-hash"),
    )
    monkeypatch.setattr(
        module,
        "_static_preflight",
        lambda *_args, **_kwargs: {
            "executor_manifest_hash": "manifest-hash",
            "anchor_boundary_hash": "anchor-hash",
        },
    )
    monkeypatch.setattr(
        module,
        "_metadata_for_plan",
        lambda *_args, **_kwargs: executable.metadata_snapshot,
    )
    monkeypatch.setattr(
        module,
        "StudyCollector",
        lambda **_kwargs: pytest.fail("collector constructed after execution-plan tamper"),
    )
    budget_path = tmp_path / "budget.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "execute_router_v2_phase.py",
            "--phase",
            PHASE_A,
            "--corpus",
            str(tmp_path / "corpus.jsonl"),
            "--plan",
            str(tmp_path / "locked.json"),
            "--execution-plan",
            str(execution_path),
            "--state",
            str(tmp_path / "state.json"),
            "--budget-ledger",
            str(budget_path),
            "--execute",
            "--allow-provider-calls",
            "--yes",
            "--reviewed-plan-hash",
            payload["execution_plan_hash"],
        ],
    )
    assert module.main() == 2
    assert not budget_path.exists() or json.loads(budget_path.read_text(encoding="utf-8"))["entries"] == []
