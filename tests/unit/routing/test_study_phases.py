from __future__ import annotations

import sys

import pytest

from theriad.routing.sandbox import CODE_GRADER_RUNTIME_ENV
from theriad.routing.study import CORE_MODELS
from theriad.routing.study_collector import (
    PHASE_A,
    PHASE_B,
    PHASE_C,
    PHASE_D,
    StudyCollectorError,
    assert_phase_gate,
    build_phase_plan,
    build_phased_study_plan,
)

from .test_study import _valid_corpus


@pytest.fixture(autouse=True)
def _pin_fixture_code_grader_runtime(monkeypatch):
    import theriad.routing.sandbox as sandbox_module

    monkeypatch.setenv(CODE_GRADER_RUNTIME_ENV, sys.executable)
    monkeypatch.setattr(sandbox_module, "sandbox_available", lambda: True)
    monkeypatch.setattr(sandbox_module, "runtime_supported", lambda runtime=None: True)


def _complete_phase_a_evidence(rows):
    return {
        row.prompt_id: {
            model: {"success": int((index + model_index) % 2 == 0)}
            for model_index, model in enumerate(CORE_MODELS)
        }
        for index, row in enumerate(rows)
        if row.split == "train"
    }


def test_phased_plan_keeps_phase_a_unmaterialized_before_provider_approval():
    rows = _valid_corpus()
    plan = build_phased_study_plan(rows)
    assert plan.phases[PHASE_A].call_count == 0
    assert plan.phases[PHASE_A].planned_call_count == 3_600
    assert plan.phases[PHASE_A].status == "locked_pending_provider_approval"
    assert plan.phases[PHASE_B].call_count == 0
    assert plan.phases[PHASE_B].planned_call_count == 200
    assert plan.phases[PHASE_C].call_count == 0
    assert plan.phases[PHASE_C].planned_call_count == 1_600
    assert plan.phases[PHASE_D].call_count == 0
    assert plan.phases[PHASE_D].planned_call_count == 1_600
    assert plan.to_dict()["provider_calls_materialized"] == 0
    assert plan.to_dict()["planned_provider_call_count"] == 7_000
    assert plan.grok_algorithm["prompt_ids"] == []
    assert plan.grok_algorithm["selected_ids_status"] == "unresolved_until_complete_phase_A_outcomes"
    assert plan.phases[PHASE_A].grok_selection["prompt_ids"] == []
    assert plan.phases[PHASE_A].grok_selection["selected_metrics"] == {}


def test_phase_a_reservation_multiplier_is_hash_bound_and_scenario_priced():
    rows = _valid_corpus()
    baseline = build_phased_study_plan(rows)
    buffered = build_phased_study_plan(
        rows,
        phase_reservation_multipliers={
            PHASE_A: {"xiaomi/mimo-v2.5": 5.0},
        },
    )

    baseline_a = baseline.phases[PHASE_A]
    buffered_a = buffered.phases[PHASE_A]
    baseline_mimo = {
        call.prompt_id: call.maximum_cost_usd
        for call in baseline_a.planned_calls
        if call.model == "xiaomi/mimo-v2.5"
    }
    buffered_mimo = {
        call.prompt_id: call.maximum_cost_usd
        for call in buffered_a.planned_calls
        if call.model == "xiaomi/mimo-v2.5"
    }

    assert buffered.plan_hash != baseline.plan_hash
    assert buffered_a.plan_hash != baseline_a.plan_hash
    assert buffered_mimo.keys() == baseline_mimo.keys()
    assert all(
        buffered_mimo[prompt_id] == pytest.approx(maximum * 5.0)
        for prompt_id, maximum in baseline_mimo.items()
    )
    assert (
        buffered_a.cost_scenarios["free_nemotron"]["worst_case_model_cost_usd"]
        > baseline_a.cost_scenarios["free_nemotron"]["worst_case_model_cost_usd"]
    )
    assert (
        buffered.phases[PHASE_C].planned_maximum_cost_usd
        == baseline.phases[PHASE_C].planned_maximum_cost_usd
    )


def test_phase_a_reservation_multiplier_can_target_one_task_format():
    rows = _valid_corpus()
    baseline = build_phased_study_plan(rows)
    buffered = build_phased_study_plan(
        rows,
        phase_reservation_multipliers={
            PHASE_A: {"xiaomi/mimo-v2.5@multiple_choice": 5.0},
        },
    )
    baseline_calls = {
        (call.prompt_id, call.model): call
        for call in baseline.phases[PHASE_A].planned_calls
    }
    buffered_calls = {
        (call.prompt_id, call.model): call
        for call in buffered.phases[PHASE_A].planned_calls
    }

    for key, call in baseline_calls.items():
        revised = buffered_calls[key]
        expected_factor = (
            5.0
            if call.model == "xiaomi/mimo-v2.5"
            and call.task_format == "multiple_choice"
            else 1.0
        )
        assert revised.maximum_cost_usd == pytest.approx(
            call.maximum_cost_usd * expected_factor
        )


def test_grok_ids_and_calls_unlock_only_after_complete_phase_a_outcomes():
    rows = _valid_corpus()
    evidence = _complete_phase_a_evidence(rows)
    locked = build_phase_plan(rows, phase=PHASE_B)
    assert locked.status.startswith("locked")
    with pytest.raises(StudyCollectorError, match="locked"):
        assert_phase_gate(locked, {"phase_status": {}})

    unlocked = build_phase_plan(
        rows,
        phase=PHASE_B,
        outcome_evidence=evidence,
        state={"phase_status": {PHASE_A: "complete"}},
    )
    assert unlocked.status == "ready"
    assert unlocked.call_count == 200
    assert all(call.split == "train" for call in unlocked.calls)
    assert_phase_gate(unlocked, {"phase_status": {PHASE_A: "complete"}})


def test_validation_and_test_are_hard_gated():
    rows = _valid_corpus()
    validation = build_phase_plan(rows, phase=PHASE_C)
    with pytest.raises(StudyCollectorError, match="locked"):
        assert_phase_gate(validation, {"phase_status": {}})
    test = build_phase_plan(
        rows,
        phase=PHASE_D,
        state={"phase_status": {PHASE_A: "complete", PHASE_B: "complete", PHASE_C: "complete"}},
    )
    assert test.call_count == 0
    with pytest.raises(StudyCollectorError, match="locked"):
        assert_phase_gate(
            test,
            {"phase_status": {PHASE_A: "complete", PHASE_B: "complete", PHASE_C: "complete"}},
        )

    manifest = {"status": "ready_for_test", "manifest_hash": "frozen"}
    unlocked = build_phase_plan(
        rows,
        phase=PHASE_D,
        state={"phase_status": {PHASE_A: "complete", PHASE_B: "complete", PHASE_C: "complete"}},
        final_freeze_manifest=manifest,
    )
    assert unlocked.call_count == 1_600
    assert all(call.split == "test" for call in unlocked.calls)
    assert_phase_gate(unlocked, {"phase_status": {PHASE_C: "complete"}}, final_freeze_manifest=manifest)
