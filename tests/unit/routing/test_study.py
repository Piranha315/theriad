from __future__ import annotations

import json
from dataclasses import replace

import pytest

from theriad.routing.study import (
    CORE_MODELS,
    StudyCorpusError,
    StudyPrompt,
    audit_corpus,
    corpus_checksum,
    grok_selection_metadata,
    load_jsonl,
    select_informative_training,
    study_call_scope,
    validate_corpus,
)
from theriad.routing.study_budget import StudyBudgetLedger
from theriad.routing.study_collector import build_study_plan
from theriad.routing.study_grading import (
    grade_response,
    merge_deterministic_and_judge,
    randomize_pairwise_positions,
)


def _prompt(index: int, split: str, label_type: str = "deterministic") -> StudyPrompt:
    tags: set[str] = set()
    if index < 100:
        tags.add("short-hard")
    elif index < 200:
        tags.add("long-easy")
    elif index < 300:
        tags.add("specialist")
    elif index < 400:
        tags.add("ambiguous")
    elif index < 500:
        tags.add("disagreement-candidate")
    if split == "test":
        if index - 1_600 < 100:
            tags.add("ood")
        elif index - 1_600 < 200:
            tags.add("representative")
    if label_type == "tool_state":
        grader = {
            "kind": "trusted_executor",
            "executor_id": "test-executor",
            "executor_version": "test-v1",
            "evidence_policy": "sha256_trusted_result",
        }
    elif label_type == "pairwise":
        grader = {
            "kind": "frozen_pairwise_judge",
            "judge_id": "test-judge",
            "judge_version": "test-v1",
            "criteria_version": "test-criteria-v1",
            "position_randomization": "prompt_id_seeded_sha256",
            "sampling_settings": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64},
            "tie_policy": "record_tie_without_forcing_binary_label",
            "abstention_policy": "judge_may_abstain_on rubric_or_context_failure",
            "disagreement_review_policy": "independent_manual_review_then_adjudication; no automatic overwrite",
            "judging_cost_status": "postponed_unpriced",
            "expected_judging_cost_usd": None,
            "worst_case_judging_cost_usd": None,
        }
    else:
        grader = {"answer": "A", "answer_type": "multiple_choice"}
    lane = "code" if index % 5 == 0 else "general"
    task_format = (
        "code_execution"
        if lane == "code"
        else "multiple_choice"
        if index % 3 == 0
        else "exact_match"
        if index % 3 == 1
        else "conversational_reasoning"
    )
    if lane == "code":
        grader = {
            **grader,
            "code_task_type": "fixture_implementation",
            "executable_tests": ["fixture_test_command"],
            "test_harness_id": "router-v2-fixture-harness",
            "test_harness_version": "fixture-v1",
        }
    return StudyPrompt(
        prompt_id=f"p-{index}",
        prompt=f"unique study prompt {index}",
        split=split,
        source=f"source-{index}",
        benchmark_family=f"family-{index}",
        domain=f"domain-{index % 4}",
        template_family=f"template-{index}",
        collection_vintage=f"vintage-{index}",
        license_name="test-license",
        license_status="approved",
        source_revision="revision-1",
        provenance_url="https://example.test/source",
        lane=lane,
        label_type=label_type,
        task_format=task_format,
        leakage_group=f"leakage-{index}",
        tags=tuple(sorted(tags)),
        grader=grader,
    )


def _valid_corpus() -> tuple[StudyPrompt, ...]:
    rows: list[StudyPrompt] = []
    for index in range(2_000):
        split = "train" if index < 1_200 else "validation" if index < 1_600 else "test"
        label = "deterministic" if index < 1_400 else "tool_state"
        rows.append(_prompt(index, split, label))
    return tuple(rows)


def test_exact_study_contract_and_coverage_scope():
    rows = _valid_corpus()
    audit = validate_corpus(rows)
    assert audit.valid
    assert audit.split_counts == {"train": 1_200, "validation": 400, "test": 400}
    assert audit.label_counts == {"deterministic": 1_400, "tool_state": 600}
    scope = study_call_scope(rows)
    assert {model: len(items) for model, items in scope.items()} == {
        "nvidia/nemotron-3-super-120b-a12b:free": 2_000,
        "xiaomi/mimo-v2.5": 2_000,
        "openai/gpt-5.6-luna": 2_000,
        "x-ai/grok-4.6": 800,
    }
    assert len(select_informative_training(rows)) == 200
    assert "model-disagreement" not in audit.stratum_counts


def test_authentic_disagreement_is_not_a_precollection_gate():
    rows = _valid_corpus()
    audit = validate_corpus(rows)
    assert audit.valid
    assert audit.test_stratum_counts["ood"] == 100
    assert audit.test_stratum_counts["representative"] == 100


def test_group_leakage_is_reported_and_rejected():
    rows = list(_valid_corpus())
    rows[1_600] = replace(rows[1_600], leakage_group=rows[0].leakage_group)
    audit = audit_corpus(rows)
    assert any(row["kind"] == "leakage_group" for row in audit.leakage)
    with pytest.raises(StudyCorpusError, match="failed promotion contract"):
        validate_corpus(rows)


@pytest.mark.parametrize("kind", ["prompt_hash", "leakage_group"])
def test_audit_reports_each_leakage_class(kind: str):
    first = _prompt(1, "train")
    second = _prompt(2, "test")
    if kind == "prompt_hash":
        second = replace(second, prompt=first.prompt)
    elif kind == "leakage_group":
        second = replace(second, leakage_group=first.leakage_group)
    audit = audit_corpus([first, second], expected_counts={"train": 1, "test": 1})
    assert any(row["kind"] == kind for row in audit.leakage)


def test_malformed_rejections_fail_closed_even_when_counts_match():
    rows = _valid_corpus()
    audit = audit_corpus(rows, rejected=[{"line": 9, "reason": "malformed_json"}])
    assert not audit.valid
    with pytest.raises(StudyCorpusError, match="rejected=1"):
        validate_corpus(rows, rejected=[{"line": 9, "reason": "malformed_json"}])


@pytest.mark.parametrize(
    ("label_type", "grader"),
    [
        ("deterministic", {}),
        ("tool_state", {"kind": "trusted_executor"}),
        ("pairwise", {"kind": "randomized_pairwise"}),
    ],
)
def test_label_contract_rejects_non_executable_grader(label_type: str, grader: dict):
    with pytest.raises(ValueError, match="require"):
        replace(_prompt(1, "train", label_type), grader=grader)


def test_plan_is_exact_and_hashable():
    plan = build_study_plan(_valid_corpus(), max_output_tokens=256)
    assert plan.call_count == 6_800
    assert plan.by_phase()["train"]["calls"] == 3_600
    assert plan.by_phase()["validation"]["calls"] == 1_600
    assert plan.by_phase()["test"]["calls"] == 1_600
    assert len(plan.plan_hash) == 64


def test_test_outcomes_cannot_change_grok_selection_or_call_scope():
    rows = _valid_corpus()
    test_ids = [row.prompt_id for row in rows if row.split == "test"]
    first_evidence = {
        prompt_id: {CORE_MODELS[0]: {"success": False}}
        for prompt_id in test_ids
    }
    second_evidence = {
        prompt_id: {CORE_MODELS[0]: {"success": True}, CORE_MODELS[1]: {"success": False}}
        for prompt_id in test_ids
    }
    first = build_study_plan(rows, outcome_evidence=first_evidence)
    second = build_study_plan(rows, outcome_evidence=second_evidence)
    assert first.grok_selection["prompt_ids"] == second.grok_selection["prompt_ids"]
    assert [(call.prompt_id, call.model) for call in first.calls] == [
        (call.prompt_id, call.model) for call in second.calls
    ]


def test_grok_selection_switches_to_actual_core_outcomes_after_collection():
    rows = _valid_corpus()
    first_train = next(row for row in rows if row.split == "train")
    evidence = {
        row.prompt_id: {
            CORE_MODELS[0]: {"success": 0 if row.prompt_id == first_train.prompt_id else 1},
            CORE_MODELS[1]: {"success": 1},
            CORE_MODELS[2]: {"success": 1},
        }
        for row in rows
        if row.split == "train"
    }
    metadata = grok_selection_metadata(rows, count=1, outcome_evidence=evidence)
    assert metadata["mode"] == "post_core_outcomes"
    assert metadata["rows_with_observed_disagreement"] == 1
    assert metadata["selected_metrics"][first_train.prompt_id]["observed_model_count"] == 3


def test_hash_tampering_and_duplicate_normalization_fail_closed(tmp_path):
    row = _prompt(1, "train")
    raw = row.to_dict()
    raw["prompt"] = "tampered prompt"
    with pytest.raises(ValueError, match="normalized_prompt_sha256"):
        StudyPrompt.from_dict(raw, require_frozen_hash=True)

    missing_hash = row.to_dict()
    missing_hash.pop("normalized_prompt_sha256")
    path = tmp_path / "missing-hash.jsonl"
    path.write_text(json.dumps(missing_hash) + "\n", encoding="utf-8")
    with pytest.raises(StudyCorpusError, match="normalized_prompt_sha256"):
        load_jsonl(path)

    duplicate = replace(_prompt(2, "test"), prompt="  unique   study prompt 1 ")
    audit = audit_corpus([row, duplicate], expected_counts={"train": 1, "test": 1})
    assert any(item["reason"] == "duplicate_prompt" for item in audit.missing_metadata)
    assert corpus_checksum([row, duplicate]) != corpus_checksum([row])


def test_deterministic_grading_wins_over_a_conflicting_judge():
    prompt = replace(_prompt(1, "train"), grader={"answer": "A", "answer_type": "multiple_choice"})
    deterministic = grade_response(prompt, "ANSWER: A")
    merged = merge_deterministic_and_judge(deterministic, False, judge_version="judge-v1")
    assert merged.label is True
    assert merged.disagreement


def test_pairwise_positions_are_randomized_but_resumable():
    first = randomize_pairwise_positions("prompt-1", "reference", "challenger", judge_version="v1")
    second = randomize_pairwise_positions("prompt-1", "reference", "challenger", judge_version="v1")
    assert first == second
    assert {first.left_answer, first.right_answer} == {"reference", "challenger"}


def test_budget_reservation_is_idempotent(tmp_path):
    ledger = StudyBudgetLedger(tmp_path / "budget.json")
    first = ledger.reserve(
        phase="train", model="model", maximum_cost_usd=1.0, idempotency_key="p:model"
    )
    second = ledger.reserve(
        phase="train", model="model", maximum_cost_usd=1.0, idempotency_key="p:model"
    )
    assert first == second
    assert ledger.status()["committed_usd"] == 1.0
