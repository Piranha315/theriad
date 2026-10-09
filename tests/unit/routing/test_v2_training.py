from __future__ import annotations

from dataclasses import replace

import pytest

from theriad.models.tiers import ModelCandidate
from theriad.routing.study import REPEATED_TRIAL_GROUP_CONTRACT
from theriad.routing.v2_training import (
    ModelOutcome,
    StudyRow,
    aggregate_router_input_groups,
    assert_grouped_splits,
    assert_study_row_contract,
    automatic_knee,
    bootstrap_group_mean_ci,
    evaluate_artifact_rows,
    expected_calibration_error,
    pareto_frontier,
    post_collection_disagreement_slices,
    train_embedding_artifact,
    train_surface_artifact,
)


def _training_candidates():
    return [
        ModelCandidate(tier="test", provider="test", model="cheap", input_cost_per_million=0.1, output_cost_per_million=0.1),
        ModelCandidate(tier="test", provider="test", model="luna", input_cost_per_million=1.0, output_cost_per_million=1.0),
    ]


def _row(prompt: str, split: str, source: str = "source") -> StudyRow:
    return StudyRow(
        prompt=prompt,
        split=split,
        source=source,
        domain="domain",
        template_family=prompt,
        collected_at="2026-08-24",
        lane="general",
        outcomes={"model": ModelOutcome(1.0, 0.01, 10, 20.0)},
        prompt_id=f"prompt-{prompt}",
        benchmark_family="benchmark",
        collection_vintage="2026-08-24",
        source_revision="revision-1",
        license_name="test-license",
        license_status="approved",
        provenance_url="https://example.test/source",
    )


def _tau_trial(prompt_id: str, *, success: float, state_identity: str) -> StudyRow:
    prompt = "same visible telecom ticket"
    normalized_hash = StudyRow(prompt=prompt, split="train", source="tau2-bench", domain="telecom", template_family="service", collected_at="2026-08-24", lane="general", outcomes={"cheap": ModelOutcome(success, 0.01, 20, 40.0)}).normalized_hash
    return StudyRow(
        prompt=prompt,
        split="train",
        source="tau2-bench",
        domain="telecom",
        template_family="service",
        collected_at="2026-08-24",
        lane="general",
        outcomes={"cheap": ModelOutcome(success, 0.01, 20, 40.0)},
        prompt_id=prompt_id,
        benchmark_family="tau2-bench:telecom",
        collection_vintage="2026-08-24",
        source_revision="tau2@revision-1",
        license_name="Apache-2.0",
        license_status="approved",
        provenance_url="https://example.test/tau2",
        label_type="tool_state",
        task_format="agent_state",
        leakage_group="tau2-bench|telecom|service|2026-08-24",
        normalized_prompt_sha256=normalized_hash,
        grader={
            "kind": "tau2_official_environment",
            "executor_id": "tau2-fixture",
            "executor_version": "v1",
            "evidence_policy": "official",
            "stateful_duplicate_prompt_allowed": True,
            "state_identity": state_identity,
            "router_input_group": normalized_hash,
            "repeated_trial_contract": REPEATED_TRIAL_GROUP_CONTRACT,
        },
    )


def _complete_training_row(index: int, split: str) -> StudyRow:
    lane = "code" if index % 2 else "general"
    task_format = "code_execution" if lane == "code" else "conversational_reasoning"
    grader = (
        {
            "kind": "sandboxed_code_tests",
            "test_harness_id": "fixture",
            "test_harness_version": "v1",
            "executable_tests": "def check(answer): assert answer(1) == 1",
        }
        if lane == "code"
        else {}
    )
    return StudyRow(
        prompt=f"complete training task {index}",
        split=split,
        source=f"source-{index}",
        domain=lane,
        template_family=f"template-{index}",
        collected_at="2026-08-24",
        lane=lane,
        outcomes={
            "cheap": ModelOutcome(float(index % 3 != 0), 0.001, 30, 50.0),
            "luna": ModelOutcome(float(index % 5 != 0), 0.01, 50, 80.0),
        },
        prompt_id=f"complete-{index}",
        benchmark_family=f"benchmark-{index}",
        collection_vintage="2026-08-24",
        source_revision="revision-1",
        license_name="test-license",
        license_status="approved",
        provenance_url="https://example.test/source",
        task_format=task_format,
        grader=grader,
    )


def test_grouped_split_rejects_exact_prompt_leakage():
    with pytest.raises(ValueError, match="prompt leakage"):
        assert_grouped_splits([_row("same", "train"), _row("same", "test")])


def _leaky_pair(kind: str) -> tuple[StudyRow, StudyRow]:
    common = {
        "source": "source-train",
        "domain": "domain-train",
        "template_family": "template-train",
        "collection_vintage": "vintage-train",
        "benchmark_family": "benchmark-train",
    }
    first = _row("first prompt", "train", source=common["source"])
    first = replace(
        first,
        domain=common["domain"],
        template_family=common["template_family"],
        collection_vintage=common["collection_vintage"],
        benchmark_family=common["benchmark_family"],
    )
    second = replace(
        first,
        prompt="second prompt",
        split="test",
        source="source-test",
        domain="domain-test",
        template_family="template-test",
        collection_vintage="vintage-test",
        benchmark_family="benchmark-test",
        leakage_group="leakage-test",
        normalized_prompt_sha256="",
    )
    if kind == "normalized_prompt":
        second = replace(second, prompt="  first   prompt ", normalized_prompt_sha256="")
    elif kind == "leakage_group":
        second = replace(second, leakage_group=first.leakage_group)
    else:
        raise AssertionError(kind)
    return first, second


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("normalized_prompt", "prompt leakage"),
        ("leakage_group", "leakage-group leakage"),
    ],
)
def test_grouped_split_rejects_each_leakage_class_independently(kind: str, message: str):
    with pytest.raises(ValueError, match=message):
        assert_grouped_splits(_leaky_pair(kind))


def test_grouped_split_accepts_disjoint_values_for_every_leakage_class():
    first, second = _leaky_pair("normalized_prompt")
    assert_grouped_splits(
        [
            first,
            replace(
                second,
                prompt="second prompt",
                source="source-test",
                domain="domain-test",
                template_family="template-test",
                collection_vintage="vintage-test",
                benchmark_family="benchmark-test",
                normalized_prompt_sha256="",
            ),
        ]
    )


def test_artifact_training_contract_rejects_incomplete_provenance():
    with pytest.raises(ValueError, match="incomplete"):
        assert_study_row_contract([replace(_row("one", "train"), provenance_url="")])


def test_repeated_visible_tau_trials_are_grouped_only_with_explicit_contract():
    first = _tau_trial("tau-trial-1", success=1.0, state_identity="state-1")
    second = _tau_trial("tau-trial-2", success=0.0, state_identity="state-2")
    assert_study_row_contract([first, second])

    finalized_shape = {
        "prompt": first.prompt,
        "split": first.split,
        "source": first.source,
        "domain": first.domain,
        "template_family": first.template_family,
        "collected_at": first.collected_at,
        "lane": first.lane,
        "outcomes": {
            "cheap": {
                "success": 1.0,
                "cost_usd": 0.01,
                "output_tokens": 20,
                "latency_ms": 40.0,
            }
        },
        "prompt_id": first.prompt_id,
        "benchmark_family": first.benchmark_family,
        "collection_vintage": first.collection_vintage,
        "source_revision": first.source_revision,
        "license": first.license_name,
        "license_status": first.license_status,
        "provenance_url": first.provenance_url,
        "label_type": first.label_type,
        "task_format": first.task_format,
        "leakage_group": first.leakage_group,
        "normalized_prompt_sha256": first.normalized_hash,
        "grader": first.grader,
        "router_input_group": first.router_input_group,
        "router_input_contract": first.router_input_contract,
    }
    reloaded = StudyRow.from_dict(finalized_shape)
    assert reloaded.is_repeated_trial

    grouped = aggregate_router_input_groups([first, second])
    assert len(grouped) == 1
    assert grouped[0].trial_count == 2
    assert grouped[0].outcomes["cheap"].success == 0.5
    assert grouped[0].effective_router_input_group == first.normalized_hash

    without_contract = replace(
        second,
        grader={**second.grader, "repeated_trial_contract": ""},
        router_input_contract="",
    )
    with pytest.raises(ValueError, match="duplicate normalized prompt"):
        assert_study_row_contract([first, without_contract])


def test_finalized_grouped_trials_are_accepted_by_trainer_without_test_inputs():
    rows = [_complete_training_row(index, "train" if index < 40 else "validation" if index < 64 else "test") for index in range(68)]
    rows[0] = _tau_trial("tau-trial-1", success=1.0, state_identity="state-1")
    rows[1] = _tau_trial("tau-trial-2", success=0.0, state_identity="state-2")

    artifact = train_surface_artifact(
        rows,
        fallback_models={"general": "luna", "code": "luna"}, model_candidates=_training_candidates(),
    )
    assert artifact["raw_trial_count"] == 68
    assert artifact["effective_sample_size"]["all"] == 67
    assert artifact["repeated_trial_count"] == 1


def test_group_bootstrap_uses_effective_visible_input_units():
    low, high = bootstrap_group_mean_ci(
        [0.0, 1.0, 1.0],
        ["same-visible-input", "same-visible-input", "other-visible-input"],
        samples=300,
        seed=4,
    )
    assert low <= 2.0 / 3.0 <= high


def test_grouped_split_rejects_group_leakage():
    first = _row("one", "train")
    second = StudyRow(
        prompt="two",
        split="test",
        source=first.source,
        domain=first.domain,
        template_family=first.template_family,
        collected_at=first.collected_at,
        lane="general",
        outcomes=first.outcomes,
    )
    with pytest.raises(ValueError, match="group leakage"):
        assert_grouped_splits([first, second])


def test_pareto_frontier_removes_dominated_points():
    points = [(0.1, 1.0, 0.7), (0.2, 2.0, 0.6), (0.3, 3.0, 0.9)]
    assert pareto_frontier(points) == [points[0], points[2]]


def test_automatic_knee_selects_curved_middle_and_is_order_independent():
    points = [
        (0.1, 0.0, 0.50),
        (0.2, 0.1, 0.80),
        (0.3, 0.4, 0.90),
        (0.4, 1.0, 1.00),
    ]
    assert automatic_knee(points)[0] == 0.2
    assert automatic_knee(list(reversed(points)))[0] == 0.2


def test_expected_calibration_error_for_perfect_bins_is_zero():
    assert expected_calibration_error([0.0, 1.0], [0.0, 1.0], bins=2) == 0.0


def test_surface_trainer_freezes_per_lane_knees_without_reading_test_labels():
    rows = []
    for index in range(68):
        split = "train" if index < 40 else "validation" if index < 64 else "test"
        lane = "general" if index % 2 == 0 else "code"
        rows.append(
            StudyRow(
                prompt=f"task {index} " + ("detail " * (index % 7)),
                split=split,
                source=f"source-{index}",
                domain=lane,
                template_family=f"template-{index}",
                collected_at="2026-08-24",
                lane=lane,
                outcomes={
                    "cheap": ModelOutcome(float(index % 3 != 0), 0.001, 30, 50.0),
                    "luna": ModelOutcome(float(index % 5 != 0), 0.01, 50, 80.0),
                },
                prompt_id=f"prompt-{index}",
                benchmark_family=f"benchmark-{split}",
                collection_vintage=f"vintage-{split}",
                source_revision="revision-1",
                license_name="test-license",
                license_status="approved",
                provenance_url="https://example.test/source",
            )
        )
    artifact = train_surface_artifact(
        rows, fallback_models={"general": "luna", "code": "luna"}, model_candidates=_training_candidates()
    )
    assert set(artifact["models"]) == {"cheap", "luna"}
    assert 0.0 <= artifact["lanes"]["general"]["knee_threshold"] <= 1.0
    assert 0.0 <= artifact["lanes"]["code"]["knee_threshold"] <= 1.0


def test_test_outcomes_cannot_change_training_or_configuration():
    rows = []
    for index in range(68):
        split = "train" if index < 40 else "validation" if index < 64 else "test"
        lane = "general" if index % 2 == 0 else "code"
        rows.append(
            StudyRow(
                prompt=f"task {index} " + ("detail " * (index % 7)),
                split=split,
                source=f"source-{index}",
                domain=lane,
                template_family=f"template-{index}",
                collected_at="2026-08-24",
                lane=lane,
                outcomes={
                    "cheap": ModelOutcome(float(index % 3 != 0), 0.001, 30, 50.0),
                    "luna": ModelOutcome(float(index % 5 != 0), 0.01, 50, 80.0),
                },
                prompt_id=f"prompt-{index}",
                benchmark_family=f"benchmark-{index}",
                collection_vintage=f"vintage-{index}",
                source_revision="revision-1",
                license_name="test-license",
                license_status="approved",
                provenance_url="https://example.test/source",
            )
        )
    first = train_surface_artifact(rows, fallback_models={"general": "luna", "code": "luna"}, model_candidates=_training_candidates())
    mutated_rows = list(rows)
    mutated_rows[-1] = replace(
        mutated_rows[-1],
        outcomes={
            "cheap": ModelOutcome(0.0, 999.0, 1_000, 99_000.0),
            "luna": ModelOutcome(1.0, 999.0, 1_000, 99_000.0),
        },
    )
    second = train_surface_artifact(
        mutated_rows, fallback_models={"general": "luna", "code": "luna"}, model_candidates=_training_candidates()
    )
    stable_keys = ("family", "feature_mean", "feature_stdev", "lanes", "models")
    assert {key: first[key] for key in stable_keys} == {key: second[key] for key in stable_keys}


def test_post_collection_disagreement_slice_keeps_missing_state_pending():
    rows = [
        replace(_row("train", "train"), prompt_id="train"),
        replace(_row("validation", "validation"), prompt_id="validation"),
        replace(_row("test", "test"), prompt_id="test"),
    ]
    report = post_collection_disagreement_slices(rows, model_ids=("model",))
    assert report["influences_training_or_tuning"] is False
    assert report["by_split"]["train"]["pending_count"] == 1
    assert report["by_split"]["validation"]["pending_count"] == 1
    assert report["by_split"]["test"]["pending_count"] == 1


def _frozen_artifact():
    return {
        "schema_version": 1,
        "router_version": "frozen-evaluation",
        "feature_mean": {},
        "feature_stdev": {},
        "lanes": {lane: {"knee_threshold": 0.0, "fallback_model": "luna"} for lane in ("general", "code")},
        "models": {
            model: {"intercept": 1.0, "expected_output_tokens": 50}
            for model in ("cheap", "luna")
        },
    }


def test_frozen_decisions_ignore_realized_cost_success_and_output_length():
    first = replace(_complete_training_row(64, "test"), outcomes={
        "cheap": ModelOutcome(0.0, 50.0, 50_000, 100.0),
        "luna": ModelOutcome(1.0, 0.001, 1, 1.0),
    })
    second = replace(first, outcomes={
        "cheap": ModelOutcome(1.0, 0.001, 1, 1.0),
        "luna": ModelOutcome(0.0, 50.0, 50_000, 100.0),
    })
    before = evaluate_artifact_rows(_frozen_artifact(), [first], model_candidates=_training_candidates())
    after = evaluate_artifact_rows(_frozen_artifact(), [second], model_candidates=_training_candidates())
    assert before[0][0] == after[0][0] == "cheap"
    assert before[0][1].success == 0.0
    assert after[0][1].success == 1.0


def test_missing_selected_outcome_never_changes_candidate_set():
    row = replace(_complete_training_row(64, "test"), outcomes={"luna": ModelOutcome(1.0, 0.001, 1, 1.0)})
    with pytest.raises(ValueError, match="missing selected outcome.*cheap"):
        evaluate_artifact_rows(_frozen_artifact(), [row], model_candidates=_training_candidates())


def test_two_successful_policies_still_receive_different_cost_scores():
    row = replace(_complete_training_row(64, "test"), outcomes={
        "cheap": ModelOutcome(1.0, 0.001, 50, 100.0),
        "luna": ModelOutcome(1.0, 0.01, 50, 100.0),
    })
    artifact = _frozen_artifact()
    cheap = evaluate_artifact_rows(artifact, [row], model_candidates=_training_candidates())
    artifact["lanes"]["general"]["fixed_model"] = "luna"
    luna = evaluate_artifact_rows(artifact, [row], model_candidates=_training_candidates())
    assert cheap[0][1].success == luna[0][1].success == 1.0
    assert cheap[0][1].cost_usd < luna[0][1].cost_usd


@pytest.mark.parametrize("family", ["surface", "semantic", "hybrid", "knn"])
def test_trainer_validation_metrics_exactly_match_runtime(family):
    rows = [_complete_training_row(index, "train" if index < 40 else "validation") for index in range(64)]
    kwargs = {"fallback_models": {"general": "luna", "code": "luna"}, "model_candidates": _training_candidates()}
    embeddings = {row.prompt_id: [float(index % 2), float(index % 3)] for index, row in enumerate(rows)}
    backend = None
    if family == "surface":
        artifact = train_surface_artifact(rows, **kwargs)
    else:
        artifact = train_embedding_artifact(
            rows, embeddings, **kwargs, family=family, embedding_model="fixture",
            include_surface=family == "hybrid", knn=family == "knn",
        )
        class Backend:
            def encode(self, text):
                return next(embeddings[row.prompt_id] for row in rows if row.prompt == text)
        backend = Backend()
    for lane in ("general", "code"):
        validation = [row for row in rows if row.split == "validation" and row.lane == lane]
        results = evaluate_artifact_rows(
            artifact, validation, model_candidates=_training_candidates(), embedding_backend=backend,
        )
        assert artifact["lanes"][lane]["validation_cost_usd"] == pytest.approx(sum(outcome.cost_usd for _, outcome in results) / len(results))
        assert artifact["lanes"][lane]["validation_success"] == pytest.approx(sum(outcome.success for _, outcome in results) / len(results))


def test_budget_matched_selection_compares_fixed_policy_and_enforces_headroom():
    rows = [_complete_training_row(index, "train" if index < 40 else "validation") for index in range(64)]
    artifact = train_surface_artifact(
        rows, fallback_models={"general": "luna", "code": "luna"},
        model_candidates=_training_candidates(), selection_cost_ratio=0.95,
        baseline_models_by_prompt={row.prompt_id: "luna" for row in rows if row.split == "validation"},
    )
    assert artifact["research_only"] is True
    for lane in artifact["lanes"].values():
        assert lane["validation_cost_usd"] <= 0.95 * lane["baseline_cost_usd"]
        assert lane["fixed_model"] == "cheap"
        assert any(choice.get("fixed_model") == "luna" for choice in lane["validation_policy_comparisons"])


def test_budget_matched_selection_rejects_infeasible_cost_constraint():
    rows = [_complete_training_row(index, "train" if index < 40 else "validation") for index in range(64)]
    with pytest.raises(ValueError, match="no general policy meets"):
        train_surface_artifact(
            rows, fallback_models={"general": "luna", "code": "luna"},
            model_candidates=_training_candidates(), selection_cost_ratio=0.95,
            baseline_models_by_prompt={row.prompt_id: "cheap" for row in rows if row.split == "validation"},
        )


def test_unknown_model_requires_explicit_pre_call_catalog():
    with pytest.raises(ValueError, match="explicit pre-call model candidates"):
        evaluate_artifact_rows(_frozen_artifact(), [_complete_training_row(64, "test")])


@pytest.mark.parametrize("embedding", [False, True])
def test_trainers_never_read_or_aggregate_locked_test_outcomes(embedding):
    class LockedOutcomes(dict):
        def __iter__(self):
            raise AssertionError("locked test outcomes were accessed")

        def __getitem__(self, key):
            raise AssertionError("locked test outcomes were accessed")

    rows = [_complete_training_row(index, "train" if index < 40 else "validation") for index in range(64)]
    locked = replace(_tau_trial("locked-1", success=1.0, state_identity="state-1"), split="test", outcomes=LockedOutcomes())
    repeated = replace(_tau_trial("locked-2", success=0.0, state_identity="state-2"), split="test", outcomes=LockedOutcomes())
    rows.extend([locked, repeated])
    kwargs = {"fallback_models": {"general": "luna", "code": "luna"}, "model_candidates": _training_candidates()}
    if embedding:
        vectors = {row.prompt_id: [float(index % 2), 1.0] for index, row in enumerate(rows) if row.split != "test"}
        artifact = train_embedding_artifact(rows, vectors, **kwargs)
    else:
        artifact = train_surface_artifact(rows, **kwargs)
    assert artifact["effective_sample_size"]["test"] == 1
    assert artifact["repeated_trial_count"] == 1


def test_repeated_trials_cannot_hide_a_missing_model_outcome():
    first = _tau_trial("trial-1", success=1.0, state_identity="state-1")
    missing = replace(_tau_trial("trial-2", success=0.0, state_identity="state-2"), outcomes={})
    with pytest.raises(ValueError, match="incomplete repeated-trial outcomes"):
        aggregate_router_input_groups([first, missing])
