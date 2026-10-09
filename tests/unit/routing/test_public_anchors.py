from __future__ import annotations

import json
from pathlib import Path

import pytest

from theriad.routing.anchor_evaluation import (
    compare_ablation_strategies,
    compare_empirical_vs_irt,
    equal_difficulty_specialization_slice,
    evaluate_selection,
)
from theriad.routing.public_anchor_import import (
    BrowseCompRunSpec,
    EvaluatedLogSpec,
    import_browsecomp_plus,
    import_evaluated_jsonl_logs,
    load_evaluation_csv,
)
from theriad.routing.public_anchors import (
    AnchorFeatureIndex,
    AnchorObservation,
    CapabilityProfile,
    PublicAnchor,
    assign_grouped_anchor_splits,
    derive_anchor_metrics,
    normalized_entropy,
    validate_anchor_leakage,
)
from theriad.routing.study_anchor_boundary import (
    PUBLIC_ANCHOR_OVERLAP_POLICY,
    StudyAnchorExclusionBoundary,
    build_study_anchor_feature_index,
)


def _observation(
    model: str,
    outcome: bool | None,
    *,
    provider: str = "provider-a",
    family: str = "family-a",
    protocol: str = "p1",
) -> AnchorObservation:
    return AnchorObservation(
        observation_id=f"prompt:{model}:{protocol}",
        model_id=model,
        protocol_id=protocol,
        outcome=outcome,
        outcome_status="graded" if outcome is not None else "missing",
        provider=provider,
        model_family=family,
    )


def _anchor(index: int, outcomes: dict[str, bool], *, split: str = "") -> PublicAnchor:
    observations = tuple(_observation(model, outcome) for model, outcome in outcomes.items())
    return PublicAnchor(
        prompt_id=f"p-{index}",
        prompt=f"unique research question {index}",
        benchmark_family="browsecomp_plus",
        domain="research",
        template_family=f"template-{index}",
        source_revision="revision-1",
        collection_vintage="2026-08-24",
        provenance_url="https://example.test/browsecomp-plus",
        split=split,
        observations=observations,
        metrics=tuple(sorted(derive_anchor_metrics(observations).items())),
    )


def test_entropy_and_balanced_difficulty_are_distinct_from_disagreement():
    assert normalized_entropy(0.0) == 0.0
    assert normalized_entropy(0.5) == pytest.approx(1.0)
    observations = (
        _observation("a1", True, provider="a", family="a"),
        _observation("a2", True, provider="a", family="a"),
        _observation("b1", False, provider="b", family="b"),
    )
    metrics = derive_anchor_metrics(observations)
    assert metrics["empirical_success_rate_unweighted"] == pytest.approx(2 / 3)
    assert metrics["empirical_success_rate_balanced"] == pytest.approx(0.5)
    assert metrics["empirical_difficulty"] == pytest.approx(0.5)
    assert metrics["model_disagreement"] == pytest.approx(1.0)


def test_balancing_policy_keeps_provider_and_family_groups_distinct():
    observations = (
        _observation("a1", True, provider="provider-a", family="shared-family"),
        _observation("a2", True, provider="provider-a", family="shared-family"),
        _observation("b1", False, provider="provider-b", family="shared-family"),
    )
    metrics = derive_anchor_metrics(observations)
    assert metrics["empirical_success_rate_balanced"] == pytest.approx(0.5)
    assert metrics["balanced_group_count"] == 2


def test_capability_upsets_are_not_collapsed_into_entropy():
    observations = (
        _observation("weak", True),
        _observation("mid", False),
        _observation("strong", False),
    )
    profile = CapabilityProfile.from_scores(
        {"weak": 0.2, "mid": 0.5, "strong": 0.9},
        estimated_from="train-only",
    )
    metrics = derive_anchor_metrics(observations, capability_profile=profile)
    assert metrics["capability_order_violation"] is True
    assert metrics["upset_count"] == 2
    assert metrics["specialization_score"] == pytest.approx(2 / 3)
    assert metrics["model_disagreement"] == pytest.approx(normalized_entropy(1 / 3))


def test_grouped_iid_splits_allow_family_overlap_but_ood_can_require_disjointness():
    rows = tuple(_anchor(index, {"weak": bool(index % 2), "strong": True}) for index in range(40))
    assigned = assign_grouped_anchor_splits(rows, seed=3)
    assert {row.split for row in assigned} == {"train", "validation", "test"}
    validate_anchor_leakage(assigned)
    with pytest.raises(ValueError, match="benchmark_family leakage"):
        validate_anchor_leakage(assigned, require_benchmark_family_disjoint=True)

    ood = assign_grouped_anchor_splits(rows, ood_benchmark_families={"browsecomp_plus"})
    assert {row.split for row in ood} == {"ood"}
    validate_anchor_leakage(ood, require_benchmark_family_disjoint=True)


def test_anchor_feature_index_exposes_neighbors_without_copying_routes():
    rows = tuple(
        _anchor(index, {"weak": index % 2 == 0, "strong": index % 3 == 0})
        for index in range(3)
    )
    index = AnchorFeatureIndex.from_embeddings(
        rows,
        {row.prompt_id: [1.0, float(i == 1)] for i, row in enumerate(rows)},
    )
    features = index.features_for_vector([1.0, 1.0], top_k=2, exclude_prompt_ids={"p-1"})
    assert features["neighbor_count"] == 2
    assert features["nearest_anchor_similarity"] is not None
    assert "neighbor_model_success_rates" in features
    assert "route" not in features


def test_tfidf_index_uses_fitted_idf_for_new_text():
    rows = tuple(
        _anchor(index, {"weak": True})
        for index in range(2)
    )
    index = AnchorFeatureIndex.from_tfidf(rows)
    features = index.features_for_text("unique research question 0", top_k=1)
    assert features["anchor_retriever_kind"] == "tfidf_baseline"
    assert features["neighbor_ids"] == ["p-0"]


def test_study_anchor_boundary_excludes_exact_overlap_from_features():
    anchors = tuple(_anchor(index, {"weak": True}) for index in range(3))
    boundary = StudyAnchorExclusionBoundary(
        schema_version=1,
        policy=PUBLIC_ANCHOR_OVERLAP_POLICY,
        study_corpus_checksum="study-checksum",
        anchor_manifest_sha256="anchor-manifest-sha256",
        overlap_hashes=(anchors[0].normalized_prompt_hash,),
        by_study_split={"test": 1},
        by_anchor_split={"test": 1},
    )
    index = build_study_anchor_feature_index(anchors, boundary)

    excluded = index.features_for_text("  unique   research question 0 ")
    assert excluded["anchor_boundary_status"] == "excluded_exact_overlap"
    assert excluded["public_anchor_outcomes_used"] is False
    assert excluded["neighbor_count"] == 0
    assert excluded["neighbor_ids"] == []

    eligible = index.features_for_text(anchors[1].prompt)
    assert eligible["anchor_boundary_status"] == "eligible_non_overlap"
    assert eligible["public_anchor_outcomes_used"] is True


def test_csv_evaluations_are_accepted_as_explicit_outcomes(tmp_path: Path):
    evaluation = tmp_path / "evaluation.csv"
    evaluation.write_text("query_id,correct\nq1,True\n", encoding="utf-8")
    assert load_evaluation_csv(evaluation) == {"q1": {"query_id": "q1", "correct": "True"}}


def test_evaluated_jsonl_importer_uses_explicit_correct_and_keeps_raw_refs(tmp_path: Path):
    evaluation = tmp_path / "evaluated.jsonl"
    evaluation.write_text(
        json.dumps(
            {
                "qid": 0,
                "question": "Which answer?",
                "correct_answer": "A",
                "gen_output": "A",
                "correct": True,
                "latency_s": 1.5,
                "total_input_tokens": 20,
                "total_output_tokens": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = EvaluatedLogSpec(
        path=evaluation,
        benchmark_family="xbench",
        model_id="model-a",
        provider="provider-a",
        model_family="family-a",
        protocol_id="xbench-serper-no-cm-v1",
        source_revision="dataset-sha",
        provenance_url="https://example.test/eval-logs",
        license_name="Apache-2.0",
        license_status="public",
        context_masking="no-CM",
    )
    anchors, rejected = import_evaluated_jsonl_logs((spec,))
    assert not rejected
    assert len(anchors) == 1
    assert anchors[0].prompt_id == "xbench:0"
    observation = anchors[0].observations[0]
    assert observation.outcome is True
    assert observation.outcome_method == "correct"
    assert observation.context_masking == "no-CM"
    assert observation.latency_ms == pytest.approx(1500.0)
    assert observation.prompt_tokens == 20
    assert observation.completion_tokens == 4
    assert dict(observation.raw_ref)["raw_outputs_retained_at_source"] is True


def test_browsecomp_import_preserves_missing_outcomes_and_trajectory_metadata(tmp_path: Path):
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query_id": "q1", "query": "What is the answer?", "answer": "42"}) + "\n",
        encoding="utf-8",
    )
    run = tmp_path / "run.jsonl"
    run.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "status": "completed",
                "search_counts": {"search": 3},
                "result": [{"type": "reasoning", "output": []}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = BrowseCompRunSpec(
        path=run,
        model_id="model-a",
        provider="provider-a",
        model_family="family-a",
        protocol_id="bm25-v1",
        source_revision="runs-sha",
        provenance_url="https://example.test/runs",
        retrieval_config="bm25",
    )
    anchors, rejected = import_browsecomp_plus(queries, (spec,))
    assert not rejected
    assert len(anchors) == 1
    observation = anchors[0].observations[0]
    assert observation.outcome is None
    assert observation.outcome_status == "missing"
    assert dict(observation.tool_call_counts) == {"search": 3}
    assert observation.trajectory_steps == 1
    assert dict(observation.raw_ref)["source_line"] == 1


def test_ablation_and_irt_run_on_synthetic_held_out_anchors():
    rows = []
    for index in range(80):
        rows.append(
            _anchor(
                index,
                {
                    "cheap": index % 5 != 0,
                    "strong": True,
                },
                split="train" if index < 60 else "test",
            )
        )
    profile = CapabilityProfile.from_scores(
        {"cheap": 0.4, "strong": 0.9},
        estimated_from="synthetic-train",
        out_of_fold=True,
    )
    train = tuple(row for row in rows if row.split == "train")
    test = tuple(row for row in rows if row.split == "test")
    result = evaluate_selection(
        train,
        test,
        strategy="difficulty_disagreement",
        costs={"cheap": 0.01, "strong": 0.10},
        quality_threshold=0.5,
        capability_profile=profile,
    )
    assert result["count"] == 20
    assert result["selected_success_rate"] is not None
    paired = compare_ablation_strategies(
        train,
        test,
        costs={"cheap": 0.01, "strong": 0.10},
        quality_threshold=0.5,
    )
    assert "comparisons" in paired
    slice_report = equal_difficulty_specialization_slice(
        train,
        test,
        costs={"cheap": 0.01, "strong": 0.10},
        capability_profile=profile,
    )
    assert "groups" in slice_report
    irt = compare_empirical_vs_irt(rows, folds=3)
    assert irt["status"] == "evaluated"
