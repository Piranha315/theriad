from __future__ import annotations

from theriad.accounting.pricing import ModelPrice, PricingTable
from theriad.routing.calibration_study import estimate_calibration_size
from theriad.routing.public_anchors import AnchorObservation, PublicAnchor
from theriad.routing.study import StudyPrompt
from theriad.routing.study_collector import build_calibration_study_plan


def _anchor(index: int) -> PublicAnchor:
    observations = (
        AnchorObservation(
            observation_id=f"p{index}:cheap",
            model_id="cheap",
            protocol_id="bcp-v1",
            outcome=index % 2 == 0,
            provider="p1",
            model_family="f1",
        ),
        AnchorObservation(
            observation_id=f"p{index}:strong",
            model_id="strong",
            protocol_id="bcp-v1",
            outcome=True,
            provider="p2",
            model_family="f2",
        ),
    )
    return PublicAnchor(
        prompt_id=f"p{index}",
        prompt=f"prompt {index}",
        benchmark_family="bcp",
        domain="research" if index % 2 else "code",
        observations=observations,
    )


def _calibration_prompt(index: int) -> StudyPrompt:
    return StudyPrompt(
        prompt_id=f"c{index}",
        prompt=f"calibration prompt {index}",
        split="validation",
        source="theriad",
        benchmark_family="theriad-traffic",
        domain="research",
        template_family=f"template-{index}",
        collection_vintage="2026-08-24",
        license_name="internal",
        license_status="internal",
        source_revision="traffic-v1",
        provenance_url="internal://traffic-v1",
        lane="general",
        label_type="deterministic",
        grader={"answer": "A", "answer_type": "multiple_choice"},
    )


def test_calibration_size_is_explicit_and_not_fixed_at_2000():
    report = estimate_calibration_size(_anchor(index) for index in range(100))
    assert report.status == "sized"
    assert report.recommended_rows != 2_000
    assert report.recommended_rows >= report.worst_case_paired_transfer_rows
    assert report.domain_count == 2


def test_calibration_plan_uses_explicit_models_and_budget_pricing():
    pricing = PricingTable(
        prices={
            "model-a": ModelPrice(0.1, 0.2),
            "model-b": ModelPrice(0.3, 0.4),
        },
        use_live=False,
    )
    plan = build_calibration_study_plan(
        tuple(_calibration_prompt(index) for index in range(3)),
        models=("model-a", "model-b"),
        pricing=pricing,
        max_output_tokens=100,
    )
    assert plan.call_count == 6
    assert set(plan.by_model()) == {"model-a", "model-b"}
    assert plan.maximum_cost_usd > 0
