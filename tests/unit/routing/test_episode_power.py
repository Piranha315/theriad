"""Offline regression checks for the actual joint bootstrap decision."""
from dataclasses import replace
import json
import math
import random

import pytest

from theriad.routing import episode_study as study


def synthetic_evidence(groups_per_family=20):
    rows, outcomes, groups = [], [], {}
    for family in study.FAMILIES:
        groups[family] = []
        for index in range(groups_per_family):
            row = study.Episode(f"{family}-{index}", f"{family}-group-{index}", family,
                                "test", f"Independent synthetic {family} unit {index}", {}, {"kind": "exact"})
            rows.append(row)
            values = (float(index % 5 != 0), float(index % 10 != 0), .02 + index * .0001, .01 + index * .0002)
            groups[family].append(values)
            for policy, success, cost in (("baseline", values[0], values[2]), ("challenger", values[1], values[3])):
                outcomes.append(study.EpisodeOutcome(row.episode_id, row.group_id, family, policy,
                                                     row.content_hash, bool(success), "completed", cost, 100, 1))
    return rows, outcomes, groups


def test_report_and_simulation_use_identical_group_bootstrap(monkeypatch, tmp_path):
    rows, outcomes, groups = synthetic_evidence()
    lock = study.freeze_test(tmp_path / "lock.json", episodes=rows, baseline={}, challenger={"kind": "frozen"}, pricing={}, power={})
    outcomes = [replace(item, policy_hash=lock["policy_hashes"][item.policy_id]) for item in outcomes]
    seen = []
    original = study._paired_bootstrap_decision

    def record(values, **kwargs):
        result = original(values, **kwargs)
        seen.append((values, result))
        return result

    monkeypatch.setattr(study, "_paired_bootstrap_decision", record)
    report = study.paired_report(rows, outcomes, seed=7, bootstrap_samples=100, lock=lock)
    assert seen[0][0] == groups
    assert report["success_lower_one_sided_95"] == seen[0][1]["quality_lower"]
    assert report["cost_per_success_ratio_upper_one_sided_95"] == seen[0][1]["cost_ratio_upper"]
    assert (report["status"] == "improved") == seen[0][1]["joint_pass"]
    assert report["status"] == "improved"
    simulation = study.simulate_joint_power(scenarios=[study.JointPowerScenario("test")], repetitions=2, bootstrap_samples=100)
    assert len(seen) == 3
    assert simulation["results"][0]["rates"]["joint_pass"]["count"] == sum(result["joint_pass"] for _, result in seen[1:])


def test_bootstrap_keeps_paired_groups_and_equal_family_weights():
    # Deliberately different family counts and different means in each group.
    groups = {family: [(0.4, .6, 2., 1.), (.8, .7, 3., 2.)] * (family_index + 1)
              for family_index, family in enumerate(study.FAMILIES)}
    actual = study._paired_bootstrap_decision(groups, seed=123, bootstrap_samples=100)
    rng = random.Random(123)
    differences, ratios = [], []
    for _ in range(100):
        chosen = [rng.choices(groups[family], k=len(groups[family])) for family in study.FAMILIES]
        family_means = [[sum(row[column] for row in group) / len(group) for column in range(4)] for group in chosen]
        a, b, ca, cb = [sum(row[column] for row in family_means) / 5 for column in range(4)]
        differences.append(b - a)
        ratios.append((cb / b) / (ca / a))
    assert actual["quality_lower"] == study._quantile(differences, .05)
    assert actual["cost_ratio_upper"] == study._quantile(ratios, .95)
    assert actual["joint_pass"] is False  # not enough independent groups


@pytest.mark.parametrize("baseline_success,challenger_success,baseline_cost", [(False, False, .02), (False, True, .02), (True, False, .02), (True, True, 0.)])
def test_undefined_cost_per_success_stays_json_serializable(tmp_path, baseline_success, challenger_success, baseline_cost):
    rows, outcomes, _ = synthetic_evidence()
    outcomes = [replace(item, success=baseline_success if item.policy_id == "baseline" else challenger_success,
                        cost_usd=baseline_cost if item.policy_id == "baseline" else .01) for item in outcomes]
    report = study.paired_report(rows, outcomes, bootstrap_samples=100)
    study.atomic_json(tmp_path / "report.json", report)
    assert json.loads((tmp_path / "report.json").read_text()) == report
    assert report["cost_per_success_ratio"] is None
    assert report["cost_per_success_ratio_upper_one_sided_95"] is None
    assert report["cost_gate"] is False
    assert "degenerate_or_zero_success_cost_ratio" in report["inference_limitations"]


def test_infinite_quantile_never_becomes_nan():
    assert study._quantile([math.inf] * 100, .95) == math.inf
    assert study._quantile([1.] * 94 + [math.inf] * 6, .95) == math.inf


def test_degenerate_bootstrap_does_not_promote_even_when_both_raw_gates_pass():
    groups = {family: [(1., 1., .02, .01)] * 20 for family in study.FAMILIES}
    result = study._paired_bootstrap_decision(groups, seed=3, bootstrap_samples=100)
    assert result["quality_gate"] and result["cost_gate"]
    assert result["joint_pass"] is False
    assert set(result["limitations"]) == {"degenerate_quality_bootstrap", "degenerate_cost_ratio_bootstrap"}


def test_joint_scenarios_are_deterministic_and_report_boundary_null_uncertainty():
    kwargs = {"groups_per_family": 20, "repetitions": 3, "bootstrap_samples": 100, "seed": 11}
    result = study.simulate_joint_power(**kwargs)
    assert study.simulate_joint_power(**kwargs) == result
    assert result["achieved_empirical_power"] is None
    assert [item["hypothesis"] for item in result["results"]] == ["joint_alternative"] * 2 + ["joint_null"] * 3
    assert all(item["joint_rate_interpretation"] == "scenario_false_pass_rate" for item in result["results"][2:])
    for item in result["results"]:
        for rate in item["rates"].values():
            lower, upper = rate["wilson_95_interval"]
            assert 0 <= lower <= rate["estimate"] + 1e-15 <= upper + 1e-15 <= 1 + 1e-15
            assert upper > lower
    json.dumps(result, allow_nan=False)
    reordered = study.simulate_joint_power(**kwargs, scenarios=list(reversed(study.DEFAULT_JOINT_POWER_SCENARIOS)))
    assert reordered["results"] == list(reversed(result["results"]))


def test_repeats_never_inflate_synthetic_independent_sample_size():
    scenario = study.JointPowerScenario("repeats", episodes_per_group=9)
    result = study.simulate_joint_power(groups_per_family=4, scenarios=[scenario], repetitions=2, bootstrap_samples=100)
    assert result["independent_groups"] == 20
    assert result["results"][0]["rates"]["joint_pass"]["count"] == 0
    groups = study._synthetic_group_values(scenario, 4, random.Random(8))
    assert all(len(group) == 4 for group in groups.values())


def test_quality_approximation_cannot_claim_budget_support_for_joint_power():
    _, outcomes, _ = synthetic_evidence()
    result = study.estimate_power(pilot=outcomes, available_test_usd=1e9)
    assert result["budget_supports_quality_only_approximation"] is True
    assert result["budget_supports_target"] is False
    assert result["joint_power_estimate"] is None
    assert result["joint_power_target"] is None
    assert result["scope"] == "quality_only_sample_size_approximation"


@pytest.mark.parametrize("changes", [{"cost_cv": float("nan")}, {"cost_per_success_ratio": 0}, {"baseline_success_probability": 0}, {"success_discordance_probability": .9}, {"episodes_per_group": 0}])
def test_invalid_scenario_assumptions_rejected(changes):
    with pytest.raises(ValueError):
        study.JointPowerScenario("invalid", **changes)
