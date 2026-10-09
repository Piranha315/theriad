from dataclasses import replace

import pytest

from theriad.routing.episode_study import (
    FAMILIES,
    Episode,
    EpisodeOutcome,
    ProductStudyBudget,
    append_event,
    estimate_power,
    execute_pair,
    freeze_test,
    paired_report,
    simulate_inference_operating_characteristics,
    reconcile_completed_pairs,
    validate_episodes,
    verify_lock,
)
from theriad.routing.study_budget import StudyBudgetExceeded


def corpus(split="test"):
    return [
        Episode(
            f"{f}-{i}",
            f"{f}-group-{i}",
            f,
            split,
            f"Solve {f} case {i}",
            {},
            {"kind": "exact", "answer": "yes"},
        )
        for f in FAMILIES
        for i in range(4)
    ]


def outcome(row, policy, success=True, cost=0.02):
    return EpisodeOutcome(
        row.episode_id,
        row.group_id,
        row.family,
        policy,
        row.content_hash,
        success,
        "completed",
        cost,
        120,
        1,
    )


def test_complete_paired_improvement_and_cost_gate(tmp_path):
    rows = [
        replace(
            row,
            episode_id=f"{row.episode_id}-{repeat}",
            group_id=f"{row.group_id}-{repeat}",
            template_id=f"{row.template_id}-{row.episode_id}-{repeat}",
            prompt=f"{row.prompt} independent synthetic unit {repeat}",
        )
        for row in corpus()
        for repeat in range(5)
    ]
    results = [
        result
        for i, row in enumerate(rows)
        for result in (
            outcome(row, "baseline", False, 0.02),
            outcome(row, "challenger", True, 0.01 + (i % 2) * 0.001),
        )
    ]
    lock = freeze_test(
        tmp_path / "lock.json",
        episodes=rows,
        baseline={"kind": "baseline"},
        challenger={"kind": "challenger"},
        pricing={},
        power={},
    )
    results = [
        replace(result, policy_hash=lock["policy_hashes"][result.policy_id]) for result in results
    ]
    report = paired_report(rows, results, bootstrap_samples=100, lock=lock)
    assert report["status"] == "inconclusive"  # zero baseline successes prohibit a cost-per-success pass
    assert report["success_difference"] == 1
    expensive = [replace(r, cost_usd=0.03) if r.policy_id == "challenger" else r for r in results]
    assert (
        paired_report(rows, expensive, bootstrap_samples=100, lock=lock)["status"] == "inconclusive"
    )


def test_same_model_mix_opposite_item_success_is_distinguished():
    rows = corpus()
    results = []
    for i, row in enumerate(rows):
        results.extend(
            [
                replace(outcome(row, "baseline", False), model="a" if i % 2 else "b"),
                replace(outcome(row, "challenger", True), model="b" if i % 2 else "a"),
            ]
        )
    assert paired_report(rows, results, bootstrap_samples=100)["success_difference"] == 1


@pytest.mark.parametrize(
    "defect", ["missing", "unknown_cost", "ungradable", "duplicate", "changed_fixture"]
)
def test_missing_or_invalid_evidence_never_imputed(defect):
    rows = corpus()
    results = [outcome(row, p) for row in rows for p in ("baseline", "challenger")]
    if defect == "missing":
        results.pop()
    elif defect == "unknown_cost":
        results[-1] = replace(results[-1], cost_usd=None)
    elif defect == "ungradable":
        results[-1] = replace(results[-1], success=None)
    elif defect == "duplicate":
        results.append(results[-1])
    else:
        results[-1] = replace(results[-1], fixture_hash="changed")
    report = paired_report(rows, results, bootstrap_samples=100)
    assert report["status"] == "incomplete"
    assert "success_lower_one_sided_95" not in report


def test_family_weights_ignore_raw_volume_and_cluster_repeats():
    rows = corpus()
    for i in range(20):
        rows.append(replace(rows[0], episode_id=f"chat-repeat-{i}", prompt=f"Chat repeat {i}"))
    results = [
        outcome(row, p, p == "challenger" and row.family == "chat")
        for row in rows
        for p in ("baseline", "challenger")
    ]
    report = paired_report(rows, results, bootstrap_samples=100)
    assert report["success_difference"] == pytest.approx(0.2)
    assert report["families"]["chat"]["independent_groups"] == 4


def test_inference_null_simulation_is_offline_and_reports_operating_characteristics():
    diagnostic = simulate_inference_operating_characteristics(
        groups_per_family=20, repetitions=3, seed=9
    )
    assert diagnostic["method"].endswith("null_simulation")
    assert 0.0 <= diagnostic["false_pass_rate_under_equal_quality_and_cost"] <= 1.0
    assert diagnostic["interpretation"].startswith("diagnostic")


def test_failed_episodes_count_as_failures_and_keep_cost():
    rows = corpus()
    results = [outcome(row, p) for row in rows for p in ("baseline", "challenger")]
    results[-1] = replace(results[-1], success=False, status="timeout", attempts=3, cost_usd=0.1)
    report = paired_report(rows, results, bootstrap_samples=100)
    assert report["families"]["agents"]["challenger"]["operational_failures"] == 1
    assert report["success_difference"] < 0


@pytest.mark.parametrize("field", ["group_id", "template_id", "prompt"])
def test_split_leakage_rejected(field):
    rows = corpus("train")
    extra = replace(
        rows[0],
        episode_id="other",
        group_id="other-group",
        split="test",
        prompt="New prompt",
        template_id="other-template",
    )
    if field == "template_id":
        rows[0] = replace(rows[0], template_id="same")
    extra = replace(extra, **{field: getattr(rows[0], field)})
    with pytest.raises(ValueError, match="split leakage"):
        validate_episodes([*rows, extra], require_all_families=False)


def test_test_lock_is_immutable_and_binds_fixtures(tmp_path):
    rows = corpus()
    kwargs = dict(
        episodes=rows,
        baseline={"id": "current"},
        challenger={"id": "one"},
        pricing={"price": 1},
        power={"target": 0.05},
    )
    lock = freeze_test(tmp_path / "lock.json", **kwargs)
    assert freeze_test(tmp_path / "lock.json", **kwargs) == lock
    verify_lock(lock, rows)
    with pytest.raises(ValueError, match="already frozen"):
        freeze_test(tmp_path / "lock.json", **{**kwargs, "challenger": {"id": "test-tuned"}})
    with pytest.raises(ValueError, match="fixtures"):
        verify_lock(lock, [replace(rows[0], prompt="new"), *rows[1:]])


@pytest.mark.asyncio
async def test_pair_reserves_both_policies_and_resume_never_recalls(tmp_path):
    budget = ProductStudyBudget(tmp_path / "budget")
    row = corpus()[0]
    called = []

    async def run(row, policy_id, policy):
        called.append(policy_id)
        return outcome(row, policy_id, cost=0.1)

    kwargs = dict(
        row=row,
        policies={"baseline": {}, "challenger": {}},
        order=["challenger", "baseline"],
        runner=run,
        maximum_costs={"baseline": 0.2, "challenger": 0.2},
        budget=budget,
        stage="test",
        output=tmp_path / "pairs.jsonl",
        lock_hash="frozen",
    )
    first = await execute_pair(**kwargs)
    assert await execute_pair(**kwargs) == first
    assert called == ["challenger", "baseline"]
    assert budget.summary()["committed_usd"] == pytest.approx(0.2)
    assert reconcile_completed_pairs(kwargs["output"], budget) == {"completed": 1, "unresolved": 0}


@pytest.mark.asyncio
async def test_budget_reservation_prevents_one_sided_pair(tmp_path):
    budget = ProductStudyBudget(tmp_path / "budget")
    called = []

    async def run(*args):
        called.append(1)

    with pytest.raises(StudyBudgetExceeded):
        await execute_pair(
            row=corpus()[0],
            policies={"baseline": {}, "challenger": {}},
            order=["baseline", "challenger"],
            runner=run,
            maximum_costs={"baseline": 6, "challenger": 6},
            budget=budget,
            stage="pilot",
            output=tmp_path / "pairs.jsonl",
            lock_hash="frozen",
        )
    assert called == []


@pytest.mark.asyncio
async def test_unknown_charge_retains_reservation(tmp_path):
    budget = ProductStudyBudget(tmp_path / "budget")

    async def run(row, policy_id, policy):
        return outcome(row, policy_id, cost=None)

    with pytest.raises(StudyBudgetExceeded, match="unknown charge"):
        await execute_pair(
            row=corpus()[0],
            policies={"baseline": {}, "challenger": {}},
            order=["baseline", "challenger"],
            runner=run,
            maximum_costs={"baseline": 0.2, "challenger": 0.2},
            budget=budget,
            stage="test",
            output=tmp_path / "pairs.jsonl",
            lock_hash="frozen",
        )
    assert budget.summary()["committed_usd"] == pytest.approx(0.4)


def test_power_never_claims_certainty_from_tiny_perfect_pilot():
    rows = corpus("pilot")[:2]
    results = [outcome(row, p, True, 0.05) for row in rows for p in ("baseline", "challenger")]
    report = estimate_power(pilot=results)
    assert report["used_variance"] >= 0.25
    assert report["required_pairs"] >= 600
    assert report["budget_supports_target"] is False


def test_corrupt_ledger_is_not_silently_skipped(tmp_path):
    path = tmp_path / "pairs.jsonl"
    append_event(path, {"event": "pair_started", "pair_id": "x"})
    with path.open("a") as f:
        f.write('{"event":')
    with pytest.raises(ValueError):
        reconcile_completed_pairs(path, ProductStudyBudget(tmp_path / "budget"))


def test_small_sample_bounds_never_establish_improvement(tmp_path):
    rows = corpus()
    lock = freeze_test(
        tmp_path / "lock.json",
        episodes=rows,
        baseline={},
        challenger={"x": 1},
        pricing={},
        power={},
    )
    results = [
        replace(
            outcome(row, policy, policy == "challenger", 0.01 if policy == "challenger" else 0.02),
            policy_hash=lock["policy_hashes"][policy],
        )
        for row in rows
        for policy in ("baseline", "challenger")
    ]
    report = paired_report(rows, results, lock=lock, bootstrap_samples=100)
    assert report["status"] == "inconclusive"
    assert report["bootstrap"]["bounds_are_diagnostic_only"]
    changed = [replace(result, policy_hash="changed") for result in results]
    assert paired_report(rows, changed, lock=lock)["status"] == "incomplete"


@pytest.mark.asyncio
async def test_pair_cannot_resume_with_a_different_policy(tmp_path):
    row = corpus()[0]
    kwargs = dict(
        row=row,
        order=["baseline", "challenger"],
        maximum_costs={"baseline": 0.1, "challenger": 0.1},
        budget=ProductStudyBudget(tmp_path / "budget"),
        stage="pilot",
        output=tmp_path / "pairs.jsonl",
        lock_hash="contract",
    )

    async def run(row, policy_id, policy):
        return outcome(row, policy_id)

    await execute_pair(policies={"baseline": {}, "challenger": {}}, runner=run, **kwargs)
    with pytest.raises(ValueError, match="resume changed"):
        await execute_pair(
            policies={"baseline": {}, "challenger": {"model": "changed"}}, runner=run, **kwargs
        )


def test_repeated_pilot_trials_do_not_inflate_independent_groups():
    rows = corpus("pilot")
    repeated = [
        replace(row, episode_id=f"{row.episode_id}-{trial}") for row in rows for trial in range(3)
    ]
    results = [outcome(row, p) for row in repeated for p in ("baseline", "challenger")]
    report = estimate_power(pilot=results)
    assert report["pilot_groups"] == len(rows)
    assert report["pilot_pairs"] == len(rows) * 3
    assert report["required_pairs"] == report["required_independent_groups"] * 3
    with pytest.raises(ValueError, match="duplicate"):
        estimate_power(pilot=results + [results[0]])


@pytest.mark.asyncio
async def test_bounded_unknown_pair_finishes_both_trajectories_and_keeps_cost_null(tmp_path):
    row = corpus('pilot')[0]
    calls = []
    async def run(row, policy, config):
        calls.append(policy)
        return outcome(row, policy, cost=None if policy == 'baseline' else .01)
    kwargs = dict(row=row, policies={'baseline': {}, 'challenger': {}}, order=['baseline', 'challenger'],
                  runner=run, maximum_costs={'baseline': .1, 'challenger': .1},
                  budget=ProductStudyBudget(tmp_path / 'budget'), stage='pilot',
                  output=tmp_path / 'pairs.jsonl', lock_hash='bounded-study', allow_bounded_unknown=True)
    for _ in range(2):
        results = await execute_pair(**kwargs)
        assert results[0].cost_usd is None
    assert calls == ['baseline', 'challenger']
    assert kwargs['budget'].summary()['committed_usd'] == .2
    assert kwargs['budget'].summary()['realized_cost_usd'] == 0
    kwargs['budget'].reserve('pilot', 'next-independent-group', .1, calls=1)
