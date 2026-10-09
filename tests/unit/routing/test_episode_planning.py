"""Offline planning regression checks: cost scales with actual configurations."""
from dataclasses import replace

import pytest

from theriad.routing.episode_fixtures import build_four_way_seed_corpus
from theriad.routing.episode_planning import propose_episode_budget


def inputs(repeats=1):
    rows = [replace(row, max_calls=2, max_input_tokens=1000, max_output_tokens=100)
            for row in build_four_way_seed_corpus(repeats=repeats)]
    catalog = {"a": {"input_per_1m": 1, "output_per_1m": 2},
               "b": {"input_per_1m": 3, "output_per_1m": 4},
               "baseline": {"input_per_1m": 2, "output_per_1m": 3}}
    plans = {row.episode_id: {"chain": [["openrouter", "baseline"]]} for row in rows}
    eligibility = {row.episode_id: {"a": {"eligible": True}, "b": {"eligible": True}} for row in rows}
    return rows, catalog, plans, eligibility


def propose(args, **kwargs):
    return propose_episode_budget(*args, alternatives=["a", "b"], **kwargs)


def test_actual_four_way_stages_count_baseline_and_every_eligible_alternative():
    result = propose(inputs(), contingency_fraction=0)
    stages = result["stages"]
    assert result["episodes"] == result["independent_groups"] == 50
    assert {key: stage["configuration_count"] for key, stage in stages.items()} == {
        "pilot": 10, "development": 105, "test": 10, "smoke": 10}
    assert stages["development"]["max_model_calls"] == 210
    assert stages["development"]["max_input_tokens"] == 210_000
    assert stages["development"]["max_output_tokens"] == 21_000
    assert stages["pilot"]["model_reservation_usd"] == pytest.approx(5 * (.0046 + .0024))
    assert stages["development"]["model_reservation_usd"] == pytest.approx(35 * (.0046 + .0024 + .0068))
    assert stages["test"]["model_reservation_usd"] == pytest.approx(5 * (.0046 + .0068))
    assert result["budget_config"]["hard_cap_usd"] == .65
    assert result["grading"]["paid_judge_calls"] == 0
    assert result["paid_calls_authorized"] is False


def test_price_bounds_eligibility_and_repeat_changes_recompute_proposal():
    args = inputs()
    first = propose(args)
    args[1]["b"]["input_per_1m"] *= 2
    repriced = propose(args)
    assert repriced["budget_config"]["hard_cap_usd"] > first["budget_config"]["hard_cap_usd"]
    assert repriced["provenance"]["catalog_hash"] != first["provenance"]["catalog_hash"]
    args[3][args[0][1].episode_id]["b"]["eligible"] = False
    excluded = propose(args)
    assert excluded["stages"]["development"]["configuration_count"] == 104
    repeated = propose(inputs(repeats=3))
    assert repeated["episodes"] == 150 and repeated["independent_groups"] == 50
    assert repeated["stages"]["development"]["model_reservation_usd"] == pytest.approx(3 * first["stages"]["development"]["model_reservation_usd"])
    row = args[0][0]
    args[0][0] = replace(row, max_calls=row.max_calls * 2, max_input_tokens=row.max_input_tokens * 2)
    bounded = propose(args)
    assert bounded["stages"]["pilot"]["model_reservation_usd"] > repriced["stages"]["pilot"]["model_reservation_usd"]


def test_fallback_baseline_and_contingency_are_reserved_conservatively():
    args = inputs()
    for row in args[0]:
        args[2][row.episode_id] = {"models": {"planner": "baseline", "synthesis": "b"}}
    proposal = propose(args, contingency_fraction=.25)
    pilot = proposal["stages"]["pilot"]
    assert pilot["model_reservation_usd"] == pytest.approx(5 * (.0068 + .0024))
    assert pilot["contingency_usd"] == pytest.approx(.25 * pilot["model_reservation_usd"])
    assert pilot["stage_cap_usd"] >= pilot["model_reservation_usd"] + pilot["contingency_usd"]
    assert proposal == propose(args, contingency_fraction=.25)


@pytest.mark.parametrize("price", [None, True, -1, float("nan"), float("inf")])
def test_invalid_or_unknown_prices_are_not_called_free(price):
    args = inputs()
    args[1]["a"]["input_per_1m"] = price
    with pytest.raises(ValueError, match="finite nonnegative"):
        propose(args)


def test_unpriced_charges_missing_eligibility_and_unknown_graders_block():
    args = inputs()
    args[1]["a"]["request_cost_usd"] = .1
    with pytest.raises(ValueError, match="bounded reservation adapter"):
        propose(args)
    args = inputs()
    del args[3][args[0][0].episode_id]["a"]
    with pytest.raises(ValueError, match="incomplete frozen eligibility"):
        propose(args)
    args = inputs()
    args[0][0] = replace(args[0][0], grader={"kind": "paid_judge"})
    with pytest.raises(ValueError, match="unsupported grading costs"):
        propose(args)


def test_documented_zero_prices_remain_zero_and_renderer_uses_actual_proposal():
    from theriad.routing.episode_planning import render_budget_proposal

    args = inputs()
    for quote in args[1].values():
        quote.update(input_per_1m=0, output_per_1m=0)
    proposal = propose(args)
    assert proposal["budget_config"]["hard_cap_usd"] == 0
    assert set(proposal["budget_config"]["stage_caps_usd"].values()) == {0}
    rendered = render_budget_proposal(proposal)
    assert "50 episode rows" in rendered
    assert "$0.00" in rendered
    assert "does not authorize paid calls" in rendered
    assert "35 | 35 | 105 | 210" in rendered
