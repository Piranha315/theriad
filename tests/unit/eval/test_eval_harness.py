"""Tests for the router evaluation harness."""
from __future__ import annotations

import json

import pytest

from theriad.eval.corpus import EvalItem, load_corpus, vintages, write_corpus
from theriad.eval.executors import CallResult, is_truncated, static_executor
from theriad.eval.grid import (
    STRATUM_EASY,
    STRATUM_UNSOLVED,
    Cell,
    ModelCoverage,
    OutcomeGrid,
    PlannedCall,
    derive_strata,
)
from theriad.eval.oracle import (
    Action,
    ExpectedValueOracle,
    Posterior,
    best_of_n_probability,
    enumerate_actions,
    score_action,
    stratum_posterior,
)
from theriad.eval.regret import evaluate_router
from theriad.eval.run import (
    call_costs,
    default_coverage,
    plan_grid,
    populate_grid,
)


def _item(item_id: str, answer: str = "A", vintage: int = 2020, **kw) -> EvalItem:
    return EvalItem(
        id=item_id,
        category=kw.pop("category", "qa"),
        prompt=f"question {item_id}",
        answer=answer,
        answer_type=kw.pop("answer_type", "multiple_choice"),
        source=kw.pop("source", "test"),
        vintage=vintage,
        **kw,
    )


# ── Corpus ────────────────────────────────────────────────────────────────────

def test_corpus_roundtrip(tmp_path):
    items = [_item("a", vintage=2020), _item("b", vintage=2026)]
    path = tmp_path / "corpus.jsonl"
    write_corpus(items, path)
    assert load_corpus(path) == items


def test_corpus_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "dupes.jsonl"
    row = json.dumps(_item("same").to_dict())
    path.write_text(row + "\n" + row + "\n")
    with pytest.raises(ValueError, match="duplicate item id"):
        load_corpus(path)


def test_corpus_rejects_malformed_lines(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json\n")
    with pytest.raises(ValueError, match="malformed"):
        load_corpus(path)


def test_corpus_skips_blank_and_comment_lines(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(f"\n# a comment\n{json.dumps(_item('x').to_dict())}\n")
    assert len(load_corpus(path)) == 1


def test_vintages_counts_and_sorts():
    items = [_item("a", vintage=2026), _item("b", vintage=2020), _item("c", vintage=2020)]
    assert list(vintages(items).items()) == [(2020, 2), (2026, 1)]


def test_verifiable_requires_a_keyed_answer_type():
    assert _item("a").verifiable is True
    assert EvalItem(id="b", category="qa", prompt="p").verifiable is False


def test_grading_delegates_to_benchmark_quality_check():
    item = _item("a", answer="C")
    assert item.grade("The answer is C") is True
    assert item.grade("The answer is D") is False


# ── Coverage ──────────────────────────────────────────────────────────────────

def test_full_coverage_covers_everything():
    cov = ModelCoverage(model="m", tier="cheap", item_fraction=1.0)
    assert all(cov.covers(f"item{i}") for i in range(50))


def test_zero_coverage_covers_nothing():
    cov = ModelCoverage(model="m", tier="frontier", item_fraction=0.0)
    assert not any(cov.covers(f"item{i}") for i in range(50))


def test_partial_coverage_is_deterministic():
    cov = ModelCoverage(model="m", tier="premium", item_fraction=0.3)
    first = [cov.covers(f"item{i}") for i in range(200)]
    second = [ModelCoverage(model="m", tier="premium", item_fraction=0.3).covers(f"item{i}")
              for i in range(200)]
    assert first == second


def test_partial_coverage_hits_approximately_the_fraction():
    cov = ModelCoverage(model="m", tier="premium", item_fraction=0.25)
    hits = sum(cov.covers(f"item{i}") for i in range(4000))
    assert 0.21 < hits / 4000 < 0.29


def test_coverage_subsets_differ_across_models():
    """Independent subsamples per model, so cheap models aren't measured on the
    same narrow slice the expensive ones happen to land on."""
    a = ModelCoverage(model="model-a", tier="t", item_fraction=0.3)
    b = ModelCoverage(model="model-b", tier="t", item_fraction=0.3)
    ids = [f"item{i}" for i in range(300)]
    assert [a.covers(i) for i in ids] != [b.covers(i) for i in ids]


def test_invalid_coverage_is_rejected():
    with pytest.raises(ValueError):
        ModelCoverage(model="m", tier="t", item_fraction=1.5)
    with pytest.raises(ValueError):
        ModelCoverage(model="m", tier="t", samples_per_item=0)


def test_default_coverage_is_asymmetric():
    plan = default_coverage(cheap=["c"], premium=["p"], frontier=["f"])
    fractions = {c.model: c.item_fraction for c in plan}
    assert fractions["c"] > fractions["p"] > fractions["f"] > 0.0


# ── Grid ──────────────────────────────────────────────────────────────────────

def test_record_accumulates_trials_and_cost():
    grid = OutcomeGrid()
    grid.record("m", "i", True, 0.01)
    grid.record("m", "i", False, 0.01)
    cell = grid.cell("m", "i")
    assert (cell.trials, cell.successes) == (2, 1)
    assert cell.cost_usd == pytest.approx(0.02)
    assert grid.total_trials == 2


def test_raw_rate_is_none_without_trials():
    assert Cell(model="m", item_id="i").raw_rate is None


def test_grid_roundtrip(tmp_path):
    grid = OutcomeGrid()
    grid.record("m", "i", True, 0.5)
    path = tmp_path / "grid.json"
    grid.save(path)
    loaded = OutcomeGrid.load(path)
    assert loaded.cell("m", "i").successes == 1
    assert loaded.total_cost_usd == pytest.approx(0.5)


def test_loading_a_missing_grid_returns_empty():
    assert len(OutcomeGrid.load("/nonexistent/grid.json")) == 0


def test_plan_requests_k_samples_per_covered_cell():
    items = [_item("a"), _item("b")]
    cov = [ModelCoverage(model="m", tier="cheap", item_fraction=1.0, samples_per_item=3)]
    plan = OutcomeGrid().plan(items, cov)
    assert {c.item_id for c in plan} == {"a", "b"}
    assert all(c.n_samples == 3 for c in plan)


def test_plan_subtracts_work_already_done():
    items = [_item("a")]
    cov = [ModelCoverage(model="m", tier="cheap", samples_per_item=5)]
    grid = OutcomeGrid()
    for _ in range(2):
        grid.record("m", "a", True)
    assert grid.plan(items, cov)[0].n_samples == 3


def test_plan_omits_completed_cells():
    items = [_item("a")]
    cov = [ModelCoverage(model="m", tier="cheap", samples_per_item=2)]
    grid = OutcomeGrid()
    for _ in range(2):
        grid.record("m", "a", True)
    assert grid.plan(items, cov) == []


def test_plan_skips_models_with_zero_coverage():
    items = [_item("a")]
    cov = [ModelCoverage(model="skipped", tier="frontier", item_fraction=0.0)]
    assert OutcomeGrid().plan(items, cov) == []


# ── Strata ────────────────────────────────────────────────────────────────────

def test_strata_derive_from_measurements_not_vintage():
    items = [_item("easy", vintage=2026), _item("nobody", vintage=2020)]
    grid = OutcomeGrid()
    for model in ("weak", "strong"):
        grid.record(model, "easy", True)
        grid.record(model, "nobody", False)
    strata = derive_strata(grid, items)
    # The 2026 item is "easy" and the 2020 item "unsolved" — the opposite of what
    # a vintage-based difficulty label would have assigned.
    assert strata.stratum("easy") == STRATUM_EASY
    assert strata.stratum("nobody") == STRATUM_UNSOLVED


def test_strata_ignore_items_with_no_measurements():
    strata = derive_strata(OutcomeGrid(), [_item("unmeasured")])
    assert "unmeasured" not in strata.by_item


# ── Posterior ─────────────────────────────────────────────────────────────────

def test_posterior_never_collapses_to_zero():
    """Zero observed successes must not read as impossible."""
    p = Posterior(successes=0, trials=5)
    assert p.mean > 0.0
    assert p.mean == pytest.approx(1 / 7)


def test_posterior_never_collapses_to_one():
    p = Posterior(successes=5, trials=5)
    assert p.mean < 1.0


def test_posterior_with_no_trials_is_uniform():
    assert Posterior(successes=0, trials=0).mean == pytest.approx(0.5)


def test_posterior_tightens_as_trials_grow():
    assert Posterior(1, 10).stdev > Posterior(10, 100).stdev


def test_interval_is_wide_at_small_k():
    """Documents why per-item estimates can't rank models: 1/10 spans ~0.02-0.41."""
    lo, hi = Posterior(successes=1, trials=10).interval()
    assert hi - lo > 0.3


def test_stratum_pooling_tightens_the_estimate():
    grid = OutcomeGrid()
    for i in range(20):
        grid.record("m", f"item{i}", i % 5 == 0)
    per_item = Posterior(*_counts(grid, "m", "item0"))
    pooled = stratum_posterior(grid, "m", [f"item{i}" for i in range(20)])
    assert pooled.stdev < per_item.stdev


def _counts(grid, model, item_id):
    cell = grid.cell(model, item_id)
    return cell.successes, cell.trials


# ── Best-of-n ─────────────────────────────────────────────────────────────────

def test_best_of_n_matches_the_closed_form():
    assert best_of_n_probability(0.08, 5) == pytest.approx(1 - 0.92**5)


def test_best_of_one_is_the_single_sample_probability():
    assert best_of_n_probability(0.3, 1) == 0.3


def test_resampling_cheap_can_beat_one_strong_call_at_equal_spend():
    """The concrete case: p=0.08 five times beats p=0.20 once when strong is 5x."""
    cheap = score_action(Action("cheap", 5), 0.08, cost_per_call=1.0, value_correct=100)
    strong = score_action(Action("strong", 1), 0.20, cost_per_call=5.0, value_correct=100)
    assert cheap.cost_usd == strong.cost_usd
    assert cheap.success_prob > strong.success_prob
    assert cheap.expected_value > strong.expected_value


def test_best_of_n_is_unavailable_without_a_verifier():
    actions = enumerate_actions(["m"], max_samples=5, verifiable=False)
    assert all(a.n_samples == 1 for a in actions)


def test_best_of_n_is_offered_when_verifiable():
    actions = enumerate_actions(["m"], max_samples=3, verifiable=True)
    assert {a.n_samples for a in actions} == {1, 2, 3}


# ── Oracle ────────────────────────────────────────────────────────────────────

def _two_model_grid():
    """cheap solves easy items only; strong solves both."""
    grid = OutcomeGrid()
    for _ in range(10):
        grid.record("cheap", "easy", True)
        grid.record("cheap", "hard", False)
        grid.record("strong", "easy", True)
        grid.record("strong", "hard", True)
    return grid


def test_oracle_prefers_cheap_when_both_succeed():
    grid = _two_model_grid()
    oracle = ExpectedValueOracle(
        grid, {"cheap": 0.001, "strong": 0.01}, value_correct=1.0, max_samples=1
    )
    assert oracle.evaluate(_item("easy")).best.action.model == "cheap"


def test_oracle_escalates_when_the_gap_justifies_the_cost():
    grid = _two_model_grid()
    oracle = ExpectedValueOracle(
        grid, {"cheap": 0.001, "strong": 0.01}, value_correct=10.0, max_samples=1
    )
    assert oracle.evaluate(_item("hard")).best.action.model == "strong"


def test_oracle_declines_to_escalate_when_correctness_is_cheap():
    """Same measurements, low V: the accuracy gain isn't worth the spend."""
    grid = _two_model_grid()
    oracle = ExpectedValueOracle(
        grid, {"cheap": 0.0, "strong": 100.0}, value_correct=0.001, max_samples=1
    )
    assert oracle.evaluate(_item("hard")).best.action.model == "cheap"


def test_value_of_correctness_changes_the_oracle_action():
    """V is load-bearing — hence sweeping it rather than fixing one value."""
    grid = _two_model_grid()
    costs = {"cheap": 0.001, "strong": 0.05}
    low = ExpectedValueOracle(grid, costs, value_correct=0.01, max_samples=1)
    high = ExpectedValueOracle(grid, costs, value_correct=100.0, max_samples=1)
    assert low.evaluate(_item("hard")).best.action.model == "cheap"
    assert high.evaluate(_item("hard")).best.action.model == "strong"


def test_oracle_ranks_all_actions():
    grid = _two_model_grid()
    oracle = ExpectedValueOracle(grid, {"cheap": 0.001, "strong": 0.01}, 1.0, max_samples=1)
    ranked = oracle.evaluate(_item("easy")).all_actions
    assert [a.expected_value for a in ranked] == sorted(
        (a.expected_value for a in ranked), reverse=True
    )


def test_unmeasured_model_falls_back_to_the_uniform_prior():
    oracle = ExpectedValueOracle(OutcomeGrid(), {"unknown": 0.0}, 1.0, max_samples=1)
    assert oracle.success_probability("unknown", _item("x")) == pytest.approx(0.5)


# ── Regret ────────────────────────────────────────────────────────────────────

def test_matching_the_oracle_yields_zero_regret():
    grid = _two_model_grid()
    items = [_item("easy"), _item("hard")]
    costs = {"cheap": 0.001, "strong": 0.01}
    oracle = ExpectedValueOracle(grid, costs, value_correct=10.0, max_samples=1)
    optimal = {i.id: oracle.evaluate(i).best.action for i in items}
    report = evaluate_router(lambda i: optimal[i.id], items, oracle, costs)
    assert report.total_regret == pytest.approx(0.0)
    assert report.match_rate == 1.0


def test_always_cheap_router_shows_accuracy_loss_not_overspend():
    grid = _two_model_grid()
    items = [_item("easy"), _item("hard")]
    costs = {"cheap": 0.001, "strong": 0.01}
    oracle = ExpectedValueOracle(grid, costs, value_correct=10.0, max_samples=1)
    report = evaluate_router(lambda i: Action("cheap"), items, oracle, costs)
    assert report.total_regret > 0
    assert report.accuracy_loss_value > 0
    assert report.overspend == pytest.approx(0.0)


def test_always_strong_router_shows_overspend_not_accuracy_loss():
    grid = _two_model_grid()
    items = [_item("easy"), _item("hard")]
    costs = {"cheap": 0.001, "strong": 0.01}
    oracle = ExpectedValueOracle(grid, costs, value_correct=10.0, max_samples=1)
    report = evaluate_router(lambda i: Action("strong"), items, oracle, costs)
    assert report.overspend > 0
    assert report.accuracy_loss_value == pytest.approx(0.0)


def test_regret_is_never_negative():
    grid = _two_model_grid()
    items = [_item("easy"), _item("hard")]
    costs = {"cheap": 0.001, "strong": 0.01}
    oracle = ExpectedValueOracle(grid, costs, value_correct=5.0, max_samples=1)
    for router in (lambda i: Action("cheap"), lambda i: Action("strong")):
        report = evaluate_router(router, items, oracle, costs)
        assert all(r.regret >= -1e-9 for r in report.items)


def test_report_breaks_down_by_vintage():
    grid = _two_model_grid()
    items = [_item("easy", vintage=2020), _item("hard", vintage=2026)]
    costs = {"cheap": 0.001, "strong": 0.01}
    oracle = ExpectedValueOracle(grid, costs, value_correct=10.0, max_samples=1)
    report = evaluate_router(lambda i: Action("cheap"), items, oracle, costs)
    assert set(report.by_vintage()) == {2020, 2026}


def test_summary_is_serialisable():
    grid = _two_model_grid()
    items = [_item("easy")]
    costs = {"cheap": 0.001, "strong": 0.01}
    oracle = ExpectedValueOracle(grid, costs, value_correct=1.0, max_samples=1)
    summary = evaluate_router(lambda i: Action("cheap"), items, oracle, costs).summary()
    json.dumps(summary)


# ── Planning and population ───────────────────────────────────────────────────

def test_plan_cost_scales_with_samples_and_coverage():
    items = [_item(f"i{n}") for n in range(10)]
    costs = {"m": 0.01}
    cheap_plan = plan_grid(
        OutcomeGrid(), items,
        [ModelCoverage(model="m", tier="c", item_fraction=1.0, samples_per_item=2)],
        costs,
    )
    assert cheap_plan.n_calls == 20
    assert cheap_plan.estimated_cost_usd == pytest.approx(0.20)


def test_populate_grades_outputs_against_the_answer_key():
    items = [_item("a", answer="A"), _item("b", answer="B")]
    grid = OutcomeGrid()
    cov = [ModelCoverage(model="m", tier="c", item_fraction=1.0, samples_per_item=1)]
    plan = plan_grid(grid, items, cov, {"m": 0.0})
    executor = static_executor(
        {("m", "a"): "The answer is A", ("m", "b"): "The answer is Z"}
    )
    populate_grid(grid, items, plan, executor)
    assert grid.cell("m", "a").successes == 1
    assert grid.cell("m", "b").successes == 0


def test_truncated_calls_are_excluded_from_trials_not_scored_as_failures():
    """A cut-off response is missing data. Scoring it wrong would bias the grid
    against models that reason at length, on exactly the hard items."""
    items = [_item("a", answer="A")]
    grid = OutcomeGrid()
    plan = plan_grid(
        grid, items,
        [ModelCoverage(model="m", tier="c", samples_per_item=1)],
        {"m": 0.02},
    )
    populate_grid(
        grid, items, plan,
        static_executor({("m", "a"): "long reasoning that got cut"},
                        cost_per_call=0.02, truncate={("m", "a")}),
    )
    cell = grid.cell("m", "a")
    assert cell.trials == 0            # not counted as an attempt at the answer
    assert cell.successes == 0
    assert cell.truncated == 1
    assert cell.cost_usd == pytest.approx(0.02)   # but we still paid
    assert cell.truncation_rate == 1.0


def test_truncated_cells_are_surfaced():
    grid = OutcomeGrid()
    grid.record("m", "a", False, 0.01, truncated=True)
    grid.record("m", "b", True, 0.01)
    flagged = grid.truncated_cells()
    assert [c.item_id for c in flagged] == ["a"]
    assert grid.total_truncated == 1


def test_truncation_retries_are_bounded():
    """A cell that always truncates must not be re-planned forever."""
    items = [_item("a")]
    cov = [ModelCoverage(model="m", tier="c", samples_per_item=2)]
    grid = OutcomeGrid()
    for _ in range(6):
        grid.record("m", "a", False, 0.0, truncated=True)
    assert grid.plan(items, cov, attempt_multiplier=3.0) == []


def test_explicit_truncated_recovery_reopens_exhausted_cells():
    items = [_item("a")]
    cov = [ModelCoverage(model="m", tier="c", samples_per_item=1)]
    grid = OutcomeGrid()
    for _ in range(6):
        grid.record("m", "a", False, 0.0, truncated=True)
    plan = grid.plan(items, cov, retry_truncated=True)
    assert plan == [PlannedCall(model="m", item_id="a", n_samples=1)]


def test_truncation_is_retried_within_budget():
    items = [_item("a")]
    cov = [ModelCoverage(model="m", tier="c", samples_per_item=2)]
    grid = OutcomeGrid()
    grid.record("m", "a", False, 0.0, truncated=True)
    assert grid.plan(items, cov, attempt_multiplier=3.0)[0].n_samples == 2


def test_is_truncated_recognises_length_finish_reasons():
    assert is_truncated("length") is True
    assert is_truncated("max_tokens") is True
    assert is_truncated("stop") is False
    assert is_truncated(None) is False


def test_attempts_counts_paid_calls_including_truncations():
    grid = OutcomeGrid()
    grid.record("m", "a", True, 0.0)
    grid.record("m", "a", False, 0.0, truncated=True)
    assert grid.cell("m", "a").attempts == 2
    assert grid.cell("m", "a").trials == 1


def test_generation_cap_defaults_to_uncapped():
    """The costing assumption must never leak into max_tokens."""
    from theriad.eval.run import DEFAULT_GENERATION_CAP

    assert DEFAULT_GENERATION_CAP is None


def test_populate_does_not_record_executor_failures():
    """An errored call is missing data, not evidence the model answered wrong.

    Recording it as an unsuccessful trial makes an outage indistinguishable from
    incompetence, and permanently poisons the cell because resume then skips it.
    """
    items = [_item("a")]
    grid = OutcomeGrid()
    plan = plan_grid(
        grid, items,
        [ModelCoverage(model="m", tier="c", samples_per_item=2)],
        {"m": 0.0},
    )

    def _boom(model, item):
        raise RuntimeError("provider exploded")

    seen = []
    populate_grid(grid, items, plan, _boom, on_error=lambda m, i, e: seen.append(m))

    assert grid.cell("m", "a") is None or grid.cell("m", "a").trials == 0
    assert seen == ["m", "m"]
    # The cell is absent, so a resume retries rather than skipping it forever.
    assert plan_grid(
        grid, items,
        [ModelCoverage(model="m", tier="c", samples_per_item=2)],
        {"m": 0.0},
    ).n_calls == 2


def test_populate_aborts_after_sustained_errors():
    """Throttling affects every remaining call; spinning through them is noise."""
    from theriad.eval.run import PopulateAborted

    items = [_item(f"i{n}") for n in range(40)]
    grid = OutcomeGrid()
    plan = plan_grid(
        grid, items,
        [ModelCoverage(model="m", tier="c", samples_per_item=1)],
        {"m": 0.0},
    )
    calls = []

    def _boom(model, item):
        calls.append(item.id)
        raise RuntimeError("429 rate limited")

    with pytest.raises(PopulateAborted):
        populate_grid(grid, items, plan, _boom, max_consecutive_errors=5)
    # Stopped at the breaker instead of burning the whole plan.
    assert len(calls) == 5


def test_populate_breaker_resets_on_success():
    """Intermittent errors must not accumulate toward the abort threshold."""
    items = [_item(f"i{n}", answer="A") for n in range(6)]
    grid = OutcomeGrid()
    plan = plan_grid(
        grid, items,
        [ModelCoverage(model="m", tier="c", samples_per_item=1)],
        {"m": 0.0},
    )
    state = {"n": 0}

    def _flaky(model, item):
        state["n"] += 1
        if state["n"] % 2:
            raise RuntimeError("transient")
        return CallResult(output="The answer is A", cost_usd=0.0)

    populate_grid(grid, items, plan, _flaky, max_consecutive_errors=3)
    assert grid.total_trials == 3


def test_populate_is_resumable():
    items = [_item("a", answer="A")]
    grid = OutcomeGrid()
    cov = [ModelCoverage(model="m", tier="c", samples_per_item=4)]
    executor = static_executor({("m", "a"): "The answer is A"})

    populate_grid(grid, items, plan_grid(grid, items, cov[:1], {"m": 0.0}), executor)
    assert grid.cell("m", "a").trials == 4
    # A second pass has nothing left to do.
    assert plan_grid(grid, items, cov, {"m": 0.0}).n_calls == 0


def test_call_costs_resolve_from_the_pricing_table():
    costs = call_costs(["openai/gpt-4o-mini"], input_tokens=1000, output_tokens=1000)
    # 0.15/1M in + 0.60/1M out over 1k tokens each.
    assert costs["openai/gpt-4o-mini"] == pytest.approx(0.00075)


def test_unpriced_models_score_as_free_and_are_reported():
    from theriad.eval.run import unpriced

    assert unpriced(["totally/unknown-model-xyz"]) == ["totally/unknown-model-xyz"]


# ── Benchmark conversion ──────────────────────────────────────────────────────

from theriad.eval.benchmarks import (  # noqa: E402
    DEFAULT_SOURCES,
    BenchmarkSource,
    build_corpus,
    row_to_item,
)

_MC = BenchmarkSource(
    name="mc", hf_dataset="x/y", vintage=2020, category="qa",
    answer_type="multiple_choice", question_field="q", answer_field="a",
    choices_field="choices",
)
_EXACT = BenchmarkSource(
    name="ex", hf_dataset="x/y", vintage=2021, category="reasoning",
    answer_type="exact_match", question_field="q", answer_field="a",
)


def test_mc_row_with_integer_answer_index():
    item = row_to_item(_MC, {"q": "Pick", "choices": ["w", "x", "y"], "a": 2}, 0)
    assert item.answer == "C"
    assert "C. y" in item.prompt


def test_mc_row_with_literal_answer_text():
    item = row_to_item(_MC, {"q": "Pick", "choices": ["w", "x"], "a": "x"}, 1)
    assert item.answer == "B"


def test_mc_row_with_letter_answer():
    item = row_to_item(_MC, {"q": "Pick", "choices": ["w", "x"], "a": "B"}, 2)
    assert item.answer == "B"


def test_mc_row_with_out_of_range_index_is_dropped():
    assert row_to_item(_MC, {"q": "Pick", "choices": ["w"], "a": 5}, 0) is None


def test_mc_row_without_choices_is_dropped():
    assert row_to_item(_MC, {"q": "Pick", "a": 0}, 0) is None


def test_exact_row_strips_gsm8k_rationale():
    item = row_to_item(_EXACT, {"q": "2+2?", "a": "reasoning here\n#### 4"}, 0)
    assert item.answer == "4"


def test_rows_missing_fields_are_dropped():
    assert row_to_item(_EXACT, {"q": "", "a": "4"}, 0) is None
    assert row_to_item(_EXACT, {"q": "x"}, 0) is None


def test_converted_items_are_never_token_capped():
    """A cap here would truncate reasoning models on exactly the hard items."""
    item = row_to_item(_EXACT, {"q": "2+2?", "a": "4"}, 0)
    assert item.max_output_tokens is None


def test_converted_items_carry_vintage_and_source():
    item = row_to_item(_MC, {"q": "Pick", "choices": ["w", "x"], "a": 0}, 0)
    assert (item.source, item.vintage) == ("mc", 2020)


def test_build_corpus_spans_sources_and_samples_down():
    def fake_fetch(source, limit, token):
        return [
            {"q": f"q{i}", "a": 0, "choices": ["w", "x"]} for i in range(limit)
        ]

    corpus = build_corpus([_MC, _MC], per_source=5, fetcher=fake_fetch)
    assert len(corpus) == 10


def test_build_corpus_is_deterministic_under_a_seed():
    def fake_fetch(source, limit, token):
        return [{"q": f"q{i}", "a": 0, "choices": ["w", "x"]} for i in range(limit)]

    a = build_corpus([_MC], per_source=5, seed=7, fetcher=fake_fetch)
    b = build_corpus([_MC], per_source=5, seed=7, fetcher=fake_fetch)
    assert [i.id for i in a] == [i.id for i in b]


def test_default_sources_span_multiple_vintages():
    years = {s.vintage for s in DEFAULT_SOURCES}
    assert len(years) >= 3
    assert min(years) <= 2021 and max(years) >= 2024


def test_default_sources_are_all_mechanically_gradable():
    assert all(
        s.answer_type in {"multiple_choice", "exact_match"} for s in DEFAULT_SOURCES
    )


# ── Source verification ───────────────────────────────────────────────────────

from theriad.eval.benchmarks import verify_source, verify_sources  # noqa: E402


def test_verify_source_passes_on_a_correct_mapping():
    check = verify_source(
        _MC, fetcher=lambda s, n, t: [{"q": "Pick", "choices": ["w", "x"], "a": 1}]
    )
    assert check.ok
    assert check.sample_answer == "B"


def test_verify_source_names_the_missing_field_and_lists_alternatives():
    """The failure mode this guards: a drifted schema yields nonsense, not an error."""
    check = verify_source(
        _MC, fetcher=lambda s, n, t: [{"prompt": "Pick", "opts": ["w"], "label": 0}]
    )
    assert not check.ok
    assert "missing field" in check.reason
    assert check.available_fields == ["label", "opts", "prompt"]


def test_verify_source_catches_fields_present_but_unconvertible():
    check = verify_source(
        _MC, fetcher=lambda s, n, t: [{"q": "Pick", "choices": [], "a": 0}]
    )
    assert not check.ok


def test_verify_source_reports_fetch_errors():
    def boom(source, limit, token):
        raise RuntimeError("gated dataset")

    check = verify_source(_MC, fetcher=boom)
    assert not check.ok and "gated" in check.reason


def test_verify_source_handles_empty_results():
    check = verify_source(_MC, fetcher=lambda s, n, t: [])
    assert not check.ok and "no rows" in check.reason


def test_verify_sources_preserves_order():
    checks = verify_sources(
        [_MC, _EXACT],
        fetcher=lambda s, n, t: [{"q": "x", "a": "1", "choices": ["w", "x"]}],
    )
    assert [c.source.name for c in checks] == ["mc", "ex"]


# ── Assembled choices (GPQA-style) ────────────────────────────────────────────

_ASSEMBLED = BenchmarkSource(
    name="gpqa_like", hf_dataset="x/y", vintage=2023, category="reasoning",
    answer_type="multiple_choice", question_field="Question",
    answer_field="Correct Answer",
    distractor_fields=("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"),
)

_GPQA_ROW = {
    "Question": "Which is correct?",
    "Correct Answer": "right",
    "Incorrect Answer 1": "wrong1",
    "Incorrect Answer 2": "wrong2",
    "Incorrect Answer 3": "wrong3",
}


def test_assembled_choices_builds_all_four_options():
    item = row_to_item(_ASSEMBLED, _GPQA_ROW, 0)
    for text in ("right", "wrong1", "wrong2", "wrong3"):
        assert text in item.prompt


def test_assembled_answer_letter_points_at_the_correct_option():
    item = row_to_item(_ASSEMBLED, _GPQA_ROW, 0)
    line = next(
        ln for ln in item.prompt.splitlines() if ln.startswith(f"{item.answer}.")
    )
    assert line.endswith("right")


def test_assembled_choices_shuffle_is_deterministic_per_item():
    """The grid is keyed on item id and accumulates across runs, so the same id
    must always produce the same option order."""
    a = row_to_item(_ASSEMBLED, _GPQA_ROW, 7)
    b = row_to_item(_ASSEMBLED, _GPQA_ROW, 7)
    assert a.prompt == b.prompt and a.answer == b.answer


def test_assembled_choices_are_not_always_in_position_a():
    """Leaving the key first would measure position bias, not capability."""
    letters = {row_to_item(_ASSEMBLED, _GPQA_ROW, i).answer for i in range(40)}
    assert len(letters) > 1


def test_assembled_source_without_distractors_is_dropped():
    row = {"Question": "q", "Correct Answer": "right"}
    assert row_to_item(_ASSEMBLED, row, 0) is None


def test_verify_checks_distractor_fields():
    check = verify_source(_ASSEMBLED, fetcher=lambda s, n, t: [_GPQA_ROW])
    assert check.ok

    partial = verify_source(
        _ASSEMBLED,
        fetcher=lambda s, n, t: [{"Question": "q", "Correct Answer": "right"}],
    )
    assert not partial.ok
    assert "Incorrect Answer 1" in partial.reason


def test_assembles_choices_flag():
    assert _ASSEMBLED.assembles_choices is True
    assert _MC.assembles_choices is False


# ── Gradability (the A-J / last-match bugs) ───────────────────────────────────

from theriad.benchmark.types import _extract_mc_letter  # noqa: E402


@pytest.mark.parametrize("letter", list("ABCDEFGHIJ"))
def test_all_ten_choice_letters_are_extractable(letter):
    """MMLU-Pro presents ten options; an A-E range made F-J ungradable."""
    assert _extract_mc_letter(f"ANSWER: {letter}") == letter


def test_answer_at_the_end_of_a_long_response_is_found():
    """The harness sends no max_tokens, so models reason at length by design and
    the conclusion lives at the end."""
    text = "Working through it. Option C looks plausible.\n" * 50 + "ANSWER: D"
    assert _extract_mc_letter(text) == "D"


def test_last_match_wins_over_options_discussed_first():
    assert _extract_mc_letter("It isn't A. It isn't B. The answer is C") == "C"


@pytest.mark.parametrize(
    "text,expected",
    [("The answer is B", "B"), ("(C)", "C"), ("A. because", "A"), ("D", "D")],
)
def test_pre_existing_formats_still_extract(text, expected):
    assert _extract_mc_letter(text) == expected


def test_prose_pronoun_does_not_become_answer_i():
    """Uncommitted prose is not an explicit choice, including the pronoun I."""
    assert _extract_mc_letter("I think it is B maybe") is None


def test_mmlu_pro_style_item_grades_end_to_end():
    item = EvalItem(
        id="p", category="qa", prompt="", answer="I",
        answer_type="multiple_choice", source="mmlu_pro", vintage=2024,
    )
    assert item.grade("Reasoning...\n" * 30 + "ANSWER: I") is True


def test_verify_rejects_a_converted_but_ungradable_source():
    """Conversion succeeding means well-formed; only the round-trip means scoreable."""
    ten = BenchmarkSource(
        name="ten", hf_dataset="x/y", vintage=2024, category="qa",
        answer_type="multiple_choice", question_field="q", answer_field="a",
        choices_field="choices",
    )
    row = {"q": "Pick", "choices": [f"opt{i}" for i in range(10)], "a": 8}
    check = verify_source(ten, fetcher=lambda s, n, t: [row])
    assert check.ok  # 'I' is now reachable; was the bug this test pins
    assert "gradable=yes" in check.reason


# ── Exact-match grading (GSM8K / HLE) ─────────────────────────────────────────

def _exact(answer: str) -> EvalItem:
    return EvalItem(
        id="e", category="reasoning", prompt="", answer=answer,
        answer_type="exact_match", source="gsm8k", vintage=2021,
    )


def test_exact_match_reads_the_answer_marker():
    """The prompts request 'ANSWER: X'; comparing the whole output scored it wrong."""
    assert _exact("18").grade("ANSWER: 18") is True


def test_exact_match_survives_a_reasoning_chain():
    """A reasoning benchmark where showing your work scores zero is backwards."""
    assert _exact("18").grade("He starts with 20...\n" * 30 + "ANSWER: 18") is True


def test_exact_match_still_accepts_a_bare_answer():
    assert _exact("18").grade("18") is True


@pytest.mark.parametrize("response", ["$18", "18.0", "18.", "ANSWER: $18"])
def test_exact_match_ignores_formatting_noise(response):
    assert _exact("18").grade(response) is True


@pytest.mark.parametrize("response", ["ANSWER: 19", "ANSWER: 1800", "ANSWER: "])
def test_exact_match_still_rejects_wrong_answers(response):
    assert _exact("18").grade(response) is False


def test_exact_match_handles_thousands_separators_either_way():
    assert _exact("1,000").grade("ANSWER: 1000") is True
    assert _exact("1000").grade("ANSWER: 1,000") is True
    assert _exact("1,000").grade("ANSWER: 100") is False


def test_exact_match_works_for_text_answers():
    item = EvalItem(
        id="t", category="qa", prompt="", answer="Paris",
        answer_type="exact_match", source="hle", vintage=2025,
    )
    assert item.grade("It is in France.\nANSWER: Paris") is True
    assert item.grade("ANSWER: London") is False


def test_last_marker_wins_for_exact_match():
    assert _exact("18").grade("ANSWER: 12\nOn reflection.\nANSWER: 18") is True


def test_verify_round_trip_now_passes_for_exact_match_sources():
    src = BenchmarkSource(
        name="gsm_like", hf_dataset="x/y", vintage=2021, category="reasoning",
        answer_type="exact_match", question_field="q", answer_field="a",
    )
    check = verify_source(
        src, fetcher=lambda s, n, t: [{"q": "2+2?", "a": "steps\n#### 4"}]
    )
    assert check.ok and "gradable=yes" in check.reason


# ── Sample correlation / best-of-n discounting ────────────────────────────────

from theriad.eval.oracle import estimate_sample_correlation  # noqa: E402
from theriad.eval.routers import (  # noqa: E402
    fixed_router,
    theriad_router,
    routed_models,
    unmeasured_models,
)


def test_zero_correlation_is_the_independent_formula():
    assert best_of_n_probability(0.08, 5, 0.0) == pytest.approx(1 - 0.92**5)


def test_full_correlation_makes_extra_samples_worthless():
    """Five variations of the same wrong answer are worth one sample."""
    assert best_of_n_probability(0.08, 5, 1.0) == pytest.approx(0.08)


def test_partial_correlation_lands_between():
    independent = best_of_n_probability(0.08, 5, 0.0)
    half = best_of_n_probability(0.08, 5, 0.5)
    assert 0.08 < half < independent


def test_oracle_defaults_to_single_call_actions():
    """Best-of-n must be opt-in: without a verifier at inference time it is a
    bound no router can reach, which collapses match rate to zero."""
    grid = _two_model_grid()
    oracle = ExpectedValueOracle(grid, {"cheap": 0.001, "strong": 0.01}, 1.0)
    assert all(a.action.n_samples == 1 for a in oracle.evaluate(_item("easy")).all_actions)


def test_correlated_samples_stop_best_of_n_dominating():
    grid = _two_model_grid()
    costs = {"cheap": 0.0001, "strong": 0.01}
    independent = ExpectedValueOracle(
        grid, costs, 1.0, max_samples=5, sample_correlation=0.0
    )
    correlated = ExpectedValueOracle(
        grid, costs, 1.0, max_samples=5, sample_correlation=1.0
    )
    assert independent.evaluate(_item("hard")).best.action.n_samples > 1
    assert correlated.evaluate(_item("hard")).best.action.n_samples == 1


def test_correlation_estimate_is_high_for_all_or_nothing_cells():
    """Perfectly correlated samples make every cell 0/k or k/k."""
    grid = OutcomeGrid()
    for i in range(20):
        outcome = i % 2 == 0
        for _ in range(5):
            grid.record("m", f"item{i}", outcome)
    assert estimate_sample_correlation(grid, "m") > 0.8


def test_correlation_estimate_is_low_for_binomial_scatter():
    import random as _random

    grid = OutcomeGrid()
    rng = _random.Random(3)
    for i in range(120):
        for _ in range(5):
            grid.record("m", f"item{i}", rng.random() < 0.5)
    assert estimate_sample_correlation(grid, "m") < 0.3


def test_correlation_estimate_returns_none_without_enough_data():
    grid = OutcomeGrid()
    grid.record("m", "a", True)
    assert estimate_sample_correlation(grid, "m") is None


# ── Routers ───────────────────────────────────────────────────────────────────

def test_fixed_router_always_plays_the_same_action():
    r = fixed_router("m")
    assert {r(_item(f"i{n}")).model for n in range(5)} == {"m"}


def test_theriad_router_selects_real_registry_models():
    r = theriad_router()
    models = routed_models(r, [_item("a"), _item("b")])
    assert models and all("/" in m for m in models)


def test_theriad_router_honours_the_policy():
    items = [_item("a")]
    cheap = routed_models(theriad_router("aggressive_savings"), items)
    quality = routed_models(theriad_router("quality_first"), items)
    assert cheap and quality


def test_unmeasured_models_flags_gaps_in_grid_coverage():
    """A model the grid never measured scores on the uniform prior, so any
    regret including it is partly fiction."""
    grid = OutcomeGrid()
    grid.record("known/model", "a", True)
    missing = unmeasured_models(fixed_router("unknown/model"), [_item("a")], grid)
    assert missing == {"unknown/model"}


def test_unmeasured_models_is_empty_when_coverage_is_complete():
    grid = OutcomeGrid()
    grid.record("known/model", "a", True)
    assert unmeasured_models(fixed_router("known/model"), [_item("a")], grid) == set()


# ── Provider pinning ──────────────────────────────────────────────────────────

from theriad.eval.executors import (  # noqa: E402
    AWS_HIGHEST_THROUGHPUT,
    DEFAULT_PROVIDER_PREFERENCES,
    META_ONLY,
)


def test_opus5_is_pinned_to_bedrock():
    assert DEFAULT_PROVIDER_PREFERENCES["anthropic/claude-opus-5"] is AWS_HIGHEST_THROUGHPUT
    assert AWS_HIGHEST_THROUGHPUT["only"] == ["amazon-bedrock"]
    assert AWS_HIGHEST_THROUGHPUT["sort"] == "throughput"


def test_muse_spark_12_is_pinned_to_meta():
    assert DEFAULT_PROVIDER_PREFERENCES["meta/muse-spark-1.2"] is META_ONLY
    assert META_ONLY["only"] == ["meta"]
    assert META_ONLY["allow_fallbacks"] is False


def test_pinned_provider_disables_fallbacks():
    """Without this, a capacity blip silently serves from another upstream and
    the grid records it as if it came from the pinned one."""
    assert AWS_HIGHEST_THROUGHPUT["allow_fallbacks"] is False


def test_new_premium_candidates_are_priceable():
    """Unpriced models resolve to $0.00, which would make them look free to the
    oracle and to the budget guard."""
    from theriad.accounting.pricing import PricingTable

    pt = PricingTable(use_live=False)
    for slug in (
        "openai/gpt-5.6-luna",
        "x-ai/grok-4.5",
        "z-ai/glm-5.3-flash",
        "meta/muse-spark-1.2",
        "anthropic/claude-opus-5",
    ):
        result = pt.lookup(slug)
        assert result is not None and result.source == "exact", slug
        assert result.price.input_per_1m > 0 and result.price.output_per_1m > 0


def test_luna_is_seeded_at_list_not_the_promo_rate():
    """A seed that understates cost is the one failure the budget guard cannot
    catch, so the promotional rate stays documented rather than baked in."""
    from theriad.accounting.pricing import PricingTable

    price = PricingTable(use_live=False).get("openai/gpt-5.6-luna")
    assert (price.input_per_1m, price.output_per_1m) == (0.20, 1.20)


def test_luna_is_cheaper_than_the_model_it_replaces():
    from theriad.eval.run import call_costs

    costs = call_costs(["openai/gpt-5.6-luna", "z-ai/glm-5.2"], 1600, 900)
    assert costs["openai/gpt-5.6-luna"] < costs["z-ai/glm-5.2"]


def test_replacement_study_model_set_contains_current_and_us_candidates():
    from theriad.eval.run import REPLACEMENT_STUDY_MODELS

    models = {model for tier in REPLACEMENT_STUDY_MODELS.values() for model in tier}
    assert {
        "qwen/qwen3.6-plus",
        "deepseek/deepseek-v4-flash-0731",
        "z-ai/glm-5.2",
        "openai/gpt-oss-120b",
        "google/gemini-2.5-flash",
        "gpt-5.4-mini",
        "openai/gpt-5.6-luna",
        "x-ai/grok-4.5",
        "z-ai/glm-5.3-flash",
        "meta/muse-spark-1.2",
        "anthropic/claude-opus-5",
    } <= models
    assert "moonshotai/kimi-k2.7-code" not in models


def test_replacement_study_model_set_rejects_manual_overrides():
    import pytest

    from theriad.eval.run import build_parser, model_lists

    args = build_parser().parse_args(
        ["--model-set", "replacement-study", "--cheap", "custom/model", "plan"]
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        model_lists(args)


def test_replacement_study_model_set_is_available_from_cli():
    from theriad.eval.run import build_parser, model_lists

    args = build_parser().parse_args(["--model-set", "replacement-study", "plan"])
    cheap, premium, frontier = model_lists(args)
    assert "openai/gpt-oss-120b" in cheap
    assert "openai/gpt-5.6-luna" in premium
    assert "z-ai/glm-5.3-flash" in frontier
    assert "meta/muse-spark-1.2" in frontier
    assert "anthropic/claude-opus-5" in frontier
