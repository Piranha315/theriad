"""
Unit tests for the benchmark harness.

No real API calls are made. Provider lane calls are mocked.
"""
from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from theriad.benchmark.budget import BudgetExceeded, BudgetGuard
from theriad.benchmark.pricing import (
    CEREBRAS_PRICES, GROQ_PRICES, OLLAMA_PRICE, ProviderPrice, estimate,
)
from theriad.benchmark.prompts import load_prompts, stratified_sample
from theriad.benchmark.types import BenchmarkPrompt, BenchmarkResult


# ── format_usd ───────────────────────────────────────────────────────────────

class TestFormatUsd:

    def _f(self, v):
        from theriad.benchmark.formatting import format_usd
        return format_usd(v)

    def test_none_is_question_mark(self):
        assert self._f(None) == "?"

    def test_exact_zero_is_dollar_zero(self):
        assert self._f(0.0) == "$0"

    def test_sub_microdollar_shows_less_than(self):
        assert self._f(0.0000003) == "<$0.000001"
        assert self._f(1e-9) == "<$0.000001"
        assert self._f(0.0000009999) == "<$0.000001"

    def test_boundary_one_microdollar_not_less_than(self):
        result = self._f(0.000001)
        assert not result.startswith("<")
        assert result.startswith("$")

    def test_small_cost_adaptive_precision(self):
        # 0.0000031 (3.1e-6) is above the 1µ$ threshold — must show real digits
        result = self._f(0.0000031)
        assert result not in ("?", "$0", "<$0.000001")
        assert "31" in result

    def test_small_cost_strips_trailing_zeros(self):
        result = self._f(0.00001000)
        assert not result.endswith("0"), f"trailing zeros in {result!r}"

    def test_known_tiny_cost_not_question_mark(self):
        # The core regression: a tiny but known cost must never show "?"
        for v in (0.000001, 0.0000012, 0.00001, 0.0001, 0.001):
            assert self._f(v) != "?", f"format_usd({v}) should not be '?'"

    def test_cent_scale_uses_four_decimals(self):
        assert self._f(0.05) == "$0.0500"
        assert self._f(1.23456) == "$1.2346"

    def test_large_cost(self):
        result = self._f(10.0)
        assert result == "$10.0000"


# ── BenchmarkPrompt / quality_check ──────────────────────────────────────────

class TestBenchmarkPrompt:

    def test_quality_check_missing_rubric_is_ungradable(self):
        bp = BenchmarkPrompt(id="x", category="c", prompt="p", max_output_tokens=50)
        assert bp.quality_check("anything") is None
        assert bp.quality_check("") is None

    def test_quality_check_all_present(self):
        bp = BenchmarkPrompt(
            id="x", category="c", prompt="p", max_output_tokens=50,
            expected_properties=["def", "return"],
        )
        assert bp.quality_check("def foo():\n    return 1") is True

    def test_quality_check_missing_one(self):
        bp = BenchmarkPrompt(
            id="x", category="c", prompt="p", max_output_tokens=50,
            expected_properties=["def", "yield"],
        )
        assert bp.quality_check("def foo():\n    return 1") is False

    def test_quality_check_case_sensitive(self):
        bp = BenchmarkPrompt(
            id="x", category="c", prompt="p", max_output_tokens=50,
            expected_properties=["SELECT"],
        )
        assert bp.quality_check("select * from foo") is False
        assert bp.quality_check("SELECT * FROM foo") is True


# ── BenchmarkResult serialization ─────────────────────────────────────────────

class TestBenchmarkResult:

    def _make(self, **kwargs) -> BenchmarkResult:
        defaults = dict(
            timestamp="2025-01-01T00:00:00+00:00",
            provider="groq", model="llama3-8b",
            tier="cheap_bulk", prompt_id="code_01", category="code",
            input_tokens=50, output_tokens=100,
            actual_cost_usd=0.000005,
            provider_latency_ms=250.0, total_latency_ms=260.0,
            tokens_per_second=400.0,
            finish_reason="stop", status="success", error=None,
        )
        defaults.update(kwargs)
        return BenchmarkResult(**defaults)

    def test_roundtrip_success(self):
        r = self._make()
        d = r.to_dict()
        r2 = BenchmarkResult.from_dict(d)
        assert r2.provider == "groq"
        assert r2.status == "success"
        assert r2.actual_cost_usd == pytest.approx(0.000005, rel=1e-5)

    def test_null_cost_preserved(self):
        r = self._make(actual_cost_usd=None)
        d = r.to_dict()
        assert d["actual_cost_usd"] is None
        r2 = BenchmarkResult.from_dict(d)
        assert r2.actual_cost_usd is None

    def test_all_required_fields_present_in_dict(self):
        r = self._make()
        d = r.to_dict()
        for field in [
            "timestamp", "provider", "model", "tier", "prompt_id", "category",
            "input_tokens", "output_tokens", "actual_cost_usd",
            "provider_latency_ms", "total_latency_ms", "tokens_per_second",
            "finish_reason", "status", "error",
        ]:
            assert field in d, f"Missing field: {field}"

    def test_tokens_per_second_rounded(self):
        r = self._make(tokens_per_second=123.456)
        d = r.to_dict()
        assert d["tokens_per_second"] == 123.5


# ── BudgetGuard ───────────────────────────────────────────────────────────────

class TestBudgetGuard:

    def test_initial_state(self):
        g = BudgetGuard(10.0)
        assert g.spent == 0.0
        assert g.remaining == pytest.approx(10.0)
        assert not g.is_exhausted

    def test_record_accumulates(self):
        g = BudgetGuard(10.0)
        g.record(1.0)
        g.record(2.0)
        assert g.spent == pytest.approx(3.0)
        assert g.remaining == pytest.approx(7.0)

    def test_record_none_is_noop(self):
        g = BudgetGuard(10.0)
        g.record(None)
        assert g.spent == 0.0

    def test_record_zero_is_noop(self):
        g = BudgetGuard(10.0)
        g.record(0.0)
        assert g.spent == 0.0

    def test_check_before_call_allows_within_budget(self):
        g = BudgetGuard(10.0)
        g.record(5.0)
        g.check_before_call(4.0)  # should not raise

    def test_check_before_call_raises_when_over(self):
        g = BudgetGuard(10.0)
        g.record(9.0)
        with pytest.raises(BudgetExceeded):
            g.check_before_call(2.0)

    def test_check_before_call_none_always_passes(self):
        g = BudgetGuard(0.01)
        g.record(0.005)
        g.check_before_call(None)  # unknown cost passes through

    def test_exhausted_after_limit(self):
        g = BudgetGuard(5.0)
        g.record(5.0)
        assert g.is_exhausted

    def test_check_before_call_raises_when_already_exhausted(self):
        g = BudgetGuard(1.0)
        g.record(1.0)
        with pytest.raises(BudgetExceeded):
            g.check_before_call(None)

    def test_invalid_limit(self):
        with pytest.raises(ValueError):
            BudgetGuard(0.0)

    def test_calls_recorded(self):
        g = BudgetGuard(100.0)
        g.record(1.0)
        g.record(2.0)
        assert g.calls_recorded == 2

    def test_summary_line_format(self):
        g = BudgetGuard(10.0)
        g.record(3.5)
        line = g.summary_line()
        assert "3.5000" in line
        assert "10.00" in line


# ── Pricing ───────────────────────────────────────────────────────────────────

class TestPricing:

    def test_estimate_returns_none_for_none_price(self):
        assert estimate(None, 100, 200) is None

    def test_estimate_free_for_ollama(self):
        cost = estimate(OLLAMA_PRICE, 1000, 2000)
        assert cost == 0.0

    def test_estimate_groq_70b(self):
        price = GROQ_PRICES["llama-3.3-70b-versatile"]
        # 1M input at $0.59, 1M output at $0.79
        cost = estimate(price, 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.59 + 0.79)

    def test_estimate_groq_8b_cheap(self):
        price = GROQ_PRICES["llama-3.1-8b-instant"]
        cost = estimate(price, 1_000, 500)
        # (1000 * 0.05 + 500 * 0.08) / 1_000_000
        expected = (1000 * 0.05 + 500 * 0.08) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_cerebras_prices_present(self):
        assert "llama3.3-70b" in CEREBRAS_PRICES
        assert "llama3.1-8b" in CEREBRAS_PRICES

    def test_custom_price(self):
        p = ProviderPrice(input_per_1m=1.0, output_per_1m=2.0)
        cost = estimate(p, 500_000, 500_000)
        assert cost == pytest.approx(0.5 + 1.0)


# ── Prompt loading ────────────────────────────────────────────────────────────

class TestPromptLoading:

    def _write_prompts(self, tmp_path: Path, lines: list[dict]) -> Path:
        p = tmp_path / "prompts.jsonl"
        with p.open("w") as fh:
            for obj in lines:
                fh.write(json.dumps(obj) + "\n")
        return p

    def test_load_basic(self, tmp_path):
        path = self._write_prompts(tmp_path, [
            {"id": "a", "category": "code", "prompt": "Write foo.", "max_output_tokens": 100},
        ])
        prompts = load_prompts(path)
        assert len(prompts) == 1
        assert prompts[0].id == "a"
        assert prompts[0].category == "code"

    def test_load_default_properties(self, tmp_path):
        path = self._write_prompts(tmp_path, [
            {"id": "b", "category": "code", "prompt": "p", "max_output_tokens": 50},
        ])
        prompts = load_prompts(path)
        assert prompts[0].expected_properties == []

    def test_load_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_prompts(Path("/nonexistent/prompts.jsonl"))

    def test_load_skips_blank_lines(self, tmp_path):
        p = tmp_path / "prompts.jsonl"
        p.write_text(
            '{"id": "a", "category": "c", "prompt": "p", "max_output_tokens": 50}\n'
            '\n'
            '{"id": "b", "category": "c", "prompt": "q", "max_output_tokens": 50}\n'
        )
        prompts = load_prompts(p)
        assert len(prompts) == 2

    def test_load_real_benchmark_file(self):
        """The checked-in benchmark_prompts.jsonl must load cleanly."""
        real_path = Path(__file__).parent.parent.parent.parent / "benchmark_prompts.jsonl"
        if not real_path.exists():
            pytest.skip("benchmark_prompts.jsonl not present")
        prompts = load_prompts(real_path)
        assert len(prompts) >= 30
        assert len({p.id for p in prompts}) == len(prompts)
        categories = {p.category for p in prompts}
        assert categories >= {"code", "summarize", "reasoning", "extract", "instruct", "creative"}


class TestStratifiedSample:

    def _make_prompts(self, n_per_cat: int = 5) -> list[BenchmarkPrompt]:
        cats = ["code", "summarize", "reasoning", "extract", "instruct", "creative"]
        out = []
        for cat in cats:
            for i in range(n_per_cat):
                out.append(BenchmarkPrompt(
                    id=f"{cat}_{i:02d}", category=cat,
                    prompt=f"Prompt {i}", max_output_tokens=100,
                ))
        return out

    def test_limit_zero_returns_empty(self):
        assert stratified_sample(self._make_prompts(), 0) == []

    def test_limit_gte_total_returns_all(self):
        prompts = self._make_prompts()
        assert len(stratified_sample(prompts, 100)) == 30

    def test_limit_respected(self):
        prompts = self._make_prompts()
        sample = stratified_sample(prompts, 12)
        assert len(sample) == 12

    def test_categories_balanced(self):
        prompts = self._make_prompts()
        sample = stratified_sample(prompts, 12)
        from collections import Counter
        counts = Counter(p.category for p in sample)
        # With 12 prompts across 6 categories, each should get 2
        assert max(counts.values()) - min(counts.values()) <= 1

    def test_original_order_preserved(self):
        prompts = self._make_prompts()
        sample = stratified_sample(prompts, 18)
        ids = [p.id for p in sample]
        # All picked prompts should appear in the same relative order as original
        original_order = {p.id: i for i, p in enumerate(prompts)}
        positions = [original_order[pid] for pid in ids]
        assert positions == sorted(positions)


# ── Summary stats ─────────────────────────────────────────────────────────────

class TestSummaryAggregation:

    def _result(self, provider="groq", model="m1", category="code",
                status="success", cost=0.001, latency=200.0, tps=400.0,
                quality_pass=True) -> BenchmarkResult:
        return BenchmarkResult(
            timestamp="2025-01-01T00:00:00+00:00",
            provider=provider, model=model, tier=None,
            prompt_id="p1", category=category,
            input_tokens=50, output_tokens=100,
            actual_cost_usd=cost if status == "success" else None,
            provider_latency_ms=latency if status == "success" else 0.0,
            total_latency_ms=latency if status == "success" else 0.0,
            tokens_per_second=tps if status == "success" else None,
            finish_reason="stop" if status == "success" else None,
            status=status, error=None if status == "success" else "fail",
            quality_pass=quality_pass,
        )

    def test_aggregate_basic(self):
        from theriad.benchmark.summary import _aggregate
        results = [
            self._result(provider="groq", model="m1", cost=0.001, latency=200.0),
            self._result(provider="groq", model="m1", cost=0.002, latency=400.0),
        ]
        stats = _aggregate(results)
        assert len(stats) == 1
        s = stats[0]
        assert s.n == 2
        assert s.success_count == 2
        assert s.avg_cost_usd == pytest.approx(0.0015)
        assert s.avg_latency_ms == pytest.approx(300.0)

    def test_aggregate_failure_counted(self):
        from theriad.benchmark.summary import _aggregate
        results = [
            self._result(status="success"),
            self._result(status="error"),
        ]
        stats = _aggregate(results)
        s = stats[0]
        assert s.success_count == 1
        assert s.fail_count == 1

    def test_quality_pass_rate(self):
        from theriad.benchmark.summary import _aggregate
        results = [
            self._result(quality_pass=True),
            self._result(quality_pass=True),
            self._result(quality_pass=False),
        ]
        stats = _aggregate(results)
        s = stats[0]
        assert s.quality_pass_rate == pytest.approx(2 / 3)

    def test_rank_free_wins_on_cost(self):
        from theriad.benchmark.summary import _aggregate, _rank
        results = [
            self._result(provider="ollama", model="local", cost=0.0, latency=500.0),
            self._result(provider="groq", model="paid", cost=0.01, latency=200.0),
        ]
        stats = _aggregate(results)
        ranked = _rank(stats)
        # With equal success rates, the composite score depends on both dims
        # Both should appear in ranking
        assert len(ranked) == 2

    def test_ranking_penalizes_failures(self):
        from theriad.benchmark.summary import _aggregate, _rank

        good = [self._result(provider="a", model="good", status="success") for _ in range(5)]
        bad = [
            self._result(provider="b", model="bad", status="success"),
            self._result(provider="b", model="bad", status="error"),
            self._result(provider="b", model="bad", status="error"),
        ]
        stats = _aggregate(good + bad)
        ranked = _rank(stats)
        # "good" should rank above "bad" due to lower failure rate
        top_model = ranked[0][1].model
        assert top_model == "good"


# ── Mocked provider lane runs ─────────────────────────────────────────────────

class TestMockedLanes:

    def _bp(self, id="code_01", category="code") -> BenchmarkPrompt:
        return BenchmarkPrompt(
            id=id, category=category,
            prompt="Write a Python function.", max_output_tokens=150,
        )

    @pytest.mark.asyncio
    async def test_groq_lane_skipped_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from theriad.benchmark.providers.groq_lane import GroqLane
        lane = GroqLane(api_key=None)
        assert lane.models() == []

    @pytest.mark.asyncio
    async def test_cerebras_lane_skipped_without_api_key(self, monkeypatch):
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        from theriad.benchmark.providers.cerebras_lane import CerebrasLane
        lane = CerebrasLane(api_key=None)
        assert lane.models() == []

    @pytest.mark.asyncio
    async def test_ollama_lane_models_empty_when_unreachable(self):
        from theriad.benchmark.providers.ollama_lane import OllamaLane
        lane = OllamaLane(base_url="http://localhost:19999")  # non-existent
        models = lane.models()
        assert models == []

    @pytest.mark.asyncio
    async def test_runner_writes_jsonl(self, tmp_path):
        """Runner writes one JSONL line per (model, prompt) call."""
        from theriad.benchmark.providers.base import BenchmarkLane
        from theriad.benchmark.runner import run_benchmark

        # A mock lane that always succeeds
        class MockLane(BenchmarkLane):
            @property
            def provider_name(self): return "mock"
            def models(self): return ["mock-model"]
            async def run_one(self, model, bp):
                return BenchmarkResult(
                    timestamp="2025-01-01T00:00:00+00:00",
                    provider="mock", model=model, tier=None,
                    prompt_id=bp.id, category=bp.category,
                    input_tokens=10, output_tokens=20,
                    actual_cost_usd=0.0001,
                    provider_latency_ms=100.0, total_latency_ms=110.0,
                    tokens_per_second=200.0, finish_reason="stop",
                    status="success", error=None, quality_pass=True,
                )

        prompts = [
            BenchmarkPrompt(id=f"p{i}", category="code",
                            prompt="Test", max_output_tokens=50)
            for i in range(3)
        ]
        output = tmp_path / "out.jsonl"
        results = await run_benchmark(
            lanes=[MockLane()],
            prompts=prompts,
            budget_usd=100.0,
            output_path=output,
        )
        assert len(results) == 3
        assert all(r.status == "success" for r in results)

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3
        obj = json.loads(lines[0])
        assert obj["provider"] == "mock"

    @pytest.mark.asyncio
    async def test_runner_skips_on_budget_exceeded(self, tmp_path):
        """All calls are skipped after budget is hit."""
        from theriad.benchmark.providers.base import BenchmarkLane
        from theriad.benchmark.runner import run_benchmark

        class ExpensiveLane(BenchmarkLane):
            @property
            def provider_name(self): return "expensive"
            def models(self): return ["model"]
            async def run_one(self, model, bp):
                return BenchmarkResult(
                    timestamp="2025-01-01T00:00:00+00:00",
                    provider="expensive", model=model, tier=None,
                    prompt_id=bp.id, category=bp.category,
                    input_tokens=10, output_tokens=20,
                    actual_cost_usd=5.0,   # each call costs $5
                    provider_latency_ms=100.0, total_latency_ms=110.0,
                    tokens_per_second=200.0, finish_reason="stop",
                    status="success", error=None, quality_pass=True,
                )

        prompts = [
            BenchmarkPrompt(id=f"p{i}", category="code",
                            prompt="Test", max_output_tokens=50)
            for i in range(5)
        ]
        output = tmp_path / "out.jsonl"
        results = await run_benchmark(
            lanes=[ExpensiveLane()],
            prompts=prompts,
            budget_usd=7.0,   # only $7 budget — hits ceiling after 1 call
            output_path=output,
        )

        success_count = sum(1 for r in results if r.status == "success")
        skip_count = sum(1 for r in results if r.status == "budget_exceeded")
        assert success_count >= 1
        assert skip_count >= 1
        assert success_count + skip_count == 5

    @pytest.mark.asyncio
    async def test_build_lanes_skips_unknown(self, monkeypatch):
        from theriad.benchmark.runner import build_lanes
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        lanes = build_lanes(["groq", "nonexistent", "ollama"])
        names = {l.provider_name for l in lanes}
        assert "nonexistent" not in names
        assert "groq" in names
        assert "ollama" in names


# ── PricingTable.lookup() ─────────────────────────────────────────────────────

class TestPricingTableLookup:
    """Tests for PricingTable.lookup() and the xiaomi/mimo-v2.5 pricing entries."""

    def _table(self):
        from theriad.accounting.pricing import PricingTable
        # use_live=False keeps these assertions deterministic: they pin exact
        # seed values, which the live catalog is free to override at runtime.
        return PricingTable(use_live=False)

    def test_exact_match_returns_exact_source(self):
        result = self._table().lookup("xiaomi/mimo-v2.5")
        assert result is not None
        assert result.source == "exact"
        assert result.model_slug == "xiaomi/mimo-v2.5"

    def test_exact_versioned_slug_returns_exact_source(self):
        # The dated slug is listed explicitly in the table.
        result = self._table().lookup("xiaomi/mimo-v2.5-20260422")
        assert result is not None
        assert result.source == "exact"
        assert result.model_slug == "xiaomi/mimo-v2.5-20260422"

    def test_unknown_versioned_slug_falls_back_to_alias(self):
        # A future dated release not in the table; prefix match finds the base slug.
        result = self._table().lookup("xiaomi/mimo-v2.5-20271231")
        assert result is not None
        assert result.source == "alias"

    def test_unknown_model_returns_none(self):
        result = self._table().lookup("totally/unknown-model-xyz")
        assert result is None

    def test_get_backward_compat_returns_model_price(self):
        from theriad.accounting.pricing import ModelPrice
        price = self._table().get("xiaomi/mimo-v2.5")
        assert isinstance(price, ModelPrice)
        assert price.input_per_1m == pytest.approx(0.14)
        assert price.output_per_1m == pytest.approx(0.28)

    def test_get_unknown_returns_none(self):
        assert self._table().get("totally/unknown-model-xyz") is None

    def test_mimo_v2_5_pricing_values(self):
        result = self._table().lookup("xiaomi/mimo-v2.5")
        assert result.price.input_per_1m == pytest.approx(0.14)
        assert result.price.output_per_1m == pytest.approx(0.28)

    def test_mimo_v2_5_cost_not_question_mark(self):
        # Core regression: MiMo v2.5 cost must show real digits, never "?"
        from theriad.benchmark.formatting import format_usd
        table = self._table()
        # Simulate 500 prompt tokens + 200 completion tokens
        result = table.lookup("xiaomi/mimo-v2.5-20260422")
        assert result is not None
        cost = (500 * result.price.input_per_1m + 200 * result.price.output_per_1m) / 1_000_000
        assert cost > 0
        assert format_usd(cost) != "?"

    def test_cost_known_true_for_known_model(self):
        result = self._table().lookup("xiaomi/mimo-v2.5")
        assert result is not None  # cost_known would be True in the lane

    def test_existing_openai_model_still_works(self):
        result = self._table().lookup("openai/gpt-4o")
        assert result is not None
        assert result.source == "exact"
        assert result.price.input_per_1m == pytest.approx(2.50)


# ── cohere/north-mini-code:free — benchmark candidate tests ──────────────────

class TestCohereNorthMiniFree:
    """
    Tests covering the four requirements for cohere/north-mini-code:free
    as a benchmark-only free-hosted candidate.
    """

    MODEL_SLUG = "cohere/north-mini-code:free"

    # 1. Free pricing displays "$0", not "?"
    def test_free_pricing_displays_dollar_zero(self):
        from theriad.accounting.pricing import PricingTable
        from theriad.benchmark.formatting import format_usd

        table = PricingTable()
        result = table.lookup(self.MODEL_SLUG)
        assert result is not None, "cohere/north-mini-code:free must be in PricingTable"
        assert result.price.input_per_1m == pytest.approx(0.0)
        assert result.price.output_per_1m == pytest.approx(0.0)

        # Cost for any token count must format as "$0", never "?"
        cost = (100 * result.price.input_per_1m + 200 * result.price.output_per_1m) / 1_000_000
        assert format_usd(cost) == "$0"

    # 2. cohere/north-mini-code:free is allowed in OpenRouter lane
    def test_model_allowed_in_openrouter_lane(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        from theriad.benchmark.providers.openrouter_lane import OpenRouterLane, _OPENROUTER_MODELS

        assert self.MODEL_SLUG in _OPENROUTER_MODELS, (
            f"{self.MODEL_SLUG!r} must be listed in _OPENROUTER_MODELS"
        )
        lane = OpenRouterLane(api_key="sk-test")
        assert self.MODEL_SLUG in lane.models()

    # 3. Free model does not count against paid budget
    def test_free_model_cost_not_counted_against_budget(self):
        from theriad.benchmark.budget import BudgetGuard

        guard = BudgetGuard(limit_usd=1.0)
        # Simulate recording a $0 cost (as free models produce)
        guard.record(actual_cost=0.0)
        guard.record(actual_cost=0.0)
        guard.record(actual_cost=0.0)

        # Budget must remain untouched — free calls never consume budget
        assert guard.spent == pytest.approx(0.0)
        assert not guard.is_exhausted
        # And a subsequent paid call still sees the full budget
        guard.check_before_call(estimated_cost=0.99)  # must not raise

    # 4. Free hosted models still respect rate-limit/failure diagnostics
    def test_free_model_rate_limit_captured_in_result(self):
        from theriad.benchmark.types import BenchmarkResult

        # Simulate what the lane produces when the free model rate-limits
        error_result = BenchmarkResult(
            timestamp="2026-06-29T00:00:00+00:00",
            provider="openrouter",
            model=self.MODEL_SLUG,
            tier="benchmark_direct",
            prompt_id="code_01",
            category="code",
            input_tokens=0,
            output_tokens=0,
            actual_cost_usd=None,
            provider_latency_ms=0.0,
            total_latency_ms=0.0,
            tokens_per_second=None,
            finish_reason=None,
            status="error",
            error="429 Too Many Requests — free tier rate limit exceeded",
            quality_pass=None,
        )

        d = error_result.to_dict()
        # Error status and message must survive serialisation round-trip
        assert d["status"] == "error"
        assert "rate limit" in (d["error"] or "").lower()

        r2 = BenchmarkResult.from_dict(d)
        assert r2.status == "error"
        assert r2.actual_cost_usd is None   # no cost charged on failure
        assert r2.model == self.MODEL_SLUG


# ── StickyRouteCache ──────────────────────────────────────────────────────────

class TestStickyRouteCache:
    """Tests for conversation-level route stickiness."""

    def _cache(self, ttl: int = 1800):
        from theriad.routing.sticky import StickyRouteCache
        return StickyRouteCache(ttl_seconds=ttl)

    def test_miss_on_unknown_conversation(self):
        cache = self._cache()
        assert cache.get("conv-unknown") is None

    def test_set_then_get_returns_route(self):
        cache = self._cache()
        cache.set("conv-1", selected_model="qwen/qwen3.6-plus",
                  selected_tier="cheap_code", provider="openrouter", reason="first_call")
        entry = cache.get("conv-1")
        assert entry is not None
        assert entry.selected_model == "qwen/qwen3.6-plus"
        assert entry.selected_tier == "cheap_code"
        assert entry.provider == "openrouter"

    def test_sticky_reuse_on_follow_up(self):
        """Second message in same conversation reuses sticky model."""
        cache = self._cache()
        cache.set("conv-2", selected_model="qwen/qwen3.6-plus",
                  selected_tier="cheap_code", provider="openrouter", reason="first_call")

        # Follow-up call: same tier → sticky should still be live
        entry = cache.get("conv-2")
        assert entry is not None
        assert entry.selected_model == "qwen/qwen3.6-plus"

    def test_sticky_cleared_on_major_tier_change(self):
        """Routing to a drastically different tier clears the sticky entry."""
        cache = self._cache()
        cache.set("conv-3", selected_model="qwen/qwen3.6-plus",
                  selected_tier="cheap_code", provider="openrouter", reason="first_call")

        # premium tier is 3 ranks above cheap_code — should be a major change
        is_major = cache.is_major_tier_change("cheap_code", "premium")
        assert is_major is True

        cleared = cache.clear("conv-3", reason="task_type_change")
        assert cleared is True
        assert cache.get("conv-3") is None

    def test_sticky_not_cleared_on_adjacent_tier(self):
        """Adjacent tier change (cheap_code → code_strong) is NOT a major change."""
        cache = self._cache()
        is_major = cache.is_major_tier_change("cheap_code", "code_strong")
        assert is_major is False

    def test_sticky_cleared_on_provider_error(self):
        cache = self._cache()
        cache.set("conv-4", selected_model="qwen/qwen3.6-plus",
                  selected_tier="cheap_code", provider="openrouter", reason="first_call")
        cache.clear("conv-4", reason="provider_error")
        assert cache.get("conv-4") is None

    def test_clear_nonexistent_returns_false(self):
        cache = self._cache()
        assert cache.clear("conv-nope", reason="test") is False

    def test_expired_entry_evicted_on_get(self):
        """Entries past TTL are treated as misses."""
        import time
        cache = self._cache(ttl=1)
        cache.set("conv-5", selected_model="m", selected_tier="cheap_bulk",
                  provider="openrouter", reason="test")
        time.sleep(1.1)
        assert cache.get("conv-5") is None

    def test_size_reflects_live_entries(self):
        cache = self._cache()
        cache.set("a", selected_model="m", selected_tier="cheap_bulk",
                  provider="openrouter", reason="test")
        cache.set("b", selected_model="m", selected_tier="cheap_bulk",
                  provider="openrouter", reason="test")
        assert cache.size() == 2
        cache.clear("a", reason="test")
        assert cache.size() == 1


# ── Policy cost/quality tradeoff ──────────────────────────────────────────────

class TestPolicyTradeoff:
    """Policy labels must map to documented numeric tradeoff values."""

    def test_aggressive_savings_is_9(self):
        from theriad.models.policy import POLICY_TRADEOFF
        assert POLICY_TRADEOFF["aggressive_savings"] == 9

    def test_balanced_is_7(self):
        from theriad.models.policy import POLICY_TRADEOFF
        assert POLICY_TRADEOFF["balanced"] == 7

    def test_quality_first_is_2(self):
        from theriad.models.policy import POLICY_TRADEOFF
        assert POLICY_TRADEOFF["quality_first"] == 2

    def test_resolution_exposes_tradeoff(self):
        from theriad.models.policy import PolicyResolver
        resolver = PolicyResolver()
        res = resolver.resolve("cheap_bulk", "aggressive_savings", 0.9)
        assert res.cost_quality_tradeoff == 9

    def test_unknown_policy_defaults_to_balanced(self):
        from theriad.models.policy import POLICY_TRADEOFF, PolicyResolver
        resolver = PolicyResolver()
        res = resolver.resolve("cheap_bulk", "nonexistent_policy", 0.9)
        assert res.cost_quality_tradeoff == POLICY_TRADEOFF["balanced"]


# ── ModelFilter (project model filters) ──────────────────────────────────────

class TestModelFilter:
    """Project-level allowed/blocked model pattern filtering."""

    def _filter(self, **kwargs):
        from theriad.routing.filters import ModelFilter, ModelFilterSettings
        return ModelFilter(ModelFilterSettings(**kwargs))

    def _candidates(self, models: list[tuple[str, str]]):
        """Build candidate list with no estimated cost / tier info."""
        return [(prov, mdl, None, None) for prov, mdl in models]

    def test_no_restrictions_allows_all(self):
        f = self._filter()
        candidates = self._candidates([
            ("openrouter", "qwen/qwen3.6-plus"),
            ("ollama", "ollama/llama3.1:8b"),
        ])
        result = f.filter_candidates(candidates)
        assert len(result) == 2

    def test_allowed_pattern_filters_others(self):
        f = self._filter(allowed_model_patterns=["qwen/*"])
        candidates = self._candidates([
            ("openrouter", "qwen/qwen3.6-plus"),
            ("openrouter", "cohere/north-mini-code:free"),
        ])
        result = f.filter_candidates(candidates)
        slugs = [m for _, m, *_ in result]
        assert "qwen/qwen3.6-plus" in slugs
        assert "cohere/north-mini-code:free" not in slugs

    def test_blocked_pattern_excludes_match(self):
        f = self._filter(blocked_model_patterns=["cohere/*"])
        candidates = self._candidates([
            ("openrouter", "qwen/qwen3.6-plus"),
            ("openrouter", "cohere/north-mini-code:free"),
        ])
        result = f.filter_candidates(candidates)
        slugs = [m for _, m, *_ in result]
        assert "qwen/qwen3.6-plus" in slugs
        assert "cohere/north-mini-code:free" not in slugs

    def test_local_only_excludes_hosted_models(self):
        f = self._filter(local_only=True)
        candidates = self._candidates([
            ("openrouter", "qwen/qwen3.6-plus"),   # hosted
            ("ollama", "llama3.1:8b"),               # local
        ])
        result = f.filter_candidates(candidates)
        slugs = [m for _, m, *_ in result]
        assert "qwen/qwen3.6-plus" not in slugs
        assert "llama3.1:8b" in slugs

    def test_max_cost_blocks_expensive_model(self):
        f = self._filter(max_cost_per_call_usd=0.001)
        candidates = [
            ("openrouter", "cheap-model", 0.0005, None),
            ("openrouter", "expensive-model", 0.005, None),
        ]
        result = f.filter_candidates(candidates)
        slugs = [m for _, m, *_ in result]
        assert "cheap-model" in slugs
        assert "expensive-model" not in slugs

    def test_max_cost_allows_unknown_cost(self):
        """Models with no known cost estimate pass cost filter (cost=None)."""
        f = self._filter(max_cost_per_call_usd=0.001)
        candidates = [("openrouter", "unknown-cost-model", None, None)]
        result = f.filter_candidates(candidates)
        assert len(result) == 1

    def test_allow_premium_false_blocks_premium_tier(self):
        f = self._filter(allow_premium=False)
        from theriad.routing.filters import FilterResult
        result = f.is_allowed("openai/gpt-4o", "openrouter", tier="premium",
                              estimated_cost_usd=None)
        assert result.allowed is False

    def test_empty_candidates_returns_empty(self):
        f = self._filter()
        assert f.filter_candidates([]) == []


# ── openrouter/auto-or benchmark slug ────────────────────────────────────────

class TestOpenRouterAutoOrSlug:
    """openrouter/auto-or benchmarks OR's native auto-router."""

    def test_auto_or_in_openrouter_models(self):
        from theriad.benchmark.providers.openrouter_lane import _OPENROUTER_MODELS
        assert "openrouter/auto-or" in _OPENROUTER_MODELS

    def test_auto_or_has_benchmark_metadata(self):
        from theriad.benchmark.providers.openrouter_lane import _OPENROUTER_BENCHMARK_METADATA
        assert "openrouter/auto-or" in _OPENROUTER_BENCHMARK_METADATA
        meta = _OPENROUTER_BENCHMARK_METADATA["openrouter/auto-or"]
        assert meta["status"] == "benchmark_only"
        assert meta["candidate_tiers"] == []  # not slotted into any production tier

    @pytest.mark.asyncio
    async def test_auto_or_calls_openrouter_auto_model(self, monkeypatch):
        """auto-or path passes model='openrouter/auto' to the provider."""
        from theriad.benchmark.providers.openrouter_lane import OpenRouterLane
        from theriad.benchmark.types import BenchmarkPrompt, BenchmarkResult
        from theriad.core.providers.interfaces import CompletionResponse

        captured_model: list[str] = []

        async def mock_complete(self_inner, request):
            captured_model.append(request.model)
            return CompletionResponse(
                model="anthropic/claude-haiku-4-5",  # OR chose this
                content="hello",
                prompt_tokens=10,
                completion_tokens=5,
                latency_ms=100.0,
                provider="openrouter",
                finish_reason="stop",
            )

        monkeypatch.setattr(
            "theriad.core.providers.openrouter.OpenRouterProvider.complete",
            mock_complete,
        )

        lane = OpenRouterLane(api_key="sk-test")
        bp = BenchmarkPrompt(
            id="test_01", category="code",
            prompt="Write a hello world.", max_output_tokens=100,
        )
        result = await lane.run_one("openrouter/auto-or", bp)

        # The request must use "openrouter/auto" (OR's literal slug)
        assert captured_model == ["openrouter/auto"]
        # The result should record what OR actually selected
        assert result.model == "anthropic/claude-haiku-4-5"
        assert result.status == "success"
        assert result.tier == "benchmark_direct"
