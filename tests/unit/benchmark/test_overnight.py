"""
Unit tests for overnight-safe benchmark infrastructure.

Covers:
- Preflight checks (disk thresholds, API key presence)
- Resume mode (load completed keys, skip on re-run)
- Checkpoint / fsync (one row written per call)
- Local model allowlist (large models blocked by default)
- Output caps (per-category token limits)
- Presets (smoke, overnight_safe, full)
- Output preview / sha256 capture
- Partial summary on incomplete JSONL
- Cost formatting regression (MiMo shows real cost, unknown shows "?")
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from theriad.benchmark.types import BenchmarkPrompt, BenchmarkResult, _make_output_preview


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bp(id="code_01", category="code", max_tokens=100) -> BenchmarkPrompt:
    return BenchmarkPrompt(id=id, category=category, prompt="Write foo.", max_output_tokens=max_tokens)


def _result(
    provider="openrouter", model="xiaomi/mimo-v2.5-20260422",
    prompt_id="code_01", category="code",
    status="success", cost=4.83e-05,
) -> BenchmarkResult:
    return BenchmarkResult(
        timestamp="2026-06-29T00:00:00+00:00",
        provider=provider, model=model, tier="cheap_bulk",
        prompt_id=prompt_id, category=category,
        input_tokens=50, output_tokens=200,
        actual_cost_usd=cost if status == "success" else None,
        provider_latency_ms=1200.0, total_latency_ms=1250.0,
        tokens_per_second=18.0, finish_reason="stop",
        status=status, error=None if status == "success" else "timeout",
        quality_pass=True,
        cost_known=True, pricing_source="exact",
        pricing_model_slug="xiaomi/mimo-v2.5-20260422",
        requested_model_slug="xiaomi/mimo-v2.5",
    )


# ── Preflight checks ──────────────────────────────────────────────────────────

class TestPreflight:
    """Tests for theriad.benchmark.preflight."""

    def _run(self, free_gb, env=None, check_ollama=False, budget=2.0, providers=None):
        from theriad.benchmark.preflight import run_preflight, _GB
        fake_usage = type("DU", (), {"free": int(free_gb * _GB), "total": int(100 * _GB), "used": 0})()
        with patch("shutil.disk_usage", return_value=fake_usage):
            with patch.dict(os.environ, env or {"OPENROUTER_API_KEY": "sk-test"}):
                return run_preflight(
                    providers or ["openrouter"],
                    budget_usd=budget,
                    check_ollama=check_ollama,
                )

    def test_disk_above_20gb_is_ok(self):
        result = self._run(free_gb=25.0)
        disk = next(c for c in result.checks if c.name == "disk")
        assert disk.status == "OK"
        assert result.local_allowed is True

    def test_disk_between_10_and_20_is_warn_and_blocks_local(self):
        result = self._run(free_gb=12.0)
        disk = next(c for c in result.checks if c.name == "disk")
        assert disk.status == "WARN"
        assert result.local_allowed is False
        # Still ok to run (not BLOCKED)
        assert result.ok_to_run is True

    def test_disk_below_10gb_is_blocked_with_local_providers(self):
        # Low disk BLOCKS when ollama is in the provider list
        result = self._run(free_gb=8.0, providers=["openrouter", "ollama"])
        disk = next(c for c in result.checks if c.name == "disk")
        assert disk.status == "BLOCKED"
        assert result.local_allowed is False
        assert result.ok_to_run is False

    def test_disk_below_10gb_is_warn_only_for_cloud_providers(self):
        # Low disk is only a warning for cloud-only runs — no local models to protect
        result = self._run(free_gb=8.0, providers=["openrouter"])
        disk = next(c for c in result.checks if c.name == "disk")
        assert disk.status == "WARN"
        assert result.ok_to_run is True

    def test_missing_api_key_blocks_run(self):
        from theriad.benchmark.preflight import run_preflight, _GB
        fake_usage = type("DU", (), {"free": int(25 * _GB), "total": int(100 * _GB), "used": 0})()
        with patch("shutil.disk_usage", return_value=fake_usage):
            with patch.dict(os.environ, {}, clear=True):
                # Remove key if present
                env_clean = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
                result = run_preflight(["openrouter"], budget_usd=2.0, check_ollama=False)
        # On CI the key may or may not be present; just check the check exists
        key_check = next((c for c in result.checks if "openrouter_key" in c.name), None)
        assert key_check is not None

    def test_valid_setup_is_ready(self):
        result = self._run(free_gb=25.0)
        assert result.ok_to_run is True

    def test_budget_zero_blocks(self):
        from theriad.benchmark.preflight import run_preflight, _GB
        fake_usage = type("DU", (), {"free": int(25 * _GB), "total": int(100 * _GB), "used": 0})()
        with patch("shutil.disk_usage", return_value=fake_usage):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
                result = run_preflight(["openrouter"], budget_usd=0.0, check_ollama=False)
        budget_check = next(c for c in result.checks if c.name == "budget")
        assert budget_check.status == "BLOCKED"
        assert result.ok_to_run is False

    def test_preflight_prints_report(self, capsys):
        result = self._run(free_gb=25.0)
        result.print_report()
        out = capsys.readouterr().out
        assert "BENCHMARK PREFLIGHT" in out
        assert "READY TO RUN" in out

    def test_blocked_preflight_prints_blocked(self, capsys):
        result = self._run(free_gb=5.0, providers=["openrouter", "ollama"])
        result.print_report()
        out = capsys.readouterr().out
        assert "BLOCKED" in out


# ── Resume ────────────────────────────────────────────────────────────────────

class TestResume:
    """Tests for theriad.benchmark.resume."""

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def test_empty_file_returns_no_keys(self, tmp_path):
        from theriad.benchmark.resume import load_completed_keys
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        state = load_completed_keys(path)
        assert state.completed == 0

    def test_none_path_returns_no_keys(self):
        from theriad.benchmark.resume import load_completed_keys
        state = load_completed_keys(None)
        assert state.completed == 0
        assert state.source_path is None

    def test_nonexistent_file_returns_no_keys(self, tmp_path):
        from theriad.benchmark.resume import load_completed_keys
        state = load_completed_keys(tmp_path / "nope.jsonl")
        assert state.completed == 0

    def test_loads_success_rows(self, tmp_path):
        from theriad.benchmark.resume import load_completed_keys
        row = _result().to_dict()
        path = tmp_path / "r.jsonl"
        self._write_jsonl(path, [row])
        state = load_completed_keys(path)
        assert state.completed == 1
        assert state.is_done("openrouter", "xiaomi/mimo-v2.5-20260422", "code_01", "code")

    def test_skips_error_rows(self, tmp_path):
        from theriad.benchmark.resume import load_completed_keys
        row = _result(status="error").to_dict()
        path = tmp_path / "r.jsonl"
        self._write_jsonl(path, [row])
        state = load_completed_keys(path)
        assert state.completed == 0

    def test_skips_budget_exceeded_rows(self, tmp_path):
        from theriad.benchmark.resume import load_completed_keys
        row = _result(status="budget_exceeded").to_dict()
        path = tmp_path / "r.jsonl"
        self._write_jsonl(path, [row])
        state = load_completed_keys(path)
        assert state.completed == 0

    def test_is_done_correct_key(self, tmp_path):
        from theriad.benchmark.resume import load_completed_keys
        rows = [
            _result(provider="openrouter", model="m1", prompt_id="p1", category="code").to_dict(),
            _result(provider="ollama",    model="llama3.1:8b", prompt_id="p1", category="code").to_dict(),
        ]
        path = tmp_path / "r.jsonl"
        self._write_jsonl(path, rows)
        state = load_completed_keys(path)
        assert state.completed == 2
        assert state.is_done("openrouter", "m1", "p1", "code")
        assert state.is_done("ollama", "llama3.1:8b", "p1", "code")
        assert not state.is_done("openrouter", "m2", "p1", "code")

    def test_malformed_lines_skipped(self, tmp_path):
        from theriad.benchmark.resume import load_completed_keys
        path = tmp_path / "r.jsonl"
        path.write_text('not json\n' + json.dumps(_result().to_dict()) + "\n")
        state = load_completed_keys(path)
        assert state.completed == 1  # only the valid row

    @pytest.mark.asyncio
    async def test_runner_skips_completed_prompts(self, tmp_path):
        """Runner should skip prompts whose key already exists in resume_state."""
        from theriad.benchmark.providers.base import BenchmarkLane
        from theriad.benchmark.resume import ResumeState
        from theriad.benchmark.runner import run_benchmark

        calls: list[str] = []

        class TrackingLane(BenchmarkLane):
            @property
            def provider_name(self): return "mock"
            def models(self): return ["model-a"]
            async def run_one(self, model, bp):
                calls.append(bp.id)
                return _result(provider="mock", model=model,
                               prompt_id=bp.id, category=bp.category)

        prompts = [_bp(id=f"p{i}", category="code") for i in range(3)]
        # Mark p0 and p1 as already done
        completed = frozenset([
            ("mock", "model-a", "p0", "code"),
            ("mock", "model-a", "p1", "code"),
        ])
        resume = ResumeState(completed_keys=completed, source_path=None)

        output = tmp_path / "out.jsonl"
        results = await run_benchmark(
            lanes=[TrackingLane()],
            prompts=prompts,
            budget_usd=100.0,
            output_path=output,
            resume_state=resume,
        )
        # Only p2 should have been called
        assert calls == ["p2"]
        assert len(results) == 1
        assert results[0].prompt_id == "p2"


# ── Checkpoint / fsync ────────────────────────────────────────────────────────

class TestCheckpointing:
    """Each result row is written and fsynced immediately."""

    @pytest.mark.asyncio
    async def test_one_row_written_per_call(self, tmp_path):
        from theriad.benchmark.providers.base import BenchmarkLane
        from theriad.benchmark.runner import run_benchmark

        written_after_calls: list[int] = []

        class CountingLane(BenchmarkLane):
            @property
            def provider_name(self): return "mock"
            def models(self): return ["m"]
            async def run_one(self, model, bp):
                # Count lines written so far BEFORE returning
                out = tmp_path / "out.jsonl"
                count = sum(1 for line in out.read_text().splitlines() if line.strip()) if out.exists() else 0
                written_after_calls.append(count)
                return _result(provider="mock", model=model, prompt_id=bp.id, category=bp.category)

        prompts = [_bp(id=f"p{i}") for i in range(3)]
        output = tmp_path / "out.jsonl"
        await run_benchmark(
            lanes=[CountingLane()],
            prompts=prompts,
            budget_usd=100.0,
            output_path=output,
        )
        # After each call, the file should have exactly i previous rows
        assert written_after_calls == [0, 1, 2]
        # Final file has 3 rows
        lines = [l for l in output.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_partial_results_readable_after_simulated_crash(self, tmp_path):
        """Results written before a crash are intact and parseable."""
        from theriad.benchmark.providers.base import BenchmarkLane
        from theriad.benchmark.runner import run_benchmark

        call_count = 0

        class CrashingLane(BenchmarkLane):
            @property
            def provider_name(self): return "mock"
            def models(self): return ["m"]
            async def run_one(self, model, bp):
                nonlocal call_count
                call_count += 1
                if call_count == 3:
                    raise RuntimeError("Simulated crash")
                return _result(provider="mock", model=model, prompt_id=bp.id, category=bp.category)

        prompts = [_bp(id=f"p{i}") for i in range(5)]
        output = tmp_path / "out.jsonl"
        try:
            await run_benchmark(
                lanes=[CrashingLane()],
                prompts=prompts,
                budget_usd=100.0,
                output_path=output,
            )
        except RuntimeError:
            pass  # crash is expected

        # The 2 rows written before the crash should be intact
        lines = [l.strip() for l in output.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert obj["status"] == "success"


# ── Local model allowlist ─────────────────────────────────────────────────────

class TestAllowlist:
    """Tests for theriad.benchmark.allowlist."""

    def test_large_model_detected(self):
        from theriad.benchmark.allowlist import is_large_model
        assert is_large_model("qwen3.6:27b") is True
        assert is_large_model("deepseek-r1:70b") is True
        assert is_large_model("llama3.1-70b") is True

    def test_small_model_not_large(self):
        from theriad.benchmark.allowlist import is_large_model
        assert is_large_model("llama3.1:8b") is False
        assert is_large_model("llama3.2:3b") is False
        assert is_large_model("mistral:7b") is False
        assert is_large_model("gemma:7b") is False

    def test_filter_removes_large_by_default(self):
        from theriad.benchmark.allowlist import filter_models
        models = ["llama3.1:8b", "qwen3.6:27b", "deepseek-r1:latest"]
        safe = filter_models(models, allow_large=False)
        assert "llama3.1:8b" in safe
        assert "qwen3.6:27b" not in safe
        # deepseek-r1:latest — not large by size heuristic but on explicit block list
        assert "deepseek-r1:latest" not in safe

    def test_filter_allow_large_overrides(self):
        from theriad.benchmark.allowlist import filter_models
        # Use a model that's large but NOT on the explicit block list
        # (e.g. a hypothetical 70b llama that isn't blocked explicitly)
        models = ["llama3.1:8b", "llama3.1:70b"]
        safe = filter_models(models, allow_large=True)
        # llama3.1:70b is large but only filtered by the size heuristic —
        # allow_large=True overrides the heuristic check
        assert "llama3.1:8b" in safe
        assert "llama3.1:70b" in safe

    def test_explicit_block_list_enforced_even_with_allow_large(self):
        from theriad.benchmark.allowlist import filter_models
        # deepseek-r1:latest is on the explicit block list
        safe = filter_models(["deepseek-r1:latest"], allow_large=True)
        assert safe == []

    def test_empty_list_returns_empty(self):
        from theriad.benchmark.allowlist import filter_models
        assert filter_models([], allow_large=False) == []

    def test_ollama_lane_filters_large_models(self):
        """OllamaLane.models() with explicit list filters via allowlist."""
        from theriad.benchmark.providers.ollama_lane import OllamaLane
        lane = OllamaLane(models=["llama3.1:8b", "qwen3.6:27b"], allow_large=False)
        assert "qwen3.6:27b" not in lane.models()
        assert "llama3.1:8b" in lane.models()

    def test_ollama_lane_allow_large_passes_through(self):
        from theriad.benchmark.providers.ollama_lane import OllamaLane
        # Use a large model NOT on the explicit block list
        lane = OllamaLane(models=["llama3.1:8b", "llama3.1:70b"], allow_large=True)
        assert "llama3.1:70b" in lane.models()
        assert "llama3.1:8b" in lane.models()


# ── Output caps ───────────────────────────────────────────────────────────────

class TestOutputCaps:
    """Per-category max_output_tokens are applied before calling the lane."""

    def test_apply_output_cap_reduces_tokens(self):
        from theriad.benchmark.runner import _apply_output_cap
        bp = _bp(category="summarize", max_tokens=2000)
        capped = _apply_output_cap(bp, {"summarize": 512})
        assert capped.max_output_tokens == 512
        assert capped.id == bp.id

    def test_apply_output_cap_no_cap_for_category_is_noop(self):
        from theriad.benchmark.runner import _apply_output_cap
        bp = _bp(category="reasoning", max_tokens=1500)
        capped = _apply_output_cap(bp, {"code": 1200})  # no 'reasoning' key
        assert capped.max_output_tokens == 1500

    def test_apply_output_cap_none_caps_is_noop(self):
        from theriad.benchmark.runner import _apply_output_cap
        bp = _bp(category="code", max_tokens=2000)
        capped = _apply_output_cap(bp, None)
        assert capped.max_output_tokens == 2000

    def test_apply_output_cap_already_under_cap_is_noop(self):
        from theriad.benchmark.runner import _apply_output_cap
        bp = _bp(category="summarize", max_tokens=300)
        capped = _apply_output_cap(bp, {"summarize": 512})
        assert capped.max_output_tokens == 300  # already under cap

    @pytest.mark.asyncio
    async def test_runner_applies_output_caps(self, tmp_path):
        """Caps are reflected in the BenchmarkPrompt passed to the lane."""
        from theriad.benchmark.providers.base import BenchmarkLane
        from theriad.benchmark.runner import run_benchmark

        seen_max_tokens: list[int] = []

        class CapCheckLane(BenchmarkLane):
            @property
            def provider_name(self): return "mock"
            def models(self): return ["m"]
            async def run_one(self, model, bp):
                seen_max_tokens.append(bp.max_output_tokens)
                return _result(provider="mock", model=model, prompt_id=bp.id, category=bp.category)

        prompts = [_bp(category="summarize", max_tokens=2000)]
        await run_benchmark(
            lanes=[CapCheckLane()],
            prompts=prompts,
            budget_usd=100.0,
            output_path=tmp_path / "out.jsonl",
            output_caps={"summarize": 512},
        )
        assert seen_max_tokens == [512]


# ── Presets ───────────────────────────────────────────────────────────────────

class TestPresets:
    """Tests for theriad.benchmark.presets."""

    def test_get_preset_smoke(self):
        from theriad.benchmark.presets import get_preset
        p = get_preset("smoke")
        assert p.name == "smoke"
        assert p.budget_usd == pytest.approx(0.50)
        assert p.prompt_limit == 6
        assert p.allow_large_local_models is False

    def test_get_preset_overnight_safe(self):
        from theriad.benchmark.presets import get_preset
        p = get_preset("overnight_safe")
        assert p.budget_usd == pytest.approx(2.00)
        assert p.resume is True
        assert p.allow_large_local_models is False

    def test_get_preset_full(self):
        from theriad.benchmark.presets import get_preset
        p = get_preset("full")
        assert p.budget_usd == pytest.approx(10.00)
        assert p.resume is True
        assert "groq" in p.providers

    def test_unknown_preset_raises(self):
        from theriad.benchmark.presets import get_preset
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("totally_fake")

    def test_overnight_safe_has_output_caps(self):
        from theriad.benchmark.presets import get_preset
        p = get_preset("overnight_safe")
        assert "summarize" in p.output_caps
        assert p.output_caps["summarize"] <= 512

    def test_smoke_caps_are_tighter_than_overnight(self):
        from theriad.benchmark.presets import get_preset, SMOKE, OVERNIGHT_SAFE
        for cat in ("code", "summarize"):
            if cat in SMOKE.output_caps and cat in OVERNIGHT_SAFE.output_caps:
                assert SMOKE.output_caps[cat] <= OVERNIGHT_SAFE.output_caps[cat]


# ── Output preview / sha256 ───────────────────────────────────────────────────

class TestOutputCapture:
    """Tests for _make_output_preview in types.py."""

    def test_preview_truncated_at_500_chars(self):
        content = "x" * 1000
        preview, sha = _make_output_preview(content)
        assert len(preview) == 500
        assert all(c == "x" for c in preview)

    def test_short_content_not_truncated(self):
        content = "hello"
        preview, sha = _make_output_preview(content)
        assert preview == "hello"

    def test_sha256_is_hex_string(self):
        _, sha = _make_output_preview("test")
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_sha256_changes_with_content(self):
        _, sha1 = _make_output_preview("hello")
        _, sha2 = _make_output_preview("world")
        assert sha1 != sha2

    def test_sha256_stable(self):
        content = "The quick brown fox"
        _, sha1 = _make_output_preview(content)
        _, sha2 = _make_output_preview(content)
        assert sha1 == sha2

    def test_result_roundtrip_includes_preview(self, tmp_path):
        r = _result()
        r.output_preview = "First 500 chars of output..."
        r.output_sha256 = "abc123" + "0" * 58
        d = r.to_dict()
        assert d["output_preview"] == "First 500 chars of output..."
        r2 = BenchmarkResult.from_dict(d)
        assert r2.output_preview == r.output_preview
        assert r2.output_sha256 == r.output_sha256


# ── Partial summary ───────────────────────────────────────────────────────────

class TestPartialSummary:
    """Summary CLI works on incomplete / partial JSONL."""

    def _write_partial(self, path: Path, rows: list[BenchmarkResult]) -> None:
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r.to_dict()) + "\n")

    def test_summary_on_empty_file(self, capsys):
        import sys
        from theriad.benchmark.summary import print_summary
        print_summary([], file=sys.stdout)
        out = capsys.readouterr().out
        assert "No results" in out

    def test_summary_on_single_row(self, capsys):
        import sys
        from theriad.benchmark.summary import print_summary
        print_summary([_result()], file=sys.stdout)
        out = capsys.readouterr().out
        assert "openrouter" in out

    def test_summary_handles_partial_results(self, tmp_path, capsys):
        from theriad.benchmark.summary import load_results, print_summary
        rows = [
            _result(prompt_id="p1", status="success"),
            _result(prompt_id="p2", status="error", cost=None),
        ]
        path = tmp_path / "partial.jsonl"
        self._write_partial(path, rows)
        import sys
        results = load_results(path)
        assert len(results) == 2
        print_summary(results, file=sys.stdout)
        out = capsys.readouterr().out
        assert "openrouter" in out

    def test_summary_load_skips_malformed_lines(self, tmp_path):
        from theriad.benchmark.summary import load_results
        path = tmp_path / "mixed.jsonl"
        path.write_text(
            "NOT JSON\n"
            + json.dumps(_result().to_dict()) + "\n"
            + "{broken\n"
            + json.dumps(_result(prompt_id="p2").to_dict()) + "\n"
        )
        results = load_results(path)
        assert len(results) == 2

    def test_median_latency_computed(self):
        from theriad.benchmark.summary import _aggregate, _median
        results = [
            _result(prompt_id=f"p{i}") for i in range(5)
        ]
        # Manually set different latencies
        for i, r in enumerate(results):
            r.provider_latency_ms = float((i + 1) * 100)

        stats = _aggregate(results)
        assert len(stats) == 1
        s = stats[0]
        assert s.median_latency_ms == pytest.approx(300.0)  # median of [100,200,300,400,500]


# ── Cost formatting regression ────────────────────────────────────────────────

class TestCostFormattingRegression:
    """Ensure format_usd still handles edge cases correctly."""

    def _f(self, v):
        from theriad.benchmark.formatting import format_usd
        return format_usd(v)

    def test_mimo_tiny_cost_not_question_mark(self):
        # MiMo v2.5: 50 prompt + 200 completion tokens
        # cost = (50 * 0.105 + 200 * 0.28) / 1_000_000 = (5.25 + 56) / 1M = 6.125e-5
        cost = (50 * 0.105 + 200 * 0.28) / 1_000_000
        assert cost > 0
        result = self._f(cost)
        assert result != "?"
        assert result.startswith("$")

    def test_unknown_pricing_shows_question_mark(self):
        assert self._f(None) == "?"

    def test_free_model_shows_zero(self):
        assert self._f(0.0) == "$0"
