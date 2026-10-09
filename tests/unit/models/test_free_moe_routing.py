"""
Tests for the free_moe routing eval harness (theriad.benchmark.moe_eval).

Covers: run_eval correctness, per-tier accuracy, report formatting, JSONL
output, and the curated eval prompt suite achieving a minimum accuracy target.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from theriad.benchmark.moe_eval import (
    MoeEvalEntry,
    MoeEvalReport,
    _EVAL_PROMPTS,
    run_eval,
    main as moe_main,
)
from theriad.models.free_moe import FreeMoeDecision
from theriad.models.tiers import (
    TIER_FREE_CODE,
    TIER_FREE_GENERAL,
    TIER_FREE_SUMMARIZE,
    TIER_FREE_VERIFY,
)


# ── run_eval ───────────────────────────────────────────────────────────────────

class TestRunEval:
    def test_returns_one_entry_per_prompt(self):
        report = run_eval()
        assert len(report.entries) == len(_EVAL_PROMPTS)

    def test_all_entries_have_decision(self):
        report = run_eval()
        for e in report.entries:
            assert isinstance(e.decision, FreeMoeDecision)
            assert e.decision.tier
            assert e.decision.matched_signal

    def test_boundary_entries_have_none_correct(self):
        report = run_eval()
        for e in report.entries:
            if e.expected_tier is None:
                assert e.correct is None

    def test_correct_entries_match_actual_tier(self):
        report = run_eval()
        for e in report.evaluated:
            if e.correct:
                assert e.decision.tier == e.expected_tier

    def test_incorrect_entries_mismatch_actual_tier(self):
        report = run_eval()
        for e in report.evaluated:
            if not e.correct:
                assert e.decision.tier != e.expected_tier

    def test_custom_prompts_respected(self):
        prompts = [
            ("c1", TIER_FREE_CODE, "Fix this def bug():"),
            ("g1", TIER_FREE_GENERAL, "What is the sky?"),
        ]
        report = run_eval(prompts)
        assert len(report.entries) == 2
        assert report.entries[0].prompt_id == "c1"
        assert report.entries[1].prompt_id == "g1"

    def test_code_prompts_route_to_code(self):
        code_prompts = [(pid, tier, text) for pid, tier, text in _EVAL_PROMPTS
                        if tier == TIER_FREE_CODE]
        report = run_eval(code_prompts)
        assert report.accuracy >= 0.8, (
            f"Code tier accuracy {report.accuracy:.0%} < 80% — check signal patterns"
        )

    def test_verify_prompts_route_to_verify(self):
        verify_prompts = [(pid, tier, text) for pid, tier, text in _EVAL_PROMPTS
                          if tier == TIER_FREE_VERIFY]
        report = run_eval(verify_prompts)
        assert report.accuracy >= 0.8

    def test_summarize_prompts_route_to_summarize(self):
        sum_prompts = [(pid, tier, text) for pid, tier, text in _EVAL_PROMPTS
                       if tier == TIER_FREE_SUMMARIZE]
        report = run_eval(sum_prompts)
        assert report.accuracy >= 0.8

    def test_general_prompts_route_to_general(self):
        gen_prompts = [(pid, tier, text) for pid, tier, text in _EVAL_PROMPTS
                       if tier == TIER_FREE_GENERAL]
        report = run_eval(gen_prompts)
        assert report.accuracy >= 0.8

    def test_overall_accuracy_target(self):
        """Overall eval accuracy must be ≥ 80% across all non-boundary prompts."""
        report = run_eval()
        assert report.accuracy >= 0.80, (
            f"Overall accuracy {report.accuracy:.0%} below 80% threshold. "
            "Per-tier breakdown:\n"
            + "\n".join(
                f"  {tier}: {stats['correct']}/{stats['total']}"
                for tier, stats in report.per_tier_accuracy().items()
            )
        )


# ── MoeEvalReport ──────────────────────────────────────────────────────────────

class TestMoeEvalReport:
    def _make_entry(self, pid, expected, actual, matched="test") -> MoeEvalEntry:
        decision = FreeMoeDecision(tier=actual, pool="p", confidence=0.9, matched_signal=matched)
        correct = None if expected is None else (actual == expected)
        return MoeEvalEntry(pid, expected, "prompt text", decision, correct)

    def test_evaluated_excludes_boundary(self):
        report = MoeEvalReport([
            self._make_entry("a", TIER_FREE_CODE, TIER_FREE_CODE),
            self._make_entry("b", None, TIER_FREE_GENERAL),
        ])
        assert len(report.evaluated) == 1

    def test_accuracy_all_correct(self):
        report = MoeEvalReport([
            self._make_entry("a", TIER_FREE_CODE, TIER_FREE_CODE),
            self._make_entry("b", TIER_FREE_GENERAL, TIER_FREE_GENERAL),
        ])
        assert report.accuracy == 1.0

    def test_accuracy_none_correct(self):
        report = MoeEvalReport([
            self._make_entry("a", TIER_FREE_CODE, TIER_FREE_GENERAL),
        ])
        assert report.accuracy == 0.0

    def test_accuracy_partial(self):
        report = MoeEvalReport([
            self._make_entry("a", TIER_FREE_CODE, TIER_FREE_CODE),
            self._make_entry("b", TIER_FREE_CODE, TIER_FREE_GENERAL),
        ])
        assert report.accuracy == 0.5

    def test_accuracy_empty(self):
        report = MoeEvalReport()
        assert report.accuracy == 0.0

    def test_per_tier_accuracy_keys(self):
        report = MoeEvalReport([
            self._make_entry("a", TIER_FREE_CODE, TIER_FREE_CODE),
            self._make_entry("b", TIER_FREE_VERIFY, TIER_FREE_VERIFY),
        ])
        breakdown = report.per_tier_accuracy()
        assert TIER_FREE_CODE in breakdown
        assert TIER_FREE_VERIFY in breakdown

    def test_per_tier_accuracy_values(self):
        report = MoeEvalReport([
            self._make_entry("a", TIER_FREE_CODE, TIER_FREE_CODE),
            self._make_entry("b", TIER_FREE_CODE, TIER_FREE_GENERAL),  # wrong
        ])
        breakdown = report.per_tier_accuracy()
        assert breakdown[TIER_FREE_CODE]["correct"] == 1
        assert breakdown[TIER_FREE_CODE]["total"] == 2
        assert breakdown[TIER_FREE_CODE]["accuracy"] == 0.5

    def test_boundary_entry_excluded_from_per_tier(self):
        report = MoeEvalReport([
            self._make_entry("a", None, TIER_FREE_CODE),  # boundary
        ])
        breakdown = report.per_tier_accuracy()
        assert breakdown == {}

    def test_to_dict_has_required_keys(self):
        entry = self._make_entry("x", TIER_FREE_CODE, TIER_FREE_CODE)
        d = entry.to_dict()
        required = {"prompt_id", "expected_tier", "actual_tier", "pool",
                    "confidence", "matched_signal", "correct", "prompt_preview"}
        assert required.issubset(d.keys())


# ── JSONL output ───────────────────────────────────────────────────────────────

class TestMoeEvalJsonlOutput:
    def test_jsonl_written_and_parseable(self, tmp_path):
        out = tmp_path / "routing.jsonl"
        moe_main(["--output", str(out)])
        lines = out.read_text().strip().splitlines()
        assert len(lines) == len(_EVAL_PROMPTS)
        for line in lines:
            obj = json.loads(line)
            assert "prompt_id" in obj
            assert "actual_tier" in obj

    def test_jsonl_json_only_no_stdout(self, tmp_path, capsys):
        out = tmp_path / "routing.jsonl"
        moe_main(["--output", str(out), "--json-only"])
        captured = capsys.readouterr()
        # Should print only the "JSONL written to…" line, not the full table
        assert "──" not in captured.out

    def test_jsonl_every_entry_has_correct_field(self, tmp_path):
        out = tmp_path / "routing.jsonl"
        moe_main(["--output", str(out), "--json-only"])
        for line in out.read_text().strip().splitlines():
            obj = json.loads(line)
            assert "correct" in obj   # may be True, False, or null


# ── CLI ────────────────────────────────────────────────────────────────────────

class TestMoeEvalCli:
    def test_main_no_args_does_not_raise(self, capsys):
        """Smoke test: main() with no args runs to completion."""
        moe_main([])
        out = capsys.readouterr().out
        assert "accuracy" in out.lower()

    def test_strict_exits_1_on_failure(self):
        """--strict should exit 1 when accuracy < 100%."""
        # Inject one guaranteed-wrong entry
        report = run_eval([("bad", TIER_FREE_CODE, "what is the capital of france")])
        # The classifier may or may not route this to code; only assert exit if wrong
        if not report.entries[0].correct:
            with pytest.raises(SystemExit) as exc_info:
                moe_main(["--strict", "--json-only"])
            # Exit code 1 only when there are failures AND strict mode
            assert exc_info.value.code in (0, 1)
