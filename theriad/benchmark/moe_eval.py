"""
free_moe_experimental routing eval — no model calls, no API keys required.

Runs a curated prompt set through ``route_free_moe()`` and reports:
  - Per-prompt routing decision (tier + matched signal)
  - Per-tier accuracy against expected assignments
  - Overall classification rate
  - JSONL output compatible with the existing benchmark schema

Usage
─────
  # Quick routing table (stdout only):
  python3 -m theriad.benchmark.moe_eval

  # Save JSONL + summary:
  python3 -m theriad.benchmark.moe_eval --output moe_routing.jsonl

  # Evaluate against a specific subset of expected tiers:
  python3 -m theriad.benchmark.moe_eval --strict

Design
──────
This eval is intentionally offline: it tests the *routing classifier* in
``theriad.models.free_moe``, not model quality.  To benchmark actual model
outputs, pass the resulting JSONL to the existing ``theriad.benchmark.runner``
with ``--policy free_moe_experimental``.

The curated prompts below cover four tiers and several boundary cases.
Adding new edge-case prompts here is the main way to improve the classifier.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from theriad.models.free_moe import FreeMoeDecision, route_free_moe
from theriad.models.tiers import (
    TIER_FREE_CODE,
    TIER_FREE_GENERAL,
    TIER_FREE_SUMMARIZE,
    TIER_FREE_VERIFY,
)

# ── Curated eval prompts ───────────────────────────────────────────────────────
# Each entry: (prompt_id, expected_tier, prompt_text)
# expected_tier=None means "any tier is acceptable" (boundary/ambiguous cases)

_EVAL_PROMPTS: list[tuple[str, str | None, str]] = [
    # ── CODE ──────────────────────────────────────────────────────────────────
    ("code_01", TIER_FREE_CODE,
     "Fix this bug: `def add(a, b): return a - b`"),
    ("code_02", TIER_FREE_CODE,
     "Write a Python function that reverses a linked list."),
    ("code_03", TIER_FREE_CODE,
     "```python\ndef fib(n): return fib(n-1) + fib(n-2)\n``` — why is this slow?"),
    ("code_04", TIER_FREE_CODE,
     "I'm getting a TypeError: 'NoneType' object is not subscriptable."),
    ("code_05", TIER_FREE_CODE,
     "Explain how to use pytest fixtures."),
    ("code_06", TIER_FREE_CODE,
     "What's the difference between `import` and `from x import y`?"),
    ("code_07", TIER_FREE_CODE,
     "Debug this: the function compiles but outputs wrong results at runtime."),
    ("code_08", TIER_FREE_CODE,
     "Write a JavaScript function that debounces user input."),
    ("code_09", TIER_FREE_CODE,
     "My Rust code has a segfault — how do I find it?"),
    ("code_10", TIER_FREE_CODE,
     "What does `class Foo(Bar):` mean in Python?"),

    # ── VERIFY ────────────────────────────────────────────────────────────────
    ("verify_01", TIER_FREE_VERIFY,
     "Can you fact-check this claim: the Amazon river is the longest river."),
    ("verify_02", TIER_FREE_VERIFY,
     "Is it true that coffee causes cancer?"),
    ("verify_03", TIER_FREE_VERIFY,
     "Please verify: according to the paper, the sample size was 200 participants."),
    ("verify_04", TIER_FREE_VERIFY,
     "Check whether this citation is correct: Smith et al. (2020)."),
    ("verify_05", TIER_FREE_VERIFY,
     "Does this source support the claim that vaccines cause autism?"),
    ("verify_06", TIER_FREE_VERIFY,
     "Does eating breakfast boost metabolism? What do sources say?"),

    # ── SUMMARIZE ─────────────────────────────────────────────────────────────
    ("sum_01", TIER_FREE_SUMMARIZE,
     "Summarize this article in three bullet points."),
    ("sum_02", TIER_FREE_SUMMARIZE,
     "TL;DR of the following research paper: [...]"),
    ("sum_03", TIER_FREE_SUMMARIZE,
     "Extract the key points from this meeting transcript."),
    ("sum_04", TIER_FREE_SUMMARIZE,
     "What are the main ideas in this document?"),
    ("sum_05", TIER_FREE_SUMMARIZE,
     "Give me bullet points summarizing the pros and cons."),
    # Long message → should hit the "long_message" fallback to summarize tier
    ("sum_06", TIER_FREE_SUMMARIZE,
     "Please read the following and tell me what you think: " + "word " * 200),

    # ── GENERAL ───────────────────────────────────────────────────────────────
    ("gen_01", TIER_FREE_GENERAL,
     "What is the capital of France?"),
    ("gen_02", TIER_FREE_GENERAL,
     "Tell me a short story about a robot and a dog."),
    ("gen_03", TIER_FREE_GENERAL,
     "How do I make pasta carbonara?"),
    ("gen_04", TIER_FREE_GENERAL,
     "What is the meaning of life?"),
    ("gen_05", TIER_FREE_GENERAL,
     "Recommend a good sci-fi novel from the 1980s."),

    # ── BOUNDARY / AMBIGUOUS (expected=None) ──────────────────────────────────
    ("boundary_01", None,
     "What are the key points of the error message?"),        # "error" → code, "key points" → summarize
    ("boundary_02", None,
     "Can you verify the syntax of this Python function?"),   # "verify" + "function" → could be either
    ("boundary_03", None,
     "Summarize the bug report."),                            # "summarize" + "bug" → could be either
]


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class MoeEvalEntry:
    prompt_id: str
    expected_tier: str | None
    prompt: str
    decision: FreeMoeDecision
    correct: bool | None  # None for boundary cases

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "expected_tier": self.expected_tier,
            "actual_tier": self.decision.tier,
            "pool": self.decision.pool,
            "confidence": self.decision.confidence,
            "matched_signal": self.decision.matched_signal,
            "correct": self.correct,
            "prompt_preview": self.prompt[:120],
        }


@dataclass
class MoeEvalReport:
    entries: list[MoeEvalEntry] = field(default_factory=list)

    @property
    def evaluated(self) -> list[MoeEvalEntry]:
        """Entries with a defined expected tier (excludes boundary cases)."""
        return [e for e in self.entries if e.expected_tier is not None]

    @property
    def accuracy(self) -> float:
        ev = self.evaluated
        if not ev:
            return 0.0
        return sum(1 for e in ev if e.correct) / len(ev)

    def per_tier_accuracy(self) -> dict[str, dict]:
        tiers: dict[str, list[bool]] = {}
        for e in self.evaluated:
            tiers.setdefault(e.expected_tier or "unknown", []).append(bool(e.correct))
        return {
            tier: {
                "correct": sum(results),
                "total": len(results),
                "accuracy": sum(results) / len(results) if results else 0.0,
            }
            for tier, results in sorted(tiers.items())
        }


# ── Eval runner ────────────────────────────────────────────────────────────────

def run_eval(prompts: list[tuple[str, str | None, str]] | None = None) -> MoeEvalReport:
    """Run the routing eval and return a report. No I/O."""
    prompts = prompts or _EVAL_PROMPTS
    report = MoeEvalReport()
    for prompt_id, expected_tier, text in prompts:
        decision = route_free_moe(text)
        correct = None if expected_tier is None else (decision.tier == expected_tier)
        report.entries.append(MoeEvalEntry(
            prompt_id=prompt_id,
            expected_tier=expected_tier,
            prompt=text,
            decision=decision,
            correct=correct,
        ))
    return report


# ── Formatting ─────────────────────────────────────────────────────────────────

_TIER_SHORT = {
    TIER_FREE_CODE:      "CODE    ",
    TIER_FREE_VERIFY:    "VERIFY  ",
    TIER_FREE_SUMMARIZE: "SUMMARIZE",
    TIER_FREE_GENERAL:   "GENERAL  ",
}


def _fmt_tier(tier: str) -> str:
    return _TIER_SHORT.get(tier, tier[:9].upper())


def print_report(report: MoeEvalReport, *, strict: bool = False) -> None:
    print("\n── free_moe_experimental routing eval ──────────────────────────────────")
    print(f"{'ID':<14} {'EXPECTED':<12} {'ACTUAL':<12} {'SIGNAL':<22} {'OK'}")
    print("─" * 72)

    for e in report.entries:
        expected_label = _fmt_tier(e.expected_tier) if e.expected_tier else "  (any)  "
        actual_label   = _fmt_tier(e.decision.tier)
        ok_label = "✓" if e.correct is True else ("✗" if e.correct is False else "~")
        print(f"{e.prompt_id:<14} {expected_label:<12} {actual_label:<12} "
              f"{e.decision.matched_signal:<22} {ok_label}")

    print("─" * 72)
    print(f"\nOverall accuracy (excluding boundary): {report.accuracy:.0%} "
          f"({sum(1 for e in report.evaluated if e.correct)}/{len(report.evaluated)})\n")

    print("Per-tier accuracy:")
    for tier, stats in report.per_tier_accuracy().items():
        short = _TIER_SHORT.get(tier, tier)
        bar_filled = int(stats["accuracy"] * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"  {short:<12} [{bar}] {stats['accuracy']:.0%}  "
              f"({stats['correct']}/{stats['total']})")

    boundary = [e for e in report.entries if e.expected_tier is None]
    if boundary:
        print(f"\nBoundary/ambiguous cases ({len(boundary)}) — no pass/fail:")
        for e in boundary:
            print(f"  {e.prompt_id:<14} → {_fmt_tier(e.decision.tier)}  "
                  f"(signal: {e.decision.matched_signal})")

    if strict and report.accuracy < 1.0:
        print("\n[strict mode] Some prompts routed to unexpected tiers.")
        sys.exit(1)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python3 -m theriad.benchmark.moe_eval",
        description="Offline routing eval for the free_moe_experimental policy.",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Write per-entry JSONL to this file path (optional).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any expected-tier prompt is misrouted.",
    )
    p.add_argument(
        "--json-only",
        action="store_true",
        dest="json_only",
        help="Suppress the human-readable report; write only the JSONL.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = run_eval()

    if not args.json_only:
        print_report(report, strict=args.strict)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            for entry in report.entries:
                fh.write(json.dumps(entry.to_dict()) + "\n")
        print(f"\nJSONL written to {args.output}  ({len(report.entries)} entries)")

    if args.strict and report.accuracy < 1.0 and args.json_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
