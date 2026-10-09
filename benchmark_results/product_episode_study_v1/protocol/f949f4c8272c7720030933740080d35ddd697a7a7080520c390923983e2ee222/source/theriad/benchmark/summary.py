"""
Benchmark summary CLI.

Reads a JSONL results file produced by the runner and prints ranked tables
grouped by provider+model and by category.

    python3 -m theriad.benchmark.summary benchmark_results/latest.jsonl
    python3 -m theriad.benchmark.summary benchmark_results/  # auto-selects latest

Ranking metric: composite score (lower = better):
    score = normalized_latency * 0.4
          + normalized_cost    * 0.4
          + failure_rate       * 0.2

Output columns (per model summary):
    provider  model  N  success%  fail%  avg_cost  avg_latency  avg_tok/s  quality%
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from theriad.benchmark.types import BenchmarkResult


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(path: Path) -> list[BenchmarkResult]:
    """Load BenchmarkResult records from a JSONL file."""
    results = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                results.append(BenchmarkResult.from_dict(obj))
            except Exception as exc:
                print(f"Warning: skipping line {lineno}: {exc}", file=sys.stderr)
    return results


def _resolve_path(path: Path) -> Path:
    """If path is a directory, return the most recently modified JSONL inside."""
    if path.is_file():
        return path
    if path.is_dir():
        jsonl_files = sorted(path.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {path}")
        return jsonl_files[-1]
    raise FileNotFoundError(f"Path not found: {path}")


# ── Aggregation ────────────────────────────────────────────────────────────────

class ModelStats(NamedTuple):
    provider: str
    model: str
    n: int
    success_count: int
    fail_count: int
    budget_skip_count: int
    avg_cost_usd: float | None
    avg_latency_ms: float | None
    median_latency_ms: float | None
    avg_tps: float | None
    quality_pass_rate: float | None   # fraction of quality-checked runs that passed


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]


def _aggregate(results: list[BenchmarkResult]) -> list[ModelStats]:
    """Aggregate results by (provider, model)."""
    buckets: dict[tuple[str, str], list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        buckets[(r.provider, r.model)].append(r)

    stats = []
    for (provider, model), records in sorted(buckets.items()):
        n = len(records)
        successes = [r for r in records if r.status == "success"]
        fails = [r for r in records if r.status in ("error", "timeout")]
        skips = [r for r in records if r.status in ("skipped", "budget_exceeded")]

        costs = [r.actual_cost_usd for r in successes if r.actual_cost_usd is not None]
        latencies = [r.provider_latency_ms for r in successes]
        tps_vals = [r.tokens_per_second for r in successes if r.tokens_per_second is not None]

        quality_checked = [r for r in successes if r.quality_pass is not None]
        quality_pass_rate = (
            sum(1 for r in quality_checked if r.quality_pass) / len(quality_checked)
            if quality_checked else None
        )

        stats.append(ModelStats(
            provider=provider,
            model=model,
            n=n,
            success_count=len(successes),
            fail_count=len(fails),
            budget_skip_count=len(skips),
            avg_cost_usd=sum(costs) / len(costs) if costs else None,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else None,
            median_latency_ms=_median(latencies),
            avg_tps=sum(tps_vals) / len(tps_vals) if tps_vals else None,
            quality_pass_rate=quality_pass_rate,
        ))
    return stats


def _aggregate_by_category(
    results: list[BenchmarkResult],
) -> dict[str, list[ModelStats]]:
    """Aggregate results split by category."""
    by_cat: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    return {cat: _aggregate(recs) for cat, recs in sorted(by_cat.items())}


# ── Ranking ────────────────────────────────────────────────────────────────────

def _rank(stats: list[ModelStats]) -> list[tuple[float, ModelStats]]:
    """
    Assign a composite score to each model. Lower = better.

    Weights:
        40% — normalised avg latency  (lower is better)
        40% — normalised avg cost     (lower is better; free = 0)
        20% — failure rate            (lower is better)
    """
    def _safe_latency(s: ModelStats) -> float:
        return s.avg_latency_ms if s.avg_latency_ms is not None else float("inf")

    def _safe_cost(s: ModelStats) -> float:
        return s.avg_cost_usd if s.avg_cost_usd is not None else float("inf")

    latencies = [_safe_latency(s) for s in stats]
    costs = [_safe_cost(s) for s in stats]

    max_lat = max(latencies) or 1.0
    max_cost = max(costs) or 1.0

    ranked = []
    for s in stats:
        fail_rate = s.fail_count / s.n if s.n > 0 else 1.0
        norm_lat = _safe_latency(s) / max_lat
        norm_cost = _safe_cost(s) / max_cost if max_cost < float("inf") else 0.0
        score = norm_lat * 0.4 + norm_cost * 0.4 + fail_rate * 0.2
        ranked.append((score, s))

    ranked.sort(key=lambda x: x[0])
    return ranked


# ── Formatting ─────────────────────────────────────────────────────────────────

_COL_W = {
    "rank":     4,
    "provider": 12,
    "model":    34,
    "n":         5,
    "ok%":       6,
    "fail%":     7,
    "cost":     12,
    "latency":  12,
    "tok/s":     9,
    "quality":   9,
}

def _header() -> str:
    cols = ["rank", "provider", "model", "n", "ok%", "fail%", "cost", "latency", "tok/s", "quality"]
    return "  ".join(c.ljust(_COL_W[c]) for c in cols)


def _row(rank: int, score: float, s: ModelStats) -> str:
    success_pct = f"{s.success_count / s.n * 100:.0f}%" if s.n else "—"
    fail_pct    = f"{s.fail_count    / s.n * 100:.0f}%" if s.n else "—"
    from theriad.benchmark.formatting import format_usd
    cost_str    = format_usd(s.avg_cost_usd)
    lat_str     = f"{s.avg_latency_ms:.0f}ms" if s.avg_latency_ms is not None else "—"
    tps_str     = f"{s.avg_tps:.0f}" if s.avg_tps is not None else "—"
    q_str       = f"{s.quality_pass_rate * 100:.0f}%" if s.quality_pass_rate is not None else "—"
    model_short = s.model[:32]

    parts = [
        f"#{rank}".ljust(_COL_W["rank"]),
        s.provider.ljust(_COL_W["provider"]),
        model_short.ljust(_COL_W["model"]),
        str(s.n).ljust(_COL_W["n"]),
        success_pct.ljust(_COL_W["ok%"]),
        fail_pct.ljust(_COL_W["fail%"]),
        cost_str.ljust(_COL_W["cost"]),
        lat_str.ljust(_COL_W["latency"]),
        tps_str.ljust(_COL_W["tok/s"]),
        q_str.ljust(_COL_W["quality"]),
    ]
    return "  ".join(parts)


# ── Main summary ──────────────────────────────────────────────────────────────

def print_summary(results: list[BenchmarkResult], file=sys.stdout) -> None:
    if not results:
        print("No results to summarise.", file=file)
        return

    total = len(results)
    successes = sum(1 for r in results if r.status == "success")
    total_cost = sum(r.actual_cost_usd for r in results if r.actual_cost_usd is not None)
    categories = sorted({r.category for r in results})

    print(f"\n{'═' * 100}", file=file)
    print(f"  BENCHMARK SUMMARY", file=file)
    print(f"  {total} results · {successes} success · ${total_cost:.4f} total spend", file=file)
    print(f"  Categories: {', '.join(categories)}", file=file)
    print(f"{'═' * 100}", file=file)

    # ── Overall ranking ───────────────────────────────────────────────────────
    print(f"\n  OVERALL RANKING  (score: 40% latency + 40% cost + 20% failure rate)", file=file)
    print(f"  {'─' * 98}", file=file)
    print(f"  {_header()}", file=file)
    print(f"  {'─' * 98}", file=file)

    stats = _aggregate(results)
    ranked = _rank(stats)
    for i, (score, s) in enumerate(ranked, 1):
        print(f"  {_row(i, score, s)}", file=file)

    # ── Per-category ranking ──────────────────────────────────────────────────
    by_cat = _aggregate_by_category(results)
    for cat, cat_stats in by_cat.items():
        if not cat_stats:
            continue
        print(f"\n  CATEGORY: {cat.upper()}", file=file)
        print(f"  {'─' * 98}", file=file)
        print(f"  {_header()}", file=file)
        print(f"  {'─' * 98}", file=file)
        ranked_cat = _rank(cat_stats)
        for i, (score, s) in enumerate(ranked_cat, 1):
            print(f"  {_row(i, score, s)}", file=file)

    print(f"\n{'═' * 100}\n", file=file)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python3 -m theriad.benchmark.summary",
        description="Summarise benchmark results from a JSONL file.",
    )
    p.add_argument(
        "path",
        type=Path,
        help="Path to a benchmark JSONL file or directory (auto-selects latest)",
    )
    p.add_argument(
        "--category", "-c",
        help="Filter to a specific category (e.g. code, reasoning)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    try:
        path = _resolve_path(args.path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {path}", file=sys.stderr)
    results = load_results(path)

    if args.category:
        results = [r for r in results if r.category == args.category]
        if not results:
            print(f"No results for category {args.category!r}", file=sys.stderr)
            sys.exit(1)

    print_summary(results)


if __name__ == "__main__":
    main()
