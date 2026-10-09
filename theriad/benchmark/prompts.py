"""
Benchmark prompt loader.

Reads benchmark_prompts.jsonl from a configurable path.
Supports stratified sampling: pick at most N prompts while keeping
category representation as balanced as possible.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .types import BenchmarkPrompt

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "benchmark_prompts.jsonl"


def load_prompts(path: Path | str | None = None) -> list[BenchmarkPrompt]:
    """Load all prompts from the JSONL file."""
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Benchmark prompt file not found: {p}. "
            "Expected benchmark_prompts.jsonl in the project root."
        )
    prompts = []
    with p.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {p}: {exc}") from exc
            prompts.append(BenchmarkPrompt(
                id=obj["id"],
                category=obj["category"],
                prompt=obj["prompt"],
                max_output_tokens=obj.get("max_output_tokens") or None,
                expected_properties=obj.get("expected_properties", []),
                answer=obj.get("answer"),
                answer_type=obj.get("answer_type"),
            ))
    return prompts


def stratified_sample(prompts: list[BenchmarkPrompt], limit: int) -> list[BenchmarkPrompt]:
    """
    Return up to `limit` prompts distributed as evenly as possible across categories.

    Within each category the original order is preserved. Categories with fewer
    prompts than their fair share contribute all their prompts; the slack is
    redistributed to remaining categories.
    """
    if limit <= 0:
        return []
    if limit >= len(prompts):
        return list(prompts)

    by_category: dict[str, list[BenchmarkPrompt]] = defaultdict(list)
    for p in prompts:
        by_category[p.category].append(p)

    categories = list(by_category.keys())
    remaining = limit
    selected: list[BenchmarkPrompt] = []

    # Greedy round: distribute quota, let short categories give back slack
    pending = {cat: by_category[cat] for cat in categories}
    while remaining > 0 and pending:
        quota = max(1, remaining // len(pending))
        next_pending = {}
        for cat, pool in pending.items():
            take = min(quota, len(pool))
            selected.extend(pool[:take])
            remaining -= take
            leftover = pool[take:]
            if leftover and remaining > 0:
                next_pending[cat] = leftover
        # If we made no progress (all categories exhausted), stop
        if len(next_pending) == len(pending):
            break
        pending = next_pending

    # Restore original ordering
    order = {p.id: i for i, p in enumerate(prompts)}
    selected.sort(key=lambda p: order[p.id])
    return selected[:limit]
