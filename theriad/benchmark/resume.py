"""
Benchmark resume helpers.

Reads an existing JSONL results file and returns the set of already-completed
(provider, model, prompt_id, category) keys so the runner can skip them.

Usage:
    from theriad.benchmark.resume import load_completed_keys, ResumeState

    state = load_completed_keys(Path("benchmark_results/latest.jsonl"))
    print(f"Loaded {state.completed} completed rows. Remaining: ...")
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from theriad.core.logging.logger import get_logger

logger = get_logger("benchmark.resume")

# The 4-tuple that uniquely identifies a benchmark row.
CompletedKey = tuple[str, str, str, str]  # (provider, model, prompt_id, category)


@dataclass
class ResumeState:
    completed_keys: frozenset[CompletedKey]
    source_path: Path | None

    @property
    def completed(self) -> int:
        return len(self.completed_keys)

    def is_done(self, provider: str, model: str, prompt_id: str, category: str) -> bool:
        return (provider, model, prompt_id, category) in self.completed_keys

    def print_summary(self, *, file=None) -> None:
        import sys
        fh = file or sys.stdout
        if self.source_path:
            print(
                f"  Resume: loaded {self.completed} completed row(s) "
                f"from {self.source_path}",
                file=fh,
            )
        else:
            print("  Resume: no existing results file — starting fresh.", file=fh)


_EMPTY = ResumeState(completed_keys=frozenset(), source_path=None)


def load_completed_keys(path: Path | None) -> ResumeState:
    """
    Parse a JSONL results file and return the set of completed benchmark keys.

    A row is considered completed if its status is "success".
    Rows with status "error", "timeout", "skipped", or "budget_exceeded" are
    NOT included — they may be retried.

    Args:
        path: Path to the JSONL file. If None or non-existent, returns empty state.

    Returns:
        ResumeState with completed keys and the source path.
    """
    if path is None or not path.exists():
        return _EMPTY

    keys: set[CompletedKey] = set()
    skipped = 0
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Resume: skipping malformed line %d: %s", lineno, exc)
                skipped += 1
                continue

            if obj.get("status") != "success":
                continue

            key: CompletedKey = (
                obj.get("provider", ""),
                obj.get("model", ""),
                obj.get("prompt_id", ""),
                obj.get("category", ""),
            )
            keys.add(key)

    state = ResumeState(completed_keys=frozenset(keys), source_path=path)
    logger.info(
        "Resume: %d completed keys loaded from %s%s",
        len(keys), path,
        f" ({skipped} malformed lines skipped)" if skipped else "",
    )
    return state
