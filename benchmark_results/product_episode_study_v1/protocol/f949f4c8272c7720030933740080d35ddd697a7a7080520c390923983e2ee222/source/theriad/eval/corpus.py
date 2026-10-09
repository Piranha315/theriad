"""
Vintage-tagged evaluation corpus.

An ``EvalItem`` is a ``BenchmarkPrompt`` plus provenance: which benchmark it came
from and when that benchmark was published. Vintage is a **covariate, not a
difficulty label** — see ``theriad.eval.grid`` for why difficulty is derived from
measured outcomes instead.

Why vintage is tracked at all
-----------------------------
Contamination correlates with a benchmark's age and popularity, because training
corpora are cumulative snapshots of the web: an item published in 2020 has had
years to be scraped, reproduced in tutorials, and folded into instruction-tuning
sets. Newer items are *less* contaminated, and several modern benchmarks are
built for it (time-windowed problem sets, held-out private splits).

That matters for a router eval in a specific and limited way. Contamination
invalidates a *model capability* measurement — you wanted reasoning and measured
recall. It does much less damage to a *router* measurement, because the router's
objective is "cheapest action that succeeds", and memorisation is a legitimate
way to succeed. What contamination breaks is **transfer**: the router's decision
boundary gets calibrated on a difficulty distribution where cheap models look
unusually strong, and production queries are not memorised.

So vintage is carried through to the report as a stratification axis. If the
router's regret is flat across vintages, the boundary is probably transferable;
if regret climbs sharply on recent items, it is fitting contamination.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from theriad.benchmark.types import BenchmarkPrompt


@dataclass(frozen=True)
class EvalItem:
    """One gradable evaluation item with provenance.

    Fields:
        id:          Stable unique identifier; used as the grid cell key.
        category:    Task category (code, qa, reason, …) for stratified sampling.
        prompt:      The question text sent to the model.
        answer:      Ground-truth answer, or None when only property checks apply.
        answer_type: "multiple_choice" | "exact_match" | None.
        source:      Benchmark name, e.g. "hle", "mmlu", "gsm8k".
        vintage:     Publication year of the source benchmark.
        expected_properties: Substring checks used when answer is None.
    """
    id: str
    category: str
    prompt: str
    answer: str | None = None
    answer_type: str | None = None
    source: str = "unknown"
    vintage: int | None = None
    expected_properties: list[str] = field(default_factory=list)
    max_output_tokens: int | None = None

    @property
    def verifiable(self) -> bool:
        """True when correctness can be checked mechanically.

        Gates best-of-n as an available action: resampling only pays if you can
        tell which sample is right. In the eval harness the answer key makes
        every keyed item verifiable, but the flag is carried explicitly because
        the *production* router only has a verifier where checking is cheap —
        tests for code, schema validity for extraction, arithmetic for math. An
        oracle that assumes a verifier the router will not have at inference
        time reports a bound the router cannot reach.
        """
        return self.answer is not None and self.answer_type in {
            "multiple_choice",
            "exact_match",
        }

    def to_prompt(self) -> BenchmarkPrompt:
        """Adapt to the existing BenchmarkPrompt so runner + grading are reused."""
        return BenchmarkPrompt(
            id=self.id,
            category=self.category,
            prompt=self.prompt,
            max_output_tokens=self.max_output_tokens,
            expected_properties=list(self.expected_properties),
            answer=self.answer,
            answer_type=self.answer_type,
        )

    def grade(self, output: str) -> bool | None:
        """Return correctness, or None when the item has no valid scoring rule."""
        return self.to_prompt().quality_check(output)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "source": self.source,
            "vintage": self.vintage,
            "expected_properties": list(self.expected_properties),
            "max_output_tokens": self.max_output_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalItem":
        return cls(
            id=d["id"],
            category=d.get("category", "unknown"),
            prompt=d["prompt"],
            answer=d.get("answer"),
            answer_type=d.get("answer_type"),
            source=d.get("source", "unknown"),
            vintage=d.get("vintage"),
            expected_properties=list(d.get("expected_properties", [])),
            max_output_tokens=d.get("max_output_tokens"),
        )


def load_corpus(path: Path | str) -> list[EvalItem]:
    """Load a JSONL corpus of EvalItems.

    Raises ValueError on duplicate ids — the id is the grid cell key, so a
    collision would silently merge two different items' outcomes.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval corpus not found: {p}")

    items: list[EvalItem] = []
    seen: set[str] = set()
    with p.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = EvalItem.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{p}:{lineno}: malformed corpus line: {exc}") from exc
            if item.id in seen:
                raise ValueError(f"{p}:{lineno}: duplicate item id {item.id!r}")
            seen.add(item.id)
            items.append(item)
    return items


def write_corpus(items: list[EvalItem], path: Path | str) -> None:
    """Write items as JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        for item in items:
            fh.write(json.dumps(item.to_dict()) + "\n")


def vintages(items: list[EvalItem]) -> dict[int | None, int]:
    """Return {vintage: item_count}, for reporting corpus balance."""
    counts: dict[int | None, int] = {}
    for item in items:
        counts[item.vintage] = counts.get(item.vintage, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0])))
