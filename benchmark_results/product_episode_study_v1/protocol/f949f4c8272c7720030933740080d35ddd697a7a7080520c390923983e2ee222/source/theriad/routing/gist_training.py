"""Dataset preparation utilities for GIST-based prompt-difficulty training.

This module deliberately contains no PyTorch or Sentence Transformers imports.
The live router must remain usable without training dependencies, and the pure
adapters here are unit-testable without downloading a model.

The target is *relative prompt difficulty*, not lexical sophistication.  Each
source supplies an outcome-derived signal:

* RouteLLM rates how likely Mixtral is to handle a prompt relative to GPT-4.
* RouteBalance scores the same prompt across several model sizes.

The raw scales are not comparable.  :func:`calibrate_target_quantiles` maps
each source to its training-split empirical quantile before sources are mixed.
The resulting value is suitable for representation learning and ranking, but
is not a calibrated probability of success.  Production calibration must be
fit later on Theriad's own model/outcome grid.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

BASE_MODEL = "avsolatorio/GIST-small-Embedding-v0"
BASE_MODEL_REVISION = "75e62fd210b9fde790430e0b2f040b0b00a021b1"

ROUTELLM_REPO = "routellm/gpt4_dataset"
ROUTELLM_REVISION = "7ef62d690ff71ce9c21f4226cfc7bbc2ef9e6227"
ROUTELLM_LICENSE = "apache-2.0"

ROUTEBALANCE_REPO = "asdwb/route_balance_model_estimator"
ROUTEBALANCE_REVISION = "48a5a7536b0adaa3635a702660210b17386245aa"
ROUTEBALANCE_LICENSE = "cc-by-4.0"

DEEPEVAL_SCORE = "deepeval-llama3.1-8b-it_reference"
SPLIT_PRIORITY = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True)
class DifficultyExample:
    """One prompt and its source-specific difficulty target in ``[0, 1]``."""

    prompt: str
    target: float
    label_source: str
    split: str
    record_id: str
    provenance: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def normalise_prompt(prompt: Any) -> str:
    """Return a stable single string from common prompt encodings.

    RouteLLM stores ordinary strings while some Arena-derived corpora store a
    JSON list of conversation turns.  Joining those turns is deterministic and
    prevents the JSON brackets themselves becoming a dataset fingerprint.
    """

    if isinstance(prompt, list):
        return "\n\n".join(str(part).strip() for part in prompt if str(part).strip())
    if not isinstance(prompt, str):
        return ""
    text = prompt.strip()
    if text.startswith("["):
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, list) and all(isinstance(part, str) for part in decoded):
            text = "\n\n".join(part.strip() for part in decoded if part.strip())
    return " ".join(text.split())


def prompt_key(prompt: str) -> str:
    return hashlib.sha256(normalise_prompt(prompt).encode("utf-8")).hexdigest()


def stable_fraction(value: str, seed: int = 42) -> float:
    """Stable value in ``[0, 1)`` used for source-local holdout splits."""

    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def route_llm_example(row: dict[str, Any], split: str) -> DifficultyExample | None:
    """Adapt one ``routellm/gpt4_dataset`` row.

    RouteLLM's score is 1--5 and higher means the weak model is more likely to
    perform well.  Difficulty therefore reverses and rescales that score.
    """

    prompt = normalise_prompt(row.get("prompt"))
    score = row.get("mixtral_score")
    if not prompt or isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    score = float(score)
    if not 1.0 <= score <= 5.0:
        return None
    sources = row.get("source") or []
    if isinstance(sources, str):
        sources = [sources]
    record_id = row.get("id") or prompt_key(prompt)[:16]
    return DifficultyExample(
        prompt=prompt,
        target=(5.0 - score) / 4.0,
        label_source="routellm_gpt4_mixtral",
        split=split,
        record_id=str(record_id),
        provenance=f"{ROUTELLM_REPO}:{','.join(map(str, sources)) or 'unknown'}",
    )


def _finite_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


def route_balance_example(
    row: dict[str, Any],
    *,
    config: str,
    split: str,
    include_harmful: bool = False,
) -> DifficultyExample | None:
    """Adapt one RouteBalance record using mean cross-model judge success.

    Truncated generations are ignored.  Safety-source prompts are excluded by
    default because successful refusal and task difficulty are different
    targets; the dataset card explicitly calls out that distinction.
    """

    if row.get("is_harmful") and not include_harmful:
        return None
    prompt = normalise_prompt(row.get("prompt"))
    if not prompt:
        return None
    scores: list[float] = []
    models = row.get("models")
    if not isinstance(models, dict):
        return None
    for result in models.values():
        if not isinstance(result, dict) or result.get("is_truncated"):
            continue
        judges = result.get("llm_judge_scores")
        if not isinstance(judges, dict):
            continue
        score = _finite_score(judges.get(DEEPEVAL_SCORE))
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    record_id = row.get("request_id") or prompt_key(prompt)[:16]
    dataset_name = str(row.get("dataset") or "unknown")
    return DifficultyExample(
        prompt=prompt,
        target=1.0 - sum(scores) / len(scores),
        label_source=f"routebalance_{config}",
        split=split,
        record_id=str(record_id),
        provenance=f"{ROUTEBALANCE_REPO}:{config}:{dataset_name}",
    )


def assign_routebalance_split(
    example: DifficultyExample,
    *,
    original_split: str,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> DifficultyExample:
    """Keep the official test split and carve validation from train by prompt."""

    if original_split == "test":
        return replace(example, split="test")
    fraction = stable_fraction(prompt_key(example.prompt), seed)
    split = "validation" if fraction < validation_fraction else "train"
    return replace(example, split=split)


def calibrate_target_quantiles(
    examples: Iterable[DifficultyExample],
) -> tuple[list[DifficultyExample], dict[str, list[float]]]:
    """Map each source's raw scale through its training empirical CDF.

    Validation and test values use training-only boundaries.  This avoids
    leaking held-out label distributions and prevents the much larger RouteLLM
    corpus from defining the meaning of RouteBalance's judge scale.
    """

    rows = list(examples)
    calibration: dict[str, list[float]] = {}
    for row in rows:
        if row.split == "train":
            calibration.setdefault(row.label_source, []).append(row.target)
    for source, values in calibration.items():
        values.sort()
        if not values:
            raise ValueError(f"no training targets for {source}")

    calibrated: list[DifficultyExample] = []
    for row in rows:
        values = calibration.get(row.label_source)
        if not values:
            raise ValueError(f"no training calibration for {row.label_source}")
        # Mid-rank for ties: repeated discrete RouteLLM scores should not spread
        # across a quantile interval merely because of input order.
        left = bisect.bisect_left(values, row.target)
        right = bisect.bisect_right(values, row.target)
        quantile = ((left + right) / 2.0) / len(values)
        calibrated.append(replace(row, target=min(1.0, max(0.0, quantile))))
    return calibrated, calibration


def deduplicate_examples(examples: Iterable[DifficultyExample]) -> list[DifficultyExample]:
    """Group duplicate prompts and keep them in the highest-priority split.

    If a prompt appears in an official test split anywhere, it is removed from
    training everywhere.  Labels from sources in that winning split are
    averaged; lower-priority labels are discarded rather than leaked.
    """

    grouped: dict[str, list[DifficultyExample]] = {}
    for row in examples:
        grouped.setdefault(prompt_key(row.prompt), []).append(row)

    out: list[DifficultyExample] = []
    for key, rows in grouped.items():
        winning_priority = max(SPLIT_PRIORITY[row.split] for row in rows)
        winners = [row for row in rows if SPLIT_PRIORITY[row.split] == winning_priority]
        first = winners[0]
        out.append(
            DifficultyExample(
                prompt=first.prompt,
                target=sum(row.target for row in winners) / len(winners),
                label_source="+".join(sorted({row.label_source for row in winners})),
                split=first.split,
                record_id=key[:16],
                provenance=" | ".join(sorted({row.provenance for row in winners})),
            )
        )
    return sorted(out, key=lambda row: (SPLIT_PRIORITY[row.split], row.record_id))


def iter_jsonl_lines(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Parse JSONL while tolerating blank lines and rejecting non-objects."""

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        yield value


def iter_utf8_lf_lines(chunks: Iterable[bytes]) -> Iterator[str]:
    """Decode UTF-8 records separated specifically by ASCII line feed.

    ``str.splitlines()`` also splits on Unicode U+2028 and U+2029. Those code
    points are legal inside JSON strings and occur in RouteBalance prompts, so
    a generic HTTP ``iter_lines`` helper corrupts otherwise valid JSONL. Byte
    framing keeps Unicode line separators inside their original record.
    """

    buffer = bytearray()
    for chunk in chunks:
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            if line.endswith(b"\r"):
                line = line[:-1]
            yield line.decode("utf-8")
    if buffer:
        if buffer.endswith(b"\r"):
            buffer = buffer[:-1]
        yield bytes(buffer).decode("utf-8")


def write_jsonl(path: Path, examples: Iterable[DifficultyExample]) -> int:
    """Write prepared examples using a stable schema and return the row count."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
            count += 1
    return count
