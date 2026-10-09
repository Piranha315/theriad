"""
Convert a sample of HLE (Humanity's Last Exam) to benchmark_prompts.jsonl format.

HLE (https://huggingface.co/datasets/cais/hle) contains ~3,000 expert-level questions
across many academic domains with multiple-choice and exact-match answer types.

The dataset requires a HuggingFace account and accepting its terms:
  1. Visit https://huggingface.co/datasets/cais/hle and click "Access repository"
  2. Get a read token from https://huggingface.co/settings/tokens
  3. Pass it via --hf-token or set the HF_TOKEN environment variable

Usage:
    python3 -m theriad.benchmark.hle_converter [options]

Options:
    --sample N        Questions to include (default: 50)
    --output PATH     Output JSONL (default: hle_benchmark_prompts.jsonl)
    --split SPLIT     Dataset split (default: test)
    --seed SEED       Random seed for sampling (default: 42)
    --all-types       Include image-based questions (default: text-only)
    --category CAT    Filter to one HLE category (e.g. Mathematics)
    --hf-token TOKEN  HuggingFace read token (or set HF_TOKEN env var)

After converting, run the benchmark with:
    python3 -m theriad.benchmark.run --prompts hle_benchmark_prompts.jsonl --preset smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

_HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=cais%2Fhle&config=default&split={split}&offset={offset}&limit={limit}"
)
_HF_INFO_URL = (
    "https://datasets-server.huggingface.co/info"
    "?dataset=cais%2Fhle&config=default"
)
_PAGE_SIZE = 100
_REQUEST_TIMEOUT = 30


# Prompt template — tells the model how to answer both question types.
_MCQ_INSTRUCTION = (
    "Answer the following multiple-choice question. "
    "Respond with ONLY the letter of the correct answer (A, B, C, D, or E). "
    "Do not explain your reasoning.\n\n"
)
_EXACT_INSTRUCTION = (
    "Answer the following question with a short, precise response. "
    "Give only the answer itself — no explanation.\n\n"
)

# No per-type output token cap — let each model use its full output capacity.
# HLE questions require chain-of-thought reasoning before the final answer;
# any fixed cap risks hitting finish_reason=length on reasoning models.
# The letter-extraction regex finds the answer letter anywhere in the output.
_MCQ_MAX_TOKENS = None
_EXACT_MAX_TOKENS = None


def _auth_headers(hf_token: str | None) -> dict:
    if hf_token:
        return {"Authorization": f"Bearer {hf_token}"}
    return {}


def _fetch_total_rows(split: str, hf_token: str | None = None) -> int:
    """Return total row count from the HF dataset info endpoint."""
    req = urllib.request.Request(_HF_INFO_URL, headers=_auth_headers(hf_token))
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        splits = data.get("dataset_info", {}).get("splits", {})
        return int(splits.get(split, {}).get("num_examples", 0))
    except Exception:
        return 0


def _fetch_page(split: str, offset: int, limit: int, hf_token: str | None = None) -> list[dict]:
    """Fetch one page of rows from the HF dataset viewer API."""
    url = _HF_ROWS_URL.format(split=split, offset=offset, limit=limit)
    req = urllib.request.Request(url, headers=_auth_headers(hf_token))
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [entry["row"] for entry in data.get("rows", [])]
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError(
                f"HTTP 401 at offset={offset}: HuggingFace requires authentication.\n"
                "  1. Accept terms at https://huggingface.co/datasets/cais/hle\n"
                "  2. Get a token from https://huggingface.co/settings/tokens\n"
                "  3. Re-run with: --hf-token <your_token>"
            ) from exc
        raise RuntimeError(
            f"HuggingFace API returned HTTP {exc.code} for offset={offset}."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch HLE rows at offset={offset}: {exc}") from exc


def _is_image_question(row: dict) -> bool:
    """Return True if this row requires image understanding."""
    img = row.get("image")
    if img is None:
        return False
    # The HF viewer may return image as a dict with 'src', a base64 string, or null.
    if isinstance(img, dict):
        return bool(img.get("src") or img.get("bytes"))
    return bool(img)


def _row_to_prompt(row: dict, index: int) -> dict:
    """Convert one HLE row to a benchmark_prompts.jsonl record."""
    answer_type = (row.get("answer_type") or "exact_match").strip().lower()
    # Normalise to canonical types
    if answer_type in ("multiple_choice", "mc", "multiplechoice"):
        answer_type = "multiple_choice"
    else:
        answer_type = "exact_match"

    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    raw_category = (row.get("category") or "general").strip()
    # Lowercase and replace spaces so category is a clean slug (e.g. "Mathematics" → "mathematics")
    category = raw_category.lower().replace(" ", "_")

    if answer_type == "multiple_choice":
        prompt = _MCQ_INSTRUCTION + question
        max_tokens = _MCQ_MAX_TOKENS
    else:
        prompt = _EXACT_INSTRUCTION + question
        max_tokens = _EXACT_MAX_TOKENS

    record: dict = {
        "id": f"hle_{index:04d}",
        "category": category,
        "prompt": prompt,
        "expected_properties": [],   # graded via answer/answer_type instead
        "answer": answer,
        "answer_type": answer_type,
    }
    if max_tokens is not None:
        record["max_output_tokens"] = max_tokens
    return record


def fetch_sample(
    *,
    sample: int,
    split: str = "test",
    text_only: bool = True,
    category_filter: str | None = None,
    seed: int = 42,
    hf_token: str | None = None,
) -> list[dict]:
    """
    Download HLE rows and return `sample` converted benchmark prompt dicts.

    Downloads pages sequentially until enough qualifying rows are collected.
    """
    rng = random.Random(seed)

    total = _fetch_total_rows(split, hf_token)
    if total == 0:
        # Fall back to a large assumed size if info endpoint fails
        total = 3000

    # Build a shuffled list of page offsets to visit
    page_offsets = list(range(0, total, _PAGE_SIZE))
    rng.shuffle(page_offsets)

    collected: list[dict] = []
    for offset in page_offsets:
        if len(collected) >= sample:
            break
        try:
            rows = _fetch_page(split, offset, _PAGE_SIZE, hf_token)
        except RuntimeError as exc:
            # Propagate auth errors immediately; warn and continue for others
            if "401" in str(exc):
                raise
            print(f"  Warning: {exc}", file=sys.stderr)
            continue

        # Shuffle within the page for randomness
        rng.shuffle(rows)

        for row in rows:
            if len(collected) >= sample:
                break
            if text_only and _is_image_question(row):
                continue
            if category_filter:
                cat = (row.get("category") or "").strip().lower()
                if cat != category_filter.strip().lower():
                    continue
            if not (row.get("question") or "").strip():
                continue
            if not (row.get("answer") or "").strip():
                continue
            collected.append(_row_to_prompt(row, len(collected)))

    return collected


def write_prompts(prompts: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python3 -m theriad.benchmark.hle_converter",
        description="Download HLE questions and convert to benchmark_prompts.jsonl.",
    )
    p.add_argument("--sample", "-n", type=int, default=50,
                   help="Number of questions to include (default: 50)")
    p.add_argument("--output", "-o", type=Path,
                   default=Path("hle_benchmark_prompts.jsonl"),
                   help="Output JSONL path (default: hle_benchmark_prompts.jsonl)")
    p.add_argument("--split", default="test",
                   help="Dataset split (default: test)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling (default: 42)")
    p.add_argument("--all-types", action="store_true", dest="all_types",
                   help="Include image-based questions (skipped by default)")
    p.add_argument("--category", default=None,
                   help="Filter to one HLE category (e.g. Mathematics)")
    p.add_argument("--hf-token", default=None, dest="hf_token",
                   help="HuggingFace read token (or set HF_TOKEN env var)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or None

    print(f"Fetching {args.sample} HLE questions from split={args.split!r} …")
    if not args.all_types:
        print("  (text-only; use --all-types to include image questions)")
    if args.category:
        print(f"  category filter: {args.category!r}")
    if not hf_token:
        print(
            "  Note: no HF_TOKEN set. This dataset requires authentication.\n"
            "  If you get a 401, visit https://huggingface.co/datasets/cais/hle\n"
            "  and re-run with --hf-token <your_token>",
            file=sys.stderr,
        )

    try:
        prompts = fetch_sample(
            sample=args.sample,
            split=args.split,
            text_only=not args.all_types,
            category_filter=args.category,
            seed=args.seed,
            hf_token=hf_token,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not prompts:
        print("ERROR: No qualifying questions found. Check connectivity or filters.", file=sys.stderr)
        return 1

    write_prompts(prompts, args.output)

    mcq = sum(1 for p in prompts if p.get("answer_type") == "multiple_choice")
    exact = len(prompts) - mcq
    categories = sorted({p["category"] for p in prompts})

    print(f"\nWrote {len(prompts)} questions → {args.output}")
    print(f"  multiple_choice: {mcq}  exact_match: {exact}")
    print(f"  categories: {', '.join(categories)}")
    print(f"\nRun benchmark:")
    print(f"  python3 -m theriad.benchmark.run --prompts {args.output} --preset smoke --yes")
    print(f"  python3 -m theriad.benchmark.run --prompts {args.output} --preset overnight_safe --budget 2.00")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
