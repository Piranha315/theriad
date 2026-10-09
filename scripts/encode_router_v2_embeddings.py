#!/usr/bin/env python3
"""Encode a frozen Router V2 corpus into a reusable local JSONL vector file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.study import load_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="allow Sentence Transformers to fetch a missing model revision",
    )
    args = parser.parse_args()
    if not args.allow_model_download:
        raise SystemExit("refusing a model download; pass --allow-model-download explicitly")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("sentence-transformers is required for embedding generation") from exc

    rows = load_jsonl(args.corpus)
    encoder = SentenceTransformer(args.model)
    vectors = encoder.encode(
        [row.prompt for row in rows],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row, vector in zip(rows, vectors, strict=True):
            handle.write(
                json.dumps(
                    {"prompt_id": row.prompt_id, "embedding": [float(value) for value in vector]},
                    separators=(",", ":"),
                )
                + "\n"
            )
    print(json.dumps({"output": str(args.output), "rows": len(rows), "model": args.model}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
