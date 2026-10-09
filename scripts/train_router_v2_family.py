#!/usr/bin/env python3
"""Train one Router V2 research family into the common artifact contract.

Embedding families consume precomputed vectors.  This script intentionally does
not download an encoder or put PyTorch on the request path.  ``route_llm_pairwise``
and ``routerdc_shared`` are representation adapters: claiming a faithful BERT
or RouterDC reproduction requires the corresponding pinned encoder/data, so the
artifact records the supplied embedding model and remains experimental until
the offline comparison is complete.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.v2_training import (  # noqa: E402
    StudyRow,
    assert_study_row_contract,
    train_embedding_artifact,
    train_surface_artifact,
)

FAMILIES = {
    "surface": {
        "embedding": False,
        "knn": False,
        "include_surface": True,
        "comparison_status": "native_baseline",
        "faithful_reproduction": True,
    },
    "semantic_knn": {
        "embedding": True,
        "knn": True,
        "include_surface": False,
        "comparison_status": "dependency_light_embedding_baseline",
        "faithful_reproduction": False,
    },
    "route_llm_pairwise": {
        "embedding": True,
        "knn": False,
        "include_surface": False,
        "comparison_status": "approximation_outside_faithful_result",
        "faithful_reproduction": False,
        "missing_for_faithful_result": "pinned RouteLLM pairwise data, BERT encoder, and training procedure",
    },
    "routerdc_shared": {
        "embedding": True,
        "knn": False,
        "include_surface": False,
        "comparison_status": "approximation_outside_faithful_result",
        "faithful_reproduction": False,
        "missing_for_faithful_result": "pinned RouterDC contrastive data, encoder, and training procedure",
    },
    "hybrid": {
        "embedding": True,
        "knn": False,
        "include_surface": True,
        "comparison_status": "local_hybrid_baseline",
        "faithful_reproduction": False,
    },
}


def _rows(path: Path) -> list[StudyRow]:
    rows: list[StudyRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(StudyRow.from_dict(json.loads(line)))
            except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SystemExit(f"{path}:{line_number}: {exc}") from exc
    try:
        assert_study_row_contract(rows)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return rows


def _embeddings(path: Path) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                key = str(raw["prompt_id"])
                vector = [float(value) for value in raw["embedding"]]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SystemExit(f"{path}:{line_number}: {exc}") from exc
            if not key or not vector:
                raise SystemExit(f"{path}:{line_number}: prompt_id and embedding are required")
            if key in result:
                raise SystemExit(f"{path}:{line_number}: duplicate embedding for {key}")
            result[key] = vector
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-jsonl", type=Path)
    parser.add_argument("--embedding-model", default="unknown")
    parser.add_argument("--general-fallback", default="openai/gpt-5.6-luna")
    parser.add_argument("--code-fallback", default="openai/gpt-5.6-luna")
    parser.add_argument("--general-model", action="append")
    parser.add_argument("--code-model", action="append")
    parser.add_argument("--runtime-rss-mb", type=float)
    parser.add_argument("--runtime-p95-ms", type=float)
    args = parser.parse_args()

    rows = _rows(args.input)
    spec = FAMILIES[args.family]
    if spec["embedding"] and args.embedding_jsonl is None:
        raise SystemExit(f"--embedding-jsonl is required for {args.family}")
    if spec["embedding"] and args.embedding_model == "unknown":
        raise SystemExit("--embedding-model is required for embedding families")
    profile = None
    if args.runtime_rss_mb is not None or args.runtime_p95_ms is not None:
        if args.runtime_rss_mb is None or args.runtime_p95_ms is None:
            raise SystemExit("runtime RSS and p95 latency must be supplied together")
        profile = {
            "incremental_rss_mb": args.runtime_rss_mb,
            "p95_latency_ms": args.runtime_p95_ms,
        }
    active = {
        "general": args.general_model or [
            "nvidia/nemotron-3-super-120b-a12b:free",
            "xiaomi/mimo-v2.5",
            "openai/gpt-5.6-luna",
            "x-ai/grok-4.6",
        ],
        "code": args.code_model or [
            "nvidia/nemotron-3-super-120b-a12b:free",
            "xiaomi/mimo-v2.5",
            "openai/gpt-5.6-luna",
            "x-ai/grok-4.6",
        ],
    }
    if args.family == "surface":
        artifact = train_surface_artifact(
            rows,
            fallback_models={"general": args.general_fallback, "code": args.code_fallback},
            active_models_by_lane=active,
        )
        artifact["family"] = "surface_logistic"
    else:
        artifact = train_embedding_artifact(
            rows,
            _embeddings(args.embedding_jsonl),
            fallback_models={"general": args.general_fallback, "code": args.code_fallback},
            active_models_by_lane=active,
            family=args.family,
            embedding_model=args.embedding_model,
            include_surface=bool(spec["include_surface"]),
            runtime_profile=profile,
            knn=bool(spec["knn"]),
        )
    artifact["family_metadata"] = {
        key: value
        for key, value in spec.items()
        if key in {"comparison_status", "faithful_reproduction", "missing_for_faithful_result"}
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "family": artifact["family"],
                "router_version": artifact["router_version"],
                "rows": len(rows),
                "embedding_model": artifact.get("embedding_model"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
