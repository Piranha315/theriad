#!/usr/bin/env python3
"""Profile an embedding artifact against Theriad's production resource gates."""
from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.v2 import SentenceTransformerEmbeddingBackend  # noqa: E402


def _rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if platform.system() == "Darwin" else value / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, help="Text file with one prompt per line")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--write-profile", action="store_true")
    args = parser.parse_args()

    raw = json.loads(args.artifact.read_text(encoding="utf-8"))
    model = raw.get("embedding_model")
    if not model:
        raise SystemExit("artifact does not declare embedding_model")
    prompts = (
        [line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.prompts
        else [
            "Summarize this contract clause.",
            "Prove that the estimator is consistent under these assumptions.",
            "Debug the asynchronous retry loop in this Python service.",
            "Return a JSON object containing the requested fields.",
        ]
    )
    before = _rss_mb()
    backend = SentenceTransformerEmbeddingBackend(str(model))
    backend.encode(prompts[0])
    latencies = []
    for index in range(max(10, args.iterations)):
        started = time.perf_counter()
        backend.encode(prompts[index % len(prompts)])
        latencies.append((time.perf_counter() - started) * 1000)
    after = _rss_mb()
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    profile = {
        "embedding_model": model,
        "incremental_rss_mb": max(0.0, after - before),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": p95,
        "iterations": len(latencies),
    }
    profile["passed"] = profile["incremental_rss_mb"] <= 120.0 and p95 <= 20.0
    if args.write_profile:
        if not profile["passed"]:
            raise SystemExit("resource gates failed; artifact was not modified")
        raw["runtime_profile"] = {
            "incremental_rss_mb": profile["incremental_rss_mb"],
            "p50_latency_ms": profile["p50_latency_ms"],
            "p95_latency_ms": profile["p95_latency_ms"],
            "iterations": profile["iterations"],
        }
        args.artifact.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2, sort_keys=True))
    return 0 if profile["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
