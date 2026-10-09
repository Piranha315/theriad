#!/usr/bin/env python3
"""Audit public BrowseComp-Plus artifacts without downloading payloads by default."""
from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

GITHUB_REPO = "texttron/BrowseComp-Plus"
HF_BENCHMARK = "Tevatron/browsecomp-plus"
HF_CORPUS = "Tevatron/browsecomp-plus-corpus"
HF_RUNS = "Tevatron/browsecomp-plus-runs"
HF_RESULTS = "Tevatron/BrowseComp-Plus-results"
HF_THIRD_PARTY_LOGS = "ZhuofengLi/bcp-eval-logs"


def _get_json(url: str) -> dict[str, Any] | list[Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "theriad-router-v2-audit/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        value = json.load(response)
    if not isinstance(value, dict | list):
        raise ValueError(f"expected JSON object/list from {url}")
    return value


def _get_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "theriad-router-v2-audit/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def _count_remote_jsonl_rows(url: str) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "theriad-router-v2-audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return sum(1 for line in response if line.strip())


def _hf_info(repo_id: str) -> dict[str, Any]:
    return dict(_get_json(f"https://huggingface.co/api/datasets/{repo_id}"))


def _hf_tree(repo_id: str) -> list[dict[str, Any]]:
    value = _get_json(
        f"https://huggingface.co/api/datasets/{repo_id}/tree/main?recursive=true&expand=false"
    )
    return [dict(row) for row in value] if isinstance(value, list) else []


def _github_info() -> dict[str, Any]:
    return dict(_get_json(f"https://api.github.com/repos/{GITHUB_REPO}"))


def _dataset_summary(info: dict[str, Any]) -> dict[str, Any]:
    card = info.get("cardData") if isinstance(info.get("cardData"), dict) else {}
    dataset_info = card.get("dataset_info") if isinstance(card.get("dataset_info"), dict) else {}
    splits = dataset_info.get("splits") if isinstance(dataset_info.get("splits"), list) else []
    return {
        "id": info.get("id"),
        "sha": info.get("sha"),
        "last_modified": info.get("lastModified"),
        "private": info.get("private"),
        "gated": info.get("gated"),
        "license": card.get("license"),
        "features": [feature.get("name") for feature in dataset_info.get("features", []) if isinstance(feature, dict)],
        "splits": [
            {"name": split.get("name"), "num_examples": split.get("num_examples")}
            for split in splits
            if isinstance(split, dict)
        ],
        "download_size_bytes": dataset_info.get("download_size"),
        "dataset_size_bytes": dataset_info.get("dataset_size"),
    }


def _aggregate_results() -> dict[str, Any]:
    url = f"https://huggingface.co/datasets/{HF_RESULTS}/resolve/main/agent_results.csv"
    records = list(csv.DictReader(io.StringIO(_get_text(url))))
    return {
        "source": url,
        "row_count": len(records),
        "models": sorted({str(row.get("LLM") or "") for row in records if row.get("LLM")}),
        "retrievers": sorted({str(row.get("Retriever") or "") for row in records if row.get("Retriever")}),
        "evaluation_dates": sorted({str(row.get("Evaluation Date") or "") for row in records if row.get("Evaluation Date")}),
        "item_level_outcomes": False,
        "note": "aggregate leaderboard rows; not prompt x model correctness records",
    }


def _run_summary(info: dict[str, Any], tree: list[dict[str, Any]], *, count_rows: bool) -> dict[str, Any]:
    files = [
        {
            "path": str(row.get("path")),
            "size_bytes": row.get("size"),
            "is_trajectory_file": str(row.get("path", "")).endswith(".jsonl"),
        }
        for row in tree
        if str(row.get("path", "")).endswith(".jsonl")
    ]
    if count_rows:
        for file in files:
            url = f"https://huggingface.co/datasets/{HF_RUNS}/resolve/main/{file['path']}"
            file["row_count"] = _count_remote_jsonl_rows(url)
    return {
        "id": info.get("id"),
        "sha": info.get("sha"),
        "last_modified": info.get("lastModified"),
        "gated": info.get("gated"),
        "private": info.get("private"),
        "files": files,
        "direct_item_level_correctness": False,
        "outcome_status": "missing_from_run_rows",
        "requires_grading_or_public_per_query_evaluation": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--count-run-rows",
        action="store_true",
        help="stream large public run files to count JSONL rows; may download >1 GB",
    )
    args = parser.parse_args()

    github = _github_info()
    benchmark = _hf_info(HF_BENCHMARK)
    corpus = _hf_info(HF_CORPUS)
    runs = _hf_info(HF_RUNS)
    results = _hf_info(HF_RESULTS)
    third_party = _hf_info(HF_THIRD_PARTY_LOGS)
    report = {
        "audit_version": "browsecomp-plus-public-artifacts-v1",
        "retrieved_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "official_repository": {
            "full_name": github.get("full_name"),
            "html_url": github.get("html_url"),
            "default_branch": github.get("default_branch"),
            "license": (github.get("license") or {}).get("spdx_id") if isinstance(github.get("license"), dict) else None,
            "pushed_at": github.get("pushed_at"),
        },
        "official_benchmark_dataset": _dataset_summary(benchmark),
        "official_corpus_dataset": _dataset_summary(corpus),
        "official_run_dataset": _run_summary(runs, _hf_tree(HF_RUNS), count_rows=args.count_run_rows),
        "official_aggregate_results": {
            "dataset": {
                "id": results.get("id"),
                "sha": results.get("sha"),
                "last_modified": results.get("lastModified"),
                "gated": results.get("gated"),
                "private": results.get("private"),
            },
            "agent_results": _aggregate_results(),
        },
        "third_party_item_logs": {
            "id": third_party.get("id"),
            "sha": third_party.get("sha"),
            "gated": third_party.get("gated"),
            "private": third_party.get("private"),
            "status": "manual_access_required",
            "note": "not treated as an official or usable source without access and schema audit",
        },
        "usable_direct_item_level_outcomes": 0,
        "provider_calls_made": 0,
        "production_changes_made": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
