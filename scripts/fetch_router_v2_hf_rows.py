#!/usr/bin/env python3
"""Fetch pinned, public Hugging Face rows into an explicit raw cache.

This is a read-only source-preparation helper for the Router V2 corpus. It
uses the Hugging Face dataset-server rows API, pins every source revision in
the manifest, and writes raw rows only below the caller-provided cache
directory. It never reads credentials or calls an inference provider.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

USER_AGENT = "theriad-router-v2-source-cache/1.0"
PAGE_SIZE = 100

SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "mmlu": {
        "dataset": "cais/mmlu",
        "config": "all",
        "split": "test",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "license_name": "MIT",
        "license_status": "public",
        "last_modified": "2024-03-08",
        "provenance_url": "https://huggingface.co/datasets/cais/mmlu",
    },
    "mmlu-pro": {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "config": "default",
        "split": "test",
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "license_name": "MIT",
        "license_status": "public",
        "last_modified": "2026-05-02",
        "provenance_url": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
    },
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "license_name": "MIT",
        "license_status": "public",
        "last_modified": "2026-03-23",
        "provenance_url": "https://huggingface.co/datasets/openai/gsm8k",
    },
    "arc-easy": {
        "dataset": "allenai/ai2_arc",
        "config": "ARC-Easy",
        "split": "test",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "license_name": "CC-BY-SA-4.0",
        "license_status": "public",
        "last_modified": "2023-12-21",
        "provenance_url": "https://huggingface.co/datasets/allenai/ai2_arc",
    },
    "routellm": {
        "dataset": "routellm/gpt4_dataset",
        "config": "default",
        "split": "train",
        "revision": "7ef62d690ff71ce9c21f4226cfc7bbc2ef9e6227",
        "license_name": "Apache-2.0",
        "license_status": "public",
        "last_modified": "2025-04-01",
        "provenance_url": "https://huggingface.co/datasets/routellm/gpt4_dataset",
    },
}


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _fetch_page(spec: dict[str, Any], offset: int, length: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": spec["dataset"],
            "config": spec["config"],
            "split": spec["split"],
            "offset": offset,
            "length": length,
            "revision": spec["revision"],
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        error = payload.get("error") if isinstance(payload, dict) else None
        raise ValueError(f"dataset-server returned no rows for {spec['dataset']}: {error}")
    normalized: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
            raise ValueError(f"dataset-server returned a malformed row for {spec['dataset']}")
        normalized.append(item["row"])
    return normalized


def _parse_request(value: str) -> tuple[str, int]:
    name, separator, count = value.partition("=")
    if not separator or name not in SOURCE_SPECS:
        raise argparse.ArgumentTypeError(
            f"source must be NAME=COUNT where NAME is one of {sorted(SOURCE_SPECS)}"
        )
    try:
        parsed = int(count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("source count must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("source count must be positive")
    return name, parsed


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        type=_parse_request,
        required=True,
        help="pinned source and row count, for example mmlu=900",
    )
    args = parser.parse_args()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "api": "https://datasets-server.huggingface.co/rows",
        "page_size": PAGE_SIZE,
        "sources": {},
    }
    for name, requested_count in args.source:
        spec = dict(SOURCE_SPECS[name])
        output = args.cache_dir / f"{name}.jsonl"
        rows: list[dict[str, Any]] = []
        while len(rows) < requested_count:
            page = _fetch_page(spec, len(rows), min(PAGE_SIZE, requested_count - len(rows)))
            if not page:
                raise RuntimeError(
                    f"dataset-server ended before {requested_count} rows for {name}"
                )
            rows.extend(page)
        _write_jsonl(output, rows[:requested_count])
        digest, size = _sha256(output)
        manifest["sources"][name] = {
            **spec,
            "requested_count": requested_count,
            "accepted_raw_rows": len(rows[:requested_count]),
            "raw_path": str(output),
            "sha256": digest,
            "size_bytes": size,
        }

    manifest_path = args.cache_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
