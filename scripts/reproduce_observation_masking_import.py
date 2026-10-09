#!/usr/bin/env python3
"""Reproduce the pinned i-DeepSearch evaluated-log import.

The command downloads only ``evaluated.jsonl`` files into an explicit cache,
verifies every file hash recorded in the manifest, and then invokes the same
item-level importer used by the local public-anchor workflow.  It never reads
credentials or calls a model/provider.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.public_anchor_import import (  # noqa: E402
    EvaluatedLogSpec,
    import_evaluated_jsonl_logs,
)
from theriad.routing.public_anchors import audit_anchors, write_anchor_jsonl  # noqa: E402

USER_AGENT = "theriad-router-v2-public-anchor-reproducer/1.0"
DATASET_ID = "i-DeepSearch/observation-masking-eval-logs"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
SOURCE_LICENSE = "Apache-2.0"
SOURCE_LICENSE_STATUS = "public"
PROVIDER_SENTINEL = "not_published_in_dataset"
VINTAGE_SENTINEL = "not_published_in_dataset"


def _get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _remote_files(dataset_id: str, revision: str) -> list[dict[str, Any]]:
    encoded_revision = urllib.parse.quote(revision, safe="")
    tree_url = (
        f"https://huggingface.co/api/datasets/{dataset_id}/tree/"
        f"{encoded_revision}?recursive=true&expand=false"
    )
    rows = _get_json(tree_url)
    if not isinstance(rows, list):
        raise ValueError("Hugging Face tree response was not a list")
    files: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        path = str(row.get("path") or "")
        if not path.endswith("evaluated.jsonl"):
            continue
        encoded_path = urllib.parse.quote(path, safe="/")
        files.append(
            {
                "path": path,
                "oid": str(row.get("oid") or ""),
                "size_bytes": int(row["size"]) if row.get("size") is not None else None,
                "remote_url": (
                    f"https://huggingface.co/datasets/{dataset_id}/resolve/"
                    f"{encoded_revision}/{encoded_path}"
                ),
            }
        )
    if not files:
        raise ValueError("the pinned revision has no evaluated.jsonl files")
    return sorted(files, key=lambda row: row["path"])


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _download_file(entry: dict[str, Any], cache_root: Path, revision: str) -> dict[str, Any]:
    relative = Path("files") / revision / entry["path"]
    target = cache_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = str(entry.get("sha256") or "")
    if target.is_file():
        actual_hash, actual_size = _sha256(target)
        if expected_hash and actual_hash != expected_hash:
            target.unlink()
        elif entry.get("size_bytes") is not None and actual_size != int(entry["size_bytes"]):
            target.unlink()
        else:
            return {**entry, "cache_relative_path": str(relative), "sha256": actual_hash}

    temporary = target.with_name(target.name + ".partial")
    request = urllib.request.Request(str(entry["remote_url"]), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
            size = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                size += len(chunk)
        actual_hash, actual_size = _sha256(temporary)
        expected_size = entry.get("size_bytes")
        if expected_size is not None and actual_size != int(expected_size):
            raise ValueError(
                f"size mismatch for {entry['path']}: expected {expected_size}, got {actual_size}"
            )
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {entry['path']}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {**entry, "cache_relative_path": str(relative), "sha256": actual_hash}


def _protocol_metadata(
    benchmark: str, directory: str
) -> tuple[str, str, str, str, str]:
    config = directory.rsplit("/", 1)[-1]
    context_masking = "no-CM"
    for suffix in ("-CM16", "-CM8", "-CM2", "-CM1", "-SUM50", "-CM"):
        if config.endswith(suffix):
            context_masking = suffix[1:]
            config = config[: -len(suffix)]
            break
    base = config
    retrieval = "not_published_in_dataset"
    model_id = base
    if benchmark == "BrowseComp-Plus":
        for suffix in ("Qwen3-Emb-8B", "AgentIR", "BM25"):
            marker = f"-{suffix}"
            if base.endswith(marker):
                retrieval = suffix
                model_id = base[: -len(marker)]
                break
    scaffold = "ablation" if directory.startswith("ablation/") else "standard"
    protocol_id = f"{benchmark}::{scaffold}::{retrieval}::{context_masking}"
    return model_id, retrieval, context_masking, protocol_id, scaffold


def _spec(entry: dict[str, Any], cache_root: Path, revision: str) -> EvaluatedLogSpec:
    parts = str(entry["path"]).split("/")
    if len(parts) < 3 or parts[-1] != "evaluated.jsonl":
        raise ValueError(f"unexpected evaluated-log path: {entry['path']}")
    benchmark = parts[0]
    directory = "/".join(parts[1:-1])
    model_id, retrieval, context_masking, protocol_id, scaffold = _protocol_metadata(
        benchmark, directory
    )
    return EvaluatedLogSpec(
        path=cache_root / entry["cache_relative_path"],
        benchmark_family=benchmark,
        model_id=model_id,
        provider=PROVIDER_SENTINEL,
        model_family=model_id,
        protocol_id=protocol_id,
        source_revision=f"{DATASET_ID}@{revision}",
        provenance_url=DATASET_URL,
        license_name=SOURCE_LICENSE,
        license_status=SOURCE_LICENSE_STATUS,
        evaluation_method="explicit_record_correct",
        domain="research",
        collection_vintage=VINTAGE_SENTINEL,
        retrieval_config=retrieval,
        context_masking=context_masking,
        prompting_protocol="dataset_evaluated_jsonl_v1",
        tool_availability="web_agent",
        agent_scaffold=scaffold,
        split="test",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--rejections", type=Path, required=True)
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="refresh the evaluated-file list at the manifest's pinned revision",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit("manifest root must be an object")
    if manifest.get("dataset_id") != DATASET_ID:
        raise SystemExit("manifest dataset_id is not the pinned i-DeepSearch dataset")
    revision = str(manifest.get("revision") or "")
    if not revision:
        raise SystemExit("manifest revision is required")
    if manifest.get("license_name") != SOURCE_LICENSE or manifest.get("license_status") != SOURCE_LICENSE_STATUS:
        raise SystemExit("manifest license metadata does not match the published dataset card")

    entries = manifest.get("files")
    if args.refresh_manifest or not isinstance(entries, list) or not entries:
        entries = _remote_files(DATASET_ID, revision)
        manifest["files"] = entries
        manifest["manifest_status"] = "file_list_refreshed"

    if not isinstance(entries, list) or not entries:
        raise SystemExit("manifest contains no evaluated files")
    downloaded: list[dict[str, Any]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict) or not raw_entry.get("path") or not raw_entry.get("remote_url"):
            raise SystemExit("manifest contains an incomplete file entry")
        downloaded.append(_download_file(raw_entry, args.cache_dir, revision))

    manifest["files"] = downloaded
    manifest["file_count"] = len(downloaded)
    if args.refresh_manifest or not manifest.get("generated_at"):
        manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["metadata_disclosure"] = {
        "provider": "The source publishes model labels but not serving-provider identity; provider uses an explicit sentinel.",
        "collection_vintage": "The source publishes no per-run collection date; collection_vintage uses an explicit sentinel.",
        "protocols": "Each benchmark/model/retriever/CM directory is a separate protocol; outcomes are not pooled across protocols.",
    }
    _write_manifest(args.manifest, manifest)
    manifest_hash, _ = _sha256(args.manifest)

    specs = tuple(_spec(entry, args.cache_dir, revision) for entry in downloaded)
    anchors, rejected = import_evaluated_jsonl_logs(specs)
    source_refs = []
    for entry in downloaded:
        spec = _spec(entry, args.cache_dir, revision)
        source_refs.append(
            {
                "dataset_id": DATASET_ID,
                "revision": revision,
                "path": entry["path"],
                "remote_url": entry["remote_url"],
                "cache_path": str(args.cache_dir / entry["cache_relative_path"]),
                "oid": entry.get("oid"),
                "size_bytes": entry.get("size_bytes"),
                "sha256": entry.get("sha256"),
                "benchmark_family": spec.benchmark_family,
                "model_id": spec.model_id,
                "provider": spec.provider,
                "model_family": spec.model_family,
                "protocol_id": spec.protocol_id,
                "retrieval_config": spec.retrieval_config,
                "context_masking": spec.context_masking,
                "agent_scaffold": spec.agent_scaffold,
                "prompting_protocol": spec.prompting_protocol,
                "tool_availability": spec.tool_availability,
                "license_name": spec.license_name,
                "license_status": spec.license_status,
            }
        )
    audit = audit_anchors(anchors, source_references=source_refs).to_dict()
    observations = [observation for anchor in anchors for observation in anchor.observations]
    audit.update(
        {
            "dataset_id": DATASET_ID,
            "dataset_revision": revision,
            "provenance_url": DATASET_URL,
            "license_name": SOURCE_LICENSE,
            "license_status": SOURCE_LICENSE_STATUS,
            "manifest_path": str(args.manifest),
            "manifest_sha256": manifest_hash,
            "file_count": len(downloaded),
            "file_hashes": [
                {
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                    "size_bytes": entry["size_bytes"],
                }
                for entry in downloaded
            ],
            "accepted_anchor_count": len(anchors),
            "accepted_observation_count": len(observations),
            "rejected_record_count": len(rejected),
            "explicit_outcome_count": sum(item.outcome is not None for item in observations),
            "model_count": len({item.model_id for item in observations}),
            "protocol_count": len({item.protocol_id for item in observations}),
            "rejections": list(rejected),
            "raw_data": "cache_only; not redistributed by this command",
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_anchor_jsonl(anchors, args.output)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.rejections.parent.mkdir(parents=True, exist_ok=True)
    with args.rejections.open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(args.manifest), "audit": audit}, indent=2, sort_keys=True))
    return 0 if anchors and audit["explicit_outcome_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
