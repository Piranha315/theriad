#!/usr/bin/env python3
"""Import public evaluated JSONL logs into Router V2 anchor observations.

This is read-only with respect to providers and the network.  Downloading or
granting access to a gated Hugging Face dataset is an explicit upstream step.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.public_anchor_import import (  # noqa: E402
    EvaluatedLogSpec,
    import_evaluated_jsonl_logs,
    load_browsecomp_queries,
    load_evaluated_log_manifest,
)
from theriad.routing.public_anchors import (  # noqa: E402
    CapabilityProfile,
    audit_anchors,
    derive_anchor_metrics,
    write_anchor_jsonl,
)


def _resolve_specs(
    specs: tuple[EvaluatedLogSpec, ...], manifest: Path
) -> tuple[EvaluatedLogSpec, ...]:
    resolved: list[EvaluatedLogSpec] = []
    for spec in specs:
        path = spec.path if spec.path.is_absolute() else manifest.parent / spec.path
        resolved.append(replace(spec, path=path))
    return tuple(resolved)


def _apply_capability_profile(anchors, profile: CapabilityProfile):
    updated = []
    for anchor in anchors:
        protocols = sorted({observation.protocol_id for observation in anchor.observations})
        metrics_by_protocol = {
            protocol: derive_anchor_metrics(
                anchor.observations,
                capability_profile=profile,
                protocol_id=protocol,
            )
            for protocol in protocols
        }
        metrics = {"metrics_by_protocol": metrics_by_protocol}
        if len(metrics_by_protocol) == 1:
            metrics.update(next(iter(metrics_by_protocol.values())))
        updated.append(replace(anchor, metrics=tuple(sorted(metrics.items()))))
    return tuple(updated)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--rejections", type=Path, required=True)
    parser.add_argument(
        "--queries",
        type=Path,
        help="optional decrypted BrowseComp query JSONL for logs without question fields",
    )
    parser.add_argument("--capability-scores", type=Path)
    parser.add_argument("--capability-source", default="caller-supplied-train-only")
    parser.add_argument("--capability-oof", action="store_true")
    parser.add_argument("--require-outcomes", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    specs = _resolve_specs(load_evaluated_log_manifest(manifest), manifest)
    queries = load_browsecomp_queries(args.queries) if args.queries else None
    anchors, rejected = import_evaluated_jsonl_logs(specs, queries=queries)
    if args.capability_scores:
        raw_scores = json.loads(args.capability_scores.read_text(encoding="utf-8"))
        if not isinstance(raw_scores, dict):
            raise SystemExit("--capability-scores must contain a JSON object")
        profile = CapabilityProfile.from_scores(
            raw_scores,
            estimated_from=args.capability_source,
            out_of_fold=args.capability_oof,
        )
        anchors = _apply_capability_profile(anchors, profile)

    source_refs = [
        {
            "path": str(spec.path),
            "benchmark_family": spec.benchmark_family,
            "model_id": spec.model_id,
            "protocol_id": spec.protocol_id,
            "source_revision": spec.source_revision,
            "provenance_url": spec.provenance_url,
            "license_name": spec.license_name,
            "license_status": spec.license_status,
        }
        for spec in specs
    ]
    audit = audit_anchors(anchors, source_references=source_refs)
    audit_payload = audit.to_dict()
    audit_payload["rejected_count"] = len(rejected)
    audit_payload["rejections"] = list(rejected)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.rejections.parent.mkdir(parents=True, exist_ok=True)
    with args.rejections.open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    if not anchors:
        print(json.dumps(audit_payload, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    write_anchor_jsonl(anchors, args.output)
    print(json.dumps({"output": str(args.output), **audit_payload}, indent=2, sort_keys=True))
    if args.require_outcomes and audit.usable_outcome_count == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
