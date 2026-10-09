"""Pre-test configuration-freeze manifest for Router V2.

This manifest is deliberately broader than the corpus checksum.  It binds the
model/provider snapshot, adapters, graders, budgets, router configuration, and
source revisions before an untouched test call can be unlocked.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .study import (
    CORE_MODELS,
    FORMAT_TOKEN_BUDGETS,
    GROK_MODEL,
    GROK_SELECTION_ALGORITHM_VERSION,
    corpus_checksum,
)

FREEZE_MANIFEST_SCHEMA_VERSION = 1
REQUIRED_FREEZE_FIELDS = (
    "corpus_and_split_hashes",
    "model_ids_and_provider_identities",
    "price_snapshot",
    "task_adapters_and_tool_capabilities",
    "sampling_settings",
    "token_limits",
    "grader_and_executor_versions",
    "feature_set",
    "router_families",
    "encoder_names_and_revisions",
    "calibration_method",
    "lower_confidence_bound_method",
    "knee_selection_algorithm",
    "grok_selection_algorithm_and_selected_ids",
    "plan_hashes",
    "source_revisions",
)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _split_hashes(rows: Sequence[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        result[split] = canonical_hash(
            [row.to_dict() for row in rows if getattr(row, "split", "") == split]
        )
    return result


def build_pretest_freeze_manifest(
    rows: Sequence[Any],
    *,
    phase_status: Mapping[str, str],
    plan_hashes: Mapping[str, str],
    price_snapshot: Mapping[str, Any],
    task_adapters: Mapping[str, Any],
    sampling_settings: Mapping[str, Any],
    grader_versions: Mapping[str, Any],
    feature_set: Sequence[str],
    router_families: Sequence[str],
    encoders: Mapping[str, Any],
    calibration_method: Mapping[str, Any],
    lower_confidence_bound_method: Mapping[str, Any],
    knee_selection_algorithm: Mapping[str, Any],
    grok_selection: Mapping[str, Any],
    source_revisions: Sequence[str],
    provider_identities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete manifest, marked locked until all gates pass."""

    corpus = corpus_checksum(rows)
    providers = dict(provider_identities or {})
    providers.setdefault("models", [*CORE_MODELS, GROK_MODEL])
    providers.setdefault("provider_identity_status", "pinned_ids_required_before_test")
    selected_ids = list(grok_selection.get("prompt_ids") or [])
    status = "ready_for_test"
    missing_preconditions: list[str] = []
    for phase in ("A_core_train", "B_grok_train", "C_validation"):
        if phase_status.get(phase) != "complete":
            missing_preconditions.append(f"{phase}_complete")
    if not selected_ids:
        missing_preconditions.append("grok_200_selected_ids")
    if missing_preconditions:
        status = "blocked_pending_phased_study"
    payload: dict[str, Any] = {
        "schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "status": status,
        "test_unlock": status == "ready_for_test",
        "missing_preconditions": missing_preconditions,
        "corpus_and_split_hashes": {
            "complete_corpus_sha256": corpus,
            "split_sha256": _split_hashes(rows),
        },
        "model_ids_and_provider_identities": providers,
        "price_snapshot": {
            "values": dict(price_snapshot),
            "sha256": canonical_hash(price_snapshot),
        },
        "task_adapters_and_tool_capabilities": dict(task_adapters),
        "sampling_settings": dict(sampling_settings),
        "token_limits": {"format_budgets": FORMAT_TOKEN_BUDGETS},
        "grader_and_executor_versions": dict(grader_versions),
        "feature_set": list(feature_set),
        "router_families": list(router_families),
        "encoder_names_and_revisions": dict(encoders),
        "calibration_method": dict(calibration_method),
        "lower_confidence_bound_method": dict(lower_confidence_bound_method),
        "knee_selection_algorithm": dict(knee_selection_algorithm),
        "grok_selection_algorithm_and_selected_ids": {
            **dict(grok_selection),
            "algorithm_version": str(
                grok_selection.get("algorithm_version")
                or GROK_SELECTION_ALGORITHM_VERSION
            ),
            "selected_ids": selected_ids,
            "selected_ids_status": str(
                grok_selection.get("selected_ids_status")
                or ("frozen" if selected_ids else "pending")
            ),
        },
        "plan_hashes": dict(plan_hashes),
        "source_revisions": sorted({str(value) for value in source_revisions if str(value)}),
        "phase_status": dict(phase_status),
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_pretest_freeze_manifest(
    manifest: Mapping[str, Any], *, require_test_unlock: bool = False
) -> None:
    if manifest.get("schema_version") != FREEZE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported Router V2 freeze-manifest schema")
    missing = [field for field in REQUIRED_FREEZE_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"freeze manifest is missing fields: {missing}")
    stored = str(manifest.get("manifest_hash") or "")
    if stored != canonical_hash({key: value for key, value in manifest.items() if key != "manifest_hash"}):
        raise ValueError("freeze manifest hash does not match its contents")
    if require_test_unlock and manifest.get("status") != "ready_for_test":
        raise ValueError("freeze manifest does not unlock untouched test")
    if require_test_unlock and manifest.get("test_unlock") is not True:
        raise ValueError("freeze manifest test_unlock is not true")
