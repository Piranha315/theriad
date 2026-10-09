#!/usr/bin/env python3
"""Assemble the frozen Router V2 corpus without provider calls.

The quotas in this file are part of the preregistered study input. They are
expressed as source/format/split allocations so a passing row count cannot
hide a one-lane or one-format corpus. Every row is copied from a pinned local
source; this command never creates paraphrases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.study import (  # noqa: E402
    GROK_SELECTION_ALGORITHM_VERSION,
    OOD_POLICY_VERSION,
    StudyPrompt,
    audit_corpus,
    corpus_checksum,
    write_jsonl,
)

BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
TAU3_REVISION = "a2c024725189473d2d7cea3a5cfdbcc67478e41f"
ANCHOR_REVISION = "19342195a5c5c9f8ae016cac6a388d937371df8e"
ANCHOR_SOURCE = "i-DeepSearch/observation-masking-eval-logs"
ANCHOR_URL = "https://huggingface.co/datasets/i-DeepSearch/observation-masking-eval-logs"


SOURCE_METADATA: dict[str, dict[str, str]] = {
    "cais/mmlu": {
        "license": "MIT",
        "revision": "cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe",
        "url": "https://huggingface.co/datasets/cais/mmlu",
        "vintage": "2024-03-08",
    },
    "TIGER-Lab/MMLU-Pro": {
        "license": "MIT",
        "revision": "TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "url": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
        "vintage": "2026-05-02",
    },
    "openai/gsm8k": {
        "license": "MIT",
        "revision": "openai/gsm8k@740312add88f781978c0658806c59bc2815b9866",
        "url": "https://huggingface.co/datasets/openai/gsm8k",
        "vintage": "2026-03-23",
    },
    "allenai/ai2_arc": {
        "license": "CC-BY-SA-4.0",
        "revision": "allenai/ai2_arc@210d026faf9955653af8916fad021475a3f00453",
        "url": "https://huggingface.co/datasets/allenai/ai2_arc",
        "vintage": "2023-12-21",
    },
    "routellm/gpt4_dataset": {
        "license": "Apache-2.0",
        "revision": "routellm/gpt4_dataset@7ef62d690ff71ce9c21f4226cfc7bbc2ef9e6227",
        "url": "https://huggingface.co/datasets/routellm/gpt4_dataset",
        "vintage": "2025-04-01",
    },
}


def _stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        return [], [{"source": str(path), "reason": f"{type(exc).__name__}: {exc}"}]
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected.append({"source": str(path), "line": line_number, "reason": str(exc)})
                continue
            if not isinstance(value, dict):
                rejected.append({"source": str(path), "line": line_number, "reason": "row_not_object"})
                continue
            rows.append(value)
    return rows, rejected


def _read_json_array(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [{"source": str(path), "reason": f"{type(exc).__name__}: {exc}"}]
    if not isinstance(value, list):
        return [], [{"source": str(path), "reason": "json_payload_not_array"}]
    rows = [item for item in value if isinstance(item, dict)]
    rejected = (
        [{"source": str(path), "reason": "non_object_task_row"}]
        if len(rows) != len(value)
        else []
    )
    return rows, rejected


def _prompt_text(value: Any) -> str:
    """Flatten benchmark chat messages without changing source wording."""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, list):
                parts.append(_prompt_text(item))
            elif isinstance(item, dict):
                content = item.get("content")
                if content is not None:
                    parts.append(str(content).strip())
            elif item is not None:
                parts.append(str(item).strip())
        return "\n\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value or "").strip()


def _choices_prompt(question: str, choices: list[Any]) -> str:
    lines = [question.strip(), "", "Options:"]
    lines.extend(f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(choices))
    lines.append("\nAnswer with the option letter.")
    return "\n".join(lines)


def _choice_letter(value: Any, choices: list[Any]) -> str:
    if isinstance(value, int) and 0 <= value < len(choices):
        return chr(ord("A") + value)
    text = str(value or "").strip()
    if len(text) == 1 and text.upper().isalpha():
        return text.upper()
    if text.isdigit() and 0 <= int(text) < len(choices):
        return chr(ord("A") + int(text))
    raise ValueError(f"cannot normalize multiple-choice answer {value!r}")


def _metadata(source: str, *, domain: str, benchmark: str) -> dict[str, str]:
    source_meta = SOURCE_METADATA[source]
    return {
        "source": source,
        "benchmark_family": benchmark,
        "domain": domain,
        "collection_vintage": source_meta["vintage"],
        "license_name": source_meta["license"],
        "license_status": "public",
        "source_revision": source_meta["revision"],
        "provenance_url": source_meta["url"],
    }


def _row(
    *,
    prompt_id: str,
    prompt: str,
    source: str,
    benchmark: str,
    domain: str,
    template: str,
    lane: str,
    label_type: str,
    task_format: str,
    grader: dict[str, Any],
    tags: tuple[str, ...] = (),
    protocol_id: str = "",
    collection_vintage: str | None = None,
    license_name: str | None = None,
    source_revision: str | None = None,
    provenance_url: str | None = None,
) -> StudyPrompt:
    if source in SOURCE_METADATA:
        metadata = _metadata(source, domain=domain, benchmark=benchmark)
    elif source == "bfcl-v4":
        metadata = {
            "source": source,
            "benchmark_family": benchmark,
            "domain": domain,
            "collection_vintage": collection_vintage or "2026-08-25",
            "license_name": license_name or "Apache-2.0",
            "license_status": "public",
            "source_revision": source_revision or f"ShishirPatil/gorilla@{BFCL_REVISION}",
            "provenance_url": provenance_url or "https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard",
        }
    elif source == "tau3":
        metadata = {
            "source": source,
            "benchmark_family": benchmark,
            "domain": domain,
            "collection_vintage": collection_vintage or "2026-08-25",
            "license_name": license_name or "MIT",
            "license_status": "public",
            "source_revision": source_revision or f"sierra-research/tau2-bench@{TAU3_REVISION}",
            "provenance_url": provenance_url or "https://github.com/sierra-research/tau2-bench",
        }
    else:
        metadata = {
            "source": source,
            "benchmark_family": benchmark,
            "domain": domain,
            "collection_vintage": collection_vintage or "",
            "license_name": license_name or "",
            "license_status": "public",
            "source_revision": source_revision or "",
            "provenance_url": provenance_url or "",
        }
    return StudyPrompt(
        prompt_id=prompt_id,
        prompt=prompt,
        split="train",
        source=metadata["source"],
        benchmark_family=metadata["benchmark_family"],
        domain=metadata["domain"],
        template_family=template,
        collection_vintage=metadata["collection_vintage"],
        license_name=metadata["license_name"],
        license_status=metadata["license_status"],
        source_revision=metadata["source_revision"],
        provenance_url=metadata["provenance_url"],
        lane=lane,
        label_type=label_type,
        task_format=task_format,
        leakage_group=f"{metadata['source']}:{prompt_id}",
        protocol_id=protocol_id,
        tags=tags,
        grader=grader,
    )


def _select(raw_rows: list[dict[str, Any]], count: int, *, key: str) -> list[dict[str, Any]]:
    if len(raw_rows) < count:
        raise ValueError(f"{key} has {len(raw_rows)} rows; need {count}")
    return sorted(
        raw_rows,
        key=lambda row: _stable_key(f"{key}:{row.get('id', '')}:{row.get('question', '')}"),
    )[:count]


def _assign_splits(rows: list[StudyPrompt], quotas: dict[str, int]) -> list[StudyPrompt]:
    if sum(quotas.values()) != len(rows):
        raise ValueError(f"split quotas {quotas} do not sum to {len(rows)}")
    ordered = sorted(rows, key=lambda row: _stable_key(row.prompt_id))
    result: list[StudyPrompt] = []
    offset = 0
    for split in ("train", "validation", "test"):
        for row in ordered[offset : offset + quotas[split]]:
            result.append(replace(row, split=split))
        offset += quotas[split]
    return result


def _build_hf_rows(cache_dir: Path, rejected: list[dict[str, Any]]) -> list[StudyPrompt]:
    accepted: list[StudyPrompt] = []
    mmlu_rows, errors = _read_jsonl(cache_dir / "mmlu.jsonl")
    rejected.extend(errors)
    mmlu_selected = _select(mmlu_rows, 300, key="mmlu")
    for raw in mmlu_selected:
        choices = list(raw.get("choices") or raw.get("options") or [])
        category = str(raw.get("subject") or "general")
        family = f"mmlu:{category}"
        accepted.append(
            _row(
                prompt_id=f"mmlu:{raw.get('id') or _stable_key(str(raw))[:12]}",
                prompt=_choices_prompt(str(raw.get("question") or ""), choices),
                source="cais/mmlu",
                benchmark=family,
                domain=category,
                template=family,
                lane="general",
                label_type="deterministic",
                task_format="multiple_choice",
                grader={"answer": _choice_letter(raw.get("answer"), choices), "answer_type": "multiple_choice"},
            )
        )

    mmlu_pro_rows, errors = _read_jsonl(cache_dir / "mmlu-pro.jsonl")
    rejected.extend(errors)
    for raw in _select(mmlu_pro_rows, 200, key="mmlu-pro"):
        choices = list(raw.get("choices") or raw.get("options") or [])
        category = str(raw.get("category") or raw.get("subject") or "general")
        family = f"mmlu-pro:{category}"
        accepted.append(
            _row(
                prompt_id=f"mmlu-pro:{raw.get('id') or _stable_key(str(raw))[:12]}",
                prompt=_choices_prompt(str(raw.get("question") or ""), choices),
                source="TIGER-Lab/MMLU-Pro",
                benchmark=family,
                domain=category,
                template=family,
                lane="general",
                label_type="deterministic",
                task_format="multiple_choice",
                grader={"answer": _choice_letter(raw.get("answer"), choices), "answer_type": "multiple_choice"},
            )
        )

    gsm_rows, errors = _read_jsonl(cache_dir / "gsm8k.jsonl")
    rejected.extend(errors)
    for raw in _select(gsm_rows, 200, key="gsm8k"):
        match = re.search(r"####\s*(.+?)\s*$", str(raw.get("answer") or ""), re.DOTALL)
        if not match:
            rejected.append({"source": "openai/gsm8k", "id": raw.get("id"), "reason": "missing_final_answer_marker"})
            continue
        accepted.append(
            _row(
                prompt_id=f"gsm8k:{raw.get('id') or _stable_key(str(raw))[:12]}",
                prompt=f"{str(raw.get('question') or '').strip()}\n\nReturn the final numeric answer.",
                source="openai/gsm8k",
                benchmark="gsm8k:word-problem",
                domain="math",
                template="gsm8k:word-problem",
                lane="general",
                label_type="deterministic",
                task_format="exact_match",
                grader={"answer": match.group(1).strip(), "answer_type": "exact_match"},
            )
        )

    arc_rows, errors = _read_jsonl(cache_dir / "arc-easy.jsonl")
    rejected.extend(errors)
    for raw in _select(arc_rows, 100, key="arc-easy"):
        choices = raw.get("choices") or {}
        if isinstance(choices, dict):
            labels = list(choices.get("label") or [])
            texts = list(choices.get("text") or [])
        else:
            labels = []
            texts = list(choices)
        answer_key = str(raw.get("answerKey") or "")
        answer = chr(ord("A") + labels.index(answer_key)) if answer_key in labels else _choice_letter(answer_key, texts)
        accepted.append(
            _row(
                prompt_id=f"arc-easy:{raw.get('id') or _stable_key(str(raw))[:12]}",
                prompt=_choices_prompt(str(raw.get("question") or ""), texts),
                source="allenai/ai2_arc",
                benchmark="arc-easy:science",
                domain="science",
                template="arc-easy:science",
                lane="general",
                label_type="deterministic",
                task_format="multiple_choice",
                grader={"answer": answer, "answer_type": "multiple_choice"},
            )
        )
    by_format = {
        "multiple_choice": [row for row in accepted if row.task_format == "multiple_choice"],
        "exact_match": [row for row in accepted if row.task_format == "exact_match"],
    }
    result: list[StudyPrompt] = []
    result.extend(_assign_splits(by_format["multiple_choice"], {"train": 360, "validation": 120, "test": 120}))
    result.extend(_assign_splits(by_format["exact_match"], {"train": 120, "validation": 40, "test": 40}))
    return result


def _build_bfcl_rows(cache_dir: Path, rejected: list[dict[str, Any]]) -> list[StudyPrompt]:
    # The cache has task definitions, but this checkout has neither the pinned
    # official BFCL executor nor a locally executable structured-provider
    # endpoint.  Keep the source inventory in the gap report and exclude every
    # BFCL row from the paid corpus until both capabilities are present.
    rejected.append(
        {
            "source": "bfcl-v4",
            "reason": "excluded_from_paid_corpus",
            "status": "blocked_missing_executable_official_checker",
            "required": [
                "pinned function definitions/tool schemas",
                "provider-native structured tool calls",
                "pinned official BFCL checker executable locally",
            ],
            "ground_truth_policy": "complete ground_truth sequence must be preserved; no first-element reduction",
        }
    )
    return []


def _build_tau_rows(cache_dir: Path, rejected: list[dict[str, Any]]) -> list[StudyPrompt]:
    # A serialized scenario is not a tau-bench run.  No pinned tau-bench
    # environment/evaluator is installed in this workspace, so these rows are
    # excluded rather than left as pending paid calls.
    rejected.append(
        {
            "source": "tau3",
            "reason": "excluded_from_paid_corpus",
            "status": "blocked_missing_executable_tau_bench_environment",
            "required": [
                "pinned tau-bench environment",
                "domain tool schemas and persisted environment state",
                "official task/state evaluator",
            ],
            "serialized_prompt_is_not_run": True,
        }
    )
    return []


def _build_route_rows(cache_dir: Path, rejected: list[dict[str, Any]]) -> list[StudyPrompt]:
    raw_rows, errors = _read_jsonl(cache_dir / "routellm.jsonl")
    rejected.extend(errors)
    rows: list[StudyPrompt] = []
    for raw in _select(raw_rows, 200, key="routellm"):
        task_id = str(raw.get("id") or raw.get("prompt_id") or _stable_key(str(raw))[:12])
        domain = str(raw.get("source") or "general").strip() or "general"
        rows.append(
            _row(
                prompt_id=f"routellm:{task_id}",
                prompt=str(raw.get("prompt") or "").strip(),
                source="routellm/gpt4_dataset",
                benchmark="routellm-gpt4",
                domain=domain,
                template="routellm:gpt4",
                lane="general",
                label_type="pairwise",
                task_format="conversational_reasoning",
                tags=("disagreement-candidate",),
                grader={
                    "kind": "frozen_pairwise_judge",
                    "judge_id": "router-v2-local-pairwise-judge",
                    "judge_version": "router-v2-pairwise-judge-v2",
                    "criteria_version": "router-v2-pairwise-criteria-v1",
                    "position_randomization": "prompt_id_seeded_sha256",
                    "sampling_settings": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": 256,
                    },
                    "tie_policy": "record_tie_without_forcing_binary_label",
                    "abstention_policy": "judge_may_abstain_on rubric_or_context_failure",
                    "disagreement_review_policy": "independent_manual_review_then_adjudication; no automatic overwrite",
                    "judging_cost_status": "postponed_unpriced",
                    "expected_judging_cost_usd": None,
                    "worst_case_judging_cost_usd": None,
                },
            )
        )
    return _assign_splits(rows, {"train": 120, "validation": 40, "test": 40})


def _preferred_protocol(raw: dict[str, Any]) -> str:
    protocols = [str(value) for value in raw.get("protocols") or [] if str(value)]
    preferred = "BrowseComp-Plus::standard::AgentIR::no-CM"
    if preferred in protocols:
        return preferred
    return sorted(protocols)[0] if protocols else ""


def _build_anchor_rows(anchor_path: Path, rejected: list[dict[str, Any]]) -> list[StudyPrompt]:
    raw_rows, errors = _read_jsonl(anchor_path)
    rejected.extend(errors)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_rows:
        if raw.get("prompt") and raw.get("reference_answer"):
            by_family[str(raw.get("benchmark_family") or "unknown")].append(raw)

    selected = _select(by_family.get("BrowseComp-Plus", []), 240, key="anchor:BrowseComp-Plus")
    for family in ("GAIA-text", "xBench", "BrowseComp-ZH"):
        selected.extend(_select(by_family.get(family, []), 20, key=f"anchor:{family}"))
    if len(selected) != 300:
        rejected.append({"source": str(anchor_path), "reason": f"selected {len(selected)} rows; need 300"})

    rows: list[StudyPrompt] = []
    for raw in selected:
        prompt_id = str(raw.get("prompt_id") or raw.get("id") or _stable_key(str(raw))[:12])
        family = str(raw.get("benchmark_family") or "unknown")
        reference = str(raw.get("reference_answer") or "").strip()
        rows.append(
            _row(
                prompt_id=f"anchor:{prompt_id}",
                prompt=str(raw.get("prompt") or "").strip(),
                source=ANCHOR_SOURCE,
                benchmark=family,
                domain=str(raw.get("domain") or "research"),
                template=f"anchor:{family}",
                lane="general",
                label_type="deterministic",
                task_format="search_research",
                protocol_id=_preferred_protocol(raw),
                collection_vintage=f"dataset_revision:{ANCHOR_REVISION}",
                license_name="Apache-2.0",
                source_revision=f"{ANCHOR_SOURCE}@{ANCHOR_REVISION}",
                provenance_url=ANCHOR_URL,
                grader={
                    "expected_properties": [reference],
                    "answer_type": "reference_answer_contains",
                    "public_anchor_outcomes_used_as_candidate_outcomes": False,
                    "protocol_isolated": True,
                },
            )
        )
    browsecomp_rows = [row for row in rows if row.benchmark_family == "BrowseComp-Plus"]
    heldout_rows = [row for row in rows if row.benchmark_family != "BrowseComp-Plus"]
    return _assign_splits(
        browsecomp_rows,
        {"train": 180, "validation": 60, "test": 0},
    ) + [replace(row, split="test") for row in heldout_rows]


def _apply_stratum_tags(rows: list[StudyPrompt]) -> list[StudyPrompt]:
    short_hard = {
        row.prompt_id
        for row in sorted(
            (row for row in rows if row.source == "TIGER-Lab/MMLU-Pro"),
            key=lambda row: _stable_key(row.prompt_id),
        )[:100]
    }
    specialist = {
        row.prompt_id
        for row in sorted(
            (row for row in rows if row.source == "cais/mmlu" and row.domain not in {"general", "miscellaneous"}),
            key=lambda row: _stable_key(row.prompt_id),
        )[:100]
    }
    ambiguous = {
        row.prompt_id
        for row in rows
        if row.source == "bfcl-v4" and row.benchmark_family == "bfcl-v4:irrelevance"
    }
    long_easy_eligible = [
        row
        for row in rows
        if (row.source == "tau3" and ("[PERSONA:Easy]" in row.prompt or len(row.prompt) >= 500))
        or (row.source == ANCHOR_SOURCE and len(row.prompt) >= 500)
    ]
    long_easy = {
        row.prompt_id
        for row in sorted(long_easy_eligible, key=lambda row: _stable_key(row.prompt_id))[:100]
    }

    train_values = {
        field: {getattr(row, field) for row in rows if row.split == "train"}
        for field in ("source", "domain", "benchmark_family", "template_family")
    }
    result: list[StudyPrompt] = []
    for row in rows:
        tags = set(row.tags)
        if row.prompt_id in short_hard:
            tags.add("short-hard")
        if row.prompt_id in specialist:
            tags.add("specialist")
        if row.prompt_id in ambiguous:
            tags.add("ambiguous")
        if row.prompt_id in long_easy:
            tags.add("long-easy")
        if row.split == "test":
            unseen = any(getattr(row, field) not in train_values[field] for field in train_values)
            tags.add("ood" if unseen else "representative")
        result.append(replace(row, tags=tuple(sorted(tags))))
    return result


def _deduplicate(rows: list[StudyPrompt], rejected: list[dict[str, Any]]) -> list[StudyPrompt]:
    seen: dict[str, str] = {}
    result: list[StudyPrompt] = []
    for row in rows:
        if row.normalized_hash in seen:
            rejected.append(
                {
                    "prompt_id": row.prompt_id,
                    "prior_prompt_id": seen[row.normalized_hash],
                    "reason": "duplicate_normalized_prompt",
                }
            )
            continue
        seen[row.normalized_hash] = row.prompt_id
        result.append(row)
    return result


def _source_split_counts(rows: list[StudyPrompt]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[row.source][row.split] += 1
    return {source: dict(sorted(value.items())) for source, value in sorted(counts.items())}


def _input_fingerprints(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(set(paths)):
        if path.exists() and path.is_file():
            result[str(path)] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    return result


def _gap_report(
    rows: list[StudyPrompt],
    rejected: list[dict[str, Any]],
    audit: Any,
    *,
    source_split_counts: dict[str, dict[str, int]],
    input_fingerprints: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    missing_strata = {
        name: minimum - audit.stratum_counts.get(name, 0)
        for name, minimum in {
            "short-hard": 100,
            "long-easy": 100,
            "specialist": 100,
            "ambiguous": 100,
            "ood": 100,
            "representative": 100,
            "disagreement-candidate": 100,
        }.items()
        if audit.stratum_counts.get(name, 0) < minimum
    }
    gated_sources = [
        {
            "source": "gaia-benchmark/GAIA",
            "status": "gated",
            "reason": "task data is gated with a non-redistribution agreement; no local authorization was supplied",
            "replacement_gap": "the public-anchor GAIA-text prompt-only subset is present, but its historical model outcomes are not candidate outcomes",
        },
        {
            "source": "Idavidrein/gpqa",
            "status": "gated",
            "reason": "the dataset card is gated and prohibits revealing examples",
            "replacement_gap": "no GPQA rows were added; an authorized local export and non-redistribution manifest are required",
        },
        {
            "source": "cais/hle",
            "status": "gated",
            "reason": "the dataset card is gated with an anti-resharing agreement",
            "replacement_gap": "no HLE rows were added; an authorized local export is required",
        },
        {
            "source": "SWE-bench/SWE-bench_Verified",
            "status": "license_review_required",
            "reason": "the dataset card does not declare a dataset license even though the official repository is MIT",
            "replacement_gap": "no SWE rows were added; an explicit dataset-license decision and pinned task export are required",
        },
        {
            "source": "Tevatron/browsecomp-plus",
            "status": "accessible_but_not_imported",
            "reason": "the MIT dataset is a large query/document artifact; this corpus uses only the pinned i-DeepSearch prompt rows",
            "replacement_gap": "a separate pinned query/answer export may replace or extend the prompt-only anchor subset after review",
        },
    ]
    return {
        "schema_version": 2,
        "status": "valid" if audit.valid else "blocked",
        "target": {"train": 1200, "validation": 400, "test": 400, "total": 2000},
        "accepted_prompt_count": len(rows),
        "accepted_prompts_by_source_and_split": source_split_counts,
        "rejected_prompts": rejected,
        "missing_strata": missing_strata,
        "post_collection_required_slices": {
            "model-disagreement": {
                "minimum": 100,
                "status": "pending_until_core_model_outcomes_exist",
                "definition": "same prompt with divergent explicit outcomes from at least two core candidate models",
                "never_used_for_training_or_tuning": True,
            }
        },
        "disagreement_candidate_boundary": {
            "tag": "disagreement-candidate",
            "basis": "frozen public-model pairwise proxy from RouteLLM source rows",
            "observed_candidate_model_disagreement": False,
            "must_not_be_described_as": "observed candidate-model disagreement",
        },
        "missing_label_types": {
            label: minimum - audit.label_counts.get(label, 0)
            for label, minimum in {"deterministic": 1_300, "tool_state": 500, "pairwise": 200}.items()
            if audit.label_counts.get(label, 0) < minimum
        },
        "executable_task_coverage": {
            "ordinary_text": {
                "status": "eligible_by_adapter_contract",
                "rows": sum(
                    1
                    for row in rows
                    if row.source not in {"bfcl-v4", "tau3"}
                    and row.task_format != "code_execution"
                ),
            },
            "bfcl": {
                "status": "excluded_until_official_checker_and_tool_endpoint_are_executable",
                "rows_in_paid_corpus": 0,
                "local_fixture_status": "adapter_fixture_available",
            },
            "tau_bench": {
                "status": "excluded_until_real_environment_and_state_evaluator_are_executable",
                "rows_in_paid_corpus": 0,
                "serialized_scenario_rows_in_paid_corpus": 0,
                "local_fixture_status": "environment-interface-fixture-available",
            },
        },
        "genuine_code_lane_replacement_gap": {
            "status": "blocked",
            "licensed_pinned_executable_code_sources_found": [],
            "required_rows_by_split": {"train": 240, "validation": 80, "test": 80},
            "rows_present_by_split": {
                split: sum(1 for row in rows if row.split == split and row.lane == "code")
                for split in ("train", "validation", "test")
            },
            "shortfall_by_split": {
                split: max(
                    0,
                    required
                    - sum(1 for row in rows if row.split == split and row.lane == "code"),
                )
                for split, required in {"train": 240, "validation": 80, "test": 80}.items()
            },
            "prohibited_substitution": "BFCL tool selection and natural-language function selection cannot satisfy this gap",
        },
        "audit": audit.to_dict(),
        "corpus_checksum": corpus_checksum(rows),
        "source_revisions": sorted({row.source_revision for row in rows}),
        "input_fingerprints": input_fingerprints,
        "ood_policy": {
            "version": OOD_POLICY_VERSION,
            "rule": "test row is OOD iff source, domain, benchmark_family, or template_family is absent from the training values; otherwise it is representative",
            "test_membership_alone_is_not_evidence": True,
            "held_out_examples": ["tau3:telecom", "GAIA-text", "xBench", "BrowseComp-ZH"],
        },
        "stratum_basis": {
            "short-hard": "100 MMLU-Pro rows selected by source/category proxy; not a model score",
            "long-easy": "100 pinned tau3/persona or serialized prompt length >=500 rows selected by source/length proxy; not a model score",
            "specialist": "100 MMLU rows from non-general subjects",
            "ambiguous": "100 pinned BFCL v4 irrelevance rows",
            "ood": "explicit train-source/domain/benchmark/template distance rule",
            "representative": "explicit train-source/domain/benchmark/template overlap rule",
            "disagreement-candidate": "frozen RouteLLM public pairwise proxy; not observed candidate-model disagreement",
        },
        "public_anchor_boundary": {
            "dataset": ANCHOR_SOURCE,
            "revision": ANCHOR_REVISION,
            "protocols_are_kept_separate": True,
            "outcomes_used_as_theriad_candidate_outcomes": False,
            "prompt_and_reference_answer_use": "search/research grading context only",
        },
        "gated_or_unresolved_sources": gated_sources,
        "unresolved_data_gaps": [
            {
                "gap": "BFCL official checker/provider-native tool-call execution",
                "status": "blocked_and_excluded",
                "reason": "no local pinned official checker or structured endpoint capability was found; rows are not paid-eligible",
            },
            {
                "gap": "tau-bench environment/evaluator execution",
                "status": "blocked_and_excluded",
                "reason": "no real pinned environment was found; serialized scenarios are not agent runs and rows are not paid-eligible",
            },
            {
                "gap": "genuine executable code tasks",
                "status": "replacement_required",
                "reason": "the available local caches contain no licensed pinned implementation/debugging tasks with executable tests",
            },
            {
                "gap": "pairwise judge outcomes and judging price",
                "status": "pending_and_unpriced",
                "reason": "criteria, judge identity/version, sampling, and position randomization are frozen, but no judge call is authorized",
            },
            {
                "gap": "authentic model-disagreement slice",
                "status": "post_collection_required",
                "reason": "requires explicit core candidate-model outcomes after collection and cannot be replaced by the public proxy",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-cache-dir", type=Path, required=True)
    parser.add_argument("--bfcl-cache-dir", type=Path, required=True)
    parser.add_argument("--tau3-cache-dir", type=Path, required=True)
    parser.add_argument("--anchor-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--gap-report", type=Path, required=True)
    args = parser.parse_args()

    rejected: list[dict[str, Any]] = []
    rows = _build_hf_rows(args.hf_cache_dir, rejected)
    rows.extend(_build_bfcl_rows(args.bfcl_cache_dir, rejected))
    rows.extend(_build_tau_rows(args.tau3_cache_dir, rejected))
    rows.extend(_build_route_rows(args.hf_cache_dir, rejected))
    rows.extend(_build_anchor_rows(args.anchor_jsonl, rejected))
    rows = _apply_stratum_tags(rows)
    rows = _deduplicate(rows, rejected)
    audit = audit_corpus(rows, rejected=rejected)
    source_split_counts = _source_split_counts(rows)
    inputs = [
        args.anchor_jsonl,
        args.hf_cache_dir / "mmlu.jsonl",
        args.hf_cache_dir / "mmlu-pro.jsonl",
        args.hf_cache_dir / "gsm8k.jsonl",
        args.hf_cache_dir / "arc-easy.jsonl",
        args.hf_cache_dir / "routellm.jsonl",
        args.bfcl_cache_dir / "BFCL_v4_irrelevance.json",
        args.bfcl_cache_dir / "BFCL_v4_simple_python.json",
        args.bfcl_cache_dir / "possible_answer" / "BFCL_v4_simple_python.json",
        args.tau3_cache_dir / "data_tau2_domains_airline_tasks.json",
        args.tau3_cache_dir / "data_tau2_domains_mock_tasks.json",
        args.tau3_cache_dir / "data_tau2_domains_retail_tasks.json",
        args.tau3_cache_dir / "data_tau2_domains_telecom_tasks.json",
    ]
    for filename in ("BFCL_v4_exec_simple.json", "BFCL_v4_exec_multiple.json", "BFCL_v4_exec_parallel.json"):
        inputs.extend(
            [
                args.bfcl_cache_dir / "unused_datasets" / "question" / filename,
                args.bfcl_cache_dir / "unused_datasets" / "possible_answer" / filename,
            ]
        )
    input_fingerprints = _input_fingerprints(inputs)
    gap_report = _gap_report(
        rows,
        rejected,
        audit,
        source_split_counts=source_split_counts,
        input_fingerprints=input_fingerprints,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, args.output)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "candidate_path": str(args.output),
                "candidate_count": len(rows),
                "corpus_checksum": corpus_checksum(rows),
                "source_split_counts": source_split_counts,
                "audit": audit.to_dict(),
                "source_revisions": sorted({row.source_revision for row in rows}),
                "input_fingerprints": input_fingerprints,
                "ood_policy_version": OOD_POLICY_VERSION,
                "grok_selection_algorithm_version": GROK_SELECTION_ALGORITHM_VERSION,
                "public_anchor_outcomes_used_as_candidate_outcomes": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    args.gap_report.parent.mkdir(parents=True, exist_ok=True)
    args.gap_report.write_text(json.dumps(gap_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gap_report, indent=2, sort_keys=True))
    return 0 if audit.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
