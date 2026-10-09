#!/usr/bin/env python3
"""Acquire, statically screen, and assemble the pinned Router V2 additions.

This command consumes only already-downloaded, pinned source files.  It never
contacts a model/provider and never executes benchmark or candidate code on
the host.  Canonical solutions are used only for executor validation through
``theriad.routing.sandbox``.  Rows that do not pass provenance, license,
grouping, or executor checks are reported rather than padded with paraphrases.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.sandbox import (  # noqa: E402
    SANDBOX_EXECUTOR_ID,
    SANDBOX_POLICY_VERSION,
    sandbox_available,
)
from theriad.routing.study import (  # noqa: E402
    OOD_POLICY_VERSION,
    REPEATED_TRIAL_GROUP_CONTRACT,
    StudyPrompt,
    audit_corpus,
    corpus_checksum,
    load_jsonl,
    write_jsonl,
)
from theriad.routing.study_adapters import (  # noqa: E402
    BFCL_ADAPTER_VERSION,
    CODE_ADAPTER_VERSION,
    TAU_ADAPTER_VERSION,
)

BIGCODE_REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
BIGCODE_REPO_REVISION = "09dd993f46c3fbf3a799465bb96d524edcb0b199"
HUMANEVAL_REVISION = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"
MBPP_REVISION = "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"
BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_OFFICIAL_CHECKER_ID = "bfcl_eval.eval_checker.ast_eval.ast_checker.ast_checker"
TAU_REVISION = "a2c024725189473d2d7cea3a5cfdbcc67478e41f"
TAU_PACKAGE_VERSION = "1.0.1"
ANCHOR_REVISION = "19342195a5c5c9f8ae016cac6a388d937371df8e"
ANCHOR_SOURCE = "i-DeepSearch/observation-masking-eval-logs"
ANCHOR_URL = "https://huggingface.co/datasets/i-DeepSearch/observation-masking-eval-logs"
SOURCE_DATE = "2026-08-25"

TARGET_DETERMINISTIC_SPLITS = {"train": 180, "validation": 60, "test": 60}
TARGET_BFCL_SPLITS = {"train": 60, "validation": 20, "test": 20}
TARGET_TAU_SPLITS = {"train": 180, "validation": 60, "test": 60}
PAIRWISE_REPLACEMENT_SPLITS = {"train": 120, "validation": 40, "test": 40}
GENERAL_REPLACEMENT_SPLITS = {"train": 120, "validation": 40, "test": 40}
GENERAL_REPLACEMENT_FAMILY_QUOTAS = {
    "BrowseComp-Plus": {"train": 72, "validation": 24, "test": 24},
    "BrowseComp-ZH": {"train": 24, "validation": 8, "test": 8},
    "GAIA-text": {"train": 12, "validation": 4, "test": 4},
    "xBench": {"train": 12, "validation": 4, "test": 4},
}
MAX_TAU_REFERENCE_ACTIONS = 8

CODE_FILES = {
    "bigcodebench": "bigcodebench/v0.1.4-00000-of-00001.parquet",
    "humanevalplus": "humanevalplus/test-00000-of-00001-5973903632b82d40.parquet",
    "mbppplus": "mbppplus/test-00000-of-00001-d5781c9c51e02795.parquet",
}

SAFE_TEST_IMPORTS = {
    "abc",
    "ast",
    "bisect",
    "collections",
    "copy",
    "csv",
    "datetime",
    "decimal",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "typing",
    "unittest",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "sympy",
    "networkx",
}
FORBIDDEN_TEST_IMPORTS = {
    "ctypes",
    "ftplib",
    "multiprocessing",
    "os",
    "pickle",
    "resource",
    "signal",
    "socket",
    "subprocess",
    "urllib",
    "webbrowser",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized(text: str) -> str:
    return " ".join(str(text).strip().split())


def _split_assign(rows: list[StudyPrompt], quotas: dict[str, int]) -> list[StudyPrompt]:
    ordered = sorted(rows, key=lambda row: _stable(row.prompt_id))
    if len(ordered) != sum(quotas.values()):
        raise ValueError(f"split quotas {quotas} do not sum to {len(ordered)} rows")
    result: list[StudyPrompt] = []
    offset = 0
    for split in ("train", "validation", "test"):
        count = quotas[split]
        result.extend(
            StudyPrompt.from_dict(
                {**row.to_dict(), "split": split}, require_frozen_hash=True
            )
            for row in ordered[offset : offset + count]
        )
        offset += count
    return result


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - acquisition environment only
        raise RuntimeError("pandas is required to read pinned parquet sources") from exc
    frame = pd.read_parquet(path)
    return [dict(record) for record in frame.to_dict("records")]


def _imports(source: str) -> tuple[set[str], str | None]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return set(), f"syntax_error:{exc.msg}"
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", 1)[0])
    return names, None


def _screen_test_source(test_source: str) -> dict[str, Any]:
    imported, parse_error = _imports(test_source)
    forbidden = sorted(imported & FORBIDDEN_TEST_IMPORTS)
    unknown = sorted(imported - SAFE_TEST_IMPORTS - FORBIDDEN_TEST_IMPORTS)
    return {
        "parse_error": parse_error,
        "imports": sorted(imported),
        "forbidden_imports": forbidden,
        "unknown_imports": unknown,
        "accepted": not parse_error and not forbidden and not unknown,
    }


def _safe_row_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value or "")


def _entry_point(raw: dict[str, Any]) -> str:
    published = _safe_row_text(raw.get("entry_point")).strip()
    if published:
        return published
    code = _safe_row_text(raw.get("code"))
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return node.name
    return ""


def _tau_prompt(raw: dict[str, Any]) -> str:
    """Render the official user request without serializing the scenario."""

    ticket = raw.get("ticket")
    if isinstance(ticket, str) and ticket.strip():
        return ticket.strip()
    instructions = (raw.get("user_scenario") or {}).get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        return instructions.strip()
    if isinstance(instructions, dict):
        labels = (
            ("reason_for_call", "Customer request"),
            ("known_info", "Known customer information"),
            ("task_instructions", "Conversation constraints"),
        )
        parts = [
            f"{label}: {instructions[key].strip()}"
            for key, label in labels
            if isinstance(instructions.get(key), str) and instructions[key].strip()
        ]
        return "\n\n".join(parts)
    return ""


def _code_grader(
    *,
    task_type: str,
    entry_point: str,
    tests: str,
    harness_id: str,
    harness_version: str,
    source_task_id: str,
    source_file_sha256: str,
    source_family: str,
    contamination: dict[str, Any],
    difficulty: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "sandboxed_code_tests",
        "code_task_type": task_type,
        "entry_point": entry_point,
        "executable_tests": tests,
        "test_harness_id": harness_id,
        "test_harness_version": harness_version,
        "test_protocol": "python_check_or_unittest",
        "runtime_id": "python3.11-compatible",
        "language": "python",
        "sandbox_executor_id": SANDBOX_EXECUTOR_ID,
        "sandbox_policy_version": SANDBOX_POLICY_VERSION,
        "adapter_version": CODE_ADAPTER_VERSION,
        "sandbox_available": sandbox_available(),
        "network_disabled": True,
        "base_filesystem_read_only": True,
        "task_directory_only_writable": True,
        "credentials_in_environment": False,
        "source_task_id": source_task_id,
        "source_file_sha256": source_file_sha256,
        "source_family": source_family,
        "contamination_metadata": contamination,
        "difficulty_screening": difficulty,
        "collection_eligible": sandbox_available(),
        "collection_block_reason": (
            "approved sandbox backend unavailable" if not sandbox_available() else ""
        ),
    }


def _make_code_row(
    *,
    raw: dict[str, Any],
    source_family: str,
    source: str,
    benchmark: str,
    revision: str,
    license_name: str,
    url: str,
    file_sha256: str,
    task_type: str,
    prompt: str,
    solution_source: str,
    test_source: str,
    entry_point: str,
    harness_id: str,
    harness_version: str,
) -> tuple[StudyPrompt | None, dict[str, Any]]:
    screen = _screen_test_source(test_source)
    solution_screen = _screen_test_source(solution_source)
    if not _normalized(prompt):
        return None, {"task_id": str(raw.get("task_id")), "reason": "empty_prompt"}
    if not test_source.strip():
        return None, {"task_id": str(raw.get("task_id")), "reason": "missing_tests"}
    if not solution_screen["accepted"]:
        return None, {
            "task_id": str(raw.get("task_id")),
            "reason": "unsafe_or_unavailable_solution_imports",
            "screen": solution_screen,
        }
    if not screen["accepted"]:
        return None, {
            "task_id": str(raw.get("task_id")),
            "reason": "unsafe_or_unparseable_test_imports",
            "screen": screen,
        }
    difficulty = {
        "method": "source-test-and-prompt-size-static-proxy-v1",
        "score": len(prompt) + len(test_source) + len(solution_source),
        "modern_model_outcome_observed": False,
        "status": "static_proxy_only",
    }
    contamination = {
        "source_published_date_available": False,
        "task_id": str(raw.get("task_id")),
        "benchmark_revision_pinned": True,
        "candidate_model_contamination_check": "not_observed_before_collection",
    }
    row = StudyPrompt(
        prompt_id=f"code:{source_family}:{raw.get('task_id')}",
        prompt=prompt,
        split="train",
        source=source,
        benchmark_family=benchmark,
        domain="software-engineering",
        template_family=f"{source_family}:python-function",
        collection_vintage=SOURCE_DATE,
        license_name=license_name,
        license_status="approved",
        source_revision=revision,
        provenance_url=url,
        lane="code",
        label_type="deterministic",
        task_format="code_execution",
        leakage_group=f"{source}:{raw.get('task_id')}",
        tags=("code", "implementation"),
        grader=_code_grader(
            task_type=task_type,
            entry_point=entry_point,
            tests=test_source,
            harness_id=harness_id,
            harness_version=harness_version,
            source_task_id=str(raw.get("task_id")),
            source_file_sha256=file_sha256,
            source_family=source_family,
            contamination=contamination,
            difficulty=difficulty,
        ),
    )
    return row, {"task_id": str(raw.get("task_id")), "reason": "accepted_static_screen"}


def _acquire_code_sources(
    code_cache: Path,
    *,
    reserved_hashes: set[str],
    executor_report: Path | None = None,
    executor_exclusions: Path | None = None,
) -> tuple[list[StudyPrompt], dict[str, Any]]:
    files = {name: code_cache / relative for name, relative in CODE_FILES.items()}
    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(path)
    records: dict[str, list[dict[str, Any]]] = {
        name: _read_parquet(path) for name, path in files.items()
    }
    candidates: dict[str, list[StudyPrompt]] = defaultdict(list)
    rejections: list[dict[str, Any]] = []
    failed_executor_keys: set[tuple[str, str]] = set()
    executor_validation: dict[str, Any] = {}
    if executor_report is not None and executor_report.exists():
        validation = json.loads(executor_report.read_text(encoding="utf-8"))
        executor_validation = validation
        failed_executor_keys = {
            (str(item.get("source")), str(item.get("source_task_id")))
            for item in validation.get("failures", [])
            if item.get("source") and item.get("source_task_id")
        }
    if executor_exclusions is not None and executor_exclusions.exists():
        exclusions = json.loads(executor_exclusions.read_text(encoding="utf-8"))
        failed_executor_keys.update(
            (str(item.get("source")), str(item.get("source_task_id")))
            for item in exclusions.get("failures", [])
            if item.get("source") and item.get("source_task_id")
        )
    for raw in records["bigcodebench"]:
        row, evidence = _make_code_row(
            raw=raw,
            source_family="bigcodebench",
            source="bigcode/bigcodebench",
            benchmark="bigcodebench:v0.1.4",
            revision=f"bigcode/bigcodebench@{BIGCODE_REVISION}:repo@{BIGCODE_REPO_REVISION}",
            license_name="Apache-2.0",
            url="https://huggingface.co/datasets/bigcode/bigcodebench",
            file_sha256=_sha256(files["bigcodebench"]),
            task_type="function_implementation",
            prompt=_safe_row_text(raw.get("instruct_prompt")),
            solution_source=(
                _safe_row_text(raw.get("code_prompt"))
                + "\n"
                + _safe_row_text(raw.get("canonical_solution"))
            ),
            test_source=_safe_row_text(raw.get("test")),
            entry_point=_safe_row_text(raw.get("entry_point")),
            harness_id="bigcodebench.unittest",
            harness_version=f"bigcodebench@{BIGCODE_REVISION}",
        )
        if row is not None:
            key = (row.source, str(row.grader.get("source_task_id")))
            if key in failed_executor_keys:
                rejections.append({"source": row.source, "task_id": key[1], "reason": "prior_sandbox_executor_failure"})
            else:
                candidates["bigcodebench"].append(row)
        else:
            rejections.append({"source": "bigcode/bigcodebench", **evidence})
    for family, source, benchmark, revision, license_name, url, task_type in (
        (
            "humanevalplus",
            "evalplus/humanevalplus",
            "evalplus:HumanEval+",
            f"evalplus/humanevalplus@{HUMANEVAL_REVISION}",
            "Apache-2.0",
            "https://huggingface.co/datasets/evalplus/humanevalplus",
            "function_implementation",
        ),
        (
            "mbppplus",
            "evalplus/mbppplus",
            "evalplus:MBPP+",
            f"evalplus/mbppplus@{MBPP_REVISION}",
            "Apache-2.0",
            "https://huggingface.co/datasets/evalplus/mbppplus",
            "function_implementation",
        ),
    ):
        for raw in records[family]:
            row, evidence = _make_code_row(
                raw=raw,
                source_family=family,
                source=source,
                benchmark=benchmark,
                revision=revision,
                license_name=license_name,
                url=url,
                file_sha256=_sha256(files[family]),
                task_type=task_type,
                prompt=_safe_row_text(raw.get("prompt")),
                solution_source=(
                    _safe_row_text(raw.get("code"))
                    if family == "mbppplus"
                    else _safe_row_text(raw.get("prompt"))
                    + "\n"
                    + _safe_row_text(raw.get("canonical_solution"))
                ),
                test_source=_safe_row_text(raw.get("test")),
                entry_point=_entry_point(raw),
                harness_id="evalplus.check",
                harness_version=f"evalplus@{revision.rsplit('@', 1)[-1]}",
            )
            if row is not None:
                key = (row.source, str(row.grader.get("source_task_id")))
                if key in failed_executor_keys:
                    rejections.append({"source": row.source, "task_id": key[1], "reason": "prior_sandbox_executor_failure"})
                else:
                    candidates[family].append(row)
            else:
                rejections.append({"source": source, **evidence})

    selected: list[StudyPrompt] = []
    used_hashes = set(reserved_hashes)
    source_plan = {"bigcodebench": 200, "humanevalplus": 50, "mbppplus": 50}
    for family, count in source_plan.items():
        ordered = sorted(candidates[family], key=lambda row: (-int(row.grader["difficulty_screening"]["score"]), row.prompt_id))
        if len(ordered) < count:
            raise RuntimeError(f"{family} has only {len(ordered)} usable rows; need {count}")
        for row in ordered:
            if row.normalized_hash in used_hashes:
                continue
            selected.append(row)
            used_hashes.add(row.normalized_hash)
            if sum(1 for value in selected if value.source == row.source) >= count:
                break
        if sum(1 for value in selected if value.source == ordered[0].source) < count:
            raise RuntimeError(
                f"{family} does not have {count} unique rows after normalization"
            )
    selected = _split_assign(selected, TARGET_DETERMINISTIC_SPLITS)
    report = {
        "candidates_by_source": {name: len(values) for name, values in candidates.items()},
        "selected_by_source": dict(Counter(row.source for row in selected)),
        "accepted_count": len(selected),
        "accepted_splits": dict(Counter(row.split for row in selected)),
        "sandbox_validation": {
            "available": sandbox_available(),
            "executor_id": SANDBOX_EXECUTOR_ID,
            "policy_version": SANDBOX_POLICY_VERSION,
            "canonical_solutions_executed": bool(executor_report),
            "validation_report": str(executor_report) if executor_report else None,
            "validation_status": executor_validation.get("status") if executor_validation else None,
            "accepted_code_rows": executor_validation.get("accepted_code_rows") if executor_validation else None,
            "passed_count": executor_validation.get("passed_count") if executor_validation else None,
            "failed_count": executor_validation.get("failed_count") if executor_validation else None,
            "prior_failed_rows_excluded": len(failed_executor_keys),
        },
        "rejections_count": len(rejections),
        "sample_rejections": rejections[:20],
    }
    return selected, report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _anchor_protocol(raw: dict[str, Any]) -> str:
    protocols = [str(value) for value in raw.get("protocols") or [] if str(value)]
    preferred = "BrowseComp-Plus::standard::AgentIR::no-CM"
    if preferred in protocols:
        return preferred
    return sorted(protocols)[0] if protocols else ""


def _deterministic_general_replacements(
    anchor_path: Path, *, reserved_hashes: set[str]
) -> tuple[list[StudyPrompt], dict[str, Any]]:
    """Select unused public-anchor reference tasks without importing outcomes."""

    raw_rows = _read_jsonl(anchor_path)
    candidates_by_family: dict[str, list[StudyPrompt]] = defaultdict(list)
    used_hashes = set(reserved_hashes)
    skipped = 0
    for raw in raw_rows:
        prompt = _normalized(str(raw.get("prompt") or ""))
        reference = _normalized(str(raw.get("reference_answer") or ""))
        family = str(raw.get("benchmark_family") or "")
        task_id = str(raw.get("prompt_id") or raw.get("id") or "")
        protocol = _anchor_protocol(raw)
        if not prompt or not reference or not family or not task_id or not protocol:
            skipped += 1
            continue
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if digest in used_hashes:
            skipped += 1
            continue
        used_hashes.add(digest)
        candidates_by_family[family].append(
            StudyPrompt(
                prompt_id=f"anchor-replacement:{task_id}",
                prompt=prompt,
                split="train",
                source=ANCHOR_SOURCE,
                benchmark_family=family,
                domain="research",
                template_family=f"anchor:{family}",
                collection_vintage=f"dataset_revision:{ANCHOR_REVISION}",
                license_name="Apache-2.0",
                license_status="public",
                source_revision=f"{ANCHOR_SOURCE}@{ANCHOR_REVISION}",
                provenance_url=ANCHOR_URL,
                lane="general",
                label_type="deterministic",
                task_format="search_research",
                protocol_id=protocol,
                leakage_group=f"{ANCHOR_SOURCE}:replacement:{task_id}",
                tags=("search-research", "deterministic-general-replacement"),
                grader={
                    "expected_properties": [reference],
                    "answer_type": "reference_answer_contains",
                    "protocol_isolated": True,
                    "protocol_id": protocol,
                    "public_anchor_outcomes_used_as_candidate_outcomes": False,
                    "source_task_id": task_id,
                    "replacement_role": "deterministic_general_for_reduced_tau2",
                },
            )
        )

    selected: list[StudyPrompt] = []
    family_counts: Counter[str] = Counter()
    for family, quotas in GENERAL_REPLACEMENT_FAMILY_QUOTAS.items():
        candidates = sorted(candidates_by_family.get(family, []), key=lambda row: _stable(row.prompt_id))
        required = sum(quotas.values())
        if len(candidates) < required:
            raise RuntimeError(
                f"anchor family {family} has only {len(candidates)} unused rows; need {required}"
            )
        family_rows = _split_assign(candidates[:required], quotas)
        selected.extend(family_rows)
        family_counts[family] = len(family_rows)
    if len(selected) != sum(GENERAL_REPLACEMENT_SPLITS.values()):
        raise RuntimeError(f"selected {len(selected)} general replacements; expected 200")
    return selected, {
        "source": ANCHOR_SOURCE,
        "revision": ANCHOR_REVISION,
        "provenance_url": ANCHOR_URL,
        "available_task_count": len(raw_rows),
        "accepted_count": len(selected),
        "accepted_splits": dict(Counter(row.split for row in selected)),
        "accepted_by_family": dict(sorted(family_counts.items())),
        "protocols": sorted({row.protocol_id for row in selected}),
        "outcomes_imported": False,
        "observed_candidate_model_disagreement": False,
        "skipped_rows": skipped,
        "decision": "accepted_deterministic_replacement_only",
    }


def _bfcl_rows(
    bfcl_cache: Path, *, reserved_hashes: set[str]
) -> tuple[list[StudyPrompt], dict[str, Any]]:
    categories = (
        ("simple_python", "simple", 25),
        ("multiple", "multiple", 25),
        ("parallel", "parallel", 25),
        ("irrelevance", "irrelevance", 25),
    )
    selected: list[StudyPrompt] = []
    used_hashes = set(reserved_hashes)
    inventory: dict[str, int] = {}
    for file_stem, mode, count in categories:
        source_file = bfcl_cache / f"BFCL_v4_{file_stem}.json"
        answer_file = bfcl_cache / "possible_answer" / f"BFCL_v4_{file_stem}.json"
        tasks = _read_jsonl(source_file)
        answers = {str(row.get("id")): row for row in _read_jsonl(answer_file)} if answer_file.exists() else {}
        inventory[file_stem] = len(tasks)
        ordered = sorted(tasks, key=lambda row: _stable(str(row.get("id"))))
        if len(ordered) < count:
            raise RuntimeError(f"BFCL {file_stem} has only {len(ordered)} rows")
        category_rows: list[StudyPrompt] = []
        for raw in ordered:
            task_id = str(raw.get("id"))
            answer = answers.get(task_id, {})
            ground_truth = answer.get("ground_truth", []) if mode != "irrelevance" else []
            schemas = list(raw.get("function") or [])
            if not schemas:
                raise RuntimeError(f"BFCL row {task_id} lacks function definitions")
            candidate = StudyPrompt(
                    prompt_id=f"bfcl:{task_id}",
                    prompt=_normalized(str(raw.get("question") or "")),
                    split="train",
                    source="bfcl-v4",
                    benchmark_family=f"bfcl-v4:{mode}",
                    domain="tool-use",
                    template_family=f"bfcl-v4:{mode}",
                    collection_vintage=SOURCE_DATE,
                    license_name="Apache-2.0",
                    license_status="approved",
                    source_revision=f"ShishirPatil/gorilla@{BFCL_REVISION}",
                    provenance_url="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard",
                    lane="code",
                    label_type="tool_state",
                    task_format="tool_use",
                    leakage_group=f"bfcl-v4:{task_id}",
                    tags=("code", "tool-use", "ambiguous"),
                    grader={
                        "kind": "bfcl_official_checker",
                        "adapter_version": BFCL_ADAPTER_VERSION,
                        "executor_id": "bfcl_eval.eval_checker.eval_runner",
                        "executor_version": f"gorilla@{BFCL_REVISION}",
                        "evidence_policy": "official_checker_result_only",
                        "official_checker_id": BFCL_OFFICIAL_CHECKER_ID,
                        "official_checker_version": f"gorilla@{BFCL_REVISION}",
                        "official_checker_available": True,
                        "official_checker_smoke": "simple,multiple,parallel,irrelevance",
                        "tool_schemas": schemas,
                        "function_definitions": schemas,
                        "ground_truth": ground_truth,
                        "ground_truth_preservation": "complete_sequence; never ground_truth[0]",
                        "task_mode": mode,
                        "call_order_preserved": True,
                        "parallel_calls_preserved": True,
                        "source_task_id": task_id,
                        "source_file_sha256": _sha256(source_file),
                        "possible_answer_file_sha256": _sha256(answer_file) if answer_file.exists() else None,
                        "collection_eligible": True,
                        "token_limit_policy": {
                            "expected_output_tokens": 128,
                            "max_output_tokens": 256,
                            "source": "bfcl.function_schema_and_call_count.v1",
                        },
                    },
                    max_output_tokens=256,
                    expected_output_tokens=128,
                )
            if candidate.normalized_hash in used_hashes:
                continue
            used_hashes.add(candidate.normalized_hash)
            category_rows.append(candidate)
            if len(category_rows) == count:
                break
        if len(category_rows) != count:
            raise RuntimeError(
                f"BFCL {file_stem} has only {len(category_rows)} unique rows after normalization"
            )
        selected.extend(category_rows)
    result = _split_assign(selected, TARGET_BFCL_SPLITS)
    return result, {
        "inventory_by_category": inventory,
        "accepted_count": len(result),
        "accepted_splits": dict(Counter(row.split for row in result)),
        "task_modes": dict(Counter(row.grader["task_mode"] for row in result)),
        "official_checker": {
            "id": BFCL_OFFICIAL_CHECKER_ID,
            "revision": BFCL_REVISION,
            "smoke_fixtures": ["simple", "multiple", "parallel", "irrelevance"],
            "provider_calls": 0,
        },
    }


def _tau_task_files(tau_source: Path) -> dict[str, Path]:
    root = tau_source / "data" / "tau2" / "domains"
    return {
        name: root / name / "tasks.json"
        for name in ("mock", "airline", "retail", "telecom", "banking_knowledge")
    }


def _tau_action_metadata(raw: dict[str, Any]) -> dict[str, int | str]:
    """Derive turn/token limits from the official action reference."""

    criteria = raw.get("evaluation_criteria")
    actions = criteria.get("actions") if isinstance(criteria, dict) else ()
    actions = actions if isinstance(actions, list) else []
    action_count = len(actions)
    argument_chars = sum(
        len(json.dumps(action.get("arguments") or {}, ensure_ascii=False, sort_keys=True))
        for action in actions
        if isinstance(action, dict)
    )
    expected_turns = max(1, action_count)
    max_turns = max(expected_turns, action_count + 1)
    if action_count <= 2 and argument_chars <= 512:
        expected_tokens, max_tokens = 128, 384
    elif action_count <= 5 and argument_chars <= 1_500:
        expected_tokens, max_tokens = 192, 512
    elif action_count <= MAX_TAU_REFERENCE_ACTIONS and argument_chars <= 3_000:
        expected_tokens, max_tokens = 256, 768
    else:
        expected_tokens, max_tokens = 384, 1_024
    return {
        "reference_action_count": action_count,
        "reference_action_argument_chars": argument_chars,
        "expected_turns": expected_turns,
        "max_turns": max_turns,
        "expected_output_tokens": expected_tokens,
        "max_output_tokens": max_tokens,
        "turn_limit_source": "tau2.evaluation_criteria.actions_count.v1",
        "token_limit_source": "tau2.evaluation_criteria.action_arguments.v1",
    }


def _tau_rows(
    tau_source: Path, *, reserved_hashes: set[str]
) -> tuple[list[StudyPrompt], dict[str, Any]]:
    files = _tau_task_files(tau_source)
    tasks_by_domain: dict[str, list[dict[str, Any]]] = {}
    for domain, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        tasks_by_domain[domain] = _read_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))
    # The 240 train/validation rows are drawn from official base domains.
    # The 60 test rows are from one held-out full-telecom state family.
    # Tau tasks can legitimately share a user ticket while differing in
    # hidden environment state.  That state is a trial identity for grading,
    # not a router-visible feature.  The explicit repeated-trial contract
    # below makes the later trainer aggregate these trials by visible input.
    base_rows: list[tuple[str, dict[str, Any]]] = []
    for domain in ("mock", "airline", "retail", "telecom"):
        base_rows.extend(
            (domain, task)
            for task in tasks_by_domain[domain]
            if _tau_action_metadata(task)["reference_action_count"]
            <= MAX_TAU_REFERENCE_ACTIONS
        )
    base_rows.extend(
        ("banking_knowledge", task)
        for task in tasks_by_domain["banking_knowledge"]
        if _tau_action_metadata(task)["reference_action_count"]
        <= MAX_TAU_REFERENCE_ACTIONS
    )
    base_rows = sorted(
        base_rows, key=lambda item: _stable(f"{item[0]}:{item[1].get('id')}"))
    test_ticket = " ".join(
        str(
            next(
                task.get("ticket")
                for task in tasks_by_domain["telecom"]
                if "mms_issue" in str(task.get("id"))
                and "abroad" in str(task.get("id"))
            )
        ).split()
    )
    test_ticket_hash = hashlib.sha256(test_ticket.encode("utf-8")).hexdigest()
    non_telecom = [item for item in base_rows if item[0] != "telecom"]
    val_rows: list[tuple[str, dict[str, Any]]] = []
    val_hashes: set[str] = set()
    for item in non_telecom:
        ticket = _tau_prompt(item[1])
        ticket_hash = hashlib.sha256(_normalized(ticket).encode("utf-8")).hexdigest()
        if ticket_hash in reserved_hashes or ticket_hash in val_hashes:
            continue
        val_rows.append(item)
        val_hashes.add(ticket_hash)
        if len(val_rows) == TARGET_TAU_SPLITS["validation"]:
            break
    if len(val_rows) != TARGET_TAU_SPLITS["validation"]:
        raise RuntimeError(
            "tau base domains cannot provide the required unique validation ticket groups"
        )
    train_pool = [
        item
        for item in base_rows
        if hashlib.sha256(_normalized(_tau_prompt(item[1])).encode("utf-8")).hexdigest()
        not in val_hashes | {test_ticket_hash} | set(reserved_hashes)
    ]
    if len(train_pool) < TARGET_TAU_SPLITS["train"]:
        raise RuntimeError(
            "tau base domains cannot provide the required train tasks after group isolation"
        )
    selected_train = train_pool[: TARGET_TAU_SPLITS["train"]]
    full_rows = [
        ("telecom_full", task)
        for task in json.loads(files["telecom"].read_text(encoding="utf-8"))
        if hashlib.sha256(_normalized(_tau_prompt(task)).encode("utf-8")).hexdigest()
        == test_ticket_hash
        and _tau_action_metadata(task)["reference_action_count"]
        <= MAX_TAU_REFERENCE_ACTIONS
    ]
    full_rows = sorted(full_rows, key=lambda item: _stable(f"telecom_full:{item[1].get('id')}"))
    if len(full_rows) < TARGET_TAU_SPLITS["test"]:
        raise RuntimeError("tau full telecom held-out family cannot provide the required test tasks")
    selected_full = full_rows[: TARGET_TAU_SPLITS["test"]]
    selected_base = selected_train + val_rows
    train_keys = {f"{domain}:{task.get('id')}" for domain, task in selected_train}
    val_keys = {f"{domain}:{task.get('id')}" for domain, task in val_rows}
    result: list[StudyPrompt] = []
    selected_state_ids: set[str] = set()
    base_prompts: list[StudyPrompt] = []
    full_prompts: list[StudyPrompt] = []
    for domain, raw in selected_base + selected_full:
        task_id = str(raw.get("id") or "")
        ticket = _tau_prompt(raw)
        if not ticket.strip() or not raw.get("evaluation_criteria"):
            raise RuntimeError(f"tau task {domain}:{task_id} lacks ticket or evaluation criteria")
        official_domain = "telecom" if domain == "telecom_full" else domain
        task_file = files.get(official_domain, files["telecom"])
        action_metadata = _tau_action_metadata(raw)
        candidate = StudyPrompt(
                prompt_id=f"tau2:{domain}:{task_id}",
                prompt=_normalized(ticket),
                split="train",
                source="tau2-bench",
                benchmark_family=f"tau2-bench:{domain}",
                domain=official_domain,
                template_family=f"tau2:{domain}",
                collection_vintage=SOURCE_DATE,
                license_name="MIT",
                license_status="approved",
                source_revision=f"sierra-research/tau2-bench@{TAU_REVISION}",
                provenance_url="https://github.com/sierra-research/tau2-bench",
                lane="general",
                label_type="tool_state",
                task_format="agent_state",
                leakage_group=f"tau2:{domain}:{task_id}",
                tags=("tool-use", "agent-state"),
                grader={
                    "kind": "tau2_official_environment",
                    "adapter_version": TAU_ADAPTER_VERSION,
                    "executor_id": "tau2.evaluate_simulation",
                    "executor_version": f"tau2=={TAU_PACKAGE_VERSION}; repo@{TAU_REVISION}",
                    "evidence_policy": "official_state_evaluator_reward_only",
                    "official_environment_id": f"tau2.domain.{official_domain}",
                    "official_environment_available": True,
                    "official_evaluator_id": "tau2.evaluator.evaluate_simulation",
                    "official_evaluator_version": f"tau2=={TAU_PACKAGE_VERSION}",
                    "environment_constructor": f"tau2.domains.{official_domain}",
                    "persist_state": True,
                    "interactive_agent_loop": True,
                    "serialized_scenario_is_not_execution": True,
                    "official_task_set": domain,
                    "official_task_id": task_id,
                    "task_payload": raw,
                    "task_file_sha256": _sha256(task_file),
                    "collection_eligible": True,
                    **action_metadata,
                    "token_limit_policy": {
                        "expected_output_tokens": action_metadata["expected_output_tokens"],
                        "max_output_tokens": action_metadata["max_output_tokens"],
                        "source": action_metadata["token_limit_source"],
                    },
                },
                max_output_tokens=int(action_metadata["max_output_tokens"]),
                expected_output_tokens=int(action_metadata["expected_output_tokens"]),
            )
        state_identity = hashlib.sha256(
            json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        candidate.grader["state_identity"] = state_identity
        candidate.grader["stateful_duplicate_prompt_allowed"] = True
        candidate.grader["repeated_trial_contract"] = REPEATED_TRIAL_GROUP_CONTRACT
        candidate.grader["router_input_group"] = candidate.normalized_hash
        candidate.grader["effective_sample_unit"] = "normalized_prompt"
        if state_identity in selected_state_ids:
            continue
        selected_state_ids.add(state_identity)
        (full_prompts if domain == "telecom_full" else base_prompts).append(candidate)
    if len(base_prompts) != sum(TARGET_TAU_SPLITS.values()) - TARGET_TAU_SPLITS["test"] or len(full_prompts) != TARGET_TAU_SPLITS["test"]:
        raise RuntimeError(
            f"tau unique rows after normalization: base={len(base_prompts)} full={len(full_prompts)}"
        )
    # All 300 accepted rows are real tau rows.  The removed 200 rows are
    # replaced by deterministic general reference-answer tasks elsewhere.
    train_rows = []
    validation_rows = []
    for row in base_prompts:
        state_key = f"{row.grader['official_task_set']}:{row.grader['official_task_id']}"
        if state_key in train_keys:
            train_rows.append(
                StudyPrompt.from_dict({**row.to_dict(), "split": "train"}, require_frozen_hash=True)
            )
        elif state_key in val_keys:
            validation_rows.append(
                StudyPrompt.from_dict(
                    {**row.to_dict(), "split": "validation"}, require_frozen_hash=True
                )
            )
    if (
        len(train_rows) != TARGET_TAU_SPLITS["train"]
        or len(validation_rows) != TARGET_TAU_SPLITS["validation"]
    ):
        raise RuntimeError(
            f"tau split assignment lost rows: train={len(train_rows)} validation={len(validation_rows)}"
        )
    result = train_rows + validation_rows
    result.extend(
        StudyPrompt.from_dict({**row.to_dict(), "split": "test"}, require_frozen_hash=True)
        for row in full_prompts
    )
    return result, {
        "official_inventory_by_task_set": {
            "mock": len(tasks_by_domain["mock"]),
            "airline": len(tasks_by_domain["airline"]),
            "retail": len(tasks_by_domain["retail"]),
            "telecom_base": len(tasks_by_domain["telecom"]),
            "banking_knowledge": len(tasks_by_domain["banking_knowledge"]),
            "telecom_full": len(json.loads(files["telecom"].read_text(encoding="utf-8"))),
        },
        "accepted_count": len(result),
        "accepted_splits": dict(Counter(row.split for row in result)),
        "target_addition_splits": TARGET_TAU_SPLITS,
        "pairwise_replacement_splits": PAIRWISE_REPLACEMENT_SPLITS,
            "reduced_from_count": 500,
        "selection_rule": {
            "max_reference_actions": MAX_TAU_REFERENCE_ACTIONS,
            "turn_limits": "expected=max(1, reference_action_count); max=reference_action_count+1",
            "token_limits": "serialized official action-argument size and action count; max 1024",
            "repeated_trial_contract": REPEATED_TRIAL_GROUP_CONTRACT,
            "router_input_group": "normalized prompt SHA-256; hidden state is never added to the prompt",
            "effective_sample_size": "unique normalized router-visible inputs after grouped aggregation",
        },
        "official_environment": {
            "package": f"tau2=={TAU_PACKAGE_VERSION}",
            "revision": TAU_REVISION,
            "interactive_state_smoke": "passed_reward_1.0_mock",
            "provider_calls": 0,
        },
    }


def _recompute_tags(rows: list[StudyPrompt]) -> list[StudyPrompt]:
    train_values = {
        field: {getattr(row, field) for row in rows if row.split == "train"}
        for field in ("source", "domain", "benchmark_family", "template_family")
    }
    anchor_rows = [row for row in rows if row.source == "i-DeepSearch/observation-masking-eval-logs"]
    disagreement_proxy = {
        row.prompt_id
        for row in sorted(anchor_rows, key=lambda row: _stable(f"public-anchor-proxy-v1:{row.prompt_id}"))[:100]
    }
    specialist = {
        row.prompt_id
        for row in sorted(
            (row for row in rows if row.source == "cais/mmlu" and row.domain not in {"general", "miscellaneous"}),
            key=lambda row: _stable(row.prompt_id),
        )[:100]
    }
    short_hard = {
        row.prompt_id
        for row in sorted(
            (row for row in rows if row.source == "TIGER-Lab/MMLU-Pro"),
            key=lambda row: _stable(row.prompt_id),
        )[:100]
    }
    long_easy = {
        row.prompt_id
        for row in sorted(
            (row for row in rows if len(row.prompt) >= 500),
            key=lambda row: _stable(row.prompt_id),
        )[:100]
    }
    result: list[StudyPrompt] = []
    for row in rows:
        tags = {
            tag
            for tag in row.tags
            if tag not in {"ood", "representative", "disagreement-candidate", "short-hard", "long-easy", "specialist"}
        }
        if row.prompt_id in disagreement_proxy:
            tags.add("disagreement-candidate")
        if row.prompt_id in specialist:
            tags.add("specialist")
        if row.prompt_id in short_hard:
            tags.add("short-hard")
        if row.prompt_id in long_easy:
            tags.add("long-easy")
        if row.split == "test":
            unseen = any(getattr(row, field) not in train_values[field] for field in train_values)
            tags.add("ood" if unseen else "representative")
        result.append(StudyPrompt.from_dict({**row.to_dict(), "tags": sorted(tags)}, require_frozen_hash=True))
    return result


def _source_report(
    *,
    code_report: dict[str, Any],
    bfcl_report: dict[str, Any],
    tau_report: dict[str, Any],
    general_report: dict[str, Any],
    official_executor_smoke: dict[str, Any],
    final_rows: list[StudyPrompt],
    old_pairwise_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": SOURCE_DATE,
        "provider_calls": 0,
        "provider_spend_usd": 0.0,
        "target_additions": {
            "total": 700,
            "deterministic_code": 300,
            "code_tool_state": 100,
            "general_agent_state": 300,
            "deterministic_general_replacement": 200,
            "gross_rows_after_pairwise_removal": 900,
            "splits": {
                "train": {"total": 420, "code": 240},
                "validation": {"total": 140, "code": 80},
                "test": {"total": 140, "code": 80},
            },
        },
        "pairwise_replacement": {
            "removed_unusable_rows": old_pairwise_count,
            "replacement_rows": 200,
            "replacement_splits": PAIRWISE_REPLACEMENT_SPLITS,
            "policy": "raw RouteLLM pairwise rows have no reference-success label; they are excluded, never coerced to success/failure",
        },
        "tau2_reduction": {
            "previous_count": 500,
            "current_count": 300,
            "removed_count": 200,
            "current_splits": TARGET_TAU_SPLITS,
            "replacement_splits": GENERAL_REPLACEMENT_SPLITS,
            "router_visible_duplicate_policy": {
                "contract": REPEATED_TRIAL_GROUP_CONTRACT,
                "raw_trials_remain_separate_for_collection_and_grading": True,
                "training_and_evaluation_effective_sample_unit": "normalized_prompt",
                "hidden_state_is_not_added_to_prompt": True,
            },
        },
        "sources": [
            {
                "candidate": "BigCodeBench v0.1.4",
                "official_url": "https://github.com/bigcode-project/bigcodebench",
                "dataset_url": "https://huggingface.co/datasets/bigcode/bigcodebench",
                "license": "Apache-2.0",
                "revision": f"dataset@{BIGCODE_REVISION}; repo@{BIGCODE_REPO_REVISION}",
                "available_task_count": 1140,
                "proposed_accepted_count": 200,
                "accepted_splits": {"train": 120, "validation": 40, "test": 40},
                "task_type": "function implementation",
                "lane": "code",
                "grader_executor": "sandboxed_code_tests / bigcodebench.unittest",
                "sandbox_requirements": "macOS sandbox-exec, network deny, read-only base, private writable task directory, limits, clean environment",
                "contamination_risk": "no per-item publication date supplied; pinned revision and task-id grouping retained",
                "decision": "accepted",
            },
            {
                "candidate": "EvalPlus HumanEval+",
                "official_url": "https://github.com/evalplus/evalplus",
                "dataset_url": "https://huggingface.co/datasets/evalplus/humanevalplus",
                "license": "Apache-2.0",
                "revision": HUMANEVAL_REVISION,
                "available_task_count": 164,
                "proposed_accepted_count": 50,
                "accepted_splits": {"train": 30, "validation": 10, "test": 10},
                "task_type": "function implementation with extended tests",
                "lane": "code",
                "grader_executor": "sandboxed_code_tests / evalplus.check",
                "sandbox_requirements": "same isolated Python policy; no host execution",
                "contamination_risk": "classic benchmark contamination risk; small supplemental share and pinned revision",
                "decision": "accepted_as_supplement",
            },
            {
                "candidate": "EvalPlus MBPP+",
                "official_url": "https://github.com/evalplus/evalplus",
                "dataset_url": "https://huggingface.co/datasets/evalplus/mbppplus",
                "license": "Apache-2.0",
                "revision": MBPP_REVISION,
                "available_task_count": 378,
                "proposed_accepted_count": 50,
                "accepted_splits": {"train": 30, "validation": 10, "test": 10},
                "task_type": "function implementation with extended tests",
                "lane": "code",
                "grader_executor": "sandboxed_code_tests / evalplus.check",
                "sandbox_requirements": "same isolated Python policy; no host execution",
                "contamination_risk": "classic benchmark contamination risk; small supplemental share and pinned revision",
                "decision": "accepted_as_supplement",
            },
            {
                "candidate": "LiveCodeBench",
                "official_url": "https://github.com/LiveCodeBench/LiveCodeBench",
                "license": "repository MIT; dataset card reports ambiguous `cc`",
                "revision": "HF revision verified during acquisition; not accepted",
                "available_task_count": 1055,
                "proposed_accepted_count": 0,
                "accepted_splits": {},
                "task_type": "generation, self-repair, execution, test prediction",
                "lane": "code",
                "grader_executor": "official checker available in repository",
                "sandbox_requirements": "would require the same isolated code sandbox",
                "contamination_risk": "date-stamped releases are useful, but redistribution license variant is not explicit enough",
                "decision": "rejected_ambiguous_dataset_license",
            },
            {
                "candidate": "SWE-bench Verified",
                "official_url": "https://github.com/SWE-bench/SWE-bench",
                "dataset_url": "https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified",
                "license": "official repository MIT; dataset card has no explicit dataset license",
                "revision": "HF revision verified during acquisition; not accepted",
                "available_task_count": 500,
                "proposed_accepted_count": 0,
                "accepted_splits": {},
                "task_type": "repository-level patch repair",
                "lane": "code",
                "grader_executor": "Docker repository harness, unavailable locally",
                "sandbox_requirements": "pinned Docker harness with network disabled and per-task resource limits",
                "contamination_risk": "human-validated and date-stamped fields exist; per-repository license audit remains incomplete",
                "decision": "rejected_license_and_sandbox_gap",
            },
            {
                "candidate": "BFCL v4",
                "official_url": "https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard",
                "license": "Apache-2.0",
                "revision": BFCL_REVISION,
                "available_task_count": sum(bfcl_report["inventory_by_category"].values()),
                "proposed_accepted_count": 100,
                "accepted_splits": TARGET_BFCL_SPLITS,
                "task_type": "structured tool selection; simple, multiple, parallel, irrelevance",
                "lane": "code capability/tool-use slice, not code execution",
                "grader_executor": "pinned official BFCL AST checker; structured provider adapter",
                "sandbox_requirements": "checker smoke ran with network denied and write isolation; candidate endpoint must support tool calls",
                "contamination_risk": "benchmark revision pinned; no candidate outcomes imported",
                "decision": "accepted_tool_state",
            },
            {
                "candidate": "i-DeepSearch observation-masking evaluated logs (reference-only rows)",
                "official_url": ANCHOR_URL,
                "license": "Apache-2.0",
                "revision": ANCHOR_REVISION,
                "available_task_count": general_report["available_task_count"],
                "proposed_accepted_count": general_report["accepted_count"],
                "accepted_splits": GENERAL_REPLACEMENT_SPLITS,
                "task_type": "deterministic search/research reference-answer tasks",
                "lane": "general",
                "grader_executor": "pinned reference answer containment; no public observations imported",
                "sandbox_requirements": "none beyond ordinary collector; no downloaded code executed",
                "contamination_risk": "public benchmark contamination possible; revision and prompt groups pinned",
                "decision": "accepted_deterministic_replacement_only",
            },
            {
                "candidate": "tau2-bench",
                "official_url": "https://github.com/sierra-research/tau2-bench",
                "license": "MIT",
                "revision": TAU_REVISION,
                "available_task_count": sum(tau_report["official_inventory_by_task_set"].values()),
                "proposed_accepted_count": 300,
                "accepted_splits": TARGET_TAU_SPLITS,
                "task_type": "interactive agent/tool/state trajectory",
                "lane": "general",
                "grader_executor": "pinned tau2 environment and official evaluate_simulation state evaluator",
                "sandbox_requirements": "official environment process with persisted state; network-denied acquisition smoke; candidate tool loop, not serialized prompt",
                "contamination_risk": "official task IDs and revision pinned; full telecom rows are held out by explicit source/template distance",
                "decision": "accepted_target_after_reduction",
            },
        ],
        "executor_results": {
            "sandbox": {
                "available": sandbox_available(),
                "executor_id": SANDBOX_EXECUTOR_ID,
                "policy_version": SANDBOX_POLICY_VERSION,
                "real_benchmark_execution_on_host": False,
            },
            "bfcl": bfcl_report["official_checker"],
            "tau2": tau_report["official_environment"],
            "deterministic_general_replacement": general_report,
            "code": code_report["sandbox_validation"],
            "official_executor_smoke": official_executor_smoke,
        },
        "accepted_distribution": {
            "rows": len(final_rows),
            "source": dict(Counter(row.source for row in final_rows)),
            "lane": {split: dict(Counter(row.lane for row in final_rows if row.split == split)) for split in ("train", "validation", "test")},
            "task_format": dict(Counter(row.task_format for row in final_rows)),
            "label": dict(Counter(row.label_type for row in final_rows)),
            "split": dict(Counter(row.split for row in final_rows)),
        },
        "ood_policy": {
            "version": OOD_POLICY_VERSION,
            "rule": "test row is OOD iff source, domain, benchmark_family, or template_family is absent from training values",
            "test_membership_alone_is_not_evidence": True,
        },
        "disagreement_candidate_boundary": {
            "tag": "disagreement-candidate",
            "basis": "public-anchor deterministic hash proxy v1",
            "selection_count": 100,
            "observed_candidate_model_disagreement": False,
            "must_not_be_described_as": "observed candidate-model disagreement",
        },
        "unresolved_data_gaps": [
            "No accepted code source with an explicit compatible license supplied repository-level repair tasks; SWE-bench remains excluded pending per-item license and sandbox review.",
            "No accepted code source with an explicit compatible license supplied a separate debugging/self-repair slice; LiveCodeBench remains excluded pending dataset-license clarification.",
            "Pairwise rows were removed and replaced; no pairwise judge calls are planned or priced in the active corpus.",
            "Authentic model-disagreement slice remains post-collection only and is absent from all training/tuning decisions.",
        ],
        "checksum": corpus_checksum(final_rows),
    }


def build_corpus(
    *,
    current_path: Path,
    code_cache: Path,
    bfcl_cache: Path,
    tau_source: Path,
    general_source: Path,
    code_executor_report: Path | None = None,
    code_executor_exclusions: Path | None = None,
    official_executor_smoke: Path | None = None,
) -> tuple[list[StudyPrompt], dict[str, Any]]:
    current = list(load_jsonl(current_path))
    old_pairwise = [row for row in current if row.label_type == "pairwise"]
    managed_existing = [
        row
        for row in current
        if row.task_format == "code_execution"
        or row.source == "bfcl-v4"
        or (row.source == "tau2-bench" and row.task_format == "agent_state")
    ]
    base = [
        row
        for row in current
        if row.label_type != "pairwise" and row not in managed_existing
    ]
    if len(base) != 1100:
        raise RuntimeError(
            "expected 1100 stable deterministic rows after removing managed "
            f"code/BFCL/tau rows, found {len(base)}; current={len(current)}, "
            f"managed={len(managed_existing)}, pairwise={len(old_pairwise)}"
        )
    reserved_hashes = {row.normalized_hash for row in base}
    code_rows, code_report = _acquire_code_sources(
        code_cache,
        reserved_hashes=reserved_hashes,
        executor_report=code_executor_report,
        executor_exclusions=code_executor_exclusions,
    )
    reserved_hashes.update(row.normalized_hash for row in code_rows)
    bfcl_rows, bfcl_report = _bfcl_rows(
        bfcl_cache, reserved_hashes=reserved_hashes
    )
    reserved_hashes.update(row.normalized_hash for row in bfcl_rows)
    general_rows, general_report = _deterministic_general_replacements(
        general_source, reserved_hashes=reserved_hashes
    )
    reserved_hashes.update(row.normalized_hash for row in general_rows)
    tau_rows, tau_report = _tau_rows(
        tau_source, reserved_hashes=reserved_hashes
    )
    rows = _recompute_tags(base + code_rows + bfcl_rows + general_rows + tau_rows)
    if len(rows) != 2000:
        raise RuntimeError(f"assembled {len(rows)} rows, expected 2000")
    audit = audit_corpus(rows)
    smoke_report = {}
    if official_executor_smoke is not None:
        smoke_report = json.loads(official_executor_smoke.read_text(encoding="utf-8"))
        if smoke_report.get("status") != "passed":
            raise RuntimeError("official executor smoke report is not passed")
    else:
        raise RuntimeError("official executor smoke report is required for corpus assembly")
    report = _source_report(
        code_report=code_report,
        bfcl_report=bfcl_report,
        tau_report=tau_report,
        general_report=general_report,
        official_executor_smoke=smoke_report,
        final_rows=rows,
        old_pairwise_count=len(old_pairwise),
    )
    report["audit_before_write"] = audit.to_dict()
    report["code_acquisition"] = code_report
    report["bfcl_acquisition"] = bfcl_report
    report["tau_acquisition"] = tau_report
    report["preserved_existing_replacement_rows"] = sum(
        1
        for row in base
        if row.grader.get("replacement_role") == "deterministic_general_for_removed_pairwise"
        or row.grader.get("replacement_role") == "deterministic_general_for_reduced_tau2"
    )
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--code-cache", type=Path, required=True)
    parser.add_argument("--bfcl-cache", type=Path, required=True)
    parser.add_argument("--tau-source", type=Path, required=True)
    parser.add_argument(
        "--general-source",
        type=Path,
        default=ROOT / "data/router-v2/observation-masking-anchors.jsonl",
    )
    parser.add_argument("--code-executor-report", type=Path)
    parser.add_argument("--code-executor-exclusions", type=Path)
    parser.add_argument("--official-executor-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows, report = build_corpus(
        current_path=args.current,
        code_cache=args.code_cache,
        bfcl_cache=args.bfcl_cache,
        tau_source=args.tau_source,
        general_source=args.general_source,
        code_executor_report=args.code_executor_report,
        code_executor_exclusions=args.code_executor_exclusions,
        official_executor_smoke=args.official_executor_smoke,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, args.output)
    report["output_path"] = str(args.output)
    report["output_sha256"] = _sha256(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["audit_before_write"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
