"""Leakage-safe Router V2 study corpus and call-plan contracts.

This module is intentionally provider-free.  It validates a locally supplied
corpus before any model call is possible, and it keeps source/licence metadata
with every prompt so a later report can distinguish accepted, rejected, and
unresolved records.

The builder does not download benchmark data.  Dataset terms and revisions are
too important to infer from a filename, and a gated or redistribution-
incompatible row must never enter a paid study by accident.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STUDY_SPLIT_COUNTS = {"train": 1_200, "validation": 400, "test": 400}
CORE_MODELS = (
    "nvidia/nemotron-3-super-120b-a12b:free",
    "xiaomi/mimo-v2.5",
    "openai/gpt-5.6-luna",
)
GROK_MODEL = "x-ai/grok-4.6"
GROK_TRAIN_COUNT = 200
LABEL_TYPES = frozenset({"deterministic", "tool_state", "pairwise"})
TASK_FORMATS = frozenset(
    {
        "multiple_choice",
        "exact_match",
        "code_execution",
        "tool_use",
        "search_research",
        "structured_output",
        "agent_state",
        "conversational_reasoning",
    }
)
# BFCL/tool selection is a capability slice, not code generation.  It may live
# in the code lane only as the explicitly labelled code-oriented tool-state
# slice; it must never satisfy the code_execution contract.
CODE_LANE_FORMATS = frozenset({"code_execution"})
TOOL_USE_FORMATS = frozenset({"tool_use"})
APPROVED_LICENSE_STATUSES = frozenset(
    {"approved", "permitted", "public", "public-domain", "internal"}
)
REQUIRED_STRATA = {
    "short-hard": 100,
    "long-easy": 100,
    "specialist": 100,
    "ambiguous": 100,
    # OOD is a source/domain-distance slice, not a test-membership label.  The
    # corpus contract requires a non-trivial held-out slice while allowing the
    # explicit source mix to determine its exact size.
    "ood": 40,
    "representative": 100,
    "disagreement-candidate": 100,
}
POST_COLLECTION_REQUIRED_STRATA = {"model-disagreement": 100}
INFORMATIVE_TAGS = frozenset(REQUIRED_STRATA)
OOD_POLICY_VERSION = "router-v2-ood-source-distance-v1"
GROK_SELECTION_ALGORITHM_VERSION = "router-v2-grok-informative-v2"
GROK_SELECTION_SEED = 17
REPEATED_TRIAL_GROUP_CONTRACT = "router-input-group-mean-v1"
MAX_SOURCE_FRACTION = 0.35
MAX_BENCHMARK_FRACTION = 0.35
MAX_TASK_FORMAT_FRACTION = 0.35
MIN_CODE_LANE_FRACTION = 0.20
MIN_TEST_OOD_ROWS = 40
MIN_TEST_REPRESENTATIVE_ROWS = 100

# These are preregistered study limits.  They are intentionally format-specific
# because one universal 256/512 budget would truncate code, research, and agent
# trajectories in a format-dependent way.
FORMAT_TOKEN_BUDGETS: dict[str, dict[str, int | str]] = {
    "multiple_choice": {
        "expected_output_tokens": 256,
        "max_output_tokens": 1024,
        "truncation_risk": "medium",
    },
    "exact_match": {
        "expected_output_tokens": 128,
        "max_output_tokens": 512,
        "truncation_risk": "low",
    },
    "code_execution": {
        "expected_output_tokens": 768,
        "max_output_tokens": 2048,
        "truncation_risk": "medium",
    },
    "tool_use": {
        "expected_output_tokens": 128,
        "max_output_tokens": 384,
        "truncation_risk": "low",
    },
    "search_research": {
        "expected_output_tokens": 768,
        "max_output_tokens": 2048,
        "truncation_risk": "high",
    },
    "structured_output": {
        "expected_output_tokens": 384,
        "max_output_tokens": 1024,
        "truncation_risk": "medium",
    },
    "agent_state": {
        "expected_output_tokens": 256,
        "max_output_tokens": 768,
        "truncation_risk": "medium",
    },
    "conversational_reasoning": {
        "expected_output_tokens": 384,
        "max_output_tokens": 1024,
        "truncation_risk": "medium",
    },
}


class StudyCorpusError(ValueError):
    """Raised when a corpus cannot be promoted to a paid study input."""


def normalize_prompt(prompt: str) -> str:
    return " ".join(str(prompt).strip().split())


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


def stable_fraction(value: str, seed: int = 17) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


@dataclass(frozen=True)
class StudyPrompt:
    """One prompt eligible for a Router V2 study.

    ``grader`` is a compact, deterministic grading contract.  It may contain
    an answer/answer_type pair, an expected JSON schema, or a trusted-executor
    label.  It must not contain credentials or raw private application state.
    """

    prompt_id: str
    prompt: str
    split: str
    source: str
    benchmark_family: str
    domain: str
    template_family: str
    collection_vintage: str
    license_name: str
    license_status: str
    source_revision: str
    provenance_url: str
    lane: str
    label_type: str
    task_format: str = "conversational_reasoning"
    leakage_group: str = ""
    protocol_id: str = ""
    tags: tuple[str, ...] = ()
    grader: dict[str, Any] = field(default_factory=dict)
    max_output_tokens: int | None = None
    expected_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.split not in STUDY_SPLIT_COUNTS:
            raise ValueError(f"invalid study split {self.split!r}")
        if self.lane not in {"general", "code"}:
            raise ValueError(f"invalid study lane {self.lane!r}")
        if self.label_type not in LABEL_TYPES:
            raise ValueError(f"invalid label type {self.label_type!r}")
        if self.task_format not in TASK_FORMATS:
            raise ValueError(f"invalid task format {self.task_format!r}")
        if not normalize_prompt(self.prompt):
            raise ValueError("study prompt cannot be empty")
        if not self.leakage_group:
            object.__setattr__(
                self,
                "leakage_group",
                "|".join(
                    (
                        self.source,
                        self.domain,
                        self.template_family,
                        self.collection_vintage,
                    )
                ),
            )
        if not isinstance(self.grader, dict):
            raise ValueError("study grader must be an object")
        budget = FORMAT_TOKEN_BUDGETS[self.task_format]
        if self.max_output_tokens is None:
            object.__setattr__(
                self, "max_output_tokens", int(budget["max_output_tokens"])
            )
        if self.expected_output_tokens is None:
            object.__setattr__(
                self,
                "expected_output_tokens",
                int(budget["expected_output_tokens"]),
            )
        if self.max_output_tokens < 1 or self.expected_output_tokens < 1:
            raise ValueError("study token budgets must be positive")
        if self.expected_output_tokens > self.max_output_tokens:
            raise ValueError("expected output tokens cannot exceed the maximum")
        token_policy = self.grader.get("token_limit_policy")
        if token_policy is not None:
            if not isinstance(token_policy, dict):
                raise ValueError("token_limit_policy must be an object")
            if (
                int(token_policy.get("expected_output_tokens", -1))
                != self.expected_output_tokens
                or int(token_policy.get("max_output_tokens", -1))
                != self.max_output_tokens
                or not str(token_policy.get("source") or "").strip()
            ):
                raise ValueError(
                    "row token limits must match the declared token_limit_policy"
                )
        if self.label_type == "deterministic":
            has_answer = self.grader.get("answer") is not None
            has_properties = bool(self.grader.get("expected_properties"))
            has_schema = isinstance(self.grader.get("json_schema"), dict)
            has_executable_code_contract = (
                self.task_format == "code_execution"
                and str(self.grader.get("kind") or "") == "sandboxed_code_tests"
                and bool(self.grader.get("executable_tests"))
                and bool(str(self.grader.get("test_harness_id") or "").strip())
                and bool(str(self.grader.get("test_harness_version") or "").strip())
            )
            if not (has_answer or has_properties or has_schema or has_executable_code_contract):
                raise ValueError(
                    "deterministic study rows require an answer, expected_properties, json_schema, or sandboxed code tests"
                )
        elif self.label_type == "tool_state":
            required = ("kind", "executor_id", "executor_version", "evidence_policy")
            if any(not str(self.grader.get(key) or "").strip() for key in required):
                raise ValueError(
                    "tool_state study rows require kind, executor_id, executor_version, and evidence_policy"
                )
        elif self.label_type == "pairwise":
            required = (
                "kind",
                "judge_id",
                "judge_version",
                "criteria_version",
                "position_randomization",
                "sampling_settings",
                "tie_policy",
                "abstention_policy",
                "disagreement_review_policy",
                "judging_cost_status",
            )
            if any(not str(self.grader.get(key) or "").strip() for key in required):
                raise ValueError(
                    "pairwise study rows require judge identity, criteria, position randomization, and sampling settings"
                )

    @property
    def normalized_hash(self) -> str:
        return prompt_hash(self.prompt)

    @property
    def group_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.source,
            self.domain,
            self.template_family,
            self.collection_vintage,
            self.leakage_group,
        )

    @property
    def informative_score(self) -> int:
        tags = set(self.tags)
        return sum(tag in tags for tag in INFORMATIVE_TAGS) + int(
            "informative" in tags
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "prompt_id": self.prompt_id,
            "prompt": normalize_prompt(self.prompt),
            "split": self.split,
            "source": self.source,
            "benchmark_family": self.benchmark_family,
            "domain": self.domain,
            "template_family": self.template_family,
            "collection_vintage": self.collection_vintage,
            "license": self.license_name,
            "license_status": self.license_status,
            "source_revision": self.source_revision,
            "provenance_url": self.provenance_url,
            "lane": self.lane,
            "label_type": self.label_type,
            "task_format": self.task_format,
            "leakage_group": self.leakage_group,
            "protocol_id": self.protocol_id,
            "normalized_prompt_sha256": self.normalized_hash,
            "tags": list(self.tags),
            "grader": self.grader,
            "max_output_tokens": self.max_output_tokens,
            "expected_output_tokens": self.expected_output_tokens,
        }
        if self.grader.get("router_input_group"):
            result["router_input_group"] = self.grader["router_input_group"]
        if self.grader.get("repeated_trial_contract"):
            result["router_input_contract"] = self.grader["repeated_trial_contract"]
        return result

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, require_frozen_hash: bool = False
    ) -> StudyPrompt:
        if not isinstance(raw, dict):
            raise ValueError("study row must be an object")
        prompt = normalize_prompt(str(raw.get("prompt") or ""))
        source = str(raw.get("source") or "").strip()
        benchmark_family = str(
            raw.get("benchmark_family") or raw.get("source") or ""
        ).strip()
        stored_hash = raw.get("normalized_prompt_sha256")
        if stored_hash is None and require_frozen_hash:
            raise ValueError("frozen study row is missing normalized_prompt_sha256")
        if stored_hash is not None:
            stored_hash = str(stored_hash).strip().lower()
            if stored_hash != prompt_hash(prompt):
                raise ValueError("normalized_prompt_sha256 does not match prompt")
        grader = raw.get("grader")
        if not isinstance(grader, dict):
            grader = {}
        else:
            grader = dict(grader)
        if raw.get("router_input_group") and "router_input_group" not in grader:
            grader["router_input_group"] = raw.get("router_input_group")
        if raw.get("router_input_contract") and "repeated_trial_contract" not in grader:
            grader["repeated_trial_contract"] = raw.get("router_input_contract")
        # Accept the existing EvalItem-shaped fields while keeping the new
        # provenance contract explicit in the serialized output.
        if "answer" in raw and "answer" not in grader:
            grader["answer"] = raw.get("answer")
        if "answer_type" in raw and "answer_type" not in grader:
            grader["answer_type"] = raw.get("answer_type")
        if "expected_properties" in raw and "expected_properties" not in grader:
            grader["expected_properties"] = list(raw.get("expected_properties") or [])
        return cls(
            prompt_id=str(raw.get("prompt_id") or raw.get("id") or "").strip(),
            prompt=prompt,
            split=str(raw.get("split") or "").strip(),
            source=source,
            benchmark_family=benchmark_family,
            domain=str(raw.get("domain") or raw.get("category") or "").strip(),
            template_family=str(raw.get("template_family") or "").strip(),
            collection_vintage=str(
                raw.get("collection_vintage") or raw.get("collected_at") or ""
            ).strip(),
            license_name=str(raw.get("license") or raw.get("license_name") or "").strip(),
            license_status=str(raw.get("license_status") or "").strip().lower(),
            source_revision=str(
                raw.get("source_revision") or raw.get("revision") or ""
            ).strip(),
            provenance_url=str(raw.get("provenance_url") or raw.get("source_url") or "").strip(),
            lane=str(raw.get("lane") or "general").strip(),
            label_type=str(raw.get("label_type") or "deterministic").strip(),
            task_format=str(
                raw.get("task_format") or "conversational_reasoning"
            ).strip(),
            leakage_group=str(raw.get("leakage_group") or "").strip(),
            protocol_id=str(raw.get("protocol_id") or "").strip(),
            tags=tuple(sorted({str(tag).strip() for tag in raw.get("tags", []) if str(tag).strip()})),
            grader=grader,
            max_output_tokens=(
                max(1, int(raw["max_output_tokens"]))
                if raw.get("max_output_tokens") is not None
                else None
            ),
            expected_output_tokens=(
                max(1, int(raw["expected_output_tokens"]))
                if raw.get("expected_output_tokens") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class CorpusAudit:
    total: int
    accepted: int
    split_counts: dict[str, int]
    lane_counts: dict[str, dict[str, int]]
    label_counts: dict[str, int]
    task_format_counts: dict[str, int]
    source_counts: dict[str, int]
    benchmark_counts: dict[str, int]
    stratum_counts: dict[str, int]
    test_stratum_counts: dict[str, int]
    rejected: tuple[dict[str, Any], ...] = ()
    leakage: tuple[dict[str, Any], ...] = ()
    missing_metadata: tuple[dict[str, Any], ...] = ()
    token_budget_issues: tuple[dict[str, Any], ...] = ()
    lane_contract_issues: tuple[dict[str, Any], ...] = ()
    grader_contract_issues: tuple[dict[str, Any], ...] = ()
    valid: bool = False

    @property
    def rejected_count(self) -> int:
        return len(self.rejected) + len(self.missing_metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "rejected_count": self.rejected_count,
            "split_counts": dict(sorted(self.split_counts.items())),
            "lane_counts": {
                split: dict(sorted(counts.items()))
                for split, counts in sorted(self.lane_counts.items())
            },
            "label_counts": dict(sorted(self.label_counts.items())),
            "task_format_counts": dict(sorted(self.task_format_counts.items())),
            "source_counts": dict(sorted(self.source_counts.items())),
            "benchmark_counts": dict(sorted(self.benchmark_counts.items())),
            "stratum_counts": dict(sorted(self.stratum_counts.items())),
            "test_stratum_counts": dict(sorted(self.test_stratum_counts.items())),
            "rejected": list(self.rejected),
            "leakage": list(self.leakage),
            "missing_metadata": list(self.missing_metadata),
            "token_budget_issues": list(self.token_budget_issues),
            "lane_contract_issues": list(self.lane_contract_issues),
            "grader_contract_issues": list(self.grader_contract_issues),
            "valid": self.valid,
        }


def is_genuine_code_task(row: StudyPrompt) -> bool:
    """Return whether a code-lane row has an approved executable contract."""

    if row.lane != "code":
        return False
    grader = row.grader
    kind = str(grader.get("kind") or "").lower()
    if row.task_format == "tool_use" and row.label_type == "tool_state":
        return (
            kind.startswith("bfcl")
            and grader.get("official_checker_available") is True
            and bool(str(grader.get("official_checker_id") or "").strip())
            and bool(str(grader.get("official_checker_version") or "").strip())
        )
    if row.task_format not in CODE_LANE_FORMATS or kind.startswith("bfcl"):
        return False
    return bool(
        str(grader.get("code_task_type") or "").strip()
        and grader.get("executable_tests")
        and str(grader.get("test_harness_id") or "").strip()
        and str(grader.get("test_harness_version") or "").strip()
    )


def _stateful_duplicate_prompt_allowed(row: StudyPrompt) -> bool:
    """Allow same ticket text only under the explicit repeated-trial contract.

    The state identity is evidence that two official environment trials are
    distinct.  It is deliberately not part of the router-visible prompt and it
    cannot by itself waive the grouped-input rule.
    """

    grader = row.grader
    return bool(
        row.label_type == "tool_state"
        and str(grader.get("kind") or "").lower() == "tau2_official_environment"
        and grader.get("stateful_duplicate_prompt_allowed") is True
        and str(grader.get("state_identity") or "").strip()
        and str(grader.get("repeated_trial_contract") or "")
        == REPEATED_TRIAL_GROUP_CONTRACT
        and str(grader.get("router_input_group") or "") == row.normalized_hash
    )


def format_token_budget(task_format: str) -> dict[str, int | str]:
    """Return a copy of the frozen token policy for one task format."""

    try:
        return dict(FORMAT_TOKEN_BUDGETS[task_format])
    except KeyError as exc:
        raise ValueError(f"no frozen token budget for {task_format!r}") from exc


def audit_corpus(
    prompts: Iterable[StudyPrompt],
    *,
    expected_counts: dict[str, int] | None = None,
    rejected: Iterable[dict[str, Any]] = (),
) -> CorpusAudit:
    """Audit counts, provenance, labels, strata, and grouped leakage."""

    rows = list(prompts)
    expected = expected_counts or STUDY_SPLIT_COUNTS
    split_counts = Counter(row.split for row in rows)
    lane_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        lane_counts[row.split][row.lane] += 1
    label_counts = Counter(row.label_type for row in rows)
    task_format_counts = Counter(row.task_format for row in rows)
    source_counts = Counter(row.source for row in rows)
    benchmark_counts = Counter(row.benchmark_family for row in rows)
    stratum_counts = Counter(tag for row in rows for tag in row.tags)
    test_stratum_counts = Counter(
        tag for row in rows if row.split == "test" for tag in row.tags
    )
    missing: list[dict[str, Any]] = []
    token_budget_issues: list[dict[str, Any]] = []
    lane_contract_issues: list[dict[str, Any]] = []
    grader_contract_issues: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    seen_hashes: dict[str, list[StudyPrompt]] = defaultdict(list)
    group_maps: dict[str, dict[str, set[str]]] = {
        "prompt_hash": defaultdict(set),
        "leakage_group": defaultdict(set),
    }
    for index, row in enumerate(rows):
        if not row.prompt_id:
            missing.append({"index": index, "reason": "missing_prompt_id"})
        elif row.prompt_id in seen_ids:
            missing.append(
                {"index": index, "reason": "duplicate_prompt_id", "prompt_id": row.prompt_id}
            )
        seen_ids[row.prompt_id] = index
        digest = row.normalized_hash
        prior_rows = seen_hashes[digest]
        if prior_rows:
            same_split = all(previous.split == row.split for previous in prior_rows)
            distinct_states = all(
                _stateful_duplicate_prompt_allowed(previous)
                and str(previous.grader.get("state_identity"))
                != str(row.grader.get("state_identity"))
                for previous in prior_rows
            )
            if not (
                _stateful_duplicate_prompt_allowed(row)
                and same_split
                and distinct_states
                and all(
                    str(previous.grader.get("router_input_group"))
                    == str(row.grader.get("router_input_group"))
                    and str(previous.grader.get("repeated_trial_contract"))
                    == REPEATED_TRIAL_GROUP_CONTRACT
                    for previous in prior_rows
                )
            ):
                missing.append(
                    {
                        "index": index,
                        "reason": "duplicate_prompt",
                        "prompt_id": row.prompt_id,
                        "prior_prompt_id": prior_rows[0].prompt_id,
                    }
                )
        seen_hashes[digest].append(row)
        metadata = {
            "source": row.source,
            "benchmark_family": row.benchmark_family,
            "domain": row.domain,
            "template_family": row.template_family,
            "collection_vintage": row.collection_vintage,
            "license": row.license_name,
            "license_status": row.license_status,
            "source_revision": row.source_revision,
            "provenance_url": row.provenance_url,
            "task_format": row.task_format,
            "leakage_group": row.leakage_group,
        }
        missing_fields = [name for name, value in metadata.items() if not value]
        if row.license_status not in APPROVED_LICENSE_STATUSES:
            missing_fields.append("license_status_not_approved")
        if missing_fields:
            missing.append(
                {
                    "index": index,
                    "prompt_id": row.prompt_id,
                    "reason": "missing_or_unapproved_metadata",
                    "fields": sorted(set(missing_fields)),
                }
            )
        budget = format_token_budget(row.task_format)
        row_policy = row.grader.get("token_limit_policy")
        if isinstance(row_policy, dict):
            budget = {
                **budget,
                "expected_output_tokens": int(row_policy["expected_output_tokens"]),
                "max_output_tokens": int(row_policy["max_output_tokens"]),
                "source": str(row_policy["source"]),
            }
        if (
            row.expected_output_tokens != int(budget["expected_output_tokens"])
            or row.max_output_tokens != int(budget["max_output_tokens"])
        ):
            token_budget_issues.append(
                {
                    "prompt_id": row.prompt_id,
                    "task_format": row.task_format,
                    "expected_output_tokens": row.expected_output_tokens,
                    "max_output_tokens": row.max_output_tokens,
                    "frozen_budget": budget,
                }
            )
        if row.lane == "code" and not is_genuine_code_task(row):
            lane_contract_issues.append(
                {
                    "prompt_id": row.prompt_id,
                    "reason": "code lane requires sandboxed code tests or the approved BFCL tool-state contract",
                    "task_format": row.task_format,
                    "grader_kind": row.grader.get("kind"),
                }
            )
        if row.lane == "general" and row.task_format == "code_execution":
            lane_contract_issues.append(
                {
                    "prompt_id": row.prompt_id,
                    "reason": "code_execution rows must be in the code lane",
                }
            )
        if row.label_type == "pairwise":
            from .study_graders import validate_pairwise_contract

            try:
                validate_pairwise_contract(row.grader)
            except ValueError as exc:
                grader_contract_issues.append(
                    {"prompt_id": row.prompt_id, "reason": str(exc)}
                )
        group_maps["prompt_hash"][digest].add(row.split)
        group_maps["leakage_group"][row.leakage_group].add(row.split)

    leakage: list[dict[str, Any]] = []
    for kind, values in group_maps.items():
        for key, splits in values.items():
            if len(splits) > 1:
                leakage.append({"kind": kind, "key": key, "splits": sorted(splits)})

    deterministic = label_counts.get("deterministic", 0) / max(len(rows), 1)
    tool_state = label_counts.get("tool_state", 0) / max(len(rows), 1)
    pairwise = label_counts.get("pairwise", 0) / max(len(rows), 1)
    label_mix_ok = (
        0.60 <= deterministic <= 0.80
        and 0.20 <= tool_state <= 0.40
        and pairwise == 0.0
    )
    counts_ok = all(split_counts.get(split, 0) == count for split, count in expected.items())
    strata_ok = all(
        stratum_counts.get(tag, 0) >= minimum
        for tag, minimum in REQUIRED_STRATA.items()
    )
    lane_ok = all(
        lane_counts.get(split, Counter()).get("code", 0)
        >= math.ceil(count * MIN_CODE_LANE_FRACTION)
        and lane_counts.get(split, Counter()).get("general", 0) > 0
        for split, count in expected.items()
    )
    non_test_ood = sum("ood" in row.tags and row.split != "test" for row in rows)
    non_test_representative = sum(
        "representative" in row.tags and row.split != "test" for row in rows
    )
    test_strata_ok = (
        test_stratum_counts.get("ood", 0) >= MIN_TEST_OOD_ROWS
        and test_stratum_counts.get("representative", 0) >= MIN_TEST_REPRESENTATIVE_ROWS
        and non_test_ood == 0
        and non_test_representative == 0
    )
    total = max(len(rows), 1)
    source_mix_ok = max(source_counts.values(), default=0) / total <= MAX_SOURCE_FRACTION
    benchmark_mix_ok = (
        max(benchmark_counts.values(), default=0) / total <= MAX_BENCHMARK_FRACTION
    )
    format_mix_ok = (
        max(task_format_counts.values(), default=0) / total <= MAX_TASK_FORMAT_FRACTION
        and task_format_counts.get("multiple_choice", 0) / total <= 0.35
    )
    supplied_rejections = tuple(rejected)
    valid = (
        not supplied_rejections
        and not missing
        and not leakage
        and not token_budget_issues
        and not lane_contract_issues
        and not grader_contract_issues
        and counts_ok
        and label_mix_ok
        and strata_ok
        and lane_ok
        and test_strata_ok
        and source_mix_ok
        and benchmark_mix_ok
        and format_mix_ok
    )
    return CorpusAudit(
        total=len(rows) + len(supplied_rejections),
        accepted=len(rows),
        split_counts=dict(split_counts),
        lane_counts={split: dict(counts) for split, counts in lane_counts.items()},
        label_counts=dict(label_counts),
        task_format_counts=dict(task_format_counts),
        source_counts=dict(source_counts),
        benchmark_counts=dict(benchmark_counts),
        stratum_counts=dict(stratum_counts),
        test_stratum_counts=dict(test_stratum_counts),
        rejected=supplied_rejections,
        leakage=tuple(leakage),
        missing_metadata=tuple(missing),
        token_budget_issues=tuple(token_budget_issues),
        lane_contract_issues=tuple(lane_contract_issues),
        grader_contract_issues=tuple(grader_contract_issues),
        valid=valid,
    )


def validate_corpus(
    prompts: Iterable[StudyPrompt],
    *,
    expected_counts: dict[str, int] | None = None,
    rejected: Iterable[dict[str, Any]] = (),
) -> CorpusAudit:
    audit = audit_corpus(prompts, expected_counts=expected_counts, rejected=rejected)
    if not audit.valid:
        reasons = []
        if audit.split_counts != (expected_counts or STUDY_SPLIT_COUNTS):
            reasons.append(f"split_counts={audit.split_counts}")
        if audit.leakage:
            reasons.append(f"leakage={len(audit.leakage)}")
        if audit.rejected:
            reasons.append(f"rejected={len(audit.rejected)}")
        if audit.missing_metadata:
            reasons.append(f"metadata={len(audit.missing_metadata)}")
        if audit.token_budget_issues:
            reasons.append(f"token_budgets={len(audit.token_budget_issues)}")
        if audit.lane_contract_issues:
            reasons.append(f"lane_contract={len(audit.lane_contract_issues)}")
        if audit.grader_contract_issues:
            reasons.append(f"grader_contract={len(audit.grader_contract_issues)}")
        if audit.label_counts:
            reasons.append(f"label_counts={audit.label_counts}")
        if not audit.lane_counts:
            reasons.append("lane_counts=missing")
        if audit.task_format_counts:
            reasons.append(f"task_formats={audit.task_format_counts}")
        if audit.test_stratum_counts:
            reasons.append(f"test_strata={audit.test_stratum_counts}")
        reasons.append(f"strata={audit.stratum_counts}")
        raise StudyCorpusError("study corpus failed promotion contract: " + "; ".join(reasons))
    return audit


def _outcome_success(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("success")
    elif hasattr(value, "success"):
        value = value.success
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result in {0.0, 1.0} else None


def _outcome_evidence_metrics(
    prompt_id: str,
    outcome_evidence: Mapping[str, Mapping[str, Any]] | None,
    *,
    model_ids: tuple[str, ...] = CORE_MODELS,
    require_all_models: bool = False,
) -> tuple[float, float, int] | None:
    if not outcome_evidence or prompt_id not in outcome_evidence:
        return None
    values = [
        result
        for model in model_ids
        if (result := _outcome_success(outcome_evidence[prompt_id].get(model))) is not None
    ]
    if (require_all_models and len(values) != len(model_ids)) or len(values) < 2:
        return None
    pair_count = len(values) * (len(values) - 1) / 2
    disagreement = sum(left != right for index, left in enumerate(values) for right in values[index + 1 :]) / pair_count
    mean_success = sum(values) / len(values)
    uncertainty = 0.0
    if 0.0 < mean_success < 1.0:
        uncertainty = -(
            mean_success * math.log2(mean_success)
            + (1.0 - mean_success) * math.log2(1.0 - mean_success)
        )
    return disagreement, uncertainty, len(values)


def grok_selection_metadata(
    prompts: Iterable[StudyPrompt],
    *,
    count: int = GROK_TRAIN_COUNT,
    seed: int = GROK_SELECTION_SEED,
    outcome_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the frozen Grok selection algorithm and selected IDs."""

    train = [row for row in prompts if row.split == "train"]
    if len(train) < count:
        raise StudyCorpusError(f"need {count} train prompts for Grok, found {len(train)}")
    train_ids = {row.prompt_id for row in train}
    train_evidence = {
        prompt_id: value
        for prompt_id, value in (outcome_evidence or {}).items()
        if prompt_id in train_ids
    }
    if train_evidence and set(train_evidence) != train_ids:
        missing = sorted(train_ids - set(train_evidence))[:5]
        raise StudyCorpusError(
            "Grok informative selection requires complete Phase A outcomes for every train prompt; "
            f"missing {len(train_ids - set(train_evidence))} prompts including {missing}"
        )
    scored: list[tuple[tuple[Any, ...], StudyPrompt, dict[str, Any]]] = []
    observed_count = 0
    disagreement_count = 0
    for row in train:
        metrics = _outcome_evidence_metrics(
            row.prompt_id,
            train_evidence or None,
            require_all_models=bool(train_evidence),
        )
        if metrics is None:
            actual_disagreement = 0.0
            actual_uncertainty = 0.0
            observed_models = 0
        else:
            actual_disagreement, actual_uncertainty, observed_models = metrics
            observed_count += 1
            disagreement_count += int(actual_disagreement > 0.0)
        proxy = int("disagreement-candidate" in row.tags)
        score = (
            -actual_disagreement,
            -actual_uncertainty,
            -proxy,
            -row.informative_score,
            stable_fraction(row.prompt_id or row.normalized_hash, seed),
            row.prompt_id,
        )
        scored.append(
            (
                score,
                row,
                {
                    "observed_disagreement": (
                        actual_disagreement if metrics is not None else None
                    ),
                    "observed_uncertainty": (
                        actual_uncertainty if metrics is not None else None
                    ),
                    "observed_model_count": observed_models,
                    "proxy": bool(proxy),
                },
            )
        )
    ordered = sorted(scored, key=lambda item: item[0])
    selected = ordered[:count]
    mode = "post_core_outcomes" if train_evidence and observed_count == len(train) else "pre_outcome_proxy"
    return {
        "algorithm_version": GROK_SELECTION_ALGORITHM_VERSION,
        "seed": seed,
        "count": count,
        "mode": mode,
        "eligible_train_count": len(train),
        "rows_with_complete_core_outcomes": observed_count,
        "rows_with_observed_disagreement": disagreement_count,
        "uses_only_train_rows": True,
        "selection_rule": [
            "observed candidate-model pairwise disagreement descending",
            "observed candidate-model Bernoulli entropy descending",
            "disagreement-candidate proxy tag descending before core outcomes exist",
            "pre-registered informative tags descending",
            "stable SHA-256 seeded tie-break, then prompt_id",
        ],
        "prompt_ids": [row.prompt_id for _score, row, _metrics in selected],
        "selected_metrics": {
            row.prompt_id: metrics for _score, row, metrics in selected
        },
    }


def select_informative_training(
    prompts: Iterable[StudyPrompt],
    count: int = GROK_TRAIN_COUNT,
    *,
    seed: int = GROK_SELECTION_SEED,
    outcome_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[StudyPrompt, ...]:
    """Apply the frozen pre-outcome or post-core-outcome Grok algorithm."""

    rows = tuple(prompts)
    metadata = grok_selection_metadata(
        rows,
        count=count,
        seed=seed,
        outcome_evidence=outcome_evidence,
    )
    by_id = {row.prompt_id: row for row in rows if row.split == "train"}
    return tuple(by_id[prompt_id] for prompt_id in metadata["prompt_ids"])


def study_call_scope(
    prompts: Iterable[StudyPrompt],
    *,
    grok_model: str = GROK_MODEL,
    grok_train_count: int = GROK_TRAIN_COUNT,
    outcome_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, tuple[StudyPrompt, ...]]:
    """Return the exact model → prompt coverage required by the study brief."""

    rows = tuple(prompts)
    train_ids = {row.prompt_id for row in rows if row.split == "train"}
    train_evidence = {
        prompt_id: value
        for prompt_id, value in (outcome_evidence or {}).items()
        if prompt_id in train_ids
    }
    # Before Phase A, Grok's IDs remain unresolved.  The frozen algorithm is
    # still auditable via grok_selection_metadata, but it cannot turn a proxy
    # tag into a pre-authorized candidate-model call.
    grok_train = (
        {
            row.prompt_id
            for row in select_informative_training(
                rows,
                grok_train_count,
                outcome_evidence=train_evidence,
            )
        }
        if train_evidence
        else set()
    )
    all_rows = tuple(row for row in rows if row.split in STUDY_SPLIT_COUNTS)
    grok_rows = tuple(
        row for row in all_rows if row.split != "train" or row.prompt_id in grok_train
    )
    return {
        **{model: all_rows for model in CORE_MODELS},
        grok_model: grok_rows,
    }


def load_jsonl(path: str | Path) -> tuple[StudyPrompt, ...]:
    """Load a candidate corpus without weakening row-level validation."""

    rows: list[StudyPrompt] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                raw = json.loads(line)
                rows.append(StudyPrompt.from_dict(raw, require_frozen_hash=True))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                raise StudyCorpusError(f"{path}:{line_number}: {exc}") from exc
    return tuple(rows)


def write_jsonl(rows: Iterable[StudyPrompt], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def corpus_checksum(prompts: Iterable[StudyPrompt]) -> str:
    """Hash the complete ordered frozen corpus representation."""

    encoded = "\n".join(
        json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in prompts
    ) + "\n"
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
