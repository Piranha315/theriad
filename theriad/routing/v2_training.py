"""Dependency-light training and evaluation utilities for Router V2.

The production router imports none of this module.  It exists to make the
grouped split, calibration, Pareto frontier, and artifact format reproducible.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from theriad.models.tiers import ModelCandidate
from theriad.routing.difficulty import featurise
from theriad.routing.study import (
    CORE_MODELS,
    REPEATED_TRIAL_GROUP_CONTRACT,
    TASK_FORMATS,
)
from theriad.routing.v2 import (
    ARTIFACT_SCHEMA_VERSION,
    EmbeddingBackend,
    RouterV2,
    RouterV2Artifact,
)


@dataclass(frozen=True)
class ModelOutcome:
    """A complete measured configuration outcome used for policy fitting.

    Episode collectors map terminal grader success to ``success``, sum model,
    retry, router and tool execution charges into ``cost_usd``, sum response
    tokens into ``output_tokens``, and use episode wall time for ``latency_ms``.
    Evaluation-only judge/collection overhead is charged to study spend in the
    ledger separately from the production policy's execution cost.
    Unknown costs and ungradable outcomes remain in the study ledger; they
    cannot become zero-valued successful observations. Complete repeated trials
    can be averaged by ``aggregate_router_input_groups``.
    """

    success: float
    cost_usd: float
    output_tokens: int
    latency_ms: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.success)) or not 0.0 <= self.success <= 1.0:
            raise ValueError("model outcome success must be finite and within [0,1]")
        if not math.isfinite(float(self.cost_usd)) or self.cost_usd < 0:
            raise ValueError("model outcome cost must be finite and non-negative")
        if isinstance(self.output_tokens, bool) or not isinstance(self.output_tokens, int):
            raise ValueError("model outcome output_tokens must be an integer")
        if self.output_tokens < 0:
            raise ValueError("model outcome output_tokens cannot be negative")
        if not math.isfinite(float(self.latency_ms)) or self.latency_ms < 0:
            raise ValueError("model outcome latency must be finite and non-negative")


@dataclass(frozen=True)
class StudyRow:
    prompt: str
    split: str
    source: str
    domain: str
    template_family: str
    collected_at: str
    lane: str
    outcomes: dict[str, ModelOutcome]
    # Provenance fields are optional for backward-compatible local fixtures, but
    # the paid-study corpus validator requires them before collection.
    prompt_id: str = ""
    benchmark_family: str = ""
    collection_vintage: str = ""
    source_revision: str = ""
    license_name: str = ""
    license_status: str = ""
    provenance_url: str = ""
    label_type: str = "deterministic"
    tags: tuple[str, ...] = ()
    task_format: str = "conversational_reasoning"
    leakage_group: str = ""
    protocol_id: str = ""
    normalized_prompt_sha256: str = ""
    grader: dict[str, Any] = field(default_factory=dict)
    router_input_group: str = ""
    router_input_contract: str = ""
    trial_count: int = 1

    def __post_init__(self) -> None:
        normalized = " ".join(self.prompt.split())
        expected_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if self.normalized_prompt_sha256:
            if self.normalized_prompt_sha256 != expected_hash:
                raise ValueError("normalized_prompt_sha256 does not match prompt")
        else:
            object.__setattr__(self, "normalized_prompt_sha256", expected_hash)
        if self.task_format not in TASK_FORMATS:
            raise ValueError(f"invalid task format {self.task_format!r}")
        if self.trial_count < 1:
            raise ValueError("trial_count must be positive")
        if not isinstance(self.grader, dict):
            raise ValueError("study row grader must be an object")
        explicit_group = self.router_input_group or str(
            self.grader.get("router_input_group") or ""
        )
        if explicit_group:
            object.__setattr__(self, "router_input_group", explicit_group)
        explicit_contract = self.router_input_contract or str(
            self.grader.get("repeated_trial_contract") or ""
        )
        if explicit_contract:
            object.__setattr__(self, "router_input_contract", explicit_contract)
        if not self.leakage_group:
            object.__setattr__(
                self,
                "leakage_group",
                "|".join(
                    (
                        self.source,
                        self.domain,
                        self.template_family,
                        self.collection_vintage or self.collected_at[:10],
                    )
                ),
            )

    @property
    def group_key(self) -> tuple[str, str, str, str, str]:
        vintage = self.collection_vintage or self.collected_at[:10]
        return self.source, self.domain, self.template_family, vintage, self.leakage_group

    @property
    def normalized_hash(self) -> str:
        return hashlib.sha256(" ".join(self.prompt.split()).encode("utf-8")).hexdigest()

    @property
    def effective_router_input_group(self) -> str:
        """Return the visible-input grouping key used for fitting.

        Unique rows use their normalized prompt hash.  Repeated τ2 trials must
        explicitly carry the same hash in ``router_input_group``; hidden state
        is never used as a router feature or as an additional sample unit.
        """

        return self.router_input_group or self.normalized_hash

    @property
    def is_repeated_trial(self) -> bool:
        return (
            self.router_input_contract == REPEATED_TRIAL_GROUP_CONTRACT
            and self.effective_router_input_group == self.normalized_hash
            and self.label_type == "tool_state"
            and str(self.grader.get("kind") or "").lower()
            == "tau2_official_environment"
            and self.grader.get("stateful_duplicate_prompt_allowed") is True
            and bool(str(self.grader.get("state_identity") or "").strip())
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StudyRow:
        if not isinstance(raw, Mapping):
            raise ValueError("study row must be an object")
        raw_outcomes = raw.get("outcomes")
        if not isinstance(raw_outcomes, Mapping) or not raw_outcomes:
            raise ValueError("study row needs a non-empty outcomes object")
        outcomes: dict[str, ModelOutcome] = {}
        for model, row in raw_outcomes.items():
            if not isinstance(row, Mapping):
                raise ValueError(f"outcome for {model} must be an object")
            outcomes[str(model)] = ModelOutcome(
                success=float(row["success"]),
                cost_usd=float(row["cost_usd"]),
                output_tokens=int(row["output_tokens"]),
                latency_ms=float(row["latency_ms"]),
            )
        split = str(raw.get("split") or "")
        lane = str(raw.get("lane") or "general")
        if split not in {"train", "calibration", "threshold_selection", "validation", "test"}:
            raise ValueError(f"invalid split {split!r}")
        if lane not in {"general", "code"}:
            raise ValueError(f"invalid lane {lane!r}")
        prompt = str(raw.get("prompt") or "").strip()
        if not prompt or not outcomes:
            raise ValueError("study row needs a prompt and outcomes")
        stored_hash = str(raw.get("normalized_prompt_sha256") or "").strip().lower()
        if not stored_hash:
            raise ValueError("study row needs normalized_prompt_sha256")
        expected_hash = hashlib.sha256(" ".join(prompt.split()).encode("utf-8")).hexdigest()
        if stored_hash != expected_hash:
            raise ValueError("normalized_prompt_sha256 does not match prompt")
        raw_grader = raw.get("grader")
        grader = dict(raw_grader) if isinstance(raw_grader, Mapping) else {}
        return cls(
            prompt=prompt,
            split=split,
            source=str(raw.get("source") or "unknown"),
            domain=str(raw.get("domain") or "unknown"),
            template_family=str(raw.get("template_family") or "unknown"),
            collected_at=str(raw.get("collected_at") or "unknown"),
            lane=lane,
            outcomes=outcomes,
            prompt_id=str(raw.get("prompt_id") or raw.get("id") or ""),
            benchmark_family=str(raw.get("benchmark_family") or raw.get("source") or ""),
            collection_vintage=str(
                raw.get("collection_vintage") or raw.get("collected_at") or ""
            ),
            source_revision=str(raw.get("source_revision") or raw.get("revision") or ""),
            license_name=str(raw.get("license") or raw.get("license_name") or ""),
            license_status=str(raw.get("license_status") or ""),
            provenance_url=str(raw.get("provenance_url") or raw.get("source_url") or ""),
            label_type=str(raw.get("label_type") or "deterministic"),
            tags=tuple(str(tag) for tag in raw.get("tags", []) if str(tag)),
            task_format=str(raw.get("task_format") or "conversational_reasoning"),
            leakage_group=str(raw.get("leakage_group") or ""),
            protocol_id=str(raw.get("protocol_id") or ""),
            normalized_prompt_sha256=stored_hash,
            grader=grader,
            router_input_group=str(
                raw.get("router_input_group") or grader.get("router_input_group") or ""
            ),
            router_input_contract=str(
                raw.get("router_input_contract")
                or grader.get("repeated_trial_contract")
                or ""
            ),
            trial_count=max(1, int(raw.get("trial_count") or 1)),
        )


def assert_grouped_splits(rows: Iterable[StudyRow]) -> None:
    """Reject every independently prohibited cross-split leakage class.

    The checks are intentionally separate.  A source can leak while its
    domain/template/vintage tuple differs, and a benchmark family can leak
    while source metadata happens to be distinct.  Collapsing these into one
    composite key would silently allow both cases.
    """

    group_maps: dict[str, dict[Any, str]] = {
        "normalized_prompt": {},
        "leakage_group": {},
    }
    prompt_rows: dict[str, StudyRow] = {}
    for row in rows:
        keys: dict[str, Any] = {
            "normalized_prompt": hashlib.sha256(
                " ".join(row.prompt.split()).encode("utf-8")
            ).hexdigest(),
            "leakage_group": row.leakage_group,
        }
        for kind, key in keys.items():
            prior = group_maps[kind].get(key)
            if prior is None:
                group_maps[kind][key] = row.split
                continue
            if prior == row.split:
                if kind == "normalized_prompt":
                    # Same-split repetitions are valid only when the corpus
                    # explicitly declares a grouped-trial contract.  The
                    # trainer aggregates them before fitting.
                    previous = prompt_rows[key]
                    if not (previous.is_repeated_trial and row.is_repeated_trial):
                        raise ValueError(
                            f"duplicate normalized prompt in {row.split}: {key!r}"
                        )
                continue
            if kind == "normalized_prompt":
                raise ValueError(f"normalized prompt leakage between {prior} and {row.split}")
            raise ValueError(f"leakage-group leakage between {prior} and {row.split}: {key!r}")
        if row.normalized_hash not in prompt_rows:
            prompt_rows[row.normalized_hash] = row


APPROVED_LICENSE_STATUSES = frozenset(
    {"approved", "permitted", "public", "public-domain", "internal"}
)
VALID_LABEL_TYPES = frozenset({"deterministic", "tool_state", "pairwise"})


def assert_study_row_contract(rows: Iterable[StudyRow]) -> None:
    """Reject incomplete post-collection rows before artifact training/evaluation."""

    materialized = list(rows)
    if not materialized:
        raise ValueError("study rows cannot be empty")
    seen_ids: set[str] = set()
    seen_prompts: dict[str, StudyRow] = {}
    required = (
        "prompt_id",
        "prompt",
        "source",
        "benchmark_family",
        "domain",
        "template_family",
        "collection_vintage",
        "source_revision",
        "license_name",
        "license_status",
        "provenance_url",
        "collected_at",
        "task_format",
        "leakage_group",
        "normalized_prompt_sha256",
    )
    for index, row in enumerate(materialized):
        missing = [name for name in required if not str(getattr(row, name, "")).strip()]
        if row.license_status not in APPROVED_LICENSE_STATUSES:
            missing.append("license_status_not_approved")
        if row.label_type not in VALID_LABEL_TYPES:
            missing.append("label_type_invalid")
        if row.task_format not in TASK_FORMATS:
            missing.append("task_format_invalid")
        if missing:
            raise ValueError(f"study row {index} is incomplete: {sorted(set(missing))}")
        if row.prompt_id in seen_ids:
            raise ValueError(f"duplicate prompt_id in study rows: {row.prompt_id}")
        prior = seen_prompts.get(row.normalized_hash)
        if prior is not None:
            if not (
                prior.is_repeated_trial
                and row.is_repeated_trial
                and prior.split == row.split
                and prior.effective_router_input_group
                == row.effective_router_input_group
                and prior.router_input_contract == row.router_input_contract
            ):
                raise ValueError(f"duplicate normalized prompt in study rows: {row.prompt_id}")
        seen_ids.add(row.prompt_id)
        seen_prompts.setdefault(row.normalized_hash, row)
    assert_grouped_splits(materialized)


def _mean_outcomes(rows: Sequence[StudyRow]) -> dict[str, ModelOutcome]:
    models = sorted({model for row in rows for model in row.outcomes})
    result: dict[str, ModelOutcome] = {}
    for model in models:
        missing = [row.prompt_id for row in rows if model not in row.outcomes]
        if missing:
            raise ValueError(f"incomplete repeated-trial outcomes for {model}: {missing}")
        outcomes = [row.outcomes[model] for row in rows]
        weights = [row.trial_count for row in rows]
        result[model] = ModelOutcome(
            success=float(np.average([value.success for value in outcomes], weights=weights)),
            cost_usd=float(np.average([value.cost_usd for value in outcomes], weights=weights)),
            output_tokens=int(round(float(np.average([value.output_tokens for value in outcomes], weights=weights)))),
            latency_ms=float(np.average([value.latency_ms for value in outcomes], weights=weights)),
        )
    return result


def aggregate_router_input_groups(rows: Iterable[StudyRow]) -> tuple[StudyRow, ...]:
    """Aggregate indistinguishable router inputs into effective sample units.

    τ2 can contain multiple official environment states with identical visible
    ticket text.  They remain separate executable trials for collection and
    grading, but the trainer/evaluator consumes one grouped row whose outcomes
    are the arithmetic mean of the completed trials.  Grouping never crosses a
    split and never accepts hidden state as a router-visible key.
    """

    materialized = tuple(rows)
    groups: dict[tuple[str, str], list[StudyRow]] = {}
    for row in materialized:
        key = (row.split, row.effective_router_input_group)
        groups.setdefault(key, []).append(row)
    aggregated: list[StudyRow] = []
    for (_split, group_key), members in groups.items():
        representative = members[0]
        if any(
            member.normalized_hash != representative.normalized_hash
            or member.prompt != representative.prompt
            or member.lane != representative.lane
            or member.task_format != representative.task_format
            or member.router_input_contract != representative.router_input_contract
            for member in members
        ):
            raise ValueError(f"router-input group {group_key!r} contains incompatible rows")
        if len(members) > 1 and not all(member.is_repeated_trial for member in members):
            raise ValueError(
                f"router-input group {group_key!r} repeats without the explicit trial contract"
            )
        if len(members) == 1:
            aggregated.append(representative)
            continue
        aggregated.append(
            StudyRow(
                prompt=representative.prompt,
                split=representative.split,
                source=representative.source,
                domain=representative.domain,
                template_family=representative.template_family,
                collected_at=representative.collected_at,
                lane=representative.lane,
                outcomes=_mean_outcomes(members),
                prompt_id=f"router-input-group:{group_key}",
                benchmark_family=representative.benchmark_family,
                collection_vintage=representative.collection_vintage,
                source_revision=representative.source_revision,
                license_name=representative.license_name,
                license_status=representative.license_status,
                provenance_url=representative.provenance_url,
                label_type=representative.label_type,
                tags=representative.tags,
                task_format=representative.task_format,
                leakage_group=representative.leakage_group,
                protocol_id=representative.protocol_id,
                normalized_prompt_sha256=representative.normalized_prompt_sha256,
                grader=dict(representative.grader),
                router_input_group=group_key,
                router_input_contract=representative.router_input_contract,
                trial_count=len(members),
            )
        )
    return tuple(aggregated)


def _row_disagreement_metrics(
    row: StudyRow, model_ids: Sequence[str]
) -> dict[str, Any]:
    values = [
        float(row.outcomes[model].success)
        for model in model_ids
        if model in row.outcomes
    ]
    if len(values) < 2:
        return {
            "status": "pending",
            "observed_model_count": len(values),
            "disagreement": None,
            "uncertainty": None,
        }
    pair_count = len(values) * (len(values) - 1) / 2
    disagreement = sum(
        left != right
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    ) / pair_count
    mean_success = sum(values) / len(values)
    uncertainty = 0.0
    if 0.0 < mean_success < 1.0:
        uncertainty = -(
            mean_success * math.log2(mean_success)
            + (1.0 - mean_success) * math.log2(1.0 - mean_success)
        )
    return {
        "status": "disagreement" if len(set(values)) > 1 else "agreement",
        "observed_model_count": len(values),
        "disagreement": float(disagreement),
        "uncertainty": float(uncertainty),
    }


def post_collection_disagreement_slices(
    rows: Iterable[StudyRow],
    *,
    model_ids: Sequence[str] = CORE_MODELS,
) -> dict[str, Any]:
    """Compute disagreement slices only from explicit collected outcomes.

    This report is intentionally not called by corpus promotion or training.
    It is a post-collection diagnostic for train, validation, and untouched test
    rows; missing outcomes remain pending rather than being imputed.
    """

    raw_rows = tuple(rows)
    materialized = aggregate_router_input_groups(raw_rows)
    report: dict[str, Any] = {
        "schema_version": 1,
        "policy": "post_collection_only",
        "required_models": list(model_ids),
        "influences_training_or_tuning": False,
        "raw_trial_count": len(raw_rows),
        "effective_sample_count": len(materialized),
        "by_split": {},
    }
    for split in ("train", "validation", "test"):
        split_rows = [row for row in materialized if row.split == split]
        metrics = {
            row.prompt_id: _row_disagreement_metrics(row, tuple(model_ids))
            for row in split_rows
        }
        report["by_split"][split] = {
            "row_count": len(split_rows),
            "eligible_count": sum(value["status"] != "pending" for value in metrics.values()),
            "disagreement_count": sum(value["status"] == "disagreement" for value in metrics.values()),
            "pending_count": sum(value["status"] == "pending" for value in metrics.values()),
            "prompt_ids": [
                prompt_id
                for prompt_id, value in metrics.items()
                if value["status"] == "disagreement"
            ],
            "metrics": metrics,
        }
    return report


def brier_score(probability: Sequence[float], target: Sequence[float]) -> float:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def log_loss(probability: Sequence[float], target: Sequence[float]) -> float:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-9, 1 - 1e-9)
    y = np.asarray(target, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(
    probability: Sequence[float], target: Sequence[float], *, bins: int = 10
) -> float:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if p.size == 0 or p.size != y.size:
        raise ValueError("probability and target must be non-empty and equal length")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & (
            p <= edges[index + 1] if index == bins - 1 else p < edges[index + 1]
        )
        if mask.any():
            total += float(mask.mean()) * abs(float(p[mask].mean() - y[mask].mean()))
    return total


def pareto_frontier(points: Sequence[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Return non-dominated ``(threshold, cost, quality)`` points."""

    ordered = sorted(points, key=lambda row: (row[1], -row[2], row[0]))
    frontier: list[tuple[float, float, float]] = []
    best_quality = -math.inf
    for point in ordered:
        if point[2] > best_quality + 1e-12:
            frontier.append(point)
            best_quality = point[2]
    return frontier


def automatic_knee(points: Sequence[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Select maximum distance above the normalized endpoint chord.

    Ties resolve to lower cost, then lower threshold, so the result is stable
    across input ordering and repeated runs.
    """

    frontier = pareto_frontier(points)
    if not frontier:
        raise ValueError("at least one policy point is required")
    if len(frontier) <= 2:
        return max(frontier, key=lambda row: (row[2] - row[1], -row[1], -row[0]))
    costs = np.asarray([row[1] for row in frontier], dtype=np.float64)
    qualities = np.asarray([row[2] for row in frontier], dtype=np.float64)
    x = (costs - costs.min()) / max(1e-12, float(costs.max() - costs.min()))
    y = (qualities - qualities.min()) / max(1e-12, float(qualities.max() - qualities.min()))
    x1, y1 = float(x[0]), float(y[0])
    x2, y2 = float(x[-1]), float(y[-1])
    denominator = math.hypot(y2 - y1, x2 - x1) or 1.0
    distances = [
        ((x2 - x1) * float(yi) - (y2 - y1) * float(xi) + y2 * x1 - x2 * y1)
        / denominator
        for xi, yi in zip(x, y, strict=True)
    ]
    best = max(range(len(frontier)), key=lambda i: (distances[i], -frontier[i][1], -frontier[i][0]))
    return frontier[best]


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 17,
    group_ids: Sequence[str] | None = None,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        raise ValueError("cannot bootstrap an empty sample")
    if group_ids is not None:
        if len(group_ids) != len(array):
            raise ValueError("group_ids must have the same length as values")
        grouped: dict[str, list[float]] = {}
        for group, value in zip(group_ids, array, strict=True):
            grouped.setdefault(str(group), []).append(float(value))
        array = np.asarray(
            [float(np.mean(grouped[group])) for group in sorted(grouped)],
            dtype=np.float64,
        )
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(samples, len(array)), replace=True), axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_group_mean_ci(
    values: Sequence[float],
    group_ids: Sequence[str],
    *,
    samples: int = 2000,
    seed: int = 17,
) -> tuple[float, float]:
    """Bootstrap effective router-input groups rather than raw trials."""

    return bootstrap_mean_ci(
        values,
        samples=samples,
        seed=seed,
        group_ids=group_ids,
    )


def _matrix(rows: Sequence[StudyRow], names: Sequence[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        raw = featurise(row.prompt)
        matrix.append([raw.get(name, 0.0) for name in names])
    return np.asarray(matrix, dtype=np.float64)


def _logistic_fit(x: np.ndarray, y: np.ndarray, *, ridge: float = 1.0) -> np.ndarray:
    """Small deterministic IRLS fit; intercept is the first column."""

    beta = np.zeros(x.shape[1], dtype=np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    for _ in range(80):
        eta = np.clip(x @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(p * (1.0 - p), 1e-6, None)
        z = eta + (y - p) / weight
        lhs = x.T @ (weight[:, None] * x) + penalty
        rhs = x.T @ (weight * z)
        try:
            updated = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            updated = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        if float(np.max(np.abs(updated - beta))) < 1e-8:
            beta = updated
            break
        beta = updated
    return beta


def _platt_fit(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.column_stack([np.ones(len(logits)), logits])
    beta = _logistic_fit(x, y, ridge=1e-3)
    return float(beta[1]), float(beta[0])


def artifact_candidates(
    artifact: RouterV2Artifact,
    model_candidates: Sequence[ModelCandidate] | None = None,
) -> tuple[ModelCandidate, ...]:
    """Build a pre-call candidate set, never using held-out outcome availability.

    Explicit candidates freeze provider/pricing and account/capability filtering.
    The registry is the backward-compatible default for the standalone trainer.
    Unknown synthetic or external models must supply their real price schedule.
    """
    if model_candidates is None:
        from theriad.models.registry import DEFAULT_REGISTRY

        model_candidates = tuple(
            candidate
            for tier in DEFAULT_REGISTRY.all_tiers()
            for candidate in DEFAULT_REGISTRY.candidates_for_tier(tier)
        )
    catalog: dict[str, ModelCandidate] = {}
    for candidate in model_candidates:
        catalog.setdefault(candidate.model, candidate)
    needed = {
        model
        for lane in artifact.lanes.values()
        for model in (lane.models or tuple(artifact.models))
    }
    missing = needed - catalog.keys()
    if missing:
        raise ValueError(f"explicit pre-call model candidates required for: {sorted(missing)}")
    return tuple(catalog[model] for model in sorted(needed))


def evaluate_artifact_rows(
    artifact: dict[str, Any],
    rows: Sequence[StudyRow],
    *,
    model_candidates: Sequence[ModelCandidate] | None = None,
    embedding_backend: EmbeddingBackend | None = None,
) -> tuple[tuple[str, ModelOutcome], ...]:
    """Score each runtime decision against that exact row's observed outcome.

    A missing selected outcome invalidates comparison rather than shrinking the
    candidate set or the evaluation denominator to favorable observations.
    """
    parsed = RouterV2Artifact.from_dict(artifact)
    router = RouterV2(parsed, embedding_backend=embedding_backend)
    candidates = artifact_candidates(parsed, model_candidates)
    results: list[tuple[str, ModelOutcome]] = []
    for row in rows:
        decision = router.predict(row.prompt, candidates, lane=row.lane)
        if decision.selected_model not in row.outcomes:
            raise ValueError(
                f"missing selected outcome for {row.prompt_id}: {decision.selected_model}"
            )
        results.append((decision.selected_model, row.outcomes[decision.selected_model]))
    return tuple(results)


def _freeze_runtime_lanes(
    artifact: dict[str, Any],
    selection_rows: Sequence[StudyRow],
    *,
    model_candidates: Sequence[ModelCandidate] | None,
    embedding_backend: EmbeddingBackend | None = None,
    selection_cost_ratio: float | None = None,
    baseline_models_by_prompt: Mapping[str, str] | None = None,
    baseline_outcomes_by_prompt: Mapping[str, ModelOutcome] | None = None,
    allow_fixed_models: bool = True,
) -> None:
    """Tune configurations, using precisely the runtime pre-call policy.

    The historical Pareto knee remains the default. A study opts into quality
    maximization under its measured baseline cost by passing 0.95 and a frozen
    baseline mapping. No test outcome is used; realized validation costs score
    complete configurations only, and never decide an individual row's model.
    """
    if selection_cost_ratio is not None and (
        not math.isfinite(selection_cost_ratio) or not 0 < selection_cost_ratio <= 1
    ):
        raise ValueError("selection_cost_ratio must be in (0,1]")
    parsed = RouterV2Artifact.from_dict(artifact)
    candidates = artifact_candidates(parsed, model_candidates)
    artifact["selection_price_snapshot"] = [
        {
            "model": candidate.model,
            "provider": candidate.provider,
            "input_cost_per_million": candidate.input_cost_per_million,
            "output_cost_per_million": candidate.output_cost_per_million,
        }
        for candidate in candidates
    ]
    for lane in ("general", "code"):
        lane_rows = [row for row in selection_rows if row.lane == lane]
        if not lane_rows:
            raise ValueError(f"threshold-selection split has no measured {lane} rows")
        config = artifact["lanes"][lane]
        choices: list[dict[str, Any]] = []
        for threshold in np.linspace(0.0, 0.99, 100):
            config["knee_threshold"] = float(threshold)
            results = evaluate_artifact_rows(
                artifact, lane_rows, model_candidates=candidates,
                embedding_backend=embedding_backend,
            )
            choices.append({
                "knee_threshold": float(threshold),
                "validation_cost_usd": float(np.mean([result.cost_usd for _, result in results])),
                "validation_success": float(np.mean([result.success for _, result in results])),
                "complexity": 1,
            })
        if selection_cost_ratio is None:
            knee = automatic_knee([
                (row["knee_threshold"], row["validation_cost_usd"], row["validation_success"])
                for row in choices
            ])
            selected = next(row for row in choices if row["knee_threshold"] == knee[0])
            config.update(selected)
            config["selection_objective"] = "legacy_pareto_knee"
            continue

        if baseline_models_by_prompt is None and baseline_outcomes_by_prompt is None:
            raise ValueError("budget-matched selection requires frozen baseline_models_by_prompt")
        baseline: list[ModelOutcome] = []
        for row in lane_rows:
            if baseline_outcomes_by_prompt is not None:
                if row.prompt_id not in baseline_outcomes_by_prompt:
                    raise ValueError(f"missing actual baseline outcome for {row.prompt_id}")
                baseline.append(baseline_outcomes_by_prompt[row.prompt_id])
                continue
            baseline_model = baseline_models_by_prompt.get(row.prompt_id)
            if baseline_model is None or baseline_model not in row.outcomes:
                raise ValueError(f"missing frozen baseline outcome for {row.prompt_id}")
            baseline.append(row.outcomes[baseline_model])
        baseline_cost = float(np.mean([outcome.cost_usd for outcome in baseline]))
        baseline_success = float(np.mean([outcome.success for outcome in baseline]))
        for model in (config["models"] or artifact["models"]) if allow_fixed_models else []:
            candidate = next(row for row in candidates if row.model == model)
            if any(
                price is None or not math.isfinite(price) or price < 0
                for price in (candidate.input_cost_per_million, candidate.output_cost_per_million)
            ):
                continue
            config["fixed_model"] = model
            results = evaluate_artifact_rows(
                artifact, lane_rows, model_candidates=candidates,
                embedding_backend=embedding_backend,
            )
            choices.append({
                "knee_threshold": 0.0,
                "fixed_model": model,
                "validation_cost_usd": float(np.mean([result.cost_usd for _, result in results])),
                "validation_success": float(np.mean([result.success for _, result in results])),
                "complexity": 0,
            })
        config.pop("fixed_model", None)
        feasible = [
            choice for choice in choices
            if ((baseline_outcomes_by_prompt is None and choice["validation_cost_usd"] <= baseline_cost * selection_cost_ratio)
                or (baseline_outcomes_by_prompt is not None and choice["validation_success"] > 0 and baseline_success > 0
                    and choice["validation_cost_usd"] / choice["validation_success"]
                    <= baseline_cost / baseline_success * selection_cost_ratio))
        ]
        if not feasible:
            raise ValueError(f"no {lane} policy meets the frozen baseline cost constraint")
        selected = min(feasible, key=lambda row: (
            -row["validation_success"], row["validation_cost_usd"], row["complexity"],
            row.get("fixed_model", ""), row["knee_threshold"],
        ))
        config.update(selected)
        config.update({
            "selection_objective": "quality_at_frozen_baseline_cost_per_success" if baseline_outcomes_by_prompt is not None else "quality_at_frozen_baseline_cost",
            "baseline_success": baseline_success,
            "fixed_model_candidates_allowed": allow_fixed_models,
            "baseline_cost_usd": baseline_cost,
            "selection_cost_ratio": selection_cost_ratio,
            "validation_policy_comparisons": choices,
        })
        artifact["research_only"] = True


def train_surface_artifact(
    rows: Sequence[StudyRow],
    *,
    fallback_models: dict[str, str],
    active_models_by_lane: dict[str, Sequence[str]] | None = None,
    feature_names: Sequence[str] | None = None,
    model_candidates: Sequence[ModelCandidate] | None = None,
    selection_cost_ratio: float | None = None,
    baseline_models_by_prompt: Mapping[str, str] | None = None,
    baseline_outcomes_by_prompt: Mapping[str, ModelOutcome] | None = None,
    allow_fixed_models: bool = True,
    calibration_split: str | None = None,
    threshold_selection_split: str | None = None,
) -> dict[str, Any]:
    """Fit the auditable surface baseline and freeze validation-set knees.

    The same artifact contract also accepts embedding weights from a heavier
    offline trainer, allowing RouterDC-style and hybrid heads to share the live
    runtime without shipping PyTorch.

    Pass ``model_candidates`` to freeze the pre-call provider/price catalog.
    Budget studies additionally pass ``selection_cost_ratio=0.95`` and
    ``baseline_models_by_prompt={validation_prompt_id: baseline_model}``; the
    IDs are those returned by ``aggregate_router_input_groups``. Selection then
    maximizes observed validation quality under each lane's actual baseline
    cost ceiling, compares fixed-model policies, and emits a research-only
    artifact. Omit the ratio to preserve the historical Pareto-knee objective.
    Missing selected outcomes, baseline outcomes, or feasible policies fail
    explicitly. Test labels and response features never determine a route.
    """

    raw_rows = tuple(rows)
    assert_study_row_contract(raw_rows)
    # Test metadata can be audited for split leakage; test outcomes are never
    # even aggregated by a trainer. The locked evaluator owns those labels.
    available_splits = {row.split for row in raw_rows}
    calibration_split = calibration_split or ("calibration" if "calibration" in available_splits else "validation")
    threshold_selection_split = threshold_selection_split or (
        "threshold_selection" if "threshold_selection" in available_splits else "validation"
    )
    if calibration_split not in available_splits or threshold_selection_split not in available_splits:
        raise ValueError("calibration and threshold-selection splits are required")
    if calibration_split == threshold_selection_split and {"calibration", "threshold_selection"} <= available_splits:
        raise ValueError("calibration and threshold-selection inputs must be disjoint")
    rows = aggregate_router_input_groups(
        row for row in raw_rows if row.split in {"train", calibration_split, threshold_selection_split}
    )
    test_rows = [row for row in raw_rows if row.split == "test"]
    test_group_count = len({row.effective_router_input_group for row in test_rows})
    train = [row for row in rows if row.split == "train"]
    calibration = [row for row in rows if row.split == calibration_split]
    threshold_selection = [row for row in rows if row.split == threshold_selection_split]
    if not train or not calibration or not threshold_selection:
        raise ValueError("train, calibration, and threshold-selection rows are required")
    names = list(feature_names or sorted(featurise(train[0].prompt)))
    raw_train = _matrix(train, names)
    mean = raw_train.mean(axis=0)
    stdev = raw_train.std(axis=0)
    stdev[stdev < 1e-9] = 1.0
    x_train = np.column_stack([np.ones(len(train)), np.clip((raw_train - mean) / stdev, -6.0, 6.0)])
    raw_calibration = _matrix(calibration, names)
    x_calibration = np.column_stack([np.ones(len(calibration)), np.clip((raw_calibration - mean) / stdev, -6.0, 6.0)])

    models = sorted({model for row in train for model in row.outcomes})
    heads: dict[str, Any] = {}
    for model in models:
        train_indices = [i for i, row in enumerate(train) if model in row.outcomes]
        calibration_indices = [i for i, row in enumerate(calibration) if model in row.outcomes]
        if len(train_indices) < max(20, len(names) + 2) or len(calibration_indices) < 10:
            continue
        y_train = np.asarray([train[i].outcomes[model].success for i in train_indices])
        beta = _logistic_fit(x_train[train_indices], y_train)
        calibration_logits = x_calibration[calibration_indices] @ beta
        y_calibration = np.asarray(
            [calibration[i].outcomes[model].success for i in calibration_indices]
        )
        platt_a, platt_b = _platt_fit(calibration_logits, y_calibration)
        probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(platt_a * calibration_logits + platt_b, -30, 30))
        )
        uncertainty = float(math.sqrt(np.mean((probabilities - y_calibration) ** 2)))
        observed = [train[i].outcomes[model] for i in train_indices]
        heads[model] = {
            "intercept": float(beta[0]),
            "weights": {name: float(value) for name, value in zip(names, beta[1:], strict=True)},
            "platt_a": platt_a,
            "platt_b": platt_b,
            "uncertainty": uncertainty,
            "sample_count": len(train_indices),
            "expected_output_tokens": float(np.mean([row.output_tokens for row in observed])),
            "latency_p50_ms": float(np.median([row.latency_ms for row in observed])),
            "latency_p95_ms": float(np.quantile([row.latency_ms for row in observed], 0.95)),
        }

    if not heads:
        raise ValueError("no model has enough train and validation observations")

    lanes = {
        lane: {
            "knee_threshold": 0.0,
            "fallback_model": fallback_models.get(lane, "openai/gpt-5.6-luna"),
            "models": list(active_models_by_lane.get(lane, ())) if active_models_by_lane else [],
        }
        for lane in ("general", "code")
    }

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "router_version": "router-v2-surface-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "family": "surface_logistic",
        "feature_mean": {name: float(value) for name, value in zip(names, mean, strict=True)},
        "feature_stdev": {name: float(value) for name, value in zip(names, stdev, strict=True)},
        "ood_max_abs_z": 4.0,
        "lanes": lanes,
        "models": heads,
        "router_input_group_contract": REPEATED_TRIAL_GROUP_CONTRACT,
        "raw_trial_count": len(raw_rows),
        "effective_sample_size": {
            "all": len(rows) + test_group_count,
            "train": len(train),
            "calibration": len(calibration),
            "threshold_selection": len(threshold_selection),
            "test": test_group_count,
        },
        "repeated_trial_count": (
            sum(max(0, row.trial_count - 1) for row in rows)
            + sum(row.trial_count for row in test_rows) - test_group_count
        ),
    }

    artifact["split_lineage"] = {
        "training": "train",
        "probability_calibration": calibration_split,
        "threshold_selection": threshold_selection_split,
        "locked_test": "test",
        "preprocessing_fit": "train_only",
    }
    _freeze_runtime_lanes(
        artifact, threshold_selection, model_candidates=model_candidates,
        selection_cost_ratio=selection_cost_ratio,
        baseline_models_by_prompt=baseline_models_by_prompt,
        baseline_outcomes_by_prompt=baseline_outcomes_by_prompt,
        allow_fixed_models=allow_fixed_models,
    )
    return artifact


def _embedding_for_row(
    row: StudyRow, embeddings: dict[str, Sequence[float]]
) -> Sequence[float]:
    """Resolve embeddings by prompt id first, then by normalized prompt hash."""

    candidates = [
        row.prompt_id,
        hashlib.sha256(" ".join(row.prompt.split()).encode("utf-8")).hexdigest(),
    ]
    for key in candidates:
        if key and key in embeddings:
            vector = embeddings[key]
            if vector:
                return vector
    raise ValueError(f"missing embedding for {row.prompt_id or row.prompt[:32]!r}")


def train_embedding_artifact(
    rows: Sequence[StudyRow],
    embeddings: dict[str, Sequence[float]],
    *,
    fallback_models: dict[str, str],
    active_models_by_lane: dict[str, Sequence[str]] | None = None,
    family: str = "semantic_linear",
    embedding_model: str = "unknown",
    include_surface: bool = False,
    runtime_profile: dict[str, float] | None = None,
    knn: bool = False,
    knn_k: int = 16,
    model_candidates: Sequence[ModelCandidate] | None = None,
    selection_cost_ratio: float | None = None,
    baseline_models_by_prompt: Mapping[str, str] | None = None,
    calibration_split: str | None = None,
    threshold_selection_split: str | None = None,
) -> dict[str, Any]:
    """Train an embedding-backed artifact using only supplied offline vectors.

    The function deliberately does not download or instantiate an encoder.  This
    makes it usable for both ``all-MiniLM-L6-v2`` and ``gte-modernbert-base``
    vectors while keeping encoder choice, revision, and runtime profiling in the
    caller's provenance.  ``knn=True`` stores train neighbours in the artifact;
    the live runtime only accepts that heavier form when its embedding resource
    gates are explicitly satisfied.

    ``model_candidates``, ``selection_cost_ratio``, and
    ``baseline_models_by_prompt`` share ``train_surface_artifact``'s frozen
    catalog and opt-in budget-matched semantics. Selection's embedding lookup
    uses only the supplied validation prompt vectors.
    """

    raw_rows = tuple(rows)
    assert_study_row_contract(raw_rows)
    available_splits = {row.split for row in raw_rows}
    calibration_split = calibration_split or ("calibration" if "calibration" in available_splits else "validation")
    threshold_selection_split = threshold_selection_split or ("threshold_selection" if "threshold_selection" in available_splits else "validation")
    if calibration_split not in available_splits or threshold_selection_split not in available_splits:
        raise ValueError("calibration and threshold-selection splits are required")
    if calibration_split == threshold_selection_split and {"calibration", "threshold_selection"} <= available_splits:
        raise ValueError("calibration and threshold-selection inputs must be disjoint")
    rows = aggregate_router_input_groups(row for row in raw_rows if row.split in {"train", calibration_split, threshold_selection_split})
    test_rows = [row for row in raw_rows if row.split == "test"]
    test_group_count = len({row.effective_router_input_group for row in test_rows})
    train = [row for row in rows if row.split == "train"]
    calibration = [row for row in rows if row.split == calibration_split]
    threshold_selection = [row for row in rows if row.split == threshold_selection_split]
    if not train or not calibration or not threshold_selection:
        raise ValueError("train, calibration, and threshold-selection rows are required")
    train_embeddings = [_embedding_for_row(row, embeddings) for row in train]
    validation_embeddings = [_embedding_for_row(row, embeddings) for row in calibration]
    dimension = len(train_embeddings[0])
    if dimension < 1 or any(len(vector) != dimension for vector in (*train_embeddings, *validation_embeddings)):
        raise ValueError("all study embeddings must have one consistent positive dimension")

    names = list(sorted(featurise(train[0].prompt))) if include_surface else []
    if include_surface:
        raw_train = _matrix(train, names)
        feature_mean = raw_train.mean(axis=0)
        feature_stdev = raw_train.std(axis=0)
        feature_stdev[feature_stdev < 1e-9] = 1.0
        surface_train = np.clip((raw_train - feature_mean) / feature_stdev, -6.0, 6.0)
        raw_validation = _matrix(calibration, names)
        surface_validation = np.clip((raw_validation - feature_mean) / feature_stdev, -6.0, 6.0)
    else:
        feature_mean = np.asarray([], dtype=np.float64)
        feature_stdev = np.asarray([], dtype=np.float64)
        surface_train = np.empty((len(train), 0), dtype=np.float64)
        surface_validation = np.empty((len(calibration), 0), dtype=np.float64)
    embedding_train = np.asarray(train_embeddings, dtype=np.float64)
    embedding_validation = np.asarray(validation_embeddings, dtype=np.float64)
    x_train = np.column_stack([np.ones(len(train)), surface_train, embedding_train])
    x_validation = np.column_stack([np.ones(len(calibration)), surface_validation, embedding_validation])

    models = sorted({model for row in train for model in row.outcomes})
    heads: dict[str, Any] = {}
    for model in models:
        train_indices = [i for i, row in enumerate(train) if model in row.outcomes]
        validation_indices = [i for i, row in enumerate(calibration) if model in row.outcomes]
        if len(train_indices) < max(20, x_train.shape[1] + 2) or len(validation_indices) < 10:
            continue
        y_train = np.asarray([train[i].outcomes[model].success for i in train_indices])
        y_validation = np.asarray([calibration[i].outcomes[model].success for i in validation_indices])
        head: dict[str, Any] = {
            "intercept": 0.0,
            "weights": {},
            "embedding_weights": [],
            "platt_a": 1.0,
            "platt_b": 0.0,
            "uncertainty": 0.10,
            "sample_count": len(train_indices),
            "expected_output_tokens": float(
                np.mean([train[i].outcomes[model].output_tokens for i in train_indices])
            ),
            "latency_p50_ms": float(
                np.median([train[i].outcomes[model].latency_ms for i in train_indices])
            ),
            "latency_p95_ms": float(
                np.quantile([train[i].outcomes[model].latency_ms for i in train_indices], 0.95)
            ),
        }
        if knn:
            neighbors = [
                {
                    "vector": embedding_train[index].astype(float).tolist(),
                    "success": float(train[index].outcomes[model].success),
                }
                for index in train_indices
            ]
            raw_probabilities: list[float] = []
            for index in validation_indices:
                distances = [
                    (
                        float(np.sum((embedding_train[train_index] - embedding_validation[index]) ** 2)),
                        float(train[train_index].outcomes[model].success),
                    )
                    for train_index in train_indices
                ]
                nearest = sorted(distances, key=lambda value: value[0])[:knn_k]
                raw_probabilities.append(
                    (sum(value for _distance, value in nearest) + 1.0)
                    / (len(nearest) + 2.0)
                )
            raw_probabilities_array = np.asarray(raw_probabilities, dtype=np.float64)
            logits = np.log(
                np.clip(raw_probabilities_array, 1e-9, 1 - 1e-9)
                / (1 - np.clip(raw_probabilities_array, 1e-9, 1 - 1e-9))
            )
            platt_a, platt_b = _platt_fit(logits, y_validation)
            probabilities = 1.0 / (
                1.0 + np.exp(-np.clip(platt_a * logits + platt_b, -30, 30))
            )
            head.update(
                {
                    "embedding_neighbors": neighbors,
                    "knn_k": knn_k,
                    "knn_alpha": 1.0,
                    "knn_beta": 1.0,
                    "platt_a": platt_a,
                    "platt_b": platt_b,
                }
            )
        else:
            beta = _logistic_fit(x_train[train_indices], y_train)
            validation_logits = x_validation[validation_indices] @ beta
            platt_a, platt_b = _platt_fit(validation_logits, y_validation)
            probabilities = 1.0 / (
                1.0 + np.exp(-np.clip(platt_a * validation_logits + platt_b, -30, 30))
            )
            surface_coefficients = beta[1 : 1 + len(names)]
            embedding_coefficients = beta[1 + len(names) :]
            head.update(
                {
                    "intercept": float(beta[0]),
                    "weights": {
                        name: float(value)
                        for name, value in zip(names, surface_coefficients, strict=True)
                    },
                    "embedding_weights": [float(value) for value in embedding_coefficients],
                    "platt_a": platt_a,
                    "platt_b": platt_b,
                }
            )
        uncertainty = float(math.sqrt(np.mean((probabilities - y_validation) ** 2)))
        head["uncertainty"] = uncertainty
        heads[model] = head

    if not heads:
        raise ValueError("no model has enough train and validation embedding observations")

    lanes = {
        lane: {
            "knee_threshold": 0.0,
            "fallback_model": fallback_models.get(lane, "openai/gpt-5.6-luna"),
            "models": list(active_models_by_lane.get(lane, ())) if active_models_by_lane else [],
        }
        for lane in ("general", "code")
    }

    artifact: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "router_version": f"router-v2-{family}-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "feature_mean": {
            name: float(value) for name, value in zip(names, feature_mean, strict=True)
        },
        "feature_stdev": {
            name: float(value) for name, value in zip(names, feature_stdev, strict=True)
        },
        "ood_max_abs_z": 4.0,
        "lanes": lanes,
        "models": heads,
        "embedding_model": embedding_model,
        "embedding_dimension": dimension,
        "embedding_training_rows": len(train),
        "router_input_group_contract": REPEATED_TRIAL_GROUP_CONTRACT,
        "raw_trial_count": len(raw_rows),
        "effective_sample_size": {
            "all": len(rows) + test_group_count,
            "train": len(train),
            "calibration": len(calibration),
            "threshold_selection": len(threshold_selection),
            "test": test_group_count,
        },
        "repeated_trial_count": (
            sum(max(0, row.trial_count - 1) for row in rows)
            + sum(row.trial_count for row in test_rows) - test_group_count
        ),
    }
    if runtime_profile is not None:
        artifact["runtime_profile"] = {
            "incremental_rss_mb": float(runtime_profile["incremental_rss_mb"]),
            "p95_latency_ms": float(runtime_profile["p95_latency_ms"]),
        }
    validation_vectors: dict[str, Sequence[float]] = {}
    for row, vector in zip(calibration, validation_embeddings, strict=True):
        if row.prompt in validation_vectors and not np.array_equal(validation_vectors[row.prompt], vector):
            raise ValueError("identical calibration prompts require identical frozen embeddings")
        validation_vectors[row.prompt] = vector

    class FrozenEmbeddingBackend:
        def encode(self, text: str) -> Sequence[float]:
            return validation_vectors[text]

    artifact["split_lineage"] = {
        "training": "train",
        "probability_calibration": calibration_split,
        "threshold_selection": threshold_selection_split,
        "locked_test": "test",
        "preprocessing_fit": "train_only",
    }
    _freeze_runtime_lanes(
        artifact, threshold_selection, model_candidates=model_candidates,
        embedding_backend=FrozenEmbeddingBackend(),
        selection_cost_ratio=selection_cost_ratio,
        baseline_models_by_prompt=baseline_models_by_prompt,
    )
    return artifact
