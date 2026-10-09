"""Public prompt-by-model anchors for Router V2 research.

This module is deliberately separate from the paid-study collector.  Public
benchmark material often contains prompts, trajectories, aggregate results,
and evaluation metadata in different files.  The importer keeps those facts
separate and refuses to turn an absent correctness label into a failure or a
success.

The canonical unit is one immutable observation for one prompt, model, and
protocol.  Derived prompt features are calculated only from observations that
actually have a binary outcome::

    prompt_id x model_id x protocol_id -> outcome

The raw source file remains the source of truth.  Normalized observations carry
an immutable source line/hash reference and all available trajectory metadata;
they do not copy potentially very large model outputs into the derived index.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ANCHOR_SCHEMA_VERSION = 1
DIFFICULTY_POLICY_VERSION = "empirical-failure-v1"
BALANCED_WEIGHT_POLICY_VERSION = "provider-model-family-balanced-v1"
DISAGREEMENT_POLICY_VERSION = "binary-entropy-v1"
CAPABILITY_POLICY_VERSION = "explicit-capability-order-v1"


def normalize_prompt(prompt: Any) -> str:
    """Normalize whitespace without attempting to rewrite task content."""

    if not isinstance(prompt, str):
        return ""
    return " ".join(prompt.strip().split())


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nonnegative(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and number >= 0 else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _bool_or_none(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "correct", "success", "pass"}:
            return True
        if normalized in {"0", "false", "no", "incorrect", "failure", "fail"}:
            return False
    return None


def _pairs(value: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    if not isinstance(value, Mapping):
        return ()
    return tuple(sorted((str(key), val) for key, val in value.items()))


def _pairs_to_dict(value: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    return {str(key): val for key, val in value}


@dataclass(frozen=True)
class WeightPolicy:
    """Versioned policy for turning model observations into a success rate."""

    name: str
    version: str
    group_field: str = "model"

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "group_field": self.group_field,
        }


UNWEIGHTED_POLICY = WeightPolicy(
    name="model-balanced",
    version="model-equal-v1",
    group_field="model_id",
)
PROVIDER_MODEL_FAMILY_POLICY = WeightPolicy(
    name="provider-model-family-balanced",
    version=BALANCED_WEIGHT_POLICY_VERSION,
    group_field="provider_model_family",
)


@dataclass(frozen=True)
class AnchorObservation:
    """One raw-derived prompt/model/protocol observation.

    ``outcome`` is intentionally nullable.  ``None`` means that the public
    artifact exposed a trajectory or a run record but no correctness result;
    it is not treated as a model failure.  ``raw_ref`` points back to the
    immutable source line so large trajectory payloads need not be copied into
    every derived feature file.
    """

    observation_id: str
    model_id: str
    protocol_id: str
    outcome: bool | None = None
    outcome_status: str = "missing"
    outcome_method: str = ""
    provider: str = ""
    model_family: str = ""
    benchmark_family: str = ""
    source_revision: str = ""
    collection_vintage: str = ""
    provenance_url: str = ""
    license_name: str = ""
    license_status: str = ""
    evaluation_date: str = ""
    prompting_protocol: str = ""
    tool_availability: str = ""
    agent_scaffold: str = ""
    attempts: int | None = None
    grader: str = ""
    retrieval_config: str = ""
    context_masking: str = ""
    tool_call_counts: tuple[tuple[str, Any], ...] = ()
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reported_cost_usd: float | None = None
    trajectory_steps: int | None = None
    trajectory_metadata: tuple[tuple[str, Any], ...] = ()
    raw_ref: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("anchor observation_id is required")
        if not self.model_id:
            raise ValueError("anchor model_id is required")
        if not self.protocol_id:
            raise ValueError("anchor protocol_id is required")
        if self.outcome not in {True, False, None}:
            raise ValueError("anchor outcome must be true, false, or null")
        if self.outcome is None and self.outcome_status == "graded":
            raise ValueError("graded anchor outcomes cannot be null")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        for count in (self.prompt_tokens, self.completion_tokens, self.attempts):
            if count is not None and count < 0:
                raise ValueError("counts cannot be negative")

    @property
    def balancing_group(self) -> str:
        """Return the least-specific available group without inventing identity."""

        if self.provider and self.model_family:
            return f"{self.provider}:{self.model_family}"
        return self.model_family or self.provider or self.model_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "model_id": self.model_id,
            "protocol_id": self.protocol_id,
            "outcome": self.outcome,
            "outcome_status": self.outcome_status,
            "outcome_method": self.outcome_method,
            "provider": self.provider,
            "model_family": self.model_family,
            "benchmark_family": self.benchmark_family,
            "source_revision": self.source_revision,
            "collection_vintage": self.collection_vintage,
            "provenance_url": self.provenance_url,
            "license_name": self.license_name,
            "license_status": self.license_status,
            "evaluation_date": self.evaluation_date,
            "prompting_protocol": self.prompting_protocol,
            "tool_availability": self.tool_availability,
            "agent_scaffold": self.agent_scaffold,
            "attempts": self.attempts,
            "grader": self.grader,
            "retrieval_config": self.retrieval_config,
            "context_masking": self.context_masking,
            "tool_call_counts": _pairs_to_dict(self.tool_call_counts),
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reported_cost_usd": self.reported_cost_usd,
            "trajectory_steps": self.trajectory_steps,
            "trajectory_metadata": _pairs_to_dict(self.trajectory_metadata),
            "raw_ref": _pairs_to_dict(self.raw_ref),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AnchorObservation:
        if not isinstance(raw, Mapping):
            raise ValueError("anchor observation must be an object")
        return cls(
            observation_id=str(raw.get("observation_id") or ""),
            model_id=str(raw.get("model_id") or ""),
            protocol_id=str(raw.get("protocol_id") or ""),
            outcome=_bool_or_none(raw.get("outcome")),
            outcome_status=str(raw.get("outcome_status") or "missing"),
            outcome_method=str(raw.get("outcome_method") or ""),
            provider=str(raw.get("provider") or ""),
            model_family=str(raw.get("model_family") or ""),
            benchmark_family=str(raw.get("benchmark_family") or ""),
            source_revision=str(raw.get("source_revision") or ""),
            collection_vintage=str(raw.get("collection_vintage") or ""),
            provenance_url=str(raw.get("provenance_url") or ""),
            license_name=str(raw.get("license_name") or ""),
            license_status=str(raw.get("license_status") or ""),
            evaluation_date=str(raw.get("evaluation_date") or ""),
            prompting_protocol=str(raw.get("prompting_protocol") or ""),
            tool_availability=str(raw.get("tool_availability") or ""),
            agent_scaffold=str(raw.get("agent_scaffold") or ""),
            attempts=_int_or_none(raw.get("attempts")),
            grader=str(raw.get("grader") or ""),
            retrieval_config=str(raw.get("retrieval_config") or ""),
            context_masking=str(raw.get("context_masking") or ""),
            tool_call_counts=_pairs(raw.get("tool_call_counts")),
            latency_ms=_nonnegative(raw.get("latency_ms")),
            prompt_tokens=_int_or_none(raw.get("prompt_tokens")),
            completion_tokens=_int_or_none(raw.get("completion_tokens")),
            reported_cost_usd=_nonnegative(raw.get("reported_cost_usd")),
            trajectory_steps=_int_or_none(raw.get("trajectory_steps")),
            trajectory_metadata=_pairs(raw.get("trajectory_metadata")),
            raw_ref=_pairs(raw.get("raw_ref")),
        )


@dataclass(frozen=True)
class CapabilityProfile:
    """Explicit model capability ordering, kept separate from price."""

    scores: tuple[tuple[str, float], ...]
    order_weak_to_strong: tuple[str, ...]
    estimated_from: str
    out_of_fold: bool = False
    version: str = CAPABILITY_POLICY_VERSION

    @classmethod
    def from_scores(
        cls,
        scores: Mapping[str, float],
        *,
        estimated_from: str,
        out_of_fold: bool = False,
    ) -> CapabilityProfile:
        if not scores:
            raise ValueError("capability scores cannot be empty")
        clean = {str(model): float(score) for model, score in scores.items()}
        if any(not math.isfinite(score) for score in clean.values()):
            raise ValueError("capability scores must be finite")
        order = tuple(sorted(clean, key=lambda model: (clean[model], model)))
        return cls(
            scores=tuple(sorted(clean.items())),
            order_weak_to_strong=order,
            estimated_from=str(estimated_from),
            out_of_fold=bool(out_of_fold),
        )

    @property
    def score_map(self) -> dict[str, float]:
        return dict(self.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": dict(self.scores),
            "order_weak_to_strong": list(self.order_weak_to_strong),
            "estimated_from": self.estimated_from,
            "out_of_fold": self.out_of_fold,
            "version": self.version,
        }


@dataclass(frozen=True)
class PublicAnchor:
    """One prompt with raw-derived observations and derived metrics."""

    prompt_id: str
    prompt: str
    benchmark_family: str
    domain: str = ""
    template_family: str = ""
    source_revision: str = ""
    collection_vintage: str = ""
    provenance_url: str = ""
    split: str = ""
    reference_answer: str | None = None
    metadata: tuple[tuple[str, Any], ...] = ()
    observations: tuple[AnchorObservation, ...] = ()
    metrics: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("anchor prompt_id is required")
        if not normalize_prompt(self.prompt):
            raise ValueError("anchor prompt cannot be empty")
        if not self.benchmark_family:
            raise ValueError("anchor benchmark_family is required")
        if self.split and self.split not in {"train", "validation", "test", "ood"}:
            raise ValueError(f"invalid anchor split {self.split!r}")

    @property
    def metric_map(self) -> dict[str, Any]:
        return _pairs_to_dict(self.metrics)

    @property
    def metadata_map(self) -> dict[str, Any]:
        return _pairs_to_dict(self.metadata)

    @property
    def normalized_prompt_hash(self) -> str:
        return prompt_hash(self.prompt)

    @property
    def group_key(self) -> tuple[str, str, str]:
        """Return the prompt/template grouping key used for split assignment.

        Source revisions and collection vintages remain provenance fields, but
        are intentionally not split keys: a template must not be separated
        merely because its public artifact was republished.
        """

        return (
            self.benchmark_family,
            self.domain,
            self.template_family or f"prompt:{self.prompt_id}",
        )

    def with_metrics(
        self,
        *,
        capability_profile: CapabilityProfile | None = None,
        protocol_id: str | None = None,
    ) -> PublicAnchor:
        metrics = derive_anchor_metrics(
            self.observations,
            capability_profile=capability_profile,
            protocol_id=protocol_id,
        )
        return PublicAnchor(
            prompt_id=self.prompt_id,
            prompt=self.prompt,
            benchmark_family=self.benchmark_family,
            domain=self.domain,
            template_family=self.template_family,
            source_revision=self.source_revision,
            collection_vintage=self.collection_vintage,
            provenance_url=self.provenance_url,
            split=self.split,
            reference_answer=self.reference_answer,
            metadata=self.metadata,
            observations=self.observations,
            metrics=tuple(sorted(metrics.items())),
        )

    def to_dict(self) -> dict[str, Any]:
        observations = [observation.to_dict() for observation in self.observations]
        by_model: dict[str, list[bool | None]] = defaultdict(list)
        by_protocol: dict[str, dict[str, bool | None]] = defaultdict(dict)
        for observation in self.observations:
            by_model[observation.model_id].append(observation.outcome)
            by_protocol[observation.protocol_id][observation.model_id] = observation.outcome
        outcomes: dict[str, bool | None] = {}
        for model, values in by_model.items():
            if len(values) == 1:
                outcomes[model] = values[0]
        return {
            "schema_version": ANCHOR_SCHEMA_VERSION,
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "benchmark_family": self.benchmark_family,
            "domain": self.domain,
            "template_family": self.template_family,
            "source_revision": self.source_revision,
            "collection_vintage": self.collection_vintage,
            "provenance_url": self.provenance_url,
            "split": self.split,
            "reference_answer": self.reference_answer,
            "metadata": _pairs_to_dict(self.metadata),
            "protocols": sorted({item.protocol_id for item in self.observations}),
            "outcomes": outcomes,
            "outcomes_by_protocol": {
                protocol: dict(sorted(values.items()))
                for protocol, values in sorted(by_protocol.items())
            },
            "observations": observations,
            "metrics": _pairs_to_dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PublicAnchor:
        if not isinstance(raw, Mapping):
            raise ValueError("public anchor must be an object")
        observations_raw = raw.get("observations") or []
        if not isinstance(observations_raw, list):
            raise ValueError("anchor observations must be a list")
        return cls(
            prompt_id=str(raw.get("prompt_id") or ""),
            prompt=str(raw.get("prompt") or ""),
            benchmark_family=str(raw.get("benchmark_family") or ""),
            domain=str(raw.get("domain") or ""),
            template_family=str(raw.get("template_family") or ""),
            source_revision=str(raw.get("source_revision") or ""),
            collection_vintage=str(raw.get("collection_vintage") or ""),
            provenance_url=str(raw.get("provenance_url") or ""),
            split=str(raw.get("split") or ""),
            reference_answer=(
                str(raw["reference_answer"]) if raw.get("reference_answer") is not None else None
            ),
            metadata=_pairs(raw.get("metadata")),
            observations=tuple(AnchorObservation.from_dict(item) for item in observations_raw),
            metrics=_pairs(raw.get("metrics")),
        )


def _filtered_observations(
    observations: Iterable[AnchorObservation], protocol_id: str | None = None
) -> list[AnchorObservation]:
    rows = [
        observation
        for observation in observations
        if observation.outcome is not None
        and (protocol_id is None or observation.protocol_id == protocol_id)
    ]
    return rows


def _model_rates(
    observations: Sequence[AnchorObservation], *, protocol_id: str | None = None
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for observation in _filtered_observations(observations, protocol_id):
        grouped[observation.model_id].append(float(bool(observation.outcome)))
    return {
        model: float(np.mean(values))
        for model, values in sorted(grouped.items())
        if values
    }


def _success_rate(
    observations: Sequence[AnchorObservation],
    *,
    policy: WeightPolicy,
    protocol_id: str | None = None,
) -> tuple[float | None, int, int]:
    rows = _filtered_observations(observations, protocol_id)
    if not rows:
        return None, 0, 0
    model_rates = _model_rates(rows)
    if policy.group_field == "model_id":
        return float(np.mean(list(model_rates.values()))), len(model_rates), len(rows)
    model_groups: dict[str, str] = {}
    for observation in rows:
        model_groups.setdefault(observation.model_id, observation.balancing_group)
    groups: dict[str, list[float]] = defaultdict(list)
    for model, rate in model_rates.items():
        groups[model_groups[model]].append(rate)
    group_rates = [float(np.mean(values)) for values in groups.values() if values]
    return (
        float(np.mean(group_rates)) if group_rates else None,
        len(groups),
        len(rows),
    )


def normalized_entropy(probability: float | None) -> float | None:
    """Binary entropy normalized to [0, 1], with endpoints equal to zero."""

    if probability is None:
        return None
    p = min(1.0, max(0.0, float(probability)))
    if p in {0.0, 1.0}:
        return 0.0
    entropy = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))
    return float(entropy / math.log(2.0))


def _capability_metrics(
    observations: Sequence[AnchorObservation],
    capability_profile: CapabilityProfile | None,
    *,
    protocol_id: str | None,
) -> dict[str, Any]:
    if capability_profile is None:
        return {
            "capability_order_violation": None,
            "upset_count": None,
            "upset_rate": None,
            "specialization_score": None,
            "capability_policy_version": None,
            "capability_profile_oof": None,
        }
    model_rates = _model_rates(
        [row for row in observations if row.outcome is not None], protocol_id=protocol_id
    )
    comparable_pairs = 0
    upset_count = 0
    for weaker_index, weaker in enumerate(capability_profile.order_weak_to_strong):
        for stronger in capability_profile.order_weak_to_strong[weaker_index + 1 :]:
            if weaker not in model_rates or stronger not in model_rates:
                continue
            comparable_pairs += 1
            if model_rates[weaker] > 0.5 and model_rates[stronger] < 0.5:
                upset_count += 1
    upset_rate = upset_count / comparable_pairs if comparable_pairs else None
    return {
        "capability_order_violation": bool(upset_count) if comparable_pairs else None,
        "upset_count": upset_count if comparable_pairs else None,
        "upset_rate": upset_rate,
        # In v1 specialization is deliberately an interpretable upset proxy,
        # not a hidden scalar that combines difficulty and disagreement.
        "specialization_score": upset_rate,
        "capability_policy_version": CAPABILITY_POLICY_VERSION,
        "capability_profile_oof": capability_profile.out_of_fold,
        "capability_profile_source": capability_profile.estimated_from,
        "capability_comparable_pairs": comparable_pairs,
    }


def derive_anchor_metrics(
    observations: Sequence[AnchorObservation],
    *,
    capability_profile: CapabilityProfile | None = None,
    protocol_id: str | None = None,
) -> dict[str, Any]:
    """Derive separate difficulty, disagreement, and specialization signals."""

    all_rows = [
        row for row in observations if protocol_id is None or row.protocol_id == protocol_id
    ]
    observed_rows = [row for row in all_rows if row.outcome is not None]
    model_count = len({row.model_id for row in all_rows})
    observed_model_count = len({row.model_id for row in observed_rows})
    unweighted, unweighted_groups, observed_trials = _success_rate(
        all_rows, policy=UNWEIGHTED_POLICY, protocol_id=protocol_id
    )
    balanced, balanced_groups, _ = _success_rate(
        all_rows, policy=PROVIDER_MODEL_FAMILY_POLICY, protocol_id=protocol_id
    )
    metrics: dict[str, Any] = {
        "difficulty_policy_version": DIFFICULTY_POLICY_VERSION,
        "disagreement_policy_version": DISAGREEMENT_POLICY_VERSION,
        "empirical_success_rate_unweighted": unweighted,
        "empirical_difficulty_unweighted": (
            1.0 - unweighted if unweighted is not None else None
        ),
        "model_disagreement_unweighted": normalized_entropy(unweighted),
        "empirical_success_rate_balanced": balanced,
        "empirical_difficulty_balanced": 1.0 - balanced if balanced is not None else None,
        "model_disagreement_balanced": normalized_entropy(balanced),
        # The balanced aliases are the default features exposed to the router.
        "empirical_difficulty": 1.0 - balanced if balanced is not None else None,
        "model_disagreement": normalized_entropy(balanced),
        "observed_model_count": observed_model_count,
        "model_count": model_count,
        "observed_trial_count": observed_trials,
        "missing_model_count": max(0, model_count - observed_model_count),
        "unweighted_group_count": unweighted_groups,
        "balanced_group_count": balanced_groups,
        "weight_policies": {
            "unweighted": UNWEIGHTED_POLICY.to_dict(),
            "balanced": PROVIDER_MODEL_FAMILY_POLICY.to_dict(),
        },
    }
    metrics.update(
        _capability_metrics(
            all_rows,
            capability_profile,
            protocol_id=protocol_id,
        )
    )
    return metrics


def capability_profile_from_observations(
    observations: Iterable[AnchorObservation],
    *,
    estimated_from: str,
    out_of_fold: bool = False,
    protocol_id: str | None = None,
) -> CapabilityProfile:
    """Estimate a capability profile from caller-selected observations.

    The function does not infer capability from price.  Callers should pass a
    train-only or out-of-fold observation set and retain the resulting profile
    metadata when using it for upset analysis.
    """

    rows = _filtered_observations(tuple(observations), protocol_id)
    rates = _model_rates(rows)
    return CapabilityProfile.from_scores(
        rates,
        estimated_from=estimated_from,
        out_of_fold=out_of_fold,
    )


def assign_grouped_anchor_splits(
    anchors: Sequence[PublicAnchor],
    *,
    ood_benchmark_families: Iterable[str] = (),
    seed: int = 17,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[PublicAnchor, ...]:
    """Assign deterministic IID and explicit OOD splits without prompt leakage."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 <= validation_fraction < 1.0 - train_fraction:
        raise ValueError("validation_fraction leaves no test split")
    ood = {str(value) for value in ood_benchmark_families}
    assigned: list[PublicAnchor] = []
    for anchor in anchors:
        if anchor.benchmark_family in ood:
            split = "ood"
        else:
            # Keep all identical source/domain/template/vintage groups together.
            group = "|".join(anchor.group_key)
            fraction = int.from_bytes(
                hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big"
            ) / float(2**64)
            if fraction < train_fraction:
                split = "train"
            elif fraction < train_fraction + validation_fraction:
                split = "validation"
            else:
                split = "test"
        assigned.append(
            PublicAnchor(
                prompt_id=anchor.prompt_id,
                prompt=anchor.prompt,
                benchmark_family=anchor.benchmark_family,
                domain=anchor.domain,
                template_family=anchor.template_family,
                source_revision=anchor.source_revision,
                collection_vintage=anchor.collection_vintage,
                provenance_url=anchor.provenance_url,
                split=split,
                reference_answer=anchor.reference_answer,
                metadata=anchor.metadata,
                observations=anchor.observations,
                metrics=anchor.metrics,
            )
        )
    validate_anchor_leakage(assigned)
    return tuple(assigned)


def validate_anchor_leakage(
    anchors: Iterable[PublicAnchor],
    *,
    require_benchmark_family_disjoint: bool = False,
) -> None:
    """Reject exact-prompt and grouped source leakage across assigned splits.

    Benchmark families may legitimately appear in both IID train/test splits;
    callers running an OOD regime can request family-level disjointness.
    """

    prompt_splits: dict[str, str] = {}
    groups: dict[str, set[str]] = defaultdict(set)
    families: dict[str, set[str]] = defaultdict(set)
    for anchor in anchors:
        if not anchor.split:
            continue
        prompt_key = anchor.normalized_prompt_hash
        prior = prompt_splits.setdefault(prompt_key, anchor.split)
        if prior != anchor.split:
            raise ValueError(f"anchor prompt leakage between {prior} and {anchor.split}")
        groups["|".join(anchor.group_key)].add(anchor.split)
        families[anchor.benchmark_family].add(anchor.split)
    checks = [("group", groups)]
    if require_benchmark_family_disjoint:
        checks.append(("benchmark_family", families))
    for kind, values in checks:
        leaked = {key: sorted(splits) for key, splits in values.items() if len(splits) > 1}
        if leaked:
            raise ValueError(f"anchor {kind} leakage: {sorted(leaked)[:3]}")


def _metric(anchor: PublicAnchor, name: str) -> float | None:
    value = anchor.metric_map.get(name)
    return _finite(value)


@dataclass(frozen=True)
class AnchorNeighbor:
    prompt_id: str
    similarity: float
    weight: float


class AnchorFeatureIndex:
    """Similarity index that exposes anchor evidence as features only.

    ``from_embeddings`` is the intended semantic path: embeddings are supplied
    by a pinned offline encoder.  ``from_tfidf`` is a dependency-light lexical
    fallback useful for smoke tests and baseline comparisons; it is labelled as
    such in the resulting features and must not be described as semantic
    retrieval.
    """

    def __init__(
        self,
        anchors: Sequence[PublicAnchor],
        vectors: Mapping[str, Sequence[float]],
        *,
        retriever_kind: str,
        token_vocabulary: Mapping[str, int] | None = None,
        token_idf: Mapping[str, float] | None = None,
    ) -> None:
        if not anchors:
            raise ValueError("anchor index cannot be empty")
        self.anchors = {anchor.prompt_id: anchor for anchor in anchors}
        self.retriever_kind = retriever_kind
        self._token_vocabulary = dict(token_vocabulary or {})
        self._token_idf = dict(token_idf or {})
        dimensions = {len(vector) for vector in vectors.values()}
        if len(dimensions) != 1:
            raise ValueError("anchor vectors must have one dimension")
        self.dimension = dimensions.pop()
        if self.dimension < 1:
            raise ValueError("anchor vectors cannot be empty")
        missing = set(self.anchors) - set(vectors)
        if missing:
            raise ValueError(f"missing vectors for {len(missing)} anchors")
        matrix = np.asarray([vectors[anchor.prompt_id] for anchor in anchors], dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms <= 0):
            raise ValueError("anchor vectors must have non-zero norm")
        self._ids = tuple(anchor.prompt_id for anchor in anchors)
        self._matrix = matrix / norms[:, None]

    @classmethod
    def from_embeddings(
        cls,
        anchors: Sequence[PublicAnchor],
        embeddings: Mapping[str, Sequence[float]],
    ) -> AnchorFeatureIndex:
        return cls(anchors, embeddings, retriever_kind="semantic_embedding")

    @classmethod
    def from_tfidf(cls, anchors: Sequence[PublicAnchor]) -> AnchorFeatureIndex:
        # ``\w`` is Unicode-aware in Python.  The public anchor set includes
        # BrowseComp-ZH rows; an ASCII-only tokenizer would turn those prompts
        # into zero vectors and make the overlap-filtered index unbuildable.
        token_pattern = re.compile(r"\w+", re.UNICODE)
        documents = [
            {token.lower() for token in token_pattern.findall(anchor.prompt)}
            for anchor in anchors
        ]
        document_frequency: Counter[str] = Counter()
        for tokens in documents:
            document_frequency.update(tokens)
        vocabulary = sorted(document_frequency)
        index = {token: position for position, token in enumerate(vocabulary)}
        matrix = np.zeros((len(anchors), len(vocabulary)), dtype=np.float64)
        for row_index, tokens in enumerate(documents):
            for token in tokens:
                matrix[row_index, index[token]] = math.log(
                    (1.0 + len(anchors)) / (1.0 + document_frequency[token])
                )
        vectors = {
            anchor.prompt_id: matrix[row_index].tolist()
            for row_index, anchor in enumerate(anchors)
        }
        # A one-token corpus can produce all-zero vectors after IDF; make the
        # fallback fail clearly rather than returning arbitrary neighbors.
        if not vocabulary or np.any(np.linalg.norm(matrix, axis=1) <= 0):
            raise ValueError("tfidf anchor index has no non-zero document vectors")
        return cls(
            anchors,
            vectors,
            retriever_kind="tfidf_baseline",
            token_vocabulary=index,
            token_idf={
                token: math.log((1.0 + len(anchors)) / (1.0 + document_frequency[token]))
                for token in vocabulary
            },
        )

    def retrieve(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 8,
        exclude_prompt_ids: Iterable[str] = (),
        exclude_benchmark_families: Iterable[str] = (),
    ) -> tuple[AnchorNeighbor, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = np.asarray(vector, dtype=np.float64)
        if query.shape != (self.dimension,):
            raise ValueError(f"query vector must have dimension {self.dimension}")
        norm = float(np.linalg.norm(query))
        if norm <= 0:
            raise ValueError("query vector must have non-zero norm")
        similarities = self._matrix @ (query / norm)
        excluded_ids = set(exclude_prompt_ids)
        excluded_families = set(exclude_benchmark_families)
        candidates = [
            (float(similarities[index]), prompt_id)
            for index, prompt_id in enumerate(self._ids)
            if prompt_id not in excluded_ids
            and self.anchors[prompt_id].benchmark_family not in excluded_families
        ]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        candidates = candidates[:top_k]
        if not candidates:
            return ()
        raw_weights = np.asarray([max(0.0, score) for score, _ in candidates], dtype=np.float64)
        if float(raw_weights.sum()) <= 0:
            raw_weights = np.exp(
                np.asarray([score for score, _ in candidates], dtype=np.float64)
                - max(score for score, _ in candidates)
            )
        raw_weights /= raw_weights.sum()
        return tuple(
            AnchorNeighbor(prompt_id=prompt_id, similarity=score, weight=float(raw_weights[i]))
            for i, (score, prompt_id) in enumerate(candidates)
        )

    def features_for_vector(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 8,
        exclude_prompt_ids: Iterable[str] = (),
        exclude_benchmark_families: Iterable[str] = (),
    ) -> dict[str, Any]:
        neighbors = self.retrieve(
            vector,
            top_k=top_k,
            exclude_prompt_ids=exclude_prompt_ids,
            exclude_benchmark_families=exclude_benchmark_families,
        )
        if not neighbors:
            return {
                "anchor_retriever_kind": self.retriever_kind,
                "neighbor_count": 0,
                "nearest_anchor_similarity": None,
                "neighbor_mean_difficulty": None,
                "neighbor_weighted_difficulty": None,
                "neighbor_mean_disagreement": None,
                "neighbor_specialization_score": None,
                "neighbor_model_success_rates": {},
                "neighbor_domain_distribution": {},
                "neighbor_benchmark_family_distribution": {},
            }
        weighted_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
        model_success: dict[str, list[tuple[float, float]]] = defaultdict(list)
        domain_weights: dict[str, float] = defaultdict(float)
        family_weights: dict[str, float] = defaultdict(float)
        for neighbor in neighbors:
            anchor = self.anchors[neighbor.prompt_id]
            metrics = anchor.metric_map
            for feature_name, metric_name in (
                ("difficulty", "empirical_difficulty"),
                ("disagreement", "model_disagreement"),
                ("specialization", "specialization_score"),
            ):
                value = _finite(metrics.get(metric_name))
                if value is not None:
                    weighted_values[feature_name].append((value, neighbor.weight))
            for observation in anchor.observations:
                if observation.outcome is not None:
                    model_success[observation.model_id].append(
                        (float(bool(observation.outcome)), neighbor.weight)
                    )
            domain_weights[anchor.domain] += neighbor.weight
            family_weights[anchor.benchmark_family] += neighbor.weight

        def weighted_mean(values: Sequence[tuple[float, float]]) -> float | None:
            if not values:
                return None
            total = sum(weight for _, weight in values)
            return sum(value * weight for value, weight in values) / total if total else None

        return {
            "anchor_retriever_kind": self.retriever_kind,
            "neighbor_count": len(neighbors),
            "nearest_anchor_similarity": neighbors[0].similarity,
            "neighbor_mean_difficulty": weighted_mean(weighted_values["difficulty"]),
            "neighbor_weighted_difficulty": weighted_mean(weighted_values["difficulty"]),
            "neighbor_mean_disagreement": weighted_mean(weighted_values["disagreement"]),
            "neighbor_specialization_score": weighted_mean(weighted_values["specialization"]),
            "neighbor_model_success_rates": {
                model: weighted_mean(values) for model, values in sorted(model_success.items())
            },
            "neighbor_domain_distribution": dict(sorted(domain_weights.items())),
            "neighbor_benchmark_family_distribution": dict(sorted(family_weights.items())),
            "neighbor_ids": [neighbor.prompt_id for neighbor in neighbors],
        }

    def features_for_text(self, prompt: str, *, top_k: int = 8) -> dict[str, Any]:
        """Query the lexical fallback; semantic indexes require an encoder."""

        if self.retriever_kind != "tfidf_baseline":
            raise ValueError("semantic index requires a caller-supplied query embedding")
        token_pattern = re.compile(r"[A-Za-z0-9_]+")
        tokens = {token.lower() for token in token_pattern.findall(prompt)}
        vector = np.zeros(self.dimension, dtype=np.float64)
        if self._token_vocabulary:
            for token in tokens:
                position = self._token_vocabulary.get(token)
                if position is not None:
                    vector[position] = self._token_idf.get(token, 1.0)
        else:
            # This fallback is retained only for indexes reconstructed from an
            # older serialized representation without its vocabulary.
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                vector[int.from_bytes(digest[:4], "big") % self.dimension] += 1.0
        return self.features_for_vector(vector, top_k=top_k)


def load_anchor_jsonl(path: str | Path) -> tuple[PublicAnchor, ...]:
    rows: list[PublicAnchor] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                rows.append(PublicAnchor.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid public anchor: {exc}") from exc
    return tuple(rows)


def write_anchor_jsonl(anchors: Iterable[PublicAnchor], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for anchor in anchors:
            handle.write(json.dumps(anchor.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


@dataclass(frozen=True)
class AnchorAudit:
    benchmark_family: str
    prompt_count: int
    observation_count: int
    usable_outcome_count: int
    missing_outcome_count: int
    unique_models: tuple[str, ...]
    protocols: tuple[str, ...]
    model_counts_by_protocol: dict[str, dict[str, int]]
    outcome_counts_by_protocol: dict[str, int]
    missing_counts_by_protocol: dict[str, int]
    source_references: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ANCHOR_SCHEMA_VERSION,
            "benchmark_family": self.benchmark_family,
            "prompt_count": self.prompt_count,
            "observation_count": self.observation_count,
            "usable_outcome_count": self.usable_outcome_count,
            "missing_outcome_count": self.missing_outcome_count,
            "unique_models": list(self.unique_models),
            "protocols": list(self.protocols),
            "model_counts_by_protocol": self.model_counts_by_protocol,
            "outcome_counts_by_protocol": self.outcome_counts_by_protocol,
            "missing_counts_by_protocol": self.missing_counts_by_protocol,
            "source_references": list(self.source_references),
        }


def audit_anchors(
    anchors: Iterable[PublicAnchor],
    *,
    source_references: Iterable[Mapping[str, Any]] = (),
) -> AnchorAudit:
    rows = tuple(anchors)
    observations = [observation for row in rows for observation in row.observations]
    protocols = sorted({observation.protocol_id for observation in observations})
    model_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    outcome_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    for observation in observations:
        model_counts[observation.protocol_id][observation.model_id] += 1
        if observation.outcome is None:
            missing_counts[observation.protocol_id] += 1
        else:
            outcome_counts[observation.protocol_id] += 1
    families = sorted({row.benchmark_family for row in rows})
    family = families[0] if len(families) == 1 else ",".join(families)
    return AnchorAudit(
        benchmark_family=family,
        prompt_count=len(rows),
        observation_count=len(observations),
        usable_outcome_count=sum(outcome_counts.values()),
        missing_outcome_count=sum(missing_counts.values()),
        unique_models=tuple(sorted({observation.model_id for observation in observations})),
        protocols=tuple(protocols),
        model_counts_by_protocol={
            protocol: dict(sorted(model_counts[protocol].items())) for protocol in protocols
        },
        outcome_counts_by_protocol=dict(sorted(outcome_counts.items())),
        missing_counts_by_protocol=dict(sorted(missing_counts.items())),
        source_references=tuple(dict(reference) for reference in source_references),
    )
