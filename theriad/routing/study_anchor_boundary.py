"""Fail-closed boundary between public anchors and the Router V2 study.

Public-anchor outcomes are an independent reference corpus.  Exact normalized
prompt overlaps with the study cannot receive anchor-derived neighbor,
difficulty, disagreement, or calibration features.  The exclusion is keyed by
the frozen overlap manifest, not by split membership alone.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .public_anchors import (
    AnchorFeatureIndex,
    AnchorNeighbor,
    PublicAnchor,
    normalize_prompt,
    prompt_hash,
)

PUBLIC_ANCHOR_OVERLAP_POLICY = "exclude_all_overlap_hashes_from_study_anchor_features_v1"
STUDY_ANCHOR_BOUNDARY_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty_features(*, reason: str, retriever_kind: str) -> dict[str, Any]:
    return {
        "anchor_boundary_status": reason,
        "public_anchor_outcomes_used": False,
        "anchor_retriever_kind": retriever_kind,
        "neighbor_count": 0,
        "nearest_anchor_similarity": None,
        "neighbor_mean_difficulty": None,
        "neighbor_weighted_difficulty": None,
        "neighbor_mean_disagreement": None,
        "neighbor_specialization_score": None,
        "neighbor_model_success_rates": {},
        "neighbor_domain_distribution": {},
        "neighbor_benchmark_family_distribution": {},
        "neighbor_ids": [],
    }


@dataclass(frozen=True)
class StudyAnchorExclusionBoundary:
    schema_version: int
    policy: str
    study_corpus_checksum: str
    anchor_manifest_sha256: str
    overlap_hashes: tuple[str, ...]
    by_study_split: dict[str, int]
    by_anchor_split: dict[str, int]
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STUDY_ANCHOR_BOUNDARY_SCHEMA_VERSION:
            raise ValueError("unsupported study-anchor boundary schema")
        if self.policy != PUBLIC_ANCHOR_OVERLAP_POLICY:
            raise ValueError("unsupported public-anchor overlap policy")
        if not self.study_corpus_checksum:
            raise ValueError("study-anchor boundary needs a study corpus checksum")
        if not self.anchor_manifest_sha256:
            raise ValueError("study-anchor boundary needs an anchor manifest hash")
        if len(set(self.overlap_hashes)) != len(self.overlap_hashes):
            raise ValueError("study-anchor boundary contains duplicate hashes")

    def excludes_prompt(self, prompt: str) -> bool:
        return prompt_hash(normalize_prompt(prompt)) in set(self.overlap_hashes)

    def excludes_hash(self, normalized_hash: str) -> bool:
        return str(normalized_hash).lower() in set(self.overlap_hashes)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "study_corpus_checksum": self.study_corpus_checksum,
            "anchor_manifest_sha256": self.anchor_manifest_sha256,
            "overlap_hashes": list(self.overlap_hashes),
            "overlap_count": len(self.overlap_hashes),
            "by_study_split": dict(sorted(self.by_study_split.items())),
            "by_anchor_split": dict(sorted(self.by_anchor_split.items())),
            "public_anchor_outcomes_used_for_study": False,
        }
        if self.manifest_hash:
            payload["manifest_hash"] = self.manifest_hash
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> StudyAnchorExclusionBoundary:
        if not isinstance(raw, Mapping):
            raise ValueError("study-anchor boundary must be an object")
        without_hash = {key: value for key, value in raw.items() if key != "manifest_hash"}
        stored = str(raw.get("manifest_hash") or "")
        if stored and hashlib.sha256(_canonical_json(without_hash).encode("utf-8")).hexdigest() != stored:
            raise ValueError("study-anchor boundary manifest hash does not match its contents")
        overlap_hashes = tuple(sorted(str(value).lower() for value in raw.get("overlap_hashes", [])))
        if int(raw.get("overlap_count") or len(overlap_hashes)) != len(overlap_hashes):
            raise ValueError("study-anchor boundary overlap count does not match hashes")
        return cls(
            schema_version=int(raw.get("schema_version") or 0),
            policy=str(raw.get("policy") or ""),
            study_corpus_checksum=str(raw.get("study_corpus_checksum") or ""),
            anchor_manifest_sha256=str(raw.get("anchor_manifest_sha256") or ""),
            overlap_hashes=overlap_hashes,
            by_study_split={str(k): int(v) for k, v in (raw.get("by_study_split") or {}).items()},
            by_anchor_split={str(k): int(v) for k, v in (raw.get("by_anchor_split") or {}).items()},
            manifest_hash=stored,
        )

    @classmethod
    def load(cls, path: str | Path) -> StudyAnchorExclusionBoundary:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"study-anchor boundary is unreadable: {path}") from exc
        return cls.from_dict(raw)


class StudyAnchorFeatureIndex:
    """Anchor index that requires a visible-input hash at feature lookup."""

    def __init__(self, index: AnchorFeatureIndex | None, boundary: StudyAnchorExclusionBoundary):
        self.index = index
        self.boundary = boundary
        self.retriever_kind = index.retriever_kind if index is not None else "empty_excluded_index"

    def retrieve(
        self,
        vector: Sequence[float],
        *,
        query_prompt_hash: str,
        top_k: int = 8,
        exclude_prompt_ids: Iterable[str] = (),
        exclude_benchmark_families: Iterable[str] = (),
    ) -> tuple[AnchorNeighbor, ...]:
        if self.boundary.excludes_hash(query_prompt_hash):
            return ()
        if self.index is None:
            return ()
        return self.index.retrieve(
            vector,
            top_k=top_k,
            exclude_prompt_ids=exclude_prompt_ids,
            exclude_benchmark_families=exclude_benchmark_families,
        )

    def features_for_vector(
        self,
        vector: Sequence[float],
        *,
        query_prompt_hash: str,
        top_k: int = 8,
        exclude_prompt_ids: Iterable[str] = (),
        exclude_benchmark_families: Iterable[str] = (),
    ) -> dict[str, Any]:
        if self.boundary.excludes_hash(query_prompt_hash):
            return _empty_features(
                reason="excluded_exact_overlap",
                retriever_kind=self.retriever_kind,
            )
        if self.index is None:
            return _empty_features(reason="empty_anchor_index", retriever_kind=self.retriever_kind)
        return {
            "anchor_boundary_status": "eligible_non_overlap",
            "public_anchor_outcomes_used": True,
            **self.index.features_for_vector(
                vector,
                top_k=top_k,
                exclude_prompt_ids=exclude_prompt_ids,
                exclude_benchmark_families=exclude_benchmark_families,
            ),
        }

    def features_for_text(self, prompt: str, *, top_k: int = 8) -> dict[str, Any]:
        if self.boundary.excludes_prompt(prompt):
            return _empty_features(
                reason="excluded_exact_overlap",
                retriever_kind=self.retriever_kind,
            )
        if self.index is None:
            return _empty_features(reason="empty_anchor_index", retriever_kind=self.retriever_kind)
        return {
            "anchor_boundary_status": "eligible_non_overlap",
            "public_anchor_outcomes_used": True,
            **self.index.features_for_text(prompt, top_k=top_k),
        }


def build_study_anchor_feature_index(
    anchors: Sequence[PublicAnchor],
    boundary: StudyAnchorExclusionBoundary,
    *,
    vectors: Mapping[str, Sequence[float]] | None = None,
    retriever: str = "tfidf_baseline",
) -> StudyAnchorFeatureIndex:
    """Build an index after removing every exact study/anchor overlap."""

    filtered = tuple(anchor for anchor in anchors if not boundary.excludes_hash(anchor.normalized_prompt_hash))
    if not filtered:
        return StudyAnchorFeatureIndex(None, boundary)
    if vectors is not None:
        selected = {anchor.prompt_id: vectors[anchor.prompt_id] for anchor in filtered}
        index = AnchorFeatureIndex.from_embeddings(filtered, selected)
    elif retriever == "tfidf_baseline":
        index = AnchorFeatureIndex.from_tfidf(filtered)
    else:
        raise ValueError("semantic anchor indexes require explicit vectors")
    return StudyAnchorFeatureIndex(index, boundary)


def anchor_file_sha256(path: str | Path) -> str:
    return _sha256_path(Path(path))


__all__ = [
    "PUBLIC_ANCHOR_OVERLAP_POLICY",
    "STUDY_ANCHOR_BOUNDARY_SCHEMA_VERSION",
    "StudyAnchorExclusionBoundary",
    "StudyAnchorFeatureIndex",
    "anchor_file_sha256",
    "build_study_anchor_feature_index",
]
