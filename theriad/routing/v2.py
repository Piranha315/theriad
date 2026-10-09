"""Artifact-backed, calibrated per-model routing.

Router V2 deliberately has no training-framework dependency.  Offline tooling
writes a small JSON artifact; the live process loads it, extracts the same
structural features used by layer 1, and predicts success/cost/latency for each
eligible model.  A missing or invalid artifact disables V2 rather than changing
the production route.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from theriad.models.tiers import ModelCandidate
from theriad.routing.difficulty import featurise

ARTIFACT_SCHEMA_VERSION = 1
VALID_MODES = frozenset({"off", "shadow", "canary", "on"})
GENERAL_LANE = "general"
CODE_LANE = "code"
DEFAULT_FALLBACK_MODEL = "openai/gpt-6-luna"


class EmbeddingBackend(Protocol):
    def encode(self, text: str) -> Sequence[float]: ...


class SentenceTransformerEmbeddingBackend:
    """Explicitly opted-in backend for profiled semantic artifacts."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ValueError("sentence-transformers is not installed") from exc
        self._model = SentenceTransformer(model_name)

    def encode(self, text: str) -> Sequence[float]:
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return [float(value) for value in vector]


class OnnxEmbeddingBackend:
    """Mean-pooled, L2-normalised sentence embeddings via onnxruntime.

    Same vectors as the sentence-transformers encoder it was exported from, but
    without importing torch: the dependency, not the model, is what breaks the
    runtime memory gate. Needs only ``onnxruntime`` and ``tokenizers``.
    """

    def __init__(self, model_path: str | Path, tokenizer_path: str | Path, max_seq_length: int = 256,
                 intra_op_threads: int = 1):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ValueError("onnxruntime and tokenizers are required for ONNX embeddings") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(intra_op_threads))
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_seq_length)

    def encode(self, text: str) -> Sequence[float]:
        encoded = self._tokenizer.encode(text)
        feeds = {
            "input_ids": [encoded.ids],
            "attention_mask": [encoded.attention_mask],
            "token_type_ids": [encoded.type_ids],
        }
        import numpy as np

        output = self._session.run(
            None, {k: np.asarray(v, dtype=np.int64) for k, v in feeds.items()}
        )[0][0]
        return [float(value) for value in output]


@dataclass(frozen=True)
class RouterPrediction:
    model: str
    eligible: bool
    exclusion_reason: str | None
    p_success: float
    p_success_lower: float
    expected_output_tokens: float
    expected_cost_usd: float | None
    expected_latency_ms: float
    uncertainty: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouterV2Decision:
    router_version: str
    mode: str
    lane: str
    knee_threshold: float
    selected_model: str
    candidate_scores: tuple[RouterPrediction, ...]
    ood_score: float
    selected_propensity: float
    shadow: bool
    applied: bool
    shadow_challenger: str | None = None
    reason: str = ""
    decision_id: str = ""
    experiment_id: str | None = None
    assignment_key: str | None = None
    assigned_arm: str | None = None
    artifact_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["candidate_scores"] = [row.to_dict() for row in self.candidate_scores]
        return raw


@dataclass(frozen=True)
class _ModelHead:
    intercept: float
    weights: dict[str, float]
    embedding_weights: tuple[float, ...] = ()
    embedding_neighbors: tuple[tuple[tuple[float, ...], float], ...] = ()
    knn_k: int = 16
    knn_alpha: float = 1.0
    knn_beta: float = 1.0
    platt_a: float = 1.0
    platt_b: float = 0.0
    uncertainty: float = 0.10
    sample_count: int = 0
    expected_output_tokens: float = 200.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0


@dataclass(frozen=True)
class _LaneConfig:
    knee_threshold: float
    fallback_model: str = DEFAULT_FALLBACK_MODEL
    models: tuple[str, ...] = ()
    fixed_model: str | None = None


@dataclass(frozen=True)
class _Projection:
    """Linear projection applied to encoder output before the heads run.

    ``basis`` is K×D (rows are principal axes), ``mean`` is D. The runtime
    computes ``z = basis @ (vector - mean)``. ``ood_reconstruction_p99``, when
    set, is the training prompts' 99th-percentile reconstruction error
    ``|| (v - mean) - basisᵀ z ||``; a request's error divided by it is its
    out-of-distribution score, so values above 1.0 are further out than 99%
    of what the artifact was fitted on.
    """
    encoder: str
    mean: tuple[float, ...]
    basis: tuple[tuple[float, ...], ...]
    ood_reconstruction_p99: float | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> "_Projection":
        if not isinstance(raw, dict):
            raise ValueError("projection must be an object")
        encoder = raw.get("encoder")
        if not isinstance(encoder, str) or not encoder.strip():
            raise ValueError("projection.encoder must be a non-empty string")
        try:
            mean = tuple(float(v) for v in raw["mean"])
            basis = tuple(tuple(float(v) for v in row) for row in raw["basis"])
            p99 = raw.get("ood_reconstruction_p99")
            p99 = None if p99 is None else float(p99)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("projection mean/basis must be numeric") from exc
        if not mean or not basis or any(len(row) != len(mean) for row in basis):
            raise ValueError("projection basis must be K×D with D = len(mean)")
        if not all(math.isfinite(v) for v in mean) or not all(
            math.isfinite(v) for row in basis for v in row
        ):
            raise ValueError("projection values must be finite")
        if p99 is not None and (not math.isfinite(p99) or p99 <= 0):
            raise ValueError("ood_reconstruction_p99 must be positive and finite")
        return cls(encoder=encoder, mean=mean, basis=basis, ood_reconstruction_p99=p99)

    def apply(self, vector: Sequence[float]) -> tuple[list[float], float | None]:
        """(projected vector, OOD score or None)."""
        if len(vector) != len(self.mean):
            raise ValueError("encoder output does not match projection dimension")
        centered = [float(v) - m for v, m in zip(vector, self.mean, strict=True)]
        z = [sum(b * c for b, c in zip(row, centered, strict=True)) for row in self.basis]
        if self.ood_reconstruction_p99 is None:
            return z, None
        residual = list(centered)
        for weight, row in zip(z, self.basis, strict=True):
            for i, b in enumerate(row):
                residual[i] -= weight * b
        error = math.sqrt(sum(r * r for r in residual))
        return z, error / self.ood_reconstruction_p99


@dataclass(frozen=True)
class RouterV2Artifact:
    router_version: str
    feature_mean: dict[str, float]
    feature_stdev: dict[str, float]
    lanes: dict[str, _LaneConfig]
    models: dict[str, _ModelHead]
    ood_max_abs_z: float = 4.0
    embedding_model: str | None = None
    embedding_incremental_rss_mb: float | None = None
    embedding_p95_latency_ms: float | None = None
    generated_at: str | None = None
    family: str = "surface_logistic"
    research_only: bool = False
    projection: _Projection | None = None
    embedding_runtime: dict[str, Any] | None = None
    """Optional {"backend": "onnx", "model_path", "tokenizer_path", "max_seq_length"},
    paths relative to the artifact file."""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RouterV2Artifact:
        if not isinstance(raw, dict):
            raise ValueError("router artifact root must be an object")
        try:
            schema_version = int(raw.get("schema_version", -1))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("router artifact schema_version must be an integer") from exc
        if schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported router artifact schema")
        version = str(raw.get("router_version") or "").strip()
        if not version:
            raise ValueError("router_version is required")
        lane_rows = raw.get("lanes")
        model_rows = raw.get("models")
        if not isinstance(lane_rows, dict) or not isinstance(model_rows, dict):
            raise ValueError("lanes and models must be objects")
        if not {GENERAL_LANE, CODE_LANE}.issubset(lane_rows):
            raise ValueError("router artifact must configure both general and code lanes")
        if not model_rows:
            raise ValueError("artifact contains no model heads")
        lanes: dict[str, _LaneConfig] = {}
        for name, row in lane_rows.items():
            if not isinstance(row, dict):
                raise ValueError(f"invalid lane configuration for {name}")
            if "knee_threshold" not in row:
                raise ValueError(f"missing knee threshold for {name}")
            try:
                threshold = float(row["knee_threshold"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid knee threshold for {name}") from exc
            if not math.isfinite(threshold):
                raise ValueError(f"invalid knee threshold for {name}")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"invalid knee threshold for {name}")
            active_models = row.get("models", [])
            if not isinstance(active_models, list):
                raise ValueError(f"models for lane {name} must be a list")
            lanes[str(name)] = _LaneConfig(
                knee_threshold=threshold,
                fallback_model=str(row.get("fallback_model") or DEFAULT_FALLBACK_MODEL),
                models=tuple(str(model) for model in active_models if str(model)),
                fixed_model=str(row["fixed_model"]) if row.get("fixed_model") else None,
            )
        models: dict[str, _ModelHead] = {}
        for model, row in model_rows.items():
            if not str(model).strip() or not isinstance(row, dict):
                raise ValueError(f"invalid model head for {model}")
            weights = row.get("weights", {})
            embedding_weights = row.get("embedding_weights", [])
            embedding_neighbors = row.get("embedding_neighbors", [])
            if (
                not isinstance(weights, dict)
                or not isinstance(embedding_weights, list)
                or not isinstance(embedding_neighbors, list)
            ):
                raise ValueError(f"invalid model head for {model}")
            neighbors: list[tuple[tuple[float, ...], float]] = []
            for neighbor in embedding_neighbors:
                if not isinstance(neighbor, dict) or not isinstance(neighbor.get("vector"), list):
                    raise ValueError(f"invalid embedding neighbor for {model}")
                try:
                    vector = tuple(float(value) for value in neighbor["vector"])
                    success = float(neighbor.get("success", 0.0))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"invalid embedding neighbor values for {model}") from exc
                if not vector or not all(math.isfinite(value) for value in vector):
                    raise ValueError(f"invalid embedding neighbor values for {model}")
                if not math.isfinite(success) or not 0.0 <= success <= 1.0:
                    raise ValueError(f"invalid embedding neighbor values for {model}")
                neighbors.append((vector, success))
            try:
                numeric = {
                    "intercept": float(row.get("intercept", 0.0)),
                    "platt_a": float(row.get("platt_a", 1.0)),
                    "platt_b": float(row.get("platt_b", 0.0)),
                    "uncertainty": float(row.get("uncertainty", 0.10)),
                    "expected_output_tokens": float(row.get("expected_output_tokens", 200.0)),
                    "latency_p50_ms": float(row.get("latency_p50_ms", 0.0)),
                    "latency_p95_ms": float(row.get("latency_p95_ms", 0.0)),
                }
                embedding_vector = tuple(float(value) for value in embedding_weights)
                weight_map = {str(k): float(v) for k, v in weights.items()}
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid numeric model head for {model}") from exc
            if any(not math.isfinite(value) for value in (*numeric.values(), *embedding_vector, *weight_map.values())):
                raise ValueError(f"invalid numeric model head for {model}")
            models[str(model)] = _ModelHead(
                intercept=numeric["intercept"],
                weights=weight_map,
                embedding_weights=embedding_vector,
                embedding_neighbors=tuple(neighbors),
                knn_k=max(1, int(row.get("knn_k", 16))),
                knn_alpha=max(1e-9, float(row.get("knn_alpha", 1.0))),
                knn_beta=max(1e-9, float(row.get("knn_beta", 1.0))),
                platt_a=numeric["platt_a"],
                platt_b=numeric["platt_b"],
                uncertainty=max(0.0, numeric["uncertainty"]),
                sample_count=max(0, int(row.get("sample_count", 0))),
                expected_output_tokens=max(1.0, numeric["expected_output_tokens"]),
                latency_p50_ms=max(0.0, numeric["latency_p50_ms"]),
                latency_p95_ms=max(0.0, numeric["latency_p95_ms"]),
            )
        for lane, config in lanes.items():
            if any(model not in models for model in config.models):
                raise ValueError(f"lane {lane} references an unknown model")
            if config.fixed_model and (
                config.fixed_model not in models
                or (config.models and config.fixed_model not in config.models)
            ):
                raise ValueError(f"lane {lane} references an ineligible fixed model")
        feature_mean_raw = raw.get("feature_mean", {})
        feature_stdev_raw = raw.get("feature_stdev", {})
        if not isinstance(feature_mean_raw, dict) or not isinstance(feature_stdev_raw, dict):
            raise ValueError("feature_mean and feature_stdev must be objects")
        try:
            feature_mean = {str(k): float(v) for k, v in feature_mean_raw.items()}
            feature_stdev = {str(k): float(v) for k, v in feature_stdev_raw.items()}
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("feature statistics must be numeric") from exc
        if any(not math.isfinite(value) for value in (*feature_mean.values(), *feature_stdev.values())):
            raise ValueError("feature statistics must be finite")
        if any(value <= 0 for value in feature_stdev.values()):
            raise ValueError("feature standard deviations must be positive")
        try:
            ood_max_abs_z = float(raw.get("ood_max_abs_z", 4.0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("ood_max_abs_z must be numeric") from exc
        if not math.isfinite(ood_max_abs_z) or ood_max_abs_z <= 0:
            raise ValueError("ood_max_abs_z must be positive and finite")
        embedding_model = raw.get("embedding_model")
        if embedding_model is not None and (
            not isinstance(embedding_model, str) or not embedding_model.strip()
        ):
            raise ValueError("embedding_model must be a non-empty string")
        runtime_profile = raw.get("runtime_profile")
        if runtime_profile is not None and not isinstance(runtime_profile, dict):
            raise ValueError("runtime_profile must be an object")
        incremental_rss = runtime_profile.get("incremental_rss_mb") if runtime_profile else None
        embedding_p95 = runtime_profile.get("p95_latency_ms") if runtime_profile else None
        for name, value in (("incremental_rss_mb", incremental_rss), ("p95_latency_ms", embedding_p95)):
            if value is not None:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"runtime profile {name} must be numeric") from exc
                if not math.isfinite(numeric_value) or numeric_value < 0:
                    raise ValueError(f"runtime profile {name} must be finite and non-negative")
        projection = (
            _Projection.from_dict(raw["projection"]) if raw.get("projection") is not None else None
        )
        if projection is not None and embedding_model and projection.encoder != embedding_model:
            raise ValueError("projection.encoder must match embedding_model")
        return cls(
            router_version=version,
            feature_mean=feature_mean,
            feature_stdev=feature_stdev,
            lanes=lanes,
            models=models,
            ood_max_abs_z=ood_max_abs_z,
            embedding_model=embedding_model,
            embedding_incremental_rss_mb=(float(incremental_rss) if incremental_rss is not None else None),
            embedding_p95_latency_ms=(float(embedding_p95) if embedding_p95 is not None else None),
            generated_at=raw.get("generated_at"),
            family=str(raw.get("family") or "surface_logistic"),
            research_only=raw.get("research_only", False) is not False,
            projection=projection,
            embedding_runtime=_embedding_runtime(raw.get("embedding_runtime")),
        )


def _embedding_runtime(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or raw.get("backend") != "onnx":
        raise ValueError("embedding_runtime must be {'backend': 'onnx', ...}")
    for key in ("model_path", "tokenizer_path"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"embedding_runtime.{key} must be a non-empty string")
    return dict(raw)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def lane_for_tier(tier: str) -> str:
    return CODE_LANE if "code" in tier else GENERAL_LANE


def normalized_prompt_hash(prompt: str) -> str:
    normalized = " ".join(prompt.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deterministic_fraction(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


_SENSITIVE_PATTERNS = (
    re.compile(r"\b(?:password|passcode|api[_ -]?key|secret|private key)\b", re.I),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
)


def is_sensitive_prompt(prompt: str) -> bool:
    """Conservative shadow-exploration exclusion; never used to block the main call."""

    return any(pattern.search(prompt) for pattern in _SENSITIVE_PATTERNS)


class RouterV2:
    def __init__(
        self,
        artifact: RouterV2Artifact,
        *,
        embedding_backend: EmbeddingBackend | None = None,
    ):
        self.artifact = artifact
        self.embedding_backend = embedding_backend

    @classmethod
    def from_path(
        cls, path: str | Path, *, embedding_backend: EmbeddingBackend | None = None
    ) -> RouterV2:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("router artifact root must be an object")
        artifact = RouterV2Artifact.from_dict(raw)
        has_embedding_heads = any(
            head.embedding_weights or head.embedding_neighbors
            for head in artifact.models.values()
        )
        if has_embedding_heads and not artifact.embedding_model:
            raise ValueError("embedding artifact must declare embedding_model")
        if artifact.embedding_model and embedding_backend is None:
            enabled = os.getenv("THERIAD_ROUTER_V2_ENABLE_EMBEDDINGS", "false").lower() in {
                "1",
                "true",
                "yes",
            }
            rss = artifact.embedding_incremental_rss_mb
            latency = artifact.embedding_p95_latency_ms
            if not enabled:
                raise ValueError("artifact requires an embedding backend")
            if rss is None or latency is None or rss > 120.0 or latency > 20.0:
                raise ValueError("embedding artifact has not passed runtime resource gates")
            runtime = artifact.embedding_runtime
            if runtime is not None:
                base = Path(path).resolve().parent
                embedding_backend = OnnxEmbeddingBackend(
                    base / runtime["model_path"],
                    base / runtime["tokenizer_path"],
                    int(runtime.get("max_seq_length", 256)),
                    int(runtime.get("intra_op_threads", 1)),
                )
            else:
                embedding_backend = SentenceTransformerEmbeddingBackend(artifact.embedding_model)
        return cls(artifact, embedding_backend=embedding_backend)

    def _features(self, prompt: str) -> tuple[dict[str, float], float]:
        if not self.artifact.feature_mean:
            # A purely learned (embedding-only) artifact carries no surface
            # features; running the regex extractor would compute nothing used.
            return {}, 0.0
        raw = featurise(prompt)
        standardized: dict[str, float] = {}
        maximum = 0.0
        for name, mean in self.artifact.feature_mean.items():
            z = (raw.get(name, mean) - mean) / self.artifact.feature_stdev.get(name, 1.0)
            maximum = max(maximum, abs(z))
            standardized[name] = max(-6.0, min(6.0, z))
        return standardized, maximum / self.artifact.ood_max_abs_z

    @staticmethod
    def _knn_probability(
        embedding: Sequence[float],
        head: _ModelHead,
    ) -> float:
        if not head.embedding_neighbors:
            return 0.5
        dimension = len(embedding)
        distances: list[tuple[float, float]] = []
        for vector, success in head.embedding_neighbors:
            if len(vector) != dimension:
                continue
            distance = sum((float(left) - float(right)) ** 2 for left, right in zip(vector, embedding, strict=True))
            distances.append((distance, success))
        if not distances:
            return 0.5
        nearest = sorted(distances, key=lambda row: row[0])[: head.knn_k]
        successes = sum(success for _distance, success in nearest)
        return (successes + head.knn_alpha) / (
            len(nearest) + head.knn_alpha + head.knn_beta
        )

    def predict(
        self,
        prompt: str,
        candidates: Sequence[ModelCandidate],
        *,
        lane: str,
        unavailable_models: frozenset[str] = frozenset(),
    ) -> RouterV2Decision:
        if lane not in self.artifact.lanes:
            raise ValueError(f"artifact has no configuration for lane {lane!r}")
        lane_config = self.artifact.lanes[lane]
        features, ood_score = self._features(prompt)
        embedding: Sequence[float] = ()
        if self.artifact.embedding_model:
            if self.embedding_backend is None:
                raise ValueError("embedding artifact has no embedding backend")
            embedding = self.embedding_backend.encode(prompt)
            if self.artifact.projection is not None:
                embedding, embedding_ood = self.artifact.projection.apply(embedding)
                if embedding_ood is not None:
                    ood_score = max(ood_score, embedding_ood)

        predictions: list[RouterPrediction] = []
        for candidate in candidates:
            head = self.artifact.models.get(candidate.model)
            reason: str | None = None
            if candidate.model in unavailable_models:
                reason = "provider_unavailable"
            elif head is None:
                reason = "unmeasured_model"
            elif lane_config.models and candidate.model not in lane_config.models:
                reason = "model_not_promoted_for_lane"
            elif not all(
                price is not None and math.isfinite(price) and price >= 0.0
                for price in (candidate.input_cost_per_million, candidate.output_cost_per_million)
            ):
                reason = "missing_price"
            elif head.embedding_weights and len(head.embedding_weights) != len(embedding):
                reason = "embedding_dimension_mismatch"

            if head is None:
                predictions.append(
                    RouterPrediction(candidate.model, False, reason, 0.0, 0.0, 0.0, None, 0.0, 1.0)
                )
                continue

            if head.embedding_neighbors and not reason:
                knn_probability = self._knn_probability(embedding, head)
                logit = math.log(
                    max(1e-9, min(1 - 1e-9, knn_probability))
                    / max(1e-9, 1 - min(1 - 1e-9, knn_probability))
                )
            else:
                logit = head.intercept + sum(
                    weight * features.get(name, 0.0) for name, weight in head.weights.items()
                )
                if head.embedding_weights and not reason:
                    logit += sum(a * b for a, b in zip(head.embedding_weights, embedding, strict=True))
            calibrated = _sigmoid(head.platt_a * logit + head.platt_b)
            sampling_var = calibrated * (1.0 - calibrated) / max(1, head.sample_count + 1)
            total_uncertainty = min(1.0, math.sqrt(head.uncertainty**2 + sampling_var))
            lower = max(0.0, calibrated - 1.645 * total_uncertainty)
            input_tokens = max(1.0, float(len(prompt.split())) * 1.3)
            prices = (candidate.input_cost_per_million, candidate.output_cost_per_million)
            expected_cost = (
                (
                    input_tokens * float(prices[0])
                    + head.expected_output_tokens * float(prices[1])
                ) / 1_000_000.0
                if all(price is not None and math.isfinite(price) and price >= 0.0 for price in prices)
                else None
            )
            predictions.append(
                RouterPrediction(
                    model=candidate.model,
                    eligible=reason is None,
                    exclusion_reason=reason,
                    p_success=calibrated,
                    p_success_lower=lower,
                    expected_output_tokens=head.expected_output_tokens,
                    expected_cost_usd=expected_cost,
                    expected_latency_ms=head.latency_p95_ms or head.latency_p50_ms,
                    uncertainty=total_uncertainty,
                )
            )

        eligible = [prediction for prediction in predictions if prediction.eligible]
        if not eligible:
            raise ValueError("artifact has no eligible measured candidate")
        if lane_config.fixed_model:
            selected = next((p for p in eligible if p.model == lane_config.fixed_model), None)
            if selected is None:
                raise ValueError("fixed policy has no eligible candidate")
            reason = "fixed_model_policy"
        elif ood_score > 1.0:
            fallback = next(
                (p for p in eligible if p.model == lane_config.fallback_model),
                max(eligible, key=lambda p: (p.p_success, -p.expected_cost_usd, p.model)),
            )
            selected = fallback
            reason = "ood_fallback"
        else:
            clears = [
                prediction
                for prediction in eligible
                if prediction.p_success_lower >= lane_config.knee_threshold
            ]
            if clears:
                selected = min(clears, key=lambda p: (p.expected_cost_usd, -p.p_success, p.model))
                reason = "cheapest_above_knee"
            else:
                selected = max(
                    eligible, key=lambda p: (p.p_success, -p.expected_cost_usd, p.model)
                )
                reason = "highest_success_fallback"
        return RouterV2Decision(
            router_version=self.artifact.router_version,
            mode="on",
            lane=lane,
            knee_threshold=lane_config.knee_threshold,
            selected_model=selected.model,
            candidate_scores=tuple(predictions),
            ood_score=ood_score,
            selected_propensity=1.0,
            shadow=False,
            applied=True,
            reason=reason,
            decision_id=str(uuid.uuid4()),
        )


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_ARTIFACT = _PROJECT_ROOT / "data" / "router-v2-shadow" / "router-v2-embedding.json"


def _resolve_artifact_path() -> str:
    explicit = os.getenv("THERIAD_ROUTER_V2_ARTIFACT", "").strip()
    if explicit:
        return explicit
    if _DEFAULT_ARTIFACT.is_file():
        return str(_DEFAULT_ARTIFACT)
    return ""


def _env_mode() -> str:
    explicit = os.getenv("THERIAD_ROUTER_V2_MODE", "").strip().lower()
    if explicit:
        return explicit if explicit in VALID_MODES else "off"
    if _resolve_artifact_path():
        return "shadow"
    return "off"


def learned_only_active() -> bool:
    """True when V2 should make every routing decision itself (mode "on" with an artifact).

    Callers use this to skip the rule/feature routing layers entirely instead of
    letting them pick the tier V2 then chooses within.
    """
    return _env_mode() == "on" and bool(_resolve_artifact_path())


def _env_percentage(name: str, default: float) -> float:
    try:
        return max(0.0, min(100.0, float(os.getenv(name, str(default)))))
    except ValueError:
        return default


@lru_cache(maxsize=4)
def _load_router(path: str) -> RouterV2 | None:
    try:
        return RouterV2.from_path(path)
    except (
        AssertionError,
        AttributeError,
        KeyError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None


def clear_router_cache() -> None:
    _load_router.cache_clear()


def resolve_from_environment(
    prompt: str,
    candidates: Sequence[ModelCandidate],
    *,
    tier: str,
    baseline_model: str,
    unavailable_models: frozenset[str] = frozenset(),
    episode_id: str | None = None,
    experiment_id: str | None = None,
) -> RouterV2Decision | None:
    """Evaluate V2 according to rollout mode; return None on any unsafe state."""

    mode = _env_mode()
    if mode == "off" or not prompt:
        return None
    artifact_path = _resolve_artifact_path()
    if not artifact_path:
        return None
    router = _load_router(artifact_path)
    if router is None or router.artifact.research_only:
        return None
    try:
        proposed = router.predict(
            prompt, candidates, lane=lane_for_tier(tier), unavailable_models=unavailable_models
        )
    except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError):
        return None

    assignment_key = str(episode_id or normalized_prompt_hash(prompt))
    bucket = deterministic_fraction(
        f"{experiment_id or 'router-v2-canary'}:{assignment_key}"
    )
    canary_pct = _env_percentage("THERIAD_ROUTER_V2_CANARY_PCT", 5.0)
    applies = mode == "on" or (mode == "canary" and bucket < canary_pct / 100.0)
    shadow = mode == "shadow" or (mode == "canary" and not applies)
    challenger = proposed.selected_model if proposed.selected_model != baseline_model else None
    if shadow and is_sensitive_prompt(prompt):
        challenger = None
    propensity = 1.0
    if mode == "canary":
        propensity = canary_pct / 100.0 if applies else 1.0 - canary_pct / 100.0
    artifact_hash = hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()
    return RouterV2Decision(
        router_version=proposed.router_version,
        mode=mode,
        lane=proposed.lane,
        knee_threshold=proposed.knee_threshold,
        selected_model=proposed.selected_model if applies else baseline_model,
        candidate_scores=proposed.candidate_scores,
        ood_score=proposed.ood_score,
        selected_propensity=propensity,
        shadow=shadow,
        applied=applies,
        shadow_challenger=challenger,
        reason=proposed.reason if applies else f"{mode}_prediction_only",
        decision_id=proposed.decision_id,
        experiment_id=experiment_id,
        assignment_key=assignment_key,
        assigned_arm="treatment" if applies else "baseline",
        artifact_hash=artifact_hash,
    )


def observation_rows(
    decision: RouterV2Decision,
    *,
    prompt: str,
    actual_model: str | None = None,
    actual_cost_usd: float | None = None,
    actual_latency_ms: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Turn one decision into privacy-preserving normalized storage rows."""

    prompt_hash = normalized_prompt_hash(prompt)
    rows: list[dict[str, Any]] = []
    for prediction in decision.candidate_scores:
        role = "selected" if prediction.model == actual_model else "candidate"
        if prediction.model == decision.shadow_challenger:
            role = "challenger"
        rows.append(
            {
                "decision_id": decision.decision_id,
                "experiment_id": decision.experiment_id,
                "assignment_key": decision.assignment_key,
                "assigned_arm": decision.assigned_arm,
                "applied": decision.applied,
                "override_reason": None if decision.applied else decision.reason,
                "artifact_hash": decision.artifact_hash,
                "prompt_hash": prompt_hash,
                "router_version": decision.router_version,
                "lane": decision.lane,
                "model": prediction.model,
                "role": role,
                "propensity": decision.selected_propensity if role in {"selected", "challenger"} else None,
                "p_success": prediction.p_success,
                "p_success_lower": prediction.p_success_lower,
                "expected_output_tokens": prediction.expected_output_tokens,
                "expected_cost_usd": prediction.expected_cost_usd,
                "expected_latency_ms": prediction.expected_latency_ms,
                "uncertainty": prediction.uncertainty,
                "eligible": prediction.eligible,
                "exclusion_reason": prediction.exclusion_reason,
                "realized_cost_usd": actual_cost_usd if prediction.model == actual_model else None,
                "realized_latency_ms": actual_latency_ms if prediction.model == actual_model else None,
                "input_tokens": input_tokens if prediction.model == actual_model else None,
                "output_tokens": output_tokens if prediction.model == actual_model else None,
            }
        )
    return rows


def canary_rollback_required(
    windows: Sequence[Mapping[str, Any]], *, serious_incident: bool = False
) -> bool:
    """Apply the preregistered disabled-canary rollback rule."""
    if serious_incident:
        return True
    if len(windows) < 2:
        return False
    for window in windows[-2:]:
        if min(int(window.get("treatment_matured", 0)), int(window.get("baseline_matured", 0))) < 100:
            return False
        if float(window.get("treatment_operational_failure_rate", 0.0)) - float(window.get("baseline_operational_failure_rate", 0.0)) > 0.02:
            continue
        if float(window.get("treatment_p95_latency_ms", 0.0)) > 1.20 * float(window.get("baseline_p95_latency_ms", 0.0)):
            continue
        return False
    return True
