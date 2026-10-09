"""Read-only importers for public benchmark prompts and model runs."""
from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .public_anchors import (
    AnchorObservation,
    PublicAnchor,
    canonical_hash,
    derive_anchor_metrics,
)


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


def _nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value = float(value)
    return value if value >= 0 else None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _outcome_from_record(record: Mapping[str, Any]) -> tuple[bool | None, str]:
    """Extract an explicitly published binary outcome, never infer from status."""

    for key in ("correct", "judge_correct", "outcome", "success"):
        if key in record:
            value = _bool_or_none(record.get(key))
            if value is not None:
                return value, key
    for parent_key in ("judge_result", "evaluation", "metrics"):
        parent = record.get(parent_key)
        if isinstance(parent, Mapping):
            for key in ("correct", "judge_correct", "outcome", "success"):
                if key in parent:
                    value = _bool_or_none(parent.get(key))
                    if value is not None:
                        return value, f"{parent_key}.{key}"
    return None, "missing"


def _query_id(record: Mapping[str, Any]) -> str:
    for key in ("query_id", "qid", "prompt_id", "id"):
        value = record.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return ""


def _prompt_from_record(record: Mapping[str, Any]) -> str:
    for key in ("question", "query", "prompt"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _reference_answer_from_record(record: Mapping[str, Any]) -> str | None:
    for key in ("correct_answer", "answer", "reference_answer"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return None


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, JSON arrays, or an evaluation summary's per-query list."""

    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped[0] in "[{":
        value = json.loads(text)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, Mapping)]
        if isinstance(value, Mapping):
            for key in ("per_query_metrics", "results", "records", "evaluations"):
                rows = value.get(key)
                if isinstance(rows, list):
                    return [dict(row) for row in rows if isinstance(row, Mapping)]
            return [dict(value)]
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(dict(value))
    return rows


@dataclass(frozen=True)
class BrowseCompRunSpec:
    """Metadata needed to compare one public run configuration."""

    path: Path
    model_id: str
    provider: str
    model_family: str
    protocol_id: str
    source_revision: str
    provenance_url: str
    license_name: str = "MIT"
    license_status: str = "public"
    retrieval_config: str = ""
    prompting_protocol: str = ""
    tool_availability: str = ""
    agent_scaffold: str = ""
    evaluation_path: Path | None = None
    evaluation_method: str = ""
    evaluation_date: str = ""
    collection_vintage: str = ""
    context_masking: str = ""

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "provider",
            "model_family",
            "protocol_id",
            "source_revision",
            "provenance_url",
        ):
            if not getattr(self, name):
                raise ValueError(f"BrowseComp run spec requires {name}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BrowseCompRunSpec:
        if not isinstance(raw, Mapping):
            raise ValueError("run manifest entries must be objects")
        path = Path(str(raw.get("path") or ""))
        evaluation = raw.get("evaluation_path")
        return cls(
            path=path,
            model_id=str(raw.get("model_id") or ""),
            provider=str(raw.get("provider") or ""),
            model_family=str(raw.get("model_family") or ""),
            protocol_id=str(raw.get("protocol_id") or ""),
            source_revision=str(raw.get("source_revision") or ""),
            provenance_url=str(raw.get("provenance_url") or ""),
            license_name=str(raw.get("license_name") or raw.get("license") or "MIT"),
            license_status=str(raw.get("license_status") or "public"),
            retrieval_config=str(raw.get("retrieval_config") or ""),
            prompting_protocol=str(raw.get("prompting_protocol") or ""),
            tool_availability=str(raw.get("tool_availability") or ""),
            agent_scaffold=str(raw.get("agent_scaffold") or ""),
            evaluation_path=Path(str(evaluation)) if evaluation else None,
            evaluation_method=str(raw.get("evaluation_method") or ""),
            evaluation_date=str(raw.get("evaluation_date") or ""),
            collection_vintage=str(raw.get("collection_vintage") or ""),
            context_masking=str(raw.get("context_masking") or ""),
        )


def load_run_manifest(path: str | Path) -> tuple[BrowseCompRunSpec, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        raw = raw.get("runs")
    if not isinstance(raw, list):
        raise ValueError("run manifest must be a JSON list or an object with runs")
    return tuple(BrowseCompRunSpec.from_dict(row) for row in raw)


@dataclass(frozen=True)
class EvaluatedLogSpec:
    """Metadata for one JSONL file with explicit per-item evaluation rows."""

    path: Path
    benchmark_family: str
    model_id: str
    provider: str
    model_family: str
    protocol_id: str
    source_revision: str
    provenance_url: str
    license_name: str
    license_status: str
    evaluation_method: str = "explicit_record_correct"
    domain: str = "research"
    template_family: str = ""
    collection_vintage: str = ""
    evaluation_date: str = ""
    retrieval_config: str = ""
    context_masking: str = ""
    prompting_protocol: str = ""
    tool_availability: str = ""
    agent_scaffold: str = ""
    split: str = "test"

    def __post_init__(self) -> None:
        for name in (
            "benchmark_family",
            "model_id",
            "provider",
            "model_family",
            "protocol_id",
            "source_revision",
            "provenance_url",
            "license_name",
            "license_status",
        ):
            if not getattr(self, name):
                raise ValueError(f"evaluated log spec requires {name}")
        if self.split not in {"train", "validation", "test", "ood"}:
            raise ValueError(f"invalid evaluated log split {self.split!r}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EvaluatedLogSpec:
        if not isinstance(raw, Mapping):
            raise ValueError("evaluated log manifest entries must be objects")
        return cls(
            path=Path(str(raw.get("path") or "")),
            benchmark_family=str(raw.get("benchmark_family") or ""),
            model_id=str(raw.get("model_id") or ""),
            provider=str(raw.get("provider") or ""),
            model_family=str(raw.get("model_family") or ""),
            protocol_id=str(raw.get("protocol_id") or ""),
            source_revision=str(raw.get("source_revision") or ""),
            provenance_url=str(raw.get("provenance_url") or ""),
            license_name=str(raw.get("license_name") or raw.get("license") or ""),
            license_status=str(raw.get("license_status") or ""),
            evaluation_method=str(raw.get("evaluation_method") or "explicit_record_correct"),
            domain=str(raw.get("domain") or "research"),
            template_family=str(raw.get("template_family") or ""),
            collection_vintage=str(raw.get("collection_vintage") or ""),
            evaluation_date=str(raw.get("evaluation_date") or ""),
            retrieval_config=str(raw.get("retrieval_config") or ""),
            context_masking=str(raw.get("context_masking") or ""),
            prompting_protocol=str(raw.get("prompting_protocol") or ""),
            tool_availability=str(raw.get("tool_availability") or ""),
            agent_scaffold=str(raw.get("agent_scaffold") or ""),
            split=str(raw.get("split") or "test"),
        )


def load_evaluated_log_manifest(path: str | Path) -> tuple[EvaluatedLogSpec, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        raw = raw.get("runs")
    if not isinstance(raw, list):
        raise ValueError("evaluated log manifest must be a JSON list or an object with runs")
    return tuple(EvaluatedLogSpec.from_dict(row) for row in raw)


def _load_evaluations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records = list(load_evaluation_csv(path).values()) if path.suffix.lower() == ".csv" else _read_json_records(path)
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        query_id = _query_id(record)
        if query_id:
            result[query_id] = record
    return result


def load_browsecomp_queries(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the decrypted query JSONL produced by the official helper."""

    records = _read_json_records(Path(path))
    queries: dict[str, dict[str, Any]] = {}
    for record in records:
        query_id = _query_id(record)
        prompt = record.get("query") or record.get("prompt")
        if not query_id or not isinstance(prompt, str) or not prompt.strip():
            continue
        queries[query_id] = {
            "prompt": prompt,
            "reference_answer": (
                str(record["answer"]) if record.get("answer") is not None else None
            ),
            "domain": str(record.get("domain") or "research"),
            "template_family": str(record.get("template_family") or ""),
            "metadata": {
                "query_record_hash": canonical_hash(record),
                "gold_doc_count": len(record.get("gold_docs") or [])
                if isinstance(record.get("gold_docs"), list)
                else None,
                "evidence_doc_count": len(record.get("evidence_docs") or [])
                if isinstance(record.get("evidence_docs"), list)
                else None,
                "negative_doc_count": len(record.get("negative_docs") or [])
                if isinstance(record.get("negative_docs"), list)
                else None,
            },
        }
    return queries


def _trajectory_fields(run: Mapping[str, Any]) -> tuple[dict[str, Any], int | None, dict[str, Any]]:
    result = run.get("result")
    trajectory_steps = len(result) if isinstance(result, list) else None
    tool_counts = run.get("tool_call_counts")
    if not isinstance(tool_counts, Mapping):
        tool_counts = run.get("search_counts")
    tool_counts = dict(tool_counts) if isinstance(tool_counts, Mapping) else {}
    metadata = {
        "status": run.get("status"),
        "search_counts": dict(run.get("search_counts"))
        if isinstance(run.get("search_counts"), Mapping)
        else None,
        "trajectory_sha256": canonical_hash(result) if result is not None else None,
        "raw_keys": sorted(str(key) for key in run),
    }
    return tool_counts, trajectory_steps, metadata


def _source_ref(path: Path, line_number: int, record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_path": str(path),
        "source_line": line_number,
        "source_record_sha256": canonical_hash(record),
        "raw_outputs_retained_at_source": True,
    }


def _evaluated_log_fields(
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], int | None, dict[str, Any]]:
    tool_counts = record.get("tool_call_counts")
    if not isinstance(tool_counts, Mapping):
        tool_counts = record.get("search_counts")
    tool_counts = dict(tool_counts) if isinstance(tool_counts, Mapping) else {}

    trajectory_steps = _nonnegative_int(record.get("trajectory_steps") or record.get("steps"))
    latency_ms = _nonnegative(record.get("latency_ms"))
    if latency_ms is None:
        latency_s = _nonnegative(record.get("latency_s"))
        latency_ms = latency_s * 1000.0 if latency_s is not None else None
    metadata = {
        "status": record.get("status"),
        "raw_keys": sorted(str(key) for key in record),
        "gen_output_sha256": (
            canonical_hash(record.get("gen_output")) if record.get("gen_output") is not None else None
        ),
        "evaluation_content_sha256": (
            canonical_hash(record.get("content")) if record.get("content") is not None else None
        ),
    }
    return tool_counts, trajectory_steps, {
        **metadata,
        "latency_ms": latency_ms,
        "prompt_tokens": _nonnegative_int(
            record.get("prompt_tokens") or record.get("total_input_tokens")
        ),
        "completion_tokens": _nonnegative_int(
            record.get("completion_tokens") or record.get("total_output_tokens")
        ),
    }


def import_evaluated_jsonl_logs(
    run_specs: Iterable[EvaluatedLogSpec],
    *,
    queries: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[tuple[PublicAnchor, ...], tuple[dict[str, Any], ...]]:
    """Import JSONL logs that publish explicit per-item correctness.

    This covers the ``evaluated.jsonl`` files in
    ``i-DeepSearch/observation-masking-eval-logs`` and can also consume local
    files from gated BCP-100 releases.  A record without ``correct`` (or an
    equivalent explicit field) is retained with a nullable outcome; status or
    the presence of a generated answer is never treated as success.
    """

    specs = tuple(run_specs)
    query_map = dict(queries or {})
    inferred_queries: dict[tuple[str, str], dict[str, Any]] = {}
    # Some published error rows omit the prompt, while another protocol for
    # the same benchmark/query retains it.  Recover only from that same
    # benchmark family; never borrow a prompt across datasets with colliding
    # numeric qids.
    for spec in specs:
        if not spec.path.is_file():
            continue
        with spec.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, Mapping):
                    continue
                query_id = _query_id(record)
                prompt = _prompt_from_record(record)
                if query_id and prompt:
                    inferred_queries.setdefault(
                        (spec.benchmark_family, query_id),
                        {
                            "prompt": prompt,
                            "reference_answer": _reference_answer_from_record(record),
                        },
                    )
    by_prompt: dict[str, list[AnchorObservation]] = {}
    prompt_rows: dict[str, dict[str, Any]] = {}
    prompt_specs: dict[str, list[EvaluatedLogSpec]] = {}
    rejected: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.path.is_file():
            rejected.append({"source": str(spec.path), "reason": "evaluation_file_missing"})
            continue
        with spec.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    rejected.append(
                        {"source": str(spec.path), "line": line_number, "reason": str(exc)}
                    )
                    continue
                if not isinstance(record, Mapping):
                    rejected.append(
                        {
                            "source": str(spec.path),
                            "line": line_number,
                            "reason": "evaluation_record_not_object",
                        }
                    )
                    continue
                query_id = _query_id(record)
                query = query_map.get(query_id) or inferred_queries.get(
                    (spec.benchmark_family, query_id), {}
                )
                prompt = _prompt_from_record(record) or str(query.get("prompt") or "")
                if not query_id or not prompt.strip():
                    rejected.append(
                        {
                            "source": str(spec.path),
                            "line": line_number,
                            "query_id": query_id,
                            "reason": "missing_query_id_or_prompt",
                        }
                    )
                    continue
                anchor_id = f"{spec.benchmark_family}:{query_id}"
                existing = prompt_rows.get(anchor_id)
                if existing is not None and existing["prompt"] != prompt:
                    rejected.append(
                        {
                            "source": str(spec.path),
                            "line": line_number,
                            "query_id": query_id,
                            "reason": "conflicting_prompt_for_benchmark_query",
                        }
                    )
                    continue
                outcome, outcome_method = _outcome_from_record(record)
                tool_counts, trajectory_steps, log_metadata = _evaluated_log_fields(record)
                observation = AnchorObservation(
                    observation_id=(
                        f"{anchor_id}:{spec.protocol_id}:{spec.model_id}:{line_number}"
                    ),
                    model_id=spec.model_id,
                    protocol_id=spec.protocol_id,
                    outcome=outcome,
                    outcome_status="graded" if outcome is not None else "missing",
                    outcome_method=outcome_method or spec.evaluation_method,
                    provider=spec.provider,
                    model_family=spec.model_family,
                    benchmark_family=spec.benchmark_family,
                    source_revision=spec.source_revision,
                    collection_vintage=spec.collection_vintage,
                    provenance_url=spec.provenance_url,
                    license_name=spec.license_name,
                    license_status=spec.license_status,
                    evaluation_date=spec.evaluation_date,
                    prompting_protocol=spec.prompting_protocol,
                    tool_availability=spec.tool_availability,
                    agent_scaffold=spec.agent_scaffold,
                    attempts=_nonnegative_int(record.get("attempts")),
                    grader=str(record.get("grader") or spec.evaluation_method),
                    retrieval_config=spec.retrieval_config,
                    context_masking=spec.context_masking,
                    tool_call_counts=tuple(sorted(tool_counts.items())),
                    latency_ms=log_metadata["latency_ms"],
                    prompt_tokens=log_metadata["prompt_tokens"],
                    completion_tokens=log_metadata["completion_tokens"],
                    reported_cost_usd=_nonnegative(record.get("reported_cost_usd")),
                    trajectory_steps=trajectory_steps,
                    trajectory_metadata=tuple(sorted(log_metadata.items())),
                    raw_ref=tuple(sorted(_source_ref(spec.path, line_number, record).items())),
                )
                by_prompt.setdefault(anchor_id, []).append(observation)
                prompt_rows.setdefault(
                    anchor_id,
                    {
                        "prompt": prompt,
                        "reference_answer": (
                            _reference_answer_from_record(record)
                            or query.get("reference_answer")
                        ),
                        "domain": str(query.get("domain") or spec.domain),
                        "template_family": str(query.get("template_family") or spec.template_family),
                        "metadata": {
                            "query_id": query_id,
                            "log_format": "evaluated_jsonl",
                        },
                    },
                )
                prompt_specs.setdefault(anchor_id, []).append(spec)

    anchors: list[PublicAnchor] = []
    for prompt_id in sorted(by_prompt):
        row = prompt_rows[prompt_id]
        observations = tuple(by_prompt[prompt_id])
        specs = prompt_specs[prompt_id]
        protocols = sorted({item.protocol_id for item in observations})
        by_protocol = {
            protocol: derive_anchor_metrics(observations, protocol_id=protocol)
            for protocol in protocols
        }
        revisions = sorted({spec.source_revision for spec in specs})
        provenance = sorted({spec.provenance_url for spec in specs})
        split_values = {spec.split for spec in specs}
        metrics: dict[str, Any] = {"metrics_by_protocol": by_protocol}
        if len(by_protocol) == 1:
            metrics.update(next(iter(by_protocol.values())))
        anchors.append(
            PublicAnchor(
                prompt_id=prompt_id,
                prompt=row["prompt"],
                benchmark_family=observations[0].benchmark_family,
                domain=row["domain"],
                template_family=row["template_family"],
                source_revision=revisions[0] if len(revisions) == 1 else "mixed",
                collection_vintage=(
                    observations[0].collection_vintage
                    if len({item.collection_vintage for item in observations}) == 1
                    else "mixed"
                ),
                provenance_url=provenance[0] if len(provenance) == 1 else "mixed",
                split=next(iter(split_values)) if len(split_values) == 1 else "",
                reference_answer=row["reference_answer"],
                metadata=tuple(sorted(row["metadata"].items())),
                observations=observations,
                metrics=tuple(sorted(metrics.items())),
            )
        )
    return tuple(anchors), tuple(rejected)


def import_browsecomp_plus(
    queries_path: str | Path,
    run_specs: Iterable[BrowseCompRunSpec],
    *,
    benchmark_family: str = "browsecomp_plus",
    domain_default: str = "research",
) -> tuple[tuple[PublicAnchor, ...], tuple[dict[str, Any], ...]]:
    """Normalize public BrowseComp-Plus queries and trajectories.

    Missing evaluation files are valid for an audit and produce nullable
    outcomes.  A run's ``status == completed`` is retained as trajectory
    metadata but is never treated as correctness.
    """

    queries = load_browsecomp_queries(queries_path)
    by_prompt: dict[str, list[AnchorObservation]] = {}
    prompt_rows: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for spec in run_specs:
        if not spec.path.is_file():
            rejected.append({"source": str(spec.path), "reason": "run_file_missing"})
            continue
        evaluations = _load_evaluations(spec.evaluation_path)
        with spec.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    run = json.loads(line)
                except json.JSONDecodeError as exc:
                    rejected.append(
                        {"source": str(spec.path), "line": line_number, "reason": str(exc)}
                    )
                    continue
                if not isinstance(run, Mapping):
                    rejected.append(
                        {"source": str(spec.path), "line": line_number, "reason": "run_not_object"}
                    )
                    continue
                query_id = _query_id(run)
                query = queries.get(query_id)
                if query is None:
                    rejected.append(
                        {
                            "source": str(spec.path),
                            "line": line_number,
                            "query_id": query_id,
                            "reason": "query_not_found",
                        }
                    )
                    continue
                evaluation = evaluations.get(query_id, {})
                outcome, outcome_method = _outcome_from_record(evaluation)
                tool_counts, trajectory_steps, trajectory_metadata = _trajectory_fields(run)
                observation_id = f"{query_id}:{spec.protocol_id}:{spec.model_id}:{line_number}"
                observation = AnchorObservation(
                    observation_id=observation_id,
                    model_id=spec.model_id,
                    protocol_id=spec.protocol_id,
                    outcome=outcome,
                    outcome_status="graded" if outcome is not None else "missing",
                    outcome_method=outcome_method or spec.evaluation_method,
                    provider=spec.provider,
                    model_family=spec.model_family,
                    benchmark_family=benchmark_family,
                    source_revision=spec.source_revision,
                    collection_vintage=spec.collection_vintage,
                    provenance_url=spec.provenance_url,
                    license_name=spec.license_name,
                    license_status=spec.license_status,
                    evaluation_date=spec.evaluation_date,
                    prompting_protocol=spec.prompting_protocol,
                    tool_availability=spec.tool_availability,
                    agent_scaffold=spec.agent_scaffold,
                    attempts=_nonnegative_int(evaluation.get("attempts")),
                    grader=str(evaluation.get("grader") or spec.evaluation_method),
                    retrieval_config=spec.retrieval_config,
                    context_masking=spec.context_masking,
                    tool_call_counts=tuple(sorted(tool_counts.items())),
                    latency_ms=_nonnegative(evaluation.get("latency_ms")),
                    prompt_tokens=_nonnegative_int(evaluation.get("prompt_tokens")),
                    completion_tokens=_nonnegative_int(evaluation.get("completion_tokens")),
                    reported_cost_usd=_nonnegative(evaluation.get("reported_cost_usd")),
                    trajectory_steps=trajectory_steps,
                    trajectory_metadata=tuple(sorted(trajectory_metadata.items())),
                    raw_ref=tuple(sorted(_source_ref(spec.path, line_number, run).items())),
                )
                by_prompt.setdefault(query_id, []).append(observation)
                prompt_rows.setdefault(
                    query_id,
                    {
                        "prompt": query["prompt"],
                        "reference_answer": query.get("reference_answer"),
                        "domain": query.get("domain") or domain_default,
                        "template_family": query.get("template_family") or "",
                        "metadata": query.get("metadata") or {},
                    },
                )

    anchors: list[PublicAnchor] = []
    for prompt_id in sorted(by_prompt):
        row = prompt_rows[prompt_id]
        observations = tuple(by_prompt[prompt_id])
        by_protocol = {
            protocol: derive_anchor_metrics(observations, protocol_id=protocol)
            for protocol in sorted({item.protocol_id for item in observations})
        }
        metrics: dict[str, Any] = {"metrics_by_protocol": by_protocol}
        if len(by_protocol) == 1:
            metrics.update(next(iter(by_protocol.values())))
        anchors.append(
            PublicAnchor(
                prompt_id=prompt_id,
                prompt=row["prompt"],
                benchmark_family=benchmark_family,
                domain=row["domain"],
                template_family=row["template_family"],
                source_revision=observations[0].source_revision,
                collection_vintage=observations[0].collection_vintage,
                provenance_url=observations[0].provenance_url,
                reference_answer=row["reference_answer"],
                metadata=tuple(sorted(row["metadata"].items())),
                observations=observations,
                metrics=tuple(sorted(metrics.items())),
            )
        )
    return tuple(anchors), tuple(rejected)


def load_evaluation_csv(path: str | Path) -> dict[str, dict[str, Any]]:
    """Adapt the official detailed CSV shape without claiming aggregate scores are item labels."""

    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            query_id = _query_id(record)
            if query_id:
                rows[query_id] = dict(record)
    return rows
