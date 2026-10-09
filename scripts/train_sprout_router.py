#!/usr/bin/env python3
"""Train a GIST prompt encoder to predict SPROUT's per-model score matrix.

The script projects only prompt metadata and the 13 scalar score columns from
SPROUT's Parquet files; model responses and judge rationales are not copied.
MuSR and RAGBench are held out as complete validation families, while the
official SPROUT test split remains untouched.  Training samples source families
uniformly and optimizes both masked score regression and within-prompt ranking.

SPROUT currently declares no license in its dataset card.  Artifacts produced
by this script are therefore research-only and must not be published or
deployed until the dataset owner grants compatible rights.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.gist_training import (  # noqa: E402
    BASE_MODEL,
    BASE_MODEL_REVISION,
)
from theriad.routing.sprout_training import (  # noqa: E402
    DEFAULT_VALIDATION_FAMILIES,
    MODEL_COLUMNS,
    SPROUT_LICENSE,
    SPROUT_REPO,
    SPROUT_REVISION,
    SproutExample,
    adapt_projected_row,
    assert_disjoint_splits,
    balanced_sample_indices,
    deduplicate_examples,
    model_score_metrics,
)

DEFAULT_OUTPUT = ROOT / "artifacts" / "gist-sprout-router"
PARQUET_SHARDS = {
    "train": ("0000", "0001", "0002"),
    "validation": ("0000",),
    "test": ("0000",),
}


def _require(module: str, extra: str):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            f"Missing {module!r}. Install with python -m pip install -e '.[{extra}]'."
        ) from exc


def parquet_alias(split: str, shard: str) -> str:
    return (
        f"hf://datasets/{SPROUT_REPO}@~parquet/default/"
        f"{split}/{shard}.parquet"
    )


def projection_sql(output: Path) -> str:
    fields = ["key", "dataset", "dataset_level", "dataset_idx", "prompt"]
    fields.extend(f'"{model}".score AS "{model}"' for model in MODEL_COLUMNS)
    escaped = str(output).replace("'", "''")
    return f"COPY (SELECT {', '.join(fields)} FROM data) TO '{escaped}' (FORMAT JSON)"


def project_shard(split: str, shard: str, output: Path) -> None:
    """Use Parquet projection so ~860 MB of response text is never materialized."""

    if shutil.which("npx") is None:
        raise SystemExit("npx is required for the lightweight Parquet projection step")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "npx",
        "-y",
        "-p",
        "parquetlens",
        "-p",
        "@parquetlens/sql",
        "parquetlens",
        parquet_alias(split, shard),
        "--sql",
        projection_sql(output),
    ]
    subprocess.run(command, check=True)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield row


def write_jsonl(path: Path, rows: list[SproutExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
    return len(rows)


def prepare(args: argparse.Namespace) -> tuple[dict[str, list[SproutExample]], dict[str, Any]]:
    raw: list[SproutExample] = []
    rejected = Counter()
    projected = Counter()
    with tempfile.TemporaryDirectory(prefix="theriad-sprout-") as temporary:
        projection_dir = Path(temporary)
        for official_split, shards in PARQUET_SHARDS.items():
            for shard in shards:
                projected_path = projection_dir / f"{official_split}-{shard}.jsonl"
                project_shard(official_split, shard, projected_path)
                for row in iter_jsonl(projected_path):
                    projected[official_split] += 1
                    example = adapt_projected_row(
                        row,
                        official_split=official_split,
                        validation_families=args.validation_families,
                    )
                    if example is None:
                        rejected[f"{official_split}_invalid"] += 1
                    else:
                        raw.append(example)

    prepared, duplicate_drops = deduplicate_examples(raw)
    assert_disjoint_splits(prepared)
    splits = {"train": [], "validation": [], "test": []}
    for row in prepared:
        splits[row.split].append(row)
    if not all(splits.values()):
        raise SystemExit("train, grouped validation, and official test must all be non-empty")

    overlap = {
        split: len({row.record_id for row in rows})
        for split, rows in splits.items()
    }
    metadata = {
        "projected_counts": dict(projected),
        "rejected_counts": dict(rejected),
        "duplicate_drops": duplicate_drops,
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "unique_record_ids": overlap,
        "source_family_counts": {
            split: dict(Counter(row.source_family for row in rows))
            for split, rows in splits.items()
        },
        "validation_families": list(args.validation_families),
        "split_policy": (
            "official test untouched; train plus official validation pooled; "
            "complete validation families held out; exact prompts deduplicated "
            "with test then validation priority"
        ),
    }
    return splits, metadata


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_kwargs_for_device(device: str) -> dict[str, Any]:
    return {"attn_implementation": "eager"} if device == "mps" else {}


def set_trainable_layers(model, last_n: int) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = last_n == -1
    if last_n > 0:
        layers = getattr(getattr(model[0].auto_model, "encoder", None), "layer", None)
        if layers is None:
            raise ValueError("model does not expose BERT-style encoder.layer blocks")
        for layer in list(layers)[-last_n:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    return {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total": sum(p.numel() for p in model.parameters()),
    }


def batches_from_indices(
    rows: list[SproutExample], indices: list[int], batch_size: int
) -> Iterator[list[SproutExample]]:
    for start in range(0, len(indices), batch_size):
        yield [rows[index] for index in indices[start : start + batch_size]]


def sequential_batches(rows: list[SproutExample], batch_size: int) -> Iterator[list[SproutExample]]:
    return batches_from_indices(rows, list(range(len(rows))), batch_size)


def model_features(model, prompts: list[str], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model.preprocess(prompts).items()
    }


def target_tensors(torch, batch: list[SproutExample], device: str):
    target = torch.tensor(
        [[score if score is not None else 0.0 for score in row.scores] for row in batch],
        dtype=torch.float32,
        device=device,
    )
    mask = torch.tensor(
        [[score is not None for score in row.scores] for row in batch],
        dtype=torch.bool,
        device=device,
    )
    return target, mask


def within_prompt_ranking_loss(torch, raw, target, mask, margin: float):
    target_difference = target[:, :, None] - target[:, None, :]
    predicted_difference = raw[:, :, None] - raw[:, None, :]
    pair_mask = mask[:, :, None] & mask[:, None, :]
    pair_mask &= target_difference.abs() >= margin
    triangle = torch.triu(
        torch.ones(raw.shape[1], raw.shape[1], dtype=torch.bool, device=raw.device),
        diagonal=1,
    )
    pair_mask &= triangle[None, :, :]
    if not torch.any(pair_mask):
        return raw.sum() * 0.0
    signs = torch.sign(target_difference[pair_mask])
    return torch.nn.functional.softplus(
        -signs * predicted_difference[pair_mask]
    ).mean()


def evaluate(model, head, rows, *, batch_size, device, torch) -> dict[str, Any]:
    model.eval()
    head.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in sequential_batches(rows, batch_size):
            features = model_features(model, [row.prompt for row in batch], device)
            embedding = model(features)["sentence_embedding"]
            predictions.append(torch.sigmoid(head(embedding)).cpu().numpy())
            target, mask = target_tensors(torch, batch, device)
            targets.append(target.cpu().numpy())
            masks.append(mask.cpu().numpy())
    prediction_array = np.concatenate(predictions)
    target_array = np.concatenate(targets)
    mask_array = np.concatenate(masks)
    metrics = model_score_metrics(prediction_array, target_array, mask_array)
    per_family = {}
    for family in sorted({row.source_family for row in rows}):
        indices = np.asarray([row.source_family == family for row in rows])
        per_family[family] = model_score_metrics(
            prediction_array[indices], target_array[indices], mask_array[indices]
        )
    metrics["per_source_family"] = per_family
    return metrics


def save_artifact(model, head, output: Path, manifest: dict[str, Any], torch) -> None:
    safetensors_torch = __import__("safetensors.torch", fromlist=["save_file"])
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output / "encoder"), safe_serialization=True)
    tensors = {
        name: value.detach().cpu().contiguous() for name, value in head.state_dict().items()
    }
    safetensors_torch.save_file(tensors, str(output / "performance_head.safetensors"))
    (output / "model_index.json").write_text(
        json.dumps({"models": list(MODEL_COLUMNS)}, indent=2) + "\n"
    )
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def train(args: argparse.Namespace, splits, data_metadata) -> dict[str, Any]:
    torch = _require("torch", "training")
    sentence_transformers = _require("sentence_transformers", "training")
    SentenceTransformer = sentence_transformers.SentenceTransformer

    device = choose_device(torch, args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = SentenceTransformer(
        args.model,
        revision=args.model_revision,
        device=device,
        trust_remote_code=False,
        model_kwargs=model_kwargs_for_device(device),
    )
    model.max_seq_length = min(args.max_length, model.max_seq_length)
    parameter_counts = set_trainable_layers(model, args.train_last_n_layers)
    head = torch.nn.Linear(model.get_embedding_dimension(), len(MODEL_COLUMNS)).to(device)
    encoder_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer_groups = [{"params": head.parameters(), "lr": args.head_learning_rate}]
    if encoder_parameters:
        optimizer_groups.append({"params": encoder_parameters, "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)

    train_rows = splits["train"]
    examples_per_epoch = args.examples_per_epoch or len(train_rows)
    best_selection_loss = float("inf")
    best_model_state = None
    best_head_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        head.train()
        running = 0.0
        examples_seen = 0
        indices = balanced_sample_indices(
            train_rows, count=examples_per_epoch, seed=args.seed + epoch
        )
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(
            batches_from_indices(train_rows, indices, args.batch_size), start=1
        ):
            features = model_features(model, [row.prompt for row in batch], device)
            embedding = model(features)["sentence_embedding"]
            raw = head(embedding)
            prediction = torch.sigmoid(raw)
            target, mask = target_tensors(torch, batch, device)
            regression = torch.nn.functional.huber_loss(
                prediction[mask], target[mask], delta=args.huber_delta
            )
            ranking = within_prompt_ranking_loss(
                torch, raw, target, mask, args.ranking_margin
            )
            loss = regression + args.ranking_weight * ranking
            (loss / args.gradient_accumulation_steps).backward()
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [*encoder_parameters, *head.parameters()], args.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().cpu()) * len(batch)
            examples_seen += len(batch)
            if args.log_every and step % args.log_every == 0:
                print(
                    f"epoch={epoch + 1} step={step} "
                    f"loss={running / examples_seen:.5f}",
                    flush=True,
                )
        remainder = step % args.gradient_accumulation_steps
        if remainder:
            scale = args.gradient_accumulation_steps / remainder
            for parameter in [*encoder_parameters, *head.parameters()]:
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(
                [*encoder_parameters, *head.parameters()], args.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        validation = evaluate(
            model,
            head,
            splits["validation"],
            batch_size=args.eval_batch_size,
            device=device,
            torch=torch,
        )
        selection_loss = float(validation["top1_regret"]) + 0.25 * float(
            validation["rmse"]
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running / examples_seen,
                "validation": validation,
                "checkpoint_selection_loss": selection_loss,
            }
        )
        print(f"epoch={epoch + 1} validation={json.dumps(validation, sort_keys=True)}")
        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_head_state = copy.deepcopy(head.state_dict())

    if best_model_state is None or best_head_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_model_state)
    head.load_state_dict(best_head_state)
    metrics = {
        "validation": evaluate(
            model,
            head,
            splits["validation"],
            batch_size=args.eval_batch_size,
            device=device,
            torch=torch,
        ),
        "test": evaluate(
            model,
            head,
            splits["test"],
            batch_size=args.eval_batch_size,
            device=device,
            torch=torch,
        ),
    }
    manifest = {
        "schema_version": 1,
        "created_at_unix": int(time.time()),
        "base_model": args.model,
        "base_model_revision": args.model_revision,
        "resolved_device": device,
        "embedding_dimension": model.get_embedding_dimension(),
        "max_sequence_length": model.max_seq_length,
        "target": "13 SPROUT per-model judge scores plus within-prompt model ranking",
        "model_columns": list(MODEL_COLUMNS),
        "parameter_counts": parameter_counts,
        "training_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output_dir"
        },
        "data": data_metadata,
        "dataset": {
            "repo": SPROUT_REPO,
            "revision": SPROUT_REVISION,
            "license": SPROUT_LICENSE,
        },
        "history": history,
        "metrics": metrics,
        "deployment_gate": (
            "RESEARCH ONLY: SPROUT declares no license. Do not publish or deploy. "
            "The 13 output IDs are also not Theriad's current provider catalog; "
            "evaluate transfer on an untouched internal grid before adaptation."
        ),
    }
    save_artifact(model, head, Path(args.output_dir), manifest, torch)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--model-revision", default=BASE_MODEL_REVISION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--validation-families",
        nargs="+",
        default=list(DEFAULT_VALIDATION_FAMILIES),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--examples-per-epoch", type=int, default=0, help="0 means train size")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--ranking-margin", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-last-n-layers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise SystemExit("epochs and batch sizes must be positive")
    if args.examples_per_epoch < 0:
        raise SystemExit("--examples-per-epoch cannot be negative")
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be positive")
    if args.train_last_n_layers < -1:
        raise SystemExit("--train-last-n-layers must be -1 or greater")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    splits, metadata = prepare(args)
    prepared_dir = Path(args.output_dir) / "prepared"
    counts = {
        split: write_jsonl(prepared_dir / f"{split}.jsonl", rows)
        for split, rows in splits.items()
    }
    preparation = {
        "schema_version": 1,
        "counts": counts,
        "metadata": metadata,
        "dataset": {
            "repo": SPROUT_REPO,
            "revision": SPROUT_REVISION,
            "license": SPROUT_LICENSE,
        },
        "model_columns": list(MODEL_COLUMNS),
        "deployment_gate": "research-only: dataset license is undeclared",
    }
    (prepared_dir / "manifest.json").write_text(
        json.dumps(preparation, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(preparation, indent=2, sort_keys=True))
    if args.prepare_only:
        return 0
    manifest = train(args, splits, metadata)
    print(json.dumps(manifest["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
