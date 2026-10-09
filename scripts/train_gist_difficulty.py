#!/usr/bin/env python3
"""Fine-tune GIST-small to rank prompts by outcome-derived difficulty.

The default mixture uses two pinned Hugging Face datasets:

* ``routellm/gpt4_dataset`` — broad weak-vs-strong routing labels.
* ``asdwb/route_balance_model_estimator`` — modern cross-model judge scores.

Data is streamed from pinned JSONL revisions, so the script does not require
``datasets`` or download multi-hundred-megabyte files to disk.  The encoder is
fine-tuned together with a one-dimensional difficulty head using Huber and
pairwise ranking losses.  The saved scalar is a relative difficulty rank, not a
success probability; calibrate it on Theriad's own held-out outcome grid before
using it for production routing.

Examples:

    # Validate adapters and write a tiny prepared-data fixture; no model load.
    python scripts/train_gist_difficulty.py --prepare-only \
        --max-routellm-train 20 --max-routebalance-train 20

    # Full run. About 436 MB is streamed from Hugging Face with default sources.
    python scripts/train_gist_difficulty.py --output-dir artifacts/gist-difficulty

    # Frozen-encoder baseline; compare its test metrics with the default run.
    python scripts/train_gist_difficulty.py --train-last-n-layers 0 \
        --output-dir artifacts/gist-difficulty-head-only
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.gist_training import (  # noqa: E402
    BASE_MODEL,
    BASE_MODEL_REVISION,
    ROUTEBALANCE_LICENSE,
    ROUTEBALANCE_REPO,
    ROUTEBALANCE_REVISION,
    ROUTELLM_LICENSE,
    ROUTELLM_REPO,
    ROUTELLM_REVISION,
    DifficultyExample,
    assign_routebalance_split,
    calibrate_target_quantiles,
    deduplicate_examples,
    iter_utf8_lf_lines,
    route_balance_example,
    route_llm_example,
    write_jsonl,
)

HF_BASE = "https://huggingface.co/datasets"
DEFAULT_OUTPUT = ROOT / "artifacts" / "gist-difficulty"


@dataclass(frozen=True)
class SourceFile:
    repo: str
    revision: str
    filename: str
    split: str
    config: str | None = None

    @property
    def url(self) -> str:
        return f"{HF_BASE}/{self.repo}/resolve/{self.revision}/{self.filename}"


ROUTELLM_FILES = (
    SourceFile(ROUTELLM_REPO, ROUTELLM_REVISION, "train.jsonl", "train"),
    SourceFile(ROUTELLM_REPO, ROUTELLM_REVISION, "valid.jsonl", "validation"),
)
ROUTEBALANCE_FILES = (
    SourceFile(ROUTEBALANCE_REPO, ROUTEBALANCE_REVISION, "train.jsonl", "train", "qwen"),
    SourceFile(ROUTEBALANCE_REPO, ROUTEBALANCE_REVISION, "test.jsonl", "test", "qwen"),
    SourceFile(
        ROUTEBALANCE_REPO,
        ROUTEBALANCE_REVISION,
        "gemma3_train.jsonl",
        "train",
        "gemma3",
    ),
    SourceFile(
        ROUTEBALANCE_REPO,
        ROUTEBALANCE_REVISION,
        "gemma3_test.jsonl",
        "test",
        "gemma3",
    ),
)


def selected_source_files(args: argparse.Namespace) -> tuple[SourceFile, ...]:
    files: list[SourceFile] = []
    if not args.no_routellm:
        files.extend(ROUTELLM_FILES)
    if not args.no_routebalance:
        files.extend(ROUTEBALANCE_FILES)
    return tuple(files)


def _require(module: str, extra: str):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise SystemExit(
            f"Missing {module!r}. Install the training dependencies with "
            f"`python -m pip install -e '.[{extra}]'`."
        ) from exc


def stream_jsonl(source: SourceFile, max_rows: int | None = None) -> Iterator[dict[str, Any]]:
    """Stream a pinned Hub JSONL file without caching it on disk."""

    httpx = _require("httpx", "training")
    timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        with client.stream("GET", source.url) as response:
            response.raise_for_status()
            seen = 0
            for line_number, line in enumerate(
                iter_utf8_lf_lines(response.iter_bytes()), start=1
            ):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source.filename}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{source.filename}:{line_number}: expected JSON object")
                yield row
                seen += 1
                if max_rows is not None and seen >= max_rows:
                    break


def _cap_for(source: SourceFile, args: argparse.Namespace) -> int | None:
    family = "train" if source.split == "train" else "eval"
    if source.repo == ROUTELLM_REPO:
        value = getattr(args, f"max_routellm_{family}")
    else:
        value = getattr(args, f"max_routebalance_{family}")
    return value or None


def load_examples(args: argparse.Namespace) -> tuple[list[DifficultyExample], dict[str, Any]]:
    """Stream, adapt, calibrate, and deduplicate all selected sources."""

    raw: list[DifficultyExample] = []
    rejected = Counter()
    streamed = Counter()

    if not args.no_routellm:
        for source in ROUTELLM_FILES:
            for row in stream_jsonl(source, _cap_for(source, args)):
                streamed[f"routellm:{source.split}"] += 1
                example = route_llm_example(row, source.split)
                if example is None:
                    rejected["routellm_invalid"] += 1
                else:
                    raw.append(example)

    if not args.no_routebalance:
        for source in ROUTEBALANCE_FILES:
            for row in stream_jsonl(source, _cap_for(source, args)):
                streamed[f"routebalance:{source.config}:{source.split}"] += 1
                example = route_balance_example(
                    row,
                    config=source.config or "default",
                    split=source.split,
                    include_harmful=args.include_harmful_routebalance,
                )
                if example is None:
                    rejected["routebalance_filtered_or_invalid"] += 1
                    continue
                example = assign_routebalance_split(
                    example,
                    original_split=source.split,
                    validation_fraction=args.routebalance_validation_fraction,
                    seed=args.seed,
                )
                raw.append(example)

    if not raw:
        raise SystemExit("No usable examples. Enable at least one source or raise the caps.")

    # Small bounded schema checks can legitimately filter every training row
    # from one source (for example, if the first RouteBalance rows are safety
    # prompts) while retaining that source's test rows.  A held-out row cannot
    # be calibrated without source-local training boundaries, so omit that
    # source explicitly instead of either leaking test labels or crashing.
    calibrated_sources = {row.label_source for row in raw if row.split == "train"}
    uncalibrated = Counter(
        row.label_source for row in raw if row.label_source not in calibrated_sources
    )
    if uncalibrated:
        rejected.update(
            {f"no_training_calibration:{source}": count for source, count in uncalibrated.items()}
        )
        raw = [row for row in raw if row.label_source in calibrated_sources]
    if not raw:
        raise SystemExit(
            "No source has usable training examples. Raise the training caps or "
            "review the filtering policy."
        )

    calibrated, calibration = calibrate_target_quantiles(raw)
    prepared = deduplicate_examples(calibrated)
    metadata = {
        "streamed": dict(streamed),
        "rejected": dict(rejected),
        "before_deduplication": len(raw),
        "after_deduplication": len(prepared),
        "split_counts": dict(Counter(row.split for row in prepared)),
        "label_source_counts": dict(Counter(row.label_source for row in prepared)),
        "training_target": "source-local empirical difficulty quantile",
        "calibration": {
            source: {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
            for source, values in calibration.items()
        },
    }
    return prepared, metadata


def split_examples(
    examples: Iterable[DifficultyExample],
) -> dict[str, list[DifficultyExample]]:
    out = {"train": [], "validation": [], "test": []}
    for row in examples:
        out[row.split].append(row)
    return out


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, implemented without SciPy."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if len(target) == 0:
        return {"count": 0, "rmse": float("nan"), "mae": float("nan"), "spearman": float("nan")}
    error = prediction - target
    rp, rt = rankdata(prediction), rankdata(target)
    rp -= rp.mean()
    rt -= rt.mean()
    denom = math.sqrt(float(np.sum(rp**2) * np.sum(rt**2)))
    spearman = float(np.sum(rp * rt) / denom) if denom else 0.0
    return {
        "count": int(len(target)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "spearman": spearman,
    }


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_kwargs_for_device(device: str) -> dict[str, Any]:
    """Use an attention backend that supports training dropout on the device."""

    # PyTorch's MPS scaled_dot_product_attention currently rejects non-zero
    # dropout during training. Transformers otherwise selects SDPA for BERT,
    # so force its eager implementation on Apple Silicon.
    return {"attn_implementation": "eager"} if device == "mps" else {}


def set_trainable_layers(model, last_n: int) -> dict[str, int]:
    """Freeze the encoder except its last N transformer blocks.

    ``0`` is a frozen-encoder baseline and ``-1`` trains the whole encoder.
    GIST-small is BERT-based, but the fallback keeps the script honest if a
    compatible Sentence Transformer exposes a different block layout.
    """

    for parameter in model.parameters():
        parameter.requires_grad = last_n == -1
    if last_n > 0:
        transformer = model[0].auto_model
        layers = getattr(getattr(transformer, "encoder", None), "layer", None)
        if layers is None:
            raise ValueError("model does not expose BERT-style encoder.layer blocks")
        for layer in list(layers)[-last_n:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    return {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total": sum(p.numel() for p in model.parameters()),
    }


def batches(
    rows: list[DifficultyExample],
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> Iterator[list[DifficultyExample]]:
    indices = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [rows[index] for index in indices[start : start + batch_size]]


def pairwise_ranking_loss(torch, raw_prediction, targets, margin: float):
    difference = targets[:, None] - targets[None, :]
    mask = difference.abs() >= margin
    mask = torch.triu(mask, diagonal=1)
    if not torch.any(mask):
        return raw_prediction.sum() * 0.0
    sign = torch.sign(difference[mask])
    predicted_difference = raw_prediction[:, None] - raw_prediction[None, :]
    # If target_i > target_j, raw_i should also be greater than raw_j.
    return torch.nn.functional.softplus(-sign * predicted_difference[mask]).mean()


def model_features(model, prompts: list[str], device: str) -> dict[str, Any]:
    """Preprocess prompts and move tensors while preserving metadata fields."""

    features = model.preprocess(prompts)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in features.items()
    }


def evaluate(model, head, rows, *, batch_size, device, torch) -> dict[str, float]:
    model.eval()
    head.eval()
    predictions: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for batch in batches(rows, batch_size=batch_size, seed=0, shuffle=False):
            features = model_features(model, [row.prompt for row in batch], device)
            embeddings = model(features)["sentence_embedding"]
            prediction = torch.sigmoid(head(embeddings).squeeze(-1))
            predictions.extend(prediction.detach().cpu().tolist())
            targets.extend(row.target for row in batch)
    return regression_metrics(np.asarray(predictions), np.asarray(targets))


def save_artifact(model, head, output: Path, manifest: dict[str, Any], torch) -> None:
    safetensors_torch = __import__("safetensors.torch", fromlist=["save_file"])
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output / "encoder"), safe_serialization=True)
    tensors = {name: value.detach().cpu().contiguous() for name, value in head.state_dict().items()}
    safetensors_torch.save_file(tensors, str(output / "difficulty_head.safetensors"))
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


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
    embedding_dimension = model.get_embedding_dimension()
    head = torch.nn.Linear(embedding_dimension, 1).to(device)

    encoder_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_groups = [{"params": head.parameters(), "lr": args.head_learning_rate}]
    if encoder_parameters:
        optimizer_groups.append({"params": encoder_parameters, "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)

    train_rows = splits["train"]
    validation_rows = splits["validation"]
    if not train_rows:
        raise SystemExit("Training split is empty; increase the training caps.")
    if not validation_rows:
        raise SystemExit("Validation split is empty; increase evaluation caps.")

    best_validation = float("inf")
    best_model_state = None
    best_head_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        head.train()
        running = 0.0
        examples_seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(
            batches(
                train_rows,
                batch_size=args.batch_size,
                seed=args.seed + epoch,
                shuffle=True,
            ),
            start=1,
        ):
            features = model_features(model, [row.prompt for row in batch], device)
            targets = torch.tensor([row.target for row in batch], dtype=torch.float32, device=device)
            embeddings = model(features)["sentence_embedding"]
            raw = head(embeddings).squeeze(-1)
            predictions = torch.sigmoid(raw)
            regression = torch.nn.functional.huber_loss(
                predictions,
                targets,
                delta=args.huber_delta,
            )
            ranking = pairwise_ranking_loss(torch, raw, targets, args.ranking_margin)
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
                    f"loss={running / max(examples_seen, 1):.5f}",
                    flush=True,
                )
        # Flush a final partial accumulation window.
        remainder = step % args.gradient_accumulation_steps
        if remainder:
            # Losses were divided by the nominal window size above. Restore
            # the mean gradient when the final window contains fewer batches.
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
            validation_rows,
            batch_size=args.eval_batch_size,
            device=device,
            torch=torch,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running / max(examples_seen, 1),
                "validation": validation,
            }
        )
        print(f"epoch={epoch + 1} validation={json.dumps(validation, sort_keys=True)}")
        if validation["rmse"] < best_validation:
            best_validation = validation["rmse"]
            best_model_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
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
            validation_rows,
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
    datasets = []
    if not args.no_routellm:
        datasets.append(
            {
                "repo": ROUTELLM_REPO,
                "revision": ROUTELLM_REVISION,
                "license": ROUTELLM_LICENSE,
                "label": "difficulty=(5-mixtral_score)/4, then training-CDF quantile",
            }
        )
    if not args.no_routebalance:
        datasets.append(
            {
                "repo": ROUTEBALANCE_REPO,
                "revision": ROUTEBALANCE_REVISION,
                "license": ROUTEBALANCE_LICENSE,
                "label": (
                    "difficulty=1-mean(cross-model DeepEval score), then "
                    "training-CDF quantile"
                ),
                "harmful_prompts_included": args.include_harmful_routebalance,
            }
        )
    manifest = {
        "schema_version": 1,
        "created_at_unix": int(time.time()),
        "base_model": args.model,
        "base_model_revision": args.model_revision,
        "resolved_device": device,
        "attention_implementation": model[0].auto_model.config._attn_implementation,
        "embedding_dimension": embedding_dimension,
        "max_sequence_length": model.max_seq_length,
        "target": "relative prompt difficulty quantile; higher is harder",
        "target_is_probability": False,
        "parameter_counts": parameter_counts,
        "training_arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output_dir"}
        },
        "data": data_metadata,
        "datasets": datasets,
        "history": history,
        "metrics": metrics,
        "deployment_gate": (
            "Not production-calibrated. Fit and evaluate a calibration layer on "
            "an untouched Theriad model/outcome grid before routing traffic."
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
    parser.add_argument("--no-routellm", action="store_true")
    parser.add_argument("--no-routebalance", action="store_true")
    parser.add_argument("--include-harmful-routebalance", action="store_true")
    parser.add_argument("--max-routellm-train", type=int, default=0, help="0 means all")
    parser.add_argument("--max-routellm-eval", type=int, default=0, help="0 means all")
    parser.add_argument("--max-routebalance-train", type=int, default=0, help="per config")
    parser.add_argument("--max-routebalance-eval", type=int, default=0, help="per config")
    parser.add_argument("--routebalance-validation-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--ranking-margin", type=float, default=0.15)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--train-last-n-layers",
        type=int,
        default=2,
        help="0=head only, -1=all encoder layers, default=last two",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.no_routellm and args.no_routebalance:
        raise SystemExit("At least one dataset source must be enabled.")
    if not 0.0 < args.routebalance_validation_fraction < 0.5:
        raise SystemExit("--routebalance-validation-fraction must be between 0 and 0.5")
    if args.epochs < 1 or args.batch_size < 2 or args.eval_batch_size < 1:
        raise SystemExit("epochs and batch sizes must be positive; training batch must be >=2")
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be positive")
    if args.train_last_n_layers < -1:
        raise SystemExit("--train-last-n-layers must be -1 or greater")
    for name in (
        "max_routellm_train",
        "max_routellm_eval",
        "max_routebalance_train",
        "max_routebalance_eval",
    ):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} cannot be negative")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    examples, metadata = load_examples(args)
    splits = split_examples(examples)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts = {
        split: write_jsonl(output / "prepared" / f"{split}.jsonl", rows)
        for split, rows in splits.items()
    }
    preparation = {
        "schema_version": 1,
        "counts": counts,
        "metadata": metadata,
        "sources": [source.url for source in selected_source_files(args)],
    }
    (output / "prepared" / "manifest.json").write_text(
        json.dumps(preparation, indent=2, sort_keys=True)
    )
    print(json.dumps(preparation, indent=2, sort_keys=True))
    if args.prepare_only:
        return 0
    manifest = train(args, splits, metadata)
    print(json.dumps(manifest["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
