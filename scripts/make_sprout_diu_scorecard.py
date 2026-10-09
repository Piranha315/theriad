#!/usr/bin/env python3
"""Build a presentation scorecard for the SPROUT embedding-router pilot.

Chart contract
--------------
Question: Does the learned prompt embedding show enough routing signal to
justify a defense-relevant follow-on evaluation?
Takeaway: Aggregate model selection improves materially over uniform model
selection, but gains vary sharply by task family.
Form: 16:9 static scorecard with KPI, absolute-score bars, and ranked family
regret-reduction bars. Bars use zero baselines and direct labels.
Palette: one blue root for the router, orange for weak-transfer exceptions,
and neutral baseline/oracle references. Color is never the only distinction.
Outputs: PNG and SVG in ``charts/`` plus reproducible presentation metrics in
the trained artifact directory.

This is deliberately marked preliminary. SPROUT declares no license and its
tasks are not a defense-domain operational evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.sprout_training import model_score_metrics  # noqa: E402

DEFAULT_ARTIFACT = ROOT / "artifacts" / "gist-sprout-router"
DEFAULT_OUTPUT = ROOT / "charts"

INK = "#152238"
MUTED = "#627083"
GRID = "#D9E0E8"
PAPER = "#FFFFFF"
SOFT = "#F4F7FA"
BLUE = "#1E63D6"
BLUE_LIGHT = "#BCD2F5"
ORANGE = "#D97706"
ORANGE_LIGHT = "#F5D4A8"
GREY = "#8D99A8"
LIGHT_GREY = "#D4DAE2"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def load_prepared_test(artifact: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    targets: list[list[float]] = []
    masks: list[list[bool]] = []
    path = artifact / "prepared" / "test.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            scores = row["scores"]
            rows.append(row)
            targets.append([score if score is not None else 0.0 for score in scores])
            masks.append([score is not None for score in scores])
    return rows, np.asarray(targets), np.asarray(masks, dtype=bool)


def infer(artifact: Path, prompts: list[str], *, device: str, batch_size: int) -> np.ndarray:
    from safetensors.torch import load_file
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(str(artifact / "encoder"), device=device)
    embedding = encoder.encode(
        prompts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    state = load_file(artifact / "performance_head.safetensors")
    weight = state["weight"].numpy()
    bias = state["bias"].numpy()
    return _sigmoid(embedding @ weight.T + bias)


def prompt_outcomes(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, np.ndarray]:
    selected_scores = []
    oracle_scores = []
    uniform_scores = []
    for predicted_row, target_row, mask_row in zip(prediction, target, mask, strict=True):
        valid = np.flatnonzero(mask_row)
        local_prediction = predicted_row[valid]
        local_target = target_row[valid]
        selected_scores.append(float(local_target[int(np.argmax(local_prediction))]))
        oracle_scores.append(float(np.max(local_target)))
        uniform_scores.append(float(np.mean(local_target)))
    return {
        "router": np.asarray(selected_scores),
        "oracle": np.asarray(oracle_scores),
        "uniform": np.asarray(uniform_scores),
    }


def regret_reduction(outcomes: dict[str, np.ndarray], indices: np.ndarray) -> float:
    baseline_regret = np.mean(outcomes["oracle"][indices] - outcomes["uniform"][indices])
    router_regret = np.mean(outcomes["oracle"][indices] - outcomes["router"][indices])
    return 1.0 - router_regret / baseline_regret


def stratified_bootstrap_ci(
    outcomes: dict[str, np.ndarray], families: list[str], *, seed: int, replicates: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    family_indices: dict[str, np.ndarray] = {}
    for family in sorted(set(families)):
        family_indices[family] = np.flatnonzero(np.asarray(families) == family)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = np.concatenate(
            [rng.choice(indices, len(indices), replace=True) for indices in family_indices.values()]
        )
        estimates[replicate] = regret_reduction(outcomes, sampled)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def presentation_metrics(
    rows: list[dict[str, Any]], prediction: np.ndarray, target: np.ndarray, mask: np.ndarray,
    *, seed: int, bootstrap_replicates: int,
) -> dict[str, Any]:
    metrics = model_score_metrics(prediction, target, mask)
    outcomes = prompt_outcomes(prediction, target, mask)
    families = [str(row["source_family"]) for row in rows]
    all_indices = np.arange(len(rows))
    reduction = regret_reduction(outcomes, all_indices)
    ci = stratified_bootstrap_ci(
        outcomes, families, seed=seed, replicates=bootstrap_replicates
    )
    per_family = {}
    for family in sorted(set(families)):
        indices = np.flatnonzero(np.asarray(families) == family)
        family_outcomes = {key: values[indices] for key, values in outcomes.items()}
        per_family[family] = {
            "prompt_count": int(len(indices)),
            "uniform_mean_score": float(np.mean(family_outcomes["uniform"])),
            "router_mean_score": float(np.mean(family_outcomes["router"])),
            "oracle_mean_score": float(np.mean(family_outcomes["oracle"])),
            "regret_reduction": regret_reduction(
                family_outcomes, np.arange(len(indices))
            ),
        }
    return {
        "scope": "SPROUT official test after exact-prompt deduplication",
        "prompt_count": len(rows),
        "candidate_model_count": prediction.shape[1],
        "observed_score_count": int(mask.sum()),
        "uniform_mean_score": float(np.mean(outcomes["uniform"])),
        "router_mean_score": float(np.mean(outcomes["router"])),
        "oracle_mean_score": float(np.mean(outcomes["oracle"])),
        "uniform_regret": float(np.mean(outcomes["oracle"] - outcomes["uniform"])),
        "router_regret": float(np.mean(outcomes["oracle"] - outcomes["router"])),
        "regret_reduction": reduction,
        "regret_reduction_prompt_bootstrap_95_ci": list(ci),
        "mean_prompt_spearman": metrics["mean_prompt_spearman"],
        "score_rmse": metrics["rmse"],
        "score_mae": metrics["mae"],
        "per_source_family": per_family,
        "definitions": {
            "uniform_mean_score": (
                "Expected judge score when selecting uniformly among the 13 candidates "
                "available for each prompt."
            ),
            "oracle_mean_score": "Best observed candidate-model score for each prompt.",
            "regret": "Oracle score minus selected-model score; lower is better.",
            "regret_reduction": "1 - router regret / uniform-selection regret.",
            "confidence_interval": (
                "Stratified prompt bootstrap; preserves test-set source-family counts. "
                "It does not include training-seed, model-catalog, or dataset-shift uncertainty."
            ),
        },
        "external_sharing_gate": (
            "PRELIMINARY RESEARCH. SPROUT has no declared license and is not a "
            "defense-domain operational benchmark. Obtain licensing review and run "
            "a frozen defense-relevant evaluation before external performance claims."
        ),
    }


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def draw_scorecard(metrics: dict[str, Any], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
        }
    )
    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    grid = fig.add_gridspec(
        12, 24, left=0.08, right=0.965, top=0.80, bottom=0.14, wspace=2.8, hspace=1.0
    )

    fig.text(
        0.055,
        0.955,
        "Theriad embedding-router feasibility results",
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.055,
        0.912,
        (
            f"Preliminary research · {metrics['prompt_count']:,} held-out prompts · "
            f"{metrics['candidate_model_count']} candidate models · SPROUT official test split"
        ),
        fontsize=11.5,
        color=MUTED,
        ha="left",
        va="top",
    )
    fig.text(
        0.945,
        0.95,
        "INTERNAL DRAFT · DO NOT DISTRIBUTE",
        fontsize=10,
        fontweight="bold",
        color=ORANGE,
        ha="right",
        va="top",
    )

    kpi = fig.add_subplot(grid[0:5, 0:7])
    kpi.set_facecolor(SOFT)
    for spine in kpi.spines.values():
        spine.set_visible(False)
    kpi.set_xticks([])
    kpi.set_yticks([])
    reduction = metrics["regret_reduction"]
    ci_low, ci_high = metrics["regret_reduction_prompt_bootstrap_95_ci"]
    kpi.text(0.08, 0.78, f"{reduction:.0%}", fontsize=55, fontweight="bold", color=BLUE, va="top")
    kpi.text(0.08, 0.50, "lower expected regret", fontsize=17, fontweight="bold", va="top")
    kpi.text(0.08, 0.39, "vs uniform model selection", fontsize=13, color=MUTED, va="top")
    kpi.text(
        0.08,
        0.19,
        f"95% prompt-bootstrap CI: {ci_low:.0%}–{ci_high:.0%}",
        fontsize=10.5,
        color=MUTED,
        va="top",
    )
    kpi.text(0.08, 0.09, "Sampling uncertainty only; one training seed", fontsize=9.5, color=MUTED, va="top")

    score_ax = fig.add_subplot(grid[0:5, 8:15])
    labels = ["Uniform selection", "Learned router", "Oracle (best observed)"]
    values = [
        metrics["uniform_mean_score"],
        metrics["router_mean_score"],
        metrics["oracle_mean_score"],
    ]
    colors = [LIGHT_GREY, BLUE, GREY]
    bars = score_ax.barh(labels, values, color=colors, height=0.52, edgecolor="none")
    score_ax.invert_yaxis()
    score_ax.set_xlim(0, 1.05)
    score_position = score_ax.get_position()
    fig.text(
        score_position.x0,
        score_position.y1 + 0.038,
        "Mean judge score",
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        score_position.x0,
        score_position.y1 + 0.012,
        "Average observed task score across the held-out prompt set",
        fontsize=9.5,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    score_ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    score_ax.grid(axis="x", color=GRID, linewidth=0.8)
    score_ax.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        score_ax.text(value + 0.018, bar.get_y() + bar.get_height() / 2, percent(value), va="center", fontsize=11, fontweight="bold")
    for side in ("top", "right", "left"):
        score_ax.spines[side].set_visible(False)
    score_ax.spines["bottom"].set_color(GRID)
    score_ax.tick_params(axis="y", length=0, labelsize=10.5)
    score_ax.tick_params(axis="x", labelsize=9)

    definition = fig.add_subplot(grid[0:5, 16:24])
    definition.axis("off")
    definition.text(0, 0.95, "What this establishes", fontsize=14, fontweight="bold", va="top")
    bullets = [
        "Prompt embeddings contain useful signal for model selection.",
        "The signal transfers unevenly across task families.",
        "This is a feasibility result—not operational validation.",
    ]
    for index, bullet in enumerate(bullets):
        y = 0.75 - index * 0.22
        definition.text(0.00, y, "•", fontsize=17, color=BLUE if index < 2 else ORANGE, va="center")
        definition.text(0.06, y, bullet, fontsize=11.5, color=INK, va="center", wrap=True)

    fig.text(
        0.055,
        0.49,
        "Regret reduction varies materially by task family",
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.055,
        0.463,
        "Relative improvement over uniform model selection; higher is better",
        fontsize=9.5,
        color=MUTED,
        ha="left",
        va="top",
    )
    family_ax = fig.add_subplot(grid[7:12, 0:24])
    family_rows = [
        (family, values["regret_reduction"], values["prompt_count"])
        for family, values in metrics["per_source_family"].items()
    ]
    family_rows.sort(key=lambda row: row[1], reverse=True)
    family_labels = [
        {
            "math": "MATH",
            "openhermes": "OpenHermes",
            "mmlu_pro": "MMLU-Pro",
            "ragbench": "RAGBench",
            "gpqa": "GPQA",
            "musr": "MuSR",
        }.get(family, family)
        for family, _, _ in family_rows
    ]
    family_values = [value for _, value, _ in family_rows]
    family_counts = [count for _, _, count in family_rows]
    y = np.arange(len(family_rows))
    family_colors = [BLUE if value >= 0.25 else ORANGE for value in family_values]
    family_bars = family_ax.barh(y, family_values, color=family_colors, height=0.54)
    family_ax.invert_yaxis()
    family_ax.set_yticks(y, family_labels)
    family_ax.set_xlim(0, 0.80)
    family_ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    family_ax.grid(axis="x", color=GRID, linewidth=0.8)
    family_ax.set_axisbelow(True)
    for bar, value, count in zip(family_bars, family_values, family_counts, strict=True):
        family_ax.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0%}   n={count:,}",
            va="center",
            fontsize=10,
            fontweight="bold" if value < 0.25 else "normal",
            color=INK,
        )
    for side in ("top", "right", "left"):
        family_ax.spines[side].set_visible(False)
    family_ax.spines["bottom"].set_color(GRID)
    family_ax.tick_params(axis="y", length=0, labelsize=10.5)
    family_ax.tick_params(axis="x", labelsize=9)

    footer = (
        "Regret = perfect-hindsight oracle score − selected-model score. Uniform baseline = expected score when choosing among 13 candidates at random. "
        "Exact prompt overlap removed.\n"
        "LIMITS: non-defense academic/general benchmark; one training seed; no cost/latency comparison; SPROUT has no declared license. "
        "Do not distribute externally pending licensing review and defense-relevant validation. Source revision: d54e3cda."
    )
    fig.text(0.055, 0.055, footer, fontsize=8.3, color=MUTED, ha="left", va="bottom", linespacing=1.45)

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "sprout_diu_scorecard.png", dpi=180, facecolor=PAPER)
    fig.savefig(output / "sprout_diu_scorecard.svg", facecolor=PAPER)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows, target, mask = load_prepared_test(args.artifact_dir)
    prediction = infer(
        args.artifact_dir,
        [str(row["prompt"]) for row in rows],
        device=args.device,
        batch_size=args.batch_size,
    )
    metrics = presentation_metrics(
        rows,
        prediction,
        target,
        mask,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    manifest = json.loads((args.artifact_dir / "training_manifest.json").read_text())
    expected = manifest["metrics"]["test"]
    for key, presentation_key in (
        ("random_mean_score", "uniform_mean_score"),
        ("selected_mean_score", "router_mean_score"),
        ("oracle_mean_score", "oracle_mean_score"),
        ("top1_regret", "router_regret"),
    ):
        if not math.isclose(expected[key], metrics[presentation_key], abs_tol=1e-6):
            raise RuntimeError(f"recomputed {presentation_key} does not match manifest")
    metrics["dataset"] = manifest["dataset"]
    metrics["artifact_created_at_unix"] = manifest["created_at_unix"]
    (args.artifact_dir / "presentation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    draw_scorecard(metrics, args.output_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"wrote {args.output_dir / 'sprout_diu_scorecard.png'}")
    print(f"wrote {args.output_dir / 'sprout_diu_scorecard.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
