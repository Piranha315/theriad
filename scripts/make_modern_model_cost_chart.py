"""Build a presentation-ready cost/quality Pareto plot for modern models.

The chart normalizes every option to the same hypothetical workload (600 input
and 1,200 output tokens per request) and plots projected spend against measured
benchmark accuracy. It is explicitly a modeled comparison, not realized spend.

Run:
    python3 scripts/make_modern_model_cost_chart.py

Outputs:
    charts/modern_model_cost_comparison.png
    charts/modern_model_cost_comparison.svg
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

from theriad.eval.corpus import load_corpus
from theriad.eval.grid import OutcomeGrid
from theriad.eval.routers import theriad_router


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "charts"
OUT.mkdir(exist_ok=True)

CORPUS = ROOT / "eval_corpus.jsonl"
GRID = ROOT / "eval_grid.json"

# Current OpenRouter list-price workload estimates, checked 2026-08-24.
# Unit: dollars per 1,000 requests at 600 input + 1,200 output tokens each.
STANDARD_COST = {
    "Nemotron Super (free)": 0.0,
    "GPT-OSS 120B": 0.2262,
    "DeepSeek V4 Flash": 0.4200,
    "MiMo V2.5": 0.4200,
    "Luna": 1.5600,
    "Qwen 3.6 Plus": 2.5350,
    "GLM 5.2": 4.2228,
    "Muse Spark 1.2": 5.8500,
    "Grok 4.6": 8.4000,
}

MODEL_SLUG = {
    "Nemotron Super (free)": "nvidia/nemotron-3-super-120b-a12b:free",
    "GPT-OSS 120B": "openai/gpt-oss-120b",
    "DeepSeek V4 Flash": "deepseek/deepseek-v4-flash-0731",
    "MiMo V2.5": "xiaomi/mimo-v2.5",
    "Luna": "openai/gpt-5.6-luna",
    "Qwen 3.6 Plus": "qwen/qwen3.6-plus",
    "GLM 5.2": "z-ai/glm-5.2",
    "Muse Spark 1.2": "meta/muse-spark-1.2",
    "Grok 4.6": "x-ai/grok-4.6",
}

# Per-request normalized costs for the models selected by the current router.
# The free endpoints intentionally remain zero.
ROUTED_STANDARD_COST = {
    "nvidia/nemotron-3-super-120b-a12b:free": 0.0,
    "xiaomi/mimo-v2.5": 0.4200 / 1_000,
    "openai/gpt-5.6-luna": 1.5600 / 1_000,
    "x-ai/grok-4.6": 8.4000 / 1_000,
    "poolside/laguna-xs-2.1:free": 0.0,
}

INK = "#18212F"
MUTED = "#687386"
GRIDLINE = "#D8DEE8"
THERIAD = "#2767E8"
LUNA = "#D69E2E"
CONTEXT = "#B7C0CF"
BG = "#FBFCFE"


def measured_accuracy(grid: OutcomeGrid, item_ids: list[str], model: str) -> tuple[float, int, int]:
    cells = [grid.cell(model, item_id) for item_id in item_ids]
    valid = [cell for cell in cells if cell is not None and cell.trials > 0]
    successes = sum(cell.successes for cell in valid)
    trials = sum(cell.trials for cell in valid)
    return successes / trials, len(valid), sum(cell.truncated for cell in cells if cell)


def build_data() -> tuple[list[dict], dict]:
    items = load_corpus(CORPUS)
    item_ids = [item.id for item in items]
    grid = OutcomeGrid.load(GRID)
    router = theriad_router()

    selections = []
    router_successes = 0
    router_trials = 0
    for item in items:
        model = router(item).model
        selections.append(model)
        cell = grid.cell(model, item.id)
        if cell is None or cell.trials == 0:
            raise RuntimeError(f"Missing measured router cell for {model} / {item.id}")
        router_successes += cell.successes
        router_trials += cell.trials
    router_standard_cost_1k = (
        sum(ROUTED_STANDARD_COST[model] for model in selections) / len(items) * 1_000
    )

    rows = []
    for label, cost in STANDARD_COST.items():
        accuracy, coverage, truncations = measured_accuracy(
            grid, item_ids, MODEL_SLUG[label]
        )
        rows.append(
            {
                "label": label,
                "cost": cost,
                "accuracy": accuracy,
                "coverage": coverage,
                "truncations": truncations,
            }
        )
    rows.append(
        {
            "label": "Theriad (routed)",
            "cost": router_standard_cost_1k,
            "accuracy": router_successes / router_trials,
            "coverage": len(items),
            "truncations": 0,
        }
    )
    rows.sort(key=lambda row: row["cost"])

    metadata = {
        "standard_ratio": STANDARD_COST["Luna"] / router_standard_cost_1k,
        "n_items": len(items),
    }
    return rows, metadata


def pareto_frontier(rows: list[dict], n_items: int) -> list[dict]:
    """Return full-coverage points not dominated on cost and accuracy."""
    eligible = sorted(
        (row for row in rows if row["coverage"] == n_items),
        key=lambda row: (row["cost"], -row["accuracy"]),
    )
    frontier = []
    best_accuracy = -1.0
    for row in eligible:
        if row["accuracy"] > best_accuracy:
            frontier.append(row)
            best_accuracy = row["accuracy"]
    return frontier


def render(rows: list[dict], metadata: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.edgecolor": GRIDLINE,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
        }
    )

    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)

    frontier = pareto_frontier(rows, metadata["n_items"])
    ax.plot(
        [row["cost"] for row in frontier],
        [row["accuracy"] for row in frontier],
        color=INK,
        linewidth=1.7,
        linestyle=(0, (4, 3)),
        zorder=2,
    )

    for row in rows:
        label = row["label"]
        complete = row["coverage"] == metadata["n_items"]
        if label == "Theriad (routed)":
            face, edge, size, linewidth = THERIAD, INK, 180, 1.2
        elif label == "Luna":
            face, edge, size, linewidth = LUNA, INK, 160, 1.2
        elif complete:
            face, edge, size, linewidth = CONTEXT, "#738095", 105, 1.0
        else:
            face, edge, size, linewidth = BG, "#738095", 105, 1.4
        ax.scatter(
            row["cost"],
            row["accuracy"],
            s=size,
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            zorder=4,
        )

    label_offsets = {
        "Nemotron Super (free)": (10, 10),
        "GPT-OSS 120B": (10, -18),
        "Theriad (routed)": (12, 10),
        "DeepSeek V4 Flash": (10, -18),
        "MiMo V2.5": (10, 10),
        "Luna": (12, 10),
        "Qwen 3.6 Plus": (10, -19),
        "GLM 5.2": (10, 10),
        "Muse Spark 1.2": (10, -18),
        "Grok 4.6": (-10, 11),
    }
    for row in rows:
        dx, dy = label_offsets[row["label"]]
        if row["coverage"] < metadata["n_items"]:
            mark = "†"
        else:
            mark = ""
        ha = "right" if dx < 0 else "left"
        weight = "bold" if row["label"] in {"Theriad (routed)", "Luna"} else "normal"
        ax.annotate(
            f"{row['label']}{mark}\n${row['cost']:.2f} / 1k",
            (row["cost"], row["accuracy"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=9.3,
            color=INK,
            fontweight=weight,
            linespacing=1.15,
            zorder=5,
        )

    ax.set_xlim(-0.15, 9.15)
    ax.set_ylim(0.80, 0.975)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Projected provider spend per 1,000 requests", fontsize=11)
    ax.set_ylabel("Observed benchmark accuracy", fontsize=11)
    ax.tick_params(axis="both", labelsize=9.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    legend = [
        Line2D([0], [0], color=INK, lw=1.7, linestyle=(0, (4, 3)), label="Pareto-efficient frontier"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CONTEXT,
               markeredgecolor="#738095", markersize=8, label="Full 200-prompt coverage"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BG,
               markeredgecolor="#738095", markersize=8, label="Incomplete coverage; excluded from frontier"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=9)

    ax.annotate(
        f"{metadata['standard_ratio']:.1f}× cheaper than Luna\nwith 5.5 pp lower accuracy",
        xy=(next(row["cost"] for row in rows if row["label"] == "Theriad (routed)"),
            next(row["accuracy"] for row in rows if row["label"] == "Theriad (routed)")),
        xytext=(2.0, 0.855),
        arrowprops={"arrowstyle": "->", "color": THERIAD, "lw": 1.4},
        color=THERIAD,
        fontsize=10.5,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": BG, "edgecolor": THERIAD, "linewidth": 0.8},
        zorder=6,
    )

    fig.suptitle(
        "Projected cost and benchmark accuracy by model",
        x=0.08,
        y=0.96,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.08,
        0.905,
        "Zero-marginal-cost Nemotron scenario · 600 input + 1,200 output tokens/request · 200-prompt benchmark",
        ha="left",
        fontsize=11,
        color=MUTED,
    )
    caveats = [
        f"{row['label'].replace(' (free)', '')}: {row['coverage']}/{metadata['n_items']} scored, "
        f"{row['truncations']} truncation{'s' if row['truncations'] != 1 else ''}"
        for row in rows
        if row["coverage"] < metadata["n_items"] or row["truncations"]
    ]
    fig.text(
        0.09,
        0.015,
        "† " + "; ".join(caveats) + ". Accuracy excludes truncated attempts; incomplete models are not used to draw the frontier.\n"
        "Modeled spend assumes access to OpenRouter's Nemotron Super :free endpoint at zero marginal cost and uses 600 input + 1,200 output tokens/request.\n"
        "This is a routing scenario, not a government procurement-price claim. Source: eval_grid.json and OpenRouter list prices, 2026-08-24.",
        ha="left",
        fontsize=7.8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.10, right=0.96, top=0.82, bottom=0.20)

    png = OUT / "modern_model_cost_comparison.png"
    svg = OUT / "modern_model_cost_comparison.svg"
    fig.savefig(png, dpi=220, facecolor=BG, bbox_inches="tight")
    fig.savefig(svg, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {svg}")


if __name__ == "__main__":
    render(*build_data())
