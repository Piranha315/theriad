"""Presentation chart for the Theriad routing benchmark.

Data source: report_final.json (2026-08-16), the first report where every model
in the comparison has full 200-item coverage and the two grader bugs found on
2026-08-15 (\\boxed{} extraction, numeric-answers-with-units) are fixed.

What this chart may and may not claim is documented in BENCHMARK_STATUS.md.
In short: the router-level comparison below is sound; per-model rankings are NOT,
because coverage across the wider registry is still asymmetric.

Run:  python3 scripts/make_benchmark_charts.py
Out:  charts/theriad_regret_curve.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Every label here is currency. Left on, mathtext reads the first "$" as the
# start of a math expression and fails to parse the rest of the sentence.
plt.rcParams["text.parse_math"] = False

OUT = Path(__file__).resolve().parent.parent / "charts"
OUT.mkdir(exist_ok=True)

# ── Measured (report_final.json, 3,043-cell grid) ─────────────────────────────
V = [0.001, 0.01, 0.10, 1.00]
SERIES = {
    "Theriad (routed)":            [0.018, 0.330, 7.843, 93.801],
    "Cheapest model only (free)":  [0.024, 0.414, 8.708, 102.475],
    "Best model only (grok-4.6)":  [8.339, 7.965, 8.620, 25.991],
}
ACC = {"Theriad (routed)": 0.869,
       "Cheapest model only (free)": 0.860,
       "Best model only (grok-4.6)": 0.945}
COST1K = {"Theriad (routed)": 0.0021,
          "Cheapest model only (free)": 0.0,
          "Best model only (grok-4.6)": 8.40}

INK, MUTED, BRAND, GREY, WARN = "#12141c", "#8b8f9e", "#2f6df6", "#c7cad4", "#d1495b"
COLOUR = {"Theriad (routed)": BRAND,
          "Cheapest model only (free)": GREY,
          "Best model only (grok-4.6)": WARN}

FOOTER = (
    "200 verifiable items (MMLU, GSM8K, GPQA, MMLU-Pro), single-turn QA · every "
    "model measured on all 200 · OpenRouter list prices, Aug 2026\n"
    "Regret = money lost vs a perfect-hindsight oracle, combining overspend and "
    "the cash value of wrong answers. Lower is better."
)


def regret_curve() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for name, ys in SERIES.items():
        c = COST1K[name]
        price = "free" if c == 0 else (f"${c:,.2f}/1k" if c >= 0.01
                                       else f"${c:.4f}/1k")
        ax.plot(V, ys, marker="o", ms=7, lw=2.4, color=COLOUR[name],
                label=f"{name} — {ACC[name]:.1%} correct, {price}", zorder=3)

    # The crossover is the product's actual boundary, so name it rather than
    # letting a reader discover it and conclude we hid it.
    ax.axvspan(0.001, 0.22, color=BRAND, alpha=0.05, zorder=0)
    ax.text(0.0014, 150, "Theriad wins here", fontsize=10.5, color=BRAND,
            fontweight="bold")
    ax.text(0.62, 150, "frontier model wins", fontsize=10.5, color=WARN,
            fontweight="bold", ha="right")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Value of one correct answer (log scale)", fontsize=10, color=MUTED)
    ax.set_ylabel("Cost of routing imperfectly, per 1,000 requests",
                  fontsize=10, color=MUTED)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"${v:,.0f}" if v >= 1 else f"${v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"${v:,.0f}" if v >= 1 else f"${v:g}"))
    ax.grid(color=GREY, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GREY); ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")

    fig.text(0.085, 0.955, "Routing beats both single-model strategies",
             fontsize=16, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.085, 0.893,
             "86.9% accurate at $0.002 per 1,000 requests — against 86.0% for "
             "the free model and 94.5% for the best\nmodel at $8.40. Above "
             "~$0.20 per correct answer, paying for the frontier model wins.",
             fontsize=10, color=MUTED, ha="left", va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.775, bottom=0.24)
    fig.text(0.5, 0.06, FOOTER, ha="center", va="center", fontsize=7.5,
             color=MUTED, linespacing=1.6)
    fig.savefig(OUT / "theriad_regret_curve.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / 'theriad_regret_curve.png'}")


def savings_bars(v: float = 0.01) -> None:
    """Total cost of 1,000 requests: money spent plus the value of wrong answers.

    Regret is the right metric but needs explaining. This is the same comparison
    in money a buyer already understands — you pay for the calls, and you pay
    again for every answer that comes back wrong. ``v`` is what one correct
    answer is worth; the ranking changes with it, which is the honest caveat and
    why the crossover is stated in the footer rather than buried.
    """
    rows = []
    for name in SERIES:
        spend = COST1K[name]
        wrong = (1.0 - ACC[name]) * 1000.0 * v
        rows.append((name, spend, wrong, spend + wrong))
    rows.sort(key=lambda r: r[3])
    best_alt = max(r[3] for r in rows)
    names = [r[0] for r in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    y = range(len(rows))
    for i, (name, spend, wrong, total) in enumerate(rows):
        ax.barh(i, spend, color=COLOUR[name], height=0.5, zorder=3)
        ax.barh(i, wrong, left=spend, color=COLOUR[name], height=0.5,
                alpha=0.35, zorder=3)
        ax.text(total + best_alt * 0.015, i, f"${total:,.2f}", va="center",
                fontsize=13, fontweight="bold", color=INK)
        if total < best_alt:
            saved = best_alt - total
            ax.text(total + best_alt * 0.145, i,
                    f"saves ${saved:,.2f} per 1,000 requests "
                    f"({saved / best_alt:.0%})",
                    va="center", fontsize=9.5, color=MUTED)

    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{n}\n{ACC[n]:.1%} correct" for n in names], fontsize=10)
    ax.set_xlim(0, best_alt * 1.42)
    ax.set_xlabel("Total cost per 1,000 requests  "
                  "(solid = money spent · faded = value of wrong answers)",
                  fontsize=10, color=MUTED)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(axis="x", color=GREY, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=MUTED, labelsize=9)

    fig.text(0.075, 0.955, "Routing costs 85% less than always using the "
             "best model", fontsize=16, fontweight="bold", color=INK,
             ha="left", va="top")
    fig.text(0.075, 0.887,
             f"Counting both what you spend and what a wrong answer costs you, "
             f"at ${v:g} per correct answer.",
             fontsize=10, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(left=0.235, right=0.97, top=0.78, bottom=0.26)
    fig.text(0.5, 0.075,
             "200 verifiable items (MMLU, GSM8K, GPQA, MMLU-Pro), single-turn QA · "
             "every model measured on all 200 · OpenRouter list prices, Aug 2026\n"
             "Ranking depends on what a correct answer is worth: above ~$0.20 the "
             "frontier model's accuracy justifies its price and it wins instead.",
             ha="center", va="center", fontsize=7.5, color=MUTED, linespacing=1.6)
    fig.savefig(OUT / "theriad_savings_bars.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / 'theriad_savings_bars.png'}")


if __name__ == "__main__":
    regret_curve()
    savings_bars()
