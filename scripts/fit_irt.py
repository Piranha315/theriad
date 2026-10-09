"""Fit an item-response model to the eval grid, and check it earns its keep.

Why
---
The grid estimates every (model, item) cell independently: a Beta posterior over
one or two trials, pooled to the stratum when strata are supplied. That estimator
has no way to use the fact that models are better or worse *overall*, or that
items are harder or easier *for everyone* — so an unmeasured cell falls back to
a prior and a measured cell with n=1 barely moves off it. Both failure modes have
already produced wrong conclusions in this project: sparse premium coverage made
escalation look harmful, and n=1 shrinkage put router accuracy at 0.62 when the
model's measured rate was 0.87.

A Rasch model says the same data much more efficiently::

    P(model i solves item j) = sigmoid(ability_i - difficulty_j)

Every observation of a model informs its ability, which then informs every
*unmeasured* cell for that model; every observation of an item informs its
difficulty, likewise. 3,043 observations estimate 17 + 200 parameters instead of
3,400 independent ones.

Two things fall out for free:
  - a difficulty score per item, which is the training label a prompt-side
    difficulty predictor needs (no extra spend);
  - a principled estimate for cells nobody measured, which is what asymmetric
    coverage keeps costing us.

Whether that is *better* is an empirical question, so this script does not just
fit — it cross-validates against the estimators actually in use and prints the
comparison. If IRT does not win here, it should not be adopted.

Run:  python3 scripts/fit_irt.py
Out:  irt_fit.json  (abilities, difficulties, CV comparison)
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "eval_grid.json"
CORPUS = ROOT / "eval_corpus.jsonl"
OUT = ROOT / "irt_fit.json"

RIDGE = 1e-2      # keeps abilities finite when a model solves (or fails) everything
LR = 0.5
STEPS = 4000
SEED = 0


def load_observations():
    """One row per trial: (model_idx, item_idx, successes, trials)."""
    cells = json.loads(GRID.read_text())["cells"]
    models = sorted({c["model"] for c in cells})
    items = sorted({c["item_id"] for c in cells})
    mi = {m: i for i, m in enumerate(models)}
    ii = {t: i for i, t in enumerate(items)}
    rows = np.array([[mi[c["model"]], ii[c["item_id"]], c["successes"], c["trials"]]
                     for c in cells if c["trials"] > 0], dtype=float)
    return models, items, rows


def fit_rasch(n_models, n_items, rows, ridge=RIDGE, steps=STEPS, lr=LR):
    """Maximum a posteriori Rasch fit by gradient ascent.

    Identifiability: difficulties are centred at 0 after each step, since only
    (ability - difficulty) is determined by the data.
    """
    m_idx = rows[:, 0].astype(int)
    i_idx = rows[:, 1].astype(int)
    succ, trials = rows[:, 2], rows[:, 3]

    ability = np.zeros(n_models)
    difficulty = np.zeros(n_items)
    for _ in range(steps):
        z = ability[m_idx] - difficulty[i_idx]
        p = 1.0 / (1.0 + np.exp(-z))
        resid = succ - trials * p            # dLL/dz per observation
        g_a = np.bincount(m_idx, weights=resid, minlength=n_models) - ridge * ability
        g_d = -np.bincount(i_idx, weights=resid, minlength=n_items) - ridge * difficulty
        ability += lr * g_a / max(1, len(rows) / n_models)
        difficulty += lr * g_d / max(1, len(rows) / n_items)
        difficulty -= difficulty.mean()
    return ability, difficulty


def fit_2pl(n_models, n_items, rows, ridge=RIDGE, disc_ridge=0.5, steps=STEPS, lr=LR):
    """2PL: P = sigmoid(discrimination_j * (ability_i - difficulty_j)).

    Rasch assumes every item separates models equally well. That is false here:
    a saturated MMLU item that all 17 models solve discriminates nothing, while a
    GPQA item that splits the field cleanly discriminates a lot. Letting each
    item have its own slope lets the fit ignore the uninformative ones instead of
    being dragged by them. Discrimination is parametrised as exp(log_disc) to
    keep it positive, and shrunk toward 1.
    """
    m_idx = rows[:, 0].astype(int)
    i_idx = rows[:, 1].astype(int)
    succ, trials = rows[:, 2], rows[:, 3]

    ability = np.zeros(n_models)
    difficulty = np.zeros(n_items)
    log_disc = np.zeros(n_items)
    scale_m = max(1, len(rows) / n_models)
    scale_i = max(1, len(rows) / n_items)
    for _ in range(steps):
        disc = np.exp(log_disc)
        gap = ability[m_idx] - difficulty[i_idx]
        z = disc[i_idx] * gap
        p = 1.0 / (1.0 + np.exp(-z))
        resid = succ - trials * p
        g_a = np.bincount(m_idx, weights=resid * disc[i_idx], minlength=n_models) \
            - ridge * ability
        g_d = -np.bincount(i_idx, weights=resid * disc[i_idx], minlength=n_items) \
            - ridge * difficulty
        g_ld = np.bincount(i_idx, weights=resid * disc[i_idx] * gap, minlength=n_items) \
            - disc_ridge * log_disc
        ability += lr * g_a / scale_m
        difficulty += lr * g_d / scale_i
        log_disc += 0.1 * lr * g_ld / scale_i
        difficulty -= difficulty.mean()
    return ability, difficulty, np.exp(log_disc)


def logloss(p, succ, trials, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return -(succ * np.log(p) + (trials - succ) * np.log(1 - p)).sum()


def derive_strata_from(rows, n_items):
    """Replicate derive_strata, but only from the rows given.

    Must be recomputed per fold: a stratum label is "what fraction of models
    solved this item", which is precisely the quantity being predicted. Deriving
    it once over the whole grid leaks the test cells into the baseline and makes
    it look far better than it is — the first version of this script did exactly
    that and appeared to show IRT losing.
    """
    solved = defaultdict(list)
    for r in rows:
        solved[int(r[1])].append(1.0 if r[2] > 0 else 0.0)
    out = {}
    for j in range(n_items):
        v = solved.get(j)
        if not v:
            out[j] = "unseen"
            continue
        f = sum(v) / len(v)
        out[j] = ("unsolved" if f == 0 else "hard" if f < 0.34
                  else "moderate" if f < 0.90 else "easy")
    return out


def cross_validate(n_models, n_items, rows, strata_of_item, folds=5):
    """Hold out cells and compare held-out log-loss across estimators.

    Baselines are the estimators the harness actually uses or could trivially
    use, so the comparison answers "should we switch?" rather than "is IRT
    better than nothing?".
    """
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    chunks = np.array_split(order, folds)
    totals = defaultdict(float)
    n_trials_total = 0.0

    for f in range(folds):
        test_idx = chunks[f]
        train_idx = np.concatenate([chunks[g] for g in range(folds) if g != f])
        train, test = rows[train_idx], rows[test_idx]
        ts, tt = test[:, 2], test[:, 3]
        n_trials_total += tt.sum()

        # 1. global mean
        g = (train[:, 2].sum() + 1) / (train[:, 3].sum() + 2)
        totals["global mean"] += logloss(np.full(len(test), g), ts, tt)

        # 2. per-model mean (Laplace-smoothed)
        ms = np.bincount(train[:, 0].astype(int), weights=train[:, 2], minlength=n_models)
        mt = np.bincount(train[:, 0].astype(int), weights=train[:, 3], minlength=n_models)
        pm = (ms + 1) / (mt + 2)
        totals["per-model mean"] += logloss(pm[test[:, 0].astype(int)], ts, tt)

        # 3. per-cell Beta(1,1) — what the grid does without strata
        cs = defaultdict(lambda: [0.0, 0.0])
        for r in train:
            c = cs[(int(r[0]), int(r[1]))]; c[0] += r[2]; c[1] += r[3]
        p_cell = np.array([(lambda c: (c[0] + 1) / (c[1] + 2))(cs[(int(r[0]), int(r[1]))])
                           for r in test])
        totals["per-cell Beta(1,1)"] += logloss(p_cell, ts, tt)

        # 4. per-(model, stratum) — what the harness uses today. Strata derived
        # from the training fold only; see derive_strata_from.
        fold_strata = derive_strata_from(train, n_items)
        ss = defaultdict(lambda: [0.0, 0.0])
        for r in train:
            k = (int(r[0]), fold_strata[int(r[1])])
            ss[k][0] += r[2]; ss[k][1] += r[3]
        p_str = np.array([(lambda c: (c[0] + 1) / (c[1] + 2))(
            ss[(int(r[0]), fold_strata[int(r[1])])]) for r in test])
        totals["per-model x stratum"] += logloss(p_str, ts, tt)

        # 4b. the leaky version, kept to quantify how much the leak was worth
        ss2 = defaultdict(lambda: [0.0, 0.0])
        for r in train:
            k = (int(r[0]), strata_of_item[int(r[1])])
            ss2[k][0] += r[2]; ss2[k][1] += r[3]
        p_leak = np.array([(lambda c: (c[0] + 1) / (c[1] + 2))(
            ss2[(int(r[0]), strata_of_item[int(r[1])])]) for r in test])
        totals["per-model x stratum (LEAKY)"] += logloss(p_leak, ts, tt)

        # 5. Rasch
        a, d = fit_rasch(n_models, n_items, train)
        z = a[test[:, 0].astype(int)] - d[test[:, 1].astype(int)]
        totals["Rasch (IRT)"] += logloss(1 / (1 + np.exp(-z)), ts, tt)

        # 6. 2PL
        a2, d2, disc = fit_2pl(n_models, n_items, train)
        z2 = disc[test[:, 1].astype(int)] * (a2[test[:, 0].astype(int)]
                                             - d2[test[:, 1].astype(int)])
        totals["2PL (IRT)"] += logloss(1 / (1 + np.exp(-z2)), ts, tt)

        # 7. 2PL blended with the stratum estimator — they use different
        # information (latent traits vs per-model stratum interactions), so the
        # average can beat both if their errors are not aligned.
        totals["2PL + stratum (mean)"] += logloss(
            0.5 * (1 / (1 + np.exp(-z2))) + 0.5 * p_str, ts, tt)
        totals["Rasch + stratum (mean)"] += logloss(
            0.5 * (1 / (1 + np.exp(-z))) + 0.5 * p_str, ts, tt)

        # CONTROLS. Averaging two probabilities shrinks confident predictions
        # toward the middle, and log-loss rewards that whenever the base
        # estimator is overconfident — which the stratum estimator is, on small
        # per-stratum counts. If blending with an *uninformative* partner gains
        # as much as blending with IRT, the gain is shrinkage, not information,
        # and adopting IRT would be cargo-culting a temperature adjustment.
        totals["CONTROL stratum + global"] += logloss(
            0.5 * p_str + 0.5 * np.full(len(test), g), ts, tt)
        totals["CONTROL stratum + per-model"] += logloss(
            0.5 * p_str + 0.5 * pm[test[:, 0].astype(int)], ts, tt)

    return {k: v / n_trials_total for k, v in totals.items()}


def main() -> None:
    models, items, rows = load_observations()
    n_m, n_i = len(models), len(items)
    print(f"{len(rows)} cells · {n_m} models · {n_i} items")

    # Difficulty strata, replicating derive_strata's definition, for baseline 4.
    solved = defaultdict(list)
    for r in rows:
        solved[int(r[1])].append(1.0 if r[2] > 0 else 0.0)
    def stratum(j):
        v = sum(solved[j]) / len(solved[j])
        return "unsolved" if v == 0 else ("hard" if v < 0.34
                                          else ("moderate" if v < 0.90 else "easy"))
    strata_of_item = {j: stratum(j) for j in range(n_i)}

    print("\nheld-out log-loss per trial (5-fold, lower is better):")
    res = cross_validate(n_m, n_i, rows, strata_of_item)
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        print(f"   {k:22s} {v:.4f}")

    ability, difficulty = fit_rasch(n_m, n_i, rows)
    print("\nmodel ability (logit; higher = better):")
    for i in np.argsort(-ability):
        print(f"   {ability[i]:+6.3f}  {models[i]}")

    src = {}
    for line in CORPUS.read_text().splitlines():
        if line.strip():
            o = json.loads(line); src[o["id"]] = o["source"]
    by_src = defaultdict(list)
    for j, item in enumerate(items):
        by_src[src.get(item, "?")].append(difficulty[j])
    print("\nmean item difficulty by source (higher = harder):")
    for k, v in sorted(by_src.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"   {np.mean(v):+6.3f}  {k:10s} n={len(v)}  sd={np.std(v):.3f}")

    OUT.write_text(json.dumps({
        "cv_logloss_per_trial": res,
        "ability": {m: float(a) for m, a in zip(models, ability)},
        "difficulty": {t: float(d) for t, d in zip(items, difficulty)},
    }, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
