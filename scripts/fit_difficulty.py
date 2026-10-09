"""Predict item difficulty from the prompt alone, and check it earns its keep.

Why
---
Cross-tier escalation is decidable only if the router can tell how hard *this*
request is. ``scripts/sim_escalation.py`` showed why: with no per-request signal
the argmax of ``quality x V - cost`` is identical for every request, so the
router collapses to one model per price point. The Rasch quality spread inside
the premium tier is 0.0017 on an easy item and 0.5528 on a hard one — all of the
routing signal lives in that conditioning.

The eval grid already supplies the label for free. ``fit_rasch`` estimates a
difficulty per item on a logit scale; those are exactly the targets a
prompt-side predictor needs, and they cost no extra provider spend.

At inference the router sees only the prompt, so the features here are the ones
``theriad.layers.layer1_features`` already extracts, plus a few cheap lexical
counts. No embeddings and no model call: this has to run on the hot path ahead
of the request it is routing.

Honesty
-------
200 items and ~20 features is a thin fit, and difficulty may simply not be
predictable from surface text. So this script does not just fit — it
cross-validates against the baselines that would replace it:

  - predict the global mean difficulty (what "no predictor" means)
  - predict the per-source mean (an oracle: source is NOT available at
    inference, so this is an upper bound on what a perfect topic classifier
    could buy, not a shippable option)

If ridge does not beat the global mean out of sample, there is no predictor here
and the honest output is to say so rather than ship a fit that dresses up noise.

Run:  python3 scripts/fit_difficulty.py
"""
from __future__ import annotations

import datetime
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

from theriad.eval import OutcomeGrid, fit_rasch, load_corpus
from theriad.layers.layer1_features.extractor import FeatureExtractor

# Imported rather than redefined: featurise is the contract between training and
# inference, and a private copy here would silently drift from the router's.
from theriad.routing.difficulty import MODEL_PATH, DifficultyModel, featurise

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval_corpus.jsonl"
GRID = ROOT / "eval_grid.json"
OUT = ROOT / "difficulty_fit.json"

FOLDS = 5
SEED = 0
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge with an unpenalised intercept."""
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    penalty = alpha * np.eye(d + 1)
    penalty[0, 0] = 0.0
    return np.linalg.solve(Xb.T @ Xb + penalty, Xb.T @ y)


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(X), 1)), X]) @ w


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation. Ranking matters more than absolute difficulty here:
    the router needs to know which requests are harder, not their logit."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def main() -> None:
    items = load_corpus(CORPUS)
    grid = OutcomeGrid.load(GRID)
    fit = fit_rasch(grid)

    extractor = FeatureExtractor()
    rows, targets, sources = [], [], []
    for item in items:
        if item.id not in fit.difficulty:
            continue
        rows.append(featurise(item.prompt, extractor))
        targets.append(fit.difficulty[item.id])
        sources.append(item.source)

    names = sorted(rows[0])
    X = np.array([[r[n] for n in names] for r in rows], dtype=float)
    y = np.array(targets, dtype=float)
    src = np.array(sources)
    print(f"{len(y)} items · {X.shape[1]} features")
    print(f"difficulty: mean={y.mean():.3f} sd={y.std():.3f} "
          f"min={y.min():.2f} max={y.max():.2f}")

    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(len(y)), FOLDS)

    preds = {k: np.zeros(len(y)) for k in
             ("global mean", "per-source mean (ORACLE)", "ridge")}
    chosen_alphas = []

    for f in range(FOLDS):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(FOLDS) if g != f])

        preds["global mean"][test] = y[train].mean()

        source_means = {s: y[train][src[train] == s].mean()
                        for s in np.unique(src[train])}
        preds["per-source mean (ORACLE)"][test] = [
            source_means.get(s, y[train].mean()) for s in src[test]
        ]

        # Standardise on the training fold only.
        mu, sd = X[train].mean(0), X[train].std(0)
        sd[sd < 1e-9] = 1.0
        Xtr, Xte = (X[train] - mu) / sd, (X[test] - mu) / sd

        # Inner split to pick alpha, so the outer fold stays untouched.
        cut = int(len(train) * 0.8)
        best_alpha, best_err = RIDGE_ALPHAS[0], float("inf")
        for alpha in RIDGE_ALPHAS:
            w = ridge_fit(Xtr[:cut], y[train][:cut], alpha)
            err = rmse(predict(Xtr[cut:], w), y[train][cut:])
            if err < best_err:
                best_alpha, best_err = alpha, err
        chosen_alphas.append(best_alpha)
        preds["ridge"][test] = predict(Xte, ridge_fit(Xtr, y[train], best_alpha))

    print(f"\nridge alphas chosen per fold: {chosen_alphas}")
    print("\nheld-out (5-fold):")
    print(f"   {'estimator':28}{'RMSE':>9}{'spearman':>11}")
    results = {}
    for name, p in preds.items():
        results[name] = {"rmse": rmse(p, y), "spearman": spearman(p, y)}
        print(f"   {name:28}{results[name]['rmse']:>9.3f}"
              f"{results[name]['spearman']:>11.3f}")

    base = results["global mean"]["rmse"]
    gain = (base - results["ridge"]["rmse"]) / base * 100
    print(f"\nridge vs global mean: {gain:+.1f}% RMSE")
    if results["ridge"]["rmse"] >= base:
        print("VERDICT: ridge does NOT beat predicting the mean. There is no "
              "prompt-side difficulty signal here; do not ship this predictor.")
    else:
        print("VERDICT: ridge beats the mean out of sample.")

    # Which features carry the signal, fitted on everything for reporting only.
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    w = ridge_fit((X - mu) / sd, y, float(np.median(chosen_alphas)))
    weights = sorted(zip(names, w[1:]), key=lambda kv: -abs(kv[1]))
    print("\nlargest standardised coefficients:")
    for name, coef in weights[:8]:
        print(f"   {coef:+7.3f}  {name}")

    by_source = defaultdict(list)
    for s, t in zip(sources, targets):
        by_source[s].append(t)
    print("\nmean difficulty by source (why the oracle row is strong):")
    for s, vals in sorted(by_source.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"   {np.mean(vals):+7.3f}  {s:10} n={len(vals)} sd={np.std(vals):.2f}")

    # Out-of-fold predictions, keyed by item. Anything downstream that routes on
    # predicted difficulty MUST use these rather than a fit over the whole set:
    # an in-sample difficulty is partly memorised, and routing on it would score
    # a predictor that does not exist at inference time.
    ids = [item.id for item in items if item.id in fit.difficulty]
    OUT.write_text(json.dumps({
        "cv": results,
        "n_items": len(y),
        "features": names,
        "coefficients": dict(zip(names, w[1:].tolist())),
        "intercept": float(w[0]),
        "oof_prediction": {i: float(p) for i, p in zip(ids, preds["ridge"])},
        "actual_difficulty": {i: float(t) for i, t in zip(ids, y)},
    }, indent=1))

    # Install the production artifact only when the fit earned it. Writing
    # coefficients that lost to the mean would give the router a confident
    # difficulty signal built from noise, which is worse than having none: the
    # empty-model path degrades to average difficulty and behaves as before.
    if results["ridge"]["rmse"] < base:
        model = DifficultyModel(
            coefficients=dict(zip(names, w[1:].tolist())),
            intercept=float(w[0]),
            mean={n: float(m) for n, m in zip(names, mu)},
            stdev={n: float(s) for n, s in zip(names, sd)},
            # Saturation bounds, not decoration: predict() clips to these
            # because a linear model has no evidence outside the difficulty
            # range it was trained on.
            difficulty_min=float(y.min()),
            difficulty_max=float(y.max()),
            spearman=results["ridge"]["spearman"],
            trained_on_items=len(y),
            generated_at=datetime.date.today().isoformat(),
        )
        MODEL_PATH.write_text(json.dumps(model.to_dict(), indent=1))
        print(f"installed {MODEL_PATH}")
    else:
        print(f"NOT installing {MODEL_PATH} — the fit did not beat the mean.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
