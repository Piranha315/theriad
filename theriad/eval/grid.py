"""
Outcome grid: measured success rates for (model × item), with asymmetric coverage.

The grid is the only expensive artifact in the harness. Everything downstream —
oracle, regret, strata, and the cold-start table for outcome-based routing — is
computed offline from it. It is cached to disk and additive, so a run can be
resumed or extended without re-spending.

Asymmetric coverage
-------------------
Cost per cell varies by ~20x across the tiers, and the number of cells needed
does not. Cheap candidates carry most production traffic and are cheap to
measure, so they get full coverage. Expensive candidates are needed only to
price a small number of escalation boundaries, so they get a stratified
subsample.

The strong version of this: the top of the escalation ladder often needs no
per-item coverage at all. If policy escalates to the ceiling tier whenever
everything below fails, then for those items the *action* is fixed regardless of
what the ceiling model would have scored — measuring it buys no decision.

Coverage is therefore per-model and includes 0.0 ("do not measure"). But note
what full skipping costs: under an expected-value oracle, whether to escalate
from premium to frontier depends on the *gap* between their success
probabilities, so with no frontier data that boundary cannot be priced — you can
only evaluate a policy that always escalates, never one that decides whether to.
``item_fraction`` around 0.2 keeps the boundary measurable at a fraction of full
cost, because stratum-level estimates are what the router needs anyway (it
cannot see item identity at inference time either).

Subsampling is deterministic: cells are chosen by hashing (model, item_id), so
the same plan reproduces across runs and a resumed run extends the same subset
rather than drifting onto new items.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import EvalItem

# ── Coverage ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelCoverage:
    """How much of the corpus to measure for one model.

    Fields:
        model:            Model slug as sent to the provider.
        tier:             Tier this model represents (reporting only).
        item_fraction:    [0, 1]. 1.0 measures every item; 0.0 skips the model
                          entirely; 0.2 takes a deterministic 20% subsample.
        samples_per_item: k. Must be ≥ 1. Single-sample cells cannot distinguish
                          a model that fails an item from one that got unlucky,
                          which is exactly the distinction the oracle needs.
    """
    model: str
    tier: str
    item_fraction: float = 1.0
    samples_per_item: int = 5

    def __post_init__(self) -> None:
        if not 0.0 <= self.item_fraction <= 1.0:
            raise ValueError(
                f"item_fraction must be in [0, 1], got {self.item_fraction}"
            )
        if self.samples_per_item < 1:
            raise ValueError(
                f"samples_per_item must be >= 1, got {self.samples_per_item}"
            )

    def covers(self, item_id: str) -> bool:
        """Deterministic membership test for this model's subsample."""
        if self.item_fraction >= 1.0:
            return True
        if self.item_fraction <= 0.0:
            return False
        digest = hashlib.sha256(f"{self.model}:{item_id}".encode()).digest()
        # First 8 bytes as a big-endian int, mapped to [0, 1).
        position = int.from_bytes(digest[:8], "big") / (1 << 64)
        return position < self.item_fraction


# ── Cells ─────────────────────────────────────────────────────────────────────


@dataclass
class Cell:
    """Accumulated outcomes for one (model, item) pair.

    ``truncated`` counts responses cut off by an output-token cap. Those are
    **excluded from trials and successes**: a truncated response is missing data,
    not a failed attempt. Scoring it as a failure biases the grid against verbose
    and reasoning-heavy models exactly on the hard items where they generate the
    longest chains — which would make the harness conclude that reasoning models
    are weak at reasoning. Cost is still accumulated, because the call was paid
    for either way.
    """
    model: str
    item_id: str
    trials: int = 0
    successes: int = 0
    cost_usd: float = 0.0
    truncated: int = 0

    @property
    def raw_rate(self) -> float | None:
        """Unsmoothed success rate, or None with no trials.

        Prefer ``theriad.eval.oracle.posterior`` for anything decision-facing:
        the raw rate collapses to exactly 0.0 or 1.0 at small k, which is the
        artifact the smoothed estimate exists to avoid.
        """
        return self.successes / self.trials if self.trials else None

    @property
    def attempts(self) -> int:
        """Total paid calls, including discarded truncations."""
        return self.trials + self.truncated

    @property
    def truncation_rate(self) -> float:
        """Fraction of paid calls discarded for truncation.

        Anything materially above zero means the output cap is too low and the
        cell's estimate is drawn from a biased subsample of that model's
        behaviour. Raise the cap and re-run rather than interpreting the number.
        """
        return self.truncated / self.attempts if self.attempts else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "item_id": self.item_id,
            "trials": self.trials,
            "successes": self.successes,
            "cost_usd": round(self.cost_usd, 8),
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Cell":
        return cls(
            model=d["model"],
            item_id=d["item_id"],
            trials=d.get("trials", 0),
            successes=d.get("successes", 0),
            cost_usd=d.get("cost_usd", 0.0),
            truncated=d.get("truncated", 0),
        )


# ── Grid ──────────────────────────────────────────────────────────────────────


@dataclass
class PlannedCall:
    """One (model, item) cell that still needs ``n_samples`` executions."""
    model: str
    item_id: str
    n_samples: int


class OutcomeGrid:
    """Sparse (model, item) → Cell store, JSON-backed and additive."""

    def __init__(self, cells: dict[tuple[str, str], Cell] | None = None):
        self._cells: dict[tuple[str, str], Cell] = cells or {}

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(
        self,
        model: str,
        item_id: str,
        success: bool,
        cost_usd: float = 0.0,
        truncated: bool = False,
    ) -> None:
        """Add one observed call.

        When ``truncated`` is True the call is counted and charged but does NOT
        contribute a trial — see Cell for why truncation is missing data rather
        than failure.
        """
        key = (model, item_id)
        cell = self._cells.get(key)
        if cell is None:
            cell = Cell(model=model, item_id=item_id)
            self._cells[key] = cell
        cell.cost_usd += cost_usd
        if truncated:
            cell.truncated += 1
            return
        cell.trials += 1
        cell.successes += int(success)

    def cell(self, model: str, item_id: str) -> Cell | None:
        return self._cells.get((model, item_id))

    def cells(self) -> list[Cell]:
        """Every recorded cell, in a stable (model, item_id) order.

        Sorted so that consumers which fold over the grid — the Rasch fit in
        ``theriad.eval.irt``, for one — are reproducible run to run rather than
        depending on dict insertion order from whatever sequence the harness
        happened to execute in.
        """
        return [self._cells[k] for k in sorted(self._cells)]

    def models(self) -> list[str]:
        return sorted({m for m, _ in self._cells})

    def item_ids(self) -> list[str]:
        return sorted({i for _, i in self._cells})

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self._cells.values())

    @property
    def total_trials(self) -> int:
        return sum(c.trials for c in self._cells.values())

    def __len__(self) -> int:
        return len(self._cells)

    # ── Planning ──────────────────────────────────────────────────────────────

    def plan(
        self,
        items: list[EvalItem],
        coverage: list[ModelCoverage],
        attempt_multiplier: float = 3.0,
        retry_truncated: bool = False,
    ) -> list[PlannedCall]:
        """Return the cells still needing execution under ``coverage``.

        Already-recorded trials are subtracted, so a resumed run only fills gaps.
        Cells whose target is already met are omitted entirely.

        Truncated calls do not count as trials, so a cell that keeps truncating
        would be re-planned forever. ``attempt_multiplier`` bounds that: once a
        cell has been *attempted* more than ``samples_per_item ×
        attempt_multiplier`` times it is abandoned rather than re-planned. That
        is a real finding, not a transient error — the output cap is too low for
        that model — so surface it with ``truncated_cells()`` and raise the cap
        instead of spending more.

        When ``retry_truncated`` is true, one fresh call is scheduled for each
        covered cell that has only truncated attempts and no usable trial,
        regardless of the normal retry budget. This explicit recovery mode is
        intended for a higher or uncapped generation limit; ordinary resume
        remains bounded.
        """
        planned: list[PlannedCall] = []
        for cov in coverage:
            if cov.item_fraction <= 0.0:
                continue
            budget = int(cov.samples_per_item * attempt_multiplier)
            for item in items:
                if not cov.covers(item.id):
                    continue
                done = self.cell(cov.model, item.id)
                have = done.trials if done else 0
                attempts = done.attempts if done else 0
                if retry_truncated and done and done.truncated > 0 and have == 0:
                    planned.append(
                        PlannedCall(model=cov.model, item_id=item.id, n_samples=1)
                    )
                    continue
                need = cov.samples_per_item - have
                if need > 0 and attempts < budget:
                    planned.append(
                        PlannedCall(
                            model=cov.model,
                            item_id=item.id,
                            n_samples=min(need, budget - attempts),
                        )
                    )
        return planned

    def truncated_cells(self, threshold: float = 0.0) -> list[Cell]:
        """Cells whose truncation rate exceeds ``threshold``, worst first."""
        return sorted(
            (c for c in self._cells.values() if c.truncation_rate > threshold),
            key=lambda c: -c.truncation_rate,
        )

    @property
    def total_truncated(self) -> int:
        return sum(c.truncated for c in self._cells.values())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "cells": [c.to_dict() for c in self._cells.values()],
        }
        # Replace atomically so an interrupted checkpoint cannot leave a
        # half-written JSON file that blocks the next resume.
        tmp = p.with_name(f".{p.name}.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        with tmp.open("rb") as fh:
            os.fsync(fh.fileno())
        tmp.replace(p)

    @classmethod
    def load(cls, path: Path | str) -> "OutcomeGrid":
        p = Path(path)
        if not p.exists():
            return cls()
        payload = json.loads(p.read_text())
        cells = {}
        for raw in payload.get("cells", []):
            cell = Cell.from_dict(raw)
            cells[(cell.model, cell.item_id)] = cell
        return cls(cells)


# ── Difficulty strata (derived, never assumed) ────────────────────────────────

STRATUM_EASY = "easy"
STRATUM_MODERATE = "moderate"
STRATUM_HARD = "hard"
STRATUM_UNSOLVED = "unsolved"

_STRATUM_ORDER = [STRATUM_EASY, STRATUM_MODERATE, STRATUM_HARD, STRATUM_UNSOLVED]


@dataclass
class Strata:
    """Difficulty buckets derived from measured solve rates.

    Difficulty is computed from the grid, never inferred from a benchmark's
    vintage or reputation. Vintage stays a reporting covariate: if regret is flat
    across vintages the router's boundary is likely transferable, and if it
    climbs on recent items the boundary is fitting contamination.

    "unsolved" means no model in the grid solved the item in any trial. That is a
    statement about this sampling budget, not a property of the item: with k=5
    per cell, a model whose true success rate is 5% will show zero successes
    about 77% of the time. The label is useful for reporting coverage of the hard
    tail; it must not be read as "impossible", and the oracle deliberately does
    not use it.
    """
    by_item: dict[str, str] = field(default_factory=dict)
    solve_rate: dict[str, float] = field(default_factory=dict)

    def stratum(self, item_id: str) -> str:
        return self.by_item.get(item_id, STRATUM_MODERATE)

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in _STRATUM_ORDER}
        for s in self.by_item.values():
            out[s] = out.get(s, 0) + 1
        return out


def derive_strata(grid: OutcomeGrid, items: list[EvalItem]) -> Strata:
    """Bucket items by the fraction of measured models that ever solved them."""
    strata = Strata()
    for item in items:
        rates = [
            cell.raw_rate
            for model in grid.models()
            if (cell := grid.cell(model, item.id)) is not None and cell.trials > 0
        ]
        if not rates:
            continue
        solved_by = sum(1 for r in rates if r and r > 0.0) / len(rates)
        strata.solve_rate[item.id] = solved_by
        if solved_by == 0.0:
            strata.by_item[item.id] = STRATUM_UNSOLVED
        elif solved_by < 0.34:
            strata.by_item[item.id] = STRATUM_HARD
        elif solved_by < 0.90:
            strata.by_item[item.id] = STRATUM_MODERATE
        else:
            strata.by_item[item.id] = STRATUM_EASY
    return strata
