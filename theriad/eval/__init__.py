"""
Router evaluation harness.

Scores a *routing policy* against a perfectly-informed oracle, rather than
scoring models against each other. The unit of analysis is the decision.

Pipeline:
    corpus  → vintage-tagged, mechanically gradable items
    grid    → measured (model × item) success rates under asymmetric coverage
    oracle  → best action per item by expected value, for a stated V
    regret  → what the router gave up against that oracle

Offline by default (D-018): only ``run populate`` makes provider calls, and it
requires --allow-paid-calls, --yes, and a budget ceiling.
"""
from .corpus import EvalItem, load_corpus, write_corpus
from .grid import ModelCoverage, OutcomeGrid, derive_strata
from .irt import RaschFit, fit_rasch
from .oracle import Action, ExpectedValueOracle, Posterior, best_of_n_probability
from .regret import RegretReport, evaluate_router

__all__ = [
    "Action",
    "EvalItem",
    "ExpectedValueOracle",
    "ModelCoverage",
    "OutcomeGrid",
    "Posterior",
    "RaschFit",
    "RegretReport",
    "best_of_n_probability",
    "derive_strata",
    "fit_rasch",
    "evaluate_router",
    "load_corpus",
    "write_corpus",
]
