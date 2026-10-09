"""Prompt-side difficulty prediction.

What this is for
----------------
Routing that cannot tell how hard a request is cannot decide when to escalate.
Scoring candidates by ``quality x V - cost`` with no per-request signal picks the
same model for every request, because the argmax does not depend on the request
— measured in ``scripts/sim_escalation.py``, where the router collapsed to one
model per price point.

This module supplies the missing conditioning: a cheap estimate of the item's
Rasch difficulty from the prompt alone, so the router can ask
``P(model solves *this*) = sigmoid(ability - difficulty)`` and pick the cheapest
candidate clearing a success bar.

Scale
-----
Predictions are on the Rasch logit scale, centred so that 0.0 is an
average-difficulty item in the eval corpus. Roughly: -3 is a routine GSM8K word
problem, +2.6 is a typical MMLU-Pro item, +10 is at the unsolved tail. It is
directly comparable to ``theriad.eval.irt.RaschFit.difficulty`` because that is
what it was trained against.

How good is it
--------------
5-fold held-out on 200 items (``scripts/fit_difficulty.py``):

    global mean               RMSE 3.260   spearman -0.071
    per-source mean (ORACLE)  RMSE 2.607   spearman  0.504
    ridge (this model)        RMSE 2.743   spearman  0.539

It beats predicting the mean by 15.9% RMSE and edges the per-source oracle on
rank correlation — and source is not available at inference, so that row was
only ever an upper bound. Rank matters more than absolute value here: the router
needs to know which requests are harder, not their exact logit.

It is a linear model on surface features, so it is weakest exactly where you
would expect: a short prompt that happens to be deeply hard reads as easy. In
``scripts/sim_threshold.py`` it captured 62% of the regret reduction that a
perfect difficulty oracle would deliver. Treat it as a useful prior, not a
measurement, and keep the fallback path honest.

Retraining
----------
    python3 scripts/fit_difficulty.py     # rewrites difficulty_model.json

Retrain whenever the grid grows materially. The script refuses to endorse a fit
that does not beat the mean out of sample; heed it rather than shipping the
coefficients anyway.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("difficulty_model.json")

#: Difficulty assumed when no model is installed. 0.0 is the corpus mean, which
#: makes an uncalibrated deployment behave as though every request were of
#: average difficulty — i.e. it degrades to the un-differentiated behaviour that
#: preceded this module rather than to a random one.
DEFAULT_DIFFICULTY = 0.0

#: Standardised features are clipped to this many training standard deviations
#: before the linear model sees them. Production prompts are not benchmark
#: items — a one-line question sits many sigma below the corpus mean on every
#: length feature — and an unclipped linear fit turns that into an arbitrarily
#: confident extrapolation.
FEATURE_CLIP = 3.0

# Lexical cues that plausibly track reasoning depth. Kept small and readable:
# with 200 training rows every extra feature is a chance to fit noise. This dict
# is the shared contract between training and inference — scripts/fit_difficulty
# imports ``featurise`` from here so the two cannot drift apart.
_CUES = {
    "cue_calculate": r"\b(calculate|compute|how many|how much|total)\b",
    "cue_why": r"\b(why|explain|because|reason)\b",
    "cue_which": r"\b(which of the following|which one|select|choose)\b",
    "cue_number": r"\d",
    "cue_unit": r"\b(mol|kg|km|cm|ml|joule|volt|hz|nm)\b",
    "cue_negation": r"\b(not|except|never|neither)\b",
}


def featurise(prompt: str, extractor=None) -> dict[str, float]:
    """Numeric features for one prompt.

    Reuses ``layer1_features``' extractor so difficulty prediction costs no new
    parsing on the hot path — layer 1 has usually run already.
    """
    if extractor is None:
        extractor = _default_extractor()
    feats = {
        k: float(v)
        for k, v in extractor.extract(prompt).to_dict().items()
        if isinstance(v, (int, float, bool))
    }
    lowered = prompt.lower()
    for name, pattern in _CUES.items():
        feats[name] = float(len(re.findall(pattern, lowered)))
    # Length in logs: a 4000-char prompt is not 10x harder than a 400-char one.
    feats["log_chars"] = math.log1p(len(prompt))
    feats["log_words"] = math.log1p(len(prompt.split()))
    return feats


@lru_cache(maxsize=1)
def _default_extractor():
    from theriad.layers.layer1_features.extractor import FeatureExtractor

    return FeatureExtractor()


@dataclass(frozen=True)
class DifficultyModel:
    """Linear predictor of Rasch difficulty from prompt features."""

    coefficients: dict[str, float] = field(default_factory=dict)
    intercept: float = DEFAULT_DIFFICULTY
    mean: dict[str, float] = field(default_factory=dict)
    stdev: dict[str, float] = field(default_factory=dict)
    difficulty_min: float = -FEATURE_CLIP
    difficulty_max: float = FEATURE_CLIP
    spearman: float = 0.0
    trained_on_items: int = 0
    generated_at: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.coefficients

    def predict(self, prompt: str, extractor=None) -> float:
        """Predicted difficulty on the Rasch logit scale.

        Returns ``DEFAULT_DIFFICULTY`` when no model is installed, so callers
        never need to special-case a fresh checkout.

        Two clamps, both load-bearing. A linear model extrapolates without
        bound, and production prompts are not benchmark items: an unclamped fit
        scored a two-line arithmetic question at -17.1 against a training range
        of -3.02..10.18, which makes every candidate clear any success bar and
        silently disables escalation. So standardised features are clipped to
        +/-FEATURE_CLIP sigma, and the result is clipped to the difficulty range
        actually observed in training. Outside that range the model has no
        evidence, and saturating is the honest behaviour.
        """
        if self.is_empty:
            return DEFAULT_DIFFICULTY
        feats = featurise(prompt, extractor)
        total = self.intercept
        for name, weight in self.coefficients.items():
            centre = self.mean.get(name, 0.0)
            scale = self.stdev.get(name, 1.0) or 1.0
            z = (feats.get(name, centre) - centre) / scale
            total += weight * max(-FEATURE_CLIP, min(FEATURE_CLIP, z))
        return max(self.difficulty_min, min(self.difficulty_max, total))

    @classmethod
    def from_dict(cls, raw: dict) -> "DifficultyModel":
        return cls(
            coefficients=dict(raw.get("coefficients", {})),
            intercept=float(raw.get("intercept", DEFAULT_DIFFICULTY)),
            mean=dict(raw.get("mean", {})),
            stdev=dict(raw.get("stdev", {})),
            difficulty_min=float(raw.get("difficulty_min", -FEATURE_CLIP)),
            difficulty_max=float(raw.get("difficulty_max", FEATURE_CLIP)),
            spearman=float(raw.get("spearman", 0.0)),
            trained_on_items=int(raw.get("trained_on_items", 0)),
            generated_at=raw.get("generated_at"),
        )

    def to_dict(self) -> dict:
        return {
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "mean": self.mean,
            "stdev": self.stdev,
            "difficulty_min": self.difficulty_min,
            "difficulty_max": self.difficulty_max,
            "spearman": self.spearman,
            "trained_on_items": self.trained_on_items,
            "generated_at": self.generated_at,
        }


@lru_cache(maxsize=1)
def load_difficulty_model(path: str | None = None) -> DifficultyModel:
    """Load the installed model, or an empty one on a fresh checkout.

    Cached: the file does not change within a process, and ``predict`` is on the
    routing hot path.
    """
    target = Path(path) if path else MODEL_PATH
    if not target.exists():
        return DifficultyModel()
    try:
        return DifficultyModel.from_dict(json.loads(target.read_text()))
    except (json.JSONDecodeError, TypeError, ValueError):
        # A corrupt artifact must not take routing down. Average difficulty is
        # the safe degradation: it reproduces pre-difficulty behaviour.
        return DifficultyModel()


def success_probability(ability: float, difficulty: float) -> float:
    """Rasch: P(a model of this ability solves an item of this difficulty)."""
    z = ability - difficulty
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
