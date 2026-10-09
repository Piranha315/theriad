"""
Optional local text classifier using sklearn.

Design:
- Only invoked when embedding confidence (Layer 2) falls below threshold.
- Uses TF-IDF + LogisticRegression: fast, interpretable, no GPU required.
- Model is serializable (joblib) for persistence between runs.
- The classifier has NO hard dependency on sklearn at import time.
  ImportError is caught and the layer degrades gracefully.
- Must be removable: nothing in core imports from this module.

Tradeoff: LogisticRegression requires labeled training data. At v1, we ship
with a minimal synthetic dataset per task type. Users can extend via fit().
"""
from __future__ import annotations

from dataclasses import dataclass

from theriad.core.logging.logger import get_logger

logger = get_logger("layer3.classifier")

# Minimal synthetic training data (task_type → example prompts)
_SYNTHETIC_DATA: list[tuple[str, str]] = [
    # code
    ("code", "Write a Python function to reverse a string"),
    ("code", "Debug this JavaScript code: var x = 1"),
    ("code", "Implement a binary search tree in Java"),
    ("code", "Fix the bug in this SQL query"),
    ("code", "Create a React component for a login form"),
    # qa
    ("qa", "What is the capital of Japan?"),
    ("qa", "How many continents are there?"),
    ("qa", "Who invented the telephone?"),
    ("qa", "What year did World War II end?"),
    ("qa", "What is the speed of light?"),
    # summarize
    ("summarize", "Summarize the following text in three sentences"),
    ("summarize", "Give me a TL;DR of this article"),
    ("summarize", "Condense this report into key bullet points"),
    ("summarize", "What are the main takeaways from this document?"),
    # translate
    ("translate", "Translate this text from English to French"),
    ("translate", "Convert the following paragraph to Spanish"),
    ("translate", "Render this Japanese text in English"),
    # analysis
    ("analysis", "Analyze the pros and cons of microservices architecture"),
    ("analysis", "Compare PostgreSQL vs MySQL for high-traffic applications"),
    ("analysis", "Write a detailed evaluation of the given business strategy"),
]


@dataclass
class ClassifierResult:
    task_type: str
    confidence: float
    tier: str           # abstract tier label; model resolved later by policy (ADR D-016)
    reason: str


# Maps classifier task_type → tier label. Model slugs belong in the registry.
_TASK_TIER_MAP = {
    "code":      "cheap_code",
    "qa":        "cheap_bulk",
    "summarize": "cheap_bulk",
    "translate": "cheap_bulk",
    "analysis":  "strong_reasoning",
}


class LocalClassifier:
    """
    TF-IDF + Logistic Regression text classifier.

    Removable: delete this file and ClassifierRoutingLayer — nothing else breaks.
    """

    def __init__(self):
        self._pipeline = None  # sklearn Pipeline, lazy loaded

    def _ensure_sklearn(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            return Pipeline, TfidfVectorizer, LogisticRegression
        except ImportError as exc:
            raise ImportError(
                "Layer 3 requires scikit-learn: pip install scikit-learn"
            ) from exc

    def fit(self, examples: list[tuple[str, str]] | None = None) -> None:
        """
        Train on examples (task_type, prompt_text).
        Defaults to _SYNTHETIC_DATA if not provided.
        """
        Pipeline, TfidfVectorizer, LogisticRegression = self._ensure_sklearn()
        data = examples or _SYNTHETIC_DATA
        texts = [t for _, t in data]
        labels = [label for label, _ in data]

        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ])
        self._pipeline.fit(texts, labels)
        logger.info("classifier trained", extra={"examples": len(data)})

    def predict(self, prompt: str) -> ClassifierResult:
        """Classify the prompt. fit() must be called first."""
        if self._pipeline is None:
            raise RuntimeError("LocalClassifier: call fit() before predict()")
        self._ensure_sklearn()

        proba = self._pipeline.predict_proba([prompt])[0]
        classes = self._pipeline.classes_
        best_idx = proba.argmax()
        task_type = classes[best_idx]
        confidence = float(proba[best_idx])
        tier = _TASK_TIER_MAP.get(task_type, "cheap_bulk")

        return ClassifierResult(
            task_type=task_type,
            confidence=confidence,
            tier=tier,
            reason=f"local_classifier: task={task_type} confidence={confidence:.3f}",
        )

    def save(self, path: str) -> None:
        """Persist trained model to disk via joblib."""
        try:
            import joblib
        except ImportError:
            import pickle as joblib  # type: ignore[no-redef]
        if self._pipeline is None:
            raise RuntimeError("Nothing to save — model not trained")
        joblib.dump(self._pipeline, path)
        logger.info("classifier saved", extra={"path": path})

    def load(self, path: str) -> None:
        """Load a previously saved model from disk."""
        try:
            import joblib
        except ImportError:
            import pickle as joblib  # type: ignore[no-redef]
        self._pipeline = joblib.load(path)
        logger.info("classifier loaded", extra={"path": path})
