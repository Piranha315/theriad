"""
Overnight-safe local model allowlist.

Controls which Ollama models are permitted to run without explicit user
override. The defaults are conservative: only small (<= 8B) models are
allowed in overnight mode.

Usage:
    from theriad.benchmark.allowlist import filter_models, is_safe_overnight

    safe = filter_models(discovered_models, allow_large=False)
    # safe = ["llama3.1:8b", "llama3.2:latest"]  # qwen3.6:27b excluded
"""
from __future__ import annotations

import re
from theriad.core.logging.logger import get_logger

logger = get_logger("benchmark.allowlist")

# ── Overnight-safe allowlist ──────────────────────────────────────────────────

# Models explicitly allowed in overnight-safe mode (case-insensitive prefix match).
_SAFE_PREFIXES: frozenset[str] = frozenset({
    "llama3.1:8b",
    "llama3.1:latest",
    "llama3.2:latest",
    "llama3.2:3b",
    "llama3.2:1b",
    "llama3:latest",
    "llama3:8b",
    "phi3:latest",
    "phi3:mini",
    "gemma:latest",
    "gemma:7b",
    "gemma2:latest",
    "gemma2:9b",
    "mistral:latest",
    "mistral:7b",
})

# Models explicitly blocked regardless of size heuristics.
_BLOCKED_EXPLICIT: frozenset[str] = frozenset({
    "qwen3.6:27b",
    "deepseek-r1:latest",
    "deepseek-r1:7b",   # block deepseek-r1 family entirely by default
    "deepseek-r1:8b",
})

# Regex patterns that indicate a large / memory-hungry model.
# Any model whose name matches one of these is blocked unless --allow-large-local-models.
_LARGE_MODEL_RE = re.compile(
    r"""
    \b(
        13b | 14b | 20b | 22b | 24b | 27b | 30b | 32b |
        34b | 40b | 47b | 65b | 70b | 72b | 235b | 671b
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


def is_large_model(model: str) -> bool:
    """Return True if the model name indicates it is likely > 14B parameters."""
    return bool(_LARGE_MODEL_RE.search(model))


def is_blocked_explicit(model: str) -> bool:
    """Return True if the model is on the explicit block list."""
    return model.lower() in _BLOCKED_EXPLICIT


def is_safe_overnight(model: str) -> bool:
    """
    Return True if the model is on the overnight-safe allowlist.
    Does NOT consult the large-model heuristic; use filter_models() for that.
    """
    lower = model.lower()
    return lower in _SAFE_PREFIXES


def filter_models(
    models: list[str],
    *,
    allow_large: bool = False,
) -> list[str]:
    """
    Filter a list of discovered Ollama model names for overnight-safe use.

    Rules applied in order:
      1. Explicit block list → always excluded.
      2. Large model heuristic (>14B) → excluded unless allow_large=True.
      3. Remaining models are allowed (conservative: safe allowlist is advisory,
         unknown small models are permitted).

    Args:
        models:      Raw model names from Ollama /api/tags.
        allow_large: If True, skip the large-model heuristic check.

    Returns:
        Filtered list of model names that may be run.
    """
    allowed: list[str] = []
    for model in models:
        if is_blocked_explicit(model):
            logger.info(
                "Skipping %r — on explicit overnight block list", model
            )
            continue
        if not allow_large and is_large_model(model):
            logger.info(
                "Skipping %r — large model disabled for overnight-safe mode "
                "(use --allow-large-local-models to override)",
                model,
            )
            continue
        allowed.append(model)
    return allowed
