"""Conservative, local model-memory learning.

The learner intentionally does not make a second LLM call. It records only
explicit first-person preferences and durable facts from the user's message,
then scopes them to the model that answered. This keeps learning predictable,
cheap, and reviewable in the Memory tab.
"""
from __future__ import annotations

import re

from .store import MemoryEntry, MemoryStore

_DURABLE_PREFIX = re.compile(
    r"^(?:"
    r"i\s+(?:prefer|like|love|use|am|['’]m|live|work)\b|"
    r"my\s+\w+\b|"
    r"please\s+(?:always|never)\b|"
    r"(?:always|never)\s+\b|"
    r"remember(?:\s+that)?\b|"
    r"call\s+me\b"
    r")",
    re.IGNORECASE,
)
_SENSITIVE = re.compile(
    r"\b(?:password|passcode|api[_ -]?key|access[_ -]?token|secret|private[_ -]?key|"
    r"seed phrase|social security|ssn|credit card|cvv)\b",
    re.IGNORECASE,
)


def model_memory_id(provider: str, model: str) -> str:
    """Return a stable, provider-qualified model identifier."""
    return f"{str(provider or '').strip().lower()}:{str(model or '').strip().lower()}"


def extract_explicit_memories(text: str, *, limit: int = 3) -> list[str]:
    """Extract only obvious durable user facts; never infer them from answers."""
    candidates: list[str] = []
    for raw in re.split(r"(?:\r?\n+)|(?<=[.!?])\s+", text or ""):
        cleaned = re.sub(r"^[-*•\s]+", "", raw).strip().strip('"“”')
        if not cleaned or "?" in cleaned or len(cleaned) < 8 or len(cleaned) > 280:
            continue
        if _SENSITIVE.search(cleaned) or not _DURABLE_PREFIX.match(cleaned):
            continue
        normalized = " ".join(cleaned.split())
        if normalized.casefold() not in {item.casefold() for item in candidates}:
            candidates.append(normalized)
        if len(candidates) >= limit:
            break
    return candidates


def learn_from_interaction(
    store: MemoryStore,
    *,
    model_id: str,
    user_message: str,
) -> list[MemoryEntry]:
    """Persist explicit user facts when the model's learning policy allows it."""
    profile = store.get_model_profile(model_id, create=False)
    if not profile or not profile.get("learning_enabled"):
        return []
    candidates = extract_explicit_memories(user_message)
    if not candidates:
        return []
    return store.add_model_memories(model_id, candidates)
