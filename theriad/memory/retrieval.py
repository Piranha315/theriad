"""Retrieve specific imported or Theriad-native conversations on explicit request."""

from __future__ import annotations

import re
from dataclasses import dataclass

from theriad.storage.database import get_db

_EXPLICIT_RECALL = re.compile(
    r"(?:@memory\b|"
    r"(?:look at|find|retrieve|search|use|recall|from)\s+(?:my\s+)?"
    r"(?:conversation|chat|thread)\b|"
    r"(?:previous|earlier|specific|named|called)\s+"
    r"(?:conversation|chat|thread|discussion)\b)",
    re.IGNORECASE,
)
_WORDS = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{2,}")
_STOP = frozenset(
    "the and for that this with from have about into look find show conversation chat thread please".split()
)


@dataclass(frozen=True)
class ConversationSource:
    id: str
    title: str
    provider: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "title": self.title, "provider": self.provider}


def _terms(text: str) -> set[str]:
    return {word.lower() for word in _WORDS.findall(text or "") if word.lower() not in _STOP}


def _score(query_terms: set[str], title: str, text: str) -> int:
    title_terms = _terms(title)
    body_terms = _terms(text)
    return len(query_terms & title_terms) * 8 + len(query_terms & body_terms)


def retrieve_conversations(
    query: str,
    *,
    imported_entries: list | None = None,
    project_id: str | None = None,
    max_chars: int = 12_000,
) -> tuple[str, list[ConversationSource]]:
    """Return directly requested original conversation text and source metadata."""
    if not _EXPLICIT_RECALL.search(query or ""):
        return "", []
    query_terms = _terms(query)
    if not query_terms:
        return "", []

    candidates: list[tuple[int, ConversationSource, str]] = []
    for entry in imported_entries or []:
        title = str(getattr(entry, "conversation_title", "") or "Imported conversation")
        text = str(getattr(entry, "text", ""))
        score = _score(query_terms, title, text)
        if score:
            candidates.append(
                (score, ConversationSource(str(entry.id), title, str(entry.source)), text)
            )

    db = get_db()
    if project_id is None:
        rows = db.execute(
            "SELECT id, title FROM conversations WHERE project_id IS NULL "
            "ORDER BY updated_at DESC LIMIT 100"
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id, title FROM conversations WHERE project_id = ? "
            "ORDER BY updated_at DESC LIMIT 100",
            (project_id,),
        ).fetchall()
    for row in rows:
        messages = db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? "
            "ORDER BY created_at ASC LIMIT 200",
            (row["id"],),
        ).fetchall()
        text = "\n".join(f"[{message['role']}]: {message['content']}" for message in messages)
        score = _score(query_terms, row["title"] or "Untitled", text)
        if score:
            candidates.append(
                (score, ConversationSource(row["id"], row["title"] or "Untitled", "theriad"), text)
            )

    if not candidates:
        return "", []
    candidates.sort(key=lambda item: item[0], reverse=True)
    blocks: list[str] = []
    sources: list[ConversationSource] = []
    used = 0
    for _, source, text in candidates[:3]:
        block = f"Conversation: {source.title} ({source.provider})\n{text.strip()}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        block = block[:remaining]
        blocks.append(block)
        sources.append(source)
        used += len(block)
    return "\n\n=== Retrieved conversations ===\n" + "\n\n".join(blocks), sources
