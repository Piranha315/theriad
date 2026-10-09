"""
Context builder for Theriad project workspace.

Assembles a context prefix from:
  - Project instructions (system note)
  - Enabled reusable skills
  - Recent conversation messages
  - Selected/attached file text

Can enforce an optional token budget and returns a truncation warning when it
is explicitly exceeded. By default, context is preserved in full; provider
context-window limits remain the final authority.
Token estimate: words * 1.33  (same heuristic used elsewhere in Theriad).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ContextPiece:
    label: str
    content: str
    tokens: int


@dataclass
class BuiltContext:
    text: str
    """Assembled context prefix ready to prepend to the user prompt."""
    pieces: list[ContextPiece] = field(default_factory=list)
    total_tokens: int = 0
    budget_tokens: int | None = None
    truncated: bool = False
    truncation_warning: str | None = None
    paid_provider_warning: str | None = None
    history_summarized: bool = False
    messages_summarized: int = 0
    recent_messages_kept: int = 0


_CHARS_PER_WORD = 5  # rough average


def _estimate_tokens(text: str) -> int:
    words = max(1, len(text.split()))
    return math.ceil(words * 1.33)


def _summarize_messages(messages: list[dict], max_tokens: int) -> str:
    """Create a deterministic, extractive history summary without another model call."""
    lines: list[str] = []
    used = 0
    for message in messages:
        role = str(message.get("role", "user")).strip().title() or "User"
        content = " ".join(str(message.get("content", "")).split())
        if not content:
            continue
        words = content.split()
        excerpt = " ".join(words[:56])
        if len(words) > 56:
            excerpt += "…"
        line = f"- {role}: {excerpt}"
        tokens = _estimate_tokens(line)
        if used + tokens > max_tokens:
            break
        lines.append(line)
        used += tokens
    return "Earlier conversation, compressed locally:\n" + "\n".join(lines) if lines else ""


def compact_conversation_history(
    messages: list[dict] | None,
    *,
    budget_tokens: int | None,
    recent_messages: int = 8,
) -> tuple[list[ContextPiece], int, int]:
    """Return history verbatim unless an explicit budget requests compaction."""
    history = [message for message in (messages or []) if str(message.get("content", "")).strip()]
    if not history or (budget_tokens is not None and budget_tokens <= 0):
        return [], 0, 0

    total = sum(_estimate_tokens(str(message.get("content", ""))) for message in history)
    if budget_tokens is None or total <= budget_tokens:
        pieces = [
            ContextPiece(
                label=f"message:{message.get('role', 'user')}",
                content=f"[{message.get('role', 'user')}]: {str(message.get('content', '')).strip()}",
                tokens=_estimate_tokens(str(message.get("content", ""))),
            )
            for message in history
        ]
        return pieces, 0, len(history)

    recent = history[-recent_messages:]
    recent_tokens = sum(_estimate_tokens(str(message.get("content", ""))) for message in recent)
    while len(recent) > 2 and recent_tokens > int(budget_tokens * 0.72):
        recent.pop(0)
        recent_tokens = sum(_estimate_tokens(str(message.get("content", ""))) for message in recent)

    older = history[: len(history) - len(recent)]
    summary_budget = max(0, budget_tokens - recent_tokens)
    summary = _summarize_messages(older, summary_budget)
    pieces: list[ContextPiece] = []
    if summary:
        pieces.append(ContextPiece("conversation_summary", summary, _estimate_tokens(summary)))
    pieces.extend(
        ContextPiece(
            label=f"message:{message.get('role', 'user')}",
            content=f"[{message.get('role', 'user')}]: {str(message.get('content', '')).strip()}",
            tokens=_estimate_tokens(str(message.get("content", ""))),
        )
        for message in recent
    )
    return pieces, len(older), len(recent)


def build_context(
    *,
    instructions: str | None = None,
    skills: str | None = None,
    messages: list[dict] | None = None,
    file_texts: list[tuple[str, str]] | None = None,  # (filename, text)
    max_context_tokens: int | None = None,
    include_paid_warning: bool = False,
) -> BuiltContext:
    """
    Build a context prefix string from the supplied components.

    Parameters
    ----------
    instructions:
        Project-level system instructions (prepended first).
    skills:
        Enabled reusable skill instructions. These are placed after project
        instructions and before files or conversation history.
    messages:
        Recent conversation history as list of {"role": …, "content": …} dicts,
        in chronological order (oldest first).
    file_texts:
        List of (filename, extracted_text) tuples for files to include.
    max_context_tokens:
        Optional hard token budget. When omitted, pieces are included in full
        and conversation history is not summarized or truncated. When set,
        pieces are included in priority order: instructions → skills → compacted
        conversation → files; the last piece to fit may be truncated at a word
        boundary.
    include_paid_warning:
        When True, append a note that attached file text will be sent to the
        provider (cost warning for users).

    Returns
    -------
    BuiltContext with assembled text and metadata.
    """
    pieces: list[ContextPiece] = []

    if instructions and instructions.strip():
        pieces.append(
            ContextPiece(
                label="project_instructions",
                content=instructions.strip(),
                tokens=_estimate_tokens(instructions),
            )
        )

    if skills and skills.strip():
        pieces.append(
            ContextPiece(
                label="enabled_skills",
                content=skills.strip(),
                tokens=_estimate_tokens(skills),
            )
        )

    history_budget = (
        max(512, min(2_048, max_context_tokens // 2))
        if max_context_tokens is not None
        else None
    )
    history_pieces, messages_summarized, recent_messages_kept = compact_conversation_history(
        messages,
        budget_tokens=history_budget,
    )
    pieces.extend(history_pieces)

    for fname, text in file_texts or []:
        if text and text.strip():
            snippet = f"--- File: {fname} ---\n{text.strip()}"
            pieces.append(
                ContextPiece(
                    label=f"file:{fname}",
                    content=snippet,
                    tokens=_estimate_tokens(snippet),
                )
            )

    # Enforce an explicitly requested budget — greedy fit in priority order
    # (instructions, skills, compacted conversation, files). The normal path
    # has no application-level request truncation.
    used_tokens = 0
    accepted: list[ContextPiece] = []
    truncated = False

    for piece in pieces:
        remaining = (
            max_context_tokens - used_tokens
            if max_context_tokens is not None
            else None
        )
        if remaining is not None and remaining <= 0:
            truncated = True
            break
        if remaining is None or piece.tokens <= remaining:
            accepted.append(piece)
            used_tokens += piece.tokens
        else:
            # Partial fit — truncate text at word boundary
            target_words = int(remaining / 1.33)
            words = piece.content.split()
            partial_text = " ".join(words[:target_words])
            if partial_text:
                partial_piece = ContextPiece(
                    label=piece.label,
                    content=partial_text + " [TRUNCATED]",
                    tokens=remaining,
                )
                accepted.append(partial_piece)
                used_tokens += remaining
            truncated = True
            break

    parts: list[str] = []
    if accepted:
        parts.append("=== Context ===")
        for p in accepted:
            parts.append(p.content)
        parts.append("=== End Context ===")

    text = "\n\n".join(parts) if parts else ""

    truncation_warning = (
        f"Context was truncated to fit within {max_context_tokens} tokens. "
        "Some file content may not be included. Earlier conversation turns were summarized locally."
        if truncated
        else None
    )

    paid_warning = (
        "Note: attached file content will be sent to the AI provider and may incur cost."
        if include_paid_warning and file_texts
        else None
    )

    return BuiltContext(
        text=text,
        pieces=accepted,
        total_tokens=used_tokens,
        budget_tokens=max_context_tokens,
        truncated=truncated,
        truncation_warning=truncation_warning,
        paid_provider_warning=paid_warning,
        history_summarized=messages_summarized > 0,
        messages_summarized=messages_summarized,
        recent_messages_kept=recent_messages_kept,
    )
