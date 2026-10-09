"""
Conversation-export parsers → named memory entries.

Supported inputs are parsed from their JSON shape, but all conversation
exports use the neutral ``import`` source. The collection's user-defined name
is the identity exposed in the memory UI.

1. Conversation exports with a ``mapping`` of message nodes.
2. Conversation exports with a ``chat_messages`` list.
3. Generic format — a list of {"title", "messages": [{"role", "content"}]}.

These are the same export files consumed by tools like Nexus AI Chat Importer,
so users who already export their history for Obsidian can feed the identical
file here.

Import policy (v1, deterministic — no model calls):
- One memory entry per conversation, built from the conversation title plus
  the first few *user* messages (assistant text is excluded: memories are
  about the user).
- Tags: auto-extracted keywords + "imported".
- Deduplication happens in MemoryStore.add_many (exact-text match).
"""
from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from datetime import datetime, timezone
from typing import Any

from .store import MemoryEntry, auto_tags

# How much of each conversation makes it into a memory entry.
MAX_USER_MESSAGES_PER_CONVERSATION = 3
MAX_MEMORY_CHARS = 500
MAX_IMPORT_FILES = 2_000
MAX_IMPORT_FILE_BYTES = 10 * 1024 * 1024
MAX_IMPORT_TOTAL_BYTES = 100 * 1024 * 1024
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".log", ".jsonl",
    ".yaml", ".yml", ".xml", ".html", ".htm", ".rst", ".py", ".js",
    ".jsx", ".ts", ".tsx", ".css", ".scss", ".java", ".go", ".rs",
    ".rb", ".php", ".swift", ".sql", ".toml", ".ini", ".conf",
}
ARCHIVE_EXTENSIONS = {".zip"}


class ImportError_(ValueError):
    """Raised when the payload matches no known export format."""


def _ts_to_iso(ts: float | None) -> str:
    if not ts:
        return datetime.now(timezone.utc).isoformat()
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return datetime.now(timezone.utc).isoformat()


def _build_entry(
    title: str,
    user_texts: list[str],
    created_at: str,
    *,
    source: str = "import",
    provider_tag: str | None = None,
) -> MemoryEntry | None:
    snippets = [t.strip() for t in user_texts if t and t.strip()]
    if not snippets:
        return None
    body = " · ".join(snippets[:MAX_USER_MESSAGES_PER_CONVERSATION])
    if len(body) > MAX_MEMORY_CHARS:
        body = body[: MAX_MEMORY_CHARS - 1] + "…"
    text = f"[{title.strip()}] {body}" if title.strip() else body
    return MemoryEntry(
        text=text,
        tags=auto_tags(body) + ["imported"] + ([provider_tag] if provider_tag else []),
        source=source,
        conversation_title=title.strip() or None,
        created_at=created_at,
        updated_at=created_at,
    )


# ── Conversation export shapes ───────────────────────────────────────────────

def _has_mapping_shape(data: Any) -> bool:
    return (
        isinstance(data, list)
        and bool(data)
        and isinstance(data[0], dict)
        and "mapping" in data[0]
    )


def _import_mapping_shape(
    data: list[dict],
    *,
    source: str = "import",
    provider_tag: str | None = None,
) -> list[MemoryEntry]:
    entries: list[MemoryEntry] = []
    for conv in data:
        if not isinstance(conv, dict):
            continue
        title = str(conv.get("title") or "")
        created = _ts_to_iso(conv.get("create_time"))
        mapping = conv.get("mapping") or {}
        if not isinstance(mapping, dict):
            continue
        # mapping is unordered; sort nodes by message create_time when present.
        nodes = []
        for node in mapping.values():
            msg = (node or {}).get("message") if isinstance(node, dict) else None
            if not msg:
                continue
            role = ((msg.get("author") or {}).get("role") or "").lower()
            if role != "user":
                continue
            parts = (msg.get("content") or {}).get("parts") or []
            text = " ".join(str(p) for p in parts if isinstance(p, str)).strip()
            if text:
                nodes.append((msg.get("create_time") or 0, text))
        nodes.sort(key=lambda pair: pair[0])
        entry = _build_entry(
            title,
            [t for _, t in nodes],
            created,
            source=source,
            provider_tag=provider_tag,
        )
        if entry:
            entries.append(entry)
    return entries


def _has_chat_messages_shape(data: Any) -> bool:
    return (
        isinstance(data, list)
        and bool(data)
        and isinstance(data[0], dict)
        and "chat_messages" in data[0]
    )


def _import_chat_messages_shape(
    data: list[dict],
    *,
    source: str = "import",
    provider_tag: str | None = None,
) -> list[MemoryEntry]:
    entries: list[MemoryEntry] = []
    for conv in data:
        if not isinstance(conv, dict):
            continue
        title = str(conv.get("name") or "")
        created = str(conv.get("created_at") or datetime.now(timezone.utc).isoformat())
        user_texts = [
            str(m.get("text") or "")
            for m in (conv.get("chat_messages") or [])
            if isinstance(m, dict) and str(m.get("sender", "")).lower() in ("human", "user")
        ]
        entry = _build_entry(
            title,
            user_texts,
            created,
            source=source,
            provider_tag=provider_tag,
        )
        if entry:
            entries.append(entry)
    return entries


def _has_generic_shape(data: Any) -> bool:
    return (
        isinstance(data, list)
        and bool(data)
        and isinstance(data[0], dict)
        and "messages" in data[0]
    )


def _import_generic_shape(
    data: list[dict],
    *,
    default_source: str = "import",
    default_provider: str | None = None,
) -> list[MemoryEntry]:
    entries: list[MemoryEntry] = []
    for conv in data:
        if not isinstance(conv, dict):
            continue
        title = str(conv.get("title") or conv.get("name") or "")
        provider = None
        if default_provider is not None:
            provider = str(conv.get("provider") or default_provider or "").strip().lower() or None
        created = str(conv.get("created_at") or datetime.now(timezone.utc).isoformat())
        user_texts = [
            str(m.get("content") or m.get("text") or "")
            for m in (conv.get("messages") or [])
            if isinstance(m, dict) and str(m.get("role", "")).lower() in ("user", "human")
        ]
        entry = _build_entry(
            title,
            user_texts,
            created,
            source=provider or default_source,
            provider_tag=provider,
        )
        if entry:
            entries.append(entry)
    return entries


def import_conversation_export(data: Any) -> tuple[str, list[MemoryEntry]]:
    """
    Parse a supported conversation shape and return (format_name, entries).

    Shape detection only chooses the parser. It never assigns a provider or
    model identity to the resulting memories.

    Raises ImportError_ when the payload matches no known shape.
    """
    if _has_mapping_shape(data):
        return "conversation", _import_mapping_shape(data)
    if _has_chat_messages_shape(data):
        return "conversation", _import_chat_messages_shape(data)
    if _has_generic_shape(data):
        return "generic", _import_generic_shape(data)
    raise ImportError_(
        "Unrecognized conversation export. Expected a JSON list with a "
        "mapping, chat_messages, or messages field."
    )


# Compatibility entry points retained for clients that still import provider
# specific exports directly. The app's current import path remains neutral and
# puts every upload into a user-named collection.
def _looks_like_chatgpt(data: Any) -> bool:
    return _has_mapping_shape(data)


def _looks_like_claude(data: Any) -> bool:
    return _has_chat_messages_shape(data)


def _looks_like_generic(data: Any) -> bool:
    return _has_generic_shape(data)


def import_chatgpt_export(data: list[dict]) -> list[MemoryEntry]:
    return _import_mapping_shape(data, source="chatgpt", provider_tag="chatgpt")


def import_claude_export(data: list[dict]) -> list[MemoryEntry]:
    return _import_chat_messages_shape(data, source="claude", provider_tag="claude")


def import_generic_export(data: list[dict], default_provider: str = "manual") -> list[MemoryEntry]:
    return _import_generic_shape(data, default_provider=default_provider)


def detect_and_import(data: Any, provider_hint: str | None = None) -> tuple[str, list[MemoryEntry]]:
    """Compatibility auto-detection entry point with provider-labelled output."""
    if _looks_like_chatgpt(data):
        return "chatgpt", import_chatgpt_export(data)
    if _looks_like_claude(data):
        return "claude", import_claude_export(data)
    if _looks_like_generic(data):
        return "generic", import_generic_export(data, provider_hint or "manual")
    raise ImportError_(
        "Unrecognized export format. Expected a ChatGPT conversations.json "
        "(list with 'mapping'), a Claude conversations.json (list with "
        "'chat_messages'), or a generic list of {title, messages:[{role, content}]} ."
    )


def _direct_text_entries(text: str, filename: str) -> list[MemoryEntry]:
    """Turn common text files into bounded, useful memory entries."""
    ext = os.path.splitext(filename.lower())[1]
    if ext in {".csv", ".tsv"}:
        rows = list(csv.DictReader(io.StringIO(text), delimiter="\t" if ext == ".tsv" else ","))
        if not rows or not any("text" in row for row in rows):
            raise ImportError_(f"{filename}: CSV/TSV must contain a 'text' column")
        return [MemoryEntry(text=row["text"].strip(), tags=_split_tags(row.get("tags", "")),
                            source=row.get("source", "import") or "import")
                for row in rows if row.get("text", "").strip()]
    if ext == ".jsonl":
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
        return _direct_entries(data, filename)
    # Markdown paragraphs are more useful than individual wrapped lines.
    chunks = (text.split("\n\n") if ext in {".md", ".markdown", ".rst"}
              else text.splitlines())
    return [MemoryEntry(text=chunk.strip().lstrip("# ").lstrip("-* "),
                        tags=[], source="import")
            for chunk in chunks if chunk.strip() and not chunk.lstrip().startswith("#")]


def _split_tags(value: str) -> list[str]:
    return [tag.strip() for tag in value.replace("|", ",").replace(";", ",").split(",") if tag.strip()]


def _direct_entries(data: Any, filename: str) -> list[MemoryEntry]:
    if isinstance(data, dict) and isinstance(data.get("memories"), list):
        data = data["memories"]
    if not isinstance(data, list):
        raise ImportError_(f"{filename}: expected a JSON list or a memory.json object")
    entries = []
    for item in data:
        if isinstance(item, dict) and str(item.get("text", "")).strip():
            entries.append(MemoryEntry(text=str(item["text"]).strip(),
                                       tags=item.get("tags", []), source=item.get("source", "import")))
    return entries


def import_uploaded_files(
    files: list[tuple[str, bytes]],
    provider_hint: str | None = None,
) -> tuple[str, list[MemoryEntry], int]:
    """Import a folder/archive upload without extracting anything to disk.

    ZIP paths are never trusted; nested archives and binary files are skipped.
    Returns (format, entries, skipped_files).
    """
    if len(files) > MAX_IMPORT_FILES:
        raise ImportError_(f"Too many files. The limit is {MAX_IMPORT_FILES:,}.")
    total = 0
    entries: list[MemoryEntry] = []
    formats: set[str] = set()
    skipped = 0
    queue = list(files)
    while queue:
        filename, payload = queue.pop(0)
        total += len(payload)
        if len(payload) > MAX_IMPORT_FILE_BYTES or total > MAX_IMPORT_TOTAL_BYTES:
            raise ImportError_("Import is too large (maximum 10 MB per file and 100 MB total).")
        ext = os.path.splitext(filename.lower())[1]
        if ext in ARCHIVE_EXTENSIONS:
            try:
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    for info in archive.infolist():
                        if info.is_dir() or info.filename.startswith(("/", "\\")) or ".." in info.filename.replace("\\", "/").split("/"):
                            skipped += 1
                            continue
                        if len(queue) + len(entries) >= MAX_IMPORT_FILES or info.file_size > MAX_IMPORT_FILE_BYTES:
                            skipped += 1
                            continue
                        queue.append((info.filename, archive.read(info)))
            except zipfile.BadZipFile as exc:
                raise ImportError_(f"{filename}: invalid ZIP archive") from exc
            continue
        try:
            if ext == ".json" or filename.lower().endswith("conversations.json"):
                parsed = json.loads(payload.decode("utf-8-sig"))
                try:
                    if provider_hint:
                        fmt, imported = detect_and_import(parsed, provider_hint)
                    else:
                        fmt, imported = import_conversation_export(parsed)
                except ImportError_:
                    imported = _direct_entries(parsed, filename)
                    fmt = "direct"
                formats.add(fmt)
            elif ext in TEXT_EXTENSIONS:
                imported = _direct_text_entries(payload.decode("utf-8", errors="replace"), filename)
                formats.add("direct")
            else:
                skipped += 1
                continue
        except (UnicodeDecodeError, json.JSONDecodeError, ImportError_) as exc:
            skipped += 1
            if isinstance(exc, ImportError_) and ext in {".json", ".jsonl", ".csv", ".tsv"}:
                raise
            continue
        entries.extend(imported)
    fmt = next(iter(formats)) if len(formats) == 1 else ("mixed" if formats else "empty")
    return fmt, entries, skipped
