"""Named memory collections and prompt injection helpers.

Imports are addressed by the alias the user gives them. There is no provider
classifier for named collections: ``@work`` and ``@research`` are simply named
collections selected from the current memory store. The small provider
capacity registry remains available for older API clients.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .store import get_memory_store

# Fixed cap for imported/shared memory context. It is intentionally independent
# of the model or provider selected for the response.
DEFAULT_INJECT_BUDGET_CHARS = 4000


@dataclass(frozen=True)
class ProviderMemory:
    id: str
    label: str
    context_window_tokens: int
    native_memory: bool
    inject_budget_chars: int


PROVIDER_MEMORY: dict[str, ProviderMemory] = {
    "claude": ProviderMemory("claude", "Claude", 1_000_000, True, 8000),
    "chatgpt": ProviderMemory("chatgpt", "ChatGPT", 128_000, True, 4000),
    "gemini": ProviderMemory("gemini", "Gemini", 1_000_000, True, 8000),
    "mistral": ProviderMemory("mistral", "Mistral", 128_000, False, 4000),
}

_MENTION_RE = re.compile(r"@(" + "|".join(PROVIDER_MEMORY) + r")\b", re.IGNORECASE)


def get_provider_memory(provider: str | None) -> ProviderMemory | None:
    if not isinstance(provider, str) or not provider:
        return None
    return PROVIDER_MEMORY.get(provider.lower())


def providers_as_dicts() -> list[dict]:
    """Return the legacy provider capacity metadata used by older clients."""
    return [asdict(provider) for provider in PROVIDER_MEMORY.values()]


def memory_mentions(store=None) -> list[dict]:
    """Return the named memory collections that can be selected with ``@``."""
    store = store or get_memory_store()
    # Model-scoped memories are injected only after the concrete response
    # model is selected. Keep them out of the shared memory picker.
    entries = [entry for entry in store.list() if entry.model_id is None]
    mentions = [{
        "id": "__all__",
        "alias": store.name,
        "label": "All memories",
        "kind": "store",
        "count": len(entries),
        "sources": sorted({entry.source for entry in entries}),
    }]
    for upload in store.upload_sources():
        mentions.append({
            "id": upload["id"],
            "alias": upload["name"],
            "label": upload["name"],
            "kind": "upload",
            "count": upload["count"],
            "sources": upload["sources"],
        })
    return mentions


def detect_mentioned_memory(text: str, store=None) -> dict | None:
    """Return the first named memory collection explicitly mentioned in text."""
    aliases = sorted(memory_mentions(store), key=lambda item: len(item["alias"]), reverse=True)
    for mention in aliases:
        alias = mention["alias"]
        if re.search(r"@" + re.escape(alias) + r"(?![A-Za-z0-9_-])", text or "", re.IGNORECASE):
            return mention
    return None


def detect_mentioned_provider(text: str) -> str | None:
    """Return a legacy provider mention or the store's named-memory alias."""
    match = _MENTION_RE.search(text or "")
    if match:
        return match.group(1).lower()
    try:
        store_name = (get_memory_store()).name
        if store_name and store_name not in PROVIDER_MEMORY:
            custom_re = re.compile(r"@" + re.escape(store_name) + r"\b", re.IGNORECASE)
            if custom_re.search(text or ""):
                return store_name
    except Exception:  # noqa: BLE001
        pass
    return None


def build_memory_context(
    provider: str | dict | None = None,
    *,
    store=None,
    query: str | None = None,
    mention: dict | None = None,
) -> tuple[str, int]:
    """
    Return (context_block, memories_used) for a named collection.

    Newest memories first, packed until the selected provider's legacy budget
    or the fixed named-memory budget is exhausted. Returns ("", 0) when the
    memory file or selected collection is empty.
    """
    if mention is not None:
        provider = mention
    provider_memory = get_provider_memory(provider if isinstance(provider, str) else None)
    budget = provider_memory.inject_budget_chars if provider_memory else DEFAULT_INJECT_BUDGET_CHARS
    memory_store = store or get_memory_store()
    mention = provider if isinstance(provider, dict) else None
    if mention is None and isinstance(provider, str) and provider.lower() not in PROVIDER_MEMORY:
        mention = detect_mentioned_memory("@" + provider, memory_store)
    # Model-scoped memories are injected only by build_model_memory_context
    # after the concrete response model is selected.
    entries = [entry for entry in memory_store.list() if entry.model_id is None]
    if mention and mention.get("kind") == "upload":
        entries = [entry for entry in entries if entry.upload_id == mention["id"]]
    if not entries:
        return "", 0

    if query:
        terms = {word.lower() for word in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{2,}", query)}

        def relevance(entry) -> tuple[int, str]:
            title = str(entry.conversation_title or "").lower()
            haystack = f"{title} {entry.text} {' '.join(entry.tags)}".lower()
            score = sum(8 if term in title else 1 for term in terms if term in haystack)
            return score, entry.updated_at

        ranked = sorted(entries, key=relevance, reverse=True)
        relevant = [entry for entry in ranked if relevance(entry)[0] > 0]
        entries = relevant or ranked
    else:
        entries = sorted(entries, key=lambda e: e.updated_at, reverse=True)

    source_label = f" @{mention['alias']}" if mention else ""
    header = f"Known facts about the user (from{source_label} memory):\n"
    lines: list[str] = []
    used = len(header)
    count = 0
    for entry in entries:
        line = f"- {entry.text}\n"
        if used + len(line) > budget:
            break
        lines.append(line)
        used += len(line)
        count += 1
    if not lines:
        return "", 0
    return header + "".join(lines), count


def build_model_memory_context(
    model_id: str,
    *,
    store=None,
    query: str | None = None,
) -> tuple[str, int]:
    """Build the opt-in memory block learned by one concrete model."""
    memory_store = store or get_memory_store()
    profile = memory_store.get_model_profile(model_id, create=False)
    if not profile or not profile.get("context_enabled"):
        return "", 0
    entries = memory_store.model_memory_entries(model_id)
    if not entries:
        return "", 0

    if query:
        terms = {word.lower() for word in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{2,}", query)}

        def relevance(entry) -> tuple[int, str]:
            haystack = f"{entry.text} {' '.join(entry.tags)}".lower()
            return sum(1 for term in terms if term in haystack), entry.updated_at

        ranked = sorted(entries, key=relevance, reverse=True)
        relevant = [entry for entry in ranked if relevance(entry)[0] > 0]
        entries = relevant or ranked
    else:
        entries = sorted(entries, key=lambda entry: entry.updated_at, reverse=True)

    header = f"Known facts learned by {model_id} from prior conversations:\n"
    lines: list[str] = []
    used = len(header)
    for entry in entries:
        line = f"- {entry.text}\n"
        if used + len(line) > DEFAULT_INJECT_BUDGET_CHARS:
            break
        lines.append(line)
        used += len(line)
    if not lines:
        return "", 0
    return header + "".join(lines), len(lines)
