"""
Theriad memory layer.

A modular, human-editable memory file that users can grow via:
  - manual entries (UI or direct file edits to data/memory.json)
  - named imports from conversation exports

Submodules:
  store      — JSON-file-backed MemoryStore (CRUD, hand-edit friendly)
  importers  — conversation export parsers → named memory entries
  providers  — named memory collection mentions and prompt injection
"""
from .providers import (
    PROVIDER_MEMORY,
    build_memory_context,
    build_model_memory_context,
    detect_mentioned_provider,
    get_provider_memory,
    providers_as_dicts,
)
from .store import MemoryEntry, MemoryStore, get_memory_store

__all__ = [
    "MemoryEntry",
    "MemoryStore",
    "get_memory_store",
    "PROVIDER_MEMORY",
    "get_provider_memory",
    "providers_as_dicts",
    "detect_mentioned_provider",
    "build_memory_context",
    "build_model_memory_context",
]
