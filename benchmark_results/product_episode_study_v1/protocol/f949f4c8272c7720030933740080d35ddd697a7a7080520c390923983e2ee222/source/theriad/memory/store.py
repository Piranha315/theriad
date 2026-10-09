"""
JSON-file-backed memory store.

Design decisions:
- The memory file is a single pretty-printed JSON document at
  <data_dir>/memory.json (default: ./data/memory.json). It is meant to be
  hand-editable: users can open it, add/remove/reword entries, and the store
  picks the changes up automatically (mtime check on every read).
- No database. Memories are small (hundreds, not millions) and a flat file
  keeps the layer modular — the whole memory is one portable artifact users
  can copy between machines, check into a dotfiles repo, or share.
- Writes are atomic (tmp file + os.replace) so a crash never corrupts the file.
- Entries are plain dicts on disk; MemoryEntry is the typed in-process view.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from theriad.core.logging.logger import get_logger

logger = get_logger("memory.store")

MEMORY_FILE_VERSION = 3

# Written into the file so people who open it by hand know the rules.
_FILE_HELP = (
    "Hand-editable Theriad memory file. Each entry needs at least 'text'. "
    "Optional: 'title', 'tags' (list of strings), 'source' "
    "(import|manual|...), 'conversation_title', 'upload_id', 'upload_name', "
    "and 'model_id'. Top-level 'model_profiles' controls learned context "
    "per provider:model. "
    "Missing ids/timestamps are filled in automatically on the next write."
)

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_alias(value: str, *, fallback: str | None = None) -> str:
    """Validate a compact alias that can be typed after ``@``."""
    name = str(value or "").strip().lower()
    if not name and fallback is not None:
        name = fallback
    if not name:
        raise ValueError("name must not be empty")
    if not re.match(r"^[a-z0-9][a-z0-9_-]{0,38}[a-z0-9]$|^[a-z0-9]$", name):
        raise ValueError(
            "name must be 1–40 characters, letters/digits/hyphens/underscores only"
        )
    return name


def _normalise_model_id(value: str) -> str:
    """Validate the stable provider:model key used for scoped memories."""
    model_id = str(value or "").strip().lower()
    if not model_id or len(model_id) > 220 or not re.match(r"^[a-z0-9][a-z0-9._:/-]*$", model_id):
        raise ValueError("model_id must be a provider:model identifier")
    return model_id


@dataclass
class MemoryEntry:
    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str | None = None
    tags: list[str] = field(default_factory=list)
    source: str = "manual"
    conversation_title: str | None = None
    upload_id: str | None = None
    upload_name: str | None = None
    model_id: str | None = None
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("title") is None:
            d.pop("title")
        if d.get("conversation_title") is None:
            d.pop("conversation_title")
        if d.get("upload_id") is None:
            d.pop("upload_id")
        if d.get("upload_name") is None:
            d.pop("upload_name")
        if d.get("model_id") is None:
            d.pop("model_id")
        return d

    @classmethod
    def from_dict(cls, raw: dict) -> MemoryEntry:
        """Tolerant constructor for hand-edited entries (fills gaps)."""
        title = raw.get("title")
        return cls(
            text=str(raw.get("text", "")).strip(),
            id=str(raw.get("id") or uuid.uuid4()),
            title=str(title).strip() or None if title is not None else None,
            tags=[str(t).strip().lower() for t in raw.get("tags", []) if str(t).strip()],
            source=str(raw.get("source", "manual")).lower(),
            conversation_title=raw.get("conversation_title"),
            upload_id=str(raw.get("upload_id")).strip() or None
            if raw.get("upload_id") is not None else None,
            upload_name=str(raw.get("upload_name")).strip() or None
            if raw.get("upload_name") is not None else None,
            model_id=str(raw.get("model_id")).strip().lower() or None
            if raw.get("model_id") is not None else None,
            created_at=str(raw.get("created_at") or _utcnow()),
            updated_at=str(raw.get("updated_at") or raw.get("created_at") or _utcnow()),
        )


# Simple keyword extraction for auto-tagging imported memories.
_STOPWORDS = frozenset(
    "the a an and or but if then else for while of to in on at is are was were "
    "be been i you he she it we they my your our their this that these those "
    "with as by from about into over after before what which who whom how why "
    "can could should would do does did not no yes me him her them us".split()
)


def auto_tags(text: str, max_tags: int = 4) -> list[str]:
    """Deterministic keyword tags: most frequent non-stopword tokens."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
    counts: dict[str, int] = {}
    for w in words:
        if w in _STOPWORDS:
            continue
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:max_tags]]


class MemoryStore:
    """Thread-safe CRUD over the memory JSON file."""

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else Path("data") / "memory.json"
        self._lock = threading.Lock()
        self._entries: list[MemoryEntry] = []
        self._model_profiles: dict[str, dict] = {}
        # Per-user files use a hashed filename for isolation, but their
        # user-facing @mention alias should still start as "memory".
        self._name: str = "memory" if self._path.parent.name == "memory-users" else self._path.stem
        self._mtime: float | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def name(self) -> str:
        """User-defined @mention alias for this memory store (e.g. 'work', 'research')."""
        with self._lock:
            self._load_unlocked()
            return self._name

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _load_unlocked(self) -> None:
        """(Re)load from disk if the file changed since the last read."""
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            self._entries = []
            self._model_profiles = {}
            self._mtime = None
            return
        if self._mtime is not None and mtime == self._mtime:
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            items = raw.get("memories", []) if isinstance(raw, dict) else raw
            self._entries = [
                MemoryEntry.from_dict(item)
                for item in items
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
            # Load stored name if present; fall back to filename stem
            if isinstance(raw, dict) and raw.get("name"):
                self._name = str(raw["name"]).strip() or self._path.stem
            raw_profiles = raw.get("model_profiles", {}) if isinstance(raw, dict) else {}
            if not isinstance(raw_profiles, dict):
                raw_profiles = {}
            self._model_profiles = {}
            for raw_model_id, raw_profile in raw_profiles.items():
                if not isinstance(raw_profile, dict):
                    continue
                try:
                    model_id = _normalise_model_id(raw_model_id)
                except ValueError:
                    continue
                try:
                    max_memories = int(raw_profile.get("max_memories", 100))
                except (TypeError, ValueError):
                    max_memories = 100
                self._model_profiles[model_id] = {
                    "model_id": model_id,
                    "context_enabled": bool(raw_profile.get("context_enabled", True)),
                    "learning_enabled": bool(raw_profile.get("learning_enabled", False)),
                    "max_memories": max(10, min(500, max_memories)),
                    "created_at": str(raw_profile.get("created_at") or _utcnow()),
                    "updated_at": str(raw_profile.get("updated_at") or _utcnow()),
                }
            self._mtime = mtime
        except (json.JSONDecodeError, OSError) as exc:
            # Never clobber a hand-edited file we can't parse; surface and keep
            # the last good in-memory copy.
            logger.warning("memory file unreadable; keeping last good state", extra={"error": str(exc)})

    def _save_unlocked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "version": MEMORY_FILE_VERSION,
            "name": self._name,
            "_help": _FILE_HELP,
            "memories": [e.to_dict() for e in self._entries],
            "model_profiles": self._model_profiles,
        }
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self._path)
        try:
            self._mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            self._mtime = None

    # ── Name ──────────────────────────────────────────────────────────────────

    def set_name(self, new_name: str) -> None:
        """
        Rename the memory store's @mention alias.

        The name is stored in the JSON header and picked up immediately by
        the named-memory mention matcher. Must be 1-40 chars, letters/digits/
        hyphens only.
        """
        new_name = _normalise_alias(new_name)
        with self._lock:
            self._load_unlocked()
            self._name = new_name
            self._save_unlocked()

    def _unique_upload_alias_unlocked(self, suggested_name: str | None) -> str:
        """Return a valid, non-colliding alias for a new imported collection."""
        base = _normalise_alias(suggested_name or "upload", fallback="upload")
        used = {self._name}
        used.update(e.upload_name for e in self._entries if e.upload_name)
        candidate = base
        suffix = 2
        while candidate in used:
            suffix_text = f"-{suffix}"
            candidate = f"{base[:40 - len(suffix_text)]}{suffix_text}"
            suffix += 1
        return candidate

    def add_upload(
        self,
        entries: list[MemoryEntry],
        *,
        suggested_name: str | None = None,
        dedupe: bool = True,
    ) -> dict | None:
        """Add imported entries as one named collection addressable with ``@``."""
        if not entries:
            return None
        with self._lock:
            self._load_unlocked()
            existing_texts = {e.text for e in self._entries} if dedupe else set()
            upload_id = str(uuid.uuid4())
            upload_name = self._unique_upload_alias_unlocked(suggested_name)
            added = 0
            for entry in entries:
                if not entry.text.strip():
                    continue
                if dedupe and entry.text in existing_texts:
                    continue
                entry.upload_id = upload_id
                entry.upload_name = upload_name
                self._entries.append(entry)
                existing_texts.add(entry.text)
                added += 1
            if not added:
                return None
            self._save_unlocked()
            return {"id": upload_id, "name": upload_name, "count": added}

    def upload_sources(self) -> list[dict]:
        """Return imported collections with their stable ids and aliases."""
        with self._lock:
            self._load_unlocked()
            groups: dict[str, dict] = {}
            for entry in self._entries:
                if not entry.upload_id or entry.model_id is not None:
                    continue
                group = groups.setdefault(
                    entry.upload_id,
                    {
                        "id": entry.upload_id,
                        "name": entry.upload_name or "upload",
                        "count": 0,
                        "sources": set(),
                        "created_at": entry.created_at,
                        "updated_at": entry.updated_at,
                    },
                )
                group["count"] += 1
                group["sources"].add(entry.source)
                group["created_at"] = min(group["created_at"], entry.created_at)
                group["updated_at"] = max(group["updated_at"], entry.updated_at)
            return [
                {**group, "sources": sorted(group["sources"])}
                for group in sorted(
                    groups.values(), key=lambda item: item["updated_at"], reverse=True
                )
            ]

    def rename_upload(self, upload_id: str, new_name: str) -> dict:
        """Rename an imported collection without changing its stable id."""
        new_name = _normalise_alias(new_name)
        with self._lock:
            self._load_unlocked()
            entries = [e for e in self._entries if e.upload_id == upload_id]
            if not entries:
                raise ValueError("memory upload not found")
            if new_name == self._name or any(
                e.upload_id != upload_id and e.upload_name == new_name
                for e in self._entries
            ):
                raise ValueError(f"memory upload alias '@{new_name}' is already in use")
            for entry in entries:
                entry.upload_name = new_name
                entry.updated_at = _utcnow()
            self._save_unlocked()
        return next(source for source in self.upload_sources() if source["id"] == upload_id)

    # ── Model-scoped memory ──────────────────────────────────────────────────

    @staticmethod
    def _default_model_profile(model_id: str) -> dict:
        now = _utcnow()
        return {
            "model_id": model_id,
            "context_enabled": True,
            "learning_enabled": False,
            "max_memories": 100,
            "created_at": now,
            "updated_at": now,
        }

    def _ensure_model_profile_unlocked(self, model_id: str) -> dict:
        profile = self._model_profiles.get(model_id)
        if profile is None:
            profile = self._default_model_profile(model_id)
            self._model_profiles[model_id] = profile
        return profile

    def get_model_profile(self, model_id: str, *, create: bool = False) -> dict | None:
        """Return one model's memory policy, optionally creating its default."""
        normalized = _normalise_model_id(model_id)
        with self._lock:
            self._load_unlocked()
            profile = self._model_profiles.get(normalized)
            if profile is None and create:
                profile = self._ensure_model_profile_unlocked(normalized)
                self._save_unlocked()
            if profile is None:
                return None
            result = dict(profile)
            result["count"] = sum(1 for entry in self._entries if entry.model_id == normalized)
            return result

    def model_profiles(self) -> list[dict]:
        """Return model policies with their current scoped-memory counts."""
        with self._lock:
            self._load_unlocked()
            counts: dict[str, int] = {}
            for entry in self._entries:
                if entry.model_id:
                    counts[entry.model_id] = counts.get(entry.model_id, 0) + 1
            return [
                {**profile, "count": counts.get(model_id, 0)}
                for model_id, profile in sorted(
                    self._model_profiles.items(),
                    key=lambda item: item[1]["updated_at"],
                    reverse=True,
                )
            ]

    def update_model_profile(
        self,
        model_id: str,
        *,
        context_enabled: bool | None = None,
        learning_enabled: bool | None = None,
        max_memories: int | None = None,
    ) -> dict:
        """Persist a model's context and learning policy."""
        normalized = _normalise_model_id(model_id)
        with self._lock:
            self._load_unlocked()
            profile = self._ensure_model_profile_unlocked(normalized)
            if context_enabled is not None:
                profile["context_enabled"] = bool(context_enabled)
            if learning_enabled is not None:
                profile["learning_enabled"] = bool(learning_enabled)
            if max_memories is not None:
                profile["max_memories"] = max(10, min(500, int(max_memories)))
            profile["updated_at"] = _utcnow()
            self._save_unlocked()
            result = dict(profile)
            result["count"] = sum(1 for entry in self._entries if entry.model_id == normalized)
            return result

    def model_memory_entries(self, model_id: str) -> list[MemoryEntry]:
        """Return only memories learned by the selected model."""
        normalized = _normalise_model_id(model_id)
        with self._lock:
            self._load_unlocked()
            return [entry for entry in self._entries if entry.model_id == normalized]

    def add_model_memories(
        self,
        model_id: str,
        texts: list[str],
        *,
        source: str = "learned",
    ) -> list[MemoryEntry]:
        """Add deduplicated, model-scoped memories and enforce the profile cap."""
        normalized = _normalise_model_id(model_id)
        with self._lock:
            self._load_unlocked()
            profile = self._ensure_model_profile_unlocked(normalized)
            existing = {entry.text.lower() for entry in self._entries if entry.model_id == normalized}
            added: list[MemoryEntry] = []
            for text in texts:
                cleaned = " ".join(str(text or "").split()).strip()
                if not cleaned or cleaned.lower() in existing:
                    continue
                entry = MemoryEntry(
                    text=cleaned,
                    title="Learned from conversation",
                    tags=["learned", "model-memory"],
                    source=source.lower(),
                    model_id=normalized,
                )
                self._entries.append(entry)
                existing.add(cleaned.lower())
                added.append(entry)

            scoped = [entry for entry in self._entries if entry.model_id == normalized]
            max_memories = int(profile.get("max_memories", 100))
            evicted_ids: set[str] = set()
            if len(scoped) > max_memories:
                keep_ids = {
                    entry.id
                    for entry in sorted(scoped, key=lambda item: item.updated_at, reverse=True)[:max_memories]
                }
                evicted_ids = {entry.id for entry in scoped if entry.id not in keep_ids}
                self._entries = [
                    entry
                    for entry in self._entries
                    if entry.model_id != normalized or entry.id in keep_ids
                ]
            if added or evicted_ids:
                profile["updated_at"] = _utcnow()
                self._save_unlocked()
            if evicted_ids:
                added = [entry for entry in added if entry.id not in evicted_ids]
            return added

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def list(
        self,
        *,
        source: str | None = None,
        tag: str | None = None,
        query: str | None = None,
    ) -> list[MemoryEntry]:
        with self._lock:
            self._load_unlocked()
            entries = list(self._entries)
        if source:
            entries = [e for e in entries if e.source == source.lower()]
        if tag:
            entries = [e for e in entries if tag.lower() in e.tags]
        if query:
            q = query.lower()
            entries = [
                e for e in entries
                if q in e.text.lower()
                or q in (e.title or "").lower()
                or q in (e.conversation_title or "").lower()
                or any(q in tag_name.lower() for tag_name in e.tags)
            ]
        return entries

    def get(self, entry_id: str) -> MemoryEntry | None:
        with self._lock:
            self._load_unlocked()
            for e in self._entries:
                if e.id == entry_id:
                    return e
        return None

    def add(
        self,
        text: str,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
        source: str = "manual",
        conversation_title: str | None = None,
    ) -> MemoryEntry:
        text = text.strip()
        if not text:
            raise ValueError("memory text must not be empty")
        entry = MemoryEntry(
            text=text,
            title=title.strip() or None if title is not None else None,
            tags=[t.strip().lower() for t in (tags or []) if t.strip()] or auto_tags(text),
            source=source.lower(),
            conversation_title=conversation_title,
        )
        with self._lock:
            self._load_unlocked()
            self._entries.append(entry)
            self._save_unlocked()
        return entry

    def add_many(self, entries: list[MemoryEntry], *, dedupe: bool = True) -> int:
        """Bulk insert (used by importers). Returns the number actually added."""
        with self._lock:
            self._load_unlocked()
            existing_texts = {e.text for e in self._entries} if dedupe else set()
            added = 0
            for entry in entries:
                if not entry.text.strip():
                    continue
                if dedupe and entry.text in existing_texts:
                    continue
                self._entries.append(entry)
                existing_texts.add(entry.text)
                added += 1
            if added:
                self._save_unlocked()
        return added

    def update(
        self,
        entry_id: str,
        *,
        text: str | None = None,
        title: str | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> MemoryEntry | None:
        with self._lock:
            self._load_unlocked()
            for e in self._entries:
                if e.id == entry_id:
                    if text is not None and text.strip():
                        e.text = text.strip()
                    if title is not None:
                        e.title = title.strip() or None
                    if tags is not None:
                        e.tags = [t.strip().lower() for t in tags if t.strip()]
                    if source is not None:
                        e.source = source.lower()
                    e.updated_at = _utcnow()
                    self._save_unlocked()
                    return e
        return None

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            self._load_unlocked()
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.id != entry_id]
            if len(self._entries) != before:
                self._save_unlocked()
                return True
        return False


# Module-level singleton — one memory file per process by default.
_store: MemoryStore | None = None
_store_lock = threading.Lock()


def get_memory_store(path: str | Path | None = None) -> MemoryStore:
    """Return the shared MemoryStore (or a dedicated one when path is given)."""
    global _store
    if path is not None:
        return MemoryStore(path)
    with _store_lock:
        if _store is None:
            _store = MemoryStore()
        return _store
